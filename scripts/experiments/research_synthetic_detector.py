"""Controlled YOLO11n real-only versus added synthetic glyphs; immutable research runs."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, random, sys, time, urllib.request
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'backend/training_runs/20261004-yolo-synthetic-v2'
DEPS=ROOT/'backend/training_runs/yolo-research-deps'
sys.path[:0]=[str(DEPS),str(ROOT),str(ROOT/'backend')]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'backend/training_runs/yolo-research-config')
os.environ['MPLCONFIGDIR']=str(ROOT/'backend/training_runs/yolo-research-config/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.nms import non_max_suppression
from scripts.train_bbox_detector import make_tiles, tile_windows
from scripts.experiments import research_full_proposals as f
from codex_model.classifier import CodexClassifier, _preprocess_image
from codex_pipeline.models.projection_head import ProjectionHead
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter

SNAPSHOT=ROOT/'backend/training_corpus/snapshots/snapshot-611c734590f272060251'
CACHE=ROOT/'backend/model_registry/versions/20261003-annotation-lanczos-warmstart/training_data/precomputed/features.pt'
PAGE_CACHE=ROOT/'backend/training_runs/20261004-full-proposal-readout-v1/full-proposal-features.pt'
NESTED=ROOT/'reports/nested-rejection-results-20261004.json'
BRANCH=ROOT/'reports/nested-branch-rejection-results-20261004.json'
PRETRAIN=ROOT/'backend/training_runs/detection-pretrained/yolo11n.pt'
URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
PRETRAIN_SHA='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
SIZE=768; STEPS=200; SYNTH_PAGES=64; GLYPHS=16; GRID=[.05,.10,.15,.20,.25,.35,.50,.65]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def resize_packet(image,boxes,ignores):
    scale=min(SIZE/image.width,SIZE/image.height)
    w,h=max(1,round(image.width*scale)),max(1,round(image.height*scale))
    ox,oy=(SIZE-w)//2,(SIZE-h)//2
    canvas=Image.new('RGB',(SIZE,SIZE),(114,114,114))
    canvas.paste(image.resize((w,h),Image.Resampling.BILINEAR),(ox,oy))
    # Use actual rounded scales for each axis, for targets and inverse inference.
    sx,sy=w/image.width,h/image.height
    transform=lambda b:[b[0]*sx+ox,b[1]*sy+oy,b[2]*sx+ox,b[3]*sy+oy]
    return canvas,[transform(b) for b in boxes],[transform(b) for b in ignores],(sx,sy,ox,oy)

def xyxy(b):
    x,y,w,h=b;return [x,y,x+w,y+h]

def cutout(path):
    image=Image.open(path).convert('RGB');arr=np.array(image)
    border=np.concatenate([arr[0],arr[-1],arr[:,0],arr[:,-1]])
    paper=np.median(border,axis=0)
    fg=(np.abs(arr.astype(float)-paper).max(2)>25)|(arr.max(2)<145)
    if fg.mean()<.015 or fg.mean()>.97:return None
    fg=binary_fill_holes(fg)
    yy,xx=np.nonzero(fg)
    if not len(xx):return None
    bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
    if min(bounds[2]-bounds[0],bounds[3]-bounds[1])<4:return None
    mask=Image.fromarray(fg.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(.35))
    image.putalpha(mask)
    return image.crop(bounds)

class IgnoredBCE(torch.nn.Module):
    def __init__(self):super().__init__();self.mask=None
    def forward(self,logits,target):
        values=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        if self.mask is not None:
            assert self.mask.shape==values.shape[:2]
            # Ignore negative anchor centres only; soft-positive assigned targets retain loss.
            values=values.masked_fill((self.mask & ~target.gt(0).any(-1)).unsqueeze(-1),0)
        return values

def check():
    x=torch.zeros(1,3,1,requires_grad=True);y=torch.tensor([[[0.],[1.],[0.]]])
    criterion=IgnoredBCE();criterion.mask=torch.tensor([[True,True,False]])
    v=criterion(x,y);v.sum().backward()
    assert v[0,0]==0 and v[0,1]>0 and x.grad[0,0]==0 and x.grad[0,1]!=0 and x.grad[0,2]!=0
    criterion.mask=None;torch.testing.assert_close(criterion(x,y),F.binary_cross_entropy_with_logits(x,y,reduction='none'))
    _,boxes,ignored,t=resize_packet(Image.new('RGB',(50,100)),[[0,0,50,100]],[[10,20,30,40]])
    assert boxes==[[192,0,576,768]] and ignored==[[268.8,153.6,422.4,307.2]]
    return dict(ignored_negatives_zero_gradient=True,positive_targets_protected=True,empty_ignore_matches_native=True,letterbox_targets_checked=True)

def prepare():
    assert not RUN.exists(),'Use a fresh run, never overwrite research evidence'
    RUN.mkdir(parents=True)
    if not PRETRAIN.exists():urllib.request.urlretrieve(URL,PRETRAIN)
    assert sha(PRETRAIN)==PRETRAIN_SHA and PRETRAIN.stat().st_size==5613764
    nested=read(NESTED);cache=torch.load(CACHE,map_location='cpu',weights_only=True)
    pages=torch.load(PAGE_CACHE,map_location='cpu',weights_only=True)['pages']
    meta={r['row_id']:r for r in csv.DictReader((SNAPSHOT/'metadata.csv').open())}
    kinds={r['row_id']:r['source_kind'] for r in read(SNAPSHOT/'snapshot_manifest.json')['rows']}
    source_paths={str(p) for p in [NESTED,BRANCH,CACHE,PAGE_CACHE,SNAPSHOT/'metadata.csv',SNAPSHOT/'snapshot_manifest.json',PRETRAIN,Path(__file__),Path(f.__file__),ROOT/'scripts/train_bbox_detector.py',ROOT/'scripts/evaluate_glyph_benchmark.py',ROOT/'backend/codex_model/classifier.py']}
    source_paths.update(nested['protocol']['input_sha256'])
    source_paths.update(str(p) for p in (DEPS/'ultralytics').rglob('*') if p.suffix in {'.py','.yaml'})
    source_paths.add(str(PRETRAIN.with_name('yolo11n-release-provenance.json')))
    all_real=[];all_synth=[];splits=[];cutouts={};random.seed(42)
    for split in nested['protocol']['splits']:
        fold=split['fold'];folder=RUN/'data'/f'fold{fold}';folder.mkdir(parents=True)
        fit=[p for p in pages if p['group'] in split['fit_pages']]
        withheld=set(split['held'])|set(split['calibration'])|set(split['excluded'])
        forbidden_pixels={r['source_pixel_sha256'] for r in meta.values() if r['source_group'] in withheld}
        eligible=[meta[cache['row_ids'][i]] for i in split['raw']]
        assert all(r['dataset_split']=='train' and r['source_group'] not in withheld and r['source_pixel_sha256'] not in forbidden_pixels for r in eligible)
        eligible_synthetic=[row for row in eligible if kinds[row['row_id']]=='legacy']
        pools=collections.defaultdict(list)
        for row in eligible_synthetic:pools[int(row['class_label'])].append(row)
        assert set(pools)==set(range(286))
        real=[];background=[];sizes=[]
        for p in fit:
            source_paths.add(p['image']);assert sha(p['image'])==p['image_sha256']
            image=Image.open(p['image']).convert('RGB')
            shapes=[xyxy(g['bbox']) for g in p['gt']+p['ignore']]
            page=dict(source_group=p['group'],image_path=p['image'],width=image.width,height=image.height,
                      boxes=p['gt'],ignore_boxes=[g['bbox'] for g in p['ignore']])
            sizes.extend(max(g['bbox'][2:]) for g in p['gt'])
            for index,tile in enumerate(make_tiles(page)):
                canvas,boxes,ignores,_=resize_packet(image.crop(tile['window']),tile['boxes'],tile['ignore_boxes'])
                path=folder/f'real_{len(real):03d}.png';canvas.save(path)
                real.append(dict(path=str(path),boxes=boxes,ignore_boxes=ignores,group=p['group']))
            # Background patches cannot intersect any known or ignored glyph rectangle.
            for y in range(0,image.height-95,96):
                for x in range(0,image.width-95,96):
                    patch=[x,y,x+96,y+96]
                    if any(not (patch[2]<=b[0]-5 or patch[0]>=b[2]+5 or patch[3]<=b[1]-5 or patch[1]>=b[3]+5) for b in shapes):continue
                    background_patch=image.crop(patch)
                    pixels=np.array(background_patch)
                    if (pixels.mean(2)<145).mean()>.001:continue
                    background.append((background_patch,p['group'],patch))
        assert real and background
        rng=random.Random(142+fold);labels=list(sorted(pools));rng.shuffle(labels);synth=[];used=set();discarded=set()
        for index in range(SYNTH_PAGES):
            patch,bg_group,bg_bbox=rng.choice(background)
            canvas=patch.resize((SIZE,SIZE),Image.Resampling.BILINEAR);boxes=[];placed=[]
            attempts=0
            while len(boxes)<GLYPHS and attempts<GLYPHS*40:
                label=labels[(index*GLYPHS+attempts)%len(labels)]
                row=rng.choice(pools[label]);attempts+=1
                rid=row['row_id'];path=SNAPSHOT/row['image_path']
                if rid not in cutouts:cutouts[rid]=cutout(path);source_paths.add(str(path))
                obj=cutouts[rid]
                if obj is None:discarded.add(rid);continue
                long_side=max(24,min(180,rng.choice(sizes)*rng.uniform(.7,1.3)))
                factor=long_side/max(obj.size);w,h=max(6,round(obj.width*factor)),max(6,round(obj.height*factor))
                x,y=rng.randint(8,SIZE-w-8),rng.randint(8,SIZE-h-8);box=[x,y,x+w,y+h]
                if any(not (box[2]+6<b[0] or box[0]-6>b[2] or box[3]+6<b[1] or box[1]-6>b[3]) for b in boxes):continue
                canvas.paste(obj.resize((w,h),Image.Resampling.LANCZOS),(x,y),obj.resize((w,h),Image.Resampling.LANCZOS))
                boxes.append(box);placed.append(dict(row_id=rid,source_group=row['source_group'],source_pixel_sha256=row['source_pixel_sha256'],class_name=row['element_name'],bbox=box));used.add(rid)
            assert len(boxes)==GLYPHS,(fold,index,len(boxes))
            path=folder/f'synth_{index:03d}.png';canvas.save(path)
            synth.append(dict(path=str(path),boxes=boxes,ignore_boxes=[],group=f'synthetic:{fold}:{index}',background_group=bg_group,background_bbox=bg_bbox,objects=placed))
        plan_rng=random.Random(20261004+fold)
        real_plan=[plan_rng.sample(range(len(real)),2) for _ in range(STEPS)]
        syn_rng=random.Random(42000+fold);synth_plan=[syn_rng.sample(range(len(synth)),2) for _ in range(STEPS)]
        splits.append(dict(fold=fold,fit_pages=split['fit_pages'],calibration=split['calibration'],held=split['held'],excluded=split['excluded'],raw_ids=split['raw'],eligible_crops=len(eligible),eligible_legacy_crops=len(eligible_synthetic),synthetic_unique_source_crops=len(used),discarded_candidate_crops=len(discarded),support_counts={c:len({r['source_pixel_sha256'] for r in eligible if int(r['class_label'])==c}) for c in range(286)},real=real,synth=synth,real_plan=real_plan,synth_plan=synth_plan))
        print('PREPARED',fold,'real tiles',len(real),'synth',len(synth),'unique crops',len(used),flush=True)
        if fold==0:
            preview=Image.new('RGB',(2*SIZE,SIZE),(255,255,255))
            for slot,row in enumerate([real[0],synth[0]]):
                img=Image.open(row['path']).convert('RGB');draw=ImageDraw.Draw(img)
                for b in row['boxes']:draw.rectangle(b,outline=(200,30,30),width=2)
                preview.paste(img,(slot*SIZE,0))
            preview.save(ROOT/'reports/yolo-synthetic-data-preview-20261004.png')
    dataset=dict(splits=splits,pages=pages);write(RUN/'dataset.json',dataset)
    paths=source_paths|{z['path'] for sp in splits for k in ['real','synth'] for z in sp[k]}|{str(RUN/'dataset.json')}
    protocol=dict(schema_version='yolo-synthetic.v1',model='YOLO11n, COCO pretrained shared weights, target nc=1',pretrained_url=URL,pretrained_sha256=sha(PRETRAIN),ultralytics_version='8.3.221',seed=42,steps=STEPS,fits=10,optimizer_updates=10*STEPS,
                  optimizer=dict(name='AdamW',lr_peak=.0005,lr_final=.00005,warmup_steps=20,weight_decay=.0005,clip_norm=5),
                  arms=['real','real_plus_synthetic'],real_batch=2,extra_synthetic_batch=2,synthetic_loss_weight=.5,
                  comparison='Same initialization, real tiles, real batch plan and optimizer updates. Mixed arm adds a separate two-image synthetic forward and half its mean native loss. Additional data/compute explicit, not compute matched.',
                  trainable='all YOLO parameters; BatchNorm running statistics frozen in both arms, affine parameters learned',
                  data='Reuse exact nested five outer folds and three independent calibration pages. Recorded components/groups/pixels excluded before synthesis. 64x16 glyphs/fold, class-cycled sampling from eligible legacy raw crops only; external/example montages excluded after v1 preflight, foreground mask estimated from border colour + dark ink, holes filled, real glyph-free and ink-poor background patches only from fit pages.',
                  ignore='Negative anchor centres inside clipped ignored/nonowner boxes get zero BCE; positive soft targets protected. Box/DFL losses unchanged. 343 known GT,7 unknown/conflict regions for readout.',
                  inference=dict(size=SIZE,views='existing 768/512 tile windows plus whole page, target independent',floor=.05,nms=.5,max_page_boxes=100),
                  selection='Fixed final update200, no checkpoint selection. Per arm/fold choose detector confidence from fixed grid maximizing mean calibration-page detection F1 at IoU>=.5; ties lower threshold. Save before outer readout.',
                  recognition='Frozen seed42 nested paired projection/prototypes per fold plus historical rare gating. Raw detector rectangle and detector-box-prompted SAM mask, padding5. Compare at semantic threshold .35 and existing branch thresholds; no new recognition fits or semantic threshold selection.',
                  calibration_grid=GRID,limitations='Same17 already examined pages, exploratory, not blind. Base historical provenance unknown. Synthetic masks/boxes estimated, not original recovered coordinates. Extra synthetic compute reported. No dev/test scoring and no activation.',checks=check(),input_sha256={p:sha(p) for p in sorted(paths)},activated=False,locked_test_scored=False)
    for sp in splits:
        for z in sp['synth']:
            assert z['background_group'] in sp['fit_pages']
            for obj in z['objects']:assert obj['source_group'] not in sp['excluded']
    write(RUN/'preregistration.json',protocol)
    print('PREPARATION_DONE',flush=True)

def anchors_for(model):
    feats=[torch.empty(1,1,SIZE//int(s),SIZE//int(s)) for s in model.stride.cpu()]
    anchors,strides=make_anchors(feats,model.stride.cpu(),.5)
    return (anchors*strides).numpy()

def packets(rows,anchors):
    result=[]
    for r in rows:
        boxes=np.array(r['boxes'],dtype=np.float32).reshape(-1,4)
        targets=np.column_stack(((boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2,boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]))/SIZE
        mask=np.zeros(len(anchors),dtype=bool)
        for b in r['ignore_boxes']:mask|=(anchors[:,0]>=b[0])&(anchors[:,0]<=b[2])&(anchors[:,1]>=b[1])&(anchors[:,1]<=b[3])
        result.append((torch.from_numpy(np.array(Image.open(r['path']).convert('RGB')).copy()).permute(2,0,1),torch.from_numpy(targets),torch.from_numpy(mask)))
    return result

def batch(items,ids):
    selected=[items[i] for i in ids]
    return dict(img=torch.stack([v[0] for v in selected]).cuda().float()/255,
                bboxes=torch.cat([v[1] for v in selected]).cuda(),
                cls=torch.zeros(sum(len(v[1]) for v in selected),1,device='cuda'),
                batch_idx=torch.cat([torch.full((len(v[1]),),i) for i,v in enumerate(selected)]).cuda(),
                ignore=torch.stack([v[2] for v in selected]).cuda())

def make_model():
    assert sha(PRETRAIN)==PRETRAIN_SHA
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    pretrained=torch.load(PRETRAIN,map_location='cpu',weights_only=False)['model'].float()
    model=DetectionModel('yolo11n.yaml',nc=1,verbose=False)
    model.load(pretrained,verbose=False)
    model=model.cuda()
    model.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
    model.criterion=model.init_criterion();model.criterion.bce=IgnoredBCE()
    return model

def loss(model,items):
    model.criterion.bce.mask=items['ignore']
    total,components=model(items)
    return total.sum()/len(items['img']),components

def freeze_bn(model):
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

@torch.no_grad()
def detect(model,page):
    image=Image.open(page['image']).convert('RGB');all_boxes=[];all_scores=[]
    for window in tile_windows(image.width,image.height):
        canvas,_,_,(sx,sy,ox,oy)=resize_packet(image.crop(window),[],[])
        inputs=torch.from_numpy(np.array(canvas)).permute(2,0,1).unsqueeze(0).cuda().float()/255
        prediction=model(inputs)[0]
        out=non_max_suppression(prediction,conf_thres=.05,iou_thres=.5,nc=1,max_det=100)[0]
        boxes=out[:,:4].clone();boxes[:,[0,2]]=(boxes[:,[0,2]]-ox)/sx+window[0];boxes[:,[1,3]]=(boxes[:,[1,3]]-oy)/sy+window[1]
        boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,image.width);boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,image.height)
        valid=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]);all_boxes.append(boxes[valid]);all_scores.append(out[valid,4])
    from torchvision.ops import nms
    boxes=torch.cat(all_boxes);scores=torch.cat(all_scores);keep=nms(boxes,scores,.5)[:100];result=[]
    for idx,(box,score) in enumerate(zip(boxes[keep].cpu().tolist(),scores[keep].cpu().tolist())):
        x1,y1,x2,y2=box;x1,y1=math.floor(x1),math.floor(y1);x2,y2=math.ceil(x2),math.ceil(y2)
        result.append(dict(bbox=[x1,y1,x2-x1,y2-y1],proposal_index=idx,similarity=score,prediction=-1,detector_confidence=score))
    return result

def summarize_geometry(pages,threshold):
    scores=[f.score_page(p['gt'],p['ignore'],p['predictions'],threshold) for p in pages]
    counts={k:sum(s['counts'][k] for s in scores) for k in scores[0]['counts']}
    return dict(counts=counts,metrics=f.metrics(counts),page_mean_f1=float(np.mean([f.metrics(s['counts'])['detection_f1'] for s in scores])))

def verify(protocol):
    assert all(sha(p)==digest for p,digest in protocol['input_sha256'].items()),'Frozen source or data changed'

def run():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    protocol=read(RUN/'final-preregistration.json');verify(protocol);data=read(RUN/'dataset.json')
    nested=read(NESTED);gtmap={p['group']:p['gt'] for p in nested['outer_pages'] if p['seed']==42}
    results=[];started=time.monotonic()
    for sp in data['splits']:
        gtmap={p['group']:[dict(g,natural_support=sp['support_counts'][str(g['label'])]) for g in p['gt']] for p in nested['outer_pages'] if p['seed']==42}
        for arm in ['real','real_plus_synthetic']:
            model=make_model();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            anchors=anchors_for(model);real=packets(sp['real'],anchors);synthetic=packets(sp['synth'],anchors) if arm!='real' else []
            opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0005)
            history=[];fit_start=time.monotonic();freeze_bn(model)
            for step,(ids,synids) in enumerate(zip(sp['real_plan'],sp['synth_plan']),1):
                lr=(.00005+(.0005-.00005)*step/20) if step<=20 else .00005+.5*(.0005-.00005)*(1+math.cos(math.pi*(step-20)/(STEPS-20)))
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad();r_loss,r_components=loss(model,batch(real,ids));r_loss.backward();s_loss=None
                if synthetic:
                    s_loss,s_components=loss(model,batch(synthetic,synids));(.5*s_loss).backward()
                assert torch.isfinite(r_loss) and (s_loss is None or torch.isfinite(s_loss))
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);opt.step()
                history.append(dict(step=step,lr=lr,real_loss=float(r_loss.detach()),synthetic_loss=None if s_loss is None else float(s_loss.detach()),real_components=r_components.detach().cpu().tolist(),grad_norm=float(grad)))
                if step%50==0:print('YOLO_FIT',sp['fold'],arm,step,'loss',history[-1]['real_loss'],flush=True)
            cp=RUN/f'{arm}_fold{sp["fold"]}_seed42.pt';torch.save(dict(model=model.state_dict(),fold=sp['fold'],arm=arm,steps=STEPS,real_plan=sp['real_plan']),cp)
            changed=sum(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items());assert changed>0
            fit_seconds=time.monotonic()-fit_start;model.eval();cal=[]
            for p in data['pages']:
                if p['group'] in sp['calibration']:cal.append(dict(group=p['group'],gt=gtmap[p['group']],ignore=p['ignore'],predictions=detect(model,p)))
            curve=[dict(threshold=t,**summarize_geometry(cal,t)) for t in GRID]
            best=max(z['page_mean_f1'] for z in curve);chosen=min((z for z in curve if z['page_mean_f1']>=best-1e-12),key=lambda z:z['threshold'])
            selection=dict(threshold=chosen['threshold'],curve=curve,objective=best)
            write(RUN/f'selection_{arm}_fold{sp["fold"]}.json',selection)
            outer=[]
            for p in data['pages']:
                if p['group'] in sp['held']:
                    predictions=detect(model,p);outer.append(dict(group=p['group'],image=p['image'],gt=gtmap[p['group']],ignore=p['ignore'],predictions=predictions,geometry=f.score_page(gtmap[p['group']],p['ignore'],predictions,chosen['threshold'])))
            initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
            if arm!='real':assert initial_hash==next(z['initial_state_sha256'] for z in results if z['fold']==sp['fold'] and z['arm']=='real')
            result=dict(fold=sp['fold'],arm=arm,initial_state_sha256=initial_hash,checkpoint=str(cp),checkpoint_sha256=sha(cp),fit_seconds=fit_seconds,real_image_exposures=2*STEPS,synthetic_image_exposures=2*STEPS if synthetic else 0,changed_tensors=changed,history=history,calibration_pages=cal,selection=selection,outer_pages=outer)
            results.append(result);write(RUN/'progress.json',dict(fits=results,activated=False));del model,opt,real,synthetic;torch.cuda.empty_cache()
            print('YOLO_OUTER',sp['fold'],arm,'threshold',chosen['threshold'],'matched',sum(p['geometry']['counts']['matched'] for p in outer),flush=True)
    verify(protocol)
    output=dict(protocol=protocol,fits=results,elapsed_seconds=time.monotonic()-started,activated=False,locked_test_scored=False,optimizer_updates=STEPS*len(results))
    write(RUN/'detection-results.json',output);print('YOLO_DETECTION_DONE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','prepare','run']);args=parser.parse_args()
    if args.phase=='check':print(json.dumps(check()))
    elif args.phase=='prepare':prepare()
    else:run()

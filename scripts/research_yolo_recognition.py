"""Frozen recognition on YOLO proposals; no classifier fitting or semantic selection."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scripts import research_synthetic_detector as s
from scripts.research_nested_branch_rejection import score as branch_score

ROOT=s.ROOT
RUN=s.RUN/'recognition'
RUNTIME=ROOT/'backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime'
PIN=ROOT/'backend/training_corpus/backbone-pins/dinov2-vits14-local.json'
SAM=Path('/home/sina/.cache/mobile_sam/mobile_sam.pt')
HEADS=ROOT/'backend/training_runs/20261004-nested-rejection-v1'


def aggregate(pages,key):
    counts={k:sum(p[key]['counts'][k] for p in pages) for k in pages[0][key]['counts']}
    return dict(counts=counts,metrics=s.f.metrics(counts),page_mean_exact_f1=float(np.mean([s.f.metrics(p[key]['counts'])['exact_f1'] for p in pages])))


def check():
    gt=[dict(bbox=[0,0,10,10],label=0,natural_support=2)]
    rows=[dict(bbox=[0,0,10,10],prediction=0,proposal_index=0,similarity=.4,fallback=True),dict(bbox=[30,0,10,10],prediction=0,proposal_index=1,similarity=.4,fallback=False)]
    z=branch_score(gt,[],rows,.7)['counts'];assert z['correct']==1 and z['accepted']==1 and z['rejected']==1
    return dict(branch_threshold_reused=True)


def prepare():
    assert not RUN.exists();RUN.mkdir()
    proto=s.read(s.RUN/'final-preregistration.json');s.verify(proto)
    paths=[Path(__file__),Path(s.__file__),s.RUN/'final-preregistration.json',s.RUN/'dataset.json',s.BRANCH,s.NESTED,PIN,SAM,ROOT/'scripts/research_nested_branch_rejection.py',ROOT/'backend/codex_model/classifier.py',ROOT/'backend/codex_pipeline/models/projection_head.py',ROOT/'backend/codex_pipeline/segmentation/mobilesam.py']
    paths+=list(RUNTIME.glob('**/*'));paths=[p for p in paths if p.is_file()]
    paths+=list(HEADS.glob('paired_fold*_seed42_step1000.pt'))
    pin=s.read(PIN);paths.append(Path(pin['weights_path']))
    paths+=list(Path(pin['repository_path']).rglob('*.py'))
    import mobile_sam
    paths+=list(Path(mobile_sam.__file__).parent.rglob('*.py'))
    p=dict(schema_version='yolo-recognition.v1',input_sha256={str(p):s.sha(p) for p in sorted(set(paths))},checks=check(),seed=42,optimization=False,optimizer_updates=0,
           selection='None: both raw detector rectangles and detector-box-prompted SAM mask published, with semantic035 and frozen nested rare-branch cuts. Detector cuts fixed by own calibration pages before outer detection. No GT boxes or labels in crops, prompts or recognition.',
           inference='Original page, detector xyxy prompt to frozen MobileSAM, multimask_output=False, batch8, padding5; DINOv2 local pin, classifier native Lanczos224/white preprocessing, features batch16. Historical and own-fold paired projections with own-fit unique-pixel support rare gating.',
           scope='17 previously examined pages,343 known objects,7 ignore regions. Compare same seed42 existing SAM and own nested heads. No dev/test readout, no activation; one seed and fixed short detector recipe, extra synthetic compute explicit.',activated=False,locked_test_scored=False)
    s.write(RUN/'preregistration.json',p);print('RECOGNITION_FROZEN',len(paths),flush=True)


@torch.no_grad()
def features(classifier,predictor,page,rows):
    pixels=np.array(Image.open(page['image']).convert('RGB'));h,w=pixels.shape[:2]
    out={'raw':[],'mask':[]};masks=[];crop_hashes=[]
    if rows:
        predictor.set_image(pixels)
        for start in range(0,len(rows),8):
            boxes=torch.tensor([s.xyxy(r['bbox']) for r in rows[start:start+8]],device='cuda',dtype=torch.float32)
            boxes=predictor.transform.apply_boxes_torch(boxes,(h,w))
            values,_,_=predictor.predict_torch(point_coords=None,point_labels=None,boxes=boxes,multimask_output=False)
            masks.extend(values[:,0].cpu().numpy())
    raw_crops=[];mask_crops=[]
    for row,mask in zip(rows,masks):
        x,y,bw,bh=row['bbox'];a,b,c,d=max(0,x-5),max(0,y-5),min(w,x+bw+5),min(h,y+bh+5)
        raw=pixels[b:d,a:c].copy();masked=raw.copy();masked[~mask[b:d,a:c]]=255
        raw_crops.append(Image.fromarray(raw));mask_crops.append(Image.fromarray(masked))
        crop_hashes.append(dict(raw=s.hashlib.sha256(raw.tobytes()).hexdigest(),mask=s.hashlib.sha256(masked.tobytes()).hexdigest()))
    for name,crops in [('raw',raw_crops),('mask',mask_crops)]:
        for start in range(0,len(crops),16):
            inputs=torch.cat([s._preprocess_image(im,224) for im in crops[start:start+16]]).cuda()
            out[name].append(classifier._backbone(inputs).cpu())
        out[name]=torch.cat(out[name]) if out[name] else torch.empty(0,384)
    return out,crop_hashes


def run():
    torch.set_num_threads(4);protocol=s.read(RUN/'preregistration.json');s.verify(protocol)
    det=s.read(s.RUN/'detection-results.json');assert len(det['fits'])==10 and det['optimizer_updates']==2000
    s.verify(det['protocol']);s.write(RUN/'detector-input.json',dict(path=str(s.RUN/'detection-results.json'),sha256=s.sha(s.RUN/'detection-results.json'),checkpoints={p['checkpoint']:p['checkpoint_sha256'] for p in det['fits']}))
    assert all(s.sha(p['checkpoint'])==p['checkpoint_sha256'] for p in det['fits'])
    classifier=s.CodexClassifier(RUNTIME,device='cuda',backbone_source=PIN)
    predictor=s.MobileSAMSegmenter(checkpoint_path=str(SAM),device='cuda',points_per_side=16)._mask_generator.predictor
    branch=s.read(s.BRANCH);data=s.read(s.RUN/'dataset.json');splits={p['fold']:p for p in data['splits']}
    results=[];cache=[];started=time.monotonic()
    for fit in det['fits']:
        fold=fit['fold'];sp=splits[fold]
        cp=torch.load(HEADS/f'paired_fold{fold}_seed42_step1000.pt',map_location='cpu',weights_only=True)
        model=s.ProjectionHead().cuda();model.load_state_dict(cp['projection']);model.eval();prototypes=cp['prototypes'].cuda()
        threshold=next(z['selection']['threshold'] for z in branch['folds'] if z['seed']==42 and z['fold']==fold)
        pages=[]
        for p in fit['outer_pages']:
            rows=[r for r in p['predictions'] if r['detector_confidence']>=fit['selection']['threshold']]
            feats,hashes=features(classifier,predictor,p,rows)
            page=dict(group=p['group'],image=p['image'],fold=fold,arm=fit['arm'],gt=p['gt'],ignore=p['ignore'],detector_generated=len(p['predictions']),detector_accepted=len(rows),detector_threshold=fit['selection']['threshold'],semantic_branch_threshold=threshold,views={})
            cache.append(dict(fold=fold,arm=fit['arm'],group=p['group'],proposal_indices=[r['proposal_index'] for r in rows],features=feats,crop_sha256=hashes))
            for view,ff in feats.items():
                with torch.no_grad():
                    old=classifier._projection(ff.cuda())@prototypes.T;new=model(ff.cuda())@prototypes.T
                    oldlabel=old.argmax(1);flag=torch.tensor([1<=sp['support_counts'][str(c)]<=4 for c in oldlabel.tolist()],device='cuda',dtype=torch.bool)
                    sc=torch.where(flag[:,None],old,new);values,labels=sc.max(1)
                predictions=[dict(r,similarity=float(values[i]),prediction=int(labels[i]),fallback=bool(flag[i]),crop_sha256=hashes[i][view]) for i,r in enumerate(rows)]
                e035=s.f.score_page(p['gt'],p['ignore'],predictions,.35);ebranch=branch_score(p['gt'],p['ignore'],predictions,threshold)
                page['views'][view]=dict(predictions=predictions,current_035=e035,rare_branch=ebranch)
            pages.append(page)
            print('YOLO_NAMES',fold,fit['arm'],p['group'][-10:],'proposals',len(rows),'correct raw/mask',[page['views'][v]['rare_branch']['counts']['correct'] for v in ['raw','mask']],flush=True)
        result=dict(fold=fold,arm=fit['arm'],pages=pages);results.append(result);s.write(RUN/'progress.json',dict(fits=results,activated=False))
    torch.save(dict(pages=cache),RUN/'features.pt')
    geometry={};recognition={}
    for arm in ['real','real_plus_synthetic']:
        fs=[p for p in det['fits'] if p['arm']==arm];ps=[p for z in results if z['arm']==arm for p in z['pages']];assert len(ps)==17
        geometry[arm]=aggregate([p for z in fs for p in z['outer_pages']],'geometry')
        recognition[arm]={v:{policy:aggregate([dict(p[policy])|{} for p in []],policy) for policy in []} for v in []}
        recognition[arm]={v:{policy:aggregate([p['views'][v] for p in ps],policy) for policy in ['current_035','rare_branch']} for v in ['raw','mask']}
    nested=s.read(s.NESTED);base035=nested['aggregate']['gated']['current_035']['seeds']['42'];basebranch=branch['aggregate']['rare_branch']['seeds']['42']
    prior=[p for p in nested['outer_pages'] if p['seed']==42]
    samgeom=aggregate([dict(geometry=s.f.score_page(p['gt'],p['ignore'],p['methods']['gated']['predictions'],None)) for p in prior],'geometry')
    s.verify(protocol);s.verify(det['protocol'])
    output=dict(schema_version='yolo-readout-summary.v1',protocol=protocol,detector_protocol_sha256=s.sha(s.RUN/'final-preregistration.json'),detection_input_sha256=s.sha(s.RUN/'detection-results.json'),fits=results,geometry=geometry,recognition=recognition,reference=dict(sam_geometry=samgeom,sam_gated_035=base035,sam_branch=basebranch),feature_sha256=s.sha(RUN/'features.pt'),elapsed_seconds=time.monotonic()-started,activated=False,locked_test_scored=False,recognition_optimizer_updates=0,detector_optimizer_updates=2000)
    s.write(RUN/'results.json',output);s.write(ROOT/'reports/yolo-synthetic-results-20261004.json',output)
    print(json.dumps(dict(geometry=geometry,recognition=recognition),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','prepare','run']);phase=parser.parse_args().phase
    if phase=='check':print(check())
    elif phase=='prepare':prepare()
    else:run()

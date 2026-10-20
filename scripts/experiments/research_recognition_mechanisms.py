#!/usr/bin/env python3
"""Paired mechanism experiments: frozen crops, rare supports, CE + cross-page contrast."""
import argparse,collections,csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts.experiments import research_recognition_stability as core
from scripts.train_contrastive_recognition import positive_mask
from scripts.train_segmentation import read_json,write_json,sha256
from codex_model.classifier import CodexClassifier,_preprocess_image
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter
RUN=core.ROOT/'backend/training_runs/20261003-recognition-mechanisms-v2'
SEEDS=[42,43,44]
SUPPORTS=[1,2,4]
METHODS={'ce_fixed':0.,'ce_crosspage_01':.1,'ce_crosspage_05':.5}
STEPS=300


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def contrastive(embedded,labels,groups,pixels,objects):
    pos=positive_mask(labels,groups,pixels)
    pos &= torch.tensor([[a!=b for b in objects] for a in objects],device=labels.device)
    active=pos.any(1)
    if not active.any():return embedded.sum()*0,pos
    # Same-class pairs lacking a distinct cross-page object are neither positive nor negative.
    denominator=pos | (labels[:,None]!=labels[None,:])
    logits=F.normalize(embedded,dim=1)@F.normalize(embedded,dim=1).T/.1
    logp=logits-torch.logsumexp(logits.masked_fill(~denominator,-torch.inf),dim=1,keepdim=True)
    losses=-logp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp_min(1)
    return losses[active].mean(),pos


def limited_supports(labels,groups,pixels,ids,k,seed):
    pools=collections.defaultdict(lambda:collections.defaultdict(list));seen=set()
    for i in ids:
        identity=(int(labels[i]),pixels[i])
        if identity in seen:continue
        seen.add(identity);pools[int(labels[i])][groups[i]].append(i)
    rng=random.Random(seed);out=[]
    for label in sorted(pools):
        pages=list(sorted(pools[label]));rng.shuffle(pages)
        for page in pages:rng.shuffle(pools[label][page])
        picked=[]
        while len(picked)<k and any(pools[label].values()):
            for page in pages:
                if pools[label][page] and len(picked)<k:picked.append(pools[label][page].pop())
        out.extend(picked)
    assert len(out)==len(set(out)) and set(out)<=set(ids)
    assert max(collections.Counter(int(labels[i]) for i in out).values())<=k
    return out


def setup():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    d=core.load();d['teacher']=core.project(d['original'],d['features'])
    rows={r['row_id']:r for r in csv.DictReader((core.SNAPSHOT/'metadata.csv').open())}
    d['pixels']=[rows[r]['source_pixel_sha256'] for r in d['cache']['row_ids']]
    d['objects']=list(d['cache']['row_ids'])
    for r in d['rt']['rows']:
        d['pixels'].append(d['pixels'][r['snapshot_index']]);d['objects'].append(r['row_id'])
    return d


def register(d):
    RUN.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path(core.__file__),core.ROOT/'scripts/train_contrastive_recognition.py',
        core.ROOT/'backend/codex_model/classifier.py',core.ROOT/'backend/codex_pipeline/segmentation/mobilesam.py',
        core.ROOT/'backend/codex_pipeline/models/projection_head.py',core.ROOT/'backend/training_runs/20261003-bbox-detector-pilot/dataset.json']
    hashes=d['before'] | {str(p):sha256(p) for p in paths}
    write_json(RUN/'preregistration.json',dict(schema_version='recognition-mechanisms.v2',input_sha256=hashes,
        seeds=SEEDS,steps=STEPS,methods=METHODS,support_limits=SUPPORTS,support_bins='reference availability before cap and actual unique supports after cap are published separately',support_methods=['historical','centroid','shrink_tau8','learned_anchor'],
        crops=['mask','rectangle','dilated3','half_background'],sam_points_per_batch=4,sam_points_per_side=16,padding=5,
        crop_fixture='new saved RGB/mask fixture; compare old cache diagnostically, require saved-pixel DINO replay at1e-4',
        phases='crop inference with frozen head first; supports with frozen projection; CE/contrast comparison uses only new masked train views',
        locked_test='no scoring and no optimization; no locked-test images extracted',
        folds='same 5 training-page folds as v1; all held raw/masked objects excluded from optimization',
        selection='CV-only exploratory; page-mean joint recall improves, rare1-4 raw class macro does not regress, every fold mean raw top1 drops <=2pp; no activation',
        contrast='local CE + weighted cross-page SupCon adaptation, not exact Khosla recipe; same-class same-page pairs excluded from denominator',
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-4),temperature=.1,
        diagnostics='active anchors, per-class positive counts, intra/inter-page negatives, CE/contrast losses and gradient norms at steps1/100/300',
        provenance_limit='original historical projection training unknown; CV tests adaptation stability, not certified blind generalization',activated=False))


def verify(d):
    p=read_json(RUN/'preregistration.json')
    assert p['input_sha256']=={k:sha256(Path(k)) for k in p['input_sha256']}
    assert d['before']=={str(p):sha256(p) for p in d['paths']}


def render_views(rgb,mask):
    raw=np.array(rgb,copy=True);m=np.array(mask,dtype=bool)
    mask_img=raw.copy();mask_img[~m]=255
    expanded=F.max_pool2d(torch.from_numpy(m.astype(np.float32))[None,None],7,stride=1,padding=3)[0,0].numpy().astype(bool)
    dilated=raw.copy();dilated[~expanded]=255
    half=raw.copy();half[~m]=np.uint8((raw[~m].astype(np.float32)+255)/2)
    return dict(mask=mask_img,rectangle=raw,dilated3=dilated,half_background=half)


@torch.no_grad()
def encode(clf,images):
    out=[]
    for start in range(0,len(images),16):
        x=torch.cat([_preprocess_image(Image.fromarray(im),224) for im in images[start:start+16]]).cuda()
        out.append(clf._backbone(x).float().cpu())
    return torch.cat(out)


def crops(d):
    register(d);started=time.monotonic()
    dataset=read_json(core.ROOT/'backend/training_runs/20261003-bbox-detector-pilot/dataset.json')
    image_hashes={p['source_group']:p['image_sha256'] for p in dataset['pages']}
    s=MobileSAMSegmenter(checkpoint_path=str(core.SAM),device='cuda',points_per_side=16)
    s._mask_generator.points_per_batch=4
    clf=CodexClassifier(model_dir=core.RUNTIME,device='cuda',backbone_source=core.PIN)
    rows=[];rgb=[];masks=[];manifest=[];images={k:[] for k in ['mask','rectangle','dilated3','half_background']}
    for page in d['app']['full_pages']['pages']:
        if page['split']=='locked_test':continue
        path=Path(page['source_image_path']);assert sha256(path)==image_hashes[page['source_group']]
        pixels=np.array(Image.open(path).convert('RGB'))
        proposals=s.extract_crops(pixels,s.segment_page(pixels),padding=5)
        by_box={tuple(p.bbox):p for p in proposals}
        assert len(proposals)==page['proposal_count'] and len(by_box)==len(proposals)
        for j,r in enumerate(d['rt']['rows']):
            if r['source_group']!=page['source_group']:continue
            assert r['split']!='locked_test'
            p=by_box[tuple(r['proposal_bbox'])];x,y,w,h=p.bbox
            left,top=max(0,x-5),max(0,y-5);right,bottom=min(pixels.shape[1],x+w+5),min(pixels.shape[0],y+h+5)
            raw=pixels[top:bottom,left:right].copy();mask=p.mask[top:bottom,left:right].copy()
            variants=render_views(raw,mask);assert np.array_equal(variants['mask'],p.crop)
            rgb.append(torch.from_numpy(raw));masks.append(torch.from_numpy(mask));rows.append(dict(r,runtime_index=j))
            manifest.append(dict(row_id=r['row_id'],group=r['source_group'],bbox=r['proposal_bbox'],image_sha256=image_hashes[r['source_group']],
                crop_shape=list(raw.shape),rgb_sha256=digest(raw),mask_sha256=digest(mask),view_sha256={k:digest(v) for k,v in variants.items()}))
            for k,v in variants.items():images[k].append(v)
        print('CACHE PAGE',page['split'],page['source_group'][-8:],flush=True)
    assert len(rows)==275 and len({r['row_id'] for r in rows})==275
    features={k:encode(clf,v) for k,v in images.items()}
    torch.save(dict(rows=rows,rgb=rgb,masks=masks,features=features),RUN/'crop-fixture.pt')
    saved=torch.load(RUN/'crop-fixture.pt',weights_only=True,map_location='cpu')
    reconstructed=[render_views(a.numpy(),b.numpy())['mask'] for a,b in zip(saved['rgb'],saved['masks'])]
    replay=encode(clf,reconstructed);error=float((replay-features['mask']).abs().max())
    assert torch.allclose(replay,features['mask'],atol=1e-4,rtol=1e-4),error
    old=d['rt']['features'][[r['runtime_index'] for r in rows]]
    differences=(features['mask']-old).abs().max(1).values
    oldz=core.project(d['original'],old.cuda());freshz=core.project(d['original'],features['mask'].cuda())
    oldrank=(oldz@d['old'].T).argmax(1);freshrank=(freshz@d['old'].T).argmax(1)
    report=dict(views={},old_fixture_comparison=dict(max_feature_error=float(differences.max()),
        rows_above_1e4=int((differences>1e-4).sum()),top1_predictions_changed=int((oldrank!=freshrank).sum()),
        differing_rows=[dict(row_id=r['row_id'],group=r['source_group'],max_error=float(differences[i]),top1_changed=bool(oldrank[i]!=freshrank[i])) for i,r in enumerate(rows) if differences[i]>1e-4]),
        replay_max_error=error,selection='none',activated=False)
    labels=d['labels'][[d['n']+r['runtime_index'] for r in rows]]
    for k,fs in features.items():
        scores=core.project(d['original'],fs.cuda())@d['old'].T;entry=dict(splits={},pages=[])
        for split in ['train','dev']:
            ids=[i for i,r in enumerate(rows) if r['split']==split]
            m=core.metrics(scores[ids],labels[ids]);m['known']=d['app']['full_pages']['by_split'][split]['ground_truth_count'];entry['splits'][split]=m
        for p in d['app']['full_pages']['pages']:
            if p['split']=='locked_test':continue
            ids=[i for i,r in enumerate(rows) if r['source_group']==p['source_group']]
            entry['pages'].append(dict(group=p['source_group'],split=p['split'],known=p['ground_truth_count'],metrics=core.metrics(scores[ids],labels[ids])))
        report['views'][k]=entry
        print('VIEW',k,entry['splits'],flush=True)
    write_json(RUN/'crop-manifest.json',dict(rows=manifest,fixture_sha256=sha256(RUN/'crop-fixture.pt'),
        saved_pixel_replay_passed=True,replay_max_error=error,preprocessing=d['cache']['preprocessing'],classifier_sha256=sha256(core.ROOT/'backend/codex_model/classifier.py')))
    verify(d);report['verification']=dict(inputs_unchanged=True,active_model_unchanged=True,saved_pixel_replay_passed=True,elapsed_seconds=time.monotonic()-started)
    write_json(RUN/'crop-results.json',report)


def load_fixture(d):
    verify(d);m=read_json(RUN/'crop-manifest.json');assert sha256(RUN/'crop-fixture.pt')==m['fixture_sha256']
    f=torch.load(RUN/'crop-fixture.pt',weights_only=True,map_location='cpu')
    assert m['saved_pixel_replay_passed'] and len(f['rows'])==275
    for i,r in enumerate(f['rows']):
        assert r['split']!='locked_test' and r['row_id']==d['rt']['rows'][r['runtime_index']]['row_id']
        assert digest(f['rgb'][i].numpy())==m['rows'][i]['rgb_sha256'] and digest(f['masks'][i].numpy())==m['rows'][i]['mask_sha256']
        d['features'][d['n']+r['runtime_index']]=f['features']['mask'][i].cuda()
    d['teacher']=core.project(d['original'],d['features'])
    return f


def evaluate(d,model,proto,held,train_raw,effective_supports=None):
    raw=[i for i in range(d['n']) if d['splits'][i]=='train' and d['groups'][i] in held]
    masked=[d['n']+j for j,r in enumerate(d['rt']['rows']) if r['split']=='train' and r['source_group'] in held]
    assert not set(raw+masked)&set(train_raw)
    ids=raw+masked;scores=core.project(model,d['features'][ids])@proto.T
    y=d['labels'][ids];pred=scores.argmax(1)
    pools=collections.defaultdict(set)
    for i in train_raw:pools[int(d['labels'][i])].add(d['pixels'][i])
    counts={c:len(p) for c,p in pools.items()}
    output=dict(raw=core.metrics(scores[:len(raw)],y[:len(raw)]),masked=core.metrics(scores[len(raw):],y[len(raw):]),pages=[],reference_availability={},effective_support_bins={})
    for group in held:
        js=[j for j,i in enumerate(ids) if i>=d['n'] and d['groups'][i]==group]
        known=next(p['ground_truth_count'] for p in d['app']['full_pages']['pages'] if p['source_group']==group)
        output['pages'].append(dict(group=group,known=known,metrics=core.metrics(scores[js],y[js])))
    for name,lo,hi in [('zero',0,0),('rare1_4',1,4),('medium5_20',5,20),('many',21,100000)]:
        js=[j for j,i in enumerate(raw) if lo<=counts.get(int(d['labels'][i]),0)<=hi]
        output['reference_availability'][name]=core.metrics(scores[js],y[js])
    actual=collections.defaultdict(set)
    for i in (train_raw if effective_supports is None else effective_supports):actual[int(d['labels'][i])].add(d['pixels'][i])
    for name,lo,hi in [('zero',0,0),('one',1,1),('two',2,2),('three_four',3,4),('five_twenty',5,20),('many',21,100000)]:
        js=[j for j,i in enumerate(raw) if lo<=len(actual[int(d['labels'][i])])<=hi]
        output['effective_support_bins'][name]=core.metrics(scores[js],y[js])
    output['raw_predictions']=[dict(row_id=d['objects'][i],label=int(d['labels'][i]),prediction=int(pred[j])) for j,i in enumerate(raw)]
    return output


def folds():
    return [f['held_groups'] for f in read_json(core.RUN/'cv-results.json')['folds']]


def supports(d):
    load_fixture(d);torch.use_deterministic_algorithms(True);out=dict(folds=[],activated=False,locked_test_scored=False)
    for fold,held in enumerate(folds()):
        full_raw=[i for i in d['train'] if i<d['n'] and d['groups'][i] not in held]
        base=evaluate(d,d['original'],d['old'],held,full_raw)
        for seed in SEEDS:
            for k in SUPPORTS:
                selected=limited_supports(d['labels'],d['groups'],d['pixels'],full_raw,k,seed)
                assert all(d['splits'][i]=='train' and d['groups'][i] not in held for i in selected)
                center,groups=core.centroid(d['teacher'],d['labels'],selected,d['groups'],fallback=d['old'])
                learned_model,learned,history=core.fit(d,'prototypes_anchor',seed,selected,steps=STEPS)
                entry=dict(fold=fold,seed=seed,k=k,held=held,selected_raw_ids=selected,selected_classes=len({int(d['labels'][i]) for i in selected}),
                    baseline=base,candidates={},learned_history=history)
                for method,model,p in [('historical',d['original'],d['old']),('centroid',d['original'],center),
                    ('shrink_tau8',d['original'],core.shrink(d['old'],center,groups,8)),('learned_anchor',learned_model,learned)]:
                    entry['candidates'][method]=evaluate(d,model,p,held,full_raw,selected)
                out['folds'].append(entry);write_json(RUN/'supports-progress.json',out)
                print('SUPPORTS',fold,seed,k,{m:e['masked']['top1_correct'] for m,e in entry['candidates'].items()},flush=True)
    verify(d);write_json(RUN/'supports-results.json',out)


def grad_norm(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(g.square().sum() for g in gs if g is not None)))


def fit_contrast(d,weight,seed,train):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=core.ProjectionHead().cuda();model.load_state_dict(d['init']);model.train()
    params=list(model.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    plan=core.batches(d['labels'],train,d['n'],seed,STEPS);history=[];coverage=collections.defaultdict(collections.Counter)
    for step,ids in enumerate(plan,1):
        embedded=model(d['features'][ids]);ce=F.cross_entropy(embedded@d['old'].T/.1,d['labels'][ids])
        group=[d['groups'][i] for i in ids];pixel=[d['pixels'][i] for i in ids];objects=[d['objects'][i] for i in ids]
        con,pos=contrastive(embedded,d['labels'][ids],group,pixel,objects)
        nums=pos.sum(1).cpu().tolist();labels=d['labels'][ids].cpu().tolist()
        for label,n in zip(labels,nums):coverage[label][n]+=1
        neg=d['labels'][ids,None]!=d['labels'][ids][None,:]
        same_page=torch.tensor([[a==b for b in group] for a in group],device='cuda')
        entry=dict(step=step,ce=float(ce.detach()),contrast=float(con.detach()),weighted_contrast=float((weight*con).detach()),
            active_anchors=int(pos.any(1).sum()),anchors=len(ids),negatives_intrapage=int((neg&same_page).sum()),negatives_interpage=int((neg&~same_page).sum()))
        if step in [1,100,300]:
            entry['ce_gradient_norm']=grad_norm(ce,params)
            entry['contrast_gradient_norm']=grad_norm(con,params) if pos.any() else 0.
        loss=ce+weight*con;assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();history.append(entry)
    assert any(e['active_anchors'] for e in history)
    model.eval();return model,history,{str(k):dict(v) for k,v in coverage.items()}


def contrasts(d):
    load_fixture(d);torch.use_deterministic_algorithms(True);out=dict(folds=[],activated=False,locked_test_scored=False)
    for fold,held in enumerate(folds()):
        train=[i for i in d['train'] if d['groups'][i] not in held]
        raw=[i for i in train if i<d['n']]
        base=evaluate(d,d['original'],d['old'],held,raw)
        assert all(d['splits'][i]=='train' and d['groups'][i] not in held for i in train)
        for seed in SEEDS:
            entry=dict(fold=fold,seed=seed,held=held,baseline=base,methods={})
            for method,weight in METHODS.items():
                model,history,coverage=fit_contrast(d,weight,seed,train)
                torch.save(dict(projection=model.state_dict(),class_names=d['cache']['class_names'],prototypes=d['old'].cpu(),seed=seed,weight=weight,train_ids=train),RUN/f'{method}_fold{fold}_seed{seed}.pt')
                e=evaluate(d,model,d['old'],held,raw)
                entry['methods'][method]=dict(evaluation=e,history=history,positive_count_histograms=coverage)
                print('CONTRAST',fold,seed,method,'SAM',e['masked']['top1_correct'],'RAW',e['raw']['top1_correct'],flush=True)
            out['folds'].append(entry);write_json(RUN/'contrast-progress.json',out)
    verify(d);write_json(RUN/'contrast-results.json',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['crops','supports','contrastive'],required=True);args=parser.parse_args()
    d=setup()
    if args.phase=='crops':crops(d)
    else:
        assert not (RUN/(('supports' if args.phase=='supports' else 'contrast')+'-results.json')).exists()
        verify(d)
        if args.phase=='supports':supports(d)
        else:contrasts(d)
    verify(d);write_json(RUN/(args.phase+'-verification.json'),dict(inputs_unchanged=True,active_model_unchanged=True,activated=False,locked_test_scored=False))
    print('COMPLETE',args.phase,flush=True)


if __name__=='__main__':main()

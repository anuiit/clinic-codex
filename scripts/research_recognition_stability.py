#!/usr/bin/env python3
"""Bounded recognition experiments on frozen DINO/SAM; no active-model writes."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backend'), str(ROOT/'scripts')]
from codex_pipeline.models.projection_head import ProjectionHead
from train_contrastive_recognition import project, metrics, CACHE, RUNTIME, SNAPSHOT, APPLICATION, SAM, PIN
from train_segmentation import read_json, write_json, sha256
SEEDS = [42,43,44]
METHODS = ['ce_fixed', 'embedding_anchor', 'logit_distill', 'learned_head_distill', 'prototypes_anchor']
STEPS = 300
RUN = ROOT/'backend/training_runs/20261003-recognition-stability-v1'
PREVIOUS = ROOT/'backend/training_runs/20261003-contrastive-recognition-pilot-distinct'


def aligned_prototypes(package, names):
    by_name = {package['class_names'][int(label)]: package['prototypes'][i]
               for i,label in enumerate(package['class_labels'])}
    if set(by_name) != set(names.values()): raise ValueError('taxonomy mismatch')
    return F.normalize(torch.stack([by_name[names[i]] for i in range(len(names))]),dim=1)


def centroid(z, labels, indices, groups=None, fallback=None):
    out=[]; counts=[]; host_labels=labels.detach().cpu().tolist()
    for c in range(286):
        ids=[i for i in indices if host_labels[i]==c]
        if not ids:
            if fallback is None: raise ValueError(f'missing class {c}')
            out.append(fallback[c]);counts.append(0);continue
        if groups is None: out.append(z[ids].mean(0));counts.append(len(ids))
        else:
            pools=collections.defaultdict(list)
            for i in ids:pools[groups[i]].append(i)
            out.append(torch.stack([z[rows].mean(0) for rows in pools.values()]).mean(0));counts.append(len(pools))
    return F.normalize(torch.stack(out),dim=1),torch.tensor(counts,device=z.device)


def shrink(old,new,counts,tau=8):
    alpha=counts.float()/(counts.float()+tau)
    return F.normalize((1-alpha[:,None])*old+alpha[:,None]*new,dim=1)


def batches(labels, train_ids, raw_size, seed, steps=STEPS):
    pools=[collections.defaultdict(list),collections.defaultdict(list)]
    host_labels=labels.detach().cpu().tolist()
    for i in train_ids:pools[int(i>=raw_size)][host_labels[i]].append(i)
    rng=random.Random(seed);result=[]
    for _ in range(steps):
        ids=[]
        for pool,nclasses,nrows in [(pools[0],64,2),(pools[1],16,4)]:
            for c in rng.sample(sorted(pool),min(nclasses,len(pool))):
                ids.extend(rng.sample(pool[c],min(nrows,len(pool[c]))))
        assert len(ids)==len(set(ids)) and set(ids)<=set(train_ids)
        result.append(ids)
    return result


def load():
    paths=[CACHE,CACHE.with_suffix('.pt.prov.json'),SNAPSHOT/'metadata.csv',SNAPSHOT/'snapshot_manifest.json',
           APPLICATION,PREVIOUS/'runtime_features.pt',PREVIOUS/'preregistration.json',RUNTIME/'weights/projection.pt',
           RUNTIME/'weights/prototypes.pt',RUNTIME/'config.json',PIN,SAM,
           ROOT/'backend/codex_model/weights/projection.pt',ROOT/'backend/codex_model/weights/prototypes.pt',ROOT/'backend/codex_model/config.json']
    before={str(p):sha256(p) for p in paths}
    prov=read_json(CACHE.with_suffix('.pt.prov.json'))
    assert prov['metadata_csv_sha256']==before[str(SNAPSHOT/'metadata.csv')]
    assert prov['snapshot']['snapshot_manifest_sha256']==before[str(SNAPSHOT/'snapshot_manifest.json')]
    previous=read_json(PREVIOUS/'preregistration.json')
    for p in [SAM,PIN,RUNTIME/'weights/projection.pt',RUNTIME/'weights/prototypes.pt',APPLICATION]:
        assert previous['input_sha256'][str(p)]==before[str(p)]
    c=torch.load(CACHE,weights_only=True,map_location='cpu')
    rt=torch.load(PREVIOUS/'runtime_features.pt',weights_only=True,map_location='cpu')
    meta={r['row_id']:r for r in csv.DictReader((SNAPSHOT/'metadata.csv').open())}
    assert len(meta)==len(c['features'])==9716
    gs=collections.defaultdict(set)
    for i,row_id in enumerate(c['row_ids']):
        r=meta[row_id];assert r['source_group']==c['source_groups'][i] and r['dataset_split']==c['dataset_splits'][i]
        assert int(r['class_label'])==int(c['labels'][i]);gs[r['source_group']].add(r['dataset_split'])
    assert all(len(s)==1 for s in gs.values())
    app=read_json(APPLICATION); expected={r['row_id']:(p,r) for p in app['full_pages']['pages'] for r in p['matched_rows']}
    assert len(rt['rows'])==len(rt['features'])==326
    for r in rt['rows']:
        i=r['snapshot_index'];p,q=expected[r['row_id']]
        assert c['row_ids'][i]==r['row_id'] and c['dataset_splits'][i]==r['split']==p['split']
        assert r['proposal_bbox']==q['proposal_bbox'] and c['source_groups'][i]==r['source_group']==p['source_group']
        assert c['class_names'][int(c['labels'][i])]==r['class_name']
    assert len({r['row_id'] for r in rt['rows']})==326
    n=len(c['features']);raw=c['features'].float().cuda();masked=rt['features'].float().cuda()
    features=torch.cat([raw,masked]);labels=torch.cat([c['labels'].long(),c['labels'][[r['snapshot_index'] for r in rt['rows']]].long()]).cuda()
    groups=c['source_groups']+[r['source_group'] for r in rt['rows']]
    splits=c['dataset_splits']+[r['split'] for r in rt['rows']]
    init=torch.load(RUNTIME/'weights/projection.pt',weights_only=True,map_location='cpu')
    old=aligned_prototypes(torch.load(RUNTIME/'weights/prototypes.pt',weights_only=True,map_location='cpu'),c['class_names']).cuda()
    original=ProjectionHead().cuda();original.load_state_dict(init)
    train=[i for i,s in enumerate(splits) if s=='train'];assert len(train)==9725
    assert len({int(labels[i]) for i in train})==286
    business={p['source_group'] for p in app['full_pages']['pages']}
    return dict(cache=c,rt=rt,app=app,features=features,labels=labels,groups=groups,splits=splits,n=n,init=init,
                old=old,original=original,train=train,business=business,before=before,paths=paths)


def evaluate(d,scores,scope='all'):
    labels=d['labels'];n=d['n'];out={'raw':{},'application':{},'pages':[]}
    splits=['dev'] if scope=='dev' else ['train','dev','locked_test']
    for split in splits:
        raw_ids=[i for i,s in enumerate(d['splits'][:n]) if s==split]
        out['raw'][split]=metrics(scores[raw_ids],labels[raw_ids])
        rows=[(n+j,r) for j,r in enumerate(d['rt']['rows']) if r['split']==split];ids=[i for i,r in rows]
        known=d['app']['full_pages']['by_split'][split]['ground_truth_count']
        m=metrics(scores[ids],labels[ids]);m.update(known_boxes=known,missed_known_boxes=known-len(ids));out['application'][split]=m
    for p in d['app']['full_pages']['pages']:
        if p['split'] not in splits:continue
        ids=[n+j for j,r in enumerate(d['rt']['rows']) if r['source_group']==p['source_group']]
        out['pages'].append(dict(group=p['source_group'],split=p['split'],known_boxes=p['ground_truth_count'],metrics=metrics(scores[ids],labels[ids])))
    if scope!='dev':
        ids=[i for i,s in enumerate(d['splits'][:n]) if s=='locked_test' and d['groups'][i]=='page:05:13:02']
        out['legacy_chimalli']=metrics(scores[ids],labels[ids])
        ids=[i for i,s in enumerate(d['splits'][:n]) if s=='locked_test'];ranking=scores[ids].argmax(1)
        out['raw_test_classes']=[dict(name=d['cache']['class_names'][int(c)],count=int((labels[ids]==c).sum()),correct=int(((labels[ids]==c)&(ranking==c)).sum())) for c in labels[ids].unique()]
    return out


def fit(d,method,seed,train_ids,steps=STEPS):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ProjectionHead().cuda();model.load_state_dict(d['init'])
    weights=torch.nn.Parameter(d['old'].clone(),requires_grad=method in ['learned_head_distill','prototypes_anchor'])
    if method=='prototypes_anchor':
        model.eval();params=[weights]
    else:model.train();params=list(model.parameters())+([weights] if weights.requires_grad else [])
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    plan=batches(d['labels'],train_ids,d['n'],seed,steps)
    history=[];seen=set();host_labels=d['labels'].detach().cpu().tolist()
    teacher=d['teacher']
    for step,ids in enumerate(plan,1):
        seen.update(host_labels[i] for i in ids)
        if method=='prototypes_anchor':embedded=teacher[ids]
        else:embedded=model(d['features'][ids])
        proto=F.normalize(weights,dim=1);scores=embedded@proto.T
        ce=F.cross_entropy(scores/.1,d['labels'][ids]);loss=ce
        replay=torch.tensor([i<d['n'] and d['groups'][i] not in d['business'] for i in ids],device='cuda')
        if method=='embedding_anchor' and replay.any():loss=loss+5*(1-(embedded[replay]*teacher[ids][replay]).sum(1)).mean()
        if method in ['logit_distill','learned_head_distill'] and replay.any():
            target=teacher[ids][replay]@d['old'].T
            loss=loss+4*F.kl_div(F.log_softmax(scores[replay]/.2,dim=1),F.softmax(target/.2,dim=1),reduction='batchmean')
        if method=='prototypes_anchor':loss=loss+(1-(proto*d['old']).sum(1)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        history.append(float(loss.detach()))
        if step%100==0:print('FIT',method,seed,step,round(history[-1],4),flush=True)
    assert seen=={host_labels[i] for i in train_ids}
    model.eval();return model,F.normalize(weights.detach(),dim=1),history


def crop_views(d):
    from codex_model.classifier import CodexClassifier,_preprocess_image
    from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter
    classifier=CodexClassifier(model_dir=RUNTIME,device='cuda',backbone_source=PIN)
    segmenter=MobileSAMSegmenter(checkpoint_path=str(SAM),device='cuda',points_per_side=16)
    by_group=collections.defaultdict(list)
    for j,r in enumerate(d['rt']['rows']):by_group[r['source_group']].append((j,r))
    variants={k:torch.empty_like(d['rt']['features']) for k in ['masked_replay','rectangle_proposal','dilated_mask3','half_background']}
    page_dataset=read_json(ROOT/'backend/training_runs/20261003-bbox-detector-pilot/dataset.json')
    digests={p['source_group']:p['image_sha256'] for p in page_dataset['pages']}
    for num,page in enumerate(d['app']['full_pages']['pages'],1):
        path=Path(page['source_image_path']);assert sha256(path)==digests[page['source_group']]
        image=np.array(Image.open(path).convert('RGB'))
        proposals=segmenter.extract_crops(image,segmenter.segment_page(image),padding=5)
        assert len(proposals)==page['proposal_count']
        boxes={tuple(p.bbox):p for p in proposals};jobs={k:[] for k in variants};indices=[]
        for j,r in by_group[page['source_group']]:
            p=boxes[tuple(r['proposal_bbox'])];x,y,w,h=p.bbox
            left,top=max(0,x-5),max(0,y-5);right,bottom=min(image.shape[1],x+w+5),min(image.shape[0],y+h+5)
            raw=image[top:bottom,left:right].copy();mask=p.mask[top:bottom,left:right]
            expanded=F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None,None],7,stride=1,padding=3)[0,0].numpy().astype(bool)
            expanded_crop=raw.copy();expanded_crop[~expanded]=255
            half=raw.copy();half[~mask]=np.uint8((raw[~mask].astype(np.float32)+255)/2)
            for k,img in [('masked_replay',p.crop),('rectangle_proposal',raw),('dilated_mask3',expanded_crop),('half_background',half)]:jobs[k].append(Image.fromarray(img))
            indices.append(j)
        for k,images in jobs.items():
            for start in range(0,len(images),16):
                batch=torch.cat([_preprocess_image(img,224) for img in images[start:start+16]]).cuda()
                with torch.no_grad():features=classifier._backbone(batch).float().cpu()
                variants[k][indices[start:start+16]]=features
        print('CROPS',num,'/',len(d['app']['full_pages']['pages']),len(indices),flush=True)
    max_error=float((variants['masked_replay']-d['rt']['features']).abs().max())
    assert torch.allclose(variants['masked_replay'],d['rt']['features'],atol=1e-4,rtol=1e-4),max_error
    torch.save(dict(variants=variants,rows=d['rt']['rows'],masked_replay_max_error=max_error),RUN/'crop_features.pt')
    del classifier,segmenter;torch.cuda.empty_cache();return variants,max_error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['fit','crops','cv'],required=True);args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    d=load();d['teacher']=project(d['original'],d['features']);start=time.monotonic()
    if args.phase=='fit':
        RUN.mkdir(parents=True,exist_ok=False)
        protocol=dict(schema_version='recognition-stability.v1',source_sha256=sha256(Path(__file__)),input_sha256=d['before'],seeds=SEEDS,methods=METHODS,steps=STEPS,
          frozen=['DINOv2','MobileSAM'],train_rows_raw=9471,train_rows_masked=254,train_classes=286,
          sampling='64 raw classes up to 2 distinct rows +16 SAM classes up to 4 distinct rows per batch; paired identical batches by seed',
          input_views='keep all raw train examples, add available SAM views of the same training examples; no test/dev optimization',
          losses={'ce_temperature':.1,'distillation_temperature':.2,'distillation_multiplier':4,'embedding_cosine_multiplier':5,'prototype_cosine_multiplier':1},
          optimizer={'name':'AdamW','lr':1e-4,'weight_decay':1e-4},
          readouts=['stored','raw_centroid','sam_replaced_centroid','dual_group_centroid','group_shrinkage_tau8','multi_old_raw_sam'],
          crop_views=['mask','rectangle same proposal','mask dilated 3px','background whitened 50%'],
          cv='5 folds over the 17 training business page groups, seed42, all 5 fit methods; no dev or locked-test rows in training or CV',
          selection='fixed final step; rank candidates by dev only, require raw dev >=59 and SAM dev >=8, then average raw/SAM macro top1; locked test is diagnostics only',
          locked_test='Previously examined, exploratory; original historical training provenance unknown; no fresh blind evidence',activated=False)
        write_json(RUN/'preregistration.json',protocol)
        z=d['teacher'];raw_train=[i for i in d['train'] if i<d['n']];sam_train=[i for i in d['train'] if i>=d['n']]
        replaced=[i for i in raw_train if d['groups'][i] not in d['business']]+sam_train
        raw_p,_=centroid(z,d['labels'],raw_train)
        replace_p,_=centroid(z,d['labels'],replaced)
        dual_p,counts=centroid(z,d['labels'],d['train'],d['groups'])
        sam_p,sam_counts=centroid(z,d['labels'],sam_train,fallback=d['old'])
        readouts={'stored':d['old'],'raw_centroid':raw_p,'sam_replaced_centroid':replace_p,'dual_group_centroid':dual_p,'group_shrinkage_tau8':shrink(d['old'],dual_p,counts)}
        report=dict(schema_version='recognition-stability-results.v1',protocol=protocol,readouts={},training={},activated=False)
        for name,proto in readouts.items():
            e=evaluate(d,z@proto.T);report['readouts'][name]=e
            print('READOUT',name,'SAM',e['application']['locked_test']['top1_correct'],'RAW',e['raw']['locked_test']['top1_correct'],flush=True)
        multi=torch.stack([z@d['old'].T,z@raw_p.T,z@sam_p.T]).max(0).values
        report['readouts']['multi_old_raw_sam']=evaluate(d,multi)
        baseline=report['readouts']['stored'];assert baseline['raw']['locked_test']['top1_correct']==97 and baseline['application']['locked_test']['top1_correct']==26
        chi=next(i for i,n in d['cache']['class_names'].items() if n=='chimalli')
        report['prototype_audit']={'chimalli_train_rows':int((d['labels'][raw_train]==chi).sum()),'chimalli_train_groups':int(counts[chi]),'old_to_raw_cosine':float((d['old'][chi]*raw_p[chi]).sum()),'old_to_dual_cosine':float((d['old'][chi]*dual_p[chi]).sum())}
        # Fit all fixed candidates before evaluating any locked-test candidate.
        for seed in SEEDS:
            for method in METHODS:
                model,proto,history=fit(d,method,seed,d['train'])
                key=f'{method}_seed_{seed}'
                torch.save(dict(projection=model.state_dict(),prototypes=proto.cpu(),class_names=d['cache']['class_names'],train_indices=d['train'],method=method,seed=seed,history=history),RUN/(key+'.pt'))
                dev=evaluate(d,project(model,d['features'])@proto.T,scope='dev')
                report['training'][key]={'dev_before_test':dev,'history':history}
                write_json(RUN/'progress.json',report)
        dev_means={}
        for method in METHODS:
            items=[report['training'][f'{method}_seed_{s}']['dev_before_test'] for s in SEEDS]
            raw=np.mean([e['raw']['dev']['top1_correct'] for e in items]);sam=np.mean([e['application']['dev']['top1_correct'] for e in items])
            rank=np.mean([(e['raw']['dev']['top1_macro']+e['application']['dev']['top1_macro'])/2 for e in items])
            dev_means[method]=dict(raw_correct=float(raw),sam_correct=float(sam),macro_rank=float(rank),eligible=bool(raw>=59 and sam>=8))
        eligible=[m for m in METHODS if dev_means[m]['eligible']]
        report['dev_selection']=dict(methods=dev_means,selected=max(eligible,key=lambda m:dev_means[m]['macro_rank']) if eligible else None,locked_before_test=True)
        write_json(RUN/'dev_selection.json',report['dev_selection'])
        for key in report['training']:
            ck=torch.load(RUN/(key+'.pt'),weights_only=True,map_location='cpu');model=ProjectionHead().cuda();model.load_state_dict(ck['projection']);proto=ck['prototypes'].cuda()
            report['training'][key]['evaluation']=evaluate(d,project(model,d['features'])@proto.T)
            e=report['training'][key]['evaluation'];print('RESULT',key,'SAM',e['application']['locked_test']['top1_correct'],'RAW',e['raw']['locked_test']['top1_correct'],'DEV',e['application']['dev']['top1_correct'],e['raw']['dev']['top1_correct'],flush=True)
        report['elapsed_seconds']=time.monotonic()-start;write_json(RUN/'results.json',report)
    elif args.phase=='crops':
        assert sha256(Path(__file__))==read_json(RUN/'preregistration.json')['source_sha256']
        variants,error=crop_views(d);report={'views':{},'masked_replay_max_error':error}
        for name,features in variants.items():
            z=d['teacher'].clone();z[d['n']:]=project(d['original'],features.cuda());e=evaluate(d,z@d['old'].T)
            report['views'][name]=e;print('VIEW',name,'SAM',e['application']['locked_test']['top1_correct'],'DEV',e['application']['dev']['top1_correct'],flush=True)
        write_json(RUN/'crop-results.json',report)
    else:
        assert sha256(Path(__file__))==read_json(RUN/'preregistration.json')['source_sha256']
        pages=sorted(p['source_group'] for p in d['app']['full_pages']['pages'] if p['split']=='train');assert len(pages)==17
        random.Random(42).shuffle(pages);folds=[pages[i::5] for i in range(5)]
        report={'schema_version':'recognition-stability-cv.v1','folds':[],'methods':METHODS,'seed':42,'selection':'all five preregistered methods, no fold selection'}
        for fold,held in enumerate(folds):
            train=[i for i in d['train'] if d['groups'][i] not in held]
            ids=[d['n']+j for j,r in enumerate(d['rt']['rows']) if r['source_group'] in held]
            known=sum(p['ground_truth_count'] for p in d['app']['full_pages']['pages'] if p['source_group'] in held)
            base=metrics(d['teacher'][ids]@d['old'].T,d['labels'][ids])
            entry=dict(fold=fold,held_groups=held,train_rows=len(train),matched=len(ids),known_boxes=known,baseline=base,candidates={})
            assert not any(d['groups'][i] in held or d['splits'][i]!='train' for i in train)
            for method in METHODS:
                model,proto,history=fit(d,method,42,train)
                scores=project(model,d['features'][ids])@proto.T
                entry['candidates'][method]=metrics(scores,d['labels'][ids])
                print('CV',fold,method,'baseline',base['top1_correct'],'candidate',entry['candidates'][method]['top1_correct'],'/',len(ids),flush=True)
                entry['candidates'][method]['pages']=[dict(group=g,metrics=metrics(scores[[j for j,i in enumerate(ids) if d['groups'][i]==g]],d['labels'][[i for i in ids if d['groups'][i]==g]])) for g in held]
            report['folds'].append(entry);write_json(RUN/'cv-progress.json',report)
        report['pooled']={method:{'correct':sum(f['candidates'][method]['top1_correct'] for f in report['folds']),'matched':sum(f['matched'] for f in report['folds']),'known':sum(f['known_boxes'] for f in report['folds'])} for method in METHODS}
        report['pooled']['baseline']={'correct':sum(f['baseline']['top1_correct'] for f in report['folds']),'matched':sum(f['matched'] for f in report['folds']),'known':sum(f['known_boxes'] for f in report['folds'])}
        write_json(RUN/'cv-results.json',report)
    assert d['before']=={str(p):sha256(p) for p in d['paths']}
    write_json(RUN/(args.phase+'-verification.json'),dict(inputs_unchanged=True,active_model_unchanged=True,only_train_rows_optimized=True,source_groups_disjoint=True,elapsed_seconds=time.monotonic()-start,activated=False))
    print('COMPLETE',args.phase,flush=True)


if __name__=='__main__':main()

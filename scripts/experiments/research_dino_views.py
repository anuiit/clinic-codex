"""Posthoc inference-view diagnostic on fixed SAM boxes; no training."""
import collections, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.experiments import research_dino_tail as study
from scripts.experiments.summarize_recognition_mechanisms import classification
from codex_model.classifier import _preprocess_image

RUN = study.ROOT/'backend/training_runs/20261004-dino-views-v1'


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    d=study.data(); torch.use_deterministic_algorithms(True)
    primary=study.prior.m.read_json(study.RUN/'results.json')
    p=study.prior.m.read_json(study.RUN/'preregistration.json'); study.prior.verify(p)
    inputs=p['input_sha256']|{str(f):study.prior.m.sha256(f) for f in [Path(__file__),study.RUN/'results.json']+list(study.RUN.glob('*_fold*.pt'))}
    RUN.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema_version='dino-views.v1',input_sha256=inputs,
        trigger='Adaptive diagnostic after primary raw gain+15 and masked gain+2; retain primary200-step result unchanged',
        arms=['best_prior','head_only','last_two_blocks'],views=['mask','rectangle'],
        scope='Same254 known SAM matched crops, identical proposal boxes/padding; only existing train pixels; no dev/test crop encoded',
        gating='Same rare1..4 rule applied to historical prediction on each input view; ungated outcomes also reported',
        selection='No fitting, hyperparameter search, activation or independent validation; descriptive posthoc comparison',
        activated=False,optimizer_updates=0,locked_test_scored=False)
    study.prior.m.write_json(RUN/'preregistration.json',protocol)
    fixture=d['fixture']; records=[(j,r) for j,r in enumerate(fixture['rows']) if r['split']=='train']
    assert len(records)==254
    b=study.backbone(); original_tail=study.Tail(b).cuda().eval().requires_grad_(False)
    rect={}; original_rect={}; max_error=0.
    for start in range(0,len(records),16):
        rr=records[start:start+16]
        x=torch.cat([_preprocess_image(Image.fromarray(fixture['rgb'][j].numpy()),224) for j,r in rr]).cuda()
        t=b.prepare_tokens_with_masks(x)
        for block in b.blocks[:-2]: t=block(t)
        f=original_tail(t)
        ref=fixture['features']['rectangle'][[j for j,r in rr]].cuda()
        max_error=max(max_error,float((f-ref).abs().max()))
        torch.testing.assert_close(f,ref,atol=1e-4,rtol=1e-4)
        for k,(j,r) in enumerate(rr):
            i=d['n']+r['runtime_index']; rect[i]=t[k].cpu(); original_rect[i]=ref[k]
    prefix=np.load(study.CACHE_RUN/'tokens.npy',mmap_mode='r')
    saved=study.prior.m.read_json(study.CACHE_RUN/'cache.json'); positions={i:j for j,i in enumerate(saved['ids'])}
    entries=[]
    for fold,held in enumerate(study.prior.m.folds()):
        raw,train,lookup,anchor,excluded=study.prior.scope(d,held); counts=study.support_counts(d,raw)
        held_ids=[d['n']+r['runtime_index'] for j,r in records if r['source_group'] in held]
        assert all(i not in train and d['groups'][i] in excluded for i in held_ids)
        for method in ['best_prior','head_only','last_two_blocks']:
            path=study.prior.RUN/f'paired_fold{fold}_seed42_step1000.pt' if method=='best_prior' else study.RUN/f'{method}_fold{fold}.pt'
            cp=torch.load(path,weights_only=True,map_location='cpu')
            head=study.prior.core.ProjectionHead().cuda().eval(); head.load_state_dict(cp['projection'])
            tail=study.Tail(b).cuda().eval().requires_grad_(False)
            if method!='best_prior': tail.load_state_dict(cp['tail'])
            for view in ['mask','rectangle']:
                rows=[]
                for start in range(0,len(held_ids),study.CHUNK):
                    ids=held_ids[start:start+study.CHUNK]
                    t=study.token_batch(prefix,positions,ids) if view=='mask' else torch.stack([rect[i] for i in ids]).cuda()
                    f=tail(t); scores=head(f)@d['old'].T
                    oldf=d['features'][ids] if view=='mask' else torch.stack([original_rect[i] for i in ids])
                    historical=(study.prior.core.project(d['original'],oldf)@d['old'].T).argmax(1)
                    pred=scores.argmax(1); use=(counts[historical]>=1)&(counts[historical]<=4)
                    gated=torch.where(use,historical,pred)
                    for k,i in enumerate(ids):
                        rows.append(dict(row_id=d['objects'][i],group=d['groups'][i],label=int(d['labels'][i]),prediction=int(pred[k]),
                            gated_prediction=int(gated[k]),historical_prediction=int(historical[k]),fallback=bool(use[k])))
                if view=='mask':
                    chosen=next(e for e in primary['folds'] if e['fold']==fold and e['method']==('head_only' if method=='best_prior' else method))
                    expected=chosen['baseline' if method=='best_prior' else 'evaluation']['mask']
                    assert [(r['row_id'],r['prediction'],r['gated_prediction']) for r in rows]==[(r['row_id'],r['prediction'],r['gated_prediction']) for r in expected]
                entries.append(dict(fold=fold,held=held,method=method,view=view,rows=rows))
    families={}
    for method in protocol['arms']:
        families[method]={}
        for view in protocol['views']:
            rows=[r for e in entries if e['method']==method and e['view']==view for r in e['rows']]
            assert len(rows)==len({r['row_id'] for r in rows})==254
            gated=[dict(r,prediction=r['gated_prediction']) for r in rows]
            families[method][view]=dict(ungated=classification(rows),gated=classification(gated),
                known_joint=sum(r['prediction']==r['label'] for r in gated)/343)
    matched={r['row_id'] for e in entries for r in e['rows']}; raw_breakdown={}
    for method in protocol['arms']:
        ee=[e for e in primary['folds'] if e['method']==('head_only' if method=='best_prior' else method)]
        field='baseline' if method=='best_prior' else 'evaluation'
        rows=[r for e in ee for r in e[field]['raw']]
        raw_breakdown[method]={}
        for name,active in [('matched',True),('missed',False)]:
            selected=[dict(r,prediction=r['gated_prediction']) for r in rows if (r['row_id'] in matched)==active]
            raw_breakdown[method][name]=classification(selected)
    study.prior.verify(protocol)
    output=dict(schema_version='dino-views-summary.v1',families=families,raw_by_sam_availability=raw_breakdown,folds=entries,
        max_original_rectangle_feature_error=max_error,optimizer_updates=0,activated=False,locked_test_scored=False,
        verification=dict(all254_primary_mask_predictions_reproduced=True,rectangle_saved_pixel_features_replayed=True,
                          inputs_active_weights_unchanged=True,only_train_fixture_rows_encoded=True),
        scope='Adaptive descriptive inference comparison; same17 repeatedly inspected pages; no independent validation')
    study.prior.m.write_json(RUN/'results.json',output)
    study.prior.m.write_json(study.ROOT/'reports/dino-views-results-20261004.json',output)
    print(json.dumps(dict(families={k:{v:x['gated']['correct'] for v,x in f.items()} for k,f in families.items()},raw_breakdown=raw_breakdown),indent=2))


if __name__=='__main__': main()

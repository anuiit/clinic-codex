"""Summarize immutable long base experiments; primary is fixed epoch100 only."""
import collections,csv,json,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.summarize_recognition_mechanisms import aggregate,classification
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'backend/training_runs/20261004-base-corpus-long-v2'
OUT=ROOT/'reports/base-corpus-long-results-20261004.json'
SNAPSHOT=ROOT/'backend/training_corpus/snapshots/snapshot-611c734590f272060251'
CACHE=ROOT/'backend/model_registry/versions/20261003-annotation-lanczos-warmstart/training_data/precomputed/features.pt'


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cv_summary(records,meta,index,epoch=None,best=False):
    entries=[]
    for r in records:
        e=r['best_evaluation'] if best else r['milestones'][str(epoch)]['evaluation']
        lookup={k:i for i,k in enumerate(index)}
        entry=dict(held=r['plan']['held_groups'],selected_raw_ids=[lookup[k] for k in r['row_exposures']])
        entries.append((entry,e))
    return aggregate(entries,meta,index)


def paired(candidate,baseline):
    a={p['group']:p for p in candidate['pages']};b={p['group']:p for p in baseline['pages']}
    assert set(a)==set(b);groups=sorted(a);delta=np.array([a[g]['joint']-b[g]['joint'] for g in groups])
    rng=np.random.default_rng(20261004);boot=np.mean(rng.choice(delta,size=(10000,len(delta)),replace=True),axis=1)
    return dict(mean_page_delta=float(delta.mean()),page_deltas={g:float(d) for g,d in zip(groups,delta)},
        wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta<-1e-10).sum()),
        descriptive_bootstrap95=[float(v) for v in np.quantile(boot,[.025,.975])],
        interpretation='Paired resampling of17 held pages, shared overlapping training sets; descriptive, not independent blind confidence')


def guards(candidate,baseline,candidate_records,baseline_records):
    by_scope={r['plan']['scope']:r for r in baseline_records}
    deltas={r['plan']['scope']:r['milestones']['100']['evaluation']['raw']['top1_micro']-by_scope[r['plan']['scope']]['milestones']['100']['evaluation']['raw']['top1_micro'] for r in candidate_records}
    checks=dict(page_mean_improves=candidate['page_mean_joint']>baseline['page_mean_joint'],
        rare_macro_no_regression=candidate['natural_support_bins']['rare1_4']['macro']>=baseline['natural_support_bins']['rare1_4']['macro'],
        every_fold_raw_drop_at_most2pp=min(deltas.values())>=-.02)
    return dict(checks=checks,eligible=all(checks.values()),fold_raw_deltas=deltas)


@torch.no_grad()
def training_crop_diagnostic(cache,parent_ids,full,historical):
    from scripts.research_recognition_stability import aligned_prototypes
    from codex_pipeline.models.projection_head import ProjectionHead
    torch.set_num_threads(4)
    lookup={k:i for i,k in enumerate(cache['row_ids'])}
    raw_ids=[i for i,k in enumerate(cache['row_ids']) if k not in parent_ids and cache['dataset_splits'][i]=='train']
    fixture=torch.load(ROOT/'backend/training_runs/20261003-recognition-mechanisms-v2/crop-fixture.pt',weights_only=True,map_location='cpu')
    mask_ids=[i for i,r in enumerate(fixture['rows']) if r['split']=='train']
    matched=[lookup[fixture['rows'][i]['row_id']] for i in mask_ids]
    assert len(raw_ids)==343 and len(mask_ids)==254 and set(matched)<=set(raw_ids)
    runtime=ROOT/'backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime'
    packages={'historical':dict(model_state_dict=torch.load(runtime/'weights/projection.pt',weights_only=True,map_location='cpu'),
        prototypes=aligned_prototypes(torch.load(runtime/'weights/prototypes.pt',weights_only=True,map_location='cpu'),cache['class_names']))}
    for arm in ['parent','combined']:
        r=next(r for r in full[arm] if r['seed']==42)
        packages[arm]=torch.load(Path(r['training_artifact_dir'])/'epoch100.pt',weights_only=True,map_location='cpu')
    out={}
    for name,package in packages.items():
        model=ProjectionHead();model.load_state_dict(package['model_state_dict']);model.eval();proto=package['prototypes']
        raw_pred=(model(cache['features'][raw_ids])@proto.T).argmax(1)
        common_pred=(model(cache['features'][matched])@proto.T).argmax(1)
        mask_pred=(model(fixture['features']['mask'][mask_ids])@proto.T).argmax(1)
        yr=cache['labels'][raw_ids];ym=cache['labels'][matched]
        a=common_pred==ym;b=mask_pred==ym
        expected=historical['application']['train'] if name=='historical' else next(r for r in full[name] if r['seed']==42)['fixed_epoch100']['application']['train']
        assert int(b.sum())==expected['top1_correct'] and len(b)==expected['count']
        out[name]=dict(annotation_raw_correct=int((raw_pred==yr).sum()),annotation_raw_count=len(raw_ids),
            matched_raw_correct=int(a.sum()),matched_mask_correct=int(b.sum()),matched_count=len(mask_ids),
            both_correct=int((a&b).sum()),raw_correct_mask_wrong=int((a&~b).sum()),raw_wrong_mask_correct=int((~a&b).sum()),both_wrong=int((~a&~b).sum()),
            examples=[dict(row_id=cache['row_ids'][i],label=cache['class_names'][int(ym[j])],raw_prediction=cache['class_names'][int(common_pred[j])],
                masked_prediction=cache['class_names'][int(mask_pred[j])]) for j,i in enumerate(matched) if a[j] and not b[j]])
    return dict(candidates=out,scope='Training business pages, fixed epoch100 seed42; exact annotated crops vs matched SAM crops. Seen-data diagnostic, not generalization',
        known=343,matched=254,missed=89,geometry_and_backbone_unchanged=True)


def main():
    p=read(RUN/'preregistration.json');assert all(sha(Path(k))==v for k,v in p['input_sha256'].items())
    dependency=read(ROOT/'backend/training_runs/20261003-recognition-mechanisms-v2/preregistration.json')
    assert all(sha(Path(k))==v for k,v in dependency['input_sha256'].items())
    d=read(RUN/'results.json');records=d['results'];assert len(records)==16
    assert { (r['plan']['scope'],r['plan']['arm'],r['plan']['seed']) for r in records }=={(x['scope'],x['arm'],x['seed']) for x in p['plans']}
    meta={r['row_id']:r for r in csv.DictReader((SNAPSHOT/'metadata.csv').open())}
    c=torch.load(CACHE,weights_only=True,map_location='cpu');index=c['row_ids'];labels=c['labels'].tolist();parent=read(SNAPSHOT.parent/'snapshot-471c74f801b749a9ad6e/snapshot_manifest.json');old=set(parent['active_row_ids'])
    unique=set();total=0
    for r in records:
        plan=r['plan'];ids=[i for i,k in enumerate(index) if k in r['row_exposures']]
        assert len(ids)==plan['train_count'] and all(c['dataset_splits'][i]=='train' for i in ids)
        assert not set(plan['held_groups'])&{c['source_groups'][i] for i in ids}
        assert sha_bytes('\n'.join(index[i] for i in ids).encode())==plan['train_row_ids_sha256']
        if plan['arm']=='parent':assert all(index[i] in old for i in ids)
        counts=collections.Counter(labels[i] for i in ids)
        assert len(counts)==286 and sum(n>=8 for n in counts.values())==plan['eligible_classes']
        assert sum(r['row_exposures'].values())==8000000 and r['updates']==50000
        assert all((r['row_exposures'][index[i]]>0)==(counts[labels[i]]>=8) for i in ids)
        assert len(r['history'])==100 and r['history'][-1]['updates']==50000
        training=Path(r['training_artifact_dir']);assert r['checkpoint_sha256']=={cp.name:sha(cp) for cp in training.glob('*.pt')}
        unique.add(str(training))
    assert len(unique)==p['unique_fits']==11
    by_arm={arm:[r for r in records if r['plan']['scope'].startswith('cv') and r['plan']['arm']==arm] for arm in ['parent','combined']}
    assert all(len(rs)==5 for rs in by_arm.values())
    for r,e in zip(by_arm['combined'],d['historical']['cv']):
        assert set(r['plan']['held_groups'])=={p['group'] for p in e['pages']}
    historical=aggregate([(dict(held=by_arm['combined'][i]['plan']['held_groups']),e) for i,e in enumerate(d['historical']['cv'])],meta,index)
    curves={arm:{str(epoch):cv_summary(rs,meta,index,epoch) for epoch in sorted(map(int,rs[0]['milestones']))} for arm,rs in by_arm.items()}
    primary={arm:curves[arm]['100'] for arm in by_arm}
    primary['paired_enriched_vs_parent']=paired(primary['combined'],primary['parent'])
    primary['paired_enriched_vs_historical']=paired(primary['combined'],historical)
    primary['guard_vs_parent']=guards(primary['combined'],primary['parent'],by_arm['combined'],by_arm['parent'])
    hist_records=[dict(plan=r['plan'],milestones={'100':{'evaluation':e}}) for r,e in zip(by_arm['combined'],d['historical']['cv'])]
    primary['guard_vs_historical']=guards(primary['combined'],historical,by_arm['combined'],hist_records)
    full={arm:[] for arm in by_arm}
    for r in records:
        if r['plan']['scope']!='full':continue
        full[r['plan']['arm']].append(dict(seed=r['plan']['seed'],fixed_epoch100=r['milestones']['100']['evaluation'],
            epoch0=r['milestones']['0']['evaluation'],best_dev_epoch=r['best_epoch'],best_dev_evaluation=r['best_evaluation'],
            final_train_loss=r['history'][-1]['train_loss'],best_dev_raw_correct=max(h['dev']['top1_correct'] for h in r['history']),
            training_artifact_dir=r['training_artifact_dir']))
    table=[]
    for arm,rs in by_arm.items():
        for r in rs:
            e=r['milestones']['100']['evaluation'];table.append(dict(fold=r['plan']['scope'],arm=arm,train=r['plan']['train_count'],new=r['plan']['new_count'],eligible=r['plan']['eligible_classes'],raw=e['raw'],masked=e['masked'],pages=e['pages']))
    output=dict(schema_version='base-corpus-long-summary.v2',run=str(RUN),activated=False,primary_checkpoint='fixed_epoch100',
        distinct_fit_count=11,comparisons=16,updates_per_fit=50000,total_optimizer_updates=550000,
        unique_training_seconds=sum(read(Path(x)/'result.json')['elapsed_seconds'] for x in unique),historical=historical,
        primary=primary,curves=curves,cv_best_dev_secondary={arm:cv_summary(rs,meta,index,best=True) for arm,rs in by_arm.items()},
        fold_table=table,full_corpus_secondary=full,historical_full_secondary=d['historical']['full'],
        training_crop_diagnostic=training_crop_diagnostic(c,old,full,d['historical']['full']),
        coverage=dict(classes=286,parent_train=9128,combined_train=9471,new_train=343,new_train_classes=99,new_pages=17,
            eligible_classes_parent=250,eligible_classes_combined=252,new_rows_eligible=337,new_rows_prototype_only=6,classes_with_no_support=0),
        verification=dict(input_hashes_unchanged=True,inherited_dependency_hashes_unchanged=True,checkpoint_hashes_passed=True,row_coverage_and_exposure_passed=True,shared_parent_fits_verified=True,
            active_model_unchanged=True,implementation_check=p['implementation_check']),
        limitations=['Historical pretraining provenance unknown','Previously examined pages and test, not blind generalization',
            'CV seed42 only; full-corpus secondary three seeds','Training uses raw crops; SAM views only evaluated','No backbone or segmentation training',
            'Joint recall of known matched objects is not detection precision or F1'])
    write(OUT,output)
    write(RUN/'summary-verification.json',dict(summary_path=str(OUT),summary_sha256=sha(OUT),schema_note='Runner schema base-corpus-long.v1 describes serialized format; run v2 adds training reuse and has separate preregistration; v1 interrupted with source archived',**output['verification']))
    print(json.dumps(dict(historical=dict(raw=historical['raw']['correct'],masked=historical['masked_correct'],page=historical['page_mean_joint']),
        primary={a:dict(raw=primary[a]['raw']['correct'],masked=primary[a]['masked_correct'],page=primary[a]['page_mean_joint'],rare=primary[a]['natural_support_bins']['rare1_4']) for a in by_arm},
        paired=primary['paired_enriched_vs_parent'],guards=primary['guard_vs_parent'],total_seconds=output['unique_training_seconds']),indent=2))


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


if __name__=='__main__':main()

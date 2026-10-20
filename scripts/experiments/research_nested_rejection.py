"""Nested page split: fit projection, calibrate rejection, then read outer pages."""
import argparse,collections,csv,json,random,statistics,time
from pathlib import Path
import torch
from scripts.experiments import research_rejection_calibration as r
f=r.f;d=r.d;ROOT=r.ROOT
OLD_TRAIN_RUN=d.RUN
RUN=ROOT/'backend/training_runs/20261004-nested-rejection-v1'
OUT=ROOT/'reports/nested-rejection-results-20261004.json'
SEEDS=r.SEEDS;METHODS=r.METHODS;STEPS=1000;SPLIT_SEED=20261004
POLICIES=['current_035','calibrated','rare_guarded']


def pick_calibration(groups,fold):
    if len(groups)<4:raise ValueError('Need calibration and fit pages')
    return sorted(random.Random(SPLIT_SEED+fold).sample(sorted(groups),3))


def select(curve,guarded=False):
    reference=next(z for z in curve if z['threshold']==.35)
    observed=reference['counts']['rare_known']
    if guarded and not observed:
        chosen=reference;reason='no_rare_calibration_objects_use_035';admissible=[.35]
    else:
        allowed=[z for z in curve if not guarded or (z['counts']['rare_correct']>=reference['counts']['rare_correct'] and z['metrics']['rare_class_macro']>=reference['metrics']['rare_class_macro'])]
        assert allowed
        best=max(z['page_mean']['exact_f1'] for z in allowed)
        chosen=min((z for z in allowed if z['page_mean']['exact_f1']>=best-1e-12),key=lambda z:z['threshold'])
        reason='rare_nonregression_then_page_f1' if guarded else 'page_f1'
        admissible=[z['threshold'] for z in allowed]
    return dict(threshold=chosen['threshold'],objective=chosen['page_mean']['exact_f1'],calibration_rare_objects=observed,calibration_rare_correct=chosen['counts']['rare_correct'],admissible=admissible,reason=reason)


def check():
    r.check();a=pick_calibration(['d','a','c','e','b'],2)
    assert len(a)==3 and a==pick_calibration(['b','e','c','a','d'],2)
    curve=[dict(threshold=t,counts=dict(rare_known=2,rare_correct=c),metrics=dict(rare_class_macro=c/2),page_mean=dict(exact_f1=v)) for t,c,v in [(.35,2,.3),(.6,2,.5),(.8,1,.8)]]
    assert select(curve)['threshold']==.8 and select(curve,True)['threshold']==.6
    for z in curve:z['counts']['rare_known']=0;z['counts']['rare_correct']=0;z['metrics']['rare_class_macro']=None
    assert select(curve,True)['threshold']==.35
    return dict(split_order_invariant=True,rare_guard_rejects_high_f1_with_loss=True,no_calibration_rares_uses_035=True,**r.check())


def support_pools(data,raw,meta):
    pools=collections.defaultdict(set)
    for i in raw:
        z=meta[data['cache']['row_ids'][i]];pools[int(z['class_label'])].add(z['source_pixel_sha256'])
    return pools


@torch.no_grad()
def predictions(data,model,features,pages,pools,outer_pools):
    result=[]
    for p in pages:
        ids=list(range(p['proposal_offset'],p['proposal_offset']+len(p['proposals'])))
        old=d.core.project(data['original'],features[ids])@data['old'].T
        new=d.core.project(model,features[ids])@data['old'].T
        flag=torch.tensor([1<=len(pools[c])<=4 for c in old.argmax(1).tolist()],device='cuda')
        scores=dict(baseline=old,paired=new,gated=torch.where(flag[:,None],old,new));methods={}
        for m,sc in scores.items():
            values,labels=sc.max(1)
            methods[m]=dict(predictions=[dict(q,prediction=int(labels[j]),similarity=float(values[j]),fallback=bool(flag[j]) if m=='gated' else False) for j,q in enumerate(p['proposals'])])
        result.append(dict(group=p['group'],gt=[dict(z,natural_support=len(pools[z['label']]),outer_reference_support=len(outer_pools[z['label']])) for z in p['gt']],ignore=p['ignore'],methods=methods))
    return result


def aggregate(pages,method,policy):
    by_seed={}
    for seed in SEEDS:
        ps=[p for p in pages if p['seed']==seed];assert len(ps)==17
        ev=[p['methods'][method]['evaluations'][policy] for p in ps]
        counts={k:sum(e['counts'][k] for e in ev) for k in ev[0]['counts']}
        rare=collections.defaultdict(list);reference_rare=collections.defaultdict(list)
        for p,e in zip(ps,ev):
            correct={z['gt_index'] for z in e['matches'] if z['correct']}
            for j,z in enumerate(p['gt']):
                if 1<=z['natural_support']<=4:rare[z['label']].append(int(j in correct))
                if 1<=z['outer_reference_support']<=4:reference_rare[z['label']].append(int(j in correct))
        values=f.metrics(counts);values['rare_class_macro']=statistics.mean(statistics.mean(v) for v in rare.values()) if rare else None
        values['reference_rare_class_macro']=statistics.mean(statistics.mean(v) for v in reference_rare.values())
        reference_counts=dict(known=sum(map(len,reference_rare.values())),correct=sum(map(sum,reference_rare.values())))
        assert reference_counts['known']==10
        by_seed[str(seed)]=dict(counts=counts,metrics=values,reference_rare=reference_counts,page_mean={k:statistics.mean(f.metrics(e['counts'])[k] for e in ev) for k in ['exact_precision','exact_recall','exact_f1']},lost_correct=sum(len(p['methods'][method]['delta'][policy]['lost_correct_gt_indices']) for p in ps),gained_correct=sum(len(p['methods'][method]['delta'][policy]['gained_correct_gt_indices']) for p in ps))
    return dict(seeds=by_seed,mean_metrics={k:statistics.mean(z['metrics'][k] for z in by_seed.values()) for k in next(iter(by_seed.values()))['metrics']},mean_page={k:statistics.mean(z['page_mean'][k] for z in by_seed.values()) for k in ['exact_precision','exact_recall','exact_f1']})


def main():
    checks=check();old=d.m.read_json(r.OUT);d.verify(old['protocol']);previous=d.m.read_json(f.OUT);d.verify(previous['protocol'])
    data=d.base.setup();data['fixture']=d.m.load_fixture(data);torch.use_deterministic_algorithms(True)
    saved=torch.load(previous['feature_cache'],weights_only=True,map_location='cpu');pages=saved['pages']
    assert saved['features'].shape==(654,384) and len(pages)==17
    meta={z['row_id']:z for z in csv.DictReader((d.core.SNAPSHOT/'metadata.csv').open())}
    splits=[];batch_plans={};folds=d.m.folds()
    for fold,held in enumerate(folds):
        outer_raw,outer_train,_,_,outer_excluded=d.scope(data,held)
        available={p['group'] for p in pages if p['group'] not in outer_excluded}
        picked=pick_calibration(available,fold)
        raw,train,lookup,anchor,excluded=d.scope(data,sorted(set(held)|set(picked)))
        calibration=sorted(g for g in available if g in excluded)
        fit_pages=sorted(g for g in available if g not in excluded)
        assert len(calibration)==3 and len(fit_pages)>=10
        assert set(calibration).isdisjoint(held) and set(fit_pages).isdisjoint(set(calibration)|set(held))
        assert set(raw)<=set(outer_raw) and {int(data['labels'][i]) for i in raw}==set(range(286))
        withheld=set(held)|set(calibration)
        assert not {data['groups'][i] for i in train}&withheld
        assert not {data['pixels'][i] for i in train}&{data['pixels'][i] for i in range(data['n']) if data['groups'][i] in withheld}
        fit_row_ids={data['objects'][i] for i in raw}
        assert all(z['in_active_train'] and z['row_id'] in fit_row_ids for p in pages if p['group'] in fit_pages for z in p['gt'])
        planned={}
        for seed in SEEDS:
            plan=d.core.batches(data['labels'],train,data['n'],seed,STEPS);batch_plans[(fold,seed)]=plan
            planned[str(seed)]=d.m.digest(torch.tensor([i for batch in plan for i in batch]).numpy())
        splits.append(dict(fold=fold,held=held,outer_excluded=outer_excluded,picked_calibration=picked,calibration=calibration,fit_pages=fit_pages,excluded=excluded,raw=raw,train=train,lookup=lookup,anchor=sorted(anchor),outer_raw=outer_raw,batch_sha256=planned))
    paths=[Path(__file__),r.OUT,Path(r.__file__),ROOT/'scripts/train_contrastive_recognition.py',ROOT/'scripts/train_segmentation.py',OLD_TRAIN_RUN/'results.json']
    protocol=dict(schema_version='nested-rejection.v1',input_sha256=old['protocol']['input_sha256']|{str(p):d.m.sha256(p) for p in paths},adaptive=True,activated=False,locked_test_scored=False,seeds=SEEDS,steps=STEPS,fit_count=15,optimizer_updates=15000,fit_recipe='Reuse unmodified research_crop_compatible.fit method paired, historical init and fixed prototypes, same AdamW/loss/batch recipe. In-memory output RUN relocation to this run and MILESTONES=[1000], fit held argument=calibration pages only; raw/train/lookup/anchor already exclude closure(outer+calibration). No outer evaluation during fit.',split_rule='For each outer fold, random.Random(20261004+fold) samples3 sorted available page groups; identical seeds; close union outer+calibration components and exclude their pixels from raw+SAM fit. Snapshot business components verified singleton; no scores or held labels used in split.',splits=splits,threshold_grid=r.GRID,selection='Primary: maximize arithmetic mean exact F1 per calibration page, ties within1e-12 lower threshold. Secondary rare_guarded: same objective only among cuts preserving .35 calibration rare_correct and rare_class_macro; if rare_known=0 use .35 without optimization. One scalar each method/fold/seed; no outer data consulted.',methods=METHODS,policies=POLICIES,primary='gated/calibrated',secondary='gated/rare_guarded',rare='Gate and calibration guard use unique raw fit pixels1..4 after exclusion outer+calibration. Publish held fit-support rare cohort plus original outer-support1..4 cohort10objects to compare with previous research.',reference='Same nested head at .35 isolates rejection; previous nonnested .35 results contextual only. Historical baseline replay must reproduce previous full-readout predictions.',geometry=previous['protocol']['primary_geometry'],limitation='Same17 previously examined pages; adaptive retrospective nested split, not blind evaluation; historical init/prototypes provenance unknown; no dev/test selection or scoring; active runtime unchanged.',checks=checks)
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol)
    features=saved['features'].cuda();entries=[];outer_pages=[];started=time.monotonic()
    prior_run,prior_milestones=d.RUN,d.MILESTONES
    d.RUN,d.MILESTONES=RUN,[1000]
    try:
        for split in splits:
            fold=split['fold'];outer_pools=support_pools(data,split['outer_raw'],meta);pools=support_pools(data,split['raw'],meta)
            cal=[p for p in pages if p['group'] in split['calibration']]
            for seed in SEEDS:
                fit=d.fit(data,'paired',seed,fold,split['calibration'],split['raw'],split['train'],split['lookup'],set(split['anchor']),batch_plans[(fold,seed)])
                assert fit['batch_sha256']==split['batch_sha256'][str(seed)]
                cp=RUN/f'paired_fold{fold}_seed{seed}_step1000.pt';package=torch.load(cp,weights_only=True,map_location='cpu')
                assert package['train_ids']==split['train']
                model=d.core.ProjectionHead().cuda();model.load_state_dict(package['projection']);model.eval()
                cal_pages=predictions(data,model,features,cal,pools,outer_pools)
                selections={}
                for method in METHODS:
                    curve=[dict(threshold=t,**r.summarize(cal_pages,method,t)) for t in r.GRID]
                    selections[method]=dict(curve=curve,calibrated=select(curve),rare_guarded=select(curve,True))
                entry=dict(fold=fold,seed=seed,calibration_pages=cal_pages,fit=fit,selections=selections,checkpoint_sha256=d.m.sha256(cp))
                d.m.write_json(RUN/f'selection_fold{fold}_seed{seed}.json',entry)
                # Freeze all selections before the first outer readout for this head.
                test=[p for p in pages if p['group'] in split['held']]
                current=predictions(data,model,features,test,pools,outer_pools)
                for p in current:
                    reference=next(q for q in previous['fold_pages'] if q['fold']==fold and q['seed']==seed and q['group']==p['group'])
                    assert all(z['outer_reference_support']==q['natural_support'] for z,q in zip(p['gt'],reference['gt']))
                    for method in METHODS:
                        rows=p['methods'][method]['predictions'];evaluations={};delta={}
                        if method=='baseline':
                            old_rows=reference['methods'][method]['predictions']
                            assert all(a['prediction']==b['prediction'] and abs(a['similarity']-b['similarity'])<1e-6 for a,b in zip(rows,old_rows))
                        for policy in POLICIES:
                            threshold=.35 if policy=='current_035' else selections[method][policy]['threshold']
                            evaluations[policy]=f.score_page(p['gt'],p['ignore'],rows,threshold)
                            before={z['gt_index'] for z in evaluations['current_035']['matches'] if z['correct']};after={z['gt_index'] for z in evaluations[policy]['matches'] if z['correct']}
                            delta[policy]=dict(lost_correct_gt_indices=sorted(before-after),gained_correct_gt_indices=sorted(after-before))
                        p['methods'][method].update(evaluations=evaluations,delta=delta)
                    outer_pages.append(dict(p,fold=fold,seed=seed))
                entries.append(entry);d.verify(protocol)
                print('NESTED',fold,seed,'cal rare',selections['gated']['calibrated']['calibration_rare_objects'],'threshold',selections['gated']['calibrated']['threshold'],'guarded',selections['gated']['rare_guarded']['threshold'],flush=True)
    finally:d.RUN,d.MILESTONES=prior_run,prior_milestones
    assert len(entries)==15 and len(list(RUN.glob('*_step1000.pt')))==15
    agg={m:{policy:aggregate(outer_pages,m,policy) for policy in POLICIES} for m in METHODS}
    guards={}
    for policy in ['calibrated','rare_guarded']:
        b=agg['gated']['current_035'];a=agg['gated'][policy]
        guards[policy]=dict(rare_correct_no_regression=all(a['seeds'][str(s)]['counts']['rare_correct']>=b['seeds'][str(s)]['counts']['rare_correct'] for s in SEEDS),rare_class_macro_no_regression=all(a['seeds'][str(s)]['metrics']['rare_class_macro']>=b['seeds'][str(s)]['metrics']['rare_class_macro'] for s in SEEDS),reference_rare_no_regression=all(a['seeds'][str(s)]['reference_rare']['correct']>=b['seeds'][str(s)]['reference_rare']['correct'] for s in SEEDS),precision_gain=a['mean_metrics']['exact_precision']>b['mean_metrics']['exact_precision'],f1_gain=a['mean_metrics']['exact_f1']>b['mean_metrics']['exact_f1'])
    d.verify(protocol)
    result=dict(schema_version='nested-rejection-summary.v1',run=str(RUN),protocol=protocol,folds=entries,outer_pages=outer_pages,aggregate=agg,guards=guards,elapsed_seconds=time.monotonic()-started,fit_seconds=sum(e['fit']['elapsed_seconds'] for e in entries),optimizer_updates=15000,activated=False,locked_test_scored=False,verification=dict(input_sources_active_files_unchanged=True,calibration_and_outer_excluded_from_fit_groups_pixels_components=True,all286_classes_fit_each_head=True,all_batch_plans_reproduced=True,exactly15_new_fixed1000_fits=True,selections_saved_before_outer_readout=True,historical_predictions_reproduced=True,no_dev_or_test_scoring=True))
    d.m.write_json(RUN/'results.json',result);d.m.write_json(OUT,result)
    print(json.dumps(dict(gated={p:dict(correct=[z['counts']['correct'] for z in a['seeds'].values()],rare=[z['counts']['rare_correct'] for z in a['seeds'].values()],rare_known=[z['counts']['rare_known'] for z in a['seeds'].values()],reference_rare=[z['reference_rare']['correct'] for z in a['seeds'].values()],mean=a['mean_metrics']) for p,a in agg['gated'].items()},guards=guards,seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:print(json.dumps(check()))
    else:main()

"""Train-page-only scalar rejection calibration on frozen proposal features/heads."""
import argparse,collections,csv,json,statistics,time
from pathlib import Path
import torch
from scripts.experiments import research_full_proposals as f
d=f.d
ROOT=f.ROOT
RUN=ROOT/'backend/training_runs/20261004-rejection-calibration-v1'
OUT=ROOT/'reports/rejection-calibration-results-20261004.json'
GRID=[i/100 for i in range(35,96,5)]
METHODS=f.METHODS
SEEDS=f.SEEDS


def summarize(pages,method,threshold):
    evaluations=[f.score_page(p['gt'],p['ignore'],p['methods'][method]['predictions'],threshold) for p in pages]
    assert evaluations
    counts={k:sum(e['counts'][k] for e in evaluations) for k in evaluations[0]['counts']}
    rare=collections.defaultdict(list)
    for p,e in zip(pages,evaluations):
        correct={z['gt_index'] for z in e['matches'] if z['correct']}
        for i,z in enumerate(p['gt']):
            if 1<=z['natural_support']<=4:rare[z['label']].append(int(i in correct))
    values=f.metrics(counts)
    values['rare_class_macro']=statistics.mean(statistics.mean(v) for v in rare.values()) if rare else None
    return dict(counts=counts,metrics=values,page_mean={k:statistics.mean(f.metrics(e['counts'])[k] for e in evaluations) for k in ['exact_precision','exact_recall','exact_f1']})


def calibrate(train_pages,method):
    if not train_pages:raise ValueError('No allowed train pages for calibration')
    curve=[dict(threshold=t,**summarize(train_pages,method,t)) for t in GRID]
    best=max(z['page_mean']['exact_f1'] for z in curve)
    chosen=min((z for z in curve if z['page_mean']['exact_f1']>=best-1e-12),key=lambda z:z['threshold'])
    return dict(threshold=chosen['threshold'],objective=chosen['page_mean']['exact_f1'],curve=curve)


def check():
    f.check()
    gt=[dict(bbox=[0,0,10,10],label=0,natural_support=2)]
    rows=[dict(bbox=[0,0,10,10],prediction=0,similarity=.9,proposal_index=0),dict(bbox=[0,0,10,10],prediction=1,similarity=.4,proposal_index=1)]
    p=dict(gt=gt,ignore=[],methods={'gated':{'predictions':rows}})
    z=calibrate([p],'gated')
    assert z['threshold']==.45 and z['objective']==1 # rejection must rematch the tied boxes
    assert summarize([p],'gated',.35)['counts']['correct']==0
    p['methods']['gated']['predictions']=[dict(rows[0],similarity=.96)]
    assert calibrate([p],'gated')['threshold']==.35 # all F1 values tied: lowest threshold
    assert f.score_page(gt,[],[dict(rows[0],similarity=.45)],.45)['counts']['accepted']==1
    try:calibrate([],'gated')
    except ValueError:pass
    else:raise AssertionError('Empty calibration must fail')
    return dict(rematching_after_rejection=True,tie_uses_lower_threshold=True,inclusive_threshold=True,empty_scope_fails=True)


def main():
    checks=check();previous=d.m.read_json(f.OUT);prior=d.m.read_json(d.RUN/'results.json');d.verify(previous['protocol'])
    assert not previous['activated'] and previous['optimizer_updates']==0
    saved=torch.load(previous['feature_cache'],weights_only=True,map_location='cpu')
    assert d.m.sha256(Path(previous['feature_cache']))==previous['feature_sha256']
    pages=saved['pages'];assert len(pages)==17 and saved['features'].shape==(654,384)
    assert sum(len(p['gt']) for p in pages)==343 and sum(len(p['proposals']) for p in pages)==654
    # Register before head scoring, threshold curves, selection, or held-out evaluation.
    paths=[Path(__file__),f.OUT,f.RUN/'preregistration.json',ROOT/'scripts/train_contrastive_recognition.py',ROOT/'scripts/train_segmentation.py']
    protocol=dict(schema_version='rejection-calibration.v1',input_sha256=previous['protocol']['input_sha256']|{str(p):d.m.sha256(p) for p in paths},adaptive=True,optimizer_updates=0,activated=False,locked_test_scored=False,seeds=SEEDS,folds=5,methods=METHODS,primary='gated',threshold_grid=GRID,reference_threshold=.35,selection='For each method/fold/seed, one scalar maximizing the arithmetic mean of per-page exact F1 on allowed train pages only; ties within 1e-12 choose lower threshold. No held labels, scores, metrics, or rare counts used for selection.',train_scope='Own fold head and natural supports after the same component closure as its fit. Use all17 business pages whose group is not excluded; never use another fold OOF predictions to calibrate.',held_scope='Reuse verified frozen full-proposal OOF predictions on direct held pages, then rematch independently at the selected threshold. No dev/test images or labels scored.',rare='Known GT class has1..4 unique natural raw train pixels after fold/component exclusion; rare counts and class macro audited after selection. A decline vs gated at0.35 forbids recommending activation; no rare-aware threshold is selected.',geometry=previous['protocol']['primary_geometry'],limitations='Train-page calibration is in-sample for the projection. Same17 previously examined pages; adaptive retrospective research, historical training provenance unknown, no independent blind validation.',checks=checks)
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol)
    started=time.monotonic();data=d.base.setup();data['fixture']=d.m.load_fixture(data)
    torch.use_deterministic_algorithms(True)
    meta={r['row_id']:r for r in csv.DictReader((d.core.SNAPSHOT/'metadata.csv').open())}
    features=saved['features'].cuda();entries=[];held_pages=[]
    for e in prior['folds']:
        fold,seed=e['fold'],e['seed'];held=set(e['held'])
        raw,train,_,_,excluded=d.scope(data,e['held']);assert raw==e['selected_raw_ids']
        pools=collections.defaultdict(set)
        for i in raw:
            r=meta[data['cache']['row_ids'][i]];pools[int(r['class_label'])].add(r['source_pixel_sha256'])
        allowed=[p for p in pages if p['group'] not in excluded]
        assert allowed and not held.intersection(p['group'] for p in allowed)
        train_business_groups={data['groups'][i] for i in train}
        assert {p['group'] for p in allowed}<=train_business_groups
        raw_rows={data['cache']['row_ids'][i] for i in raw}
        assert all(z['in_active_train'] and z['row_id'] in raw_rows for p in allowed for z in p['gt'])
        cp=d.RUN/f'paired_fold{fold}_seed{seed}_step1000.pt'
        assert d.m.sha256(cp)==e['methods']['paired']['milestones']['1000']['checkpoint_sha256']
        model=d.core.ProjectionHead().cuda();model.load_state_dict(torch.load(cp,weights_only=True,map_location='cpu')['projection']);model.eval()
        calibration_pages=[]
        for p in allowed:
            ids=list(range(p['proposal_offset'],p['proposal_offset']+len(p['proposals'])))
            with torch.no_grad():
                old=d.core.project(data['original'],features[ids])@data['old'].T
                new=d.core.project(model,features[ids])@data['old'].T
                flag=torch.tensor([1<=len(pools[c])<=4 for c in old.argmax(1).tolist()],device='cuda')
                scores=dict(baseline=old,paired=new,gated=torch.where(flag[:,None],old,new))
            methods={}
            for method,sc in scores.items():
                values,labels=sc.max(1)
                rows=[dict(q,prediction=int(labels[j]),similarity=float(values[j]),fallback=bool(flag[j]) if method=='gated' else False) for j,q in enumerate(p['proposals'])]
                methods[method]=dict(predictions=rows)
            calibration_pages.append(dict(group=p['group'],gt=[dict(z,natural_support=len(pools[z['label']])) for z in p['gt']],ignore=p['ignore'],methods=methods))
        selections={m:calibrate(calibration_pages,m) for m in METHODS}
        entry=dict(fold=fold,seed=seed,held=sorted(held),excluded=sorted(excluded),selected_raw_ids=raw,calibration_groups=[p['group'] for p in calibration_pages],train_pages=calibration_pages,selections=selections)
        # Selection is now complete; held data enters only below this boundary.
        current=[p for p in previous['fold_pages'] if p['fold']==fold and p['seed']==seed]
        assert {p['group'] for p in current}==held
        for p in current:
            assert all(z['natural_support']==len(pools[z['label']]) for z in p['gt'])
            methods={}
            for m in METHODS:
                rows=p['methods'][m]['predictions'];old=p['methods'][m]['evaluations']['current_035']
                assert f.score_page(p['gt'],p['ignore'],rows,.35)==old
                selected=f.score_page(p['gt'],p['ignore'],rows,selections[m]['threshold'])
                before={z['gt_index'] for z in old['matches'] if z['correct']};after={z['gt_index'] for z in selected['matches'] if z['correct']}
                methods[m]=dict(predictions=rows,threshold=selections[m]['threshold'],evaluations=dict(current_035=old,calibrated=selected),delta=dict(lost_correct_gt_indices=sorted(before-after),gained_correct_gt_indices=sorted(after-before),lost_rare_gt_indices=sorted(i for i in before-after if 1<=p['gt'][i]['natural_support']<=4)))
            held_pages.append(dict(fold=fold,seed=seed,group=p['group'],gt=p['gt'],ignore=p['ignore'],methods=methods))
        entries.append(entry)
        d.m.write_json(RUN/f'fold{fold}_seed{seed}.json',entry)
        print('CALIBRATED',fold,seed,'trainpages',len(calibration_pages),'thresholds',{m:z['threshold'] for m,z in selections.items()},flush=True)
    aggregate={}
    for m in METHODS:
        aggregate[m]={}
        for policy in ['current_035','calibrated']:
            by_seed={}
            for seed in SEEDS:
                ps=[p for p in held_pages if p['seed']==seed];assert len(ps)==17
                ev=[p['methods'][m]['evaluations'][policy] for p in ps]
                c={k:sum(z['counts'][k] for z in ev) for k in ev[0]['counts']}
                rare=collections.defaultdict(list)
                for p,z in zip(ps,ev):
                    correct={r['gt_index'] for r in z['matches'] if r['correct']}
                    for i,r in enumerate(p['gt']):
                        if 1<=r['natural_support']<=4:rare[r['label']].append(int(i in correct))
                values=f.metrics(c);values['rare_class_macro']=statistics.mean(statistics.mean(v) for v in rare.values())
                by_seed[str(seed)]=dict(counts=c,metrics=values,page_mean={k:statistics.mean(f.metrics(z['counts'])[k] for z in ev) for k in ['exact_precision','exact_recall','exact_f1']},lost_correct=sum(len(p['methods'][m]['delta']['lost_correct_gt_indices']) for p in ps) if policy=='calibrated' else 0,gained_correct=sum(len(p['methods'][m]['delta']['gained_correct_gt_indices']) for p in ps) if policy=='calibrated' else 0)
                if policy=='current_035':
                    reference=previous['aggregate'][m][policy]['seeds'][str(seed)]
                    assert c==reference['counts'] and all(values[k]==v for k,v in reference['metrics'].items())
            aggregate[m][policy]=dict(seeds=by_seed,mean_metrics={k:statistics.mean(z['metrics'][k] for z in by_seed.values()) for k in next(iter(by_seed.values()))['metrics']},mean_page={k:statistics.mean(z['page_mean'][k] for z in by_seed.values()) for k in ['exact_precision','exact_recall','exact_f1']})
    before=aggregate['gated']['current_035'];after=aggregate['gated']['calibrated']
    guard=dict(rare_correct_no_regression=all(after['seeds'][str(s)]['counts']['rare_correct']>=before['seeds'][str(s)]['counts']['rare_correct'] for s in SEEDS),rare_class_macro_no_regression=all(after['seeds'][str(s)]['metrics']['rare_class_macro']>=before['seeds'][str(s)]['metrics']['rare_class_macro'] for s in SEEDS),precision_gain=after['mean_metrics']['exact_precision']>before['mean_metrics']['exact_precision'],f1_gain=after['mean_metrics']['exact_f1']>before['mean_metrics']['exact_f1'])
    d.verify(protocol)
    result=dict(schema_version='rejection-calibration-summary.v1',run=str(RUN),protocol=protocol,folds=entries,held_pages=held_pages,aggregate=aggregate,guard=guard,verification=dict(inputs_sources_active_files_unchanged=True,own_fold_heads_used_for_train_calibration=True,all_train_pages_after_component_closure=True,held_pages_enter_after_selection=True,baseline_readout_reproduced=True,rematching_after_each_threshold=True,no_optimization=True,no_dev_test_scoring=True),elapsed_seconds=time.monotonic()-started,activated=False,optimizer_updates=0,locked_test_scored=False)
    d.m.write_json(RUN/'results.json',result);d.m.write_json(OUT,result)
    print(json.dumps(dict(aggregate={m:{p:dict(correct=[z['counts']['correct'] for z in a['seeds'].values()],accepted=[z['counts']['accepted'] for z in a['seeds'].values()],rare=[z['counts']['rare_correct'] for z in a['seeds'].values()],mean=a['mean_metrics']) for p,a in policies.items()} for m,policies in aggregate.items()},guard=guard,seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:print(json.dumps(check()))
    else:main()


"""Adaptive nested readout: historic rare branch keeps .35, calibrate other outputs."""
import argparse,json,statistics,time
from pathlib import Path
from scripts.experiments import research_nested_rejection as n
from scripts.experiments import research_rejection_calibration as r
f=n.f;d=n.d;ROOT=n.ROOT
RUN=ROOT/'backend/training_runs/20261004-nested-branch-rejection-v1'
OUT=ROOT/'reports/nested-branch-rejection-results-20261004.json'


def score(gt,ignore,rows,threshold):
    accepted=[z for z in rows if z['similarity']>=(.35 if z.get('fallback',False) else threshold)]
    result=f.score_page(gt,ignore,accepted,None)
    result['counts'].update(generated=len(rows),rejected=len(rows)-len(accepted),fallback_all=sum(z.get('fallback',False) for z in rows))
    return result


def calibrate(pages):
    curve=[]
    for threshold in r.GRID:
        ev=[score(p['gt'],p['ignore'],p['methods']['gated']['predictions'],threshold) for p in pages]
        counts={k:sum(z['counts'][k] for z in ev) for k in ev[0]['counts']}
        curve.append(dict(threshold=threshold,counts=counts,metrics=f.metrics(counts),page_mean_f1=statistics.mean(f.metrics(z['counts'])['exact_f1'] for z in ev)))
    best=max(z['page_mean_f1'] for z in curve)
    chosen=min((z for z in curve if z['page_mean_f1']>=best-1e-12),key=lambda z:z['threshold'])
    return dict(threshold=chosen['threshold'],objective=chosen['page_mean_f1'],curve=curve)


def check():
    n.check();gt=[dict(bbox=[0,0,10,10],label=0,natural_support=2)]
    rows=[dict(bbox=[0,0,10,10],prediction=0,similarity=.5,proposal_index=0,fallback=True),dict(bbox=[0,0,10,10],prediction=1,similarity=.4,proposal_index=1,fallback=False)]
    assert score(gt,[],rows,.35)==f.score_page(gt,[],rows,.35)
    e=score(gt,[],rows,.8);assert e['counts']['correct']==1 and e['counts']['rejected']==1 and e['counts']['generated']==2
    rows[0]['similarity']=.3;assert score(gt,[],rows,.8)['counts']['accepted']==0
    rows[0]['similarity']=.5
    p=dict(gt=gt,ignore=[],methods={'gated':dict(predictions=rows)})
    assert calibrate([p])['threshold']==.45
    return dict(reference035_exactly_reproduced=True,rare_branch_keeps035_not_force_acceptance=True,nonrare_filter_and_rematching=True,tie_lower_threshold=True)


def main():
    checks=check();prior=d.m.read_json(n.OUT);d.verify(prior['protocol'])
    paths=[Path(__file__),n.OUT,n.RUN/'preregistration.json']+list(n.RUN.glob('paired_*_step1000.pt'))
    assert len(list(n.RUN.glob('paired_*_step1000.pt')))==15
    protocol=dict(schema_version='nested-branch-rejection.v1',input_sha256=prior['protocol']['input_sha256']|{str(p):d.m.sha256(p) for p in paths},adaptive=True,activated=False,optimizer_updates=0,locked_test_scored=False,trigger='Nested primary loses rares; guarded scalar falls back to035 on12/15heads due absent calibration rares. Same already examined pages, no independent confirmation.',rule='Only gated method: fallback=True outputs retain historical threshold035; others use one scalar selected on own3 fit-excluded calibration pages, grid035..095 step005, maximize mean per-page exact F1, tieswithin1e-12 lower. Flag is pre-existing historic top1 raw-fit support1..4; never held GT.',heads='Same15 nested checkpoints, frozen; no fit, no pixel extraction, no change to existing readout or prior rules',reference='Same nested gated at035; nested unconstrained and scalar rare-guard policies remain reported separately',scope='Each head threshold chosen and saved before reading its outer pages; all inherited component/pixel closures retained; no dev/test scoring; adaptive exploratory readout only.',check=checks)
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol)
    started=time.monotonic();entries=[];outer=[]
    for e in prior['folds']:
        for p in e['calibration_pages']:
            rows=p['methods']['gated']['predictions'];assert score(p['gt'],p['ignore'],rows,.35)==f.score_page(p['gt'],p['ignore'],rows,.35)
        selected=calibrate(e['calibration_pages'])
        entry=dict(fold=e['fold'],seed=e['seed'],selection=selected,checkpoint_sha256=e['checkpoint_sha256'])
        d.m.write_json(RUN/f'selection_fold{e["fold"]}_seed{e["seed"]}.json',entry)
        current=[p for p in prior['outer_pages'] if p['fold']==e['fold'] and p['seed']==e['seed']]
        for p in current:
            rows=p['methods']['gated']['predictions'];before=p['methods']['gated']['evaluations']['current_035']
            assert score(p['gt'],p['ignore'],rows,.35)==before
            after=score(p['gt'],p['ignore'],rows,selected['threshold'])
            old={z['gt_index'] for z in before['matches'] if z['correct']};new={z['gt_index'] for z in after['matches'] if z['correct']}
            outer.append(dict(group=p['group'],fold=p['fold'],seed=p['seed'],gt=p['gt'],ignore=p['ignore'],methods={'gated':dict(predictions=rows,evaluations={'current_035':before,'rare_branch':after},delta={'current_035':dict(lost_correct_gt_indices=[],gained_correct_gt_indices=[]),'rare_branch':dict(lost_correct_gt_indices=sorted(old-new),gained_correct_gt_indices=sorted(new-old))})}))
        entries.append(entry)
        print('BRANCH',e['fold'],e['seed'],selected['threshold'],flush=True)
    before=n.aggregate(outer,'gated','current_035');after=n.aggregate(outer,'gated','rare_branch')
    assert before==prior['aggregate']['gated']['current_035']
    guard=dict(rare_correct_no_regression=all(after['seeds'][str(s)]['counts']['rare_correct']>=before['seeds'][str(s)]['counts']['rare_correct'] for s in n.SEEDS),rare_class_macro_no_regression=all(after['seeds'][str(s)]['metrics']['rare_class_macro']>=before['seeds'][str(s)]['metrics']['rare_class_macro'] for s in n.SEEDS),reference_rare_no_regression=all(after['seeds'][str(s)]['reference_rare']['correct']>=before['seeds'][str(s)]['reference_rare']['correct'] for s in n.SEEDS),precision_gain=after['mean_metrics']['exact_precision']>before['mean_metrics']['exact_precision'],f1_gain=after['mean_metrics']['exact_f1']>before['mean_metrics']['exact_f1'])
    d.verify(protocol)
    result=dict(schema_version='nested-branch-rejection-summary.v1',run=str(RUN),protocol=protocol,folds=entries,outer_pages=outer,aggregate={'current_035':before,'rare_branch':after},guard=guard,verification=dict(inputs_active_sources_checkpoints_unchanged=True,no_training_or_extraction=True,own_fit_excluded_calibration_only=True,selection_saved_before_outer=True,reference035_reproduced=True,matching_and_counts_recomputed=True),optimizer_updates=0,activated=False,locked_test_scored=False,elapsed_seconds=time.monotonic()-started)
    d.m.write_json(RUN/'results.json',result);d.m.write_json(OUT,result)
    print(json.dumps(dict(correct=[z['counts']['correct'] for z in after['seeds'].values()],rare=[z['counts']['rare_correct'] for z in after['seeds'].values()],lost=[z['lost_correct'] for z in after['seeds'].values()],mean=after['mean_metrics'],guard=guard,seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:print(json.dumps(check()))
    else:main()

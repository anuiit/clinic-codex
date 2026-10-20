"""Summarize initialization control; never select best-dev for conclusions."""
import collections,csv,json,statistics
from pathlib import Path
import torch
from scripts import summarize_base_corpus_long as summary
RUN=summary.ROOT/'backend/training_runs/20261004-base-init-control-v1'
OUT=summary.ROOT/'reports/base-init-control-results-20261004.json'


def main():
    p=summary.read(RUN/'preregistration.json');d=summary.read(RUN/'results.json')
    assert all(summary.sha(Path(k))==v for k,v in p['input_sha256'].items())
    results=d['results'];unique={r['training_artifact_dir'] for r in results}
    assert len(unique)==d['unique_fit_count']==(6 if d['extended'] else 2)
    assert len(results)==(10 if d['extended'] else 2)
    cache=torch.load(summary.CACHE,weights_only=True,map_location='cpu');index=cache['row_ids']
    meta={r['row_id']:r for r in csv.DictReader((summary.SNAPSHOT/'metadata.csv').open())}
    warm=summary.read(summary.OUT)
    for r in results:
        prior=summary.read(summary.RUN/f"{r['plan']['scope']}-{r['plan']['arm']}-seed42/result.json")
        assert r['episode_plan_sha256']==prior['episode_plan_sha256']
        assert r['plan']==prior['plan'] and len(r['history'])==100 and r['updates']==50000
        assert r['row_exposures']==prior['row_exposures']
        folder=Path(r['training_artifact_dir']);assert r['checkpoint_sha256']=={cp.name:summary.sha(cp) for cp in folder.glob('*.pt')}
    by_arm={arm:[r for r in results if r['plan']['arm']==arm] for arm in ['parent','combined']}
    first={}
    for arm,rs in by_arm.items():
        r=rs[0];e=r['milestones']['100']['evaluation']
        first[arm]=dict(raw=e['raw'],masked=e['masked'],page_mean_joint=statistics.mean(x['metrics']['top1_correct']/x['known'] for x in e['pages']),
            pages=e['pages'],raw_predictions=e['raw_predictions'],final_train_loss=r['history'][-1]['train_loss'])
    primary=None;curves=None
    if d['extended']:
        primary={arm:summary.cv_summary(rs,meta,index,100) for arm,rs in by_arm.items()}
        primary['paired_enriched_vs_parent']=summary.paired(primary['combined'],primary['parent'])
        primary['paired_enriched_vs_warm_enriched']=summary.paired(primary['combined'],warm['primary']['combined'])
        primary['paired_enriched_vs_historical']=summary.paired(primary['combined'],warm['historical'])
        primary['guard_vs_parent']=summary.guards(primary['combined'],primary['parent'],by_arm['combined'],by_arm['parent'])
        warm_records=[summary.read(summary.RUN/f"cv{i}-combined-seed42/result.json") for i in range(5)]
        primary['guard_vs_warm_enriched']=summary.guards(primary['combined'],warm['primary']['combined'],by_arm['combined'],warm_records)
        hist_records=[dict(plan=r['plan'],milestones={'100':{'evaluation':e}}) for r,e in zip(by_arm['combined'],d['historical'])]
        primary['guard_vs_historical']=summary.guards(primary['combined'],warm['historical'],by_arm['combined'],hist_records)
        curves={arm:{str(epoch):summary.cv_summary(rs,meta,index,epoch) for epoch in sorted(map(int,rs[0]['milestones']))} for arm,rs in by_arm.items()}
    out=dict(schema_version='base-init-control-summary.v1',initialization='Fresh standard ProjectionHead, seed42; pretrained DINO features unchanged',
        distinct_fit_count=len(unique),comparisons=len(results),optimizer_updates=50000*len(unique),primary_checkpoint='fixed_epoch100',
        first_fold=first,primary_all_folds=primary,curves=curves,extension=summary.read(RUN/'extension-decision.json'),
        verification=dict(input_hashes_unchanged=True,checkpoint_hashes_passed=True,episode_plans_identical=True,row_exposures_identical=True,active_model_unchanged=True),
        limitation='Diagnostic extension gate registered after warm first-fold observation, before any fresh fit; previously inspected corpus, not blind hypothesis test',
        activated=False,locked_test_scored=False)
    summary.write(OUT,out);print(json.dumps(dict(extended=d['extended'],fits=len(unique),first_fold={a:dict(raw=e['raw']['top1_correct'],masked=e['masked']['top1_correct']) for a,e in first.items()},
        primary=None if primary is None else {a:dict(raw=primary[a]['raw']['correct'],masked=primary[a]['masked_correct'],page_mean=primary[a]['page_mean_joint']) for a in by_arm}),indent=2))


if __name__=='__main__':main()

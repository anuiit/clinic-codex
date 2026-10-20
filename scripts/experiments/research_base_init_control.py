"""Bounded initialization control, identical base recipe and cached inputs."""
import json,time
from pathlib import Path
import torch
from scripts.experiments import research_base_corpus_long as base
from scripts.train_segmentation import read_json,write_json,sha256
RUN=base.core.ROOT/'backend/training_runs/20261004-base-init-control-v1'
WARM=base.RUN
THRESHOLD=.10


def main():
    torch.set_num_threads(4);d=base.setup();base.configure_determinism(42)
    fresh=base.ProjectionHead().state_dict();initial=RUN.with_suffix('.initial_projection.pt')
    if initial.exists():
        saved=torch.load(initial,weights_only=True,map_location='cpu')
        assert saved.keys()==fresh.keys() and all(torch.equal(saved[k],fresh[k]) for k in fresh)
    else:torch.save(fresh,initial)
    d['init']=torch.load(initial,weights_only=True,map_location='cpu')
    d['extra_paths'] += [Path(__file__),initial,WARM/'cv0-parent-seed42/result.json',WARM/'cv0-combined-seed42/result.json']
    base.RUN=RUN;base.SEEDS=[42];base.register(d)
    p=read_json(RUN/'preregistration.json');p['plans']=[x for x in p['plans'] if x['scope'].startswith('cv')]
    p.update(schema_version='base-init-control.v1',initial_projection=str(initial),
        initialization='Standard newly initialized ProjectionHead, CPU torch seed42; pretrained DINO frozen',
        primary='Diagnostic initialization comparison at fixed epoch100; no head or hyperparameter selection',
        secondary='Best-dev recorded separately, no full-corpus or locked-test scoring in this control',
        total_comparisons=len(p['plans']),unique_fits=len({(x['seed'],x['train_row_ids_sha256']) for x in p['plans']}),
        staged_extension=dict(first_fold='cv0',absolute_masked_top1_difference_threshold=THRESHOLD,
            criterion='If either fresh arm differs from its corresponding warm arm by at least10pp, extend both arms to all5 folds',
            warning='Gate was registered after warm first-fold inspection, before any fresh fit; exploratory diagnostic, not blind selection'),
        max_unique_fits=6,locked_test_scored=False)
    assert p['unique_fits']<=p['max_unique_fits']==6
    write_json(RUN/'preregistration.json',p);base.verify_inputs()
    historical=[]
    for held in base.mechanism.folds():
        ids,_=base.train_indices(d,'combined',held)
        historical.append(base.mechanism.evaluate(d,d['original'],d['old'],held,ids))
    results=[base.fit(d,x) for x in p['plans'] if x['scope']=='cv0'];base.verify_inputs()
    differences={}
    for r in results:
        arm=r['plan']['arm'];warm=read_json(WARM/f'cv0-{arm}-seed42/result.json')
        assert r['episode_plan_sha256']==warm['episode_plan_sha256']
        differences[arm]=r['milestones']['100']['evaluation']['masked']['top1_micro']-warm['milestones']['100']['evaluation']['masked']['top1_micro']
    extended=any(abs(v)>=THRESHOLD for v in differences.values())
    write_json(RUN/'extension-decision.json',dict(differences=differences,threshold=THRESHOLD,extended=extended,metric='fixed_epoch100_masked_top1',episode_plans_identical=True))
    if extended:
        for x in p['plans']:
            if x['scope']=='cv0':continue
            r=base.fit(d,x)
            warm=read_json(WARM/f"{x['scope']}-{x['arm']}-seed42/result.json")
            assert r['episode_plan_sha256']==warm['episode_plan_sha256']
            results.append(r);base.verify_inputs()
    unique_fit_count=len({r['training_artifact_dir'] for r in results})
    assert unique_fit_count==(6 if extended else 2) and unique_fit_count<=p['max_unique_fits']
    write_json(RUN/'results.json',dict(schema_version='base-init-control.v1',results=results,historical=historical,unique_fit_count=unique_fit_count,
        extended=extended,locked_test_scored=False,verification=dict(input_hashes_unchanged=True,active_model_unchanged=True,episode_plans_equal_to_warm=True),activated=False))
    print('CONTROL_DONE',len(results),'comparisons','extended',extended,flush=True)


if __name__=='__main__':main()

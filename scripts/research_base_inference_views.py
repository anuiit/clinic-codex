"""Fixed-checkpoint inference views: identical SAM proposals, no training or new DINO/SAM runs."""
import csv,json,statistics,hashlib
from pathlib import Path
import torch
from scripts import summarize_base_corpus_long as summary
from scripts.research_recognition_stability import aligned_prototypes
from codex_pipeline.models.projection_head import ProjectionHead
ROOT=summary.ROOT
OUT=ROOT/'reports/base-inference-views-results-20261004.json'


def main():
    torch.set_num_threads(4)
    warm=summary.read(summary.RUN/'results.json');cache=torch.load(summary.CACHE,weights_only=True,map_location='cpu')
    fixture_path=ROOT/'backend/training_runs/20261003-recognition-mechanisms-v2/crop-fixture.pt'
    manifest=summary.read(fixture_path.parent/'crop-manifest.json');assert summary.sha(fixture_path)==manifest['fixture_sha256']
    f=torch.load(fixture_path,weights_only=True,map_location='cpu');lookup={k:i for i,k in enumerate(cache['row_ids'])}
    labels=cache['labels'][[lookup[r['row_id']] for r in f['rows']]]
    runtime=ROOT/'backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime'
    historical=dict(model_state_dict=torch.load(runtime/'weights/projection.pt',weights_only=True,map_location='cpu'),
        prototypes=aligned_prototypes(torch.load(runtime/'weights/prototypes.pt',weights_only=True,map_location='cpu'),cache['class_names']))
    families={'warm_parent':[r for r in warm['results'] if r['plan']['arm']=='parent' and r['plan']['scope'].startswith('cv')],
        'warm_combined':[r for r in warm['results'] if r['plan']['arm']=='combined' and r['plan']['scope'].startswith('cv')]}
    cold_path=ROOT/'backend/training_runs/20261004-base-init-control-v1/results.json'
    if cold_path.exists():
        cold=summary.read(cold_path)
        if cold['extended']:
            families.update({f'fresh_{arm}':[r for r in cold['results'] if r['plan']['arm']==arm] for arm in ['parent','combined']})
    families['historical']=families['warm_combined']
    out={};verification=[]
    for family,records in families.items():
        views={v:dict(correct=0,count=0,pages=[]) for v in f['features']}
        model=ProjectionHead()
        for r in records:
            cp=historical if family=='historical' else torch.load(Path(r['training_artifact_dir'])/'epoch100.pt',weights_only=True,map_location='cpu')
            if family!='historical':assert summary.sha(Path(r['training_artifact_dir'])/'epoch100.pt')==r['checkpoint_sha256']['epoch100.pt']
            model.load_state_dict(cp['model_state_dict']);model.eval()
            held=r['plan']['held_groups'];ids=[i for i,row in enumerate(f['rows']) if row['split']=='train' and row['source_group'] in held]
            pages={p['group']:p['known'] for p in r['milestones']['100']['evaluation']['pages']}
            for view,features in f['features'].items():
                with torch.no_grad():pred=(model(features[ids])@cp['prototypes'].T).argmax(1)
                correct=pred==labels[ids];views[view]['correct']+=int(correct.sum());views[view]['count']+=len(ids)
                for g in held:
                    js=[j for j,i in enumerate(ids) if f['rows'][i]['source_group']==g]
                    value=int(correct[js].sum());views[view]['pages'].append(dict(group=g,known=pages[g],matched=len(js),correct=value,joint=value/pages[g]))
                if view=='mask':
                    expected=warm['historical']['cv'][int(r['plan']['scope'][2:])]['masked'] if family=='historical' else r['milestones']['100']['evaluation']['masked']
                    assert int(correct.sum())==expected['top1_correct'] and len(ids)==expected['count']
        for view,e in views.items():
            assert e['count']==254 and len(e['pages'])==17 and len({p['group'] for p in e['pages']})==17
            e.update(known=343,conditional_accuracy=e['correct']/254,joint_micro=e['correct']/343,page_mean_joint=statistics.mean(p['joint'] for p in e['pages']))
        out[family]=views
    summary.write(OUT,dict(schema_version='base-inference-views.v1',families=out,activated=False,
        paired_warm_rectangle_vs_historical_mask=summary.paired(out['warm_combined']['rectangle'],out['historical']['mask']),
        primary_checkpoint='epoch100_fixed',fixture_sha256=summary.sha(fixture_path),
        verification=dict(mask_readouts_equal_to_original=True,checkpoint_hashes_passed=True,held_groups_excluded_during_training=True),
        scope='5 business held-page folds seed42; same proposals and pretrained DINO, 4 previously audited saved crop-view features; no optimization or new image extraction',
        limitation='Exploratory view comparison on previously inspected pages; no automatic view selection or activation',fresh_extended_included='fresh_combined' in out))
    print(json.dumps({a:{v:dict(correct=e['correct'],page_mean=e['page_mean_joint']) for v,e in vs.items()} for a,vs in out.items()},indent=2))


if __name__=='__main__':main()

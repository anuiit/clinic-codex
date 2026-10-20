"""Exploratory full-proposal OOF readout; frozen weights and current rejection."""
import collections,csv,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.experiments import research_crop_compatible as d
from scripts.evaluate_glyph_benchmark import Box,_greedy_match,box_iou
from codex_model.classifier import CodexClassifier
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter
ROOT=d.core.ROOT
RUN=ROOT/'backend/training_runs/20261004-full-proposal-readout-v2'
SOURCE_RUN=ROOT/'backend/training_runs/20261004-full-proposal-readout-v1'
OUT=ROOT/'reports/full-proposal-results-20261004.json'
SEEDS=[42,43,44]
METHODS=['baseline','paired','gated']

def box(bbox):
    x,y,w,h=bbox
    return {'box':Box(float(x),float(y),float(x+w),float(y+h))}

def score_page(gt,ignore,rows,threshold):
    # ponytail: reuse geometric greedy matching; tied IoUs follow stable input indices.
    accepted=[r for r in rows if threshold is None or r['similarity']>=threshold]
    matches=_greedy_match([box(r['bbox']) for r in gt],[box(r['bbox']) for r in accepted],.5)
    matched={p for g,p,iou in matches}
    ignored={j for j,r in enumerate(accepted) if j not in matched and any(box_iou(box(r['bbox'])['box'],box(z['bbox'])['box'])>=.5 for z in ignore)}
    unmatched=[j for j in range(len(accepted)) if j not in matched|ignored]
    duplicates=sum(any(box_iou(box(accepted[j]['bbox'])['box'],box(z['bbox'])['box'])>=.5 for z in gt) for j in unmatched)
    correct=sum(gt[g]['label']==accepted[p]['prediction'] for g,p,iou in matches)
    rare=[(j,z) for j,z in enumerate(gt) if 1<=z['natural_support']<=4]
    by_gt={g:p for g,p,iou in matches}
    rare_correct=sum(j in by_gt and z['label']==accepted[by_gt[j]]['prediction'] for j,z in rare)
    counts=dict(known=len(gt),ignore_regions=len(ignore),generated=len(rows),accepted=len(accepted),rejected=len(rows)-len(accepted),ignored_predictions=len(ignored),scored_predictions=len(accepted)-len(ignored),matched=len(matches),correct=correct,wrong_name=len(matches)-correct,unmatched_predictions=len(unmatched),duplicate_predictions=duplicates,unmatched_other=len(unmatched)-duplicates,missed=len(gt)-len(matches),rare_known=len(rare),rare_correct=rare_correct,fallback_all=sum(r.get('fallback',False) for r in rows),fallback_accepted=sum(r.get('fallback',False) for r in accepted))
    assert counts['scored_predictions']==counts['matched']+counts['unmatched_predictions']
    assert counts['known']==counts['matched']+counts['missed']
    return dict(counts=counts,matches=[dict(gt_index=g,proposal_index=accepted[p]['proposal_index'],iou=iou,correct=gt[g]['label']==accepted[p]['prediction']) for g,p,iou in matches],ignored_proposal_indices=[accepted[j]['proposal_index'] for j in sorted(ignored)],unmatched_proposal_indices=[accepted[j]['proposal_index'] for j in unmatched])

def metrics(c):
    def ratio(a,b):return a/b if b else 0.
    def f1(p,r):return 2*p*r/(p+r) if p+r else 0.
    dp=ratio(c['matched'],c['scored_predictions']);dr=ratio(c['matched'],c['known'])
    ep=ratio(c['correct'],c['scored_predictions']);er=ratio(c['correct'],c['known'])
    return dict(detection_precision=dp,detection_recall=dr,detection_f1=f1(dp,dr),exact_precision=ep,exact_recall=er,exact_f1=f1(ep,er),conditional_top1=ratio(c['correct'],c['matched']),rare_joint_recall=ratio(c['rare_correct'],c['rare_known']))

def check():
    gt=[dict(bbox=[0,0,10,10],label=0,natural_support=2)]
    ignored=[dict(bbox=[30,0,10,10])]
    rows=[dict(bbox=b,prediction=l,similarity=s,proposal_index=i) for i,(b,l,s) in enumerate([([0,0,10,10],1,.5),([0,0,10,10],0,.6),([30,0,10,10],0,.8),([60,0,10,10],0,.2)])]
    a=score_page(gt,ignored,rows,None)['counts']
    assert (a['matched'],a['ignored_predictions'],a['unmatched_predictions'],a['duplicate_predictions'])==(1,1,2,1)
    assert a['correct']==1 # equal-IoU stable higher proposal index, independent of labels
    rows[1]['prediction']=1
    assert score_page(gt,ignored,rows,None)['counts']['correct']==0
    b=score_page(gt,ignored,rows,.35)['counts'];assert b['rejected']==1 and b['ignored_predictions']==1 and b['scored_predictions']==2
    c=score_page(gt,ignored,rows,.9)['counts'];assert c['missed']==1 and c['scored_predictions']==0
    assert metrics(c)['exact_f1']==0

def ground_truth(data):
    snapshot=d.m.read_json(d.core.SNAPSHOT/'snapshot_manifest.json')
    imported=d.m.read_json(ROOT/'reports/annotation-import-applied-20261003.json')
    labels={name:int(label) for label,name in data['cache']['class_names'].items()}
    active=collections.defaultdict(dict)
    for z in snapshot['rows']:
        if z['dataset_split']=='train' and z['source_group'] in data['business'] and z.get('bbox'):
            active[z['source_group']][tuple(z['bbox'])]=z
    pages=[]
    for p in data['app']['full_pages']['pages']:
        if p['split']!='train':continue
        group=p['source_group'];geometries=collections.defaultdict(list)
        for r in imported['rows']:
            if 'image-pixels:'+r['source_image_pixel_sha256']==group and r['archive_decision']['status']=='approved':geometries[tuple(r['bbox'])].append(r)
        assert set(active[group])<=set(geometries)
        gt=[];ignore=[]
        for coords,rows in sorted(geometries.items()):
            names={r['class_name'] for r in rows}
            if len(names)!=1 or not names<=set(labels):
                assert coords not in active[group]
                ignore.append(dict(bbox=list(coords),names=sorted(names),reason='conflict' if len(names)>1 else 'unknown_class'))
            else:
                name=next(iter(names));z=active[group].get(coords)
                if z:assert z['class_name']==name
                gt.append(dict(bbox=list(coords),label=labels[name],class_name=name,row_id=z['row_id'] if z else None,annotation_keys=[r['key'] for r in rows],in_active_train=z is not None))
        assert sum(z['in_active_train'] for z in gt)==p['ground_truth_count']
        pages.append(dict(group=group,image=p['source_image_path'],image_sha256=d.m.sha256(Path(p['source_image_path'])),gt=gt,ignore=ignore,expected_proposals=p['proposal_count']))
    assert len(pages)==17 and sum(z['in_active_train'] for p in pages for z in p['gt'])==343
    return pages

def main():
    check();data=d.base.setup();fixture=data['fixture']=d.m.load_fixture(data);prior=d.m.read_json(d.RUN/'results.json');old_protocol=d.m.read_json(d.RUN/'preregistration.json');d.verify(old_protocol)
    readout=d.m.read_json(ROOT/'reports/rare-readout-results-20261004.json');rare_protocol=d.m.read_json(Path(readout['run'])/'preregistration.json');d.verify(rare_protocol)
    threshold=d.m.read_json(d.core.RUNTIME/'config.json')['rejection_threshold'];assert threshold==.35
    pages=ground_truth(data)
    paths=[SOURCE_RUN/'full-proposal-features.pt',SOURCE_RUN/'preregistration.json',SOURCE_RUN/'runner-source.py',SOURCE_RUN/'aborted.json',Path(__file__),ROOT/'scripts/evaluate_glyph_benchmark.py',ROOT/'scripts/research_rare_readout.py',ROOT/'reports/annotation-import-applied-20261003.json',ROOT/'reports/rare-readout-results-20261004.json']+[Path(p['image']) for p in pages]+list(d.RUN.glob('paired_*_step1000.pt'))
    protocol=dict(schema_version='full-proposal-readout.v1',input_sha256=old_protocol['input_sha256']|rare_protocol['input_sha256']|{str(p):d.m.sha256(p) for p in paths},adaptive=True,optimizer_updates=0,activated=False,locked_test_scored=False,seeds=SEEDS,folds=5,methods=METHODS,thresholds=[None,threshold],threshold_selection='none; 0.35 from historical config, plus no-rejection diagnostic; no transfer calibration claim',sam=dict(points_per_side=16,points_per_batch=4,padding=5),primary_geometry=dict(iou=.5,matching='existing geometry-only greedy, descending IoU then stable gt/proposal index tie-break; rematch after rejection; known GT first then ignore unmatched overlaps',duplicates='unmatched accepted proposals with IoU>=0.5 against any known GT'),ground_truth=dict(exhaustive='human confirmed all glyphs on each page',active_known=343,known=sum(len(p['gt']) for p in pages),ignored_regions=sum(len(p['ignore']) for p in pages),extra_known=sum(not z['in_active_train'] for p in pages for z in p['gt']),policy='all distinct approved archive geometries; deduplicated train crops still count as objects at distinct locations; conflict and out-of-taxonomy regions ignored, not background'),scope='Same previously examined17 train pages, OOF adaptation only, historical training unknown; exploratory and not independent; dev/test pages not processed')
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol);d.m.write_json(RUN/'ground-truth.json',dict(pages=pages))
    started=time.monotonic()
    previous=d.m.read_json(SOURCE_RUN/'preregistration.json');aborted=d.m.read_json(SOURCE_RUN/'aborted.json')
    assert d.m.sha256(SOURCE_RUN/'runner-source.py')==previous['input_sha256'][str(Path(__file__))]==aborted['source_sha256']
    assert d.m.sha256(SOURCE_RUN/'full-proposal-features.pt')==aborted['full_proposal_features_sha256']
    for path,digest in previous['input_sha256'].items():
        if path!=str(Path(__file__)):assert d.m.sha256(Path(path))==digest
    saved=torch.load(SOURCE_RUN/'full-proposal-features.pt',weights_only=True,map_location='cpu')
    new_pages=saved['pages'];all_features=[saved['features']]
    assert len(new_pages)==17 and saved['features'].shape==(654,384)
    for p,q in zip(pages,new_pages):
        assert all(p[k]==q[k] for k in p)
        assert len(q['proposals'])==p['expected_proposals']
        by_box={tuple(z['bbox']):z for z in q['proposals']};assert len(by_box)==len(q['proposals'])
        replayed=0
        for j,r in enumerate(fixture['rows']):
            if r['source_group']!=p['group']:continue
            z=by_box[tuple(r['proposal_bbox'])]
            assert z['crop_sha256']==d.m.read_json(d.m.RUN/'crop-manifest.json')['rows'][j]['view_sha256']['mask']
            assert torch.allclose(saved['features'][q['proposal_offset']+z['proposal_index']],fixture['features']['mask'][j],atol=1e-4,rtol=1e-4)
            replayed+=1
        assert replayed==q['replayed_matched']
    assert sum(p['replayed_matched'] for p in new_pages)==254
    print('VERIFIED FULL PROPOSAL CACHE',654,'proposals;254 matched crops replayed',flush=True)
    features=torch.cat(all_features).cuda();entries=[]
    metadata={r['row_id']:r for r in csv.DictReader((d.core.SNAPSHOT/'metadata.csv').open())}
    for e in prior['folds']:
        fold,seed=e['fold'],e['seed'];held=set(e['held'])
        raw,_,_,_,_=d.scope(data,e['held']);assert raw==e['selected_raw_ids']
        pools=collections.defaultdict(set)
        for i in raw:
            z=metadata[data['cache']['row_ids'][i]];pools[int(z['class_label'])].add(z['source_pixel_sha256'])
        model=d.core.ProjectionHead().cuda();cp=d.RUN/f'paired_fold{fold}_seed{seed}_step1000.pt'
        model.load_state_dict(torch.load(cp,weights_only=True,map_location='cpu')['projection']);model.eval()
        for p in new_pages:
            if p['group'] not in held:continue
            ids=list(range(p['proposal_offset'],p['proposal_offset']+len(p['proposals'])))
            with torch.no_grad():
                old=d.core.project(data['original'],features[ids])@data['old'].T
                new=d.core.project(model,features[ids])@data['old'].T
                oldlabel=old.argmax(1);flag=torch.tensor([1<=len(pools[c])<=4 for c in oldlabel.tolist()],device='cuda')
                scores=dict(baseline=old,paired=new,gated=torch.where(flag[:,None],old,new))
            gt=[dict(z,natural_support=len(pools[z['label']])) for z in p['gt']];methods={}
            for method,sc in scores.items():
                values,labels=sc.max(1);top=sc.topk(3,dim=1)
                rows=[dict(q,prediction=int(labels[j]),similarity=float(values[j]),fallback=bool(flag[j]) if method=='gated' else False,top3_labels=top.indices[j].tolist(),top3_similarities=top.values[j].tolist()) for j,q in enumerate(p['proposals'])]
                methods[method]=dict(predictions=rows,evaluations={key:score_page(gt,p['ignore'],rows,t) for key,t in [('no_rejection',None),('current_035',threshold)]})
                reference=next(z for z in readout['folds'] if z['fold']==fold and z['seed']==seed)['details']['mask']['predictions']
                expected=[z for z in reference if metadata[z['row_id']]['source_group']==p['group']]
                source_key='historical_prediction' if method=='baseline' else 'paired_prediction' if method=='paired' else 'prediction'
                by_bbox={tuple(q['bbox']):q for q in rows};fixture_boxes={r['row_id']:tuple(r['proposal_bbox']) for r in fixture['rows'] if r['source_group']==p['group']}
                assert all(by_bbox[fixture_boxes[z['row_id']]]['prediction']==z[source_key] for z in expected)
            entries.append(dict(fold=fold,seed=seed,group=p['group'],gt=gt,ignore=p['ignore'],methods=methods))
        print('FULL READOUT',fold,seed,flush=True)
    aggregate={}
    for method in METHODS:
        aggregate[method]={}
        for policy in ['no_rejection','current_035']:
            by_seed={}
            for seed in SEEDS:
                es=[e for e in entries if e['seed']==seed];assert len(es)==17
                c={k:sum(e['methods'][method]['evaluations'][policy]['counts'][k] for e in es) for k in es[0]['methods'][method]['evaluations'][policy]['counts']}
                by_seed[str(seed)]=dict(counts=c,metrics=metrics(c),page_mean_exact_recall=statistics.mean(e['methods'][method]['evaluations'][policy]['counts']['correct']/len(e['gt']) for e in es))
            aggregate[method][policy]=dict(seeds=by_seed,mean_metrics={k:statistics.mean(e['metrics'][k] for e in by_seed.values()) for k in next(iter(by_seed.values()))['metrics']})
    d.verify(protocol)
    result=dict(schema_version='full-proposal-summary.v1',run=str(RUN),protocol=protocol,aggregate=aggregate,fold_pages=entries,verification=dict(preregistered_sources_inputs_active_files_unchanged=True,sam_proposal_counts_and_unique_boxes_reproduced=True,all254_matched_pixels_features_and_predictions_reproduced=True,fold_supports_after_component_closure=True,no_optimization=True,no_dev_or_test_images_processed=True),feature_cache=str(SOURCE_RUN/'full-proposal-features.pt'),feature_sha256=d.m.sha256(SOURCE_RUN/'full-proposal-features.pt'),elapsed_seconds=time.monotonic()-started,activated=False,optimizer_updates=0,locked_test_scored=False)
    d.m.write_json(RUN/'results.json',result);d.m.write_json(OUT,result)
    print(json.dumps(dict(gt=protocol['ground_truth'],results={m:{p:dict(correct=[e['counts']['correct'] for e in a['seeds'].values()],accepted=[e['counts']['accepted'] for e in a['seeds'].values()],unmatched=[e['counts']['unmatched_predictions'] for e in a['seeds'].values()],wrong_name=[e['counts']['wrong_name'] for e in a['seeds'].values()],rare=[e['counts']['rare_correct'] for e in a['seeds'].values()],mean=a['mean_metrics']) for p,a in z.items()} for m,z in aggregate.items()}),indent=2))

if __name__=='__main__':main()

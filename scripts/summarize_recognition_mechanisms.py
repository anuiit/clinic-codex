#!/usr/bin/env python3
"""Derive paired summaries from immutable experiments; no model or test scoring."""
import collections,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'backend/training_runs/20261003-recognition-mechanisms-v2'
OUT=ROOT/'reports/recognition-mechanisms-results-20261003.json'
SEEDS=[42,43,44]

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def mean(items):return statistics.mean(items) if items else None

def classification(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['label']].append(int(r['label']==r['prediction']))
    return dict(count=len(rows),classes=len(groups),correct=sum(sum(v) for v in groups.values()),
        micro=mean([r['label']==r['prediction'] for r in rows]),macro=mean([mean(v) for v in groups.values()]))

def support_counts(rows):
    pools=collections.defaultdict(set)
    for r in rows:pools[int(r['class_label'])].add(r['source_pixel_sha256'])
    return {c:len(v) for c,v in pools.items()}

def bin_name(n,effective=False):
    if n==0:return 'zero'
    if effective and n<=4:return 'one' if n==1 else 'two' if n==2 else 'three_four'
    return 'rare1_4' if n<=4 else ('five_twenty' if effective else 'medium5_20') if n<=20 else 'many'

def aggregate(entries,meta,index):
    raw=[];pages=[];natural=collections.defaultdict(list);effective=collections.defaultdict(list)
    masked_correct=masked_count=known=0
    for entry,e in entries:
        full=[r for r in meta.values() if r['dataset_split']=='train' and r['source_group'] not in entry['held']]
        counts=support_counts(full)
        selected=[meta[index[i]] for i in entry['selected_raw_ids']] if 'selected_raw_ids' in entry else full
        actual=support_counts(selected)
        assert all(r['source_group'] not in entry['held'] and r['dataset_split']=='train' for r in selected)
        assert len({(r['class_label'],r['source_pixel_sha256']) for r in selected})==len(selected) or 'selected_raw_ids' not in entry
        bins=collections.defaultdict(list);effective_bins=collections.defaultdict(list)
        for r in e['raw_predictions']:
            m=meta[r['row_id']];assert int(m['class_label'])==r['label'] and m['source_group'] in entry['held']
            bins[bin_name(counts.get(r['label'],0))].append(r)
            effective_bins[bin_name(actual.get(r['label'],0),True)].append(r)
        for key,m in e['reference_availability'].items():
            c=classification(bins[key]);assert c['count']==m['count']
            if c['count']:assert c['correct']==m['top1_correct'] and abs(c['macro']-m['top1_macro'])<1e-6
            natural[key].extend(bins[key])
        for key,m in e['effective_support_bins'].items():
            c=classification(effective_bins[key]);assert c['count']==m['count']
            if c['count']:assert c['correct']==m['top1_correct'] and abs(c['macro']-m['top1_macro'])<1e-6
            effective[key].extend(effective_bins[key])
        raw.extend(e['raw_predictions']);masked_correct+=e['masked']['top1_correct'];masked_count+=e['masked']['count']
        for p in e['pages']:
            known+=p['known'];pages.append(dict(group=p['group'],known=p['known'],correct=p['metrics']['top1_correct'],matched=p['metrics']['count'],joint=p['metrics']['top1_correct']/p['known']))
    assert len(raw)==343 and len({r['row_id'] for r in raw})==343
    assert len(pages)==17 and len({p['group'] for p in pages})==17 and known==343 and masked_count==254
    return dict(raw=classification(raw),masked_correct=masked_correct,masked_count=masked_count,known=known,
        joint_micro=masked_correct/known,page_mean_joint=mean([p['joint'] for p in pages]),pages=pages,
        natural_support_bins={k:classification(v) for k,v in natural.items()},effective_support_bins={k:classification(v) for k,v in effective.items()})

def across_seeds(items):
    return dict(seeds={str(seed):value for seed,value in zip(SEEDS,items)},masked_correct=[e['masked_correct'] for e in items],
        raw_correct=[e['raw']['correct'] for e in items],mean_masked_correct=mean([e['masked_correct'] for e in items]),
        page_mean_joint=mean([e['page_mean_joint'] for e in items]),raw_macro=mean([e['raw']['macro'] for e in items]),
        rare_macro=mean([e['natural_support_bins']['rare1_4']['macro'] for e in items]),
        rare_count=items[0]['natural_support_bins']['rare1_4']['count'],rare_classes=items[0]['natural_support_bins']['rare1_4']['classes'])

def paired_pages(candidate,baseline):
    groups=sorted(p['group'] for p in candidate['seeds']['42']['pages']);delta=[]
    for group in groups:
        delta.append(mean([next(p['joint'] for p in candidate['seeds'][str(s)]['pages'] if p['group']==group)-next(p['joint'] for p in baseline['seeds'][str(s)]['pages'] if p['group']==group) for s in SEEDS]))
    rng=np.random.default_rng(20261003)
    bootstrap=np.mean(rng.choice(np.array(delta),size=(10000,len(delta)),replace=True),axis=1)
    return dict(page_deltas=dict(zip(groups,delta)),mean_delta=mean(delta),wins=sum(d>1e-10 for d in delta),
        ties=sum(abs(d)<=1e-10 for d in delta),losses=sum(d<-1e-10 for d in delta),
        bootstrap_95_interval=[float(v) for v in np.quantile(bootstrap,[.025,.975])],
        interpretation='descriptive paired resampling of 17 pages, seeds averaged before resampling; not an independent blind test')

def diagnostics(records,weight):
    history=[h for r in records for h in r['history']]
    active=sum(h['active_anchors'] for h in history);anchors=sum(h['anchors'] for h in history)
    gradients=[weight*h['contrast_gradient_norm']/h['ce_gradient_norm'] for h in history if h.get('ce_gradient_norm',0)>0]
    coverage=[]
    for r in records:
        hs=r['positive_count_histograms'];coverage.append(dict(classes_observed=len(hs),classes_with_positives=sum(any(int(k)>0 and n for k,n in hist.items()) for hist in hs.values())))
        assert sum(sum(hist.values()) for hist in hs.values())==sum(h['anchors'] for h in r['history'])
    last=[h for r in records for h in r['history'][-20:]]
    return dict(active_anchor_exposures=active,anchor_exposures=anchors,active_fraction=active/anchors,
        intra_page_negatives=sum(h['negatives_intrapage'] for h in history),inter_page_negatives=sum(h['negatives_interpage'] for h in history),
        weighted_to_ce_gradient_ratio_mean=mean(gradients),weighted_to_ce_gradient_ratio_range=[min(gradients),max(gradients)],
        final20_ce_mean=mean([h['ce'] for h in last]),final20_contrast_mean=mean([h['contrast'] for h in last]),
        final20_weighted_contrast_mean=mean([h['weighted_contrast'] for h in last]),class_coverage_per_fit=coverage)

def contrast_summary(contrast,protocol,meta,index,run):
    assert len(contrast['folds'])==15
    assert {(e['fold'],e['seed']) for e in contrast['folds']}=={(f,s) for f in range(5) for s in SEEDS}
    methods=['historical']+list(protocol['methods']);result={}
    for method in methods:
        result[method]=across_seeds([aggregate([(e,e['baseline'] if method=='historical' else e['methods'][method]['evaluation']) for e in contrast['folds'] if e['seed']==s],meta,index) for s in SEEDS])
    for method,value in result.items():
        value['paired_pages_vs_historical']=paired_pages(value,result['historical'])
        value['paired_pages_vs_ce']=paired_pages(value,result['ce_fixed'])
        if method!='historical':
            records=[e['methods'][method] for e in contrast['folds']]
            assert all(len(r['positive_count_histograms'])==286 for r in records)
            value['diagnostics']=diagnostics(records,protocol['methods'][method])
            fold_deltas={str(f):mean([e['methods'][method]['evaluation']['raw']['top1_micro']-e['baseline']['raw']['top1_micro'] for e in contrast['folds'] if e['fold']==f]) for f in range(5)}
            guards=dict(page_mean_improves=value['page_mean_joint']>result['historical']['page_mean_joint'],
                rare_macro_no_regression=value['rare_macro']>=result['historical']['rare_macro'],every_fold_raw_drop_at_most_2pp=min(fold_deltas.values())>=-.02)
            value['eligibility']=dict(guards=guards,eligible=all(guards.values()),raw_fold_mean_deltas=fold_deltas,
                interpretation='CV exploratory admissibility only, not an activation decision')
    assert len(list(run.glob('ce*_fold*_seed*.pt')))==15*len(protocol['methods'])
    return result

def main():
    protocol=read(RUN/'preregistration.json')
    assert all(sha(Path(p))==h for p,h in protocol['input_sha256'].items())
    snapshot=ROOT/'backend/training_corpus/snapshots/snapshot-611c734590f272060251'
    meta={r['row_id']:r for r in csv.DictReader((snapshot/'metadata.csv').open())}
    cache=torch.load(ROOT/'backend/model_registry/versions/20261003-annotation-lanczos-warmstart/training_data/precomputed/features.pt',weights_only=True,map_location='cpu')
    index=cache['row_ids']
    supports=read(RUN/'supports-results.json');assert len(supports['folds'])==45
    assert {(e['fold'],e['seed'],e['k']) for e in supports['folds']}=={(f,s,k) for f in range(5) for s in SEEDS for k in [1,2,4]}
    for e in supports['folds']:
        h=e['candidates']['historical'];assert h['raw_predictions']==e['baseline']['raw_predictions'] and h['masked']==e['baseline']['masked']
        for candidate in e['candidates'].values():
            for key in h['effective_support_bins']:assert h['effective_support_bins'][key]['count']==candidate['effective_support_bins'][key]['count']
    erratum=dict(field='supports-results.json folds[].baseline.effective_support_bins',
        explanation='The outer baseline is a full-availability reference. Its effective bins are not comparable with capped candidates.',
        comparable_baseline='folds[].candidates.historical, evaluated with exactly the same selected_raw_ids as every capped candidate',
        checks='All 45 entries: raw predictions and masked metrics equal the outer reference; candidate bin denominators match the capped historical baseline.',
        immutable_outputs_modified=False)
    write(RUN/'report-erratum.json',erratum)
    crops=read(RUN/'crop-results.json');crop_summary={}
    for name,e in crops['views'].items():
        crop_summary[name]=dict(splits=e['splits'],train_page_mean_joint=mean([p['metrics']['top1_correct']/p['known'] for p in e['pages'] if p['split']=='train']))
    out=dict(schema_version='recognition-mechanisms-summary.v2',activated=False,locked_test_scored=False,fit_steps=300,
        support_fit_count=45,contrast_fit_count=0,crops=crop_summary,crop_replay=crops['old_fixture_comparison']|dict(saved_pixel_replay_max_error=crops['replay_max_error']),supports={},contrast={},erratum=erratum,
        macro_definition='Pool held raw predictions over five folds separately per seed and method, then average accuracy over represented classes. Natural rare bins use eligible unique train pixels before any cap.',
        joint_definition='Correct label of an IoU-matched SAM proposal divided by all known annotated glyphs; misses count as failures. Page mean is the unweighted mean over 17 held pages.',
        limitation='Historical checkpoint provenance unknown; previously examined corpus; measures adaptation stability, not certified blind generalization.')
    for k in [1,2,4]:
        result={}
        for method in ['historical','centroid','shrink_tau8','learned_anchor']:
            result[method]=across_seeds([aggregate([(e,e['candidates'][method]) for e in supports['folds'] if e['seed']==s and e['k']==k],meta,index) for s in SEEDS])
        for method,value in result.items():value['paired_pages_vs_historical']=paired_pages(value,result['historical'])
        out['supports'][str(k)]=result
    if (RUN/'contrast-results.json').exists():
        out['contrast']=contrast_summary(read(RUN/'contrast-results.json'),protocol,meta,index,RUN)
        out['contrast_fit_count']=45
    batch_run=ROOT/'backend/training_runs/20261003-crosspage-batches-v3'
    out['paired_batches']={};out['paired_batches_fit_count']=0
    if (batch_run/'results.json').exists():
        recipe=read(batch_run/'preregistration.json');assert all(sha(Path(p))==h for p,h in recipe['input_sha256'].items())
        out['paired_batches']=contrast_summary(read(batch_run/'results.json'),recipe,meta,index,batch_run)
        out['paired_batches_fit_count']=30
        out['pairing_verification']=read(batch_run/'pairing-verification.json')
        out['paired_batches_artifact_sha256']={p.name:sha(p) for p in [batch_run/'preregistration.json',batch_run/'results.json',batch_run/'pairing-verification.json']}
    out['trained_runs']=out['support_fit_count']+out['contrast_fit_count']+out['paired_batches_fit_count']
    out['artifact_sha256']={p.name:sha(p) for p in [RUN/'preregistration.json',RUN/'crop-manifest.json',RUN/'crop-results.json',RUN/'supports-results.json']+([RUN/'contrast-results.json'] if out['contrast_fit_count'] else [])}
    out['analysis_source_sha256']=sha(Path(__file__))
    write(OUT,out)

    print(json.dumps(dict(trained_runs=out['trained_runs'],crops={k:v['splits']['train']['top1_correct'] for k,v in out['crops'].items()},
        supports={k:{m:dict(masked=v['masked_correct'],raw=v['raw_correct'],rare=round(v['rare_macro'],4),page_delta=round(v['paired_pages_vs_historical']['mean_delta']*100,3)) for m,v in methods.items()} for k,methods in out['supports'].items()},
        contrast={m:dict(masked=v['masked_correct'],raw=v['raw_correct'],rare=round(v['rare_macro'],4),page_delta=round(v['paired_pages_vs_historical']['mean_delta']*100,3),eligibility=v.get('eligibility')) for m,v in out['contrast'].items()}),ensure_ascii=False,indent=2))

if __name__=='__main__':
    assert classification([dict(label=0,prediction=0)]*9+[dict(label=1,prediction=0)])['macro']==.5
    assert [bin_name(n,True) for n in [0,1,2,3,4,5,20,21]]==['zero','one','two','three_four','three_four','five_twenty','five_twenty','many']
    main()






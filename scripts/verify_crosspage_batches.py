#!/usr/bin/env python3
"""CPU replay of immutable batch plans against recorded fit hashes and coverage."""
import collections,csv,json
from pathlib import Path
import torch
from scripts import research_crosspage_batches as b
m=b.m

def active(ids,d):
    pools=collections.defaultdict(list)
    for i in ids:pools[int(d['labels'][i])].append(i)
    return sum(any(d['groups'][i]!=d['groups'][j] and d['pixels'][i]!=d['pixels'][j] and d['objects'][i]!=d['objects'][j] for j in pools[int(d['labels'][i])]) for i in ids)

def raw_pairs(ids,d):
    pools=collections.defaultdict(list)
    for i in ids:
        if i<d['n']:pools[int(d['labels'][i])].append(i)
    return sum(len(js)==2 and d['groups'][js[0]]!=d['groups'][js[1]] and d['pixels'][js[0]]!=d['pixels'][js[1]] for js in pools.values())

def main():
    torch.set_num_threads(4)
    recipe=m.read_json(b.RUN/'preregistration.json')
    assert all(m.sha256(Path(p))==h for p,h in recipe['input_sha256'].items())
    cache=torch.load(m.core.CACHE,weights_only=True,map_location='cpu')
    rt=torch.load(m.core.PREVIOUS/'runtime_features.pt',weights_only=True,map_location='cpu')
    meta={r['row_id']:r for r in csv.DictReader((m.core.SNAPSHOT/'metadata.csv').open())}
    labels=torch.cat([cache['labels'],cache['labels'][[r['snapshot_index'] for r in rt['rows']]]]).long()
    objects=cache['row_ids']+[r['row_id'] for r in rt['rows']]
    d=dict(n=len(cache['row_ids']),labels=labels,groups=cache['source_groups']+[r['source_group'] for r in rt['rows']],
        splits=cache['dataset_splits']+[r['split'] for r in rt['rows']],pixels=[meta[r]['source_pixel_sha256'] for r in objects],objects=objects)
    result=m.read_json(b.RUN/'results.json');assert len(result['folds'])==15
    verification=[]
    for entry in result['folds']:
        train=[i for i,s in enumerate(d['splits']) if s=='train' and d['groups'][i] not in entry['held']]
        old=b.ORIGINAL_BATCHES(labels,train,d['n'],entry['seed'],300)
        new=b.paired_batches(d,train,entry['seed'],300)
        plan_hash=m.digest(torch.tensor([i for ids in new for i in ids],dtype=torch.int64).numpy())
        assert plan_hash==entry['batches_sha256']
        seen={int(labels[i]) for ids in new for i in ids};assert seen=={int(labels[i]) for i in train}==set(range(286))
        for before,after in zip(old,new):
            assert len(before)==len(after)==len(set(after))
            assert collections.Counter(int(labels[i]) for i in before)==collections.Counter(int(labels[i]) for i in after)
            assert [i for i in before if i>=d['n']]==[i for i in after if i>=d['n']]
            assert set(after)<=set(train)
        active_old=[active(ids,d) for ids in old];active_new=[active(ids,d) for ids in new]
        counts=collections.Counter(int(labels[i]) for ids in new for i in ids)
        for method,e in entry['methods'].items():
            assert [h['active_anchors'] for h in e['history']]==active_new
            assert {int(c):sum(hist.values()) for c,hist in e['positive_count_histograms'].items()}==dict(counts)
            ck=torch.load(b.RUN/f'{method}_fold{entry["fold"]}_seed{entry["seed"]}.pt',weights_only=True,map_location='cpu')
            assert ck['batches_sha256']==plan_hash and ck['train_ids']==train
        total=sum(len(ids) for ids in new)
        verification.append(dict(fold=entry['fold'],seed=entry['seed'],classes_supervised=len(seen),batch_plan_sha256=plan_hash,
            replacements=sum(a!=z for before,after in zip(old,new) for a,z in zip(before,after)),
            raw_qualified_pairs_before=sum(raw_pairs(ids,d) for ids in old),raw_qualified_pairs_after=sum(raw_pairs(ids,d) for ids in new),
            old_active_anchors=sum(active_old),new_active_anchors=sum(active_new),anchors=total,
            old_active_fraction=sum(active_old)/total,new_active_fraction=sum(active_new)/total))
    out=dict(source_and_inputs_unchanged=True,all_286_classes_supervised=True,all_held_dev_test_excluded=True,
        original_class_counts_and_sam_rows_preserved=True,recorded_batch_hashes_and_positive_coverage_reproduced=True,
        verification_source_sha256=m.sha256(Path(__file__)),fits=verification,
        old_active_fraction=sum(e['old_active_anchors'] for e in verification)/sum(e['anchors'] for e in verification),
        new_active_fraction=sum(e['new_active_anchors'] for e in verification)/sum(e['anchors'] for e in verification),
        total_replacements=sum(e['replacements'] for e in verification),activated=False)
    m.write_json(b.RUN/'pairing-verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='fits'},indent=2))

if __name__=='__main__':
    toy=dict(n=3,labels=torch.tensor([0,0,0]),groups=['a','b','a'],pixels=['x','y','x'],objects=['one','two','one'])
    assert active([0,1,2],toy)==3 and raw_pairs([0,1],toy)==1
    main()

#!/usr/bin/env python3
"""Follow-up: change cross-page pair availability, keep CE/contrast fits paired."""
import collections,json,time
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import research_recognition_mechanisms as m
RUN=m.core.ROOT/'backend/training_runs/20261003-crosspage-batches-v3'
METHODS={'ce_fixed':0.,'ce_crosspage_01':.1}
ORIGINAL_BATCHES=m.core.batches

def pair_rows(ids,labels,groups,pixels,pools,raw_size,rng):
    result=list(ids);positions=collections.defaultdict(list)
    for j,i in enumerate(ids):
        if i<raw_size:positions[labels[i]].append(j)
    for label,js in positions.items():
        if len(js)<2:continue
        a,b=(ids[j] for j in js[:2])
        if groups[a]!=groups[b] and pixels[a]!=pixels[b]:continue
        candidates=[i for i in pools[label] if groups[i]!=groups[a] and pixels[i]!=pixels[a]]
        if candidates:result[js[1]]=rng.choice(candidates)
    assert len(result)==len(ids)==len(set(result))
    assert collections.Counter(labels[i] for i in result)==collections.Counter(labels[i] for i in ids)
    assert all(i in pools[labels[i]] for i in result if i<raw_size)
    return result

def paired_batches(d,train,seed,steps):
    import random
    labels=d['labels'].cpu().tolist();pools=collections.defaultdict(list)
    for i in train:
        if i<d['n']:pools[labels[i]].append(i)
    original=ORIGINAL_BATCHES(d['labels'],train,d['n'],seed,steps)
    rng=random.Random(seed+10000)
    return [pair_rows(ids,labels,d['groups'],d['pixels'],pools,d['n'],rng) for ids in original]

def main():
    d=m.setup();m.load_fixture(d);m.verify(d)
    RUN.mkdir(exist_ok=False);started=time.monotonic()
    inputs=m.read_json(m.RUN/'preregistration.json')['input_sha256']|{str(Path(__file__)):m.sha256(Path(__file__)),str(m.RUN/'crop-manifest.json'):m.sha256(m.RUN/'crop-manifest.json'),str(m.RUN/'crop-fixture.pt'):m.sha256(m.RUN/'crop-fixture.pt')}
    m.write_json(RUN/'preregistration.json',dict(schema_version='crosspage-batches.v3',input_sha256=inputs,seeds=m.SEEDS,steps=300,methods=METHODS,
        trigger='After the completed v2 comparison: active contrast anchors about one fifth, lower SAM recognition with added contrast; this adaptive follow-up changes pairing only.',
        batches='Original class draws, per-class counts and SAM rows kept; replace second raw row with distinct pixels from another source group where possible; all classes remain supervised by CE.',
        implementation='Temporary scoped replacement of core.batches during existing fit_contrast; no source or model files modified.',
        paired='Both methods receive identical replacement batches, seed, initialization, optimizer, losses and steps.',
        losses='Same CE and cross-page contrast as mechanisms v2; lambda0 vs0.1; fixed historical prototypes; no prototype recomputation.',
        selection='No locked test; publish every fold/seed; same exploratory page mean, pooled rare macro and per-fold raw safety guards as v2; no activation.',
        limitation='Adaptive follow-up on already examined 17-page corpus; historical warmstart provenance unknown.',activated=False))
    torch.use_deterministic_algorithms(True)
    output=dict(folds=[],activated=False,locked_test_scored=False)
    for fold,held in enumerate(m.folds()):
        train=[i for i in d['train'] if d['groups'][i] not in held];raw=[i for i in train if i<d['n']]
        assert all(d['splits'][i]=='train' and d['groups'][i] not in held for i in train)
        base=m.evaluate(d,d['original'],d['old'],held,raw)
        for seed in m.SEEDS:
            entry=dict(fold=fold,seed=seed,held=held,baseline=base,methods={})
            plan=paired_batches(d,train,seed,300)
            batches_sha256=m.digest(torch.tensor([i for ids in plan for i in ids],dtype=torch.int64).numpy())
            entry['batches_sha256']=batches_sha256
            for method,weight in METHODS.items():
                with patch.object(m.core,'batches',lambda labels,ids,n,s,steps: plan):
                    model,history,coverage=m.fit_contrast(d,weight,seed,train)
                assert m.core.batches is ORIGINAL_BATCHES
                torch.save(dict(projection=model.state_dict(),class_names=d['cache']['class_names'],prototypes=d['old'].cpu(),seed=seed,weight=weight,train_ids=train,batches_sha256=batches_sha256),RUN/f'{method}_fold{fold}_seed{seed}.pt')
                evaluation=m.evaluate(d,model,d['old'],held,raw)
                entry['methods'][method]=dict(evaluation=evaluation,history=history,positive_count_histograms=coverage)
                print('PAIRED',fold,seed,method,'SAM',evaluation['masked']['top1_correct'],'RAW',evaluation['raw']['top1_correct'],'active',round(sum(h['active_anchors'] for h in history)/sum(h['anchors'] for h in history),3),flush=True)
            assert [h['active_anchors'] for h in entry['methods']['ce_fixed']['history']]==[h['active_anchors'] for h in entry['methods']['ce_crosspage_01']['history']]
            output['folds'].append(entry);m.write_json(RUN/'progress.json',output)
    m.verify(d);assert all(m.sha256(Path(p))==h for p,h in inputs.items())
    m.write_json(RUN/'results.json',output);m.write_json(RUN/'verification.json',dict(source_and_inputs_unchanged=True,active_model_unchanged=True,activated=False,locked_test_scored=False,trained_runs=30,elapsed_seconds=time.monotonic()-started))
    print('COMPLETE crosspage batches',flush=True)

if __name__=='__main__':
    import random
    assert pair_rows([0,1,4],[0,0,0,0,1],['a','a','b','c','z'],['x','y','x','w','q'],{0:[0,1,2,3],1:[4]},5,random.Random(42))==[0,3,4]
    assert pair_rows([0,1],[0,0],['a','a'],['x','y'],{0:[0,1]},2,random.Random(42))==[0,1]
    main()

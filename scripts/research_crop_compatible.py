"""Paired crop-compatible CE adaptation with class balance, replay and consistency."""
import argparse,collections,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts import research_base_corpus_long as base
from scripts import research_recognition_mechanisms as m
from scripts import research_recognition_stability as core
from scripts import summarize_recognition_mechanisms as summarizer
RUN=core.ROOT/'backend/training_runs/20261004-crop-compatible-v2'
STEPS=1000
SEEDS=[42,43,44]
METHODS=['ce','balanced','preserve','paired']
MILESTONES=[300,1000]


def class_mean(values,labels):
    classes,inverse,counts=labels.unique(sorted=True,return_inverse=True,return_counts=True)
    weights=1/(counts[inverse].to(values.dtype)*len(classes))
    return (values*weights).sum(),weights


def check():
    y=torch.tensor([0,0,0,1]);values=torch.tensor([1.,2.,3.,8.],requires_grad=True)
    loss,weights=class_mean(values,y)
    assert loss==5 and torch.allclose(weights,torch.tensor([1/6,1/6,1/6,.5]))
    loss.backward();torch.testing.assert_close(values.grad,weights)
    logits=torch.tensor([[2.,-1.],[0.,1.]],requires_grad=True)
    target=F.softmax(logits.detach()/.2,dim=1)
    kl=F.kl_div(F.log_softmax(logits/.2,dim=1),target,reduction='none').sum(1).mean()
    assert abs(float(kl))<1e-6
    kl.backward();assert torch.isfinite(logits.grad).all()
    a=F.normalize(torch.tensor([[1.,0.],[1.,1.]],requires_grad=True),dim=1)
    pair=(1-(a*a).sum(1)).mean();assert abs(float(pair))<1e-6
    return dict(class_mean_equal_mass=True,gradient_weights_passed=True,teacher_identity_kl_zero=True,pair_identity_zero=True)


def scope(d,held):
    raw,excluded=base.train_indices(d,'combined',held)
    raw_set=set(raw);lookup={};sam=[]
    for r in d['fixture']['rows']:
        if r['split']!='train' or r['source_group'] in excluded:continue
        i=d['n']+r['runtime_index'];j=r['snapshot_index']
        assert j in raw_set and d['objects'][i]==d['objects'][j]==r['row_id']
        assert d['groups'][i]==d['groups'][j]==r['source_group'] and d['labels'][i]==d['labels'][j]
        assert i not in lookup;lookup[i]=j;sam.append(i)
    sam.sort();train=raw+sam
    assert set(train)=={i for i in d['train'] if d['groups'][i] not in excluded}
    assert all(d['splits'][i]=='train' and d['groups'][i] not in excluded for i in train)
    held_pixels={d['pixels'][i] for i in range(d['n']) if d['groups'][i] in held}
    assert not held_pixels.intersection(d['pixels'][i] for i in train)
    parent=[i for i in raw if d['cache']['row_ids'][i] in d['parent_ids']]
    with torch.no_grad():teacher_scores=d['teacher'][parent]@d['old'].T
    anchor={i for j,i in enumerate(parent) if int(teacher_scores[j].argmax())==int(d['labels'][i])}
    assert anchor<=raw_set and all(d['splits'][i]=='train' for i in anchor)
    return raw,train,lookup,anchor,excluded


def verify(protocol):
    changed=[p for p,h in protocol['input_sha256'].items() if m.sha256(Path(p))!=h]
    assert not changed,changed


def fit(d,method,seed,fold,held,raw,train,lookup,anchor,plan):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=core.ProjectionHead().cuda();model.load_state_dict(d['init']);model.train()
    params=list(model.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    labels=d['labels'].cpu().tolist();exposure=collections.Counter();extra=collections.Counter();mass=collections.Counter();klmass=collections.Counter()
    history=[];milestones={};started=time.monotonic();reproduction=None
    for step,ids in enumerate(plan,1):
        y=d['labels'][ids];z=model(d['features'][ids]);scores=z@d['old'].T
        values=F.cross_entropy(scores/.1,y,reduction='none')
        if method=='ce':ce=F.cross_entropy(scores/.1,y);weights=torch.full_like(values,1/len(ids))
        else:ce,weights=class_mean(values,y)
        loss=ce;kl=ce*0;pair=ce*0
        replay=[j for j,i in enumerate(ids) if i in anchor]
        pairids=[i for i in ids if i in lookup]
        if method in ['preserve','paired'] and replay:
            target=(d['teacher'][[ids[j] for j in replay]]@d['old'].T/.2).detach()
            rows=F.kl_div(F.log_softmax(scores[replay]/.2,dim=1),F.softmax(target,dim=1),reduction='none').sum(1)
            kl,kw=class_mean(rows,y[replay]);loss=loss+4*kl
            for j,w in zip(replay,kw.detach().cpu().tolist()):klmass[labels[ids[j]]]+=w
        if method=='paired' and pairids:
            # Deterministic pair forwards keep the shared CE dropout stream identical across arms.
            model.eval()
            a=model(d['features'][[lookup[i] for i in pairids]]);b=model(d['features'][pairids])
            pair=(1-(a*b).sum(1)).mean();loss=loss+pair
            model.train();extra.update(pairids);extra.update(lookup[i] for i in pairids)
        assert torch.isfinite(loss)
        h=dict(step=step,ce=float(ce.detach()),kl=float(kl.detach()),weighted_kl=float((4*kl).detach()),pair=float(pair.detach()),loss=float(loss.detach()),
            classes=int(y.unique().numel()),replay_rows=len(replay) if method in ['preserve','paired'] else 0,pair_rows=len(pairids) if method=='paired' else 0)
        if step in [1,300,1000]:
            h['ce_gradient_norm']=m.grad_norm(ce,params)
            if method in ['preserve','paired'] and replay:h['weighted_kl_gradient_norm']=m.grad_norm(4*kl,params)
            if method=='paired' and pairids:h['pair_gradient_norm']=m.grad_norm(pair,params)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step()
        exposure.update(ids)
        for i,w in zip(ids,weights.detach().cpu().tolist()):mass[labels[i]]+=w
        history.append(h)
        if step in MILESTONES:
            evaluation=m.evaluate(d,model,d['old'],held,raw)
            cp=RUN/f'{method}_fold{fold}_seed{seed}_step{step}.pt'
            torch.save(dict(projection=model.state_dict(),prototypes=d['old'].cpu(),train_ids=train,seed=seed,method=method,step=step),cp)
            milestones[str(step)]=dict(evaluation=evaluation,checkpoint_sha256=m.sha256(cp),exposure=dict(exposure),extra_pair_forwards=dict(extra),ce_class_mass=dict(mass),kl_class_mass=dict(klmass))
            if method=='ce' and step==300:
                old=m.read_json(m.RUN/'contrast-results.json')
                prior=next(r for r in old['folds'] if r['fold']==fold and r['seed']==seed)['methods']['ce_fixed']['evaluation']
                assert evaluation==prior,'Reference CE metrics or predictions changed'
                ref=torch.load(m.RUN/f'ce_fixed_fold{fold}_seed{seed}.pt',weights_only=True,map_location='cpu')
                assert set(train)==set(ref['train_ids'])
                error=max(float((p.detach().cpu()-ref['projection'][k]).abs().max()) for k,p in model.state_dict().items())
                for k,p in model.state_dict().items():torch.testing.assert_close(p.detach().cpu(),ref['projection'][k],atol=2e-6,rtol=1e-4)
                reproduction=dict(predictions_and_metrics_equal=True,max_weight_error=error)
            model.train()
            print('FIT',fold,seed,method,step,'SAM',evaluation['masked']['top1_correct'],'RAW',evaluation['raw']['top1_correct'],flush=True)
    assert set(exposure)<=set(train) and {labels[i] for i in exposure}==set(range(286))
    assert set(extra)<=set(train)
    return dict(milestones=milestones,history=history,reference300_reproduction=reproduction,batch_sha256=m.digest(torch.tensor([i for batch in plan for i in batch]).numpy()),
        train_row_ids=[d['objects'][i] for i in train],unique_rows_seen=len(exposure),eligible_rows=len(train),elapsed_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','run']);args=parser.parse_args()
    if args.phase=='check':print(json.dumps(check()));return
    d=base.setup();d['fixture']=m.load_fixture(d);torch.use_deterministic_algorithms(True)
    paths=d['extra_paths']+[Path(__file__),Path(summarizer.__file__),m.RUN/'contrast-results.json']
    paths += list(m.RUN.glob('ce_fixed_fold*_seed*.pt'));assert len(list(m.RUN.glob('ce_fixed_fold*_seed*.pt')))==15
    hashes=d['before']|{str(p):m.sha256(p) for p in paths}
    protocol=dict(schema_version='crop-compatible.v1',input_sha256=hashes,seeds=SEEDS,steps=STEPS,milestones=MILESTONES,methods=METHODS,
        primary='fixed final step1000; step300 diagnostic, no dev selection',optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1,scheduler=None),
        losses=dict(ce_temperature=.1,balanced='mean CE within each present class, mean across classes',kl_temperature=.2,kl_multiplier=4,kl_T_squared=False,
            kl='class mean KL on eligible parent raw train objects correctly recognized by detached historical eval teacher',pair_multiplier=1,pair='mean cosine distance on deterministic raw annotation and SAM masked pair forwards'),
        init='historical projection and fixed historical prototypes',batches='core.batches unchanged; same1000 prefix plans for all arms; original CE reproduced at300',
        folds='same5 business train folds; held source-group component closure and held pixels excluded; raw+SAM pair joins row_id/source_group',
        guard='page mean joint improves vs historical, pooled raw rare1-4 macro does not regress, each fold mean raw top1 drop <=2pp, seed-averaged; no activation',
        limitation='Adaptive continuation after inspecting this corpus and earlier research; historical training provenance unknown; exploratory, not blind generalization',
        pair_exposure='paired arm adds deterministic forward passes of already eligible raw and masked objects; publish additional exposures; no new pixels or features',
        fit_count=60,optimizer_updates=60000,check=check(),activated=False,locked_test_scored=False)
    RUN.mkdir(exist_ok=False);m.write_json(RUN/'preregistration.json',protocol);output=dict(folds=[],activated=False,locked_test_scored=False)
    for fold,held in enumerate(m.folds()):
        raw,train,lookup,anchor,excluded=scope(d,held)
        baseline=m.evaluate(d,d['original'],d['old'],held,raw)
        for seed in SEEDS:
            plan=core.batches(d['labels'],train,d['n'],seed,STEPS)
            assert plan[:300]==core.batches(d['labels'],train,d['n'],seed,300)
            entry=dict(fold=fold,seed=seed,held=held,excluded=excluded,selected_raw_ids=raw,baseline=baseline,
                anchor_row_ids=[d['objects'][i] for i in sorted(anchor)],anchor_row_ids_sha256=m.digest(torch.tensor(sorted(anchor)).numpy()),
                pairs={d['objects'][i]:dict(raw_index=j,sam_index=i,source_group=d['groups'][i]) for i,j in lookup.items()},methods={})
            for method in METHODS:
                entry['methods'][method]=fit(d,method,seed,fold,held,raw,train,lookup,anchor,plan)
            assert len({r['batch_sha256'] for r in entry['methods'].values()})==1
            assert all(r['milestones']['1000']['exposure']==entry['methods']['ce']['milestones']['1000']['exposure'] for r in entry['methods'].values())
            output['folds'].append(entry);m.write_json(RUN/'progress.json',output)
            verify(protocol)
    assert len(output['folds'])==15 and len(list(RUN.glob('*_step1000.pt')))==60
    verify(protocol);m.write_json(RUN/'results.json',output)
    m.write_json(RUN/'verification.json',dict(inputs_and_active_files_unchanged=True,paired_batches_and_primary_exposures_equal=True,reference300_all_predictions_equal=True,held_groups_pixels_and_components_excluded=True,all286_classes_supervised_each_fit=True,fits=60,updates=60000,activated=False,locked_test_scored=False))
    print('DONE crop compatible',flush=True)


if __name__=='__main__':main()

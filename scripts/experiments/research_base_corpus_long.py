"""Long paired base-recipe experiment; frozen DINO/SAM, no active-model writes."""
import argparse,collections,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scripts.experiments import research_recognition_mechanisms as mechanism
from scripts.experiments import research_recognition_stability as core
from scripts.train_segmentation import read_json,write_json,sha256
from codex_pipeline.data.cached_dataset import CachedFeatureDataset,CachedEpisodicSampler,collate_cached_episodes
from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead
from codex_pipeline.models.prototypical import PrototypicalLoss

RUN=core.ROOT/'backend/training_runs/20261004-base-corpus-long-v2'
CONFIG=core.ROOT/'backend/codex_pipeline/config/snapshot.yaml'
MILESTONES={0,1,5,10,20,40,60,80,100}
SEEDS=[42,43,44]


def vector_loss(support,query,temperature=.1):
    # Episodes are sorted by class, with exactly three distinct supports per class.
    proto=F.normalize(support.reshape(20,3,-1).mean(1),dim=1)
    logits=query@proto.T/temperature
    target=torch.arange(20,device=query.device).repeat_interleave(5)
    return F.cross_entropy(logits,target),logits


def check():
    configure_determinism(42);torch.set_num_threads(4)
    features=torch.randn(160,384)
    labels=torch.arange(20).repeat_interleave(8)
    ds=CachedFeatureDataset(features,labels)
    indices=next(iter(CachedEpisodicSampler(ds,20,3,5,1)))
    blocks=np.array(indices).reshape(20,8)
    blocks=blocks[np.argsort(labels[blocks[:,0]].numpy())]
    x=features[blocks].reshape(20,8,384)
    ref_s,ref_sl,ref_q,ref_ql=collate_cached_episodes((features[indices],labels[indices]),20,3,5)
    assert torch.equal(x[:,:3].reshape(60,384),ref_s)
    assert torch.equal(x[:,3:].reshape(100,384),ref_q)
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        a=ProjectionHead().to(device);b=ProjectionHead().to(device);b.load_state_dict(a.state_dict())
        opta=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
        optb=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.0001)
        torch.manual_seed(71);torch.cuda.manual_seed_all(71)
        ref=PrototypicalLoss(.1)(a(ref_s.to(device)),ref_sl.to(device),a(ref_q.to(device)),ref_ql.to(device))
        ref['loss'].backward()
        torch.manual_seed(71);torch.cuda.manual_seed_all(71)
        loss,logits=vector_loss(b(x[:,:3].reshape(60,384).to(device)),b(x[:,3:].reshape(100,384).to(device)))
        loss.backward()
        torch.testing.assert_close(loss,ref['loss'],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(logits,ref['logits'],atol=1e-5,rtol=1e-5)
        error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()))
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)
        for model,opt in [(a,opta),(b,optb)]:torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=2e-5,rtol=1e-4)
        results.append(dict(device=device,gradient_max_error=error,loss=float(loss)))
    return dict(passed=True,collation_equal=True,loss_logits_gradients_adamw_step_equal=True,devices=results)


def setup():
    d=mechanism.setup();mechanism.load_fixture(d)
    manifest=read_json(core.SNAPSHOT/'snapshot_manifest.json')
    parent_path=core.SNAPSHOT.parent/manifest['parent_snapshot_id']/'snapshot_manifest.json'
    parent=read_json(parent_path);d['parent_ids']=set(parent['active_row_ids'])
    assert d['parent_ids']<=set(d['cache']['row_ids']) and len(d['parent_ids'])==9266
    d['cfg']=yaml.safe_load(CONFIG.read_text())
    assert d['cfg']['training']==dict(episodes_per_epoch=500,num_epochs=100,n_way=20,k_shot=3,q_queries=5,learning_rate=.001,weight_decay=.0001,lr_scheduler='cosine',warmup_epochs=5,temperature=.1,device='cuda',seed=42)
    d['components']=manifest['components']
    d['extra_paths']=[parent_path,CONFIG,Path(__file__),Path(core.__file__),Path(mechanism.__file__),
        core.ROOT/'backend/codex_pipeline/data/cached_dataset.py',core.ROOT/'backend/codex_pipeline/models/prototypical.py',
        core.ROOT/'backend/codex_pipeline/models/projection_head.py',core.ROOT/'backend/codex_pipeline/determinism.py',
        core.ROOT/'backend/codex_pipeline/scripts/train.py',mechanism.RUN/'crop-fixture.pt',mechanism.RUN/'crop-manifest.json',core.RUN/'cv-results.json']
    return d


def train_indices(d,arm,held):
    excluded=set(held)
    for component in d['components']:
        if excluded.intersection(component['source_groups']):excluded.update(component['source_groups'])
    ids=[i for i in range(d['n']) if d['splits'][i]=='train' and d['groups'][i] not in excluded
        and (arm=='combined' or d['cache']['row_ids'][i] in d['parent_ids'])]
    held_ids=[i for i in range(d['n']) if d['groups'][i] in held]
    assert not {d['pixels'][i] for i in ids}&{d['pixels'][i] for i in held_ids}
    assert not set(ids)&set(held_ids)
    assert len({int(d['labels'][i]) for i in ids})==286,'Base comparison requires support for every class'
    return ids,sorted(excluded)


@torch.no_grad()
def readout(d,model,ids):
    embedded=core.project(model,d['features'][ids])
    labels=d['labels'][ids]
    return F.normalize(torch.stack([embedded[labels==c].mean(0) for c in range(286)]),dim=1)


@torch.no_grad()
def dev_score(d,model,proto):
    ids=[i for i in range(d['n']) if d['splits'][i]=='dev']
    scores=core.project(model,d['features'][ids])@proto.T
    return core.metrics(scores,d['labels'][ids])


@torch.no_grad()
def evaluate(d,model,proto,ids,held):
    if held:
        reference,_=train_indices(d,'combined',held)
        return mechanism.evaluate(d,model,proto,held,reference,effective_supports=ids)
    return core.evaluate(d,core.project(model,d['features'])@proto.T)


def register(d):
    hashes=d['before']|{str(p):sha256(p) for p in d['extra_paths']}
    if (RUN/'preregistration.json').exists():
        p=read_json(RUN/'preregistration.json');assert p['input_sha256']==hashes
        return
    RUN.mkdir(parents=True,exist_ok=False)
    plans=[]
    for scope,held,seeds in [('cv'+str(i),h,[42]) for i,h in enumerate(mechanism.folds())]+[('full',[],SEEDS)]:
        for seed in seeds:
            for arm in ['parent','combined']:
                ids,excluded=train_indices(d,arm,held)
                labels=d['labels'][ids].cpu().tolist();counts=collections.Counter(labels)
                new=[i for i in ids if d['cache']['row_ids'][i] not in d['parent_ids']]
                plans.append(dict(scope=scope,seed=seed,arm=arm,held_groups=held,excluded_groups=excluded,
                    train_count=len(ids),new_count=len(new),eligible_classes=sum(n>=8 for n in counts.values()),
                    new_eligible_count=sum(counts[int(d['labels'][i])]>=8 for i in new),
                    train_row_ids_sha256=hashlib.sha256('\n'.join(d['cache']['row_ids'][i] for i in ids).encode()).hexdigest(),
                    prototype_only_classes={d['cache']['class_names'][c]:n for c,n in counts.items() if n<8}))
    write_json(RUN/'preregistration.json',dict(schema_version='base-corpus-long.v1',input_sha256=hashes,
        config=d['cfg'],plans=plans,updates_per_fit=50000,total_comparisons=len(plans),unique_fits=len({(x['seed'],x['train_row_ids_sha256']) for x in plans}),milestones=sorted(MILESTONES),
        reuse='Identical training-row IDs and seed/config/init reuse one fit, separate fold/full readouts reconstructed at all fixed milestones',
        initial_projection=str(core.RUNTIME/'weights/projection.pt'),initialization='Historical projection, same weights for both arms; no fresh-head or backbone training',
        training='Raw annotation crops only, original CachedEpisodicSampler; ordered vector loss equivalent to original loss',
        rarity='Reference availability uses combined train-only corpus for both arms; effective supports use each actual arm',
        primary='5 page-held folds at seed42, fixed epoch100, recomputed raw train-only centroids; paired old vs enriched corpus',
        secondary='Full corpus seeds42/43/44; dev and previously inspected locked test scored diagnostically; not blind test',
        checkpoint_selection='Fixed epoch100 primary; best among100 epochs by raw dev99 top1 secondary, first ties retained',
        guard='Compare page-mean joint recall and raw micro against parent and historical; rare1-4 macro must not regress; fold raw drop <=2pp. No automatic promotion',
        provenance_limit='Historical base projection and prototypes training provenance unknown; excludes adaptation leakage, not historical pretraining leakage',
        unmodified=dict(DINO=True,SAM=True,active_model=True),activation=False,implementation_check=check()))


def verify_inputs():
    p=read_json(RUN/'preregistration.json')
    assert p['input_sha256']=={k:sha256(Path(k)) for k in p['input_sha256']}
    return p


def reuse(d,plan,source,folder):
    started=time.monotonic();ids,_=train_indices(d,plan['arm'],plan['held_groups'])
    assert source['plan']['seed']==plan['seed']
    assert source['plan']['train_row_ids_sha256']==plan['train_row_ids_sha256']
    source_folder=Path(source['training_artifact_dir'])
    assert source['checkpoint_sha256']=={p.name:sha256(p) for p in source_folder.glob('*.pt')}
    model=ProjectionHead().cuda();model.load_state_dict(d['init']);milestones={}
    for epoch,value in source['milestones'].items():
        if int(epoch)==0:
            model.load_state_dict(d['init']);proto=readout(d,model,ids)
        else:
            cp=torch.load(source_folder/f'epoch{int(epoch):03d}.pt',weights_only=True,map_location='cuda')
            model.load_state_dict(cp['model_state_dict']);proto=cp['prototypes']
        milestones[epoch]=dict(dev=value['dev'],evaluation=evaluate(d,model,proto,ids,plan['held_groups']))
    cp=torch.load(source_folder/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(cp['model_state_dict'])
    result=dict(source,plan=plan,milestones=milestones,
        best_evaluation=evaluate(d,model,cp['prototypes'],ids,plan['held_groups']),
        reused_training_from=str(source_folder),elapsed_seconds=time.monotonic()-started)
    write_json(folder/'history.json',dict(history=result['history'],milestones=milestones,best_epoch=result['best_epoch']))
    write_json(folder/'result.json',result)
    print('REUSE',folder.name,'FROM',source_folder.name,flush=True);return result


def fit(d,plan):
    name=f"{plan['scope']}-{plan['arm']}-seed{plan['seed']}"
    folder=RUN/name
    if (folder/'result.json').exists():return read_json(folder/'result.json')
    folder.mkdir(exist_ok=True);started=time.monotonic()
    for previous in RUN.glob('*/result.json'):
        source=read_json(previous)
        if source['plan']['seed']==plan['seed'] and source['plan']['train_row_ids_sha256']==plan['train_row_ids_sha256']:
            return reuse(d,plan,source,folder)
    ids,excluded=train_indices(d,plan['arm'],plan['held_groups'])
    assert len(ids)==plan['train_count'] and excluded==plan['excluded_groups']
    configure_determinism(plan['seed'])
    model=ProjectionHead().cuda();model.load_state_dict(d['init'])
    ds=CachedFeatureDataset(d['cache']['features'][ids],d['cache']['labels'][ids])
    sampler=CachedEpisodicSampler(ds,20,3,5,500)
    labels=ds.labels.numpy();features=d['features'][ids];exposures=np.zeros(len(ids),dtype=np.int64)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=95)
    warmup=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.01,total_iters=5)
    history=[];best=-1;best_epoch=None;milestones={};episode_digest=hashlib.sha256()
    baseline_proto=readout(d,model,ids)
    milestones['0']=dict(dev=dev_score(d,model,baseline_proto),evaluation=evaluate(d,model,baseline_proto,ids,plan['held_groups']))
    for epoch in range(100):
        blocks=np.array(list(sampler),dtype=np.int64).reshape(500,20,8)
        order=np.argsort(labels[blocks[:,:,0]],axis=1)
        blocks=np.take_along_axis(blocks,order[:,:,None],axis=1)
        assert np.all(np.diff(np.sort(blocks.reshape(500,160),axis=1),axis=1)>0)
        episode_digest.update(blocks.tobytes());exposures+=np.bincount(blocks.ravel(),minlength=len(ids))
        batch_features=features[torch.from_numpy(blocks).cuda()]
        losses=[];model.train();lr=optimizer.param_groups[0]['lr']
        for x in batch_features:
            support=model(x[:,:3].reshape(60,384));query=model(x[:,3:].reshape(100,384))
            loss,_=vector_loss(support,query)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(loss.detach())
        train_loss=float(torch.stack(losses).mean())
        assert np.isfinite(train_loss)
        proto=readout(d,model,ids);dev=dev_score(d,model,proto)
        history.append(dict(epoch=epoch+1,updates=(epoch+1)*500,train_loss=train_loss,learning_rate=lr,dev=dev))
        if dev['top1_correct']>best:
            best=dev['top1_correct'];best_epoch=epoch+1
            torch.save(dict(model_state_dict=model.state_dict(),prototypes=proto,epoch=epoch+1,dev=dev),folder/'best.pt')
        if epoch<5:warmup.step()
        else:cosine.step()
        if epoch+1 in MILESTONES:
            milestones[str(epoch+1)]=dict(dev=dev,evaluation=evaluate(d,model,proto,ids,plan['held_groups']))
            torch.save(dict(model_state_dict=model.state_dict(),prototypes=proto,epoch=epoch+1,dev=dev),folder/f'epoch{epoch+1:03d}.pt')
            print('MILESTONE',name,epoch+1,'loss',round(train_loss,4),'dev',dev['top1_correct'],'elapsed',round(time.monotonic()-started,1),flush=True)
        write_json(folder/'history.json',dict(history=history,milestones=milestones,best_epoch=best_epoch))
    assert sum(exposures)==8000000
    for c in ds.classes:
        ci=ds.class_to_indices[c]
        assert bool((exposures[ci]>0).all()) if len(ci)>=8 else bool((exposures[ci]==0).all())
    best_ckpt=torch.load(folder/'best.pt',weights_only=True,map_location='cuda');model.load_state_dict(best_ckpt['model_state_dict'])
    result=dict(plan=plan,updates=50000,training_artifact_dir=str(folder),history=history,milestones=milestones,best_epoch=best_epoch,
        best_evaluation=evaluate(d,model,best_ckpt['prototypes'],ids,plan['held_groups']),
        episode_plan_sha256=episode_digest.hexdigest(),row_exposures={d['cache']['row_ids'][i]:int(exposures[j]) for j,i in enumerate(ids)},
        checkpoint_sha256={p.name:sha256(p) for p in folder.glob('*.pt')},elapsed_seconds=time.monotonic()-started,activated=False)
    write_json(folder/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['check','run']);args=parser.parse_args()
    if args.command=='check':print(json.dumps(check(),indent=2));return
    torch.set_num_threads(4);d=setup();register(d);p=verify_inputs()
    historical=dict(cv=[],full=core.evaluate(d,core.project(d['original'],d['features'])@d['old'].T))
    for held in mechanism.folds():
        ids,_=train_indices(d,'combined',held)
        historical['cv'].append(mechanism.evaluate(d,d['original'],d['old'],held,ids))
    write_json(RUN/'historical-baseline.json',historical)
    results=[]
    for plan in p['plans']:
        results.append(fit(d,plan));verify_inputs()
        write_json(RUN/'progress.json',dict(completed=len(results),total=len(p['plans']),results=[dict(plan=r['plan'],elapsed_seconds=r['elapsed_seconds']) for r in results]))
    write_json(RUN/'results.json',dict(schema_version='base-corpus-long.v1',historical=historical,results=results,
        verification=dict(input_hashes_unchanged=True,active_model_unchanged=True),activated=False))
    print('DONE',RUN,len(results),flush=True)


if __name__=='__main__':main()

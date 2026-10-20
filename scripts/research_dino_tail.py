"""Controlled last-two-block DINO adaptation; research artifacts only."""
import argparse, collections, copy, csv, json, os, random, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts import research_crop_compatible as prior
from codex_model.classifier import CodexClassifier, _preprocess_image
ROOT = prior.core.ROOT
RUN = ROOT / 'backend/training_runs/20261004-dino-tail-v2'
CACHE_RUN = ROOT / 'backend/training_runs/20261004-dino-tail-v1'
STEPS, SEED, CHUNK = 200, 42, 64


class Tail(torch.nn.Module):
    def __init__(self, b):
        super().__init__()
        assert not b.chunked_blocks and len(b.blocks) == 12
        self.blocks = torch.nn.Sequential(*[copy.deepcopy(v) for v in b.blocks[-2:]])
        self.norm = copy.deepcopy(b.norm)
    def forward(self, x):
        return self.norm(self.blocks(x))[:, 0]


def backbone():
    return CodexClassifier(model_dir=prior.core.RUNTIME, device='cuda', backbone_source=prior.core.PIN)._backbone.eval().requires_grad_(False)


def data():
    d = prior.base.setup()
    d['fixture'] = prior.m.load_fixture(d)
    return d


def check():
    prior.check()
    torch.manual_seed(5)
    a = torch.nn.Linear(3, 4); b = copy.deepcopy(a)
    x = torch.randn(7, 3); y = torch.tensor([0, 0, 1, 1, 1, 2, 3])
    loss, w = prior.class_mean(F.cross_entropy(a(x), y, reduction='none'), y)
    loss.backward()
    for s in range(0, len(y), 3):
        (F.cross_entropy(b(x[s:s+3]), y[s:s+3], reduction='none') * w[s:s+3]).sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)
    counts = torch.tensor([1, 5, 4, 0]); old = torch.arange(4); new = torch.tensor([1, 2, 3, 0])
    gate = (counts[old] >= 1) & (counts[old] <= 4)
    assert torch.equal(torch.where(gate, old, new), torch.tensor([0, 2, 2, 0]))
    b = backbone(); tail = Tail(b).cuda().requires_grad_(True)
    x = torch.randn(2, 3, 224, 224, device='cuda')
    with torch.no_grad():
        t = b.prepare_tokens_with_masks(x)
        for block in b.blocks[:-2]: t = block(t)
        torch.testing.assert_close(tail(t), b(x), atol=1e-5, rtol=1e-5)
    tail(t).square().mean().backward()
    assert not t.requires_grad and all(not p.requires_grad and p.grad is None for p in b.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tail.parameters())
    return dict(microbatch_gradients_equal=True, prefix_tail_matches_full_backbone=True,
                only_tail_receives_gradients=True, historical_rare_gate_checked=True)


def register(d):
    if (RUN/'preregistration.json').exists():
        p = prior.m.read_json(RUN/'preregistration.json'); prior.verify(p)
        return p
    inputs = d['before'] | {str(p):prior.m.sha256(p) for p in d['extra_paths']}
    paths = [Path(__file__), Path(prior.__file__), prior.RUN/'results.json', ROOT/'reports/rare-readout-results-20261004.json']
    paths += [CACHE_RUN/'cache.json', CACHE_RUN/'tokens.npy']
    paths += [prior.RUN/f'paired_fold{fold}_seed{SEED}_step1000.pt' for fold in range(5)]
    inputs.update({str(p):prior.m.sha256(p) for p in paths})
    pin = prior.m.read_json(prior.core.PIN)
    inputs[pin['weights_path']] = prior.m.sha256(Path(pin['weights_path']))
    assert inputs[pin['weights_path']] == pin['weights_sha256']
    RUN.mkdir(parents=True, exist_ok=False)
    p = dict(schema_version='dino-tail.v2', input_sha256=inputs, seed=SEED, folds=5,
        initialization='Existing fold-specific paired step1000 projection; original DINO tail; historical prototypes',
        methods=['head_only','last_two_blocks'], steps=STEPS, diagnostic_steps=[0,100],
        primary='Fixed final step200; no checkpoint, hyperparameter or threshold selection on held labels',
        optimizer=dict(name='AdamW', head_lr=1e-5, tail_lr=1e-5, weight_decay=1e-4, clip_norm=1),
        precision='float32; no autocast/TF32; strict deterministic algorithms and CUBLAS :4096:8; identical cached tokens and FP32 tail in both arms',
        trainable='Projection in both arms; blocks10/11 and final norm additionally in last_two_blocks',
        losses='Class-balanced CE T=.1; parent-correct historical KL T=.2 multiplier4; raw/SAM pair cosine multiplier1',
        batches='Identical core.batches seed42 positions1001..1200; full-batch loss weights, microbatches64; RNG reset after model loading in both arms',
        augmentation='None; same Lanczos/pad224 and saved SAM masks as paired comparator',
        readouts=['ungated','historical_rare1_4_fallback'],
        localization='Same254 SAM matches on343 known annotations; no SAM/bbox training',
        data_scope='9471 raw train plus254 masked train crops cached, including held images through a fixed prefix; held source components/pixels excluded from gradients/losses; no dev/test pixels processed',
        evaluation='OOF raw343 and mask254 from17 previously examined training pages; same five folds',
        limitation='Exploratory single seed; historical head/prototype provenance unknown; incomplete coverage prevents false-positive precision claims',
        activation=False, locked_test_scored=False)
    prior.m.write_json(RUN/'preregistration.json', p)
    return p


def cache(d, protocol):
    saved = prior.m.read_json(CACHE_RUN/'cache.json')
    assert saved['ids'] == d['train'] and len(saved['ids']) == 9725
    assert all(d['splits'][i] == 'train' for i in saved['ids'])
    assert prior.m.sha256(CACHE_RUN/'tokens.npy') == saved['tokens_sha256']
    return saved


def token_batch(tokens, positions, ids):
    # ponytail: ~3.84GB mmap cache; stream smaller caches if the corpus outgrows RAM.
    return torch.from_numpy(np.array(tokens[[positions[i] for i in ids]],copy=True)).cuda()


def support_counts(d, raw):
    pixels = collections.defaultdict(set)
    for i in raw: pixels[int(d['labels'][i])].add(d['pixels'][i])
    return torch.tensor([len(pixels[c]) for c in range(286)],device='cuda')


@torch.no_grad()
def evaluate(d, head, tail, tokens, positions, held, raw):
    head.eval()
    if tail is not None: tail.eval()
    counts = support_counts(d,raw)
    raw_ids = [i for i in range(d['n']) if d['splits'][i]=='train' and d['groups'][i] in held]
    mask_ids = [d['n']+r['runtime_index'] for r in d['fixture']['rows'] if r['split']=='train' and r['source_group'] in held]
    assert not set(raw_ids+mask_ids)&set(raw)
    result = dict(raw=[],mask=[])
    for view,ids in [('raw',raw_ids),('mask',mask_ids)]:
        for s in range(0,len(ids),CHUNK):
            js = ids[s:s+CHUNK]
            f = d['features'][js] if tail is None else tail(token_batch(tokens,positions,js))
            scores = head(f)@d['old'].T
            historical = (d['teacher'][js]@d['old'].T).argmax(1); pred = scores.argmax(1)
            fallback = (counts[historical]>=1)&(counts[historical]<=4)
            gated = torch.where(fallback,historical,pred); top3 = scores.topk(3,dim=1).indices
            for j,i in enumerate(js):
                result[view].append(dict(row_id=d['objects'][i],group=d['groups'][i],label=int(d['labels'][i]),
                    prediction=int(pred[j]),gated_prediction=int(gated[j]),historical_prediction=int(historical[j]),
                    fallback=bool(fallback[j]),natural_support=int(counts[int(d['labels'][i])]),top3=top3[j].tolist()))
    return result


def fit(d, fold, held, method, tokens, positions, protocol):
    dest = RUN/f'{method}_fold{fold}.json'
    if dest.exists(): return prior.m.read_json(dest)
    raw,train,lookup,anchor,excluded = prior.scope(d,held)
    assert all(i in positions and d['splits'][i]=='train' and d['groups'][i] not in excluded for i in train)
    assert not {d['pixels'][i] for i in train}&{d['pixels'][i] for i in range(d['n']) if d['groups'][i] in held}
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    head = prior.core.ProjectionHead().cuda()
    saved = torch.load(prior.RUN/f'paired_fold{fold}_seed{SEED}_step1000.pt',weights_only=True,map_location='cpu')
    assert set(saved['train_ids']) == set(train)
    head.load_state_dict(saved['projection'])
    tail = Tail(backbone()).cuda().requires_grad_(method=='last_two_blocks')
    assert all(p.requires_grad == (method=='last_two_blocks') for p in tail.parameters())
    tail_before = {k:p.detach().cpu().clone() for k,p in tail.state_dict().items()} if tail else {}
    baseline = evaluate(d,head,None,tokens,positions,held,raw)
    if tail: assert evaluate(d,head,tail,tokens,positions,held,raw)==baseline, 'Initial predictions changed'
    reference = prior.m.read_json(ROOT/'reports/rare-readout-results-20261004.json')
    expected = next(e for e in reference['folds'] if e['seed']==SEED and e['fold']==fold)
    for view in ['raw','mask']:
        assert [r['gated_prediction'] for r in baseline[view]] == [r['prediction'] for r in expected['details'][view]['predictions']]
    params = list(head.parameters())+(list(tail.parameters()) if method=='last_two_blocks' else [])
    opt = torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-4)
    plan = prior.core.batches(d['labels'],train,d['n'],SEED,1000+STEPS)[1000:]
    history,exposure,extra,diagnostics = [],collections.Counter(),collections.Counter(),{}
    started = time.monotonic(); head.train()
    if tail: tail.train()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    for step,ids in enumerate(plan,1):
        y = d['labels'][ids]; _,w = prior.class_mean(torch.zeros(len(ids),device='cuda'),y)
        replay = [j for j,i in enumerate(ids) if i in anchor]; rw = {}
        if replay:
            _,kw = prior.class_mean(torch.zeros(len(replay),device='cuda'),y[replay])
            rw = {j:kw[k] for k,j in enumerate(replay)}
        opt.zero_grad(); totals = dict(ce=0.,kl=0.,pair=0.)
        for s in range(0,len(ids),CHUNK):
            js = ids[s:s+CHUNK]
            f = d['features'][js] if tail is None else tail(token_batch(tokens,positions,js))
            scores = head(f)@d['old'].T
            ce = (F.cross_entropy(scores/.1,d['labels'][js],reduction='none')*w[s:s+len(js)]).sum()
            loss = ce; totals['ce'] += float(ce.detach())
            rr = [j for j in range(len(js)) if s+j in rw]
            if rr:
                target = (d['teacher'][[js[j] for j in rr]]@d['old'].T/.2).detach()
                v = F.kl_div(F.log_softmax(scores[rr]/.2,dim=1),F.softmax(target,dim=1),reduction='none').sum(1)
                kl = (v*torch.stack([rw[s+j] for j in rr])).sum()
                loss = loss+4*kl; totals['kl'] += float(kl.detach())
            assert torch.isfinite(loss); loss.backward()
        pairs = [i for i in ids if i in lookup]; head.eval()
        for s in range(0,len(pairs),CHUNK):
            js = pairs[s:s+CHUNK]; rs = [lookup[i] for i in js]
            a = d['features'][rs] if tail is None else tail(token_batch(tokens,positions,rs))
            b = d['features'][js] if tail is None else tail(token_batch(tokens,positions,js))
            pair = (1-(head(a)*head(b)).sum(1)).sum()/len(pairs)
            assert torch.isfinite(pair); pair.backward(); totals['pair'] += float(pair.detach())
            extra.update(js); extra.update(rs)
        head.train(); grad = torch.nn.utils.clip_grad_norm_(params,1)
        assert torch.isfinite(grad); opt.step(); exposure.update(ids)
        history.append(dict(step=step,**totals,grad_norm=float(grad)))
        if step%25==0 or step==1:
            print('TRAIN',method,'fold',fold,'step',step,'/',STEPS,'loss',round(totals['ce']+4*totals['kl']+totals['pair'],4),'seconds',round(time.monotonic()-started),flush=True)
        if step==100:
            diagnostics[str(step)] = evaluate(d,head,tail,tokens,positions,held,raw); head.train()
            if tail: tail.train()
    assert {int(d['labels'][i]) for i in exposure} == set(range(286))
    result = dict(fold=fold,seed=SEED,held=held,excluded=excluded,method=method,baseline=baseline,
        evaluation=evaluate(d,head,tail,tokens,positions,held,raw),diagnostics=diagnostics,history=history,
        eligible_ids=train,raw_train_ids=raw,exposure=dict(exposure),pair_exposure=dict(extra),
        batch_sha256=prior.m.digest(np.array([i for batch in plan for i in batch],dtype=np.int64)),
        trainable_parameters=sum(p.numel() for p in params),elapsed_seconds=time.monotonic()-started,
        tail_max_weight_change=max([float((tail.state_dict()[k].detach().cpu()-p).abs().max()) for k,p in tail_before.items()] or [0.]))
    cp = RUN/f'{method}_fold{fold}.pt'
    torch.save(dict(projection=head.state_dict(),tail=tail.state_dict() if tail else None,prototypes=d['old'].cpu(),
        train_ids=train,fold=fold,seed=SEED,step=STEPS,prefix_backbone_pin=prior.m.read_json(prior.core.PIN)),cp)
    result['checkpoint_sha256'] = prior.m.sha256(cp)
    prior.m.write_json(dest,result); prior.verify(protocol)
    print('RESULT',method,fold,{v:sum(r['gated_prediction']==r['label'] for r in result['evaluation'][v]) for v in ['raw','mask']},flush=True)
    return result


def summarize(d, entries):
    from scripts.summarize_recognition_mechanisms import classification
    families = {}
    for method,field in [('best_prior','baseline'),('head_only','evaluation'),('last_two_blocks','evaluation')]:
        chosen = [e for e in entries if e['method']==('head_only' if method=='best_prior' else method)]
        rows = {v:[r for e in chosen for r in e[field][v]] for v in ['raw','mask']}
        assert len(rows['raw'])==343 and len(rows['mask'])==254 and len({r['row_id'] for r in rows['raw']})==343
        pages = []
        for page in d['app']['full_pages']['pages']:
            if page['split']!='train': continue
            rr = [r for r in rows['mask'] if r['group']==page['source_group']]
            pages.append(dict(group=page['source_group'],known=page['ground_truth_count'],matched=len(rr),correct=sum(r['gated_prediction']==r['label'] for r in rr)))
        fam = dict(pages=pages)
        for v in ['raw','mask']:
            gated = [dict(r,prediction=r['gated_prediction']) for r in rows[v]]
            fam[v] = dict(ungated=classification(rows[v]),gated=classification(gated),
                rare_gated=classification([r for r in gated if 1<=r['natural_support']<=4]))
        correct = fam['mask']['gated']['correct']; fam['joint_known'] = dict(correct=correct,total=343,rate=correct/343)
        families[method] = fam
    assert families['best_prior']['mask']['gated']['correct']==180
    comparisons = {}
    for ref in ['best_prior','head_only']:
        a = families['last_two_blocks']['pages']; b = families[ref]['pages']
        assert [p['group'] for p in a]==[p['group'] for p in b]
        rng = random.Random(20261004); samples = []
        for _ in range(10000):
            ids = rng.choices(range(len(a)),k=len(a))
            samples.append(sum(a[i]['correct']-b[i]['correct'] for i in ids)/sum(a[i]['known'] for i in ids))
        comparisons[ref] = dict(delta_correct=sum(p['correct'] for p in a)-sum(p['correct'] for p in b),
            delta_joint_pp=100*(families['last_two_blocks']['joint_known']['rate']-families[ref]['joint_known']['rate']),
            exploratory_page_bootstrap_95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),
            bootstrap='10000 fixed-prediction page clusters; no refitting; single seed, adaptive corpus')
    out = dict(schema_version='dino-tail-summary.v1',run=str(RUN),families=families,comparisons=comparisons,folds=entries,
        activated=False,locked_test_scored=False,seed=SEED,total_optimizer_updates=len(entries)*STEPS,
        scope='Exploratory OOF17 train pages; known annotation recall, no false-positive precision')
    prior.m.write_json(RUN/'results.json',out); prior.m.write_json(ROOT/'reports/dino-tail-results-20261004.json',out)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase',choices=['check','cache','run','verify']); args = p.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True)
    if args.phase=='check': print(json.dumps(check())); return
    d = data(); torch.use_deterministic_algorithms(True); protocol = register(d)
    if args.phase=='verify':
        saved = prior.m.read_json(CACHE_RUN/'cache.json')
        assert prior.m.sha256(CACHE_RUN/'tokens.npy')==saved['tokens_sha256']
        assert all(prior.m.sha256(Path(p))==sha for p,sha in saved['image_sha256'].items())
        results = prior.m.read_json(RUN/'results.json'); assert len(results['folds'])==10
        for e in results['folds']:
            assert prior.m.sha256(RUN/f"{e['method']}_fold{e['fold']}.pt")==e['checkpoint_sha256']
        prior.verify(protocol)
        prior.m.write_json(RUN/'verification.json',dict(inputs_active_weights_and_images_unchanged=True,cache_and_checkpoint_hashes_verified=True,
            all_five_held_folds_complete=True,no_dev_test_pixels_processed=True,no_activation=True))
        print('VERIFIED'); return
    saved = cache(d,protocol)
    if args.phase=='cache': print(json.dumps({k:v for k,v in saved.items() if k not in ['ids','image_sha256']})); return
    tokens = np.load(CACHE_RUN/'tokens.npy',mmap_mode='r'); positions = {i:j for j,i in enumerate(saved['ids'])}; entries = []
    for fold,held in enumerate(prior.m.folds()):
        for method in ['head_only','last_two_blocks']: entries.append(fit(d,fold,held,method,tokens,positions,protocol))
        assert entries[-1]['exposure']==entries[-2]['exposure'] and entries[-1]['pair_exposure']==entries[-2]['pair_exposure']
        assert entries[-1]['batch_sha256']==entries[-2]['batch_sha256']
        prior.m.write_json(RUN/'progress.json',dict(folds=entries))
    out = summarize(d,entries)
    print(json.dumps(dict(families={k:dict(raw=v['raw']['gated']['correct'],mask=v['mask']['gated']['correct']) for k,v in out['families'].items()},comparisons=out['comparisons']),indent=2))


if __name__=='__main__': main()

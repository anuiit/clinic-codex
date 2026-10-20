"""Bounded adaptive ablation: remove KL only from the paired recipe."""
import json,time
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.experiments import research_crop_compatible as d
RUN=d.core.ROOT/'backend/training_runs/20261004-crop-compatible-no-kl-v2'
PRIOR=d.RUN
KL=d.F.kl_div


def zero_kl(*args,**kwargs):return KL(*args,**kwargs)*0


def main():
    previous=d.m.read_json(PRIOR/'results.json');prior_p=d.m.read_json(PRIOR/'preregistration.json');d.verify(prior_p)
    data=d.base.setup();data['fixture']=d.m.load_fixture(data);torch.use_deterministic_algorithms(True)
    hashes=prior_p['input_sha256']|{str(Path(__file__)):d.m.sha256(Path(__file__)),str(PRIOR/'results.json'):d.m.sha256(PRIOR/'results.json')}
    protocol=dict(schema_version='crop-compatible-no-kl.v1',input_sha256=hashes,trigger='After complete1000-step four-arm results: KL adds0.33 SAM glyph on average vs balanced, so test whether it is needed inside paired arm',
        methods=['paired_no_kl'],fit_count=15,steps=1000,seeds=d.SEEDS,primary='fixed step1000; step300 secondary',
        change='Only set KL contribution and gradient to zero via scoped F.kl_div multiplier0; same train data, pairs, CE balance, cosine, dropout stream, batches, initialization, prototypes, optimizer, milestones and evaluation',
        activated=False,locked_test_scored=False,adaptive=True)
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol);d.RUN=RUN;out=dict(folds=[],activated=False,locked_test_scored=False)
    # Verify the override removes gradients without changing RNG.
    logits=torch.randn(3,4,device='cuda',requires_grad=True);state=torch.cuda.get_rng_state()
    z=zero_kl(d.F.log_softmax(logits,dim=1),d.F.softmax(logits.detach(),dim=1),reduction='none')
    z.sum().backward();assert not logits.grad.any() and torch.equal(state,torch.cuda.get_rng_state())
    for old in previous['folds']:
        raw,train,lookup,anchor,excluded=d.scope(data,old['held']);plan=d.core.batches(data['labels'],train,data['n'],old['seed'],1000)
        with patch.object(d.F,'kl_div',zero_kl):
            r=d.fit(data,'paired',old['seed'],old['fold'],old['held'],raw,train,lookup,anchor,plan)
        assert d.F.kl_div is KL
        control=old['methods']['paired'];assert r['batch_sha256']==control['batch_sha256']
        for step,mile in r['milestones'].items():
            assert {str(k):v for k,v in mile['exposure'].items()}==control['milestones'][step]['exposure']
            assert {str(k):v for k,v in mile['extra_pair_forwards'].items()}==control['milestones'][step]['extra_pair_forwards']
        assert all(h['kl']==h['weighted_kl']==0 for h in r['history'])
        out['folds'].append(dict(fold=old['fold'],seed=old['seed'],held=old['held'],excluded=excluded,selected_raw_ids=raw,baseline=old['baseline'],methods={'paired_no_kl':r}))
        d.m.write_json(RUN/'progress.json',out);d.verify(protocol)
    assert len(out['folds'])==len(list(RUN.glob('*_step1000.pt')))==15
    d.verify(protocol);d.m.write_json(RUN/'results.json',out)
    d.m.write_json(RUN/'verification.json',dict(inputs_and_active_files_unchanged=True,plans_ce_and_pair_exposures_identical=True,kl_values_and_gradients_zero=True,fits=15,updates=15000,activated=False))
    print('DONE no KL ablation',flush=True)


if __name__=='__main__':main()

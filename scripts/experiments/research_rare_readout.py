"""Adaptive fixed-weight rare-class fallback; no optimization or activation."""
import collections,csv,json,statistics
from pathlib import Path
import torch
from scripts.experiments import research_crop_compatible as d
from scripts.experiments import summarize_recognition_mechanisms as s
RUN=d.core.ROOT/'backend/training_runs/20261004-rare-readout-v1'
OUT=d.core.ROOT/'reports/rare-readout-results-20261004.json'


class RareReadout(torch.nn.Module):
    def __init__(self,old,new,proto,counts):
        super().__init__();self.old=old;self.new=new
        self.register_buffer('proto',proto);self.register_buffer('counts',torch.tensor([counts.get(c,0) for c in range(286)],device='cuda'))
    def forward(self,x):
        a=self.old(x);b=self.new(x);label=(a@self.proto.T).argmax(1)
        use_old=(self.counts[label]>=1)&(self.counts[label]<=4)
        return torch.where(use_old[:,None],a,b)


def main():
    original=d.m.read_json(d.RUN/'results.json');p=d.m.read_json(d.RUN/'preregistration.json');d.verify(p)
    paths=[Path(__file__),d.RUN/'results.json']+list(d.RUN.glob('paired_*_step1000.pt'))
    protocol=dict(schema_version='rare-readout.v1',input_sha256=p['input_sha256']|{str(f):d.m.sha256(f) for f in paths},
        trigger='Posthoc actual SAM rare evaluation: historical8/10 vs paired3/10 despite passing raw rare guard',
        rule='Compute distinct raw train pixels per class after held/component exclusion; if historical top1 class has1..4 supports, use historical embedding/class, else paired embedding/class. Same fixed historical prototypes; no use of held ground truth to gate',
        folds=5,seeds=d.SEEDS,checkpoint='paired fixed step1000',rare_definition='Same natural1..4 raw training support bins as before; no threshold sweep',
        extended_guard='Old pagejoint/raw rare/foldraw guards plus no decline in actual SAM known-rare class macro; exploratory, not blind',
        optimizer_updates=0,activated=False,locked_test_scored=False,adaptive=True)
    RUN.mkdir(exist_ok=False);d.m.write_json(RUN/'preregistration.json',protocol)
    data=d.base.setup();data['fixture']=d.m.load_fixture(data);torch.use_deterministic_algorithms(True)
    meta={r['row_id']:r for r in csv.DictReader((d.core.SNAPSHOT/'metadata.csv').open())};index=data['cache']['row_ids'];entries=[]
    for e in original['folds']:
        raw,train,lookup,anchor,excluded=d.scope(data,e['held'])
        counts=s.support_counts([meta[index[i]] for i in raw]);model=d.core.ProjectionHead().cuda()
        cp=d.RUN/f"paired_fold{e['fold']}_seed{e['seed']}_step1000.pt"
        assert d.m.sha256(cp)==e['methods']['paired']['milestones']['1000']['checkpoint_sha256']
        model.load_state_dict(torch.load(cp,weights_only=True,map_location='cpu')['projection']);model.eval()
        gate=RareReadout(data['original'],model,data['old'],counts).cuda().eval()
        ev=d.m.evaluate(data,gate,data['old'],e['held'],raw)
        mask=[data['n']+r['runtime_index'] for r in data['fixture']['rows'] if r['split']=='train' and r['source_group'] in e['held']]
        rid=[i for i in range(data['n']) if data['splits'][i]=='train' and data['groups'][i] in e['held']]
        details={}
        for view,ids in [('raw',rid),('mask',mask)]:
            with torch.no_grad():
                a=d.core.project(data['original'],data['features'][ids])@data['old'].T;b=d.core.project(model,data['features'][ids])@data['old'].T
                historical=a.argmax(1);candidate=b.argmax(1);chosen=(gate(data['features'][ids])@data['old'].T).argmax(1)
            flags=[j for j,label in enumerate(historical.tolist()) if 1<=counts.get(label,0)<=4]
            assert all(chosen[j]==historical[j] for j in flags)
            records=[dict(row_id=data['objects'][i],label=int(data['labels'][i]),prediction=int(chosen[j]),historical_prediction=int(historical[j]),paired_prediction=int(candidate[j]),fallback=j in flags,natural_support=counts[int(data['labels'][i])]) for j,i in enumerate(ids)]
            details[view]=dict(fallback_rows=len(flags),corrected_vs_paired=sum(r['prediction']==r['label'] and r['paired_prediction']!=r['label'] for r in records),broken_vs_paired=sum(r['prediction']!=r['label'] and r['paired_prediction']==r['label'] for r in records),predictions=records)
        entries.append(dict(fold=e['fold'],seed=e['seed'],held=e['held'],selected_raw_ids=raw,baseline=e['baseline'],paired=e['methods']['paired']['milestones']['1000']['evaluation'],gated=ev,details=details))
    agg={method:s.across_seeds([s.aggregate([(e,e[method]) for e in entries if e['seed']==seed],meta,index) for seed in d.SEEDS]) for method in ['baseline','paired','gated']}
    rare={};gating={}
    for method in ['baseline','paired','gated']:
        rare[method]={}
        for seed in d.SEEDS:
            rows=[r for e in entries if e['seed']==seed for r in e['details']['mask']['predictions'] if 1<=r['natural_support']<=4]
            assert len(rows)==10
            key='historical_prediction' if method=='baseline' else 'paired_prediction' if method=='paired' else 'prediction'
            rare[method][str(seed)]=s.classification([dict(label=r['label'],prediction=r[key]) for r in rows])
        gating[method]=dict(raw_guard=None,sam_rare_macro=statistics.mean(r['macro'] for r in rare[method].values()))
    from scripts.experiments.summarize_crop_compatible import guards
    control_entries=[dict(e,methods={'gated':{'milestones':{'1000':{'evaluation':e['gated']}}}}) for e in entries]
    guard=guards(control_entries,'gated','historical',1000,agg['gated'],agg['baseline']);guard['checks']['sam_rare_macro_no_regression']=gating['gated']['sam_rare_macro']>=gating['baseline']['sam_rare_macro'];guard['eligible']=all(guard['checks'].values())
    agg['gated']['paired_vs_historical']=s.paired_pages(agg['gated'],agg['baseline']);agg['gated']['paired_vs_ungated']=s.paired_pages(agg['gated'],agg['paired'])
    d.verify(protocol);output=dict(schema_version='rare-readout-summary.v1',run=str(RUN),families=agg,rare_sam=rare,guard=guard,folds=entries,
        verification=dict(inputs_and_active_files_unchanged=True,no_optimization=True,held_truth_not_used_for_gating=True,historical_rare_answers_retained=True),activated=False,optimizer_updates=0,locked_test_scored=False,
        scope='Adaptive fixed-weight readout on previously examined held pages; actual SAM rare10objects, all matched. No automatic activation')
    s.write(OUT,output);s.write(RUN/'results.json',output)
    print(json.dumps(dict(families={k:dict(sam=v['masked_correct'],raw=v['raw_correct'],page=v['page_mean_joint'],rare_raw=v['rare_macro'],rare_sam=[r['correct'] for r in rare[k].values()]) for k,v in agg.items()},guard=guard),indent=2))


if __name__=='__main__':main()

"""Post-hoc fixed fold0 train geometry diagnostic, never model selection."""
import argparse,time
from pathlib import Path
import torch
from scripts.experiments import research_synthetic_detector as s
ROOT=s.ROOT;RUN=ROOT/'backend/training_runs/20261004-yolo-train-diagnostic-v1'
SOURCES={200:s.RUN,1000:ROOT/'backend/training_runs/20261004-yolo-synthetic-1000-v1'}

def prepare():
    assert not RUN.exists();RUN.mkdir();paths=[Path(__file__),Path(s.__file__)]
    for folder in SOURCES.values():
        paths.extend([folder/'dataset.json',folder/'final-preregistration.json'])
        for arm in ['real','real_plus_synthetic']:paths.extend([folder/f'{arm}_fold0_seed42.pt',folder/f'selection_{arm}_fold0.json'])
    s.write(RUN/'preregistration.json',dict(input_sha256={str(p):s.sha(p) for p in paths},posthoc=True,selection='Fixed fold0, both arms and both durations; own fit pages only. Existing confidence cuts, no tuning, no training. Publish train-versus-own-outer geometry, neither independent nor a promotion criterion.',optimizer_updates=0,activated=False))
    print('DIAGNOSTIC_FROZEN',flush=True)

def run():
    torch.set_num_threads(4);p=s.read(RUN/'preregistration.json');s.verify(p);out=[]
    for steps,folder in SOURCES.items():
        data=s.read(folder/'dataset.json');sp=data['splits'][0]
        for arm in ['real','real_plus_synthetic']:
            model=s.make_model();cp=folder/f'{arm}_fold0_seed42.pt';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.eval()
            threshold=s.read(folder/f'selection_{arm}_fold0.json')['threshold'];pages=[]
            for page in data['pages']:
                if page['group'] not in sp['fit_pages']:continue
                gt=[dict(z,natural_support=sp['support_counts'][str(z['label'])]) for z in page['gt']]
                rows=s.detect(model,page);pages.append(dict(group=page['group'],gt=gt,ignore=page['ignore'],predictions=rows,geometry=s.f.score_page(gt,page['ignore'],rows,threshold)))
            counts={k:sum(z['geometry']['counts'][k] for z in pages) for k in pages[0]['geometry']['counts']}
            det=s.read(folder/'detection-results.json');fit=next(z for z in det['fits'] if z['fold']==0 and z['arm']==arm)
            c={k:sum(z['geometry']['counts'][k] for z in fit['outer_pages']) for k in fit['outer_pages'][0]['geometry']['counts']}
            out.append(dict(steps=steps,arm=arm,threshold=threshold,train=dict(counts=counts,metrics=s.f.metrics(counts),pages=pages),outer=dict(counts=c,metrics=s.f.metrics(c))))
            print('TRAIN_DIAGNOSTIC',steps,arm,'train/outer recall',out[-1]['train']['metrics']['detection_recall'],out[-1]['outer']['metrics']['detection_recall'],flush=True)
            del model;torch.cuda.empty_cache()
    s.verify(p);s.write(RUN/'results.json',dict(protocol=p,results=out,activated=False,optimizer_updates=0));print('DIAGNOSTIC_DONE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);phase=parser.parse_args().phase
    prepare() if phase=='prepare' else run()

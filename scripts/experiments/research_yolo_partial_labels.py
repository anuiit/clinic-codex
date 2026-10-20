"""Controlled partial-label loss ablation; frozen real batches/cuts from pilot200."""
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scripts.experiments import research_synthetic_detector as s
from scripts.experiments import research_yolo_recognition as r
from scripts.train_bbox_detector import clipped
from scripts.experiments.research_nested_branch_rejection import score as branch_score
ROOT=s.ROOT;SOURCE=s.RUN;RUN=ROOT/'backend/training_runs/20261004-yolo-partial-label-v2';STEPS=200

def prepare():
    assert not RUN.exists();RUN.mkdir();old=s.read(SOURCE/'final-preregistration.json');s.verify(old)
    audit_path=ROOT/'reports/yolo-annotation-coverage-audit-20261004.json';audit=s.read(audit_path)
    confirmation=ROOT/'reports/yolo-annotation-coverage-confirmation-20261004.json';assert s.read(confirmation)['export_is_exhaustive'] is False
    data=s.read(SOURCE/'dataset.json');sam=torch.load(s.PAGE_CACHE,map_location='cpu',weights_only=True)['pages'];sam={p['group']:p for p in sam}
    extras={};paths=[Path(__file__),Path(r.__file__),s.PAGE_CACHE,audit_path,confirmation,SOURCE/'recognition/results.json',SOURCE/'detection-results.json']
    for sp in data['splits']:
        index=0;added=0
        for p in data['pages']:
            if p['group'] not in sp['fit_pages']:continue
            assert p['group'] not in set(sp['held'])|set(sp['calibration'])|set(sp['excluded'])
            unknown=[q['bbox'] for q in sam[p['group']]['proposals'] if max([s.f.box_iou(s.f.box(q['bbox'])['box'],s.f.box(g['bbox'])['box']) for g in p['gt']+p['ignore']],default=0)<.5]
            extras[p['group']]=unknown;image=Image.open(p['image']).convert('RGB')
            page=dict(width=image.width,height=image.height,image_path=p['image'],source_group=p['group'],boxes=p['gt'],ignore_boxes=[z['bbox'] for z in p['ignore']])
            for tile in s.make_tiles(page):
                row=sp['real'][index];index+=1;assert row['group']==p['group']
                extra=[c for b in unknown if (c:=clipped(b,tile['window'])) is not None]
                canvas,boxes,ignores,_=s.resize_packet(image.crop(tile['window']),tile['boxes'],tile['ignore_boxes']+extra)
                assert boxes==row['boxes'] and np.array_equal(np.array(canvas),np.array(Image.open(row['path'])))
                row['ignore_boxes']=ignores;row['extra_ignored_boxes']=len(extra);added+=len(extra)
        assert index==len(sp['real']);sp['extra_ignored_boxes']=added
    s.write(RUN/'dataset.json',data);s.write(RUN/'uncertain-sam-regions.json',extras)
    paths.extend([RUN/'dataset.json',RUN/'uncertain-sam-regions.json'])
    p=dict(input_sha256=old['input_sha256']|{str(p):s.sha(p) for p in paths},steps=200,fits=5,new_optimizer_updates=1000,adaptive=True,
           trigger='Human confirmed visible head absent from archive should be annotated, contradicting previous exhaustiveness assurance. Do not invent GT or retrain names.',
           change='Only add ignored-negative anchor centre regions from frozen SAM proposals on own fit pages with maxIoU<.5 against any known/ignored geometry. No positive pseudo labels. Positive soft targets stay protected, native bbox/DFL losses unchanged.',
           constants='Same native model/pretrain/seed42/init, real pixel tensors and200 batch plan, all optimizer/BN settings and LR horizon200 as pilot real arm. Five new fits only, no synthetic forwards.',
           inference='Reuse own-fold pilot real detector confidence cuts, no new calibration or selection. Same views/NMS and same frozen nested recognition/rare gating/semantic035 and branch cuts. Report known-annotation recall; unmatched predictions cannot be assumed background with partial labels.',
           metrics_policy='Only known-annotation recall and conditional naming are interpretable. Precision/F1 fields computed by legacy helper are accounting diagnostics on the partial export, not exhaustive performance or a promotion objective.',
           limitation='SAM may miss unknown glyphs or mark true background. Its uncertain rectangles approximate partial-label ignore, not exhaustive labels. Same examined17pages, one seed, adaptive correction, no blind evaluation.',activated=False,locked_test_scored=False)
    s.write(RUN/'preregistration.json',p);s.verify(p);print('PARTIAL_PROTOCOL_FROZEN',[sp['extra_ignored_boxes'] for sp in data['splits']],flush=True)


def fit():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;protocol=s.read(RUN/'preregistration.json');s.verify(protocol);data=s.read(RUN/'dataset.json');control=s.read(SOURCE/'detection-results.json');results=[]
    for sp in data['splits']:
        ref=next(f for f in control['fits'] if f['fold']==sp['fold'] and f['arm']=='real');model=s.make_model();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        initial_hash=s.hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest();assert initial_hash==ref['initial_state_sha256']
        packets=s.packets(sp['real'],s.anchors_for(model));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0005);s.freeze_bn(model);history=[];started=time.monotonic()
        for step,ids in enumerate(sp['real_plan'],1):
            lr=(.00005+(.0005-.00005)*step/20) if step<=20 else .00005+.5*(.0005-.00005)*(1+math.cos(math.pi*(step-20)/(200-20)))
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad();loss,components=s.loss(model,s.batch(packets,ids));loss.backward();assert torch.isfinite(loss)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);opt.step()
            history.append(dict(step=step,lr=lr,real_loss=float(loss.detach()),real_components=components.detach().cpu().tolist(),grad_norm=float(grad)))
        cp=RUN/f'partial_fold{sp["fold"]}_seed42.pt';torch.save(dict(model=model.state_dict(),steps=200,fold=sp['fold'],arm='partial_ignore'),cp);model.eval();outer=[]
        for page in ref['outer_pages']:
            predictions=s.detect(model,page);outer.append(dict(page,predictions=predictions,geometry=s.f.score_page(page['gt'],page['ignore'],predictions,ref['selection']['threshold'])))
        results.append(dict(fold=sp['fold'],arm='partial_ignore',initial_state_sha256=initial_hash,checkpoint=str(cp),checkpoint_sha256=s.sha(cp),selection=dict(threshold=ref['selection']['threshold'],source='frozen pilot real calibration, reused without new tuning'),history=history,fit_seconds=time.monotonic()-started,real_image_exposures=400,outer_pages=outer))
        s.write(RUN/'progress.json',dict(fits=results,activated=False));print('PARTIAL_FIT',sp['fold'],'matched',sum(p['geometry']['counts']['matched'] for p in outer),flush=True)
        del model,opt,packets;torch.cuda.empty_cache()
    s.verify(protocol);s.write(RUN/'detection-results.json',dict(protocol=protocol,fits=results,optimizer_updates=1000,activated=False,locked_test_scored=False))


def recognize():
    torch.set_num_threads(4);det=s.read(RUN/'detection-results.json');s.verify(det['protocol']);assert len(det['fits'])==5
    # Freeze this inference source and original readout helpers before extracting pixels.
    paths=[Path(__file__),Path(r.__file__),s.BRANCH,RUN/'detection-results.json']+list(r.HEADS.glob('paired_fold*_seed42_step1000.pt'))
    rp=s.read(SOURCE/'recognition/preregistration.json');s.verify(rp)
    protocol=dict(input_sha256=rp['input_sha256']|{str(p):s.sha(p) for p in paths},optimizer_updates=0,activated=False,selection='None; both raw/mask and035/branch, detector cuts reused from control.')
    assert not (RUN/'recognition-preregistration.json').exists();s.write(RUN/'recognition-preregistration.json',protocol)
    classifier=s.CodexClassifier(r.RUNTIME,device='cuda',backbone_source=r.PIN);predictor=s.MobileSAMSegmenter(checkpoint_path=str(r.SAM),device='cuda',points_per_side=16)._mask_generator.predictor
    data=s.read(RUN/'dataset.json');spmap={sp['fold']:sp for sp in data['splits']};branch=s.read(s.BRANCH);pages=[];cache=[]
    for fit in det['fits']:
        fold=fit['fold'];sp=spmap[fold];cp=torch.load(r.HEADS/f'paired_fold{fold}_seed42_step1000.pt',map_location='cpu',weights_only=True);model=s.ProjectionHead().cuda();model.load_state_dict(cp['projection']);model.eval();prototypes=cp['prototypes'].cuda()
        threshold=next(z['selection']['threshold'] for z in branch['folds'] if z['seed']==42 and z['fold']==fold)
        for p in fit['outer_pages']:
            rows=[q for q in p['predictions'] if q['detector_confidence']>=fit['selection']['threshold']];features,hashes=r.features(classifier,predictor,p,rows)
            page=dict(group=p['group'],fold=fold,gt=p['gt'],ignore=p['ignore'],views={})
            cache.append(dict(group=p['group'],features=features,proposal_indices=[q['proposal_index'] for q in rows],crop_sha256=hashes))
            for view,ff in features.items():
                with torch.no_grad():
                    old=classifier._projection(ff.cuda())@prototypes.T;new=model(ff.cuda())@prototypes.T;oldlabel=old.argmax(1);flag=torch.tensor([1<=sp['support_counts'][str(c)]<=4 for c in oldlabel.tolist()],dtype=torch.bool,device='cuda');sc=torch.where(flag[:,None],old,new);values,labels=sc.max(1)
                predictions=[dict(q,similarity=float(values[i]),prediction=int(labels[i]),fallback=bool(flag[i])) for i,q in enumerate(rows)]
                page['views'][view]=dict(predictions=predictions,current_035=s.f.score_page(p['gt'],p['ignore'],predictions,.35),rare_branch=branch_score(p['gt'],p['ignore'],predictions,threshold))
            pages.append(page)
        print('PARTIAL_NAMES',fold,flush=True)
    assert len(pages)==17;torch.save(dict(pages=cache),RUN/'features.pt');s.verify(protocol);s.verify(det['protocol'])
    geometry=r.aggregate([p for fit in det['fits'] for p in fit['outer_pages']],'geometry');recognition={v:{policy:r.aggregate([p['views'][v] for p in pages],policy) for policy in ['current_035','rare_branch']} for v in ['raw','mask']}
    s.write(RUN/'results.json',dict(protocol=protocol,geometry=geometry,recognition=recognition,pages=pages,reference=s.read(SOURCE/'recognition/results.json')['recognition']['real'],feature_sha256=s.sha(RUN/'features.pt'),metrics_policy='Partial labels: precision and F1 are not exhaustive performance; publish known-annotation recall and conditional naming only.',new_detector_fits=5,new_optimizer_updates=1000,activated=False,locked_test_scored=False))
    print(json.dumps(dict(geometry=geometry,recognition=recognition),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','fit','recognize']);phase=parser.parse_args().phase
    {'prepare':prepare,'fit':fit,'recognize':recognize}[phase]()


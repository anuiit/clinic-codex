#!/usr/bin/env python3
"""One-page causal crop replay probe; no crop ablation metrics or optimization."""
import os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts import research_recognition_stability as core
from scripts.train_segmentation import read_json,write_json,sha256
from codex_model.classifier import CodexClassifier,_preprocess_image
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter


def bitmap_sha(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    d=core.load();page=d['app']['full_pages']['pages'][0]
    assert sha256(Path(core.__file__))==read_json(core.RUN/'preregistration.json')['source_sha256']
    record=core.RUN/'crop-probe-preregistration.json';assert not record.exists()
    write_json(record,dict(source_sha256=sha256(Path(__file__)),page_group=page['source_group'],seed=42,
        sam_microbatches=[4,1],dino_batches=[1,16],optimization=False,activated=False,
        purpose='diagnose failed crop replay; no selection or ablation accuracy metrics'))
    started=time.monotonic();path=Path(page['source_image_path'])
    ds=read_json(core.ROOT/'backend/training_runs/20261003-bbox-detector-pilot/dataset.json')
    assert sha256(path)==next(p['image_sha256'] for p in ds['pages'] if p['source_group']==page['source_group'])
    pixels=np.array(Image.open(path).convert('RGB'));s=MobileSAMSegmenter(checkpoint_path=str(core.SAM),device='cuda',points_per_side=16)
    views={};report=dict(page_group=page['source_group'],sam={},rows=[],activated=False)
    rows=[(j,r) for j,r in enumerate(d['rt']['rows']) if r['source_group']==page['source_group']]
    for batch in [4,1]:
        s._mask_generator.points_per_batch=batch
        ps=s.extract_crops(pixels,s.segment_page(pixels),padding=5);by_box={tuple(p.bbox):p for p in ps}
        assert len(ps)==page['proposal_count'] and len(by_box)==len(ps)
        views[batch]=[by_box[tuple(r['proposal_bbox'])] for j,r in rows]
        report['sam'][str(batch)]=dict(proposal_count=len(ps),matched_boxes_present=True)
        print('PROBE SAM',batch,len(ps),flush=True)
    del s;torch.cuda.empty_cache()
    clf=CodexClassifier(model_dir=core.RUNTIME,device='cuda',backbone_source=core.PIN)
    batch=torch.cat([_preprocess_image(Image.fromarray(p.crop),224) for p in views[4]]).cuda()
    with torch.no_grad():
        many=torch.cat([clf._backbone(batch[i:i+16]).float().cpu() for i in range(0,len(batch),16)])
        singles=torch.cat([clf._backbone(batch[i:i+1]).float().cpu() for i in range(len(batch))])
    expected=d['rt']['features'][[j for j,r in rows]]
    for i,(j,r) in enumerate(rows):
        p,q=views[4][i],views[1][i]
        report['rows'].append(dict(row_id=r['row_id'],bbox=r['proposal_bbox'],mask_same=bitmap_sha(p.mask)==bitmap_sha(q.mask),
            crop_same=bitmap_sha(p.crop)==bitmap_sha(q.crop),mask_different_pixels=int((p.mask!=q.mask).sum()),
            crop_sha_batch4=bitmap_sha(p.crop),crop_sha_batch1=bitmap_sha(q.crop),
            dino_batch_max_error=float((many[i]-singles[i]).abs().max()),
            cache_replay_max_error=float((many[i]-expected[i]).abs().max()),
            cache_replay_cosine=float(F.cosine_similarity(many[i,None],expected[i,None])[0])))
    report['summary']=dict(sam_masks_changed=sum(not r['mask_same'] for r in report['rows']),
        sam_crops_changed=sum(not r['crop_same'] for r in report['rows']),rows=len(rows),
        dino_batch_max_error=float((many-singles).abs().max()),cache_replay_max_error=float((many-expected).abs().max()),
        cache_replay_passed=bool(torch.allclose(many,expected,atol=1e-4,rtol=1e-4)))
    torch.save(dict(rows=[r for j,r in rows],features_batch16=many,features_batch1=singles,
        crops_batch4=[torch.from_numpy(p.crop.copy()) for p in views[4]],crops_batch1=[torch.from_numpy(p.crop.copy()) for p in views[1]]),core.RUN/'crop-probe-data.pt')
    assert d['before']=={str(p):sha256(p) for p in d['paths']}
    report['verification']=dict(inputs_unchanged=True,active_model_unchanged=True,optimization=False,elapsed_seconds=time.monotonic()-started)
    write_json(core.RUN/'crop-probe-results.json',report);print(report['summary'],flush=True)


if __name__=='__main__':main()

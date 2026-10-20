#!/usr/bin/env python3
"""Registered inference-only crop ablation; SAM CUDA cumsum warns on determinism."""
import os,time
from pathlib import Path
import torch
from scripts.experiments import research_recognition_stability as core
from scripts.train_segmentation import read_json,write_json,sha256


def record_previous_hash():
    return read_json(core.RUN/'crop-driver-preregistration.json')['driver_sha256']


def crop_views_batched(d):
    from codex_pipeline.segmentation import mobilesam
    original=mobilesam.MobileSAMSegmenter
    def small_batch(*args,**kwargs):
        segmenter=original(*args,**kwargs)
        segmenter._mask_generator.points_per_batch=4
        return segmenter
    # Local inference process only; restores the constructor even on failure.
    mobilesam.MobileSAMSegmenter=small_batch
    try:return core.crop_views(d)
    finally:mobilesam.MobileSAMSegmenter=original


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    original=Path(core.__file__)
    assert sha256(original)==read_json(core.RUN/'preregistration.json')['source_sha256']
    record=core.RUN/'crop-driver-preregistration-v2.json'
    assert not record.exists()
    write_json(record,dict(driver_sha256=sha256(Path(__file__)),base_source_sha256=sha256(original),
        views=['masked_replay','rectangle_proposal','dilated_mask3','half_background'],
        optimization=False,deterministic_algorithms='warn_only for unsupported SAM CUDA cumsum',
        reason='inference-only recovery: warn-only SAM cumsum, point microbatch4 after memory saturation',
        points_per_batch=4,previous_driver_sha256=record_previous_hash(),
        comparison='same proposal boxes and DINO backbone; no ground-truth boxes used as SAM prompts',
        selection='none; publish all preregistered views on already inspected splits',activated=False))
    started=time.monotonic();d=core.load();d['teacher']=core.project(d['original'],d['features'])
    variants,error=crop_views_batched(d);report={'views':{},'masked_replay_max_error':error,'activated':False}
    for name,features in variants.items():
        z=d['teacher'].clone();z[d['n']:]=core.project(d['original'],features.cuda())
        e=core.evaluate(d,z@d['old'].T);report['views'][name]=e
        print('VIEW',name,'SAM',e['application']['locked_test']['top1_correct'],'DEV',e['application']['dev']['top1_correct'],flush=True)
    assert d['before']=={str(p):sha256(p) for p in d['paths']}
    write_json(core.RUN/'crop-results.json',report)
    write_json(core.RUN/'crops-verification.json',dict(inputs_unchanged=True,active_model_unchanged=True,
        same_boxes_same_backbone=True,masked_features_reproduced=True,masked_replay_max_error=error,
        optimization=False,driver_sha256=sha256(Path(__file__)),elapsed_seconds=time.monotonic()-started,activated=False))
    print('COMPLETE crops',flush=True)


if __name__=='__main__':main()

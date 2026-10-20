import torch
from scripts.research_recognition_stability import aligned_prototypes,batches,shrink

def test_label_alignment_sampler_and_shrinkage():
    package={'prototypes':torch.tensor([[0.,1.],[1.,0.]]),'class_labels':torch.tensor([15,8]),'class_names':{15:'b',8:'a'}}
    assert torch.equal(aligned_prototypes(package,{0:'a',1:'b'}),torch.eye(2))
    labels=torch.tensor([c for c in range(25) for _ in range(2)]+[c for c in range(5) for _ in range(2)])
    train=[i for i in range(60) if i not in [0,1,52]]
    one=batches(labels,train,50,42,3)
    assert one==batches(labels,train,50,42,3)
    assert all(len(x)==len(set(x)) and set(x)<=set(train) for x in one)
    old=torch.eye(2);new=torch.flip(old,[0]);counts=torch.tensor([0,8])
    blended=shrink(old,new,counts,8)
    assert torch.equal(blended[0],old[0])
    assert torch.allclose(blended[1],torch.tensor([2**-.5,2**-.5]))
    assert torch.allclose(blended.norm(dim=1),torch.ones(2))


def test_crop_microbatch_restores_constructor(monkeypatch):
    import pytest
    from scripts import research_crop_ablation as driver
    from codex_pipeline.segmentation import mobilesam
    class FakeSegmenter:
        def __init__(self):
            from types import SimpleNamespace
            self._mask_generator=SimpleNamespace(points_per_batch=64)
    monkeypatch.setattr(mobilesam,'MobileSAMSegmenter',FakeSegmenter)
    def fails_after_construction(data):
        assert mobilesam.MobileSAMSegmenter()._mask_generator.points_per_batch==4
        raise RuntimeError('extraction failed')
    monkeypatch.setattr(driver.core,'crop_views',fails_after_construction)
    with pytest.raises(RuntimeError,match='extraction failed'):
        driver.crop_views_batched({})
    assert mobilesam.MobileSAMSegmenter is FakeSegmenter

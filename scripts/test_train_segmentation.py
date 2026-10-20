"""Small checks for weak-box training and full-page evaluation contracts."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from scripts.train_segmentation import (box_mask, evaluate, grid_points, mask_loss,
    mask_box_overlap, prepare_teacher_cache, projection_loss)


def test_runtime_grid_is_the_sam_grid():
    pytest.importorskip("mobile_sam", reason="mobile_sam is not installed (not in requirements-dev.txt)")
    from mobile_sam.utils.amg import build_point_grid
    expected = build_point_grid(16) * np.array([320, 640])
    np.testing.assert_array_equal(grid_points(320, 640), expected)


def test_box_projection_has_gradients_and_penalizes_wrong_placement():
    box = box_mask([2, 3, 6, 7], 8)[None]
    same = box.clone()
    moved = torch.roll(box, 2, dims=-1).requires_grad_()
    assert projection_loss(same, box).item() < 1e-6
    loss = projection_loss(moved, box).sum()
    loss.backward()
    assert loss.item() > 0 and moved.grad.abs().sum() > 0


def test_only_one_ambiguous_mask_token_is_supervised():
    target = box_mask([2, 2, 6, 6], 8)[None]
    logits = torch.stack([target[0] * 8 - 4, torch.zeros(8, 8), torch.ones(8, 8)], dim=0)[None].requires_grad_()
    loss = mask_loss(logits, target, target).mean()
    loss.backward()
    assert torch.isfinite(loss)
    active = logits.grad.abs().sum((-2, -1)) > 0
    assert active.sum().item() == 1 and active[0, 0]


def test_empty_and_bad_teacher_masks_are_not_box_labels():
    box = box_mask([1, 1, 7, 7], 8)
    assert mask_box_overlap(torch.zeros_like(box), box) == 0
    assert mask_box_overlap(box_mask([1, 1, 2, 2], 8), box) < 0.5
    assert mask_box_overlap(box, box) == 1


def test_teacher_cache_refuses_locked_test_before_loading_images():
    predictor = SimpleNamespace()
    segmenter = SimpleNamespace(_mask_generator=SimpleNamespace(predictor=predictor))
    with pytest.raises(ValueError, match="Held-out"):
        prepare_teacher_cache(segmenter, [{"split": "locked_test"}], 8)


def test_evaluation_uses_full_page_without_ground_truth_prompts(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32), "white").save(image)
    proposal = SimpleNamespace(bbox=(2, 2, 8, 8))
    class Segmenter:
        def segment_page(self, pixels):
            assert pixels.shape == (32, 32, 3)
            return [proposal, SimpleNamespace(bbox=(20, 20, 5, 5))]
    page = {"image_path": str(image), "source_group": "p", "split": "dev",
            "width": 32, "height": 32, "boxes": [{"bbox": [2, 2, 8, 8]}]}
    result = evaluate(Segmenter(), [page])
    assert result["counts"]["matched_boxes"] == 1
    assert result["counts"]["proposal_count"] == 2
    assert result["counts"]["known_box_coverage"] == 1
    assert result["precision"] is None and result["mask_iou"] is None

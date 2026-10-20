from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_s14_v5.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_s14_v5", SCRIPT)
assert SPEC and SPEC.loader
v5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v5
SPEC.loader.exec_module(v5)


def test_exact_47_plus_30_matrix_and_baselines():
    v5.assert_specs()
    assert len(v5.SPECS) == 77
    assert sum(item.family == "v3" for item in v5.SPECS) == 47
    assert sum(item.family == "v4" for item in v5.SPECS) == 30
    assert v5.SPECS[v5.CONTROL_INDEX].name == "v3-control-b1-ncm"
    assert v5.SPECS[v5.KNN1_INDEX].name == "v4-baseline-cosine-k1"


def test_projection_path_and_sha_are_the_actual_b1_runtime():
    expected = ROOT / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/runtime/weights/projection.pt"
    assert v5.PROJECTION == expected
    assert v5.EXPECTED_PROJECTION_SHA == "52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03"
    assert v5.assert_projection_identity() == {"path": str(expected.resolve()), "sha256": v5.EXPECTED_PROJECTION_SHA}


def test_projection_identity_fails_closed_on_path(tmp_path):
    copy = tmp_path / "projection.pt"
    copy.write_bytes(v5.PROJECTION.read_bytes())
    with pytest.raises(ValueError, match="path mismatch"):
        v5.assert_projection_identity(copy)


def test_projection_identity_fails_closed_on_hash(monkeypatch):
    monkeypatch.setattr(v5, "sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        v5.assert_projection_identity()


def test_manifest_pins_projection_identity_and_selection():
    manifest = v5.frozen_manifest()
    assert manifest["projection"]["path"] == str(v5.PROJECTION.resolve())
    assert manifest["projection"]["sha256"] == v5.EXPECTED_PROJECTION_SHA
    assert manifest["expected_projection_sha256"] == v5.EXPECTED_PROJECTION_SHA
    assert manifest["created_before_results"] is True
    assert len(manifest["specification_hash"]) == 64
    assert manifest["selection"]["outer_confirm_top_n"] == 2
    assert manifest["selection"]["balanced_logloss_used_for_gate"] is False
    assert manifest["cv"]["confirmation"]["seeds"] == [42, 73, 101]
    assert manifest["cv"]["confirmation"]["selection_seed_excluded"] is True


def test_cache_methods_are_not_claimed_runtime_compatible():
    for spec in v5.SPECS:
        original = v5.source_spec(spec)
        if original.method in {"knn", "kernel", "balanced_kernel", "hybrid", "adaptive", "fallback"}:
            assert spec.runtime_compatible is False


def test_loader_reads_only_the_physical_train_cache(monkeypatch):
    labels = torch.arange(v5.EXPECTED_CLASSES).repeat_interleave(2)
    labels = torch.cat((labels, torch.zeros(v5.EXPECTED_ROWS-len(labels), dtype=torch.long)))
    cache = {"features": torch.zeros(v5.EXPECTED_ROWS, 3), "labels": labels, "source_row_indices": torch.arange(v5.EXPECTED_ROWS), "class_names": None, "hidden_dim": 3, "partition": "train"}
    seen = []
    def fake_load(path, **kwargs):
        seen.append(Path(path))
        return cache
    monkeypatch.setattr(v5.torch, "load", fake_load)
    features, loaded_labels = v5.load_train_cache()
    assert seen == [v5.SEALED]
    assert len(features) == len(loaded_labels) == v5.EXPECTED_ROWS


def test_true_b1_scores_are_deterministic_and_finite():
    features = torch.randn(v5.EXPECTED_CLASSES*2, 384, generator=torch.Generator().manual_seed(9))
    labels = torch.arange(v5.EXPECTED_CLASSES).repeat_interleave(2)
    projection = v5.load_projection_b1()
    for index in (v5.CONTROL_INDEX, v5.KNN1_INDEX):
        first = v5.score_spec(v5.SPECS[index], features, labels, features[:7], projection)
        second = v5.score_spec(v5.SPECS[index], features, labels, features[:7], projection)
        assert first.shape == (7, v5.EXPECTED_CLASSES)
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()

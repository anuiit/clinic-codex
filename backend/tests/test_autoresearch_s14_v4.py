from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_s14_v4.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_s14_v4", SCRIPT)
assert SPEC and SPEC.loader
v4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v4
SPEC.loader.exec_module(v4)


def test_matrix_is_exactly_30_unique_and_baselines_are_fixed():
    v4.assert_specs()
    assert len(v4.SPECS) == 30
    assert len({item.name for item in v4.SPECS}) == 30
    assert [item.name for item in v4.SPECS[:2]] == ["baseline-b1-ncm", "baseline-cosine-k1"]


def test_targeted_families_and_grids_are_present():
    assert {item.method for item in v4.SPECS} == {"ncm", "knn", "kernel", "hybrid", "balanced_kernel", "adaptive", "fallback"}
    assert {item.kwargs["k"] for item in v4.SPECS if item.method == "knn" and item.transform == "b1"} >= {1, 2, 3, 4, 5, 7, 9, 15}
    assert {item.kwargs["alpha"] for item in v4.SPECS if item.method == "hybrid"} == {.25, .5, .75}
    assert {item.kwargs["margin"] for item in v4.SPECS if item.method == "fallback"} == {.02, .05, .1}
    assert {item.transform for item in v4.SPECS} >= {"b1", "centered", "power050", "power075", "zscore"}


def test_only_physical_train_cache_is_configured():
    assert v4.SEALED.name == "train-only.features.pt"
    assert v4.EXPECTED_ROWS == 9128
    assert v4.EXPECTED_CLASSES == 286


def test_loader_reads_exactly_the_sealed_cache(monkeypatch):
    seen = []
    labels = torch.arange(v4.EXPECTED_CLASSES).repeat_interleave(2)
    labels = torch.cat((labels, torch.zeros(v4.EXPECTED_ROWS-len(labels), dtype=torch.long)))
    cache = {
        "features": torch.zeros(v4.EXPECTED_ROWS, 2),
        "labels": labels,
        "source_row_indices": torch.arange(v4.EXPECTED_ROWS),
        "class_names": None,
        "hidden_dim": 2,
        "partition": "train",
    }

    def fake_load(path, **kwargs):
        seen.append(Path(path))
        return cache

    monkeypatch.setattr(v4.torch, "load", fake_load)
    features, loaded_labels = v4.load_train_cache()
    assert seen == [v4.SEALED]
    assert len(features) == len(loaded_labels) == v4.EXPECTED_ROWS


def test_transformations_are_finite_and_train_fitted():
    generator = torch.Generator().manual_seed(4)
    train = torch.randn(20, 8, generator=generator)
    query = torch.randn(4, 8, generator=generator)
    for name in ("b1", "centered", "power050", "power075", "zscore"):
        left, right = v4.transform_pair(train, query, name)
        assert left.shape == train.shape and right.shape == query.shape
        assert torch.isfinite(left).all() and torch.isfinite(right).all()


def test_neighbour_and_hybrid_scores_are_deterministic_and_finite():
    generator = torch.Generator().manual_seed(5)
    train = torch.randn(v4.EXPECTED_CLASSES*2, 16, generator=generator)
    labels = torch.arange(v4.EXPECTED_CLASSES).repeat_interleave(2)
    query = train[:13] + .01
    names = ("baseline-b1-ncm", "baseline-cosine-k1", "kernel-k15-beta10", "hybrid-k1-alpha050", "balanced-k31-beta5-a050", "adaptive-k-1-3-5", "margin-fallback-005")
    for name in names:
        spec = next(item for item in v4.SPECS if item.name == name)
        first = v4.score_spec(spec, train, labels, query)
        second = v4.score_spec(spec, train, labels, query)
        assert first.shape == (len(query), v4.EXPECTED_CLASSES)
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()


def test_euclidean_and_cosine_k1_agree_on_normalized_data():
    generator = torch.Generator().manual_seed(6)
    train = torch.nn.functional.normalize(torch.randn(100, 12, generator=generator), dim=1)
    query = torch.nn.functional.normalize(torch.randn(9, 12, generator=generator), dim=1)
    _, cosine = v4.neighbour_table(train, query, "cosine", 1)
    _, euclidean = v4.neighbour_table(train, query, "euclidean", 1)
    assert torch.equal(cosine, euclidean)


def test_selection_contract_is_frozen_before_results():
    assert v4.TOP_N == 5
    assert v4.SEEDS == (17, 42, 73)
    assert v4.MIN_GAIN_VS_KNN1 == pytest.approx(.002)
    assert v4.BUCKET_TOLERANCE == pytest.approx(.01)
    manifest = v4.frozen_manifest()
    assert manifest["created_before_results"] is True
    assert len(manifest["specification_hash"]) == 64

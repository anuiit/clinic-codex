from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_s14_v3.py"
MODULE_SPEC = importlib.util.spec_from_file_location("autoresearch_s14_v3", SCRIPT)
assert MODULE_SPEC and MODULE_SPEC.loader
v3 = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = v3
MODULE_SPEC.loader.exec_module(v3)


def test_preregistered_matrix_has_at_least_35_unique_specs():
    v3.assert_specs()
    assert len(v3.SPECS) >= 35
    assert len({item.name for item in v3.SPECS}) == len(v3.SPECS)
    assert v3.SPECS[0].name == "control-b1-ncm"


def test_required_method_families_and_values_are_preregistered():
    methods = {item.method for item in v3.SPECS}
    assert {"ncm", "simpleshot", "power", "zscore", "pca_whiten", "robust", "shrink", "ridge", "knn", "mahal_diag", "fecam_diag", "relu_ridge"} <= methods
    assert {item.kwargs.get("lam") for item in v3.SPECS if item.method == "ridge"} == {1e-4, 1e-3, 1e-2, .1, 1., 10., 100.}
    assert {item.kwargs.get("k") for item in v3.SPECS if item.method == "knn" and "beta" not in item.kwargs} == {1, 3, 5, 9, 15}
    assert {item.kwargs.get("width") for item in v3.SPECS if item.method == "relu_ridge"} == {256, 512, 1024}


def synthetic_labels() -> torch.Tensor:
    # Contract-sized labels; every class has two rows and the remainder class 0.
    labels = torch.arange(v3.EXPECTED_CLASSES).repeat_interleave(2)
    return torch.cat((labels, torch.zeros(v3.EXPECTED_ROWS-len(labels), dtype=torch.long)))


def test_stratified_folds_are_deterministic_exhaustive_disjoint_and_retain_classes():
    labels = synthetic_labels()
    for fold_count in (3, 5):
        first = v3.stratified_folds(labels, fold_count, 17)
        second = v3.stratified_folds(labels, fold_count, 17)
        assert all(torch.equal(left, right) for left, right in zip(first, second))
        flattened = torch.cat(first)
        assert len(flattened) == len(flattened.unique()) == v3.EXPECTED_ROWS
        assert set(flattened.tolist()) == set(range(v3.EXPECTED_ROWS))
        for holdout in first:
            mask = torch.ones(len(labels), dtype=torch.bool)
            mask[holdout] = False
            assert labels[mask].unique().numel() == v3.EXPECTED_CLASSES


def test_train_cache_validation_is_physical_and_rejects_extra_metadata():
    labels = synthetic_labels()
    cache = {
        "features": torch.zeros(v3.EXPECTED_ROWS, 3),
        "labels": labels,
        "source_row_indices": torch.arange(v3.EXPECTED_ROWS),
        "class_names": None,
        "hidden_dim": 3,
        "partition": "train",
    }
    v3.validate_train_cache(cache)
    cache["image_paths"] = ["x"] * v3.EXPECTED_ROWS
    with pytest.raises(ValueError, match="unexpected keys"):
        v3.validate_train_cache(cache)


def test_metric_math_support_buckets_and_gates():
    support = torch.full((v3.EXPECTED_CLASSES,), 8, dtype=torch.long)
    support[0] = 2
    records = []
    for label, count in enumerate(support.tolist()):
        for row in range(count):
            correct = row != 0
            records.append({"row_index": len(records), "label": label, "prediction": label if correct else (label+1) % v3.EXPECTED_CLASSES, "top3": [label] if correct else [0, 1, 2], "loss": 0.25})
    metrics = v3.metric_summary(records, support)
    expected_macro = sum((count-1)/count for count in support.tolist())/v3.EXPECTED_CLASSES
    assert metrics["macro_top1"] == pytest.approx(expected_macro)
    assert metrics["balanced_logloss"] == pytest.approx(.25)
    control = dict(metrics)
    candidate = dict(metrics)
    candidate["macro_top1"] += .006
    assert v3.screening_pass(candidate, control)
    bootstrap = {"lower_95": .001}
    candidate["macro_top1"] = control["macro_top1"] + .011
    assert v3.outer_pass(candidate, control, bootstrap)


def test_specs_and_random_features_are_deterministic():
    assert v3.stable_hash([item.name for item in v3.SPECS]) == v3.stable_hash([item.name for item in v3.SPECS])
    for width in (256, 512, 1024):
        assert v3.matrix_hash(128, width) == v3.matrix_hash(128, width)
        assert torch.equal(v3.random_matrix(128, width, torch.device("cpu")), v3.random_matrix(128, width, torch.device("cpu")))


def test_source_boundary_and_selection_do_not_consume_non_train_rows():
    source_text = SCRIPT.read_text(encoding="utf-8").lower()
    assert 'mark == "train"' in source_text
    assert "image_paths" not in {"features", "labels", "source_row_indices", "class_names", "hidden_dim", "partition"}
    assert v3.OUTER_SEEDS == (17, 42, 73)
    assert v3.TOP_N == 8
    assert v3.SCREEN_GAIN == pytest.approx(.005)
    assert v3.OUTER_GAIN == pytest.approx(.01)


def test_projection_and_core_scores_have_expected_shape_and_are_finite():
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(v3.EXPECTED_CLASSES*2, 8, generator=generator)
    y = torch.arange(v3.EXPECTED_CLASSES).repeat_interleave(2)
    q = x[:17]
    specs = (
        v3.Spec("n", "ncm", space="raw"),
        v3.Spec("s", "shrink", v3._p(lam=1.0), space="raw"),
        v3.Spec("r", "ridge", v3._p(lam=1.0), space="raw"),
        v3.Spec("k", "knn", v3._p(k=1), space="raw"),
        v3.Spec("m", "mahal_diag", v3._p(shrink=.5), space="raw"),
    )
    for spec in specs:
        scores = v3.scores_for(spec, x, y, q)
        assert scores.shape == (len(q), v3.EXPECTED_CLASSES)
        assert torch.isfinite(scores).all()

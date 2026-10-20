from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/autoresearch_dual_dataset_architectures.py"
SPEC = importlib.util.spec_from_file_location("dual_arch", SCRIPT)
assert SPEC and SPEC.loader
dual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dual
SPEC.loader.exec_module(dual)

requires_b1_projection = pytest.mark.skipif(
    not dual.B1_PROJECTION.exists(),
    reason="requires unshipped research artifact: B1 runtime weights/projection.pt",
)
requires_protocol_inputs = pytest.mark.skipif(
    not all(path.exists() for path in (dual.OLD_CACHE, dual.NEW_CACHE, dual.B1_PROJECTION)),
    reason="requires unshipped research artifact: B1 projection.pt and precomputed features.pt caches",
)


def dataset(name: str, train_x: torch.Tensor, train_y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor):
    return dual.Dataset(
        name=name,
        train_x=train_x,
        train_y=train_y,
        dev_x=dev_x,
        dev_y=dev_y,
        dev_groups=[f"g-{i}" for i in range(len(dev_y))],
        class_names={i: str(i) for i in range(286)},
        counts={"train": len(train_y), "dev": len(dev_y), "locked_test": 1},
    )


def test_architectures_are_normalized_and_distinct_sizes():
    x = torch.randn(5, 384)
    counts = []
    for kind in dual.ARCHITECTURES:
        model = dual.Architecture(kind).eval()
        output = model(x)
        assert output.shape == (5, 128)
        assert torch.allclose(output.norm(dim=1), torch.ones(5), atol=1e-5)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == len(counts)


def test_combined_train_removes_dev_overlap_and_same_label_duplicate():
    a, b, c, d = (torch.full((384,), float(i)) for i in range(4))
    old = dataset("old", torch.stack((a, b)), torch.tensor((0, 1)), torch.stack((c,)), torch.tensor((2,)))
    new = dataset("new", torch.stack((a, c, d)), torch.tensor((0, 2, 3)), torch.stack((b,)), torch.tensor((1,)))
    # The production guard also requires all 286 classes; isolate the dedup core by padding them.
    padding = torch.stack([torch.full((384,), float(i + 10)) for i in range(286)])
    new.train_x = torch.cat((new.train_x, padding))
    new.train_y = torch.cat((new.train_y, torch.arange(286)))
    features, labels, audit = dual.combined_train(old, new)
    assert audit["removed_exact_dev_features"] == 2
    assert audit["removed_exact_train_duplicates"] == 1
    assert len(features) == len(labels)
    assert set(labels.tolist()) == set(range(286))


def test_balanced_sampler_can_draw_singletons():
    groups = dual.class_indices(torch.tensor((0, 0, 1)))
    sample = dual.sample_balanced(groups, __import__("random").Random(3), 100)
    assert set(sample.tolist()) == {0, 1, 2}


@pytest.mark.research_artifacts
@requires_protocol_inputs
def test_protocol_has_exact_matrix_and_closed_boundaries():
    value = dual.protocol()
    assert len(value["screen_specs"]) == 12
    assert value["training"]["steps"] == 1500
    assert value["boundaries"] == {
        "locked_test_scored": False,
        "locked_test_rows_loaded_then_filtered": True,
        "business_final_read": False,
        "runtime_write": False,
        "promotion": False,
    }


def test_evaluate_forces_dropout_off():
    model = dual.Architecture("v0_mlp").train()
    train_x = torch.randn(286, 384)
    train_y = torch.arange(286)
    data = dataset("x", train_x, train_y, torch.randn(8, 384), torch.arange(8))
    dual.evaluate(model, train_x, train_y, data, data)
    assert not model.training
    assert not next(module for module in model.modules() if isinstance(module, torch.nn.Dropout)).training


@pytest.mark.research_artifacts
@requires_b1_projection
def test_b1_checkpoint_loads_into_v0():
    model = dual.Architecture("v0_mlp")
    state = torch.load(dual.B1_PROJECTION, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_confirmation_selection_excludes_control():
    rows = [
        {"architecture": "v0_mlp", "metrics": {"mean_macro_top1": 1.0}},
        {"architecture": "v1_ln_mlp", "metrics": {"mean_macro_top1": 0.8}},
        {"architecture": "v2_ln_linear", "metrics": {"mean_macro_top1": 0.7}},
        {"architecture": "v3_residual", "metrics": {"mean_macro_top1": 0.6}},
    ]
    assert [row["architecture"] for row in dual.select_top_candidates(rows)] == [
        "v1_ln_mlp",
        "v2_ln_linear",
    ]

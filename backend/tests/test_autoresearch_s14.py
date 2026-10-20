from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "autoresearch_s14.py"
_SPEC = importlib.util.spec_from_file_location("autoresearch_s14", _SCRIPT)
assert _SPEC and _SPEC.loader
s14 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = s14
_SPEC.loader.exec_module(s14)


def test_fixed_sweep_has_exactly_eleven_one_factor_specs():
    s14.assert_frozen_specs()
    assert len(s14.SPECS) == 11
    assert [spec.changed_factors(s14.CONTROL) for spec in s14.SPECS[1:]] == [
        ["learning_rate"], ["learning_rate"], ["epochs"], ["epochs"], ["weight_decay"],
        ["weight_decay"], ["temperature"], ["temperature"], ["noise_std"], ["mixup_prob"],
    ]


def _cache(splits: list[str]) -> dict:
    return {
        "features": torch.arange(len(splits) * 2, dtype=torch.float32).reshape(len(splits), 2),
        "labels": torch.tensor(([0] * (splits.count("train") - 1)) + [1] + ([0] * splits.count("dev")) + ([0] * splits.count("locked_test"))),
        "dataset_splits": splits,
        "source_groups": [f"group-{index}" for index in range(len(splits))],
    }


def test_sealed_split_guard_excludes_locked_rows():
    splits = ["train"] * 9128 + ["dev"] * 56
    train_features, train_labels, dev_features, dev_labels, dev_groups = s14.split_train_dev(_cache(splits))
    assert len(train_features) == len(train_labels) == 9128
    assert len(dev_features) == len(dev_labels) == len(dev_groups) == 56
    assert train_features[:, 0].max().item() < 2 * 9128
    assert dev_features[:, 0].min().item() == 2 * 9128


def test_persisted_split_guard_rejects_invalid_or_missing_lock_partition():
    with pytest.raises(ValueError, match="sealed cache"):
        s14.split_train_dev(_cache(["train"] * 9128 + ["dev"] * 56 + ["locked_test"] * 82))


def test_seal_cache_physically_omits_all_locked_rows(tmp_path):
    splits = ["train"] * 9128 + ["dev"] * 56 + ["locked_test"] * 82
    source = _cache(splits)
    source.update({"class_names": {0: "zero", 1: "one"}, "hidden_dim": 2, "image_paths": [f"p-{i}" for i in range(len(splits))]})
    source_path, sealed_path = tmp_path / "source.pt", tmp_path / "sealed.pt"
    torch.save(source, source_path)

    provenance = s14.seal_cache(source_path, sealed_path)
    sealed = torch.load(sealed_path, map_location="cpu", weights_only=False)

    assert provenance["sha256"] == s14.sha256(sealed_path)
    assert len(sealed["features"]) == 9184
    assert set(sealed["dataset_splits"]) == {"train", "dev"}
    assert "locked_test" not in sealed["dataset_splits"]
    assert "image_paths" not in sealed


def test_metric_math_and_keep_rule():
    control = {"J": 0.60, "per_class": {"0": {"accuracy": 0.8}, "1": {"accuracy": 0.5}}}
    candidate = {"J": 0.611, "per_class": {"0": {"accuracy": 0.7}, "1": {"accuracy": 0.5}}}
    assert s14.decision(candidate, control) == ("KEEP", [])
    candidate["per_class"]["0"]["accuracy"] = 0.59
    assert s14.decision(candidate, control) == ("DISCARD", ["0"])
    assert 0.75 * 0.8 + 0.25 * 0.4 == pytest.approx(0.7)


def test_forbidden_paths_are_rejected_and_protocol_inputs_are_allowed():
    for path in (Path("/tmp/business-final/features.pt"), Path("/tmp/annotations/features.pt"), Path("/tmp/locked_test/features.pt")):
        with pytest.raises(ValueError, match="forbidden"):
            s14.assert_allowed_input_path(path)
    s14.assert_allowed_input_path(s14.CACHE)
    s14.assert_allowed_input_path(s14.INIT_PROJECTION)


def test_training_invocation_uses_sealed_cache_and_pinned_environment(tmp_path):
    sealed = tmp_path / "sealed.pt"
    command = s14.training_command(s14.SPECS[-1], tmp_path / "config.yaml", tmp_path / "checkpoints", sealed)
    assert command[command.index("--features") + 1] == str(sealed)
    assert str(s14.CACHE) not in command
    assert command[-2:] == ["--mixup-prob", "0.25"]
    environment = s14.subprocess_environment()
    assert environment["PYTHONHASHSEED"] == "42"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

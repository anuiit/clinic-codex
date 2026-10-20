import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/autoresearch_tlachia_sparse.py"
spec = importlib.util.spec_from_file_location("tlachia_sparse", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_fixed_eight_specs():
    runner.assert_specs()
    assert len(runner.SPECS) == 8


def test_sparse_episode_has_disjoint_source_rows():
    by_class = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12, 13],
                4: [14, 15, 16, 17, 18, 19], 5: [20, 21, 22, 23, 24, 25, 26]}
    by_class.update({100 + i: list(range(1000 + 8 * i, 1008 + 8 * i)) for i in range(20)})
    episode = runner.episode_for_spec(by_class, runner.Spec("test", 6), __import__("random").Random(4))
    assert not (set(episode["support"]) & set(episode["query"]))
    assert len(episode["support"]) == 60 and len(episode["query"]) == 100


def test_singleton_is_prototype_only():
    by_class = {0: [999]}
    by_class.update({i: list(range(i * 10, i * 10 + 8)) for i in range(1, 22)})
    episode = runner.episode_for_spec(by_class, runner.Spec("test", 0), __import__("random").Random(4))
    assert 999 not in episode["support"] + episode["query"]


@pytest.mark.parametrize("path", ["/x/annotations/y.pt", "/x/business-final/y.pt", "/x/locked_test/y.pt"])
def test_forbidden_inputs(path):
    with pytest.raises(ValueError): runner.assert_allowed_input(Path(path))


def test_sealed_cache_rejects_locked_rows():
    cache = {"features": torch.zeros(2, 3), "labels": torch.tensor([0, 0]), "dataset_splits": ["train", "locked_test"], "source_groups": ["x", "x"], "row_ids": ["a", "b"]}
    with pytest.raises(ValueError): runner.validate_cache(cache)


def test_metric_bins_and_gate():
    candidate = {"macro_global": .6, "primary_sparse_macro": .7, "per_class": {"1": {"train_support": 2, "accuracy": .7}, "2": {"train_support": 5, "accuracy": .7}}}
    control = {"macro_global": .59, "primary_sparse_macro": .65, "per_class": {"1": {"train_support": 2, "accuracy": .6}, "2": {"train_support": 5, "accuracy": .6}}}
    outcome = runner.gate(candidate, control, [.02, .03])
    assert outcome["pass"] is True

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("b1_adapter", SCRIPTS / "autoresearch_b1_residual_adapter.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def test_zero_gate_reproduces_b1_embeddings():
    state = torch.load(adapter.dual.B1_PROJECTION, map_location="cpu", weights_only=True)
    model = adapter.ResidualAdapter(state).eval()
    x = torch.randn(7, 384)
    assert torch.allclose(model(x), model.base(x), atol=1e-6, rtol=0.0)


def test_training_adapter_keeps_frozen_b1_deterministic():
    state = torch.load(adapter.dual.B1_PROJECTION, map_location="cpu", weights_only=True)
    model = adapter.ResidualAdapter(state).train()
    assert model.training
    assert not model.base.training
    assert not next(module for module in model.base.modules() if isinstance(module, torch.nn.Dropout)).training


def test_protocol_is_explicitly_exploratory_and_closed():
    value = adapter.protocol()
    assert "exploratory" in value["selection_caveat"]
    assert value["training"]["seeds"] == [42, 73, 101]
    assert not value["boundaries"]["locked_test_scored"]
    assert value["boundaries"]["locked_test_rows_loaded_then_filtered"]
    assert not value["boundaries"]["business_final_read"]
    assert not value["boundaries"]["runtime_write"]
    assert not value["boundaries"]["promotion"]

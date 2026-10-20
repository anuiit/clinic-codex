from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluate_s14_v5_business.py"
SPEC = importlib.util.spec_from_file_location("evaluate_s14_v5_business", SCRIPT)
assert SPEC and SPEC.loader
terminal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terminal
SPEC.loader.exec_module(terminal)


def test_protocol_has_exactly_four_frozen_methods_and_no_tuning():
    protocol = terminal.protocol_spec()
    assert tuple(protocol["method_order"]) == terminal.METHOD_NAMES
    assert len(protocol["methods"]) == 4
    assert protocol["eligible_for_training_or_tuning"] is False
    assert protocol["interpretation"]["no_selection_or_tuning"] is True
    assert protocol["interpretation"]["runtime_compatible"] is False


def test_margin_fallback_exact_rule():
    cache = torch.tensor([[9.0, 1.0, 0.0], [8.0, 2.0, 0.0]])
    ncm = torch.tensor([[1.0, 8.0, 0.0], [1.0, 8.0, 0.0]])
    neighbours = torch.tensor([[.90, .87], [.90, .89]])
    scores, use_cache = terminal.fallback_scores(cache, ncm, neighbours, margin=.02)
    assert use_cache.tolist() == [True, False]
    assert torch.equal(scores[0], cache[0])
    assert torch.equal(scores[1], ncm[1])


def test_fallback_rejects_non_top_two_shape():
    with pytest.raises(ValueError, match="exactly two"):
        terminal.fallback_scores(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3))


def test_projection_path_and_hash_fail_closed(tmp_path, monkeypatch):
    copied = tmp_path / "projection.pt"
    copied.write_bytes(terminal.B1_PROJECTION.read_bytes())
    with pytest.raises(ValueError, match="path mismatch"):
        terminal.assert_expected_file(copied, terminal.B1_PROJECTION, terminal.EXPECTED_PROJECTION_SHA, "B1 projection")
    monkeypatch.setattr(terminal, "sha256", lambda path: "0"*64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        terminal.assert_expected_file(terminal.B1_PROJECTION, terminal.B1_PROJECTION, terminal.EXPECTED_PROJECTION_SHA, "B1 projection")


def test_caller_hash_rejects_bad_format_and_bad_content(tmp_path):
    path = tmp_path / "input.pt"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        terminal.assert_caller_hash(path, "BAD", "business features")
    with pytest.raises(ValueError, match="caller SHA mismatch"):
        terminal.assert_caller_hash(path, "0"*64, "business features")


def test_write_once_refuses_existing_output(tmp_path):
    output = tmp_path / "result.json"
    terminal.write_json_once(output, {"first": True})
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        terminal.write_json_once(output, {"second": True})
    assert output.read_bytes() == original


def test_metrics_include_per_page_per_class_and_rows():
    scores = torch.tensor([[3., 2., 1.], [1., 3., 2.], [3., 1., 2.]])
    labels = torch.tensor([0, 1, 2])
    metrics, rows, correct = terminal.metrics_for(scores, labels, ["a", "a", "b"], ["a", "b", "c"], ["r0", "r1", "r2"])
    assert metrics["page_count"] == 2
    assert len(metrics["per_page"]) == 2
    assert len(metrics["per_class"]) == 3
    assert len(rows) == 3
    assert correct.tolist() == [True, True, False]


def test_preflight_api_has_no_business_parameters():
    assert terminal.preflight.__annotations__.keys() == {"spec_output", "return"}
    parser = terminal.parser()
    args = parser.parse_args(["preflight", "--spec-output", "x.json"])
    assert set(vars(args)) == {"command", "spec_output"}


def _identity(suffix: str = "0"):
    return terminal.terminal_identity(
        business_content_sha256=suffix*64,
        business_manifest_sha256="1"*64,
        business_features_sha256="2"*64,
        business_provenance_sha256="3"*64,
        specification_sha256="4"*64,
    )


def test_global_one_shot_refuses_second_different_request(tmp_path):
    ledger, output = tmp_path/"terminal-ledger.json", tmp_path/"result.json"
    terminal._reserve_one_shot_at(_identity("0"), ledger_path=ledger, output_path=output)
    with pytest.raises(RuntimeError, match="already reserved"):
        terminal._reserve_one_shot_at(_identity("a"), ledger_path=ledger, output_path=tmp_path/"different-result.json")


def test_reserved_crash_state_is_durable_and_cannot_auto_retry(tmp_path):
    ledger, output = tmp_path/"terminal-ledger.json", tmp_path/"result.json"
    identity = _identity("b")
    terminal._reserve_one_shot_at(identity, ledger_path=ledger, output_path=output)
    persisted = json.loads(ledger.read_text(encoding="utf-8"))
    assert persisted["state"] == "reserved"
    assert "started_at" not in persisted and "completed_at" not in persisted
    with pytest.raises(RuntimeError, match="already reserved"):
        terminal._reserve_one_shot_at(identity, ledger_path=ledger, output_path=output)


def test_started_crash_state_is_auditable(tmp_path):
    ledger, output = tmp_path/"terminal-ledger.json", tmp_path/"result.json"
    identity = _identity("c")
    reserved = terminal._reserve_one_shot_at(identity, ledger_path=ledger, output_path=output)
    terminal.transition_ledger(ledger_path=ledger, terminal_key=reserved["terminal_key"], expected_state="reserved", new_state="started")
    persisted = json.loads(ledger.read_text(encoding="utf-8"))
    assert persisted["state"] == "started" and "started_at" in persisted
    with pytest.raises(RuntimeError, match="already reserved"):
        terminal._reserve_one_shot_at(identity, ledger_path=ledger, output_path=output)


def test_frozen_spec_substitution_is_rejected(tmp_path):
    substitute = tmp_path/"frozen-spec.json"
    substitute.write_bytes(terminal.FROZEN_SPEC.read_bytes())
    with pytest.raises(ValueError, match="exact fixed frozen spec path"):
        terminal.validate_frozen_spec(substitute)


def test_exact_frozen_spec_hash_is_enforced(monkeypatch):
    original = terminal.sha256
    monkeypatch.setattr(terminal, "sha256", lambda path: "0"*64 if Path(path).resolve() == terminal.FROZEN_SPEC.resolve() else original(Path(path)))
    with pytest.raises(ValueError, match="spec file SHA mismatch"):
        terminal.validate_frozen_spec(terminal.FROZEN_SPEC)


def test_device_batch_and_output_are_not_cli_overridable():
    protocol = terminal.protocol_spec()
    assert protocol["execution"]["device"] == "cpu"
    assert protocol["execution"]["batch_size"] == 4096
    assert protocol["execution"]["fixed_output"] == str(terminal.TERMINAL_OUTPUT.resolve())
    parser = terminal.parser()
    subparsers = next(action for action in parser._actions if hasattr(action, "choices") and isinstance(action.choices, dict) and "evaluate" in action.choices)
    options = {option for action in subparsers.choices["evaluate"]._actions for option in action.option_strings}
    assert "--device" not in options
    assert "--batch-size" not in options
    assert "--output" not in options
    assert "--spec" not in options

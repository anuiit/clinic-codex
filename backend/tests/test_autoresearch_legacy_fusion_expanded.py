from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/autoresearch_legacy_fusion_expanded.py"
SPEC = importlib.util.spec_from_file_location("legacy_fusion_expanded", SCRIPT)
assert SPEC and SPEC.loader
expanded = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = expanded
SPEC.loader.exec_module(expanded)


def _row(group: str, index: int, classes: list[str]) -> dict:
    return {
        "source_group": group,
        "archive_name": f"a{index}.zip",
        "entry_path": f"g{index}.jpg",
        "candidate_runtime_classes": classes,
    }


def test_selection_is_deterministic_and_group_bounded(monkeypatch):
    monkeypatch.setattr(expanded, "TARGET_FILES", 6)
    monkeypatch.setattr(expanded, "MIN_GROUPS", 3)
    monkeypatch.setattr(expanded, "MAX_PER_GROUP", 2)
    monkeypatch.setattr(expanded, "_taxonomy", lambda: ({"a", "b", "c"}, {"a"}))
    rows = [
        _row("g1", 1, ["a"]), _row("g1", 2, ["b"]),
        _row("g2", 3, ["c"]), _row("g2", 4, ["a"]),
        _row("g3", 5, ["b"]), _row("g3", 6, ["c"]),
        _row("g4", 7, ["a"]),
    ]
    first = expanded.select_entries(rows, {"a"}, target=6)
    second = expanded.select_entries(list(reversed(rows)), {"a"}, target=6)
    assert [(r["archive_name"], r["entry_path"]) for r in first] == [
        (r["archive_name"], r["entry_path"]) for r in second
    ]
    assert len(first) == 6
    assert len({r["source_group"] for r in first}) >= 3
    assert {c for r in first for c in r["candidate_runtime_classes"]} == {"a", "b", "c"}


def test_frozen_budgets_are_bounded():
    assert expanded.LIMITS == {
        "max_files": 2_000,
        "max_groups": 400,
        "max_compressed_bytes": 80_000_000,
        "max_uncompressed_bytes": 100_000_000,
        "max_requests": 4_000,
        "max_pixel_count": 1_000_000_000,
    }
    assert expanded.DEV_GROUPS == 50
    assert expanded.MAX_PER_GROUP == 10


def test_p4_dev_groups_are_seeded_before_general_selection(monkeypatch):
    monkeypatch.setattr(expanded, "TARGET_FILES", 4)
    monkeypatch.setattr(expanded, "MIN_GROUPS", 2)
    monkeypatch.setattr(expanded, "MAX_PER_GROUP", 2)
    monkeypatch.setitem(expanded.LIMITS, "max_groups", 3)
    monkeypatch.setattr(expanded, "P4_FROZEN_DEV_GROUPS", frozenset({"p4"}))
    monkeypatch.setattr(expanded, "_taxonomy", lambda: ({"a", "b"}, {"a"}))
    rows = [
        _row("p4", 1, ["b"]),
        _row("other", 2, ["a"]),
        _row("other", 3, ["b"]),
        _row("third", 4, ["a"]),
        _row("p4", 5, ["a"]),
    ]
    selected = expanded.select_entries(rows, {"a"}, target=4)
    assert "p4" in {row["source_group"] for row in selected}


def test_download_validation_rejects_non_expanded_artifact(monkeypatch):
    fake = {"payload": {"research_subpurpose": "other"}}
    monkeypatch.setattr(expanded, "verify_artifact", lambda *_: None)
    try:
        expanded.validate_expanded_plan(fake)
    except expanded.PolicyError as error:
        assert "not an expanded" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-expanded plan was accepted")


def test_written_artifact_round_trips_semantic_hash(tmp_path, monkeypatch):
    payload = {"purpose": "multiview_self_supervised", "entries": [], "counts": {}}
    value = expanded.artifact(expanded.MEDIA_PLAN_SCHEMA, payload)
    output = tmp_path / "plan.json"
    expanded.write_json(output, value, write_once=True)
    reread = expanded.read_json(output)
    expanded.verify_artifact(reread, expanded.MEDIA_PLAN_SCHEMA)
    assert reread["payload_sha256"] == value["payload_sha256"]


def test_expanded_evaluation_never_auto_authorizes_global_training():
    source = Path(expanded.__file__).read_text(encoding="utf-8")
    assert '"automatic_global_training_allowed": False' in source
    assert '"light_model_iteration_gate_passed"' in source
    assert 'artifact("legacy-fusion-expanded-evaluation.v1"' not in source


def test_shared_rerank_preserves_every_b1_only_position():
    baseline = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    candidate = np.array([[0.1, 0.8, 1.0, 0.6]], dtype=np.float32)
    result = expanded.rerank_shared_in_place(baseline, candidate, (0, 2))
    assert np.array_equal(result[:, (1, 3)], baseline[:, (1, 3)])
    assert sorted(result[0, (0, 2)]) == sorted(baseline[0, (0, 2)])
    assert result[0, 2] > result[0, 0]


def test_expanded_rows_bind_files_and_apply_dynamic_run50_audit(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    entries, files = [], []
    for index, split in enumerate(("train", "dev")):
        relative = f"a{index}/g{index}.jpg"
        path = root / relative
        path.parent.mkdir()
        Image.new("RGB", (2, 2), color=(255 * index, 0, 0)).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {
            "archive_name": f"a{index}.zip", "entry_path": f"g{index}.jpg",
            "source_group": f"group-{index}", "split_group": f"group-{index}",
            "physical_source_group": f"group-{index}", "benchmark_split": split,
            "roles": ["glyph_occurrence"], "multilabel_evidence_kind": "explicit_descendant_multilabel_evidence",
            "candidate_runtime_classes": ["a"], "mapping_statuses": ["accepted_exact"],
            "mapping_ids": [f"m{index}"], "node_keys": [f"n{index}"],
        }
        entries.append(entry)
        files.append({
            **{key: entry[key] for key in ("archive_name", "entry_path", "physical_source_group", "candidate_runtime_classes", "mapping_statuses", "mapping_ids", "node_keys")},
            "output_path": relative, "byte_sha256": digest,
            "image": {"pixel_sha256": digest},
        })
    fake = SimpleNamespace(
        semantic_hash=lambda value: hashlib.sha256(repr(value).encode()).hexdigest(),
        _pixel_identity=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        _dhash64=lambda _path: "0000000000000000",
        load_run50_seen_pixels=lambda: (set(), set(), {"rows": 0}),
        exclude_leakage=lambda rows, *_: (list(rows), {"reasons": {}}),
    )
    monkeypatch.setattr(expanded, "validate_expanded_plan", lambda _plan: None)
    monkeypatch.setattr(expanded, "verify_download", lambda *_: None)
    monkeypatch.setattr(expanded, "_load_fusion_module", lambda: fake)
    monkeypatch.setattr(expanded, "DEV_GROUPS", 1)
    rows, audit = expanded.load_expanded_rows(
        {"payload": {"entries": entries}}, {"payload": {"files": files}}, root,
        SimpleNamespace(runtime_names=("a", "b")),
    )
    assert [row["split"] for row in rows] == ["train", "dev"]
    assert audit["strict_rows_after_run50_exclusion"] == 2
    assert len(audit["retained_rows_binding_sha256"]) == 64
    assert audit["retained_split_rows"] == {"train": 1, "dev": 1}
    files[0]["byte_sha256"] = "0" * 64
    try:
        expanded.load_expanded_rows(
            {"payload": {"entries": entries}}, {"payload": {"files": files}}, root,
            SimpleNamespace(runtime_names=("a", "b")),
        )
    except expanded.PolicyError as error:
        assert "byte identity" in str(error)
    else:  # pragma: no cover
        raise AssertionError("byte drift was accepted")

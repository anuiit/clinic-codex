from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_split_feasibility.py"
SPEC = importlib.util.spec_from_file_location("audit_split_feasibility", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _manifest() -> dict:
    class_order = ["alpha", "beta", "gamma"]
    definitions = [
        ("component-a1", "group-a1", "train", [("alpha", 2)]),
        ("component-a2", "group-a2", "locked_test", [("alpha", 1)]),
        ("component-b1", "group-b1", "train", [("beta", 3)]),
        ("component-g1", "group-g1", "train", [("gamma", 2)]),
        ("component-g2", "group-g2", "dev", [("gamma", 1)]),
    ]
    rows = []
    components = []
    assignments = {}
    counters = {name: 0 for name in class_order}
    for component_id, group, split, class_counts in definitions:
        row_ids = []
        class_names = []
        for class_name, count in class_counts:
            class_names.append(class_name)
            for _ in range(count):
                counters[class_name] += 1
                row_id = f"row-{class_name}-{counters[class_name]}"
                row_ids.append(row_id)
                rows.append(
                    {
                        "row_id": row_id,
                        "class_name": class_name,
                        "class_label": class_order.index(class_name),
                        "source_group": group,
                        "dataset_split": split,
                    }
                )
        assignments[group] = split
        components.append(
            {
                "component_id": component_id,
                "row_ids": row_ids,
                "source_groups": [group],
                "class_names": class_names,
                "dataset_split": split,
            }
        )
    split_counts = {
        split: sum(row["dataset_split"] == split for row in rows) for split in audit.SPLITS
    }
    return {
        "schema_version": "training-snapshot.v2",
        "snapshot_id": "snapshot-test",
        "content_sha256": "a" * 64,
        "class_order": class_order,
        "class_count": len(class_order),
        "row_count": len(rows),
        "source_group_count": len(assignments),
        "component_count": len(components),
        "split_counts": split_counts,
        "source_group_assignments": assignments,
        "policy": {"dev_fraction": 0.2, "locked_test_fraction": 0.2},
        "rows": rows,
        "components": components,
    }


def test_audit_distinguishes_upper_bound_and_feasible_lower_bound() -> None:
    report = audit.analyze_manifest(
        _manifest(), min_eval_rows=3, min_eval_classes=3, min_train_rows_per_class=1
    )

    feasibility = report["feasibility"]
    assert feasibility["verdict"] == "impossible_with_current_corpus_under_constraints"
    assert feasibility["holdout_eligible_class_upper_bound"] == 2
    assert feasibility["holdout_eligible_class_names"] == ["alpha", "gamma"]
    assert feasibility["optimistic_eval_row_upper_bound"] == 6
    assert feasibility["upper_bound_is_jointly_achievable"] is False
    assert feasibility["greedy_feasible_lower_bound"]["class_count"] == 2
    assert feasibility["greedy_feasible_lower_bound"]["row_count"] == 4
    beta = next(item for item in report["classes"] if item["class_name"] == "beta")
    assert beta["holdout_eligible"] is False
    assert beta["holdout_ineligible_reason"] == "single_duplicate_connected_component"
    assert report["collection"]["fresh_repartition_optimistic"][
        "additional_holdout_eligible_classes_needed"
    ] == 1
    assert report["collection"]["append_only_locked_test"] == {
        "additional_eval_rows_required": 2,
        "additional_eval_classes_required": 2,
        "new_independent_rows_lower_bound": 2,
        "new_class_component_incidences_lower_bound": 2,
        "ideal_new_rows_for_both_eval_splits_at_gate": 4,
        "ideal_new_rows_to_restore_requested_fractions": 3,
        "lower_bound_is_sufficient": False,
        "operational_note": (
            "Collect fresh source components disjoint from every persisted group. "
            "The row minima assume every usable new row reaches the intended heldout "
            "split; fixed hashing, balancing, group atomicity, exact-pixel "
            "deduplication, and conflicts can require a larger candidate pool."
        ),
    }


def test_current_split_metrics_detect_target_deficit_and_dominance() -> None:
    report = audit.analyze_manifest(_manifest(), min_eval_rows=2, min_eval_classes=1)

    locked = report["current_split"]["metrics"]["locked_test"]
    assert locked["row_count"] == 1
    assert locked["class_count"] == 1
    assert locked["largest_class_fraction"] == 1.0
    assert locked["target_row_deficit"] == 0
    assert "current_locked_test_class_dominance_above_25pct" in report[
        "feasibility"
    ]["blocking_reasons"]


def test_audit_rejects_incomplete_component_partition() -> None:
    manifest = _manifest()
    manifest["components"][0]["row_ids"].pop()

    with pytest.raises(ValueError, match="omits rows"):
        audit.analyze_manifest(manifest)


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), -0.1, 1.0])
def test_audit_rejects_invalid_split_fractions(fraction: float) -> None:
    manifest = _manifest()
    manifest["policy"]["dev_fraction"] = fraction

    with pytest.raises(ValueError, match="split fractions"):
        audit.analyze_manifest(manifest)


def test_audit_rejects_heldout_fraction_sum_at_least_one() -> None:
    manifest = _manifest()
    manifest["policy"].update({"dev_fraction": 0.5, "locked_test_fraction": 0.5})

    with pytest.raises(ValueError, match="must sum to less than 1"):
        audit.analyze_manifest(manifest)


def test_write_once_accepts_identical_content_and_refuses_drift(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    assert audit._write_once(target, b"first\n") == "created"
    assert audit._write_once(target, b"first\n") == "verified_noop"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit._write_once(target, b"second\n")


def test_cli_writes_deterministic_reports(tmp_path: Path) -> None:
    source = tmp_path / "snapshot.json"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    source.write_text(json.dumps(_manifest()), encoding="utf-8")

    args = [
        "--snapshot-manifest",
        str(source),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_md),
        "--min-eval-rows",
        "3",
        "--min-eval-classes",
        "3",
    ]
    assert audit.main(args) == 0
    first_json = output_json.read_bytes()
    first_md = output_md.read_bytes()
    assert audit.main(args) == 0
    assert output_json.read_bytes() == first_json
    assert output_md.read_bytes() == first_md
    assert b"impossible_with_current_corpus_under_constraints" in first_md

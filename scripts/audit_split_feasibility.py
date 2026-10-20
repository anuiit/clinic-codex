#!/usr/bin/env python3
"""Audit whether a training snapshot can support a leakage-safe holdout.

The audit is deliberately read-only.  It validates the persisted snapshot,
measures the current split against its requested fractions, and computes
conservative feasibility bounds at duplicate-connected source-component
granularity.  It never rewrites the snapshot or searches model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "split-feasibility-audit.v1"
SNAPSHOT_SCHEMA = "training-snapshot.v2"
SPLITS = ("train", "dev", "locked_test")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot manifest root must be an object")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(
            f"expected {SNAPSHOT_SCHEMA}, got {payload.get('schema_version')!r}"
        )
    return payload


def _validate_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    class_order = manifest.get("class_order")
    rows = manifest.get("rows")
    components = manifest.get("components")
    assignments = manifest.get("source_group_assignments")
    policy = manifest.get("policy")
    if not isinstance(class_order, list) or not class_order or not all(
        isinstance(item, str) and item for item in class_order
    ):
        raise ValueError("class_order must be a non-empty list of names")
    if len(class_order) != len(set(class_order)):
        raise ValueError("class_order contains duplicates")
    if not isinstance(rows, list) or not isinstance(components, list):
        raise ValueError("snapshot rows/components must be lists")
    if not isinstance(assignments, dict) or not isinstance(policy, dict):
        raise ValueError("snapshot assignments/policy must be objects")

    row_by_id: dict[str, dict[str, Any]] = {}
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actual_split_counts: Counter[str] = Counter()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{position}] must be an object")
        row_id = row.get("row_id")
        class_name = row.get("class_name")
        class_label = row.get("class_label")
        source_group = row.get("source_group")
        split = row.get("dataset_split")
        if not isinstance(row_id, str) or not row_id or row_id in row_by_id:
            raise ValueError(f"invalid or duplicate row_id: {row_id!r}")
        if not isinstance(class_name, str) or class_name not in class_order:
            raise ValueError(f"unknown class for {row_id}: {class_name!r}")
        expected_label = class_order.index(class_name)
        if class_label != expected_label:
            raise ValueError(
                f"class label mismatch for {row_id}: {class_label!r} != {expected_label}"
            )
        if not isinstance(source_group, str) or not source_group:
            raise ValueError(f"invalid source_group for {row_id}")
        if split not in SPLITS:
            raise ValueError(f"invalid dataset_split for {row_id}: {split!r}")
        if assignments.get(source_group) != split:
            raise ValueError(f"source-group assignment mismatch for {row_id}")
        row_by_id[row_id] = row
        rows_by_group[source_group].append(row)
        actual_split_counts[split] += 1

    component_by_id: dict[str, dict[str, Any]] = {}
    row_to_component: dict[str, str] = {}
    groups_to_component: dict[str, str] = {}
    for position, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"components[{position}] must be an object")
        component_id = component.get("component_id")
        row_ids = component.get("row_ids")
        source_groups = component.get("source_groups")
        split = component.get("dataset_split")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_by_id
        ):
            raise ValueError(f"invalid or duplicate component_id: {component_id!r}")
        if not isinstance(row_ids, list) or not row_ids:
            raise ValueError(f"component {component_id} has no row_ids")
        if not isinstance(source_groups, list) or not source_groups:
            raise ValueError(f"component {component_id} has no source_groups")
        if split not in SPLITS:
            raise ValueError(f"component {component_id} has invalid split: {split!r}")
        component_rows: list[dict[str, Any]] = []
        for row_id in row_ids:
            if row_id not in row_by_id:
                raise ValueError(f"component {component_id} references unknown row {row_id!r}")
            if row_id in row_to_component:
                raise ValueError(f"row belongs to multiple components: {row_id}")
            row_to_component[row_id] = component_id
            component_rows.append(row_by_id[row_id])
        actual_groups = {row["source_group"] for row in component_rows}
        if actual_groups != set(source_groups):
            raise ValueError(f"component {component_id} source_groups do not match rows")
        if {row["dataset_split"] for row in component_rows} != {split}:
            raise ValueError(f"component {component_id} spans persisted splits")
        for group in source_groups:
            if group in groups_to_component:
                raise ValueError(f"source group belongs to multiple components: {group}")
            groups_to_component[group] = component_id
        component_by_id[component_id] = {
            **component,
            "rows_resolved": component_rows,
        }

    if set(row_to_component) != set(row_by_id):
        missing = sorted(set(row_by_id) - set(row_to_component))
        raise ValueError(f"component partition omits rows: {missing[:5]}")
    if set(groups_to_component) != set(rows_by_group):
        raise ValueError("component partition does not cover source groups exactly")
    if manifest.get("row_count") != len(rows):
        raise ValueError("row_count does not match rows")
    if manifest.get("class_count") != len(class_order):
        raise ValueError("class_count does not match class_order")
    if manifest.get("component_count") != len(components):
        raise ValueError("component_count does not match components")
    if manifest.get("source_group_count") != len(rows_by_group):
        raise ValueError("source_group_count does not match rows")
    recorded_counts = manifest.get("split_counts")
    if not isinstance(recorded_counts, dict) or any(
        recorded_counts.get(split) != actual_split_counts[split] for split in SPLITS
    ):
        raise ValueError("split_counts do not match rows")

    return {
        "class_order": class_order,
        "rows": rows,
        "row_by_id": row_by_id,
        "components": component_by_id,
        "rows_by_group": rows_by_group,
        "split_counts": actual_split_counts,
        "policy": policy,
    }


def _component_counts(validated: dict[str, Any]) -> dict[str, Counter[str]]:
    return {
        component_id: Counter(row["class_name"] for row in component["rows_resolved"])
        for component_id, component in validated["components"].items()
    }


def _greedy_feasible_holdout(
    component_counts: dict[str, Counter[str]],
    component_sizes: dict[str, int],
    total_class_counts: Counter[str],
    *,
    min_train_rows_per_class: int,
) -> dict[str, Any]:
    remaining = Counter(total_class_counts)
    covered: set[str] = set()
    selected: list[str] = []
    remaining_ids = set(component_counts)
    while True:
        eligible = [
            component_id
            for component_id in remaining_ids
            if all(
                remaining[class_name] - count >= min_train_rows_per_class
                for class_name, count in component_counts[component_id].items()
            )
        ]
        if not eligible:
            break
        eligible.sort(
            key=lambda component_id: (
                -len(set(component_counts[component_id]) - covered),
                -component_sizes[component_id],
                component_id,
            )
        )
        chosen = eligible[0]
        selected.append(chosen)
        remaining.subtract(component_counts[chosen])
        covered.update(component_counts[chosen])
        remaining_ids.remove(chosen)
    return {
        "component_ids": selected,
        "component_count": len(selected),
        "row_count": sum(component_sizes[item] for item in selected),
        "class_count": len(covered),
        "class_names": sorted(covered),
        "method": "deterministic_greedy_new_class_then_rows",
        "is_optimal": False,
    }


def analyze_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    min_eval_rows: int = 600,
    min_eval_classes: int = 100,
    min_train_rows_per_class: int = 1,
) -> dict[str, Any]:
    min_eval_rows = _require_int(min_eval_rows, name="min_eval_rows", minimum=1)
    min_eval_classes = _require_int(min_eval_classes, name="min_eval_classes", minimum=1)
    min_train_rows_per_class = _require_int(
        min_train_rows_per_class,
        name="min_train_rows_per_class",
        minimum=1,
    )
    validated = _validate_snapshot(manifest)
    rows = validated["rows"]
    class_order: list[str] = validated["class_order"]
    components: dict[str, dict[str, Any]] = validated["components"]
    component_counts = _component_counts(validated)
    component_sizes = {
        component_id: len(component["rows_resolved"])
        for component_id, component in components.items()
    }
    total_class_counts = Counter(row["class_name"] for row in rows)
    class_components: dict[str, set[str]] = defaultdict(set)
    class_groups: dict[str, set[str]] = defaultdict(set)
    class_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for component_id, component in components.items():
        for class_name in component_counts[component_id]:
            class_components[class_name].add(component_id)
        for row in component["rows_resolved"]:
            class_groups[row["class_name"]].add(row["source_group"])
            class_split_counts[row["class_name"]][row["dataset_split"]] += 1

    individually_cedable = {
        component_id
        for component_id, counts in component_counts.items()
        if all(
            total_class_counts[class_name] - count >= min_train_rows_per_class
            for class_name, count in counts.items()
        )
    }
    upper_classes = sorted(
        {
            class_name
            for component_id in individually_cedable
            for class_name in component_counts[component_id]
        }
    )
    optimistic_upper_rows = sum(component_sizes[item] for item in individually_cedable)
    optimistic_upper_groups = len(
        {
            group
            for component_id in individually_cedable
            for group in components[component_id]["source_groups"]
        }
    )
    greedy = _greedy_feasible_holdout(
        component_counts,
        component_sizes,
        total_class_counts,
        min_train_rows_per_class=min_train_rows_per_class,
    )

    policy = validated["policy"]
    row_count = len(rows)
    requested_fractions = {
        "dev": float(policy.get("dev_fraction", 0.0)),
        "locked_test": float(policy.get("locked_test_fraction", 0.0)),
    }
    if any(
        not math.isfinite(fraction) or not 0.0 <= fraction < 1.0
        for fraction in requested_fractions.values()
    ):
        raise ValueError("snapshot split fractions must be finite values in [0, 1)")
    if sum(requested_fractions.values()) >= 1.0:
        raise ValueError("snapshot heldout split fractions must sum to less than 1")
    requested_rows = {
        split: max(1, int(row_count * fraction)) if fraction else 0
        for split, fraction in requested_fractions.items()
    }
    current_counts: Counter[str] = validated["split_counts"]
    current_metrics: dict[str, Any] = {}
    for split in ("dev", "locked_test"):
        split_rows = [row for row in rows if row["dataset_split"] == split]
        class_counts = Counter(row["class_name"] for row in split_rows)
        group_counts = Counter(row["source_group"] for row in split_rows)
        current_metrics[split] = {
            "row_count": len(split_rows),
            "class_count": len(class_counts),
            "source_group_count": len(group_counts),
            "fraction_achieved": len(split_rows) / row_count if row_count else 0.0,
            "fraction_requested": requested_fractions[split],
            "target_row_count": requested_rows[split],
            "target_row_deficit": max(0, requested_rows[split] - len(split_rows)),
            "largest_class": class_counts.most_common(1)[0][0] if class_counts else None,
            "largest_class_fraction": (
                class_counts.most_common(1)[0][1] / len(split_rows) if split_rows else 0.0
            ),
            "largest_source_group": (
                group_counts.most_common(1)[0][0] if group_counts else None
            ),
            "largest_source_group_fraction": (
                group_counts.most_common(1)[0][1] / len(split_rows) if split_rows else 0.0
            ),
        }

    current_train_counts = Counter(
        row["class_name"] for row in rows if row["dataset_split"] == "train"
    )
    remaining_eligible_train_components = []
    for component_id, component in components.items():
        if component["dataset_split"] != "train":
            continue
        if all(
            current_train_counts[class_name] - count >= min_train_rows_per_class
            for class_name, count in component_counts[component_id].items()
        ):
            remaining_eligible_train_components.append(component_id)
    target_deficits = {
        split: current_metrics[split]["target_row_deficit"]
        for split in ("dev", "locked_test")
    }
    splitter_exhausted = any(target_deficits.values()) and not remaining_eligible_train_components

    class_details: list[dict[str, Any]] = []
    for class_name in class_order:
        eligible_components = sorted(class_components[class_name] & individually_cedable)
        component_count = len(class_components[class_name])
        if eligible_components:
            reason = None
        elif component_count <= 1:
            reason = "single_duplicate_connected_component"
        else:
            reason = "all_components_violate_train_floor"
        class_details.append(
            {
                "class_label": class_order.index(class_name),
                "class_name": class_name,
                "row_count": total_class_counts[class_name],
                "component_count": component_count,
                "source_group_count": len(class_groups[class_name]),
                "split_counts": {
                    split: class_split_counts[class_name][split] for split in SPLITS
                },
                "individually_cedable_component_count": len(eligible_components),
                "individually_cedable_component_ids": eligible_components,
                "holdout_eligible": bool(eligible_components),
                "holdout_ineligible_reason": reason,
                "independent_components_needed_for_one_holdout": max(0, 2 - component_count),
                "independent_components_needed_for_dev_and_test": max(0, 3 - component_count),
            }
        )

    ineligible = [item for item in class_details if not item["holdout_eligible"]]
    locked_class_names = {
        row["class_name"] for row in rows if row["dataset_split"] == "locked_test"
    }
    collection_priorities = sorted(
        [item for item in class_details if item["class_name"] not in locked_class_names],
        key=lambda item: (
            item["split_counts"]["train"],
            item["class_label"],
        ),
    )
    upper_class_deficit = max(0, min_eval_classes - len(upper_classes))
    upper_row_deficit = max(0, min_eval_rows - optimistic_upper_rows)
    collection_lower_bound = max(upper_class_deficit, upper_row_deficit)
    incremental_class_deficit = max(0, min_eval_classes - len(locked_class_names))
    incremental_row_deficit = max(
        0,
        min_eval_rows - current_metrics["locked_test"]["row_count"],
    )
    incremental_new_row_lower_bound = max(
        incremental_class_deficit,
        incremental_row_deficit,
    )
    both_eval_minimum = min_eval_rows * 2
    current_heldout_rows = (
        current_metrics["dev"]["row_count"]
        + current_metrics["locked_test"]["row_count"]
    )
    balanced_eval_new_row_lower_bound = max(0, both_eval_minimum - current_heldout_rows)
    heldout_fraction = sum(requested_fractions.values())
    if heldout_fraction:
        strict_fraction_new_row_lower_bound = max(
            0,
            math.ceil(
                (heldout_fraction * row_count - current_heldout_rows)
                / (1.0 - heldout_fraction)
            ),
        )
    else:
        strict_fraction_new_row_lower_bound = 0

    blocking_reasons: list[str] = []
    if len(upper_classes) < min_eval_classes:
        blocking_reasons.append("evaluable_class_upper_bound_below_gate")
    if optimistic_upper_rows < min_eval_rows:
        blocking_reasons.append("evaluable_row_upper_bound_below_gate")
    if current_metrics["locked_test"]["class_count"] < min_eval_classes:
        blocking_reasons.append("current_locked_test_class_count_below_gate")
    if current_metrics["locked_test"]["row_count"] < min_eval_rows:
        blocking_reasons.append("current_locked_test_row_count_below_gate")
    if current_metrics["locked_test"]["largest_class_fraction"] > 0.25:
        blocking_reasons.append("current_locked_test_class_dominance_above_25pct")
    if current_metrics["locked_test"]["largest_source_group_fraction"] > 0.25:
        blocking_reasons.append("current_locked_test_source_group_dominance_above_25pct")

    if len(upper_classes) < min_eval_classes or optimistic_upper_rows < min_eval_rows:
        feasibility = "impossible_with_current_corpus_under_constraints"
    elif greedy["class_count"] >= min_eval_classes and greedy["row_count"] >= min_eval_rows:
        feasibility = "constructively_feasible_by_deterministic_greedy"
    else:
        feasibility = "undetermined_between_bounds"

    result = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": {
            "path": manifest_path.resolve().as_posix() if manifest_path else None,
            "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
            "snapshot_id": manifest.get("snapshot_id"),
            "content_sha256": manifest.get("content_sha256"),
            "row_count": row_count,
            "class_count": len(class_order),
            "source_group_count": len(validated["rows_by_group"]),
            "component_count": len(components),
        },
        "constraints": {
            "unit": "duplicate_connected_source_component",
            "minimum_train_rows_per_class": min_train_rows_per_class,
            "promotion_minimum_eval_rows": min_eval_rows,
            "promotion_minimum_eval_classes": min_eval_classes,
            "source_group_leakage_allowed": False,
            "duplicate_component_leakage_allowed": False,
        },
        "current_split": {
            "counts": {split: current_counts[split] for split in SPLITS},
            "requested_rows": requested_rows,
            "metrics": current_metrics,
            "remaining_individually_movable_train_component_count": len(
                remaining_eligible_train_components
            ),
            "remaining_individually_movable_train_component_ids": sorted(
                remaining_eligible_train_components
            ),
            "target_deficits": target_deficits,
            "splitter_exhausted_before_requested_fractions": splitter_exhausted,
            "append_only_locked_test": {
                "parent_assignments_are_immutable": bool(
                    policy.get("parent_assignments_are_immutable", False)
                ),
                "existing_row_count": current_metrics["locked_test"]["row_count"],
                "existing_class_count": current_metrics["locked_test"]["class_count"],
                "movable_existing_train_component_count": len(
                    remaining_eligible_train_components
                ),
                "additional_eval_rows_required": incremental_row_deficit,
                "additional_eval_classes_required": incremental_class_deficit,
                "new_independent_rows_lower_bound": incremental_new_row_lower_bound,
                "new_class_component_incidences_lower_bound": incremental_class_deficit,
                "ideal_new_rows_for_both_eval_splits_at_gate": (
                    balanced_eval_new_row_lower_bound
                ),
                "ideal_new_rows_to_restore_requested_fractions": (
                    strict_fraction_new_row_lower_bound
                ),
                "lower_bound_is_sufficient": False,
            },
        },
        "feasibility": {
            "verdict": feasibility,
            "blocking_reasons": blocking_reasons,
            "individually_cedable_component_count": len(individually_cedable),
            "holdout_eligible_class_upper_bound": len(upper_classes),
            "holdout_eligible_class_names": upper_classes,
            "optimistic_eval_row_upper_bound": optimistic_upper_rows,
            "optimistic_eval_source_group_upper_bound": optimistic_upper_groups,
            "upper_bound_is_jointly_achievable": False,
            "greedy_feasible_lower_bound": greedy,
            "classes_with_at_least_two_components": sum(
                item["component_count"] >= 2 for item in class_details
            ),
            "classes_with_at_least_three_components": sum(
                item["component_count"] >= 3 for item in class_details
            ),
            "scenario": "fresh_repartition_of_current_corpus",
        },
        "collection": {
            "fresh_repartition_optimistic": {
                "additional_holdout_eligible_classes_needed": upper_class_deficit,
                "additional_eval_rows_needed": upper_row_deficit,
                "new_independent_rows_lower_bound": collection_lower_bound,
                "lower_bound_is_sufficient": False,
            },
            "append_only_locked_test": {
                "additional_eval_rows_required": incremental_row_deficit,
                "additional_eval_classes_required": incremental_class_deficit,
                "new_independent_rows_lower_bound": incremental_new_row_lower_bound,
                "new_class_component_incidences_lower_bound": incremental_class_deficit,
                "ideal_new_rows_for_both_eval_splits_at_gate": (
                    balanced_eval_new_row_lower_bound
                ),
                "ideal_new_rows_to_restore_requested_fractions": (
                    strict_fraction_new_row_lower_bound
                ),
                "lower_bound_is_sufficient": False,
                "operational_note": (
                    "Collect fresh source components disjoint from every persisted group. "
                    "The row minima assume every usable new row reaches the intended heldout "
                    "split; fixed hashing, balancing, group atomicity, exact-pixel "
                    "deduplication, and conflicts can require a larger candidate pool."
                ),
            },
            "priority_classes": collection_priorities,
        },
        "classes": class_details,
        "methodology": {
            "individual_upper_bound": (
                "Union of components that can each be removed while retaining the "
                "minimum train rows for every class they contain; simultaneous removal "
                "may violate that floor, so this is optimistic."
            ),
            "greedy_lower_bound": (
                "A deterministic feasible construction prioritizing new class coverage, "
                "then row count; it is not an optimizer and cannot prove impossibility."
            ),
            "gate_decision": (
                "Impossibility is claimed only when an optimistic upper bound misses a gate."
            ),
        },
    }
    result["audit_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def render_markdown(report: dict[str, Any], *, priority_limit: int = 30) -> str:
    snapshot = report["snapshot"]
    current = report["current_split"]
    feasibility = report["feasibility"]
    collection = report["collection"]
    append_only = collection["append_only_locked_test"]
    locked = current["metrics"]["locked_test"]
    dev = current["metrics"]["dev"]
    lines = [
        "# P5a — Audit de faisabilité du split",
        "",
        f"Snapshot : `{snapshot['snapshot_id']}` (`{snapshot['manifest_sha256']}`).",
        "",
        "## Verdict",
        "",
        f"**{feasibility['verdict']}**",
        "",
        (
            f"Le locked test actuel contient **{locked['row_count']} lignes / "
            f"{locked['class_count']} classes / {locked['source_group_count']} groupes**, "
            f"contre un gate de {report['constraints']['promotion_minimum_eval_rows']} lignes / "
            f"{report['constraints']['promotion_minimum_eval_classes']} classes."
        ),
        (
            f"La borne supérieure optimiste du corpus est de **"
            f"{feasibility['holdout_eligible_class_upper_bound']} classes / "
            f"{feasibility['optimistic_eval_row_upper_bound']} lignes**; elle n'est pas "
            "présentée comme simultanément réalisable et suppose une répartition depuis zéro."
        ),
        (
            "Avec les assignations existantes immuables, aucune composante train n'est "
            f"déplaçable : il faut au minimum **{append_only['additional_eval_rows_required']} "
            f"nouvelles lignes couvrant {append_only['additional_eval_classes_required']} "
            "classes supplémentaires** dans des sources indépendantes."
        ),
        "",
        "## Split actuel",
        "",
        "| Split | Lignes | Cible | Classes | Groupes | Plus grosse classe | Plus gros groupe |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| dev | {dev['row_count']} | {dev['target_row_count']} | "
            f"{dev['class_count']} | {dev['source_group_count']} | "
            f"{dev['largest_class_fraction']:.1%} | "
            f"{dev['largest_source_group_fraction']:.1%} |"
        ),
        (
            f"| locked_test | {locked['row_count']} | {locked['target_row_count']} | "
            f"{locked['class_count']} | {locked['source_group_count']} | "
            f"{locked['largest_class_fraction']:.1%} | "
            f"{locked['largest_source_group_fraction']:.1%} |"
        ),
        "",
        (
            "Épuisement du splitter avant les fractions demandées : **"
            f"{str(current['splitter_exhausted_before_requested_fractions']).lower()}** "
            f"({current['remaining_individually_movable_train_component_count']} composante "
            "train encore déplaçable individuellement)."
        ),
        "",
        "## Bornes de faisabilité",
        "",
        f"- Composantes individuellement cédables : {feasibility['individually_cedable_component_count']}",
        f"- Classes avec au moins deux composantes : {feasibility['classes_with_at_least_two_components']}",
        f"- Classes avec au moins trois composantes : {feasibility['classes_with_at_least_three_components']}",
        (
            "- Construction gloutonne réalisable : "
            f"{feasibility['greedy_feasible_lower_bound']['row_count']} lignes / "
            f"{feasibility['greedy_feasible_lower_bound']['class_count']} classes"
        ),
        "",
        "## Collecte minimale à planifier (contrat append-only)",
        "",
        (
            f"Borne inférieure : **{append_only['new_independent_rows_lower_bound']} "
            "nouvelles lignes indépendantes**, avec au moins "
            f"{append_only['new_class_component_incidences_lower_bound']} nouvelles incidences "
            "classe-composante. Cette borne est nécessaire, pas suffisante : dédoublonnage, "
            "conflits et atomicité des pages peuvent augmenter la collecte réelle."
        ),
        (
            "Si dev et locked_test doivent tous deux atteindre 600 lignes, la borne idéale "
            f"monte à {append_only['ideal_new_rows_for_both_eval_splits_at_gate']} lignes; "
            "restaurer strictement les fractions 10 % / 10 % demanderait au moins "
            f"{append_only['ideal_new_rows_to_restore_requested_fractions']} lignes si chaque "
            "nouvelle ligne était utilisable en heldout. Le hash réel ne garantit aucune de ces bornes."
        ),
        "",
        "Classes absentes du locked test, triées par support train croissant :",
        "",
        "| Classe | Lignes | Composantes | Groupes | Besoin pour 1 holdout |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in collection["priority_classes"][:priority_limit]:
        lines.append(
            f"| {item['class_name']} | {item['row_count']} | {item['component_count']} | "
            f"{item['source_group_count']} | "
            f"{item['independent_components_needed_for_one_holdout']} |"
        )
    lines.extend(
        [
            "",
            "## Décision P5b",
            "",
            "Ne pas réassigner le snapshot existant ni entraîner pour promotion. Le prochain lot doit "
            "rendre le splitter fail-closed lorsque les cibles sont irréalisables, déclarer les classes "
            "mono-composante `holdout_ineligible`, puis définir un plan de collecte par source indépendante.",
            "",
            f"Audit SHA-256 : `{report['audit_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_once(path: Path, content: bytes) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
        return "created"
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different report: {path}")
        return "verified_noop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--min-eval-rows", type=int, default=600)
    parser.add_argument("--min-eval-classes", type=int, default=100)
    parser.add_argument("--min-train-rows-per-class", type=int, default=1)
    parser.add_argument("--priority-limit", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.snapshot_manifest.expanduser().resolve()
        report = analyze_manifest(
            _load_manifest(manifest_path),
            manifest_path=manifest_path,
            min_eval_rows=args.min_eval_rows,
            min_eval_classes=args.min_eval_classes,
            min_train_rows_per_class=args.min_train_rows_per_class,
        )
        writes: dict[str, str] = {}
        if args.output_json:
            writes["json"] = _write_once(
                args.output_json,
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )
        if args.output_markdown:
            writes["markdown"] = _write_once(
                args.output_markdown,
                render_markdown(report, priority_limit=args.priority_limit).encode("utf-8"),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"split feasibility audit failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        summary = {
            "snapshot_id": report["snapshot"]["snapshot_id"],
            "verdict": report["feasibility"]["verdict"],
            "current_locked_test": report["current_split"]["metrics"]["locked_test"],
            "eligible_class_upper_bound": report["feasibility"][
                "holdout_eligible_class_upper_bound"
            ],
            "optimistic_eval_row_upper_bound": report["feasibility"][
                "optimistic_eval_row_upper_bound"
            ],
            "writes": writes,
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

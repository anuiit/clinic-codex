#!/usr/bin/env python3
"""Materialize a clean-room Tlachia simulation as training-snapshot.v2.

Only rows retained by the simulation are copied. Observed conflicts and
business-test similarities remain recorded as quarantine evidence; none may
intersect the active rows. This adapter intentionally produces a research-only
snapshot when the declared minimum support is one image per runtime class.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.codex_pipeline.data.class_order import load_runtime_class_order  # noqa: E402
from backend.codex_pipeline.data.tlachia.core import (  # noqa: E402
    IntegrityError,
    read_json,
    verify_artifact,
)
from backend.codex_pipeline.data.tlachia.media import MEDIA_MANIFEST_SCHEMA  # noqa: E402
from backend.codex_pipeline.data.tlachia.simulation import (  # noqa: E402
    STANDALONE_SIMULATION_SCHEMA,
)
from scripts.build_training_snapshot import (  # noqa: E402
    _semantic_manifest,
    canonical_json_sha,
    materialize_training_snapshot,
    pixel_sha256,
    repo_path,
    sha256_file,
)


def build_manifest(
    *,
    simulation_path: Path,
    media_manifest_path: Path,
    runtime_config_path: Path,
) -> dict[str, Any]:
    simulation_path = simulation_path.resolve()
    media_manifest_path = media_manifest_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    simulation = read_json(simulation_path)
    media = read_json(media_manifest_path)
    verify_artifact(simulation, STANDALONE_SIMULATION_SCHEMA)
    verify_artifact(media, MEDIA_MANIFEST_SCHEMA)

    payload = simulation["payload"]
    gates = payload.get("gates", {})
    required_gates = (
        "full_286_class_coverage",
        "minimum_train_support",
        "zero_unresolved_label_conflicts",
        "zero_unresolved_business_test_overlap",
        "quarantine_accounting_complete",
        "source_groups_split_disjoint",
        "all_splits_nonempty",
        "research_snapshot_materialization_ready",
    )
    failed = [name for name in required_gates if gates.get(name) is not True]
    if failed:
        raise IntegrityError(f"standalone simulation is not materialization-ready: {failed}")
    if payload.get("media_manifest_payload_sha256") != media.get("payload_sha256"):
        raise IntegrityError("simulation binds a different media manifest")

    class_order = list(load_runtime_class_order(runtime_config_path))
    if payload.get("class_order") != class_order or len(class_order) != 286:
        raise IntegrityError("simulation class order differs from the runtime ABI")

    media_root = Path(media["payload"]["output_root"]).resolve()
    media_by_entry = {
        (row["archive_name"], row["entry_path"]): row
        for row in media["payload"]["files"]
    }
    quarantine_entries = {row["entry"] for row in payload.get("quarantine", [])}
    active_entries: set[str] = set()
    active_rows: list[dict[str, Any]] = []
    for raw in payload.get("rows", []):
        key = (raw["archive_name"], raw["entry_path"])
        media_row = media_by_entry.get(key)
        if media_row is None:
            raise IntegrityError(f"simulation row is absent from media manifest: {key}")
        entry = f"{key[0]}:{key[1]}"
        if entry in quarantine_entries:
            raise IntegrityError(f"quarantined media survived into active rows: {entry}")
        active_entries.add(entry)
        source = (media_root / media_row["output_path"]).resolve()
        if media_root != source and media_root not in source.parents:
            raise IntegrityError(f"media path escapes output root: {source}")
        if not source.is_file():
            raise IntegrityError(f"retained media is missing: {source}")
        if sha256_file(source) != media_row["byte_sha256"]:
            raise IntegrityError(f"retained media byte hash drifted: {entry}")
        if pixel_sha256(source) != media_row["image"]["pixel_sha256"]:
            raise IntegrityError(f"retained media pixel hash drifted: {entry}")
        if raw["class_label"] != class_order.index(raw["class_name"]):
            raise IntegrityError(f"runtime label drifted: {entry}")
        split = raw.get("proposed_split")
        if split not in {"train", "dev", "locked_test"}:
            raise IntegrityError(f"invalid persisted split for {entry}: {split!r}")
        active_rows.append(
            {
                "row_id": raw["row_id"],
                "source_kind": "tlachia_theme_reference",
                "source_id": raw["source_id"],
                "class_label": raw["class_label"],
                "class_name": raw["class_name"],
                "source_group": raw["source_group"],
                "source_path": repo_path(source),
                "source_sha256": media_row["byte_sha256"],
                "source_pixel_sha256": media_row["image"]["pixel_sha256"],
                "decision": "approved_tlachia_clean_room",
                "duplicate_of": None,
                "dataset_split": split,
                "archive_name": raw["archive_name"],
                "entry_path": raw["entry_path"],
                "mapping_ids": list(raw["mapping_ids"]),
            }
        )
    if active_entries & quarantine_entries:
        raise IntegrityError("active/quarantine entry sets overlap")

    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in active_rows:
        rows_by_group[row["source_group"]].append(row)
    assignments: dict[str, str] = {}
    components: list[dict[str, Any]] = []
    for group, rows in sorted(rows_by_group.items()):
        splits = {row["dataset_split"] for row in rows}
        if len(splits) != 1:
            raise IntegrityError(f"source group spans persisted splits: {group}")
        split = next(iter(splits))
        assignments[group] = split
        components.append(
            {
                "component_id": "group-" + canonical_json_sha([group]),
                "source_groups": [group],
                "dataset_split": split,
                "assignment_origin": "new",
                "row_ids": sorted(row["row_id"] for row in rows),
                "class_names": sorted({row["class_name"] for row in rows}),
            }
        )

    semantic = _semantic_manifest(
        class_order=class_order,
        runtime_config_path=runtime_config_path,
        source_manifests=[
            {
                "kind": STANDALONE_SIMULATION_SCHEMA,
                "path": repo_path(simulation_path),
                "sha256": sha256_file(simulation_path),
                "payload_sha256": simulation["payload_sha256"],
            },
            {
                "kind": MEDIA_MANIFEST_SCHEMA,
                "path": repo_path(media_manifest_path),
                "sha256": sha256_file(media_manifest_path),
                "payload_sha256": media["payload_sha256"],
            },
        ],
        active_rows=active_rows,
        duplicate_rows=[],
        conflicts=[],
        source_group_assignments=assignments,
        components=components,
        parent_snapshot_id=None,
        split_salt=payload["policy"]["split_salt"],
        dev_fraction=float(payload["policy"]["requested_dev_fraction"]),
        locked_test_fraction=float(payload["policy"]["requested_test_fraction"]),
        min_train_rows_per_class=int(payload["policy"]["minimum_train_rows_per_class"]),
        allow_underfilled_holdouts=True,
    )
    semantic["tlachia_clean_room"] = {
        "simulation_payload_sha256": simulation["payload_sha256"],
        "media_manifest_payload_sha256": media["payload_sha256"],
        "business_final": payload["business_final"],
        "quarantine_accounting": payload["quarantine_accounting"],
        "observed_counts": {
            key: payload["counts"][key]
            for key in (
                "exact_label_conflicts",
                "internal_perceptual_conflicts",
                "business_overlaps",
                "quarantined",
            )
        },
        "research_only": True,
        "research_only_reasons": [
            "minimum_train_support_relaxed_to_one",
            "observed_conflicts_quarantined",
            "business_near_duplicate_quarantined",
            "promotion_holdout_not_validated",
        ],
        "prototype_only_classes": payload.get("insufficient_train_class_names", []),
    }
    content_sha = canonical_json_sha(semantic)
    return {
        **semantic,
        "snapshot_id": "snapshot-tlachia-" + content_sha[:20],
        "content_sha256": content_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path, default=REPO_ROOT / "backend/codex_model/config.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "backend/training_corpus/snapshots"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(
            simulation_path=args.simulation,
            media_manifest_path=args.media_manifest,
            runtime_config_path=args.runtime_config,
        )
        destination = None if args.dry_run else materialize_training_snapshot(
            manifest, args.output_root
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "snapshot_id": manifest["snapshot_id"],
                "content_sha256": manifest["content_sha256"],
                "row_count": manifest["row_count"],
                "split_counts": manifest["split_counts"],
                "ready_for_training": manifest["ready_for_training"],
                "promotion_evaluation_ready": manifest["promotion_evaluation_ready"],
                "destination": str(destination) if destination else None,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit live and historical annotation stores without mutating either one.

The report replaces path/mtime-dependent review fingerprints with a proposed
content identity for comparison only.  It identifies exact cross-store
matches, duplicate source pages, label conflicts, stale legacy decisions, and
rows that require manual re-attestation.  Decisions are read through
AnnotationReviewStore (review-state.sqlite3; review-index.json only for legacy
stores) and are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402


SCHEMA_VERSION = "annotation-store-audit.v2"


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(f"RGB:{width}x{height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def normalize_bbox(value: object) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"bbox must contain four finite numbers: {value!r}")
    result: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"bbox must contain four finite numbers: {value!r}")
        result.append(int(item) if float(item).is_integer() else round(float(item), 6))
    return result


def normalize_class(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid class name: {value!r}")
    return unicodedata.normalize("NFC", value.strip())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def review_decisions(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read decisions through AnnotationReviewStore (SQLite is authoritative).

    review-index.json is read directly only for legacy stores without a
    review database. The identity names which store was read and its hash.
    """
    store = AnnotationReviewStore(path)
    if store.has_database():
        return store.export_review_manifest()["decisions"], {
            "review_store": "sqlite",
            "review_index_path": store.db_path.as_posix(),
            "review_index_sha256": store.review_manifest_sha256(),
        }
    if not store.manifest_path.is_file():
        raise FileNotFoundError(
            f"no review store (review-state.sqlite3 or review-index.json): {path}"
        )
    return store.export_review_manifest()["decisions"], {
        "review_store": "legacy_json",
        "review_index_path": store.manifest_path.as_posix(),
        "review_index_sha256": sha256_file(store.manifest_path),
    }


def _queue_by_key(store: AnnotationReviewStore) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    queue = store.list_queue()
    by_key: dict[str, dict[str, Any]] = {}
    for analysis in queue.get("analyses", []):
        for element in analysis.get("elements", []):
            by_key[str(element["key"])] = element
    return by_key, list(queue.get("diagnostics", []))


def scan_store(path: Path, *, role: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    decisions, review_identity = review_decisions(path)
    queue, diagnostics = _queue_by_key(AnnotationReviewStore(path))

    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for analysis_dir in sorted(
        item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")
    ):
        metadata_path = analysis_dir / "metadata.json"
        image_path = analysis_dir / "image.png"
        if not metadata_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"analysis is incomplete: {analysis_dir}")
        metadata = _load_json(metadata_path)
        analysis_id = metadata.get("analysis_id")
        if analysis_id != analysis_dir.name:
            raise ValueError(f"analysis id/path mismatch: {analysis_dir}")
        annotations = metadata.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"metadata annotations must be a list: {metadata_path}")
        image_pixels = pixel_sha256(image_path)
        inventory.extend(
            [
                {"path": f"{analysis_dir.name}/metadata.json", "sha256": sha256_file(metadata_path)},
                {"path": f"{analysis_dir.name}/image.png", "sha256": sha256_file(image_path)},
            ]
        )
        seen: set[int] = set()
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"annotations[{position}] is not an object: {metadata_path}")
            index = annotation.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError(f"invalid or duplicate annotation index: {metadata_path}:{index!r}")
            seen.add(index)
            crop_path = analysis_dir / "elements" / f"{index}.png"
            if not crop_path.is_file():
                raise FileNotFoundError(f"annotation crop is missing: {crop_path}")
            crop_pixels = pixel_sha256(crop_path)
            crop_bytes = sha256_file(crop_path)
            inventory.append(
                {"path": f"{analysis_dir.name}/elements/{index}.png", "sha256": crop_bytes}
            )
            class_name = normalize_class(annotation.get("class_name"))
            bbox = normalize_bbox(annotation.get("bbox"))
            key = f"{analysis_id}:{index}"
            decision = decisions.get(key) if isinstance(decisions.get(key), dict) else {}
            queue_row = queue.get(key, {})
            content_identity = {
                "source_image_pixel_sha256": image_pixels,
                "crop_pixel_sha256": crop_pixels,
                "bbox": bbox,
                "class_name": class_name,
            }
            rows.append(
                {
                    "store_role": role,
                    "analysis_id": analysis_id,
                    "index": index,
                    "key": key,
                    "class_name": class_name,
                    "bbox": bbox,
                    "bbox_requires_normalization_policy": any(
                        not isinstance(item, int) for item in bbox
                    ),
                    "source_group": f"image-pixels:{image_pixels}",
                    "source_image_pixel_sha256": image_pixels,
                    "crop_sha256": crop_bytes,
                    "crop_pixel_sha256": crop_pixels,
                    "content_fingerprint_v2": canonical_json_sha(content_identity),
                    "persisted_status": decision.get("status", "undecided"),
                    "legacy_fingerprint_v1": decision.get("source_fingerprint"),
                    "reviewed_at": decision.get("reviewed_at"),
                    "reviewed_by": decision.get("reviewed_by"),
                    "persisted_dataset_split_v1": decision.get("dataset_split"),
                    "current_status": queue_row.get("review_status", "missing_from_queue"),
                    "current_trainable": bool(queue_row.get("trainable", False)),
                    "current_fingerprint_v1": queue_row.get("source_fingerprint"),
                    "stale_decision": bool(queue_row.get("stale_decision", False)),
                }
            )

    inventory.sort(key=lambda item: item["path"])
    rows.sort(key=lambda item: (item["analysis_id"], item["index"]))
    annotation_keys = {row["key"] for row in rows}
    orphaned_decisions: list[dict[str, Any]] = []
    for key in sorted(str(item) for item in decisions):
        if key in annotation_keys:
            continue
        decision = decisions.get(key)
        orphaned_decisions.append(
            {
                "store_role": role,
                "key": key,
                "decision": decision,
            }
        )
    return {
        "role": role,
        "path": str(path),
        "review_store": review_identity["review_store"],
        "review_index_sha256": review_identity["review_index_sha256"],
        "source_files_sha256": canonical_json_sha(inventory),
        "analysis_count": len({row["analysis_id"] for row in rows}),
        "row_count": len(rows),
        "decision_counts": dict(sorted(Counter(row["persisted_status"] for row in rows).items())),
        "current_trainable_count": sum(row["current_trainable"] for row in rows),
        "stale_decision_count": sum(row["stale_decision"] for row in rows),
        "orphaned_decision_count": len(orphaned_decisions),
        "orphaned_decisions": orphaned_decisions,
        "diagnostics": diagnostics,
        "rows": rows,
    }


def build_audit(live_store: Path, historical_store: Path) -> dict[str, Any]:
    live = scan_store(live_store, role="live")
    historical = scan_store(historical_store, role="historical")
    all_rows = live["rows"] + historical["rows"]
    orphaned_decisions = sorted(
        live["orphaned_decisions"] + historical["orphaned_decisions"],
        key=lambda item: (item["store_role"], item["key"]),
    )

    by_element_shape: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        shape_key = (
            row["source_image_pixel_sha256"],
            row["crop_pixel_sha256"],
            tuple(row["bbox"]),
        )
        by_element_shape[shape_key].append(row)
        by_fingerprint[row["content_fingerprint_v2"]].append(row)

    conflicts: list[dict[str, Any]] = []
    conflict_fingerprints: set[str] = set()
    for shape_key, cluster in sorted(by_element_shape.items(), key=lambda item: str(item[0])):
        classes = sorted({row["class_name"] for row in cluster})
        statuses = sorted({row["persisted_status"] for row in cluster if row["persisted_status"] != "undecided"})
        if len(classes) > 1 or ({"approved", "rejected"} <= set(statuses)):
            for row in cluster:
                conflict_fingerprints.add(row["content_fingerprint_v2"])
            conflicts.append(
                {
                    "source_image_pixel_sha256": shape_key[0],
                    "crop_pixel_sha256": shape_key[1],
                    "bbox": list(shape_key[2]),
                    "class_names": classes,
                    "statuses": statuses,
                    "keys": sorted(row["key"] for row in cluster),
                    "resolution": "needs_manual_review",
                }
            )

    exact_matches: list[dict[str, Any]] = []
    for fingerprint, cluster in sorted(by_fingerprint.items()):
        roles = {row["store_role"] for row in cluster}
        if roles == {"live", "historical"}:
            exact_matches.append(
                {
                    "content_fingerprint_v2": fingerprint,
                    "live_keys": sorted(row["key"] for row in cluster if row["store_role"] == "live"),
                    "historical_keys": sorted(
                        row["key"] for row in cluster if row["store_role"] == "historical"
                    ),
                    "class_name": cluster[0]["class_name"],
                    "conflicted": fingerprint in conflict_fingerprints,
                }
            )

    live_fingerprints = {row["content_fingerprint_v2"] for row in live["rows"]}
    rebind_actions: list[dict[str, Any]] = []
    for row in historical["rows"]:
        if row["persisted_status"] not in {"approved", "rejected"}:
            action = "manual_review_unreviewed"
        elif row["content_fingerprint_v2"] in conflict_fingerprints:
            action = "manual_review_conflict"
        elif row["content_fingerprint_v2"] in live_fingerprints:
            action = "manual_reaffirmation_exact_content_match"
        else:
            action = "manual_review_historical_only"
        rebind_actions.append(
            {
                "historical_key": row["key"],
                "persisted_status": row["persisted_status"],
                "content_fingerprint_v2": row["content_fingerprint_v2"],
                "recommended_action": action,
            }
        )

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "live_store_is_operational_source": True,
            "historical_decisions_are_never_auto_rebound": True,
            "content_identity": "source_image_pixels+crop_pixels+bbox+class",
            "conflicts_require_manual_review": True,
        },
        "stores": [live, historical],
        "orphaned_decisions": orphaned_decisions,
        "exact_cross_store_matches": exact_matches,
        "conflicts": conflicts,
        "historical_rebind_actions": rebind_actions,
        "counts": {
            "exact_cross_store_match_count": len(exact_matches),
            "conflict_count": len(conflicts),
            "orphaned_decision_count": len(orphaned_decisions),
            "live_orphaned_decision_count": live["orphaned_decision_count"],
            "historical_orphaned_decision_count": historical["orphaned_decision_count"],
            "manual_reaffirmation_count": sum(
                action["recommended_action"] == "manual_reaffirmation_exact_content_match"
                for action in rebind_actions
            ),
            "manual_review_count": sum(
                action["recommended_action"] != "manual_reaffirmation_exact_content_match"
                for action in rebind_actions
            ),
        },
    }
    return {**semantic, "audit_sha256": canonical_json_sha(semantic)}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-store",
        type=Path,
        default=REPO_ROOT / "backend" / "annotations",
    )
    parser.add_argument(
        "--historical-store",
        type=Path,
        default=REPO_ROOT / "annotations" / "annotations",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        report = build_audit(args.live_store, args.historical_store)
        if args.output:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"audit output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema_version": report["schema_version"],
        "audit_sha256": report["audit_sha256"],
        "stores": [
            {
                "role": store["role"],
                "analysis_count": store["analysis_count"],
                "row_count": store["row_count"],
                "decision_counts": store["decision_counts"],
                "current_trainable_count": store["current_trainable_count"],
                "stale_decision_count": store["stale_decision_count"],
                "orphaned_decision_count": store["orphaned_decision_count"],
            }
            for store in report["stores"]
        ],
        **report["counts"],
        "output": str(args.output.resolve()) if args.output else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

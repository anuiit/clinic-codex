#!/usr/bin/env python3
"""Freeze newly business-reviewed annotation pages as an immutable test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402
from backend.codex_pipeline.data.class_order import (  # noqa: E402
    class_order_sha256,
    load_runtime_class_order,
)


SCHEMA_VERSION = "business-test-set.v1"
CHECKSUM_SCHEMA_VERSION = "business-test-set-checksums.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_payload(image: Image.Image) -> tuple[bytes, int, int]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    return rgb.tobytes(), width, height


def pixel_sha256_image(image: Image.Image) -> str:
    payload, width, height = _rgb_payload(image)
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}x{height}:".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        return pixel_sha256_image(image)


def _bmp_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="BMP")
    return buffer.getvalue()


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


def _analysis_ids(path: Path) -> set[str]:
    return {
        item.name
        for item in path.iterdir()
        if item.is_dir() and (item / "metadata.json").is_file()
    }


def _normalise_class(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid class name: {value!r}")
    return unicodedata.normalize("NFC", value.strip())


def _normalise_bbox(value: object, *, page_size: tuple[int, int]) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"bbox must contain four integers: {value!r}")
    x, y, width, height = value
    page_width, page_height = page_size
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > page_width
        or y + height > page_height
    ):
        raise ValueError(f"bbox is outside page bounds {page_size}: {value!r}")
    return list(value)


def _reference_training_pixels(manifest_path: Path) -> tuple[str, set[str]]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "training-snapshot.v2":
        raise ValueError("reference training manifest must use training-snapshot.v2")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference training manifest has no rows")
    pixels = {
        row.get("source_pixel_sha256")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("source_pixel_sha256"), str)
    }
    if len(pixels) != len(rows):
        raise ValueError("reference training rows lack unique source pixel hashes")
    return sha256_file(manifest_path), pixels


def plan_business_test_set(
    *,
    annotations_dir: Path,
    baseline_annotations_dir: Path,
    runtime_config: Path,
    reference_training_manifest: Path,
    prior_annotation_dirs: list[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    annotations_dir = annotations_dir.expanduser().resolve()
    baseline_annotations_dir = baseline_annotations_dir.expanduser().resolve()
    runtime_config = runtime_config.expanduser().resolve()
    reference_training_manifest = reference_training_manifest.expanduser().resolve()
    decisions, review_identity = review_decisions(annotations_dir)

    current_ids = _analysis_ids(annotations_dir)
    baseline_ids = _analysis_ids(baseline_annotations_dir)
    selected_ids = sorted(current_ids - baseline_ids)
    if not selected_ids:
        raise ValueError("no new analysis pages found relative to baseline")

    class_order = [
        _normalise_class(name) for name in load_runtime_class_order(runtime_config)
    ]
    class_labels = {name: index for index, name in enumerate(class_order)}
    training_manifest_sha, training_pixels = _reference_training_pixels(
        reference_training_manifest
    )

    prior_page_pixels: set[str] = set()
    prior_sources = [baseline_annotations_dir, *(prior_annotation_dirs or [])]
    prior_inventories: list[dict[str, Any]] = []
    for source in prior_sources:
        source = source.expanduser().resolve()
        page_hashes: list[str] = []
        if source.is_dir():
            for analysis_id in sorted(_analysis_ids(source)):
                page_path = source / analysis_id / "image.png"
                page_hash = pixel_sha256(page_path)
                prior_page_pixels.add(page_hash)
                page_hashes.append(page_hash)
        prior_inventories.append(
            {
                "path": source.as_posix(),
                "page_count": len(page_hashes),
                "page_pixel_hashes_sha256": canonical_json_sha(sorted(page_hashes)),
            }
        )

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    crop_payloads: dict[str, bytes] = {}
    page_hashes: dict[str, str] = {}
    crop_hash_to_key: dict[str, str] = {}
    page_hash_to_id: dict[str, str] = {}
    selected_keys: set[str] = set()

    for analysis_id in selected_ids:
        analysis_dir = annotations_dir / analysis_id
        metadata_path = analysis_dir / "metadata.json"
        page_path = analysis_dir / "image.png"
        metadata = _load_json(metadata_path)
        if metadata.get("analysis_id") != analysis_id:
            raise ValueError(f"analysis id/path mismatch: {analysis_dir}")
        annotations = metadata.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"metadata annotations must be a list: {metadata_path}")
        with Image.open(page_path) as raw_page:
            page = ImageOps.exif_transpose(raw_page).convert("RGB")
        page_pixel_sha = pixel_sha256_image(page)
        if page_pixel_sha in page_hash_to_id:
            raise ValueError(
                f"new test pages are exact duplicates: {page_hash_to_id[page_pixel_sha]}, "
                f"{analysis_id}"
            )
        if page_pixel_sha in prior_page_pixels:
            raise ValueError(f"new test page overlaps prior annotations: {analysis_id}")
        page_hash_to_id[page_pixel_sha] = analysis_id
        page_hashes[analysis_id] = page_pixel_sha

        seen_indices: set[int] = set()
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"annotation {position} must be an object: {metadata_path}")
            index = annotation.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index in seen_indices
            ):
                raise ValueError(f"invalid or duplicate annotation index: {analysis_id}:{index!r}")
            seen_indices.add(index)
            key = f"{analysis_id}:{index}"
            selected_keys.add(key)
            decision = decisions.get(key)
            if not isinstance(decision, dict):
                raise ValueError(f"business decision is missing: {key}")
            class_name = _normalise_class(annotation.get("class_name"))
            bbox = _normalise_bbox(annotation.get("bbox"), page_size=page.size)
            if any(
                decision.get(field) != expected
                for field, expected in (
                    ("analysis_id", analysis_id),
                    ("index", index),
                    ("class_name", class_name),
                    ("bbox", bbox),
                )
            ):
                raise ValueError(f"business decision does not match metadata: {key}")
            status = decision.get("status")
            if status not in {"approved", "rejected"}:
                raise ValueError(f"business decision is not final: {key}:{status!r}")

            x, y, width, height = bbox
            crop = page.crop((x, y, x + width, y + height))
            crop_pixel_sha = pixel_sha256_image(crop)
            existing_crop = analysis_dir / "elements" / f"{index}.png"
            crop_origin = "reconstructed_from_page_bbox"
            existing_crop_sha = None
            if existing_crop.is_file():
                with Image.open(existing_crop) as raw_existing:
                    existing = ImageOps.exif_transpose(raw_existing).convert("RGB")
                if existing.size != crop.size or existing.tobytes() != crop.tobytes():
                    raise ValueError(f"stored crop differs from page bbox: {key}")
                crop_origin = "stored_crop_verified_against_page_bbox"
                existing_crop_sha = sha256_file(existing_crop)

            base_record = {
                "analysis_id": analysis_id,
                "index": index,
                "class_name": class_name,
                "bbox": bbox,
                "review_status": status,
                "reviewed_at": decision.get("reviewed_at"),
                "legacy_source_fingerprint": decision.get("source_fingerprint"),
                "legacy_source_fingerprint_reused": False,
                "page_path": page_path.as_posix(),
                "page_sha256": sha256_file(page_path),
                "page_pixel_sha256": page_pixel_sha,
                "crop_origin": crop_origin,
                "stored_crop_sha256": existing_crop_sha,
                "crop_pixel_sha256": crop_pixel_sha,
                "source_group": f"business-page:{page_pixel_sha}",
            }
            if status == "rejected":
                excluded.append({**base_record, "exclusion_reason": "business_rejected"})
                continue
            if class_name not in class_labels:
                excluded.append(
                    {**base_record, "exclusion_reason": "absent_from_runtime_taxonomy"}
                )
                continue
            if crop_pixel_sha in training_pixels:
                raise ValueError(f"business test crop overlaps reference training data: {key}")
            if crop_pixel_sha in crop_hash_to_key:
                raise ValueError(
                    f"business test contains duplicate crops: {crop_hash_to_key[crop_pixel_sha]}, {key}"
                )
            crop_hash_to_key[crop_pixel_sha] = key
            row_id = "business-row-" + canonical_json_sha(
                {
                    "page_pixel_sha256": page_pixel_sha,
                    "bbox": bbox,
                    "class_name": class_name,
                    "crop_pixel_sha256": crop_pixel_sha,
                }
            )
            relative_output = (
                Path("Elements")
                / f"{class_labels[class_name] + 1:04d}-{class_name}"
                / f"{row_id}.bmp"
            ).as_posix()
            payload = _bmp_bytes(crop)
            crop_payloads[relative_output] = payload
            included.append(
                {
                    **base_record,
                    "row_id": row_id,
                    "class_label": class_labels[class_name],
                    "dataset_split": "locked_test",
                    "output_path": relative_output,
                    "output_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    orphaned_selected_decisions = sorted(
        key
        for key in decisions
        if key.split(":", 1)[0] in set(selected_ids) and key not in selected_keys
    )
    if orphaned_selected_decisions:
        raise ValueError(
            "review index contains orphaned decisions for selected pages: "
            + ", ".join(orphaned_selected_decisions[:5])
        )
    if not included:
        raise ValueError("business test set contains no evaluable approved rows")

    included.sort(key=lambda row: (row["analysis_id"], row["index"]))
    excluded.sort(key=lambda row: (row["analysis_id"], row["index"]))
    class_counts = Counter(row["class_name"] for row in included)
    page_counts = Counter(row["analysis_id"] for row in included)
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "authority": "business_reviewed_final_test",
        "source": {
            "annotations_dir": annotations_dir.as_posix(),
            "baseline_annotations_dir": baseline_annotations_dir.as_posix(),
            "review_store": review_identity["review_store"],
            "review_index_path": review_identity["review_index_path"],
            "review_index_sha256": review_identity["review_index_sha256"],
            "selected_analysis_ids": selected_ids,
            "selection_rule": "analysis_ids_present_in_annotations_dir_but_absent_from_baseline",
        },
        "runtime_config_path": runtime_config.as_posix(),
        "runtime_config_sha256": sha256_file(runtime_config),
        "class_order": class_order,
        "class_order_sha256": class_order_sha256(class_order),
        "reference_training_manifest_path": reference_training_manifest.as_posix(),
        "reference_training_manifest_sha256": training_manifest_sha,
        "prior_annotation_inventories": prior_inventories,
        "analysis_count": len(selected_ids),
        "row_count": len(included),
        "class_count": len(class_counts),
        "excluded_count": len(excluded),
        "class_counts": dict(sorted(class_counts.items())),
        "analysis_row_counts": dict(sorted(page_counts.items())),
        "maximum_class_fraction": max(class_counts.values()) / len(included),
        "maximum_page_fraction": max(page_counts.values()) / len(included),
        "independence": {
            "exact_page_overlap_with_prior_annotations": 0,
            "exact_crop_overlap_with_reference_training": 0,
            "internal_exact_page_duplicates": 0,
            "internal_exact_crop_duplicates": 0,
        },
        "rows": included,
        "exclusions": excluded,
    }
    content_sha = canonical_json_sha(semantic)
    return (
        {
            **semantic,
            "test_set_id": "business-final-" + content_sha[:20],
            "content_sha256": content_sha,
        },
        crop_payloads,
    )


def materialize_business_test_set(
    manifest: dict[str, Any], crop_payloads: dict[str, bytes], output_root: Path
) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / manifest["test_set_id"]
    if destination.exists():
        raise FileExistsError(f"business test set already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=".business-test-staging-", dir=output_root) as temp:
        staging = Path(temp)
        for relative, payload in sorted(crop_payloads.items()):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        manifest_path = staging / "test_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_path = staging / "metadata.csv"
        fields = [
            "image_path",
            "element_id",
            "element_name",
            "class_label",
            "row_id",
            "source_group",
            "dataset_split",
            "output_sha256",
        ]
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in manifest["rows"]:
                writer.writerow(
                    {
                        "image_path": str((staging / row["output_path"]).resolve()),
                        "element_id": row["class_label"] + 1,
                        "element_name": row["class_name"],
                        "class_label": row["class_label"],
                        "row_id": row["row_id"],
                        "source_group": row["source_group"],
                        "dataset_split": row["dataset_split"],
                        "output_sha256": row["output_sha256"],
                    }
                )
        # Metadata paths must refer to the final immutable directory, not staging.
        text = metadata_path.read_text(encoding="utf-8").replace(
            staging.as_posix(), destination.as_posix()
        )
        metadata_path.write_text(text, encoding="utf-8")
        element_inventory = [
            {"path": row["output_path"], "sha256": row["output_sha256"]}
            for row in manifest["rows"]
        ]
        checksums = {
            "schema_version": CHECKSUM_SCHEMA_VERSION,
            "test_set_id": manifest["test_set_id"],
            "test_manifest_sha256": sha256_file(manifest_path),
            "metadata_csv_sha256": sha256_file(metadata_path),
            "element_count": len(element_inventory),
            "elements_sha256": canonical_json_sha(element_inventory),
        }
        (staging / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=REPO_ROOT / "annotations")
    parser.add_argument(
        "--baseline-annotations-dir",
        type=Path,
        default=REPO_ROOT / "annotations" / "annotations",
    )
    parser.add_argument(
        "--prior-annotations-dir", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=REPO_ROOT / "backend" / "codex_model" / "config.json",
    )
    parser.add_argument("--reference-training-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "backend" / "training_corpus" / "evaluation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest, payloads = plan_business_test_set(
            annotations_dir=args.annotations_dir,
            baseline_annotations_dir=args.baseline_annotations_dir,
            runtime_config=args.runtime_config,
            reference_training_manifest=args.reference_training_manifest,
            prior_annotation_dirs=args.prior_annotations_dir,
        )
        destination = None
        if not args.dry_run:
            destination = materialize_business_test_set(
                manifest, payloads, args.output_root
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = {
        "test_set_id": manifest["test_set_id"],
        "content_sha256": manifest["content_sha256"],
        "analysis_count": manifest["analysis_count"],
        "row_count": manifest["row_count"],
        "class_count": manifest["class_count"],
        "excluded_count": manifest["excluded_count"],
        "maximum_class_fraction": manifest["maximum_class_fraction"],
        "maximum_page_fraction": manifest["maximum_page_fraction"],
        "destination": str(destination) if destination else None,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize admin-approved annotations as an Elements/ classifier dataset."""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402
from backend.codex_pipeline.data.class_order import class_order_sha256, load_runtime_class_order  # noqa: E402


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def _folder_name(class_label: int, class_name: str) -> str:
    return f"{class_label + 1:04d}-{_slug(class_name)}"


def _dest_filename(row: dict[str, Any]) -> str:
    source = _slug(f"{row['analysis_id']}_{row['index']}")
    # The existing metadata builder accepts arbitrary instance ids after the
    # first dash; keep the codex/folio/page fields explicit but synthetic.
    return f"999_000_000-{source}.bmp"


def export_approved_annotations(
    annotations_dir: Path,
    output_dir: Path,
    *,
    clean: bool = True,
    runtime_config: Path | None = None,
) -> dict[str, Any]:
    store = AnnotationReviewStore(annotations_dir)
    queue = store.list_queue()
    rows = list(store.iter_approved_annotations())
    runtime_classes = load_runtime_class_order(runtime_config) if runtime_config else None
    eligible_keys = {(row["analysis_id"], row["index"]) for row in rows}
    excluded: list[dict[str, Any]] = []
    for analysis in queue["analyses"]:
        for element in analysis["elements"]:
            reason = None
            if runtime_classes is not None and element["class_name"] not in runtime_classes:
                reason = "unknown_class"
            elif (element["analysis_id"], element["index"]) not in eligible_keys:
                reason = element["split_reason"]
            if reason:
                excluded.append({
                    "analysis_id": element["analysis_id"],
                    "index": element["index"],
                    "class_name": element["class_name"],
                    "review_status": element["review_status"],
                    "reason": reason,
                })
    if runtime_classes is not None:
        rows = [row for row in rows if row["class_name"] in runtime_classes]
    class_names = sorted({row["class_name"] for row in rows})
    class_labels = {
        class_name: idx for idx, class_name in enumerate(runtime_classes or class_names)
    }

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    source_groups: dict[str, str] = {}
    for row in rows:
        class_name = row["class_name"]
        class_label = class_labels[class_name]
        class_dir = output_dir / _folder_name(class_label, class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / _dest_filename(row)

        with Image.open(row["crop_path"]) as image:
            image.convert("RGB").save(dest, format="BMP")

        image_path = row["image_path"]
        if image_path not in source_groups:
            with Image.open(image_path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
                digest.update(rgb.tobytes())
                source_groups[image_path] = f"image-pixels:{digest.hexdigest()}"
        exported.append(
            {
                "analysis_id": row["analysis_id"],
                "index": row["index"],
                "class_name": class_name,
                "class_label": class_label,
                "source_crop_path": row["crop_path"],
                "output_path": str(dest),
                "output_relative_path": dest.relative_to(output_dir).as_posix(),
                "source_group": source_groups[image_path],
                "source_fingerprint": row["source_fingerprint"],
                "dataset_split": row["dataset_split"],
            }
        )

    manifest = {
        "schema_version": 1,
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "exported_count": len(exported),
        "class_count": len(class_names),
        "classes": class_names,
        "rows": exported,
        "total_annotation_count": queue["counts"]["total"],
        "split_counts": dict(sorted(Counter(row["dataset_split"] for row in exported).items())),
        "excluded_count": len(excluded),
        "excluded_counts": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "excluded_rows": excluded,
        "diagnostics": queue["diagnostics"],
        "runtime_config": str(runtime_config) if runtime_config else None,
        "runtime_class_order_sha256": class_order_sha256(runtime_classes) if runtime_classes else None,

    }
    (output_dir / "_approved_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export only admin-approved annotation crops to an Elements/ dataset."
    )
    parser.add_argument(
        "--annotations-dir",
        default=str(REPO_ROOT / "backend" / "annotations"),
        help="Submitted annotations directory containing review-index.json.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "backend" / "training_data" / "approved" / "Elements"),
        help="Generated Elements output directory.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory before exporting.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit successfully even if no approved annotations are exported.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=REPO_ROOT / "backend" / "codex_model" / "config.json",
        help="Runtime taxonomy: excludes unknown classes and preserves integer labels.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    summary = export_approved_annotations(
        annotations_dir=Path(args.annotations_dir),
        output_dir=Path(args.output),
        clean=not args.keep_existing,
        runtime_config=args.runtime_config,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Exported {summary['exported_count']} approved annotations "
            f"across {summary['class_count']} classes to {summary['output_dir']}"
            f"; splits={summary['split_counts']}; exclusions={summary['excluded_counts']}"
        )

    if summary["exported_count"] == 0 and not args.allow_empty:
        print("ERROR: no admin-approved annotations were exported.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

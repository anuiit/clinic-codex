#!/usr/bin/env python3
"""Import a reviewed ZIP into a new isolated store; preview unless --apply.

Legacy path/mtime fingerprints cannot be verified after a ZIP transfer.
--re-attest-verified explicitly creates local decisions after checking labels,
boxes and source pixels. Archived decisions remain in the provenance report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.codex_pipeline.data.class_order import load_runtime_class_order
from backend.services.annotation_review import (
    AnnotationReviewStore, _dataset_split_for_key, _source_fingerprint,
    _validate_analysis_id, _validate_index, utc_now_iso,
)
from backend.services.annotation_storage import sanitize_class_name
from backend.services.image_limits import ensure_image_within_limits


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        ensure_image_within_limits(image)
        return ImageOps.exif_transpose(image).convert("RGB")


def _pixel_sha(image: Image.Image) -> str:
    digest = hashlib.sha256(f"RGB:{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def import_annotation_archive(
    archive: Path, output_dir: Path, runtime_config: Path, *,
    apply: bool = False, re_attest_verified: bool = False,
) -> dict[str, Any]:
    archive = archive.resolve()
    output_dir = output_dir.resolve()
    archive_sha = _sha256_file(archive)
    classes = load_runtime_class_order(runtime_config)
    report: dict[str, Any] = {
        "schema_version": "annotation-archive-import.v1",
        "archive_path": str(archive), "archive_sha256": archive_sha,
        "runtime_config_sha256": _sha256_file(runtime_config),
        "annotations_dir": str(output_dir), "applied": apply,
        "re_attest_verified": re_attest_verified,
        "legacy_fingerprint_verified": False,
        "split_policy": "preserve archive; missing split uses existing review-key 80/10/10 hash policy",
        "created_at": utc_now_iso(), "rows": [],
    }
    analyses = []
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        if len(infos) > 100_000 or sum(info.file_size for info in infos) > 1024**3:
            raise ValueError("annotation archive exceeds size limits")
        for info in infos:
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or ":" in info.filename or info.flag_bits & 1
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.file_size > 128 * 1024**2):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        members = set(names)
        manifest = json.loads(z.read("annotations/review-index.json"))
        decisions = manifest.get("decisions")
        if not isinstance(decisions, dict):
            raise ValueError("archive review decisions must be an object")
        seen_keys = set()
        for member in sorted(members):
            parts = PurePosixPath(member).parts
            if len(parts) != 3 or parts[0] != "annotations" or parts[2] != "metadata.json":
                continue
            metadata = json.loads(z.read(member))
            analysis_id = metadata.get("analysis_id")
            _validate_analysis_id(analysis_id)
            if analysis_id != parts[1] or not isinstance(metadata.get("annotations"), list):
                raise ValueError(f"invalid analysis metadata: {member}")
            folder = f"annotations/{analysis_id}"
            page_member = f"{folder}/image.png"
            page = _rgb(z.read(page_member))
            image_sha = _pixel_sha(page)
            local_annotations = []
            for annotation in metadata["annotations"]:
                index = annotation.get("index")
                _validate_index(index)
                class_name = annotation.get("class_name")
                if not isinstance(class_name, str) or sanitize_class_name(class_name) != class_name:
                    raise ValueError(f"invalid class name: {member}:{index}")
                bbox = annotation.get("bbox")
                if (not isinstance(bbox, list) or len(bbox) != 4
                        or any(type(value) not in {int, float} or not math.isfinite(value) for value in bbox)):
                    raise ValueError(f"invalid finite bbox: {member}:{index}")
                x, y, w, h = bbox
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > page.width or y+h > page.height:
                    raise ValueError(f"bbox outside source page: {member}:{index}")
                key = f"{analysis_id}:{index}"
                if key in seen_keys:
                    raise ValueError(f"duplicate annotation index: {key}")
                seen_keys.add(key)
                # Absolute source paths are provenance only; never follow them.
                basename = str(annotation.get("crop_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
                if not basename or basename in {".", ".."}:
                    basename = f"{index}.png"
                crop_member = f"{folder}/elements/{basename}"
                decision = decisions.get(key) or {}
                if not isinstance(decision, dict):
                    raise ValueError(f"invalid decision: {key}")
                reason = None
                crop_sha = crop_pixels = None
                if crop_member not in members:
                    reason = "missing_crop"
                else:
                    crop_data = z.read(crop_member)
                    crop = _rgb(crop_data)
                    crop_sha = hashlib.sha256(crop_data).hexdigest()
                    crop_pixels = _pixel_sha(crop)
                    if crop_pixels != _pixel_sha(page.crop((x, y, x+w, y+h))):
                        reason = "crop_source_mismatch"
                if not reason and decision and (
                    decision.get("analysis_id") != analysis_id or decision.get("index") != index
                    or decision.get("class_name") != class_name or decision.get("bbox") != bbox
                ):
                    reason = "decision_metadata_mismatch"
                status = decision.get("status", "pending")
                archived_split = decision.get("dataset_split")
                if not reason and status != "approved":
                    reason = "rejected_review" if status == "rejected" else "pending_review"
                if not reason and class_name not in classes:
                    reason = "unknown_class"
                if not reason and archived_split not in {None, "train", "val", "test", "excluded"}:
                    reason = "invalid_dataset_split"
                if not reason and archived_split == "excluded":
                    reason = "explicit_exclusion"
                if not reason and not re_attest_verified:
                    reason = "needs_reattestation"
                split = "excluded" if reason else archived_split or _dataset_split_for_key(key)
                row = {
                    "key": key, "analysis_id": analysis_id, "index": index,
                    "class_name": class_name, "bbox": bbox,
                    "archive_member": crop_member, "crop_sha256": crop_sha,
                    "crop_pixel_sha256": crop_pixels, "source_image_pixel_sha256": image_sha,
                    "archive_decision": decision, "dataset_split": split,
                    "split_source": "archive" if archived_split else "existing_review_key_hash",
                    "exclusion_reason": reason,
                }
                report["rows"].append(row)
                local_annotations.append({**annotation, "crop_path": f"elements/{index}.png"})
            analyses.append((folder, page_member, {**metadata, "annotations": local_annotations}))
        if not analyses:
            raise ValueError("archive contains no annotation analyses")
        report["orphan_decision_keys"] = sorted(set(decisions) - seen_keys)
        report["annotation_count"] = len(report["rows"])
        report["page_count"] = len(analyses)
        report["retained_count"] = sum(row["exclusion_reason"] is None for row in report["rows"])
        report["split_counts"] = dict(sorted(Counter(row["dataset_split"] for row in report["rows"]).items()))
        report["excluded_counts"] = dict(sorted(Counter(
            row["exclusion_reason"] for row in report["rows"] if row["exclusion_reason"]
        ).items()))
        report["fallback_split_count"] = sum(
            row["exclusion_reason"] is None and row["split_source"] == "existing_review_key_hash"
            for row in report["rows"]
        )
        if _sha256_file(archive) != archive_sha:
            raise ValueError("archive changed during validation")
        if apply:
            # New store only: leave the live store and previous imports untouched.
            output_dir.mkdir(parents=True, exist_ok=False)
            for folder, page_member, metadata in analyses:
                analysis_dir = output_dir / metadata["analysis_id"]
                (analysis_dir / "elements").mkdir(parents=True)
                (analysis_dir / "image.png").write_bytes(z.read(page_member))
                (analysis_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            local_decisions = {}
            for row in report["rows"]:
                crop_path = output_dir / row["analysis_id"] / "elements" / f"{row['index']}.png"
                if row["crop_sha256"]:
                    crop_path.write_bytes(z.read(row["archive_member"]))
                active_status = (
                    "approved" if row["exclusion_reason"] is None
                    or (re_attest_verified and row["exclusion_reason"] == "explicit_exclusion")
                    else "rejected" if re_attest_verified and row["exclusion_reason"] == "rejected_review"
                    else None
                )
                if active_status:
                    metadata = next(meta for _, _, meta in analyses if meta["analysis_id"] == row["analysis_id"])
                    annotation = next(a for a in metadata["annotations"] if a["index"] == row["index"])
                    local_decisions[row["key"]] = {
                        "analysis_id": row["analysis_id"], "index": row["index"], "status": active_status,
                        "class_name": row["class_name"], "bbox": row["bbox"],
                        "reviewed_at": report["created_at"], "reviewed_by": "verified-archive-import",
                        "source_fingerprint": _source_fingerprint(
                            analysis_id=row["analysis_id"], uploaded_at=metadata.get("uploaded_at"),
                            annotation=annotation, crop_path=crop_path,
                        ),
                        "dataset_split": row["dataset_split"],
                        "archive_sha256": archive_sha, "archive_decision": row["archive_decision"],
                    }
            # Publish decisions last; an incomplete import cannot be trainable.
            (output_dir / "import-provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            (output_dir / "review-index.json").write_text(json.dumps({"schema_version": 1, "decisions": local_decisions}, indent=2)+"\n", encoding="utf-8")
            actual = list(AnnotationReviewStore(output_dir).iter_approved_annotations())
            if len(actual) != report["retained_count"]:
                raise RuntimeError("import eligibility differs from validated plan")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=REPO_ROOT / "backend/codex_model/config.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--re-attest-verified", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.output or REPO_ROOT / "backend/training_data/imports" / _sha256_file(args.archive) / "annotations"
    report = import_annotation_archive(args.archive, output, args.runtime_config, apply=args.apply, re_attest_verified=args.re_attest_verified)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

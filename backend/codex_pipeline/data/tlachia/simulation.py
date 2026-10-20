from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .core import IntegrityError, PolicyError, artifact, read_json, semantic_hash, verify_artifact
from .media import MEDIA_MANIFEST_SCHEMA
from .p3 import MAPPING_SCHEMA, REVIEWABLE_STATUSES, TRAINABLE_MAPPING_STATUSES


SIMULATION_SCHEMA = "tlachia-snapshot-simulation.v1"
STANDALONE_SIMULATION_SCHEMA = "tlachia-standalone-snapshot-simulation.v1"
EXPECTED_BUSINESS_TEST_ID = "business-final-ea447440c1ae1d12e7b4"
EXPECTED_BUSINESS_CONTENT_SHA256 = (
    "ea447440c1ae1d12e7b47d2d8774911593b6328de7181a011823fa564f8cf652"
)


def _dhash64(path: Path) -> str:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        grayscale = rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(grayscale.get_flattened_data())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:016x}"


def _pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def _verified_media_identity(path: Path, file_row: dict[str, Any]) -> int:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != file_row["byte_sha256"]:
        raise IntegrityError(f"downloaded media byte hash mismatch: {path}")
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    if digest.hexdigest() != file_row["image"]["pixel_sha256"]:
        raise IntegrityError(f"downloaded media pixel hash mismatch: {path}")
    return int(_dhash64(path), 16)


def _hash_split(group_key: str, *, salt: str, dev: float, test: float) -> str:
    bucket = int(
        hashlib.sha256(f"{salt}:{group_key}".encode("utf-8")).hexdigest()[:16], 16
    ) / 2**64
    if bucket < test:
        return "locked_test"
    if bucket < test + dev:
        return "dev"
    return "train"


def _assign_splits_atomically(
    rows: list[dict[str, Any]],
    *,
    split_salt: str,
    dev_fraction: float,
    test_fraction: float,
    minimum_train_rows_per_class: int,
) -> list[str]:
    for row in rows:
        row["proposed_split"] = _hash_split(
            row["source_group"],
            salt=split_salt,
            dev=dev_fraction,
            test=test_fraction,
        )
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
        rows_by_group[row["source_group"]].append(row)
    for class_rows in by_class.values():
        train_count = sum(row["proposed_split"] == "train" for row in class_rows)
        for group in sorted({row["source_group"] for row in class_rows}):
            if train_count >= min(minimum_train_rows_per_class, len(class_rows)):
                break
            group_rows = rows_by_group[group]
            if group_rows[0]["proposed_split"] != "train":
                for group_row in group_rows:
                    group_row["proposed_split"] = "train"
                train_count += len(group_rows)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["source_group"]].add(row["proposed_split"])
    return sorted(group for group, splits in group_splits.items() if len(splits) != 1)


def _load_business_reference(
    business_manifest_path: Path,
    *,
    expected_test_id: str,
    expected_content_sha256: str,
) -> tuple[dict[str, Any], set[str], list[tuple[int, str, str]]]:
    if not business_manifest_path.is_file():
        raise PolicyError("the immutable business-final manifest is mandatory")
    business = read_json(business_manifest_path)
    if business.get("schema_version") not in {"business-final-test.v1", "business-test-set.v1"}:
        raise IntegrityError("business manifest schema mismatch")
    actual_test_id = business.get("test_id") or business.get("test_set_id")
    if actual_test_id != expected_test_id:
        raise IntegrityError("business-final test identity does not match the frozen authority")
    if business.get("content_sha256") != expected_content_sha256:
        raise IntegrityError("business-final content hash does not match the frozen authority")
    semantic_business = {
        key: value
        for key, value in business.items()
        if key not in {"test_id", "test_set_id", "content_sha256"}
    }
    recomputed_content_sha256 = hashlib.sha256(
        json.dumps(
            semantic_business,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if recomputed_content_sha256 != expected_content_sha256:
        raise IntegrityError("business-final semantic content hash is invalid")
    if not business.get("rows") or business.get("row_count", len(business["rows"])) != len(
        business["rows"]
    ):
        raise IntegrityError("business-final row count is empty or inconsistent")
    pixels = {
        str(value)
        for row in business.get("rows", [])
        for value in (row.get("crop_pixel_sha256"), row.get("page_pixel_sha256"))
        if value
    }
    hashes: list[tuple[int, str, str]] = []
    seen_paths: set[Path] = set()
    for row in business.get("rows", []):
        candidates: list[Path] = []
        if row.get("output_path"):
            candidates.append(business_manifest_path.parent / str(row["output_path"]))
        if row.get("page_path"):
            candidates.append(Path(str(row["page_path"])))
        if not candidates:
            raise IntegrityError("business-final row has no auditable image path")
        for candidate in candidates:
            path = candidate.resolve()
            if path in seen_paths:
                continue
            if not path.is_file():
                raise IntegrityError(f"business-final image is missing: {path}")
            is_crop = row.get("output_path") and path == (
                business_manifest_path.parent / str(row["output_path"])
            ).resolve()
            recorded_pixel_sha = row.get(
                "crop_pixel_sha256" if is_crop else "page_pixel_sha256"
            )
            if not recorded_pixel_sha or _pixel_sha256(path) != recorded_pixel_sha:
                raise IntegrityError(f"business-final pixel hash mismatch: {path}")
            recorded_byte_sha = row.get("output_sha256" if is_crop else "page_sha256")
            if recorded_byte_sha and hashlib.sha256(path.read_bytes()).hexdigest() != recorded_byte_sha:
                raise IntegrityError(f"business-final byte hash mismatch: {path}")
            seen_paths.add(path)
            hashes.append(
                (
                    int(_dhash64(path), 16),
                    str(row.get("row_id", row.get("analysis_id", path.name))),
                    str(row.get("class_name", "business-page")),
                )
            )
    binding = {
        "path": business_manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(business_manifest_path.read_bytes()).hexdigest(),
        "content_sha256": business.get("content_sha256"),
        "test_id": business.get("test_id") or business.get("test_set_id"),
    }
    return binding, pixels, hashes


def simulate_standalone_snapshot(
    *,
    mapping: dict[str, Any],
    media_manifest: dict[str, Any],
    media_manifest_path: Path,
    business_manifest_path: Path,
    split_salt: str = "clinic-codex-tlachia-standalone-v1",
    dev_fraction: float = 0.1,
    test_fraction: float = 0.1,
    near_duplicate_hamming_threshold: int = 4,
    minimum_train_rows_per_class: int = 2,
    expected_business_test_id: str = EXPECTED_BUSINESS_TEST_ID,
    expected_business_content_sha256: str = EXPECTED_BUSINESS_CONTENT_SHA256,
) -> dict[str, Any]:
    """Simulate a clean-room Tlachia corpus without consulting prior training rows."""
    if near_duplicate_hamming_threshold < 0 or near_duplicate_hamming_threshold > 64:
        raise PolicyError("near-duplicate Hamming threshold must be between 0 and 64")
    if minimum_train_rows_per_class < 1:
        raise PolicyError("minimum train support must be positive")
    if (
        not split_salt
        or not math.isfinite(dev_fraction)
        or not math.isfinite(test_fraction)
        or dev_fraction < 0
        or test_fraction < 0
        or dev_fraction + test_fraction >= 1
    ):
        raise PolicyError("invalid standalone split policy")
    verify_artifact(mapping, MAPPING_SCHEMA)
    verify_artifact(media_manifest, MEDIA_MANIFEST_SCHEMA)
    if media_manifest["payload"].get("purpose") != "training_source":
        raise PolicyError("standalone simulation requires a training_source media manifest")

    class_order = [row["runtime_class_name"] for row in mapping["payload"]["runtime_classes"]]
    if len(class_order) != 286 or len(set(class_order)) != 286:
        raise IntegrityError("standalone runtime taxonomy must contain 286 unique classes")
    for label, row in enumerate(mapping["payload"]["runtime_classes"]):
        if row.get("runtime_class_label") != label:
            raise IntegrityError("standalone runtime class labels do not preserve ABI order")
    mapping_by_id = {row["mapping_id"]: row for row in mapping["payload"]["mappings"]}
    business_binding, business_pixels, business_hashes = _load_business_reference(
        business_manifest_path,
        expected_test_id=expected_business_test_id,
        expected_content_sha256=expected_business_content_sha256,
    )
    media_by_entry = {
        (row["archive_name"], row["entry_path"]): row
        for row in media_manifest["payload"]["files"]
    }
    media_root_value = media_manifest["payload"].get("output_root")
    if not isinstance(media_root_value, str) or not media_root_value:
        raise IntegrityError("media manifest does not bind its output root")
    media_root = Path(media_root_value).resolve()

    candidates: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for file_row in media_manifest["payload"]["files"]:
        unknown_mapping_ids = sorted(set(file_row["mapping_ids"]) - set(mapping_by_id))
        if unknown_mapping_ids:
            raise IntegrityError(
                f"media manifest refers to unknown mapping IDs: {unknown_mapping_ids[:3]}"
            )
        rows = [mapping_by_id[mapping_id] for mapping_id in file_row["mapping_ids"]]
        statuses = sorted({row["status"] for row in rows})
        classes = sorted(
            {
                row["candidate_runtime_class_name"]
                for row in rows
                if row["status"] in TRAINABLE_MAPPING_STATUSES
                and row.get("candidate_runtime_class_name")
            }
        )
        if statuses != sorted(file_row.get("mapping_statuses", [])):
            raise IntegrityError("media manifest mapping statuses drifted from mapping")
        if classes != sorted(file_row.get("candidate_runtime_classes", [])):
            raise IntegrityError("media manifest runtime classes drifted from mapping")
        expected_group = (
            f"tlachia:archive:{file_row['archive_name']}:"
            f"theme-reference:{file_row['entry_path'].casefold()}"
        )
        if file_row.get("physical_source_group") != expected_group:
            raise IntegrityError("media manifest physical source group is non-canonical")
        entry = f"{file_row['archive_name']}:{file_row['entry_path']}"
        if any(status not in TRAINABLE_MAPPING_STATUSES for status in statuses) or len(classes) != 1:
            quarantine.append({"entry": entry, "reason": "mapping_not_uniquely_trainable", "runtime_classes": classes})
            continue
        class_name = classes[0]
        class_label = class_order.index(class_name)
        pixel_sha = file_row["image"]["pixel_sha256"]
        if pixel_sha in business_pixels:
            quarantine.append({"entry": entry, "reason": "business_final_pixel_overlap", "runtime_classes": classes})
            continue
        candidates.append(
            {
                "row_id": "tlachia-" + semantic_hash(
                    {"archive": file_row["archive_name"], "entry": file_row["entry_path"], "pixel": pixel_sha, "class": class_name}
                )[:24],
                "class_name": class_name,
                "class_label": class_label,
                "source_kind": "tlachia_theme_reference",
                "source_id": f"{file_row['archive_name']}:{file_row['entry_path']}",
                "source_path": f"tlachia://{file_row['archive_name']}/{file_row['entry_path']}",
                "source_group": file_row["physical_source_group"],
                "archive_name": file_row["archive_name"],
                "entry_path": file_row["entry_path"],
                "output_path": file_row["output_path"],
                "byte_sha256": file_row["byte_sha256"],
                "pixel_sha256": pixel_sha,
                "source_sha256": file_row["byte_sha256"],
                "source_pixel_sha256": pixel_sha,
                "mapping_ids": file_row["mapping_ids"],
            }
        )

    by_pixel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_pixel[row["pixel_sha256"]].append(row)
    deduplicated: list[dict[str, Any]] = []
    exact_duplicate_groups = 0
    exact_duplicate_rows = 0
    exact_conflicts = 0
    for pixel_sha, rows in sorted(by_pixel.items()):
        classes = sorted({row["class_name"] for row in rows})
        if len(classes) != 1:
            exact_conflicts += 1
            quarantine.extend(
                {"entry": f"{row['archive_name']}:{row['entry_path']}", "reason": "conflicting_internal_pixel_duplicate", "runtime_classes": classes}
                for row in rows
            )
            continue
        ordered = sorted(rows, key=lambda row: (row["archive_name"], row["entry_path"]))
        winner = dict(ordered[0])
        winner["duplicate_entries"] = [f"{row['archive_name']}:{row['entry_path']}" for row in ordered[1:]]
        if len(ordered) > 1:
            exact_duplicate_groups += 1
            exact_duplicate_rows += len(ordered) - 1
        deduplicated.append(winner)

    business_near_duplicates: list[dict[str, Any]] = []
    eligible: list[tuple[int, dict[str, Any]]] = []
    for row in deduplicated:
        media_row = media_by_entry[(row["archive_name"], row["entry_path"])]
        path = (media_root / media_row["output_path"]).resolve()
        if media_root != path and media_root not in path.parents:
            raise IntegrityError("downloaded media path escapes its bound output root")
        if not path.is_file():
            raise IntegrityError(f"downloaded media file is missing: {path}")
        value = _verified_media_identity(path, media_row)
        business_matches = [
            {"distance": (value ^ other).bit_count(), "row_id": row_id, "class_name": class_name}
            for other, row_id, class_name in business_hashes
            if (value ^ other).bit_count() <= near_duplicate_hamming_threshold
        ]
        if business_matches:
            business_near_duplicates.append({"row_id": row["row_id"], "entry": f"{row['archive_name']}:{row['entry_path']}", "matches": sorted(business_matches, key=lambda item: (item["distance"], item["row_id"]))[:10]})
            quarantine.append({"entry": f"{row['archive_name']}:{row['entry_path']}", "reason": "business_final_perceptual_near_duplicate", "runtime_classes": [row["class_name"]]})
            continue
        eligible.append((value, row))

    # Perceptual duplicates are atomic split components. This prevents visually
    # equivalent rows from leaking across train/dev/test. Any component carrying
    # more than one class is quarantined in full.
    parent = list(range(len(eligible)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(eligible)):
        left_hash = eligible[left][0]
        for right in range(left + 1, len(eligible)):
            if (left_hash ^ eligible[right][0]).bit_count() <= near_duplicate_hamming_threshold:
                union(left, right)

    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, (_value, row) in enumerate(eligible):
        components[find(index)].append(row)
    retained: list[dict[str, Any]] = []
    internal_near_conflicts: list[dict[str, Any]] = []
    perceptual_duplicate_components = 0
    for rows in components.values():
        rows = sorted(rows, key=lambda item: item["row_id"])
        classes = sorted({row["class_name"] for row in rows})
        if len(classes) > 1:
            conflict = {
                "component_id": "near-" + semantic_hash([row["row_id"] for row in rows])[:24],
                "row_ids": [row["row_id"] for row in rows],
                "runtime_classes": classes,
            }
            internal_near_conflicts.append(conflict)
            quarantine.extend(
                {
                    "entry": f"{row['archive_name']}:{row['entry_path']}",
                    "reason": "conflicting_internal_perceptual_component",
                    "runtime_classes": classes,
                }
                for row in rows
            )
            continue
        if len(rows) > 1:
            perceptual_duplicate_components += 1
            component_group = "tlachia:perceptual-component:" + semantic_hash(
                [row["row_id"] for row in rows]
            )[:24]
            for row in rows:
                row["source_group"] = component_group
        retained.extend(rows)

    # A class must always retain training support. Move whole deterministic
    # groups to train; this is policy-driven and independent of outcomes.
    split_group_conflicts = _assign_splits_atomically(
        retained,
        split_salt=split_salt,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
        minimum_train_rows_per_class=minimum_train_rows_per_class,
    )
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        by_class[row["class_name"]].append(row)

    split_counts = Counter(row["proposed_split"] for row in retained)
    train_counts = Counter(row["class_name"] for row in retained if row["proposed_split"] == "train")
    covered_classes = sorted(by_class)
    missing_classes = [name for name in class_order if name not in by_class]
    insufficient_train_classes = [
        name for name in covered_classes if train_counts[name] < minimum_train_rows_per_class
    ]
    observed_conflicts = bool(exact_conflicts or internal_near_conflicts)
    observed_business_overlap = bool(business_near_duplicates) or any(
        row["reason"] == "business_final_pixel_overlap" for row in quarantine
    )
    retained_entries = {
        f"{row['archive_name']}:{row['entry_path']}" for row in retained
    }
    quarantined_entries = {row["entry"] for row in quarantine}
    quarantine_retained_intersection = sorted(
        retained_entries & quarantined_entries
    )
    quarantine_accounting_complete = not quarantine_retained_intersection
    # Conflicts and business-test similarities are fail-closed above: every
    # affected physical member is removed before split assignment.  Keep the
    # observed evidence visible, while gating the research snapshot on what is
    # still unresolved in active rows rather than on already-neutralised rows.
    unresolved_conflicts = bool(quarantine_retained_intersection)
    unresolved_business_overlap = bool(quarantine_retained_intersection)
    materialization_ready = (
        not missing_classes
        and not insufficient_train_classes
        and not unresolved_conflicts
        and not unresolved_business_overlap
        and quarantine_accounting_complete
        and not split_group_conflicts
        and all(split_counts.get(split, 0) > 0 for split in ("train", "dev", "locked_test"))
        and bool(retained)
    )
    payload = {
        "mode": "standalone_clean_room_rebuild",
        "mapping_payload_sha256": mapping["payload_sha256"],
        "media_manifest_payload_sha256": media_manifest["payload_sha256"],
        "business_final": business_binding,
        "class_order": class_order,
        "class_order_sha256": mapping["payload"]["class_order_sha256"],
        "policy": {
            "prior_training_dataset_consulted": False,
            "reference_unit": "one archive-scoped theme reference",
            "exact_pixel_dedup": True,
            "business_exact_and_dhash_quarantine": True,
            "near_duplicate_hamming_threshold": near_duplicate_hamming_threshold,
            "split": "deterministic source-group hash with train coverage safeguard",
            "split_salt": split_salt,
            "requested_dev_fraction": str(dev_fraction),
            "requested_test_fraction": str(test_fraction),
            "minimum_train_rows_per_class": minimum_train_rows_per_class,
        },
        "counts": {
            "downloaded_files": len(media_manifest["payload"]["files"]),
            "retained_rows": len(retained),
            "runtime_classes": len(class_order),
            "covered_runtime_classes": len(covered_classes),
            "missing_runtime_classes": len(missing_classes),
            "insufficient_train_classes": len(insufficient_train_classes),
            "exact_duplicate_groups": exact_duplicate_groups,
            "exact_duplicate_rows_removed": exact_duplicate_rows,
            "exact_label_conflicts": exact_conflicts,
            "internal_perceptual_conflicts": len(internal_near_conflicts),
            "perceptual_duplicate_components": perceptual_duplicate_components,
            "business_overlaps": sum("business_final" in row["reason"] for row in quarantine),
            "quarantined": len(quarantine),
            "split_group_conflicts": len(split_group_conflicts),
        },
        "split_counts": {split: split_counts.get(split, 0) for split in ("train", "dev", "locked_test")},
        "class_support": {
            name: {"total": len(by_class[name]), "train": train_counts[name]}
            for name in covered_classes
        },
        "missing_class_names": missing_classes,
        "insufficient_train_class_names": insufficient_train_classes,
        "rows": sorted(retained, key=lambda row: (row["class_label"], row["source_group"], row["row_id"])),
        "business_near_duplicates": business_near_duplicates,
        "internal_perceptual_conflicts": internal_near_conflicts,
        "quarantine": sorted(quarantine, key=lambda row: (row["reason"], row["entry"])),
        "quarantine_accounting": {
            "observed_label_conflict_groups": exact_conflicts
            + len(internal_near_conflicts),
            "observed_business_overlaps": sum(
                "business_final" in row["reason"] for row in quarantine
            ),
            "quarantined_entries": len(quarantined_entries),
            "retained_entry_intersection": quarantine_retained_intersection,
            "complete": quarantine_accounting_complete,
        },
        "gates": {
            "full_286_class_coverage": not missing_classes,
            "minimum_train_support": not insufficient_train_classes,
            "observed_label_conflicts_absent": not observed_conflicts,
            "observed_business_test_overlap_absent": not observed_business_overlap,
            "zero_unresolved_label_conflicts": not unresolved_conflicts,
            "zero_unresolved_business_test_overlap": not unresolved_business_overlap,
            "quarantine_accounting_complete": quarantine_accounting_complete,
            # Backward-compatible names now describe the retained corpus.
            "zero_label_conflicts": not unresolved_conflicts,
            "zero_business_test_overlap": not unresolved_business_overlap,
            "source_groups_split_disjoint": not split_group_conflicts,
            "all_splits_nonempty": all(
                split_counts.get(split, 0) > 0 for split in ("train", "dev", "locked_test")
            ),
            "research_snapshot_materialization_ready": materialization_ready,
            "training_allowed": False,
            "training_blockers": sorted(
                reason
                for condition, reason in (
                    (bool(missing_classes), "runtime_class_coverage_incomplete"),
                    (bool(insufficient_train_classes), "minimum_train_support_incomplete"),
                    (unresolved_conflicts, "unresolved_pixel_or_perceptual_label_conflicts"),
                    (unresolved_business_overlap, "unresolved_business_final_overlap"),
                    (not quarantine_accounting_complete, "quarantine_accounting_incomplete"),
                    (bool(split_group_conflicts), "source_group_split_conflict"),
                    (
                        any(split_counts.get(split, 0) == 0 for split in ("train", "dev", "locked_test")),
                        "one_or_more_splits_empty",
                    ),
                    (True, "simulation_not_materialized_as_training_snapshot_v2"),
                )
                if condition
            ),
        },
    }
    return artifact(STANDALONE_SIMULATION_SCHEMA, payload)


def simulate_child_snapshot(
    *,
    parent_manifest_path: Path,
    mapping: dict[str, Any],
    media_manifest: dict[str, Any],
    media_manifest_path: Path,
    business_manifest_path: Path,
    near_duplicate_hamming_threshold: int = 4,
) -> dict[str, Any]:
    parent = read_json(parent_manifest_path)
    if parent.get("schema_version") != "training-snapshot.v2":
        raise IntegrityError("parent must be a training-snapshot.v2 manifest")
    if not parent.get("ready_for_training"):
        raise PolicyError("parent snapshot is not research-ready")
    if parent.get("row_count") != len(parent.get("rows", [])):
        raise IntegrityError("parent snapshot row count mismatch")
    verify_artifact(mapping, MAPPING_SCHEMA)
    verify_artifact(media_manifest, MEDIA_MANIFEST_SCHEMA)
    if media_manifest["payload"].get("purpose") != "training_source":
        raise PolicyError("snapshot simulation requires a training_source media manifest")
    if mapping["payload"].get("class_order_sha256") != parent.get("class_order_sha256"):
        raise IntegrityError("mapping taxonomy order does not match the parent snapshot ABI")
    parent_class_order = parent.get("class_order")
    mapping_class_order = [
        row["runtime_class_name"] for row in mapping["payload"]["runtime_classes"]
    ]
    if mapping_class_order != parent_class_order:
        raise IntegrityError("mapping runtime classes do not exactly match the parent order")
    for row in mapping["payload"]["mappings"]:
        target = row.get("candidate_runtime_class_name")
        label = row.get("runtime_class_label")
        if target is None:
            if label is not None:
                raise IntegrityError("unmapped Tlachia row carries a numeric runtime label")
        elif not isinstance(label, int) or parent_class_order[label] != target:
            raise IntegrityError("Tlachia mapping name/label violates the parent ABI")

    review_rows = [
        row
        for row in mapping["payload"]["mappings"]
        if row["status"] in REVIEWABLE_STATUSES
    ]
    mapping_by_id = {
        row["mapping_id"]: row for row in mapping["payload"]["mappings"]
    }
    parent_by_pixel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parent["rows"]:
        parent_by_pixel[row["source_pixel_sha256"]].append(row)

    business_pixels: set[str] = set()
    if business_manifest_path is None or not business_manifest_path.is_file():
        raise PolicyError("the immutable business-final manifest is mandatory")
    business = read_json(business_manifest_path)
    if business.get("schema_version") not in {
        "business-final-test.v1",
        "business-test-set.v1",
    }:
        raise IntegrityError("business manifest schema mismatch")
    business_pixels = {
        str(row["crop_pixel_sha256"])
        for row in business.get("rows", [])
        if row.get("crop_pixel_sha256")
    }
    business_pixels.update(
        str(row["page_pixel_sha256"])
        for row in business.get("rows", [])
        if row.get("page_pixel_sha256")
    )
    business_binding: dict[str, Any] = {
        "path": business_manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(business_manifest_path.read_bytes()).hexdigest(),
        "content_sha256": business.get("content_sha256"),
        "test_id": business.get("test_id") or business.get("test_set_id"),
    }

    new_candidates: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for file_row in media_manifest["payload"]["files"]:
        mapping_rows = [mapping_by_id[mapping_id] for mapping_id in file_row["mapping_ids"]]
        trainable_rows = [
            row for row in mapping_rows if row["status"] in TRAINABLE_MAPPING_STATUSES
        ]
        class_names = sorted(
            {
                row["candidate_runtime_class_name"]
                for row in trainable_rows
                if row["candidate_runtime_class_name"]
            }
        )
        if len(class_names) != 1:
            quarantined.append(
                {
                    "entry": f"{file_row['archive_name']}:{file_row['entry_path']}",
                    "reason": "mapping_not_uniquely_trainable",
                    "runtime_classes": class_names,
                }
            )
            continue
        class_name = class_names[0]
        class_label = next(
            row["runtime_class_label"]
            for row in trainable_rows
            if row["candidate_runtime_class_name"] == class_name
        )
        pixel_sha = file_row["image"]["pixel_sha256"]
        if pixel_sha in business_pixels:
            quarantined.append(
                {
                    "entry": f"{file_row['archive_name']}:{file_row['entry_path']}",
                    "reason": "business_final_pixel_overlap",
                    "runtime_classes": class_names,
                }
            )
            continue
        parent_matches = parent_by_pixel.get(pixel_sha, [])
        if parent_matches:
            parent_classes = sorted({row["class_name"] for row in parent_matches})
            quarantined.append(
                {
                    "entry": f"{file_row['archive_name']}:{file_row['entry_path']}",
                    "reason": (
                        "consistent_parent_duplicate"
                        if parent_classes == [class_name]
                        else "conflicting_parent_duplicate"
                    ),
                    "runtime_classes": sorted(set(parent_classes + [class_name])),
                }
            )
            continue
        new_candidates.append(
            {
                "candidate_id": "tlachia-row-"
                + semantic_hash(
                    {
                        "archive": file_row["archive_name"],
                        "entry": file_row["entry_path"],
                        "pixel_sha256": pixel_sha,
                        "class_name": class_name,
                    }
                )[:24],
                "source_kind": "tlachia_theme_reference",
                "source_group": file_row["physical_source_group"],
                "archive_name": file_row["archive_name"],
                "entry_path": file_row["entry_path"],
                "byte_sha256": file_row["byte_sha256"],
                "pixel_sha256": pixel_sha,
                "class_name": class_name,
                "class_label": class_label,
                "mapping_ids": file_row["mapping_ids"],
            }
        )

    # Collapse exact new pixels only after detecting class conflicts.
    by_pixel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in new_candidates:
        by_pixel[row["pixel_sha256"]].append(row)
    additions: list[dict[str, Any]] = []
    for pixel_sha, rows in sorted(by_pixel.items()):
        classes = sorted({row["class_name"] for row in rows})
        if len(classes) != 1:
            quarantined.append(
                {
                    "entry": ";".join(
                        sorted(f"{row['archive_name']}:{row['entry_path']}" for row in rows)
                    ),
                    "reason": "conflicting_new_pixel_duplicate",
                    "runtime_classes": classes,
                }
            )
            continue
        winner = sorted(rows, key=lambda row: (row["archive_name"], row["entry_path"]))[0]
        winner = dict(winner)
        winner["duplicate_entries"] = sorted(
            f"{row['archive_name']}:{row['entry_path']}" for row in rows[1:]
        )
        additions.append(winner)

    if near_duplicate_hamming_threshold < 0 or near_duplicate_hamming_threshold > 64:
        raise PolicyError("near-duplicate Hamming threshold must be between 0 and 64")
    parent_hashes: list[tuple[int, str, str]] = []
    for row in parent["rows"]:
        path = (parent_manifest_path.parent / row["output_path"]).resolve()
        if not path.is_file():
            raise IntegrityError(f"parent snapshot image is missing: {path}")
        parent_hashes.append((int(_dhash64(path), 16), row["row_id"], row["class_name"]))
    business_hashes: list[tuple[int, str, str]] = []
    seen_business_paths: set[Path] = set()
    for row in business.get("rows", []):
        candidates: list[Path] = []
        if row.get("output_path"):
            candidates.append(business_manifest_path.parent / str(row["output_path"]))
        if row.get("page_path"):
            candidates.append(Path(str(row["page_path"])))
        if not candidates:
            raise IntegrityError("business-final row has no auditable image path")
        for candidate in candidates:
            path = candidate.resolve()
            if path in seen_business_paths:
                continue
            if not path.is_file():
                raise IntegrityError(f"business-final image is missing: {path}")
            seen_business_paths.add(path)
            business_hashes.append(
                (
                    int(_dhash64(path), 16),
                    str(row.get("row_id", row.get("analysis_id", path.name))),
                    str(row.get("class_name", "business-page")),
                )
            )
    media_file_by_entry = {
        (row["archive_name"], row["entry_path"]): row
        for row in media_manifest["payload"]["files"]
    }
    near_duplicate_evidence: list[dict[str, Any]] = []
    retained_additions: list[dict[str, Any]] = []
    retained_hashes: list[tuple[int, dict[str, Any]]] = []
    for row in additions:
        media_row = media_file_by_entry[(row["archive_name"], row["entry_path"])]
        path = (media_manifest_path.parent / media_row["output_path"]).resolve()
        if not path.is_file():
            raise IntegrityError(f"downloaded media file is missing: {path}")
        value = int(_dhash64(path), 16)
        matches = [
            {
                "distance": (value ^ parent_value).bit_count(),
                "kind": "parent",
                "row_id": row_id,
                "class_name": class_name,
            }
            for parent_value, row_id, class_name in parent_hashes
            if (value ^ parent_value).bit_count() <= near_duplicate_hamming_threshold
        ]
        matches.extend(
            {
                "distance": (value ^ other_value).bit_count(),
                "kind": "new",
                "row_id": other["candidate_id"],
                "class_name": other["class_name"],
            }
            for other_value, other in retained_hashes
            if (value ^ other_value).bit_count() <= near_duplicate_hamming_threshold
        )
        matches.extend(
            {
                "distance": (value ^ business_value).bit_count(),
                "kind": "business_final",
                "row_id": row_id,
                "class_name": class_name,
            }
            for business_value, row_id, class_name in business_hashes
            if (value ^ business_value).bit_count() <= near_duplicate_hamming_threshold
        )
        matches.sort(key=lambda item: (item["distance"], item["kind"], item["row_id"]))
        if matches:
            near_duplicate_evidence.append(
                {
                    "candidate_id": row["candidate_id"],
                    "entry": f"{row['archive_name']}:{row['entry_path']}",
                    "class_name": row["class_name"],
                    "dhash64": f"{value:016x}",
                    "matches": matches[:10],
                }
            )
            has_business_match = any(match["kind"] == "business_final" for match in matches)
            quarantined.append(
                {
                    "entry": f"{row['archive_name']}:{row['entry_path']}",
                    "reason": (
                        "business_final_perceptual_near_duplicate"
                        if has_business_match
                        else "perceptual_near_duplicate"
                    ),
                    "runtime_classes": sorted(
                        {row["class_name"]} | {match["class_name"] for match in matches}
                    ),
                }
            )
            continue
        retained_additions.append(row)
        retained_hashes.append((value, row))
    additions = retained_additions

    salt = str(parent.get("policy", {}).get("split_salt", "clinic-codex-training-snapshot-v2"))
    dev_fraction = float(parent.get("policy", {}).get("requested_dev_fraction", 0.1))
    test_fraction = float(parent.get("policy", {}).get("requested_test_fraction", 0.1))
    # Fractions are converted to fixed decimal strings in the output; floats are not hashed.
    for row in additions:
        row["proposed_split"] = _hash_split(
            row["source_group"], salt=salt, dev=dev_fraction, test=test_fraction
        )

    parent_split_counts = Counter(parent["split_counts"])
    addition_split_counts = Counter(row["proposed_split"] for row in additions)
    projected_split_counts = {
        split: int(parent["split_counts"].get(split, 0)) + addition_split_counts.get(split, 0)
        for split in ("train", "dev", "locked_test")
    }
    projected_classes = set(parent["class_order"])
    added_classes = {row["class_name"] for row in additions}
    review_complete = not review_rows
    conflicts = [row for row in quarantined if "conflicting" in row["reason"]]
    business_overlaps = [
        row
        for row in quarantined
        if row["reason"]
        in {"business_final_pixel_overlap", "business_final_perceptual_near_duplicate"}
    ]
    research_ready = bool(additions) and review_complete and not conflicts and not business_overlaps
    payload = {
        "parent": {
            "path": parent_manifest_path.as_posix(),
            "snapshot_id": parent["snapshot_id"],
            "content_sha256": parent["content_sha256"],
            "manifest_sha256": hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest(),
            "row_count": parent["row_count"],
            "split_counts": parent["split_counts"],
        },
        "mapping_payload_sha256": mapping["payload_sha256"],
        "media_manifest_payload_sha256": media_manifest["payload_sha256"],
        "business_final": business_binding,
        "policy": {
            "parent_assignments_immutable": True,
            "new_group_assignment": "locked-source-group hash only; no existing assignment changes",
            "exact_pixel_dedup": True,
            "near_duplicate_audit": {
                "algorithm": "dhash64",
                "hamming_threshold": near_duplicate_hamming_threshold,
                "parent_rows_scanned": len(parent_hashes),
                "business_images_scanned": len(business_hashes),
                "matches_quarantined": len(near_duplicate_evidence),
            },
            "reference_unit": "one archive-scoped theme reference, never occurrence multiplication",
        },
        "counts": {
            "parent_rows": parent["row_count"],
            "candidate_files": len(media_manifest["payload"]["files"]),
            "net_additions": len(additions),
            "added_runtime_classes": len(added_classes),
            "projected_runtime_classes": len(projected_classes),
            "quarantined": len(quarantined),
            "conflicts": len(conflicts),
            "business_pixel_overlaps": len(business_overlaps),
            "pending_business_review_rows": len(review_rows),
            "near_duplicate_matches": len(near_duplicate_evidence),
        },
        "projected_split_counts": projected_split_counts,
        "addition_split_counts": dict(sorted(addition_split_counts.items())),
        "added_class_names": sorted(added_classes),
        "additions": additions,
        "near_duplicate_evidence": near_duplicate_evidence,
        "quarantine": sorted(quarantined, key=lambda row: (row["reason"], row["entry"])),
        "gates": {
            "parent_research_ready": True,
            "mapping_review_complete": review_complete,
            "nonempty_net_additions": bool(additions),
            "zero_label_conflicts": not conflicts,
            "zero_business_pixel_overlap": not business_overlaps,
            "near_duplicate_audit_complete": True,
            "research_snapshot_materialization_ready": research_ready,
            "training_allowed": False,
            "training_blockers": sorted(
                [
                    reason
                    for condition, reason in (
                        (not review_complete, "business_mapping_review_incomplete"),
                        (not additions, "no_net_tlachia_additions"),
                        (bool(conflicts), "pixel_label_conflicts"),
                        (bool(business_overlaps), "business_final_pixel_overlap"),
                        (True, "simulation_not_materialized_as_training_snapshot_v2"),
                    )
                    if condition
                ]
            ),
        },
    }
    return artifact(SIMULATION_SCHEMA, payload)

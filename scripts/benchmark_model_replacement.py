"""Leakage-resistant research benchmark for registered classifier packages.

This command is deliberately not a promotion gate.  It binds an immutable
``training-snapshot.v2`` manifest to a validated cached-feature provenance,
fits fair comparison prototypes from ``train`` only, and/or replays the
prototypes shipped in each package.  The evaluation split is always explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(1, str(BACKEND_ROOT))

from backend.codex_model.classifier import (  # noqa: E402
    ModelPackageValidationError,
    _validate_runtime_integrity,
)
from backend.codex_pipeline.data.class_order import (  # noqa: E402
    class_order_sha256,
    load_runtime_class_order,
)
from backend.codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402

BENCHMARK_SCHEMA_VERSION = "benchmark-model-replacement.v2"
EXPECTED_CLASS_COUNT = 286
TRAIN_SPLIT = "train"
ALLOWED_EVAL_SPLITS = {"dev", "locked_test"}
PROTOTYPE_MODES = {"train_recomputed", "artifact", "both"}
DEFAULT_BATCH_SIZE = 4096
MIN_PROMOTION_EVAL_EXAMPLES = 600
MIN_PROMOTION_EVAL_CLASSES = 100
# Predeclared structural safeguards.  They are intentionally simple and are
# not fitted or calibrated against any dev/locked-test result.
MIN_PROMOTION_EVAL_SOURCE_GROUPS = 20
MAX_PROMOTION_EVAL_CLASS_DOMINANCE = 0.25
MAX_PROMOTION_EVAL_SOURCE_GROUP_DOMINANCE = 0.25
EVAL_STRUCTURE_POLICY = "predeclared_conservative_v1_not_calibrated_on_eval"
PROTOTYPE_MODE_ROLES = {
    "artifact": "historical_behavior_or_packaging_fidelity",
    "train_recomputed": "comparable_train_only_generalization",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    version_root: Path
    manifest_path: Path
    checksums_path: Path
    config_path: Path
    weights_dir: Path
    projection_path: Path
    prototypes_path: Path
    class_names: list[str]
    num_classes: int
    hidden_dim: int
    embedding_dim: int
    rejection_threshold: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_package_root(path: Path) -> Path:
    path = path.resolve()
    if (
        (path / "config.json").is_file()
        and (path / "weights" / "projection.pt").is_file()
        and (path / "weights" / "prototypes.pt").is_file()
    ):
        return path
    runtime_dir = path / "runtime"
    if (
        (runtime_dir / "config.json").is_file()
        and (runtime_dir / "weights" / "projection.pt").is_file()
        and (runtime_dir / "weights" / "prototypes.pt").is_file()
    ):
        return runtime_dir.resolve()
    raise FileNotFoundError(
        "registered package must contain runtime/config.json and runtime/weights "
        f"(got {path})"
    )


def _load_model_package(path: Path) -> ModelPackage:
    root = _resolve_package_root(path)
    # Reuse the runtime's audited registry manifest/checksum validator.  It
    # rejects missing metadata, escapes, undeclared files, size drift and hash
    # drift before any tensor is deserialized.
    _validate_runtime_integrity(root)
    version_root = root.parent
    config_path = root / "config.json"
    weights_dir = root / "weights"
    config = _load_json_object(config_path)
    class_names = load_runtime_class_order(config_path)
    num_classes = int(config.get("num_classes", len(class_names)))
    if num_classes != EXPECTED_CLASS_COUNT or len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"registered package must preserve the 286-class ABI: {config_path}"
        )
    return ModelPackage(
        root=root,
        version_root=version_root,
        manifest_path=version_root / "manifest.json",
        checksums_path=version_root / "checksums.sha256",
        config_path=config_path,
        weights_dir=weights_dir,
        projection_path=weights_dir / "projection.pt",
        prototypes_path=weights_dir / "prototypes.pt",
        class_names=class_names,
        num_classes=num_classes,
        hidden_dim=int(config["hidden_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        rejection_threshold=float(config.get("rejection_threshold", 0.35)),
    )


def _normalise_split_name(value: Any) -> str:
    return str(value).strip().lower()


def _get_record_field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"snapshot row is missing required field(s): {', '.join(names)}")


def _extract_split_name(record: dict[str, Any]) -> str:
    return _normalise_split_name(
        _get_record_field(record, "dataset_split", "split", "subset", "partition")
    )


def _extract_label(record: dict[str, Any]) -> int:
    return int(_get_record_field(record, "class_label", "label", "element_label"))


def _extract_path_key(record: dict[str, Any]) -> str | None:
    for key in ("output_path", "image_path", "source_path", "path"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _normalise_path(value: str, *, base_dir: Path) -> str:
    path = Path(value.replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False).as_posix()


def _require_unique(values: list[str], *, label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _validate_candidate_provenance_binding(
    package: ModelPackage,
    *,
    snapshot_manifest: dict[str, Any],
    split_manifest_path: Path,
    features_path: Path,
    features_provenance_path: Path,
) -> dict[str, Any]:
    package_manifest = _load_json_object(package.manifest_path)
    data = package_manifest.get("data")
    training = package_manifest.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("candidate manifest is missing data/training provenance")

    snapshot = data.get("training_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("candidate manifest is missing data.training_snapshot")
    expected_snapshot = {
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "content_sha256": snapshot_manifest["content_sha256"],
        "manifest_sha256": sha256_file(split_manifest_path),
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"candidate training_snapshot {key} mismatch")

    def validate_file_reference(
        value: Any,
        *,
        label: str,
        expected_path: Path | None = None,
    ) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError(f"candidate manifest is missing {label} provenance")
        raw_path = value.get("path")
        expected_sha = value.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"candidate {label} path is invalid")
        if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"candidate {label} sha256 is invalid")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = package.version_root / path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"candidate {label} file is missing: {path}") from exc
        if not resolved.is_file():
            raise ValueError(f"candidate {label} is not a file: {resolved}")
        if expected_path is not None and resolved != expected_path.resolve(strict=True):
            raise ValueError(f"candidate {label} path does not match benchmark input")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise ValueError(f"candidate {label} checksum mismatch")
        return {"path": str(resolved), "sha256": actual_sha}

    features_ref = validate_file_reference(
        data.get("features_cache"),
        label="features_cache",
        expected_path=features_path,
    )
    features_provenance_ref = validate_file_reference(
        data.get("features_provenance"),
        label="features_provenance",
        expected_path=features_provenance_path,
    )
    config_ref = validate_file_reference(training.get("config"), label="training.config")
    checkpoint_ref = validate_file_reference(
        training.get("checkpoint"), label="training.checkpoint"
    )
    return {
        "validated": True,
        "snapshot": expected_snapshot,
        "features_cache": features_ref,
        "features_provenance": features_provenance_ref,
        "training_config": config_ref,
        "checkpoint": checkpoint_ref,
    }


def _validate_snapshot_and_cache_provenance(
    *,
    split_manifest_path: Path,
    records: list[dict[str, Any]],
    features_provenance_path: Path,
    features_payload: dict[str, Any],
    features_len: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    manifest = _load_json_object(split_manifest_path)
    if manifest.get("schema_version") != "training-snapshot.v2":
        raise ValueError("split manifest must use training-snapshot.v2")
    snapshot_id = manifest.get("snapshot_id")
    content_sha = manifest.get("content_sha256")
    if (
        not isinstance(content_sha, str)
        or _SHA256_RE.fullmatch(content_sha) is None
        or snapshot_id != f"snapshot-{content_sha[:20]}"
    ):
        raise ValueError("snapshot_id/content_sha256 binding is invalid")
    if manifest.get("row_count") != len(records) or len(records) != features_len:
        raise ValueError("snapshot/cache row counts differ")

    class_order = manifest.get("class_order")
    if (
        not isinstance(class_order, list)
        or len(class_order) != EXPECTED_CLASS_COUNT
        or any(not isinstance(value, str) or not value for value in class_order)
    ):
        raise ValueError("snapshot must carry the exact 286-class order")
    expected_class_hash = class_order_sha256(class_order)
    if manifest.get("class_order_sha256") != expected_class_hash:
        raise ValueError("snapshot class_order_sha256 mismatch")
    if manifest.get("class_count") != EXPECTED_CLASS_COUNT:
        raise ValueError("snapshot class_count must be 286")

    row_ids = [str(_get_record_field(record, "row_id")) for record in records]
    if any(not value.strip() for value in row_ids):
        raise ValueError("snapshot row_id values must be non-empty")
    _require_unique(row_ids, label="snapshot row_id values")
    snapshot_paths_raw = [_extract_path_key(record) for record in records]
    if any(value is None for value in snapshot_paths_raw):
        raise ValueError("snapshot rows must carry output/image paths")
    snapshot_paths = [
        _normalise_path(str(value), base_dir=split_manifest_path.parent)
        for value in snapshot_paths_raw
    ]
    _require_unique(snapshot_paths, label="snapshot paths")

    snapshot_source_groups: list[str] = []
    row_split_by_id: dict[str, str] = {}
    row_group_by_id: dict[str, str] = {}

    actual_split_counts: dict[str, int] = {}
    for record in records:
        split = _extract_split_name(record)
        actual_split_counts[split] = actual_split_counts.get(split, 0) + 1
        label = _extract_label(record)
        if label < 0 or label >= EXPECTED_CLASS_COUNT:
            raise ValueError(f"snapshot class_label out of range: {label}")
        class_name = record.get("class_name", record.get("element_name"))
        if class_name != class_order[label]:
            raise ValueError(f"snapshot class_name/class_label mismatch for label {label}")
        source_group = record.get("source_group")
        if not isinstance(source_group, str) or not source_group.strip():
            raise ValueError("snapshot source_group values must be non-empty")
        snapshot_source_groups.append(source_group)
        row_id = str(record["row_id"])
        row_split_by_id[row_id] = split
        row_group_by_id[row_id] = source_group
    if manifest.get("split_counts") != actual_split_counts:
        raise ValueError("snapshot split_counts do not match its rows")

    group_assignments = manifest.get("source_group_assignments")
    if group_assignments is not None:
        if not isinstance(group_assignments, dict):
            raise ValueError("snapshot source_group_assignments must be an object")
        for record, source_group in zip(records, snapshot_source_groups):
            if group_assignments.get(source_group) != _extract_split_name(record):
                raise ValueError(
                    f"snapshot source_group assignment mismatch: {source_group}"
                )
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("snapshot components must be a non-empty exhaustive partition")
    active_row_ids = set(row_split_by_id)
    active_source_groups = set(snapshot_source_groups)
    seen_component_rows: set[str] = set()
    group_to_component: dict[str, str] = {}
    row_to_component: dict[str, str] = {}
    seen_component_ids: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("snapshot component entries must be objects")
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("snapshot component_id must be non-empty")
        if component_id in seen_component_ids:
            raise ValueError(f"duplicate snapshot component_id: {component_id}")
        seen_component_ids.add(component_id)
        component_split = _normalise_split_name(component.get("dataset_split"))
        component_rows_raw = component.get("row_ids")
        component_groups_raw = component.get("source_groups")
        if not isinstance(component_rows_raw, list) or not component_rows_raw:
            raise ValueError(f"snapshot component rows must be non-empty: {component_id}")
        if not isinstance(component_groups_raw, list) or not component_groups_raw:
            raise ValueError(f"snapshot component source_groups must be non-empty: {component_id}")
        component_rows = [str(value) for value in component_rows_raw]
        component_groups = [str(value) for value in component_groups_raw]
        if len(set(component_rows)) != len(component_rows):
            raise ValueError(f"snapshot component repeats a row: {component_id}")
        if len(set(component_groups)) != len(component_groups):
            raise ValueError(f"snapshot component repeats a source_group: {component_id}")

        unknown_rows = sorted(set(component_rows) - active_row_ids)
        if unknown_rows:
            raise ValueError(
                f"snapshot component contains inactive row: {unknown_rows[0]}"
            )
        derived_groups = {row_group_by_id[row_id] for row_id in component_rows}
        if set(component_groups) != derived_groups:
            raise ValueError(
                f"snapshot component source_groups mismatch derived rows: {component_id}"
            )
        derived_splits = {row_split_by_id[row_id] for row_id in component_rows}
        if derived_splits != {component_split}:
            raise ValueError(f"snapshot component split mismatch: {component_id}")

        for row_id in component_rows:
            if row_id in seen_component_rows:
                raise ValueError(f"snapshot row appears in multiple components: {row_id}")
            seen_component_rows.add(row_id)
            row_to_component[row_id] = component_id
        for source_group in component_groups:
            if source_group in group_to_component:
                raise ValueError(
                    "snapshot source_group appears in multiple components: "
                    f"{source_group}"
                )
            group_to_component[source_group] = component_id

    missing_rows = sorted(active_row_ids - seen_component_rows)
    if missing_rows:
        raise ValueError(f"snapshot components omit active row: {missing_rows[0]}")
    missing_groups = sorted(active_source_groups - set(group_to_component))
    if missing_groups:
        raise ValueError(f"snapshot components omit active source_group: {missing_groups[0]}")
    extra_groups = sorted(set(group_to_component) - active_source_groups)
    if extra_groups:
        raise ValueError(f"snapshot components contain inactive source_group: {extra_groups[0]}")

    duplicates = manifest.get("duplicates")
    if not isinstance(duplicates, list):
        raise ValueError("snapshot duplicates must be a list")
    seen_duplicate_rows: set[str] = set()
    for duplicate in duplicates:
        if not isinstance(duplicate, dict):
            raise ValueError("snapshot duplicate entries must be objects")
        duplicate_row_id = str(duplicate.get("row_id"))
        duplicate_of = str(duplicate.get("duplicate_of"))
        source_group = duplicate.get("source_group")
        if duplicate_row_id in seen_duplicate_rows:
            raise ValueError(f"snapshot duplicate row repeated: {duplicate_row_id}")
        seen_duplicate_rows.add(duplicate_row_id)
        if duplicate_of not in row_to_component:
            raise ValueError(f"snapshot duplicate target is not active: {duplicate_of}")
        if not isinstance(source_group, str) or source_group not in group_to_component:
            raise ValueError(
                f"snapshot duplicate source_group is not active: {source_group}"
            )
        if row_to_component[duplicate_of] != group_to_component[source_group]:
            raise ValueError(
                "snapshot duplicate-connected rows cross components/splits: "
                f"{duplicate_row_id}"
            )

    checksums_path = split_manifest_path.parent / "checksums.json"
    checksums = _load_json_object(checksums_path)
    if checksums.get("schema_version") != "training-snapshot-checksums.v1":
        raise ValueError("snapshot checksums.json schema mismatch")
    if checksums.get("snapshot_id") != snapshot_id:
        raise ValueError("snapshot checksums bind a different snapshot_id")
    if checksums.get("snapshot_manifest_sha256") != sha256_file(split_manifest_path):
        raise ValueError("snapshot manifest checksum mismatch")
    if checksums.get("element_count") != len(records):
        raise ValueError("snapshot checksum element_count mismatch")

    provenance = _load_json_object(features_provenance_path)
    if provenance.get("schema_version") != "features-cache.v1":
        raise ValueError("features provenance must use features-cache.v1")
    cache_row_ids_raw = features_payload.get("row_ids")
    if cache_row_ids_raw is None:
        cache_row_ids: list[str] = []
    elif isinstance(cache_row_ids_raw, list) and len(cache_row_ids_raw) == features_len:
        cache_row_ids = [str(value) for value in cache_row_ids_raw]
        if any(not value.strip() for value in cache_row_ids):
            raise ValueError("cached row_id values must be non-empty")
        _require_unique(cache_row_ids, label="cached row_id values")
    else:
        raise ValueError("cached row_ids must match the cached row count")

    cache_paths_raw = features_payload.get("image_paths")
    if not isinstance(cache_paths_raw, list) or len(cache_paths_raw) != features_len:
        raise ValueError("cached image_paths must match the cached row count")
    cache_paths = [
        _normalise_path(str(value), base_dir=split_manifest_path.parent)
        for value in cache_paths_raw
    ]
    _require_unique(cache_paths, label="cached image paths")

    cache_splits_raw = features_payload.get("dataset_splits")
    if not isinstance(cache_splits_raw, list) or len(cache_splits_raw) != features_len:
        raise ValueError("cached dataset_splits must match the cached row count")
    cache_splits = [_normalise_split_name(value) for value in cache_splits_raw]
    invalid_splits = sorted(set(cache_splits) - {"train", "dev", "locked_test"})
    if invalid_splits:
        raise ValueError(f"cached dataset_splits contain invalid values: {invalid_splits}")
    cache_source_groups_raw = features_payload.get("source_groups")
    if not isinstance(cache_source_groups_raw, list) or len(cache_source_groups_raw) != features_len:
        raise ValueError("cached source_groups must match the cached row count")
    cache_source_groups = [str(value) for value in cache_source_groups_raw]
    if any(not value.strip() for value in cache_source_groups):
        raise ValueError("cached source_group values must be non-empty")

    if provenance.get("runtime_class_order_sha256") != expected_class_hash:
        raise ValueError("features provenance class-order hash mismatch")
    if provenance.get("dataset_split_sha256") != _canonical_json_sha(cache_splits):
        raise ValueError("features provenance dataset_split hash mismatch")
    expected_row_ids_hash = _canonical_json_sha(cache_row_ids) if cache_row_ids else None
    if provenance.get("row_ids_sha256") != expected_row_ids_hash:
        raise ValueError("features provenance row_ids hash mismatch")
    if provenance.get("image_count") != features_len:
        raise ValueError("features provenance image_count mismatch")
    if provenance.get("class_count") != EXPECTED_CLASS_COUNT:
        raise ValueError("features provenance class_count must be 286")
    cached_features = features_payload.get("features")
    if (
        not isinstance(cached_features, torch.Tensor)
        or provenance.get("hidden_dim") != cached_features.size(1)
        or features_payload.get("hidden_dim") != cached_features.size(1)
    ):
        raise ValueError("features provenance hidden_dim mismatch")
    if provenance.get("metadata_csv_sha256") != checksums.get("metadata_csv_sha256"):
        raise ValueError("features provenance metadata checksum mismatch")

    snapshot_provenance = provenance.get("snapshot")
    if not isinstance(snapshot_provenance, dict):
        raise ValueError("features provenance is not bound to a training snapshot")
    required_snapshot_provenance = {
        "snapshot_id": snapshot_id,
        "snapshot_content_sha256": content_sha,
        "snapshot_manifest_sha256": sha256_file(split_manifest_path),
        "snapshot_checksums_sha256": sha256_file(checksums_path),
        "snapshot_elements_sha256": checksums.get("elements_sha256"),
        "verified_element_count": len(records),
    }
    for key, expected in required_snapshot_provenance.items():
        if snapshot_provenance.get(key) != expected:
            raise ValueError(f"features provenance snapshot {key} mismatch")

    return (
        manifest,
        provenance,
        class_order,
        cache_row_ids,
        cache_paths,
        cache_source_groups,
    )


def _resolve_indices(
    records: list[dict[str, Any]],
    *,
    cache_row_ids: list[str],
    cache_paths: list[str],
    split_manifest_path: Path,
) -> tuple[list[int], dict[str, int]]:
    row_id_to_index = {value: index for index, value in enumerate(cache_row_ids)}
    path_to_index = {value: index for index, value in enumerate(cache_paths)}
    indices: list[int] = []
    modes = {"row_id": 0, "path": 0}
    for record in records:
        row_id_value = record.get("row_id")
        if row_id_value is not None and cache_row_ids:
            row_id = str(row_id_value)
            if row_id not in row_id_to_index:
                raise ValueError(f"snapshot row_id not found in features payload: {row_id}")
            index = row_id_to_index[row_id]
            modes["row_id"] += 1
        else:
            raw_path = _extract_path_key(record)
            if raw_path is None:
                raise ValueError("snapshot row needs row_id or path to match cached features")
            path = _normalise_path(raw_path, base_dir=split_manifest_path.parent)
            if path not in path_to_index:
                raise ValueError(f"snapshot path not found in features payload: {raw_path}")
            index = path_to_index[path]
            modes["path"] += 1
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("snapshot rows map to overlapping cached feature indices")
    return indices, modes


def _tensor_to_batches(tensor: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [tensor[start : start + batch_size] for start in range(0, tensor.size(0), batch_size)]


@torch.no_grad()
def _project_features(
    package: ModelPackage,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model = ProjectionHead(input_dim=package.hidden_dim, embedding_dim=package.embedding_dim).to(device)
    state_dict = torch.load(package.projection_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    outputs = [model(batch.to(device)).cpu() for batch in _tensor_to_batches(features, batch_size)]
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty((0, package.embedding_dim), dtype=features.dtype)


def _compute_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, *, class_count: int
) -> torch.Tensor:
    prototypes: list[torch.Tensor] = []
    missing: list[int] = []
    for class_label in range(class_count):
        mask = labels == class_label
        if not mask.any():
            missing.append(class_label)
            continue
        prototypes.append(F.normalize(embeddings[mask].mean(dim=0), p=2, dim=-1))
    if missing:
        raise ValueError(
            "train split is missing class(es) required for fair 286-way prototypes: "
            f"{missing[:10]}"
        )
    return torch.stack(prototypes, dim=0)


def _load_artifact_prototypes(
    package: ModelPackage,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(package.prototypes_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("prototypes"), torch.Tensor):
        raise ValueError(f"invalid prototypes artifact: {package.prototypes_path}")
    prototypes = payload["prototypes"]
    if (
        not prototypes.is_floating_point()
        or tuple(prototypes.shape) != (package.num_classes, package.embedding_dim)
        or not torch.isfinite(prototypes).all()
    ):
        raise ValueError(f"prototype tensor does not match package config: {package.prototypes_path}")
    norms = prototypes.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3):
        raise ValueError(f"artifact prototype rows are not L2-normalized: {package.prototypes_path}")
    labels_raw = payload.get("class_labels")
    if isinstance(labels_raw, torch.Tensor):
        labels_raw = labels_raw.tolist()
    if not isinstance(labels_raw, list) or len(labels_raw) != package.num_classes:
        raise ValueError(f"artifact prototype labels are unavailable: {package.prototypes_path}")
    labels: list[int] = []
    for value in labels_raw:
        if isinstance(value, bool):
            raise ValueError(f"artifact prototype labels are invalid: {package.prototypes_path}")
        try:
            label = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"artifact prototype labels are invalid: {package.prototypes_path}"
            ) from exc
        if str(label) != str(value) and not isinstance(value, int):
            raise ValueError(f"artifact prototype labels are ambiguous: {package.prototypes_path}")
        labels.append(label)
    if len(set(labels)) != package.num_classes:
        raise ValueError(f"artifact prototype labels are not bijective: {package.prototypes_path}")

    class_names_raw = payload.get("class_names")
    if not isinstance(class_names_raw, dict):
        raise ValueError(f"artifact prototype class_names are invalid: {package.prototypes_path}")
    class_names: dict[int, str] = {}
    for raw_label, name in class_names_raw.items():
        if isinstance(raw_label, bool) or not isinstance(name, str) or not name:
            raise ValueError(f"artifact prototype class_names are invalid: {package.prototypes_path}")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"artifact prototype class_names are invalid: {package.prototypes_path}"
            ) from exc
        if label in class_names:
            raise ValueError(f"artifact prototype class_names are ambiguous: {package.prototypes_path}")
        class_names[label] = name
    if set(labels) != set(class_names):
        raise ValueError(f"artifact labels/class_names disagree: {package.prototypes_path}")
    row_names = [class_names[label] for label in labels]
    if len(set(row_names)) != package.num_classes or set(row_names) != set(package.class_names):
        raise ValueError(f"artifact prototype taxonomy differs from config: {package.prototypes_path}")

    row_by_name = {name: row for row, name in enumerate(row_names)}
    reorder = torch.tensor(
        [row_by_name[name] for name in package.class_names], dtype=torch.long
    )
    reordered = prototypes.index_select(0, reorder)
    protocol = (
        "canonical_zero_based"
        if labels == list(range(package.num_classes)) and row_names == package.class_names
        else "legacy_bijective_name_mapping"
    )
    return reordered, {"label_protocol": protocol, "source_labels": labels}


def _metrics_from_prototypes(
    *,
    package: ModelPackage,
    eval_embeddings: torch.Tensor,
    eval_labels: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_mode: str,
) -> dict[str, Any]:
    similarities = torch.mm(eval_embeddings, prototypes.t())
    top3_values, top3_indices = similarities.topk(3, dim=1)
    top1_indices = top3_indices[:, 0]
    top1_values = top3_values[:, 0]
    top1_correct = top1_indices.eq(eval_labels)
    top3_correct = top3_indices.eq(eval_labels.unsqueeze(1)).any(dim=1)
    accepted = top1_values.ge(package.rejection_threshold)

    observed_labels = sorted(set(int(value) for value in eval_labels.tolist()))
    per_class_top1: list[float] = []
    per_class_top3: list[float] = []
    per_class_coverage: list[float] = []
    for class_label in observed_labels:
        mask = eval_labels == class_label
        per_class_top1.append(top1_correct[mask].float().mean().item())
        per_class_top3.append(top3_correct[mask].float().mean().item())
        per_class_coverage.append(accepted[mask].float().mean().item())

    return {
        "prototype_mode": prototype_mode,
        "role": PROTOTYPE_MODE_ROLES[prototype_mode],
        "available": True,
        "eval_examples": int(eval_labels.numel()),
        "eval_observed_class_count": len(observed_labels),
        "eval_observed_class_labels": observed_labels,
        "top1_micro": float(top1_correct.float().mean().item()),
        "top3_micro": float(top3_correct.float().mean().item()),
        "top1_macro": float(sum(per_class_top1) / len(per_class_top1)),
        "top3_macro": float(sum(per_class_top3) / len(per_class_top3)),
        "coverage": float(accepted.float().mean().item()),
        "coverage_macro": float(sum(per_class_coverage) / len(per_class_coverage)),
    }


def _package_report(package: ModelPackage) -> dict[str, Any]:
    return {
        "package_root": str(package.root),
        "registry_version_root": str(package.version_root),
        "registry_manifest_path": str(package.manifest_path),
        "registry_manifest_sha256": sha256_file(package.manifest_path),
        "registry_checksums_path": str(package.checksums_path),
        "registry_checksums_sha256": sha256_file(package.checksums_path),
        "config_path": str(package.config_path),
        "weights_dir": str(package.weights_dir),
        "num_classes": package.num_classes,
        "hidden_dim": package.hidden_dim,
        "embedding_dim": package.embedding_dim,
        "rejection_threshold": package.rejection_threshold,
    }


def _reference_provenance(package: ModelPackage) -> dict[str, Any]:
    """Describe reference provenance without rejecting legacy packages."""

    manifest = _load_json_object(package.manifest_path)
    data = manifest.get("data")
    training = manifest.get("training")
    data_present = isinstance(data, dict) and bool(data)
    training_present = isinstance(training, dict) and bool(training)
    missing_sections = [
        section
        for section, present in (
            ("data", data_present),
            ("training", training_present),
        )
        if not present
    ]
    return {
        "status": "documented" if not missing_sections else "unverified",
        "verification_basis": (
            "data_and_training_sections_present_after_package_integrity_validation;"
            "contents_not_independently_attested"
        ),
        "manifest_path": str(package.manifest_path),
        "manifest_sha256": sha256_file(package.manifest_path),
        "data": data if isinstance(data, dict) else None,
        "training": training if isinstance(training, dict) else None,
        "missing_or_empty_sections": missing_sections,
    }


def _evaluation_structure(
    *,
    eval_labels: list[int],
    eval_source_groups: list[str],
    class_order: list[str],
) -> dict[str, Any]:
    """Apply predeclared, non-calibrated structural promotion safeguards."""

    if len(eval_labels) != len(eval_source_groups) or not eval_labels:
        raise ValueError("evaluation structure inputs must be non-empty and aligned")
    class_counts = Counter(eval_labels)
    source_group_counts = Counter(eval_source_groups)
    sample_count = len(eval_labels)
    dominant_class_label, dominant_class_count = max(
        class_counts.items(), key=lambda item: (item[1], -item[0])
    )
    dominant_source_group, dominant_source_group_count = max(
        source_group_counts.items(), key=lambda item: (item[1], item[0])
    )
    class_dominance = dominant_class_count / sample_count
    source_group_dominance = dominant_source_group_count / sample_count
    violations: list[str] = []
    if len(source_group_counts) < MIN_PROMOTION_EVAL_SOURCE_GROUPS:
        violations.append("too_few_source_groups")
    if class_dominance > MAX_PROMOTION_EVAL_CLASS_DOMINANCE:
        violations.append("class_dominance_exceeds_limit")
    if source_group_dominance > MAX_PROMOTION_EVAL_SOURCE_GROUP_DOMINANCE:
        violations.append("source_group_dominance_exceeds_limit")
    return {
        "policy": EVAL_STRUCTURE_POLICY,
        "thresholds": {
            "minimum_source_group_count": MIN_PROMOTION_EVAL_SOURCE_GROUPS,
            "maximum_class_dominance": MAX_PROMOTION_EVAL_CLASS_DOMINANCE,
            "maximum_source_group_dominance": MAX_PROMOTION_EVAL_SOURCE_GROUP_DOMINANCE,
        },
        "sample_count": sample_count,
        "class_count": len(class_counts),
        "source_group_count": len(source_group_counts),
        "class_counts": {
            str(label): class_counts[label] for label in sorted(class_counts)
        },
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "dominant_class": {
            "label": dominant_class_label,
            "name": class_order[dominant_class_label],
            "count": dominant_class_count,
            "fraction": class_dominance,
        },
        "dominant_source_group": {
            "source_group": dominant_source_group,
            "count": dominant_source_group_count,
            "fraction": source_group_dominance,
        },
        "violations": violations,
        "valid_for_promotion": not violations,
    }


def _evaluate_package(
    package: ModelPackage,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: list[int],
    eval_indices: list[int],
    *,
    prototype_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    train_labels = labels[train_indices]
    train_embeddings = _project_features(
        package, features[train_indices], batch_size=batch_size, device=device
    )
    eval_embeddings = _project_features(
        package, features[eval_indices], batch_size=batch_size, device=device
    )
    modes = (
        ["train_recomputed", "artifact"]
        if prototype_mode == "both"
        else [prototype_mode]
    )
    mode_reports: dict[str, Any] = {}
    for mode in modes:
        if mode == "train_recomputed":
            prototypes = _compute_prototypes(
                train_embeddings, train_labels, class_count=package.num_classes
            )
            artifact_metadata: dict[str, Any] = {}
        else:
            try:
                prototypes, artifact_metadata = _load_artifact_prototypes(package)
            except (OSError, RuntimeError, ValueError) as exc:
                mode_reports[mode] = {
                    "prototype_mode": mode,
                    "role": PROTOTYPE_MODE_ROLES[mode],
                    "available": False,
                    "reason": str(exc),
                }
                continue
        mode_reports[mode] = {
            **_metrics_from_prototypes(
                package=package,
                eval_embeddings=eval_embeddings,
                eval_labels=labels[eval_indices],
                prototypes=prototypes,
                prototype_mode=mode,
            ),
            **artifact_metadata,
        }
    return {
        **_package_report(package),
        "train_examples": len(train_indices),
        "train_observed_class_count": len(set(int(value) for value in train_labels.tolist())),
        "prototype_modes": mode_reports,
    }


def _comparison(
    runtime: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_provenance_status: str,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for mode, candidate_metrics in candidate["prototype_modes"].items():
        runtime_metrics = runtime["prototype_modes"][mode]
        interpretation = {
            "role": PROTOTYPE_MODE_ROLES[mode],
            "generalization_evidence_eligible": mode == "train_recomputed",
        }
        if mode == "artifact":
            interpretation.update(
                {
                    "reference_provenance_status": reference_provenance_status,
                    "reference_artifact_advantage_is_generalization_evidence": False,
                    "interpretation_guard": (
                        "prohibited_unverified_reference_provenance"
                        if reference_provenance_status == "unverified"
                        else "artifact_mode_is_behavior_or_packaging_fidelity_only"
                    ),
                }
            )
        if not runtime_metrics.get("available") or not candidate_metrics.get("available"):
            comparisons[mode] = {
                **interpretation,
                "available": False,
                "runtime_reason": runtime_metrics.get("reason"),
                "candidate_reason": candidate_metrics.get("reason"),
            }
            continue
        comparisons[mode] = {**interpretation, "available": True}
        comparisons[mode].update(
            {
                f"{metric}_delta": candidate_metrics[metric] - runtime_metrics[metric]
                for metric in (
                "top1_micro",
                "top1_macro",
                "top3_micro",
                "top3_macro",
                "coverage",
                "coverage_macro",
            )
            }
        )
    return comparisons


def benchmark_model_replacement(
    *,
    features_path: Path,
    split_manifest_path: Path,
    runtime_dir: Path,
    candidate_dir: Path,
    eval_split: str,
    prototype_mode: str = "both",
    features_provenance_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
) -> dict[str, Any]:
    eval_split = _normalise_split_name(eval_split)
    if eval_split not in ALLOWED_EVAL_SPLITS:
        raise ValueError(
            f"--eval-split must be explicit and one of {sorted(ALLOWED_EVAL_SPLITS)}"
        )
    if prototype_mode not in PROTOTYPE_MODES:
        raise ValueError(f"unsupported prototype mode: {prototype_mode}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"benchmark evidence output already exists: {output_path}")

    features_path = features_path.resolve()
    split_manifest_path = split_manifest_path.resolve()
    features_provenance_path = (
        features_provenance_path.resolve()
        if features_provenance_path is not None
        else Path(str(features_path) + ".prov.json")
    )
    features_payload = torch.load(features_path, map_location="cpu", weights_only=True)
    if not isinstance(features_payload, dict):
        raise ValueError("cached features payload must be a mapping")
    features = features_payload.get("features")
    labels = features_payload.get("labels")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("cached features tensor must be 2D")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("cached labels tensor must be 1D")
    labels = labels.to(torch.long)
    if features.size(0) != labels.size(0):
        raise ValueError("cached features and labels must have the same row count")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= EXPECTED_CLASS_COUNT):
        raise ValueError("cached labels fall outside the 286-class ABI")

    manifest_payload = _load_json_object(split_manifest_path)
    records = manifest_payload.get("rows")
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError("training-snapshot.v2 manifest must contain object rows")
    (
        manifest,
        provenance,
        class_order,
        cache_row_ids,
        cache_paths,
        cache_source_groups,
    ) = (
        _validate_snapshot_and_cache_provenance(
            split_manifest_path=split_manifest_path,
            records=records,
            features_provenance_path=features_provenance_path,
            features_payload=features_payload,
            features_len=features.size(0),
        )
    )

    cache_splits = [_normalise_split_name(value) for value in features_payload["dataset_splits"]]
    all_indices, mapping_modes = _resolve_indices(
        records,
        cache_row_ids=cache_row_ids,
        cache_paths=cache_paths,
        split_manifest_path=split_manifest_path,
    )
    if set(all_indices) != set(range(features.size(0))):
        raise ValueError("snapshot rows do not map bijectively to cached features")
    for record, index in zip(records, all_indices):
        manifest_label = _extract_label(record)
        if manifest_label != int(labels[index].item()):
            raise ValueError(f"snapshot/cache label mismatch for row_id {record.get('row_id')}")
        manifest_split = _extract_split_name(record)
        if manifest_split != cache_splits[index]:
            raise ValueError(f"snapshot/cache split mismatch for row_id {record.get('row_id')}")
        manifest_group = str(record.get("source_group"))
        if manifest_group != cache_source_groups[index]:
            raise ValueError(f"snapshot/cache source_group mismatch for row_id {record.get('row_id')}")

    train_indices = [
        index for record, index in zip(records, all_indices) if _extract_split_name(record) == TRAIN_SPLIT
    ]
    eval_indices = [
        index for record, index in zip(records, all_indices) if _extract_split_name(record) == eval_split
    ]
    if not train_indices:
        raise ValueError("snapshot contains no train rows")
    if not eval_indices:
        raise ValueError(f"snapshot contains no rows for explicit evaluation split: {eval_split}")
    overlap = set(train_indices) & set(eval_indices)
    if overlap:
        raise ValueError("train and evaluation rows overlap in the feature cache")
    train_source_groups = {cache_source_groups[index] for index in train_indices}
    eval_source_groups = {cache_source_groups[index] for index in eval_indices}
    source_group_overlap = sorted(train_source_groups & eval_source_groups)
    if source_group_overlap:
        raise ValueError(
            "train and evaluation source_groups overlap: " + ", ".join(source_group_overlap[:10])
        )

    features_class_names = features_payload.get("class_names")
    if not isinstance(features_class_names, dict):
        raise ValueError("cached class_names must be a mapping")
    if [features_class_names.get(index) for index in range(EXPECTED_CLASS_COUNT)] != class_order:
        raise ValueError("cached class order differs from snapshot class order")

    runtime_package = _load_model_package(runtime_dir)
    candidate_package = _load_model_package(candidate_dir)
    if runtime_package.class_names != class_order or candidate_package.class_names != class_order:
        raise ValueError("registered package class order differs from snapshot class order")
    if runtime_package.hidden_dim != features.size(1) or candidate_package.hidden_dim != features.size(1):
        raise ValueError("package hidden_dim differs from cached feature dimension")
    candidate_binding = _validate_candidate_provenance_binding(
        candidate_package,
        snapshot_manifest=manifest,
        split_manifest_path=split_manifest_path,
        features_path=features_path,
        features_provenance_path=features_provenance_path,
    )
    reference_provenance = _reference_provenance(runtime_package)

    device_obj = torch.device(device)
    runtime_metrics = _evaluate_package(
        runtime_package,
        features,
        labels,
        train_indices,
        eval_indices,
        prototype_mode=prototype_mode,
        batch_size=batch_size,
        device=device_obj,
    )
    candidate_metrics = _evaluate_package(
        candidate_package,
        features,
        labels,
        train_indices,
        eval_indices,
        prototype_mode=prototype_mode,
        batch_size=batch_size,
        device=device_obj,
    )

    eval_observed_labels = sorted(set(int(labels[index].item()) for index in eval_indices))
    evaluation_structure = _evaluation_structure(
        eval_labels=[int(labels[index].item()) for index in eval_indices],
        eval_source_groups=[cache_source_groups[index] for index in eval_indices],
        class_order=class_order,
    )
    blocking_reasons: list[str] = []
    if eval_split != "locked_test":
        blocking_reasons.append("eval_split:not_locked_test")
    if len(eval_indices) < MIN_PROMOTION_EVAL_EXAMPLES:
        blocking_reasons.append("volume:insufficient_evaluation_examples")
    if len(eval_observed_labels) < MIN_PROMOTION_EVAL_CLASSES:
        blocking_reasons.append("classes:insufficient_evaluation_classes")
    if manifest.get("promotion_evaluation_ready") is not True:
        blocking_reasons.append("snapshot_readiness:not_promotion_evaluation_ready")
    blocking_reasons.extend(
        f"evaluation_structure:{violation}"
        for violation in evaluation_structure["violations"]
    )
    coverage_sufficient = not blocking_reasons
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "research_only": True,
        "promotion_decision": "blocked",
        "inputs": {
            "features_path": str(features_path),
            "features_sha256": sha256_file(features_path),
            "features_provenance_path": str(features_provenance_path),
            "features_provenance_sha256": sha256_file(features_provenance_path),
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_content_sha256": manifest["content_sha256"],
            "runtime_dir": str(runtime_package.root),
            "candidate_dir": str(candidate_package.root),
        },
        "contract": {
            "fit_split": TRAIN_SPLIT,
            "eval_split": eval_split,
            "prototype_mode": prototype_mode,
            "prototype_mode_roles": PROTOTYPE_MODE_ROLES,
            "locked_test_used_for_fit": False,
            "class_count": EXPECTED_CLASS_COUNT,
            "class_order_sha256": class_order_sha256(class_order),
            "cache_provenance_schema": provenance["schema_version"],
        },
        "mapping": {
            "strategy": "row_id_then_normalized_path",
            "row_id_matches": mapping_modes["row_id"],
            "path_fallback_matches": mapping_modes["path"],
            "bijective": True,
            "train_eval_disjoint": True,
            "train_source_group_count": len(train_source_groups),
            "eval_source_group_count": len(eval_source_groups),
            "train_eval_source_groups_disjoint": True,
        },
        "reference_provenance": reference_provenance,
        "candidate_provenance_binding": candidate_binding,
        "split_counts": manifest["split_counts"],
        "train_examples": len(train_indices),
        "eval_examples": len(eval_indices),
        "evaluation_coverage": {
            "observed_class_count": len(eval_observed_labels),
            "observed_class_labels": eval_observed_labels,
            "observed_class_names": [class_order[label] for label in eval_observed_labels],
            "minimum_examples_for_promotion": MIN_PROMOTION_EVAL_EXAMPLES,
            "minimum_classes_for_promotion": MIN_PROMOTION_EVAL_CLASSES,
            "snapshot_promotion_evaluation_ready": manifest.get("promotion_evaluation_ready"),
            "sufficient_for_promotion": coverage_sufficient,
            "blocking_reasons": blocking_reasons,
            "warning": (
                None
                if coverage_sufficient
                else "promotion blocked: " + ", ".join(blocking_reasons)
            ),
        },
        "evaluation_structure": evaluation_structure,
        "models": {"runtime": runtime_metrics, "candidate": candidate_metrics},
        "comparison": {
            "prototype_modes": _comparison(
                runtime_metrics,
                candidate_metrics,
                reference_provenance_status=reference_provenance["status"],
            )
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path, help="Cached features.pt tensor dump.")
    parser.add_argument(
        "--features-provenance",
        type=Path,
        default=None,
        help="features-cache.v1 provenance (defaults to <features>.prov.json).",
    )
    parser.add_argument(
        "--split-manifest", required=True, type=Path, help="training-snapshot.v2 manifest."
    )
    parser.add_argument("--runtime-dir", required=True, type=Path, help="Registered runtime version/runtime path.")
    parser.add_argument("--candidate-dir", required=True, type=Path, help="Registered candidate version/runtime path.")
    parser.add_argument(
        "--eval-split",
        required=True,
        choices=sorted(ALLOWED_EVAL_SPLITS),
        help="Explicit held-out split; use locked_test for the P4 research report.",
    )
    parser.add_argument(
        "--prototype-mode",
        default="both",
        choices=sorted(PROTOTYPE_MODES),
        help="Fair train-recomputed head, package-fidelity artifact head, or both.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = benchmark_model_replacement(
            features_path=args.features,
            features_provenance_path=args.features_provenance,
            split_manifest_path=args.split_manifest,
            runtime_dir=args.runtime_dir,
            candidate_dir=args.candidate_dir,
            eval_split=args.eval_split,
            prototype_mode=args.prototype_mode,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ModelPackageValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

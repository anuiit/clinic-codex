#!/usr/bin/env python3
"""Evaluate registered model packages on an immutable business test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.codex_pipeline.data.class_order import class_order_sha256  # noqa: E402
from scripts import benchmark_model_replacement_research as benchmark  # noqa: E402


SCHEMA_VERSION = "business-model-evaluation.v1"


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _validate_business_cache(
    *, manifest_path: Path, features_path: Path, provenance_path: Path
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, list[str]]:
    manifest_path = manifest_path.resolve()
    features_path = features_path.resolve()
    provenance_path = provenance_path.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "business-test-set.v1":
        raise ValueError("business manifest must use business-test-set.v1")
    semantic = {
        key: value
        for key, value in manifest.items()
        if key not in {"test_set_id", "content_sha256"}
    }
    content_sha = canonical_json_sha(semantic)
    if manifest.get("content_sha256") != content_sha:
        raise ValueError("business manifest content_sha256 mismatch")
    if manifest.get("test_set_id") != f"business-final-{content_sha[:20]}":
        raise ValueError("business manifest test_set_id/content binding mismatch")
    rows = manifest.get("rows")
    class_order = manifest.get("class_order")
    if not isinstance(rows, list) or manifest.get("row_count") != len(rows):
        raise ValueError("business manifest row_count mismatch")
    if (
        not isinstance(class_order, list)
        or len(class_order) != benchmark.EXPECTED_CLASS_COUNT
        or manifest.get("class_order_sha256") != class_order_sha256(class_order)
    ):
        raise ValueError("business manifest class-order contract mismatch")
    independence = manifest.get("independence")
    if not isinstance(independence, dict) or any(
        independence.get(key) != 0
        for key in (
            "exact_page_overlap_with_prior_annotations",
            "exact_crop_overlap_with_reference_training",
            "internal_exact_page_duplicates",
            "internal_exact_crop_duplicates",
        )
    ):
        raise ValueError("business test independence checks are not clean")

    checksums_path = manifest_path.parent / "checksums.json"
    checksums = _load_json(checksums_path)
    if checksums.get("schema_version") != "business-test-set-checksums.v1":
        raise ValueError("business checksums schema mismatch")
    if checksums.get("test_set_id") != manifest.get("test_set_id"):
        raise ValueError("business checksums bind another test set")
    if checksums.get("test_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("business manifest checksum mismatch")
    metadata_path = manifest_path.parent / "metadata.csv"
    if checksums.get("metadata_csv_sha256") != sha256_file(metadata_path):
        raise ValueError("business metadata checksum mismatch")
    inventory = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("business test rows must be objects")
        relative = row.get("output_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("business test output paths must be relative")
        output = (manifest_path.parent / relative).resolve()
        try:
            output.relative_to((manifest_path.parent / "Elements").resolve())
        except ValueError as exc:
            raise ValueError("business test output path escapes Elements") from exc
        if sha256_file(output) != row.get("output_sha256"):
            raise ValueError(f"business test element checksum mismatch: {relative}")
        inventory.append({"path": relative, "sha256": row["output_sha256"]})
    if checksums.get("element_count") != len(rows):
        raise ValueError("business test element_count mismatch")
    if checksums.get("elements_sha256") != canonical_json_sha(inventory):
        raise ValueError("business test aggregate element checksum mismatch")

    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("business feature cache must be a mapping")
    features = payload.get("features")
    labels = payload.get("labels")
    row_ids = payload.get("row_ids")
    image_paths = payload.get("image_paths")
    source_groups = payload.get("source_groups")
    splits = payload.get("dataset_splits")
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or features.size(0) != len(rows)
        or labels.size(0) != len(rows)
    ):
        raise ValueError("business feature cache tensor shape mismatch")
    expected_row_ids = [row["row_id"] for row in rows]
    expected_paths = [
        str((manifest_path.parent / row["output_path"]).resolve()) for row in rows
    ]
    expected_groups = [row["source_group"] for row in rows]
    expected_labels = [row["class_label"] for row in rows]
    if row_ids != expected_row_ids:
        raise ValueError("business feature row_ids differ from manifest")
    if [str(Path(value).resolve()) for value in image_paths] != expected_paths:
        raise ValueError("business feature paths differ from manifest")
    if source_groups != expected_groups:
        raise ValueError("business feature source_groups differ from manifest")
    if splits != ["locked_test"] * len(rows):
        raise ValueError("business feature split must be locked_test")
    if labels.tolist() != expected_labels:
        raise ValueError("business feature labels differ from manifest")
    sparse_names = payload.get("class_names")
    if not isinstance(sparse_names, dict) or any(
        sparse_names.get(label) != class_order[label] for label in set(expected_labels)
    ):
        raise ValueError("business feature class names differ from runtime ABI")

    provenance = _load_json(provenance_path)
    if provenance.get("schema_version") != "features-cache.v1":
        raise ValueError("business feature provenance schema mismatch")
    if provenance.get("runtime_class_subset_allowed") is not True:
        raise ValueError("business feature provenance lacks explicit class-subset mode")
    if provenance.get("snapshot") is not None:
        raise ValueError("business evaluation features must not bind a training snapshot")
    if provenance.get("metadata_csv_sha256") != sha256_file(metadata_path):
        raise ValueError("business feature provenance metadata mismatch")
    if provenance.get("runtime_class_order_sha256") != manifest.get(
        "class_order_sha256"
    ):
        raise ValueError("business feature provenance class-order mismatch")
    if provenance.get("row_ids_sha256") != canonical_json_sha(expected_row_ids):
        raise ValueError("business feature provenance row-id mismatch")
    if provenance.get("dataset_split_sha256") != canonical_json_sha(splits):
        raise ValueError("business feature provenance split mismatch")
    if provenance.get("image_count") != len(rows):
        raise ValueError("business feature provenance image_count mismatch")
    if provenance.get("hidden_dim") != features.size(1):
        raise ValueError("business feature provenance hidden_dim mismatch")
    return manifest, provenance, features, labels, expected_groups


def _load_training_cache(
    *, manifest_path: Path, features_path: Path, provenance_path: Path
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[int]]:
    manifest_path = manifest_path.resolve()
    features_path = features_path.resolve()
    provenance_path = provenance_path.resolve()
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("training features must be a mapping")
    features = payload.get("features")
    labels = payload.get("labels")
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("training cache tensors are unavailable")
    records = _load_json(manifest_path).get("rows")
    if not isinstance(records, list):
        raise ValueError("training snapshot rows are unavailable")
    manifest, _prov, _order, row_ids, paths, _groups = (
        benchmark._validate_snapshot_and_cache_provenance(
            split_manifest_path=manifest_path,
            records=records,
            features_provenance_path=provenance_path,
            features_payload=payload,
            features_len=features.size(0),
        )
    )
    indices, _modes = benchmark._resolve_indices(
        records,
        cache_row_ids=row_ids,
        cache_paths=paths,
        split_manifest_path=manifest_path,
    )
    train_indices = [
        index
        for index, record in zip(indices, records)
        if benchmark._extract_split_name(record) == "train"
    ]
    if not train_indices:
        raise ValueError("training snapshot has no train rows")
    return manifest, features, labels, train_indices


def _validate_candidate_training_binding(
    package: benchmark.ModelPackage,
    *,
    snapshot_manifest: dict[str, Any],
    snapshot_manifest_path: Path,
    supplied_features_path: Path,
    supplied_provenance_path: Path,
) -> dict[str, Any]:
    """Validate a candidate's own training artifacts by content.

    B0 and B1 intentionally carry version-local copies of the same frozen feature
    tensor.  Their provenance JSON differs only where the training recipe differs,
    so requiring identical absolute paths would reject a legitimate comparison.
    """

    package_manifest = _load_json(package.manifest_path)
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
        "manifest_sha256": sha256_file(snapshot_manifest_path.resolve()),
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"candidate training_snapshot {key} mismatch")

    def validate_reference(value: Any, *, label: str) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError(f"candidate manifest is missing {label} provenance")
        raw_path = value.get("path")
        expected_sha = value.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"candidate {label} path is invalid")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"candidate {label} sha256 is invalid")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = package.version_root / path
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"candidate {label} is not a file: {resolved}")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise ValueError(f"candidate {label} checksum mismatch")
        return {"path": str(resolved), "sha256": actual_sha}

    features_ref = validate_reference(data.get("features_cache"), label="features_cache")
    if features_ref["sha256"] != sha256_file(supplied_features_path.resolve()):
        raise ValueError("candidate feature content differs from evaluation training cache")
    provenance_ref = validate_reference(
        data.get("features_provenance"), label="features_provenance"
    )
    candidate_provenance = _load_json(Path(provenance_ref["path"]))
    supplied_provenance = _load_json(supplied_provenance_path.resolve())
    stable_provenance_fields = (
        "schema_version",
        "backbone",
        "class_count",
        "dataset_split_sha256",
        "hidden_dim",
        "image_count",
        "metadata_csv_sha256",
        "row_ids_sha256",
        "runtime_class_order_sha256",
        "snapshot",
        "source_group_sha256",
    )
    for key in stable_provenance_fields:
        if candidate_provenance.get(key) != supplied_provenance.get(key):
            raise ValueError(f"candidate feature provenance {key} mismatch")
    config_ref = validate_reference(training.get("config"), label="training.config")
    checkpoint_ref = validate_reference(
        training.get("checkpoint"), label="training.checkpoint"
    )
    return {
        "validated": True,
        "binding_mode": "content_equivalent_version_local_cache",
        "snapshot": expected_snapshot,
        "features_cache": features_ref,
        "features_provenance": provenance_ref,
        "training_config": config_ref,
        "checkpoint": checkpoint_ref,
    }


def _mode_metrics(
    *,
    package: benchmark.ModelPackage,
    eval_embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    class_order: list[str],
    mode: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    similarities = torch.mm(eval_embeddings, prototypes.t())
    top3_values, top3_indices = similarities.topk(3, dim=1)
    top1 = top3_indices[:, 0]
    correct1 = top1.eq(labels)
    correct3 = top3_indices.eq(labels.unsqueeze(1)).any(dim=1)
    accepted = top3_values[:, 0].ge(package.rejection_threshold)
    observed = sorted(set(labels.tolist()))
    per_class = []
    for label in observed:
        mask = labels == label
        per_class.append(
            {
                "class_label": label,
                "class_name": class_order[label],
                "count": int(mask.sum()),
                "top1": float(correct1[mask].float().mean()),
                "top3": float(correct3[mask].float().mean()),
                "coverage": float(accepted[mask].float().mean()),
            }
        )
    confusions = Counter(
        (int(truth), int(prediction))
        for truth, prediction in zip(labels.tolist(), top1.tolist())
        if truth != prediction
    )
    return (
        {
            "prototype_mode": mode,
            "role": benchmark.PROTOTYPE_MODE_ROLES[mode],
            "eval_examples": int(labels.numel()),
            "eval_observed_class_count": len(observed),
            "top1_micro": float(correct1.float().mean()),
            "top3_micro": float(correct3.float().mean()),
            "top1_macro": sum(item["top1"] for item in per_class) / len(per_class),
            "top3_macro": sum(item["top3"] for item in per_class) / len(per_class),
            "coverage": float(accepted.float().mean()),
            "coverage_macro": sum(item["coverage"] for item in per_class) / len(per_class),
            "per_class": per_class,
            "top_confusions": [
                {
                    "true_label": truth,
                    "true_name": class_order[truth],
                    "predicted_label": prediction,
                    "predicted_name": class_order[prediction],
                    "count": count,
                }
                for (truth, prediction), count in confusions.most_common(20)
            ],
        },
        correct1,
    )


def _evaluate_package(
    *,
    package: benchmark.ModelPackage,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_indices: list[int],
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    class_order: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    train_embeddings = benchmark._project_features(
        package, train_features[train_indices], batch_size=batch_size, device=device
    )
    eval_embeddings = benchmark._project_features(
        package, eval_features, batch_size=batch_size, device=device
    )
    train_prototypes = benchmark._compute_prototypes(
        train_embeddings,
        train_labels[train_indices],
        class_count=package.num_classes,
    )
    artifact_prototypes, artifact_metadata = benchmark._load_artifact_prototypes(package)
    modes: dict[str, Any] = {}
    correctness: dict[str, torch.Tensor] = {}
    for mode, prototypes in (
        ("train_recomputed", train_prototypes),
        ("artifact", artifact_prototypes),
    ):
        metrics, correct = _mode_metrics(
            package=package,
            eval_embeddings=eval_embeddings,
            labels=eval_labels,
            prototypes=prototypes,
            class_order=class_order,
            mode=mode,
        )
        if mode == "artifact":
            metrics.update(artifact_metadata)
        modes[mode] = metrics
        correctness[mode] = correct
    return (
        {
            **benchmark._package_report(package),
            "projection_sha256": sha256_file(package.projection_path),
            "prototypes_sha256": sha256_file(package.prototypes_path),
            "prototype_modes": modes,
        },
        correctness,
    )


def _bootstrap_delta(
    reference: torch.Tensor, candidate: torch.Tensor, *, draws: int = 10000
) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired correctness vectors must align")
    n = reference.numel()
    generator = torch.Generator().manual_seed(20260812)
    deltas = []
    ref = reference.float()
    cand = candidate.float()
    for _ in range(draws):
        indices = torch.randint(n, (n,), generator=generator)
        deltas.append(float(cand[indices].mean() - ref[indices].mean()))
    ordered = sorted(deltas)
    return {
        "delta": float(cand.mean() - ref.mean()),
        "ci95_low": ordered[math.floor(0.025 * (draws - 1))],
        "ci95_high": ordered[math.ceil(0.975 * (draws - 1))],
        "draws": draws,
    }


def _cluster_bootstrap_delta(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    groups: list[str],
    *,
    draws: int = 10000,
) -> dict[str, Any]:
    """Paired bootstrap that resamples whole source pages, not individual crops."""

    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired correctness vectors must align")
    if len(groups) != reference.numel() or not groups:
        raise ValueError("bootstrap groups must be non-empty and align with predictions")
    unique_groups = sorted(set(groups))
    group_indices = [
        torch.tensor([index for index, value in enumerate(groups) if value == group])
        for group in unique_groups
    ]
    generator = torch.Generator().manual_seed(20260812)
    ref = reference.float()
    cand = candidate.float()
    deltas = []
    for _ in range(draws):
        sampled = torch.randint(
            len(group_indices), (len(group_indices),), generator=generator
        )
        indices = torch.cat([group_indices[int(index)] for index in sampled])
        deltas.append(float(cand[indices].mean() - ref[indices].mean()))
    ordered = sorted(deltas)
    return {
        "unit": "source_group_page",
        "group_count": len(unique_groups),
        "delta": float(cand.mean() - ref.mean()),
        "ci95_low": ordered[math.floor(0.025 * (draws - 1))],
        "ci95_high": ordered[math.ceil(0.975 * (draws - 1))],
        "draws": draws,
    }


def evaluate_business_test_set(
    *,
    business_manifest_path: Path,
    eval_features_path: Path,
    eval_provenance_path: Path,
    training_manifest_path: Path,
    training_features_path: Path,
    training_provenance_path: Path,
    reference_dir: Path,
    candidates: list[tuple[str, Path]],
    output_path: Path | None,
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict[str, Any]:
    business, eval_provenance, eval_features, eval_labels, eval_groups = (
        _validate_business_cache(
            manifest_path=business_manifest_path,
            features_path=eval_features_path,
            provenance_path=eval_provenance_path,
        )
    )
    training, train_features, train_labels, train_indices = _load_training_cache(
        manifest_path=training_manifest_path,
        features_path=training_features_path,
        provenance_path=training_provenance_path,
    )
    if training.get("class_order") != business.get("class_order"):
        raise ValueError("business and training class orders differ")
    class_order = business["class_order"]
    torch_device = torch.device(device)
    packages = [("historical", reference_dir), *candidates]
    if len({name for name, _ in packages}) != len(packages):
        raise ValueError("model names must be unique")

    model_reports: dict[str, Any] = {}
    correctness: dict[str, dict[str, torch.Tensor]] = {}
    package_objects: dict[str, benchmark.ModelPackage] = {}
    for name, path in packages:
        package = benchmark._load_model_package(path)
        package_objects[name] = package
        if package.class_names != class_order:
            raise ValueError(f"model {name} class order differs from business test")
        if package.hidden_dim != eval_features.size(1):
            raise ValueError(f"model {name} hidden_dim differs from evaluation cache")
        report, correct = _evaluate_package(
            package=package,
            train_features=train_features,
            train_labels=train_labels,
            train_indices=train_indices,
            eval_features=eval_features,
            eval_labels=eval_labels,
            class_order=class_order,
            batch_size=batch_size,
            device=torch_device,
        )
        model_reports[name] = report
        correctness[name] = correct

    candidate_bindings: dict[str, Any] = {}
    for name, _path in candidates:
        candidate_bindings[name] = _validate_candidate_training_binding(
            package_objects[name],
            snapshot_manifest=training,
            snapshot_manifest_path=training_manifest_path.resolve(),
            supplied_features_path=training_features_path.resolve(),
            supplied_provenance_path=training_provenance_path.resolve(),
        )

    comparisons: dict[str, Any] = {}
    for name, _path in candidates:
        modes = {}
        for mode in ("artifact", "train_recomputed"):
            reference_metrics = model_reports["historical"]["prototype_modes"][mode]
            candidate_metrics = model_reports[name]["prototype_modes"][mode]
            ref_correct = correctness["historical"][mode]
            cand_correct = correctness[name][mode]
            modes[mode] = {
                "role": benchmark.PROTOTYPE_MODE_ROLES[mode],
                "top1_micro": {
                    "cluster_bootstrap": _cluster_bootstrap_delta(
                        ref_correct, cand_correct, eval_groups
                    ),
                    "crop_level_bootstrap_exploratory": _bootstrap_delta(
                        ref_correct, cand_correct
                    ),
                },
                "top3_micro_delta": (
                    candidate_metrics["top3_micro"] - reference_metrics["top3_micro"]
                ),
                "top1_macro_delta": (
                    candidate_metrics["top1_macro"] - reference_metrics["top1_macro"]
                ),
                "top3_macro_delta": (
                    candidate_metrics["top3_macro"] - reference_metrics["top3_macro"]
                ),
                "candidate_wins": int((cand_correct & ~ref_correct).sum()),
                "candidate_losses": int((ref_correct & ~cand_correct).sum()),
            }
        comparisons[name] = {"prototype_modes": modes}

    class_counts = Counter(eval_labels.tolist())
    group_counts = Counter(eval_groups)
    report = {
        "schema_version": SCHEMA_VERSION,
        "business_test": {
            "test_set_id": business["test_set_id"],
            "content_sha256": business["content_sha256"],
            "manifest_path": str(business_manifest_path.resolve()),
            "manifest_sha256": sha256_file(business_manifest_path),
            "authority": business["authority"],
            "analysis_count": business["analysis_count"],
            "row_count": business["row_count"],
            "class_count": business["class_count"],
            "excluded_count": business["excluded_count"],
            "class_counts": business["class_counts"],
            "analysis_row_counts": business["analysis_row_counts"],
            "dominant_class": {
                "label": max(class_counts, key=class_counts.get),
                "name": class_order[max(class_counts, key=class_counts.get)],
                "count": max(class_counts.values()),
                "fraction": max(class_counts.values()) / len(eval_labels),
            },
            "dominant_page_fraction": max(group_counts.values()) / len(eval_labels),
            "independence": business["independence"],
            "independence_scope": (
                "exact RGB page/crop identity only; source-lineage and perceptual "
                "near-duplicate disjointness are not established"
            ),
            "limitations": [
                "covers_87_of_286_runtime_classes",
                "303_examples_below_previous_generic_600_example_gate",
                "10_pages_below_previous_generic_20_group_gate",
                "dominant_class_fraction_above_previous_25pct_gate",
                "business_authority_overrides_generic_coverage_gate_for_final_report_only",
                "no_persisted_source_lineage_or_perceptual_near_duplicate_audit_against_train",
            ],
        },
        "features": {
            "path": str(eval_features_path.resolve()),
            "sha256": sha256_file(eval_features_path),
            "provenance_path": str(eval_provenance_path.resolve()),
            "provenance_sha256": sha256_file(eval_provenance_path),
            "backbone": eval_provenance.get("backbone"),
        },
        "training_reference": {
            "snapshot_id": training["snapshot_id"],
            "manifest_sha256": sha256_file(training_manifest_path),
            "features_sha256": sha256_file(training_features_path),
            "train_examples": len(train_indices),
        },
        "candidate_training_bindings": candidate_bindings,
        "models": model_reports,
        "comparisons_to_historical": comparisons,
        "interpretation": {
            "authoritative_for_current_model_results": True,
            "eligible_for_training_or_tuning": False,
            "artifact_mode": "exact_packaged_runtime_behavior",
            "train_recomputed_mode": "fair_common_train_prototype_protocol",
            "production_promotion_automatic": False,
        },
    }
    report["report_sha256"] = canonical_json_sha(report)
    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--business-manifest", type=Path, required=True)
    parser.add_argument("--eval-features", type=Path, required=True)
    parser.add_argument("--eval-provenance", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--training-provenance", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Registered candidate package; repeatable",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    candidates: list[tuple[str, Path]] = []
    try:
        for raw in args.candidate:
            name, separator, path = raw.partition("=")
            if not separator or not name.strip() or not path.strip():
                raise ValueError("--candidate must use NAME=PATH")
            candidates.append((name.strip(), Path(path.strip())))
        report = evaluate_business_test_set(
            business_manifest_path=args.business_manifest,
            eval_features_path=args.eval_features,
            eval_provenance_path=args.eval_provenance,
            training_manifest_path=args.training_manifest,
            training_features_path=args.training_features,
            training_provenance_path=args.training_provenance,
            reference_dir=args.reference,
            candidates=candidates,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "test_set_id": report["business_test"]["test_set_id"],
                    "models": {
                        name: {
                            mode: {
                                key: values[key]
                                for key in ("top1_micro", "top3_micro", "top1_macro", "top3_macro")
                            }
                            for mode, values in model["prototype_modes"].items()
                        }
                        for name, model in report["models"].items()
                    },
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

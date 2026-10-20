from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_model_replacement.py"
spec = importlib.util.spec_from_file_location("benchmark_model_replacement", SCRIPT)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)

from backend.codex_pipeline.models.projection_head import ProjectionHead


CLASS_COUNT = 286
LOCKED_CLASS_COUNT = 7


def _write_package(
    version_root: Path,
    *,
    shifted_artifact: bool = False,
    benchmark_inputs: tuple[Path, Path] | None = None,
    legacy_one_based_labels: bool = False,
    unavailable_artifact: bool = False,
    reference_provenance: bool = False,
) -> Path:
    runtime_root = version_root / "runtime"
    weights_dir = runtime_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    class_names = [f"class_{idx:03d}" for idx in range(CLASS_COUNT)]
    config = {
        "model_version": version_root.name,
        "backbone": "dinov2_vits14",
        "embedding_dim": CLASS_COUNT,
        "hidden_dim": CLASS_COUNT,
        "image_size": 224,
        "rejection_threshold": 0.35,
        "num_classes": CLASS_COUNT,
        "class_names": class_names,
    }
    (runtime_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    head = ProjectionHead(input_dim=CLASS_COUNT, embedding_dim=CLASS_COUNT)
    with torch.no_grad():
        first = head.net[0]
        second = head.net[3]
        first.weight.zero_()
        first.bias.zero_()
        first.weight.copy_(torch.eye(CLASS_COUNT))
        second.weight.zero_()
        second.bias.zero_()
        second.weight.copy_(torch.eye(CLASS_COUNT))
    torch.save(head.state_dict(), weights_dir / "projection.pt")
    prototypes = torch.eye(CLASS_COUNT)
    if shifted_artifact:
        prototypes = torch.roll(prototypes, shifts=1, dims=0)
    artifact_labels = (
        list(range(1, CLASS_COUNT + 1))
        if legacy_one_based_labels
        else list(range(CLASS_COUNT))
    )
    artifact_names = {
        label: class_names[index] for index, label in enumerate(artifact_labels)
    }
    if unavailable_artifact:
        artifact_names[artifact_labels[-1]] = artifact_names[artifact_labels[0]]
    torch.save(
        {
            "prototypes": prototypes,
            "class_labels": artifact_labels,
            "class_names": artifact_names,
            "embedding_dim": CLASS_COUNT,
        },
        weights_dir / "prototypes.pt",
    )

    extra_artifacts: list[str] = []
    data: dict[str, object] = {}
    training: dict[str, object] = {}
    if reference_provenance:
        data = {"dataset_contract": "synthetic-reference.v1"}
        training = {"recipe": "synthetic-reference-training.v1"}
    if benchmark_inputs is not None:
        features_path, snapshot_manifest_path = benchmark_inputs
        features_provenance_path = Path(str(features_path) + ".prov.json")
        snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        training_config = version_root / "training" / "research.yaml"
        checkpoint = version_root / "checkpoints" / "best.pt"
        training_config.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        training_config.write_text("training:\n  seed: 42\n", encoding="utf-8")
        checkpoint.write_text("checkpoint", encoding="utf-8")
        extra_artifacts.extend(["training/research.yaml", "checkpoints/best.pt"])
        data = {
            "training_snapshot": {
                "snapshot_id": snapshot_manifest["snapshot_id"],
                "content_sha256": snapshot_manifest["content_sha256"],
                "manifest_sha256": benchmark.sha256_file(snapshot_manifest_path),
            },
            "features_cache": {
                "path": str(features_path.resolve()),
                "sha256": benchmark.sha256_file(features_path),
            },
            "features_provenance": {
                "path": str(features_provenance_path.resolve()),
                "sha256": benchmark.sha256_file(features_provenance_path),
            },
        }
        training = {
            "config": {
                "path": str(training_config.resolve()),
                "sha256": benchmark.sha256_file(training_config),
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": benchmark.sha256_file(checkpoint),
            },
        }

    artifact_records = []
    checksum_lines = []
    for relative in [
        "runtime/config.json",
        "runtime/weights/projection.pt",
        "runtime/weights/prototypes.pt",
        *extra_artifacts,
    ]:
        path = version_root / relative
        digest = benchmark.sha256_file(path)
        artifact_records.append(
            {"path": relative, "sha256": digest, "size": path.stat().st_size}
        )
        checksum_lines.append(f"{digest}  {relative}")
    manifest = {
        "schema_version": 1,
        "model_id": "codex_classifier",
        "version_id": version_root.name,
        "status": "candidate",
        "data": data,
        "training": training,
        "artifacts": artifact_records,
    }
    (version_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (version_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return version_root


def _write_dataset(root: Path) -> tuple[Path, Path]:
    features_path = root / "precomputed" / "features.pt"
    manifest_path = root / "snapshot_manifest.json"
    features: list[torch.Tensor] = []
    labels: list[int] = []
    row_ids: list[str] = []
    image_paths: list[str] = []
    dataset_splits: list[str] = []
    source_groups: list[str] = []
    rows: list[dict[str, object]] = []
    class_names = [f"class_{idx:03d}" for idx in range(CLASS_COUNT)]

    def append_row(class_label: int, split: str) -> None:
        row_id = f"row-{split}-{class_label:03d}"
        relative = f"Elements/{split}/{class_label:03d}.bmp"
        feature = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        feature[class_label] = 1.0
        features.append(feature)
        labels.append(class_label)
        row_ids.append(row_id)
        image_paths.append(str((root / relative).resolve()))
        dataset_splits.append(split)
        source_group = f"group-{split}-{class_label:03d}"
        source_groups.append(source_group)
        rows.append(
            {
                "row_id": row_id,
                "dataset_split": split,
                "class_label": class_label,
                "class_name": class_names[class_label],
                "output_path": relative,
                "source_group": source_group,
            }
        )

    for class_label in range(CLASS_COUNT):
        append_row(class_label, "train")
    for class_label in range(LOCKED_CLASS_COUNT):
        append_row(class_label, "locked_test")

    payload = {
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_paths": image_paths,
        "dataset_splits": dataset_splits,
        "source_groups": source_groups,
        "row_ids": row_ids,
        "class_names": {index: name for index, name in enumerate(class_names)},
        "hidden_dim": CLASS_COUNT,
    }
    manifest = {
        "schema_version": "training-snapshot.v2",
        "snapshot_id": "snapshot-" + "a" * 20,
        "content_sha256": "a" * 64,
        "class_order": class_names,
        "class_order_sha256": benchmark.class_order_sha256(class_names),
        "class_count": CLASS_COUNT,
        "row_count": len(rows),
        "split_counts": {"train": CLASS_COUNT, "locked_test": LOCKED_CLASS_COUNT},
        "promotion_evaluation_ready": False,
        "source_group_assignments": {
            row["source_group"]: row["dataset_split"] for row in rows
        },
        "components": [
            {
                "component_id": f"component-{index:03d}",
                "dataset_split": row["dataset_split"],
                "row_ids": [row["row_id"]],
                "source_groups": [row["source_group"]],
            }
            for index, row in enumerate(rows)
        ],
        "duplicates": [],
        "rows": rows,
    }
    features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, features_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _refresh_provenance(features_path, manifest_path)
    return features_path, manifest_path


def _refresh_provenance(features_path: Path, manifest_path: Path) -> None:
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = {
        "schema_version": "training-snapshot-checksums.v1",
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_manifest_sha256": benchmark.sha256_file(manifest_path),
        "metadata_csv_sha256": "b" * 64,
        "element_count": len(manifest["rows"]),
        "elements_sha256": "c" * 64,
    }
    checksums_path = manifest_path.parent / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    row_ids = payload.get("row_ids")
    provenance = {
        "schema_version": "features-cache.v1",
        "runtime_class_order_sha256": manifest["class_order_sha256"],
        "dataset_split_sha256": benchmark._canonical_json_sha(payload["dataset_splits"]),
        "row_ids_sha256": benchmark._canonical_json_sha(row_ids) if row_ids else None,
        "class_count": CLASS_COUNT,
        "image_count": len(payload["labels"]),
        "hidden_dim": CLASS_COUNT,
        "metadata_csv_sha256": checksums["metadata_csv_sha256"],
        "snapshot": {
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_content_sha256": manifest["content_sha256"],
            "snapshot_manifest_sha256": benchmark.sha256_file(manifest_path),
            "snapshot_checksums_sha256": benchmark.sha256_file(checksums_path),
            "snapshot_elements_sha256": checksums["elements_sha256"],
            "verified_element_count": len(manifest["rows"]),
        },
    }
    Path(str(features_path) + ".prov.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def _run(
    root: Path,
    *,
    prototype_mode: str = "both",
    shifted_candidate: bool = False,
) -> dict[str, object]:
    features_path, manifest_path = _write_dataset(root / "snapshot")
    runtime_version = _write_package(root / "runtime-version")
    candidate_version = _write_package(
        root / "candidate-version",
        shifted_artifact=shifted_candidate,
        benchmark_inputs=(features_path, manifest_path),
    )
    return benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode=prototype_mode,
        output_path=root / "benchmark.json",
        batch_size=64,
        device="cpu",
    )


def test_benchmark_reports_fair_and_artifact_heads_without_promotion(tmp_path: Path) -> None:
    report = _run(tmp_path, shifted_candidate=True)

    assert report["schema_version"] == "benchmark-model-replacement.v2"
    assert report["research_only"] is True
    assert report["promotion_decision"] == "blocked"
    assert report["contract"]["eval_split"] == "locked_test"
    assert report["contract"]["locked_test_used_for_fit"] is False
    assert report["mapping"]["row_id_matches"] == CLASS_COUNT + LOCKED_CLASS_COUNT
    assert report["mapping"]["path_fallback_matches"] == 0
    assert report["mapping"]["train_eval_disjoint"] is True
    assert report["mapping"]["train_eval_source_groups_disjoint"] is True
    assert report["candidate_provenance_binding"]["validated"] is True
    assert report["reference_provenance"]["status"] == "unverified"
    assert report["reference_provenance"]["data"] == {}
    assert report["evaluation_coverage"]["observed_class_count"] == LOCKED_CLASS_COUNT
    assert report["evaluation_coverage"]["sufficient_for_promotion"] is False
    assert report["evaluation_coverage"]["warning"].startswith("promotion blocked:")
    assert report["evaluation_coverage"]["blocking_reasons"] == [
        "volume:insufficient_evaluation_examples",
        "classes:insufficient_evaluation_classes",
        "snapshot_readiness:not_promotion_evaluation_ready",
        "evaluation_structure:too_few_source_groups",
    ]

    runtime_modes = report["models"]["runtime"]["prototype_modes"]
    candidate_modes = report["models"]["candidate"]["prototype_modes"]
    assert set(runtime_modes) == {"train_recomputed", "artifact"}
    assert runtime_modes["train_recomputed"]["top1_micro"] == 1.0
    assert runtime_modes["artifact"]["top1_micro"] == 1.0
    assert candidate_modes["train_recomputed"]["top1_micro"] == 1.0
    assert candidate_modes["artifact"]["top1_micro"] == 0.0
    assert runtime_modes["artifact"]["role"] == "historical_behavior_or_packaging_fidelity"
    assert runtime_modes["train_recomputed"]["role"] == "comparable_train_only_generalization"
    artifact_comparison = report["comparison"]["prototype_modes"]["artifact"]
    assert artifact_comparison["generalization_evidence_eligible"] is False
    assert artifact_comparison["reference_artifact_advantage_is_generalization_evidence"] is False
    assert artifact_comparison["interpretation_guard"] == "prohibited_unverified_reference_provenance"
    assert report["evaluation_structure"]["source_group_count"] == LOCKED_CLASS_COUNT
    assert report["evaluation_structure"]["valid_for_promotion"] is False
    assert (tmp_path / "benchmark.json").is_file()


def test_eval_split_is_required_and_locked_test_is_not_silently_replaced() -> None:
    parser = benchmark.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--features", "features.pt",
                "--split-manifest", "snapshot.json",
                "--runtime-dir", "runtime",
                "--candidate-dir", "candidate",
            ]
        )


def test_populated_reference_data_and_training_are_reported_documented(
    tmp_path: Path,
) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(
        tmp_path / "runtime-version", reference_provenance=True
    )
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="artifact",
    )
    reference = report["reference_provenance"]
    assert reference["status"] == "documented"
    assert "contents_not_independently_attested" in reference["verification_basis"]
    assert reference["data"]["dataset_contract"] == "synthetic-reference.v1"
    assert reference["training"]["recipe"] == "synthetic-reference-training.v1"
    artifact = report["comparison"]["prototype_modes"]["artifact"]
    assert artifact["generalization_evidence_eligible"] is False
    assert artifact["interpretation_guard"] == "artifact_mode_is_behavior_or_packaging_fidelity_only"


def test_current_like_eval_structure_is_blocked_without_eval_calibration() -> None:
    labels = [0] * 72 + [1] * 10  # 72/82 = 87.8% dominant class.
    source_groups = [f"group-{index % 4}" for index in range(82)]
    structure = benchmark._evaluation_structure(
        eval_labels=labels,
        eval_source_groups=source_groups,
        class_order=[f"class_{index:03d}" for index in range(CLASS_COUNT)],
    )
    assert structure["policy"] == "predeclared_conservative_v1_not_calibrated_on_eval"
    assert structure["source_group_count"] == 4
    assert structure["dominant_class"]["fraction"] == pytest.approx(72 / 82)
    assert structure["valid_for_promotion"] is False
    assert "too_few_source_groups" in structure["violations"]
    assert "class_dominance_exceeds_limit" in structure["violations"]


def test_invalid_locked_eval_structure_forces_promotion_sufficiency_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["promotion_evaluation_ready"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_provenance(features_path, manifest_path)
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    monkeypatch.setattr(benchmark, "MIN_PROMOTION_EVAL_EXAMPLES", 1)
    monkeypatch.setattr(benchmark, "MIN_PROMOTION_EVAL_CLASSES", 1)
    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="artifact",
    )
    assert report["evaluation_structure"]["valid_for_promotion"] is False
    assert report["evaluation_coverage"]["sufficient_for_promotion"] is False
    assert report["evaluation_coverage"]["blocking_reasons"] == [
        "evaluation_structure:too_few_source_groups"
    ]
    assert report["evaluation_coverage"]["warning"] == (
        "promotion blocked: evaluation_structure:too_few_source_groups"
    )


def test_row_id_mapping_takes_priority_over_different_paths(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    payload["image_paths"] = [str(tmp_path / "other" / f"{index}.bmp") for index in range(len(payload["labels"]))]
    torch.save(payload, features_path)
    _refresh_provenance(features_path, manifest_path)
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )

    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="artifact",
    )
    assert report["mapping"]["row_id_matches"] == len(payload["labels"])
    assert report["mapping"]["path_fallback_matches"] == 0


def test_normalized_path_mapping_is_used_when_cache_has_no_row_ids(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    payload["row_ids"] = None
    payload["image_paths"] = [
        str(Path(path).relative_to(manifest_path.parent)) for path in payload["image_paths"]
    ]
    torch.save(payload, features_path)
    _refresh_provenance(features_path, manifest_path)
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )

    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="artifact",
    )
    assert report["mapping"]["row_id_matches"] == 0
    assert report["mapping"]["path_fallback_matches"] == len(payload["labels"])


def test_benchmark_rejects_train_eval_overlap(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][-1]["row_id"] = manifest["rows"][0]["row_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row_id values must be unique"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=tmp_path / "unused-runtime",
            candidate_dir=tmp_path / "unused-candidate",
            eval_split="locked_test",
        )


def test_benchmark_rejects_snapshot_cache_label_mismatch(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    payload["labels"][-1] = 8
    torch.save(payload, features_path)
    with pytest.raises(ValueError, match="label mismatch"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=tmp_path / "unused-runtime",
            candidate_dir=tmp_path / "unused-candidate",
            eval_split="locked_test",
        )


def test_benchmark_rejects_snapshot_cache_split_mismatch(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    payload["dataset_splits"][-1] = "dev"
    torch.save(payload, features_path)
    _refresh_provenance(features_path, manifest_path)
    with pytest.raises(ValueError, match="split mismatch"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=tmp_path / "unused-runtime",
            candidate_dir=tmp_path / "unused-candidate",
            eval_split="locked_test",
        )


def test_benchmark_rejects_cache_provenance_hash_mismatch(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    provenance_path = Path(str(features_path) + ".prov.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_class_order_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="class-order hash mismatch"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=tmp_path / "unused-runtime",
            candidate_dir=tmp_path / "unused-candidate",
            eval_split="locked_test",
        )


def test_benchmark_rejects_invalid_registered_package_before_projection(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    with (candidate_version / "runtime" / "weights" / "projection.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=runtime_version,
            candidate_dir=candidate_version,
            eval_split="locked_test",
            prototype_mode="artifact",
        )


def test_benchmark_rejects_absent_candidate_provenance(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(tmp_path / "candidate-version")
    with pytest.raises(ValueError, match="data.training_snapshot"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=runtime_version,
            candidate_dir=candidate_version,
            eval_split="locked_test",
            prototype_mode="artifact",
        )


def test_benchmark_rejects_mismatched_candidate_cache_provenance(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    package_manifest_path = candidate_version / "manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    package_manifest["data"]["features_cache"]["sha256"] = "0" * 64
    package_manifest_path.write_text(json.dumps(package_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="features_cache checksum mismatch"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=runtime_version,
            candidate_dir=candidate_version,
            eval_split="locked_test",
            prototype_mode="artifact",
        )


def test_benchmark_rejects_source_group_leakage(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leaked_group = manifest["rows"][0]["source_group"]
    manifest["rows"][-1]["source_group"] = leaked_group
    manifest["components"][-1]["source_groups"] = [leaked_group]
    manifest.pop("source_group_assignments")
    payload["source_groups"][-1] = leaked_group
    torch.save(payload, features_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_provenance(features_path, manifest_path)
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    with pytest.raises(ValueError, match="source_group appears in multiple components"):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=runtime_version,
            candidate_dir=candidate_version,
            eval_split="locked_test",
            prototype_mode="artifact",
        )


def test_benchmark_output_is_write_once(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.json"
    output_path.write_text("sealed evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark.benchmark_model_replacement(
            features_path=tmp_path / "missing-features.pt",
            split_manifest_path=tmp_path / "missing-snapshot.json",
            runtime_dir=tmp_path / "missing-runtime",
            candidate_dir=tmp_path / "missing-candidate",
            eval_split="locked_test",
            output_path=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "sealed evidence\n"


def test_artifact_mode_supports_legacy_bijective_labels(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(
        tmp_path / "runtime-version", legacy_one_based_labels=True
    )
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="artifact",
    )
    runtime_artifact = report["models"]["runtime"]["prototype_modes"]["artifact"]
    assert runtime_artifact["available"] is True
    assert runtime_artifact["label_protocol"] == "legacy_bijective_name_mapping"
    assert runtime_artifact["top1_micro"] == 1.0


def test_both_mode_keeps_fair_head_when_artifact_protocol_is_unavailable(tmp_path: Path) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
        unavailable_artifact=True,
    )
    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="locked_test",
        prototype_mode="both",
    )
    candidate_modes = report["models"]["candidate"]["prototype_modes"]
    assert candidate_modes["train_recomputed"]["available"] is True
    assert candidate_modes["train_recomputed"]["top1_micro"] == 1.0
    assert candidate_modes["artifact"]["available"] is False
    assert "taxonomy differs" in candidate_modes["artifact"]["reason"]
    assert report["comparison"]["prototype_modes"]["artifact"]["available"] is False


def test_dev_coverage_is_never_sufficient_for_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["dataset_splits"] = [
        "dev" if split == "locked_test" else split
        for split in payload["dataset_splits"]
    ]
    for row in manifest["rows"]:
        if row["dataset_split"] == "locked_test":
            row["dataset_split"] = "dev"
    for component in manifest["components"]:
        if component["dataset_split"] == "locked_test":
            component["dataset_split"] = "dev"
    manifest["split_counts"] = {"train": CLASS_COUNT, "dev": LOCKED_CLASS_COUNT}
    manifest["source_group_assignments"] = {
        row["source_group"]: row["dataset_split"] for row in manifest["rows"]
    }
    manifest["promotion_evaluation_ready"] = True
    torch.save(payload, features_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_provenance(features_path, manifest_path)
    runtime_version = _write_package(tmp_path / "runtime-version")
    candidate_version = _write_package(
        tmp_path / "candidate-version",
        benchmark_inputs=(features_path, manifest_path),
    )
    monkeypatch.setattr(benchmark, "MIN_PROMOTION_EVAL_EXAMPLES", 1)
    monkeypatch.setattr(benchmark, "MIN_PROMOTION_EVAL_CLASSES", 1)
    report = benchmark.benchmark_model_replacement(
        features_path=features_path,
        split_manifest_path=manifest_path,
        runtime_dir=runtime_version,
        candidate_dir=candidate_version,
        eval_split="dev",
        prototype_mode="artifact",
    )
    assert report["evaluation_coverage"]["sufficient_for_promotion"] is False


def _assert_component_contract_error(
    tmp_path: Path,
    *,
    mutate,
    match: str,
) -> None:
    features_path, manifest_path = _write_dataset(tmp_path / "snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        benchmark.benchmark_model_replacement(
            features_path=features_path,
            split_manifest_path=manifest_path,
            runtime_dir=tmp_path / "unused-runtime",
            candidate_dir=tmp_path / "unused-candidate",
            eval_split="locked_test",
        )


def test_components_must_cover_every_active_row(tmp_path: Path) -> None:
    _assert_component_contract_error(
        tmp_path,
        mutate=lambda manifest: manifest["components"].pop(),
        match="components omit active row",
    )


def test_source_group_cannot_belong_to_two_components(tmp_path: Path) -> None:
    def duplicate_group(manifest: dict[str, object]) -> None:
        rows = manifest["rows"]
        components = manifest["components"]
        rows[1]["source_group"] = rows[0]["source_group"]
        components[1]["source_groups"] = [rows[0]["source_group"]]

    _assert_component_contract_error(
        tmp_path,
        mutate=duplicate_group,
        match="source_group appears in multiple components",
    )


def test_component_declared_groups_must_equal_groups_derived_from_rows(
    tmp_path: Path,
) -> None:
    _assert_component_contract_error(
        tmp_path,
        mutate=lambda manifest: manifest["components"][0].update(
            {"source_groups": ["group-not-derived"]}
        ),
        match="source_groups mismatch derived rows",
    )


def test_component_cannot_cross_train_and_eval_splits(tmp_path: Path) -> None:
    def cross_splits(manifest: dict[str, object]) -> None:
        components = manifest["components"]
        evaluation = components.pop()
        components[0]["row_ids"].extend(evaluation["row_ids"])
        components[0]["source_groups"].extend(evaluation["source_groups"])

    _assert_component_contract_error(
        tmp_path,
        mutate=cross_splits,
        match="component split mismatch",
    )


def test_duplicate_connected_groups_cannot_cross_components_or_splits(
    tmp_path: Path,
) -> None:
    def leak_duplicate(manifest: dict[str, object]) -> None:
        manifest["duplicates"].append(
            {
                "row_id": "row-excluded-duplicate",
                "duplicate_of": manifest["rows"][0]["row_id"],
                "source_group": manifest["rows"][-1]["source_group"],
                "dataset_split": "excluded",
            }
        )

    _assert_component_contract_error(
        tmp_path,
        mutate=leak_duplicate,
        match="duplicate-connected rows cross components/splits",
    )

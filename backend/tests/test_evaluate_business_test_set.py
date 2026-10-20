from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_business_test_set import (
    _bootstrap_delta,
    _cluster_bootstrap_delta,
    _validate_candidate_training_binding,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_paired_bootstrap_reports_exact_direction() -> None:
    reference = torch.tensor([True, False, False, True])
    candidate = torch.tensor([True, True, False, True])
    result = _bootstrap_delta(reference, candidate, draws=1000)
    assert result["delta"] == pytest.approx(0.25)
    assert result["ci95_low"] <= result["delta"] <= result["ci95_high"]


def test_cluster_bootstrap_resamples_whole_source_pages() -> None:
    reference = torch.tensor([True, True, False, False])
    candidate = torch.tensor([False, False, True, True])
    result = _cluster_bootstrap_delta(
        reference, candidate, ["page-a", "page-a", "page-b", "page-b"], draws=1000
    )
    assert result["unit"] == "source_group_page"
    assert result["group_count"] == 2
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci95_low"] == pytest.approx(-1.0)
    assert result["ci95_high"] == pytest.approx(1.0)


def test_candidate_binding_accepts_equal_version_local_feature_content(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = {"snapshot_id": "snapshot-abc", "content_sha256": "abc"}
    _write_json(snapshot_path, snapshot)
    supplied_features = tmp_path / "supplied.pt"
    candidate_features = tmp_path / "candidate.pt"
    supplied_features.write_bytes(b"same frozen tensor")
    candidate_features.write_bytes(b"same frozen tensor")
    stable = {
        "schema_version": "features-cache.v1",
        "backbone": {"name": "dinov2_vits14"},
        "class_count": 286,
        "dataset_split_sha256": "split",
        "hidden_dim": 384,
        "image_count": 10,
        "metadata_csv_sha256": "metadata",
        "row_ids_sha256": "rows",
        "runtime_class_order_sha256": "classes",
        "snapshot": {"snapshot_id": "snapshot-abc"},
        "source_group_sha256": "groups",
    }
    supplied_provenance = tmp_path / "supplied.prov.json"
    candidate_provenance = tmp_path / "candidate.prov.json"
    _write_json(supplied_provenance, {**stable, "config_sha256": "b0"})
    _write_json(candidate_provenance, {**stable, "config_sha256": "b1"})
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("training: {}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "data": {
                "training_snapshot": {
                    **snapshot,
                    "manifest_sha256": sha256_file(snapshot_path),
                },
                "features_cache": {
                    "path": str(candidate_features),
                    "sha256": sha256_file(candidate_features),
                },
                "features_provenance": {
                    "path": str(candidate_provenance),
                    "sha256": sha256_file(candidate_provenance),
                },
            },
            "training": {
                "config": {"path": str(config), "sha256": sha256_file(config)},
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                },
            },
        },
    )
    package = SimpleNamespace(manifest_path=manifest_path, version_root=tmp_path)

    result = _validate_candidate_training_binding(
        package,
        snapshot_manifest=snapshot,
        snapshot_manifest_path=snapshot_path,
        supplied_features_path=supplied_features,
        supplied_provenance_path=supplied_provenance,
    )
    assert result["validated"] is True
    assert result["binding_mode"] == "content_equivalent_version_local_cache"

    candidate_features.write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _validate_candidate_training_binding(
            package,
            snapshot_manifest=snapshot,
            snapshot_manifest_path=snapshot_path,
            supplied_features_path=supplied_features,
            supplied_provenance_path=supplied_provenance,
        )

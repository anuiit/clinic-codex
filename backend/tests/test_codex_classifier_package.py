# type: ignore
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend.codex_model.classifier import (
    CodexClassifier,
    ModelPackageValidationError,
    _ProjectionHead,
    _DEFAULT_LOCAL_BACKBONE_MANIFESTS,
)
from backend.services.model_registry import ModelRegistry


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones((inputs.shape[0], self.hidden_dim), dtype=torch.float32)


def _write_package(
    tmp_path: Path,
    *,
    version_id: str,
    backbone: str,
    hidden_dim: int,
    class_names: list[str] | None = None,
) -> tuple[ModelRegistry, Path]:
    names = class_names or ["atl", "calli"]
    source = tmp_path / f"source-{version_id}"
    weights = source / "weights"
    weights.mkdir(parents=True)
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.save(
        {
            "prototypes": prototypes,
            "class_names": {4: "atl", 9: "calli"},
            "class_labels": torch.tensor([4, 9], dtype=torch.int64),
            "embedding_dim": 2,
        },
        weights / "prototypes.pt",
    )
    torch.save(
        _ProjectionHead(hidden_dim=hidden_dim, embedding_dim=2).state_dict(),
        weights / "projection.pt",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": version_id,
                "backbone": backbone,
                "hidden_dim": hidden_dim,
                "embedding_dim": 2,
                "image_size": 224,
                "rejection_threshold": 0.35,
                "num_classes": 2,
                "class_names": names,
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path)
    registry.create_version_from_artifacts(
        version_id,
        artifact_sources={
            "runtime/config.json": source / "config.json",
            "runtime/weights/prototypes.pt": weights / "prototypes.pt",
            "runtime/weights/projection.pt": weights / "projection.pt",
        },
    )
    return registry, registry.resolve_runtime_package(version_id)


@pytest.mark.parametrize(
    ("backbone", "hidden_dim"),
    [("dinov2_vits14", 384), ("dinov2_vitb14", 768)],
)
def test_classifier_loads_validated_runtime_packages(
    tmp_path, monkeypatch, backbone, hidden_dim
):
    registry, runtime_dir = _write_package(
        tmp_path,
        version_id=f"valid-{hidden_dim}",
        backbone=backbone,
        hidden_dim=hidden_dim,
    )
    loaded: list[str] = []

    def fake_load(_repository, requested_backbone, *, pretrained):
        assert pretrained is True
        loaded.append(requested_backbone)
        return _FakeBackbone(hidden_dim)

    monkeypatch.setattr(torch.hub, "load", fake_load)

    classifier = CodexClassifier(
        model_dir=registry.resolve_runtime_package("candidate"),
        device="cpu",
        backbone_source="online",
    )
    result = classifier.classify(np.zeros((12, 8, 3), dtype=np.uint8), top_k=2)

    assert runtime_dir.name == "runtime"
    assert classifier.num_classes == 2
    assert result["class_label"] in {4, 9}
    assert loaded == [backbone]


def test_classifier_rejects_taxonomy_before_loading_backbone(tmp_path, monkeypatch):
    _registry, runtime_dir = _write_package(
        tmp_path,
        version_id="invalid-taxonomy",
        backbone="dinov2_vits14",
        hidden_dim=384,
        class_names=["calli", "atl"],
    )
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *_args, **_kwargs: pytest.fail("backbone loaded before taxonomy validation"),
    )

    with pytest.raises(ModelPackageValidationError, match="taxonomy"):
        CodexClassifier(model_dir=runtime_dir, device="cpu")


def test_classifier_preserves_legacy_weights_directory_fallback(tmp_path, monkeypatch):
    _, runtime_dir = _write_package(tmp_path, version_id="legacy", backbone="dinov2_vits14", hidden_dim=384)
    monkeypatch.setattr(sys.modules[CodexClassifier.__module__], "__file__", str(runtime_dir / "classifier.py"))
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *_args, **_kwargs: _FakeBackbone(384),
    )

    classifier = CodexClassifier(device="cpu", backbone_source="online")

    assert classifier.num_classes == 2
    assert classifier.config["backbone"] == "dinov2_vits14"


def test_classifier_loads_audited_local_backbone_without_remote_hub(
    tmp_path, monkeypatch
):
    _registry, runtime_dir = _write_package(
        tmp_path,
        version_id="offline-local-pin",
        backbone="dinov2_vits14",
        hidden_dim=384,
    )
    repository = tmp_path / "dinov2-source"
    repository.mkdir()
    hubconf_bytes = b"# local test pin\n"
    (repository / "hubconf.py").write_bytes(hubconf_bytes)
    weights_path = tmp_path / "dinov2_vits14_pretrain.pth"
    torch.save({}, weights_path)
    manifest_path = tmp_path / "dinov2-vits14-local.json"
    source_digest = hashlib.sha256(b"hubconf.py\0" + hubconf_bytes).hexdigest()
    weights_digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "dinov2-local-pin.v1",
                "backbone": "dinov2_vits14",
                "repository_path": str(repository),
                "weights_path": str(weights_path),
                "weights_sha256": weights_digest,
                "source_tree_sha256": source_digest,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_load(requested_repository, requested_backbone, *, source, pretrained):
        calls.append(
            (requested_repository, requested_backbone, source, pretrained)
        )
        return _FakeBackbone(384)

    monkeypatch.setattr(torch.hub, "load", fake_load)
    monkeypatch.setitem(
        _DEFAULT_LOCAL_BACKBONE_MANIFESTS, "dinov2_vits14", manifest_path
    )

    classifier = CodexClassifier(
        model_dir=runtime_dir,
        device="cpu",
        backbone_source="local",
    )

    assert calls == [(str(repository), "dinov2_vits14", "local", False)]
    assert classifier.backbone_identity == {
        "mode": "local_pin",
        "backbone": "dinov2_vits14",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "repository_path": str(repository),
        "repository_sha256": source_digest,
        "weights_path": str(weights_path),
        "weights_sha256": weights_digest,
        "state_sha256": hashlib.sha256(b"").hexdigest(),
    }


def test_model_identity_changes_when_loaded_backbone_changes(tmp_path, monkeypatch):
    _, runtime = _write_package(tmp_path, version_id="backbone-identity", backbone="dinov2_vits14", hidden_dim=384)
    def load(value):
        backbone = _FakeBackbone(384)
        backbone.register_buffer("identity_weight", torch.tensor([value]))
        monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: backbone)
        model = CodexClassifier(model_dir=runtime, device="cpu", backbone_source="online")
        return model.classify(np.zeros((8, 8, 3), dtype=np.uint8))["model_provenance"]
    first, second = load(1.0), load(2.0)
    assert first["sha256"] == second["sha256"]
    assert first["backbone"]["state_sha256"] != second["backbone"]["state_sha256"]
    assert first["artifact_id"] != second["artifact_id"]

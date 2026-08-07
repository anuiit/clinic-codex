"""Shared IO, determinism, and deployed-runtime helpers for Lot A scripts.

Zero-new-data arbitrage instrumentation (docs/lot-a-plan-20260807.md).
CPU only. Read-only against backend/codex_model/ — never writes there.
"""

from __future__ import annotations

import os

# Best-effort: only takes effect if this module is the first thing in the
# process to touch torch's OpenMP runtime. Callers should import
# scripts.lot_a.common before importing torch themselves.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_ROWS_PATH = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260805-vicreg-backbone-renomination-v19-r2/iteration-0003/prediction_rows.jsonl"
)
V9_CACHES_DIR = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-self-supervised-v9/iteration-0001/caches"
)
DEPLOYED_RUNTIME_DIR = REPO_ROOT / "backend/codex_model"

DEFAULT_SEED = 20260807

EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (17, 42, 73)

PREDICTION_ROW_REQUIRED_KEYS = (
    "b0_topk",
    "c1_topk",
    "candidate_topk",
    "label",
    "row_id",
    "outer_fold",
    "seed",
    "provenance_component",
    "class_name",
    "decoded_pixel_sha256",
)
_TOPK_FIELDS = ("b0_topk", "c1_topk", "candidate_topk")
_STRING_FIELDS = ("row_id", "provenance_component", "class_name", "decoded_pixel_sha256")
_INT_FIELDS = ("label", "outer_fold", "seed")


class LotASchemaError(ValueError):
    """Raised when an input file does not match the expected Lot A schema."""


class RuntimeDriftError(RuntimeError):
    """Raised when the deployed runtime artifacts change after being loaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value under the repo's canonical JSON contract."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def set_determinism(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG used by Lot A scripts and force deterministic kernels.

    CPU-only: does not touch CUDA state. OMP_NUM_THREADS is set at import
    time above (best effort); torch's own thread pool is pinned here too as
    a defensive measure independent of the environment variable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _validate_topk(row: dict, field: str, row_index: int) -> None:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise LotASchemaError(
            f"row {row_index}: field {field!r} must be a list of exactly 3 ints, got {value!r}"
        )


def _validate_prediction_row(row: Any, row_index: int) -> None:
    if not isinstance(row, dict):
        raise LotASchemaError(
            f"row {row_index}: expected a JSON object, got {type(row).__name__}"
        )
    missing = [key for key in PREDICTION_ROW_REQUIRED_KEYS if key not in row]
    if missing:
        raise LotASchemaError(f"row {row_index}: missing required keys: {sorted(missing)}")
    for field in _TOPK_FIELDS:
        _validate_topk(row, field, row_index)
    for field in _INT_FIELDS:
        if isinstance(row[field], bool) or not isinstance(row[field], int):
            raise LotASchemaError(f"row {row_index}: {field!r} must be an int")
    for field in _STRING_FIELDS:
        if not isinstance(row[field], str) or not row[field]:
            raise LotASchemaError(f"row {row_index}: {field!r} must be a non-empty string")


def load_prediction_rows(path: Path) -> list[dict]:
    """Load and schema-validate the prediction_rows.jsonl records.

    Raises LotASchemaError on the first row that violates the contract
    (missing key, wrong type, or a topk field that is not a 3-int list).
    """
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LotASchemaError(f"row {index}: invalid JSON: {exc}") from exc
            _validate_prediction_row(row, index)
            rows.append(row)
    return rows


def verify_cache_sha(cache_path: Path) -> tuple[bool, str, str]:
    """Compare a fold cache's sha256 against its `<name>.pt.prov.json` sidecar.

    Returns (matches, actual_sha256, expected_sha256). The sidecar field is
    `cache_sha256` (confirmed against runs/20260803-self-supervised-v9's
    fold cache sidecars).
    """
    cache_path = Path(cache_path)
    sidecar_path = cache_path.with_name(cache_path.name + ".prov.json")
    if not sidecar_path.is_file():
        raise LotASchemaError(f"missing provenance sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = sidecar.get("cache_sha256")
    if not isinstance(expected, str) or not expected:
        raise LotASchemaError(f"sidecar missing 'cache_sha256' field: {sidecar_path}")
    actual = sha256_file(cache_path)
    return actual == expected, actual, expected


def load_fold_cache(path: Path) -> dict:
    """Load a v9 fold cache dict (torch.load, map_location=cpu).

    Fold caches are plain python dicts of tensors/lists (not weights-only
    tensor state), so weights_only=False is required — this repo trusts its
    own research caches, it never deserializes untrusted third-party files.
    """
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Deployed runtime (read-only)
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Mirrors backend/codex_model/classifier.py::_ProjectionHead (and the
    identical backend/codex_pipeline/models/projection_head.py::ProjectionHead
    used by the v9/v19r2 research checkpoints).

    Not imported directly from backend.codex_model: importing that package's
    CodexClassifier pulls in torch.hub.load of the DINOv2 backbone (network
    access, heavyweight, irrelevant here since Lot A only ever consumes
    already-extracted 384-d backbone features from caches). The architecture
    below is copied verbatim: Linear(hidden,hidden) -> GELU -> Dropout(0.1)
    -> Linear(hidden,embedding), L2-normalized output. Shared by
    DeployedRuntime (runtime weights) and preflight.py's calibration check
    (research checkpoints — same architecture, different weights).
    """

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class DeployedRuntime:
    """Read-only wrapper around the deployed backend/codex_model artifacts.

    Loads config.json + weights/projection.pt + weights/prototypes.pt,
    records the sha256 of the three files at load time, and can assert
    later that they are unchanged. Never writes to runtime_dir.
    """

    def __init__(self, runtime_dir: Path = DEPLOYED_RUNTIME_DIR):
        self.runtime_dir = Path(runtime_dir)
        self.config_path = self.runtime_dir / "config.json"
        self.projection_path = self.runtime_dir / "weights" / "projection.pt"
        self.prototypes_path = self.runtime_dir / "weights" / "prototypes.pt"

        self.sha256_at_load = self._current_sha256()

        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        hidden_dim = int(self.config["hidden_dim"])
        embedding_dim = int(self.config["embedding_dim"])

        proj_state = torch.load(self.projection_path, map_location="cpu", weights_only=True)
        self._projection = ProjectionHead(hidden_dim, embedding_dim)
        self._projection.load_state_dict(proj_state, strict=True)
        self._projection.eval()

        proto_data = torch.load(self.prototypes_path, map_location="cpu", weights_only=True)
        prototypes = proto_data["prototypes"]
        if not torch.isfinite(prototypes).all():
            raise LotASchemaError("deployed prototypes contain non-finite values")
        self._prototypes = prototypes
        self._class_labels = [int(value) for value in proto_data["class_labels"].tolist()]

    def _current_sha256(self) -> dict[str, str]:
        return {
            "config": sha256_file(self.config_path),
            "projection": sha256_file(self.projection_path),
            "prototypes": sha256_file(self.prototypes_path),
        }

    def assert_unchanged(self) -> None:
        """Re-hash the three runtime artifacts and raise if any drifted."""
        current = self._current_sha256()
        drifted = {
            name: {"loaded": self.sha256_at_load[name], "now": current[name]}
            for name in current
            if current[name] != self.sha256_at_load[name]
        }
        if drifted:
            raise RuntimeDriftError(f"deployed runtime artifacts changed: {drifted}")

    @torch.no_grad()
    def predict_topk(self, features_384: torch.Tensor, k: int = 3) -> list[list[int]]:
        """projection head forward -> L2-normalize -> cosine vs prototypes -> topk labels.

        Returns, for each input row, the top-k class labels (mapped through
        the prototype store's class_labels, not raw column indices).
        """
        features = features_384.to(torch.float32)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        embeddings = self._projection(features)  # already L2-normalized
        similarities = embeddings @ self._prototypes.to(torch.float32).t()
        k = min(k, similarities.shape[1])
        top_indices = similarities.topk(k, dim=1).indices
        return [[self._class_labels[idx] for idx in row] for row in top_indices.tolist()]

"""Synthetic-fixture tests for scripts/lot_a/common.py.

No test in this module reads the real .omc/ caches or the real
backend/codex_model weights — everything here is built from scratch in a
tmp_path fixture.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn.functional as F

from scripts.lot_a import common

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _good_row(**overrides) -> dict:
    row = {
        "b0_topk": [1, 2, 3],
        "c1_topk": [4, 5, 6],
        "candidate_topk": [7, 8, 9],
        "label": 1,
        "row_id": "external:0",
        "outer_fold": 1,
        "seed": 17,
        "provenance_component": "component:abc",
        "class_name": "cacahuatl",
        "decoded_pixel_sha256": "0" * 64,
    }
    row.update(overrides)
    return row


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_prediction_rows_accepts_a_good_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_good_row()])
    rows = common.load_prediction_rows(path)
    assert len(rows) == 1
    assert rows[0]["row_id"] == "external:0"


def test_load_prediction_rows_rejects_bad_topk_length(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_good_row(b0_topk=[1, 2])])
    with pytest.raises(common.LotASchemaError, match="b0_topk"):
        common.load_prediction_rows(path)


def test_load_prediction_rows_rejects_missing_key(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = _good_row()
    del row["label"]
    _write_jsonl(path, [row])
    with pytest.raises(common.LotASchemaError, match="missing required keys"):
        common.load_prediction_rows(path)


def test_load_prediction_rows_rejects_non_int_topk_entries(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_good_row(c1_topk=[1, 2, "3"])])
    with pytest.raises(common.LotASchemaError, match="c1_topk"):
        common.load_prediction_rows(path)


# ---------------------------------------------------------------------------
# sha256 sidecar verification
# ---------------------------------------------------------------------------


def test_verify_cache_sha_matches(tmp_path):
    cache_path = tmp_path / "fold-01-views08.pt"
    cache_path.write_bytes(b"synthetic cache payload")
    expected = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "fold-01-views08.pt.prov.json"
    sidecar_path.write_text(json.dumps({"cache_sha256": expected}), encoding="utf-8")

    matches, actual, expected_out = common.verify_cache_sha(cache_path)
    assert matches is True
    assert actual == expected
    assert expected_out == expected


def test_verify_cache_sha_detects_mismatch(tmp_path):
    cache_path = tmp_path / "fold-01-views08.pt"
    cache_path.write_bytes(b"synthetic cache payload")
    sidecar_path = tmp_path / "fold-01-views08.pt.prov.json"
    sidecar_path.write_text(json.dumps({"cache_sha256": "0" * 64}), encoding="utf-8")

    matches, actual, expected_out = common.verify_cache_sha(cache_path)
    assert matches is False
    assert actual != expected_out


def test_verify_cache_sha_missing_sidecar_raises(tmp_path):
    cache_path = tmp_path / "fold-01-views08.pt"
    cache_path.write_bytes(b"synthetic cache payload")
    with pytest.raises(common.LotASchemaError, match="missing provenance sidecar"):
        common.verify_cache_sha(cache_path)


# ---------------------------------------------------------------------------
# DeployedRuntime.predict_topk on a synthetic tiny runtime
# ---------------------------------------------------------------------------


def _build_synthetic_runtime(tmp_path):
    """A 4-d runtime with an identity projection head (both linear layers
    are the identity matrix, zero bias) and 3 orthogonal-ish prototypes
    chosen so cosine similarity has a strict, hand-verifiable ranking with
    no ties.
    """
    runtime_dir = tmp_path / "runtime"
    weights_dir = runtime_dir / "weights"
    weights_dir.mkdir(parents=True)

    hidden_dim = embedding_dim = 4
    config = {
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "backbone": "dinov2_vits14",
    }
    (runtime_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    identity = torch.eye(hidden_dim)
    zero_bias = torch.zeros(hidden_dim)
    projection_state = {
        "net.0.weight": identity.clone(),
        "net.0.bias": zero_bias.clone(),
        "net.3.weight": identity.clone(),
        "net.3.bias": zero_bias.clone(),
    }
    torch.save(projection_state, weights_dir / "projection.pt")

    # p0 is exactly aligned with the (post-GELU, normalized) embedding;
    # p1 partially aligned; p2 orthogonal. Strict ranking: p0 > p1 > p2.
    prototypes_raw = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    prototypes = F.normalize(prototypes_raw, p=2, dim=1)
    class_labels = torch.tensor([5, 6, 7], dtype=torch.int64)
    proto_data = {
        "prototypes": prototypes,
        "class_names": {5: "p0", 6: "p1", 7: "p2"},
        "class_labels": class_labels,
        "embedding_dim": embedding_dim,
    }
    torch.save(proto_data, weights_dir / "prototypes.pt")

    return runtime_dir


def test_deployed_runtime_predict_topk_matches_analytical_ranking(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    runtime = common.DeployedRuntime(runtime_dir)

    # Large-magnitude input: GELU(20) ~= 20, GELU(-20) ~= 0 to float32
    # precision, so the identity net maps this to ~[20, 20, 0, 0] before
    # the final L2-normalize, i.e. direction [1/sqrt2, 1/sqrt2, 0, 0] —
    # exactly aligned with prototype p0 (label 5), 0.707-cosine with p1
    # (label 6), and orthogonal to p2 (label 7).
    features = torch.tensor([[20.0, 20.0, -20.0, -20.0]])

    topk = runtime.predict_topk(features, k=3)

    assert topk == [[5, 6, 7]]


def test_deployed_runtime_assert_unchanged_passes_when_untouched(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    runtime = common.DeployedRuntime(runtime_dir)
    runtime.assert_unchanged()  # must not raise


def test_deployed_runtime_assert_unchanged_detects_drift(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    runtime = common.DeployedRuntime(runtime_dir)

    (runtime_dir / "weights" / "projection.pt").write_bytes(b"tampered")

    with pytest.raises(common.RuntimeDriftError):
        runtime.assert_unchanged()


# ---------------------------------------------------------------------------
# set_determinism
# ---------------------------------------------------------------------------


def test_set_determinism_is_repeatable(tmp_path):
    common.set_determinism(1234)
    first = torch.rand(4)
    common.set_determinism(1234)
    second = torch.rand(4)
    assert torch.equal(first, second)

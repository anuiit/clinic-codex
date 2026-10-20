"""Synthetic-fixture tests for scripts/lot_a/a3_neardup.py.

No test in this module reads the real .omc/ caches, the real
prediction_rows.jsonl, or the real backend/codex_model weights — every
fixture (full-data cache, v9 fold caches, prediction rows, and the
deployed-runtime directory) is built from scratch in a tmp_path fixture.

Cosine-similarity fixtures use an orthonormal-basis construction so exact
values are hand-verifiable: for unit vectors e1=[1,0,0,0], e2=[0,1,0,0],
v(theta) = cos(theta)*e1 + sin(theta)*e2 satisfies dot(e1, v(theta)) ==
cos(theta) exactly (both vectors already unit norm), independent of any
L2-normalization the code under test applies.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.lot_a import a3_neardup as a3
from scripts.lot_a import common

# ---------------------------------------------------------------------------
# Shared corpus fixture: 6 rows, one planted near-dup pair at cos=0.985 and
# one planted exact-dup pair (shared sha, cross-component, cross-fold).
# ---------------------------------------------------------------------------

TARGET_COS = 0.985
_THETA = math.acos(TARGET_COS)

ROW_IDS = ["external:A", "external:B", "external:C", "legacy:D", "legacy:E", "external:F"]
COMPONENTS = [
    "component:1",
    "component:2",
    "component:1",
    "component:3",
    "component:4",
    "component:1",
]
FOLDS = [1, 1, 2, 3, 4, 2]
SHAS = ["sha_A", "sha_B", "sha_C", "sha_D_dup", "sha_D_dup", "sha_F"]


def _corpus_features() -> torch.Tensor:
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    e4 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    v_b = torch.tensor([math.cos(_THETA), math.sin(_THETA), 0.0, 0.0])
    return torch.stack([e1, v_b, e3, e4, -e4, -e1])


# ---------------------------------------------------------------------------
# Item 1 — exact duplicates
# ---------------------------------------------------------------------------


def test_exact_duplicate_clusters_detects_planted_pair():
    result = a3.exact_duplicate_clusters(ROW_IDS, SHAS, COMPONENTS, FOLDS, example_cap=50)

    assert result["n_clusters_size_ge_2"] == 1
    assert result["cluster_size_distribution"] == {"2": 1}
    assert result["total_rows_in_duplicate_clusters"] == 2


def test_exact_duplicate_clusters_flags_cross_component_and_cross_fold():
    result = a3.exact_duplicate_clusters(ROW_IDS, SHAS, COMPONENTS, FOLDS, example_cap=50)

    assert result["n_cross_component_pairs"] == 1
    assert result["n_cross_fold_pairs"] == 1
    pair = result["cross_component_pair_examples"][0]
    assert {pair["row_id_a"], pair["row_id_b"]} == {"legacy:D", "legacy:E"}


def test_exact_duplicate_clusters_no_duplicates_yields_empty():
    result = a3.exact_duplicate_clusters(
        ["a", "b", "c"], ["s1", "s2", "s3"], ["c1", "c2", "c3"], [1, 2, 3], example_cap=50
    )
    assert result["n_clusters_size_ge_2"] == 0
    assert result["total_rows_in_duplicate_clusters"] == 0
    assert result["n_cross_component_pairs"] == 0


def test_exact_duplicate_clusters_handles_missing_fold_info():
    result = a3.exact_duplicate_clusters(
        ["a", "b"], ["dup", "dup"], ["c1", "c2"], None, example_cap=50
    )
    assert result["n_clusters_size_ge_2"] == 1
    assert result["n_cross_component_pairs"] == 1
    assert result["n_cross_fold_pairs"] is None
    assert result["cross_fold_pair_examples"] is None


def test_exact_duplicates_fallback_from_prediction_rows_dedupes_by_row_id():
    rows = []
    for seed in (17, 42):
        rows.append(
            {
                "row_id": "legacy:D",
                "seed": seed,
                "decoded_pixel_sha256": "sha_D_dup",
                "provenance_component": "component:3",
                "outer_fold": 3,
            }
        )
        rows.append(
            {
                "row_id": "legacy:E",
                "seed": seed,
                "decoded_pixel_sha256": "sha_D_dup",
                "provenance_component": "component:4",
                "outer_fold": 4,
            }
        )
    result = a3.exact_duplicates_fallback_from_prediction_rows(rows, example_cap=50)
    assert result["n_rows_analyzed"] == 2  # deduped from 4 to 2 unique row_ids
    assert result["n_clusters_size_ge_2"] == 1
    assert result["n_cross_component_pairs"] == 1


# ---------------------------------------------------------------------------
# Item 2 — cosine-NN near-duplicates + sweep + knee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [2, 3, 64])
def test_near_duplicate_sweep_band_boundary(chunk_size):
    features = _corpus_features()
    result = a3.near_duplicate_analysis(
        features, ROW_IDS, COMPONENTS, FOLDS, a3.SWEEP_THRESHOLDS, chunk_size, example_cap=50
    )

    sweep = result["sweep"]
    # Planted pair at cos=0.985: present at 0.95 and 0.97, absent at 0.99/0.999.
    assert sweep["0.95"]["n_pairs"] == 1
    assert sweep["0.97"]["n_pairs"] == 1
    assert sweep["0.99"]["n_pairs"] == 0
    assert sweep["0.999"]["n_pairs"] == 0


def test_near_duplicate_sweep_flags_cross_component_pair():
    features = _corpus_features()
    result = a3.near_duplicate_analysis(
        features, ROW_IDS, COMPONENTS, FOLDS, a3.SWEEP_THRESHOLDS, chunk_size=4, example_cap=50
    )

    entry = result["sweep"]["0.95"]
    assert entry["n_cross_component_pairs"] == 1
    example = entry["example_pairs"][0]
    assert {example["row_id_a"], example["row_id_b"]} == {"external:A", "external:B"}
    assert example["cross_component"] is True


def test_near_duplicate_pairs_counted_once_not_twice():
    """Each unordered pair must be counted exactly once (upper triangle),
    never once per direction."""
    features = _corpus_features()
    result = a3.near_duplicate_analysis(
        features, ROW_IDS, COMPONENTS, FOLDS, a3.SWEEP_THRESHOLDS, chunk_size=1, example_cap=50
    )
    assert result["sweep"]["0.95"]["n_pairs"] == 1


def test_near_duplicate_analysis_deterministic_across_calls():
    features = _corpus_features()
    r1 = a3.near_duplicate_analysis(
        features, ROW_IDS, COMPONENTS, FOLDS, a3.SWEEP_THRESHOLDS, chunk_size=2, example_cap=50
    )
    r2 = a3.near_duplicate_analysis(
        features, ROW_IDS, COMPONENTS, FOLDS, a3.SWEEP_THRESHOLDS, chunk_size=2, example_cap=50
    )
    r1.pop("wall_time_seconds")
    r2.pop("wall_time_seconds")
    assert common.canonical_json_dumps(a3._sanitize_floats(r1)) == common.canonical_json_dumps(
        a3._sanitize_floats(r2)
    )


def test_knee_detection_finds_sharp_break():
    # A curve that is flat near 1.0 then drops sharply: the knee should sit
    # near the drop, not at either endpoint.
    curve = np.array([0.99, 0.98, 0.97, 0.96, 0.10, 0.05, 0.02])
    result = a3.knee_detection(curve)
    assert result["available"] is True
    assert result["auxiliary_only"] is True
    assert 2 <= result["knee_rank_index"] <= 4


def test_knee_detection_unavailable_for_tiny_input():
    result = a3.knee_detection(np.array([0.9, 0.8]))
    assert result["available"] is False


# ---------------------------------------------------------------------------
# Item 3 — OOF-vs-corpus probe
# ---------------------------------------------------------------------------


def test_oof_vs_corpus_probe_finds_planted_neighbor_and_excludes_self():
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    v_b = torch.tensor([math.cos(_THETA), math.sin(_THETA), 0.0, 0.0])

    corpus_row_ids = ["external:100", "external:200", "external:300"]
    corpus_features = torch.stack([e1, v_b, e3])
    corpus_components = ["component:X", "component:Y", "component:Z"]
    corpus_folds = [1, 2, 3]

    unique_meta = {
        "external:100": {
            "label": 0,
            "component": "component:X",
            "fold": 1,
            "class_name": "foo",
        }
    }
    oof_features_by_row = {"external:100": e1.clone()}

    result = a3.oof_vs_corpus_probe(
        unique_meta,
        oof_features_by_row,
        corpus_row_ids,
        corpus_features,
        corpus_components,
        corpus_folds,
        a3.SWEEP_THRESHOLDS,
    )

    assert result["status"] == "ok"
    assert result["n_missing_from_corpus"] == 0
    row = result["per_row"][0]
    # Without self-exclusion the argmax would trivially pick "external:100"
    # itself (cos=1.0). The exclusion rule must route it to the planted
    # neighbor instead.
    assert row["nn_row_id"] == "external:200"
    assert row["nn_similarity"] == pytest.approx(TARGET_COS, abs=1e-4)
    assert row["same_component_as_query"] is False


def test_oof_vs_corpus_probe_reports_missing_rows_as_partial():
    corpus_row_ids = ["external:100"]
    corpus_features = torch.tensor([[1.0, 0.0]])
    corpus_components = ["component:X"]
    corpus_folds = [1]

    unique_meta = {
        "external:100": {"label": 0, "component": "component:X", "fold": 1, "class_name": "foo"},
        "external:999": {"label": 1, "component": "component:Y", "fold": 2, "class_name": "bar"},
    }
    oof_features_by_row = {
        "external:100": torch.tensor([1.0, 0.0]),
        "external:999": torch.tensor([0.0, 1.0]),
    }

    result = a3.oof_vs_corpus_probe(
        unique_meta,
        oof_features_by_row,
        corpus_row_ids,
        corpus_features,
        corpus_components,
        corpus_folds,
        a3.SWEEP_THRESHOLDS,
    )

    assert result["status"] == "partial"
    assert result["n_missing_from_corpus"] == 1
    assert result["missing_row_ids"] == ["external:999"]


def test_oof_vs_corpus_probe_stratifies_by_prefix():
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0])

    corpus_row_ids = ["external:1", "external:2", "legacy:1", "legacy:2"]
    corpus_features = torch.stack([e1, e2, e3, -e3])
    corpus_components = ["c1", "c2", "c3", "c4"]
    corpus_folds = [1, 1, 2, 2]

    unique_meta = {
        "external:1": {"label": 0, "component": "c1", "fold": 1, "class_name": "x"},
        "legacy:1": {"label": 0, "component": "c3", "fold": 2, "class_name": "x"},
    }
    oof_features_by_row = {"external:1": e1.clone(), "legacy:1": e3.clone()}

    result = a3.oof_vs_corpus_probe(
        unique_meta,
        oof_features_by_row,
        corpus_row_ids,
        corpus_features,
        corpus_components,
        corpus_folds,
        a3.SWEEP_THRESHOLDS,
    )

    strat = result["stratified_by_prefix"]
    assert set(strat) == {"external:", "legacy:"}
    assert strat["external:"]["n"] == 1
    assert strat["legacy:"]["n"] == 1


# ---------------------------------------------------------------------------
# Item 4 — provenance census
# ---------------------------------------------------------------------------


def test_provenance_census_jaccard_on_hand_built_sets():
    result = a3.provenance_census(
        COMPONENTS, FOLDS, instrument_components=["component:1", "component:2", "component:9"]
    )
    instrument = result["instrument_comparison"]
    # corpus components = {1,2,3,4}; instrument = {1,2,9}
    # intersection = {1,2} (2), union = {1,2,3,4,9} (5) -> jaccard = 0.4
    assert instrument["status"] == "ok"
    assert instrument["n_instrument_components_in_corpus"] == 2
    assert instrument["n_instrument_components_not_in_corpus"] == 1
    assert instrument["jaccard_instrument_vs_corpus"] == pytest.approx(0.4)


def test_provenance_census_flags_components_split_across_folds():
    result = a3.provenance_census(COMPONENTS, FOLDS, instrument_components=None)
    # component:1 appears at folds {1, 2} (rows A@fold1, C@fold2, F@fold2) -> split.
    assert result["components_split_across_folds"] == {"component:1": [1, 2]}
    assert result["n_components_split_across_folds"] == 1
    assert result["instrument_comparison"]["status"] == "skipped"


def test_provenance_census_size_distribution():
    result = a3.provenance_census(COMPONENTS, FOLDS, instrument_components=None)
    # component:1 has 3 rows (A, C, F); others have 1 each.
    summary = result["component_size_distribution_summary"]
    assert summary["max"] == 3
    assert summary["min"] == 1
    assert result["full_corpus_component_count"] == 4


# ---------------------------------------------------------------------------
# Full report assembly (IO) — deployed-runtime fixture, full-data cache,
# v9 fold caches, prediction rows.
# ---------------------------------------------------------------------------


def _build_synthetic_runtime(tmp_path, hidden_dim: int = 4):
    runtime_dir = tmp_path / "runtime"
    weights_dir = runtime_dir / "weights"
    weights_dir.mkdir(parents=True)

    config = {"hidden_dim": hidden_dim, "embedding_dim": hidden_dim, "backbone": "dinov2_vits14"}
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

    prototypes = F.normalize(torch.eye(2, hidden_dim), p=2, dim=1)
    proto_data = {
        "prototypes": prototypes,
        "class_names": {0: "p0", 1: "p1"},
        "class_labels": torch.tensor([0, 1], dtype=torch.int64),
        "embedding_dim": hidden_dim,
    }
    torch.save(proto_data, weights_dir / "prototypes.pt")
    return runtime_dir


def _write_full_data_cache(path, row_ids, shas, components, folds, features):
    payload = {
        "row_id": row_ids,
        "decoded_pixel_sha256": shas,
        "component_id": components,
        "fold": folds,
        "features": features,
        "class_label": [0] * len(row_ids),
        "class_name": ["dummy"] * len(row_ids),
    }
    torch.save(payload, path)
    sidecar = path.with_name(path.name + ".prov.json")
    sidecar.write_text(
        json.dumps({"cache_sha256": common.sha256_file(path), "row_count": len(row_ids)}),
        encoding="utf-8",
    )


def _write_prediction_row(row_id, fold, component, class_name="dummy", seed=17):
    return {
        "b0_topk": [0, 1, 2],
        "c1_topk": [0, 1, 2],
        "candidate_topk": [0, 1, 2],
        "label": 0,
        "row_id": row_id,
        "outer_fold": fold,
        "seed": seed,
        "provenance_component": component,
        "class_name": class_name,
        "decoded_pixel_sha256": "0" * 64,
    }


def _write_prediction_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_fold_cache(path, oof_row_ids, oof_components, oof_folds, features):
    payload = {
        "train": {"base_features": torch.zeros((1, features.shape[1])), "class_label": [0]},
        "oof": {
            "base_features": features,
            "row_id": oof_row_ids,
            "class_label": [0] * len(oof_row_ids),
            "class_name": ["dummy"] * len(oof_row_ids),
            "component_id": oof_components,
            "decoded_pixel_sha256": ["0" * 64] * len(oof_row_ids),
            "fold": oof_folds,
        },
    }
    torch.save(payload, path)
    sidecar = path.with_name(path.name + ".prov.json")
    sidecar.write_text(
        json.dumps({"cache_sha256": common.sha256_file(path), "oof_rows": len(oof_row_ids)}),
        encoding="utf-8",
    )


def _build_full_synthetic_environment(tmp_path):
    """4 rows split across 2 folds, identical universe in the full-data
    corpus and in the v9 fold caches (so the OOF-vs-corpus probe reaches
    status 'ok' with zero missing rows)."""
    runtime_dir = _build_synthetic_runtime(tmp_path)

    row_ids = ["external:1", "external:2", "legacy:1", "legacy:2"]
    components = ["component:A", "component:B", "component:A", "component:C"]
    folds = [1, 1, 2, 2]
    shas = ["s1", "s2", "s3", "s4"]
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    full_data_path = tmp_path / "full-data-views08.pt"
    _write_full_data_cache(full_data_path, row_ids, shas, components, folds, features)

    prediction_rows_path = tmp_path / "prediction_rows.jsonl"
    _write_prediction_rows(
        prediction_rows_path,
        [
            _write_prediction_row("external:1", 1, "component:A"),
            _write_prediction_row("external:2", 1, "component:B"),
            _write_prediction_row("legacy:1", 2, "component:A"),
            _write_prediction_row("legacy:2", 2, "component:C"),
        ],
    )

    caches_dir = tmp_path / "caches"
    caches_dir.mkdir()
    _write_fold_cache(
        caches_dir / "fold-01-views08.pt",
        ["external:1", "external:2"],
        ["component:A", "component:B"],
        [1, 1],
        features[:2],
    )
    _write_fold_cache(
        caches_dir / "fold-02-views08.pt",
        ["legacy:1", "legacy:2"],
        ["component:A", "component:C"],
        [2, 2],
        features[2:],
    )
    for fold in (3, 4, 5):
        _write_fold_cache(
            caches_dir / f"fold-{fold:02d}-views08.pt",
            [],
            [],
            [],
            torch.zeros((0, 4)),
        )

    return {
        "runtime_dir": runtime_dir,
        "full_data_path": full_data_path,
        "prediction_rows_path": prediction_rows_path,
        "caches_dir": caches_dir,
    }


def test_build_core_report_ok_path_runs_all_four_items(tmp_path):
    env = _build_full_synthetic_environment(tmp_path)
    report = a3.build_core_report(
        env["full_data_path"],
        env["prediction_rows_path"],
        env["caches_dir"],
        env["runtime_dir"],
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    assert report["status"] == "ok"
    assert report["exact_duplicates"]["n_rows_analyzed"] == 4
    assert report["near_duplicates"]["status"] == "ok"
    assert report["oof_vs_corpus_probe"]["status"] == "ok"
    assert report["oof_vs_corpus_probe"]["n_missing_from_corpus"] == 0
    assert report["provenance_census"]["instrument_comparison"]["status"] == "ok"
    assert report["runtime_guard"]["verified_unchanged_before_analysis"] is True
    assert report["runtime_guard"]["verified_unchanged_after_analysis"] is True


def test_build_core_report_deterministic_double_run(tmp_path):
    env = _build_full_synthetic_environment(tmp_path)
    r1 = a3.build_core_report(
        env["full_data_path"],
        env["prediction_rows_path"],
        env["caches_dir"],
        env["runtime_dir"],
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    r2 = a3.build_core_report(
        env["full_data_path"],
        env["prediction_rows_path"],
        env["caches_dir"],
        env["runtime_dir"],
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    # Strip the only non-deterministic field (wall clock) before hashing.
    r1["near_duplicates"].pop("wall_time_seconds", None)
    r2["near_duplicates"].pop("wall_time_seconds", None)
    r1["oof_vs_corpus_probe"].pop("wall_time_seconds", None)
    r2["oof_vs_corpus_probe"].pop("wall_time_seconds", None)
    assert common.sha256_json(r1) == common.sha256_json(r2)


def test_build_core_report_missing_full_data_cache_falls_back_to_exact_dup_only(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    prediction_rows_path = tmp_path / "prediction_rows.jsonl"
    _write_prediction_rows(
        prediction_rows_path,
        [
            _write_prediction_row("legacy:D", 3, "component:3"),
            _write_prediction_row("legacy:E", 4, "component:4"),
        ],
    )
    # Both rows share the same decoded_pixel_sha256 override below.
    with prediction_rows_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    for line in lines:
        line["decoded_pixel_sha256"] = "sha_D_dup"
    _write_prediction_rows(prediction_rows_path, lines)

    missing_cache_path = tmp_path / "does-not-exist.pt"

    report = a3.build_core_report(
        missing_cache_path,
        prediction_rows_path,
        tmp_path / "caches",
        runtime_dir,
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )

    assert report["status"] == "partial"
    assert report["exact_duplicates"]["n_clusters_size_ge_2"] == 1
    assert report["near_duplicates"]["status"] == "skipped"
    assert report["oof_vs_corpus_probe"]["status"] == "skipped"
    assert report["provenance_census"]["status"] == "skipped"


def test_build_core_report_missing_everything_is_blocked(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    report = a3.build_core_report(
        tmp_path / "no-cache.pt",
        tmp_path / "no-rows.jsonl",
        tmp_path / "no-caches",
        runtime_dir,
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    assert report["status"] == "blocked"
    assert report["exact_duplicates"] is None


def test_build_core_report_oof_universe_unavailable_but_corpus_present(tmp_path):
    env = _build_full_synthetic_environment(tmp_path)
    report = a3.build_core_report(
        env["full_data_path"],
        tmp_path / "no-prediction-rows.jsonl",  # missing -> OOF universe unavailable
        env["caches_dir"],
        env["runtime_dir"],
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    assert report["status"] == "ok"
    assert report["exact_duplicates"]["n_rows_analyzed"] == 4
    assert report["near_duplicates"]["status"] == "ok"
    assert report["oof_vs_corpus_probe"]["status"] == "skipped"
    assert report["provenance_census"]["instrument_comparison"]["status"] == "skipped"


def test_build_core_report_runtime_drift_raises(tmp_path):
    env = _build_full_synthetic_environment(tmp_path)

    # Corrupt the runtime weights mid-flight by monkeypatching assert_unchanged
    # is undesirable (tests should exercise real behavior) — instead, verify
    # the guard by tampering the file directly and checking a manual
    # DeployedRuntime + assert_unchanged sequence detects it, mirroring the
    # same code path build_core_report relies on.
    runtime = common.DeployedRuntime(env["runtime_dir"])
    (env["runtime_dir"] / "weights" / "projection.pt").write_bytes(b"tampered")
    with pytest.raises(common.RuntimeDriftError):
        runtime.assert_unchanged()


# ---------------------------------------------------------------------------
# Markdown rendering smoke test
# ---------------------------------------------------------------------------


def test_render_markdown_ok_report_smoke(tmp_path):
    env = _build_full_synthetic_environment(tmp_path)
    report = a3.build_core_report(
        env["full_data_path"],
        env["prediction_rows_path"],
        env["caches_dir"],
        env["runtime_dir"],
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    report["reproducibility"] = {"hash_run_1": "abc", "hash_run_2": "abc", "matched": True}
    markdown = a3.render_markdown(report)
    assert "Phase A3" in markdown
    assert "Doublons exacts" in markdown
    assert "Census de provenance" in markdown


def test_render_markdown_blocked_report_smoke(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    report = a3.build_core_report(
        tmp_path / "no-cache.pt",
        tmp_path / "no-rows.jsonl",
        tmp_path / "no-caches",
        runtime_dir,
        seed=1234,
        chunk_size=2,
        example_cap=10,
    )
    markdown = a3.render_markdown(report)
    assert "BLOCKED" in markdown or "blocked" in markdown.lower()

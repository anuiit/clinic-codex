"""Synthetic-fixture tests for scripts/lot_a/a2_bootstrap.py.

No test in this module reads the real .omc/ caches, the real
prediction_rows.jsonl, or the real backend/codex_model weights — every
fixture (rows, fold caches, and the deployed-runtime directory) is built
from scratch in a tmp_path fixture. The coverage-simulation test is a pure
numpy Monte-Carlo fixture with a planted effect/ICC, unrelated to any real
data.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.lot_a import a2_bootstrap as a2
from scripts.lot_a import common

SEEDS = (17, 42, 73)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _row(
    row_id: str,
    seed: int,
    label: int,
    b0_topk: list[int],
    c1_topk: list[int],
    candidate_topk: list[int],
    outer_fold: int = 1,
    provenance_component: str = "component:default",
    class_name: str = "cacahuatl",
) -> dict:
    return {
        "row_id": row_id,
        "seed": seed,
        "label": label,
        "b0_topk": b0_topk,
        "c1_topk": c1_topk,
        "candidate_topk": candidate_topk,
        "outer_fold": outer_fold,
        "provenance_component": provenance_component,
        "class_name": class_name,
        "decoded_pixel_sha256": "0" * 64,
    }


CORRECT = [1, 2, 3]  # topk[0] == label(1) -> correct
WRONG = [2, 1, 3]  # topk[0] == 2 != label(1) -> wrong


def _rows_for_all_seeds(row_specs: list[dict]) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for spec in row_specs:
            rows.append(_row(seed=seed, **spec))
    return rows


def _build_synthetic_runtime(tmp_path, hidden_dim: int = 4):
    """Identity-projection runtime with 2 orthogonal prototypes, labels
    0 and 1, so predict_topk(k=1) has a hand-verifiable ranking."""
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


def _write_fold_cache(path, oof_row_ids, oof_labels, oof_components, features, folds):
    payload = {
        "train": {"base_features": torch.zeros((1, features.shape[1])), "class_label": [0]},
        "oof": {
            "base_features": features,
            "row_id": oof_row_ids,
            "class_label": oof_labels,
            "class_name": ["dummy"] * len(oof_row_ids),
            "component_id": oof_components,
            "decoded_pixel_sha256": ["0" * 64] * len(oof_row_ids),
            "fold": folds,
        },
    }
    torch.save(payload, path)
    sidecar = path.with_name(path.name + ".prov.json")
    sidecar.write_text(
        json.dumps({"cache_sha256": common.sha256_file(path), "oof_rows": len(oof_row_ids)}),
        encoding="utf-8",
    )


def _build_synthetic_fold_caches(tmp_path, hidden_dim: int = 4):
    """Two folds. Fold 1 has row_id p0 (features aligned with prototype 0,
    label 0 -> deployed correct) and p1x (aligned with prototype 1 but
    labeled 0 -> deployed wrong). Fold 2 has p1 (aligned prototype 1, label
    1 -> deployed correct)."""
    caches_dir = tmp_path / "caches"
    caches_dir.mkdir(parents=True)

    big = 20.0
    aligned_p0 = torch.tensor([[big, -big, -big, -big]])
    aligned_p1 = torch.tensor([[-big, big, -big, -big]])

    _write_fold_cache(
        caches_dir / "fold-01-views08.pt",
        oof_row_ids=["external:p0_correct", "external:p1_wrong"],
        oof_labels=[0, 0],
        oof_components=["component:A", "component:A"],
        features=torch.cat([aligned_p0, aligned_p1], dim=0),
        folds=[1, 1],
    )
    _write_fold_cache(
        caches_dir / "fold-02-views08.pt",
        oof_row_ids=["external:p1_correct"],
        oof_labels=[1],
        oof_components=["component:B"],
        features=aligned_p1.clone(),
        folds=[2, 2][:1],
    )
    return caches_dir


# ---------------------------------------------------------------------------
# Normal CDF / PPF sanity (dependency-free implementation)
# ---------------------------------------------------------------------------


def test_norm_ppf_matches_known_quantiles():
    assert math.isclose(a2._norm_ppf(0.975), 1.9599639845400545, abs_tol=1e-6)
    assert math.isclose(a2._norm_ppf(0.025), -1.9599639845400545, abs_tol=1e-6)
    assert math.isclose(a2._norm_ppf(0.80), 0.8416212335729143, abs_tol=1e-6)
    assert math.isclose(a2._norm_ppf(0.5), 0.0, abs_tol=1e-9)


def test_norm_cdf_ppf_round_trip():
    for p in (0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99):
        z = a2._norm_ppf(p)
        assert math.isclose(a2._norm_cdf(z), p, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_deployed_predictions: synthetic runtime + synthetic fold caches
# ---------------------------------------------------------------------------


def test_compute_deployed_predictions_matches_hand_verified_ranking(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    caches_dir = _build_synthetic_fold_caches(tmp_path)

    rows = _rows_for_all_seeds(
        [
            dict(
                row_id="external:p0_correct", label=0, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=1, provenance_component="component:A",
            ),
            dict(
                row_id="external:p1_wrong", label=0, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=1, provenance_component="component:A",
            ),
            dict(
                row_id="external:p1_correct", label=1, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=2, provenance_component="component:B",
            ),
        ]
    )

    per_row, meta = a2.compute_deployed_predictions(rows, caches_dir, runtime_dir)

    assert per_row["external:p0_correct"]["correct_deployed"] is True
    assert per_row["external:p1_wrong"]["correct_deployed"] is False
    assert per_row["external:p1_correct"]["correct_deployed"] is True
    assert meta["n_unique_rows"] == 3
    assert meta["n_predicted"] == 3
    assert meta["per_fold_coverage"]["1"] == 2
    assert meta["per_fold_coverage"]["2"] == 1
    assert set(meta["sha256_at_load"]) == {"config", "projection", "prototypes"}


def test_compute_deployed_predictions_raises_on_fold_mismatch(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    caches_dir = _build_synthetic_fold_caches(tmp_path)

    # row_id "external:p0_correct" actually lives in fold-01's cache, but the
    # prediction_rows row claims outer_fold=2 — must raise, not silently pass.
    rows = _rows_for_all_seeds(
        [
            dict(
                row_id="external:p0_correct", label=0, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=2, provenance_component="component:A",
            ),
        ]
    )
    with pytest.raises(common.LotASchemaError, match="outer_fold"):
        a2.compute_deployed_predictions(rows, caches_dir, runtime_dir)


def test_compute_deployed_predictions_raises_on_missing_row(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    caches_dir = _build_synthetic_fold_caches(tmp_path)

    rows = _rows_for_all_seeds(
        [
            dict(
                row_id="external:not_in_any_cache", label=0, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=1, provenance_component="component:A",
            ),
        ]
    )
    with pytest.raises(common.LotASchemaError, match="not found in any v9 fold"):
        a2.compute_deployed_predictions(rows, caches_dir, runtime_dir)


# ---------------------------------------------------------------------------
# build_paired_records: alignment correctness against a hand-computed fixture
# ---------------------------------------------------------------------------


def test_build_paired_records_known_deltas():
    rows = _rows_for_all_seeds(
        [
            dict(
                row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG,
                provenance_component="component:A",
            ),
            dict(
                row_id="r2", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT,
                provenance_component="component:A",
            ),
        ]
    )
    deployed = {
        "r1": {"correct_deployed": True, "component": "component:A", "fold": 1, "class_name": "cacahuatl"},
        "r2": {"correct_deployed": False, "component": "component:A", "fold": 1, "class_name": "cacahuatl"},
    }
    records = a2.build_paired_records(rows, deployed)

    # r1: b0 always wrong -> avg_correct_b0=0, c1 always correct -> 1, r2(cand) always wrong -> 0
    # deployed correct=True -> deployed_correct=1.0
    assert records["r1"]["avg_correct_b0"] == 0.0
    assert records["r1"]["avg_correct_c1"] == 1.0
    assert records["r1"]["avg_correct_r2"] == 0.0
    assert records["r1"]["delta_b0"] == 0.0 - 1.0
    assert records["r1"]["delta_c1"] == 1.0 - 1.0
    assert records["r1"]["delta_r2"] == 0.0 - 1.0

    # r2: all arms always correct -> avg_correct=1 for all; deployed correct=False -> 0.0
    assert records["r2"]["avg_correct_b0"] == 1.0
    assert records["r2"]["delta_b0"] == 1.0 - 0.0
    assert records["r2"]["delta_c1"] == 1.0 - 0.0
    assert records["r2"]["delta_r2"] == 1.0 - 0.0

    assert records["r1"]["prefix"] == "(no-prefix)"


def test_build_paired_records_rejects_inconsistent_metadata():
    rows = [
        _row("r1", 17, 1, WRONG, CORRECT, WRONG, outer_fold=1, provenance_component="component:A"),
        _row("r1", 42, 1, WRONG, CORRECT, WRONG, outer_fold=2, provenance_component="component:A"),
    ]
    deployed = {"r1": {"correct_deployed": True, "component": "component:A", "fold": 1, "class_name": "x"}}
    with pytest.raises(common.LotASchemaError, match="inconsistent"):
        a2.build_paired_records(rows, deployed)


# ---------------------------------------------------------------------------
# Cluster bootstrap: resampling preserves whole components
# ---------------------------------------------------------------------------


def test_stat_from_draws_idx_preserves_whole_components():
    # component A: 2 rows summing to 3.0 ; component B: 1 row summing to 6.0
    comp_sum = np.array([3.0, 6.0])
    comp_n = np.array([2.0, 1.0])
    # A manually chosen draw picking component A twice and B once.
    draws_idx = np.array([[0, 0, 1]])
    result = a2.stat_from_draws_idx(comp_sum, comp_n, draws_idx)
    expected = (2 * 3.0 + 1 * 6.0) / (2 * 2.0 + 1 * 1.0)
    assert result[0] == pytest.approx(expected)


def test_joint_cluster_bootstrap_never_mixes_rows_across_components():
    # 3 components, each internally homogeneous (constant value) but distinct
    # across components. Any resample statistic must be a weighted average
    # of {1.0, 2.0, 3.0} only — never an intermediate value that couldn't
    # arise from whole-component draws (e.g. it must be achievable as a
    # rational combination of the 3 base values with integer weights).
    comp_sum = {"x": np.array([1.0 * 5, 2.0 * 5, 3.0 * 5])}  # 5 rows/component, homogeneous
    comp_n = np.array([5.0, 5.0, 5.0])
    rng = np.random.default_rng(123)
    draws_idx, draws = a2.joint_cluster_bootstrap(comp_sum, comp_n, 2000, rng)
    values = draws["x"]
    assert np.all(values >= 1.0 - 1e-9)
    assert np.all(values <= 3.0 + 1e-9)
    # every draw statistic must be a weighted mean of {1,2,3} with weights
    # that are multiples of 5/15 (since each component always contributes
    # complete 5-row blocks) -> value*3 must be very close to an integer
    # multiple of 1 across the achievable set; verify a coarser invariant:
    # rounding value to the nearest of the achievable rational grid recovers
    # the value almost exactly.
    counts_per_draw = np.stack(
        [np.sum(draws_idx == comp, axis=1) for comp in range(3)], axis=0
    )  # (3, n_draws), each column sums to K=3
    recomputed = (counts_per_draw[0] * 1.0 + counts_per_draw[1] * 2.0 + counts_per_draw[2] * 3.0) / 3.0
    assert np.allclose(values, recomputed)


def test_joint_cluster_bootstrap_shares_resample_across_arms():
    comp_sum = {"b0": np.array([1.0, 2.0, 3.0]), "c1": np.array([10.0, 20.0, 30.0])}
    comp_n = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(7)
    draws_idx, draws = a2.joint_cluster_bootstrap(comp_sum, comp_n, 500, rng)
    # c1 must always be exactly 10x b0 since both are proportional and the
    # same draws_idx is used for every arm.
    assert np.allclose(draws["c1"], draws["b0"] * 10.0)


# ---------------------------------------------------------------------------
# BCa vs percentile on a known-skew fixture
# ---------------------------------------------------------------------------


def test_bca_differs_from_percentile_on_skewed_fixture():
    # 20 components: 19 with small continuous noise around 0, one extreme
    # outlier -> right-skewed cluster-mean sampling distribution for small K.
    rng = np.random.default_rng(2026)
    k = 20
    comp_n = np.full(k, 5.0)
    comp_values = rng.normal(0.0, 1.0, k)
    comp_values[0] = 100.0  # one big outlier component
    comp_sum = comp_values * comp_n

    observed = a2.observed_stat(comp_sum, comp_n)
    jackknife = a2.leave_one_component_out_jackknife(comp_sum, comp_n)

    draws_idx = a2._resample_component_indices(k, 5000, rng)
    boot_draws = a2.stat_from_draws_idx(comp_sum, comp_n, draws_idx)

    bca = a2.bca_ci(observed, boot_draws, jackknife)
    pct = a2.percentile_ci(boot_draws)

    assert bca["lo"] != pytest.approx(pct["lo"], abs=1e-6)
    assert bca["hi"] != pytest.approx(pct["hi"], abs=1e-6)
    # the outlier drives a strongly positive acceleration/bias correction,
    # so BCa's adjusted percentiles must diverge from the raw 0.025/0.975.
    assert bca["z0"] != 0.0
    assert bca["acceleration"] != 0.0
    assert bca["alpha1_percentile"] != pytest.approx(0.025, abs=1e-6)
    assert bca["alpha2_percentile"] != pytest.approx(0.975, abs=1e-6)


def test_percentile_ci_matches_numpy_percentile_directly():
    draws = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = a2.percentile_ci(draws, alpha=0.20)
    lo, hi = np.percentile(draws, [10, 90])
    assert result["lo"] == pytest.approx(lo)
    assert result["hi"] == pytest.approx(hi)


# ---------------------------------------------------------------------------
# Max-statistic joint band is wider-or-equal to any single-arm marginal band
# ---------------------------------------------------------------------------


def test_joint_band_wider_or_equal_to_marginal():
    rng = np.random.default_rng(99)
    n_draws = 4000
    observed = {"b0": 0.02, "c1": 0.05, "r2": -0.01}
    boot_draws = {
        "b0": observed["b0"] + rng.normal(0, 0.10, n_draws),
        "c1": observed["c1"] + rng.normal(0, 0.08, n_draws) + 0.3 * rng.normal(0, 0.10, n_draws),
        "r2": observed["r2"] + rng.normal(0, 0.12, n_draws),
    }
    joint = a2.max_stat_joint_bands(observed, boot_draws, alpha=0.05)

    for arm in ("b0", "c1", "r2"):
        se = float(np.std(boot_draws[arm], ddof=1))
        marginal_t = np.abs((boot_draws[arm] - observed[arm]) / se)
        marginal_q = float(np.percentile(marginal_t, 95))
        assert joint["critical_value_q"] >= marginal_q - 1e-12
        band = joint["bands"][arm]
        marginal_half_width = marginal_q * se
        joint_half_width = (band["hi"] - band["lo"]) / 2.0
        assert joint_half_width >= marginal_half_width - 1e-9


# ---------------------------------------------------------------------------
# compute_component_stats (ICC / MDE gate) sanity
# ---------------------------------------------------------------------------


def test_compute_component_stats_zero_between_variance_gives_low_icc():
    # every component has the exact same set of values -> zero between-
    # component variance -> ICC should clip to (near) 0.
    per_component = {f"c{i}": [1.0, 2.0, 3.0, 4.0] for i in range(6)}
    stats = a2.compute_component_stats(per_component)
    assert stats["icc_clipped_0_1"] == pytest.approx(0.0, abs=1e-9)


def test_compute_component_stats_high_between_variance_gives_high_icc():
    # each component is internally constant but differs strongly across
    # components -> ICC should be close to 1.
    per_component = {f"c{i}": [float(i) * 100.0] * 5 for i in range(10)}
    stats = a2.compute_component_stats(per_component)
    assert stats["icc_clipped_0_1"] > 0.95
    assert stats["half_width_95_analytical"] > 0
    assert stats["mde_80pct_analytical"] > stats["half_width_95_analytical"]


# ---------------------------------------------------------------------------
# Coverage check (critique item 2 verification): planted-effect, planted-ICC
# Monte Carlo, empirical BCa coverage of the nominal 95% CI.
# ---------------------------------------------------------------------------


def test_bca_coverage_simulation_meets_88pct_floor():
    """200 clustered synthetic datasets (25 components x 26 rows), planted
    effect 0.05 and planted ICC 0.15, 500 bootstrap draws each. Asserts
    empirical coverage of the nominal 95% BCa CI is >= 0.88 (critique item
    2's verification) and prints the measured value."""
    n_components = 25
    rows_per_component = 26
    n_replicates = 200
    n_draws = 500
    planted_effect = 0.05
    planted_icc = 0.15
    total_variance = 0.04  # arbitrary scale in the same order as accuracy deltas

    tau2 = planted_icc * total_variance
    sigma2 = (1 - planted_icc) * total_variance

    rng = np.random.default_rng(20260807)
    comp_n = np.full(n_components, float(rows_per_component))

    covered = 0
    for _ in range(n_replicates):
        cluster_effects = rng.normal(0.0, math.sqrt(tau2), size=n_components)
        row_noise = rng.normal(0.0, math.sqrt(sigma2), size=(n_components, rows_per_component))
        values = planted_effect + cluster_effects[:, None] + row_noise  # (K, m)
        comp_sum = values.sum(axis=1)

        observed = a2.observed_stat(comp_sum, comp_n)
        jackknife = a2.leave_one_component_out_jackknife(comp_sum, comp_n)
        draws_idx = a2._resample_component_indices(n_components, n_draws, rng)
        boot_draws = a2.stat_from_draws_idx(comp_sum, comp_n, draws_idx)

        ci = a2.bca_ci(observed, boot_draws, jackknife)
        if ci["lo"] <= planted_effect <= ci["hi"]:
            covered += 1

    coverage = covered / n_replicates
    print(f"\nmeasured BCa coverage over {n_replicates} replicates: {coverage:.4f}")
    assert coverage >= 0.88, f"measured coverage {coverage:.4f} below the 0.88 floor"


# ---------------------------------------------------------------------------
# Determinism: identical fixture run twice -> identical canonical JSON bytes
# ---------------------------------------------------------------------------


def _synthetic_paired_records(n_components: int = 5, rows_per_component: int = 8) -> dict[str, dict]:
    rng = np.random.default_rng(555)
    records: dict[str, dict] = {}
    rid = 0
    for comp in range(n_components):
        for i in range(rows_per_component):
            deployed = bool(rng.random() > 0.5)
            avg_correct = {
                arm: float(min(1.0, max(0.0, rng.normal(0.5 + 0.03 * idx, 0.15))))
                for idx, arm in enumerate(a2.ARMS)
            }
            prefix = "external:" if rid % 2 == 0 else "legacy:"
            records[f"r{rid}"] = {
                "component": f"component:{comp}",
                "fold": (comp % 5) + 1,
                "prefix": prefix,
                "n_seeds": 3,
                "correct_deployed": deployed,
                **{f"avg_correct_{arm}": avg_correct[arm] for arm in a2.ARMS},
                **{
                    f"delta_{arm}": avg_correct[arm] - (1.0 if deployed else 0.0)
                    for arm in a2.ARMS
                },
            }
            rid += 1
    return records


def test_build_statistical_report_determinism_same_fixture_twice():
    paired = _synthetic_paired_records()
    stats_1, _ = a2.build_statistical_report(paired, bootstrap_seed_base=777, bootstrap_draws=300)
    stats_2, _ = a2.build_statistical_report(paired, bootstrap_seed_base=777, bootstrap_draws=300)

    assert common.canonical_json_dumps(stats_1) == common.canonical_json_dumps(stats_2)
    assert common.sha256_json(stats_1) == common.sha256_json(stats_2)


def test_build_statistical_report_different_seed_changes_bootstrap_ci():
    paired = _synthetic_paired_records()
    stats_1, _ = a2.build_statistical_report(paired, bootstrap_seed_base=1, bootstrap_draws=300)
    stats_2, _ = a2.build_statistical_report(paired, bootstrap_seed_base=2, bootstrap_draws=300)
    # observed statistics are seed-independent, but bootstrap CIs are not
    assert stats_1["observed_delta"] == stats_2["observed_delta"]
    assert stats_1["bootstrap"]["bca_headline"] != stats_2["bootstrap"]["bca_headline"]


# ---------------------------------------------------------------------------
# Full pipeline (compute_deployed_predictions + build_paired_records +
# build_statistical_report) determinism, via build_core_report on a fully
# synthetic runtime + fold caches + prediction_rows fixture.
# ---------------------------------------------------------------------------


def _full_synthetic_fixture(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    caches_dir = _build_synthetic_fold_caches(tmp_path)
    rows = _rows_for_all_seeds(
        [
            dict(
                row_id="external:p0_correct", label=0, b0_topk=WRONG, c1_topk=CORRECT,
                candidate_topk=CORRECT, outer_fold=1, provenance_component="component:A",
            ),
            dict(
                row_id="external:p1_wrong", label=0, b0_topk=CORRECT, c1_topk=WRONG,
                candidate_topk=CORRECT, outer_fold=1, provenance_component="component:A",
            ),
            dict(
                row_id="external:p1_correct", label=1, b0_topk=CORRECT, c1_topk=CORRECT,
                candidate_topk=WRONG, outer_fold=2, provenance_component="component:B",
            ),
        ]
    )
    return rows, caches_dir, runtime_dir


def test_build_core_report_determinism_same_fixture_twice(tmp_path):
    rows, caches_dir, runtime_dir = _full_synthetic_fixture(tmp_path)

    report_1, _ = a2.build_core_report(rows, caches_dir, runtime_dir, bootstrap_seed_base=333, bootstrap_draws=200)
    report_2, _ = a2.build_core_report(rows, caches_dir, runtime_dir, bootstrap_seed_base=333, bootstrap_draws=200)

    assert common.canonical_json_dumps(report_1) == common.canonical_json_dumps(report_2)
    assert common.sha256_json(report_1) == common.sha256_json(report_2)


def test_build_core_report_runtime_guard_and_coverage(tmp_path):
    rows, caches_dir, runtime_dir = _full_synthetic_fixture(tmp_path)
    report, _ = a2.build_core_report(rows, caches_dir, runtime_dir, bootstrap_seed_base=1, bootstrap_draws=200)

    assert report["runtime_guard"]["verified_unchanged_before_analysis"] is True
    assert report["runtime_guard"]["verified_unchanged_after_analysis"] is True
    assert report["deployed_prediction_coverage"]["n_unique_rows"] == 3
    assert report["statistics"]["n_unique_rows"] == 3
    assert report["statistics"]["n_components"] == 2


# ---------------------------------------------------------------------------
# Markdown rendering smoke test
# ---------------------------------------------------------------------------


def test_render_markdown_contains_key_sections(tmp_path):
    rows, caches_dir, runtime_dir = _full_synthetic_fixture(tmp_path)
    report, _ = a2.build_core_report(rows, caches_dir, runtime_dir, bootstrap_seed_base=1, bootstrap_draws=200)
    report["reproducibility"] = {"hash_run_1": "aa" * 32, "hash_run_2": "aa" * 32, "matched": True}

    markdown = a2.render_markdown(report)

    assert "Gate MDE" in markdown
    assert "PRE-ANNONCE" in markdown
    assert "Bandes jointes" in markdown
    assert "Courbe de faisabilite" in markdown
    assert "Reproductibilite" in markdown


# ---------------------------------------------------------------------------
# Preflight gate / abort rule
# ---------------------------------------------------------------------------


def test_load_preflight_status_missing_file(tmp_path):
    status = a2.load_preflight_status(tmp_path / "does-not-exist.json")
    assert status["ok"] is False


def test_load_preflight_status_fails_when_calibration_not_pass(tmp_path):
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "checks": {
                    "1_prediction_rows": {"status": "pass"},
                    "2_fold_caches": {"status": "pass"},
                    "3_calibration": {"status": "blocked"},
                }
            }
        ),
        encoding="utf-8",
    )
    status = a2.load_preflight_status(path)
    assert status["ok"] is False
    assert "3_calibration" in status["reason"]


def test_load_preflight_status_passes_when_all_pass(tmp_path):
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "checks": {
                    "1_prediction_rows": {"status": "pass"},
                    "2_fold_caches": {"status": "pass"},
                    "3_calibration": {"status": "pass"},
                }
            }
        ),
        encoding="utf-8",
    )
    status = a2.load_preflight_status(path)
    assert status["ok"] is True


def test_write_blocked_report_writes_json_and_md(tmp_path):
    output_json = tmp_path / "a2.json"
    output_md = tmp_path / "a2.md"
    a2.write_blocked_report(output_json, output_md, {"reason": "calibration failed", "preflight_path": "x"})

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["status"] == "blocked"
    assert "calibration failed" in data["reason"]
    assert "BLOQUE" in output_md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Deliverable curve + v10 reconciliation
# ---------------------------------------------------------------------------


def test_deliverable_curve_formula():
    curve = a2.deliverable_curve(hw_25=0.05, targets=(0.01, 0.05), current_k=25)
    assert curve["0.05"]["required_components"] == 25
    assert curve["0.01"]["required_components"] == math.ceil(25 * (0.05 / 0.01) ** 2)


def test_v10_reconciliation_replicates_documented_42(tmp_path):
    # Replicates the exact v10 power_sensitivity block shape (real values
    # from reports/autoresearch-v10-model-improvement-20260803.md /
    # runs/20260803-self-supervised-v10/iteration-0004/results/summary.json)
    v10_power = {
        "v10_B14_minus_S14": {
            "half_width": 0.012860569083102687,
            "current_evaluable_components": 25,
            "conditional_required_components": {"half_width_0.005": 166, "half_width_0.010": 42},
        }
    }
    curve = a2.deliverable_curve(hw_25=0.02, targets=(0.01,), current_k=25)
    recon = a2.build_v10_reconciliation(hw25_c1=0.02, curve=curve, k_current=25, v10_power=v10_power)
    assert recon["available"] is True
    assert recon["formula_replication_check"]["matches_v10_claim_hw_0.01"] is True
    assert recon["formula_replication_check"]["matches_v10_claim_hw_0.005"] is True


def test_v10_reconciliation_handles_missing_source():
    recon = a2.build_v10_reconciliation(hw25_c1=0.02, curve={"0.01": {"required_components": 1}}, k_current=25, v10_power=None)
    assert recon["available"] is False

#!/usr/bin/env python3
"""Lot A Phase A2 — grouped paired bootstrap vs the deployed runtime
(docs/lot-a-plan-20260807.md, Phase A2 + Revisions post-critique items 2-4).

Unit = row_id (653 unique OOF rows). Deployed predictions are deterministic
(no seed dimension): proj(base_features) -> L2 -> cosine vs prototypes.pt ->
top-1, read via common.DeployedRuntime against the read-only
backend/codex_model runtime. Research arms (b0, c1, r2 == candidate) are
averaged over the 3 seeds per row before differencing against the deployed
outcome, per the plan's exact procedure:

    delta_i(arm) = mean_seeds(correct_arm_i) - correct_deployed_i
    statistic(arm) = mean_i delta_i(arm)

giving one paired vector of 653 deltas per arm. Inference is a cluster
bootstrap over provenance_component (25 components in the frozen 653-row
universe): resample K components with replacement, keep every row of each
sampled component (whole-cluster resampling — no row is ever split from its
component). Headline CI = BCa (jackknife acceleration, leave-one-component-
out); percentile CI is annex only. A max-statistic joint band across the 3
arms (same resample per draw, so the arm-to-arm correlation induced by the
shared deployed control is preserved) replaces the abolished Holm
correction (B3). The MDE gate (ICC-based analytical half-width + MDE_80%)
is computed and placed in the report BEFORE the bootstrap results (B5 pre-
announcement requirement) — the code below computes it before calling the
bootstrap for the same reason.

CPU only. Never writes under backend/codex_model/ (DeployedRuntime asserted
unchanged before and after the whole run).

Abort rule (no proxy ever): if reports/lot-a-20260807/preflight.json does
not show checks 1/2/3 as "pass", A2 is not executed — a blocked report is
written instead (docs/lot-a-plan-20260807.md, Phase A2 "Abort").
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lot_a import common  # noqa: E402

DEFAULT_OUTPUT_JSON = ROOT / "reports/lot-a-20260807/a2.json"
DEFAULT_OUTPUT_MD = ROOT / "reports/lot-a-20260807/a2.md"
DEFAULT_OUTPUT_NPZ = ROOT / "reports/lot-a-20260807/a2_bootstrap_draws.npz"
DEFAULT_PREFLIGHT_JSON = ROOT / "reports/lot-a-20260807/preflight.json"

# v10's coverage/power audit (reconciliation target, Revision item 4 /
# critique B5). Read-only citation, never re-derived from scratch.
V10_POWER_SUMMARY_JSON = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-self-supervised-v10/iteration-0004/results/summary.json"
)
V10_REPORT_PATH = ROOT / "reports/autoresearch-v10-model-improvement-20260803.md"

ARM_FIELDS = {"b0": "b0_topk", "c1": "c1_topk", "r2": "candidate_topk"}
ARMS = ("b0", "c1", "r2")
PRIMARY_ARM = "c1"
BOOTSTRAP_ALPHA = 0.05
POWER_TARGET = 0.80
DEFAULT_BOOTSTRAP_DRAWS = 10000
DELIVERABLE_TARGETS = (0.01, 0.02, 0.05)
PREFLIGHT_REQUIRED_CHECKS = ("1_prediction_rows", "2_fold_caches", "3_calibration")

PREPROCESSING_CAVEAT_TEXT = (
    "Les predictions «deployees» sont calculees a partir des features mises en "
    "cache par le pipeline de recherche (base_features des caches OOF v9, backbone "
    "dinov2_vits14, image_size 224, S/14, white-pad — parite verifiee en Phase 0 face a "
    "la config du runtime deploye), et non par une relecture directe des images brutes "
    "a travers le runtime deploye. L'equivalence deploye repose donc sur cette parite "
    "de pipeline plus la preuve du gate de calibration C1 (Phase 0, check 3 : "
    "50/50 correspondances exactes top-1, 50/50 top-3 sur le pilote 50-lignes). "
    "Important : ce gate de calibration valide que le checkpoint de RECHERCHE C1 "
    "reproduit les c1_topk stockes — il ne valide PAS que les PROTOTYPES DEPLOYES "
    "(prototypes.pt) sont bien calibres sur ces memes features en cache. La precision "
    "deployee mesuree ici sur les 653 lignes OOF est faible (voir la section "
    "'Verification de sanite'), et une verification independante sur le split TRAIN "
    "(bien plus grand, mêmes fichiers de cache) montre une precision egalement faible "
    "(quelques %), ce qui suggere un ecart plus large que le seul bruit de petit "
    "echantillon (U2 du registre d'incertitudes) plutot qu'une anomalie isolee au "
    "sous-ensemble OOF."
)

U1_CAVEAT_TEXT = (
    "Incertitude U1 (registre du plan) : le runtime deploye a potentiellement vu des "
    "sources apparentees a certaines des 653 lignes (prefixe row_id «external:» = "
    "corpus externe, provenance disjointe d'apres l'audit de provenance v8 ; prefixe "
    "«legacy:» = lignes historiques potentiellement provenance-adjacentes a "
    "l'entrainement deploye). Le delta mesure ici vis-a-vis du deploye peut donc "
    "refleter en partie de la fuite de provenance plutot que uniquement la qualite du "
    "candidat. Voir la ventilation par prefixe en annexe (bras de sensibilite U1) : si "
    "le delta differe materiellement entre 'external:' et 'legacy:', c'est signale "
    "explicitement."
)


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _row_id_prefix(row_id: str) -> str:
    return row_id.split(":", 1)[0] + ":" if ":" in row_id else "(no-prefix)"


def _sanitize_floats(value: Any) -> Any:
    """Recursively replace NaN/Inf with None so the canonical JSON contract
    never emits the non-standard NaN/Infinity tokens Python's json module
    would otherwise allow."""
    if isinstance(value, dict):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_floats(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# ---------------------------------------------------------------------------
# Normal CDF / inverse CDF — dependency-free (no scipy), deterministic.
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF. Peter Acklam's rational approximation
    plus one Halley refinement step (full float64 precision). Used instead
    of scipy.stats.norm so this script has no new third-party dependency."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    # One Halley step for full double precision.
    e = _norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return x


Z_975 = _norm_ppf(0.975)
Z_80 = _norm_ppf(0.80)


# ---------------------------------------------------------------------------
# Deployed predictions (IO: v9 fold OOF caches + backend/codex_model)
# ---------------------------------------------------------------------------


def compute_deployed_predictions(
    rows: list[dict], v9_caches_dir: Path, runtime_dir: Path
) -> tuple[dict[str, dict], dict]:
    """Deployed top-1 outcome for every unique row_id, read from whichever
    v9 fold's OOF cache holds that row (each row_id lives in exactly one
    fold — verified below, never assumed). Returns (per_row, meta)."""
    unique_meta: dict[str, dict] = {}
    for row in rows:
        row_id = row["row_id"]
        entry = {
            "label": row["label"],
            "component": row["provenance_component"],
            "fold": row["outer_fold"],
            "class_name": row["class_name"],
        }
        if row_id in unique_meta:
            if unique_meta[row_id] != entry:
                raise common.LotASchemaError(
                    f"row_id {row_id!r} has inconsistent metadata across seed "
                    f"duplicates: {unique_meta[row_id]} != {entry}"
                )
        else:
            unique_meta[row_id] = entry

    runtime = common.DeployedRuntime(runtime_dir)
    runtime.assert_unchanged()  # before

    per_row: dict[str, dict] = {}
    per_fold_coverage: dict[str, int] = {}
    train_sanity_correct = 0
    train_sanity_n = 0

    for fold in common.EXPECTED_FOLDS:
        cache_path = v9_caches_dir / f"fold-{fold:02d}-views08.pt"
        if not cache_path.is_file():
            per_fold_coverage[str(fold)] = 0
            continue
        sha_ok, actual_sha, expected_sha = common.verify_cache_sha(cache_path)
        if not sha_ok:
            raise common.LotASchemaError(
                f"fold {fold} cache sha256 mismatch: actual={actual_sha} "
                f"expected={expected_sha}"
            )
        cache = common.load_fold_cache(cache_path)
        oof = cache["oof"]
        oof_row_ids = list(oof["row_id"])
        oof_component_ids = list(oof.get("component_id", []))
        base_features = oof["base_features"].to(torch.float32)

        wanted_indices = [i for i, rid in enumerate(oof_row_ids) if rid in unique_meta]
        per_fold_coverage[str(fold)] = len(wanted_indices)
        if wanted_indices:
            wanted_row_ids = [oof_row_ids[i] for i in wanted_indices]
            features = base_features[wanted_indices]
            topk = runtime.predict_topk(features, k=1)

            for idx, row_id, top1_row in zip(wanted_indices, wanted_row_ids, topk):
                meta = unique_meta[row_id]
                if meta["fold"] != fold:
                    raise common.LotASchemaError(
                        f"row_id {row_id!r}: prediction_rows outer_fold={meta['fold']} "
                        f"but found in fold-{fold:02d} oof cache"
                    )
                if oof_component_ids and oof_component_ids[idx] != meta["component"]:
                    raise common.LotASchemaError(
                        f"row_id {row_id!r}: provenance_component mismatch — cache="
                        f"{oof_component_ids[idx]!r} prediction_rows={meta['component']!r}"
                    )
                if row_id in per_row:
                    raise common.LotASchemaError(
                        f"row_id {row_id!r} appears in more than one fold's oof cache"
                    )
                per_row[row_id] = {
                    "correct_deployed": bool(top1_row[0] == meta["label"]),
                    "component": meta["component"],
                    "fold": meta["fold"],
                    "class_name": meta["class_name"],
                }

        # Descriptive-only sanity check (not part of the primary A2 design):
        # deployed accuracy on this fold's much larger TRAIN split (same
        # cache file, already loaded — no extra IO), to contextualize
        # whether a low/zero OOF accuracy reflects the true underlying rate
        # or a low-N artifact on the 653-row holdout (U2 in the plan's
        # uncertainty register: cached-feature/deployed-pipeline parity).
        train = cache.get("train")
        if train is not None and len(train.get("class_label", [])) > 0:
            train_features = train["base_features"].to(torch.float32)
            train_labels = train["class_label"]
            train_topk = runtime.predict_topk(train_features, k=1)
            train_sanity_correct += sum(
                1 for i, row in enumerate(train_topk) if row[0] == train_labels[i]
            )
            train_sanity_n += len(train_labels)

    runtime.assert_unchanged()  # after

    missing = sorted(set(unique_meta) - set(per_row))
    if missing:
        raise common.LotASchemaError(
            f"{len(missing)} row_id(s) from prediction_rows not found in any v9 fold "
            f"oof cache: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    meta = {
        "sha256_at_load": runtime.sha256_at_load,
        "per_fold_coverage": per_fold_coverage,
        "n_unique_rows": len(unique_meta),
        "n_predicted": len(per_row),
        "train_split_sanity_check": {
            "label": (
                "descriptif uniquement, hors design A2 primaire — precision deployee "
                "sur le split TRAIN (bien plus grand, non independamment evaluable, "
                "meme fichiers de cache que les 653 OOF) pour contextualiser un taux "
                "OOF faible/nul : artefact de petit N ou motif plus large (U2) ?"
            ),
            "n": train_sanity_n,
            "correct": train_sanity_correct,
            "accuracy": (train_sanity_correct / train_sanity_n) if train_sanity_n > 0 else None,
        },
    }
    return per_row, meta


# ---------------------------------------------------------------------------
# Paired records: delta_i(arm) = mean_seeds(correct_arm_i) - correct_deployed_i
# ---------------------------------------------------------------------------


def build_paired_records(rows: list[dict], deployed: dict[str, dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["row_id"]].append(row)

    records: dict[str, dict] = {}
    for row_id, group in grouped.items():
        if row_id not in deployed:
            raise common.LotASchemaError(f"row_id {row_id!r} has no deployed prediction")
        labels = {r["label"] for r in group}
        components = {r["provenance_component"] for r in group}
        folds = {r["outer_fold"] for r in group}
        if len(labels) != 1 or len(components) != 1 or len(folds) != 1:
            raise common.LotASchemaError(
                f"row_id {row_id!r}: inconsistent label/component/fold across seed "
                "duplicates"
            )
        n_seeds = len(group)
        avg_correct = {
            arm: sum(r[field][0] == r["label"] for r in group) / n_seeds
            for arm, field in ARM_FIELDS.items()
        }
        deployed_correct = 1.0 if deployed[row_id]["correct_deployed"] else 0.0

        records[row_id] = {
            "component": components.pop(),
            "fold": folds.pop(),
            "prefix": _row_id_prefix(row_id),
            "n_seeds": n_seeds,
            "correct_deployed": deployed[row_id]["correct_deployed"],
            **{f"avg_correct_{arm}": avg_correct[arm] for arm in ARMS},
            **{f"delta_{arm}": avg_correct[arm] - deployed_correct for arm in ARMS},
        }
    return records


# ---------------------------------------------------------------------------
# ANOVA-style ICC + analytical MDE gate (pre-bootstrap, pre-registered)
# ---------------------------------------------------------------------------


def compute_component_stats(per_component_values: dict[str, list[float]]) -> dict:
    """Unbalanced one-way random-effects ANOVA (ICC, reported as a
    diagnostic — pre-registered per the plan) PLUS a with-replacement-PSU
    linearized ratio-variance estimate (Woodruff 1971 / delete-a-group
    jackknife equivalent) for the actual resampling scheme used downstream:
    K draws with replacement of the K available provenance_components,
    keeping every row of each sampled component whole. That resampling
    scheme's variance depends on the BETWEEN-component dispersion of each
    component's (size-weighted) contribution, not on within-component
    residual noise — a naive design-effect/ICC formula (SE_iid * DEFF)
    conflates the two and degenerates to 0 whenever a component is
    internally homogeneous, which is wrong for this resampling unit. This
    is the pre-bootstrap MDE gate (B5): computed and reported BEFORE the
    actual bootstrap draws are taken."""
    components = sorted(per_component_values)
    k = len(components)
    ns = np.array([len(per_component_values[c]) for c in components], dtype=np.float64)
    means = np.array(
        [float(np.mean(per_component_values[c])) for c in components], dtype=np.float64
    )
    n_total = float(ns.sum())
    grand_mean = float(np.sum(ns * means) / n_total) if n_total > 0 else float("nan")

    ss_between = float(np.sum(ns * (means - grand_mean) ** 2))
    ms_between = ss_between / (k - 1) if k > 1 else float("nan")

    ss_within = 0.0
    for i, c in enumerate(components):
        values = np.asarray(per_component_values[c], dtype=np.float64)
        ss_within += float(np.sum((values - means[i]) ** 2))
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else float("nan")

    if k > 1:
        n0 = (n_total - float(np.sum(ns**2)) / n_total) / (k - 1)
    else:
        n0 = float("nan")

    if k > 1 and n0 > 0 and not math.isnan(ms_within) and (ms_between + (n0 - 1) * ms_within) != 0:
        icc_raw = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
        icc = max(0.0, min(1.0, icc_raw))
    else:
        icc_raw = float("nan")
        icc = float("nan")

    m_bar = n_total / k if k > 0 else float("nan")

    # With-replacement PSU (cluster) linearized ratio variance for the
    # statistic actually computed downstream: statistic = sum(comp_sum) /
    # sum(comp_n) over K clusters resampled with replacement. z_k is the
    # size-weighted residual contribution of component k; sum(z_k) == 0 by
    # construction. Var_hat = sum(z_k^2) / (K*(K-1)*m_bar^2).
    if k > 1 and n_total > 0:
        comp_sums = ns * means
        z = comp_sums - ns * grand_mean
        var_hat = float(np.sum(z**2)) / (k * (k - 1) * m_bar**2)
        se_analytical = math.sqrt(var_hat) if var_hat >= 0 else float("nan")
    else:
        se_analytical = float("nan")

    half_width_95_analytical = Z_975 * se_analytical if not math.isnan(se_analytical) else float("nan")
    mde_80_analytical = (
        (Z_975 + Z_80) * se_analytical if not math.isnan(se_analytical) else float("nan")
    )

    return {
        "k_components": k,
        "n_total_rows": int(n_total),
        "grand_mean": grand_mean,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "kish_correction_n0": n0,
        "icc_raw": icc_raw,
        "icc_clipped_0_1": icc,
        "mean_cluster_size": m_bar,
        "se_analytical_cluster": se_analytical,
        "half_width_95_analytical": half_width_95_analytical,
        "mde_80pct_analytical": mde_80_analytical,
        "methodology": (
            "ICC(1) diagnostique via ANOVA a un facteur non equilibre : "
            "ICC = (MSB-MSW)/(MSB+(n0-1)*MSW), n0 = correction de Kish pour tailles "
            "inegales = (N - sum(n_k^2)/N)/(K-1). SE analytique du gate MDE : "
            "estimateur de variance lineaire pour un ratio en echantillonnage par "
            "grappes AVEC REMISE (Woodruff 1971, equivalent au jackknife "
            "delete-a-group) — adapte au schema de rechantillonnage reellement utilise "
            "plus bas (K composantes tirees avec remise parmi les K disponibles, "
            "toutes les lignes d'une composante tiree ensemble) : z_k = sum_delta(k) - "
            "n_k*moyenne_globale (residu pondere par taille, somme nulle par "
            "construction) ; Var_hat = sum(z_k^2) / (K*(K-1)*taille_moyenne^2) ; "
            f"demi-largeur 95% analytique = z_0.975*SE (z_0.975={Z_975:.6f}) ; "
            f"MDE_80% analytique = (z_0.975+z_0.80)*SE (z_0.80={Z_80:.6f}), "
            "approximation normale bilaterale alpha=0.05 / puissance=80%."
        ),
    }


# ---------------------------------------------------------------------------
# Cluster bootstrap primitives (component-level, whole-cluster resampling)
# ---------------------------------------------------------------------------


def _resample_component_indices(k: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, k, size=(n_draws, k))


def stat_from_draws_idx(
    comp_sum: np.ndarray, comp_n: np.ndarray, draws_idx: np.ndarray
) -> np.ndarray:
    sums = comp_sum[draws_idx].sum(axis=1)
    counts = comp_n[draws_idx].sum(axis=1)
    return sums / counts


def joint_cluster_bootstrap(
    comp_sums: dict[str, np.ndarray],
    comp_n: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One shared component resample per draw applied to every arm — this
    is what preserves the arm-to-arm correlation induced by the shared
    deployed control (plan step 4: 'meme resample -> preserve la
    correlation')."""
    draws_idx = _resample_component_indices(len(comp_n), n_draws, rng)
    draws = {arm: stat_from_draws_idx(comp_sums[arm], comp_n, draws_idx) for arm in comp_sums}
    return draws_idx, draws


def leave_one_component_out_jackknife(comp_sum: np.ndarray, comp_n: np.ndarray) -> np.ndarray:
    total_sum = comp_sum.sum()
    total_n = comp_n.sum()
    return (total_sum - comp_sum) / (total_n - comp_n)


def observed_stat(comp_sum: np.ndarray, comp_n: np.ndarray) -> float:
    return float(comp_sum.sum() / comp_n.sum())


# ---------------------------------------------------------------------------
# BCa / percentile confidence intervals
# ---------------------------------------------------------------------------


def bca_ci(
    observed: float,
    boot_draws: np.ndarray,
    jackknife_stats: np.ndarray,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    n = len(boot_draws)
    prop_less = float(
        np.mean(boot_draws < observed) + 0.5 * np.mean(boot_draws == observed)
    )
    # Smoothing correction so norm.ppf never sees exactly 0 or 1.
    prop_less = min(max(prop_less, 1.0 / (n + 1)), n / (n + 1))
    z0 = _norm_ppf(prop_less)

    jack_mean = float(np.mean(jackknife_stats))
    diffs = jack_mean - jackknife_stats
    num = float(np.sum(diffs**3))
    den = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    acceleration = num / den if den != 0 else 0.0

    z_lo = _norm_ppf(alpha / 2)
    z_hi = _norm_ppf(1 - alpha / 2)

    def _adjusted_percentile(z_val: float) -> float:
        denom = 1 - acceleration * (z0 + z_val)
        if denom == 0:
            return 0.5
        return _norm_cdf(z0 + (z0 + z_val) / denom)

    alpha1 = min(max(_adjusted_percentile(z_lo), 0.0), 1.0)
    alpha2 = min(max(_adjusted_percentile(z_hi), 0.0), 1.0)

    lo, hi = np.percentile(boot_draws, [alpha1 * 100, alpha2 * 100])
    if lo > hi:
        lo, hi = hi, lo
    return {
        "lo": float(lo),
        "hi": float(hi),
        "z0": z0,
        "acceleration": acceleration,
        "alpha1_percentile": alpha1,
        "alpha2_percentile": alpha2,
    }


def percentile_ci(boot_draws: np.ndarray, alpha: float = BOOTSTRAP_ALPHA) -> dict:
    lo, hi = np.percentile(boot_draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------------------
# Joint (simultaneous) bands across the 3 arms — max-statistic method
# ---------------------------------------------------------------------------


def max_stat_joint_bands(
    observed: dict[str, float],
    boot_draws: dict[str, np.ndarray],
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    """Simultaneous (1-alpha) coverage across all arms via the max-t / sup-t
    method: scale each arm's centered bootstrap draws by its own bootstrap
    SE, take the max |t| across arms per draw, and use the (1-alpha)
    quantile of that max as a single shared multiplier — replaces Holm
    (B3), using the same joint resample so correlation is honored rather
    than assumed away."""
    arms = sorted(observed)
    se = {arm: float(np.std(boot_draws[arm], ddof=1)) for arm in arms}
    t_stack = np.stack(
        [
            (boot_draws[arm] - observed[arm]) / (se[arm] if se[arm] > 0 else 1.0)
            for arm in arms
        ],
        axis=0,
    )
    max_abs_t = np.max(np.abs(t_stack), axis=0)
    q = float(np.percentile(max_abs_t, (1 - alpha) * 100))
    bands = {
        arm: {
            "lo": observed[arm] - q * se[arm],
            "hi": observed[arm] + q * se[arm],
            "se": se[arm],
        }
        for arm in arms
    }
    return {"critical_value_q": q, "alpha": alpha, "bands": bands}


# ---------------------------------------------------------------------------
# Deliverable curve k(HW_target) + v10 reconciliation
# ---------------------------------------------------------------------------


def deliverable_curve(
    hw_25: float, targets: tuple[float, ...] = DELIVERABLE_TARGETS, current_k: int = 25
) -> dict:
    return {
        f"{target:.3f}".rstrip("0").rstrip("."): {
            "target_half_width": target,
            "required_components": math.ceil(current_k * (hw_25 / target) ** 2),
        }
        for target in targets
    }


def load_v10_power_sensitivity(path: Path = V10_POWER_SUMMARY_JSON) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data.get("power_sensitivity")


def build_v10_reconciliation(
    hw25_c1: float, curve: dict, k_current: int, v10_power: dict | None
) -> dict:
    if not v10_power or "v10_B14_minus_S14" not in v10_power:
        return {
            "available": False,
            "reason": "bloc power_sensitivity v10 introuvable au chemin documente",
            "expected_path": str(V10_POWER_SUMMARY_JSON),
        }
    v10 = v10_power["v10_B14_minus_S14"]
    v10_hw = float(v10["half_width"])
    v10_k_current = int(v10["current_evaluable_components"])
    v10_target_001 = int(v10["conditional_required_components"]["half_width_0.010"])
    v10_target_005 = int(v10["conditional_required_components"]["half_width_0.005"])
    replicated_001 = math.ceil(v10_k_current * (v10_hw / 0.01) ** 2)
    replicated_005 = math.ceil(v10_k_current * (v10_hw / 0.005) ** 2)
    our_001 = curve["0.01"]["required_components"]

    return {
        "available": True,
        "v10_source": (
            "reports/autoresearch-v10-model-improvement-20260803.md + "
            ".omc/autoresearch/elements-baseline-replacement/runs/"
            "20260803-self-supervised-v10/iteration-0004/results/summary.json"
            "#power_sensitivity.v10_B14_minus_S14"
        ),
        "v10_comparison_label": (
            "v10_B14_minus_S14 (backbone de recherche B/14 vs S/14, audit "
            "coverage/power, memes 25 composantes evaluables)"
        ),
        "a2_comparison_label": "C1 vs runtime deploye (comparaison primaire de ce rapport)",
        "v10_measured_half_width": v10_hw,
        "v10_current_evaluable_components": v10_k_current,
        "v10_claimed_required_components_hw_0.01": v10_target_001,
        "v10_claimed_required_components_hw_0.005": v10_target_005,
        "formula_replication_check": {
            "note": "reproduit le calcul de v10 a partir de ses propres entrees, pour verifier que la formule est identique",
            "replicated_required_components_hw_0.01": replicated_001,
            "replicated_required_components_hw_0.005": replicated_005,
            "matches_v10_claim_hw_0.01": replicated_001 == v10_target_001,
            "matches_v10_claim_hw_0.005": replicated_005 == v10_target_005,
        },
        "a2_measured_half_width_c1": hw25_c1,
        "a2_current_evaluable_components": k_current,
        "a2_required_components_hw_0.01": our_001,
        "difference_required_components_hw_0.01": our_001 - v10_target_001,
        "explanation": (
            "La formule k(HW_cible) = ceil(k_actuel * (HW_actuel/HW_cible)^2) "
            "(extrapolation descriptive en 1/sqrt(n)) est identique entre v10 et A2 — "
            "verifie ci-dessus par replication exacte des chiffres v10 a partir de ses "
            "propres entrees (formula_replication_check). Une eventuelle divergence "
            "entre le '42' de v10 (pour HW_cible=0.01) et le chiffre A2 ne vient donc "
            "pas de la formule mais de la demi-largeur de base HW_25 mesuree : "
            "v10_B14_minus_S14 compare deux backbones de recherche (B/14 vs S/14, "
            "memes 25 composantes, comparaison de recherche appariee), tandis que A2 "
            "compare C1 au runtime deploye (comparaison differente, variance "
            "inter-composantes potentiellement differente). Les deux demi-largeurs ne "
            "sont pas censees coincider ; si elles different, c'est rapporte comme un "
            "resultat en soi, pas force a l'accord."
        ),
    }


# ---------------------------------------------------------------------------
# Stratifications (annex): per outer_fold, per row_id prefix (U1)
# ---------------------------------------------------------------------------


def stratify_by(
    paired_records: dict[str, dict], key_fn: Callable[[dict], str]
) -> dict[str, dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id, rec in paired_records.items():
        groups[key_fn(rec)].append(row_id)

    result: dict[str, dict] = {}
    for key in sorted(groups):
        row_ids = groups[key]
        components = sorted({paired_records[r]["component"] for r in row_ids})
        result[key] = {
            "n_rows": len(row_ids),
            "n_components": len(components),
            "mean_delta_b0": float(np.mean([paired_records[r]["delta_b0"] for r in row_ids])),
            "mean_delta_c1": float(np.mean([paired_records[r]["delta_c1"] for r in row_ids])),
            "mean_delta_r2": float(np.mean([paired_records[r]["delta_r2"] for r in row_ids])),
        }
    return result


def flag_material_prefix_difference(strat_prefix: dict, hw25_c1: float) -> dict:
    keys = sorted(strat_prefix)
    if len(keys) < 2:
        return {"flagged": False, "reason": "moins de 2 prefixes presents", "pairwise": []}
    pairwise = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            diff = abs(strat_prefix[a]["mean_delta_c1"] - strat_prefix[b]["mean_delta_c1"])
            pairwise.append(
                {
                    "prefix_a": a,
                    "prefix_b": b,
                    "abs_diff_mean_delta_c1": diff,
                    "exceeds_hw25_c1": diff > hw25_c1,
                }
            )
    flagged = any(p["exceeds_hw25_c1"] for p in pairwise)
    return {
        "flagged": flagged,
        "threshold_definition": (
            "materiel si |diff mean_delta_c1 entre 2 prefixes| > HW_25_c1 (demi-largeur "
            "BCa mesuree de C1, meme unite que le delta)"
        ),
        "threshold_hw25_c1": hw25_c1,
        "pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# Statistics assembly (pure function — no IO, directly testable)
# ---------------------------------------------------------------------------


def build_statistical_report(
    paired_records: dict[str, dict],
    bootstrap_seed_base: int,
    bootstrap_draws: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    row_ids = sorted(paired_records)
    if not row_ids:
        raise common.LotASchemaError("build_statistical_report: no paired records")

    components = sorted({paired_records[r]["component"] for r in row_ids})
    k = len(components)
    comp_index = {c: i for i, c in enumerate(components)}

    comp_n = np.zeros(k, dtype=np.float64)
    comp_sum = {arm: np.zeros(k, dtype=np.float64) for arm in ARMS}
    per_component_values: dict[str, dict[str, list[float]]] = {
        arm: defaultdict(list) for arm in ARMS
    }

    for row_id in row_ids:
        rec = paired_records[row_id]
        idx = comp_index[rec["component"]]
        comp_n[idx] += 1
        for arm in ARMS:
            value = rec[f"delta_{arm}"]
            comp_sum[arm][idx] += value
            per_component_values[arm][rec["component"]].append(value)

    n_total = int(comp_n.sum())
    observed = {arm: observed_stat(comp_sum[arm], comp_n) for arm in ARMS}

    deployed_accuracy = sum(
        1 for r in row_ids if paired_records[r]["correct_deployed"]
    ) / n_total
    arm_accuracy_avg_over_seeds = {
        arm: float(np.mean([paired_records[r][f"avg_correct_{arm}"] for r in row_ids]))
        for arm in ARMS
    }

    # --- MDE gate (pre-bootstrap, pre-registered — B5) ---
    mde_gate_pre_bootstrap = {
        arm: compute_component_stats(per_component_values[arm]) for arm in ARMS
    }

    # --- cluster bootstrap (joint resample preserves arm correlation — B3) ---
    rng = np.random.default_rng(bootstrap_seed_base)
    draws_idx, boot_draws = joint_cluster_bootstrap(comp_sum, comp_n, bootstrap_draws, rng)
    jackknife = {arm: leave_one_component_out_jackknife(comp_sum[arm], comp_n) for arm in ARMS}

    bca = {
        arm: bca_ci(observed[arm], boot_draws[arm], jackknife[arm], BOOTSTRAP_ALPHA)
        for arm in ARMS
    }
    percentile = {arm: percentile_ci(boot_draws[arm], BOOTSTRAP_ALPHA) for arm in ARMS}
    joint = max_stat_joint_bands(observed, boot_draws, BOOTSTRAP_ALPHA)

    hw25_c1 = (bca[PRIMARY_ARM]["hi"] - bca[PRIMARY_ARM]["lo"]) / 2.0
    curve = deliverable_curve(hw25_c1, DELIVERABLE_TARGETS, current_k=k)
    v10_power = load_v10_power_sensitivity()
    reconciliation = build_v10_reconciliation(hw25_c1, curve, k, v10_power)

    strat_fold = stratify_by(paired_records, lambda rec: str(rec["fold"]))
    strat_prefix = stratify_by(paired_records, lambda rec: rec["prefix"])
    prefix_flag = flag_material_prefix_difference(strat_prefix, hw25_c1)

    statistics: dict[str, Any] = {
        "definitions": {
            "unit": f"row_id ({n_total} uniques)",
            "correct_arm": "arm_topk[0] == label, moyenne sur les seeds disponibles pour ce row_id",
            "correct_deployed": (
                "predict_topk(base_features, k=1)[0] == label, deterministe "
                "(pas de dimension seed)"
            ),
            "delta": "delta_i(arm) = moyenne_seeds(correct_arm_i) - correct_deployed_i",
            "statistic_arm": f"moyenne sur les {n_total} lignes de delta_i(arm)",
            "primary_arm": PRIMARY_ARM,
            "secondary_arms": [a for a in ARMS if a != PRIMARY_ARM],
        },
        "n_unique_rows": n_total,
        "n_components": k,
        "bootstrap_seed_base": bootstrap_seed_base,
        "bootstrap_draws": bootstrap_draws,
        "deployed_accuracy": deployed_accuracy,
        "arm_accuracy_avg_over_seeds": arm_accuracy_avg_over_seeds,
        "observed_delta": observed,
        "mde_gate_pre_bootstrap": mde_gate_pre_bootstrap,
        "bootstrap": {
            "bca_headline": bca,
            "percentile_annex": percentile,
        },
        "joint_bands": joint,
        "deliverable_curve": {
            "formula": "k(HW_cible) = ceil(k_actuel * (HW_actuel/HW_cible)^2)",
            "headline_half_width_c1": hw25_c1,
            "current_k_components": k,
            "targets": curve,
        },
        "v10_reconciliation": reconciliation,
        "stratifications": {
            "by_outer_fold": strat_fold,
            "by_row_id_prefix": strat_prefix,
            "prefix_material_difference_flag_u1": prefix_flag,
        },
    }

    raw_draws: dict[str, np.ndarray] = {
        "components": np.array(components),
        "comp_n": comp_n,
        "draws_idx": draws_idx,
        "bootstrap_seed_base": np.array(bootstrap_seed_base),
        **{f"comp_sum_{arm}": comp_sum[arm] for arm in ARMS},
        **{f"stat_{arm}_draws": boot_draws[arm] for arm in ARMS},
        **{f"jackknife_{arm}": jackknife[arm] for arm in ARMS},
    }
    return statistics, raw_draws


# ---------------------------------------------------------------------------
# Full report assembly (IO + statistics)
# ---------------------------------------------------------------------------


def build_core_report(
    rows: list[dict],
    v9_caches_dir: Path,
    runtime_dir: Path,
    bootstrap_seed_base: int,
    bootstrap_draws: int,
    prediction_rows_path: Path | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    common.set_determinism(bootstrap_seed_base)

    deployed, deployed_meta = compute_deployed_predictions(rows, v9_caches_dir, runtime_dir)
    paired_records = build_paired_records(rows, deployed)
    statistics, raw_draws = build_statistical_report(
        paired_records, bootstrap_seed_base, bootstrap_draws
    )

    report: dict[str, Any] = {
        "schema_version": "lot-a-a2-v1",
        "git_sha": get_git_sha(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "runtime_guard": {
            "sha256_at_load": deployed_meta["sha256_at_load"],
            "verified_unchanged_before_analysis": True,
            "verified_unchanged_after_analysis": True,
        },
        "deployed_prediction_coverage": {
            "per_fold_coverage": deployed_meta["per_fold_coverage"],
            "n_unique_rows": deployed_meta["n_unique_rows"],
            "n_predicted": deployed_meta["n_predicted"],
        },
        "deployed_runtime_sanity_check": deployed_meta["train_split_sanity_check"],
        "preprocessing_caveat": PREPROCESSING_CAVEAT_TEXT,
        "u1_sensitivity_caveat": U1_CAVEAT_TEXT,
        "statistics": statistics,
    }
    if prediction_rows_path is not None:
        report["inputs"] = {
            "prediction_rows_path": str(prediction_rows_path),
            "prediction_rows_sha256": common.sha256_file(prediction_rows_path),
            "v9_caches_dir": str(v9_caches_dir),
            "runtime_dir": str(runtime_dir),
        }

    report = _sanitize_floats(report)
    return report, raw_draws


# ---------------------------------------------------------------------------
# Preflight gate (Abort rule: no proxy, ever)
# ---------------------------------------------------------------------------


def load_preflight_status(path: Path) -> dict:
    if not path.is_file():
        return {"ok": False, "reason": f"rapport preflight introuvable a {path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "reason": f"rapport preflight illisible ({exc})"}
    checks = data.get("checks", {})
    failing = [
        name for name in PREFLIGHT_REQUIRED_CHECKS if checks.get(name, {}).get("status") != "pass"
    ]
    if failing:
        return {
            "ok": False,
            "reason": f"checks preflight non 'pass' : {failing}",
            "preflight_path": str(path),
        }
    return {"ok": True, "preflight_path": str(path)}


def write_blocked_report(output_json: Path, output_md: Path, status: dict) -> None:
    report = {
        "schema_version": "lot-a-a2-v1",
        "status": "blocked",
        "reason": status["reason"],
        "preflight_path": status.get("preflight_path"),
        "note": (
            "A2 non execute : gate de calibration Phase 0 non satisfait "
            "(docs/lot-a-plan-20260807.md, Phase A2, Abort). Aucun proxy calcule."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(common.canonical_json_dumps(report), encoding="utf-8")

    md = (
        "# Lot A - Phase A2 : BLOQUE\n\n"
        f"{report['note']}\n\n"
        f"Raison : {report['reason']}\n"
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown report (French, descriptive only)
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    stats = report["statistics"]
    lines.append("# Lot A - Phase A2 : bootstrap apparie groupe vs runtime deploye")
    lines.append("")
    lines.append(f"git {report['git_sha'][:12]} - schema {report['schema_version']}")
    lines.append("")
    lines.append(
        "Langage descriptif uniquement : aucune formulation de promotion. "
        f"Unite = {stats['definitions']['unit']}."
    )
    lines.append("")

    lines.append("## Garde runtime deploye")
    lines.append("")
    guard = report["runtime_guard"]
    lines.append(
        f"- SHA-256 au chargement : projection={guard['sha256_at_load']['projection'][:16]}..., "
        f"prototypes={guard['sha256_at_load']['prototypes'][:16]}..., "
        f"config={guard['sha256_at_load']['config'][:16]}..."
    )
    lines.append(
        f"- Inchange avant analyse : {guard['verified_unchanged_before_analysis']} ; "
        f"inchange apres analyse : {guard['verified_unchanged_after_analysis']}"
    )
    lines.append("")

    lines.append("## Couverture des predictions deployees")
    lines.append("")
    cov = report["deployed_prediction_coverage"]
    lines.append(
        f"- Lignes uniques : {cov['n_unique_rows']} ; predites : {cov['n_predicted']}"
    )
    lines.append(f"- Par fold : {cov['per_fold_coverage']}")
    lines.append("")

    sanity = report.get("deployed_runtime_sanity_check")
    if sanity and sanity.get("n"):
        lines.append("## Verification de sanite du runtime deploye (descriptif, hors design A2 primaire)")
        lines.append("")
        lines.append(f"- {sanity['label']}")
        lines.append(
            f"- Split TRAIN (5 folds, memes fichiers de cache) : "
            f"{sanity['correct']}/{sanity['n']} = {_fmt(sanity['accuracy'])}"
        )
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(f"- **Pretraitement** : {report['preprocessing_caveat']}")
    lines.append("")
    lines.append(f"- **U1 (provenance)** : {report['u1_sensitivity_caveat']}")
    lines.append("")

    lines.append("## Gate MDE — PRE-ANNONCE (avant les tirages bootstrap)")
    lines.append("")
    lines.append(
        "Les chiffres de cette section sont calcules analytiquement (ANOVA / ICC) "
        "AVANT le bootstrap ci-dessous, conformement au pre-enregistrement (B5)."
    )
    lines.append("")
    lines.append(
        "| bras | ICC (diagnostique) | SE analytique (cluster) | demi-largeur 95% analytique | MDE 80% analytique |"
    )
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        gate = stats["mde_gate_pre_bootstrap"][arm]
        lines.append(
            f"| {arm} | {_fmt(gate['icc_clipped_0_1'])} | "
            f"{_fmt(gate['se_analytical_cluster'])} | {_fmt(gate['half_width_95_analytical'])} | "
            f"{_fmt(gate['mde_80pct_analytical'])} |"
        )
    lines.append("")
    lines.append(f"Methodologie ({PRIMARY_ARM}) : {stats['mde_gate_pre_bootstrap'][PRIMARY_ARM]['methodology']}")
    lines.append("")

    lines.append("## Resultats bootstrap (cluster sur provenance_component)")
    lines.append("")
    lines.append(
        f"- Tirages : {stats['bootstrap_draws']} ; seed : {stats['bootstrap_seed_base']} ; "
        f"composantes (K) : {stats['n_components']} ; lignes : {stats['n_unique_rows']}"
    )
    lines.append(f"- Precision deployee : {_fmt(stats['deployed_accuracy'])}")
    lines.append(
        "- Precision par bras (moyenne sur les seeds) : "
        + ", ".join(f"{arm}={_fmt(stats['arm_accuracy_avg_over_seeds'][arm])}" for arm in ARMS)
    )
    lines.append("")
    lines.append(
        "| bras | delta observe | BCa lo | BCa hi | percentile lo (annexe) | percentile hi (annexe) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        bca = stats["bootstrap"]["bca_headline"][arm]
        pct = stats["bootstrap"]["percentile_annex"][arm]
        lines.append(
            f"| {arm} | {_fmt(stats['observed_delta'][arm])} | {_fmt(bca['lo'])} | {_fmt(bca['hi'])} | "
            f"{_fmt(pct['lo'])} | {_fmt(pct['hi'])} |"
        )
    lines.append("")

    lines.append("## Bandes jointes (max-statistique, remplace Holm)")
    lines.append("")
    joint = stats["joint_bands"]
    lines.append(f"- q critique (max-|t| a {int((1 - joint['alpha']) * 100)}%) : {_fmt(joint['critical_value_q'])}")
    lines.append("")
    lines.append("| bras | bande jointe lo | bande jointe hi | SE bootstrap |")
    lines.append("|---|---|---|---|")
    for arm in ARMS:
        band = joint["bands"][arm]
        lines.append(f"| {arm} | {_fmt(band['lo'])} | {_fmt(band['hi'])} | {_fmt(band['se'])} |")
    lines.append("")

    lines.append("## Courbe de faisabilite k(HW_cible) et reconciliation v10")
    lines.append("")
    curve = stats["deliverable_curve"]
    lines.append(f"- Formule : `{curve['formula']}`")
    lines.append(
        f"- Demi-largeur mesuree (BCa, C1) HW_25 = {_fmt(curve['headline_half_width_c1'])} "
        f"avec k_actuel = {curve['current_k_components']} composantes"
    )
    lines.append("")
    lines.append("| cible HW | composantes requises |")
    lines.append("|---|---|")
    for key in sorted(curve["targets"], key=float):
        entry = curve["targets"][key]
        lines.append(f"| {entry['target_half_width']} | {entry['required_components']} |")
    lines.append("")
    recon = stats["v10_reconciliation"]
    if recon["available"]:
        lines.append(
            f"- v10 ({recon['v10_comparison_label']}) : HW={_fmt(recon['v10_measured_half_width'])}, "
            f"k_actuel={recon['v10_current_evaluable_components']}, "
            f"composantes requises (HW=0.01)={recon['v10_claimed_required_components_hw_0.01']} "
            "(le rapport v10 cite ~42)"
        )
        lines.append(
            f"- A2 ({recon['a2_comparison_label']}) : composantes requises (HW=0.01)="
            f"{recon['a2_required_components_hw_0.01']} "
            f"(difference vs v10 : {recon['difference_required_components_hw_0.01']:+d})"
        )
        lines.append(
            f"- Verification de la formule (replication des chiffres v10 a partir de ses "
            f"propres entrees) : {recon['formula_replication_check']}"
        )
        lines.append("")
        lines.append(recon["explanation"])
    else:
        lines.append(f"- Reconciliation v10 indisponible : {recon['reason']}")
    lines.append("")

    lines.append("## Ventilations annexes")
    lines.append("")
    lines.append("### Par outer_fold")
    lines.append("")
    lines.append("| fold | n lignes | n composantes | delta b0 | delta c1 | delta r2 |")
    lines.append("|---|---|---|---|---|---|")
    for key, entry in stats["stratifications"]["by_outer_fold"].items():
        lines.append(
            f"| {key} | {entry['n_rows']} | {entry['n_components']} | "
            f"{_fmt(entry['mean_delta_b0'])} | {_fmt(entry['mean_delta_c1'])} | {_fmt(entry['mean_delta_r2'])} |"
        )
    lines.append("")
    lines.append("### Par prefixe de row_id (external: vs legacy: — bras de sensibilite U1)")
    lines.append("")
    lines.append("| prefixe | n lignes | n composantes | delta b0 | delta c1 | delta r2 |")
    lines.append("|---|---|---|---|---|---|")
    for key, entry in stats["stratifications"]["by_row_id_prefix"].items():
        lines.append(
            f"| {key} | {entry['n_rows']} | {entry['n_components']} | "
            f"{_fmt(entry['mean_delta_b0'])} | {_fmt(entry['mean_delta_c1'])} | {_fmt(entry['mean_delta_r2'])} |"
        )
    lines.append("")
    flag = stats["stratifications"]["prefix_material_difference_flag_u1"]
    lines.append(f"- Difference materielle signalee (U1) : **{flag['flagged']}** ({flag.get('threshold_definition', '')})")
    lines.append("")

    if "reproducibility" in report:
        repro = report["reproducibility"]
        lines.append("## Reproductibilite")
        lines.append("")
        lines.append(
            f"- hash run 1 : {repro['hash_run_1'][:16]} ; hash run 2 : "
            f"{repro['hash_run_2'][:16]} ; identiques : {repro['matched']}"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-rows", type=Path, default=common.PREDICTION_ROWS_PATH)
    parser.add_argument("--v9-caches-dir", type=Path, default=common.V9_CACHES_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=common.DEPLOYED_RUNTIME_DIR)
    parser.add_argument("--preflight-json", type=Path, default=DEFAULT_PREFLIGHT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--skip-preflight-gate",
        action="store_true",
        help="bypass the preflight status gate (never use for the real run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_preflight_gate:
        status = load_preflight_status(args.preflight_json)
        if not status["ok"]:
            write_blocked_report(args.output_json, args.output_md, status)
            print(f"A2 BLOQUE : {status['reason']}")
            print(f"wrote {args.output_json}")
            print(f"wrote {args.output_md}")
            return 1

    rows = common.load_prediction_rows(args.prediction_rows)

    report_1, raw_draws = build_core_report(
        rows, args.v9_caches_dir, args.runtime_dir, args.seed, args.bootstrap_draws, args.prediction_rows
    )
    report_2, _ = build_core_report(
        rows, args.v9_caches_dir, args.runtime_dir, args.seed, args.bootstrap_draws, args.prediction_rows
    )
    hash_1 = common.sha256_json(report_1)
    hash_2 = common.sha256_json(report_2)
    matched = hash_1 == hash_2

    final_report = dict(report_1)
    final_report["reproducibility"] = {
        "hash_run_1": hash_1,
        "hash_run_2": hash_2,
        "matched": matched,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(common.canonical_json_dumps(final_report), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(final_report), encoding="utf-8")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **raw_draws)

    stats = final_report["statistics"]
    print(f"Lot A A2 - git {final_report['git_sha'][:12]}")
    print(f"deployed_accuracy={stats['deployed_accuracy']:.4f} n_rows={stats['n_unique_rows']} k={stats['n_components']}")
    for arm in ARMS:
        bca = stats["bootstrap"]["bca_headline"][arm]
        print(
            f"  {arm}: delta={stats['observed_delta'][arm]:.4f} "
            f"BCa=[{bca['lo']:.4f}, {bca['hi']:.4f}]"
        )
    print(f"reproducibility: hash_run_1={hash_1[:16]} hash_run_2={hash_2[:16]} matched={matched}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_npz}")

    if not matched:
        print(
            "WARNING: two in-process runs produced different reports — investigate "
            "before trusting this report",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

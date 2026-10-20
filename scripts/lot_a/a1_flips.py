#!/usr/bin/env python3
"""Lot A Phase A1 — flip/churn intersection analysis (docs/lot-a-plan-20260807.md).

Unit = (row_id, seed). correct(arm) = arm_topk[0] == label. Arms: b0
(b0_topk), c1 (c1_topk), r2 (candidate_topk). Transitions: T1 = B0->C1,
T2 = C1->R2, direct D = B0->R2. churn_T(seed) = set of row_id where
correct(arm) changes (either direction) under T, for that seed.

Pre-registered verdict rule (argos-hardened, blocker B6): the Jaccard
index of churn_T1/churn_T2 is NOT the verdict statistic. The verdict is
"instabilite partagee / un phenomene" if P(churn_T1 | churn_T2) >= 0.5
for >= 2 of the 3 seeds; otherwise "deux phenomenes distincts". Jaccard
is reported only as a secondary, descriptive statistic. This matters
because a strict-subset churn relationship (churn_T2 subset of churn_T1)
can have low Jaccard yet P(churn_T1 | churn_T2) == 1.0 — see
tests/test_a1_flips.py::test_strict_subset_churn_yields_un_phenomene.

Structural note: the *improvement* sets of T1 and T2 (not the net-gain
scalars) are disjoint by construction — a C1->R2 improvement requires C1
to be wrong (the pre-condition), while a B0->C1 improvement requires C1
to be right (the post-condition). This analysis therefore targets churn
(instability) overlap, never improvement overlap; the disjointness is
asserted, not merely claimed.

Argos review round 1 (no blockers, fixes applied): the pre-registered
verdict rule is never changed post-hoc. Instead, `verdict.sensitivity`
reports what the *reciprocal* rule (P(churn_T2 | churn_T1) >= 0.5) would
have concluded, side by side with the registered verdict, so directional
sensitivity is visible without touching the registered statistic.

Report scope: everything is computed per seed AND aggregated (min/mean/
max) across the 3 seeds — the 1959 rows are never pooled as though they
were 1959 independent units (they are 653 units x 3 seeds).

CPU only. Never writes under backend/codex_model/ (read-only runtime
guard: DeployedRuntime SHA-256 asserted unchanged before and after).
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lot_a import common  # noqa: E402

DEFAULT_OUTPUT_JSON = ROOT / "reports/lot-a-20260807/a1.json"
DEFAULT_OUTPUT_MD = ROOT / "reports/lot-a-20260807/a1.md"

DEFAULT_BOOTSTRAP_DRAWS = 2000
VERDICT_THRESHOLD = 0.5
# Minimum seeds required for the verdict is a parametric majority
# (len(seeds) // 2 + 1), computed in compute_verdict() — not a hardcoded
# constant, so the rule stays correct if the seed count ever changes.

ARM_FIELDS = {"B0": "b0_topk", "C1": "c1_topk", "R2": "candidate_topk"}
TRANSITION_ARMS = {
    "T1_B0_to_C1": ("b0_topk", "c1_topk"),
    "T2_C1_to_R2": ("c1_topk", "candidate_topk"),
    "D_B0_to_R2_direct": ("b0_topk", "candidate_topk"),
}


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data hygiene (report item 7)
# ---------------------------------------------------------------------------


def detect_malformed_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """A topk list with duplicate class indices is malformed. The whole row
    is excluded (not just the offending field) so every downstream
    transition operates on an identical row_id universe per seed — this is
    what makes the net(D) == net(T1) + net(T2) identity exact rather than
    approximate."""
    clean: list[dict] = []
    malformed: list[dict] = []
    for row in rows:
        bad_fields = [
            field for field in ("b0_topk", "c1_topk", "candidate_topk")
            if len(set(row[field])) != len(row[field])
        ]
        if bad_fields:
            malformed.append(
                {
                    "row_id": row["row_id"],
                    "seed": row["seed"],
                    "outer_fold": row["outer_fold"],
                    "malformed_fields": bad_fields,
                }
            )
        else:
            clean.append(row)
    return clean, malformed


def _row_id_prefix(row_id: str) -> str:
    return row_id.split(":", 1)[0] + ":" if ":" in row_id else "(no-prefix)"


def _sign(value: float | int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _classify_sign_agreement(signs: list[int]) -> str:
    """3/3 = unanimous nonzero sign; 2/3 = majority (>= half) share a
    nonzero sign; dispersed = no nonzero sign reaches a majority (includes
    the all-zero case, which carries no directional signal)."""
    if not signs:
        return "no_data"
    counts = Counter(signs)
    nonzero_counts = {sign: count for sign, count in counts.items() if sign != 0}
    if not nonzero_counts:
        return "dispersed"
    best_sign, best_count = max(nonzero_counts.items(), key=lambda item: item[1])
    if best_count == len(signs):
        return "3/3" if len(signs) == 3 else "unanimous"
    if best_count * 2 >= len(signs):
        return "2/3" if len(signs) == 3 else "majority"
    return "dispersed"


def _sign_test_exact_p(successes: int, n: int) -> float:
    """Two-sided exact sign test p-value under Binomial(n, 0.5):
    2 * min(P(X >= successes), P(X <= successes)), capped at 1.0.
    Descriptive/illustrative given the small n used here (n = number of
    seeds) — no confidence interval is derived from it."""
    if n == 0:
        return 1.0
    pmf = [comb(n, k) * (0.5**n) for k in range(n + 1)]
    p_le = sum(pmf[: successes + 1])
    p_ge = sum(pmf[successes:])
    return min(1.0, 2 * min(p_le, p_ge))


def correct(row: dict, arm_field: str) -> bool:
    return row[arm_field][0] == row["label"]


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def transition_sets(rows_by_row_id: dict[str, dict], before_field: str, after_field: str) -> dict:
    churn: set[str] = set()
    improvements: set[str] = set()
    degradations: set[str] = set()
    for row_id, row in rows_by_row_id.items():
        before = correct(row, before_field)
        after = correct(row, after_field)
        if before == after:
            continue
        churn.add(row_id)
        if after and not before:
            improvements.add(row_id)
        else:
            degradations.add(row_id)
    return {"churn": churn, "improvements": improvements, "degradations": degradations}


def _transition_summary(transition: dict, net: int) -> dict:
    return {
        "improvements": len(transition["improvements"]),
        "degradations": len(transition["degradations"]),
        "net": net,
        "churn_size": len(transition["churn"]),
    }


# ---------------------------------------------------------------------------
# Cluster bootstrap over provenance_component (report item 4, descriptive)
# ---------------------------------------------------------------------------


def cluster_bootstrap_conditionals(
    rows_by_row_id: dict[str, dict],
    component_of: dict[str, str],
    churn_t1: set[str],
    churn_t2: set[str],
    n_draws: int,
    rng_seed: tuple[int, int],
) -> dict:
    rows_by_component: dict[str, list[str]] = defaultdict(list)
    for row_id in rows_by_row_id:
        rows_by_component[component_of[row_id]].append(row_id)
    components = sorted(rows_by_component)
    n_components = len(components)

    n1_arr = np.empty(n_components, dtype=np.int64)
    n2_arr = np.empty(n_components, dtype=np.int64)
    n11_arr = np.empty(n_components, dtype=np.int64)
    nunion_arr = np.empty(n_components, dtype=np.int64)
    for i, component in enumerate(components):
        row_ids = rows_by_component[component]
        in1 = [row_id in churn_t1 for row_id in row_ids]
        in2 = [row_id in churn_t2 for row_id in row_ids]
        n1_arr[i] = sum(in1)
        n2_arr[i] = sum(in2)
        n11_arr[i] = sum(a and b for a, b in zip(in1, in2))
        nunion_arr[i] = sum(a or b for a, b in zip(in1, in2))

    if n_components == 0:
        empty = {"n_valid_draws": 0, "ci95_low": None, "ci95_high": None}
        return {
            "n_draws": n_draws,
            "n_components_resampled_per_draw": 0,
            "label": "descriptif - pas une inference de reference",
            "p_churn_t1_given_churn_t2": empty,
            "p_churn_t2_given_churn_t1": empty,
            "jaccard_secondary": empty,
        }

    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, n_components, size=(n_draws, n_components))
    n1_sum = n1_arr[draws].sum(axis=1)
    n2_sum = n2_arr[draws].sum(axis=1)
    n11_sum = n11_arr[draws].sum(axis=1)
    nunion_sum = nunion_arr[draws].sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        p12 = np.where(n2_sum > 0, n11_sum / np.maximum(n2_sum, 1), np.nan)
        p21 = np.where(n1_sum > 0, n11_sum / np.maximum(n1_sum, 1), np.nan)
        jac = np.where(nunion_sum > 0, n11_sum / np.maximum(nunion_sum, 1), np.nan)

    def summarize(arr: np.ndarray) -> dict:
        finite = arr[~np.isnan(arr)]
        if finite.size == 0:
            return {"n_valid_draws": 0, "ci95_low": None, "ci95_high": None}
        lo, hi = np.percentile(finite, [2.5, 97.5])
        return {"n_valid_draws": int(finite.size), "ci95_low": float(lo), "ci95_high": float(hi)}

    return {
        "n_draws": n_draws,
        "n_components_resampled_per_draw": n_components,
        "label": "descriptif - pas une inference de reference",
        "p_churn_t1_given_churn_t2": summarize(p12),
        "p_churn_t2_given_churn_t1": summarize(p21),
        "jaccard_secondary": summarize(jac),
    }


# ---------------------------------------------------------------------------
# Breakdowns (report item 6)
# ---------------------------------------------------------------------------


def breakdown_by(
    rows_by_row_id: dict[str, dict],
    key_fn: Callable[[dict], Any],
    transitions: dict[str, dict],
) -> dict[str, dict[str, dict[str, int]]]:
    groups: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {name: {"improvements": 0, "degradations": 0} for name in transitions}
    )
    for row_id, row in rows_by_row_id.items():
        group_key = str(key_fn(row))
        for name, sets in transitions.items():
            if row_id in sets["improvements"]:
                groups[group_key][name]["improvements"] += 1
            elif row_id in sets["degradations"]:
                groups[group_key][name]["degradations"] += 1
    return dict(groups)


def per_fold_deltas(rows_by_row_id: dict[str, dict]) -> dict[str, dict]:
    folds = sorted({row["outer_fold"] for row in rows_by_row_id.values()})
    result: dict[str, dict] = {}
    for fold in folds:
        fold_rows = [row for row in rows_by_row_id.values() if row["outer_fold"] == fold]
        n = len(fold_rows)
        acc_b0 = sum(correct(row, "b0_topk") for row in fold_rows) / n
        acc_c1 = sum(correct(row, "c1_topk") for row in fold_rows) / n
        acc_r2 = sum(correct(row, "candidate_topk") for row in fold_rows) / n
        result[str(fold)] = {
            "n_rows": n,
            "delta_accuracy_T1_c1_minus_b0": acc_c1 - acc_b0,
            "delta_accuracy_T2_r2_minus_c1": acc_r2 - acc_c1,
            "delta_accuracy_D_r2_minus_b0": acc_r2 - acc_b0,
        }
    return result


# ---------------------------------------------------------------------------
# Per-seed report
# ---------------------------------------------------------------------------


def compute_seed_report(
    seed: int,
    rows_by_row_id: dict[str, dict],
    bootstrap_draws: int,
    bootstrap_seed_base: int,
) -> dict:
    t1 = transition_sets(rows_by_row_id, *TRANSITION_ARMS["T1_B0_to_C1"])
    t2 = transition_sets(rows_by_row_id, *TRANSITION_ARMS["T2_C1_to_R2"])
    d = transition_sets(rows_by_row_id, *TRANSITION_ARMS["D_B0_to_R2_direct"])

    net_t1 = len(t1["improvements"]) - len(t1["degradations"])
    net_t2 = len(t2["improvements"]) - len(t2["degradations"])
    net_d = len(d["improvements"]) - len(d["degradations"])
    if net_d != net_t1 + net_t2:
        raise RuntimeError(
            f"arithmetic identity violated for seed {seed}: "
            f"net_D={net_d} != net_T1+net_T2={net_t1 + net_t2}"
        )

    gains_disjoint = not (t1["improvements"] & t2["improvements"])
    if not gains_disjoint:
        raise RuntimeError(
            f"structural invariant violated for seed {seed}: T1 and T2 improvement "
            "sets are not disjoint (should be impossible by construction)"
        )

    churn_t1, churn_t2 = t1["churn"], t2["churn"]
    intersection = churn_t1 & churn_t2
    union = churn_t1 | churn_t2

    p_t1_given_t2 = (len(intersection) / len(churn_t2)) if churn_t2 else None
    p_t2_given_t1 = (len(intersection) / len(churn_t1)) if churn_t1 else None
    jaccard = (len(intersection) / len(union)) if union else None

    component_of = {row_id: row["provenance_component"] for row_id, row in rows_by_row_id.items()}
    bootstrap = cluster_bootstrap_conditionals(
        rows_by_row_id,
        component_of,
        churn_t1,
        churn_t2,
        n_draws=bootstrap_draws,
        rng_seed=(bootstrap_seed_base, seed),
    )

    transitions_named = {"T1": t1, "T2": t2, "D": d}
    breakdowns = {
        "by_outer_fold": breakdown_by(rows_by_row_id, lambda r: r["outer_fold"], transitions_named),
        "by_provenance_component": breakdown_by(
            rows_by_row_id, lambda r: r["provenance_component"], transitions_named
        ),
        "by_class_name": breakdown_by(rows_by_row_id, lambda r: r["class_name"], transitions_named),
        "by_row_id_prefix": breakdown_by(
            rows_by_row_id, lambda r: _row_id_prefix(r["row_id"]), transitions_named
        ),
    }

    return {
        "seed": seed,
        "n_rows": len(rows_by_row_id),
        "transitions": {
            "T1_B0_to_C1": _transition_summary(t1, net_t1),
            "T2_C1_to_R2": _transition_summary(t2, net_t2),
            "D_B0_to_R2_direct": _transition_summary(d, net_d),
        },
        "arithmetic_identity_net_d_eq_t1_plus_t2": True,
        "gain_sets_disjoint_t1_t2": gains_disjoint,
        "churn_overlap": {
            "churn_t1_size": len(churn_t1),
            "churn_t2_size": len(churn_t2),
            "intersection_size": len(intersection),
            "union_size": len(union),
            "p_churn_t1_given_churn_t2": p_t1_given_t2,
            "p_churn_t2_given_churn_t1": p_t2_given_t1,
            "jaccard_secondary": jaccard,
            "verdict_condition_met_t1_given_t2_ge_threshold": (
                p_t1_given_t2 is not None and p_t1_given_t2 >= VERDICT_THRESHOLD
            ),
        },
        "cluster_bootstrap_descriptive": bootstrap,
        "breakdowns": breakdowns,
        "per_fold_delta_accuracy": per_fold_deltas(rows_by_row_id),
    }


# ---------------------------------------------------------------------------
# Verdict, aggregation, sign consistency (report items 3, 1, 5)
# ---------------------------------------------------------------------------


def _sensitivity_narrative(
    seeds: list[int],
    min_seeds: int,
    met_count: int,
    verdict: str,
    reciprocal_met_count: int,
    reciprocal_values: dict[int, float | None],
    met_reciprocal: dict[int, bool],
    reciprocal_would_be_un_phenomene: bool,
) -> str:
    total = len(seeds)
    reciprocal_verdict_label = (
        "un phenomene" if reciprocal_would_be_un_phenomene else "deux phenomenes distincts"
    )
    detail = ", ".join(
        f"{seed}: {reciprocal_values[seed]:.3f}" for seed in seeds if met_reciprocal[seed]
    )
    detail_suffix = f" ({detail})" if detail else ""

    primary_is_un_phenomene = verdict.startswith("un phenomene")
    if reciprocal_would_be_un_phenomene and not primary_is_un_phenomene:
        reading = (
            "Lecture substantielle : recouvrement partiel asymetrique -- la majorite "
            "du churn T1 est aussi churn T2, l'inverse non."
        )
    elif primary_is_un_phenomene and not reciprocal_would_be_un_phenomene:
        reading = (
            "Lecture substantielle : recouvrement partiel asymetrique -- la majorite "
            "du churn T2 est aussi churn T1, l'inverse non."
        )
    elif primary_is_un_phenomene and reciprocal_would_be_un_phenomene:
        reading = "Lecture substantielle : les deux regles (directe et reciproque) concordent."
    else:
        reading = (
            "Lecture substantielle : les deux regles (directe et reciproque) restent "
            "sous le seuil de majorite."
        )

    return (
        f"Regle pre-enregistree P(churn_T1|churn_T2) >= {VERDICT_THRESHOLD} : "
        f"{met_count}/{total} seeds => verdict \"{verdict}\". "
        f"Sensibilite : sous la regle reciproque P(churn_T2|churn_T1) >= "
        f"{VERDICT_THRESHOLD}, {reciprocal_met_count}/{total} seeds{detail_suffix} "
        f"auraient donne \"{reciprocal_verdict_label}\". {reading}"
    )


def compute_verdict(seed_reports: dict[int, dict]) -> dict:
    seeds = sorted(seed_reports)
    # Parametric majority, not a hardcoded constant: correct for any seed count.
    min_seeds = len(seeds) // 2 + 1

    met_primary = {
        seed: seed_reports[seed]["churn_overlap"]["verdict_condition_met_t1_given_t2_ge_threshold"]
        for seed in seeds
    }
    met_count = sum(met_primary.values())
    verdict = (
        "un phenomene (instabilite partagee)"
        if met_count >= min_seeds
        else "deux phenomenes distincts"
    )

    # Sensitivity (argos review round 1): the registered rule above is never
    # changed post-hoc. This reciprocal-rule readout is reported alongside
    # it, purely descriptive, to make directional sensitivity transparent.
    reciprocal_values = {
        seed: seed_reports[seed]["churn_overlap"]["p_churn_t2_given_churn_t1"] for seed in seeds
    }
    met_reciprocal = {
        seed: (value is not None and value >= VERDICT_THRESHOLD)
        for seed, value in reciprocal_values.items()
    }
    reciprocal_met_count = sum(met_reciprocal.values())
    reciprocal_would_be_un_phenomene = reciprocal_met_count >= min_seeds

    sensitivity = {
        "primary_rule": f"P(churn_T1|churn_T2) >= {VERDICT_THRESHOLD} pour >= {min_seeds}/{len(seeds)} seeds",
        "primary_seeds_meeting_condition": met_count,
        "primary_verdict": verdict,
        "reciprocal_rule": f"P(churn_T2|churn_T1) >= {VERDICT_THRESHOLD} pour >= {min_seeds}/{len(seeds)} seeds",
        "reciprocal_seeds_meeting_condition": reciprocal_met_count,
        "reciprocal_seeds_meeting_condition_detail": {
            str(seed): reciprocal_values[seed] for seed in seeds
        },
        "reciprocal_verdict_would_be": (
            "un phenomene (instabilite partagee)"
            if reciprocal_would_be_un_phenomene
            else "deux phenomenes distincts"
        ),
        "narrative": _sensitivity_narrative(
            seeds,
            min_seeds,
            met_count,
            verdict,
            reciprocal_met_count,
            reciprocal_values,
            met_reciprocal,
            reciprocal_would_be_un_phenomene,
        ),
    }

    return {
        "rule": (
            "instabilite partagee / un phenomene si P(churn_T1 | churn_T2) >= "
            f"{VERDICT_THRESHOLD} pour >= {min_seeds} des {len(seeds)} seeds (majorite) ; "
            "sinon deux phenomenes distincts. Jaccard n'est jamais la statistique "
            "de verdict (aboli, cf. B6). Regle pre-enregistree, non modifiee apres "
            "lecture des donnees (voir sensitivity pour la regle reciproque, "
            "descriptive uniquement)."
        ),
        "threshold": VERDICT_THRESHOLD,
        "min_seeds_required": min_seeds,
        "seeds_meeting_condition": met_count,
        "seeds_total": len(seeds),
        "seeds_meeting_condition_detail": {str(seed): met_primary[seed] for seed in seeds},
        "verdict": verdict,
        "sensitivity": sensitivity,
    }


def _agg(values: list[float | int | None]) -> dict | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {"min": min(present), "mean": sum(present) / len(present), "max": max(present)}


def aggregate_across_seeds(seed_reports: dict[int, dict]) -> dict:
    seeds = sorted(seed_reports)
    return {
        "net_T1": _agg([seed_reports[s]["transitions"]["T1_B0_to_C1"]["net"] for s in seeds]),
        "net_T2": _agg([seed_reports[s]["transitions"]["T2_C1_to_R2"]["net"] for s in seeds]),
        "net_D": _agg([seed_reports[s]["transitions"]["D_B0_to_R2_direct"]["net"] for s in seeds]),
        "churn_T1_size": _agg([seed_reports[s]["churn_overlap"]["churn_t1_size"] for s in seeds]),
        "churn_T2_size": _agg([seed_reports[s]["churn_overlap"]["churn_t2_size"] for s in seeds]),
        "p_churn_T1_given_churn_T2": _agg(
            [seed_reports[s]["churn_overlap"]["p_churn_t1_given_churn_t2"] for s in seeds]
        ),
        "p_churn_T2_given_churn_T1": _agg(
            [seed_reports[s]["churn_overlap"]["p_churn_t2_given_churn_t1"] for s in seeds]
        ),
        "jaccard_secondary": _agg(
            [seed_reports[s]["churn_overlap"]["jaccard_secondary"] for s in seeds]
        ),
    }


def aggregate_breakdowns(seed_reports: dict[int, dict], breakdown_name: str) -> dict:
    seeds = sorted(seed_reports)
    all_keys: set[str] = set()
    for seed in seeds:
        all_keys.update(seed_reports[seed]["breakdowns"][breakdown_name].keys())

    result: dict[str, dict] = {}
    for key in sorted(all_keys):
        result[key] = {}
        for transition in ("T1", "T2", "D"):
            imp_vals = []
            deg_vals = []
            for seed in seeds:
                data = seed_reports[seed]["breakdowns"][breakdown_name].get(key, {}).get(transition)
                imp_vals.append(data["improvements"] if data else 0)
                deg_vals.append(data["degradations"] if data else 0)
            result[key][transition] = {
                "improvements": _agg(imp_vals),
                "degradations": _agg(deg_vals),
            }
    return result


def sign_consistency_report(seed_reports: dict[int, dict]) -> dict:
    seeds = sorted(seed_reports)
    result: dict[str, dict] = {}
    for transition_key, delta_key in (
        ("T1_B0_to_C1", "delta_accuracy_T1_c1_minus_b0"),
        ("T2_C1_to_R2", "delta_accuracy_T2_r2_minus_c1"),
        ("D_B0_to_R2_direct", "delta_accuracy_D_r2_minus_b0"),
    ):
        seed_signs = [_sign(seed_reports[seed]["transitions"][transition_key]["net"]) for seed in seeds]
        overall_category = _classify_sign_agreement(seed_signs)
        successes = sum(1 for s in seed_signs if s > 0)
        alpha = 1 + successes
        beta = 1 + (len(seed_signs) - successes)
        posterior_mean = alpha / (alpha + beta)
        sign_test_exact_p = _sign_test_exact_p(successes, len(seed_signs))

        folds = sorted(
            {
                fold
                for seed in seeds
                for fold in seed_reports[seed]["per_fold_delta_accuracy"]
            }
        )
        per_fold_result: dict[str, dict] = {}
        category_counts: Counter[str] = Counter()
        for fold in folds:
            fold_signs = []
            for seed in seeds:
                fold_data = seed_reports[seed]["per_fold_delta_accuracy"].get(fold)
                if fold_data is None:
                    continue
                fold_signs.append(_sign(fold_data[delta_key]))
            category = _classify_sign_agreement(fold_signs) if fold_signs else "no_data"
            category_counts[category] += 1
            per_fold_result[fold] = {"seed_signs": fold_signs, "agreement_category": category}

        result[transition_key] = {
            "overall_seed_level": {
                "seeds_ordered": seeds,
                "seed_signs": seed_signs,
                "agreement_category": overall_category,
                "sign_test_exact_p": sign_test_exact_p,
                "sign_test_exact_p_label": (
                    f"bilateral, Binomial({len(seed_signs)}, 0.5) -- descriptif, aucun "
                    "intervalle de confiance calcule"
                ),
                "beta_1_1_posterior": {
                    "alpha": alpha,
                    "beta": beta,
                    "mean": posterior_mean,
                    "label": (
                        f"prior de reference uniforme, illustratif -- n={len(seed_signs)}, "
                        "aucun intervalle de confiance calcule"
                    ),
                },
            },
            "per_fold": per_fold_result,
            "per_fold_agreement_category_counts": dict(category_counts),
        }
    return result


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_core_report(
    rows: list[dict],
    runtime_dir: Path,
    bootstrap_seed_base: int,
    bootstrap_draws: int,
    prediction_rows_path: Path | None = None,
) -> dict:
    common.set_determinism(bootstrap_seed_base)

    runtime = common.DeployedRuntime(runtime_dir)
    runtime.assert_unchanged()  # before

    clean_rows, malformed = detect_malformed_rows(rows)

    rows_by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in clean_rows:
        seed, row_id = row["seed"], row["row_id"]
        if row_id in rows_by_seed[seed]:
            raise common.LotASchemaError(
                f"duplicate (row_id, seed) after hygiene filter: {row_id}, {seed}"
            )
        rows_by_seed[seed][row_id] = row

    seeds_present = sorted(rows_by_seed) if rows_by_seed else list(common.EXPECTED_SEEDS)
    seed_reports = {
        seed: compute_seed_report(
            seed, rows_by_seed.get(seed, {}), bootstrap_draws, bootstrap_seed_base
        )
        for seed in seeds_present
    }

    verdict = compute_verdict(seed_reports)
    aggregated = aggregate_across_seeds(seed_reports)
    sign_consistency = sign_consistency_report(seed_reports)
    aggregated_breakdowns = {
        name: aggregate_breakdowns(seed_reports, name)
        for name in ("by_outer_fold", "by_provenance_component", "by_class_name", "by_row_id_prefix")
    }

    runtime.assert_unchanged()  # after

    report: dict[str, Any] = {
        "schema_version": "lot-a-a1-v1",
        "git_sha": get_git_sha(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "definitions": {
            "unit": "(row_id, seed)",
            "correct_arm": "arm_topk[0] == label",
            "arms": {"b0": "b0_topk", "c1": "c1_topk", "r2": "candidate_topk"},
            "T1": "B0 -> C1",
            "T2": "C1 -> R2",
            "D": "B0 -> R2 (direct)",
        },
        "structural_note_improvement_sets_disjoint": (
            "Les ensembles d'ameliorations (pas les gains nets, qui sont des scalaires) "
            "de T1 et T2 sont disjoints par construction : une amelioration B0->C1 exige "
            "C1 correct (post-condition), une amelioration C1->R2 exige C1 incorrect "
            "(pre-condition). L'analyse porte donc sur le recouvrement du churn "
            "(instabilite), jamais sur le recouvrement des ameliorations."
        ),
        "bootstrap_seed_base": bootstrap_seed_base,
        "bootstrap_draws": bootstrap_draws,
        "runtime_guard": {
            "sha256_at_load": runtime.sha256_at_load,
            "verified_unchanged_before_analysis": True,
            "verified_unchanged_after_analysis": True,
        },
        "data_hygiene": {
            "malformed_row_count": len(malformed),
            "malformed_rows": malformed,
            "total_input_rows": len(rows),
            "clean_rows": len(clean_rows),
        },
        "verdict": verdict,
        "per_seed": {str(seed): seed_reports[seed] for seed in seeds_present},
        "aggregated": aggregated,
        "aggregated_breakdowns": aggregated_breakdowns,
        "sign_consistency": sign_consistency,
    }
    if prediction_rows_path is not None:
        report["inputs"] = {
            "prediction_rows_path": str(prediction_rows_path),
            "prediction_rows_sha256": common.sha256_file(prediction_rows_path),
        }
    return report


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Lot A - Phase A1 : intersection des flips B0->C1->R2")
    lines.append("")
    lines.append(f"git {report['git_sha'][:12]} - schema {report['schema_version']}")
    lines.append("")
    lines.append(
        "Langage descriptif uniquement : aucune formulation de promotion. "
        "Unite = (row_id, seed). correct(arm) = arm_topk[0] == label."
    )
    lines.append("")

    lines.append("## Regle de verdict pre-enregistree (avant les chiffres)")
    lines.append("")
    lines.append(report["verdict"]["rule"])
    lines.append("")
    lines.append(report["structural_note_improvement_sets_disjoint"])
    lines.append("")

    lines.append("## Hygiene des donnees")
    lines.append("")
    hygiene = report["data_hygiene"]
    lines.append(
        f"- Lignes en entree : {hygiene['total_input_rows']} ; lignes propres : "
        f"{hygiene['clean_rows']} ; lignes malformees (topk avec doublons, exclues) : "
        f"{hygiene['malformed_row_count']}"
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

    lines.append("## Verdict")
    lines.append("")
    verdict = report["verdict"]
    lines.append(
        f"- Seeds satisfaisant P(churn_T1 | churn_T2) >= {verdict['threshold']} : "
        f"{verdict['seeds_meeting_condition']}/{verdict['seeds_total']} "
        f"(majorite requise : {verdict['min_seeds_required']})"
    )
    lines.append(f"- **Verdict (regle pre-enregistree, non modifiee) : {verdict['verdict']}**")
    lines.append("")
    lines.append("### Sensibilite directionnelle (descriptif, ne modifie pas le verdict)")
    lines.append("")
    sensitivity = verdict["sensitivity"]
    lines.append(sensitivity["narrative"])
    lines.append("")
    lines.append(
        f"- Regle reciproque : {sensitivity['reciprocal_rule']} -> "
        f"{sensitivity['reciprocal_seeds_meeting_condition']}/{verdict['seeds_total']} "
        f"seeds, verdict qu'elle aurait donne : {sensitivity['reciprocal_verdict_would_be']}"
    )
    reciprocal_detail = sensitivity["reciprocal_seeds_meeting_condition_detail"]
    reciprocal_detail_str = ", ".join(
        f"{seed}: {value:.3f}" if value is not None else f"{seed}: n/a"
        for seed, value in reciprocal_detail.items()
    )
    lines.append(f"- P(churn_T2 | churn_T1) par seed : {reciprocal_detail_str}")
    lines.append("")

    lines.append("## Resultats par seed")
    lines.append("")
    lines.append(
        "| seed | n lignes | net T1 | net T2 | net D | churn T1 | churn T2 | churn D | "
        "P(T1\\|T2) | P(T2\\|T1) | Jaccard (secondaire) | IC95 descriptif P(T1\\|T2) | "
        "identite net(D)=T1+T2 | ameliorations T1/T2 disjointes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for seed_key, seed_report in report["per_seed"].items():
        t1 = seed_report["transitions"]["T1_B0_to_C1"]
        t2 = seed_report["transitions"]["T2_C1_to_R2"]
        d = seed_report["transitions"]["D_B0_to_R2_direct"]
        overlap = seed_report["churn_overlap"]
        boot = seed_report["cluster_bootstrap_descriptive"]["p_churn_t1_given_churn_t2"]
        ci = (
            f"[{boot['ci95_low']:.3f}, {boot['ci95_high']:.3f}]"
            if boot["ci95_low"] is not None
            else "n/a"
        )

        def fmt(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "n/a"

        lines.append(
            f"| {seed_key} | {seed_report['n_rows']} | {t1['net']} | {t2['net']} | {d['net']} | "
            f"{overlap['churn_t1_size']} | {overlap['churn_t2_size']} | {d['churn_size']} | "
            f"{fmt(overlap['p_churn_t1_given_churn_t2'])} | "
            f"{fmt(overlap['p_churn_t2_given_churn_t1'])} | "
            f"{fmt(overlap['jaccard_secondary'])} | {ci} | "
            f"{seed_report['arithmetic_identity_net_d_eq_t1_plus_t2']} | "
            f"{seed_report['gain_sets_disjoint_t1_t2']} |"
        )
    lines.append("")

    component_counts = {
        seed_report["cluster_bootstrap_descriptive"]["n_components_resampled_per_draw"]
        for seed_report in report["per_seed"].values()
    }
    if len(component_counts) == 1:
        component_count_str = str(next(iter(component_counts)))
    elif component_counts:
        component_count_str = f"{min(component_counts)}-{max(component_counts)} (variable selon seed)"
    else:
        component_count_str = "n/a"
    n_seeds = len(report["per_seed"])
    lines.append(
        f"IC bootstrap = descriptif ({component_count_str} composantes, "
        f"{report['bootstrap_draws']} tirages, percentile 95%) — n'est pas une "
        f"inference de reference sur ces {n_seeds} seeds."
    )
    lines.append("")

    lines.append("## Agregation min/moyenne/max sur les 3 seeds")
    lines.append("")
    agg = report["aggregated"]
    lines.append("| metrique | min | moyenne | max |")
    lines.append("|---|---|---|---|")
    for name, stats in agg.items():
        if stats is None:
            lines.append(f"| {name} | n/a | n/a | n/a |")
        else:
            lines.append(f"| {name} | {stats['min']:.3f} | {stats['mean']:.3f} | {stats['max']:.3f} |")
    lines.append("")

    lines.append("## Coherence inter-seeds (n=3, pas d'intervalle de confiance)")
    lines.append("")
    for transition_key, data in report["sign_consistency"].items():
        overall = data["overall_seed_level"]
        posterior = overall["beta_1_1_posterior"]
        lines.append(f"### {transition_key}")
        lines.append(
            f"- Signes par seed (ordre {overall['seeds_ordered']}) : {overall['seed_signs']} "
            f"-> categorie {overall['agreement_category']}"
        )
        lines.append(
            f"- Test des signes exact ({overall['sign_test_exact_p_label']}) : "
            f"p = {overall['sign_test_exact_p']:.3f}"
        )
        lines.append(
            f"- Posterieur Beta(1,1) : alpha={posterior['alpha']}, beta={posterior['beta']}, "
            f"moyenne={posterior['mean']:.3f} ({posterior['label']})"
        )
        n_folds = len(data["per_fold"])
        lines.append(
            f"- Repartition par fold ({n_folds} folds) des categories d'accord : "
            f"{data['per_fold_agreement_category_counts']}"
        )
        lines.append("")

    lines.append("## Ventilation par prefixe de row_id (external: vs legacy:)")
    lines.append("")
    prefix_breakdown = report["aggregated_breakdowns"]["by_row_id_prefix"]
    lines.append("| prefixe | transition | ameliorations (min/moy/max) | degradations (min/moy/max) |")
    lines.append("|---|---|---|---|")
    for prefix, transitions in sorted(prefix_breakdown.items()):
        for transition_name, counts in transitions.items():
            imp = counts["improvements"]
            deg = counts["degradations"]
            imp_str = f"{imp['min']:.1f}/{imp['mean']:.1f}/{imp['max']:.1f}" if imp else "n/a"
            deg_str = f"{deg['min']:.1f}/{deg['mean']:.1f}/{deg['max']:.1f}" if deg else "n/a"
            lines.append(f"| {prefix} | {transition_name} | {imp_str} | {deg_str} |")
    lines.append("")

    lines.append(
        "Les ventilations completes par outer_fold, provenance_component et class_name "
        "sont disponibles dans a1.json (cles aggregated_breakdowns et per_seed.*.breakdowns)."
    )
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
    parser.add_argument("--runtime-dir", type=Path, default=common.DEPLOYED_RUNTIME_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rows = common.load_prediction_rows(args.prediction_rows)

    report_1 = build_core_report(
        rows, args.runtime_dir, args.seed, args.bootstrap_draws, args.prediction_rows
    )
    report_2 = build_core_report(
        rows, args.runtime_dir, args.seed, args.bootstrap_draws, args.prediction_rows
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

    print(f"Lot A A1 - git {final_report['git_sha'][:12]}")
    print(f"verdict: {final_report['verdict']['verdict']}")
    print(f"reproducibility: hash_run_1={hash_1[:16]} hash_run_2={hash_2[:16]} matched={matched}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")

    if not matched:
        print(
            "WARNING: two in-process runs produced different reports — investigate "
            "before trusting this report",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

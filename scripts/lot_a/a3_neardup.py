#!/usr/bin/env python3
"""Lot A Phase A3 — near-duplicate audit + provenance census
(docs/lot-a-plan-20260807.md, Phase A3 + Revisions post-critique item 8).

Four descriptive analyses over the full-data v11 corpus (9990 rows, S/14
384-d `features`, dinov2_vits14, white-pad, image_size 224 — same backbone
and feature space as the v9 OOF caches consumed by A1/A2; confirmed
bit-identical for shared row_id, see `load_oof_universe`):

1. Exact duplicates: cluster rows by `decoded_pixel_sha256` (present
   natively in the v11 cache — no bit-identical-feature fallback needed
   here, the field exists). Pairs inside a cluster that cross `fold` or
   `component_id` are leakage candidates.
2. Cosine-NN near-duplicates on L2-normalized 384-d features: nearest
   non-self neighbor per row (auxiliary, feeds the knee detector only) plus
   a pre-registered, independent threshold sweep {0.95, 0.97, 0.99, 0.999}
   over ALL row pairs (not just NN pairs) — each pair counted once
   (upper-triangle, i<j), counted independently per threshold (not
   exclusive bands). Knee detection on the sorted NN-similarity curve is
   reported as AUXILIARY ONLY (Revision item 8): the 4 sweep thresholds are
   the pre-registered result, never a post-hoc single cut.
3. OOF-vs-corpus leakage probe (the A2 follow-up): for each of the 653 OOF
   rows (base_features loaded from the v9 fold caches, sha-verified, same
   pattern as A2's compute_deployed_predictions), the nearest neighbor in
   the FULL corpus excluding the corpus row sharing that row_id (row_id is
   unique in the v11 corpus — verified below; OOF base_features and v11
   corpus features are bit-identical for a shared row_id, verified via
   decoded_pixel_sha256 equality during development). Distribution reported
   stratified by row_id prefix (external: vs legacy:) — this is the direct
   numeric follow-up to A2's finding that deployed accuracy is 0.9955 on
   legacy: rows vs 0.8376 on external: rows.
4. Provenance census: component -> row-count distribution over the full
   corpus, comparison to the 25 provenance components pre-specified by the
   653-row instrument subset (Jaccard of component-ID sets + size
   distribution summary, per Revision item 8), components split across
   folds flagged.

Purely descriptive throughout: distributions and counts are reported, the
OOF-vs-corpus hypothesis (near-dup/same-provenance neighbors explaining
legacy: accuracy) is stated, but no causal verdict is drawn.

Abort/degradation (docs/lot-a-plan-20260807.md, Phase A3 "Abort"): if the
full-data v11 cache is absent, A3 falls back to an exact-dup-only partial
audit computed from prediction_rows.jsonl's decoded_pixel_sha256 field
(653 unique rows) — near-dup, census, and the OOF-vs-corpus probe all need
the full corpus and are skipped, explicitly. If the full-data cache IS
present but the 653-row OOF universe (prediction_rows.jsonl + the v9 fold
caches) is unavailable or fails its own hygiene checks, items 1/2/4 (which
only need the full-data cache) still run to completion; only item 3 and
the census's instrument-vs-corpus comparison are skipped, explicitly (this
is a documented extension beyond the plan's literal "exact-dup seul"
wording, chosen because items 1/2/4 have no OOF dependency — stated here
so the deviation is visible, not silent).

CPU only. Never writes under backend/codex_model/ (DeployedRuntime SHA-256
asserted unchanged before and after, per the plan's per-phase Gate 4, even
though A3's computations do not otherwise use the runtime).
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lot_a import common  # noqa: E402

DEFAULT_OUTPUT_JSON = ROOT / "reports/lot-a-20260807/a3.json"
DEFAULT_OUTPUT_MD = ROOT / "reports/lot-a-20260807/a3.md"

FULL_DATA_V11_CACHE_PATH = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-full-data-v11/cache/full-data-views08.pt"
)

# Pre-registered sweep (plan Phase A3 item 2 + Revision item 8): the 4
# thresholds are the result, never a post-hoc single cut. Not a CLI flag —
# changing this after seeing the data would defeat the pre-registration.
SWEEP_THRESHOLDS = (0.95, 0.97, 0.99, 0.999)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_EXAMPLE_CAP = 50


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
    never emits the non-standard NaN/Infinity tokens (mirrors a2_bootstrap's
    helper of the same name)."""
    if isinstance(value, dict):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_floats(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _strip_wall_time_fields(report: dict) -> dict:
    """Wall-clock timings (near_duplicates/oof_vs_corpus_probe.wall_time_seconds)
    are logged for the "log wall time" requirement but are, by construction,
    never bit-reproducible across runs — they must be excluded from the
    reproducibility hash, the same way A1/A2's reports have no such field to
    begin with. Returns a shallow-enough copy; only the two known
    wall-clock leaves are touched, everything else is shared by reference
    (never written back)."""
    stripped = dict(report)
    for key in ("near_duplicates", "oof_vs_corpus_probe"):
        section = stripped.get(key)
        if isinstance(section, dict) and "wall_time_seconds" in section:
            stripped[key] = {k: v for k, v in section.items() if k != "wall_time_seconds"}
    return stripped


def _factorize(values: list[str]) -> np.ndarray:
    """Deterministic first-seen-order string -> int-code mapping (stable
    across identical inputs, which is all the canonical-JSON reproducibility
    check requires)."""
    codes: dict[str, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        if value not in codes:
            codes[value] = len(codes)
        out[i] = codes[value]
    return out


# ---------------------------------------------------------------------------
# Item 1 — exact duplicates (decoded_pixel_sha256 clustering)
# ---------------------------------------------------------------------------


def exact_duplicate_clusters(
    row_ids: list[str],
    shas: list[str],
    components: list[str],
    folds: list[int] | None,
    example_cap: int,
) -> dict:
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, sha in enumerate(shas):
        by_sha[sha].append(index)
    clusters = {sha: indices for sha, indices in by_sha.items() if len(indices) > 1}

    size_distribution = Counter(len(indices) for indices in clusters.values())
    total_rows_in_clusters = sum(len(indices) for indices in clusters.values())

    cross_component_examples: list[dict] = []
    cross_fold_examples: list[dict] = []
    n_cross_component = 0
    n_cross_fold = 0

    for sha, indices in clusters.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                comp_cross = components[i] != components[j]
                fold_cross = folds is not None and folds[i] != folds[j]
                if comp_cross:
                    n_cross_component += 1
                    if len(cross_component_examples) < example_cap:
                        cross_component_examples.append(
                            {
                                "decoded_pixel_sha256": sha,
                                "row_id_a": row_ids[i],
                                "row_id_b": row_ids[j],
                                "component_a": components[i],
                                "component_b": components[j],
                                "fold_a": folds[i] if folds is not None else None,
                                "fold_b": folds[j] if folds is not None else None,
                            }
                        )
                if fold_cross:
                    n_cross_fold += 1
                    if len(cross_fold_examples) < example_cap:
                        cross_fold_examples.append(
                            {
                                "decoded_pixel_sha256": sha,
                                "row_id_a": row_ids[i],
                                "row_id_b": row_ids[j],
                                "fold_a": folds[i],
                                "fold_b": folds[j],
                                "component_a": components[i],
                                "component_b": components[j],
                            }
                        )

    return {
        "hash_field": "decoded_pixel_sha256",
        "n_rows_analyzed": len(row_ids),
        "n_clusters_size_ge_2": len(clusters),
        "cluster_size_distribution": {
            str(size): count for size, count in sorted(size_distribution.items())
        },
        "total_rows_in_duplicate_clusters": total_rows_in_clusters,
        "n_cross_component_pairs": n_cross_component,
        "n_cross_fold_pairs": n_cross_fold if folds is not None else None,
        "example_cap": example_cap,
        "cross_component_pair_examples": cross_component_examples,
        "cross_fold_pair_examples": cross_fold_examples if folds is not None else None,
    }


def exact_duplicates_fallback_from_prediction_rows(rows: list[dict], example_cap: int) -> dict:
    """Full-data-cache-absent fallback (plan Abort clause): dedupe the 1959
    prediction_rows.jsonl records to the 653 unique row_ids first (they
    repeat 3x for seeds, with identical decoded_pixel_sha256 per row_id),
    then run the exact same clustering as the primary path."""
    unique: dict[str, dict] = {}
    for row in rows:
        row_id = row["row_id"]
        if row_id not in unique:
            unique[row_id] = row
    row_ids = sorted(unique)
    shas = [unique[rid]["decoded_pixel_sha256"] for rid in row_ids]
    components = [unique[rid]["provenance_component"] for rid in row_ids]
    folds = [unique[rid]["outer_fold"] for rid in row_ids]
    return exact_duplicate_clusters(row_ids, shas, components, folds, example_cap)


# ---------------------------------------------------------------------------
# Item 2 — cosine-NN near-duplicates + pre-registered sweep + knee (auxiliary)
# ---------------------------------------------------------------------------


def knee_detection(sorted_desc: np.ndarray) -> dict:
    """Simplified kneedle: max perpendicular distance from the [0,1]x[0,1]
    -normalized sorted-descending curve to the chord joining its first and
    last point. AUXILIARY ONLY (Revision item 8) — never used to pick the
    sweep thresholds, which are fixed and pre-registered above."""
    n = len(sorted_desc)
    if n < 3:
        return {"available": False, "reason": "moins de 3 points"}

    x = np.arange(n, dtype=np.float64)
    x_norm = x / (n - 1)
    y = sorted_desc.astype(np.float64)
    y_min, y_max = float(y.min()), float(y.max())
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)

    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    a = y1 - y0
    b = -(x1 - x0)
    c = x1 * y0 - y1 * x0
    denom = math.sqrt(a * a + b * b)
    distances = np.abs(a * x_norm + b * y_norm + c) / denom if denom > 0 else np.zeros_like(y)
    knee_pos = int(np.argmax(distances))

    return {
        "available": True,
        "auxiliary_only": True,
        "method": (
            "kneedle simplifie : distance maximale a la corde reliant le premier et le "
            "dernier point de la courbe NN-similarite triee decroissante, x et y "
            "normalises [0,1]"
        ),
        "knee_rank_index": knee_pos,
        "knee_similarity_value": float(sorted_desc[knee_pos]),
        "n_points": n,
    }


def near_duplicate_analysis(
    features: torch.Tensor,
    row_ids: list[str],
    components: list[str],
    folds: list[int] | None,
    thresholds: tuple[float, ...],
    chunk_size: int,
    example_cap: int,
) -> dict:
    t0 = time.perf_counter()

    norm = F.normalize(features.to(torch.float32), p=2, dim=1)
    n_rows = norm.shape[0]
    comp_codes = _factorize(components)
    fold_codes = np.asarray(folds, dtype=np.int64) if folds is not None else None

    nn_sim = np.full(n_rows, -2.0, dtype=np.float64)
    threshold_stats = {
        t: {"n_pairs": 0, "n_cross_component_pairs": 0, "n_cross_fold_pairs": 0, "examples": []}
        for t in thresholds
    }

    col_idx = np.arange(n_rows).reshape(1, -1)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = norm[start:end]
        sim = (chunk @ norm.t()).numpy().astype(np.float64)  # [c, n_rows]
        rows_global = np.arange(start, end)

        # Nearest non-self neighbor (auxiliary — feeds the knee curve only).
        sim_nn = sim.copy()
        sim_nn[np.arange(end - start), rows_global] = -2.0
        nn_sim[start:end] = sim_nn.max(axis=1)

        # Pair sweep: strict upper triangle (j > global row i), each pair
        # counted exactly once; every threshold counted independently
        # (not exclusive bands).
        upper_mask = col_idx > rows_global.reshape(-1, 1)
        comp_row = comp_codes[rows_global].reshape(-1, 1)
        comp_col = comp_codes.reshape(1, -1)
        cross_component = comp_row != comp_col
        if fold_codes is not None:
            fold_row = fold_codes[rows_global].reshape(-1, 1)
            fold_col = fold_codes.reshape(1, -1)
            cross_fold = fold_row != fold_col
        else:
            cross_fold = None

        for t in thresholds:
            sel = (sim >= t) & upper_mask
            n_pairs = int(sel.sum())
            if n_pairs == 0:
                continue
            stat = threshold_stats[t]
            stat["n_pairs"] += n_pairs
            stat["n_cross_component_pairs"] += int((sel & cross_component).sum())
            if cross_fold is not None:
                stat["n_cross_fold_pairs"] += int((sel & cross_fold).sum())

            examples = stat["examples"]
            if len(examples) < example_cap:
                rr, cc = np.nonzero(sel)
                for k in range(len(rr)):
                    if len(examples) >= example_cap:
                        break
                    i_global = int(rows_global[rr[k]])
                    j_global = int(cc[k])
                    examples.append(
                        {
                            "row_id_a": row_ids[i_global],
                            "row_id_b": row_ids[j_global],
                            "cosine_similarity": float(sim[rr[k], cc[k]]),
                            "cross_component": bool(components[i_global] != components[j_global]),
                            "cross_fold": (
                                bool(folds[i_global] != folds[j_global])
                                if folds is not None
                                else None
                            ),
                        }
                    )

    wall_time = time.perf_counter() - t0

    finite_nn = nn_sim[nn_sim > -2.0]
    if finite_nn.size:
        percentiles = {
            str(p): float(np.percentile(finite_nn, p)) for p in (1, 5, 25, 50, 75, 95, 99)
        }
        nn_summary = {
            "min": float(finite_nn.min()),
            "max": float(finite_nn.max()),
            "mean": float(finite_nn.mean()),
            "median": float(np.median(finite_nn)),
            "percentiles": percentiles,
        }
        knee = knee_detection(np.sort(finite_nn)[::-1])
    else:
        nn_summary = {"min": None, "max": None, "mean": None, "median": None, "percentiles": {}}
        knee = {"available": False, "reason": "aucune similarite NN calculee"}

    sweep = {}
    for t in thresholds:
        stat = threshold_stats[t]
        sweep[f"{t}"] = {
            "threshold": t,
            "n_pairs": stat["n_pairs"],
            "n_cross_component_pairs": stat["n_cross_component_pairs"],
            "n_cross_fold_pairs": stat["n_cross_fold_pairs"] if folds is not None else None,
            "example_cap": example_cap,
            "example_pairs": stat["examples"],
        }

    return {
        "status": "ok",
        "n_rows": n_rows,
        "chunk_size": chunk_size,
        "wall_time_seconds": wall_time,
        "nn_similarity_summary_auxiliary": nn_summary,
        "knee_detection_auxiliary": knee,
        "sweep_note": (
            "seuils pre-enregistres {0.95, 0.97, 0.99, 0.999} (item 8) ; chaque seuil "
            "compte les paires independamment (pas de bandes exclusives) ; le coude "
            "ci-dessus est auxiliaire uniquement, jamais un seuil unique choisi post-hoc"
        ),
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------
# Item 3 — OOF-vs-corpus leakage probe
# ---------------------------------------------------------------------------


def load_oof_universe(
    prediction_rows_path: Path, v9_caches_dir: Path
) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """Same two-source join as a2_bootstrap.compute_deployed_predictions:
    row metadata (label/component/fold/class_name) from prediction_rows.jsonl,
    base_features from the sha-verified v9 fold OOF caches. Raises
    LotASchemaError / FileNotFoundError on any hygiene violation — callers
    treat that as "OOF universe unavailable", not a silent partial result."""
    rows = common.load_prediction_rows(prediction_rows_path)
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
                    f"row_id {row_id!r} has inconsistent metadata across seed duplicates: "
                    f"{unique_meta[row_id]} != {entry}"
                )
        else:
            unique_meta[row_id] = entry

    oof_features_by_row: dict[str, torch.Tensor] = {}
    for fold in common.EXPECTED_FOLDS:
        cache_path = v9_caches_dir / f"fold-{fold:02d}-views08.pt"
        if not cache_path.is_file():
            continue
        sha_ok, actual_sha, expected_sha = common.verify_cache_sha(cache_path)
        if not sha_ok:
            raise common.LotASchemaError(
                f"fold {fold} cache sha256 mismatch: actual={actual_sha} expected={expected_sha}"
            )
        cache = common.load_fold_cache(cache_path)
        oof = cache["oof"]
        oof_row_ids = list(oof["row_id"])
        base_features = oof["base_features"].to(torch.float32)
        for idx, row_id in enumerate(oof_row_ids):
            if row_id not in unique_meta:
                continue
            meta = unique_meta[row_id]
            if meta["fold"] != fold:
                raise common.LotASchemaError(
                    f"row_id {row_id!r}: prediction_rows outer_fold={meta['fold']} but found "
                    f"in fold-{fold:02d} oof cache"
                )
            if row_id in oof_features_by_row:
                raise common.LotASchemaError(
                    f"row_id {row_id!r} appears in more than one fold's oof cache"
                )
            oof_features_by_row[row_id] = base_features[idx]

    missing = sorted(set(unique_meta) - set(oof_features_by_row))
    if missing:
        raise common.LotASchemaError(
            f"{len(missing)} row_id(s) from prediction_rows not found in any v9 fold oof "
            f"cache: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return unique_meta, oof_features_by_row


def oof_vs_corpus_probe(
    unique_meta: dict[str, dict],
    oof_features_by_row: dict[str, torch.Tensor],
    corpus_row_ids: list[str],
    corpus_features: torch.Tensor,
    corpus_components: list[str],
    corpus_folds: list[int] | None,
    thresholds: tuple[float, ...],
) -> dict:
    t0 = time.perf_counter()

    corpus_index: dict[str, int] = {}
    for i, rid in enumerate(corpus_row_ids):
        if rid in corpus_index:
            raise common.LotASchemaError(
                f"corpus row_id {rid!r} is not unique — row_id-based self-exclusion is invalid"
            )
        corpus_index[rid] = i

    corpus_norm = F.normalize(corpus_features.to(torch.float32), p=2, dim=1)

    row_ids_sorted = sorted(unique_meta)
    matched_rows = [rid for rid in row_ids_sorted if rid in corpus_index]
    missing_rows = [rid for rid in row_ids_sorted if rid not in corpus_index]

    per_row: list[dict] = []
    for rid in matched_rows:
        oof_vec = F.normalize(oof_features_by_row[rid].unsqueeze(0), p=2, dim=1)
        sim = (oof_vec @ corpus_norm.t()).squeeze(0).numpy().astype(np.float64)
        self_idx = corpus_index[rid]
        sim[self_idx] = -2.0
        nn_idx = int(sim.argmax())
        nn_sim = float(sim[nn_idx])
        meta = unique_meta[rid]
        per_row.append(
            {
                "row_id": rid,
                "prefix": _row_id_prefix(rid),
                "component": meta["component"],
                "nn_row_id": corpus_row_ids[nn_idx],
                "nn_similarity": nn_sim,
                "nn_component": corpus_components[nn_idx],
                "same_component_as_query": corpus_components[nn_idx] == meta["component"],
                "nn_fold": corpus_folds[nn_idx] if corpus_folds is not None else None,
                "same_fold_as_query": (
                    (corpus_folds[nn_idx] == meta["fold"]) if corpus_folds is not None else None
                ),
            }
        )

    wall_time = time.perf_counter() - t0

    stratified: dict[str, dict] = {}
    for prefix in sorted({r["prefix"] for r in per_row}):
        group = [r for r in per_row if r["prefix"] == prefix]
        sims = np.array([r["nn_similarity"] for r in group], dtype=np.float64)
        stratified[prefix] = {
            "n": len(group),
            "median_nn_similarity": float(np.median(sims)) if sims.size else None,
            "mean_nn_similarity": float(np.mean(sims)) if sims.size else None,
            "counts_ge_threshold": {
                f"{t}": int((sims >= t).sum()) for t in thresholds
            },
            "n_same_component_as_query": sum(1 for r in group if r["same_component_as_query"]),
            "n_same_fold_as_query": (
                sum(1 for r in group if r["same_fold_as_query"])
                if corpus_folds is not None
                else None
            ),
        }

    return {
        "status": "ok" if not missing_rows else "partial",
        "hypothesis_under_test": (
            "A2 a trouve une accuracy deployee de 0.9955 sur les lignes legacy: contre 0.8376 "
            "sur external: (reports/lot-a-20260807/a2.md). Cette sonde teste si les lignes "
            "legacy: ont un plus proche voisin dans le corpus d'entrainement complet plus "
            "proche (quasi-duplicat) ou de meme provenance que les lignes external: — purement "
            "descriptif, aucun verdict causal n'est tire ici."
        ),
        "exclusion_rule": (
            "pour chaque ligne OOF, le plus proche voisin est cherche dans le corpus full-data "
            "complet en excluant la ligne du corpus partageant le meme row_id (row_id verifie "
            "unique dans le corpus ci-dessus) ; les features OOF (v9 base_features) et les "
            "features corpus (v11 full-data) sont bit-identiques pour un row_id partage "
            "(verifie en developpement via egalite exacte de decoded_pixel_sha256 et des "
            "tenseurs de features), donc cette exclusion par row_id equivaut a exclure la "
            "ligne elle-meme."
        ),
        "n_oof_rows": len(row_ids_sorted),
        "n_matched_to_corpus": len(matched_rows),
        "n_missing_from_corpus": len(missing_rows),
        "missing_row_ids": missing_rows,
        "wall_time_seconds": wall_time,
        "per_row": per_row,
        "stratified_by_prefix": stratified,
    }


# ---------------------------------------------------------------------------
# Item 4 — provenance census
# ---------------------------------------------------------------------------


def provenance_census(
    components: list[str],
    folds: list[int] | None,
    instrument_components: list[str] | None,
) -> dict:
    counts = Counter(components)
    sizes = np.array(sorted(counts.values()), dtype=np.int64)
    size_percentiles = (
        {str(p): float(np.percentile(sizes, p)) for p in (5, 25, 50, 75, 95)} if sizes.size else {}
    )

    split_across_folds: dict[str, list[int]] = {}
    if folds is not None:
        comp_folds: dict[str, set[int]] = defaultdict(set)
        for comp, fold in zip(components, folds):
            comp_folds[comp].add(fold)
        for comp, fold_set in comp_folds.items():
            if len(fold_set) > 1:
                split_across_folds[comp] = sorted(fold_set)

    result: dict[str, Any] = {
        "full_corpus_row_count": len(components),
        "full_corpus_component_count": len(counts),
        "component_size_distribution_summary": {
            "min": int(sizes.min()) if sizes.size else None,
            "max": int(sizes.max()) if sizes.size else None,
            "mean": float(sizes.mean()) if sizes.size else None,
            "percentiles": size_percentiles,
        },
        "component_row_counts": {comp: int(n) for comp, n in sorted(counts.items())},
        "components_split_across_folds": {
            comp: folds_list for comp, folds_list in sorted(split_across_folds.items())
        },
        "n_components_split_across_folds": len(split_across_folds),
    }

    if instrument_components is not None:
        instrument_set = set(instrument_components)
        corpus_set = set(counts)
        intersection = instrument_set & corpus_set
        union = instrument_set | corpus_set
        instrument_sizes = np.array(
            sorted(counts[c] for c in instrument_set if c in counts), dtype=np.int64
        )
        result["instrument_comparison"] = {
            "status": "ok",
            "instrument_component_count": len(instrument_set),
            "instrument_component_ids": sorted(instrument_set),
            "n_instrument_components_in_corpus": len(intersection),
            "n_instrument_components_not_in_corpus": len(instrument_set - corpus_set),
            "jaccard_instrument_vs_corpus": (
                len(intersection) / len(union) if union else None
            ),
            "instrument_component_size_in_corpus_summary": {
                "min": int(instrument_sizes.min()) if instrument_sizes.size else None,
                "max": int(instrument_sizes.max()) if instrument_sizes.size else None,
                "mean": float(instrument_sizes.mean()) if instrument_sizes.size else None,
            },
        }
    else:
        result["instrument_comparison"] = {
            "status": "skipped",
            "reason": "univers OOF des 653 lignes / 25 composantes indisponible",
        }

    return result


# ---------------------------------------------------------------------------
# Report assembly (IO + orchestration)
# ---------------------------------------------------------------------------


def build_core_report(
    full_data_cache_path: Path,
    prediction_rows_path: Path,
    v9_caches_dir: Path,
    runtime_dir: Path,
    seed: int,
    chunk_size: int,
    example_cap: int,
) -> dict:
    common.set_determinism(seed)
    runtime = common.DeployedRuntime(runtime_dir)
    runtime.assert_unchanged()  # before

    report: dict[str, Any] = {
        "schema_version": "lot-a-a3-v1",
        "git_sha": get_git_sha(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "runtime_guard": {
            "sha256_at_load": runtime.sha256_at_load,
            "verified_unchanged_before_analysis": True,
            "verified_unchanged_after_analysis": True,
        },
        "sweep_thresholds": list(SWEEP_THRESHOLDS),
    }

    if not full_data_cache_path.is_file():
        try:
            rows = common.load_prediction_rows(prediction_rows_path)
            exact = exact_duplicates_fallback_from_prediction_rows(rows, example_cap)
            fallback_status = "partial"
            fallback_reason = None
        except FileNotFoundError as exc:
            exact = None
            fallback_status = "blocked"
            fallback_reason = str(exc)

        runtime.assert_unchanged()  # after
        report.update(
            {
                "status": fallback_status,
                "abort_reason": "cache full-data (v11) absent au chemin attendu",
                "full_data_cache_path": str(full_data_cache_path),
                "note": (
                    "Audit partiel : exact-dup seul, calcule sur decoded_pixel_sha256 de "
                    "prediction_rows.jsonl (653 lignes uniques) faute de cache full-data "
                    "(docs/lot-a-plan-20260807.md, Phase A3, Abort). Near-dup cosine, census "
                    "de provenance et sonde OOF-vs-corpus non executes (necessitent le corpus "
                    "complet)."
                    if fallback_status == "partial"
                    else f"Aucun repli possible : {fallback_reason}"
                ),
                "exact_duplicates": exact,
                "near_duplicates": {
                    "status": "skipped",
                    "reason": "corpus full-data indisponible",
                },
                "oof_vs_corpus_probe": {
                    "status": "skipped",
                    "reason": "corpus full-data indisponible",
                },
                "provenance_census": {
                    "status": "skipped",
                    "reason": "corpus full-data indisponible",
                },
            }
        )
        return _sanitize_floats(report)

    sha_ok, actual_sha, expected_sha = common.verify_cache_sha(full_data_cache_path)
    if not sha_ok:
        raise common.LotASchemaError(
            f"full-data cache sha256 mismatch: actual={actual_sha} expected={expected_sha}"
        )
    cache = torch.load(full_data_cache_path, map_location="cpu", weights_only=False)

    required_keys = ("row_id", "decoded_pixel_sha256", "component_id", "features")
    missing_keys = [key for key in required_keys if key not in cache]
    if missing_keys:
        raise common.LotASchemaError(f"full-data cache missing required keys: {missing_keys}")

    row_ids: list[str] = list(cache["row_id"])
    shas: list[str] = list(cache["decoded_pixel_sha256"])
    components: list[str] = list(cache["component_id"])
    folds: list[int] | None = list(cache["fold"]) if "fold" in cache else None
    features: torch.Tensor = cache["features"]

    if not torch.isfinite(features.to(torch.float32)).all():
        raise common.LotASchemaError("full-data cache features contain non-finite values")

    exact = exact_duplicate_clusters(row_ids, shas, components, folds, example_cap)
    near = near_duplicate_analysis(
        features, row_ids, components, folds, SWEEP_THRESHOLDS, chunk_size, example_cap
    )

    try:
        unique_meta, oof_features_by_row = load_oof_universe(prediction_rows_path, v9_caches_dir)
        oof_skip_reason: str | None = None
    except (FileNotFoundError, common.LotASchemaError) as exc:
        unique_meta, oof_features_by_row = None, None
        oof_skip_reason = str(exc)

    if unique_meta is not None:
        oof_probe = oof_vs_corpus_probe(
            unique_meta,
            oof_features_by_row,
            row_ids,
            features,
            components,
            folds,
            SWEEP_THRESHOLDS,
        )
        instrument_components = sorted({meta["component"] for meta in unique_meta.values()})
    else:
        oof_probe = {"status": "skipped", "reason": oof_skip_reason}
        instrument_components = None

    census = provenance_census(components, folds, instrument_components)

    runtime.assert_unchanged()  # after

    report.update(
        {
            "status": "ok",
            "inputs": {
                "full_data_cache_path": str(full_data_cache_path),
                "full_data_cache_sha256": actual_sha,
                "full_data_row_count": len(row_ids),
                "prediction_rows_path": str(prediction_rows_path),
                "prediction_rows_sha256": (
                    common.sha256_file(prediction_rows_path)
                    if prediction_rows_path.is_file()
                    else None
                ),
                "v9_caches_dir": str(v9_caches_dir),
            },
            "exact_duplicates": exact,
            "near_duplicates": near,
            "oof_vs_corpus_probe": oof_probe,
            "provenance_census": census,
        }
    )
    return _sanitize_floats(report)


# ---------------------------------------------------------------------------
# Markdown report (French, descriptive only)
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Lot A - Phase A3 : audit near-duplicate + census de provenance")
    lines.append("")
    lines.append(f"git {report['git_sha'][:12]} - schema {report['schema_version']}")
    lines.append("")
    lines.append(
        "Langage descriptif uniquement : aucune formulation de promotion. "
        "«Elements convertible en instrument equitable» n'est PAS conclu ici — "
        "seules les conditions chiffrees sont livrees."
    )
    lines.append("")

    guard = report["runtime_guard"]
    lines.append("## Garde runtime deploye")
    lines.append("")
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

    if report["status"] != "ok":
        lines.append(f"## Statut : {report['status'].upper()}")
        lines.append("")
        lines.append(report.get("note", report.get("abort_reason", "")))
        lines.append("")
        exact = report.get("exact_duplicates")
        if exact:
            lines.append("### Exact-dup (repli, sans corpus full-data)")
            lines.append("")
            lines.append(
                f"- Lignes analysees : {exact['n_rows_analyzed']} ; clusters (taille>=2) : "
                f"{exact['n_clusters_size_ge_2']} ; lignes dans un cluster : "
                f"{exact['total_rows_in_duplicate_clusters']}"
            )
            lines.append(
                f"- Paires traversant une composante : {exact['n_cross_component_pairs']} ; "
                f"traversant un fold : {exact['n_cross_fold_pairs']}"
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

    inputs = report["inputs"]
    lines.append("## Entrees")
    lines.append("")
    lines.append(
        f"- Cache full-data : `{inputs['full_data_cache_path']}` "
        f"({inputs['full_data_row_count']} lignes, sha256={inputs['full_data_cache_sha256'][:16]}...)"
    )
    lines.append("")

    exact = report["exact_duplicates"]
    lines.append("## 1. Doublons exacts (decoded_pixel_sha256)")
    lines.append("")
    lines.append(
        f"- Lignes analysees : {exact['n_rows_analyzed']} ; clusters (taille>=2) : "
        f"{exact['n_clusters_size_ge_2']} ; lignes dans un cluster de doublons : "
        f"{exact['total_rows_in_duplicate_clusters']}"
    )
    lines.append(f"- Distribution des tailles de cluster : {exact['cluster_size_distribution']}")
    lines.append(
        f"- Paires (dans un cluster de doublons exacts) traversant une composante de "
        f"provenance : **{exact['n_cross_component_pairs']}** ; traversant un fold : "
        f"**{exact['n_cross_fold_pairs']}**"
    )
    lines.append("")

    near = report["near_duplicates"]
    lines.append("## 2. Quasi-doublons cosinus (features S/14 384-d, L2-normalisees)")
    lines.append("")
    lines.append(
        f"- Lignes : {near['n_rows']} ; taille de bloc : {near['chunk_size']} ; "
        f"temps d'execution : {_fmt(near['wall_time_seconds'], 1)}s"
    )
    nn = near["nn_similarity_summary_auxiliary"]
    lines.append(
        f"- NN-similarite (auxiliaire, alimente uniquement la detection de coude) : "
        f"min={_fmt(nn['min'])}, mediane={_fmt(nn['median'])}, moyenne={_fmt(nn['mean'])}, "
        f"max={_fmt(nn['max'])}"
    )
    knee = near["knee_detection_auxiliary"]
    if knee.get("available"):
        lines.append(
            f"- Coude (AUXILIAIRE, item 8 — ne fait jamais foi) : rang={knee['knee_rank_index']}, "
            f"similarite={_fmt(knee['knee_similarity_value'])}"
        )
    lines.append("")
    lines.append(
        "| seuil (pre-enregistre) | n paires | paires cross-composante | paires cross-fold |"
    )
    lines.append("|---|---|---|---|")
    for t in report["sweep_thresholds"]:
        entry = near["sweep"][f"{t}"]
        lines.append(
            f"| {t} | {entry['n_pairs']} | {entry['n_cross_component_pairs']} | "
            f"{entry['n_cross_fold_pairs']} |"
        )
    lines.append("")

    probe = report["oof_vs_corpus_probe"]
    lines.append("## 3. Sonde OOF-vs-corpus (suivi d'A2)")
    lines.append("")
    if probe.get("status") == "skipped":
        lines.append(f"- Non execute : {probe.get('reason')}")
    else:
        lines.append(f"- Statut : {probe['status']} ; {probe['hypothesis_under_test']}")
        lines.append(f"- Regle d'exclusion : {probe['exclusion_rule']}")
        lines.append(
            f"- Lignes OOF : {probe['n_oof_rows']} ; appariees au corpus : "
            f"{probe['n_matched_to_corpus']} ; manquantes : {probe['n_missing_from_corpus']} "
            f"({_fmt(probe['wall_time_seconds'], 1)}s)"
        )
        lines.append("")
        lines.append(
            "| prefixe | n | mediane NN-sim | moyenne NN-sim | n>=0.99 | n>=0.999 | "
            "n meme composante |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for prefix, strat in sorted(probe["stratified_by_prefix"].items()):
            counts = strat["counts_ge_threshold"]
            lines.append(
                f"| {prefix} | {strat['n']} | {_fmt(strat['median_nn_similarity'])} | "
                f"{_fmt(strat['mean_nn_similarity'])} | {counts.get('0.99', 'n/a')} | "
                f"{counts.get('0.999', 'n/a')} | {strat['n_same_component_as_query']} |"
            )
    lines.append("")

    census = report["provenance_census"]
    lines.append("## 4. Census de provenance")
    lines.append("")
    lines.append(
        f"- Corpus complet : {census['full_corpus_row_count']} lignes, "
        f"{census['full_corpus_component_count']} composantes"
    )
    size_summary = census["component_size_distribution_summary"]
    lines.append(
        f"- Taille de composante : min={size_summary['min']}, moyenne="
        f"{_fmt(size_summary['mean'], 2)}, max={size_summary['max']}"
    )
    lines.append(
        f"- Composantes scindees entre folds : {census['n_components_split_across_folds']}"
    )
    instrument = census["instrument_comparison"]
    if instrument.get("status") == "ok":
        lines.append(
            f"- Instrument (25 composantes des 653 lignes OOF) : "
            f"{instrument['n_instrument_components_in_corpus']}/"
            f"{instrument['instrument_component_count']} presentes dans le corpus ; "
            f"Jaccard instrument-vs-corpus = {_fmt(instrument['jaccard_instrument_vs_corpus'])}"
        )
    else:
        lines.append(f"- Comparaison instrument-vs-corpus : {instrument.get('reason')}")
    lines.append("")
    lines.append(
        "Distributions completes (comptes par composante, exemples de paires, lignes OOF "
        "individuelles) disponibles dans a3.json."
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
    parser.add_argument("--full-data-cache", type=Path, default=FULL_DATA_V11_CACHE_PATH)
    parser.add_argument("--prediction-rows", type=Path, default=common.PREDICTION_ROWS_PATH)
    parser.add_argument("--v9-caches-dir", type=Path, default=common.V9_CACHES_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=common.DEPLOYED_RUNTIME_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--example-cap", type=int, default=DEFAULT_EXAMPLE_CAP)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report_1 = build_core_report(
        args.full_data_cache,
        args.prediction_rows,
        args.v9_caches_dir,
        args.runtime_dir,
        args.seed,
        args.chunk_size,
        args.example_cap,
    )
    report_2 = build_core_report(
        args.full_data_cache,
        args.prediction_rows,
        args.v9_caches_dir,
        args.runtime_dir,
        args.seed,
        args.chunk_size,
        args.example_cap,
    )
    hash_1 = common.sha256_json(_strip_wall_time_fields(report_1))
    hash_2 = common.sha256_json(_strip_wall_time_fields(report_2))
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

    print(f"Lot A A3 - git {final_report['git_sha'][:12]}")
    print(f"status: {final_report['status']}")
    if final_report["status"] == "ok":
        exact = final_report["exact_duplicates"]
        near = final_report["near_duplicates"]
        print(
            f"exact-dup clusters: {exact['n_clusters_size_ge_2']} "
            f"(cross-component pairs: {exact['n_cross_component_pairs']})"
        )
        for t in final_report["sweep_thresholds"]:
            entry = near["sweep"][f"{t}"]
            print(f"near-dup @ {t}: {entry['n_pairs']} pairs")
    print(f"reproducibility: hash_run_1={hash_1[:16]} hash_run_2={hash_2[:16]} matched={matched}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")

    if not matched:
        print(
            "WARNING: two in-process runs produced different reports — investigate before "
            "trusting this report",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

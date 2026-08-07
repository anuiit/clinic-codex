#!/usr/bin/env python3
"""Lot A Phase 0 preflight gates (docs/lot-a-plan-20260807.md).

Six checks, run against the frozen v19-r2 prediction rows, the v9 fold
caches, the v9/v19-r2 research checkpoints, the full-data caches, and the
deployed backend/codex_model runtime (read-only). Writes a deterministic
report to reports/lot-a-20260807/preflight.json: the script runs its own
check pipeline twice in-process and logs both canonical-JSON hashes so
non-determinism is caught rather than assumed away.

CPU only. Never writes under backend/codex_model/.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lot_a import common  # noqa: E402

V9_RUN_DIR = (
    ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
)
V9_CHECKPOINTS_DIR = V9_RUN_DIR / "iteration-0001/results/checkpoints"
# v9's own (pre-v19r2) prediction rows carry baseline/candidate checkpoint
# sha256 per fold+seed group — used to cross-check the checkpoints located
# under V9_CHECKPOINTS_DIR actually are the ones the b0_topk/c1_topk values
# in the v19-r2 prediction rows were produced from.
V9_ITERATION_PREDICTION_ROWS = V9_RUN_DIR / "iteration-0001/results/prediction_rows.jsonl"

FULL_DATA_V11_CACHE_DIR = (
    ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11/cache"
)
FULL_DATA_R2_CACHE_DIR = (
    ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260805-full-data-r2/cache"
)

DEFAULT_OUTPUT = ROOT / "reports/lot-a-20260807/preflight.json"

FULL_DATA_ROW_BOUNDS = (4405, 9990)

# Calibration protocol (argos-hardened): 10-row pilot first; a systematic
# pilot mismatch stops the gate before the 50-row primary run rather than
# spending compute on a checkpoint layout that clearly cannot reproduce the
# stored predictions. "Reproduction" = exact top-1 index match.
CALIBRATION_PILOT_N = 10
CALIBRATION_PILOT_MIN_MATCHES = 8
CALIBRATION_PRIMARY_N = 50
CALIBRATION_PRIMARY_MIN_MATCHES = 49
CALIBRATION_SEED = common.EXPECTED_SEEDS[0]

# Sourced from scripts/autoresearch_vicreg_backbone_v19r2.py::EXPECTED_RUNTIME_SHA256
# (the script that produced the v19-r2 prediction rows this preflight validates
# against) — logged here for an informational cross-check, not re-derived.
DOCUMENTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}


def _fail(status: str, **details: Any) -> dict:
    return {"status": status, **details}


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Check 1 — prediction_rows.jsonl cardinality and schema
# ---------------------------------------------------------------------------


def check_1_prediction_rows(rows: list[dict]) -> dict:
    total_rows = len(rows)
    row_ids = [row["row_id"] for row in rows]
    unique_row_ids = sorted(set(row_ids))

    row_id_seed_pairs = Counter((row["row_id"], row["seed"]) for row in rows)
    duplicate_pairs = sorted(
        [list(pair) for pair, count in row_id_seed_pairs.items() if count > 1]
    )

    seeds_present = sorted({row["seed"] for row in rows})
    folds_present = sorted({row["outer_fold"] for row in rows})

    row_ids_by_seed = {
        seed: sorted({row["row_id"] for row in rows if row["seed"] == seed})
        for seed in seeds_present
    }
    seeds_share_row_id_set = all(ids == unique_row_ids for ids in row_ids_by_seed.values())

    prefix_counts: Counter[str] = Counter()
    for row_id in row_ids:
        prefix = row_id.split(":", 1)[0] + ":" if ":" in row_id else "(no-prefix)"
        prefix_counts[prefix] += 1

    label_to_row_ids: dict[int, set[str]] = {}
    for row in rows:
        label_to_row_ids.setdefault(row["label"], set()).add(row["row_id"])
    distinct_label_count = len(label_to_row_ids)
    classes_with_min5_unique_rowids = sum(
        1 for ids in label_to_row_ids.values() if len(ids) >= 5
    )

    provenance_components = sorted({row["provenance_component"] for row in rows})

    reasons = []
    if total_rows != 1959:
        reasons.append(f"expected exactly 1959 rows, found {total_rows}")
    if len(unique_row_ids) != 653:
        reasons.append(f"expected exactly 653 unique row_id, found {len(unique_row_ids)}")
    if duplicate_pairs:
        reasons.append(f"{len(duplicate_pairs)} duplicate (row_id, seed) pairs")
    if seeds_present != list(common.EXPECTED_SEEDS):
        reasons.append(f"expected seeds {list(common.EXPECTED_SEEDS)}, found {seeds_present}")
    if not seeds_share_row_id_set:
        reasons.append("row_id set differs across seeds")
    if folds_present != list(common.EXPECTED_FOLDS):
        reasons.append(f"expected folds {list(common.EXPECTED_FOLDS)}, found {folds_present}")

    status = "pass" if not reasons else "fail"
    return {
        "status": status,
        "reasons": reasons,
        "total_rows": total_rows,
        "unique_row_id_count": len(unique_row_ids),
        "duplicate_row_id_seed_pairs": duplicate_pairs,
        "seeds_present": seeds_present,
        "folds_present": folds_present,
        "seeds_share_row_id_set": seeds_share_row_id_set,
        "row_id_prefix_distribution": dict(sorted(prefix_counts.items())),
        "evaluable_class_count": distinct_label_count,
        "classes_with_min5_unique_rowids": classes_with_min5_unique_rowids,
        "provenance_component_count": len(provenance_components),
    }


# ---------------------------------------------------------------------------
# Check 2 — v9 fold caches: sha256, oof row_id union, finiteness, shape
# ---------------------------------------------------------------------------


def check_2_fold_caches(v9_caches_dir: Path, expected_row_ids: list[str]) -> dict:
    per_fold: dict[str, dict] = {}
    union_oof_row_ids: set[str] = set()
    all_ok = True

    for fold in common.EXPECTED_FOLDS:
        cache_path = v9_caches_dir / f"fold-{fold:02d}-views08.pt"
        entry: dict[str, Any] = {"cache_path": str(cache_path)}
        if not cache_path.is_file():
            entry["status"] = "fail"
            entry["reason"] = "cache file not found"
            per_fold[str(fold)] = entry
            all_ok = False
            continue

        sha_ok, actual_sha, expected_sha = common.verify_cache_sha(cache_path)
        cache = common.load_fold_cache(cache_path)
        oof = cache["oof"]
        base_features = oof["base_features"]
        finite_ok = bool(np.isfinite(base_features.to(torch.float32).numpy()).all())
        shape_ok = base_features.ndim == 2 and base_features.shape[1] == 384
        oof_row_ids = list(oof["row_id"])
        union_oof_row_ids.update(oof_row_ids)

        entry.update(
            {
                "sha256_matches_sidecar": sha_ok,
                "actual_sha256": actual_sha,
                "expected_sha256": expected_sha,
                "oof_row_count": len(oof_row_ids),
                "base_features_shape": list(base_features.shape),
                "base_features_finite": finite_ok,
                "base_features_shape_ok": shape_ok,
                "status": "pass" if (sha_ok and finite_ok and shape_ok) else "fail",
            }
        )
        if entry["status"] != "pass":
            all_ok = False
        per_fold[str(fold)] = entry

    union_sorted = sorted(union_oof_row_ids)
    union_matches_prediction_rows = union_sorted == sorted(expected_row_ids)
    if not union_matches_prediction_rows:
        all_ok = False

    return {
        "status": "pass" if all_ok else "fail",
        "per_fold": per_fold,
        "oof_row_id_union_count": len(union_sorted),
        "oof_row_id_union_matches_prediction_rows": union_matches_prediction_rows,
    }


# ---------------------------------------------------------------------------
# Check 3 — C1 calibration gate (reproduce stored c1_topk from checkpoint + cache)
# ---------------------------------------------------------------------------


def _pick_calibration_fold(v9_caches_dir: Path) -> int | None:
    """First fold (ascending) whose sidecar reports enough oof rows for the
    50-row primary check. Deterministic and documented rather than picked
    post-hoc from results."""
    for fold in common.EXPECTED_FOLDS:
        sidecar_path = v9_caches_dir / f"fold-{fold:02d}-views08.pt.prov.json"
        if not sidecar_path.is_file():
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("oof_rows", 0) >= CALIBRATION_PRIMARY_N:
            return fold
    return None


def check_3_calibration(
    v9_caches_dir: Path,
    v9_checkpoints_dir: Path,
    prediction_rows: list[dict],
) -> dict:
    fold = _pick_calibration_fold(v9_caches_dir)
    if fold is None:
        return _fail(
            "blocked",
            reason=(
                "no fold cache reports >= "
                f"{CALIBRATION_PRIMARY_N} oof rows; cannot run the primary "
                "50-row calibration check on any single fold"
            ),
        )
    seed = CALIBRATION_SEED
    checkpoint_path = v9_checkpoints_dir / f"fold-{fold:02d}-seed-{seed}-C1.pt"
    if not checkpoint_path.is_file():
        return _fail(
            "blocked",
            reason=f"C1 checkpoint not found: {checkpoint_path}",
            fold=fold,
            seed=seed,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    state_dict_keys = sorted(state_dict.keys()) if isinstance(state_dict, dict) else []
    expected_keys = {"net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"}
    if not isinstance(state_dict, dict) or set(state_dict_keys) != expected_keys:
        return _fail(
            "blocked",
            reason=(
                "checkpoint state_dict does not match the fixed 2-layer "
                "ProjectionHead structure (net.0/net.3) — readout may be "
                "episodic with support sets rather than a fixed head; no "
                "proxy computed"
            ),
            fold=fold,
            seed=seed,
            checkpoint_path=str(checkpoint_path),
            state_dict_keys=state_dict_keys,
        )

    hidden_dim = state_dict["net.0.weight"].shape[1]
    embedding_dim = state_dict["net.3.weight"].shape[0]
    projection = common.ProjectionHead(hidden_dim, embedding_dim)
    projection.load_state_dict(state_dict, strict=True)
    projection.eval()

    cache_path = v9_caches_dir / f"fold-{fold:02d}-views08.pt"
    cache = common.load_fold_cache(cache_path)
    train, oof = cache["train"], cache["oof"]

    with torch.no_grad():
        train_labels = torch.tensor(train["class_label"], dtype=torch.long)
        train_embeddings = projection(train["base_features"].to(torch.float32))
        prototype_labels = train_labels.unique(sorted=True)
        prototypes = []
        for label in prototype_labels.tolist():
            mean_embedding = train_embeddings[train_labels == label].mean(dim=0)
            prototypes.append(torch.nn.functional.normalize(mean_embedding, dim=0))
        prototype_tensor = torch.stack(prototypes)

        oof_embeddings = projection(oof["base_features"].to(torch.float32))
        logits = oof_embeddings @ prototype_tensor.T
        k = min(3, len(prototype_labels))
        top_indices = logits.topk(k, dim=1).indices
        computed_topk = prototype_labels[top_indices].tolist()

    computed_by_row_id = dict(zip(oof["row_id"], computed_topk))
    actual_by_row_id = {
        row["row_id"]: row["c1_topk"]
        for row in prediction_rows
        if row["outer_fold"] == fold and row["seed"] == seed
    }

    common_row_ids = sorted(set(computed_by_row_id) & set(actual_by_row_id))
    n_available = len(common_row_ids)

    if n_available < CALIBRATION_PILOT_N:
        return _fail(
            "blocked",
            reason=(
                f"only {n_available} row_id overlap between fold {fold}/seed "
                f"{seed} oof cache and prediction_rows — fewer than the "
                f"{CALIBRATION_PILOT_N}-row pilot requires"
            ),
            fold=fold,
            seed=seed,
            checkpoint_path=str(checkpoint_path),
            state_dict_keys=state_dict_keys,
        )

    pilot_ids = common_row_ids[:CALIBRATION_PILOT_N]
    pilot_top1_matches = sum(
        computed_by_row_id[rid][0] == actual_by_row_id[rid][0] for rid in pilot_ids
    )

    base = {
        "fold": fold,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "state_dict_keys": state_dict_keys,
        "n_available_row_ids": n_available,
        "pilot_n": len(pilot_ids),
        "pilot_top1_matches": pilot_top1_matches,
    }

    if pilot_top1_matches < CALIBRATION_PILOT_MIN_MATCHES:
        return _fail(
            "blocked",
            reason=(
                f"pilot mismatch: {pilot_top1_matches}/{len(pilot_ids)} exact "
                "top-1 matches on the 10-row pilot — stopping before the "
                "50-row primary run per protocol (no proxy computed)"
            ),
            **base,
        )

    primary_ids = common_row_ids[:CALIBRATION_PRIMARY_N]
    primary_top1_matches = sum(
        computed_by_row_id[rid][0] == actual_by_row_id[rid][0] for rid in primary_ids
    )
    primary_top3_exact_matches = sum(
        computed_by_row_id[rid] == actual_by_row_id[rid] for rid in primary_ids
    )
    primary_n = len(primary_ids)
    status = "pass" if primary_top1_matches >= CALIBRATION_PRIMARY_MIN_MATCHES else "fail"

    return {
        "status": status,
        **base,
        "primary_n": primary_n,
        "primary_top1_matches": primary_top1_matches,
        "primary_top1_match_rate": primary_top1_matches / primary_n,
        "primary_top3_exact_matches": primary_top3_exact_matches,
        "primary_top3_exact_match_rate": primary_top3_exact_matches / primary_n,
    }


# ---------------------------------------------------------------------------
# Check 4 — B0 + C1 checkpoint inventory (5 folds x 3 seeds x 2 arms)
# ---------------------------------------------------------------------------


def _v9_checkpoint_sha_by_fold_seed(v9_iteration_prediction_rows: Path) -> dict[tuple[int, int], dict]:
    """(fold, seed) -> {baseline_checkpoint_sha256, candidate_checkpoint_sha256}
    read from v9's own (pre-v19r2) prediction rows, first row per group."""
    if not v9_iteration_prediction_rows.is_file():
        return {}
    result: dict[tuple[int, int], dict] = {}
    with v9_iteration_prediction_rows.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["outer_fold"], row["seed"])
            if key not in result:
                result[key] = {
                    "baseline_checkpoint_sha256": row.get("baseline_checkpoint_sha256"),
                    "candidate_checkpoint_sha256": row.get("candidate_checkpoint_sha256"),
                }
    return result


def check_4_checkpoint_inventory(
    v9_checkpoints_dir: Path, v9_iteration_prediction_rows: Path
) -> dict:
    provenance_shas = _v9_checkpoint_sha_by_fold_seed(v9_iteration_prediction_rows)

    found: dict[str, dict] = {}
    b0_found_count = 0
    c1_found_count = 0
    b0_sha_matches = 0
    c1_sha_matches = 0
    total_expected = len(common.EXPECTED_FOLDS) * len(common.EXPECTED_SEEDS)

    for fold in common.EXPECTED_FOLDS:
        for seed in common.EXPECTED_SEEDS:
            key = f"fold-{fold:02d}-seed-{seed}"
            entry: dict[str, Any] = {}
            group_sha = provenance_shas.get((fold, seed), {})
            for arm, sha_field in (
                ("B0", "baseline_checkpoint_sha256"),
                ("C1", "candidate_checkpoint_sha256"),
            ):
                path = v9_checkpoints_dir / f"{key}-{arm}.pt"
                arm_entry: dict[str, Any] = {"path": str(path), "found": path.is_file()}
                if arm_entry["found"]:
                    actual_sha = common.sha256_file(path)
                    arm_entry["sha256"] = actual_sha
                    expected_sha = group_sha.get(sha_field)
                    arm_entry["expected_sha256"] = expected_sha
                    arm_entry["sha256_matches_provenance"] = (
                        expected_sha is not None and actual_sha == expected_sha
                    )
                    if arm == "B0":
                        b0_found_count += 1
                        if arm_entry["sha256_matches_provenance"]:
                            b0_sha_matches += 1
                    else:
                        c1_found_count += 1
                        if arm_entry["sha256_matches_provenance"]:
                            c1_sha_matches += 1
                entry[arm] = arm_entry
            found[key] = entry

    b0_complete = b0_found_count == total_expected
    c1_complete = c1_found_count == total_expected
    if b0_complete and c1_complete and b0_sha_matches == total_expected and c1_sha_matches == total_expected:
        status = "pass"
    elif b0_found_count == 0:
        status = "blocked"
    else:
        status = "partial"

    return {
        "status": status,
        "total_expected_per_arm": total_expected,
        "b0_found_count": b0_found_count,
        "c1_found_count": c1_found_count,
        "b0_sha256_matches_provenance": b0_sha_matches,
        "c1_sha256_matches_provenance": c1_sha_matches,
        "b0_descope": not b0_complete,
        "checkpoints": found,
    }


# ---------------------------------------------------------------------------
# Check 5 — full-data caches (v11 / r2)
# ---------------------------------------------------------------------------


def check_5_full_data_caches(v11_dir: Path, r2_dir: Path) -> dict:
    cache_files: list[Path] = []
    for directory in (v11_dir, r2_dir):
        if directory.is_dir():
            cache_files.extend(sorted(directory.glob("*.pt")))

    if not cache_files:
        return _fail("blocked", reason="no full-data cache files found under v11 or r2 cache dirs")

    listing = []
    reasons = []
    for path in cache_files:
        size_bytes = path.stat().st_size
        sidecar_path = path.with_name(path.name + ".prov.json")
        row_count = None
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            row_count = sidecar.get("row_count")
        listing.append({"path": str(path), "size_bytes": size_bytes, "row_count": row_count})
        if row_count is not None and not (
            FULL_DATA_ROW_BOUNDS[0] <= row_count <= FULL_DATA_ROW_BOUNDS[1]
        ):
            reasons.append(f"{path}: row_count {row_count} outside bounds {FULL_DATA_ROW_BOUNDS}")

    smallest = min(cache_files, key=lambda p: p.stat().st_size)
    cache = torch.load(smallest, map_location="cpu", weights_only=False)
    schema_keys = sorted(cache.keys()) if isinstance(cache, dict) else []
    loaded_row_count = None
    for key in ("features", "labels", "row_id"):
        value = cache.get(key)
        if hasattr(value, "shape"):
            loaded_row_count = value.shape[0]
            break
        if isinstance(value, list):
            loaded_row_count = len(value)
            break
    if loaded_row_count is not None and not (
        FULL_DATA_ROW_BOUNDS[0] <= loaded_row_count <= FULL_DATA_ROW_BOUNDS[1]
    ):
        reasons.append(f"{smallest}: loaded row_count {loaded_row_count} outside bounds")

    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "cache_files": listing,
        "smallest_cache_path": str(smallest),
        "smallest_cache_schema_keys": schema_keys,
        "smallest_cache_row_count": loaded_row_count,
    }


# ---------------------------------------------------------------------------
# Check 6 — deployed runtime load + predict_topk smoke
# ---------------------------------------------------------------------------


def check_6_deployed_runtime(runtime_dir: Path, sample_cache_path: Path | None) -> dict:
    runtime = common.DeployedRuntime(runtime_dir)
    sha_at_load = runtime.sha256_at_load
    matches_documented = {
        name: sha_at_load.get(name) == DOCUMENTED_RUNTIME_SHA256.get(name)
        for name in DOCUMENTED_RUNTIME_SHA256
    }

    top1_class_labels: list[int] | None = None
    sample_source = None
    if sample_cache_path is not None and sample_cache_path.is_file():
        cache = torch.load(sample_cache_path, map_location="cpu", weights_only=False)
        features = cache.get("features")
        if features is not None:
            sample = features[:5].to(torch.float32)
            topk = runtime.predict_topk(sample, k=3)
            top1_class_labels = [row[0] for row in topk]
            sample_source = str(sample_cache_path)

    runtime.assert_unchanged()

    status = "pass" if all(matches_documented.values()) and top1_class_labels else "fail"
    return {
        "status": status,
        "sha256_at_load": sha_at_load,
        "matches_documented_expected_sha256": matches_documented,
        "sample_source_cache": sample_source,
        "sample_predict_topk_top1_class_labels": top1_class_labels,
        "note": (
            "descriptive smoke test only — no correctness claim; features "
            "come from a full-data training-fit cache, not an evaluation set"
        ),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_core_report(args: argparse.Namespace) -> dict:
    common.set_determinism(args.seed)

    runtime = common.DeployedRuntime(args.runtime_dir)
    runtime.assert_unchanged()  # sanity: hashes just computed twice must agree

    rows = common.load_prediction_rows(args.prediction_rows)
    unique_row_ids = sorted({row["row_id"] for row in rows})

    check_1 = check_1_prediction_rows(rows)
    check_2 = check_2_fold_caches(args.v9_caches_dir, unique_row_ids)
    check_3 = check_3_calibration(args.v9_caches_dir, args.v9_checkpoints_dir, rows)
    check_4 = check_4_checkpoint_inventory(
        args.v9_checkpoints_dir, args.v9_iteration_prediction_rows
    )
    check_5 = check_5_full_data_caches(args.full_data_v11_dir, args.full_data_r2_dir)

    smallest_full_data_cache = None
    if args.full_data_v11_dir.is_dir():
        v11_caches = sorted(args.full_data_v11_dir.glob("*.pt"))
        if v11_caches:
            smallest_full_data_cache = v11_caches[0]
    check_6 = check_6_deployed_runtime(args.runtime_dir, smallest_full_data_cache)

    runtime.assert_unchanged()  # final: unchanged across the whole check pipeline

    return {
        "schema_version": "lot-a-preflight-v1",
        "git_sha": get_git_sha(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "seed": args.seed,
        "inputs": {
            "prediction_rows_path": str(args.prediction_rows),
            "prediction_rows_sha256": common.sha256_file(args.prediction_rows),
            "v9_caches_dir": str(args.v9_caches_dir),
            "v9_checkpoints_dir": str(args.v9_checkpoints_dir),
            "runtime_dir": str(args.runtime_dir),
        },
        "checks": {
            "1_prediction_rows": check_1,
            "2_fold_caches": check_2,
            "3_calibration": check_3,
            "4_checkpoint_inventory": check_4,
            "5_full_data_caches": check_5,
            "6_deployed_runtime": check_6,
        },
    }


def _summarize(report: dict) -> str:
    lines = [f"Lot A preflight — git {report['git_sha'][:12]}"]
    for name, check in report["checks"].items():
        lines.append(f"  {name}: {check['status']}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-rows", type=Path, default=common.PREDICTION_ROWS_PATH)
    parser.add_argument("--v9-caches-dir", type=Path, default=common.V9_CACHES_DIR)
    parser.add_argument("--v9-checkpoints-dir", type=Path, default=V9_CHECKPOINTS_DIR)
    parser.add_argument(
        "--v9-iteration-prediction-rows", type=Path, default=V9_ITERATION_PREDICTION_ROWS
    )
    parser.add_argument("--full-data-v11-dir", type=Path, default=FULL_DATA_V11_CACHE_DIR)
    parser.add_argument("--full-data-r2-dir", type=Path, default=FULL_DATA_R2_CACHE_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=common.DEPLOYED_RUNTIME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report_1 = build_core_report(args)
    report_2 = build_core_report(args)
    hash_1 = common.sha256_json(report_1)
    hash_2 = common.sha256_json(report_2)
    matched = hash_1 == hash_2

    final_report = dict(report_1)
    final_report["reproducibility"] = {
        "hash_run_1": hash_1,
        "hash_run_2": hash_2,
        "matched": matched,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(common.canonical_json_dumps(final_report), encoding="utf-8")

    print(_summarize(final_report))
    print(f"reproducibility: hash_run_1={hash_1[:16]} hash_run_2={hash_2[:16]} matched={matched}")
    print(f"wrote {args.output}")

    if not matched:
        print("WARNING: two in-process runs produced different reports — investigate before trusting this preflight", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

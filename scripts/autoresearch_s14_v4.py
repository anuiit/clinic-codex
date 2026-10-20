#!/usr/bin/env python3
"""Adaptive, train-cache-only neighbourhood research around cosine 1-NN."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
V3_SCRIPT = REPO / "scripts/autoresearch_s14_v3.py"
V3_SPEC = importlib.util.spec_from_file_location("autoresearch_s14_v3_for_v4", V3_SCRIPT)
assert V3_SPEC and V3_SPEC.loader
v3 = importlib.util.module_from_spec(V3_SPEC)
sys.modules[V3_SPEC.name] = v3
V3_SPEC.loader.exec_module(v3)

SEALED = REPO / "reports/autoresearch-s14-v3-20260812/sealed/train-only.features.pt"
REPORT_DIR = REPO / "reports/autoresearch-s14-v4-20260812"
MANIFEST = REPORT_DIR / "frozen-specs.json"
RESULTS = REPORT_DIR / "results.json"
DOC = REPO / "docs/autoresearch-s14-v4-20260812.md"
SCREEN_FOLDS = 3
OUTER_FOLDS = 5
SEEDS = (17, 42, 73)
TOP_N = 5
MIN_GAIN_VS_KNN1 = 0.002
BUCKET_TOLERANCE = 0.01
EXPECTED_ROWS = 9128
EXPECTED_CLASSES = 286
TEMPERATURE = 0.1


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    method: str
    params: tuple[tuple[str, Any], ...] = ()
    transform: str = "b1"
    runtime_compatible: bool = True

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)


def _p(**kwargs: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(kwargs.items()))


# Frozen adaptive matrix: baselines first, followed by exactly 28 challengers.
SPECS: tuple[Spec, ...] = (
    Spec("baseline-b1-ncm", "ncm"),
    Spec("baseline-cosine-k1", "knn", _p(k=1, metric="cosine")),
    Spec("euclidean-k1", "knn", _p(k=1, metric="euclidean")),
    Spec("centered-cosine-k1", "knn", _p(k=1, metric="cosine"), transform="centered"),
    Spec("power050-cosine-k1", "knn", _p(k=1, metric="cosine"), transform="power050"),
    Spec("power075-cosine-k1", "knn", _p(k=1, metric="cosine"), transform="power075"),
    Spec("zscore-cosine-k1", "knn", _p(k=1, metric="cosine"), transform="zscore"),
    Spec("cosine-k2", "knn", _p(k=2, metric="cosine")),
    Spec("cosine-k3", "knn", _p(k=3, metric="cosine")),
    Spec("cosine-k4", "knn", _p(k=4, metric="cosine")),
    Spec("cosine-k5", "knn", _p(k=5, metric="cosine")),
    Spec("cosine-k7", "knn", _p(k=7, metric="cosine")),
    Spec("cosine-k9", "knn", _p(k=9, metric="cosine")),
    Spec("cosine-k15", "knn", _p(k=15, metric="cosine")),
    Spec("kernel-k15-beta2", "kernel", _p(k=15, beta=2.0)),
    Spec("kernel-k15-beta5", "kernel", _p(k=15, beta=5.0)),
    Spec("kernel-k15-beta10", "kernel", _p(k=15, beta=10.0)),
    Spec("kernel-k15-beta20", "kernel", _p(k=15, beta=20.0)),
    Spec("hybrid-k1-alpha025", "hybrid", _p(alpha=0.25)),
    Spec("hybrid-k1-alpha050", "hybrid", _p(alpha=0.50)),
    Spec("hybrid-k1-alpha075", "hybrid", _p(alpha=0.75)),
    Spec("balanced-k31-beta5-a050", "balanced_kernel", _p(k=31, beta=5.0, support_alpha=0.5)),
    Spec("balanced-k31-beta5-a100", "balanced_kernel", _p(k=31, beta=5.0, support_alpha=1.0)),
    Spec("balanced-k31-beta10-a050", "balanced_kernel", _p(k=31, beta=10.0, support_alpha=0.5)),
    Spec("balanced-k31-beta10-a100", "balanced_kernel", _p(k=31, beta=10.0, support_alpha=1.0)),
    Spec("adaptive-k-1-3", "adaptive", _p(low_max=7, mid_max=10**9, mid_k=3, high_k=3)),
    Spec("adaptive-k-1-3-5", "adaptive", _p(low_max=7, mid_max=20, mid_k=3, high_k=5)),
    Spec("margin-fallback-002", "fallback", _p(margin=0.02)),
    Spec("margin-fallback-005", "fallback", _p(margin=0.05)),
    Spec("margin-fallback-010", "fallback", _p(margin=0.10)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assert_specs(specs: tuple[Spec, ...] = SPECS) -> None:
    if len(specs) != 30:
        raise ValueError("v4 requires exactly 30 frozen specs")
    if [item.method for item in specs[:2]] != ["ncm", "knn"]:
        raise ValueError("v4 baseline ordering changed")
    if len({item.name for item in specs}) != len(specs):
        raise ValueError("v4 spec names are not unique")
    allowed = {"ncm", "knn", "kernel", "hybrid", "balanced_kernel", "adaptive", "fallback"}
    if {item.method for item in specs} - allowed:
        raise ValueError("v4 has an unknown method")


def load_train_cache() -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(SEALED, map_location="cpu", weights_only=False)
    v3.validate_train_cache(cache)
    if len(cache["features"]) != EXPECTED_ROWS or cache["labels"].unique().numel() != EXPECTED_CLASSES:
        raise ValueError("v4 train cache contract mismatch")
    return cache["features"].float(), cache["labels"].long()


def transform_pair(train: torch.Tensor, query: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "b1":
        return v3.normalize(train), v3.normalize(query)
    if name == "centered":
        center = train.mean(0, keepdim=True)
        return v3.normalize(train-center), v3.normalize(query-center)
    if name.startswith("power"):
        power = int(name.removeprefix("power"))/100.0
        operation = lambda x: torch.sign(x)*x.abs().pow(power)
        return v3.normalize(operation(train)), v3.normalize(operation(query))
    if name == "zscore":
        center = train.mean(0, keepdim=True)
        scale = train.std(0, keepdim=True).clamp_min(1e-4)
        return v3.normalize((train-center)/scale), v3.normalize((query-center)/scale)
    raise ValueError(name)


def neighbour_table(train: torch.Tensor, query: torch.Tensor, metric: str, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "cosine":
        affinity = v3.normalize(query) @ v3.normalize(train).T
    elif metric == "euclidean":
        affinity = -torch.cdist(query, train).square()
    else:
        raise ValueError(metric)
    return affinity.topk(k, dim=1)


def scatter_votes(labels: torch.Tensor, values: torch.Tensor, neighbours: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    neighbour_labels = labels[neighbours]
    weights = torch.ones_like(values) if weights is None else weights
    scores = torch.zeros(len(neighbours), EXPECTED_CLASSES, device=neighbours.device)
    scores.scatter_add_(1, neighbour_labels, weights)
    return scores


def score_spec(spec: Spec, train: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    train, query = transform_pair(train, query, spec.transform)
    kw = spec.kwargs
    prototypes = v3.normalize(v3.class_means(train, labels))
    ncm = query @ prototypes.T
    if spec.method == "ncm":
        return ncm
    if spec.method == "knn":
        values, neighbours = neighbour_table(train, query, str(kw["metric"]), int(kw["k"]))
        return scatter_votes(labels, values, neighbours)
    if spec.method in {"kernel", "balanced_kernel"}:
        values, neighbours = neighbour_table(train, query, "cosine", int(kw["k"]))
        weights = torch.exp(float(kw["beta"])*(values-values[:, :1]))
        scores = scatter_votes(labels, values, neighbours, weights)
        if spec.method == "balanced_kernel":
            support = torch.bincount(labels, minlength=EXPECTED_CLASSES).to(scores.dtype)
            scores = scores/support.clamp_min(1).pow(float(kw["support_alpha"]))
        return scores/weights.sum(1, keepdim=True).clamp_min(1e-8)
    if spec.method == "hybrid":
        values, neighbours = neighbour_table(train, query, "cosine", 1)
        nearest = scatter_votes(labels, values, neighbours)
        alpha = float(kw["alpha"])
        return alpha*nearest + (1-alpha)*ncm
    if spec.method == "adaptive":
        values, neighbours = neighbour_table(train, query, "cosine", int(kw["high_k"]))
        support = torch.bincount(labels, minlength=EXPECTED_CLASSES)
        nearest_support = support[labels[neighbours[:, 0]]]
        chosen_k = torch.where(nearest_support <= int(kw["low_max"]), 1, torch.where(nearest_support <= int(kw["mid_max"]), int(kw["mid_k"]), int(kw["high_k"])))
        scores = torch.zeros(len(query), EXPECTED_CLASSES, device=query.device)
        neighbour_labels = labels[neighbours]
        for k in sorted(set(chosen_k.tolist())):
            mask = chosen_k.eq(k)
            partial = torch.zeros(int(mask.sum()), EXPECTED_CLASSES, device=query.device)
            partial.scatter_add_(1, neighbour_labels[mask, :k], torch.ones_like(values[mask, :k]))
            scores[mask] = partial
        return scores
    if spec.method == "fallback":
        values, neighbours = neighbour_table(train, query, "cosine", 2)
        nearest = scatter_votes(labels, values[:, :1], neighbours[:, :1])
        confident = (values[:, 0]-values[:, 1]).ge(float(kw["margin"]))
        return torch.where(confident[:, None], nearest, ncm)
    raise ValueError(spec.method)


def metric_summary(records: list[dict[str, Any]], total_support: torch.Tensor) -> dict[str, Any]:
    return v3.metric_summary(records, total_support)


def evaluate(spec: Spec, features: torch.Tensor, labels: torch.Tensor, folds: list[torch.Tensor], device: torch.device) -> dict[str, Any]:
    projection = v3.load_projection()
    embeddings = v3.apply_projection(features, projection).to(device)
    labels_device = labels.to(device)
    all_rows = torch.arange(len(labels))
    records: list[dict[str, Any]] = []
    fold_summaries = []
    started = time.time()
    for fold_index, holdout in enumerate(folds):
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[holdout] = False
        train_rows = all_rows[mask]
        score = score_spec(spec, embeddings[train_rows.to(device)], labels_device[train_rows.to(device)], embeddings[holdout.to(device)])
        probabilities = F.softmax(score/TEMPERATURE, dim=1)
        top3 = score.topk(3, dim=1).indices.cpu()
        truth = labels[holdout]
        loss = -probabilities.cpu()[torch.arange(len(holdout)), truth].clamp_min(1e-12).log()
        fold_records = [{"row_index": int(row), "label": int(label), "prediction": int(tops[0]), "top3": [int(value) for value in tops], "loss": float(item_loss)} for row, label, tops, item_loss in zip(holdout, truth, top3, loss)]
        records.extend(fold_records)
        fold_summaries.append({"fold": fold_index, "rows": len(fold_records), "micro_top1": sum(int(item["label"] == item["prediction"]) for item in fold_records)/len(fold_records)})
    records.sort(key=lambda item: item["row_index"])
    return {"metrics": metric_summary(records, torch.bincount(labels, minlength=EXPECTED_CLASSES)), "folds": fold_summaries, "predictions": records, "duration_seconds": time.time()-started}


def bootstrap(candidate: list[float], baseline: list[float], seed: int) -> dict[str, float]:
    return v3.paired_class_bootstrap(candidate, baseline, seed=seed)


def frozen_manifest() -> dict[str, Any]:
    assert_specs()
    value = {
        "protocol": "s14-adaptive-neighbourhood.v4",
        "created_before_results": True,
        "adaptive_basis": {"method": "cosine 1-NN", "outer_macro_gain": 0.0063527869839601825, "class_bootstrap_interval": [0.0032825646206470216, 0.00953351581134581]},
        "sealed": {"path": str(SEALED), "sha256": sha256(SEALED)},
        "projection": {"path": str(v3.PROJECTION), "sha256": sha256(v3.PROJECTION)},
        "v3_runner": {"path": str(V3_SCRIPT), "sha256": sha256(V3_SCRIPT)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "cv": {"kind": "in-page row-stratified", "screen": {"folds": SCREEN_FOLDS, "seed": SEEDS[0]}, "outer": {"folds": OUTER_FOLDS, "seed": SEEDS[0]}, "multi_seed": list(SEEDS)},
        "selection": {"screen_top_n_regardless": TOP_N, "outer_winner_regardless": True, "final_min_gain_vs_knn1": MIN_GAIN_VS_KNN1, "paired_class_bootstrap_lower_positive_vs_both": True, "bucket_tolerance_vs_knn1": BUCKET_TOLERANCE},
        "specs": [json.loads(json.dumps(dataclasses.asdict(item))) for item in SPECS],
    }
    value["specification_hash"] = stable_hash({"cv": value["cv"], "selection": value["selection"], "specs": value["specs"]})
    return value


def prepare() -> dict[str, Any]:
    if not SEALED.is_file():
        raise FileNotFoundError(SEALED)
    load_train_cache()
    manifest = frozen_manifest()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("v4 frozen manifest drift")
    write_json(MANIFEST, manifest)
    if not RESULTS.exists():
        write_json(RESULTS, {"protocol": manifest["protocol"], "specification_hash": manifest["specification_hash"], "screening": [], "outer": [], "confirmation": []})
    return manifest


def run_screen(start: int, stop: int) -> None:
    prepare()
    features, labels = load_train_cache()
    folds = v3.stratified_folds(labels, SCREEN_FOLDS, SEEDS[0])
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(results["screening"]) != start:
        raise RuntimeError(f"expected screening start {len(results['screening'])}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index in range(start, min(stop, len(SPECS))):
        spec = SPECS[index]
        try:
            outcome = evaluate(spec, features, labels, folds, device)
            outcome.update({"index": index, "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
        except Exception as exc:
            outcome = {"index": index, "name": spec.name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "spec": dataclasses.asdict(spec)}
        write_json(REPORT_DIR / f"screening/{index:02d}-{spec.name}.json", outcome)
        results["screening"].append(outcome)
        write_json(RESULTS, results)
        print(json.dumps({"phase": "screen", "completed": index+1, "total": len(SPECS), "name": spec.name, "macro_top1": outcome.get("metrics", {}).get("macro_top1"), "status": outcome["status"]}), flush=True)


def screen_selection(results: dict[str, Any]) -> list[dict[str, Any]]:
    valid = [item for item in results["screening"] if item["status"] == "COMPLETED"]
    if len(valid) != len(SPECS):
        raise RuntimeError("v4 screening is incomplete")
    challengers = sorted(valid[2:], key=lambda item: (-item["metrics"]["macro_top1"], item["index"]))[:TOP_N]
    return [valid[0], valid[1], *challengers]


def run_outer() -> None:
    prepare()
    features, labels = load_train_cache()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if results["outer"]:
        raise RuntimeError("v4 outer already exists")
    selected = screen_selection(results)
    folds = v3.stratified_folds(labels, OUTER_FOLDS, SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for order, item in enumerate(selected):
        spec = SPECS[item["index"]]
        outcome = evaluate(spec, features, labels, folds, device)
        outcome.update({"index": item["index"], "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
        write_json(REPORT_DIR / f"outer/{order:02d}-{spec.name}.json", outcome)
        results["outer"].append(outcome)
        write_json(RESULTS, results)
        print(json.dumps({"phase": "outer", "completed": order+1, "total": len(selected), "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    ncm, knn1 = results["outer"][:2]
    for item in results["outer"]:
        item["bootstrap_vs_ncm"] = bootstrap(item["metrics"]["per_class_top1"], ncm["metrics"]["per_class_top1"], 404)
        item["bootstrap_vs_knn1"] = bootstrap(item["metrics"]["per_class_top1"], knn1["metrics"]["per_class_top1"], 405)
    write_json(RESULTS, results)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("macro_top1", "macro_top3", "micro_top1", "balanced_logloss", "low_support_macro_top1", "high_support_macro_top1")
    return {key: sum(item["metrics"][key] for item in items)/len(items) for key in keys}


def run_confirmation() -> None:
    prepare()
    features, labels = load_train_cache()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(results["outer"]) < 3:
        raise RuntimeError("v4 outer phase missing")
    winner = max(results["outer"][2:], key=lambda item: item["metrics"]["macro_top1"])
    indices = (0, 1, winner["index"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        folds = v3.stratified_folds(labels, OUTER_FOLDS, seed)
        for index in indices:
            spec = SPECS[index]
            if any(item["seed"] == seed and item["index"] == index for item in results["confirmation"]):
                continue
            outcome = evaluate(spec, features, labels, folds, device)
            outcome.update({"seed": seed, "index": index, "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
            write_json(REPORT_DIR / f"confirmation/seed-{seed}-{spec.name}.json", outcome)
            results["confirmation"].append(outcome)
            write_json(RESULTS, results)
            print(json.dumps({"phase": "confirm", "seed": seed, "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    grouped = {index: [item for item in results["confirmation"] if item["index"] == index] for index in indices}
    summaries = {SPECS[index].name: aggregate(items) for index, items in grouped.items()}
    candidate = summaries[SPECS[winner["index"]].name]
    ncm = summaries[SPECS[0].name]
    knn1 = summaries[SPECS[1].name]
    candidate_per_class = [sum(item["metrics"]["per_class_top1"][label] for item in grouped[winner["index"]])/len(SEEDS) for label in range(EXPECTED_CLASSES)]
    ncm_per_class = [sum(item["metrics"]["per_class_top1"][label] for item in grouped[0])/len(SEEDS) for label in range(EXPECTED_CLASSES)]
    knn1_per_class = [sum(item["metrics"]["per_class_top1"][label] for item in grouped[1])/len(SEEDS) for label in range(EXPECTED_CLASSES)]
    versus_ncm = bootstrap(candidate_per_class, ncm_per_class, 504)
    versus_knn1 = bootstrap(candidate_per_class, knn1_per_class, 505)
    passed = (
        candidate["macro_top1"] >= knn1["macro_top1"] + MIN_GAIN_VS_KNN1
        and versus_ncm["lower_95"] > 0 and versus_knn1["lower_95"] > 0
        and candidate["low_support_macro_top1"] >= knn1["low_support_macro_top1"]-BUCKET_TOLERANCE
        and candidate["high_support_macro_top1"] >= knn1["high_support_macro_top1"]-BUCKET_TOLERANCE
    )
    results["confirmation_summary"] = {"winner": SPECS[winner["index"]].name, "metrics": summaries, "bootstrap_vs_ncm": versus_ncm, "bootstrap_vs_knn1": versus_knn1, "minimum_gain_vs_knn1": MIN_GAIN_VS_KNN1, "passed": passed}
    results["verdict"] = "RESEARCH_CHALLENGER" if passed else "PLATEAU_NO_V4_CHALLENGER"
    write_json(RESULTS, results)
    write_document(results)


def write_document(results: dict[str, Any]) -> None:
    summary = results["confirmation_summary"]
    rows = [
        "# S/14 autoresearch v4 — adaptive neighbourhood phase (2026-08-12)", "",
        "V4 is a separate, preregistered follow-up to the positive cosine 1-NN signal in V3. It reads only the physical train-only cache and remains an in-page row-CV experiment; it does not estimate new-page generalization.", "",
        f"- Frozen variants: {len(SPECS)}; screening completed: {len(results['screening'])}.",
        f"- Outer finalists: {len(results['outer'])-2}, each compared with B1 NCM and cosine 1-NN.",
        f"- Multi-seed winner: `{summary['winner']}` on seeds 17/42/73.",
        f"- Verdict: `{results['verdict']}`.", "",
        "| Screening # | Variant | Macro top-1 | Low support | High support |", "| ---: | --- | ---: | ---: | ---: |",
    ]
    for item in results["screening"]:
        if item["status"] == "COMPLETED":
            metric = item["metrics"]
            rows.append(f"| {item['index']} | {item['name']} | {metric['macro_top1']:.6f} | {metric['low_support_macro_top1']:.6f} | {metric['high_support_macro_top1']:.6f} |")
    rows += ["", "## Multi-seed confirmation", ""]
    for name, metric in summary["metrics"].items():
        rows.append(f"- `{name}`: macro top-1 `{metric['macro_top1']:.6f}`, macro top-3 `{metric['macro_top3']:.6f}`, micro top-1 `{metric['micro_top1']:.6f}`, balanced logloss `{metric['balanced_logloss']:.6f}`.")
    rows += ["", f"Candidate bootstrap versus NCM: `{summary['bootstrap_vs_ncm']}`.", f"Candidate bootstrap versus 1-NN: `{summary['bootstrap_vs_knn1']}`.", "", "Conceptual basis: cosine nearest-neighbour cache models, class-normalized kernel voting, prototype/cache score interpolation, and confidence-based fallback. Implementations are original to this harness. Relevant foundations include [SimpleShot](https://arxiv.org/abs/1911.04623) and [Tip-Adapter](https://arxiv.org/abs/2207.09519).", "", f"Machine-readable evidence: `{RESULTS.relative_to(REPO)}`.", ""]
    DOC.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    screen = sub.add_parser("screen")
    screen.add_argument("--start", type=int, default=0)
    screen.add_argument("--stop", type=int, default=len(SPECS))
    sub.add_parser("outer")
    sub.add_parser("confirm")
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    elif args.command == "screen":
        run_screen(args.start, args.stop)
    elif args.command == "outer":
        run_outer()
    else:
        run_confirmation()


if __name__ == "__main__":
    main()

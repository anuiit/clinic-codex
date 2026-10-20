#!/usr/bin/env python3
"""True-B1, fail-closed replay of the frozen V3 and V4 research matrices."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3_PATH = REPO / "scripts/autoresearch_s14_v3.py"
V4_PATH = REPO / "scripts/autoresearch_s14_v4.py"
v3 = import_file("autoresearch_s14_v3_for_v5", V3_PATH)
v4 = import_file("autoresearch_s14_v4_for_v5", V4_PATH)

SEALED = REPO / "reports/autoresearch-s14-v3-20260812/sealed/train-only.features.pt"
PROJECTION = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/runtime/weights/projection.pt"
EXPECTED_PROJECTION_SHA = "52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03"
REPORT_DIR = REPO / "reports/autoresearch-s14-v5-true-b1-20260812"
MANIFEST = REPORT_DIR / "frozen-specs.json"
RESULTS = REPORT_DIR / "results.json"
DOC = REPO / "docs/autoresearch-s14-v5-true-b1-20260812.md"
EXPECTED_ROWS = 9128
EXPECTED_CLASSES = 286
SCREEN_FOLDS = 3
OUTER_FOLDS = 5
SELECTION_SEED = 17
CONFIRMATION_SEEDS = (42, 73, 101)
TOP_N = 10
CONFIRM_TOP_N = 2
MIN_GAIN_VS_CONTROL = 0.005
BUCKET_TOLERANCE = 0.01
TEMPERATURE = 0.1


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    family: str
    source_index: int
    runtime_compatible: bool


_NON_RUNTIME_METHODS = {"knn", "kernel", "balanced_kernel", "hybrid", "adaptive", "fallback"}
SPECS: tuple[Spec, ...] = tuple(
    [Spec(f"v3-{item.name}", "v3", index, item.runtime_compatible and item.method not in _NON_RUNTIME_METHODS) for index, item in enumerate(v3.SPECS)]
    + [Spec(f"v4-{item.name}", "v4", index, item.runtime_compatible and item.method not in _NON_RUNTIME_METHODS) for index, item in enumerate(v4.SPECS)]
)
CONTROL_INDEX = 0
KNN1_INDEX = len(v3.SPECS) + 1


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
    if len(specs) != 77 or len({item.name for item in specs}) != 77:
        raise ValueError("v5 requires the immutable 47+30 replay matrix")
    if sum(item.family == "v3" for item in specs) != 47 or sum(item.family == "v4" for item in specs) != 30:
        raise ValueError("v5 family counts changed")
    if specs[CONTROL_INDEX].name != "v3-control-b1-ncm" or specs[KNN1_INDEX].name != "v4-baseline-cosine-k1":
        raise ValueError("v5 baselines changed")


def assert_projection_identity(path: Path = PROJECTION) -> dict[str, str]:
    resolved = path.resolve()
    if resolved != PROJECTION.resolve():
        raise ValueError(f"projection path mismatch: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256(resolved)
    if actual != EXPECTED_PROJECTION_SHA:
        raise ValueError(f"B1 projection SHA mismatch: {actual}")
    return {"path": str(resolved), "sha256": actual}


def load_projection_b1(path: Path = PROJECTION) -> dict[str, torch.Tensor]:
    assert_projection_identity(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"}
    if set(state) != expected:
        raise ValueError("B1 projection tensor schema mismatch")
    return {key: value.float() for key, value in state.items()}


def load_train_cache() -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(SEALED, map_location="cpu", weights_only=False)
    v3.validate_train_cache(cache)
    if len(cache["features"]) != EXPECTED_ROWS or cache["labels"].unique().numel() != EXPECTED_CLASSES:
        raise ValueError("v5 cache contract mismatch")
    return cache["features"].float(), cache["labels"].long()


def source_spec(spec: Spec):
    return v3.SPECS[spec.source_index] if spec.family == "v3" else v4.SPECS[spec.source_index]


def score_spec(spec: Spec, raw_train: torch.Tensor, train_labels: torch.Tensor, raw_query: torch.Tensor, projection: dict[str, torch.Tensor]) -> torch.Tensor:
    original = source_spec(spec)
    if spec.family == "v3":
        if original.space == "b1":
            train = v3.apply_projection(raw_train, projection)
            query = v3.apply_projection(raw_query, projection)
        else:
            train, query = v3.normalize(raw_train), v3.normalize(raw_query)
        return v3.scores_for(original, train, train_labels, query)
    train = v3.apply_projection(raw_train, projection)
    query = v3.apply_projection(raw_query, projection)
    return v4.score_spec(original, train, train_labels, query)


def evaluate(spec: Spec, features: torch.Tensor, labels: torch.Tensor, folds: list[torch.Tensor], device: torch.device) -> dict[str, Any]:
    projection = {key: value.to(device) for key, value in load_projection_b1().items()}
    features = features.to(device)
    labels_device = labels.to(device)
    all_rows = torch.arange(len(labels))
    records: list[dict[str, Any]] = []
    fold_summaries = []
    started = time.time()
    for fold_index, holdout in enumerate(folds):
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[holdout] = False
        train_rows = all_rows[mask]
        score = score_spec(spec, features[train_rows.to(device)], labels_device[train_rows.to(device)], features[holdout.to(device)], projection)
        probability = F.softmax(score/TEMPERATURE, dim=1)
        top3 = score.topk(3, dim=1).indices.cpu()
        truth = labels[holdout]
        loss = -probability.cpu()[torch.arange(len(holdout)), truth].clamp_min(1e-12).log()
        fold_records = [{"row_index": int(row), "label": int(label), "prediction": int(tops[0]), "top3": [int(value) for value in tops], "loss": float(item_loss)} for row, label, tops, item_loss in zip(holdout, truth, top3, loss)]
        records.extend(fold_records)
        fold_summaries.append({"fold": fold_index, "rows": len(fold_records), "micro_top1": sum(int(item["label"] == item["prediction"]) for item in fold_records)/len(fold_records)})
    records.sort(key=lambda item: item["row_index"])
    return {"metrics": v3.metric_summary(records, torch.bincount(labels, minlength=EXPECTED_CLASSES)), "folds": fold_summaries, "predictions": records, "duration_seconds": time.time()-started}


def frozen_manifest() -> dict[str, Any]:
    assert_specs()
    projection = assert_projection_identity()
    value = {
        "protocol": "s14-true-b1-replay.v5",
        "created_before_results": True,
        "invalidated_predecessor_claim": "V3/V4 outputs used historical projection SHA 0ad6ce... and are not B1 evidence.",
        "sealed": {"path": str(SEALED.resolve()), "sha256": sha256(SEALED)},
        "projection": projection,
        "expected_projection_sha256": EXPECTED_PROJECTION_SHA,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "method_sources": {"v3": {"path": str(V3_PATH), "sha256": sha256(V3_PATH)}, "v4": {"path": str(V4_PATH), "sha256": sha256(V4_PATH)}},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "cv": {"kind": "in-page row-stratified", "screen": {"folds": SCREEN_FOLDS, "seed": SELECTION_SEED}, "outer": {"folds": OUTER_FOLDS, "seed": SELECTION_SEED}, "confirmation": {"folds": OUTER_FOLDS, "seeds": list(CONFIRMATION_SEEDS), "selection_seed_excluded": True}},
        "selection": {"screen_top_n": TOP_N, "outer_confirm_top_n": CONFIRM_TOP_N, "baselines_always_confirmed": [SPECS[CONTROL_INDEX].name, SPECS[KNN1_INDEX].name], "minimum_gain_vs_control": MIN_GAIN_VS_CONTROL, "class_bootstrap_lower_positive_vs_control": True, "support_bucket_tolerance": BUCKET_TOLERANCE, "balanced_logloss_used_for_gate": False},
        "specs": [dataclasses.asdict(item) | {"source_spec": json.loads(json.dumps(dataclasses.asdict(source_spec(item))))} for item in SPECS],
    }
    value["specification_hash"] = stable_hash({"projection": value["projection"], "cv": value["cv"], "selection": value["selection"], "specs": value["specs"]})
    return value


def prepare() -> dict[str, Any]:
    load_train_cache()
    manifest = frozen_manifest()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("v5 frozen manifest drift")
    write_json(MANIFEST, manifest)
    if not RESULTS.exists():
        write_json(RESULTS, {"protocol": manifest["protocol"], "specification_hash": manifest["specification_hash"], "screening": [], "outer": [], "confirmation": []})
    return manifest


def run_screen(start: int, stop: int) -> None:
    prepare()
    features, labels = load_train_cache()
    folds = v3.stratified_folds(labels, SCREEN_FOLDS, SELECTION_SEED)
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


def selected(results: dict[str, Any]) -> list[dict[str, Any]]:
    valid = [item for item in results["screening"] if item["status"] == "COMPLETED"]
    if len(valid) != len(SPECS):
        raise RuntimeError("v5 screening incomplete")
    excluded = {CONTROL_INDEX, KNN1_INDEX}
    top = sorted((item for item in valid if item["index"] not in excluded), key=lambda item: (-item["metrics"]["macro_top1"], item["index"]))[:TOP_N]
    return [valid[CONTROL_INDEX], valid[KNN1_INDEX], *top]


def run_outer() -> None:
    prepare()
    features, labels = load_train_cache()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if results["outer"]:
        raise RuntimeError("v5 outer already exists")
    finalists = selected(results)
    folds = v3.stratified_folds(labels, OUTER_FOLDS, SELECTION_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for order, item in enumerate(finalists):
        spec = SPECS[item["index"]]
        outcome = evaluate(spec, features, labels, folds, device)
        outcome.update({"index": item["index"], "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
        write_json(REPORT_DIR / f"outer/{order:02d}-{spec.name}.json", outcome)
        results["outer"].append(outcome)
        write_json(RESULTS, results)
        print(json.dumps({"phase": "outer", "completed": order+1, "total": len(finalists), "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    control, knn1 = results["outer"][:2]
    for item in results["outer"]:
        item["bootstrap_vs_control"] = v3.paired_class_bootstrap(item["metrics"]["per_class_top1"], control["metrics"]["per_class_top1"], seed=605)
        item["bootstrap_vs_knn1"] = v3.paired_class_bootstrap(item["metrics"]["per_class_top1"], knn1["metrics"]["per_class_top1"], seed=606)
    write_json(RESULTS, results)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("macro_top1", "macro_top3", "micro_top1", "balanced_logloss", "low_support_macro_top1", "high_support_macro_top1")
    return {key: sum(item["metrics"][key] for item in items)/len(items) for key in keys}


def average_per_class(items: list[dict[str, Any]]) -> list[float]:
    return [sum(item["metrics"]["per_class_top1"][label] for item in items)/len(items) for label in range(EXPECTED_CLASSES)]


def run_confirmation() -> None:
    prepare()
    features, labels = load_train_cache()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(results["outer"]) != TOP_N + 2:
        raise RuntimeError("v5 outer incomplete")
    winners = sorted(results["outer"][2:], key=lambda item: (-item["metrics"]["macro_top1"], item["index"]))[:CONFIRM_TOP_N]
    indices = [CONTROL_INDEX, KNN1_INDEX, *(item["index"] for item in winners)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in CONFIRMATION_SEEDS:
        folds = v3.stratified_folds(labels, OUTER_FOLDS, seed)
        for index in indices:
            if any(item["seed"] == seed and item["index"] == index for item in results["confirmation"]):
                continue
            spec = SPECS[index]
            outcome = evaluate(spec, features, labels, folds, device)
            outcome.update({"seed": seed, "index": index, "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
            write_json(REPORT_DIR / f"confirmation/seed-{seed}-{spec.name}.json", outcome)
            results["confirmation"].append(outcome)
            write_json(RESULTS, results)
            print(json.dumps({"phase": "confirm", "seed": seed, "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    grouped = {index: [item for item in results["confirmation"] if item["index"] == index] for index in indices}
    summaries = []
    control_metrics = aggregate(grouped[CONTROL_INDEX])
    control_class = average_per_class(grouped[CONTROL_INDEX])
    knn1_class = average_per_class(grouped[KNN1_INDEX])
    for index in indices:
        metrics = aggregate(grouped[index])
        class_values = average_per_class(grouped[index])
        boot_control = v3.paired_class_bootstrap(class_values, control_class, seed=705)
        boot_knn1 = v3.paired_class_bootstrap(class_values, knn1_class, seed=706)
        eligible = (
            index not in {CONTROL_INDEX, KNN1_INDEX}
            and metrics["macro_top1"] >= control_metrics["macro_top1"] + MIN_GAIN_VS_CONTROL
            and boot_control["lower_95"] > 0
            and metrics["low_support_macro_top1"] >= control_metrics["low_support_macro_top1"]-BUCKET_TOLERANCE
            and metrics["high_support_macro_top1"] >= control_metrics["high_support_macro_top1"]-BUCKET_TOLERANCE
        )
        summaries.append({"index": index, "name": SPECS[index].name, "metrics": metrics, "bootstrap_vs_control": boot_control, "bootstrap_vs_knn1": boot_knn1, "eligible_research_challenger": eligible})
    eligible = [item for item in summaries if item["eligible_research_challenger"]]
    results["confirmation_summary"] = summaries
    results["verdict"] = "TRUE_B1_RESEARCH_CHALLENGER" if eligible else "PLATEAU_NO_TRUE_B1_CHALLENGER"
    results["research_challenger"] = max(eligible, key=lambda item: item["metrics"]["macro_top1"], default=None)
    write_json(RESULTS, results)
    write_document(results)


def write_document(results: dict[str, Any]) -> None:
    rows = [
        "# S/14 autoresearch v5 — true B1 replay (2026-08-12)", "",
        "V5 replaces the invalid B1 interpretation of V3/V4. It fail-closes on the actual B1 runtime projection path and SHA-256 before loading any projection tensor.", "",
        f"- True B1 projection SHA-256: `{EXPECTED_PROJECTION_SHA}`.",
        f"- Frozen matrix: {len(SPECS)} variants (47 V3 + 30 adaptive V4), all screened on the physical train-only cache.",
        f"- Outer: top {TOP_N} plus the NCM and cosine 1-NN baselines.",
        f"- Confirmation: top {CONFIRM_TOP_N} outer challengers plus both baselines, five folds on fresh seeds 42/73/101; selection seed 17 is excluded.",
        "- Balanced logloss is descriptive only and is never used to gate unlike scoring methods.",
        "- Research cache methods and the selected challenger are always `runtime_compatible=false`; V5 never authorizes packaging.",
        "- This is in-page row CV and is not evidence of new-page or business-set generalization.",
        f"- Verdict: `{results['verdict']}`.", "",
        "## Multi-seed confirmation", "",
        "| Method | Macro top-1 | Macro top-3 | Micro top-1 | Balanced logloss | Gain vs B1 NCM | CI95 vs B1 NCM | Eligible |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    control = results["confirmation_summary"][0]["metrics"]
    for item in results["confirmation_summary"]:
        metric = item["metrics"]
        boot = item["bootstrap_vs_control"]
        rows.append(f"| {item['name']} | {metric['macro_top1']:.6f} | {metric['macro_top3']:.6f} | {metric['micro_top1']:.6f} | {metric['balanced_logloss']:.6f} | {metric['macro_top1']-control['macro_top1']:+.6f} | [{boot['lower_95']:+.6f}, {boot['upper_95']:+.6f}] | {str(item['eligible_research_challenger']).lower()} |")
    rows += ["", "The historical-projection V3/V4 reports remain preserved solely as an audit trail; their method rankings motivated V5 but their claimed B1 metrics are superseded.", "", f"Machine-readable evidence: `{RESULTS.relative_to(REPO)}`.", ""]
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

#!/usr/bin/env python3
"""Bounded sparse-class episodic research for the sealed Tlachia cache.

This is intentionally a research-only runner.  It cannot read the business
test, annotations, or locked-test rows, and it never changes a runtime model.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SEALED_CACHE = REPO / "autoresearch/orchestrator-260813-1330/tlachia-s14/sealed/train-dev.features.pt"
INIT_PROJECTION = REPO / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime/weights/projection.pt"
REPORT_DIR = REPO / "autoresearch/orchestrator-260813-1330/tlachia-s14-sparse-v1"
DOCUMENT = REPO / "docs/autoresearch-tlachia-sparse-20260813.md"
EXPECTED_SPLITS = Counter(train=3763, dev=465)
FORBIDDEN_PARTS = {"business-final", "annotations", "locked_test", "locked-test"}
COMMON = {"learning_rate": 1e-5, "weight_decay": 1e-4, "epochs": 3,
          "episodes": 100, "n_way": 20, "k_shot": 3, "q_queries": 5,
          "temperature": 0.1}
SCREEN_SEED = 42
CONFIRM_SEEDS = (73, 101)


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    sparse_slots: int


SPECS = (Spec("control", 0),) + tuple(Spec(f"sparse-slots-{n}", n) for n in (2, 4, 6, 8, 10, 14, 20))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_once_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"write-once artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def assert_allowed_input(path: Path) -> None:
    if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"forbidden research input path: {path}")


def assert_specs() -> None:
    if len(SPECS) != 8 or SPECS[0] != Spec("control", 0):
        raise ValueError("protocol requires exactly one control and seven sparse specs")
    if [item.sparse_slots for item in SPECS] != [0, 2, 4, 6, 8, 10, 14, 20]:
        raise ValueError("sparse specification changed")


def validate_cache(cache: dict[str, Any]) -> None:
    required = {"features", "labels", "dataset_splits", "source_groups", "row_ids"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"sealed cache missing required fields: {sorted(missing)}")
    lengths = {len(cache[key]) for key in ("features", "labels", "dataset_splits", "source_groups", "row_ids")}
    if len(lengths) != 1:
        raise ValueError("sealed cache fields have inconsistent lengths")
    if Counter(cache["dataset_splits"]) != EXPECTED_SPLITS:
        raise ValueError(f"sealed cache must contain train/dev only: {dict(Counter(cache['dataset_splits']))}")
    if len(set(cache["row_ids"])) != len(cache["row_ids"]):
        raise ValueError("cache row IDs must be unique")


def load_cache(path: Path = SEALED_CACHE) -> dict[str, Any]:
    assert_allowed_input(path)
    if path.resolve() != SEALED_CACHE.resolve():
        raise ValueError("only the preregistered sealed cache is allowed")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_cache(cache)
    return cache


def partition(cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    validate_cache(cache)
    train_idx = [i for i, split in enumerate(cache["dataset_splits"]) if split == "train"]
    dev_idx = [i for i, split in enumerate(cache["dataset_splits"]) if split == "dev"]
    labels = cache["labels"]
    train_labels, dev_labels = labels[train_idx], labels[dev_idx]
    missing = sorted(set(dev_labels.tolist()) - set(train_labels.tolist()))
    if missing:
        raise ValueError(f"dev labels absent from train prototypes: {missing}")
    return (cache["features"][train_idx], train_labels, cache["features"][dev_idx], dev_labels,
            [str(cache["source_groups"][i]) for i in dev_idx], [str(cache["row_ids"][i]) for i in train_idx])


def train_index_by_class(labels: torch.Tensor) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        result[int(label)].append(index)
    return dict(result)


def episode_for_spec(by_class: dict[int, list[int]], spec: Spec, rng: random.Random) -> dict[str, list[int]]:
    """Make one row-disjoint episode; singleton classes deliberately remain prototype-only."""
    standard = sorted(label for label, rows in by_class.items() if len(rows) >= 8)
    sparse = sorted(label for label, rows in by_class.items() if 2 <= len(rows) <= 7)
    if len(standard) < COMMON["n_way"] - spec.sparse_slots or len(sparse) < spec.sparse_slots:
        raise ValueError("insufficient classes for preregistered episode")
    labels = rng.sample(sparse, spec.sparse_slots) + rng.sample(standard, COMMON["n_way"] - spec.sparse_slots)
    rng.shuffle(labels)
    support: list[int] = []
    query: list[int] = []
    episode_labels: list[int] = []
    for episode_label, class_label in enumerate(labels):
        rows = by_class[class_label]
        if len(rows) <= 7:
            a = min(3, len(rows) - 1)
            picked = rng.sample(rows, a + (len(rows) - a))
            support_pool, query_pool = picked[:a], picked[a:]
            # Replacement happens only after the two original row pools are disjoint.
            support.extend(rng.choices(support_pool, k=COMMON["k_shot"]))
            query.extend(rng.choices(query_pool, k=COMMON["q_queries"]))
        else:
            picked = rng.sample(rows, COMMON["k_shot"] + COMMON["q_queries"])
            support.extend(picked[:COMMON["k_shot"]])
            query.extend(picked[COMMON["k_shot"]:])
        episode_labels.extend([episode_label] * (COMMON["k_shot"] + COMMON["q_queries"]))
    support_labels = [label for label in range(COMMON["n_way"]) for _ in range(COMMON["k_shot"])]
    query_labels = [label for label in range(COMMON["n_way"]) for _ in range(COMMON["q_queries"])]
    if set(support) & set(query):
        raise AssertionError("support/query row overlap")
    return {"support": support, "query": query, "support_labels": support_labels, "query_labels": query_labels}


def make_episode_plan(by_class: dict[int, list[int]], spec: Spec, seed: int) -> list[dict[str, list[int]]]:
    rng = random.Random(seed)
    return [episode_for_spec(by_class, spec, rng) for _ in range(COMMON["episodes"] * COMMON["epochs"])]


def load_projection(input_dim: int, device: torch.device):
    assert_allowed_input(INIT_PROJECTION)
    sys.path.insert(0, str(REPO / "backend"))
    from codex_pipeline.models.projection_head import ProjectionHead
    model = ProjectionHead(input_dim=input_dim, embedding_dim=128).to(device)
    payload = torch.load(INIT_PROJECTION, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    return model


def train(spec: Spec, seed: int, train_features: torch.Tensor, train_labels: torch.Tensor, run_dir: Path) -> Path:
    del train_labels
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_class = train_index_by_class(train_labels) if False else None  # type guard kept below
    # The plan must be constructed from labels; its hash makes sampling auditable.
    # `train_labels` is passed through a local alias to keep this routine explicit.
    raise AssertionError("unreachable")


def train_from_plan(spec: Spec, seed: int, features: torch.Tensor, labels: torch.Tensor, plan: list[dict[str, list[int]]], run_dir: Path) -> Path:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_projection(features.shape[1], device)
    sys.path.insert(0, str(REPO / "backend"))
    from codex_pipeline.models.prototypical import PrototypicalLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON["learning_rate"], weight_decay=COMMON["weight_decay"])
    criterion = PrototypicalLoss(temperature=COMMON["temperature"])
    logs: list[dict[str, float | int]] = []
    model.train()
    for index, episode in enumerate(plan):
        support = features[episode["support"]].to(device)
        query = features[episode["query"]].to(device)
        support_labels = torch.tensor(episode["support_labels"], device=device)
        query_labels = torch.tensor(episode["query_labels"], device=device)
        output = criterion(model(support), support_labels, model(query), query_labels)
        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if (index + 1) % COMMON["episodes"] == 0:
            logs.append({"epoch": (index + 1) // COMMON["episodes"], "train_loss": float(output["loss"].detach().cpu()), "train_accuracy": float(output["accuracy"].detach().cpu())})
    checkpoint = run_dir / "checkpoints/latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": COMMON["epochs"] - 1, "model_state_dict": model.cpu().state_dict(), "config": common_config(spec, seed), "episode_plan_sha256": sha256(run_dir / "episode-plan.json")}, checkpoint)
    write_once_json(run_dir / "epoch-log.json", logs)
    return checkpoint


def common_config(spec: Spec, seed: int) -> dict[str, Any]:
    return {"name": spec.name, "sparse_slots": spec.sparse_slots, "seed": seed, **COMMON,
            "device_requested": "cuda", "device_resolved": "cuda" if torch.cuda.is_available() else "cpu"}


def evaluate(checkpoint: Path, train_features: torch.Tensor, train_labels: torch.Tensor, dev_features: torch.Tensor, dev_labels: torch.Tensor, dev_groups: list[str]) -> dict[str, Any]:
    del dev_groups
    sys.path.insert(0, str(REPO / "backend"))
    from codex_pipeline.models.projection_head import ProjectionHead
    from codex_pipeline.models.prototypical import compute_prototypes
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    model = ProjectionHead(input_dim=train_features.shape[1], embedding_dim=128)
    model.load_state_dict(state, strict=True); model.eval()
    with torch.no_grad():
        train_emb, dev_emb = model(train_features), model(dev_features)
        proto_labels = train_labels.unique(sorted=True)
        prototypes = compute_prototypes(train_emb, train_labels)
        target_map = {int(label): i for i, label in enumerate(proto_labels.tolist())}
        targets = torch.tensor([target_map[int(label)] for label in dev_labels.tolist()])
        logits = dev_emb @ prototypes.T / COMMON["temperature"]
        losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = proto_labels[logits.argmax(dim=1)]
    supports = Counter(int(x) for x in train_labels.tolist())
    correct = predictions.eq(dev_labels)
    per_class: dict[str, dict[str, float | int | str]] = {}
    for label in dev_labels.unique(sorted=True).tolist():
        mask = dev_labels.eq(label); support = supports[int(label)]
        bin_name = "1" if support == 1 else "2-3" if support <= 3 else "4-7" if support <= 7 else ">=8"
        per_class[str(label)] = {"train_support": support, "dev_support": int(mask.sum()), "bin": bin_name,
                                 "accuracy": float(correct[mask].float().mean()), "logloss": float(losses[mask].mean())}
    bins = {name: [float(v["accuracy"]) for v in per_class.values() if v["bin"] == name] for name in ("1", "2-3", "4-7", ">=8")}
    macro = sum(float(v["accuracy"]) for v in per_class.values()) / len(per_class)
    sparse_values = bins["2-3"] + bins["4-7"]
    return {"macro_global": macro, "micro": float(correct.float().mean()),
            "balanced_logloss": sum(float(v["logloss"]) for v in per_class.values()) / len(per_class),
            "macro_by_train_support": {name: (sum(values) / len(values) if values else None) for name, values in bins.items()},
            "primary_sparse_macro": sum(sparse_values) / len(sparse_values) if sparse_values else None, "per_class": per_class}


def paired_bootstrap(candidate: dict[str, Any], control: dict[str, Any], replicates: int = 2000, seed: int = 20260813) -> dict[str, float]:
    labels = sorted(set(candidate["per_class"]) & set(control["per_class"]))
    labels = [label for label in labels if 2 <= int(candidate["per_class"][label]["train_support"]) <= 7]
    if not labels: raise ValueError("no sparse dev classes for bootstrap")
    deltas = [float(candidate["per_class"][x]["accuracy"]) - float(control["per_class"][x]["accuracy"]) for x in labels]
    rng = random.Random(seed); samples = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(replicates))
    return {"mean": sum(deltas) / len(deltas), "lower_95": samples[int(.025 * replicates)], "upper_95": samples[int(.975 * replicates)]}


def gate(candidate_metrics: dict[str, Any], control_metrics: dict[str, Any], fresh_deltas: Iterable[float]) -> dict[str, Any]:
    bootstrap = paired_bootstrap(candidate_metrics, control_metrics)
    drops = [label for label, item in candidate_metrics["per_class"].items() if float(item["accuracy"]) < float(control_metrics["per_class"][label]["accuracy"]) - .20]
    sparse_gain = float(candidate_metrics["primary_sparse_macro"]) - float(control_metrics["primary_sparse_macro"])
    fresh = list(fresh_deltas)
    passes = {"sparse_gain_at_least_0_01": sparse_gain >= .01, "paired_bootstrap_lower_95_positive": bootstrap["lower_95"] > 0,
              "global_macro_not_more_than_0_005_below_control": float(candidate_metrics["macro_global"]) >= float(control_metrics["macro_global"]) - .005,
              "both_fresh_seed_gains_positive": len(fresh) == 2 and all(x > 0 for x in fresh), "no_class_drop_over_0_20": not drops}
    return {"pass": all(passes.values()), "passes": passes, "sparse_gain": sparse_gain, "paired_class_bootstrap": bootstrap, "fresh_seed_deltas": fresh, "class_drops_over_0_20": drops, "promotion_eligible": False}


def manifest() -> dict[str, Any]:
    assert_specs()
    return {"protocol": "tlachia-sparse-episodic.v1", "sealed_cache": {"path": str(SEALED_CACHE), "sha256": sha256(SEALED_CACHE)},
            "init_projection": {"path": str(INIT_PROJECTION), "sha256": sha256(INIT_PROJECTION)},
            "specs": [dataclasses.asdict(x) | COMMON for x in SPECS], "screen_seed": SCREEN_SEED, "confirmation_seeds": list(CONFIRM_SEEDS),
            "selection": {"primary": "macro dev accuracy of classes with train support 2-7", "top_sparse_specs": 2,
                          "final_gate": "gain>=.01, bootstrap_lower95>0, global>=control-.005, both fresh positive, no drop>.20"},
            "forbidden_inputs": sorted(FORBIDDEN_PARTS), "promotion_eligible": False,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}}


def prepare() -> dict[str, Any]:
    for path in (SEALED_CACHE, INIT_PROJECTION):
        assert_allowed_input(path)
        if not path.is_file(): raise FileNotFoundError(path)
    load_cache(); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    value = manifest(); write_once_json(REPORT_DIR / "frozen-manifest.json", value)
    return value


def one_run(spec: Spec, seed: int, cache: dict[str, Any], phase: str) -> dict[str, Any]:
    train_features, train_labels, dev_features, dev_labels, dev_groups, train_row_ids = partition(cache)
    by_class = train_index_by_class(train_labels)
    run_dir = REPORT_DIR / "runs" / f"{phase}-{spec.name}-seed{seed}"
    result_path = run_dir / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    plan = make_episode_plan(by_class, spec, seed)
    plan_rows = [{key: [train_row_ids[i] for i in values] if key in {"support", "query"} else values for key, values in episode.items()} for episode in plan]
    write_once_json(run_dir / "config.json", common_config(spec, seed))
    write_once_json(run_dir / "invocation.json", {"phase": phase, "spec": spec.name, "seed": seed, "argv": [sys.executable, str(Path(__file__).resolve()), "run-all"], "environment": {"PYTHONHASHSEED": str(seed)}})
    write_once_json(run_dir / "episode-plan.json", plan_rows)
    started = time.time(); checkpoint = train_from_plan(spec, seed, train_features, train_labels, plan, run_dir)
    metrics = evaluate(checkpoint, train_features, train_labels, dev_features, dev_labels, dev_groups)
    result = {"phase": phase, "spec": dataclasses.asdict(spec), "seed": seed, "metrics": metrics, "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)}, "elapsed_seconds": time.time() - started}
    write_once_json(result_path, result)
    (run_dir / "train.log").write_text(f"completed {spec.name} seed={seed} phase={phase}\n", encoding="utf-8")
    return result


def run_all() -> dict[str, Any]:
    prepare(); cache = load_cache(); screening = [one_run(spec, SCREEN_SEED, cache, "screening") for spec in SPECS]
    sparse = sorted(screening[1:], key=lambda item: float(item["metrics"]["primary_sparse_macro"]), reverse=True)[:2]
    selected = [next(spec for spec in SPECS if spec.name == item["spec"]["name"]) for item in sparse]
    confirmations: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        confirmations.append(one_run(SPECS[0], seed, cache, "confirmation"))
        for spec in selected: confirmations.append(one_run(spec, seed, cache, "confirmation"))
    control_screen = screening[0]
    gates = []
    for candidate in sparse:
        name = candidate["spec"]["name"]
        fresh = []
        for seed in CONFIRM_SEEDS:
            c = next(x for x in confirmations if x["seed"] == seed and x["spec"]["name"] == name)
            b = next(x for x in confirmations if x["seed"] == seed and x["spec"]["name"] == "control")
            fresh.append(float(c["metrics"]["primary_sparse_macro"]) - float(b["metrics"]["primary_sparse_macro"]))
        gates.append({"name": name, **gate(candidate["metrics"], control_screen["metrics"], fresh)})
    final = {"screening": screening, "selected_sparse_specs": [x.name for x in selected], "confirmations": confirmations, "gates": gates, "promotion_eligible": False}
    write_once_json(REPORT_DIR / "results.json", final); write_document(final)
    return final


def write_document(results: dict[str, Any]) -> None:
    lines = ["# Tlachia sparse-class autoresearch (2026-08-13)", "", "Research only. Inputs are the preregistered sealed train+dev cache and historical projection initializer; business-final, annotations and locked-test are excluded physically and by path guard.", "", "| Screening run | Sparse macro (primary) | Global macro | Micro |", "| --- | ---: | ---: | ---: |"]
    for item in results["screening"]:
        m = item["metrics"]; lines.append(f"| {item['spec']['name']} | {m['primary_sparse_macro']:.4f} | {m['macro_global']:.4f} | {m['micro']:.4f} |")
    lines.extend(["", "## Final gates", "", "| Candidate | Pass | Sparse gain | Bootstrap lower 95% |", "| --- | --- | ---: | ---: |"])
    for item in results["gates"]: lines.append(f"| {item['name']} | {item['pass']} | {item['sparse_gain']:.4f} | {item['paired_class_bootstrap']['lower_95']:.4f} |")
    lines.extend(["", "Promotion eligible: **false**. This campaign cannot alter registry or runtime artifacts.", ""])
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")


def verify() -> dict[str, Any]:
    value = prepare(); cache = load_cache(); validate_cache(cache)
    return {"ok": True, "manifest_sha256": hashlib.sha256(canonical(value).encode()).hexdigest(), "cache_sha256": sha256(SEALED_CACHE), "promotion_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-all", "verify"))
    args = parser.parse_args()
    outcome = prepare() if args.command == "prepare" else run_all() if args.command == "run-all" else verify()
    print(json.dumps(outcome if args.command == "verify" else {"command": args.command, "report_dir": str(REPORT_DIR), "promotion_eligible": False}, sort_keys=True))


if __name__ == "__main__": main()

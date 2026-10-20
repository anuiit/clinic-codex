#!/usr/bin/env python3
"""Cold-start architecture comparison on the legacy and Tlachia element caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OLD_CACHE = ROOT / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/training_data/precomputed/features.pt"
NEW_CACHE = ROOT / "backend/training_corpus/snapshots/snapshot-tlachia-6d98f8520a3b435b5dd8/precomputed/features.pt"
B1_PROJECTION = ROOT / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/runtime/weights/projection.pt"
OUTPUT = ROOT / "reports/autoresearch-dual-architectures-20260814-v4"
FORBIDDEN = ("business-final", "annotations", "locked_test")
ARCHITECTURES = ("v0_mlp", "v1_ln_mlp", "v2_ln_linear", "v3_residual")
REGIMES = ("old", "new", "combined")
SCREEN_SEED = 42
CONFIRM_SEEDS = (73, 101)
STEPS = 1500
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LOGIT_SCALE = 16.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def assert_allowed(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN):
        raise ValueError(f"forbidden research input: {path}")


@dataclass
class Dataset:
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    dev_x: torch.Tensor
    dev_y: torch.Tensor
    dev_groups: list[str]
    class_names: dict[int, str]
    counts: dict[str, int]


def load_dataset(name: str, path: Path) -> Dataset:
    assert_allowed(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = {"features", "labels", "dataset_splits", "source_groups", "class_names", "row_ids"}
    if required - set(data):
        raise ValueError(f"{name} cache missing {sorted(required - set(data))}")
    features = data["features"].float().contiguous()
    labels = data["labels"].long().contiguous()
    splits = list(data["dataset_splits"])
    groups = list(data["source_groups"])
    if features.ndim != 2 or features.shape[1] != 384 or not len(features) or len(features) != len(labels):
        raise ValueError(f"{name} feature contract mismatch")
    if not torch.isfinite(features).all() or len(features) != len(splits) or len(labels) != len(groups):
        raise ValueError(f"{name} cache contains invalid rows")
    counts = dict(Counter(splits))
    if set(counts) != {"train", "dev", "locked_test"}:
        raise ValueError(f"{name} persisted split contract mismatch: {counts}")
    train_idx = torch.tensor([i for i, split in enumerate(splits) if split == "train"], dtype=torch.long)
    dev_idx = torch.tensor([i for i, split in enumerate(splits) if split == "dev"], dtype=torch.long)
    train_y, dev_y = labels[train_idx], labels[dev_idx]
    if set(range(286)) != set(train_y.tolist()) or not set(dev_y.tolist()).issubset(set(train_y.tolist())):
        raise ValueError(f"{name} class coverage mismatch")
    class_names = {int(key): str(value) for key, value in data["class_names"].items()}
    if sorted(class_names) != list(range(286)):
        raise ValueError(f"{name} taxonomy mismatch")
    return Dataset(
        name=name,
        train_x=features[train_idx],
        train_y=train_y,
        dev_x=features[dev_idx],
        dev_y=dev_y,
        dev_groups=[groups[i] for i in dev_idx.tolist()],
        class_names=class_names,
        counts=counts,
    )


def feature_digest(row: torch.Tensor) -> str:
    return hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()


def combined_train(old: Dataset, new: Dataset) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    dev_hashes = {feature_digest(row) for row in torch.cat((old.dev_x, new.dev_x), dim=0)}
    kept: dict[str, tuple[torch.Tensor, int]] = {}
    conflicts: set[str] = set()
    removed_dev = 0
    removed_duplicates = 0
    for row, label in zip(torch.cat((old.train_x, new.train_x)), torch.cat((old.train_y, new.train_y))):
        digest = feature_digest(row)
        if digest in dev_hashes:
            removed_dev += 1
            continue
        if digest in kept:
            if kept[digest][1] != int(label):
                conflicts.add(digest)
            else:
                removed_duplicates += 1
            continue
        kept[digest] = (row, int(label))
    for digest in conflicts:
        kept.pop(digest, None)
    if not kept:
        raise ValueError("combined train became empty")
    values = list(kept.values())
    labels = torch.tensor([value[1] for value in values], dtype=torch.long)
    if set(labels.tolist()) != set(range(286)):
        raise ValueError("combined train lost runtime classes")
    return torch.stack([value[0] for value in values]), labels, {
        "rows": len(values),
        "removed_exact_dev_features": removed_dev,
        "removed_exact_train_duplicates": removed_duplicates,
        "removed_conflicting_feature_groups": len(conflicts),
    }


class Architecture(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind == "v0_mlp":
            self.net = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.1), nn.Linear(384, 128))
        elif kind == "v1_ln_mlp":
            self.net = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.1), nn.Linear(384, 128))
        elif kind == "v2_ln_linear":
            self.net = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128))
        elif kind == "v3_residual":
            self.norm = nn.LayerNorm(384)
            self.trunk = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.1), nn.Linear(384, 128))
            self.skip = nn.Linear(384, 128)
            self.net = None
        else:
            raise ValueError(f"unknown architecture {kind}")
        self.kind = kind

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.kind == "v3_residual":
            value = self.norm(features)
            output = (self.trunk(value) + self.skip(value)) / math.sqrt(2.0)
        else:
            output = self.net(features)
        return F.normalize(output, p=2, dim=-1)


def init_prototypes(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    sums = torch.zeros(286, 128, device=device)
    counts = torch.zeros(286, device=device)
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            x = features[start : start + 2048].to(device)
            y = labels[start : start + 2048].to(device)
            z = model(x)
            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
    if (counts == 0).any():
        raise ValueError("prototype initialization lost a class")
    return F.normalize(sums / counts[:, None], p=2, dim=1)


def class_indices(labels: torch.Tensor) -> dict[int, list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_class[int(label)].append(index)
    return dict(by_class)


def sample_balanced(by_class: dict[int, list[int]], generator: random.Random, batch_size: int) -> torch.Tensor:
    classes = sorted(by_class)
    return torch.tensor([generator.choice(by_class[generator.choice(classes)]) for _ in range(batch_size)], dtype=torch.long)


def train_model(kind: str, features: torch.Tensor, labels: torch.Tensor, seed: int, device: torch.device) -> Architecture:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Architecture(kind).to(device)
    weights = nn.Parameter(init_prototypes(model, features, labels, device))
    optimizer = torch.optim.AdamW(
        [*model.parameters(), weights], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = random.Random(seed)
    by_class = class_indices(labels)
    model.train()
    for _ in range(STEPS):
        index = sample_balanced(by_class, generator, min(BATCH_SIZE, len(labels)))
        x, y = features[index].to(device), labels[index].to(device)
        logits = LOGIT_SCALE * model(x) @ F.normalize(weights, p=2, dim=1).T
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss for {kind}")
    return model.cpu().eval()


def prototypes(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros(286, 128)
    counts = torch.zeros(286)
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            z = model(features[start : start + 2048])
            y = labels[start : start + 2048]
            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))
    return F.normalize(sums / counts[:, None], p=2, dim=1)


def metrics(model: nn.Module, proto: torch.Tensor, dataset: Dataset) -> dict[str, Any]:
    with torch.no_grad():
        embedding = model(dataset.dev_x)
        logits = LOGIT_SCALE * embedding @ proto.T
        prediction = logits.argmax(dim=1)
        top3 = logits.topk(3, dim=1).indices
        loss = F.cross_entropy(logits, dataset.dev_y, reduction="none")
    correct = prediction.eq(dataset.dev_y)
    classes = dataset.dev_y.unique(sorted=True)
    class_accuracy = [float(correct[dataset.dev_y.eq(label)].float().mean()) for label in classes]
    class_loss = [float(loss[dataset.dev_y.eq(label)].mean()) for label in classes]
    return {
        "rows": len(dataset.dev_y),
        "observed_classes": len(classes),
        "top1_micro": float(correct.float().mean()),
        "top1_macro": sum(class_accuracy) / len(class_accuracy),
        "top3_micro": float(top3.eq(dataset.dev_y[:, None]).any(dim=1).float().mean()),
        "balanced_logloss": sum(class_loss) / len(class_loss),
        "prediction_sha256": semantic_hash(prediction.tolist()),
    }


def evaluate(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, old: Dataset, new: Dataset) -> dict[str, Any]:
    model.eval()
    proto = prototypes(model, train_x, train_y)
    old_metrics, new_metrics = metrics(model, proto, old), metrics(model, proto, new)
    return {
        "old_dev": old_metrics,
        "new_dev": new_metrics,
        "mean_macro_top1": (old_metrics["top1_macro"] + new_metrics["top1_macro"]) / 2.0,
        "min_macro_top1": min(old_metrics["top1_macro"], new_metrics["top1_macro"]),
    }


def training_data(regime: str, old: Dataset, new: Dataset, combined: tuple[torch.Tensor, torch.Tensor, dict[str, int]]):
    if regime == "old":
        return old.train_x, old.train_y, {"rows": len(old.train_y)}
    if regime == "new":
        return new.train_x, new.train_y, {"rows": len(new.train_y)}
    return combined


def b1_reference(regime: str, old: Dataset, new: Dataset, combined, device: torch.device) -> dict[str, Any]:
    model = Architecture("v0_mlp")
    state = torch.load(B1_PROJECTION, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    train_x, train_y, audit = training_data(regime, old, new, combined)
    return {"name": "b1_warm", "regime": regime, "seed": None, "training": audit, "metrics": evaluate(model, train_x, train_y, old, new)}


def protocol() -> dict[str, Any]:
    specs = [{"architecture": architecture, "regime": regime, "seed": SCREEN_SEED} for regime in REGIMES for architecture in ARCHITECTURES]
    return {
        "schema": "autoresearch-dual-dataset-architectures.v1",
        "status": "frozen_before_first_candidate_score",
        "runner_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "old_cache": {"path": str(OLD_CACHE), "sha256": sha256_file(OLD_CACHE)},
            "new_cache": {"path": str(NEW_CACHE), "sha256": sha256_file(NEW_CACHE)},
            "b1_projection": {"path": str(B1_PROJECTION), "sha256": sha256_file(B1_PROJECTION)},
        },
        "architectures": list(ARCHITECTURES),
        "regimes": list(REGIMES),
        "screen_specs": specs,
        "confirmation": {"top_k": 2, "seeds": list(CONFIRM_SEEDS), "paired_v0": True},
        "training": {"steps": STEPS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "logit_scale": LOGIT_SCALE},
        "selection": {"primary": "mean_macro_top1", "minimum_gain": 0.01, "maximum_domain_drop": 0.005},
        "boundaries": {
            "locked_test_scored": False,
            "locked_test_rows_loaded_then_filtered": True,
            "business_final_read": False,
            "runtime_write": False,
            "promotion": False,
        },
    }


def validate_protocol(value: dict[str, Any]) -> None:
    expected = protocol()
    if value != expected:
        raise ValueError("frozen protocol or input hash drift")


def select_top_candidates(screen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (row for row in screen if row["architecture"] != "v0_mlp"),
        key=lambda row: row["metrics"]["mean_macro_top1"],
        reverse=True,
    )[:2]


def run(output: Path, device: torch.device) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"write-once output already exists: {output}")
    output.mkdir(parents=True)
    frozen = protocol()
    write_json_once(output / "protocol.json", frozen)
    old, new = load_dataset("old", OLD_CACHE), load_dataset("new", NEW_CACHE)
    if old.class_names != new.class_names:
        raise ValueError("old/new runtime taxonomy differs")
    combined = combined_train(old, new)
    references = [b1_reference(regime, old, new, combined, device) for regime in REGIMES]
    screen: list[dict[str, Any]] = []
    for spec in frozen["screen_specs"]:
        train_x, train_y, audit = training_data(spec["regime"], old, new, combined)
        model = train_model(spec["architecture"], train_x, train_y, spec["seed"], device)
        screen.append({**spec, "training": audit, "parameter_count": sum(p.numel() for p in model.parameters()), "metrics": evaluate(model, train_x, train_y, old, new)})
    top = select_top_candidates(screen)
    confirmation: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for winner in top:
        for seed in CONFIRM_SEEDS:
            for architecture in ("v0_mlp", winner["architecture"]):
                key = (architecture, winner["regime"], seed)
                if key in seen:
                    continue
                seen.add(key)
                train_x, train_y, audit = training_data(winner["regime"], old, new, combined)
                model = train_model(architecture, train_x, train_y, seed, device)
                confirmation.append({"architecture": architecture, "regime": winner["regime"], "seed": seed, "training": audit, "parameter_count": sum(p.numel() for p in model.parameters()), "metrics": evaluate(model, train_x, train_y, old, new)})
    aggregates = []
    for winner in top:
        candidate = [winner] + [row for row in confirmation if row["architecture"] == winner["architecture"] and row["regime"] == winner["regime"]]
        controls = [row for row in screen + confirmation if row["architecture"] == "v0_mlp" and row["regime"] == winner["regime"] and row["seed"] in {item["seed"] for item in candidate}]
        candidate_by_seed = {row["seed"]: row for row in candidate}
        control_by_seed = {row["seed"]: row for row in controls}
        seeds = sorted(set(candidate_by_seed) & set(control_by_seed))
        deltas = [candidate_by_seed[seed]["metrics"]["mean_macro_top1"] - control_by_seed[seed]["metrics"]["mean_macro_top1"] for seed in seeds]
        old_drops = [candidate_by_seed[seed]["metrics"]["old_dev"]["top1_macro"] - control_by_seed[seed]["metrics"]["old_dev"]["top1_macro"] for seed in seeds]
        new_drops = [candidate_by_seed[seed]["metrics"]["new_dev"]["top1_macro"] - control_by_seed[seed]["metrics"]["new_dev"]["top1_macro"] for seed in seeds]
        mean_delta = sum(deltas) / len(deltas)
        keep = mean_delta >= 0.01 and min(old_drops) >= -0.005 and min(new_drops) >= -0.005
        aggregates.append({"architecture": winner["architecture"], "regime": winner["regime"], "seeds": seeds, "mean_delta_vs_paired_v0": mean_delta, "old_domain_deltas": old_drops, "new_domain_deltas": new_drops, "verdict": "KEEP" if keep else "DISCARD"})
    result = {
        "schema": "autoresearch-dual-dataset-architectures.results.v1",
        "protocol_sha256": semantic_hash(frozen),
        "device": str(device),
        "dataset_counts": {"old": old.counts, "new": new.counts, "combined_train": combined[2]},
        "references": references,
        "screen": screen,
        "selected_for_confirmation": [{"architecture": row["architecture"], "regime": row["regime"]} for row in top],
        "confirmation": confirmation,
        "aggregates": aggregates,
        "automatic_promotion_allowed": False,
    }
    result["payload_sha256"] = semantic_hash(result)
    write_json_once(output / "results.json", result)
    return result


def verify(output: Path) -> dict[str, Any]:
    frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(frozen)
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    claimed = result.pop("payload_sha256")
    if semantic_hash(result) != claimed or result["protocol_sha256"] != semantic_hash(frozen):
        raise ValueError("result hash mismatch")
    if len(result["screen"]) != 12 or len(result["references"]) != 3:
        raise ValueError("incomplete experiment matrix")
    if any(row["metrics"]["old_dev"]["rows"] != 56 or row["metrics"]["new_dev"]["rows"] != 465 for row in result["screen"]):
        raise ValueError("dev boundary mismatch")
    return {"status": "VERIFIED", "result_sha256": sha256_file(output / "results.json"), "screen_runs": len(result["screen"]), "confirmation_runs": len(result["confirmation"]), "verdicts": result["aggregates"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-all", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "run-all":
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        result = run(args.output, device)
        print(json.dumps({"status": "COMPLETED", "output": str(args.output), "payload_sha256": result["payload_sha256"]}, sort_keys=True))
    else:
        print(json.dumps(verify(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

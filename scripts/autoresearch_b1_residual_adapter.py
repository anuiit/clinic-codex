#!/usr/bin/env python3
"""Evaluate a zero-initialized residual adapter on top of frozen B1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import autoresearch_dual_dataset_architectures as dual


OUTPUT = dual.ROOT / "reports/autoresearch-b1-residual-adapter-20260814-v2"
SEEDS = (42, 73, 101)
STEPS = 500
LEARNING_RATE = 3e-4


class ResidualAdapter(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.base = dual.Architecture("v0_mlp")
        self.base.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.adapter = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        correction = torch.tanh(self.gate) * self.adapter(features)
        return F.normalize(self.base(features) + correction, p=2, dim=-1)

    def train(self, mode: bool = True) -> "ResidualAdapter":
        super().train(mode)
        self.base.eval()
        return self


def protocol() -> dict[str, Any]:
    return {
        "schema": "autoresearch-b1-residual-adapter.v1",
        "status": "frozen_after_exploratory_hyperparameter_probe_before_multiseed_confirmation",
        "runner_sha256": dual.sha256_file(Path(__file__)),
        "dependency_sha256": dual.sha256_file(Path(dual.__file__)),
        "inputs": dual.protocol()["inputs"],
        "architecture": "frozen_b1_plus_zero_initialized_ln_linear_384x128_scalar_gate",
        "training": {"regime": "combined", "steps": STEPS, "batch_size": dual.BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": dual.WEIGHT_DECAY, "logit_scale": dual.LOGIT_SCALE, "seeds": list(SEEDS)},
        "selection_caveat": "learning_rate_and_steps_were_chosen_after_an_exploratory_read_of_the_same_dev_sets",
        "boundaries": {
            "locked_test_scored": False,
            "locked_test_rows_loaded_then_filtered": True,
            "business_final_read": False,
            "runtime_write": False,
            "promotion": False,
        },
    }


def train(features: torch.Tensor, labels: torch.Tensor, seed: int, device: torch.device) -> ResidualAdapter:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    state = torch.load(dual.B1_PROJECTION, map_location="cpu", weights_only=True)
    model = ResidualAdapter(state).to(device)
    weights = nn.Parameter(dual.init_prototypes(model, features, labels, device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad] + [weights],
        lr=LEARNING_RATE,
        weight_decay=dual.WEIGHT_DECAY,
    )
    generator, by_class = random.Random(seed), dual.class_indices(labels)
    model.train()
    for _ in range(STEPS):
        index = dual.sample_balanced(by_class, generator, dual.BATCH_SIZE)
        x, y = features[index].to(device), labels[index].to(device)
        logits = dual.LOGIT_SCALE * model(x) @ F.normalize(weights, p=2, dim=1).T
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite adapter loss")
    return model.cpu().eval()


def run(output: Path, device: torch.device) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"write-once output exists: {output}")
    output.mkdir(parents=True)
    frozen = protocol()
    dual.write_json_once(output / "protocol.json", frozen)
    old = dual.load_dataset("old", dual.OLD_CACHE)
    new = dual.load_dataset("new", dual.NEW_CACHE)
    combined = dual.combined_train(old, new)
    features, labels, audit = combined
    reference = dual.b1_reference("combined", old, new, combined, device)
    runs = []
    for seed in SEEDS:
        model = train(features, labels, seed, device)
        runs.append({
            "seed": seed,
            "gate": float(torch.tanh(model.gate)),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "metrics": dual.evaluate(model, features, labels, old, new),
        })
    old_delta = [row["metrics"]["old_dev"]["top1_macro"] - reference["metrics"]["old_dev"]["top1_macro"] for row in runs]
    new_delta = [row["metrics"]["new_dev"]["top1_macro"] - reference["metrics"]["new_dev"]["top1_macro"] for row in runs]
    mean_delta = sum(row["metrics"]["mean_macro_top1"] - reference["metrics"]["mean_macro_top1"] for row in runs) / len(runs)
    promising = mean_delta >= 0.01 and min(old_delta) >= -0.005 and min(new_delta) >= -0.005
    result = {
        "schema": "autoresearch-b1-residual-adapter.results.v1",
        "protocol_sha256": dual.semantic_hash(frozen),
        "training": audit,
        "reference": reference,
        "runs": runs,
        "aggregate": {
            "mean_macro_delta_vs_b1": mean_delta,
            "old_macro_deltas": old_delta,
            "new_macro_deltas": new_delta,
            "verdict": "PROMISING_EXPLORATORY" if promising else "DISCARD",
            "automatic_keep_allowed": False,
            "reason": "same dev sets were used for the preceding hyperparameter probe",
        },
        "automatic_promotion_allowed": False,
    }
    result["payload_sha256"] = dual.semantic_hash(result)
    dual.write_json_once(output / "results.json", result)
    return result


def verify(output: Path) -> dict[str, Any]:
    frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if frozen != protocol():
        raise ValueError("adapter protocol drift")
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    claimed = result.pop("payload_sha256")
    if dual.semantic_hash(result) != claimed or result["protocol_sha256"] != dual.semantic_hash(frozen):
        raise ValueError("adapter result hash mismatch")
    if [row["seed"] for row in result["runs"]] != list(SEEDS):
        raise ValueError("adapter seed matrix incomplete")
    return {"status": "VERIFIED", "result_sha256": dual.sha256_file(output / "results.json"), "aggregate": result["aggregate"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-all", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "run-all":
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        result = run(args.output, device)
        print(json.dumps({"status": "COMPLETED", "payload_sha256": result["payload_sha256"], "output": str(args.output)}, sort_keys=True))
    else:
        print(json.dumps(verify(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

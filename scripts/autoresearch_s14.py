#!/usr/bin/env python3
"""Frozen, development-only S14 projection-head sweep.

This is deliberately a research harness, not a registry or runtime workflow.
It only reads the supplied immutable feature cache and initializer, trains
independent projections, and writes evidence below reports/.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml


REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "reports" / "autoresearch-s14-20260812-v2"
DOCUMENT = REPO / "docs" / "autoresearch-s14-20260812.md"
CACHE = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/training_data/precomputed/features.pt"
CACHE_PROVENANCE = CACHE.with_suffix(".pt.prov.json")
INIT_PROJECTION = REPO / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime/weights/projection.pt"
TRAIN = REPO / "backend/codex_pipeline/scripts/train.py"
PYTHON = REPO / "backend/.venv/bin/python3"


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    learning_rate: float = 1e-5
    epochs: int = 3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    noise_std: float = 0.0
    mixup_prob: float = 0.0

    def changed_factors(self, control: "Spec") -> list[str]:
        return [
            field.name
            for field in dataclasses.fields(self)
            if field.name != "name" and getattr(self, field.name) != getattr(control, field.name)
        ]


CONTROL = Spec(name="control")
SPECS = (
    CONTROL,
    Spec(name="lr-3e-6", learning_rate=3e-6),
    Spec(name="lr-3e-5", learning_rate=3e-5),
    Spec(name="epochs-1", epochs=1),
    Spec(name="epochs-6", epochs=6),
    Spec(name="wd-0", weight_decay=0.0),
    Spec(name="wd-1e-3", weight_decay=1e-3),
    Spec(name="temp-0.05", temperature=0.05),
    Spec(name="temp-0.2", temperature=0.2),
    Spec(name="noise-0.01", noise_std=0.01),
    Spec(name="mixup-0.25", mixup_prob=0.25),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_specs(specs: tuple[Spec, ...] = SPECS) -> None:
    if len(specs) != 11 or specs[0] != CONTROL:
        raise ValueError("S14 requires the fixed control plus ten candidates")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("S14 spec names must be unique")
    for spec in specs[1:]:
        changes = spec.changed_factors(CONTROL)
        if len(changes) != 1:
            raise ValueError(f"{spec.name} must change exactly one factor, got {changes}")


def assert_allowed_input_path(path: Path) -> None:
    """Reject paths that could widen the frozen research input boundary."""
    blocked = {"business-final", "annotations", "locked_test"}
    if any(part.lower() in blocked for part in path.parts):
        raise ValueError(f"forbidden research input path: {path}")


def validate_sealed_cache(cache: dict[str, Any]) -> None:
    """Prove this cache contains exactly the permitted train/dev partitions."""
    required = {"features", "labels", "dataset_splits", "source_groups"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"cache missing required fields: {sorted(missing)}")
    features, labels = cache["features"], cache["labels"]
    splits, source_groups = list(cache["dataset_splits"]), list(cache["source_groups"])
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("features and labels must be tensors")
    if len(features) != len(labels) or len(labels) != len(splits) or len(labels) != len(source_groups):
        raise ValueError("cache fields have inconsistent lengths")
    split_counts = Counter(splits)
    expected = {"train": 9128, "dev": 56}
    if split_counts != expected:
        raise ValueError(f"sealed cache must contain only train/dev rows: {dict(split_counts)}")
    if "locked_test" in splits:
        raise ValueError("sealed cache contains a locked row")


def split_train_dev(cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Select the only two partitions allowed by an already sealed cache."""
    validate_sealed_cache(cache)
    features, labels = cache["features"], cache["labels"]
    splits, source_groups = list(cache["dataset_splits"]), list(cache["source_groups"])
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    dev_indices = [index for index, split in enumerate(splits) if split == "dev"]
    if len(train_indices) != 9128 or len(dev_indices) != 56:
        raise ValueError("train/dev split guard failed")
    train_index = torch.tensor(train_indices, dtype=torch.long)
    dev_index = torch.tensor(dev_indices, dtype=torch.long)
    train_labels, dev_labels = labels[train_index], labels[dev_index]
    absent = sorted(set(dev_labels.tolist()) - set(train_labels.tolist()))
    if absent:
        raise ValueError(f"dev labels missing from train-only prototypes: {absent}")
    return features[train_index], train_labels, features[dev_index], dev_labels, [source_groups[i] for i in dev_indices]


def seal_cache(source_cache: Path, target: Path) -> dict[str, Any]:
    """Create once a physical train+dev cache that omits every locked row."""
    if target.exists():
        sealed = torch.load(target, map_location="cpu", weights_only=False)
        validate_sealed_cache(sealed)
        return {"path": str(target), "sha256": sha256(target)}
    source = torch.load(source_cache, map_location="cpu", weights_only=False)
    required = {"features", "labels", "dataset_splits", "source_groups", "class_names", "hidden_dim"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"source cache missing fields: {sorted(missing)}")
    splits = list(source["dataset_splits"])
    if Counter(splits) != {"train": 9128, "dev": 56, "locked_test": 82}:
        raise ValueError("source cache does not match the preregistered persisted split")
    selected = torch.tensor([index for index, split in enumerate(splits) if split in {"train", "dev"}], dtype=torch.long)
    sealed = {
        "features": source["features"][selected].clone(),
        "labels": source["labels"][selected].clone(),
        "source_groups": [source["source_groups"][index] for index in selected.tolist()],
        "dataset_splits": [splits[index] for index in selected.tolist()],
        "class_names": source["class_names"],
        "hidden_dim": source["hidden_dim"],
    }
    validate_sealed_cache(sealed)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sealed, target)
    return {"path": str(target), "sha256": sha256(target)}


def evaluate_projection(
    projection_path: Path,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
    dev_source_groups: list[str],
    temperature: float,
) -> dict[str, Any]:
    """Evaluate a checkpoint with prototypes computed from train rows only."""
    sys.path.insert(0, str(REPO / "backend"))
    from codex_pipeline.models.projection_head import ProjectionHead
    from codex_pipeline.models.prototypical import compute_prototypes

    checkpoint = torch.load(projection_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model = ProjectionHead(input_dim=train_features.shape[1], embedding_dim=128)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        train_embeddings = model(train_features)
        dev_embeddings = model(dev_features)
        prototype_labels = train_labels.unique(sorted=True)
        prototypes = compute_prototypes(train_embeddings, train_labels)
        logits = dev_embeddings @ prototypes.T / temperature
        label_to_index = {int(label): index for index, label in enumerate(prototype_labels.tolist())}
        targets = torch.tensor([label_to_index[int(label)] for label in dev_labels.tolist()])
        losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = prototype_labels[logits.argmax(dim=1)]

    correct = predictions.eq(dev_labels)
    per_class: dict[str, dict[str, float | int]] = {}
    for label in dev_labels.unique(sorted=True).tolist():
        mask = dev_labels.eq(label)
        per_class[str(label)] = {
            "support": int(mask.sum()),
            "accuracy": float(correct[mask].float().mean()),
            "logloss": float(losses[mask].mean()),
        }
    per_group: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(dev_source_groups)):
        mask = torch.tensor([item == group for item in dev_source_groups], dtype=torch.bool)
        per_group[group] = {
            "support": int(mask.sum()),
            "accuracy": float(correct[mask].float().mean()),
            "logloss": float(losses[mask].mean()),
        }
    macro_class = sum(item["accuracy"] for item in per_class.values()) / len(per_class)
    macro_group = sum(item["accuracy"] for item in per_group.values()) / len(per_group)
    balanced_logloss = sum(item["logloss"] for item in per_class.values()) / len(per_class)
    micro = float(correct.float().mean())
    groups_are_class_equivalent = (
        len(per_class) == len(per_group) == 8
        and all(len(set(dev_labels[torch.tensor([item == group for item in dev_source_groups], dtype=torch.bool)].tolist())) == 1
                for group in per_group)
    )
    return {
        "macro_class": macro_class,
        "macro_source_group": macro_group,
        "micro": micro,
        "balanced_logloss": balanced_logloss,
        "J": 0.75 * macro_class + 0.25 * macro_group,
        "source_group_metric_redundant": groups_are_class_equivalent,
        "per_class": per_class,
        "per_source_group": per_group,
    }


def decision(metrics: dict[str, Any], control_metrics: dict[str, Any]) -> tuple[str, list[str]]:
    drops = [
        label for label, item in metrics["per_class"].items()
        if float(item["accuracy"]) < float(control_metrics["per_class"].get(label, {}).get("accuracy", 0.0)) - 0.20
    ]
    if metrics["J"] > control_metrics["J"] + 0.01 and not drops:
        return "KEEP", []
    return "DISCARD", drops


def config_for(spec: Spec, checkpoint_dir: Path) -> dict[str, Any]:
    return {
        "paths": {"checkpoint_dir": str(checkpoint_dir)},
        "data": {"split_strategy": "persisted", "val_fraction": 0.15, "num_workers": 0},
        "model": {"embedding_dim": 128},
        "training": {
            "episodes_per_epoch": 100, "num_epochs": spec.epochs, "n_way": 20,
            "k_shot": 3, "q_queries": 5, "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay, "lr_scheduler": "cosine", "warmup_epochs": 0,
            "temperature": spec.temperature, "device": "cuda", "seed": 42,
            "checkpoint_selection": "latest",
        },
        "evaluation": {"num_eval_episodes": 100},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runner_provenance() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "train_py": {"path": str(TRAIN), "sha256": sha256(TRAIN)},
        "python": {"executable": sys.executable, "version": sys.version},
        "torch": {"version": torch.__version__, "cuda_available": cuda_available, "cuda_version": torch.version.cuda},
        "device": {"requested": "cuda", "resolved": "cuda" if cuda_available else "unavailable", "name": torch.cuda.get_device_name(0) if cuda_available else None},
        "subprocess_environment": {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    }


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    return environment


def training_command(spec: Spec, config_path: Path, checkpoint_dir: Path, sealed_cache: Path) -> list[str]:
    command = [str(PYTHON), str(TRAIN), "--config", str(config_path), "--features", str(sealed_cache), "--checkpoint-dir", str(checkpoint_dir), "--init-projection", str(INIT_PROJECTION)]
    if spec.noise_std:
        command.extend(["--noise-std", str(spec.noise_std)])
    if spec.mixup_prob:
        command.extend(["--mixup-prob", str(spec.mixup_prob)])
    return command


def frozen_manifest(sealed_cache: dict[str, Any]) -> dict[str, Any]:
    assert_frozen_specs()
    return {
        "protocol": "s14-dev-only-fixed-sweep.v2",
        "source_cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
        "source_cache_provenance": {"path": str(CACHE_PROVENANCE), "sha256": sha256(CACHE_PROVENANCE)},
        "sealed_cache": sealed_cache,
        "init_projection": {"path": str(INIT_PROJECTION), "sha256": sha256(INIT_PROJECTION)},
        "runner_provenance": runner_provenance(),
        "selection": {"J": "0.75*macro_class + 0.25*macro_source_group", "minimum_j_gain": 0.01, "max_per_class_drop": 0.20},
        "specs": [dataclasses.asdict(spec) for spec in SPECS],
    }


def write_document(results: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    rows = [
        "# S14 development-only fixed sweep (2026-08-12)", "",
        "This v2 evidence is research-only: all 11 runs and every evaluation use a separately sealed physical cache containing only persisted train and dev rows. No locked rows are present in that cache, and no registry/runtime artifacts are modified.", "",
        f"- Source cache SHA-256: `{manifest['source_cache']['sha256']}`",
        f"- Source cache provenance SHA-256: `{manifest['source_cache_provenance']['sha256']}`",
        f"- Sealed train+dev cache SHA-256: `{manifest['sealed_cache']['sha256']}`",
        f"- Initial projection SHA-256: `{manifest['init_projection']['sha256']}`",
        "- Subprocess environment pins `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.",
        "- `source_group_metric_redundant=true`: development data has eight classes and eight source groups with one class per group.",
        "- Fixed protocol: 11 independent runs, same initial projection and seed; each candidate changes one factor from control.",
        "- Selection: `J = .75 macro_class + .25 macro_source_group`; KEEP requires `J > control + .01` and no per-class accuracy drop greater than .20.", "",
        "| Run | Changed factor | Macro class | Macro group | Micro | Balanced logloss | J | Decision |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        metrics = result["metrics"]
        rows.append(
            f"| {result['name']} | {', '.join(result['changed_factors']) or 'control'} | "
            f"{metrics['macro_class']:.4f} | {metrics['macro_source_group']:.4f} | {metrics['micro']:.4f} | "
            f"{metrics['balanced_logloss']:.4f} | {metrics['J']:.4f} | {result['decision']} |"
        )
    rows.extend(["", f"Machine-readable v2 evidence: `{REPORT_DIR.relative_to(REPO)}/results.json`.", ""])
    DOCUMENT.write_text("\n".join(rows), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    for path in (CACHE, CACHE_PROVENANCE, INIT_PROJECTION, TRAIN, PYTHON):
        assert_allowed_input_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    assert_frozen_specs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sealed_cache = seal_cache(CACHE, REPORT_DIR / "sealed" / "train-dev.features.pt")
    manifest = frozen_manifest(sealed_cache)
    manifest_path = REPORT_DIR / "frozen-specs.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("frozen spec manifest differs from existing run; refusing to alter protocol")
    write_json(manifest_path, manifest)
    sealed_path = Path(sealed_cache["path"])
    cache = torch.load(sealed_path, map_location="cpu", weights_only=False)
    train_features, train_labels, dev_features, dev_labels, dev_groups = split_train_dev(cache)
    results_path = REPORT_DIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    if len(results) != args.start:
        raise RuntimeError(f"expected {args.start} prior results, found {len(results)}")
    stop = min(args.stop, len(SPECS))
    if not 0 <= args.start <= stop:
        raise ValueError("invalid run range")
    for index in range(args.start, stop):
        spec = SPECS[index]
        run_dir = REPORT_DIR / "runs" / f"{index:02d}-{spec.name}"
        checkpoint_dir = run_dir / "checkpoints"
        config_path = run_dir / "config.yaml"
        config = config_for(spec, checkpoint_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        command = training_command(spec, config_path, checkpoint_dir, sealed_path)
        environment = subprocess_environment()
        write_json(run_dir / "invocation.json", {"argv": command, "environment": {key: environment[key] for key in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")}})
        completed = subprocess.run(command, cwd=REPO, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15 * 60, check=False)
        (run_dir / "train.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"run {spec.name} failed; see {run_dir / 'train.log'}")
        checkpoint = checkpoint_dir / "latest.pt"
        metrics = evaluate_projection(checkpoint, train_features, train_labels, dev_features, dev_labels, dev_groups, spec.temperature)
        result: dict[str, Any] = {
            "index": index, "name": spec.name, "changed_factors": spec.changed_factors(CONTROL),
            "spec": dataclasses.asdict(spec), "config_sha256": sha256(config_path),
            "checkpoint_sha256": sha256(checkpoint), "metrics": metrics,
        }
        if index == 0:
            result["decision"] = "CONTROL"
            result["per_class_drops_over_0_20"] = []
        else:
            result["decision"], result["per_class_drops_over_0_20"] = decision(metrics, results[0]["metrics"])
        write_json(run_dir / "result.json", result)
        results.append(result)
        write_json(results_path, results)
        write_document(results, manifest)
        print(json.dumps({"completed": index + 1, "total": len(SPECS), "name": spec.name, "J": metrics["J"], "decision": result["decision"]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="first frozen spec index to execute")
    parser.add_argument("--stop", type=int, default=len(SPECS), help="exclusive frozen spec index")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

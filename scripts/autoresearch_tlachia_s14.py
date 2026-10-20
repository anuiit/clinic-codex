#!/usr/bin/env python3
"""Frozen dev-only S/14 sweep for the clean-room Tlachia snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import autoresearch_s14 as base


REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "backend/training_corpus/snapshots/snapshot-tlachia-6d98f8520a3b435b5dd8"
SOURCE_CACHE = SNAPSHOT / "precomputed/features.pt"
SOURCE_PROVENANCE = SOURCE_CACHE.with_suffix(".pt.prov.json")
REPORT_DIR = REPO / "autoresearch/orchestrator-260813-1330/tlachia-s14"
DOCUMENT = REPO / "docs/autoresearch-tlachia-s14-20260813.md"
INIT_PROJECTION = (
    REPO
    / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original"
    / "runtime/weights/projection.pt"
)
EXPECTED_SOURCE_SPLITS = Counter(train=3763, dev=465, locked_test=447)
EXPECTED_SEALED_SPLITS = Counter(train=3763, dev=465)


def validate_sealed_cache(cache: dict[str, Any]) -> None:
    required = {"features", "labels", "dataset_splits", "source_groups", "row_ids"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"sealed cache missing required fields: {sorted(missing)}")
    lengths = {
        len(cache["features"]),
        len(cache["labels"]),
        len(cache["dataset_splits"]),
        len(cache["source_groups"]),
        len(cache["row_ids"]),
    }
    if len(lengths) != 1:
        raise ValueError("sealed cache fields have inconsistent lengths")
    counts = Counter(cache["dataset_splits"])
    if counts != EXPECTED_SEALED_SPLITS or "locked_test" in counts:
        raise ValueError(f"sealed cache split mismatch: {dict(counts)}")
    if len(set(cache["row_ids"])) != len(cache["row_ids"]):
        raise ValueError("sealed cache row IDs are not unique")


def seal_cache(source_cache: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        sealed = torch.load(target, map_location="cpu", weights_only=False)
        validate_sealed_cache(sealed)
        return {"path": str(target), "sha256": base.sha256(target)}
    source = torch.load(source_cache, map_location="cpu", weights_only=False)
    if Counter(source["dataset_splits"]) != EXPECTED_SOURCE_SPLITS:
        raise ValueError("source cache persisted split differs from frozen protocol")
    selected_indices = [
        index
        for index, split in enumerate(source["dataset_splits"])
        if split in {"train", "dev"}
    ]
    selected = torch.tensor(selected_indices, dtype=torch.long)
    sealed = {
        "features": source["features"][selected].clone(),
        "labels": source["labels"][selected].clone(),
        "source_groups": [source["source_groups"][index] for index in selected_indices],
        "dataset_splits": [source["dataset_splits"][index] for index in selected_indices],
        "row_ids": [source["row_ids"][index] for index in selected_indices],
        "class_names": source["class_names"],
        "hidden_dim": source["hidden_dim"],
    }
    validate_sealed_cache(sealed)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sealed, target)
    return {"path": str(target), "sha256": base.sha256(target)}


def split_train_dev(cache: dict[str, Any]):
    validate_sealed_cache(cache)
    splits = list(cache["dataset_splits"])
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    dev_indices = [index for index, split in enumerate(splits) if split == "dev"]
    train_index = torch.tensor(train_indices, dtype=torch.long)
    dev_index = torch.tensor(dev_indices, dtype=torch.long)
    train_labels = cache["labels"][train_index]
    dev_labels = cache["labels"][dev_index]
    absent = sorted(set(dev_labels.tolist()) - set(train_labels.tolist()))
    if absent:
        raise ValueError(f"dev labels missing from train-only prototypes: {absent}")
    return (
        cache["features"][train_index],
        train_labels,
        cache["features"][dev_index],
        dev_labels,
        [cache["source_groups"][index] for index in dev_indices],
    )


def frozen_manifest(sealed_cache: dict[str, Any]) -> dict[str, Any]:
    base.assert_frozen_specs()
    return {
        "protocol": "tlachia-s14-dev-only-fixed-sweep.v1",
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "snapshot_manifest": {
            "path": str(SNAPSHOT / "snapshot_manifest.json"),
            "sha256": base.sha256(SNAPSHOT / "snapshot_manifest.json"),
        },
        "source_cache": {"path": str(SOURCE_CACHE), "sha256": base.sha256(SOURCE_CACHE)},
        "source_cache_provenance": {
            "path": str(SOURCE_PROVENANCE),
            "sha256": base.sha256(SOURCE_PROVENANCE),
        },
        "sealed_cache": sealed_cache,
        "persisted_splits": dict(EXPECTED_SOURCE_SPLITS),
        "init_projection": {
            "path": str(INIT_PROJECTION),
            "sha256": base.sha256(INIT_PROJECTION),
        },
        "selection": {
            "input": "dev only; locked_test physically absent",
            "J": "0.75*macro_class + 0.25*macro_source_group",
            "minimum_j_gain": 0.01,
            "max_per_class_drop": 0.20,
        },
        "specs": [base.dataclasses.asdict(spec) for spec in base.SPECS],
        "runner_provenance": base.runner_provenance(),
        "promotion_eligible": False,
    }


def write_document(results: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Tlachia S/14 dev-only autoresearch (2026-08-13)",
        "",
        "Research-only sweep on the clean-room Tlachia snapshot. The sealed cache contains train+dev only; locked_test and the business-final set are never inputs to training or selection.",
        "",
        f"- Snapshot manifest SHA-256: `{manifest['snapshot_manifest']['sha256']}`",
        f"- Sealed cache SHA-256: `{manifest['sealed_cache']['sha256']}`",
        f"- Historical initializer SHA-256: `{manifest['init_projection']['sha256']}`",
        "- Five classes have one train image; 86 classes have fewer than the 8 distinct rows required by k=3/q=5 and remain outside episodic head updates (but keep train-only prototypes).",
        "- Promotion eligible: false.",
        "",
        "| Run | Factor | Macro class | Macro group | Micro | J | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['name']} | {', '.join(result['changed_factors']) or 'control'} | "
            f"{metrics['macro_class']:.4f} | {metrics['macro_source_group']:.4f} | "
            f"{metrics['micro']:.4f} | {metrics['J']:.4f} | {result['decision']} |"
        )
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_base() -> None:
    base.REPORT_DIR = REPORT_DIR
    base.DOCUMENT = DOCUMENT
    base.CACHE = SOURCE_CACHE
    base.CACHE_PROVENANCE = SOURCE_PROVENANCE
    base.INIT_PROJECTION = INIT_PROJECTION
    base.validate_sealed_cache = validate_sealed_cache
    base.seal_cache = seal_cache
    base.split_train_dev = split_train_dev
    base.frozen_manifest = frozen_manifest
    base.write_document = write_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=len(base.SPECS))
    args = parser.parse_args()
    configure_base()
    base.run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train-only, preregistered S/14 classifier research.

The runner creates a physical 9,128-row feature cache, then performs
row-stratified cross-validation.  It is research-only: it never writes a model
package, registry entry, or application runtime.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/training_data/precomputed/features.pt"
SOURCE_PROVENANCE = SOURCE.with_suffix(".pt.prov.json")
PROJECTION = REPO / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime/weights/projection.pt"
REPORT_DIR = REPO / "reports/autoresearch-s14-v3-20260812"
SEALED = REPORT_DIR / "sealed/train-only.features.pt"
MANIFEST = REPORT_DIR / "frozen-specs.json"
RESULTS = REPORT_DIR / "results.json"
DOC = REPO / "docs/autoresearch-s14-v3-20260812.md"
EXPECTED_ROWS = 9128
EXPECTED_CLASSES = 286
SCREEN_SEED = 17
OUTER_SEEDS = (17, 42, 73)
SCREEN_FOLDS = 3
OUTER_FOLDS = 5
TOP_N = 8
SCREEN_GAIN = 0.005
OUTER_GAIN = 0.01
BUCKET_TOLERANCE = 0.01
OUTER_BUCKET_TOLERANCE = 0.02
TEMPERATURE = 0.1


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    method: str
    params: tuple[tuple[str, Any], ...] = ()
    space: str = "b1"
    runtime_compatible: bool = True

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)


def _p(**kwargs: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(kwargs.items()))


# This tuple is the preregistration. Do not add candidates after inspecting scores.
SPECS: tuple[Spec, ...] = (
    Spec("control-b1-ncm", "ncm"),
    Spec("raw-ncm", "ncm", space="raw"),
    Spec("simpleshot-b1", "simpleshot"),
    Spec("simpleshot-raw", "simpleshot", space="raw"),
    Spec("power-b1-p025", "power", _p(power=0.25)),
    Spec("power-b1-p050", "power", _p(power=0.50)),
    Spec("power-b1-p075", "power", _p(power=0.75)),
    Spec("power-raw-p025", "power", _p(power=0.25), space="raw"),
    Spec("power-raw-p050", "power", _p(power=0.50), space="raw"),
    Spec("power-raw-p075", "power", _p(power=0.75), space="raw"),
    Spec("zscore-b1", "zscore"),
    Spec("zscore-raw", "zscore", space="raw"),
    Spec("pca-whiten-b1-e001", "pca_whiten", _p(eps=0.01)),
    Spec("pca-whiten-b1-e010", "pca_whiten", _p(eps=0.10)),
    Spec("pca-whiten-raw-e001", "pca_whiten", _p(eps=0.01), space="raw"),
    Spec("pca-diag-b1-e001", "zscore", _p(eps=0.01)),
    Spec("pca-diag-b1-e010", "zscore", _p(eps=0.10)),
    Spec("coord-median-b1", "robust", _p(estimator="median")),
    Spec("geomedian-b1", "robust", _p(estimator="geomedian")),
    Spec("trimmed-b1-p010", "robust", _p(estimator="trimmed", trim=0.10)),
    Spec("trimmed-b1-p020", "robust", _p(estimator="trimmed", trim=0.20)),
    Spec("tukey-biweight-b1-c4685", "robust", _p(estimator="biweight", c=4.685)),
    Spec("centroid-shrink-b1-l025", "shrink", _p(lam=0.25)),
    Spec("centroid-shrink-b1-l1", "shrink", _p(lam=1.0)),
    Spec("centroid-shrink-b1-l4", "shrink", _p(lam=4.0)),
    Spec("centroid-shrink-b1-l8", "shrink", _p(lam=8.0)),
    Spec("ridge-cos-b1-l00001", "ridge", _p(lam=1e-4)),
    Spec("ridge-cos-b1-l0001", "ridge", _p(lam=1e-3)),
    Spec("ridge-cos-b1-l001", "ridge", _p(lam=1e-2)),
    Spec("ridge-cos-b1-l01", "ridge", _p(lam=0.1)),
    Spec("ridge-cos-b1-l1", "ridge", _p(lam=1.0)),
    Spec("ridge-cos-b1-l10", "ridge", _p(lam=10.0)),
    Spec("ridge-cos-b1-l100", "ridge", _p(lam=100.0)),
    Spec("knn-b1-k1", "knn", _p(k=1)),
    Spec("knn-b1-k3", "knn", _p(k=3)),
    Spec("knn-b1-k5", "knn", _p(k=5)),
    Spec("knn-b1-k9", "knn", _p(k=9)),
    Spec("knn-b1-k15", "knn", _p(k=15)),
    Spec("knn-k15-beta2", "knn", _p(k=15, beta=2.0)),
    Spec("knn-k15-beta5", "knn", _p(k=15, beta=5.0)),
    Spec("knn-k15-beta10", "knn", _p(k=15, beta=10.0)),
    Spec("mahal-global-diag-s01", "mahal_diag", _p(shrink=0.1)),
    Spec("mahal-global-diag-s05", "mahal_diag", _p(shrink=0.5)),
    Spec("fecam-diag-s05", "fecam_diag", _p(shrink=0.5)),
    Spec("relu-ridge-m256", "relu_ridge", _p(width=256, lam=1.0), runtime_compatible=False),
    Spec("relu-ridge-m512", "relu_ridge", _p(width=512, lam=1.0), runtime_compatible=False),
    Spec("relu-ridge-m1024", "relu_ridge", _p(width=1024, lam=1.0), runtime_compatible=False),
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
    if len(specs) < 35:
        raise ValueError("at least 35 preregistered specs are required")
    if specs[0].name != "control-b1-ncm" or specs[0].method != "ncm":
        raise ValueError("the first spec must be the B1 NCM control")
    if len({item.name for item in specs}) != len(specs):
        raise ValueError("spec names must be unique")
    valid = {"ncm", "simpleshot", "power", "zscore", "pca_whiten", "robust", "shrink", "ridge", "knn", "mahal_diag", "fecam_diag", "relu_ridge"}
    unknown = {item.method for item in specs} - valid
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")


def seal_train_cache(source_path: Path = SOURCE, target: Path = SEALED) -> dict[str, Any]:
    """Physically copy only rows marked train; omit paths and source metadata."""
    if target.exists():
        validate_train_cache(torch.load(target, map_location="cpu", weights_only=False))
        return {"path": str(target), "sha256": sha256(target)}
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    required = {"features", "labels", "dataset_splits"}
    if required - set(source):
        raise ValueError("source cache lacks required tensors or split marks")
    indices = torch.tensor([i for i, mark in enumerate(source["dataset_splits"]) if mark == "train"], dtype=torch.long)
    sealed = {
        "features": source["features"][indices].clone(),
        "labels": source["labels"][indices].clone(),
        "source_row_indices": indices.clone(),
        "class_names": source.get("class_names"),
        "hidden_dim": int(source["features"].shape[1]),
        "partition": "train",
    }
    validate_train_cache(sealed)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sealed, target)
    return {"path": str(target), "sha256": sha256(target)}


def validate_train_cache(cache: dict[str, Any]) -> None:
    allowed = {"features", "labels", "source_row_indices", "class_names", "hidden_dim", "partition"}
    if set(cache) - allowed:
        raise ValueError(f"sealed cache has unexpected keys: {sorted(set(cache) - allowed)}")
    if cache.get("partition") != "train":
        raise ValueError("sealed cache is not train-only")
    x, y, rows = cache["features"], cache["labels"], cache["source_row_indices"]
    if not all(isinstance(item, torch.Tensor) for item in (x, y, rows)):
        raise ValueError("sealed cache fields must be tensors")
    if len(x) != EXPECTED_ROWS or len(y) != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
        raise ValueError("sealed cache row count differs from contract")
    if y.unique().numel() != EXPECTED_CLASSES or int(y.min()) != 0 or int(y.max()) != EXPECTED_CLASSES - 1:
        raise ValueError("sealed cache class coverage differs from contract")
    if len(rows.unique()) != EXPECTED_ROWS:
        raise ValueError("sealed row indices are not unique")


def stratified_folds(labels: torch.Tensor, folds: int, seed: int) -> list[torch.Tensor]:
    """Deterministic exhaustive row-level folds, retaining every class in train."""
    buckets: list[list[int]] = [[] for _ in range(folds)]
    generator = torch.Generator().manual_seed(seed)
    for label in labels.unique(sorted=True).tolist():
        rows = torch.where(labels.eq(label))[0]
        rows = rows[torch.randperm(len(rows), generator=generator)]
        for offset, row in enumerate(rows.tolist()):
            buckets[offset % folds].append(row)
    result = [torch.tensor(sorted(bucket), dtype=torch.long) for bucket in buckets]
    assert_fold_contract(labels, result)
    return result


def assert_fold_contract(labels: torch.Tensor, folds: list[torch.Tensor]) -> None:
    flattened = torch.cat(folds)
    if len(flattened) != len(labels) or len(flattened.unique()) != len(labels):
        raise ValueError("fold holdouts are not exhaustive and disjoint")
    if set(flattened.tolist()) != set(range(len(labels))):
        raise ValueError("folds do not cover every row index")
    all_rows = torch.arange(len(labels))
    all_classes = set(labels.unique().tolist())
    for holdout in folds:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[holdout] = False
        train = all_rows[mask]
        if set(labels[train].unique().tolist()) != all_classes:
            raise ValueError("a fold removes a class from its training side")
        if set(train.tolist()) & set(holdout.tolist()):
            raise ValueError("train and holdout intersect")


def load_projection() -> dict[str, torch.Tensor]:
    state = torch.load(PROJECTION, map_location="cpu", weights_only=True)
    return {key: value.float() for key, value in state.items()}


def apply_projection(features: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = F.linear(features, state["net.0.weight"], state["net.0.bias"])
    hidden = F.gelu(hidden)
    return F.normalize(F.linear(hidden, state["net.3.weight"], state["net.3.bias"]), dim=1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x), dim=1, eps=1e-8)


def class_means(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros(EXPECTED_CLASSES, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, y, x)
    counts = torch.bincount(y, minlength=EXPECTED_CLASSES).to(x.dtype).unsqueeze(1)
    return sums / counts.clamp_min(1)


def robust_prototypes(x: torch.Tensor, y: torch.Tensor, estimator: str, **kw: Any) -> torch.Tensor:
    values = []
    for label in range(EXPECTED_CLASSES):
        group = x[y.eq(label)]
        if estimator == "median":
            center = group.median(dim=0).values
        elif estimator == "trimmed":
            trim = int(math.floor(len(group) * float(kw["trim"])))
            ordered = group.sort(dim=0).values
            center = ordered[trim:len(group)-trim].mean(dim=0) if trim and len(group) > 2 * trim else group.mean(dim=0)
        elif estimator == "geomedian":
            center = group.mean(dim=0)
            for _ in range(20):
                distance = torch.linalg.vector_norm(group - center, dim=1).clamp_min(1e-6)
                updated = (group / distance[:, None]).sum(0) / (1.0 / distance).sum()
                if torch.linalg.vector_norm(updated - center) < 1e-5:
                    break
                center = updated
        elif estimator == "biweight":
            median = group.median(dim=0).values
            mad = (group - median).abs().median(dim=0).values.clamp_min(1e-6)
            u = (group - median) / (float(kw["c"]) * mad)
            weights = ((1 - u.square()).clamp_min(0).square())
            center = (weights * group).sum(0) / weights.sum(0).clamp_min(1e-6)
        else:
            raise ValueError(estimator)
        values.append(center)
    return normalize(torch.stack(values))


def ridge_scores(train_x: torch.Tensor, train_y: torch.Tensor, query_x: torch.Tensor, lam: float) -> torch.Tensor:
    dim = train_x.shape[1]
    target = F.one_hot(train_y, EXPECTED_CLASSES).to(train_x.dtype)
    gram = train_x.T @ train_x
    weights = torch.linalg.solve(gram + lam * torch.eye(dim, device=train_x.device), train_x.T @ target)
    return normalize(query_x) @ normalize(weights.T).T


def random_matrix(input_dim: int, width: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(310_000 + width)
    matrix = torch.randn(input_dim, width, generator=generator) / math.sqrt(input_dim)
    return matrix.to(device)


def matrix_hash(input_dim: int, width: int) -> str:
    return hashlib.sha256(random_matrix(input_dim, width, torch.device("cpu")).numpy().tobytes()).hexdigest()


def scores_for(spec: Spec, train_x: torch.Tensor, train_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
    kw = spec.kwargs
    if spec.method == "simpleshot":
        center = train_x.mean(0, keepdim=True)
        train_x, query_x = normalize(train_x - center), normalize(query_x - center)
    elif spec.method == "power":
        power = float(kw["power"])
        train_x = torch.sign(train_x) * train_x.abs().pow(power)
        query_x = torch.sign(query_x) * query_x.abs().pow(power)
        train_x, query_x = normalize(train_x), normalize(query_x)
    elif spec.method == "zscore":
        center = train_x.mean(0, keepdim=True)
        scale = train_x.std(0, keepdim=True).clamp_min(float(kw.get("eps", 1e-4)))
        train_x, query_x = normalize((train_x-center)/scale), normalize((query_x-center)/scale)
    elif spec.method == "pca_whiten":
        center = train_x.mean(0, keepdim=True)
        centered = train_x - center
        covariance = centered.T @ centered / max(1, len(train_x)-1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        transform = eigenvectors @ torch.diag((eigenvalues + float(kw["eps"])).rsqrt()) @ eigenvectors.T
        train_x, query_x = normalize(centered @ transform), normalize((query_x-center) @ transform)

    if spec.method == "ridge":
        return ridge_scores(train_x, train_y, query_x, float(kw["lam"]))
    if spec.method == "relu_ridge":
        width = int(kw["width"])
        matrix = random_matrix(train_x.shape[1], width, train_x.device)
        return ridge_scores(F.relu(train_x @ matrix), train_y, F.relu(query_x @ matrix), float(kw["lam"]))
    if spec.method == "knn":
        k = int(kw["k"])
        similarities = normalize(query_x) @ normalize(train_x).T
        values, indices = similarities.topk(k, dim=1)
        neighbour_labels = train_y[indices]
        weights = torch.ones_like(values) if "beta" not in kw else torch.exp(float(kw["beta"]) * (values - values[:, :1]))
        scores = torch.zeros(len(query_x), EXPECTED_CLASSES, device=query_x.device)
        scores.scatter_add_(1, neighbour_labels, weights)
        return scores / weights.sum(1, keepdim=True).clamp_min(1e-6)
    if spec.method == "mahal_diag":
        means = class_means(train_x, train_y)
        variance = (train_x - means[train_y]).square().mean(0)
        shrink = float(kw["shrink"])
        variance = (1-shrink)*variance + shrink*variance.mean()
        return -((query_x[:, None, :] - means[None, :, :]).square() / variance.clamp_min(1e-5)).mean(2)
    if spec.method == "fecam_diag":
        means = class_means(train_x, train_y)
        global_variance = (train_x - means[train_y]).square().mean(0)
        variances = []
        shrink = float(kw["shrink"])
        for label in range(EXPECTED_CLASSES):
            group = train_x[train_y.eq(label)]
            local = group.var(0, unbiased=False) if len(group) > 1 else global_variance
            variances.append((1-shrink)*local + shrink*global_variance)
        variance = torch.stack(variances).clamp_min(1e-5)
        return -((query_x[:, None, :] - means[None, :, :]).square() / variance[None, :, :]).mean(2)
    if spec.method == "robust":
        prototypes = robust_prototypes(train_x, train_y, **kw)
    else:
        prototypes = class_means(train_x, train_y)
        if spec.method == "shrink":
            counts = torch.bincount(train_y, minlength=EXPECTED_CLASSES).to(train_x.dtype).unsqueeze(1)
            lam = float(kw["lam"])
            prototypes = (counts*prototypes + lam*train_x.mean(0, keepdim=True))/(counts+lam)
        prototypes = normalize(prototypes)
    return normalize(query_x) @ prototypes.T


def metric_summary(records: list[dict[str, Any]], total_support: torch.Tensor) -> dict[str, Any]:
    correct = torch.zeros(EXPECTED_CLASSES, dtype=torch.float64)
    top3 = torch.zeros_like(correct)
    loss = torch.zeros_like(correct)
    support = torch.zeros(EXPECTED_CLASSES, dtype=torch.long)
    for item in records:
        label = int(item["label"])
        support[label] += 1
        correct[label] += int(item["prediction"] == label)
        top3[label] += int(label in item["top3"])
        loss[label] += float(item["loss"])
    if bool((support == 0).any()):
        raise ValueError("metrics require all classes")
    accuracy = correct/support
    top3_accuracy = top3/support
    class_loss = loss/support
    low = total_support <= 7
    high = ~low
    return {
        "macro_top1": float(accuracy.mean()),
        "macro_top3": float(top3_accuracy.mean()),
        "micro_top1": float(correct.sum()/support.sum()),
        "balanced_logloss": float(class_loss.mean()),
        "low_support_macro_top1": float(accuracy[low].mean()),
        "high_support_macro_top1": float(accuracy[high].mean()),
        "per_class_top1": [float(value) for value in accuracy],
        "per_class_support": [int(value) for value in support],
    }


def evaluate_spec(spec: Spec, features: torch.Tensor, labels: torch.Tensor, folds: list[torch.Tensor], device: torch.device) -> dict[str, Any]:
    projection = load_projection()
    raw = features.float()
    base = apply_projection(raw, projection) if spec.space == "b1" else normalize(raw)
    base = base.to(device)
    labels_device = labels.to(device)
    all_rows = torch.arange(len(labels))
    records: list[dict[str, Any]] = []
    fold_summaries = []
    started = time.time()
    for fold_index, holdout in enumerate(folds):
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[holdout] = False
        train = all_rows[mask]
        score = scores_for(spec, base[train.to(device)], labels_device[train.to(device)], base[holdout.to(device)])
        probabilities = F.softmax(score/TEMPERATURE, dim=1)
        top3 = score.topk(3, dim=1).indices.cpu()
        predictions = top3[:, 0]
        query_labels = labels[holdout]
        losses = -probabilities.cpu()[torch.arange(len(holdout)), query_labels].clamp_min(1e-12).log()
        fold_records = [
            {"row_index": int(row), "label": int(label), "prediction": int(pred), "top3": [int(v) for v in tops], "loss": float(loss)}
            for row, label, pred, tops, loss in zip(holdout, query_labels, predictions, top3, losses)
        ]
        records.extend(fold_records)
        fold_summaries.append({
            "fold": fold_index,
            "rows": len(fold_records),
            "micro_top1": sum(int(item["prediction"] == item["label"]) for item in fold_records) / len(fold_records),
        })
    records.sort(key=lambda item: item["row_index"])
    metrics = metric_summary(records, torch.bincount(labels, minlength=EXPECTED_CLASSES))
    return {"metrics": metrics, "folds": fold_summaries, "predictions": records, "duration_seconds": time.time()-started}


def paired_class_bootstrap(candidate: list[float], control: list[float], seed: int = 20260812, samples: int = 10000) -> dict[str, float]:
    delta = torch.tensor(candidate, dtype=torch.float64) - torch.tensor(control, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(delta), (samples, len(delta)), generator=generator)
    distribution = delta[indices].mean(1).sort().values
    return {"mean": float(delta.mean()), "lower_95": float(distribution[int(0.025*samples)]), "upper_95": float(distribution[int(0.975*samples)-1])}


def screening_pass(metrics: dict[str, Any], control: dict[str, Any]) -> bool:
    return (
        metrics["macro_top1"] > control["macro_top1"] + SCREEN_GAIN
        and metrics["low_support_macro_top1"] >= control["low_support_macro_top1"] - BUCKET_TOLERANCE
        and metrics["high_support_macro_top1"] >= control["high_support_macro_top1"] - BUCKET_TOLERANCE
    )


def outer_pass(metrics: dict[str, Any], control: dict[str, Any], bootstrap: dict[str, float]) -> bool:
    return (
        metrics["macro_top1"] >= control["macro_top1"] + OUTER_GAIN
        and bootstrap["lower_95"] > 0
        and metrics["low_support_macro_top1"] >= control["low_support_macro_top1"] - OUTER_BUCKET_TOLERANCE
        and metrics["high_support_macro_top1"] >= control["high_support_macro_top1"] - OUTER_BUCKET_TOLERANCE
    )


def frozen_manifest(sealed: dict[str, Any]) -> dict[str, Any]:
    assert_specs()
    random_hashes = {str(spec.kwargs["width"]): matrix_hash(128, int(spec.kwargs["width"])) for spec in SPECS if spec.method == "relu_ridge"}
    value = {
        "protocol": "s14-train-only-row-cv.v3",
        "created_before_results": True,
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "source_provenance": {"path": str(SOURCE_PROVENANCE), "sha256": sha256(SOURCE_PROVENANCE)},
        "sealed": sealed,
        "projection": {"path": str(PROJECTION), "sha256": sha256(PROJECTION)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "cv": {"kind": "in-page row-stratified", "screen": {"folds": 3, "seed": 17}, "outer": {"folds": 5, "seeds": list(OUTER_SEEDS)}},
        "selection": {"screen_gain": SCREEN_GAIN, "screen_bucket_tolerance": BUCKET_TOLERANCE, "top_n_regardless": TOP_N, "outer_gain": OUTER_GAIN, "bootstrap_lower_strictly_positive": True, "outer_bucket_tolerance": OUTER_BUCKET_TOLERANCE},
        "random_feature_matrix_sha256": random_hashes,
        "specs": [json.loads(json.dumps(dataclasses.asdict(spec))) for spec in SPECS],
    }
    value["specification_hash"] = stable_hash({"cv": value["cv"], "selection": value["selection"], "specs": value["specs"]})
    return value


def prepare() -> dict[str, Any]:
    for path in (SOURCE, SOURCE_PROVENANCE, PROJECTION):
        if not path.is_file():
            raise FileNotFoundError(path)
    sealed = seal_train_cache()
    manifest = frozen_manifest(sealed)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("frozen manifest drift; refusing to mutate the experiment")
    write_json(MANIFEST, manifest)
    if not RESULTS.exists():
        write_json(RESULTS, {"protocol": manifest["protocol"], "specification_hash": manifest["specification_hash"], "screening": [], "outer": [], "confirmation": []})
    return manifest


def load_data() -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(SEALED, map_location="cpu", weights_only=False)
    validate_train_cache(cache)
    return cache["features"], cache["labels"].long()


def run_screen(start: int, stop: int) -> None:
    prepare()
    features, labels = load_data()
    folds = stratified_folds(labels, SCREEN_FOLDS, SCREEN_SEED)
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    completed = len(results["screening"])
    if start != completed:
        raise RuntimeError(f"screening resume expected start={completed}, got {start}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index in range(start, min(stop, len(SPECS))):
        spec = SPECS[index]
        try:
            outcome = evaluate_spec(spec, features, labels, folds, device)
            outcome.update({"index": index, "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
        except Exception as exc:
            outcome = {"index": index, "name": spec.name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "spec": dataclasses.asdict(spec)}
        write_json(REPORT_DIR / f"screening/{index:02d}-{spec.name}.json", outcome)
        results["screening"].append(outcome)
        write_json(RESULTS, results)
        metric = outcome.get("metrics", {}).get("macro_top1")
        print(json.dumps({"phase": "screen", "completed": index+1, "total": len(SPECS), "name": spec.name, "macro_top1": metric, "status": outcome["status"]}), flush=True)


def selected_screening(results: dict[str, Any]) -> list[dict[str, Any]]:
    valid = [item for item in results["screening"] if item["status"] == "COMPLETED"]
    if len(valid) < 30 or valid[0]["name"] != SPECS[0].name:
        raise RuntimeError("screening is incomplete or control is unavailable")
    control = valid[0]["metrics"]
    for item in valid:
        item["screening_pass"] = screening_pass(item["metrics"], control) if item["index"] else True
    challengers = sorted(valid[1:], key=lambda item: (-item["metrics"]["macro_top1"], item["index"]))[:TOP_N]
    return [valid[0], *challengers]


def run_outer() -> None:
    prepare()
    features, labels = load_data()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if results["outer"]:
        raise RuntimeError("outer phase already exists")
    selected = selected_screening(results)
    folds = stratified_folds(labels, OUTER_FOLDS, SCREEN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for order, item in enumerate(selected):
        spec = SPECS[item["index"]]
        outcome = evaluate_spec(spec, features, labels, folds, device)
        outcome.update({"index": item["index"], "name": spec.name, "status": "COMPLETED", "screening_pass": item["screening_pass"], "spec": dataclasses.asdict(spec)})
        write_json(REPORT_DIR / f"outer/{order:02d}-{spec.name}.json", outcome)
        results["outer"].append(outcome)
        write_json(RESULTS, results)
        print(json.dumps({"phase": "outer", "completed": order+1, "total": len(selected), "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    control = results["outer"][0]
    for item in results["outer"]:
        boot = paired_class_bootstrap(item["metrics"]["per_class_top1"], control["metrics"]["per_class_top1"])
        item["class_paired_bootstrap"] = boot
        item["outer_pass"] = item["index"] != 0 and outer_pass(item["metrics"], control["metrics"], boot)
    write_json(RESULTS, results)


def aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("macro_top1", "macro_top3", "micro_top1", "balanced_logloss", "low_support_macro_top1", "high_support_macro_top1")
    return {key: sum(item["metrics"][key] for item in seed_results)/len(seed_results) for key in keys}


def run_confirmation() -> None:
    prepare()
    features, labels = load_data()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    passing = [item for item in results["outer"] if item.get("outer_pass")]
    if not passing:
        results["verdict"] = "PLATEAU_NO_RESEARCH_CHALLENGER"
        write_json(RESULTS, results)
        write_document(results)
        print(json.dumps({"phase": "confirmation", "verdict": results["verdict"]}), flush=True)
        return
    top2 = sorted(passing, key=lambda item: -item["metrics"]["macro_top1"])[:2]
    candidates = [SPECS[0], *(SPECS[item["index"]] for item in top2)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in OUTER_SEEDS:
        folds = stratified_folds(labels, OUTER_FOLDS, seed)
        for spec in candidates:
            existing = [item for item in results["confirmation"] if item["seed"] == seed and item["name"] == spec.name]
            if existing:
                continue
            outcome = evaluate_spec(spec, features, labels, folds, device)
            outcome.update({"seed": seed, "index": SPECS.index(spec), "name": spec.name, "status": "COMPLETED", "spec": dataclasses.asdict(spec)})
            write_json(REPORT_DIR / f"confirmation/seed-{seed}-{spec.name}.json", outcome)
            results["confirmation"].append(outcome)
            write_json(RESULTS, results)
            print(json.dumps({"phase": "confirmation", "seed": seed, "name": spec.name, "macro_top1": outcome["metrics"]["macro_top1"]}), flush=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results["confirmation"]:
        grouped[item["name"]].append(item)
    control_metrics = aggregate_seed_results(grouped[SPECS[0].name])
    summaries = []
    for spec in candidates[1:]:
        aggregate = aggregate_seed_results(grouped[spec.name])
        summaries.append({"name": spec.name, "metrics": aggregate, "gain": aggregate["macro_top1"]-control_metrics["macro_top1"]})
    results["confirmation_summary"] = summaries
    qualified = [item for item in summaries if item["gain"] >= OUTER_GAIN and item["metrics"]["low_support_macro_top1"] >= control_metrics["low_support_macro_top1"]-OUTER_BUCKET_TOLERANCE and item["metrics"]["high_support_macro_top1"] >= control_metrics["high_support_macro_top1"]-OUTER_BUCKET_TOLERANCE]
    results["verdict"] = "RESEARCH_CHALLENGER" if qualified else "PLATEAU_NO_RESEARCH_CHALLENGER"
    results["research_challenger"] = max(qualified, key=lambda item: item["metrics"]["macro_top1"], default=None)
    write_json(RESULTS, results)
    write_document(results)


def write_document(results: dict[str, Any]) -> None:
    completed = [item for item in results["screening"] if item["status"] == "COMPLETED"]
    rows = [
        "# S/14 autoresearch v3 — train-only row CV (2026-08-12)", "",
        "This is a research-only, in-page row-stratified experiment. It does not estimate source-page generalization and it does not promote or package a model.", "",
        f"- Frozen specifications: {len(SPECS)}; completed screening variants: {len(completed)}.",
        f"- Physical sealed cache: {EXPECTED_ROWS} train rows, {EXPECTED_CLASSES} classes; no image paths or non-train rows.",
        "- Screening: deterministic stratified 3-fold CV, seed 17. The top eight advance regardless of the preregistered screening gate.",
        "- Confirmation: deterministic stratified 5-fold CV; paired bootstrap resamples class deltas.",
        "- Final multi-seed confirmation (17/42/73) runs only when at least one candidate clears the outer gate.",
        f"- Verdict: `{results.get('verdict', 'IN_PROGRESS')}`.", "",
        "| # | Method | Macro top-1 | Low support | High support | Screening gate | Status |", "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    if completed:
        control = completed[0]["metrics"]
        for item in completed:
            metric = item["metrics"]
            passed = "control" if item["index"] == 0 else str(screening_pass(metric, control)).lower()
            rows.append(f"| {item['index']} | {item['name']} | {metric['macro_top1']:.4f} | {metric['low_support_macro_top1']:.4f} | {metric['high_support_macro_top1']:.4f} | {passed} | {item['status']} |")
    rows += ["", "Conceptual method families: nearest-class means and SimpleShot centering; power normalization; robust location estimators; shrinkage; ridge regression; cosine nearest neighbours; diagonal covariance classifiers; deterministic random-feature ridge. Implementations are original to this harness.", "", f"Machine-readable evidence: `{RESULTS.relative_to(REPO)}`.", ""]
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
    random.seed(0)
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

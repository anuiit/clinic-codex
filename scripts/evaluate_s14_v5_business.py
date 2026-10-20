#!/usr/bin/env python3
"""One-shot terminal business evaluation for the true-B1 S/14 challenger.

`preflight` never accepts or opens business inputs. `evaluate` is deliberately
separate, requires caller-supplied input hashes, validates the immutable
business bundle with the existing validator, and writes its report once.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import benchmark_model_replacement as benchmark  # noqa: E402
from scripts import evaluate_business_test_set as business_eval  # noqa: E402


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3_PATH = REPO / "scripts/autoresearch_s14_v3.py"
V5_PATH = REPO / "scripts/autoresearch_s14_v5.py"
v3 = import_file("autoresearch_s14_v3_for_terminal", V3_PATH)
v5 = import_file("autoresearch_s14_v5_for_terminal", V5_PATH)

SCHEMA_VERSION = "s14-v5-terminal-business-evaluation.v1"
SPEC_SCHEMA_VERSION = "s14-v5-terminal-business-spec.v1"
B1_PACKAGE = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1"
B1_PROJECTION = B1_PACKAGE / "runtime/weights/projection.pt"
EXPECTED_PROJECTION_SHA = "52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03"
SEALED_TRAIN = REPO / "reports/autoresearch-s14-v3-20260812/sealed/train-only.features.pt"
EXPECTED_SEALED_SHA = "3ffbf3e48c9c192eec4b370b6a626e122d59a068f165653fa29993380a6a527a"
SOURCE_TRAIN = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/training_data/precomputed/features.pt"
EXPECTED_SOURCE_SHA = "1e1ee289b2a091811b26724edd1472b62b881356ecb50ce05d0cbefe25c067d3"
SOURCE_PROVENANCE = SOURCE_TRAIN.with_suffix(".pt.prov.json")
EXPECTED_SOURCE_PROVENANCE_SHA = "bd91d10f144d20dd9cca6885ac55bb52e4f59ec3e84bb4c084edeb01c4eec2d8"
V5_RESULTS = REPO / "reports/autoresearch-s14-v5-true-b1-20260812/results.json"
EXPECTED_V5_RESULTS_SHA = "ff2dbb309ab79662f93cfab3c38a64282061a3d3da161326e17df09c022045f5"
SPEC_DIR = REPO / "reports/evaluate-s14-v5-business-spec-20260812"
FROZEN_SPEC = SPEC_DIR / "frozen-spec.json"
EXPECTED_FROZEN_SPEC_FILE_SHA = "ef3d5e0e8bc0d46cafdf111fcdbcbbd8ea6cdf30dc7d04550e4832e55c191362"
EXPECTED_SPECIFICATION_SHA = "51725972aec0f8fe987d85c005624394c098d25bfbf9212897cbedd44e2a2dc1"
TERMINAL_DIR = REPO / "reports/evaluate-s14-v5-business-terminal-20260812"
TERMINAL_LEDGER = TERMINAL_DIR / "terminal-ledger.json"
TERMINAL_OUTPUT = TERMINAL_DIR / "result.json"
METHOD_NAMES = ("packaged_b1_artifact", "b1_ncm_recomputed", "cosine_knn1", "margin_fallback_002")
MARGIN = 0.02
EXPECTED_ROWS = 9128
EXPECTED_CLASSES = 286
BOOTSTRAP_DRAWS = 10000
EVALUATION_DEVICE = "cpu"
EVALUATION_BATCH_SIZE = 4096


def sha256(path: Path) -> str:
    return business_eval.sha256_file(path)


def canonical_sha(value: Any) -> str:
    return business_eval.canonical_json_sha(value)


def write_json_once(path: Path, value: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"terminal output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_sha_literal(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase SHA-256")


def terminal_identity(*, business_content_sha256: str, business_manifest_sha256: str, business_features_sha256: str, business_provenance_sha256: str, specification_sha256: str) -> dict[str, Any]:
    for label, value in (
        ("business content SHA", business_content_sha256),
        ("business manifest SHA", business_manifest_sha256),
        ("business features SHA", business_features_sha256),
        ("business provenance SHA", business_provenance_sha256),
        ("specification SHA", specification_sha256),
    ):
        validate_sha_literal(value, label)
    value = {
        "business_content_sha256": business_content_sha256,
        "business_manifest_sha256": business_manifest_sha256,
        "business_features_sha256": business_features_sha256,
        "business_provenance_sha256": business_provenance_sha256,
        "specification_sha256": specification_sha256,
    }
    return {**value, "terminal_key": canonical_sha(value)}


def _reserve_one_shot_at(identity: dict[str, Any], *, ledger_path: Path, output_path: Path) -> dict[str, Any]:
    """Testable primitive; production always passes the two fixed paths."""
    ledger_path, output_path = ledger_path.resolve(), output_path.resolve()
    if ledger_path.exists():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"global terminal already reserved: state={existing.get('state')} key={existing.get('terminal_key')}")
    if output_path.exists():
        raise RuntimeError(f"fixed terminal output exists without an available reservation: {output_path}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": "s14-v5-terminal-ledger.v1", "state": "reserved", "reserved_at": utc_now(), "fixed_output": str(output_path), **identity}
    with ledger_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def reserve_one_shot(identity: dict[str, Any]) -> dict[str, Any]:
    """Atomically reserve the one global terminal at its non-overridable paths."""
    return _reserve_one_shot_at(identity, ledger_path=TERMINAL_LEDGER, output_path=TERMINAL_OUTPUT)


def transition_ledger(*, ledger_path: Path, terminal_key: str, expected_state: str, new_state: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if new_state not in {"started", "completed"}:
        raise ValueError("invalid terminal ledger transition")
    ledger_path = ledger_path.resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("terminal_key") != terminal_key or ledger.get("state") != expected_state:
        raise RuntimeError("terminal ledger key/state mismatch; no automatic retry allowed")
    ledger["state"] = new_state
    ledger[f"{new_state}_at"] = utc_now()
    if details:
        ledger.update(details)
    temporary = ledger_path.with_suffix(".json.transition")
    if temporary.exists():
        raise RuntimeError("stale terminal transition exists; human audit required")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, ledger_path)
    return ledger


def assert_expected_file(path: Path, expected_path: Path, expected_sha: str, label: str) -> dict[str, str]:
    resolved = path.resolve()
    if resolved != expected_path.resolve():
        raise ValueError(f"{label} path mismatch: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256(resolved)
    if actual != expected_sha:
        raise ValueError(f"{label} SHA mismatch: {actual}")
    return {"path": str(resolved), "sha256": actual}


def assert_caller_hash(path: Path, expected_sha: str, label: str) -> dict[str, str]:
    validate_sha_literal(expected_sha, f"{label} expected SHA")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = sha256(resolved)
    if actual != expected_sha:
        raise ValueError(f"{label} caller SHA mismatch: {actual}")
    return {"path": str(resolved), "sha256": actual}


def validate_train_source_binding() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Validate the physical train-only cache against its true B1 source rows."""
    sealed_ref = assert_expected_file(SEALED_TRAIN, SEALED_TRAIN, EXPECTED_SEALED_SHA, "sealed train cache")
    source_ref = assert_expected_file(SOURCE_TRAIN, SOURCE_TRAIN, EXPECTED_SOURCE_SHA, "source train cache")
    provenance_ref = assert_expected_file(SOURCE_PROVENANCE, SOURCE_PROVENANCE, EXPECTED_SOURCE_PROVENANCE_SHA, "source provenance")
    sealed = torch.load(SEALED_TRAIN, map_location="cpu", weights_only=False)
    v3.validate_train_cache(sealed)
    source = torch.load(SOURCE_TRAIN, map_location="cpu", weights_only=False)
    required = {"features", "labels", "dataset_splits"}
    if required - set(source):
        raise ValueError("source cache lacks train binding fields")
    train_indices = torch.tensor([index for index, split in enumerate(source["dataset_splits"]) if split == "train"], dtype=torch.long)
    if len(train_indices) != EXPECTED_ROWS or not torch.equal(sealed["source_row_indices"], train_indices):
        raise ValueError("sealed source row-index binding mismatch")
    if not torch.equal(sealed["features"], source["features"][train_indices]) or not torch.equal(sealed["labels"], source["labels"][train_indices]):
        raise ValueError("sealed tensors differ from bound source train rows")
    provenance = json.loads(SOURCE_PROVENANCE.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != "features-cache.v1" or provenance.get("image_count") != len(source["features"]) or provenance.get("hidden_dim") != source["features"].shape[1]:
        raise ValueError("source feature provenance contract mismatch")
    return ({"sealed": sealed_ref, "source": source_ref, "source_provenance": provenance_ref, "source_train_rows": len(train_indices)}, sealed["features"].float(), sealed["labels"].long())


def validate_b1_package() -> tuple[benchmark.ModelPackage, dict[str, Any]]:
    projection = assert_expected_file(B1_PROJECTION, B1_PROJECTION, EXPECTED_PROJECTION_SHA, "B1 projection")
    package = benchmark._load_model_package(B1_PACKAGE)
    if package.projection_path.resolve() != B1_PROJECTION.resolve() or package.num_classes != EXPECTED_CLASSES or package.hidden_dim != 384 or package.embedding_dim != 128:
        raise ValueError("B1 package ABI or projection path mismatch")
    return package, {"root": str(B1_PACKAGE.resolve()), "projection": projection, "config_sha256": sha256(package.config_path), "prototypes_sha256": sha256(package.prototypes_path), "manifest_sha256": sha256(package.manifest_path)}


def protocol_spec() -> dict[str, Any]:
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "terminal_only": True,
        "eligible_for_training_or_tuning": False,
        "business_opened_by_preflight": False,
        "runner": {"path": str(Path(__file__).resolve()), "semantic_version": SCHEMA_VERSION},
        "dependencies": {
            "business_validator": {"path": str(Path(business_eval.__file__).resolve()), "sha256": sha256(Path(business_eval.__file__).resolve())},
            "benchmark_evaluator": {"path": str(Path(benchmark.__file__).resolve()), "sha256": sha256(Path(benchmark.__file__).resolve())},
            "v5_research_results": {"path": str(V5_RESULTS.resolve()), "sha256": EXPECTED_V5_RESULTS_SHA},
        },
        "fixed_inputs": {
            "b1_package": str(B1_PACKAGE.resolve()),
            "b1_projection_sha256": EXPECTED_PROJECTION_SHA,
            "sealed_train_cache_sha256": EXPECTED_SEALED_SHA,
            "source_train_cache_sha256": EXPECTED_SOURCE_SHA,
            "source_train_provenance_sha256": EXPECTED_SOURCE_PROVENANCE_SHA,
        },
        "business_inputs": {"provided_only_to_evaluate": True, "caller_sha256_required": ["manifest", "raw_features", "features_provenance"], "validator": "scripts.evaluate_business_test_set._validate_business_cache"},
        "methods": [
            {"name": "packaged_b1_artifact", "rule": "B1 packaged projection plus packaged prototypes", "runtime_compatible": True},
            {"name": "b1_ncm_recomputed", "rule": "B1 projection plus train-only mean prototypes", "runtime_compatible": False},
            {"name": "cosine_knn1", "rule": "nearest train embedding by cosine; top3 ranks nearest per class", "runtime_compatible": False},
            {"name": "margin_fallback_002", "rule": "if top-two train similarity gap >= 0.02 use cache class ranking, else NCM", "runtime_compatible": False},
        ],
        "method_order": list(METHOD_NAMES),
        "fallback_margin": MARGIN,
        "metrics": ["top1_micro", "top1_macro", "top3_micro", "top3_macro", "per_page", "per_class"],
        "comparisons": {"unit": "source_group_page", "bootstrap_draws": BOOTSTRAP_DRAWS, "references": ["packaged_b1_artifact", "b1_ncm_recomputed"]},
        "execution": {"device": EVALUATION_DEVICE, "batch_size": EVALUATION_BATCH_SIZE, "fixed_output": str(TERMINAL_OUTPUT.resolve()), "fixed_ledger": str(TERMINAL_LEDGER.resolve()), "global_one_shot": True, "retry_policy": "no_automatic_retry_after_reserved"},
        "interpretation": {"single_terminal_pass": True, "runtime_compatible": False, "automatic_promotion": False, "no_selection_or_tuning": True},
    }


def preflight(spec_output: Path = FROZEN_SPEC) -> dict[str, Any]:
    """Validate only code/train/B1 inputs. No business path exists in this API."""
    assert_expected_file(V5_RESULTS, V5_RESULTS, EXPECTED_V5_RESULTS_SHA, "V5 research results")
    train_binding, _features, _labels = validate_train_source_binding()
    _package, package_binding = validate_b1_package()
    spec = protocol_spec()
    spec["validated_fixed_inputs"] = {"training": train_binding, "package": package_binding}
    spec["specification_sha256"] = canonical_sha(spec)
    if spec_output.exists():
        existing = json.loads(spec_output.read_text(encoding="utf-8"))
        if existing != spec:
            raise RuntimeError("frozen terminal spec drift")
    else:
        write_json_once(spec_output, spec)
    return spec


def validate_frozen_spec(path: Path) -> dict[str, Any]:
    if path.resolve() != FROZEN_SPEC.resolve():
        raise ValueError("terminal requires the exact fixed frozen spec path")
    if not path.is_file() or sha256(path) != EXPECTED_FROZEN_SPEC_FILE_SHA:
        raise ValueError("terminal frozen spec file SHA mismatch")
    expected = preflight(path)
    if expected.get("specification_sha256") != canonical_sha({key: value for key, value in expected.items() if key != "specification_sha256"}):
        raise ValueError("terminal spec self-hash mismatch")
    if expected["specification_sha256"] != EXPECTED_SPECIFICATION_SHA:
        raise ValueError("terminal specification SHA differs from the pinned contract")
    if expected.get("execution") != {"device": EVALUATION_DEVICE, "batch_size": EVALUATION_BATCH_SIZE, "fixed_output": str(TERMINAL_OUTPUT.resolve()), "fixed_ledger": str(TERMINAL_LEDGER.resolve()), "global_one_shot": True, "retry_policy": "no_automatic_retry_after_reserved"}:
        raise ValueError("terminal execution contract drift")
    return expected


def cache_class_scores(similarities: torch.Tensor, train_labels: torch.Tensor) -> torch.Tensor:
    values = []
    for label in range(EXPECTED_CLASSES):
        mask = train_labels.eq(label)
        if not bool(mask.any()):
            raise ValueError(f"train cache lacks class {label}")
        values.append(similarities[:, mask].max(dim=1).values)
    return torch.stack(values, dim=1)


def fallback_scores(cache_scores: torch.Tensor, ncm_scores: torch.Tensor, top_two_similarities: torch.Tensor, margin: float = MARGIN) -> tuple[torch.Tensor, torch.Tensor]:
    if top_two_similarities.ndim != 2 or top_two_similarities.shape[1] != 2:
        raise ValueError("fallback needs exactly two neighbour similarities")
    use_cache = (top_two_similarities[:, 0] - top_two_similarities[:, 1]).ge(margin)
    return torch.where(use_cache[:, None], cache_scores, ncm_scores), use_cache


def metrics_for(scores: torch.Tensor, labels: torch.Tensor, groups: list[str], class_order: list[str], row_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    top_values, top_indices = scores.topk(3, dim=1)
    correct1 = top_indices[:, 0].eq(labels)
    correct3 = top_indices.eq(labels[:, None]).any(1)
    observed = sorted(set(labels.tolist()))
    per_class = []
    for label in observed:
        mask = labels.eq(label)
        per_class.append({"class_label": label, "class_name": class_order[label], "count": int(mask.sum()), "top1": float(correct1[mask].float().mean()), "top3": float(correct3[mask].float().mean())})
    per_page = []
    for group in sorted(set(groups)):
        mask = torch.tensor([value == group for value in groups], dtype=torch.bool)
        per_page.append({"source_group": group, "count": int(mask.sum()), "top1": float(correct1[mask].float().mean()), "top3": float(correct3[mask].float().mean())})
    rows = [{"row_id": row_id, "source_group": group, "true_label": int(truth), "predicted_label": int(indices[0]), "top3_labels": [int(value) for value in indices], "top3_scores": [float(value) for value in values], "top1_correct": bool(ok1), "top3_correct": bool(ok3)} for row_id, group, truth, indices, values, ok1, ok3 in zip(row_ids, groups, labels, top_indices, top_values, correct1, correct3)]
    return ({"row_count": len(labels), "observed_class_count": len(observed), "page_count": len(set(groups)), "top1_micro": float(correct1.float().mean()), "top3_micro": float(correct3.float().mean()), "top1_macro": sum(item["top1"] for item in per_class)/len(per_class), "top3_macro": sum(item["top3"] for item in per_class)/len(per_class), "per_class": per_class, "per_page": per_page}, rows, correct1)


def evaluate_terminal(*, business_content_sha256: str, business_manifest: Path, business_manifest_sha256: str, business_features: Path, business_features_sha256: str, business_provenance: Path, business_provenance_sha256: str) -> dict[str, Any]:
    spec = validate_frozen_spec(FROZEN_SPEC)
    identity = terminal_identity(business_content_sha256=business_content_sha256, business_manifest_sha256=business_manifest_sha256, business_features_sha256=business_features_sha256, business_provenance_sha256=business_provenance_sha256, specification_sha256=spec["specification_sha256"])
    ledger = reserve_one_shot(identity)
    transition_ledger(ledger_path=TERMINAL_LEDGER, terminal_key=ledger["terminal_key"], expected_state="reserved", new_state="started")
    # The first business path is opened only after the durable `started` state.
    manifest_ref = assert_caller_hash(business_manifest, business_manifest_sha256, "business manifest")
    features_ref = assert_caller_hash(business_features, business_features_sha256, "business features")
    provenance_ref = assert_caller_hash(business_provenance, business_provenance_sha256, "business provenance")
    business, eval_provenance, eval_features, eval_labels, groups = business_eval._validate_business_cache(manifest_path=business_manifest, features_path=business_features, provenance_path=business_provenance)
    if business.get("content_sha256") != business_content_sha256:
        raise ValueError("business content SHA differs from reserved terminal identity")
    train_binding, train_features, train_labels = validate_train_source_binding()
    package, package_binding = validate_b1_package()
    if business["class_order"] != package.class_names or eval_features.shape[1] != package.hidden_dim:
        raise ValueError("business cache differs from B1 runtime ABI")
    torch_device = torch.device(EVALUATION_DEVICE)
    train_embeddings = benchmark._project_features(package, train_features, batch_size=EVALUATION_BATCH_SIZE, device=torch_device)
    eval_embeddings = benchmark._project_features(package, eval_features, batch_size=EVALUATION_BATCH_SIZE, device=torch_device)
    artifact_prototypes, artifact_metadata = benchmark._load_artifact_prototypes(package)
    ncm_prototypes = benchmark._compute_prototypes(train_embeddings, train_labels, class_count=EXPECTED_CLASSES)
    artifact_scores = eval_embeddings @ artifact_prototypes.T
    ncm_scores = eval_embeddings @ ncm_prototypes.T
    similarities = eval_embeddings @ train_embeddings.T
    top_two = similarities.topk(2, dim=1).values
    cache_scores = cache_class_scores(similarities, train_labels)
    fallback, use_cache = fallback_scores(cache_scores, ncm_scores, top_two)
    score_map = {"packaged_b1_artifact": artifact_scores, "b1_ncm_recomputed": ncm_scores, "cosine_knn1": cache_scores, "margin_fallback_002": fallback}
    if tuple(score_map) != METHOD_NAMES:
        raise RuntimeError("terminal method set drift")
    row_ids = [str(row["row_id"]) for row in business["rows"]]
    method_reports, row_predictions, correctness = {}, {}, {}
    runtime_flags = {item["name"]: item["runtime_compatible"] for item in spec["methods"]}
    for name, scores in score_map.items():
        metrics, rows, correct = metrics_for(scores.cpu(), eval_labels.cpu(), groups, business["class_order"], row_ids)
        method_reports[name] = {"runtime_compatible": runtime_flags[name], "metrics": metrics}
        row_predictions[name] = rows
        correctness[name] = correct
    method_reports["packaged_b1_artifact"]["artifact_metadata"] = artifact_metadata
    method_reports["margin_fallback_002"]["cache_decision_count"] = int(use_cache.sum())
    method_reports["margin_fallback_002"]["ncm_fallback_count"] = int((~use_cache).sum())
    comparisons = {}
    for reference in ("packaged_b1_artifact", "b1_ncm_recomputed"):
        comparisons[reference] = {}
        for candidate in METHOD_NAMES:
            if candidate == reference:
                continue
            comparisons[reference][candidate] = business_eval._cluster_bootstrap_delta(correctness[reference], correctness[candidate], groups, draws=BOOTSTRAP_DRAWS)
    per_row = []
    for index, row_id in enumerate(row_ids):
        per_row.append({"row_index": index, "row_id": row_id, "source_group": groups[index], "true_label": int(eval_labels[index]), "methods": {name: row_predictions[name][index] for name in METHOD_NAMES}})
    report = {
        "schema_version": SCHEMA_VERSION,
        "specification_sha256": spec["specification_sha256"],
        "business": {"test_set_id": business["test_set_id"], "content_sha256": business["content_sha256"], "manifest": manifest_ref, "features": features_ref, "features_provenance": provenance_ref, "validated_provenance": eval_provenance},
        "training": train_binding,
        "package": package_binding,
        "methods": method_reports,
        "method_order": list(METHOD_NAMES),
        "per_row_predictions": per_row,
        "group_bootstrap_deltas": comparisons,
        "interpretation": {"terminal_single_pass": True, "eligible_for_training_or_tuning": False, "runtime_compatible": False, "automatic_promotion": False, "business_results_must_not_modify_this_spec": True},
    }
    report["report_sha256"] = canonical_sha(report)
    write_json_once(TERMINAL_OUTPUT, report)
    transition_ledger(ledger_path=TERMINAL_LEDGER, terminal_key=ledger["terminal_key"], expected_state="started", new_state="completed", details={"report_path": str(TERMINAL_OUTPUT.resolve()), "report_sha256": sha256(TERMINAL_OUTPUT)})
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="validate fixed inputs without accepting business paths")
    pre.add_argument("--spec-output", type=Path, default=FROZEN_SPEC)
    run = sub.add_parser("evaluate", help="perform the one terminal business pass")
    run.add_argument("--business-content-sha256", required=True)
    run.add_argument("--business-manifest", type=Path, required=True)
    run.add_argument("--business-manifest-sha256", required=True)
    run.add_argument("--business-features", type=Path, required=True)
    run.add_argument("--business-features-sha256", required=True)
    run.add_argument("--business-provenance", type=Path, required=True)
    run.add_argument("--business-provenance-sha256", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.spec_output)
            print(json.dumps({"status": "READY_NO_BUSINESS_OPENED", "specification_sha256": result["specification_sha256"], "spec": str(args.spec_output.resolve())}, indent=2, sort_keys=True))
        else:
            report = evaluate_terminal(business_content_sha256=args.business_content_sha256, business_manifest=args.business_manifest, business_manifest_sha256=args.business_manifest_sha256, business_features=args.business_features, business_features_sha256=args.business_features_sha256, business_provenance=args.business_provenance, business_provenance_sha256=args.business_provenance_sha256)
            print(json.dumps({"status": "TERMINAL_COMPLETE", "report_sha256": report["report_sha256"], "output": str(TERMINAL_OUTPUT.resolve())}, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

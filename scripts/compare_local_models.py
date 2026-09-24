#!/usr/bin/env python3
"""Compare exact model packages on identical captured crops; never activate a model."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.codex_model.classifier import CodexClassifier
from backend.services.annotation_review import AnnotationReviewStore
from backend.services.model_registry import (
    ModelRegistry, RUNTIME_ARTIFACTS, atomic_write_json, safe_relative_path, sha256_file,
)
from backend.services.training_catalogue import digest


def under(root: Path, relative: str) -> Path:
    path = (root / safe_relative_path(relative)).resolve(strict=True)
    path.relative_to(root.resolve())
    if not path.is_file():
        raise ValueError("comparison media must be a file")
    return path


def capture_samples(args, runtime: Path, run: Path) -> list[dict]:
    watched, rows = {}, []
    store = None
    if args.analysis_id:
        store = AnnotationReviewStore(args.annotations_dir, args.review_manifest)
        before = store.export_review_manifest()
        analysis = next((item for item in store.list_queue()["analyses"]
                         if item["analysis_id"] == args.analysis_id), None)
        if analysis is None:
            raise ValueError("analysis not found")
        source_rows = [
            {**element, "analysis_id": args.analysis_id, "image_name": analysis.get("image_name"),
             "image_path": store.image_path_for(args.analysis_id),
             "crop_path": store.crop_path_for(args.analysis_id, element["index"]),
             "expected_class": element["class_name"] if element["review_status"] == "approved" else None,
             "scope": "ad_hoc"} for element in analysis["elements"]
        ]
    else:
        root = runtime.parent / "annotations"
        snapshot = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if snapshot.get("schema_version") != "local-approved-snapshot.v2":
            raise ValueError("This legacy candidate has no captured pages; select an existing analysis instead.")
        source_rows = []
        for item in snapshot["rows"]:
            if item["dataset_split"] not in ("train", "locked_test"):
                raise ValueError("unsupported snapshot split")
            source_rows.append({**item, "image_path": under(root, item["snapshot_image"]),
                                "crop_path": under(root, item["snapshot_crop"]),
                                "expected_class": item["class_name"], "scope": item["dataset_split"]})
    for ordinal, item in enumerate(source_rows):
        media = {}
        for kind, source, expected in (
            ("crop", Path(item["crop_path"]), item.get("crop_sha256")),
            ("source_image", Path(item["image_path"]), item.get("source_image_sha256")),
        ):
            checksum = sha256_file(source)
            if expected and expected != checksum:
                raise ValueError("snapshot media checksum mismatch")
            name = f"samples/{ordinal:06d}.image" if kind == "crop" else f"pages/{checksum}.image"
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256_file(target) != checksum:
                raise ValueError("media changed during capture")
            watched[source] = checksum
            media[kind + "_file"] = name
            media[kind + "_sha256"] = checksum
        with Image.open(run / media["source_image_file"]) as source_image:
            media["source_image_size"] = list(source_image.size)
        rows.append({key: item.get(key) for key in
                     ("analysis_id", "index", "image_name", "bbox", "expected_class", "scope", "review_status")} |
                    {"sample_id": f"sample-{ordinal:06d}", **media})
    if not rows:
        raise ValueError("no comparison samples")
    if store and store.export_review_manifest() != before:
        raise ValueError("reviews changed during capture; retry")
    if any(sha256_file(path) != checksum for path, checksum in watched.items()):
        raise ValueError("source media changed during capture; retry")
    return rows


def classify(model, rows: list[dict], run: Path) -> tuple[list[dict], float]:
    results, elapsed = [], 0.0
    for offset in range(0, len(rows), 16):
        images = []
        for row in rows[offset:offset + 16]:
            with Image.open(run / row["crop_file"]) as image:
                images.append(image.convert("RGB").copy())
        started = time.perf_counter()
        results.extend(model.classify_batch(images, top_k=3))
        elapsed += (time.perf_counter() - started) * 1000
    if len(results) != len(rows):
        raise ValueError("classifier returned an unexpected number of results")
    return results, elapsed / len(rows)


def correct(result: dict, expected: str | None) -> bool:
    return bool(expected and not result["rejected"] and result["class_name"] == expected)


def candidate_exposure(rows: list[dict], runtime: Path) -> None:
    manifest_path = runtime.parent / "annotations" / "manifest.json"
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if snapshot.get("schema_version") != "local-approved-snapshot.v2":
        for row in rows:
            row["candidate_exposure"] = "unknown"
        return
    seen = {}
    for item in snapshot.get("rows", []):
        checksum = item.get("source_image_sha256")
        if checksum:
            seen.setdefault(checksum, set()).add(item.get("dataset_split"))
    for row in rows:
        if row["scope"] in ("train", "locked_test"):
            row["candidate_exposure"] = row["scope"]
        else:
            splits = seen.get(row["source_image_sha256"], set())
            row["candidate_exposure"] = "train" if "train" in splits else (
                "locked_test" if "locked_test" in splits else "unseen_exact")


def metrics(rows: list[dict]) -> dict:
    common = [row for row in rows if row["expected_class"] and row["base_supported"]]
    new = [row for row in rows if row["expected_class"] and not row["base_supported"]]
    def count(model, subset):
        return sum(correct(row[model], row["expected_class"]) for row in subset)
    def top3(model, subset):
        return sum(any(pred["class_name"] == row["expected_class"] for pred in row[model]["top_k"]) for row in subset)
    return {
        "common": {"support": len(common), "base_correct": count("base", common),
                   "candidate_correct": count("candidate", common),
                   **{key: sum(row["outcome"] == value for row in common) for key, value in
                      (("gains", "gain"), ("regressions", "regression"),
                       ("unchanged", "unchanged"), ("both_wrong", "both_wrong"))}},
        "new_classes": {"support": len(new), "candidate_correct": count("candidate", new)},
        "coverage": {model: sum(not row[model]["rejected"] for row in rows) / len(rows)
                     for model in ("base", "candidate")},
        "top3": {model + "_correct": top3(model, common) for model in ("base", "candidate")},
        "per_class": {name: {"support": len(subset), "base_correct": count("base", subset),
                             "candidate_correct": count("candidate", subset),
                             "base_top3": top3("base", subset), "candidate_top3": top3("candidate", subset)}
                      for name in sorted({row["expected_class"] for row in rows if row["expected_class"]})
                      for subset in [[row for row in rows if row["expected_class"] == name]]},
    }


def compare(args) -> dict:
    active = args.backend_root / "codex_model"
    active_sources = {name: active.joinpath(*parts) for name, parts in RUNTIME_ARTIFACTS}
    active_before = {name: sha256_file(path) for name, path in active_sources.items()}
    pin_before = sha256_file(args.backbone_manifest)
    registry = ModelRegistry(args.registry_dir, repo_root=args.backend_root.parent, runtime_model_dir=active)
    runtime = registry.resolve_runtime_package(args.version_id)
    if runtime.parent.name != args.version_id:
        raise ValueError("comparison requires an exact version ID, not an alias")
    def candidate_hashes():
        return {str(path.relative_to(runtime.parent)): sha256_file(path) for path in
                [runtime.parent / "manifest.json", runtime.parent / "checksums.sha256"] +
                [runtime.joinpath(*parts) for _, parts in RUNTIME_ARTIFACTS]}
    candidate_before = candidate_hashes()
    run = args.output.parent
    if any(path.exists() for path in (args.output, run / "reference_registry", run / "samples", run / "pages")):
        raise ValueError("comparison output already exists")
    run.mkdir(parents=True, exist_ok=True)
    reference = ModelRegistry(run / "reference_registry", repo_root=args.backend_root.parent, runtime_model_dir=active)
    reference.create_version_from_artifacts("reference", artifact_sources=active_sources, status="candidate")
    reference_runtime = reference.resolve_runtime_package("reference")
    if {name: sha256_file(reference_runtime.joinpath(*parts)) for name, parts in RUNTIME_ARTIFACTS} != active_before:
        raise ValueError("active runtime changed while being captured")
    print("Stage: capture des exemples de comparaison", flush=True)
    rows = capture_samples(args, runtime, run)
    candidate_exposure(rows, runtime)
    predictions, latency, loading = {}, {}, {}
    for name, model_path in (("base", reference_runtime), ("candidate", runtime)):
        print(f"Stage: inférence du modèle {name}", flush=True)
        start = time.perf_counter()
        model = CodexClassifier(model_path, device="cpu", backbone_manifest=args.backbone_manifest)
        loading[name] = (time.perf_counter() - start) * 1000
        predictions[name], latency[name] = classify(model, rows, run)
        if name == "base":
            base_names = set(model.class_names)
        del model
    for row, base, candidate in zip(rows, predictions["base"], predictions["candidate"]):
        expected = row["expected_class"]
        supported = expected in base_names
        b_ok, c_ok = correct(base, expected), correct(candidate, expected)
        if expected and not supported:
            outcome = "new_class"
        elif c_ok and not b_ok:
            outcome = "gain"
        elif b_ok and not c_ok:
            outcome = "regression"
        elif expected and not b_ok and not c_ok:
            outcome = "both_wrong"
        elif not expected and (base["class_name"], base["rejected"]) != (candidate["class_name"], candidate["rejected"]):
            outcome = "disagreement"
        else:
            outcome = "unchanged"
        row.update(base=base, candidate=candidate, base_supported=supported, outcome=outcome)
    if (candidate_hashes() != candidate_before or sha256_file(args.backbone_manifest) != pin_before or
            {name: sha256_file(path) for name, path in active_sources.items()} != active_before):
        raise ValueError("model or backbone changed during comparison; retry")
    scopes = sorted({row["scope"] for row in rows})
    primary = "locked_test" if "locked_test" in scopes else scopes[0]
    by_scope = {scope: metrics([row for row in rows if row["scope"] == scope]) for scope in scopes}
    warnings = ["L'indépendance des images vis-à-vis de l'entraînement historique du modèle de base est inconnue.",
                "La séparation détecte les doublons exacts, pas toutes les variantes redimensionnées."]
    if primary != "locked_test":
        warnings.append("Aucun test tenu à l'écart : ces résultats ne prouvent pas la généralisation.")
    if any(row["candidate_exposure"] == "train" for row in rows):
        warnings.append("Certaines pages ont servi à l'apprentissage du candidat : leurs résultats ne mesurent pas sa généralisation.")
    if any(row["candidate_exposure"] == "unseen_exact" for row in rows):
        warnings.append("Page absente du snapshot par empreinte exacte seulement : les variantes proches ne sont pas exclues.")
    if any(row["expected_class"] and not row["base_supported"] for row in rows):
        warnings.append("Les nouvelles classes ne sont pas connues du modèle de base.")
    report = {
        "candidate_version_id": args.version_id, "sample_count": len(rows), "warnings": warnings,
        "protocol": {"evaluation_scope": primary, "metrics_scope": primary,
                     "base_historical_independence": "unknown", "device": "cpu",
                     "latency_scope": "all_samples_batched_cpu_no_warmup"},
        "metrics": {**by_scope[primary], "by_scope": by_scope, "latency_ms": latency, "loading_latency_ms": loading},
        "rows": rows, "integrity": {"active_runtime": active_before, "candidate_runtime": candidate_before,
                                   "backbone_manifest_sha256": pin_before, "samples_revision": digest(rows)},
    }
    atomic_write_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("backend-root", "annotations-dir", "review-manifest", "registry-dir", "backbone-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--analysis-id")
    args = parser.parse_args()
    from scripts.retrain_local import training_lock
    with training_lock(args.backend_root):
        report = compare(args)
    print(json.dumps({"candidate_version_id": report["candidate_version_id"], "sample_count": report["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

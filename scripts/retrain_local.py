#!/usr/bin/env python3
"""Create a local candidate from the shipped prior and current approved crops.

No optimizer, private corpus, network access, or activation. Replaying all current
approvals against the fixed prior makes reruns idempotent and handles revocations.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from backend.codex_model.classifier import PREPROCESSING_VERSION, _ProjectionHead, _preprocess_image
from backend.codex_pipeline.scripts.export_model import _prototype_contract, export_model
from backend.services.annotation_review import AnnotationReviewStore
from backend.services.model_registry import ModelRegistry
from backend.services.training_jobs import _process_is_alive, _read_lock_pid
from scripts.build_training_snapshot import pixel_sha256
from scripts.pin_dinov2 import DINOV2_WEIGHTS_SHA256, load_backbone, sha256_file


def merge_prior(prior: dict, embeddings: torch.Tensor, labels: list[int],
                new_class_names: dict[int, str] | None = None) -> dict:
    """Reconstruct old sums and append prototypes only for confirmed new labels."""
    ordered, names = _prototype_contract(prior, label="shipped base")
    if embeddings.shape != (len(labels), prior["prototypes"].shape[1]):
        raise ValueError("new embeddings and labels do not align")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise ValueError("annotation labels must be integers")
    if not torch.isfinite(embeddings).all() or not torch.allclose(
        embeddings.norm(dim=1), torch.ones(len(labels)), atol=1e-4
    ):
        raise ValueError("new embeddings must be finite unit vectors")
    added = {} if new_class_names is None else new_class_names
    if not isinstance(added, dict) or set(added) != set(labels) - set(ordered):
        raise ValueError("explicit names are required exactly for new annotation labels")
    if any(isinstance(label, bool) or not isinstance(label, int) or label <= max(ordered, default=-1)
           for label in added):
        raise ValueError("new class IDs must follow the historical numeric contract")
    if (any(not isinstance(name, str) or not name.strip() for name in added.values())
            or len(set(added.values())) != len(added) or set(added.values()) & set(names.values())):
        raise ValueError("new class names must be unique and not collide with the base")
    result = copy.deepcopy(prior)
    for index, label in enumerate(ordered):
        meta = prior.get("class_meta", {}).get(label, {})
        count, variance = meta.get("count"), meta.get("variance")
        if variance is None and count == 1:
            variance = 0.0
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"base class {label}: missing valid historical count")
        if isinstance(variance, bool) or not isinstance(variance, (int, float)) or not math.isfinite(variance) or not 0 <= variance <= 1:
            raise ValueError(f"base class {label}: missing valid cosine variance")
        selected = [i for i, value in enumerate(labels) if value == label]
        if not selected:
            continue
        vector = count * (1 - variance) * prior["prototypes"][index] + embeddings[selected].sum(dim=0)
        norm = vector.norm().item()
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"base class {label}: degenerate updated prototype")
        result["prototypes"][index] = vector / norm
        result["class_meta"][label] = {**meta, "count": count + len(selected),
                                     "variance": max(0.0, 1 - norm / (count + len(selected)))}
    for label in sorted(added):
        selected = [i for i, value in enumerate(labels) if value == label]
        vector = embeddings[selected].sum(dim=0)
        norm = vector.norm().item()
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"new class {label}: degenerate prototype")
        result["prototypes"] = torch.cat((result["prototypes"], (vector / norm).unsqueeze(0)))
        result["class_names"][label] = added[label]
        result.setdefault("class_meta", {})[label] = {"count": len(selected), "variance": max(0.0, 1 - norm / len(selected))}
    result["class_labels"] = torch.tensor([*ordered, *sorted(added)], dtype=prior["class_labels"].dtype)
    _prototype_contract(result, label="local candidate")
    return result


def capture_approvals(store: AnnotationReviewStore, destination: Path, names: dict[int, str]) -> dict:
    """Capture immutable sources, deduplicate and reserve whole pages before fitting."""
    from scripts.build_training_snapshot import _deduplicate_and_group, assign_splits, canonical_json_sha
    destination.mkdir(parents=True, exist_ok=False)
    review = store.export_review_manifest()
    records = list(store.iter_approved_annotations())
    if not records:
        raise ValueError("no current approved annotations; approve an element first")
    labels_by_name = {name: label for label, name in names.items()}
    unknown = sorted({record["class_name"] for record in records} - set(labels_by_name))
    if unknown:
        raise ValueError("Confirm new classes in Classes before training: " + ", ".join(unknown))
    rows, pages = [], {}
    for position, record in enumerate(records):
        source = Path(record["crop_path"])
        source_image = Path(record["image_path"])
        source_hash, image_hash = sha256_file(source), sha256_file(source_image)
        if image_hash not in pages:
            image_target = destination / "images" / f"{len(pages):06d}.image"
            image_target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_image, image_target)
            if sha256_file(image_target) != image_hash:
                raise ValueError("annotation source changed during capture; retry")
            pages[image_hash] = (image_target.relative_to(destination).as_posix(), pixel_sha256(image_target))
        page_path, page_pixels = pages[image_hash]
        target = destination / f"{position:06d}.image"
        shutil.copyfile(source, target)
        if sha256_file(target) != source_hash or sha256_file(source) != source_hash:
            raise ValueError("annotation crop changed during capture; retry")
        pixel_hash = pixel_sha256(target)
        source_id = f"{record['analysis_id']}:{record['index']}"
        rows.append({
            **record, "class_label": labels_by_name[record["class_name"]],
            "snapshot_crop": target.name, "snapshot_image": page_path,
            "crop_sha256": source_hash, "pixel_sha256": pixel_hash,
            "source_pixel_sha256": pixel_hash, "source_image_sha256": image_hash,
            "source_group": "image-pixels:" + page_pixels, "source_kind": "live_annotation",
            "source_id": source_id, "row_id": canonical_json_sha([source_id, pixel_hash]),
        })
    selected, duplicates, _, groups = _deduplicate_and_group(rows, exclude_conflicts=False)
    selected, assignments, components = assign_splits(
        selected, groups, parent_assignments={}, salt="local-page-holdout.v1",
        dev_fraction=0.0, test_fraction=0.2, min_train_rows_per_class=1,
    )
    if store.export_review_manifest() != review or list(store.iter_approved_annotations()) != records:
        raise ValueError("annotation reviews changed during capture; retry")
    for row in rows:
        if (sha256_file(Path(row["crop_path"])) != row["crop_sha256"]
                or sha256_file(Path(row["image_path"])) != row["source_image_sha256"]):
            raise ValueError("annotation source changed during capture; retry")
    (destination / "review-index.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    holdout_count = sum(row["dataset_split"] == "locked_test" for row in selected)
    snapshot = {
        "schema_version": "local-approved-snapshot.v2",
        "review_index_sha256": sha256_file(destination / "review-index.json"),
        "approved_count": len(records), "unique_count": len(selected),
        "duplicate_count": len(duplicates), "rows": selected, "duplicates": duplicates,
        "source_group_assignments": assignments, "components": components,
        "holdout_count": holdout_count, "train_count": len(selected) - holdout_count,
        "evaluation_scope": "page_holdout" if holdout_count else "training_fit_only_no_holdout",
        "base_historical_independence": "unknown",
    }
    (destination / "manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def create_candidate(args) -> dict:
    import tempfile
    from backend.services.training_catalogue import training_data_state, validate_local_base

    store = AnnotationReviewStore(args.annotations_dir, args.review_manifest)
    store.ensure_database()
    state = training_data_state(store, args.prior)
    if state["errors"]:
        raise ValueError("; ".join(state["errors"]))
    if not state["records"]:
        raise ValueError("no current approved annotations; approve an element first")
    if getattr(args, "expected_data_revision", None) not in (None, state["data_revision"]):
        raise ValueError("annotations or confirmed classes changed after preparation; verify again")
    registry = ModelRegistry(args.registry_dir, repo_root=ROOT, runtime_model_dir=args.base_model_dir)
    version = registry.version_dir(args.version_id)
    if version.exists():
        raise ValueError("candidate version already exists")
    protected = [args.prior, args.base_model_dir / "config.json",
                 args.base_model_dir / "weights/projection.pt", args.base_model_dir / "weights/prototypes.pt",
                 args.backbone_manifest]
    before = {str(path): sha256_file(path) for path in protected}
    prior, active, projection_state, config = validate_local_base(args.prior, args.base_model_dir)
    ordered, names = _prototype_contract(prior, label="shipped base")
    pin = json.loads(args.backbone_manifest.read_text(encoding="utf-8"))
    if pin.get("weights_sha256") != DINOV2_WEIGHTS_SHA256:
        raise ValueError("local retraining requires the official shipped-base DINOv2 weights")
    merge_prior(prior, torch.empty((0, prior["embedding_dim"])), [])
    registry.versions_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=registry.versions_dir) as temporary:
        working = Path(temporary) / "candidate"
        inputs = working / "inputs"
        inputs.mkdir(parents=True)
        for path, name in zip(protected, ("prior.pt", "config.json", "projection.pt", "active-prototypes.pt", "backbone.json")):
            shutil.copyfile(path, inputs / name)
            if sha256_file(inputs / name) != before[str(path)]:
                raise ValueError("model changed during capture; retry")
        print("Stage: capture des annotations et séparation des pages", flush=True)
        snapshot = capture_approvals(store, working / "annotations", state["names"])
        snapshot.update({"taxonomy_revision": state["revision"], "data_revision": state["data_revision"],
                         "taxonomy": {str(label): name for label, name in state["names"].items()}})
        (working / "annotations/manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        if training_data_state(store, args.prior)["data_revision"] != state["data_revision"]:
            raise ValueError("annotations changed during preparation; retry")
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(args.device)
        print("Stage: vérification du modèle et du backbone local", flush=True)
        backbone, backbone_provenance = load_backbone(config["backbone"], device, inputs / "backbone.json")
        if args.dry_run:
            if any(sha256_file(Path(path)) != value for path, value in before.items()) or training_data_state(store, args.prior)["data_revision"] != state["data_revision"]:
                raise ValueError("model or annotations changed during verification; verify again")
            print(f"Dry-run passed: {snapshot['unique_count']} unique approvals; no candidate created.", flush=True)
            return {"dry_run": True, "unique_count": snapshot["unique_count"], "data_revision": state["data_revision"]}
        projection = _ProjectionHead(prior["hidden_dim"], prior["embedding_dim"]).to(device).eval()
        projection.load_state_dict(projection_state, strict=True)
        features = []
        print("Stage: extraction des caractéristiques", flush=True)
        with torch.inference_mode():
            for offset in range(0, len(snapshot["rows"]), args.batch_size):
                batch = []
                for row in snapshot["rows"][offset:offset + args.batch_size]:
                    path = working / "annotations" / row["snapshot_crop"]
                    if sha256_file(path) != row["crop_sha256"]:
                        raise ValueError("captured crop checksum mismatch")
                    with Image.open(path) as image:
                        batch.append(_preprocess_image(image, config["image_size"]).squeeze(0))
                features.append(projection(backbone(torch.stack(batch).to(device))).cpu())
        embeddings = torch.cat(features)
        train_indices = [index for index, row in enumerate(snapshot["rows"]) if row["dataset_split"] == "train"]
        labels = [snapshot["rows"][index]["class_label"] for index in train_indices]
        training_embeddings = embeddings[train_indices]
        new_names = {label: state["names"][label] for label in set(labels) - set(ordered)}
        print("Stage: mise à jour des prototypes et export du candidat", flush=True)
        updated = merge_prior(prior, training_embeddings, labels, new_names)
        source = working / "prototypes.pt"
        torch.save(updated, source)
        runtime = working / "runtime"
        export_model(source, runtime / "weights", inputs / "config.json",
                     config_out_path=runtime / "config.json", base_model_dir=args.base_model_dir,
                     allow_new_classes=bool(new_names))
        shutil.copyfile(inputs / "projection.pt", runtime / "weights/projection.pt")
        output = torch.load(runtime / "weights/prototypes.pt", map_location="cpu", weights_only=True)
        if not torch.equal(output["prototypes"], updated["prototypes"]):
            raise ValueError("exported candidate differs from computed prototypes")
        def score(model, indices):
            model_labels, _ = _prototype_contract(model, label="evaluation")
            selected = embeddings[indices]
            expected = torch.tensor([snapshot["rows"][index]["class_label"] for index in indices])
            predictions = torch.tensor(model_labels)[(selected @ model["prototypes"].T).argmax(dim=1)]
            return int((predictions == expected).sum())
        holdout_indices = [index for index, row in enumerate(snapshot["rows"]) if row["dataset_split"] == "locked_test"]
        report = {
            "schema_version": "local-retraining-result.v2", "training_mode": "local_prior",
            "evaluation_scope": snapshot["evaluation_scope"], "generalization_validated": False,
            "base_historical_independence": "unknown", "approved_count": snapshot["approved_count"],
            "unique_count": snapshot["unique_count"], "train_count": len(labels), "holdout_count": len(holdout_indices),
            "duplicate_count": snapshot["duplicate_count"],
            "updated_classes": sorted({state["names"][label] for label in labels}),
            "new_classes": sorted(new_names.values()), "taxonomy_revision": state["revision"],
            "data_revision": state["data_revision"],
            "base_correct": score(prior, train_indices), "active_correct": score(active, train_indices),
            "candidate_correct": score(output, train_indices),
            "holdout": {"support": len(holdout_indices), "base_correct": score(active, holdout_indices),
                        "candidate_correct": score(output, holdout_indices)},
            "device": str(device), "preprocessing": PREPROCESSING_VERSION, "inputs_sha256": before,
            "backbone": backbone_provenance, "model_version_id": args.version_id, "activation": "none",
        }
        evaluation = working / "evaluation"
        evaluation.mkdir()
        (evaluation / "local.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if any(sha256_file(Path(path)) != value for path, value in before.items()):
            raise ValueError("base or backbone changed during training; candidate not registered")
        if training_data_state(store, args.prior)["data_revision"] != state["data_revision"]:
            raise ValueError("annotations or classes changed during training; candidate not registered")
        os.replace(working, version)
    registry.write_manifest(
        args.version_id, artifact_paths=[path for path in version.rglob("*") if path.is_file()],
        metadata={
            "training": {"mode": "local_prior", "projection_frozen": True, "preprocessing": PREPROCESSING_VERSION},
            "data": {"approved_count": snapshot["approved_count"], "train_count": len(labels),
                     "unique_count": snapshot["unique_count"], "holdout_count": len(holdout_indices),
                     "taxonomy_revision": state["revision"], "data_revision": state["data_revision"]},
            "metrics": report,
            "promotion": {"blocked": True, "reason": "Candidate only; evaluation does not authorize activation."},
        },
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


@contextmanager
def training_lock(backend_root: Path):
    # The OS releases this guard even after SIGKILL; retain the legacy PID file
    # so the advanced scripts and backend can still identify the active job.
    lock = backend_root / ".retrain.lock"
    with (backend_root / ".retrain.guard").open("a+b") as guard:
        if os.name == "nt":
            import msvcrt
            guard.seek(0)
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if lock.exists():
            pid = _read_lock_pid(lock)
            if pid is not None and _process_is_alive(pid) is not False:
                raise RuntimeError(f"retraining already running (PID {pid})")
            lock.unlink()
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        try:
            yield
        finally:
            if _read_lock_pid(lock) == os.getpid():
                lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", type=Path, default=ROOT / "backend")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-data-revision")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256:
        parser.error("batch size must be between 1 and 256")
    args.prior = args.backend_root / "prototypes/prototypes.pt"
    args.base_model_dir = args.backend_root / "codex_model"
    with training_lock(args.backend_root):
        create_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

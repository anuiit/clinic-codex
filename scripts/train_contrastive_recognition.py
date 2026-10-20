#!/usr/bin/env python3
"""Controlled prototypical/SupCon projection pilot; never changes active weights.

Reuse the reviewed snapshot and actual automatic-SAM crops. Both objectives
receive identical cross-page batches, initialization, architecture and updates.
The old inspected splits are exploratory; new blind pages are still required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "scripts")]
from codex_model.classifier import CodexClassifier, _preprocess_image
from codex_pipeline.models.projection_head import ProjectionHead
from codex_pipeline.models.prototypical import PrototypicalLoss
from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter
from autoresearch_component_contrastive_projection_v18 import supervised_contrastive_loss
from scripts.train_segmentation import read_json, write_json, sha256

SNAPSHOT = ROOT / "backend/training_corpus/snapshots/snapshot-611c734590f272060251"
CACHE = ROOT / "backend/model_registry/versions/20261003-annotation-lanczos-warmstart/training_data/precomputed/features.pt"
RUNTIME = ROOT / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime"
APPLICATION = ROOT / "reports/annotation-retraining-application-20261003.json"
SAM = Path("/home/sina/.cache/mobile_sam/mobile_sam.pt")
PIN = ROOT / "backend/training_corpus/backbone-pins/dinov2-vits14-local.json"
SEEDS = [42, 43, 44]
STEPS = 300


def cross_page_batches(labels, groups, train_indices, seed, steps=STEPS, pixels=None):
    pools = defaultdict(lambda: defaultdict(list))
    seen = set()
    for index in train_indices:
        identity = (int(labels[index]), groups[index], pixels[index] if pixels else index)
        if identity not in seen:
            pools[int(labels[index])][groups[index]].append(index)
            seen.add(identity)
    eligible = sorted(label for label, pages in pools.items() if len(pages) >= 2)
    if len(eligible) < 20:
        raise ValueError("Need 20 classes with at least two training source groups")
    rng = random.Random(seed)
    for _ in range(steps):
        indices, local_labels, support = [], [], []
        for local, label in enumerate(rng.sample(eligible, 20)):
            a, b = rng.sample(sorted(pools[label]), 2)
            count = min(4, len(pools[label][a]), len(pools[label][b]))
            for is_support, group in [(True, a), (False, b)]:
                rows = rng.sample(pools[label][group], count)
                indices.extend(rows); local_labels.extend([local]*count); support.extend([is_support]*count)
        yield indices, torch.tensor(local_labels), torch.tensor(support)


def positive_mask(labels, groups, pixels):
    same_class = labels[:, None] == labels[None, :]
    different_page = torch.tensor([[a != b for b in groups] for a in groups], device=labels.device)
    different_pixels = torch.tensor([[a != b for b in pixels] for a in pixels], device=labels.device)
    return same_class & different_page & different_pixels


@torch.no_grad()
def project(model, features):
    model.eval()
    return torch.cat([model(features[i:i+512]) for i in range(0, len(features), 512)])


@torch.no_grad()
def prototypes(model, features, labels, train_indices):
    counts = torch.bincount(labels[train_indices], minlength=286)
    if not bool((counts > 0).all()):
        raise ValueError("a classifier class has no training exemplars")
    embedded = project(model, features[train_indices])
    targets = labels[train_indices]
    return F.normalize(torch.stack([embedded[targets == label].mean(0) for label in range(286)]), dim=1)


def metrics(scores, labels):
    if len(labels) == 0:
        return {"count": 0}
    ranking = scores.topk(3, dim=1).indices
    correct = ranking[:, 0] == labels
    return {"count": len(labels), "top1_correct": int(correct.sum()),
            "top3_correct": int((ranking == labels[:, None]).any(1).sum()),
            "top1_micro": float(correct.float().mean()),
            "top1_macro": float(torch.stack([correct[labels == c].float().mean() for c in labels.unique()]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-cache", type=Path, help="Reuse an audited earlier extraction without rerunning SAM")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    paths = [CACHE, CACHE.with_suffix(".pt.prov.json"), SNAPSHOT / "metadata.csv",
             SNAPSHOT / "snapshot_manifest.json", APPLICATION, PIN, SAM,
             RUNTIME / "config.json", RUNTIME / "weights/projection.pt", RUNTIME / "weights/prototypes.pt",
             ROOT / "backend/codex_model/weights/projection.pt", ROOT / "backend/codex_model/weights/prototypes.pt"]
    if args.runtime_cache:
        paths.extend([args.runtime_cache, args.runtime_cache.parent / "preregistration.json"])
    paths.extend([ROOT / "backend/training_runs/20261003-bbox-detector-pilot/dataset.json",
                  ROOT / "reports/annotation-retraining-results-20261003.json",
                  ROOT / "backend/codex_pipeline/models/projection_head.py",
                  ROOT / "backend/codex_pipeline/models/prototypical.py",
                  ROOT / "scripts/autoresearch_component_contrastive_projection_v18.py"])
    before = {str(p): sha256(p) for p in paths}
    provenance = read_json(CACHE.with_suffix(".pt.prov.json"))
    assert provenance["snapshot"]["snapshot_manifest_sha256"] == before[str(SNAPSHOT / "snapshot_manifest.json")]
    assert provenance["metadata_csv_sha256"] == before[str(SNAPSHOT / "metadata.csv")]
    recipe = {"schema_version": "contrastive-recognition-pilot.v2", "source_sha256": sha256(Path(__file__)),
              "input_sha256": before, "seeds": SEEDS, "steps_per_method_seed": STEPS,
              "batch": "20 classes; 1-to-4 support and the same number of query, distinct pixels without replacement, different page groups",
              "sam": "frozen automatic generator, 16x16 grid; runtime masks, padding 5, white background",
              "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
              "trainable": "whole 384-to-384-to-128 projection; DINO and SAM frozen",
              "temperature": .1, "selection": "fixed final update, no checkpoint/threshold tuning",
              "objectives": ["prototypical", "supervised_contrastive"],
              "heldout_status": "Previously inspected persisted page splits; exploratory, not fresh blind evidence",
              "corrections": "not consulted during feature extraction, training or evaluation", "activated": False}
    write_json(args.output / "preregistration.json", recipe)
    cache = torch.load(CACHE, weights_only=True, map_location="cpu")
    with (SNAPSHOT / "metadata.csv").open() as stream:
        metadata = {row["row_id"]: row for row in csv.DictReader(stream)}
    assert len(metadata) == len(cache["features"]) == 9716
    for i, row_id in enumerate(cache["row_ids"]):
        row = metadata[row_id]
        assert row["source_group"] == cache["source_groups"][i]
        assert row["dataset_split"] == cache["dataset_splits"][i]
        assert int(row["class_label"]) == int(cache["labels"][i])
    group_splits = defaultdict(set)
    for group, split in zip(cache["source_groups"], cache["dataset_splits"]): group_splits[group].add(split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    app = read_json(APPLICATION)
    page_dataset = read_json(ROOT / "backend/training_runs/20261003-bbox-detector-pilot/dataset.json")
    page_sha = {p["source_group"]: p["image_sha256"] for p in page_dataset["pages"]}
    row_index = {r: i for i, r in enumerate(cache["row_ids"])}
    assert before[str(SAM)] == app["full_pages"]["sam_checkpoint_sha256"]
    expected_hashes = read_json(ROOT / "reports/annotation-retraining-results-20261003.json")["verification"]["runtime_model_artifact_sha256"]
    assert all(before[str(RUNTIME / name)] == digest for name, digest in expected_hashes.items())
    if args.runtime_cache:
        cache_recipe = read_json(args.runtime_cache.parent / "preregistration.json")
        assert cache_recipe["input_sha256"][str(SAM)] == before[str(SAM)]
        assert all(cache_recipe["input_sha256"][str(RUNTIME / name)] == digest for name, digest in expected_hashes.items())
        extracted = torch.load(args.runtime_cache, weights_only=True, map_location="cpu")
        runtime_features, runtime_rows = extracted["features"], extracted["rows"]
        print("REUSE_VERIFIED_RUNTIME_CROPS", len(runtime_rows), flush=True)
    else:
        classifier = CodexClassifier(model_dir=RUNTIME, device="cuda", backbone_source=PIN)
        segmenter = MobileSAMSegmenter(checkpoint_path=str(SAM), device="cuda", points_per_side=16)
        runtime_features, runtime_rows = [], []
        for number, page in enumerate(app["full_pages"]["pages"], 1):
            assert sha256(Path(page["source_image_path"])) == page_sha[page["source_group"]]
            pixels = np.array(Image.open(page["source_image_path"]).convert("RGB"))
            proposals = segmenter.extract_crops(pixels, segmenter.segment_page(pixels), padding=5)
            valid = [p for p in proposals if p.crop is not None]
            assert len({tuple(p.bbox) for p in valid}) == len(valid), "duplicate proposal boxes"
            by_box = {tuple(p.bbox): p.crop for p in valid}
            rows = page["matched_rows"]
            for offset in range(0, len(rows), 16):
                part = rows[offset:offset+16]
                batch = torch.cat([_preprocess_image(by_box[tuple(r["proposal_bbox"])], 224) for r in part]).cuda()
                with torch.no_grad(): features = classifier._backbone(batch).float().cpu()
                for feature, row in zip(features, part):
                    index = row_index[row["row_id"]]
                    assert cache["dataset_splits"][index] == page["split"]
                    assert cache["source_groups"][index] == page["source_group"]
                    assert cache["class_names"][int(cache["labels"][index])] == row["class_name"]
                    runtime_features.append(feature)
                    runtime_rows.append({**row, "source_group": page["source_group"], "snapshot_index": index})
            print("RUNTIME_CROPS", number, "of", len(app["full_pages"]["pages"]), len(rows), flush=True)
        runtime_features = torch.stack(runtime_features)
        assert len(runtime_rows) == 326
        torch.save({"features": runtime_features, "rows": runtime_rows}, args.output / "runtime_features.pt")
        del classifier, segmenter
        torch.cuda.empty_cache()
    assert len(runtime_rows) == len(runtime_features) == len({r["row_id"] for r in runtime_rows}) == 326
    assert len({r["snapshot_index"] for r in runtime_rows}) == 326
    expected_rows = {r["row_id"]: (page, r) for page in app["full_pages"]["pages"] for r in page["matched_rows"]}
    for r in runtime_rows:
        page, expected = expected_rows[r["row_id"]]
        i = r["snapshot_index"]
        assert cache["row_ids"][i] == r["row_id"] and cache["dataset_splits"][i] == page["split"]
        assert cache["source_groups"][i] == page["source_group"] and r["proposal_bbox"] == expected["proposal_bbox"]
    torch.save({"features": runtime_features, "rows": runtime_rows}, args.output / "runtime_features.pt")
    device = torch.device("cuda")
    raw_features = cache["features"].float().to(device)
    features = raw_features.clone()
    labels = cache["labels"].long().to(device)
    runtime_indices = [r["snapshot_index"] for r in runtime_rows]
    features[runtime_indices] = runtime_features.to(device)
    business_groups = {p["source_group"] for p in app["full_pages"]["pages"]}
    matched_indices = set(runtime_indices)
    train_indices = [i for i, s in enumerate(cache["dataset_splits"]) if s == "train"
                     and (cache["source_groups"][i] not in business_groups or i in matched_indices)]
    assert all(cache["dataset_splits"][i] == "train" for i in train_indices)
    assert not set(train_indices) & {i for i, s in enumerate(cache["dataset_splits"]) if s != "train"}
    pools = defaultdict(set)
    for i in train_indices: pools[int(labels[i])].add(cache["source_groups"][i])
    class_eligibility = {str(c): len(g) >= 2 for c, g in pools.items()}
    init = torch.load(RUNTIME / "weights/projection.pt", weights_only=True, map_location="cpu")
    def fresh():
        model = ProjectionHead().to(device); model.load_state_dict(init); return model
    original_prototypes = torch.load(RUNTIME / "weights/prototypes.pt", weights_only=True)["prototypes"].to(device)
    original = fresh()
    models = {"current_stored_prototypes": (original, original_prototypes),
              "initial_train_recomputed": (original, prototypes(original, features, labels, train_indices))}
    histories = {}
    loss_prototype = PrototypicalLoss(temperature=.1)
    pixel_ids = [metadata[r]["source_pixel_sha256"] for r in cache["row_ids"]]
    pixel_groups = defaultdict(set)
    for i in train_indices: pixel_groups[pixel_ids[i]].add(cache["source_groups"][i])
    assert all(len(g) == 1 for g in pixel_groups.values()), "identical pixels cross training page groups"
    for seed in SEEDS:
        batches = list(cross_page_batches(labels, cache["source_groups"], train_indices, seed, pixels=pixel_ids))
        for objective in recipe["objectives"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = fresh(); model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            history = []
            for step, (indices, local, support) in enumerate(batches, 1):
                local, support = local.to(device), support.to(device)
                embedded = model(features[indices])
                if objective == "prototypical":
                    loss = loss_prototype(embedded[support], local[support], embedded[~support], local[~support])["loss"]
                else:
                    positives = positive_mask(local, [cache["source_groups"][i] for i in indices], [pixel_ids[i] for i in indices])
                    loss = supervised_contrastive_loss(embedded, positives, temperature=.1)
                if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                history.append(float(loss.detach()))
                if step % 100 == 0: print("FIT", objective, seed, step, "LOSS", history[-1], flush=True)
            key = f"{objective}_seed_{seed}"
            proto = prototypes(model, features, labels, train_indices)
            torch.save({"projection": model.state_dict(), "prototypes": proto.cpu(), "class_names": cache["class_names"],
                        "train_indices": train_indices, "seed": seed, "objective": objective}, args.output / f"{key}.pt")
            models[key] = (model, proto)
            histories[key] = history
    report = {"schema_version": "contrastive-recognition-results.v1", "recipe": recipe,
              "train_rows": len(train_indices), "runtime_crops": len(runtime_rows),
              "eligible_cross_page_classes": sum(class_eligibility.values()), "class_eligibility": class_eligibility,
              "matched_training_rows": sum(r["split"] == "train" for r in runtime_rows),
              "excluded_unmatched_business_training_rows": sum(s == "train" for s in cache["dataset_splits"])-len(train_indices),
              "evaluation": {}, "history": histories, "activated": False,
              "limitations": ["SAM misses stay misses; contrastive only tests recognition of existing proposals",
                              "Classes without cross-page positives stay in prototypes but have no direct training anchors; see eligibility",
                              "Historical clean rectangular exemplars remain; business training uses actual masked SAM crops",
                              "Heldout pages previously inspected; new blind pages and confirmed business installation required"]}
    for key, (model, proto) in models.items():
        scores = project(model, features) @ proto.T
        evaluation = {"snapshot_raw_crops": {}, "application": {}, "pages": []}
        raw_scores = project(model, raw_features) @ proto.T
        for split in ["train", "dev", "locked_test"]:
            indices = [i for i, s in enumerate(cache["dataset_splits"]) if s == split]
            evaluation["snapshot_raw_crops"][split] = metrics(raw_scores[indices], labels[indices])
            rows = [r for r in runtime_rows if r["split"] == split]
            found = [r["snapshot_index"] for r in rows]
            result = metrics(scores[found], labels[found])
            result["known_boxes"] = app["full_pages"]["by_split"][split]["ground_truth_count"]
            result["correct_over_known_boxes"] = result["top1_correct"] / result["known_boxes"]
            result["missed_known_boxes"] = result["known_boxes"]-len(found)
            evaluation["application"][split] = result
        for page in app["full_pages"]["pages"]:
            rows = [r for r in runtime_rows if r["source_group"] == page["source_group"]]
            found = [r["snapshot_index"] for r in rows]
            ranking = scores[found].topk(3,dim=1).indices.cpu().tolist()
            evaluation["pages"].append({"source_group": page["source_group"], "split": page["split"],
                "known_boxes": page["ground_truth_count"], "metrics": metrics(scores[found], labels[found]),
                "rows": [{"row_id": r["row_id"], "class_name": r["class_name"], "proposal_bbox": r["proposal_bbox"],
                          "top3": [cache["class_names"][c] for c in ranks]} for r, ranks in zip(rows, ranking)]})
        report["evaluation"][key] = evaluation
        print("RESULT", key, {s: m["top1_correct"] for s,m in evaluation["application"].items()}, flush=True)
    baseline = report["evaluation"]["current_stored_prototypes"]["application"]
    assert [baseline[s]["top1_correct"] for s in ["train","dev","locked_test"]] == [160,8,26]
    assert before == {str(p): sha256(p) for p in paths}
    report["verification"] = {"input_files_unchanged": True, "baseline_reproduced_exactly": True,
                              "source_groups_disjoint": True, "training_only_train_indices": True,
                              "both_objectives_identical_batches_and_initial_state": True,
                              "fixed_last_checkpoint_no_test_selection": True}
    report["checkpoint_sha256"] = {p.name: sha256(p) for p in args.output.glob("*.pt")}
    report["verification"]["distinct_batch_rows_without_replacement"] = True
    report["verification"]["deterministic_torch_algorithms"] = True
    report["verification"]["runtime_sha_and_checkpoint_match_previous_application_report"] = True
    write_json(args.output / "results.json", report)
    print("COMPLETE", args.output / "results.json", flush=True)


if __name__ == "__main__": main()

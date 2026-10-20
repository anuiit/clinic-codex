"""Adapt MobileSAM mask output layers from reviewed boxes; evaluate full pages.

Boxes supervise mask projections; box-prompted frozen SAM supplies weak mask
labels. Student prompts are runtime-grid points only. Outside each target
instance is background for that instance; page annotations need not be exhaustive.
Automatic generation and its thresholds are unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from backend.services.annotation_review import AnnotationReviewStore
from scripts.evaluate_glyph_benchmark import Box, _greedy_match, box_iou
from scripts.import_annotation_archive import _pixel_sha


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def grid_points(width, height, side=16):
    # Identical half-cell grid to SAM build_point_grid, in page pixels.
    axis = (np.arange(side, dtype=np.float32) + 0.5) / side
    x, y = np.meshgrid(axis * width, axis * height)
    return np.stack([x.ravel(), y.ravel()], axis=1)


def build_dataset(import_report, snapshot, root=ROOT):
    imported, frozen = read_json(import_report), read_json(snapshot)
    if not imported.get("applied") or not imported.get("re_attest_verified"):
        raise ValueError("Use a verified, applied annotation import")
    store = Path(imported["annotations_dir"]).resolve()
    if not store.is_relative_to(root.resolve()):
        raise ValueError("Imported annotation store outside repository")
    queue = AnnotationReviewStore(store).list_queue()
    expected = {r["key"]: r for r in imported["rows"]}
    pages, image_info = {}, {}
    approved, duplicates = 0, 0

    def add(path, bbox, key, classes, origin):
        nonlocal duplicates
        path = Path(path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Page path outside repository")
        if path not in image_info:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image_info[path] = (_pixel_sha(image), *image.size, sha256(path))
        pixel_sha, width, height, image_sha = image_info[path]
        group = "image-pixels:" + pixel_sha
        split = frozen["source_group_assignments"].get(group)
        if split not in {"train", "dev", "locked_test"}:
            raise ValueError("No frozen page split for " + group)
        if (len(bbox) != 4 or any(not math.isfinite(v) for v in bbox)
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0
                or bbox[0] + bbox[2] > width or bbox[1] + bbox[3] > height):
            raise ValueError("Invalid page bbox: " + key)
        page = pages.setdefault(group, {"source_group": group, "split": split,
            "image_path": str(path), "image_sha256": image_sha,
            "width": width, "height": height, "boxes": {}})
        geometry = tuple(bbox)
        if geometry in page["boxes"]:
            duplicates += 1
            box = page["boxes"][geometry]
            box["annotation_keys"].append(key)
            box["class_names"] = sorted(set(box["class_names"]) | set(classes))
        else:
            page["boxes"][geometry] = {"bbox": list(bbox), "annotation_keys": [key],
                                       "class_names": classes, "origin": origin}
        return pixel_sha

    for analysis in queue["analyses"]:
        for element in analysis["elements"]:
            if element["review_status"] != "approved" or element["stale_decision"]:
                continue
            source = expected[element["key"]]
            if source["exclusion_reason"] not in {None, "unknown_class"}:
                continue
            if not element["crop_exists"] or not analysis["image_exists"]:
                raise ValueError("Approved geometry missing source image/crop")
            if source["bbox"] != element["bbox"] or source["class_name"] != element["class_name"]:
                raise ValueError("Verified annotation changed")
            digest = add(analysis["image_path"], element["bbox"], element["key"],
                         [element["class_name"]], "imported")
            if digest != source["source_image_pixel_sha256"]:
                raise ValueError("Verified page pixels changed")
            approved += 1
    # Geometry remains useful for class-agnostic segmentation when names conflict.
    for row in frozen["rows"]:
        if row["source_group"] in pages and row.get("bbox") and row.get("source_image_path"):
            if tuple(row["bbox"]) not in pages[row["source_group"]]["boxes"]:
                digest = add(root / row["source_image_path"], row["bbox"],
                             row["row_id"], [row["class_name"]], "parent")
                if "image-pixels:" + digest != row["source_group"]:
                    raise ValueError("Frozen parent page pixels changed")
    for page in pages.values():
        page["boxes"] = list(page["boxes"].values())
    result = {"schema_version": "box-supervised-segmentation-dataset.v1",
        "snapshot_id": frozen["snapshot_id"], "snapshot_sha256": sha256(snapshot),
        "import_report_sha256": sha256(import_report), "approved_imported_rows": approved,
        "duplicate_imported_geometries": duplicates,
        "pages": sorted(pages.values(), key=lambda p: p["source_group"]),
        "annotation_exhaustiveness": "unknown; extra proposals are not false positives",
        "mask_ground_truth": "absent",
        "unknown_class_policy": "geometry is usable only after current review approval; pending archive labels remain excluded",
        "excluded_pending_unknown_class_rows": sum(e["review_status"] != "approved" and expected[e["key"]]["exclusion_reason"] == "unknown_class" for a in queue["analyses"] for e in a["elements"])}
    result["split_box_counts"] = {s: sum(len(p["boxes"]) for p in result["pages"] if p["split"] == s)
                                  for s in ["train", "dev", "locked_test"]}
    result["split_page_counts"] = dict(Counter(p["split"] for p in result["pages"]))
    if any(not result["split_box_counts"][s] for s in ["train", "dev", "locked_test"]):
        raise ValueError("Train, dev and locked_test require separate pages")
    return result


def projection_loss(probabilities, boxes):
    def dice(a, b):
        return 1 - (2 * (a * b).sum(-1) + 1e-6) / (a.sum(-1) + b.sum(-1) + 1e-6)
    return dice(probabilities.amax(-2), boxes.amax(-2)) + dice(probabilities.amax(-1), boxes.amax(-1))


def mask_loss(logits, targets, boxes):
    """One selected token per point; other ambiguous masks remain unsupervised."""
    targets, boxes = targets[:, None], boxes[:, None]
    probabilities = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits), reduction="none")
    positive = (bce * targets).sum((-2, -1)) / targets.sum((-2, -1)).clamp_min(1)
    negative = (bce * (1 - targets)).sum((-2, -1)) / (1 - targets).sum((-2, -1)).clamp_min(1)
    dice = 1 - (2 * (probabilities * targets).sum((-2, -1)) + 1e-6) / (
        probabilities.sum((-2, -1)) + targets.sum((-2, -1)) + 1e-6)
    losses = 0.5 * (positive + negative) + dice + projection_loss(probabilities, boxes)
    winner = losses.detach().argmin(1)
    return losses[torch.arange(len(losses), device=logits.device), winner]


def box_mask(box, size=256):
    x1, y1, x2, y2 = box
    mask = torch.zeros(size, size)
    mask[max(0, math.floor(y1)):min(size, math.ceil(y2)),
         max(0, math.floor(x1)):min(size, math.ceil(x2))] = 1
    return mask


def mask_box_overlap(mask, rectangle):
    def bounds(a):
        y, x = torch.where(a > 0)
        return Box(float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1))
    return box_iou(bounds(mask), bounds(rectangle)) if mask.any() and rectangle.any() else 0.0


def prepare_teacher_cache(segmenter, pages, batch_size):
    predictor = segmenter._mask_generator.predictor
    cache, exclusions = [], Counter()
    for page in pages:
        if page["split"] != "train":
            raise ValueError("Held-out page passed to teacher fitting cache")
        image = np.array(Image.open(page["image_path"]).convert("RGB"))
        predictor.set_image(image)
        boxes = np.array([xyxy(b["bbox"]) for b in page["boxes"]], dtype=np.float32)
        transformed = predictor.transform.apply_boxes(boxes, image.shape[:2])
        points = grid_points(page["width"], page["height"])
        transformed_points = predictor.transform.apply_coords(points, image.shape[:2])
        samples = []
        for start in range(0, len(boxes), batch_size):
            b = torch.as_tensor(transformed[start:start + batch_size], device=segmenter.device)
            _, _, logits = predictor.predict_torch(None, None, boxes=b, multimask_output=False)
            for offset, target_logits in enumerate(logits[:, 0].cpu()):
                index = start + offset
                rectangle = box_mask(transformed[index] / 4)
                target = (target_logits > 0).float() * rectangle
                if mask_box_overlap(target, rectangle) < 0.5:
                    exclusions["empty_or_box_incoherent_teacher_mask"] += 1
                    continue
                px, py = (transformed_points / 4).astype(int).T
                eligible = target[py.clip(0, 255), px.clip(0, 255)].numpy() > 0
                x1, y1, x2, y2 = boxes[index]
                eligible &= ((points[:, 0] >= x1) & (points[:, 0] < x2)
                             & (points[:, 1] >= y1) & (points[:, 1] < y2))
                if not eligible.any():
                    exclusions["no_runtime_grid_point_in_teacher_mask"] += 1
                    continue
                samples.append({"points": torch.from_numpy(transformed_points[eligible].copy()),
                    "target": target, "box": rectangle, "annotation_keys": page["boxes"][index]["annotation_keys"]})
        cache.append({"source_group": page["source_group"],
            "features": predictor.features.detach().cpu().clone(), "samples": samples})
        print("TEACHER", page["source_group"], len(samples), "/", len(boxes), flush=True)
    if not sum(len(p["samples"]) for p in cache):
        raise ValueError("No train boxes have valid teacher masks and runtime-grid prompts")
    return cache, dict(exclusions)


def evaluate(segmenter, pages):
    results = []
    for page in pages:
        image = np.array(Image.open(page["image_path"]).convert("RGB"))
        proposals = segmenter.segment_page(image)
        gt = [{"box": Box(*xyxy(b["bbox"]))} for b in page["boxes"]]
        pred = [{"box": Box(*xyxy(p.bbox))} for p in proposals]
        matches = _greedy_match(gt, pred, 0.5)
        points = grid_points(page["width"], page["height"])
        reachable = [i for i, b in enumerate(page["boxes"]) if np.any(
            (points[:, 0] >= b["bbox"][0]) & (points[:, 0] < b["bbox"][0] + b["bbox"][2]) &
            (points[:, 1] >= b["bbox"][1]) & (points[:, 1] < b["bbox"][1] + b["bbox"][3]))]
        results.append({"source_group": page["source_group"], "split": page["split"],
            "known_boxes": len(gt), "proposal_count": len(pred), "matched_boxes": len(matches),
            "matched_at_iou_025": len(_greedy_match(gt, pred, 0.25)),
            "matched_at_iou_075": len(_greedy_match(gt, pred, 0.75)),
            "reachable_boxes": len(reachable), "matched_reachable_boxes": sum(g in reachable for g, _, _ in matches),
            "matched_iou_sum": sum(o for _, _, o in matches),
            "matches": [{"ground_truth_index": g, "proposal_index": p, "iou": o} for g, p, o in matches],
            "proposal_bboxes": [list(p.bbox) for p in proposals]})
    if not results:
        raise ValueError("No evaluation pages")
    counts = {k: sum(p[k] for p in results) for k in ["known_boxes", "proposal_count", "matched_boxes",
        "matched_at_iou_025", "matched_at_iou_075", "reachable_boxes", "matched_reachable_boxes", "matched_iou_sum"]}
    counts["known_box_coverage"] = counts["matched_boxes"] / counts["known_boxes"]
    counts["reachable_box_coverage"] = counts["matched_reachable_boxes"] / max(1, counts["reachable_boxes"])
    counts["mean_matched_iou"] = counts["matched_iou_sum"] / max(1, counts["matched_boxes"])
    return {"counts": counts, "pages": results, "precision": None, "mask_iou": None,
        "limitation": "Partial bbox annotations: known-box coverage only, no exhaustive recall, precision or pixel-mask accuracy"}


def state_fingerprint(model, prefix=None):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if prefix is None or name.startswith(prefix):
            digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decode(sam, features, points):
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=(points[:, None], torch.ones(len(points), 1, device=points.device)), boxes=None, masks=None)
    return sam.mask_decoder(image_embeddings=features, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["import-report", "snapshot", "checkpoint", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("Invalid training recipe")
    if args.output_dir.exists():
        parser.error("Output directory already exists; preserve previous experiments")
    dataset = build_dataset(args.import_report, args.snapshot)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "dataset.json", dataset)
    recipe = {"schema_version": "box-supervised-mobilesam-training.v1", "seed": args.seed,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "base_checkpoint": str(args.checkpoint.resolve()), "base_checkpoint_sha256": sha256(args.checkpoint),
        "dataset_sha256": sha256(args.output_dir / "dataset.json"), "script_sha256": sha256(__file__),
        "trainable": ["mask_decoder.output_upscaling", "mask_decoder.output_hypernetworks_mlps.1-3"],
        "frozen": ["image_encoder", "prompt_encoder", "mask_decoder.transformer", "mask_decoder.tokens", "mask_decoder.iou_prediction_head"],
        "supervision": "one-token balanced BCE + Dice to box-prompted frozen teacher mask, plus differentiable bbox projection Dice",
        "prompts": "runtime 16x16 grid points inside teacher foreground; no bbox prompts to student",
        "teacher_mask_minimum_box_iou": 0.5, "runtime_points_per_side": 16,
        "selection": "dev only: most IoU>=0.5 matches, then mean matched IoU; proposal budget <=1.5x baseline+5; latest research fallback if none eligible",
        "sources": ["https://github.com/ChaoningZhang/MobileSAM", "https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html"],
        "promotion": "none; exploratory candidate with weak masks and partial ground truth"}
    write_json(args.output_dir / "preregistration.json", recipe)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    from codex_pipeline.segmentation.mobilesam import MobileSAMSegmenter
    segmenter = MobileSAMSegmenter(checkpoint_path=str(args.checkpoint), device=args.device, points_per_side=16)
    sam = segmenter._mask_generator.predictor.model
    initial = state_fingerprint(sam)
    frozen_prefixes = ["image_encoder", "prompt_encoder", "mask_decoder.transformer", "mask_decoder.iou", "mask_decoder.mask_tokens"]
    frozen_before = {p: state_fingerprint(sam, p) for p in frozen_prefixes}
    train_pages = [p for p in dataset["pages"] if p["split"] == "train"]
    dev_pages = [p for p in dataset["pages"] if p["split"] == "dev"]
    baseline_dev = evaluate(segmenter, dev_pages)
    cache, exclusions = prepare_teacher_cache(segmenter, train_pages, args.batch_size)
    torch.save(cache, args.output_dir / "teacher_cache.pt")
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    heads = [sam.mask_decoder.output_upscaling, *list(sam.mask_decoder.output_hypernetworks_mlps)[1:]]
    for head in heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    parameters = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    reference = next(p for p in cache if p["samples"])
    check_features = reference["features"].to(args.device)
    check_points = reference["samples"][0]["points"][:1].to(args.device)
    with torch.no_grad():
        reference_scores = decode(sam, check_features, check_points)[1].clone()
    history, best_score, best_epoch = [], None, None
    for epoch in range(1, args.epochs + 1):
        random.shuffle(cache)
        total_loss, examples = 0.0, 0
        for page in cache:
            samples = page["samples"][:]; random.shuffle(samples)
            features = page["features"].to(args.device)
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                points = torch.stack([s["points"][random.randrange(len(s["points"]))] for s in batch]).to(args.device)
                targets = torch.stack([s["target"] for s in batch]).to(args.device)
                boxes = torch.stack([s["box"] for s in batch]).to(args.device)
                logits, _ = decode(sam, features, points)
                loss = mask_loss(logits, targets, boxes).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite segmentation loss")
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
                total_loss += float(loss.detach()) * len(batch); examples += len(batch)
        torch.save(sam.state_dict(), args.output_dir / "latest.pt")
        dev = evaluate(segmenter, dev_pages)
        counts = dev["counts"]
        within_budget = counts["proposal_count"] <= 1.5 * baseline_dev["counts"]["proposal_count"] + 5
        score = (counts["matched_boxes"], counts["mean_matched_iou"])
        if within_budget and (best_score is None or score > best_score):
            best_score, best_epoch = score, epoch
            torch.save(sam.state_dict(), args.output_dir / "best.pt")
        history.append({"epoch": epoch, "mean_train_loss": total_loss / examples,
            "train_examples": examples, "dev": dev, "within_proposal_budget": within_budget})
        write_json(args.output_dir / "history.json", history)
        print("EPOCH", epoch, "LOSS", round(total_loss / examples, 4), "DEV", counts, flush=True)
    selection_eligible = best_epoch is not None
    if not selection_eligible:
        best_epoch = args.epochs
        torch.save(sam.state_dict(), args.output_dir / "best.pt")
    sam.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=True))
    with torch.no_grad():
        final_scores = decode(sam, check_features, check_points)[1]
    assert torch.equal(reference_scores, final_scores), "Frozen IoU scores changed"
    assert state_fingerprint(sam) != initial, "No segmentation weights changed"
    assert all(state_fingerprint(sam, p) == v for p, v in frozen_before.items())
    candidate = {s: evaluate(segmenter, [p for p in dataset["pages"] if p["split"] == s])
                 for s in ["train", "dev", "locked_test"]}
    sam.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    baseline = {s: evaluate(segmenter, [p for p in dataset["pages"] if p["split"] == s])
                for s in ["train", "locked_test"]}
    baseline["dev"] = baseline_dev
    result = {"schema_version": "box-supervised-segmentation-results.v1", "dataset": dataset,
        "recipe": recipe, "teacher": {"eligible_train_boxes": sum(len(p["samples"]) for p in cache), "exclusions": exclusions},
        "selected_epoch": best_epoch, "selection_eligible": selection_eligible,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "history": history, "baseline": baseline, "candidate": candidate,
        "verification": {"frozen_components_unchanged": True, "iou_scores_same_for_identical_inputs": True,
            "selected_checkpoint_sha256": sha256(args.output_dir / "best.pt"), "segmentation_weights_changed": True,
            "base_checkpoint_unchanged": sha256(args.checkpoint) == recipe["base_checkpoint_sha256"],
            "no_locked_test_used_for_fit_or_selection": True}, "activated": False}
    write_json(args.output_dir / "results.json", result)
    print("RESULTS", args.output_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()

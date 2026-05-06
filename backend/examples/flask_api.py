"""
REST API for the Codex Element Classifier.

Endpoints:
    POST /classify       - classify a single image
    POST /classify-batch - classify multiple images
    POST /segment        - segment + classify a glyph with MobileSAM
    POST /similar        - crop + classify a bbox region with prototype similarities
    POST /trust          - crop + classify a bbox region with trust signals
    GET  /classes        - list all 286 element classes

Usage:
    python flask_api.py
    # Then: curl -X POST -F "image=@element.jpg" http://localhost:5000/classify
"""

import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_model import CodexClassifier
from codex_pipeline.segmentation import MobileSAMSegmenter

app = Flask(__name__)
# Limit request payloads to 50MB to avoid large uploads
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Environment-driven configuration
# PORT, HOST, CORS_ORIGINS, MODEL_DIR
PORT = int(os.environ.get("PORT", "7117"))
HOST = os.environ.get("HOST", "0.0.0.0")
_raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:7118")
# Allow comma-separated origins
ALLOWED_CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

MODEL_DIR = os.environ.get("MODEL_DIR", "")

# Initialize classifier with optional model_dir
if MODEL_DIR:
    clf = CodexClassifier(model_dir=MODEL_DIR)
else:
    clf = CodexClassifier()

sam_segmenter: MobileSAMSegmenter | None = None
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_INDEX: dict[str, list[dict[str, str]]] = {}


def _sample_class_name(class_dir_name: str) -> str:
    if class_dir_name.endswith("-glyph"):
        return class_dir_name[: -len("-glyph")]

    if "-" in class_dir_name:
        prefix, remainder = class_dir_name.split("-", 1)
        if prefix.isdigit() and remainder:
            return remainder

    return class_dir_name


def _build_sample_index() -> None:
    sample_index: dict[str, list[dict[str, str]]] = {}
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}

    for subdir in ["elements_sample", "glyphs_sample"]:
        sample_dir = DATA_DIR / subdir
        if not sample_dir.exists():
            continue

        for class_dir in sample_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = _sample_class_name(class_dir.name)
            sample_index.setdefault(class_name, [])

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in valid_suffixes:
                    sample_index[class_name].append({"path": str(image_path), "class_name": class_name})

    SAMPLE_INDEX.clear()
    SAMPLE_INDEX.update(sample_index)


_build_sample_index()
print(f"Sample index: {len(SAMPLE_INDEX)} classes, {sum(len(v) for v in SAMPLE_INDEX.values())} images")


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    # If the request Origin matches one of the allowed origins, echo it.
    if origin and origin in ALLOWED_CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # Fallback to the first allowed origin (useful for simple clients)
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_CORS_ORIGINS[0] if ALLOWED_CORS_ORIGINS else ""
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    return jsonify(clf.classify(img))


@app.route("/classify-batch", methods=["POST"])
def classify_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No 'images' files in request"}), 400
    images: list[Image.Image | np.ndarray] = [np.array(Image.open(io.BytesIO(f.read())).convert("RGB")) for f in files]
    return jsonify(clf.classify_batch(images))


@app.route("/segment", methods=["POST"])
def segment():
    global sam_segmenter

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    img = np.array(Image.open(io.BytesIO(request.files["image"].read())).convert("RGB"))

    h, w = img.shape[:2]

    if sam_segmenter is None:
        sam_segmenter = MobileSAMSegmenter(points_per_side=16)

    proposals = sam_segmenter.segment_page(img)
    proposals = sam_segmenter.extract_crops(img, proposals)

    elements = []
    for proposal in proposals:
        if proposal.crop is None:
            continue

        x, y, bw, bh = proposal.bbox
        result = clf.classify(proposal.crop)
        elements.append({"bbox": [int(x), int(y), int(bw), int(bh)], **result})

    return jsonify({"num_elements": len(elements), "image_size": [int(w), int(h)], "elements": elements})


@app.route("/similar", methods=["POST"])
def similar():
    data = request.get_json()
    if not data or "image_base64" not in data or "bbox" not in data:
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}}), 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": {"code": "INVALID_IMAGE", "message": str(e)}}), 400

    bbox = data["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox must be [x, y, w, h]"}}), 400

    x, y, w, h = bbox
    iw, ih = img.size
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox out of image bounds"}}), 400

    crop = img.crop((x, y, x + w, y + h))
    limit = data.get("limit", 5)
    result = clf.classify(crop, top_k=limit)

    def band(sim):
        if sim >= 0.60:
            return "high"
        if sim >= 0.35:
            return "moderate"
        return "low"

    results = []
    for rank, item in enumerate(result.get("top_k", []), 1):
        results.append({
            "rank": rank,
            "match_type": "class_prototype",
            "class_name": item["class_name"],
            "class_label": item.get("class_label"),
            "similarity": item["confidence"],
            "band": band(item["confidence"]),
            "asset": None,
        })

    return jsonify({
        "query": {"bbox": bbox, "mode": "prototype"},
        "best_match": {
            "class_name": result["class_name"],
            "similarity": result["confidence"],
            "rejected": result["rejected"],
        },
        "results": results,
    })


@app.route("/trust", methods=["POST"])
def trust():
    data = request.get_json()
    if not data or "image_base64" not in data or "bbox" not in data:
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}}), 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": {"code": "INVALID_IMAGE", "message": str(e)}}), 400

    bbox = data["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox must be [x, y, w, h]"}}), 400

    x, y, w, h = bbox
    iw, ih = img.size
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox out of image bounds"}}), 400

    crop = img.crop((x, y, x + w, y + h))
    top_k = data.get("top_k", 10)
    result = clf.classify(crop, top_k=top_k)

    predicted_class = data.get("predicted_class", result["class_name"])
    top_k_list = result.get("top_k", [])

    predicted_rank = None
    predicted_sim = None
    for rank, item in enumerate(top_k_list, 1):
        if item["class_name"] == predicted_class:
            predicted_rank = rank
            predicted_sim = item["confidence"]
            break

    if predicted_rank is None:
        predicted_rank = -1
        predicted_sim = 0.0

    margin = 0.0
    if len(top_k_list) >= 2:
        margin = top_k_list[0]["confidence"] - top_k_list[1]["confidence"]

    entropy = 0.0
    if top_k_list:
        total = sum(item["confidence"] for item in top_k_list)
        if total > 0:
            probs = [item["confidence"] / total for item in top_k_list]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)

    return jsonify({
        "query": {"bbox": bbox, "predicted_class": predicted_class},
        "trust": {
            "predicted_class_rank": predicted_rank,
            "predicted_class_similarity": predicted_sim,
            "top1_class": result["class_name"],
            "top1_similarity": result["confidence"],
            "margin_to_second": round(margin, 4),
            "above_rejection_threshold": result["confidence"] >= 0.35,
            "rejection_threshold": 0.35,
            "ambiguous": margin < 0.05,
            "entropy": round(entropy, 4),
            "top_k": top_k_list,
        },
    })


@app.route("/sample-image", methods=["GET"])
def sample_image():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Image not found"}), 404

    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(DATA_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Image not found"}), 404

    if not os.path.exists(resolved_path):
        return jsonify({"error": "Image not found"}), 404

    return send_file(resolved_path)


@app.route("/similar-samples", methods=["POST"])
def similar_samples():
    data = request.get_json()
    if not data or "image_base64" not in data or "bbox" not in data:
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}}), 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": {"code": "INVALID_IMAGE", "message": str(e)}}), 400

    bbox = data["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox must be [x, y, w, h]"}}), 400

    x, y, w, h = bbox
    iw, ih = img.size
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
        return jsonify({"error": {"code": "INVALID_BBOX", "message": "bbox out of image bounds"}}), 400

    crop = img.crop((x, y, x + w, y + h))
    result = clf.classify(crop, top_k=3)
    predicted_class = result["class_name"]

    limit = data.get("limit", 4)
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "limit must be an integer"}}), 400

    samples = SAMPLE_INDEX.get(predicted_class, [])[:limit]
    exemplars = [
        {
            "image_url": f"/sample-image?path={sample['path']}",
            "class_name": sample["class_name"],
            "source": "sample",
        }
        for sample in samples
    ]

    return jsonify({
        "query": {"bbox": bbox, "predicted_class": predicted_class},
        "exemplars": exemplars,
        "has_samples": len(exemplars) > 0,
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    config_path = Path(__file__).resolve().parents[1] / "codex_model" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return jsonify({"num_classes": config["num_classes"], "class_names": config["class_names"]})


if __name__ == "__main__":
    print(f"Codex Classifier API - http://{HOST}:{PORT}")
    print("  POST /classify, /classify-batch, /segment, /similar, /trust")
    print("  GET  /classes")
    app.run(host=HOST, port=PORT, debug=False)

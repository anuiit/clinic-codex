from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from backend.app.errors import ApiError
from backend.services.approved_corrections import image_pixel_sha256, overlaps_correction
from PIL import Image
from backend.app.services.image_service import InvalidImageError, decode_uploaded_image, invalid_image_payload

bp = Blueprint("segment", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _invalid_image_response():
    return jsonify(invalid_image_payload()), 400


def _image_decode_kwargs():
    settings = current_app.config["CLINIC_SETTINGS"]
    return {
        "max_pixels": settings.max_image_pixels,
        "max_dimension": settings.max_image_dimension,
    }


@bp.post("/segment")
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    try:
        img = np.array(decode_uploaded_image(request.files["image"], **_image_decode_kwargs()))
    except InvalidImageError:
        return _invalid_image_response()
    h, w = img.shape[:2]

    use_corrections = request.form.get("use_corrections", "true")
    if use_corrections not in {"true", "false"}:
        raise ApiError(code="INVALID_CORRECTION_MODE", message="use_corrections must be true or false", status_code=400)
    identity = image_pixel_sha256(Image.fromarray(img))
    memory = {"elements": [], "conflicts": [], "image_pixel_sha256": identity}
    lookup = getattr(_services(), "approved_page_corrections", None)
    if use_corrections == "true" and lookup is not None:
        memory = lookup(img)
    if memory["conflicts"]:
        raise ApiError(code="APPROVED_CORRECTION_CONFLICT",
                       message="Cette planche possède des corrections approuvées incompatibles. Une revue est nécessaire.",
                       status_code=409)
    corrections = memory["elements"]
    valid_proposals = []
    crops = []
    for proposal in _services().segment_page(img):
        if proposal.crop is None or overlaps_correction(proposal.bbox, corrections):
            continue

        valid_proposals.append(proposal)
        crops.append(proposal.crop)

    elements = list(corrections)
    models = {}
    if crops:
        results = _services().classify_batch(crops)
        if len(results) != len(valid_proposals):
            raise ApiError(
                code="SEGMENT_CLASSIFICATION_MISMATCH",
                message="segment classifier result count did not match proposal count",
                status_code=500,
            )
        for proposal, result in zip(valid_proposals, results):
            x, y, bw, bh = proposal.bbox
            elements.append({"bbox": [int(x), int(y), int(bw), int(bh)], "source": "model", **result})
            model = result.get("model_provenance")
            if model:
                models[model["artifact_id"]] = model

    segmenter_identity = getattr(_services(), "segmenter_provenance", lambda: None)()
    return jsonify({"num_elements": len(elements), "image_size": [int(w), int(h)], "elements": elements,
                    "provenance": {"image_pixel_sha256": identity, "correction_count": len(corrections),
                                   "corrections_enabled": use_corrections == "true",
                                   "classifier_models": list(models.values()), "segmenter": segmenter_identity}})

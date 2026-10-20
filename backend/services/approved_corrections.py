"""Replay reviewed regions only when the canonical page pixels match exactly."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .annotation_storage import sanitize_class_name


def image_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def stored_image_digest(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            return image_pixel_sha256(image)
    except (OSError, ValueError):
        return None


def approved_page_corrections(store, image: Image.Image) -> dict:
    """Do not treat partial reviewed sets as exhaustive page ground truth."""
    image = image.convert("RGB")
    identity = image_pixel_sha256(image)
    corrections = {}
    conflicts = set()
    # ponytail: linear scan; add an invalidated page index if the review store grows.
    # The SQLite review store is transactional; a decision approved between the two
    # reads below is simply picked up on the next request.
    decisions = store._load_manifest()["decisions"]
    for analysis in store.list_queue()["analyses"]:
        approved = [e for e in analysis["elements"] if e["review_status"] == "approved"
                    and not e["stale_decision"] and e["crop_exists"]]
        if not approved or not analysis["image_exists"]:
            continue
        try:
            with Image.open(analysis["image_path"]) as page:
                if page.size != image.size or image_pixel_sha256(page) != identity:
                    continue
        except (OSError, ValueError):
            continue
        for element in approved:
            decision = decisions.get(element["key"], {})
            if decision.get("source_image_pixel_sha256") not in (None, identity):
                continue
            bbox = element["bbox"]
            if (len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in bbox)):
                continue
            x, y, w, h = bbox
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.width or y+h > image.height:
                continue
            try:
                name = sanitize_class_name(element["class_name"])
                # Legacy reviews bind crop metadata, so also verify actual pixels.
                with Image.open(element["crop_path"]) as crop:
                    if image_pixel_sha256(crop) != image_pixel_sha256(image.crop((x, y, x+w, y+h))):
                        continue
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            key = tuple(bbox)
            previous = corrections.get(key)
            if previous and previous["class_name"] != name:
                conflicts.add(key)
            item = {"bbox": list(bbox), "class_name": name, "class_label": None,
                    "confidence": 1.0, "confidence_kind": "human_approval",
                    "rejected": False, "top_k": [], "source": "approved_correction",
                    "review": {"decision_id": element["key"],
                               "reviewed_at": decision.get("reviewed_at"),
                               "image_pixel_sha256": identity}}
            if previous is None or (decision.get("reviewed_at") or "") > (previous["review"]["reviewed_at"] or ""):
                corrections[key] = item
    return {"image_pixel_sha256": identity,
            "elements": [] if conflicts else list(corrections.values()),
            "conflicts": [list(b) for b in sorted(conflicts)]}


def overlaps_correction(bbox, corrections) -> bool:
    """Suppress a model proposal only if it matches a reviewed box at IoU >= .5."""
    x, y, w, h = bbox
    for item in corrections:
        a, b, c, d = item["bbox"]
        intersection = max(0, min(x+w, a+c)-max(x, a)) * max(0, min(y+h, b+d)-max(y, b))
        union = w*h + c*d - intersection
        if union > 0 and intersection / union >= 0.5:
            return True
    return False

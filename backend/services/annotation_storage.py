from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sanitize_class_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("class_name is empty after strip")
    if "/" in name:
        raise ValueError("class_name contains '/'")
    if "\\" in name:
        raise ValueError("class_name contains '\\'")
    if ".." in name:
        raise ValueError("class_name contains '..'")
    if "\x00" in name:
        raise ValueError("class_name contains null byte")
    return name


def decode_image_data_url(data_url: str) -> Image.Image:
    if data_url.startswith("data:"):
        comma = data_url.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma found")
        b64 = data_url[comma + 1:]
    else:
        b64 = data_url
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"PIL could not open image: {exc}") from exc


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0:
        raise ValueError(f"clamped width is {w} <= 0")
    if h <= 0:
        raise ValueError(f"clamped height is {h} <= 0")
    return (x, y, w, h)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def save_annotation(
    analysis_id: str,
    image: Image.Image,
    annotations: list[dict],
    base_dir: Path,
    elements_dir: Path,
) -> dict:
    if not _SAFE_ID.match(analysis_id):
        raise ValueError(
            f"analysis_id '{analysis_id}' must be alphanumeric/dash/underscore only"
        )

    target_dir = base_dir / analysis_id

    if target_dir.exists():
        target_abs = target_dir.resolve()
        for symlink in elements_dir.rglob("*"):
            if symlink.is_symlink():
                try:
                    link_target = Path(os.readlink(symlink))
                    if not link_target.is_absolute():
                        link_target = (symlink.parent / link_target).resolve()
                    if str(link_target).startswith(str(target_abs)):
                        symlink.unlink()
                except OSError:
                    pass
        shutil.rmtree(target_dir)

    (target_dir / "elements").mkdir(parents=True)

    img_w, img_h = image.size
    image.save(target_dir / "image.png", format="PNG")

    saved_annotations = []
    classes_seen: set[str] = set()

    for ann in annotations:
        idx = ann["index"]
        raw_class = ann["class_name"]
        bbox_raw = ann["bbox"]

        cls = sanitize_class_name(raw_class)
        x, y, w, h = clamp_bbox(tuple(bbox_raw), img_w, img_h)
        crop = image.crop((x, y, x + w, y + h))
        crop_filename = f"{idx}.png"
        crop.save(target_dir / "elements" / crop_filename, format="PNG")

        sym_dir = elements_dir / cls
        sym_dir.mkdir(parents=True, exist_ok=True)
        sym_path = sym_dir / f"{analysis_id}_el{idx}.png"
        sym_target_abs = target_dir / "elements" / crop_filename
        rel_target = os.path.relpath(sym_target_abs, sym_path.parent)
        if sym_path.exists() or sym_path.is_symlink():
            sym_path.unlink()
        sym_path.symlink_to(rel_target)

        classes_seen.add(cls)
        saved_annotations.append(
            {
                "index": idx,
                "class_name": cls,
                "bbox": [x, y, w, h],
                "crop_path": str(target_dir / "elements" / crop_filename),
            }
        )

    metadata = {
        "analysis_id": analysis_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "annotations": saved_annotations,
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "status": "ok",
        "analysis_id": analysis_id,
        "saved_count": len(saved_annotations),
        "classes": sorted(classes_seen),
    }

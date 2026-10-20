"""Single source of truth: backend/annotations/<analysis_id>/. No writes to training_data/."""
from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import os
import re
import shutil
import uuid
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from PIL import Image

try:
    from backend.services.image_limits import ImageSizeLimitError, ensure_image_within_limits
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from services.image_limits import ImageSizeLimitError, ensure_image_within_limits  # type: ignore


class AnnotationStorageError(Exception):
    """Base class for annotation storage errors."""


class AnnotationPermissionError(AnnotationStorageError, PermissionError):
    """Raised when the annotations directory is not writable."""


class AnnotationDiskFullError(AnnotationStorageError, OSError):
    """Raised when the disk has no space left."""


class AnnotationConflictError(AnnotationStorageError):
    """An analysis ID already belongs to a different submission."""


def sanitize_image_name(name: str | None) -> str | None:
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("image_name must be a string")
    clean = unicodedata.normalize("NFC", name.replace("\\", "/").rsplit("/", 1)[-1].strip())
    if not clean or any(ord(char) < 32 for char in clean) or len(clean) > 255:
        raise ValueError("image_name must be a filename of at most 255 characters")
    return clean


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


NOTE_MAX_LENGTH = 2000


def sanitize_note(note) -> str | None:
    """Research note attached to an element. Free text, never exported to the
    training dataset (see scripts/export_approved_annotations.py)."""
    if note is None:
        return None
    if not isinstance(note, str):
        raise ValueError("note must be a string")
    note = note.strip()
    if not note:
        return None
    if "\x00" in note:
        raise ValueError("note contains null byte")
    if len(note) > NOTE_MAX_LENGTH:
        raise ValueError(f"note exceeds {NOTE_MAX_LENGTH} characters")
    return note


def decode_image_data_url(
    data_url: str,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> Image.Image:
    if data_url.startswith("data:"):
        comma = data_url.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma found")
        b64 = data_url[comma + 1:]
    else:
        b64 = data_url
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}") from exc
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            ensure_image_within_limits(probe, max_pixels=max_pixels, max_dimension=max_dimension)
            probe.verify()
        with Image.open(io.BytesIO(raw)) as image:
            ensure_image_within_limits(image, max_pixels=max_pixels, max_dimension=max_dimension)
            rgb = image.convert("RGB")
            rgb.load()
            return rgb
    except ImageSizeLimitError as exc:
        raise ValueError(str(exc)) from exc
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


def normalize_bbox_to_int_pixels(bbox: Sequence[int | float]) -> tuple[int, int, int, int]:
    """Normalize frontend bbox values to integer pixels before crop/save.

    Phase 0 intentionally uses Python's documented ``round()`` behavior for
    every coordinate and dimension, including banker-rounding ties.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, w, h]")
    x, y, w, h = bbox
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def save_annotation(
    analysis_id: str,
    image: Image.Image,
    annotations: list[dict],
    base_dir: Path,
    elements_dir: Path,  # kept for backward compat — not used
    author_id: str | None = None,
    image_name: str | None = None,
) -> dict:
    if not _SAFE_ID.match(analysis_id):
        raise ValueError(
            f"analysis_id '{analysis_id}' must be alphanumeric/dash/underscore only"
        )

    image_name = sanitize_image_name(image_name)
    base_dir = Path(base_dir).resolve()
    target_dir = base_dir / analysis_id
    tmp_dir = base_dir / f".tmp-{analysis_id}-{os.getpid()}-{uuid.uuid4().hex}"

    try:
        (tmp_dir / "elements").mkdir(parents=True)

        img_w, img_h = image.size
        image.save(tmp_dir / "image.png", format="PNG")

        saved_annotations = []
        classes_seen: set[str] = set()

        for ann in annotations:
            idx = ann["index"]
            raw_class = ann["class_name"]
            bbox_raw = ann["bbox"]

            cls = sanitize_class_name(raw_class)
            normalized_bbox = normalize_bbox_to_int_pixels(bbox_raw)
            x, y, w, h = clamp_bbox(normalized_bbox, img_w, img_h)
            crop = image.crop((x, y, x + w, y + h))
            crop_filename = f"{idx}.png"
            crop.save(tmp_dir / "elements" / crop_filename, format="PNG")

            classes_seen.add(cls)
            saved = {
                "index": idx,
                "class_name": cls,
                "bbox": [x, y, w, h],
                "crop_path": str(target_dir / "elements" / crop_filename),
            }
            note = sanitize_note(ann.get("note"))
            if note is not None:
                saved["note"] = note
            saved_annotations.append(saved)

        metadata = {
            "analysis_id": analysis_id,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "submitted_by": author_id,
            "image_name": image_name,
            "annotations": saved_annotations,
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        if target_dir.is_symlink():
            raise AnnotationConflictError("analysis_id is not a regular submission directory")
        if target_dir.exists():
            _assert_same_submission(target_dir, metadata, tmp_dir / "image.png")
        else:
            try:
                os.rename(tmp_dir, target_dir)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                _assert_same_submission(target_dir, metadata, tmp_dir / "image.png")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    except PermissionError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise AnnotationPermissionError(str(e)) from e
    except OSError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if e.errno == errno.ENOSPC:
            raise AnnotationDiskFullError(str(e)) from e
        raise
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return {
        "status": "ok",
        "analysis_id": analysis_id,
        "saved_count": len(saved_annotations),
        "classes": sorted(classes_seen),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def _assert_same_submission(target_dir: Path, incoming: dict, incoming_image: Path) -> None:
    try:
        existing = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        old_image = target_dir / "image.png"
        with Image.open(old_image) as probe:
            old = probe.convert("RGB")
            with Image.open(incoming_image) as candidate:
                new = candidate.convert("RGB")
                same_image = old.size == new.size and hashlib.sha256(old.tobytes()).digest() == hashlib.sha256(new.tobytes()).digest()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AnnotationConflictError("existing submission is unreadable; refusing to replace it") from exc
    if existing.get("submitted_by") != incoming["submitted_by"]:
        raise AnnotationConflictError("analysis_id already belongs to another submitter")
    def fields(metadata: dict) -> list[dict]:
        return [
            {key: item.get(key) for key in ("index", "class_name", "bbox", "note")}
            for item in metadata.get("annotations", [])
        ]
    if not same_image or fields(existing) != fields(incoming) or existing.get("image_name") != incoming["image_name"]:
        raise AnnotationConflictError("analysis_id already has different submitted content")

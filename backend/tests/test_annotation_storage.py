# type: ignore
# pyright: reportMissingImports=false
"""Tests for backend/services/annotation_storage.py."""
import errno
import base64
import json
import os
import sys
import types
from pathlib import Path

import pytest

def _stub_pil():
    try:
        import PIL  # noqa: F401
    except ImportError:
        pil_mod = types.ModuleType("PIL")
        class _Image:
            @staticmethod
            def open(fp):
                return _Image()
        setattr(pil_mod, "Image", _Image)
        sys.modules["PIL"] = pil_mod
        sys.modules["PIL.Image"] = types.ModuleType("PIL.Image")


_stub_pil()

_BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from services.annotation_storage import (  # noqa: E402
    AnnotationDiskFullError,
    AnnotationConflictError,
    AnnotationPermissionError,
    decode_image_data_url,
    sanitize_note,
    save_annotation,
)

def _make_image():
    """Return a minimal real PIL Image (10×10 white PNG)."""
    from PIL import Image
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


def _make_annotations():
    return [
        {"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]},
        {"index": 1, "class_name": "calli", "bbox": [2, 2, 4, 4]},
    ]


def test_save_annotation_happy_path(tmp_path):
    """save_annotation writes image.png, elements/0.png, elements/1.png, metadata.json."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "test-abc123"

    result = save_annotation(
        analysis_id,
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    assert result["status"] == "ok"
    assert result["analysis_id"] == analysis_id
    assert result["saved_count"] == 2

    ann_dir = base_dir / analysis_id
    assert (ann_dir / "image.png").is_file()
    assert (ann_dir / "elements" / "0.png").is_file()
    assert (ann_dir / "elements" / "1.png").is_file()

    meta = json.loads((ann_dir / "metadata.json").read_text())
    assert meta["analysis_id"] == analysis_id
    assert len(meta["annotations"]) == 2


def test_no_writes_under_training_data(tmp_path):
    """save_annotation must NOT write anything under training_data/."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    training_data = tmp_path / "training_data"
    training_data.mkdir()

    save_annotation(
        "no-training-write",
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=training_data / "Elements",
    )

    for root, dirs, files in os.walk(training_data):
        for name in files:
            pytest.fail(f"Unexpected file under training_data/: {os.path.join(root, name)}")
        for name in dirs:
            p = os.path.join(root, name)
            if os.path.islink(p):
                pytest.fail(f"Unexpected symlink under training_data/: {p}")


@pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 not meaningful on Windows")
def test_permission_error_raises_annotation_permission_error(tmp_path):
    """chmod 000 on base_dir → AnnotationPermissionError."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    base_dir.chmod(0o000)
    try:
        with pytest.raises(AnnotationPermissionError):
            save_annotation(
                "perm-test",
                _make_image(),
                _make_annotations(),
                base_dir=base_dir,
                elements_dir=tmp_path / "training_data" / "Elements",
            )
    finally:
        base_dir.chmod(0o755)


def test_enospc_raises_annotation_disk_full_error(tmp_path, monkeypatch):
    """Monkeypatched OSError(ENOSPC) → AnnotationDiskFullError."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()

    import services.annotation_storage as _mod

    def _raise_enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(_mod.Path, "mkdir", lambda *a, **kw: _raise_enospc())

    with pytest.raises(AnnotationDiskFullError):
        save_annotation(
            "disk-full-test",
            _make_image(),
            _make_annotations(),
            base_dir=base_dir,
            elements_dir=tmp_path / "training_data" / "Elements",
        )


def test_resave_same_analysis_id_is_idempotent_or_conflicts(tmp_path):
    """A submitted image and its annotations remain immutable."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "overwrite-test"

    first = save_annotation(
        analysis_id,
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    retry = save_annotation(analysis_id, _make_image(), _make_annotations(), base_dir, base_dir)
    assert retry["saved_count"] == first["saved_count"]
    with pytest.raises(AnnotationConflictError):
        save_annotation(analysis_id, _make_image(), [{"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]}], base_dir, base_dir)

    ann_dir = base_dir / analysis_id
    meta = json.loads((ann_dir / "metadata.json").read_text())
    assert len(meta["annotations"]) == 2
    assert (ann_dir / "elements" / "1.png").exists()


def test_save_annotation_rounds_float_bbox_before_crop_and_metadata(tmp_path):
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "float-bbox-test"

    save_annotation(
        analysis_id,
        _make_image(),
        [{"index": 0, "class_name": "atl", "bbox": [1.2, 2.6, 3.4, 4.6]}],
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    ann_dir = base_dir / analysis_id
    meta = json.loads((ann_dir / "metadata.json").read_text())
    assert meta["annotations"][0]["bbox"] == [1, 3, 3, 5]

    from PIL import Image

    with Image.open(ann_dir / "elements" / "0.png") as crop:
        assert crop.size == (3, 5)


def test_decode_image_data_url_uses_strict_base64_validation():
    valid_prefix = base64.b64encode(b"not an image").decode("ascii")
    invalid = valid_prefix[:-2] + "$$"

    with pytest.raises(ValueError, match="base64 decode failed"):
        decode_image_data_url(invalid)


def test_resave_same_analysis_id_preserves_original_directory(tmp_path):
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "unique-temp-test"

    save_annotation(
        analysis_id,
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )
    before = (base_dir / analysis_id / "metadata.json").read_bytes()
    save_annotation(
        analysis_id,
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    assert (base_dir / analysis_id / "metadata.json").read_bytes() == before
    assert not list(base_dir.glob(".tmp-*"))


def test_sanitize_note_validation():
    assert sanitize_note(None) is None
    assert sanitize_note("") is None
    assert sanitize_note("   ") is None
    assert sanitize_note("  lecture incertaine  ") == "lecture incertaine"
    with pytest.raises(ValueError, match="must be a string"):
        sanitize_note(42)
    with pytest.raises(ValueError, match="null byte"):
        sanitize_note("bad\x00note")
    with pytest.raises(ValueError, match="exceeds"):
        sanitize_note("x" * 2001)


def test_save_annotation_persists_note_and_omits_absent_note(tmp_path):
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "test-note1"

    annotations = [
        {"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5], "note": "contour partiel"},
        {"index": 1, "class_name": "calli", "bbox": [2, 2, 4, 4]},
    ]
    result = save_annotation(
        analysis_id,
        _make_image(),
        annotations,
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )
    assert result["status"] == "ok"

    metadata = json.loads((base_dir / analysis_id / "metadata.json").read_text())
    by_index = {ann["index"]: ann for ann in metadata["annotations"]}
    assert by_index[0]["note"] == "contour partiel"
    assert "note" not in by_index[1]


def test_save_annotation_rejects_invalid_note(tmp_path):
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()

    with pytest.raises(ValueError, match="note must be a string"):
        save_annotation(
            "test-note2",
            _make_image(),
            [{"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5], "note": 7}],
            base_dir=base_dir,
            elements_dir=tmp_path / "training_data" / "Elements",
        )
    # failed save must not leave the target directory behind
    assert not (base_dir / "test-note2").exists()


def test_submission_owner_and_original_filename_are_stable(tmp_path):
    root = tmp_path / "annotations"
    image = _make_image()
    rows = _make_annotations()
    save_annotation("owned", image, rows, root, root, author_id="alice", image_name=r"C:\pages\épreuve 1.png")
    original = (root / "owned" / "metadata.json").read_bytes()
    save_annotation("owned", image, rows, root, root, author_id="alice", image_name="épreuve 1.png")
    assert (root / "owned" / "metadata.json").read_bytes() == original
    assert json.loads(original)["image_name"] == "épreuve 1.png"
    with pytest.raises(AnnotationConflictError):
        save_annotation("owned", image, rows, root, root, author_id="bob", image_name="épreuve 1.png")
    assert (root / "owned" / "metadata.json").read_bytes() == original

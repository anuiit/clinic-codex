# type: ignore
# pyright: reportMissingImports=false
"""Tests for backend/services/annotation_storage.py."""
import errno
import json
import os
import sys
import types

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
    AnnotationPermissionError,
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


def test_resave_same_analysis_id_overwrites(tmp_path):
    """Re-saving the same analysis_id replaces the previous data atomically."""
    base_dir = tmp_path / "annotations"
    base_dir.mkdir()
    analysis_id = "overwrite-test"

    save_annotation(
        analysis_id,
        _make_image(),
        _make_annotations(),
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    save_annotation(
        analysis_id,
        _make_image(),
        [{"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]}],
        base_dir=base_dir,
        elements_dir=tmp_path / "training_data" / "Elements",
    )

    ann_dir = base_dir / analysis_id
    meta = json.loads((ann_dir / "metadata.json").read_text())
    assert len(meta["annotations"]) == 1
    assert not (ann_dir / "elements" / "1.png").exists()

# type: ignore
# pyright: reportMissingImports=false
"""Tests for the /save-annotation Flask endpoint."""
import base64
import importlib
import io
import json
import sys
import types

import pytest

def _insert_stubs():
    mod = types.ModuleType("codex_model")

    class CodexClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify(self, *args, **kwargs):
            return {"class_name": "stub", "confidence": 1.0, "rejected": False, "top_k": []}

        def classify_batch(self, *args, **kwargs):
            return []

    setattr(mod, "CodexClassifier", CodexClassifier)
    sys.modules["codex_model"] = mod

    pkg = types.ModuleType("codex_pipeline")
    seg_mod = types.ModuleType("codex_pipeline.segmentation")

    class MobileSAMSegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment_page(self, img):
            return []

        def extract_crops(self, img, proposals):
            return []

    setattr(seg_mod, "MobileSAMSegmenter", MobileSAMSegmenter)
    sys.modules["codex_pipeline"] = pkg
    sys.modules["codex_pipeline.segmentation"] = seg_mod

    try:
        import PIL  # noqa: F401
    except Exception:
        pil_mod = types.ModuleType("PIL")
        setattr(pil_mod, "Image", types.SimpleNamespace(open=lambda *a, **k: None))
        sys.modules["PIL"] = pil_mod


_insert_stubs()

import os

_BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

flask_mod = importlib.import_module("examples.flask_api")
app = flask_mod.app

def _png_data_url():
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), color=(200, 200, 200)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _valid_payload():
    return {
        "analysis_id": "test-endpoint-001",
        "image_data_url": _png_data_url(),
        "annotations": [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 5, 5]},
        ],
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(flask_mod, "BACKEND_ROOT", tmp_path)
    (tmp_path / "annotations").mkdir()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_save_annotation_happy_path(client):
    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["analysis_id"] == "test-endpoint-001"


def test_save_annotation_permission_denied(client, monkeypatch):
    from services.annotation_storage import AnnotationPermissionError

    monkeypatch.setattr(flask_mod, "save_annotation", lambda *a, **kw: (_ for _ in ()).throw(AnnotationPermissionError("denied")))

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error_code"] == "PERMISSION_DENIED"
    assert "Droits" in body["message"]


def test_save_annotation_disk_full(client, monkeypatch):
    from services.annotation_storage import AnnotationDiskFullError

    monkeypatch.setattr(flask_mod, "save_annotation", lambda *a, **kw: (_ for _ in ()).throw(AnnotationDiskFullError("no space")))

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 507
    body = resp.get_json()
    assert body["error_code"] == "DISK_FULL"


def test_save_annotation_unknown_error(client, monkeypatch):
    monkeypatch.setattr(flask_mod, "save_annotation", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    resp = client.post(
        "/save-annotation",
        data=json.dumps(_valid_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["error_code"] == "INTERNAL"
    assert "id=" in body["message"]

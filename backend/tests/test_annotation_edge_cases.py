"""Persistence edge cases exercised on both Windows and Linux."""
import json
from pathlib import Path

import pytest
from PIL import Image

from backend.services.annotation_storage import save_annotation
from backend.services.annotation_review import AnnotationReviewStore, AnnotationReviewValidationError


def test_one_crop_only_fingerprints_the_requested_region(tmp_path, monkeypatch):
    from backend.services import annotation_review as module
    store = AnnotationReviewStore(tmp_path / "annotations")
    save_annotation("page", Image.new("RGB", (20, 20), "red"),
                    [{"index": i, "class_name": "atl", "bbox": [0, 0, 20, 20]} for i in range(50)],
                    base_dir=store.annotations_dir, elements_dir=tmp_path / "unused")
    original = module._source_fingerprint
    calls = []
    def counted(**kwargs):
        calls.append(kwargs["annotation"]["index"])
        return original(**kwargs)
    monkeypatch.setattr(module, "_source_fingerprint", counted)
    assert store.crop_path_for("page", 40).is_file()
    assert calls == [40]


def test_relative_root_modify_and_restore_keep_valid_crops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = AnnotationReviewStore(Path("annotations"))
    save_annotation("page", Image.new("RGB", (20, 20), "red"),
                    [{"index": 0, "class_name": "atl", "bbox": [0, 0, 20, 20]}],
                    base_dir=Path("annotations"), elements_dir=Path("unused"))
    store.set_status("page", 0, "approved")
    changed = store.modify_element("page", 0, class_name="calli", bbox=[2, 2, 10, 10], status="approved",
                                   expected_revision=1)
    assert changed["element"]["trainable"]
    assert store.crop_path_for("page", 0).is_file()
    restored = store.restore_element("page", 0, target_revision=0, expected_revision=2)
    assert restored["element"]["review_status"] == "pending"
    assert restored["element"]["class_name"] == "atl"
    assert store.crop_path_for("page", 0).is_file()


@pytest.mark.parametrize("bad", [{"class_name": 123}, {"bbox": [0, 0, -1, 10]},
                                {"bbox": [False, 0, 10, 10]}, {"crop_path": "../outside.png"},
                                {"source_fingerprint": "not-a-sha"}])
def test_legacy_import_refuses_invalid_decisions_without_creating_db(tmp_path, bad):
    decision = {"analysis_id": "page", "index": 0, "status": "approved", "class_name": "atl",
                "bbox": [0, 0, 10, 10], "source_fingerprint": "a" * 64,
                "reviewed_at": "2026-09-17T00:00:00Z", **bad}
    store = AnnotationReviewStore(tmp_path)
    store.manifest_path.write_text(json.dumps({"schema_version": 1, "decisions": {"page:0": decision}}))
    with pytest.raises(AnnotationReviewValidationError, match="page:0"):
        store.import_legacy_manifest()
    assert not store.db_path.exists()

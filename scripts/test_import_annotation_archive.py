from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend.services.annotation_review import AnnotationReviewStore
from scripts.import_annotation_archive import import_annotation_archive
from scripts.export_approved_annotations import export_approved_annotations


def _png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _archive(tmp_path, *, changed_crop=False, changed_decision=False, extra=None, split="train", float_bbox=False):
    page = Image.new("RGB", (8, 8), (10, 20, 30))
    annotations = []
    decisions = {}
    files = {"annotations/page-1/image.png": _png(page)}
    for index, (name, status) in enumerate([
        ("atl", "approved"), ("calli", "approved"),
        ("unknown", "approved"), ("atl", "rejected"),
    ]):
        bbox = [0.2, 0.2, 3.1, 3.1] if float_bbox and index == 0 else [index, index, 3, 3]
        basename = f"{index}-modified.png"
        annotation = {"index": index, "class_name": name, "bbox": bbox,
                      "crop_path": f"F:\\clinic-codex\\backend\\annotations\\page-1\\elements\\{basename}"}
        annotations.append(annotation)
        crop = page.crop((index, index, index+3, index+3))
        if changed_crop and index == 0:
            crop = Image.new("RGB", crop.size, (200, 0, 0))
        files[f"annotations/page-1/elements/{basename}"] = _png(crop)
        decision = {"analysis_id": "page-1", "index": index, "class_name": name,
                    "bbox": bbox, "status": status, "source_fingerprint": "unverifiable-on-another-machine"}
        if index == 0:
            decision["dataset_split"] = split
        if changed_decision and index == 0:
            decision["class_name"] = "calli"
        decisions[f"page-1:{index}"] = decision
    files["annotations/page-1/metadata.json"] = json.dumps({
        "analysis_id": "page-1", "uploaded_at": "2026-09-01T10:00:00+00:00",
        "annotations": annotations,
    }).encode()
    files["annotations/review-index.json"] = json.dumps({"schema_version": 1, "decisions": decisions}).encode()
    path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        if extra:
            z.writestr(*extra)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"class_names": ["calli", "atl"]}))
    return path, config


def test_import_preview_never_creates_or_reapproves_store(tmp_path):
    archive, config = _archive(tmp_path)
    output = tmp_path / "imported"
    report = import_annotation_archive(archive, output, config)
    assert not output.exists()
    assert report["retained_count"] == 0
    assert report["excluded_counts"] == {"needs_reattestation": 2, "unknown_class": 1, "rejected_review": 1}
    assert report["legacy_fingerprint_verified"] is False


def test_verified_import_is_traceable_and_uses_actual_crop_filename(tmp_path):
    archive, config = _archive(tmp_path)
    output = tmp_path / "imported"
    report = import_annotation_archive(archive, output, config, apply=True, re_attest_verified=True)
    assert report["annotation_count"] == 4
    assert report["retained_count"] == 2
    assert report["fallback_split_count"] == 1
    rows = list(AnnotationReviewStore(output).iter_approved_annotations())
    assert {row["index"] for row in rows} == {0, 1}
    assert next(row for row in rows if row["index"] == 0)["dataset_split"] == "train"
    assert (output / "page-1/elements/0.png").is_file()
    local = json.loads((output / "review-index.json").read_text())["decisions"]
    assert local["page-1:0"]["archive_decision"]["source_fingerprint"] == "unverifiable-on-another-machine"
    assert local["page-1:0"]["archive_sha256"] == report["archive_sha256"]
    assert local["page-1:0"]["source_fingerprint"] != "unverifiable-on-another-machine"
    assert local["page-1:3"]["status"] == "rejected"
    assert "page-1:2" not in local
    exported = export_approved_annotations(output, tmp_path / "Elements", runtime_config=config)
    assert exported["exported_count"] == 2
    assert exported["excluded_counts"] == {"unknown_class": 1, "rejected_review": 1}
    before = (output / "review-index.json").read_bytes()
    with pytest.raises(FileExistsError):
        import_annotation_archive(archive, output, config, apply=True, re_attest_verified=True)
    assert (output / "review-index.json").read_bytes() == before
    (output / "page-1/elements/0.png").unlink()
    assert {row["index"] for row in AnnotationReviewStore(output).iter_approved_annotations()} == {1}


@pytest.mark.parametrize("options,reason", [
    ({"changed_crop": True}, "crop_source_mismatch"),
    ({"changed_decision": True}, "decision_metadata_mismatch"),
    ({"split": "excluded"}, "explicit_exclusion"),
    ({"split": "invalid"}, "invalid_dataset_split"),
])
def test_unverified_or_excluded_rows_never_enter_export(tmp_path, options, reason):
    archive, config = _archive(tmp_path, **options)
    output = tmp_path / "imported"
    report = import_annotation_archive(archive, output, config, apply=True, re_attest_verified=True)
    assert report["rows"][0]["exclusion_reason"] == reason
    assert {row["index"] for row in AnnotationReviewStore(output).iter_approved_annotations()} == {1}


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "annotations\\escape"])
def test_unsafe_zip_member_rejected_before_writing(tmp_path, name):
    archive, config = _archive(tmp_path, extra=(name, b"bad"))
    output = tmp_path / "imported"
    with pytest.raises(ValueError, match="unsafe ZIP"):
        import_annotation_archive(archive, output, config, apply=True, re_attest_verified=True)
    assert not output.exists()


def test_symlink_zip_member_rejected_before_writing(tmp_path):
    info = zipfile.ZipInfo("annotations/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive, config = _archive(tmp_path, extra=(info, b"../../outside"))
    with pytest.raises(ValueError, match="unsafe ZIP"):
        import_annotation_archive(archive, tmp_path / "imported", config, apply=True)


def test_valid_float_bbox_is_preserved_without_reinterpreting_geometry(tmp_path):
    archive, config = _archive(tmp_path, float_bbox=True)
    output = tmp_path / "imported"
    report = import_annotation_archive(archive, output, config, apply=True, re_attest_verified=True)
    assert report["retained_count"] == 2
    assert report["rows"][0]["bbox"] == [0.2, 0.2, 3.1, 3.1]
    assert next(AnnotationReviewStore(output).iter_approved_annotations())["bbox"] == [0.2, 0.2, 3.1, 3.1]

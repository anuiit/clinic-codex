# type: ignore
# pyright: reportMissingImports=false
"""Tests for local admin annotation review manifest behavior."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from backend.services.annotation_review import (
    AnnotationReviewNotFoundError,
    AnnotationReviewConflictError,
    AnnotationReviewStore,
)
from backend.services.annotation_storage import AnnotationConflictError, save_annotation


def _make_image():
    return Image.new("RGB", (12, 12), color=(245, 245, 245))


def _annotation(index: int, class_name: str):
    return {"index": index, "class_name": class_name, "bbox": [index, index, 4, 4]}


def _save(annotations_dir: Path, analysis_id: str, count: int = 3):
    return save_annotation(
        analysis_id,
        _make_image(),
        [_annotation(i, f"class-{i}") for i in range(count)],
        base_dir=annotations_dir,
        elements_dir=annotations_dir.parent / "training_data" / "Elements",
    )


def _status_map(queue: dict, analysis_id: str) -> dict[int, str]:
    [analysis] = [item for item in queue["analyses"] if item["analysis_id"] == analysis_id]
    return {element["index"]: element["review_status"] for element in analysis["elements"]}


def test_missing_manifest_defaults_to_pending(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "analysis-1", count=2)

    queue = AnnotationReviewStore(annotations_dir).list_queue()

    assert queue["local_only"] is True
    assert "not production-secured" in queue["warning"]
    assert queue["counts"] == {
        "total": 2,
        "pending": 2,
        "approved": 0,
        "rejected": 0,
        "trainable": 0,
    }
    assert _status_map(queue, "analysis-1") == {0: "pending", 1: "pending"}
    assert {
        (element["dataset_split"], element["split_reason"])
        for element in queue["analyses"][0]["elements"]
    } == {("excluded", "pending_review")}


def test_element_level_status_persists_and_can_be_mixed(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "mixed-1", count=3)

    store = AnnotationReviewStore(annotations_dir)
    store.set_status("mixed-1", 0, "approved")
    store.set_status("mixed-1", 1, "rejected")

    reloaded = AnnotationReviewStore(annotations_dir)
    queue = reloaded.list_queue()

    assert _status_map(queue, "mixed-1") == {
        0: "approved",
        1: "rejected",
        2: "pending",
    }
    assert queue["counts"]["trainable"] == 1
    manifest = store.export_review_manifest()
    assert sorted(manifest["decisions"]) == ["mixed-1:0", "mixed-1:1"]
    approved = queue["analyses"][0]["elements"][0]
    rejected = queue["analyses"][0]["elements"][1]
    assert approved["dataset_split"] in {"train", "val", "test"}
    assert approved["split_reason"] == "trainable_hash_80_10_10"
    assert manifest["decisions"]["mixed-1:0"]["dataset_split"] == approved["dataset_split"]
    assert manifest["decisions"]["mixed-1:0"]["split_reason"] == "trainable_hash_80_10_10"
    assert rejected["dataset_split"] == "excluded"
    assert rejected["split_reason"] == "rejected_review"


def test_existing_approved_decision_gets_stable_dataset_split_without_fingerprint_hash(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "legacy-split-1", count=1)
    store = AnnotationReviewStore(annotations_dir)
    result = store.set_status("legacy-split-1", 0, "approved")
    original_split = result["element"]["dataset_split"]
    assert original_split in {"train", "val", "test"}

    manifest = store.export_review_manifest()
    legacy_decision = manifest["decisions"]["legacy-split-1:0"]
    legacy_decision.pop("dataset_split")
    legacy_decision.pop("split_reason")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE review_decisions SET decision_json = ? WHERE key = ?",
            (json.dumps(legacy_decision), "legacy-split-1:0"),
        )

    queue = AnnotationReviewStore(annotations_dir).list_queue()
    [element] = queue["analyses"][0]["elements"]

    assert element["review_status"] == "approved"
    assert element["dataset_split"] == original_split
    assert element["split_reason"] == "trainable_hash_80_10_10"


def test_approved_only_iterator_excludes_rejected_pending_missing_crop_and_orphan(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "eligibility-1", count=4)

    store = AnnotationReviewStore(annotations_dir)
    store.set_status("eligibility-1", 0, "approved")
    store.set_status("eligibility-1", 1, "rejected")
    store.set_status("eligibility-1", 3, "approved")
    (annotations_dir / "eligibility-1" / "elements" / "3.png").unlink()

    orphan = {
        "analysis_id": "orphan-1",
        "index": 0,
        "status": "approved",
        "reviewed_at": "2026-05-26T00:00:00+00:00",
        "source_fingerprint": "missing",
    }
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "INSERT INTO review_decisions VALUES (?, ?, ?)",
            ("orphan-1:0", 1, json.dumps(orphan)),
        )

    queue = AnnotationReviewStore(annotations_dir).list_queue()
    approved = list(AnnotationReviewStore(annotations_dir).iter_approved_annotations())

    assert [row["index"] for row in approved] == [0]
    assert approved[0]["analysis_id"] == "eligibility-1"
    assert {item["code"] for item in queue["diagnostics"]} >= {
        "missing_crop",
        "orphan_decision",
    }
    # Deleting an approved crop changes the source fingerprint, so the stale
    # decision is treated as pending and remains non-trainable.
    assert queue["counts"]["approved"] == 1
    assert queue["counts"]["trainable"] == 1


def test_resubmitting_changed_analysis_is_rejected_without_losing_approval(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "replace-1", count=1)
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("replace-1", 0, "approved")

    try:
        save_annotation(
            "replace-1",
            _make_image(),
            [{"index": 0, "class_name": "new-class", "bbox": [2, 2, 4, 4]}],
            base_dir=annotations_dir,
            elements_dir=annotations_dir.parent / "training_data" / "Elements",
        )
    except AnnotationConflictError:
        pass
    else:
        raise AssertionError("changed submission was accepted")

    queue = AnnotationReviewStore(annotations_dir).list_queue()

    assert _status_map(queue, "replace-1") == {0: "approved"}
    assert len(list(AnnotationReviewStore(annotations_dir).iter_approved_annotations())) == 1


def test_metadata_crop_path_must_resolve_inside_analysis_folder(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "path-guard-1", count=1)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not a crop")
    metadata_path = annotations_dir / "path-guard-1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["annotations"][0]["crop_path"] = str(outside)
    metadata_path.write_text(json.dumps(metadata))

    queue = AnnotationReviewStore(annotations_dir).list_queue()
    [element] = queue["analyses"][0]["elements"]

    assert element["crop_exists"] is True
    assert Path(element["crop_path"]).parts[-3:] == ("path-guard-1", "elements", "0.png")
    assert str(outside) != element["crop_path"]


def test_metadata_analysis_id_must_be_safe(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "safe-folder-1", count=1)
    metadata_path = annotations_dir / "safe-folder-1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["analysis_id"] = "../unsafe"
    metadata_path.write_text(json.dumps(metadata))

    queue = AnnotationReviewStore(annotations_dir).list_queue()

    assert queue["analyses"] == []
    assert any(item["code"] == "invalid_metadata" for item in queue["diagnostics"])


def test_invalid_or_missing_mutations_raise_clear_errors(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "valid-1", count=1)
    store = AnnotationReviewStore(annotations_dir)

    try:
        store.set_status("valid-1", 99, "approved")
    except AnnotationReviewNotFoundError as exc:
        assert "valid-1:99" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing element did not raise")

    try:
        store.set_status("../bad", 0, "approved")
    except ValueError as exc:
        assert "analysis_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid analysis id did not raise")

    try:
        store.set_status("valid-1", 0, "validated")
    except ValueError as exc:
        assert "status must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid status did not raise")


def test_modify_element_preserves_submission_and_resets_to_pending(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "modify-1", count=1)
    image_path = annotations_dir / "modify-1" / "image.png"
    with Image.open(image_path) as source:
        patterned_source = source.convert("RGB")
    for y in range(patterned_source.height):
        for x in range(patterned_source.width):
            patterned_source.putpixel((x, y), (x * 17, y * 19, (x + y) * 11))
    patterned_source.save(image_path)
    store = AnnotationReviewStore(annotations_dir)
    approved = store.set_status("modify-1", 0, "approved")["element"]
    metadata_before = (annotations_dir / "modify-1" / "metadata.json").read_bytes()

    result = store.modify_element(
        "modify-1",
        0,
        class_name=" edited-class ",
        bbox=[2.2, 3.6, 5.1, 4.9],
    )

    element = result["element"]
    assert element["class_name"] == "edited-class"
    assert element["bbox"] == [2, 4, 5, 5]
    assert element["review_status"] == "pending"
    assert element["trainable"] is False
    assert element["source_fingerprint"] != approved["source_fingerprint"]
    assert result["counts"]["pending"] == 1
    assert list(AnnotationReviewStore(annotations_dir).iter_approved_annotations()) == []

    metadata = json.loads((annotations_dir / "modify-1" / "metadata.json").read_text())
    [annotation] = metadata["annotations"]
    assert (annotations_dir / "modify-1" / "metadata.json").read_bytes() == metadata_before
    assert annotation["class_name"] == "class-0"
    assert annotation["bbox"] == [0, 0, 4, 4]
    with Image.open(image_path) as source, Image.open(element["crop_path"]) as crop:
        assert crop.size == (5, 5)
        expected = source.convert("RGB").crop((2, 4, 7, 9))
        assert crop.convert("RGB").tobytes() == expected.tobytes()


def test_modify_element_can_save_and_approve_with_fresh_fingerprint(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "modify-approve-1", count=1)
    store = AnnotationReviewStore(annotations_dir)

    result = store.modify_element(
        "modify-approve-1",
        0,
        class_name="approved-class",
        bbox=[1, 1, 4, 4],
        status="approved",
    )

    element = result["element"]
    assert element["review_status"] == "approved"
    assert element["trainable"] is True
    approved = list(AnnotationReviewStore(annotations_dir).iter_approved_annotations())
    assert [(row["class_name"], row["bbox"]) for row in approved] == [("approved-class", [1, 1, 4, 4])]


def test_modify_element_rejects_invalid_class_bbox_and_status(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir, "modify-errors-1", count=1)
    store = AnnotationReviewStore(annotations_dir)

    for kwargs in [
        {"class_name": "../bad", "bbox": [1, 1, 4, 4]},
        {"class_name": "ok", "bbox": [1, 1, -4, 4]},
        {"class_name": "ok", "bbox": [1, 1, 4]},
        {"class_name": "ok", "bbox": [1, 1, 4, 4], "status": "validated"},
    ]:
        try:
            store.modify_element("modify-errors-1", 0, **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid modify payload accepted: {kwargs}")


def test_two_reviewers_cannot_commit_the_same_revision(tmp_path):
    root = tmp_path / "annotations"
    _save(root, "race-1", count=1)
    def mutate(status):
        return AnnotationReviewStore(root).set_status("race-1", 0, status, expected_revision=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(mutate, status) for status in ("approved", "rejected")]
    outcomes = [future.exception() for future in futures]
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, AnnotationReviewConflictError) for outcome in outcomes) == 1
    assert AnnotationReviewStore(root).list_queue()["analyses"][0]["elements"][0]["revision"] == 1


def test_legacy_import_requires_explicit_apply_and_keeps_verified_backup(tmp_path):
    root = tmp_path / "annotations"
    _save(root, "legacy-1", count=1)
    store = AnnotationReviewStore(root)
    source = store.list_queue()["analyses"][0]["elements"][0]
    manifest = {"schema_version": 1, "decisions": {"legacy-1:0": {
        "analysis_id": "legacy-1", "index": 0, "status": "approved",
        "source_fingerprint": source["source_fingerprint"], "class_name": "class-0",
        "bbox": source["bbox"], "reviewed_at": "2026-01-01T00:00:00+00:00",
    }}}
    legacy_path = root / "review-index.json"
    legacy_path.write_text(json.dumps(manifest), encoding="utf-8")
    preview = store.preview_legacy_import()
    assert preview["decisions"] == 1
    assert not store.db_path.exists()
    assert store.list_queue()["counts"]["trainable"] == 1
    try:
        store.set_status("legacy-1", 0, "pending", expected_revision=0)
    except AnnotationReviewConflictError:
        pass
    else:
        raise AssertionError("legacy store was mutated without import")
    imported = store.import_legacy_manifest()
    assert imported["status"] == "imported"
    assert Path(imported["backup"]).read_bytes() == legacy_path.read_bytes()
    assert store.import_legacy_manifest()["status"] == "already_imported"
    assert store.list_queue()["analyses"][0]["elements"][0]["revision"] == 1
    assert store.set_status("legacy-1", 0, "pending", expected_revision=1)["element"]["revision"] == 2

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import AnnotationReviewStore
from backend.services.annotation_storage import save_annotation
from scripts.export_approved_annotations import export_approved_annotations


def _make_image():
    return Image.new("RGB", (12, 12), color=(255, 255, 255))


def _save(annotations_dir: Path):
    return save_annotation(
        "approved-export-1",
        _make_image(),
        [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]},
            {"index": 1, "class_name": "calli", "bbox": [1, 1, 4, 4]},
            {"index": 2, "class_name": "tochtli", "bbox": [2, 2, 4, 4]},
            {"index": 3, "class_name": "missing-crop", "bbox": [3, 3, 4, 4]},
        ],
        base_dir=annotations_dir,
        elements_dir=annotations_dir.parent / "training_data" / "Elements",
    )


def _patch_decisions(store: AnnotationReviewStore, splits: dict[str, str]) -> None:
    """Edit persisted SQLite review decisions in place (simulates legacy/imported splits)."""
    with closing(sqlite3.connect(store.db_path)) as connection, connection:
        for key, split in splits.items():
            (raw,) = connection.execute("SELECT decision_json FROM review_decisions WHERE key = ?", (key,)).fetchone()
            decision = json.loads(raw)
            decision["dataset_split"] = split
            connection.execute("UPDATE review_decisions SET decision_json = ? WHERE key = ?", (json.dumps(decision), key))


def test_export_approved_annotations_materializes_only_trainable_approved_elements(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("approved-export-1", 0, "approved")
    store.set_status("approved-export-1", 1, "rejected")
    store.set_status("approved-export-1", 3, "approved")
    (annotations_dir / "approved-export-1" / "elements" / "3.png").unlink()

    orphan = {
        "analysis_id": "orphan-1",
        "index": 0,
        "status": "approved",
        "reviewed_at": "2026-05-26T00:00:00+00:00",
        "source_fingerprint": "missing",
    }
    with closing(sqlite3.connect(store.db_path)) as connection, connection:
        connection.execute("INSERT INTO review_decisions VALUES (?, ?, ?)",
                           ("orphan-1:0", 1, json.dumps(orphan)))

    output_dir = tmp_path / "approved" / "Elements"
    summary = export_approved_annotations(annotations_dir, output_dir)

    exported_files = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.glob("*/*.bmp"))
    assert summary["exported_count"] == 1
    assert summary["class_count"] == 1
    assert summary["classes"] == ["atl"]
    assert exported_files == ["0001-atl/999_000_000-approved-export-1_0.bmp"]
    assert summary["rows"][0]["dataset_split"] in {"train", "val", "test"}
    approved_row = next(iter(AnnotationReviewStore(annotations_dir).iter_approved_annotations()))
    assert summary["rows"][0]["dataset_split"] == approved_row["dataset_split"]
    manifest_path = output_dir / "_approved_export_manifest.json"
    assert manifest_path.is_file()
    persisted = json.loads(manifest_path.read_text())
    assert persisted["rows"][0]["dataset_split"] == summary["rows"][0]["dataset_split"]


def test_export_approved_annotations_clean_removes_stale_output(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    AnnotationReviewStore(annotations_dir).set_status("approved-export-1", 0, "approved")
    output_dir = tmp_path / "approved" / "Elements"
    stale = output_dir / "9999-stale" / "999_000_000-stale.bmp"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    export_approved_annotations(annotations_dir, output_dir, clean=True)

    assert not stale.exists()


def test_export_excludes_research_note_from_training_manifest(tmp_path):
    """F2 contract: element notes are annotator memos, never training data."""
    annotations_dir = tmp_path / "annotations"
    save_annotation(
        "noted-export-1",
        _make_image(),
        [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4], "note": "lecture douteuse"},
        ],
        base_dir=annotations_dir,
        elements_dir=annotations_dir.parent / "training_data" / "Elements",
    )
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("noted-export-1", 0, "approved")

    # The review queue surfaces the note for human reviewers...
    queue = store.list_queue()
    element = queue["analyses"][0]["elements"][0]
    assert element["note"] == "lecture douteuse"

    # ...but neither the approved row stream nor the export manifest carry it.
    approved_row = next(iter(store.iter_approved_annotations()))
    assert "note" not in approved_row

    output_dir = tmp_path / "approved" / "Elements"
    summary = export_approved_annotations(annotations_dir, output_dir)
    assert "note" not in summary["rows"][0]
    persisted = json.loads((output_dir / "_approved_export_manifest.json").read_text())
    assert all("note" not in row for row in persisted["rows"])
    assert '"note"' not in json.dumps(persisted)


def test_approved_explicitly_excluded_annotation_is_never_reassigned_or_exported(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("approved-export-1", 0, "approved")
    _patch_decisions(store, {"approved-export-1:0": "excluded"})

    element = store.list_queue()["analyses"][0]["elements"][0]
    assert element["dataset_split"] == "excluded"
    assert element["trainable"] is False
    assert export_approved_annotations(annotations_dir, tmp_path / "output")["exported_count"] == 0
    assert store.set_status("approved-export-1", 0, "approved")["element"]["trainable"] is False
    store.set_status("approved-export-1", 0, "pending")
    assert store.set_status("approved-export-1", 0, "approved")["element"]["trainable"] is True


def test_export_reports_every_exclusion_and_preserves_runtime_labels(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("approved-export-1", 0, "approved")
    store.set_status("approved-export-1", 1, "rejected")
    store.set_status("approved-export-1", 2, "approved")
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"class_names": ["calli", "atl", "missing-crop"]}))

    report = export_approved_annotations(annotations_dir, tmp_path / "output", runtime_config=config)
    assert report["total_annotation_count"] == 4
    assert report["exported_count"] == 1
    assert report["excluded_counts"] == {"rejected_review": 1, "unknown_class": 1, "pending_review": 1}
    assert report["exported_count"] + report["excluded_count"] == report["total_annotation_count"]
    assert sum(report["split_counts"].values()) == 1
    assert report["rows"][0]["class_label"] == 1
    assert Path(report["rows"][0]["output_path"]).parent.name == "0002-atl"


def test_approved_metadata_keeps_labels_splits_and_source_pages(tmp_path):
    from backend.codex_pipeline.data.metadata import build_metadata
    from backend.codex_pipeline.data.cached_dataset import CachedFeatureDataset
    import torch
    import pytest

    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    store = AnnotationReviewStore(annotations_dir)
    for index in [0, 1, 2]:
        store.set_status("approved-export-1", index, "approved")
    _patch_decisions(store, {f"approved-export-1:{index}": split for index, split in enumerate(["train", "val", "test"])})
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"class_names": ["calli", "unused", "atl", "tochtli", "missing-crop"]}))
    output = tmp_path / "Elements"
    report = export_approved_annotations(annotations_dir, output, runtime_config=config)
    metadata = build_metadata(str(output))
    assert dict(zip(metadata.element_name, metadata.class_label)) == {"atl": 2, "calli": 0, "tochtli": 3}
    assert dict(zip(metadata.element_name, metadata.dataset_split)) == {"atl": "train", "calli": "dev", "tochtli": "locked_test"}
    assert metadata.source_group.nunique() == 1
    assert metadata.source_group.iloc[0].startswith("image-pixels:")
    features = tmp_path / "features.pt"
    torch.save({"features": torch.arange(3).float().unsqueeze(1), "labels": torch.tensor(metadata.class_label.tolist()), "dataset_splits": metadata.dataset_split.tolist()}, features)
    train, _ = CachedFeatureDataset.from_file(str(features), split="train", split_strategy="persisted")
    val, _ = CachedFeatureDataset.from_file(str(features), split="val", split_strategy="persisted")
    assert train.labels.tolist() == [2]
    assert val.labels.tolist() == [0]
    with pytest.raises(ValueError, match="preserve heldouts"):
        CachedFeatureDataset.from_file(str(features), split="train")
    relative = report["rows"][0]["output_relative_path"]
    (output / relative).unlink()
    with pytest.raises(ValueError, match="missing images"):
        build_metadata(str(output))

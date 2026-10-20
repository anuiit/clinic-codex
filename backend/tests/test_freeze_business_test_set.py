from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.freeze_business_test_set import (
    materialize_business_test_set,
    plan_business_test_set,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    annotations = tmp_path / "annotations"
    baseline = annotations / "annotations"
    (baseline / "old").mkdir(parents=True)
    _write_json(
        baseline / "old" / "metadata.json",
        {"analysis_id": "old", "annotations": []},
    )
    Image.new("RGB", (2, 2), "white").save(baseline / "old" / "image.png")

    analysis = annotations / "new"
    (analysis / "elements").mkdir(parents=True)
    page = Image.new("RGB", (4, 2), "white")
    page.putpixel((0, 0), (255, 0, 0))
    page.putpixel((2, 0), (0, 0, 255))
    page.save(analysis / "image.png")
    metadata = {
        "analysis_id": "new",
        "annotations": [
            {"index": 0, "class_name": "alpha", "bbox": [0, 0, 2, 2]},
            {"index": 1, "class_name": "outside", "bbox": [2, 0, 2, 2]},
            {"index": 2, "class_name": "beta", "bbox": [1, 0, 1, 1]},
        ],
    }
    _write_json(analysis / "metadata.json", metadata)
    page.crop((0, 0, 2, 2)).save(analysis / "elements" / "0.png")
    decisions = {
        f"new:{row['index']}": {
            "analysis_id": "new",
            "index": row["index"],
            "class_name": row["class_name"],
            "bbox": row["bbox"],
            "status": "rejected" if row["index"] == 2 else "approved",
            "reviewed_at": "2026-08-12T00:00:00Z",
            "source_fingerprint": "legacy",
        }
        for row in metadata["annotations"]
    }
    _write_json(annotations / "review-index.json", {"decisions": decisions})

    runtime = tmp_path / "runtime.json"
    _write_json(runtime, {"class_names": ["alpha", "beta"]})
    training = tmp_path / "training.json"
    _write_json(
        training,
        {
            "schema_version": "training-snapshot.v2",
            "rows": [{"source_pixel_sha256": "0" * 64}],
        },
    )
    return annotations, baseline, runtime, training


def test_freeze_is_deterministic_and_write_once(tmp_path: Path) -> None:
    annotations, baseline, runtime, training = _fixture(tmp_path)
    kwargs = {
        "annotations_dir": annotations,
        "baseline_annotations_dir": baseline,
        "runtime_config": runtime,
        "reference_training_manifest": training,
    }
    first, payloads = plan_business_test_set(**kwargs)
    second, second_payloads = plan_business_test_set(**kwargs)

    assert first == second
    assert payloads == second_payloads
    assert first["row_count"] == 1
    assert first["class_count"] == 1
    assert first["rows"][0]["crop_origin"] == "stored_crop_verified_against_page_bbox"
    assert {row["exclusion_reason"] for row in first["exclusions"]} == {
        "absent_from_runtime_taxonomy",
        "business_rejected",
    }

    output = materialize_business_test_set(first, payloads, tmp_path / "output")
    assert (output / "test_manifest.json").is_file()
    assert len(list((output / "Elements").rglob("*.bmp"))) == 1
    with pytest.raises(FileExistsError):
        materialize_business_test_set(first, payloads, tmp_path / "output")


def test_freeze_rejects_training_pixel_overlap(tmp_path: Path) -> None:
    annotations, baseline, runtime, training = _fixture(tmp_path)
    manifest, _ = plan_business_test_set(
        annotations_dir=annotations,
        baseline_annotations_dir=baseline,
        runtime_config=runtime,
        reference_training_manifest=training,
    )
    _write_json(
        training,
        {
            "schema_version": "training-snapshot.v2",
            "rows": [
                {"source_pixel_sha256": manifest["rows"][0]["crop_pixel_sha256"]}
            ],
        },
    )
    with pytest.raises(ValueError, match="overlaps reference training"):
        plan_business_test_set(
            annotations_dir=annotations,
            baseline_annotations_dir=baseline,
            runtime_config=runtime,
            reference_training_manifest=training,
        )

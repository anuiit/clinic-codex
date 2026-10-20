from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from backend.services.annotation_review import AnnotationReviewStore


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_annotation_stores.py"
SPEC = importlib.util.spec_from_file_location("audit_annotation_stores", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_store(
    root: Path,
    analysis_id: str,
    annotations: list[tuple[str, list[int], tuple[int, int, int]]],
    decisions: dict[int, str] | None = None,
) -> None:
    analysis = root / analysis_id
    elements = analysis / "elements"
    elements.mkdir(parents=True)
    Image.new("RGB", (32, 32), (240, 240, 240)).save(analysis / "image.png")
    metadata_rows = []
    for index, (class_name, bbox, color) in enumerate(annotations):
        crop = elements / f"{index}.png"
        Image.new("RGB", (8, 8), color).save(crop)
        metadata_rows.append(
            {
                "index": index,
                "class_name": class_name,
                "bbox": bbox,
                "crop_path": str(crop),
            }
        )
    (analysis / "metadata.json").write_text(
        json.dumps(
            {
                "analysis_id": analysis_id,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
                "annotations": metadata_rows,
            }
        ),
        encoding="utf-8",
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "decisions": {}}
    for index, status in (decisions or {}).items():
        row = metadata_rows[index]
        manifest["decisions"][f"{analysis_id}:{index}"] = {
            "analysis_id": analysis_id,
            "index": index,
            "class_name": row["class_name"],
            "bbox": row["bbox"],
            "status": status,
            "reviewed_at": "2026-01-02T00:00:00+00:00",
            "source_fingerprint": "legacy-path-dependent-fingerprint",
        }
    (root / "review-index.json").write_text(json.dumps(manifest), encoding="utf-8")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_audit_is_read_only_and_requires_manual_reaffirmation(tmp_path: Path) -> None:
    live = tmp_path / "live"
    historical = tmp_path / "historical"
    _write_store(
        live,
        "live-1",
        [
            ("alpha", [1, 2, 8, 8], (255, 0, 0)),
            ("gamma", [10, 11, 8, 8], (0, 0, 255)),
        ],
    )
    _write_store(
        historical,
        "history-1",
        [
            ("alpha", [1, 2, 8, 8], (255, 0, 0)),
            ("gamma", [10, 11, 8, 8], (0, 0, 255)),
            ("beta", [1, 2, 8, 8], (255, 0, 0)),
        ],
        decisions={0: "approved", 1: "approved", 2: "approved"},
    )
    before = (_tree_hash(live), _tree_hash(historical))
    report = audit.build_audit(live, historical)

    assert (_tree_hash(live), _tree_hash(historical)) == before
    assert report["schema_version"] == "annotation-store-audit.v2"
    assert report["counts"]["exact_cross_store_match_count"] == 2
    assert report["counts"]["conflict_count"] == 1
    actions = {
        row["historical_key"]: row["recommended_action"]
        for row in report["historical_rebind_actions"]
    }
    assert actions["history-1:0"] == "manual_review_conflict"
    assert actions["history-1:1"] == "manual_reaffirmation_exact_content_match"
    assert actions["history-1:2"] == "manual_review_conflict"
    assert all(not store["current_trainable_count"] for store in report["stores"])


def test_audit_output_is_write_once(tmp_path: Path) -> None:
    live = tmp_path / "live"
    historical = tmp_path / "historical"
    _write_store(live, "live-1", [("alpha", [1, 2, 3, 4], (1, 2, 3))])
    _write_store(historical, "history-1", [("alpha", [1, 2, 3, 4], (1, 2, 3))])
    output = tmp_path / "audit.json"

    assert audit.main(
        [
            "--live-store",
            str(live),
            "--historical-store",
            str(historical),
            "--output",
            str(output),
            "--json",
        ]
    ) == 0
    persisted = json.loads(output.read_text())
    assert persisted["audit_sha256"] == audit.build_audit(live, historical)["audit_sha256"]
    assert audit.main(
        [
            "--live-store",
            str(live),
            "--historical-store",
            str(historical),
            "--output",
            str(output),
        ]
    ) == 2


def test_audit_reads_decisions_from_sqlite_review_store(tmp_path: Path) -> None:
    live = tmp_path / "live"
    historical = tmp_path / "historical"
    _write_store(live, "live-1", [("alpha", [1, 2, 8, 8], (255, 0, 0))])
    _write_store(historical, "history-1", [("alpha", [1, 2, 8, 8], (255, 0, 0))])
    (live / "review-index.json").unlink()
    store = AnnotationReviewStore(live)
    store.set_status("live-1", 0, "approved", allow_self_review=True)
    (live / ".review-backups").mkdir()  # migration leaves this next to the analyses

    report = audit.build_audit(live, historical)

    stores = {item["role"]: item for item in report["stores"]}
    assert stores["live"]["review_store"] == "sqlite"
    assert stores["live"]["review_index_sha256"] == store.review_manifest_sha256()
    assert stores["live"]["decision_counts"].get("approved") == 1
    assert stores["live"]["current_trainable_count"] == 1
    assert stores["historical"]["review_store"] == "legacy_json"


def test_audit_reports_review_decisions_without_annotations_as_orphans(tmp_path: Path) -> None:
    live = tmp_path / "live"
    historical = tmp_path / "historical"
    _write_store(live, "live-1", [("alpha", [1, 2, 3, 4], (1, 2, 3))])
    _write_store(historical, "history-1", [("alpha", [1, 2, 3, 4], (1, 2, 3))])
    review_path = live / "review-index.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    orphan = {
        "analysis_id": "deleted-analysis",
        "index": 7,
        "class_name": "alpha",
        "bbox": [1, 2, 3, 4],
        "status": "approved",
        "reviewed_at": "2026-01-02T00:00:00+00:00",
    }
    review["decisions"]["deleted-analysis:7"] = orphan
    review_path.write_text(json.dumps(review), encoding="utf-8")

    report = audit.build_audit(live, historical)

    assert report["orphaned_decisions"] == [
        {"store_role": "live", "key": "deleted-analysis:7", "decision": orphan}
    ]
    assert report["counts"]["orphaned_decision_count"] == 1
    assert report["counts"]["live_orphaned_decision_count"] == 1
    assert report["counts"]["historical_orphaned_decision_count"] == 0
    stores = {store["role"]: store for store in report["stores"]}
    assert stores["live"]["orphaned_decision_count"] == 1
    assert stores["historical"]["orphaned_decision_count"] == 0

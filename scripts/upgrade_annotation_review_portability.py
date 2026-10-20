"""Rebase valid local review paths/fingerprints before moving annotations to another PC."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.annotation_review import AnnotationReviewStore, _is_relative_to  # noqa: E402


def _portable_path(raw: object, analysis_dir: Path) -> object:
    if not raw:
        return raw
    path = Path(raw)
    if not path.is_absolute():
        return raw
    if not _is_relative_to(path, analysis_dir):
        raise ValueError(f"crop path is outside its analysis: {path}")
    return path.relative_to(analysis_dir).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = AnnotationReviewStore(args.annotations_dir)
    queue = store.list_queue()
    elements = {element["key"]: element for analysis in queue["analyses"] for element in analysis["elements"]}
    changes = []
    unresolved = []
    with sqlite3.connect(store.db_path) as connection:
        for key, raw in connection.execute("SELECT key, decision_json FROM review_decisions"):
            old = json.loads(raw)
            element = elements.get(key)
            if element is None or element["stale_decision"] or element["review_status"] != old["status"]:
                unresolved.append(key)
                continue
            analysis_dir = store.annotations_dir / old["analysis_id"]
            portable = element["base_source_fingerprint"]
            history = []
            try:
                decision = {**old, "source_fingerprint": portable}
                if old.get("crop_path"):
                    decision["crop_path"] = _portable_path(old["crop_path"], analysis_dir)
                for revision, state_json in connection.execute(
                    "SELECT revision, state_json FROM review_history WHERE key = ?", (key,)
                ):
                    state = json.loads(state_json)
                    if state.get("source_fingerprint") not in (old["source_fingerprint"], portable):
                        raise ValueError("history has a different source fingerprint")
                    state["source_fingerprint"] = portable
                    if state.get("crop_path"):
                        state["crop_path"] = _portable_path(state["crop_path"], analysis_dir)
                    history.append((revision, state))
            except (TypeError, ValueError):
                unresolved.append(key)
                continue
            if decision != old or any(json.dumps(state, sort_keys=True) != json.dumps(
                    json.loads(connection.execute(
                        "SELECT state_json FROM review_history WHERE key = ? AND revision = ?", (key, revision)
                    ).fetchone()[0]), sort_keys=True) for revision, state in history):
                changes.append((key, decision, history))
        if args.apply and changes:
            backup_dir = store.annotations_dir / ".review-backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = backup_dir / f"before-portability-{stamp}-{uuid.uuid4().hex[:8]}.sqlite3"
            with sqlite3.connect(backup) as saved:
                connection.backup(saved)
            for key, decision, history in changes:
                connection.execute("UPDATE review_decisions SET decision_json = ? WHERE key = ?",
                                   (json.dumps(decision, sort_keys=True), key))
                for revision, state in history:
                    connection.execute("UPDATE review_history SET state_json = ? WHERE key = ? AND revision = ?",
                                       (json.dumps(state, sort_keys=True), key, revision))
    print(json.dumps({"upgraded": len(changes) if args.apply else 0,
                      "pending_upgrade": len(changes), "unresolved": unresolved,
                      "backup": str(backup) if args.apply and changes else None}))
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())

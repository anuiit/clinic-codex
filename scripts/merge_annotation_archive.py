"""Add an imported annotation archive to a local installation without replacing it."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Migrated archive annotations directory")
    parser.add_argument("--target", type=Path, required=True, help="Application annotations directory")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("source and target must be separate directories")
    source_db, target_db = source / "review-state.sqlite3", target / "review-state.sqlite3"
    if not source_db.is_file() or not target_db.is_file():
        raise ValueError("both review databases must exist before merging")
    folders = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))
    if any(p.is_symlink() or not (p / "metadata.json").is_file() for p in folders):
        raise ValueError("source contains an invalid analysis directory")
    collisions = [p.name for p in folders if (target / p.name).exists()]
    if collisions:
        raise ValueError(f"analysis ID collision; refusing merge: {collisions}")
    with sqlite3.connect(source_db) as incoming, sqlite3.connect(target_db) as existing:
        keys = {row[0] for row in incoming.execute("SELECT key FROM review_decisions")}
        overlap = keys & {row[0] for row in existing.execute("SELECT key FROM review_decisions")}
        if overlap:
            raise ValueError(f"review decision collision; refusing merge: {sorted(overlap)[:5]}")
    result = {"analyses": len(folders), "decisions": len(keys), "applied": False}
    if args.apply:
        backup_dir = target / ".review-backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"before-archive-merge-{stamp}.sqlite3"
        with sqlite3.connect(target_db) as existing, sqlite3.connect(backup) as saved:
            existing.backup(saved)
        for folder in folders:
            shutil.copytree(folder, target / folder.name)
        with sqlite3.connect(target_db, timeout=10) as existing:
            existing.execute("ATTACH DATABASE ? AS incoming", (str(source_db),))
            for table in ("review_decisions", "review_history"):
                existing.execute(f"INSERT INTO {table} SELECT * FROM incoming.{table}")
        result.update(applied=True, backup=str(backup))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

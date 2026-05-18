#!/usr/bin/env python3
"""One-shot migration: remove symlinks under backend/training_data/Elements
that point into backend/annotations/.

Usage:
  --dry-run (default): report what would be removed
  --apply: actually unlink and remove now-empty parent dirs under Elements/
"""
from pathlib import Path
import argparse
import os
import sys


def find_annotation_symlinks(elements_dir: Path, annotations_dir: Path):
    """Yield (path, link_target_str, resolved_target_path) for symlinks under
    elements_dir whose resolved target is inside annotations_dir.
    """
    annotations_resolved = annotations_dir.resolve()
    for p in elements_dir.rglob("*"):
        # count every entry visited (files, dirs, symlinks)
        if not p.exists() and not p.is_symlink():
            # broken non-symlink (very rare) - treat as scanned but skip
            continue
        if p.is_symlink():
            # Get the stored link target (may be relative) for user-friendly output
            try:
                link_target = os.readlink(str(p))
            except OSError:
                link_target = "<unreadable>"
            # Resolve without strict to avoid exceptions for missing targets
            target_resolved = p.resolve(strict=False)
            # Compare path components to avoid false-positive prefix matches
            try:
                if target_resolved.resolve().parts[: len(annotations_resolved.parts)] == annotations_resolved.parts:
                    yield p, link_target, target_resolved
            except Exception:
                # If resolve() fails for some reason, skip this symlink
                continue


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Migrate annotation symlinks from Elements/")
    parser.add_argument("--apply", action="store_true", help="Actually remove symlinks and clean empty dirs (default: dry-run)")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent.parent.resolve(), help="Repo root (default: auto-detected)")
    args = parser.parse_args(argv)

    root = args.root
    elements_dir = root / "backend" / "training_data" / "Elements"
    annotations_dir = root / "backend" / "annotations"

    if not elements_dir.exists():
        print("Nothing to migrate.")
        return 0

    # Count scanned entries (files/dirs/symlinks under Elements)
    scanned = 0
    for _ in elements_dir.rglob("*"):
        scanned += 1

    # Collect symlinks to remove
    removals = []
    for p, link_target, target_resolved in find_annotation_symlinks(elements_dir, annotations_dir):
        removals.append((p, link_target, target_resolved))

    if not args.apply:
        for p, link_target, _ in removals:
            # Print path relative to repo root for readability
            try:
                rel = p.relative_to(root)
            except Exception:
                rel = p
            print(f"[DRY-RUN] Would remove: {rel} -> {link_target}")
        print(f"Scanned: {scanned} entries, found: {len(removals)} symlinks pointing to annotations/ (dry-run, not removed)")
        return 0

    # Apply mode: unlink symlinks, then remove empty parent dirs under Elements/
    removed_symlinks = 0
    removed_dirs = 0
    removed_dirs_set = set()
    for p, link_target, _ in removals:
        if p.is_symlink():
            try:
                p.unlink()
                removed_symlinks += 1
                print(f"Removed symlink: {p.relative_to(root)} -> {link_target}")
            except Exception as e:
                print(f"Failed to remove symlink {p}: {e}")
                continue
            # Clean up empty parent dirs up to but not including Elements/
            current = p.parent
            while True:
                if current == elements_dir:
                    break
                try:
                    # If directory is empty, remove it and continue upwards
                    if not any(current.iterdir()):
                        try:
                            current.rmdir()
                            removed_dirs_set.add(current)
                            # move up
                            current = current.parent
                            continue
                        except Exception:
                            break
                    else:
                        break
                except Exception:
                    break

    removed_dirs = len(removed_dirs_set)
    print(f"Removed: {removed_symlinks} symlinks, {removed_dirs} empty dirs cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())

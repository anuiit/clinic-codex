#!/usr/bin/env python3
"""Atomically capture the current local classifier runtime as ``original``."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.model_registry import (  # noqa: E402
    RUNTIME_ARTIFACTS,
    ModelRegistry,
    ModelRegistryValidationError,
    file_info,
    sha256_file,
)
from scripts.promote_model import PromotionError, promotion_lock  # noqa: E402


class ImportOriginalError(RuntimeError):
    """Raised when importing the historical runtime is unsafe."""


def _is_link_or_reparse(path: Path) -> bool:
    """Reject POSIX symlinks and Windows reparse points without following them."""

    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _require_safe_path(path: Path, *, label: str, required: bool = False) -> None:
    if _is_link_or_reparse(path):
        raise ImportOriginalError(f"{label} must not be a symlink or reparse point: {path}")
    if required and not path.exists():
        raise ImportOriginalError(f"{label} does not exist: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _validate_layout(registry: ModelRegistry, version_id: str) -> None:
    """Reject overlapping or link-backed roots before the lock may write."""

    registry_root = registry.root
    runtime_root = registry.runtime_model_dir
    _require_safe_path(registry_root, label="registry root")
    _require_safe_path(runtime_root, label="runtime root", required=True)
    if not runtime_root.is_dir():
        raise ImportOriginalError(f"runtime root is not a directory: {runtime_root}")
    if _is_within(registry_root, runtime_root) or _is_within(runtime_root, registry_root):
        raise ImportOriginalError("registry root and runtime root must not overlap")

    versions_dir = registry.versions_dir
    _require_safe_path(versions_dir, label="registry versions directory")
    version_dir = registry.version_dir(version_id)
    _require_safe_path(version_dir, label="registry version target")
    staging_parent = registry_root / ".import-original-staging"
    _require_safe_path(staging_parent, label="registry staging directory")
    lock_dir = registry_root / ".locks"
    _require_safe_path(lock_dir, label="registry lock directory")
    _require_safe_path(lock_dir / "promotion.lock", label="registry promotion lock")


def _runtime_sources(runtime_dir: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    missing: list[str] = []
    for relative, runtime_parts in RUNTIME_ARTIFACTS:
        source = runtime_dir.joinpath(*runtime_parts)
        _require_safe_path(source.parent, label="runtime artifact parent", required=True)
        _require_safe_path(source, label="runtime artifact")
        if source.is_file():
            sources[relative] = source
        else:
            missing.append(str(source))
    if missing:
        raise ImportOriginalError(
            "cannot import original runtime; missing required artifact(s): " + ", ".join(missing)
        )
    return sources


def _hash_sources(sources: dict[str, Path]) -> dict[str, str]:
    return {relative: sha256_file(source) for relative, source in sorted(sources.items())}


def _runtime_package_hashes(runtime_package: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, runtime_parts in RUNTIME_ARTIFACTS:
        artifact = runtime_package.joinpath(*runtime_parts)
        _require_safe_path(artifact, label="validated package artifact")
        if not artifact.is_file():
            raise ImportOriginalError(f"validated runtime package is missing artifact: {artifact}")
        hashes[relative] = sha256_file(artifact)
    return hashes


def _copy_plan(registry: ModelRegistry, version_id: str, sources: dict[str, Path], hashes: dict[str, str]) -> list[dict[str, str]]:
    version_dir = registry.version_dir(version_id)
    return [
        {
            "destination": str(version_dir / relative),
            "path": relative,
            "sha256": hashes[relative],
            "source": str(sources[relative]),
        }
        for relative in sorted(sources)
    ]


def _alias_state(index: dict[str, Any], original_version: str) -> dict[str, str | None]:
    aliases = index.get("aliases")
    if not isinstance(aliases, dict):
        raise ImportOriginalError("model registry aliases must be an object")
    state = {
        "original": aliases.get("original"),
        "promoted": aliases.get("promoted"),
        "original_version": index.get("original_version"),
        "promoted_version": index.get("promoted_version"),
    }
    if state["original"] != original_version or state["original_version"] != original_version:
        raise ImportOriginalError("original aliases do not point to the imported original version")
    if not isinstance(state["promoted"], str) or not state["promoted"]:
        raise ImportOriginalError("promoted alias is not assigned after original import")
    if state["promoted"] != state["promoted_version"]:
        raise ImportOriginalError("promoted alias and promoted_version disagree")
    return state


def _validate_preexisting_promoted(registry: ModelRegistry, index: dict[str, Any]) -> None:
    aliases = index.get("aliases")
    if not isinstance(aliases, dict):
        raise ImportOriginalError("model registry aliases must be an object")
    promoted = aliases.get("promoted")
    promoted_version = index.get("promoted_version")
    if promoted is None and promoted_version is None:
        return
    if not isinstance(promoted, str) or not promoted or promoted != promoted_version:
        raise ImportOriginalError("preexisting promoted alias and promoted_version must agree")
    registry.resolve_runtime_package("promoted")


def _existing_original(registry: ModelRegistry, version_id: str, runtime_hashes: dict[str, str]) -> dict[str, Any] | None:
    index = registry.read_index()
    _validate_preexisting_promoted(registry, index)
    existing_version = index["aliases"].get("original")
    if not existing_version:
        if version_id in index["versions"] or registry.version_dir(version_id).exists():
            raise ImportOriginalError(f"version id already exists and is not the original: {version_id}")
        return None
    if existing_version != version_id:
        raise ImportOriginalError(
            f"original is already registered as {existing_version}; requested version id is {version_id}"
        )
    runtime_package = registry.resolve_runtime_package("original")
    if _runtime_package_hashes(runtime_package) != runtime_hashes:
        raise ImportOriginalError(
            "original already exists but current runtime hashes differ from the registered original"
        )
    return {"aliases": _alias_state(index, version_id), "runtime_package": str(runtime_package)}


def _copy_sources_to_staging(sources: dict[str, Path], staging_runtime: Path) -> None:
    """Copy all sources into a private runtime-shaped directory."""

    for relative, source in sources.items():
        _relative, runtime_parts = next(item for item in RUNTIME_ARTIFACTS if item[0] == relative)
        destination = staging_runtime.joinpath(*runtime_parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _import_metadata(
    registry: ModelRegistry,
    *,
    sources: dict[str, Path],
    runtime_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build final provenance before the immutable manifest is first written."""

    source_artifacts = []
    for relative, source in sorted(sources.items()):
        info = file_info(source)
        if info["sha256"] != runtime_hashes[relative]:
            raise ImportOriginalError("runtime changed while recording immutable import provenance")
        source_artifacts.append({"path": relative, **info})
    return {
        "promotion": {
            "imported_as_original": True,
            "runtime_model_dir": str(registry.runtime_model_dir),
        },
        "source": {
            "artifacts": source_artifacts,
            "git_commit": registry.git_short(),
            "kind": "runtime_import",
            "runtime_model_dir": str(registry.runtime_model_dir),
        },
        "training": {"command": "import_current_runtime"},
    }


def _snapshot_index(registry: ModelRegistry) -> tuple[bool, bytes | None]:
    if registry.index_path.exists():
        _require_safe_path(registry.index_path, label="registry index")
        return True, registry.index_path.read_bytes()
    return False, None


def _restore_index(registry: ModelRegistry, snapshot: tuple[bool, bytes | None]) -> None:
    existed, contents = snapshot
    if existed:
        assert contents is not None
        temporary = registry.index_path.with_name(f".{registry.index_path.name}.restore-{uuid.uuid4().hex}")
        temporary.write_bytes(contents)
        os.replace(temporary, registry.index_path)
    elif registry.index_path.exists():
        _require_safe_path(registry.index_path, label="registry index")
        registry.index_path.unlink()


def _remove_created_tree(path: Path, *, root: Path, label: str) -> None:
    """Remove only an expected direct child below an already-safe registry root."""

    if not path.exists():
        return
    _require_safe_path(path, label=label)
    _require_safe_path(root, label="registry root", required=True)
    if not _is_within(path, root) or path.resolve(strict=False) == root.resolve(strict=False):
        raise ImportOriginalError(f"refusing to remove {label} outside registry root: {path}")
    shutil.rmtree(path)


def _rollback_publication(
    registry: ModelRegistry,
    *,
    index_snapshot: tuple[bool, bytes | None],
    version_dir: Path,
) -> None:
    failures: list[str] = []
    try:
        _restore_index(registry, index_snapshot)
    except OSError as exc:
        failures.append(f"index restore failed: {exc}")
    try:
        _remove_created_tree(version_dir, root=registry.versions_dir, label="created version directory")
    except OSError as exc:
        failures.append(f"version cleanup failed: {exc}")
    if failures:
        raise ImportOriginalError("publication rollback incomplete: " + "; ".join(failures))


def _staging_runtime(registry: ModelRegistry) -> tuple[Path, Path]:
    staging_parent = registry.root / ".import-original-staging"
    staging_dir = staging_parent / uuid.uuid4().hex
    return staging_dir, staging_dir / "runtime"


def _result(*, action: str, aliases: dict[str, str | None] | None, copy_plan: list[dict[str, str]], dry_run: bool, hashes: dict[str, str], runtime_package: str | None, version_id: str, warnings: list[str] | None = None, would_mutate: bool) -> dict[str, Any]:
    return {
        "action": action,
        "aliases": aliases,
        "copy_plan": copy_plan,
        "dry_run": dry_run,
        "ok": True,
        "runtime_hashes": hashes,
        "runtime_package": runtime_package,
        "version_id": version_id,
        "warnings": warnings or [],
        "would_mutate": would_mutate,
    }


def import_original(*, registry: ModelRegistry, version_id: str, dry_run: bool) -> dict[str, Any]:
    """Plan or atomically import the original using the promotion's shared lock."""

    registry.version_dir(version_id)  # validates the explicit ID before mutation
    _validate_layout(registry, version_id)
    sources = _runtime_sources(registry.runtime_model_dir)
    before_hashes = _hash_sources(sources)
    plan = _copy_plan(registry, version_id, sources, before_hashes)

    # Dry-run deliberately takes neither a lock nor a staging write.
    if dry_run:
        existing = _existing_original(registry, version_id, before_hashes)
        return _result(
            action="noop" if existing else "import",
            aliases=existing["aliases"] if existing else None,
            copy_plan=plan,
            dry_run=True,
            hashes=before_hashes,
            runtime_package=existing["runtime_package"] if existing else None,
            version_id=version_id,
            would_mutate=False,
        )

    with promotion_lock(registry):
        # Re-read every input after acquiring the shared promotion/import lock.
        _validate_layout(registry, version_id)
        sources = _runtime_sources(registry.runtime_model_dir)
        before_hashes = _hash_sources(sources)
        plan = _copy_plan(registry, version_id, sources, before_hashes)
        existing = _existing_original(registry, version_id, before_hashes)
        if existing:
            after_hashes = _hash_sources(_runtime_sources(registry.runtime_model_dir))
            if after_hashes != before_hashes:
                raise ImportOriginalError("runtime changed while validating existing original; refusing no-op")
            return _result(
                action="noop",
                aliases=existing["aliases"],
                copy_plan=plan,
                dry_run=False,
                hashes=after_hashes,
                runtime_package=existing["runtime_package"],
                version_id=version_id,
                would_mutate=False,
            )

        index_snapshot = _snapshot_index(registry)
        version_dir = registry.version_dir(version_id)
        staging_dir, staging_runtime = _staging_runtime(registry)
        published = False
        result: dict[str, Any] | None = None
        failure: BaseException | None = None
        try:
            _copy_sources_to_staging(sources, staging_runtime)
            staged_hashes = _runtime_package_hashes(staging_runtime)
            if staged_hashes != before_hashes:
                raise ImportOriginalError("staging hashes differ from the pre-import runtime snapshot")
            after_staging_hashes = _hash_sources(_runtime_sources(registry.runtime_model_dir))
            if after_staging_hashes != before_hashes:
                raise ImportOriginalError("runtime changed while creating staging; refusing import")

            staged_sources = {
                relative: staging_runtime.joinpath(*runtime_parts)
                for relative, runtime_parts in RUNTIME_ARTIFACTS
            }
            published = True
            manifest = registry.create_version_from_artifacts(
                version_id,
                artifact_sources=staged_sources,
                status="original",
                metadata=_import_metadata(registry, sources=sources, runtime_hashes=before_hashes),
            )
            if manifest.get("version_id") != version_id or manifest.get("status") != "original":
                raise ImportOriginalError("registry did not return the requested original package")
            runtime_package = registry.resolve_runtime_package("original")
            if _runtime_package_hashes(runtime_package) != before_hashes:
                raise ImportOriginalError("imported original hashes differ from the staged runtime snapshot")
            final_hashes = _hash_sources(_runtime_sources(registry.runtime_model_dir))
            if final_hashes != before_hashes:
                raise ImportOriginalError("runtime changed during publication; import rolled back")
            result = _result(
                action="import",
                aliases=_alias_state(registry.read_index(), version_id),
                copy_plan=plan,
                dry_run=False,
                hashes=final_hashes,
                runtime_package=str(runtime_package),
                version_id=version_id,
                would_mutate=True,
            )
        except BaseException as exc:
            failure = exc
            if published:
                _rollback_publication(registry, index_snapshot=index_snapshot, version_dir=version_dir)
            raise
        finally:
            try:
                _remove_created_tree(staging_dir, root=registry.root / ".import-original-staging", label="private staging directory")
            except Exception as exc:
                if result is not None:
                    result["warnings"].append(f"published import left bounded staging residue at {staging_dir}: {exc}")
                elif failure is not None:
                    failure.add_note(f"staging cleanup also failed: {exc}")
                else:
                    active_exception = sys.exception()
                    if active_exception is not None:
                        active_exception.add_note(f"staging cleanup also failed: {exc}")
                    else:
                        raise
        assert result is not None
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import the current classifier runtime as immutable original.")
    parser.add_argument("--registry-dir", required=True, help="Model registry root to write.")
    parser.add_argument("--runtime-dir", required=True, help="Current runtime directory to import.")
    parser.add_argument("--version-id", required=True, help="Explicit immutable version ID for the original.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the import plan without mutation.")
    parser.add_argument("--json", action="store_true", help="Explicitly request JSON (the CLI always emits JSON).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ModelRegistry(Path(args.registry_dir), repo_root=REPO_ROOT, runtime_model_dir=Path(args.runtime_dir))
    try:
        result = import_original(registry=registry, version_id=args.version_id, dry_run=args.dry_run)
    except (ImportOriginalError, ModelRegistryValidationError, PromotionError, OSError) as exc:
        print(json.dumps({"action": "import", "dry_run": bool(args.dry_run), "error": str(exc), "ok": False, "version_id": args.version_id}, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"action": "import", "dry_run": bool(args.dry_run), "error": f"unexpected error: {exc}", "ok": False, "version_id": args.version_id}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

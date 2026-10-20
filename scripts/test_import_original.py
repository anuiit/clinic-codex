from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.model_registry import ModelRegistry, sha256_file  # noqa: E402
import scripts.import_original as import_original_module  # noqa: E402


VERSION_ID = "20260810T000000Z-nogit-original"


def _write_runtime(runtime_dir: Path, label: str) -> None:
    (runtime_dir / "weights").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "weights" / "prototypes.pt").write_bytes(f"{label}-prototypes".encode("utf-8"))
    (runtime_dir / "weights" / "projection.pt").write_bytes(f"{label}-projection".encode("utf-8"))
    (runtime_dir / "config.json").write_text(json.dumps({"model_version": label}), encoding="utf-8")


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run_paths(
        tmp_path / "backend" / "model_registry",
        tmp_path / "backend" / "codex_model",
        *extra,
    )


def _command(registry_dir: Path, runtime_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        "scripts/import_original.py",
        "--registry-dir",
        str(registry_dir),
        "--runtime-dir",
        str(runtime_dir),
        "--version-id",
        VERSION_ID,
        "--json",
        *extra,
    ]


def _run_paths(registry_dir: Path, runtime_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(registry_dir, runtime_dir, *extra),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _runtime_hashes(runtime: Path) -> dict[str, str]:
    return {
        "runtime/config.json": sha256_file(runtime / "config.json"),
        "runtime/weights/projection.pt": sha256_file(runtime / "weights" / "projection.pt"),
        "runtime/weights/prototypes.pt": sha256_file(runtime / "weights" / "prototypes.pt"),
    }


def _create_candidate(registry: ModelRegistry, tmp_path: Path, version_id: str = "20260810T000000Z-r2-candidate") -> None:
    source = tmp_path / "candidate-source"
    _write_runtime(source, "r2-like")
    registry.create_version_from_artifacts(
        version_id,
        artifact_sources={
            "runtime/config.json": source / "config.json",
            "runtime/weights/projection.pt": source / "weights" / "projection.pt",
            "runtime/weights/prototypes.pt": source / "weights" / "prototypes.pt",
        },
        metadata={"training": {"backbone": "dinov2_vitb14", "run": "r2-like"}},
    )


def _tree_contents(path: Path) -> dict[str, bytes]:
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def _promote_candidate_pointer(registry: ModelRegistry, version_id: str) -> None:
    index = registry.read_index()
    index["aliases"]["promoted"] = version_id
    index["promoted_version"] = version_id
    registry.write_index(index)


def test_dry_run_validates_and_never_creates_registry(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "historical")
    before = _runtime_hashes(runtime)

    result = _run(tmp_path, "--dry-run")

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["action"] == "import"
    assert body["dry_run"] is True
    assert body["would_mutate"] is False
    assert body["runtime_hashes"] == before
    assert [item["path"] for item in body["copy_plan"]] == sorted(before)
    assert not (tmp_path / "backend" / "model_registry").exists()
    assert _runtime_hashes(runtime) == before


def test_import_copies_hashes_sets_aliases_and_second_call_is_noop(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    before = _runtime_hashes(runtime)

    imported = _run(tmp_path)

    assert imported.returncode == 0, imported.stderr
    imported_body = json.loads(imported.stdout)
    assert imported_body["action"] == "import"
    assert imported_body["runtime_hashes"] == before
    assert imported_body["aliases"] == {
        "original": VERSION_ID,
        "original_version": VERSION_ID,
        "promoted": VERSION_ID,
        "promoted_version": VERSION_ID,
    }
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    package = registry.resolve_runtime_package("original")
    assert _runtime_hashes(package) == before
    index = registry.read_index()
    assert index["aliases"]["original"] == VERSION_ID
    assert index["aliases"]["promoted"] == VERSION_ID
    assert index["original_version"] == VERSION_ID
    assert index["promoted_version"] == VERSION_ID

    second = _run(tmp_path)

    assert second.returncode == 0, second.stderr
    second_body = json.loads(second.stdout)
    assert second_body["action"] == "noop"
    assert second_body["would_mutate"] is False
    assert second_body["runtime_hashes"] == before


def test_runtime_drift_refuses_existing_original_without_registry_mutation(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    assert _run(tmp_path).returncode == 0
    index_before = (registry_dir / "index.json").read_bytes()
    manifest_before = (registry_dir / "versions" / VERSION_ID / "manifest.json").read_bytes()
    (runtime / "config.json").write_text('{"model_version":"drifted"}', encoding="utf-8")

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "hashes differ" in json.loads(result.stdout)["error"]
    assert (registry_dir / "index.json").read_bytes() == index_before
    assert (registry_dir / "versions" / VERSION_ID / "manifest.json").read_bytes() == manifest_before


def test_incomplete_runtime_is_refused_without_creating_registry(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    (runtime / "weights").mkdir(parents=True)
    (runtime / "weights" / "prototypes.pt").write_bytes(b"only-one-artifact")

    result = _run(tmp_path, "--dry-run")

    assert result.returncode == 2
    assert "missing required artifact" in json.loads(result.stdout)["error"]
    assert not (tmp_path / "backend" / "model_registry").exists()


def test_drift_during_first_import_leaves_no_alias_or_version(tmp_path, monkeypatch):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    original_copy = import_original_module._copy_sources_to_staging

    def copy_then_drift(sources, staging_runtime):
        original_copy(sources, staging_runtime)
        (runtime / "config.json").write_text('{"model_version":"drifted"}', encoding="utf-8")

    monkeypatch.setattr(import_original_module, "_copy_sources_to_staging", copy_then_drift)

    with pytest.raises(import_original_module.ImportOriginalError, match="changed while creating staging"):
        import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert not (registry_dir / "index.json").exists()
    assert not registry.version_dir(VERSION_ID).exists()
    staging = registry_dir / ".import-original-staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_two_subprocess_calls_serialize_as_import_then_noop(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    first = _run(tmp_path)
    second = _run(tmp_path)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["action"] == "import"
    assert json.loads(second.stdout)["action"] == "noop"
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    assert registry.resolve_runtime_package("original").is_dir()


def test_subprocess_refuses_explicitly_held_promotion_lock(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)

    with import_original_module.promotion_lock(registry):
        blocked = _run(tmp_path)

    assert blocked.returncode == 2
    assert "promotion lock already exists" in json.loads(blocked.stdout)["error"]


def test_overlap_between_registry_and_runtime_is_refused(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "historical")

    result = _run_paths(runtime, runtime)

    assert result.returncode == 2
    assert "must not overlap" in json.loads(result.stdout)["error"]


def test_symlink_source_and_destination_are_refused_when_supported(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    source_target = tmp_path / "source-target.json"
    source_target.write_text('{"model_version":"historical"}', encoding="utf-8")
    config = runtime / "config.json"
    config.unlink()
    try:
        config.symlink_to(source_target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    source_result = _run(tmp_path, "--dry-run")

    assert source_result.returncode == 2
    assert "symlink or reparse" in json.loads(source_result.stdout)["error"]

    _write_runtime(runtime, "historical")
    target = tmp_path / "outside-version"
    target.mkdir()
    version_target = registry_dir / "versions" / VERSION_ID
    version_target.parent.mkdir(parents=True)
    try:
        version_target.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    destination_result = _run(tmp_path, "--dry-run")

    assert destination_result.returncode == 2
    assert "symlink or reparse" in json.loads(destination_result.stdout)["error"]


def test_io_failure_after_publication_is_structured_and_rolls_back(tmp_path, monkeypatch):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    original_create = ModelRegistry.create_version_from_artifacts

    def create_then_fail(self, version_id, **kwargs):
        original_create(self, version_id, **kwargs)
        raise OSError("simulated registry I/O failure")

    monkeypatch.setattr(ModelRegistry, "create_version_from_artifacts", create_then_fail)

    with pytest.raises(OSError, match="simulated registry I/O failure"):
        import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert not (registry_dir / "index.json").exists()
    assert not registry.version_dir(VERSION_ID).exists()
    staging = registry_dir / ".import-original-staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_import_preserves_candidate_and_preexisting_promoted_and_records_real_provenance(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    candidate_id = "20260810T000000Z-r2-candidate"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    _create_candidate(registry, tmp_path, candidate_id)
    _promote_candidate_pointer(registry, candidate_id)
    candidate_before = _tree_contents(registry.version_dir(candidate_id))
    candidate_meta_before = json.loads(json.dumps(registry.read_index()["versions"][candidate_id]))

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    index = registry.read_index()
    assert index["aliases"]["candidate"] == candidate_id
    assert index["aliases"]["promoted"] == candidate_id
    assert index["promoted_version"] == candidate_id
    assert index["versions"][candidate_id] == candidate_meta_before
    assert _tree_contents(registry.version_dir(candidate_id)) == candidate_before
    manifest_path = registry.version_dir(VERSION_ID) / "manifest.json"
    model_card_path = registry.version_dir(VERSION_ID) / "model-card.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["runtime_model_dir"] == str(runtime)
    assert manifest["promotion"]["runtime_model_dir"] == str(runtime)
    assert ".import-original-staging" not in manifest_path.read_text(encoding="utf-8")
    assert ".import-original-staging" not in model_card_path.read_text(encoding="utf-8")
    package = registry.resolve_runtime_package("original")
    for relative, runtime_parts in import_original_module.RUNTIME_ARTIFACTS:
        source = runtime.joinpath(*runtime_parts)
        copied = package.joinpath(*runtime_parts)
        assert copied.stat().st_size == source.stat().st_size, relative
        assert copied.stat().st_mtime_ns == source.stat().st_mtime_ns, relative


def test_rollback_restores_nonempty_index_byte_for_byte_and_keeps_candidate(tmp_path, monkeypatch):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    candidate_id = "20260810T000000Z-r2-candidate"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    _create_candidate(registry, tmp_path, candidate_id)
    _promote_candidate_pointer(registry, candidate_id)
    index_before = registry.index_path.read_bytes()
    candidate_before = _tree_contents(registry.version_dir(candidate_id))
    original_create = ModelRegistry.create_version_from_artifacts

    def create_then_fail(self, version_id, **kwargs):
        original_create(self, version_id, **kwargs)
        raise OSError("simulated nonempty-registry failure")

    monkeypatch.setattr(ModelRegistry, "create_version_from_artifacts", create_then_fail)

    with pytest.raises(OSError, match="simulated nonempty-registry failure"):
        import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert registry.index_path.read_bytes() == index_before
    assert _tree_contents(registry.version_dir(candidate_id)) == candidate_before
    assert not registry.version_dir(VERSION_ID).exists()


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_process_interrupt_after_publication_rolls_back(tmp_path, monkeypatch, interrupt_type):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    original_create = ModelRegistry.create_version_from_artifacts

    def create_then_interrupt(self, version_id, **kwargs):
        original_create(self, version_id, **kwargs)
        raise interrupt_type("simulated interrupt")

    monkeypatch.setattr(ModelRegistry, "create_version_from_artifacts", create_then_interrupt)

    with pytest.raises(interrupt_type, match="simulated interrupt"):
        import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert not registry.index_path.exists()
    assert not registry.version_dir(VERSION_ID).exists()


def test_staging_cleanup_failure_after_success_returns_warning(tmp_path, monkeypatch):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    original_remove = import_original_module._remove_created_tree

    def fail_private_staging(path, *, root, label):
        if label == "private staging directory":
            raise OSError("simulated staging cleanup failure")
        return original_remove(path, root=root, label=label)

    monkeypatch.setattr(import_original_module, "_remove_created_tree", fail_private_staging)

    result = import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert result["ok"] is True
    assert result["action"] == "import"
    assert result["warnings"] and "bounded staging residue" in result["warnings"][0]
    assert registry.resolve_runtime_package("original").is_dir()


def test_import_failure_preserves_original_cause_when_staging_cleanup_also_fails(tmp_path, monkeypatch):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    registry = ModelRegistry(registry_dir, repo_root=tmp_path, runtime_model_dir=runtime)
    original_create = ModelRegistry.create_version_from_artifacts
    original_remove = import_original_module._remove_created_tree

    def create_then_fail(self, version_id, **kwargs):
        original_create(self, version_id, **kwargs)
        raise OSError("primary publication failure")

    def fail_private_staging(path, *, root, label):
        if label == "private staging directory":
            raise OSError("secondary staging cleanup failure")
        return original_remove(path, root=root, label=label)

    monkeypatch.setattr(ModelRegistry, "create_version_from_artifacts", create_then_fail)
    monkeypatch.setattr(import_original_module, "_remove_created_tree", fail_private_staging)

    with pytest.raises(OSError, match="primary publication failure") as raised:
        import_original_module.import_original(registry=registry, version_id=VERSION_ID, dry_run=False)

    assert any("staging cleanup also failed" in note for note in raised.value.__notes__)
    assert not registry.index_path.exists()
    assert not registry.version_dir(VERSION_ID).exists()
    assert (registry_dir / ".import-original-staging").exists()


def test_lock_directory_symlink_is_refused_before_dry_run_lock_creation(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    registry_dir = tmp_path / "backend" / "model_registry"
    _write_runtime(runtime, "historical")
    target = tmp_path / "outside-locks"
    target.mkdir()
    registry_dir.mkdir(parents=True)
    try:
        (registry_dir / ".locks").symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    result = _run(tmp_path, "--dry-run")

    assert result.returncode == 2
    assert "registry lock directory must not be a symlink or reparse point" in json.loads(result.stdout)["error"]
    assert not (registry_dir / "index.json").exists()


def test_main_serializes_unexpected_exception_without_catching_interrupt(tmp_path, monkeypatch, capsys):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "historical")

    def unexpected(**_kwargs):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(import_original_module, "import_original", unexpected)

    status = import_original_module.main(
        [
            "--registry-dir",
            str(tmp_path / "backend" / "model_registry"),
            "--runtime-dir",
            str(runtime),
            "--version-id",
            VERSION_ID,
            "--json",
        ]
    )

    assert status == 2
    assert json.loads(capsys.readouterr().out)["error"] == "unexpected error: simulated unexpected failure"

    def interrupt(**_kwargs):
        raise KeyboardInterrupt("do not serialize")

    monkeypatch.setattr(import_original_module, "import_original", interrupt)
    with pytest.raises(KeyboardInterrupt, match="do not serialize"):
        import_original_module.main(
            [
                "--registry-dir",
                str(tmp_path / "backend" / "model_registry"),
                "--runtime-dir",
                str(runtime),
                "--version-id",
                VERSION_ID,
                "--json",
            ]
        )

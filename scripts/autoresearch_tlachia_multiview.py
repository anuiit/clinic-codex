#!/usr/bin/env python3
"""P4a Tlachia multi-view research harness.

The runner is deliberately research-only.  It freezes and validates the
reviewed multi-view pilot, extracts local-pinned DINOv2-S/14 features, trains
projection-head-only VICReg challengers, and evaluates exclusively on the
sealed element train/dev cache and a group-disjoint structural development
split.  It never reads the terminal business set, annotation stores, or a
locked test, and it cannot promote a model.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "reports/p4-multiview-20260813/plans/pilot-v1.json"
DOWNLOAD_MANIFEST = REPO / "reports/p4-multiview-20260813/download/pilot-v1.json"
DOWNLOAD_VERIFICATION = REPO / "reports/p4-multiview-20260813/download/pilot-v1-verification.json"
IMAGES_ROOT = REPO / "backend/training_corpus/external/tlachia-multiview-pilot-v1"
ELEMENT_CACHE = REPO / "autoresearch/orchestrator-260813-1330/tlachia-s14/sealed/train-dev.features.pt"
BACKBONE_PIN = REPO / "backend/training_corpus/backbone-pins/dinov2-vits14-local.json"
HISTORICAL_PROJECTION = REPO / "backend/model_registry/versions/20260810T143106Z-7e40c6d-original/runtime/weights/projection.pt"
B1_PROJECTION = REPO / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/runtime/weights/projection.pt"
REPORT_DIR = REPO / "autoresearch/orchestrator-260813-1330/tlachia-multiview-p4a-v1"
FEATURE_CACHE = REPORT_DIR / "features/multiview.features.pt"
FEATURE_PROVENANCE = FEATURE_CACHE.with_suffix(FEATURE_CACHE.suffix + ".prov.json")

EXPECTED_FILE_SHA256 = {
    PLAN: "4db90b3f2829d55100d049acfa5f104f1b4883dc23ee180d6587aa83b7564fe8",
    DOWNLOAD_MANIFEST: "b3b6a53b985924327cc2c299c6f3c2d4b76bf936a1c8547262fcc3d7f82a66cc",
    DOWNLOAD_VERIFICATION: "e56b7b19edb7e8c43d1f91ad52859cbfa3d43c9c4cdf16b5e25a5d7f155d72dd",
    ELEMENT_CACHE: "0bd4a9bb6332459fa6dbb04441809c2e8e568d26bf4bb76c87b584c33815a40c",
    BACKBONE_PIN: "f6cb3eb246ad907a7cab8c6f6d21d55ce8bafd9cbc3e402bb10660ca48cc9cc2",
    HISTORICAL_PROJECTION: "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    B1_PROJECTION: "52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03",
}
EXPECTED_PAYLOAD_SHA256 = {
    "plan": "5d51ec319e8e66d098af9574b5877b87ef9cdafbaa4012e6ac573cd233963343",
    "manifest": "0ff9f6973d79b465de8f63ff71696ad418b6fd2025200837a094bb1175ceb59f",
    "verification": "6cadfbafbb9ec630b3bd7c895c15d072d64ed7e9bf7afd50f5e31ea23918787a",
}
EXPECTED_SCHEMAS = {
    "plan": "tlachia-media-download-plan.v1",
    "manifest": "tlachia-media-download-manifest.v1",
    "verification": "tlachia-multiview-verification.v1",
}
EXPECTED_ELEMENT_SPLITS = Counter(train=3763, dev=465)
EXPECTED_ROWS = 219
EXPECTED_GROUPS = 24
CLASS_COUNT = 286
DEV_GROUP_COUNT = 6
FEATURE_DIM = 384
EMBEDDING_DIM = 128
IMAGE_SIZE = 224
VIEW_SEED = 20260813
VIEW_NAMES = ("base", "mild-0", "mild-1", "strong-0", "strong-1")
ALLOWED_ROLES = frozenset({"glyph_occurrence", "zone", "planche"})
SAFE_VIEW_OPERATIONS = (
    "resize_longest_side",
    "white_pad",
    "affine_rotation_translation",
    "brightness",
    "contrast",
    "mild_blur_or_noise",
)
FORBIDDEN_VIEW_OPERATIONS = (
    "crop",
    "cutmix",
    "elastic_transform",
    "horizontal_flip",
    "hue_shift",
    "mixup",
    "perspective",
    "vertical_flip",
)
FORBIDDEN_PATH_TOKENS = ("annotations", "business-final", "locked-test", "locked_test")
SCREEN_SEED = 42
CONFIRMATION_SEEDS = (73, 101, 202)
VICREG_BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1
SUPERVISED_REFRESH = {
    "episodes": 100,
    "n_way": 20,
    "k_shot": 3,
    "q_queries": 5,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "temperature": TEMPERATURE,
}


@dataclasses.dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    roles: tuple[str, ...] = ()
    views: str = "none"
    learning_rate: float = 0.0
    epochs: int = 0
    balanced_roles: bool = False
    supervised_refresh: bool = False
    initializer: str = "historical"


SPECS = (
    Spec("historical-control", "control"),
    Spec("b1-control", "control", initializer="b1"),
    Spec("vicreg-glyph-mild-lr1e-5-e1", "vicreg", ("glyph_occurrence",), "mild", 1e-5, 1),
    Spec("vicreg-glyph-strong-lr3e-5-e5", "vicreg", ("glyph_occurrence",), "strong", 3e-5, 5),
    Spec("vicreg-glyph-both-lr1e-4-e10", "vicreg", ("glyph_occurrence",), "both", 1e-4, 10),
    Spec("vicreg-context-mild-lr3e-5-e1", "vicreg", ("zone", "planche"), "mild", 3e-5, 1),
    Spec("vicreg-context-strong-lr1e-4-e5", "vicreg", ("zone", "planche"), "strong", 1e-4, 5),
    Spec("vicreg-context-both-lr1e-5-e10", "vicreg", ("zone", "planche"), "both", 1e-5, 10),
    Spec("vicreg-all-mild-lr1e-4-e1", "vicreg", tuple(sorted(ALLOWED_ROLES)), "mild", 1e-4, 1),
    Spec("vicreg-all-strong-lr1e-5-e5", "vicreg", tuple(sorted(ALLOWED_ROLES)), "strong", 1e-5, 5),
    Spec("vicreg-all-both-lr3e-5-e10", "vicreg", tuple(sorted(ALLOWED_ROLES)), "both", 3e-5, 10),
    Spec("vicreg-glyph-both-balanced-lr3e-5-e5", "vicreg", ("glyph_occurrence",), "both", 3e-5, 5, True),
    Spec("vicreg-context-both-balanced-lr1e-4-e10", "vicreg", ("zone", "planche"), "both", 1e-4, 10, True),
    Spec("hybrid-glyph-both-lr1e-5-e5", "hybrid", ("glyph_occurrence",), "both", 1e-5, 5, False, True),
    Spec("hybrid-context-both-lr3e-5-e5", "hybrid", ("zone", "planche"), "both", 3e-5, 5, False, True),
    Spec("hybrid-all-both-balanced-lr1e-4-e10", "hybrid", tuple(sorted(ALLOWED_ROLES)), "both", 1e-4, 10, True, True),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files = (
        path
        for path in root.rglob("*")
        if path.is_file()
        and not excluded.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in {".pyc", ".pyo"}
    )
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats are forbidden in hashed payloads")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def semantic_hash(value: Any) -> str:
    encoded = json.dumps(
        _normalise(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16) & 0x7FFFFFFF


def write_once_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_once_torch(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable tensor artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
    except FileExistsError as error:  # pragma: no cover - concurrent prepare guard
        raise RuntimeError(f"immutable tensor artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def publish_feature_bundle(cache: dict[str, Any], provenance: dict[str, Any]) -> None:
    """Atomically publish cache and sidecar through one directory rename.

    The final ``features`` directory never exists with only one of its two
    required files.  A crash before rename leaves an unreferenced staging
    directory; a crash after rename leaves the complete immutable bundle.
    """
    target_dir = FEATURE_CACHE.parent
    if target_dir.exists():
        raise RuntimeError(f"immutable feature bundle already exists: {target_dir}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    staging = REPORT_DIR / f".features-stage-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    staged_cache = staging / FEATURE_CACHE.name
    staged_sidecar = staging / FEATURE_PROVENANCE.name
    torch.save(cache, staged_cache)
    published_provenance = dict(provenance)
    published_provenance["cache_path"] = str(FEATURE_CACHE.resolve())
    published_provenance["cache_sha256"] = sha256_file(staged_cache)
    staged_sidecar.write_text(
        json.dumps(published_provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.rename(staging, target_dir)
    except FileExistsError as error:  # pragma: no cover - concurrent prepare guard
        raise RuntimeError(f"immutable feature bundle already exists: {target_dir}") from error


def assert_allowed_input(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if any(token in part for token in FORBIDDEN_PATH_TOKENS for part in lowered):
        raise ValueError(f"forbidden research input path: {path}")


def assert_specs() -> None:
    if len(SPECS) != 16:
        raise ValueError("P4a protocol requires exactly 16 frozen specifications")
    if [spec.name for spec in SPECS[:2]] != ["historical-control", "b1-control"]:
        raise ValueError("the two frozen controls changed")
    challengers = SPECS[2:]
    if len(challengers) != 14 or any(spec.initializer != "historical" for spec in challengers):
        raise ValueError("all 14 challengers must independently start from historical")
    if sum(spec.kind == "hybrid" and spec.supervised_refresh for spec in challengers) < 3:
        raise ValueError("at least three VICReg plus supervised-refresh hybrids are required")
    if {spec.views for spec in challengers} != {"mild", "strong", "both"}:
        raise ValueError("mild, strong, and combined views must all be covered")
    if {spec.epochs for spec in challengers} != {1, 5, 10}:
        raise ValueError("the frozen epoch grid changed")
    if {spec.learning_rate for spec in challengers} != {1e-5, 3e-5, 1e-4}:
        raise ValueError("the frozen learning-rate grid changed")


def spec_record(spec: Spec) -> dict[str, Any]:
    """Return a JSON-stable representation (dataclasses preserve tuples)."""
    value = dataclasses.asdict(spec)
    value["roles"] = list(spec.roles)
    return value


def read_json(path: Path) -> dict[str, Any]:
    assert_allowed_input(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify_hashed_artifact(value: dict[str, Any], schema: str, expected_payload: str) -> None:
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected artifact schema: {value.get('schema_version')!r}")
    if value.get("canonicalization_version") != "canonical-json-nfc-v1":
        raise ValueError("reviewed artifact canonicalization changed")
    if value.get("hash_algorithm") != "sha256":
        raise ValueError("reviewed artifact hash algorithm changed")
    payload = value.get("payload")
    if not isinstance(payload, dict) or semantic_hash(payload) != value.get("payload_sha256"):
        raise ValueError("reviewed artifact semantic hash mismatch")
    if value.get("payload_sha256") != expected_payload:
        raise ValueError("reviewed artifact payload identity changed")


def _pixel_identity(path: Path) -> tuple[str, int]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest(), image.width * image.height


def validate_reviewed_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in EXPECTED_FILE_SHA256.items():
        assert_allowed_input(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 changed: {path}")
    if IMAGES_ROOT.resolve() != (REPO / "backend/training_corpus/external/tlachia-multiview-pilot-v1").resolve():
        raise ValueError("multi-view image root changed")
    plan = read_json(PLAN)
    manifest = read_json(DOWNLOAD_MANIFEST)
    verification = read_json(DOWNLOAD_VERIFICATION)
    verify_hashed_artifact(plan, EXPECTED_SCHEMAS["plan"], EXPECTED_PAYLOAD_SHA256["plan"])
    verify_hashed_artifact(manifest, EXPECTED_SCHEMAS["manifest"], EXPECTED_PAYLOAD_SHA256["manifest"])
    verify_hashed_artifact(
        verification, EXPECTED_SCHEMAS["verification"], EXPECTED_PAYLOAD_SHA256["verification"]
    )
    pp, mp, vp = plan["payload"], manifest["payload"], verification["payload"]
    if pp.get("purpose") != "multiview_self_supervised" or mp.get("purpose") != pp["purpose"]:
        raise ValueError("reviewed pilot purpose changed")
    if mp.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise ValueError("download manifest is not bound to the reviewed plan")
    if vp.get("plan_payload_sha256") != plan["payload_sha256"] or vp.get(
        "manifest_payload_sha256"
    ) != manifest["payload_sha256"]:
        raise ValueError("verification does not bind the reviewed plan and manifest")
    required_verification_gates = {
        "cross_group_pixel_duplicates_absent",
        "group_and_class_coverage_verified",
        "hashes_crc_and_budgets_verified",
        "plan_manifest_binding",
        "single_label_supervision_absent",
    }
    if any(vp.get("gates", {}).get(name) is not True for name in required_verification_gates):
        raise ValueError("reviewed download verification gate failed")
    if pp.get("gates", {}).get("promotion_allowed") is not False or vp.get("gates", {}).get(
        "promotion_allowed"
    ) is not False:
        raise ValueError("research pilot must remain non-promotable")
    if mp.get("gates", {}).get("download_complete") is not True or mp.get("gates", {}).get(
        "training_usable_without_snapshot_quarantine"
    ) is not True:
        raise ValueError("reviewed download is incomplete or quarantined")
    if Path(mp.get("output_root", "")).resolve() != IMAGES_ROOT.resolve():
        raise ValueError("manifest output root changed")

    entries = pp.get("entries")
    files = mp.get("files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_ROWS:
        raise ValueError("reviewed plan row count changed")
    if not isinstance(files, list) or len(files) != EXPECTED_ROWS:
        raise ValueError("reviewed download row count changed")
    files_by_key = {(row["archive_name"], row["entry_path"]): row for row in files}
    if len(files_by_key) != EXPECTED_ROWS:
        raise ValueError("duplicate physical file in reviewed manifest")
    pixel_groups: dict[str, set[str]] = defaultdict(set)
    groups: set[str] = set()
    classes: set[str] = set()
    for entry in entries:
        key = (entry.get("archive_name"), entry.get("entry_path"))
        row = files_by_key.get(key)
        if row is None:
            raise ValueError("plan and download physical keys differ")
        roles = set(entry.get("roles", ()))
        if not roles or not roles <= ALLOWED_ROLES:
            raise ValueError("structural entry contains an unsupported role")
        if entry.get("supervision_kind") != "unlabeled_ssl" or entry.get(
            "single_label_class"
        ) is not None:
            raise ValueError("structural media may only be unlabeled SSL input")
        group = str(entry.get("source_group", ""))
        if not group or entry.get("split_group") != group or entry.get("physical_source_group") != group:
            raise ValueError("structural source/split/physical group mismatch")
        if row.get("physical_source_group") != group:
            raise ValueError("manifest source group differs from plan")
        if row.get("candidate_runtime_classes") != entry.get("candidate_runtime_classes"):
            raise ValueError("manifest descendant class evidence differs from plan")
        if row.get("entry_crc32") != entry.get("entry", {}).get("crc32"):
            raise ValueError("manifest CRC evidence differs from plan")
        path = (IMAGES_ROOT / str(row.get("output_path", ""))).resolve()
        try:
            path.relative_to(IMAGES_ROOT.resolve())
        except ValueError as error:
            raise ValueError("downloaded media escapes the reviewed image root") from error
        if not path.is_file() or sha256_file(path) != row.get("byte_sha256"):
            raise ValueError("downloaded media byte SHA-256 mismatch")
        pixel_sha, pixels = _pixel_identity(path)
        if pixel_sha != row.get("image", {}).get("pixel_sha256") or pixels != row.get(
            "image", {}
        ).get("pixel_count"):
            raise ValueError("downloaded media decoded-pixel evidence mismatch")
        pixel_groups[pixel_sha].add(group)
        groups.add(group)
        classes.update(str(name) for name in entry.get("candidate_runtime_classes", ()))
    if any(len(owners) > 1 for owners in pixel_groups.values()):
        raise ValueError("pixel-identical structural media crosses planche groups")
    if len(groups) != EXPECTED_GROUPS or len(classes) != CLASS_COUNT:
        raise ValueError("reviewed structural group/class evidence changed")
    return plan, manifest, verification


def validate_element_cache(cache: dict[str, Any]) -> None:
    required = {
        "features",
        "labels",
        "dataset_splits",
        "source_groups",
        "row_ids",
        "class_names",
    }
    if required - set(cache):
        raise ValueError(f"element cache missing fields: {sorted(required - set(cache))}")
    lengths = {
        len(cache[key]) for key in ("features", "labels", "dataset_splits", "source_groups", "row_ids")
    }
    if lengths != {sum(EXPECTED_ELEMENT_SPLITS.values())}:
        raise ValueError("element cache field lengths changed")
    if Counter(cache["dataset_splits"]) != EXPECTED_ELEMENT_SPLITS:
        raise ValueError("element cache must contain sealed train/dev rows only")
    if cache["features"].shape != (sum(EXPECTED_ELEMENT_SPLITS.values()), FEATURE_DIM):
        raise ValueError("element feature tensor shape changed")
    if len(set(str(value) for value in cache["row_ids"])) != len(cache["row_ids"]):
        raise ValueError("element cache row IDs are not unique")
    class_names = {int(key): str(value) for key, value in cache["class_names"].items()}
    if sorted(class_names) != list(range(CLASS_COUNT)) or len(set(class_names.values())) != CLASS_COUNT:
        raise ValueError("element runtime class order changed")


def load_element_cache() -> dict[str, Any]:
    assert_allowed_input(ELEMENT_CACHE)
    cache = torch.load(ELEMENT_CACHE, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError("element cache is not a dictionary")
    validate_element_cache(cache)
    return cache


def choose_dev_groups(rows: Sequence[dict[str, Any]], count: int = DEV_GROUP_COUNT) -> tuple[str, ...]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[str(row["source_group"])].update(str(name) for name in row.get("candidate_runtime_classes", ()))
    if len(by_group) < count:
        raise ValueError("insufficient planche groups for structural development split")
    selected: list[str] = []
    covered: set[str] = set()
    remaining = set(by_group)
    while len(selected) < count:
        group = min(
            remaining,
            key=lambda value: (-len(by_group[value] - covered), stable_hash("p4a-dev-group", value)),
        )
        selected.append(group)
        covered.update(by_group[group])
        remaining.remove(group)
    return tuple(selected)


def assert_group_isolation(groups: Sequence[str], splits: Sequence[str]) -> None:
    train = {group for group, split in zip(groups, splits) if split == "structural_train"}
    dev = {group for group, split in zip(groups, splits) if split == "structural_dev"}
    if train & dev:
        raise ValueError("planche source-group overlap across structural train/dev")
    if set(splits) != {"structural_train", "structural_dev"}:
        raise ValueError("unexpected structural split name")


def build_rows(plan: dict[str, Any], manifest: dict[str, Any], class_names: dict[int, str]) -> list[dict[str, Any]]:
    entries = plan["payload"]["entries"]
    manifest_rows = {
        (row["archive_name"], row["entry_path"]): row for row in manifest["payload"]["files"]
    }
    name_to_label = {name: label for label, name in class_names.items()}
    dev_groups = set(choose_dev_groups(entries))
    result: list[dict[str, Any]] = []
    for entry in entries:
        key = (entry["archive_name"], entry["entry_path"])
        downloaded = manifest_rows[key]
        names = sorted(set(str(name) for name in entry.get("candidate_runtime_classes", ())))
        unknown = sorted(set(names) - set(name_to_label))
        if unknown:
            raise ValueError(f"descendant evidence contains unknown runtime classes: {unknown}")
        row_id = "tlachia-mv-" + stable_hash(key[0], key[1], downloaded["byte_sha256"])[:24]
        result.append(
            {
                "row_id": row_id,
                "image_path": str((IMAGES_ROOT / downloaded["output_path"]).resolve()),
                "byte_sha256": downloaded["byte_sha256"],
                "pixel_sha256": downloaded["image"]["pixel_sha256"],
                "roles": sorted(set(entry["roles"])),
                "source_group": entry["source_group"],
                "descendant_class_names": names,
                "descendant_class_labels": sorted(name_to_label[name] for name in names),
                "split": "structural_dev" if entry["source_group"] in dev_groups else "structural_train",
            }
        )
    result.sort(key=lambda row: row["row_id"])
    assert_group_isolation(
        [str(row["source_group"]) for row in result], [str(row["split"]) for row in result]
    )
    return result


def resize_and_pad(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size), (255, 255, 255))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return padded


def safe_ssl_view(base: Image.Image, row_id: str, strength: str, index: int) -> Image.Image:
    if strength not in {"mild", "strong"} or index not in {0, 1}:
        raise ValueError("safe structural views are exactly mild/strong x two")
    seed = stable_seed("p4a-view", VIEW_SEED, row_id, strength, index)
    rng = random.Random(seed)
    limit = 4.0 if strength == "mild" else 8.0
    shift = 0.02 if strength == "mild" else 0.04
    image = base.rotate(rng.uniform(-limit, limit), resample=Image.Resampling.BILINEAR, fillcolor="white")
    dx = round(rng.uniform(-shift, shift) * image.width)
    dy = round(rng.uniform(-shift, shift) * image.height)
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor="white",
    )
    photometric = 0.06 if strength == "mild" else 0.14
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - photometric, 1 + photometric))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - photometric, 1 + photometric))
    if index == 0:
        radius = rng.uniform(0.0, 0.35 if strength == "mild" else 0.8)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        array = np.asarray(image, dtype=np.float32)
        sigma = rng.uniform(0.0, 1.5 if strength == "mild" else 3.5)
        noise = np.random.default_rng(seed ^ 0x5A17).normal(0.0, sigma, array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    return image


def image_tensor(image: Image.Image) -> torch.Tensor:
    value = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy((value - mean) / std).permute(2, 0, 1)


class StructuralDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            base = resize_and_pad(source)
        views = [base]
        views.extend(safe_ssl_view(base, row["row_id"], "mild", i) for i in range(2))
        views.extend(safe_ssl_view(base, row["row_id"], "strong", i) for i in range(2))
        return torch.stack([image_tensor(view) for view in views])


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_backbone(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    pin = read_json(BACKBONE_PIN)
    if pin.get("schema_version") != "dinov2-local-pin.v1" or pin.get("backbone") != "dinov2_vits14":
        raise ValueError("local DINOv2-S/14 pin changed")
    repository = Path(str(pin.get("repository_path", "")))
    weights = Path(str(pin.get("weights_path", "")))
    assert_allowed_input(repository)
    assert_allowed_input(weights)
    if not (repository / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("local DINOv2 source or weights are missing")
    if sha256_tree(repository) != pin.get("source_tree_sha256"):
        raise ValueError("local DINOv2 source tree changed")
    if sha256_file(weights) != pin.get("weights_sha256"):
        raise ValueError("local DINOv2 weights changed")
    model = torch.hub.load(str(repository), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "backbone": "dinov2_vits14",
        "pin_path": str(BACKBONE_PIN.resolve()),
        "pin_sha256": sha256_file(BACKBONE_PIN),
        "repository_path": str(repository.resolve()),
        "source_tree_sha256": pin["source_tree_sha256"],
        "weights_path": str(weights.resolve()),
        "weights_sha256": pin["weights_sha256"],
    }
    return model, provenance


@torch.inference_mode()
def extract_features(
    rows: Sequence[dict[str, Any]], device: torch.device, batch_size: int, num_workers: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    backbone, provenance = load_backbone(device)
    loader = DataLoader(
        StructuralDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    outputs: list[torch.Tensor] = []
    for batch in loader:
        actual, views, channels, height, width = batch.shape
        flattened = batch.reshape(actual * views, channels, height, width).to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = backbone(flattened)
        else:
            features = backbone(flattened)
        if not isinstance(features, torch.Tensor) or features.shape != (actual * views, FEATURE_DIM):
            raise ValueError("DINOv2-S/14 returned an unexpected feature shape")
        features = features.float().reshape(actual, views, FEATURE_DIM).cpu()
        if not torch.isfinite(features).all():
            raise FloatingPointError("non-finite DINOv2 structural features")
        outputs.append(features.to(torch.float16))
    return torch.cat(outputs), provenance


def feature_cache_payload(rows: Sequence[dict[str, Any]], features: torch.Tensor) -> dict[str, Any]:
    return {
        "schema_version": "tlachia-multiview-p4a-features.v1",
        "features": features,
        "row_ids": [row["row_id"] for row in rows],
        "image_paths": [row["image_path"] for row in rows],
        "byte_sha256": [row["byte_sha256"] for row in rows],
        "pixel_sha256": [row["pixel_sha256"] for row in rows],
        "roles": [row["roles"] for row in rows],
        "source_groups": [row["source_group"] for row in rows],
        "descendant_class_names": [row["descendant_class_names"] for row in rows],
        "descendant_class_labels": [row["descendant_class_labels"] for row in rows],
        "splits": [row["split"] for row in rows],
        "view_names": list(VIEW_NAMES),
        "feature_dim": FEATURE_DIM,
        "promotion_eligible": False,
    }


def validate_feature_cache(
    cache: dict[str, Any],
    *,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_groups: int | None = EXPECTED_GROUPS,
    require_reviewed_paths: bool = True,
) -> None:
    metadata = (
        "row_ids",
        "image_paths",
        "byte_sha256",
        "pixel_sha256",
        "roles",
        "source_groups",
        "descendant_class_names",
        "descendant_class_labels",
        "splits",
    )
    if cache.get("schema_version") != "tlachia-multiview-p4a-features.v1":
        raise ValueError("structural feature cache schema changed")
    features = cache.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("structural feature tensor must be rank three")
    rows = int(features.shape[0])
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("structural feature row count changed")
    if tuple(features.shape[1:]) != (len(VIEW_NAMES), FEATURE_DIM) or features.dtype != torch.float16:
        raise ValueError("structural feature tensor view/dimension/dtype changed")
    if any(not isinstance(cache.get(key), list) or len(cache[key]) != rows for key in metadata):
        raise ValueError("structural feature metadata lengths differ")
    if cache.get("view_names") != list(VIEW_NAMES) or cache.get("feature_dim") != FEATURE_DIM:
        raise ValueError("structural cache view names or feature dimension changed")
    if cache.get("promotion_eligible") is not False:
        raise ValueError("structural research cache must remain non-promotable")
    if len(set(cache["row_ids"])) != rows:
        raise ValueError("structural cache row IDs are not unique")
    for raw_path in cache["image_paths"]:
        path = Path(str(raw_path))
        assert_allowed_input(path)
        if require_reviewed_paths:
            try:
                path.resolve().relative_to(IMAGES_ROOT.resolve())
            except ValueError as error:
                raise ValueError("structural cache image escapes the reviewed root") from error
    if any(not set(roles) or not set(roles) <= ALLOWED_ROLES for roles in cache["roles"]):
        raise ValueError("structural cache contains an unsupported role")
    if any(
        sorted(set(map(int, labels))) != list(map(int, labels))
        or any(label < 0 or label >= CLASS_COUNT for label in labels)
        for labels in cache["descendant_class_labels"]
    ):
        raise ValueError("structural descendant label sets are invalid")
    assert_group_isolation(cache["source_groups"], cache["splits"])
    if expected_groups is not None and len(set(cache["source_groups"])) != expected_groups:
        raise ValueError("structural feature source-group count changed")
    if len({g for g, split in zip(cache["source_groups"], cache["splits"]) if split == "structural_dev"}) != DEV_GROUP_COUNT:
        raise ValueError("structural development group count changed")
    if not torch.isfinite(features.float()).all():
        raise FloatingPointError("structural feature cache contains non-finite values")


def load_feature_cache() -> dict[str, Any]:
    cache = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise ValueError("structural feature cache is not a dictionary")
    validate_feature_cache(cache)
    sidecar = read_json(FEATURE_PROVENANCE)
    if sidecar.get("schema_version") != "tlachia-multiview-p4a-features.provenance.v1":
        raise ValueError("structural feature provenance schema changed")
    if sidecar.get("cache_sha256") != sha256_file(FEATURE_CACHE):
        raise ValueError("structural feature provenance does not bind the cache")
    if sidecar.get("row_ids_sha256") != semantic_hash(cache["row_ids"]):
        raise ValueError("structural feature row identity hash changed")
    expected_provenance = {
        "plan_payload_sha256": EXPECTED_PAYLOAD_SHA256["plan"],
        "manifest_payload_sha256": EXPECTED_PAYLOAD_SHA256["manifest"],
        "verification_payload_sha256": EXPECTED_PAYLOAD_SHA256["verification"],
        "element_cache_sha256": EXPECTED_FILE_SHA256[ELEMENT_CACHE],
        "view_seed": VIEW_SEED,
        "view_names": list(VIEW_NAMES),
        "safe_view_operations": list(SAFE_VIEW_OPERATIONS),
        "forbidden_view_operations": list(FORBIDDEN_VIEW_OPERATIONS),
        "business_or_locked_read": False,
        "promotion_eligible": False,
    }
    for key, expected in expected_provenance.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"structural feature provenance field changed: {key}")
    return cache


def assert_cache_matches_rows(cache: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    expected = feature_cache_payload(rows, cache["features"])
    for key in (
        "row_ids",
        "image_paths",
        "byte_sha256",
        "pixel_sha256",
        "roles",
        "source_groups",
        "descendant_class_names",
        "descendant_class_labels",
        "splits",
    ):
        if cache.get(key) != expected[key]:
            raise ValueError(f"structural feature cache differs from reviewed rows: {key}")


def projection_model(input_dim: int = FEATURE_DIM) -> nn.Module:
    sys.path.insert(0, str(REPO / "backend"))
    from codex_pipeline.models.projection_head import ProjectionHead

    return ProjectionHead(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)


def load_projection(path: Path, device: torch.device) -> nn.Module:
    assert_allowed_input(path)
    model = projection_model().to(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    return model


class VICRegExpander(nn.Module):
    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size, other = matrix.shape
    if size != other:
        raise ValueError("covariance matrix must be square")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(first: torch.Tensor, second: torch.Tensor) -> dict[str, torch.Tensor]:
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2:
        raise ValueError("VICReg expects equal 2D batches with at least two rows")
    invariance = F.mse_loss(first, second)
    centered_first = first - first.mean(dim=0)
    centered_second = second - second.mean(dim=0)
    std_first = torch.sqrt(first.var(dim=0, unbiased=False) + 1e-4)
    std_second = torch.sqrt(second.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (F.relu(1.0 - std_first).mean() + F.relu(1.0 - std_second).mean())
    divisor = max(1, first.shape[0] - 1)
    covariance = (
        off_diagonal(centered_first.T @ centered_first / divisor).pow(2).sum() / first.shape[1]
        + off_diagonal(centered_second.T @ centered_second / divisor).pow(2).sum() / first.shape[1]
    )
    total = 25.0 * invariance + 25.0 * variance + covariance
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite VICReg loss")
    return {
        "total": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "minimum_std": torch.minimum(std_first.min(), std_second.min()).detach(),
    }


def eligible_indices(cache: dict[str, Any], spec: Spec) -> list[int]:
    required = set(spec.roles)
    return [
        index
        for index, (roles, split) in enumerate(zip(cache["roles"], cache["splits"]))
        if split == "structural_train" and set(roles) & required
    ]


def view_indices(view_set: str) -> tuple[int, ...]:
    mapping = {"mild": (1, 2), "strong": (3, 4), "both": (1, 2, 3, 4)}
    if view_set not in mapping:
        raise ValueError(f"unsupported frozen view set: {view_set}")
    return mapping[view_set]


def _primary_role(roles: Sequence[str]) -> str:
    for role in ("glyph_occurrence", "zone", "planche"):
        if role in roles:
            return role
    raise ValueError("structural row has no supported role")


def build_ssl_plan(cache: dict[str, Any], spec: Spec, seed: int) -> list[dict[str, Any]]:
    indices = eligible_indices(cache, spec)
    if len(indices) < 2:
        raise ValueError(f"insufficient eligible structural train rows for {spec.name}")
    allowed_views = view_indices(spec.views)
    by_role: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_role[_primary_role(cache["roles"][index])].append(index)
    result: list[dict[str, Any]] = []
    for epoch in range(spec.epochs):
        rng = random.Random(stable_seed("p4a-ssl-plan", spec.name, seed, epoch))
        if spec.balanced_roles:
            roles = sorted(by_role)
            epoch_indices = [rng.choice(by_role[roles[position % len(roles)]]) for position in range(len(indices))]
            rng.shuffle(epoch_indices)
        else:
            epoch_indices = list(indices)
            rng.shuffle(epoch_indices)
        for index in epoch_indices:
            first, second = rng.sample(allowed_views, 2)
            result.append(
                {
                    "epoch": epoch,
                    "row_index": index,
                    "row_id": cache["row_ids"][index],
                    "first_view": first,
                    "second_view": second,
                    "first_view_name": VIEW_NAMES[first],
                    "second_view_name": VIEW_NAMES[second],
                    "primary_role": _primary_role(cache["roles"][index]),
                }
            )
    return result


def element_partition(cache: dict[str, Any]) -> dict[str, Any]:
    validate_element_cache(cache)
    train_idx = [i for i, split in enumerate(cache["dataset_splits"]) if split == "train"]
    dev_idx = [i for i, split in enumerate(cache["dataset_splits"]) if split == "dev"]
    return {
        "train_features": cache["features"][train_idx],
        "train_labels": cache["labels"][train_idx].long(),
        "train_row_ids": [cache["row_ids"][i] for i in train_idx],
        "dev_features": cache["features"][dev_idx],
        "dev_labels": cache["labels"][dev_idx].long(),
        "dev_row_ids": [cache["row_ids"][i] for i in dev_idx],
        "class_names": {int(key): str(value) for key, value in cache["class_names"].items()},
    }


def build_supervised_plan(element: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(element["train_labels"].tolist()):
        by_class[int(label)].append(index)
    classes = sorted(label for label, rows in by_class.items() if len(rows) >= 8)
    if len(classes) < SUPERVISED_REFRESH["n_way"]:
        raise ValueError("insufficient element classes for fixed supervised refresh")
    rng = random.Random(stable_seed("p4a-supervised-refresh", seed))
    plan: list[dict[str, Any]] = []
    for episode in range(SUPERVISED_REFRESH["episodes"]):
        selected = rng.sample(classes, SUPERVISED_REFRESH["n_way"])
        support: list[int] = []
        query: list[int] = []
        for label in selected:
            rows = rng.sample(
                by_class[label], SUPERVISED_REFRESH["k_shot"] + SUPERVISED_REFRESH["q_queries"]
            )
            support.extend(rows[: SUPERVISED_REFRESH["k_shot"]])
            query.extend(rows[SUPERVISED_REFRESH["k_shot"] :])
        if set(support) & set(query):
            raise AssertionError("supervised support/query row overlap")
        plan.append(
            {
                "episode": episode,
                "class_labels": selected,
                "support": support,
                "query": query,
                "support_row_ids": [element["train_row_ids"][i] for i in support],
                "query_row_ids": [element["train_row_ids"][i] for i in query],
            }
        )
    return plan


def train_vicreg(
    model: nn.Module,
    cache: dict[str, Any],
    spec: Spec,
    seed: int,
    plan: Sequence[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, float | int]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    expander = VICRegExpander().to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(expander.parameters()),
        lr=spec.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    features = cache["features"]
    logs: list[dict[str, float | int]] = []
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        by_epoch[int(row["epoch"])].append(row)
    for epoch in range(spec.epochs):
        epoch_terms: list[float] = []
        rows = by_epoch[epoch]
        for offset in range(0, len(rows), VICREG_BATCH_SIZE):
            batch = rows[offset : offset + VICREG_BATCH_SIZE]
            if len(batch) < 2:
                continue
            first = torch.stack(
                [features[row["row_index"], row["first_view"]] for row in batch]
            ).to(device=device, dtype=torch.float32)
            second = torch.stack(
                [features[row["row_index"], row["second_view"]] for row in batch]
            ).to(device=device, dtype=torch.float32)
            terms = vicreg_loss(expander(model.net(first)), expander(model.net(second)))
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(expander.parameters()), 5.0)
            optimizer.step()
            epoch_terms.append(float(terms["total"].detach().cpu()))
        if not epoch_terms:
            raise RuntimeError("VICReg produced no training batch")
        logs.append({"stage": 1, "epoch": epoch + 1, "vicreg_loss": sum(epoch_terms) / len(epoch_terms)})
    return logs


def supervised_refresh(
    model: nn.Module,
    element: dict[str, Any],
    plan: Sequence[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, float | int]]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SUPERVISED_REFRESH["learning_rate"],
        weight_decay=SUPERVISED_REFRESH["weight_decay"],
    )
    features = element["train_features"]
    logs: list[dict[str, float | int]] = []
    for row in plan:
        support = features[row["support"]].to(device)
        query = features[row["query"]].to(device)
        n_way = len(row["class_labels"])
        support_labels = torch.arange(n_way, device=device).repeat_interleave(SUPERVISED_REFRESH["k_shot"])
        query_labels = torch.arange(n_way, device=device).repeat_interleave(SUPERVISED_REFRESH["q_queries"])
        support_embeddings = model(support)
        query_embeddings = model(query)
        prototypes = torch.stack(
            [F.normalize(support_embeddings[support_labels == label].mean(dim=0), dim=0) for label in range(n_way)]
        )
        logits = query_embeddings @ prototypes.T / SUPERVISED_REFRESH["temperature"]
        loss = F.cross_entropy(logits, query_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (int(row["episode"]) + 1) % 20 == 0:
            logs.append(
                {
                    "stage": 2,
                    "episode": int(row["episode"]) + 1,
                    "supervised_loss": float(loss.detach().cpu()),
                    "supervised_accuracy": float((logits.argmax(dim=1) == query_labels).float().mean().cpu()),
                }
            )
    return logs


def sample_average_precision(scores: Sequence[float], positives: Iterable[int]) -> float:
    positive_set = {int(value) for value in positives}
    if not positive_set:
        raise ValueError("sample AP requires at least one positive class")
    if any(value < 0 or value >= len(scores) for value in positive_set):
        raise ValueError("sample AP positive class is outside the score vector")
    ranked = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    hits = 0
    total = 0.0
    for rank, label in enumerate(ranked, start=1):
        if label in positive_set:
            hits += 1
            total += hits / rank
    return total / len(positive_set)


def structural_metrics(
    scores: torch.Tensor, label_sets: Sequence[Sequence[int]], groups: Sequence[str]
) -> dict[str, Any]:
    if scores.ndim != 2 or scores.shape[1] != CLASS_COUNT:
        raise ValueError("structural scores must cover all 286 runtime classes")
    if scores.shape[0] != len(label_sets) or len(label_sets) != len(groups):
        raise ValueError("structural score metadata lengths differ")
    sample_ap: list[float] = []
    recall5: list[float] = []
    per_group_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row_scores, labels, group in zip(scores.tolist(), label_sets, groups):
        ap = sample_average_precision(row_scores, labels)
        top5 = set(sorted(range(CLASS_COUNT), key=lambda i: (-row_scores[i], i))[:5])
        recall = float(bool(top5 & set(labels)))
        sample_ap.append(ap)
        recall5.append(recall)
        per_group_values[str(group)].append((ap, recall))
    per_group = {
        group: {
            "samples": len(values),
            "sample_ap": sum(value[0] for value in values) / len(values),
            "recall_at_5": sum(value[1] for value in values) / len(values),
        }
        for group, values in sorted(per_group_values.items())
    }
    return {
        "samples": len(sample_ap),
        "groups": len(per_group),
        "sample_ap": sum(sample_ap) / len(sample_ap),
        "recall_at_5": sum(recall5) / len(recall5),
        "macro_planche_ap": sum(value["sample_ap"] for value in per_group.values()) / len(per_group),
        "per_group": per_group,
    }


@torch.inference_mode()
def evaluate(model: nn.Module, element: dict[str, Any], structural: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model.eval()
    train_features = element["train_features"].to(device)
    dev_features = element["dev_features"].to(device)
    train_labels = element["train_labels"]
    dev_labels = element["dev_labels"]
    train_embeddings = model(train_features).cpu()
    dev_embeddings = model(dev_features).cpu()
    prototype_labels = train_labels.unique(sorted=True)
    if prototype_labels.tolist() != list(range(CLASS_COUNT)):
        raise ValueError("element train does not provide all 286 runtime prototypes")
    prototypes = torch.stack(
        [F.normalize(train_embeddings[train_labels == label].mean(dim=0), dim=0) for label in prototype_labels]
    )
    element_logits = dev_embeddings @ prototypes.T / TEMPERATURE
    predictions = element_logits.argmax(dim=1)
    correct = predictions.eq(dev_labels)
    losses = F.cross_entropy(element_logits, dev_labels, reduction="none")
    per_class: dict[str, dict[str, float | int]] = {}
    for label in dev_labels.unique(sorted=True).tolist():
        mask = dev_labels.eq(label)
        per_class[str(label)] = {
            "samples": int(mask.sum()),
            "accuracy": float(correct[mask].float().mean()),
            "logloss": float(losses[mask].mean()),
        }
    element_result = {
        "micro": float(correct.float().mean()),
        "macro": sum(float(row["accuracy"]) for row in per_class.values()) / len(per_class),
        "balanced_logloss": sum(float(row["logloss"]) for row in per_class.values()) / len(per_class),
        "per_class": per_class,
    }

    indices = [
        index
        for index, (roles, split, labels) in enumerate(
            zip(structural["roles"], structural["splits"], structural["descendant_class_labels"])
        )
        if split == "structural_dev" and "glyph_occurrence" in roles and labels
    ]
    if not indices:
        raise ValueError("structural development split has no glyph descendant evidence")
    base_features = structural["features"][indices, 0].to(device=device, dtype=torch.float32)
    structural_scores = model(base_features).cpu() @ prototypes.T
    structural_result = structural_metrics(
        structural_scores,
        [structural["descendant_class_labels"][i] for i in indices],
        [structural["source_groups"][i] for i in indices],
    )
    return {"element_dev": element_result, "structural_dev": structural_result}


def paired_planche_bootstrap(
    candidate_fresh: Sequence[dict[str, Any]],
    control_fresh: Sequence[dict[str, Any]],
    *,
    replicates: int = 4000,
    seed: int = 20260813,
) -> dict[str, float | int]:
    """Hierarchical paired bootstrap over fresh seed, then planche group.

    Resampling the seed first preserves seed-level dependence.  Resampling
    planche groups inside that seed preserves the paired candidate/control
    comparison and prevents the screening seed from entering the interval.
    """
    if len(candidate_fresh) != len(CONFIRMATION_SEEDS) or len(control_fresh) != len(
        CONFIRMATION_SEEDS
    ):
        raise ValueError("paired planche bootstrap requires exactly three fresh seed pairs")
    deltas_by_seed: list[list[float]] = []
    expected_groups: tuple[str, ...] | None = None
    for candidate, control in zip(candidate_fresh, control_fresh):
        candidate_groups = candidate["structural_dev"]["per_group"]
        control_groups = control["structural_dev"]["per_group"]
        groups = tuple(sorted(candidate_groups))
        if not groups or groups != tuple(sorted(control_groups)):
            raise ValueError("paired planche bootstrap requires identical non-empty groups")
        if expected_groups is None:
            expected_groups = groups
        elif groups != expected_groups:
            raise ValueError("fresh confirmation seeds must contain identical planche groups")
        deltas = [
            float(candidate_groups[group]["sample_ap"])
            - float(control_groups[group]["sample_ap"])
            for group in groups
        ]
        observed = sum(deltas) / len(deltas)
        macro_delta = float(candidate["structural_dev"]["macro_planche_ap"]) - float(
            control["structural_dev"]["macro_planche_ap"]
        )
        if not math.isclose(observed, macro_delta, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("macro planche AP differs from its per-group evidence")
        deltas_by_seed.append(deltas)
    if expected_groups is None:  # pragma: no cover - guarded above
        raise AssertionError("missing fresh planche groups")
    observed_mean = sum(sum(values) / len(values) for values in deltas_by_seed) / len(
        deltas_by_seed
    )
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled_seed_means: list[float] = []
        for _seed_draw in range(len(deltas_by_seed)):
            seed_values = rng.choice(deltas_by_seed)
            sampled_seed_means.append(
                sum(rng.choice(seed_values) for _ in range(len(seed_values))) / len(seed_values)
            )
        samples.append(sum(sampled_seed_means) / len(sampled_seed_means))
    samples.sort()
    return {
        "seeds": len(deltas_by_seed),
        "groups_per_seed": len(expected_groups),
        "paired_seed_group_units": len(deltas_by_seed) * len(expected_groups),
        "replicates": replicates,
        "mean": observed_mean,
        "lower_95": samples[math.floor(0.025 * replicates)],
        "upper_95": samples[min(replicates - 1, math.floor(0.975 * replicates))],
    }


def final_gate(
    candidate_fresh: Sequence[dict[str, Any]],
    control_fresh: Sequence[dict[str, Any]],
    decision_seeds: Sequence[int],
) -> dict[str, Any]:
    """Decide only from preregistered fresh confirmations, never seed 42."""
    if tuple(decision_seeds) != CONFIRMATION_SEEDS:
        raise ValueError("final decision requires preregistered seeds 73, 101, and 202")
    if len(candidate_fresh) != 3 or len(control_fresh) != 3:
        raise ValueError("final decision requires exactly three fresh metric pairs")
    bootstrap = paired_planche_bootstrap(candidate_fresh, control_fresh)
    fresh_primary_deltas = [
        float(candidate["structural_dev"]["macro_planche_ap"])
        - float(control["structural_dev"]["macro_planche_ap"])
        for candidate, control in zip(candidate_fresh, control_fresh)
    ]
    primary_gain = sum(fresh_primary_deltas) / len(fresh_primary_deltas)
    if not math.isclose(primary_gain, float(bootstrap["mean"]), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("fresh primary gain differs from paired seed/group evidence")
    candidate_element_macro = sum(
        float(metrics["element_dev"]["macro"]) for metrics in candidate_fresh
    ) / len(candidate_fresh)
    control_element_macro = sum(
        float(metrics["element_dev"]["macro"]) for metrics in control_fresh
    ) / len(control_fresh)
    class_sets = [set(metrics["element_dev"]["per_class"]) for metrics in candidate_fresh]
    class_sets.extend(set(metrics["element_dev"]["per_class"]) for metrics in control_fresh)
    if not class_sets or any(labels != class_sets[0] for labels in class_sets[1:]):
        raise ValueError("fresh element confirmations must contain identical observed classes")
    class_drops: list[str] = []
    per_class_fresh: dict[str, dict[str, float]] = {}
    for label in sorted(class_sets[0], key=int):
        candidate_mean = sum(
            float(metrics["element_dev"]["per_class"][label]["accuracy"])
            for metrics in candidate_fresh
        ) / len(candidate_fresh)
        control_mean = sum(
            float(metrics["element_dev"]["per_class"][label]["accuracy"])
            for metrics in control_fresh
        ) / len(control_fresh)
        delta = candidate_mean - control_mean
        per_class_fresh[label] = {
            "candidate_mean": candidate_mean,
            "control_mean": control_mean,
            "delta": delta,
        }
        if delta < -0.20:
            class_drops.append(label)
    passes = {
        "primary_gain_at_least_0_01": primary_gain >= 0.01,
        "paired_planche_bootstrap_lower_95_positive": float(bootstrap["lower_95"]) > 0.0,
        "element_macro_not_more_than_0_005_below_control": candidate_element_macro
        >= control_element_macro - 0.005,
        "no_observed_element_class_drop_over_0_20": not class_drops,
        "at_least_two_of_three_fresh_seed_gains_positive": sum(
            delta > 0.0 for delta in fresh_primary_deltas
        )
        >= 2,
    }
    return {
        "pass": all(passes.values()),
        "passes": passes,
        "decision_source": "fresh_confirmation_only",
        "decision_seeds": list(decision_seeds),
        "screen_seed_used_for_decision": False,
        "primary_gain": primary_gain,
        "paired_planche_bootstrap": bootstrap,
        "fresh_seed_deltas": fresh_primary_deltas,
        "fresh_element_macro": {
            "candidate_mean": candidate_element_macro,
            "control_mean": control_element_macro,
            "delta": candidate_element_macro - control_element_macro,
        },
        "fresh_per_class": per_class_fresh,
        "class_drops_over_0_20": class_drops,
        "promotion_eligible": False,
    }


def frozen_protocol() -> dict[str, Any]:
    assert_specs()
    return {
        "protocol": "tlachia-multiview-p4a.v1",
        "inputs": {
            str(path.relative_to(REPO)): {"sha256": digest}
            for path, digest in EXPECTED_FILE_SHA256.items()
        },
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "structural_split": {
            "method": "six whole planche groups; greedy max new descendant-class evidence then stable hash",
            "dev_groups": DEV_GROUP_COUNT,
            "train_groups": EXPECTED_GROUPS - DEV_GROUP_COUNT,
        },
        "features": {
            "backbone": "local-pinned dinov2_vits14",
            "shape": [EXPECTED_ROWS, len(VIEW_NAMES), FEATURE_DIM],
            "dtype": "float16",
            "views": list(VIEW_NAMES),
            "safe_operations": list(SAFE_VIEW_OPERATIONS),
            "forbidden_operations": list(FORBIDDEN_VIEW_OPERATIONS),
        },
        "specs": [spec_record(spec) for spec in SPECS],
        "screen_seed": SCREEN_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selection": {
            "primary": "structural dev macro_planche_AP on glyph descendant evidence",
            "top_challengers": 3,
            "screening_seed": SCREEN_SEED,
            "screening_use": "candidate selection only; excluded from every final gate statistic",
            "screening_element_guard": "element macro >= historical control - 0.005",
            "final_decision_seeds": list(CONFIRMATION_SEEDS),
            "final_aggregation": "mean fresh seed deltas; hierarchical paired bootstrap resampling seed then planche groups",
            "gate": "fresh-only gain>=.01; fresh hierarchical paired planche bootstrap lower95>0; fresh mean element macro guard; no fresh mean class drop>.20; >=2/3 fresh positive",
        },
        "supervised_refresh": SUPERVISED_REFRESH,
        "forbidden_inputs": list(FORBIDDEN_PATH_TOKENS),
        "promotion_eligible": False,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }


def prepare(device_name: str = "auto", batch_size: int = 16, num_workers: int = 0) -> dict[str, Any]:
    plan, manifest, verification = validate_reviewed_inputs()
    element_cache = load_element_cache()
    protocol = frozen_protocol()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_once_json(REPORT_DIR / "frozen-manifest.json", protocol)
    write_once_json(
        REPORT_DIR / "frozen-specs.json",
        {
            "protocol": protocol["protocol"],
            "specs": protocol["specs"],
            "screen_seed": SCREEN_SEED,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "sha256": semantic_hash(protocol["specs"]),
        },
    )
    rows = build_rows(
        plan,
        manifest,
        {int(key): str(value) for key, value in element_cache["class_names"].items()},
    )
    if not FEATURE_CACHE.exists():
        device = resolve_device(device_name)
        features, backbone = extract_features(rows, device, batch_size, num_workers)
        cache = feature_cache_payload(rows, features)
        validate_feature_cache(cache)
        publish_feature_bundle(
            cache,
            {
                "schema_version": "tlachia-multiview-p4a-features.provenance.v1",
                "row_ids_sha256": semantic_hash(cache["row_ids"]),
                "source_groups_sha256": semantic_hash(cache["source_groups"]),
                "split_sha256": semantic_hash(cache["splits"]),
                "descendant_class_labels_sha256": semantic_hash(cache["descendant_class_labels"]),
                "plan_payload_sha256": plan["payload_sha256"],
                "manifest_payload_sha256": manifest["payload_sha256"],
                "verification_payload_sha256": verification["payload_sha256"],
                "element_cache_sha256": sha256_file(ELEMENT_CACHE),
                "backbone": backbone,
                "view_seed": VIEW_SEED,
                "view_names": list(VIEW_NAMES),
                "safe_view_operations": list(SAFE_VIEW_OPERATIONS),
                "forbidden_view_operations": list(FORBIDDEN_VIEW_OPERATIONS),
                "business_or_locked_read": False,
                "promotion_eligible": False,
            },
        )
    cache = load_feature_cache()
    assert_cache_matches_rows(cache, rows)
    return {
        "ok": True,
        "protocol": protocol["protocol"],
        "feature_cache": str(FEATURE_CACHE),
        "feature_cache_sha256": sha256_file(FEATURE_CACHE),
        "rows": len(cache["row_ids"]),
        "train_groups": len(
            {g for g, split in zip(cache["source_groups"], cache["splits"]) if split == "structural_train"}
        ),
        "dev_groups": len(
            {g for g, split in zip(cache["source_groups"], cache["splits"]) if split == "structural_dev"}
        ),
        "promotion_eligible": False,
    }


def _run_dir(phase: str, spec: Spec, seed: int) -> Path:
    return REPORT_DIR / "runs" / f"{phase}-{spec.name}-seed{seed}"


def run_one(
    spec: Spec,
    seed: int,
    phase: str,
    element: dict[str, Any],
    structural: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    run_dir = _run_dir(phase, spec, seed)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return read_json(result_path)
    config = {
        "phase": phase,
        "seed": seed,
        "spec": spec_record(spec),
        "initializer_sha256": EXPECTED_FILE_SHA256[
            B1_PROJECTION if spec.initializer == "b1" else HISTORICAL_PROJECTION
        ],
        "feature_cache_sha256": sha256_file(FEATURE_CACHE),
        "promotion_eligible": False,
    }
    write_once_json(run_dir / "config.json", config)
    started = time.time()
    initial = B1_PROJECTION if spec.initializer == "b1" else HISTORICAL_PROJECTION
    model = load_projection(initial, device)
    ssl_plan: list[dict[str, Any]] = []
    supervised_plan: list[dict[str, Any]] = []
    if spec.kind != "control":
        ssl_plan = build_ssl_plan(structural, spec, seed)
    if spec.supervised_refresh:
        supervised_plan = build_supervised_plan(element, seed)
    write_once_json(
        run_dir / "training-plan.json",
        {
            "ssl": ssl_plan,
            "supervised_refresh": supervised_plan,
            "element_train_only": True,
            "structural_train_only": True,
            "plan_sha256": semantic_hash({"ssl": ssl_plan, "supervised_refresh": supervised_plan}),
        },
    )
    logs: list[dict[str, float | int]] = []
    if ssl_plan:
        logs.extend(train_vicreg(model, structural, spec, seed, ssl_plan, device))
    if supervised_plan:
        logs.extend(supervised_refresh(model, element, supervised_plan, device))
    checkpoint = run_dir / "checkpoints/latest.pt"
    write_once_torch(
        checkpoint,
        {
            "model_state_dict": copy.deepcopy(model.cpu().state_dict()),
            "config": config,
            "training_plan_sha256": sha256_file(run_dir / "training-plan.json"),
            "promotion_eligible": False,
        },
    )
    model = model.to(device)
    metrics = evaluate(model, element, structural, device)
    write_once_json(run_dir / "epoch-log.json", logs)
    write_once_json(
        run_dir / "invocation.json",
        {
            "argv": [sys.executable, str(Path(__file__).resolve()), "run-all"],
            "phase": phase,
            "seed": seed,
            "spec": spec.name,
        },
    )
    result = {
        "phase": phase,
        "seed": seed,
        "spec": spec_record(spec),
        "metrics": metrics,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "elapsed_seconds": time.time() - started,
        "promotion_eligible": False,
    }
    write_once_json(result_path, result)
    write_once_json(
        run_dir / "run-log.json",
        {"status": "completed", "elapsed_seconds": result["elapsed_seconds"], "result_sha256": sha256_file(result_path)},
    )
    return result


def run_all(device_name: str = "auto") -> dict[str, Any]:
    prepare(device_name=device_name)
    assert_specs()
    element = element_partition(load_element_cache())
    structural = load_feature_cache()
    device = resolve_device(device_name)
    screening = [run_one(spec, SCREEN_SEED, "screening", element, structural, device) for spec in SPECS]
    control = screening[0]
    eligible = [
        result
        for result in screening[2:]
        if float(result["metrics"]["element_dev"]["macro"])
        >= float(control["metrics"]["element_dev"]["macro"]) - 0.005
    ]
    selected_results = sorted(
        eligible,
        key=lambda result: (
            -float(result["metrics"]["structural_dev"]["macro_planche_ap"]),
            result["spec"]["name"],
        ),
    )[:3]
    if len(selected_results) != 3:
        raise RuntimeError("fewer than three challengers satisfy the element macro screening guard")
    selected = [next(spec for spec in SPECS if spec.name == result["spec"]["name"]) for result in selected_results]
    confirmations: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        confirmations.append(run_one(SPECS[0], seed, "confirmation", element, structural, device))
        for spec in selected:
            confirmations.append(run_one(spec, seed, "confirmation", element, structural, device))
    gates: list[dict[str, Any]] = []
    for candidate in selected_results:
        name = candidate["spec"]["name"]
        candidate_fresh = [
            next(
                row
                for row in confirmations
                if row["seed"] == seed and row["spec"]["name"] == name
            )["metrics"]
            for seed in CONFIRMATION_SEEDS
        ]
        control_fresh = [
            next(
                row
                for row in confirmations
                if row["seed"] == seed and row["spec"]["name"] == "historical-control"
            )["metrics"]
            for seed in CONFIRMATION_SEEDS
        ]
        gates.append(
            {
                "name": name,
                "selection_seed": SCREEN_SEED,
                "selection_primary": float(
                    candidate["metrics"]["structural_dev"]["macro_planche_ap"]
                ),
                **final_gate(candidate_fresh, control_fresh, CONFIRMATION_SEEDS),
            }
        )
    result = {
        "protocol": "tlachia-multiview-p4a.v1",
        "screening": screening,
        "selected_challengers": [spec.name for spec in selected],
        "confirmations": confirmations,
        "gates": gates,
        "accepted_candidates": [gate["name"] for gate in gates if gate["pass"]],
        "business_or_locked_read": False,
        "promotion_eligible": False,
    }
    write_once_json(REPORT_DIR / "results.json", result)
    write_once_json(
        REPORT_DIR / "final.json",
        {
            "protocol": result["protocol"],
            "results_sha256": sha256_file(REPORT_DIR / "results.json"),
            "accepted_candidates": result["accepted_candidates"],
            "gate_passed": bool(result["accepted_candidates"]),
            "business_or_locked_read": False,
            "promotion_eligible": False,
        },
    )
    return result


def verify() -> dict[str, Any]:
    prepared = prepare(device_name="cpu")
    assert_specs()
    structural = load_feature_cache()
    element = load_element_cache()
    validate_feature_cache(structural)
    validate_element_cache(element)
    protocol = read_json(REPORT_DIR / "frozen-manifest.json")
    frozen_specs = read_json(REPORT_DIR / "frozen-specs.json")
    if protocol != frozen_protocol():
        raise ValueError("frozen protocol differs from the current runner")
    if frozen_specs.get("specs") != [spec_record(spec) for spec in SPECS]:
        raise ValueError("frozen specifications differ from the preregistration")
    result_path = REPORT_DIR / "results.json"
    campaign: dict[str, Any] = {"completed": False}
    if result_path.exists():
        results = read_json(result_path)
        if len(results.get("screening", ())) != 16:
            raise ValueError("completed campaign does not contain 16 screening runs")
        if len(results.get("confirmations", ())) != 12:
            raise ValueError("completed campaign does not contain control plus top-three across three seeds")
        if len(results.get("gates", ())) != 3 or results.get("promotion_eligible") is not False:
            raise ValueError("completed campaign final gate shape changed")
        for row in results["screening"] + results["confirmations"]:
            checkpoint = Path(row["checkpoint"]["path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != row["checkpoint"]["sha256"]:
                raise ValueError("campaign result checkpoint integrity failed")
        campaign = {
            "completed": True,
            "results_sha256": sha256_file(result_path),
            "accepted_candidates": results.get("accepted_candidates", []),
        }
    return {
        "ok": True,
        "prepared": prepared,
        "frozen_specs_sha256": frozen_specs["sha256"],
        "campaign": campaign,
        "business_or_locked_read": False,
        "promotion_eligible": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="validate inputs and extract immutable features")
    prepare_parser.add_argument("--device", default="auto")
    prepare_parser.add_argument("--batch-size", type=int, default=16)
    prepare_parser.add_argument("--num-workers", type=int, default=0)
    run_parser = subparsers.add_parser("run-all", help="execute the frozen screening and confirmation protocol")
    run_parser.add_argument("--device", default="auto")
    subparsers.add_parser("verify", help="verify frozen inputs, cache, and any completed campaign")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.device, args.batch_size, args.num_workers)
        elif args.command == "run-all":
            result = run_all(args.device)
        else:
            result = verify()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

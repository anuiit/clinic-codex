#!/usr/bin/env python3
"""Research-only Tlachia glyph fusion spike for run_50 EfficientNet-B3 + B1.

The harness has three deliberately separate phases:

``verify``
    Re-hash and validate every reviewed input.  No model is loaded and no
    output is written.
``prepare --apply``
    Run the real run_50 model on glyph JPGs and the real B1 classifier on
    MobileSAM crops, then publish one immutable score cache.
``run-all``
    Evaluate the frozen score-only variants and publish immutable results.

This is a bounded glyph-level sanity experiment.  It never reads business,
annotation, or locked-test inputs; never writes to the historical project or
the model registry; never trains or promotes a model; and its terminal verdict
is always ``INSUFFICIENT_FOR_ELEMENT_GATE`` until a larger fresh element-level
benchmark exists.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"

P4_PLAN = REPO / "reports/p4-multiview-20260813/plans/pilot-v1.json"
P4_MANIFEST = REPO / "reports/p4-multiview-20260813/download/pilot-v1.json"
P4_VERIFICATION = (
    REPO / "reports/p4-multiview-20260813/download/pilot-v1-verification.json"
)
P4_IMAGES = REPO / "backend/training_corpus/external/tlachia-multiview-pilot-v1"

LEGACY_ROOT = Path("/mnt/f/CODEX/AI_clinic_class")
RUN50_CHECKPOINT = LEGACY_ROOT / "runs/run_50/best_model.pt"
RUN50_CONFIG = LEGACY_ROOT / "runs/run_50/config.json"
RUN50_MODEL_SOURCE = LEGACY_ROOT / "src/models/model.py"
RUN50_LABELS = LEGACY_ROOT / "app/backend/labels.json"
RUN50_TRANSFORMS = LEGACY_ROOT / "app/backend/transforms.py"
RUN50_TRAINING_CSV = LEGACY_ROOT / "data_csv/training_data.csv"

B1_RUNTIME = (
    REPO
    / "backend/model_registry/versions/20260811-s14-snapshot-warmstart-b1/runtime"
)
B1_CONFIG = B1_RUNTIME / "config.json"
B1_PROJECTION = B1_RUNTIME / "weights/projection.pt"
B1_PROTOTYPES = B1_RUNTIME / "weights/prototypes.pt"
B1_PACKAGE_MANIFEST = B1_RUNTIME.parent / "manifest.json"
B1_PACKAGE_CHECKSUMS = B1_RUNTIME.parent / "checksums.sha256"
B1_BACKBONE_PIN = REPO / "backend/training_corpus/backbone-pins/dinov2-vits14-local.json"
MOBILE_SAM_CHECKPOINT = Path("/home/sina/.cache/mobile_sam/mobile_sam.pt")
B1_CLASSIFIER_SOURCE = REPO / "backend/codex_model/classifier.py"
B1_CLASSIFIER_INIT = REPO / "backend/codex_model/__init__.py"
MOBILE_SAM_SOURCE = REPO / "backend/codex_pipeline/segmentation/mobilesam.py"
MOBILE_SAM_INIT = REPO / "backend/codex_pipeline/segmentation/__init__.py"
P4_FEATURE_CACHE = (
    REPO
    / "autoresearch/orchestrator-260813-1330/tlachia-multiview-p4a-v1/features/multiview.features.pt"
)
P4_FEATURE_PROVENANCE = Path(str(P4_FEATURE_CACHE) + ".prov.json")

REPORT_DIR = REPO / "autoresearch/legacy-fusion-20260813"
PREPARED_DIR = REPORT_DIR / "prepared"
PROFILE_WHOLE_GLYPH_DIRECT_B1 = "whole_glyph_direct_b1"
PROFILE_WHOLE_GLYPH_CACHED_B1 = "whole_glyph_cached_b1"
PROFILE_MOBILE_SAM_B1 = "mobile_sam_b1"
SCORING_PROFILES = (
    PROFILE_WHOLE_GLYPH_DIRECT_B1,
    PROFILE_WHOLE_GLYPH_CACHED_B1,
    PROFILE_MOBILE_SAM_B1,
)
DEFAULT_SCORING_PROFILE = PROFILE_WHOLE_GLYPH_DIRECT_B1
CACHE_PATH = PREPARED_DIR / DEFAULT_SCORING_PROFILE / "glyph-score-cache.npz"
CACHE_PROVENANCE = CACHE_PATH.with_name("manifest.json")
RESULTS_PATH = REPORT_DIR / "results.json"
FINAL_PATH = REPORT_DIR / "final.json"

RUN50_CHECKPOINT_SHA256 = (
    "9cfd73f37655bce2773f5b45b90bed66415144c16db72e0560a76685e95eefcd"
)
EXPECTED_FILE_SHA256 = {
    P4_PLAN: "4db90b3f2829d55100d049acfa5f104f1b4883dc23ee180d6587aa83b7564fe8",
    P4_MANIFEST: "b3b6a53b985924327cc2c299c6f3c2d4b76bf936a1c8547262fcc3d7f82a66cc",
    P4_VERIFICATION: "e56b7b19edb7e8c43d1f91ad52859cbfa3d43c9c4cdf16b5e25a5d7f155d72dd",
    RUN50_CHECKPOINT: RUN50_CHECKPOINT_SHA256,
    RUN50_CONFIG: "f91040ff0e062b34ef60fc3ae429c81b39ecb869dd8fd404a8c94e806c0ce245",
    RUN50_MODEL_SOURCE: "e221c7b77c3478bb337cd8e36371742ed36de3a5c86222d86109715d91d3e65f",
    RUN50_LABELS: "6231a929786aa50b15a6610311355364cbebc707476ebe9391068b5e465470be",
    RUN50_TRANSFORMS: "d5d48b8d243c82aaa1eaa849b213dc79cc6ac6b128cdf9b80eb3f800cabb0456",
    RUN50_TRAINING_CSV: "c94ae6e51eb3cc6a969ca317c8f2e46cc8855b252f88a910e71090dbd6aa9da6",
    B1_CONFIG: "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
    B1_PROJECTION: "52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03",
    B1_PROTOTYPES: "55f9ccf2f3d49389502e8f0667ec6b7de2599b1f7da2172db4f827d86b59f62c",
    B1_PACKAGE_MANIFEST: "1d09df51ae469fd44a24ebfc1b7dcdf66df49baa74e8c61981b57418a3f1b165",
    B1_PACKAGE_CHECKSUMS: "aa90bd6522c6b5b4386e26a11b2ed508d01b14b856b2d7cfd6636cae1f9c717c",
    B1_BACKBONE_PIN: "f6cb3eb246ad907a7cab8c6f6d21d55ce8bafd9cbc3e402bb10660ca48cc9cc2",
    MOBILE_SAM_CHECKPOINT: "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    B1_CLASSIFIER_SOURCE: "db4eae131b2b32f68745e22744e36e297b6db7d7cbaccebbb7d050be24129f08",
    B1_CLASSIFIER_INIT: "c7c6a0bf95bc387f04967fc704b260b05f1ef83bad0ebe8806f41672d348bef2",
    MOBILE_SAM_SOURCE: "f71cd7bd5a0db742dfd95f517b4f13e96e955dc5261f4d4d7221ba62bdf2fd6f",
    MOBILE_SAM_INIT: "bf6c7388b7dfbdd53e4affab14959fe3b945b124cae0524ac83355c12ddc6c4a",
    P4_FEATURE_CACHE: "c2c3ecbd38e870ac55bbaf5f517b60f8d2489c11ccd5666dcddba351be641506",
    P4_FEATURE_PROVENANCE: "2475e3dfb210c0d6fc88f073ad3e29a4be7c09f96898e37ee095900070883002",
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

CLASS_COUNT = 286
LEGACY_CLASS_COUNT = 33
SHARED_CLASS_COUNT = 31
B1_ONLY_CLASS_COUNT = 255
TOTAL_PAGE_GROUPS = 24
DEV_PAGE_GROUPS = 6
EXPECTED_P4_ROWS = 219
EXPECTED_GLYPH_ROWS = 180
EXPECTED_STRICT_GLYPH_ROWS = 176
EXPECTED_RUN50_TRAIN_ROWS = 4175
EXPECTED_RUN50_OVERLAP = 19
EXPECTED_RUN50_DEV_OVERLAP = 3
EXPECTED_ADDITIONAL_DHASH_OVERLAP = 0
DHASH_HAMMING_THRESHOLD = 4
MAX_SEGMENTS_PER_GLYPH = 32
DIRECT_B1_BATCH_SIZE = 16
B1_PRIMARY_MODE = "segmented_crops"
B1_FALLBACK_MODE = "whole_glyph_b1"
B1_CACHED_MODE = "whole_glyph_cached_base"
B1_DIRECT_MODE = "whole_glyph_direct"
B1_FALLBACK_POLICY = "zero valid MobileSAM crops => classify whole glyph exactly once"
MOBILE_SAM_CHECKPOINT_SIZE = 40_728_226
MOBILE_SAM_CHECKPOINT_KEYS = 439
MOBILE_SAM_KEY_INVENTORY_SHA256 = (
    "d17a0b2f33d58da4ac45bffefe9551510c925ead8971423152da14a616e77cf1"
)
ALLOWED_MAPPING_STATUSES = frozenset({"accepted_exact", "approved_manual"})
FORBIDDEN_INPUT_TOKENS = (
    "annotations",
    "business-final",
    "business_final",
    "locked-test",
    "locked_test",
)

CACHE_SCHEMA = "autoresearch-legacy-fusion-glyph-cache.v4"
CACHE_MANIFEST_SCHEMA = "autoresearch-legacy-fusion-cache-manifest.v4"
PROTOCOL_SCHEMA = "autoresearch-legacy-fusion-protocol.v3"
RESULTS_SCHEMA = "autoresearch-legacy-fusion-results.v1"
CANONICALIZATION = "canonical-json-nfc-v1"
SPLIT_HASH_NAMESPACE = "p4a-dev-group"
B1_SIMILARITY_THRESHOLD = 0.35
B1_CALIBRATION_TEMPERATURE = 0.10
PRIOR_WEIGHT = 0.50
CASCADE_THRESHOLD = 0.55
SET_THRESHOLD = 0.50
LATENCY_LIMIT_MS = 250.0
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260813
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.50, 2.00)
PRIOR_WEIGHT_GRID = (0.00, 0.25, 0.50, 1.00, 2.00)
CASCADE_THRESHOLD_GRID = (0.25, 0.50, 0.55, 0.75, 0.90)

VARIANTS: dict[str, dict[str, Any]] = {
    "b1": {
        "candidate": False,
        "description": "raw deployed MobileSAM crop max + B1 cache scores",
    },
    "b1_calibrated": {
        "candidate": False,
        "description": "explicit train-only temperature calibration, shared labels only",
    },
    "legacy": {"candidate": False, "description": "run_50 glyph logits, shared labels only"},
    "prior_global": {"candidate": True, "description": "B1 plus global run_50 shared-label log prior"},
    "cascade": {"candidate": True, "description": "apply run_50 prior only above confidence threshold"},
    "oracle": {"candidate": False, "description": "descriptive truth-aware upper bound"},
}


@dataclasses.dataclass(frozen=True)
class Taxonomy:
    runtime_names: tuple[str, ...]
    legacy_names: tuple[str, ...]
    shared_names: tuple[str, ...]
    b1_only_names: tuple[str, ...]
    old_only_names: tuple[str, ...]
    shared_runtime_indices: tuple[int, ...]
    b1_only_indices: tuple[int, ...]
    legacy_to_runtime: tuple[int | None, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
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
            raise TypeError("non-finite float in canonical payload")
        return value
    if isinstance(value, np.generic):
        return _normalise(value.item())
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
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def write_once_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(_normalise(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _torch() -> Any:
    import torch

    return torch


def assert_allowed_input(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
        raise ValueError(f"forbidden research input: {path}")


def assert_safe_output_dir(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise ValueError("research outputs must stay inside the repository") from error
    forbidden_roots = (B1_RUNTIME.parent.parent.resolve(), LEGACY_ROOT.resolve())
    for root in forbidden_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"research output targets a protected tree: {resolved}")


def verify_hashed_artifact(value: Any, schema: str, payload_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("hashed artifact must be an object")
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected artifact schema: {value.get('schema_version')}")
    if value.get("canonicalization_version") != CANONICALIZATION:
        raise ValueError("canonicalization contract changed")
    if value.get("hash_algorithm") != "sha256":
        raise ValueError("hash algorithm changed")
    payload = value.get("payload")
    if value.get("payload_sha256") != semantic_hash(payload):
        raise ValueError("artifact semantic hash mismatch")
    if value["payload_sha256"] != payload_sha256:
        raise ValueError("artifact payload pin mismatch")
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be an object")
    return payload


def _read_pinned_json(path: Path, expected_sha256: str) -> Any:
    assert_allowed_input(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _input_paths_for_profile(profile: str) -> dict[str, Path]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    mobile_only = {
        "mobile_sam_checkpoint_sha256",
        "mobile_sam_source_sha256",
        "mobile_sam_import_source_sha256",
    }
    cached_only = {
        "p4_feature_cache_sha256",
        "p4_feature_provenance_sha256",
    }
    excluded: set[str] = set()
    if profile != PROFILE_MOBILE_SAM_B1:
        excluded.update(mobile_only)
    if profile != PROFILE_WHOLE_GLYPH_CACHED_B1:
        excluded.update(cached_only)
    return {name: path for name, path in _CACHE_INPUT_PATHS.items() if name not in excluded}


def verify_static_hashes(profile: str = DEFAULT_SCORING_PROFILE) -> dict[str, str]:
    observed: dict[str, str] = {}
    for path in _input_paths_for_profile(profile).values():
        expected = EXPECTED_FILE_SHA256[path]
        assert_allowed_input(path)
        if not path.is_file():
            raise FileNotFoundError(f"required pinned input absent: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        observed[str(path.resolve())] = actual
    return observed


def validate_mobile_sam_checkpoint(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_keys: int,
    expected_key_inventory_sha256: str,
) -> dict[str, Any]:
    """Validate the MobileSAM state dict safely before model construction."""

    assert_allowed_input(path)
    if not path.is_file():
        raise FileNotFoundError(f"MobileSAM checkpoint absent: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            "MobileSAM checkpoint size mismatch: "
            f"expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "MobileSAM checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    torch = _torch()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "MobileSAM checkpoint is not loadable with torch weights_only=True"
        ) from error
    if not isinstance(state, Mapping):
        raise ValueError("MobileSAM weights_only payload is not a state mapping")
    keys = sorted(str(key) for key in state)
    if len(keys) != expected_keys or len(set(keys)) != expected_keys:
        raise ValueError(
            f"MobileSAM state key count changed: expected {expected_keys}, got {len(keys)}"
        )
    inventory_sha256 = semantic_hash(keys)
    if inventory_sha256 != expected_key_inventory_sha256:
        raise ValueError("MobileSAM state key inventory hash changed")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError("MobileSAM weights_only payload contains a non-tensor value")
    return {
        "path": str(path.resolve()),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "weights_only_loadable": True,
        "tensor_keys": len(keys),
        "key_inventory_sha256": inventory_sha256,
    }


def validate_taxonomy(runtime_names: Sequence[str], legacy_names: Sequence[str]) -> Taxonomy:
    runtime = tuple(str(name) for name in runtime_names)
    legacy = tuple(str(name) for name in legacy_names)
    if len(runtime) != CLASS_COUNT or len(set(runtime)) != CLASS_COUNT:
        raise ValueError("B1 taxonomy must contain 286 unique classes")
    if len(legacy) != LEGACY_CLASS_COUNT or len(set(legacy)) != LEGACY_CLASS_COUNT:
        raise ValueError("legacy taxonomy must contain 33 unique classes")
    runtime_index = {name: index for index, name in enumerate(runtime)}
    shared = tuple(name for name in legacy if name in runtime_index)
    old_only = tuple(name for name in legacy if name not in runtime_index)
    shared_set = set(shared)
    b1_only = tuple(name for name in runtime if name not in shared_set)
    if len(shared) != SHARED_CLASS_COUNT:
        raise ValueError(f"taxonomy must expose exactly 31 shared classes, got {len(shared)}")
    if len(b1_only) != B1_ONLY_CLASS_COUNT:
        raise ValueError("taxonomy must expose exactly 255 B1-only classes")
    if old_only != ("?", "barra"):
        raise ValueError(f"legacy old-only taxonomy changed: {old_only}")
    legacy_to_runtime = tuple(runtime_index.get(name) for name in legacy)
    return Taxonomy(
        runtime_names=runtime,
        legacy_names=legacy,
        shared_names=shared,
        b1_only_names=b1_only,
        old_only_names=old_only,
        shared_runtime_indices=tuple(runtime_index[name] for name in shared),
        b1_only_indices=tuple(runtime_index[name] for name in b1_only),
        legacy_to_runtime=legacy_to_runtime,
    )


def load_taxonomy_from_pins() -> Taxonomy:
    runtime_config = _read_pinned_json(B1_CONFIG, EXPECTED_FILE_SHA256[B1_CONFIG])
    runtime_names = runtime_config.get("class_names")
    if runtime_config.get("num_classes") != CLASS_COUNT or not isinstance(runtime_names, list):
        raise ValueError("B1 runtime taxonomy contract changed")
    fallback = _read_pinned_json(RUN50_LABELS, EXPECTED_FILE_SHA256[RUN50_LABELS])
    if not isinstance(fallback, list):
        raise ValueError("legacy fallback labels must be a list")
    return validate_taxonomy(runtime_names, fallback)


def _row_classes(row: Mapping[str, Any]) -> set[str]:
    values = row.get("candidate_runtime_classes", row.get("truth_labels", ()))
    return {str(value) for value in values}


def choose_dev_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = str(row.get("page_group", row.get("source_group", "")))
        if not group:
            raise ValueError("row lacks page/source group")
        by_group[group].update(_row_classes(row))
    if len(by_group) != TOTAL_PAGE_GROUPS:
        raise ValueError(f"expected exactly 24 planche groups, got {len(by_group)}")
    selected: list[str] = []
    covered: set[str] = set()
    remaining = set(by_group)
    while len(selected) < DEV_PAGE_GROUPS:
        group = min(
            remaining,
            key=lambda value: (
                -len(by_group[value] - covered),
                stable_hash(SPLIT_HASH_NAMESPACE, value),
            ),
        )
        selected.append(group)
        covered.update(by_group[group])
        remaining.remove(group)
    return tuple(selected)


def assign_page_split(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dev = set(choose_dev_groups(rows))
    result: list[dict[str, Any]] = []
    group_split: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        group = str(row.get("page_group", row.get("source_group", "")))
        split = "dev" if group in dev else "train"
        previous = group_split.setdefault(group, split)
        if previous != split:
            raise AssertionError("page group split changed within assignment")
        row["page_group"] = group
        row["split"] = split
        result.append(row)
    train_groups = {row["page_group"] for row in result if row["split"] == "train"}
    dev_groups = {row["page_group"] for row in result if row["split"] == "dev"}
    if len(train_groups) != 18 or len(dev_groups) != 6 or train_groups & dev_groups:
        raise ValueError("page split must be exactly 18 train / 6 dev and disjoint")
    return result


def exclude_leakage(
    rows: Sequence[Mapping[str, Any]],
    run50_seen_pixels: set[str],
    run50_seen_dhashes: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dhashes = run50_seen_dhashes or set()
    bmp_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        digest = row.get("bmp_sha256")
        if digest:
            bmp_splits[str(digest)].add(str(row.get("split")))
    crossing = {digest for digest, splits in bmp_splits.items() if len(splits) > 1}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        row_reasons: list[str] = []
        if str(row.get("pixel_sha256")) in run50_seen_pixels:
            row_reasons.append("run50_pixel_identical")
        elif dhashes and row.get("dhash64") is not None:
            current = int(str(row["dhash64"]), 16)
            if min((current ^ seen).bit_count() for seen in dhashes) <= DHASH_HAMMING_THRESHOLD:
                row_reasons.append("run50_dhash_hamming_le_4")
        if row.get("bmp_sha256") and str(row["bmp_sha256"]) in crossing:
            row_reasons.append("bmp_crosses_train_dev")
        if row_reasons:
            for reason in row_reasons:
                reasons[reason] += 1
            excluded.append({**row, "exclusion_reasons": row_reasons})
        else:
            kept.append(row)
    return kept, {
        "reasons": dict(reasons),
        "dhash_hamming_threshold": DHASH_HAMMING_THRESHOLD,
        "additional_dhash_overlaps": reasons.get("run50_dhash_hamming_le_4", 0),
        "excluded_rows": excluded,
        "excluded_row_ids_sha256": semantic_hash(
            sorted(str(row.get("row_id")) for row in excluded)
        ),
    }


def _pixel_identity(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def _dhash64(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize(
            (9, 8), resample=Image.Resampling.LANCZOS
        )
        values = np.asarray(grayscale, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return f"{result:016x}"


def _resolve_windows_path(value: str) -> Path:
    windows = PureWindowsPath(value)
    if not windows.drive or len(windows.drive) != 2 or windows.drive[1] != ":":
        raise ValueError(f"run_50 CSV path is not an absolute Windows path: {value}")
    return Path("/mnt") / windows.drive[0].lower() / Path(*windows.parts[1:])


def load_run50_seen_pixels() -> tuple[set[str], set[int], dict[str, Any]]:
    assert_allowed_input(RUN50_TRAINING_CSV)
    if sha256_file(RUN50_TRAINING_CSV) != EXPECTED_FILE_SHA256[RUN50_TRAINING_CSV]:
        raise ValueError("run_50 training CSV hash changed")
    seen: set[str] = set()
    path_pixels: list[tuple[str, str]] = []
    ordered_dhashes: list[str] = []
    with RUN50_TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "full_image_path" not in (reader.fieldnames or ()):
            raise ValueError("run_50 training CSV lacks full_image_path")
        for row in reader:
            raw = row.get("full_image_path")
            if not raw:
                raise ValueError("run_50 training CSV contains an empty image path")
            path = _resolve_windows_path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"run_50 training image absent: {path}")
            pixel = _pixel_identity(path)
            path_pixels.append((str(path), pixel))
            seen.add(pixel)
            ordered_dhashes.append(_dhash64(path))
    if len(path_pixels) != EXPECTED_RUN50_TRAIN_ROWS or len(seen) != EXPECTED_RUN50_TRAIN_ROWS:
        raise ValueError("run_50 training image inventory changed")
    seen_dhashes = {int(value, 16) for value in ordered_dhashes}
    return seen, seen_dhashes, {
        "rows": len(path_pixels),
        "unique_pixel_sha256": len(seen),
        "ordered_path_pixel_inventory_sha256": semantic_hash(
            path_pixels
        ),
        "unique_dhash64": len(seen_dhashes),
        "ordered_dhash64_inventory_sha256": semantic_hash(ordered_dhashes),
    }


def load_p4_rows(taxonomy: Taxonomy) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = {
        "plan": _read_pinned_json(P4_PLAN, EXPECTED_FILE_SHA256[P4_PLAN]),
        "manifest": _read_pinned_json(P4_MANIFEST, EXPECTED_FILE_SHA256[P4_MANIFEST]),
        "verification": _read_pinned_json(
            P4_VERIFICATION, EXPECTED_FILE_SHA256[P4_VERIFICATION]
        ),
    }
    payloads = {
        name: verify_hashed_artifact(
            artifact, EXPECTED_SCHEMAS[name], EXPECTED_PAYLOAD_SHA256[name]
        )
        for name, artifact in artifacts.items()
    }
    plan = payloads["plan"]
    manifest = payloads["manifest"]
    verification = payloads["verification"]
    if manifest.get("plan_payload_sha256") != EXPECTED_PAYLOAD_SHA256["plan"]:
        raise ValueError("P4 manifest is not bound to the pinned plan")
    if verification.get("plan_payload_sha256") != EXPECTED_PAYLOAD_SHA256["plan"]:
        raise ValueError("P4 verification is not bound to the pinned plan")
    if verification.get("manifest_payload_sha256") != EXPECTED_PAYLOAD_SHA256["manifest"]:
        raise ValueError("P4 verification is not bound to the pinned manifest")
    required_gates = {
        "cross_group_pixel_duplicates_absent",
        "group_and_class_coverage_verified",
        "hashes_crc_and_budgets_verified",
        "plan_manifest_binding",
        "single_label_supervision_absent",
    }
    if any(verification.get("gates", {}).get(gate) is not True for gate in required_gates):
        raise ValueError("a pinned P4 verification gate is not true")
    if verification.get("gates", {}).get("promotion_allowed") is not False:
        raise ValueError("P4 research input unexpectedly allows promotion")

    entries = plan.get("entries")
    files = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_P4_ROWS:
        raise ValueError("P4 plan row count changed")
    if not isinstance(files, list) or len(files) != EXPECTED_P4_ROWS:
        raise ValueError("P4 manifest row count changed")
    files_by_key = {(row.get("archive_name"), row.get("entry_path")): row for row in files}
    if len(files_by_key) != EXPECTED_P4_ROWS:
        raise ValueError("P4 manifest physical keys are not unique")
    runtime_index = {name: index for index, name in enumerate(taxonomy.runtime_names)}
    rows: list[dict[str, Any]] = []
    pixel_groups: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        key = (entry.get("archive_name"), entry.get("entry_path"))
        downloaded = files_by_key.get(key)
        if downloaded is None:
            raise ValueError("P4 plan/download physical keys differ")
        group = str(entry.get("source_group", ""))
        if (
            not group
            or entry.get("split_group") != group
            or entry.get("physical_source_group") != group
            or downloaded.get("physical_source_group") != group
        ):
            raise ValueError("P4 codex+planche group binding changed")
        if downloaded.get("candidate_runtime_classes") != entry.get(
            "candidate_runtime_classes"
        ):
            raise ValueError("P4 exact glyph-to-element relation differs from download")
        if downloaded.get("node_keys") != entry.get("node_keys"):
            raise ValueError("P4 exact glyph node relation differs from download")
        if downloaded.get("mapping_ids") != entry.get("mapping_ids"):
            raise ValueError("P4 exact mapping relation differs from download")
        path = (P4_IMAGES / str(downloaded.get("output_path", ""))).resolve()
        try:
            path.relative_to(P4_IMAGES.resolve())
        except ValueError as error:
            raise ValueError("P4 image escapes pinned image root") from error
        if not path.is_file() or sha256_file(path) != downloaded.get("byte_sha256"):
            raise ValueError("P4 image byte hash mismatch")
        pixel_sha = _pixel_identity(path)
        if pixel_sha != downloaded.get("image", {}).get("pixel_sha256"):
            raise ValueError("P4 image pixel hash mismatch")
        pixel_groups[pixel_sha].add(group)
        names = [str(name) for name in entry.get("candidate_runtime_classes", ())]
        unknown = sorted(set(names) - set(runtime_index))
        if unknown:
            raise ValueError(f"P4 relation contains unknown runtime classes: {unknown}")
        roles = tuple(sorted(set(str(role) for role in entry.get("roles", ()))))
        statuses = tuple(str(status) for status in entry.get("mapping_statuses", ()))
        strict = bool(
            roles == ("glyph_occurrence",)
            and names
            and entry.get("multilabel_evidence_kind")
            == "explicit_descendant_multilabel_evidence"
            and statuses
            and set(statuses) <= ALLOWED_MAPPING_STATUSES
            and entry.get("mapping_ids")
        )
        suffix = path.suffix.lower()
        rows.append(
            {
                "row_id": "tlachia-glyph-" + stable_hash(*key, pixel_sha)[:24],
                "archive_name": key[0],
                "entry_path": key[1],
                "image_path": str(path),
                "byte_sha256": downloaded["byte_sha256"],
                "pixel_sha256": pixel_sha,
                "dhash64": _dhash64(path),
                "bmp_sha256": downloaded["byte_sha256"] if suffix == ".bmp" else None,
                "page_group": group,
                "roles": list(roles),
                "node_keys": list(entry.get("node_keys", ())),
                "mapping_ids": list(entry.get("mapping_ids", ())),
                "mapping_statuses": list(statuses),
                "candidate_runtime_classes": names,
                "truth_labels": sorted(runtime_index[name] for name in set(names)),
                "strict_metric_eligible": strict,
            }
        )
    if any(len(groups) != 1 for groups in pixel_groups.values()):
        raise ValueError("pixel-identical P4 input crosses planche groups")
    if len({row["page_group"] for row in rows}) != TOTAL_PAGE_GROUPS:
        raise ValueError("P4 planche group count changed")
    assigned = assign_page_split(rows)
    glyph_rows = [row for row in assigned if row["roles"] == ["glyph_occurrence"]]
    strict_rows = [row for row in glyph_rows if row["strict_metric_eligible"]]
    if len(glyph_rows) != EXPECTED_GLYPH_ROWS:
        raise ValueError("P4 glyph row count changed")
    if len(strict_rows) != EXPECTED_STRICT_GLYPH_ROWS:
        raise ValueError("P4 fully-resolved glyph relation count changed")
    return assigned, {
        "all_rows": len(assigned),
        "glyph_rows": len(glyph_rows),
        "strict_glyph_rows": len(strict_rows),
        "train_groups": len({row["page_group"] for row in assigned if row["split"] == "train"}),
        "dev_groups": len({row["page_group"] for row in assigned if row["split"] == "dev"}),
        "relation_sha256": semantic_hash(
            [
                [row["node_keys"], row["mapping_ids"], row["truth_labels"]]
                for row in glyph_rows
            ]
        ),
    }


def frozen_protocol(profile: str = DEFAULT_SCORING_PROFILE) -> dict[str, Any]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "scope": "research-only glyph sanity spike",
        "scoring_profile": profile,
        "default_scoring_profile": DEFAULT_SCORING_PROFILE,
        "inputs": {
            "p4_plan_payload_sha256": EXPECTED_PAYLOAD_SHA256["plan"],
            "p4_manifest_payload_sha256": EXPECTED_PAYLOAD_SHA256["manifest"],
            "run50_checkpoint_sha256": RUN50_CHECKPOINT_SHA256,
            "b1_projection_sha256": EXPECTED_FILE_SHA256[B1_PROJECTION],
            "mobile_sam_sha256": (
                EXPECTED_FILE_SHA256[MOBILE_SAM_CHECKPOINT]
                if profile == PROFILE_MOBILE_SAM_B1
                else None
            ),
            "p4_feature_cache_sha256": (
                EXPECTED_FILE_SHA256[P4_FEATURE_CACHE]
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else None
            ),
            "p4_feature_provenance_sha256": (
                EXPECTED_FILE_SHA256[P4_FEATURE_PROVENANCE]
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else None
            ),
        },
        "split": {
            "unit": "codex+planche",
            "train_groups": 18,
            "dev_groups": 6,
            "selection": "greedy max-new descendant evidence then stable hash",
            "hash_namespace": SPLIT_HASH_NAMESPACE,
        },
        "exclusions": [
            "pixel-identical glyph already seen by run_50",
            "dHash-64 near-duplicate of run_50 at Hamming distance <=4",
            "physical BMP digest crossing train/dev",
            "not a glyph occurrence",
            "descendant relation not fully resolved",
        ],
        "b1_glyph_aggregate": {
            "backend": (
                "whole JPEG via deployed CodexClassifier.classify_batch(top_k=286)"
                if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                else (
                    "pinned P4 base-view DINO features then B1 projection/prototypes"
                    if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                    else "MobileSAM real crops then B1 classify_batch(top_k=286)"
                )
            ),
            "backbone_extraction_in_timed_run": (
                profile in {PROFILE_WHOLE_GLYPH_DIRECT_B1, PROFILE_MOBILE_SAM_B1}
            ),
            "runtime_preprocess_exact": profile != PROFILE_WHOLE_GLYPH_CACHED_B1,
            "direct_batch_size": (
                DIRECT_B1_BATCH_SIZE
                if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                else None
            ),
            "max_segments_per_glyph": (
                MAX_SEGMENTS_PER_GLYPH
                if profile == PROFILE_MOBILE_SAM_B1
                else None
            ),
            "segment_aggregation": (
                "per-class maximum cosine over crops"
                if profile == PROFILE_MOBILE_SAM_B1
                else None
            ),
            "score_semantics": (
                "raw deployed classifier cosine confidence"
                if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                else "similarity converted with frozen threshold/temperature"
            ),
            "similarity_to_logit": {
                "threshold": B1_SIMILARITY_THRESHOLD,
                "temperature": B1_CALIBRATION_TEMPERATURE,
            }
            if profile != PROFILE_WHOLE_GLYPH_DIRECT_B1
            else None,
            "zero_crop_policy": (
                {
                    "mode": B1_FALLBACK_MODE,
                    "action": B1_FALLBACK_POLICY,
                    "adaptive_exclusion": False,
                }
                if profile == PROFILE_MOBILE_SAM_B1
                else None
            ),
            "feature_cache": (
                {
                    "path": str(P4_FEATURE_CACHE),
                    "sha256": EXPECTED_FILE_SHA256[P4_FEATURE_CACHE],
                    "provenance_sha256": EXPECTED_FILE_SHA256[
                        P4_FEATURE_PROVENANCE
                    ],
                    "view": "base",
                    "semantic_version": "tlachia-multiview-p4a-features.v1",
                }
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else None
            ),
        },
        "inference_imports": (
            {
                "projection": (
                    "from codex_model.classifier import _ProjectionHead "
                    "(codex_model 1.0.0)"
                )
            }
            if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
            else (
                {
                    "classifier": (
                        "from codex_model import CodexClassifier (codex_model 1.0.0)"
                    )
                }
                if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                else {
                "classifier": (
                    "from codex_model import CodexClassifier (codex_model 1.0.0)"
                ),
                "segmenter": (
                    "from codex_pipeline.segmentation import MobileSAMSegmenter"
                ),
                }
            )
        ),
        "variants": VARIANTS,
        "baseline_contract": {
            "name": "b1",
            "scores": "bit-identical copy of immutable prepared B1 score matrix",
            "post_cache_temperature_or_recalibration": False,
            "all_candidate_gates_compare_to": "raw b1",
            "production_runtime_equivalent": (
                profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
            ),
            "status": (
                "production_runtime_baseline"
                if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                else "research_diagnostic_only"
            ),
        },
        "train_only_selection": {
            "temperature_grid": list(TEMPERATURE_GRID),
            "temperature_objective": "multilabel binary cross-entropy",
            "b1_calibration_scope": "31 shared logits; 255 B1-only logits untouched",
            "prior_weight_grid": list(PRIOR_WEIGHT_GRID),
            "cascade_threshold_grid": list(CASCADE_THRESHOLD_GRID),
            "fusion_objective": "train-only glyph sample AP",
            "tie_break": "smallest parameter",
        },
        "fusion": {
            "b1_only_invariant": "all 255 B1-only logits bit-identical to B1",
        },
        "metrics": [
            "glyph macro AP",
            "glyph sample AP",
            "glyph Recall@5",
            "glyph set-F1",
            "multilabel ECE-10 and Brier",
            "latency ms/glyph",
            "paired planche bootstrap",
            (
                "global-286 ranking sample AP, Recall@5 and top1 hit rate, "
                "conditional on glyph truth containing B1-only"
            ),
        ],
        "sanity_gate": {
            "mean_planche_sample_ap_gain": 0.01,
            "paired_planche_ci_lower": ">0",
            "latency_ms_per_glyph_max": LATENCY_LIMIT_MS,
            "b1_only_logits": "bit-identical to raw B1",
            "b1_only_ranking": (
                "no degradation in global-286 sample AP, Recall@5, top1 hit rate "
                "conditional on glyph truth containing B1-only"
            ),
        },
        "terminal_verdict": "INSUFFICIENT_FOR_ELEMENT_GATE",
        "fresh_expanded_element_benchmark_required": True,
        "automatic_global_training_allowed": False,
        "promotion_eligible": False,
    }


def preflight(
    profile: str = DEFAULT_SCORING_PROFILE,
) -> tuple[list[dict[str, Any]], Taxonomy, dict[str, Any]]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    hashes = verify_static_hashes(profile)
    mobile_sam_audit = None
    if profile == PROFILE_MOBILE_SAM_B1:
        mobile_sam_audit = validate_mobile_sam_checkpoint(
            MOBILE_SAM_CHECKPOINT,
            expected_size=MOBILE_SAM_CHECKPOINT_SIZE,
            expected_sha256=EXPECTED_FILE_SHA256[MOBILE_SAM_CHECKPOINT],
            expected_keys=MOBILE_SAM_CHECKPOINT_KEYS,
            expected_key_inventory_sha256=MOBILE_SAM_KEY_INVENTORY_SHA256,
        )
    taxonomy = load_taxonomy_from_pins()
    all_rows, p4_audit = load_p4_rows(taxonomy)
    seen, seen_dhashes, run50_inventory = load_run50_seen_pixels()
    glyph_rows = [row for row in all_rows if row["roles"] == ["glyph_occurrence"]]
    kept, leakage = exclude_leakage(glyph_rows, seen, seen_dhashes)
    overlap = leakage["reasons"].get("run50_pixel_identical", 0)
    dev_overlap = sum(
        row["split"] == "dev"
        and "run50_pixel_identical" in row.get("exclusion_reasons", ())
        for row in leakage["excluded_rows"]
    )
    if overlap != EXPECTED_RUN50_OVERLAP or dev_overlap != EXPECTED_RUN50_DEV_OVERLAP:
        raise ValueError(
            f"run_50 overlap changed: total={overlap}, dev={dev_overlap}; expected 19/3"
        )
    if leakage["additional_dhash_overlaps"] != EXPECTED_ADDITIONAL_DHASH_OVERLAP:
        raise ValueError(
            "run_50 dHash near-overlap count changed: "
            f"{leakage['additional_dhash_overlaps']} != "
            f"{EXPECTED_ADDITIONAL_DHASH_OVERLAP}"
        )
    strict = [row for row in kept if row["strict_metric_eligible"]]
    if not strict or not any(row["split"] == "dev" for row in strict):
        raise ValueError("no strict leakage-free glyph benchmark remains")
    audit = {
        "scoring_profile": profile,
        "protocol_sha256": semantic_hash(frozen_protocol(profile)),
        "input_file_sha256": hashes,
        "mobile_sam_checkpoint": mobile_sam_audit,
        "b1_zero_crop_fallback_policy": (
            {
                "mode": B1_FALLBACK_MODE,
                "action": B1_FALLBACK_POLICY,
                "adaptive_exclusion": False,
            }
            if profile == PROFILE_MOBILE_SAM_B1
            else None
        ),
        "direct_b1_batch_size": (
            DIRECT_B1_BATCH_SIZE
            if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
            else None
        ),
        "p4": p4_audit,
        "run50_training_inventory": run50_inventory,
        "leakage": leakage,
        "strict_rows_after_exclusion": len(strict),
        "strict_train_rows": sum(row["split"] == "train" for row in strict),
        "strict_dev_rows": sum(row["split"] == "dev" for row in strict),
        "strict_dev_groups": len(
            {row["page_group"] for row in strict if row["split"] == "dev"}
        ),
        "taxonomy": {
            "runtime": len(taxonomy.runtime_names),
            "legacy": len(taxonomy.legacy_names),
            "shared": len(taxonomy.shared_names),
            "b1_only": len(taxonomy.b1_only_names),
            "old_only": list(taxonomy.old_only_names),
        },
        "verdict": "READY_FOR_EXPLICIT_PREPARE",
        "promotion_eligible": False,
    }
    if profile == PROFILE_WHOLE_GLYPH_CACHED_B1:
        _, feature_audit = load_pinned_base_features(strict)
        audit["whole_glyph_cached_b1"] = feature_audit
    return strict, taxonomy, audit


def _load_run50_model(device: str, expected_labels: Sequence[str]) -> Any:
    torch = _torch()
    import torch.nn as nn
    from torchvision.models import efficientnet_b3

    config = _read_pinned_json(RUN50_CONFIG, EXPECTED_FILE_SHA256[RUN50_CONFIG])
    exact = {
        "model": "efficientnet_b3",
        "dropout": 0.3,
        "use_bottleneck": True,
        "bottleneck_dim": 768,
        "hidden": [512, 256],
        "custom_head": False,
    }
    if any(config.get(key) != value for key, value in exact.items()):
        raise ValueError("run_50 architecture config differs from pinned exact contract")
    checkpoint = torch.load(RUN50_CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("run_50 checkpoint lacks model_state_dict")
    labels = checkpoint.get("label_columns")
    if labels is None:
        labels = _read_pinned_json(RUN50_LABELS, EXPECTED_FILE_SHA256[RUN50_LABELS])
    if list(labels) != list(expected_labels):
        raise ValueError("run_50 checkpoint labels differ from pinned taxonomy")

    backbone = efficientnet_b3(weights=None)
    in_features = backbone.classifier[1].in_features
    if in_features != 1536:
        raise ValueError("torchvision EfficientNet-B3 feature size changed")
    backbone.classifier = nn.Identity()
    head = nn.Sequential(
        nn.Linear(1536, 768),
        nn.BatchNorm1d(768),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, LEGACY_CLASS_COUNT),
    )

    class Run50Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = backbone
            self.head = head

        def forward(self, value: Any) -> Any:
            return self.head(self.base_model(value))

    model = Run50Model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model


def _run50_tensor(image: Any) -> Any:
    torch = _torch()
    from PIL import Image, ImageOps

    rgb = ImageOps.exif_transpose(image).convert("RGB").resize(
        (300, 300), resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy((array - mean) / std).permute(2, 0, 1)


def score_run50(rows: Sequence[Mapping[str, Any]], model: Any, device: str, batch_size: int) -> tuple[np.ndarray, float]:
    torch = _torch()
    from PIL import Image

    output: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            tensors = []
            for row in rows[offset : offset + batch_size]:
                with Image.open(str(row["image_path"])) as image:
                    tensors.append(_run50_tensor(image))
            batch = torch.stack(tensors).to(torch.device(device))
            output.append(model(batch).detach().cpu().numpy().astype(np.float32))
    elapsed = time.perf_counter() - started
    logits = np.concatenate(output, axis=0)
    if logits.shape != (len(rows), LEGACY_CLASS_COUNT) or not np.isfinite(logits).all():
        raise ValueError("run_50 inference output shape/values changed")
    return logits, elapsed * 1000.0 / len(rows)


def score_direct_b1_whole_glyph(
    rows: Sequence[Mapping[str, Any]], classifier: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score whole JPEG glyphs through the deployed classifier API unchanged."""

    from PIL import Image

    if not rows:
        raise ValueError("direct B1 scoring requires at least one glyph")
    class_names = [str(name) for name in classifier.class_names]
    if len(class_names) != CLASS_COUNT or len(set(class_names)) != CLASS_COUNT:
        raise ValueError("B1 classifier class taxonomy changed")
    class_index = {name: index for index, name in enumerate(class_names)}
    outputs: list[np.ndarray] = []
    batches = 0
    started = time.perf_counter()
    for offset in range(0, len(rows), DIRECT_B1_BATCH_SIZE):
        images = []
        for row in rows[offset : offset + DIRECT_B1_BATCH_SIZE]:
            image_path = Path(str(row["image_path"]))
            if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"direct B1 input is not JPEG: {image_path}")
            with Image.open(image_path) as image:
                images.append(image.copy())
        results = classifier.classify_batch(images, top_k=CLASS_COUNT)
        batches += 1
        if len(results) != len(images):
            raise ValueError("B1 result count differs from direct JPEG batch")
        for result in results:
            top = result.get("top_k")
            if not isinstance(top, list) or len(top) != CLASS_COUNT:
                raise ValueError("B1 full score vector is unavailable")
            scores = np.full(CLASS_COUNT, np.nan, dtype=np.float32)
            for item in top:
                name = str(item.get("class_name"))
                if name not in class_index:
                    raise ValueError(f"B1 returned unknown class: {name}")
                index = class_index[name]
                if np.isfinite(scores[index]):
                    raise ValueError(f"B1 returned duplicate class: {name}")
                scores[index] = float(item["confidence"])
            if not np.isfinite(scores).all():
                raise ValueError("B1 did not return each runtime class exactly once")
            outputs.append(scores)
    elapsed = time.perf_counter() - started
    matrix = np.stack(outputs)
    if matrix.shape != (len(rows), CLASS_COUNT):
        raise ValueError("direct B1 score matrix shape changed")
    return matrix, {
        "b1_ms_per_glyph": elapsed * 1000.0 / len(rows),
        "b1_direct_batch_size": DIRECT_B1_BATCH_SIZE,
        "b1_direct_batches": batches,
    }


def _load_b1_classifier(device: str) -> Any:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    import codex_model
    from codex_model import CodexClassifier

    if getattr(codex_model, "__version__", None) != "1.0.0":
        raise ValueError("codex_model import version changed")
    if CodexClassifier.__module__ != "codex_model.classifier":
        raise ValueError("CodexClassifier import semantics changed")
    return CodexClassifier(
        model_dir=B1_RUNTIME, device=device, backbone_source=B1_BACKBONE_PIN
    )


def _load_b1_and_segmenter(device: str) -> tuple[Any, Any]:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from codex_pipeline.segmentation import MobileSAMSegmenter

    if MobileSAMSegmenter.__module__ != "codex_pipeline.segmentation.mobilesam":
        raise ValueError("MobileSAMSegmenter import semantics changed")
    classifier = _load_b1_classifier(device)
    segmenter = MobileSAMSegmenter(
        checkpoint_path=str(MOBILE_SAM_CHECKPOINT),
        device=device,
        min_area=50,
        max_area_ratio=0.85,
        min_stability_score=0.8,
        iou_threshold=0.5,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
    )
    return classifier, segmenter


def score_b1_glyph_aggregate(
    rows: Sequence[Mapping[str, Any]], classifier: Any, segmenter: Any, max_segments: int
) -> tuple[np.ndarray, list[int], list[str], dict[str, Any]]:
    from PIL import Image, ImageOps

    all_logits: list[np.ndarray] = []
    segment_counts: list[int] = []
    fallback_modes: list[str] = []
    fallback_classify_seconds = 0.0
    started = time.perf_counter()
    class_index = {name: index for index, name in enumerate(classifier.class_names)}
    if len(class_index) != CLASS_COUNT:
        raise ValueError("B1 classifier class taxonomy changed")
    for row in rows:
        with Image.open(str(row["image_path"])) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(rgb)
        proposals = segmenter.segment_page(array)
        proposals = segmenter.extract_crops(array, proposals[:max_segments])
        crops = [proposal.crop for proposal in proposals if proposal.crop is not None]
        if not crops:
            fallback_started = time.perf_counter()
            crops = [array]
            results = classifier.classify_batch(crops, top_k=CLASS_COUNT)
            fallback_classify_seconds += time.perf_counter() - fallback_started
            segment_counts.append(0)
            fallback_modes.append(B1_FALLBACK_MODE)
        else:
            results = classifier.classify_batch(crops, top_k=CLASS_COUNT)
            segment_counts.append(len(crops))
            fallback_modes.append(B1_PRIMARY_MODE)
        if len(results) != len(crops):
            raise ValueError("B1 result count differs from MobileSAM crop count")
        similarities = np.full((len(results), CLASS_COUNT), -np.inf, dtype=np.float32)
        for crop_index, result in enumerate(results):
            top = result.get("top_k")
            if not isinstance(top, list) or len(top) != CLASS_COUNT:
                raise ValueError("B1 full score vector is unavailable")
            for item in top:
                name = str(item.get("class_name"))
                if name not in class_index:
                    raise ValueError(f"B1 returned unknown class: {name}")
                similarities[crop_index, class_index[name]] = float(item["confidence"])
        if not np.isfinite(similarities).all():
            raise ValueError("B1 did not return each runtime class exactly once")
        maximum = similarities.max(axis=0)
        calibrated_logits = (
            maximum - B1_SIMILARITY_THRESHOLD
        ) / B1_CALIBRATION_TEMPERATURE
        all_logits.append(calibrated_logits.astype(np.float32))
    elapsed = time.perf_counter() - started
    scores = np.stack(all_logits)
    if scores.shape != (len(rows), CLASS_COUNT) or not np.isfinite(scores).all():
        raise ValueError("B1 glyph aggregate output shape/values changed")
    fallback_count = fallback_modes.count(B1_FALLBACK_MODE)
    return scores, segment_counts, fallback_modes, {
        "b1_ms_per_glyph": elapsed * 1000.0 / len(rows),
        "whole_glyph_b1_fallback_count": fallback_count,
        "whole_glyph_b1_fallback_rate": fallback_count / len(rows),
        "whole_glyph_b1_fallback_classify_ms_total": fallback_classify_seconds
        * 1000.0,
        "whole_glyph_b1_fallback_classify_ms_mean": (
            fallback_classify_seconds * 1000.0 / fallback_count
            if fallback_count
            else 0.0
        ),
    }


def load_pinned_base_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path = P4_FEATURE_CACHE,
    provenance_path: Path = P4_FEATURE_PROVENANCE,
) -> tuple[Any, dict[str, Any]]:
    """Load reviewed base-view features after hash/provenance validation."""

    expected_cache_hash = EXPECTED_FILE_SHA256.get(cache_path)
    expected_provenance_hash = EXPECTED_FILE_SHA256.get(provenance_path)
    if expected_cache_hash is None or expected_provenance_hash is None:
        raise ValueError("P4 feature cache/provenance path is not pinned")
    provenance = _read_pinned_json(provenance_path, expected_provenance_hash)
    required_provenance = {
        "schema_version": "tlachia-multiview-p4a-features.provenance.v1",
        "cache_sha256": expected_cache_hash,
        "plan_payload_sha256": EXPECTED_PAYLOAD_SHA256["plan"],
        "manifest_payload_sha256": EXPECTED_PAYLOAD_SHA256["manifest"],
        "verification_payload_sha256": EXPECTED_PAYLOAD_SHA256["verification"],
        "promotion_eligible": False,
        "view_names": ["base", "mild-0", "mild-1", "strong-0", "strong-1"],
    }
    for key, expected in required_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"P4 feature provenance binding changed: {key}")
    backbone = provenance.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("P4 feature provenance lacks backbone binding")
    expected_backbone = {
        "backbone": "dinov2_vits14",
        "pin_sha256": EXPECTED_FILE_SHA256[B1_BACKBONE_PIN],
        "source_tree_sha256": "a64fa885acf9ca2f6b8138ba1735e597a9a3367e5afffec2a8457bbaf780ab3a",
        "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    }
    if any(backbone.get(key) != value for key, value in expected_backbone.items()):
        raise ValueError("P4 feature provenance backbone source pins changed")
    if sha256_file(cache_path) != expected_cache_hash:
        raise ValueError("P4 feature cache SHA-256 changed")
    torch = _torch()
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("P4 feature cache is not weights_only-loadable") from error
    if not isinstance(cache, Mapping):
        raise ValueError("P4 feature cache is not a mapping")
    required = {
        "schema_version",
        "features",
        "row_ids",
        "image_paths",
        "byte_sha256",
        "pixel_sha256",
        "roles",
        "source_groups",
        "descendant_class_names",
        "descendant_class_labels",
        "splits",
        "view_names",
        "feature_dim",
        "promotion_eligible",
    }
    if set(cache) != required:
        raise ValueError("P4 feature cache fields changed")
    if cache["schema_version"] != "tlachia-multiview-p4a-features.v1":
        raise ValueError("P4 feature cache semantic version changed")
    if cache["promotion_eligible"] is not False:
        raise ValueError("P4 feature cache unexpectedly allows promotion")
    if cache["view_names"] != required_provenance["view_names"]:
        raise ValueError("P4 feature cache view order changed")
    features = cache["features"]
    if not torch.is_tensor(features) or tuple(features.shape) != (219, 5, 384):
        raise ValueError("P4 feature cache tensor shape changed")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("P4 feature cache contains invalid features")
    if cache["feature_dim"] != 384:
        raise ValueError("P4 feature cache feature dimension changed")
    for key, provenance_key in (
        ("row_ids", "row_ids_sha256"),
        ("source_groups", "source_groups_sha256"),
        ("descendant_class_labels", "descendant_class_labels_sha256"),
        ("splits", "split_sha256"),
    ):
        if semantic_hash(cache[key]) != provenance.get(provenance_key):
            raise ValueError(f"P4 feature cache/provenance mismatch: {key}")
    pixels = [str(value) for value in cache["pixel_sha256"]]
    cache_row_ids = [str(value) for value in cache["row_ids"]]
    if len(pixels) != 219 or len(set(pixels)) != 219 or len(set(cache_row_ids)) != 219:
        raise ValueError("P4 feature cache row/pixel identity is not unique")
    pixel_index = {pixel: index for index, pixel in enumerate(pixels)}
    missing = [str(row["row_id"]) for row in rows if str(row["pixel_sha256"]) not in pixel_index]
    if missing:
        raise ValueError(f"P4 feature cache misses reviewed row IDs: {missing[:3]}")
    indices = [pixel_index[str(row["pixel_sha256"])] for row in rows]
    if len(set(indices)) != len(rows):
        raise ValueError("reviewed rows map non-uniquely into P4 feature cache")
    base_index = cache["view_names"].index("base")
    base = features[indices, base_index].contiguous()
    return base, {
        "cache_sha256": expected_cache_hash,
        "provenance_sha256": expected_provenance_hash,
        "semantic_version": cache["schema_version"],
        "base_view_index": base_index,
        "selected_rows": len(rows),
        "selected_cache_row_ids_sha256": semantic_hash(
            [cache_row_ids[index] for index in indices]
        ),
        "selected_pixel_sha256_sha256": semantic_hash(
            [pixels[index] for index in indices]
        ),
        "backbone_extraction_in_timed_run": False,
    }


def score_cached_b1_base_features(
    features: Any, taxonomy: Taxonomy, device: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run only the pinned B1 projection/prototypes on cached base features."""

    torch = _torch()
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from codex_model.classifier import _ProjectionHead

    projection_state = torch.load(B1_PROJECTION, map_location="cpu", weights_only=True)
    prototype_state = torch.load(B1_PROTOTYPES, map_location="cpu", weights_only=True)
    if not isinstance(projection_state, Mapping) or not isinstance(
        prototype_state, Mapping
    ):
        raise ValueError("B1 projection/prototypes payload changed")
    required_proto = {"prototypes", "class_names", "class_labels", "embedding_dim"}
    if set(prototype_state) != required_proto:
        raise ValueError("B1 prototype payload fields changed")
    class_names = prototype_state["class_names"]
    ordered_names = tuple(str(class_names[index]) for index in range(CLASS_COUNT))
    if ordered_names != taxonomy.runtime_names:
        raise ValueError("B1 prototype taxonomy differs from pinned runtime")
    prototypes = prototype_state["prototypes"]
    if tuple(prototypes.shape) != (CLASS_COUNT, 128) or not torch.isfinite(
        prototypes
    ).all():
        raise ValueError("B1 prototype tensor changed")
    projection = _ProjectionHead(hidden_dim=384, embedding_dim=128)
    projection.load_state_dict(projection_state, strict=True)
    target = torch.device(device)
    projection.to(target).eval()
    prototypes = prototypes.to(target)
    source = features.to(target)
    started = time.perf_counter()
    with torch.inference_mode():
        similarities = projection(source) @ prototypes.t()
        logits = (
            (similarities - B1_SIMILARITY_THRESHOLD)
            / B1_CALIBRATION_TEMPERATURE
        ).cpu()
    elapsed = time.perf_counter() - started
    scores = logits.numpy().astype(np.float32, copy=False)
    if scores.shape != (len(features), CLASS_COUNT) or not np.isfinite(scores).all():
        raise ValueError("cached whole-glyph B1 score shape/values changed")
    return scores, {
        "b1_ms_per_glyph": elapsed * 1000.0 / len(features),
        "b1_cached_forward_ms_per_glyph": elapsed * 1000.0 / len(features),
        "b1_backbone_extraction_included": False,
    }


_CACHE_INPUT_PATHS = {
    "p4_plan_file_sha256": P4_PLAN,
    "p4_manifest_file_sha256": P4_MANIFEST,
    "p4_verification_file_sha256": P4_VERIFICATION,
    "run50_checkpoint_sha256": RUN50_CHECKPOINT,
    "run50_config_sha256": RUN50_CONFIG,
    "run50_model_source_sha256": RUN50_MODEL_SOURCE,
    "run50_labels_sha256": RUN50_LABELS,
    "run50_transforms_sha256": RUN50_TRANSFORMS,
    "run50_training_csv_sha256": RUN50_TRAINING_CSV,
    "b1_config_sha256": B1_CONFIG,
    "b1_projection_sha256": B1_PROJECTION,
    "b1_prototypes_sha256": B1_PROTOTYPES,
    "b1_package_manifest_sha256": B1_PACKAGE_MANIFEST,
    "b1_package_checksums_sha256": B1_PACKAGE_CHECKSUMS,
    "b1_backbone_pin_sha256": B1_BACKBONE_PIN,
    "mobile_sam_checkpoint_sha256": MOBILE_SAM_CHECKPOINT,
    "b1_classifier_source_sha256": B1_CLASSIFIER_SOURCE,
    "b1_classifier_import_source_sha256": B1_CLASSIFIER_INIT,
    "mobile_sam_source_sha256": MOBILE_SAM_SOURCE,
    "mobile_sam_import_source_sha256": MOBILE_SAM_INIT,
    "p4_feature_cache_sha256": P4_FEATURE_CACHE,
    "p4_feature_provenance_sha256": P4_FEATURE_PROVENANCE,
}


def expected_cache_input_binding() -> dict[str, str]:
    return expected_cache_input_binding_for_profile(DEFAULT_SCORING_PROFILE)


def expected_cache_input_binding_for_profile(profile: str) -> dict[str, str]:
    binding = {
        name: EXPECTED_FILE_SHA256[path]
        for name, path in _input_paths_for_profile(profile).items()
    }
    binding.update(
        {
            "p4_plan_payload_sha256": EXPECTED_PAYLOAD_SHA256["plan"],
            "p4_manifest_payload_sha256": EXPECTED_PAYLOAD_SHA256["manifest"],
            "p4_verification_payload_sha256": EXPECTED_PAYLOAD_SHA256["verification"],
        }
    )
    return binding


def observed_cache_input_binding(
    audit: Mapping[str, Any], profile: str = DEFAULT_SCORING_PROFILE
) -> dict[str, str]:
    observed_files = audit.get("input_file_sha256")
    if not isinstance(observed_files, Mapping):
        raise ValueError("preflight audit lacks input file hashes")
    binding: dict[str, str] = {}
    for name, path in _input_paths_for_profile(profile).items():
        resolved = str(path.resolve())
        value = observed_files.get(resolved)
        if not isinstance(value, str):
            raise ValueError(f"preflight audit lacks cache input hash: {resolved}")
        binding[name] = value
    binding.update(
        {
            "p4_plan_payload_sha256": EXPECTED_PAYLOAD_SHA256["plan"],
            "p4_manifest_payload_sha256": EXPECTED_PAYLOAD_SHA256["manifest"],
            "p4_verification_payload_sha256": EXPECTED_PAYLOAD_SHA256["verification"],
        }
    )
    return binding


def benchmark_binding(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prepared benchmark binding cannot be empty")
    return {
        "rows": len(rows),
        "row_ids_sha256": semantic_hash([str(row["row_id"]) for row in rows]),
        "pixel_sha256_sha256": semantic_hash(
            [str(row["pixel_sha256"]) for row in rows]
        ),
        "splits_sha256": semantic_hash([str(row["split"]) for row in rows]),
        "page_groups_sha256": semantic_hash(
            [str(row["page_group"]) for row in rows]
        ),
        "truth_labels_sha256": semantic_hash(
            [list(row["truth_labels"]) for row in rows]
        ),
    }


def prepared_paths(profile: str) -> tuple[Path, Path]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    path = PREPARED_DIR / profile / "glyph-score-cache.npz"
    return path, path.with_name("manifest.json")


def result_paths(profile: str) -> tuple[Path, Path]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    directory = REPORT_DIR / profile
    return directory / "results.json", directory / "final.json"


def fallback_binding(
    rows: Sequence[Mapping[str, Any]], fallback_modes: Sequence[str]
) -> dict[str, Any]:
    if len(rows) != len(fallback_modes):
        raise ValueError("fallback mode row count differs from benchmark")
    allowed = {B1_PRIMARY_MODE, B1_FALLBACK_MODE, B1_CACHED_MODE, B1_DIRECT_MODE}
    modes = [str(mode) for mode in fallback_modes]
    if set(modes) - allowed:
        raise ValueError("cache fallback mode changed")
    fallback_row_ids = [
        str(row["row_id"])
        for row, mode in zip(rows, modes, strict=True)
        if mode == B1_FALLBACK_MODE
    ]
    return {
        "policy": B1_FALLBACK_POLICY,
        "fallback_mode": B1_FALLBACK_MODE,
        "adaptive_exclusion": False,
        "fallback_count": len(fallback_row_ids),
        "fallback_row_ids": fallback_row_ids,
        "row_modes_sha256": semantic_hash(modes),
    }


def inference_binding(profile: str) -> dict[str, Any]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    return {
        "scoring_profile": profile,
        "production_runtime_equivalent": (
            profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
        ),
        "direct_b1_batch_size": (
            DIRECT_B1_BATCH_SIZE
            if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
            else None
        ),
        "max_segments_per_glyph": (
            MAX_SEGMENTS_PER_GLYPH if profile == PROFILE_MOBILE_SAM_B1 else None
        ),
        "classifier_import": (
            "from codex_model.classifier import _ProjectionHead"
            if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
            else "from codex_model import CodexClassifier"
        ),
        "classifier_package_version": "1.0.0",
        "segmenter_import": (
            "from codex_pipeline.segmentation import MobileSAMSegmenter"
            if profile == PROFILE_MOBILE_SAM_B1
            else None
        ),
        "p4_feature_cache_sha256": (
            EXPECTED_FILE_SHA256[P4_FEATURE_CACHE]
            if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
            else None
        ),
        "base_view": "base" if profile == PROFILE_WHOLE_GLYPH_CACHED_B1 else None,
        "latency_scope": (
            "whole-JPEG decode plus exact runtime preprocess/backbone/projection/prototypes; "
            "model construction excluded"
            if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
            else (
                "cached projection/prototypes forward; backbone extraction excluded"
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else "MobileSAM segmentation/crops plus B1 inference; model construction excluded"
            )
        ),
    }


def build_prepared_manifest(
    cache_path: Path,
    rows: Sequence[Mapping[str, Any]],
    input_binding: Mapping[str, str],
    fallback_modes: Sequence[str] | None = None,
    profile: str = DEFAULT_SCORING_PROFILE,
) -> dict[str, Any]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    expected_inputs = expected_cache_input_binding_for_profile(profile)
    if dict(input_binding) != expected_inputs:
        raise ValueError("cache input binding differs from pinned inputs")
    modes = (
        list(fallback_modes)
        if fallback_modes is not None
        else [
            (
                B1_CACHED_MODE
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else (
                    B1_DIRECT_MODE
                    if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                    else B1_PRIMARY_MODE
                )
            )
        ]
        * len(rows)
    )
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "cache_schema_version": CACHE_SCHEMA,
        "cache_format": "numpy-npz-allow-pickle-false",
        "cache_sha256": sha256_file(cache_path),
        "protocol_sha256": semantic_hash(frozen_protocol(profile)),
        "scoring_profile": profile,
        "input_binding": expected_inputs,
        "benchmark_binding": benchmark_binding(rows),
        "fallback_binding": fallback_binding(rows, modes),
        "inference_binding": inference_binding(profile),
        "promotion_eligible": False,
    }


def validate_prepared_manifest_before_tensor_load(
    manifest: Mapping[str, Any],
    cache_path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    observed_inputs: Mapping[str, str],
    profile: str = DEFAULT_SCORING_PROFILE,
) -> None:
    required = {
        "schema_version",
        "cache_schema_version",
        "cache_format",
        "cache_sha256",
        "protocol_sha256",
        "scoring_profile",
        "input_binding",
        "benchmark_binding",
        "fallback_binding",
        "inference_binding",
        "promotion_eligible",
    }
    if set(manifest) != required:
        raise ValueError("cache manifest fields changed")
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("cache manifest schema changed")
    if manifest.get("cache_schema_version") != CACHE_SCHEMA:
        raise ValueError("cache manifest cache schema changed")
    if manifest.get("cache_format") != "numpy-npz-allow-pickle-false":
        raise ValueError("cache manifest format is not safe NPZ")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("cache manifest unexpectedly allows promotion")
    if manifest.get("scoring_profile") != profile:
        raise ValueError("cache manifest scoring profile changed")
    if manifest.get("protocol_sha256") != semantic_hash(frozen_protocol(profile)):
        raise ValueError("cache manifest protocol binding changed")
    expected_inputs = expected_cache_input_binding_for_profile(profile)
    if dict(observed_inputs) != expected_inputs:
        raise ValueError("observed cache input hashes differ from pins")
    if manifest.get("input_binding") != expected_inputs:
        raise ValueError("cache manifest input binding changed")
    if manifest.get("benchmark_binding") != benchmark_binding(expected_rows):
        raise ValueError("cache manifest benchmark binding changed")
    fallback = manifest.get("fallback_binding")
    if not isinstance(fallback, Mapping):
        raise ValueError("cache manifest fallback binding changed")
    fallback_ids = fallback.get("fallback_row_ids")
    if not isinstance(fallback_ids, list) or len(set(fallback_ids)) != len(fallback_ids):
        raise ValueError("cache manifest fallback binding changed")
    expected_row_ids = {str(row["row_id"]) for row in expected_rows}
    if set(fallback_ids) - expected_row_ids:
        raise ValueError("cache manifest fallback binding changed")
    modes = [
        (
            B1_FALLBACK_MODE
            if str(row["row_id"]) in set(fallback_ids)
            else (
                B1_CACHED_MODE
                if profile == PROFILE_WHOLE_GLYPH_CACHED_B1
                else (
                    B1_DIRECT_MODE
                    if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
                    else B1_PRIMARY_MODE
                )
            )
        )
        for row in expected_rows
    ]
    if fallback != fallback_binding(expected_rows, modes):
        raise ValueError("cache manifest fallback binding changed")
    expected_inference = inference_binding(profile)
    if manifest.get("inference_binding") != expected_inference:
        raise ValueError("cache manifest inference binding changed")
    actual_cache_hash = sha256_file(cache_path)
    if manifest.get("cache_sha256") != actual_cache_hash:
        raise ValueError("cache SHA-256 differs from write-once manifest")


_NPZ_MEMBERS = {
    "schema_version",
    "protocol_sha256",
    "scoring_profile",
    "b1_source_semantics",
    "row_ids",
    "image_paths",
    "pixel_sha256",
    "page_groups",
    "splits",
    "truth_matrix",
    "b1_glyph_scores",
    "legacy_logits",
    "segment_counts",
    "fallback_modes",
    "latency_ms",
    "promotion_eligible",
    "max_segments_per_glyph",
}


def _truth_matrix_for_cache(truth_labels: Sequence[Sequence[int]]) -> np.ndarray:
    return _truth_matrix(truth_labels, CLASS_COUNT).astype(np.uint8)


def _write_npz_cache(path: Path, cache: Mapping[str, Any]) -> None:
    latency = cache["latency"]
    arrays = {
        "schema_version": np.asarray(cache["schema_version"]),
        "protocol_sha256": np.asarray(cache["protocol_sha256"]),
        "scoring_profile": np.asarray(cache["scoring_profile"]),
        "b1_source_semantics": np.asarray(cache["b1_source_semantics"]),
        "row_ids": np.asarray(cache["row_ids"], dtype=np.str_),
        "image_paths": np.asarray(cache["image_paths"], dtype=np.str_),
        "pixel_sha256": np.asarray(cache["pixel_sha256"], dtype=np.str_),
        "page_groups": np.asarray(cache["page_groups"], dtype=np.str_),
        "splits": np.asarray(cache["splits"], dtype=np.str_),
        "truth_matrix": _truth_matrix_for_cache(cache["truth_labels"]),
        "b1_glyph_scores": np.asarray(cache["b1_glyph_scores"], dtype=np.float32),
        "legacy_logits": np.asarray(cache["legacy_logits"], dtype=np.float32),
        "segment_counts": np.asarray(cache["segment_counts"], dtype=np.int32),
        "fallback_modes": np.asarray(cache["fallback_modes"], dtype=np.str_),
        "latency_ms": np.asarray(
            [
                latency["b1_ms_per_glyph"],
                latency["legacy_ms_per_glyph"],
                latency["serial_fusion_ms_per_glyph"],
                latency["whole_glyph_b1_fallback_count"],
                latency["whole_glyph_b1_fallback_rate"],
                latency["whole_glyph_b1_fallback_classify_ms_total"],
                latency["whole_glyph_b1_fallback_classify_ms_mean"],
                latency["b1_direct_batch_size"],
                latency["b1_direct_batches"],
            ],
            dtype=np.float64,
        ),
        "promotion_eligible": np.asarray(0, dtype=np.uint8),
        "max_segments_per_glyph": np.asarray(
            cache["max_segments_per_glyph"], dtype=np.int32
        ),
    }
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _npz_scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"safe NPZ scalar text field changed: {name}")
    return str(array.item())


def _load_npz_cache(path: Path) -> dict[str, Any]:
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("prepared cache is not a safe NPZ archive") from error
    with archive_context as archive:
        if set(archive.files) != _NPZ_MEMBERS:
            raise ValueError("safe NPZ cache members changed")
        truth = np.asarray(archive["truth_matrix"])
        if truth.ndim != 2 or truth.shape[1] != CLASS_COUNT:
            raise ValueError("safe NPZ truth matrix shape changed")
        if truth.dtype != np.uint8 or not np.isin(truth, (0, 1)).all():
            raise ValueError("safe NPZ truth matrix is not binary uint8")
        latency = np.asarray(archive["latency_ms"], dtype=np.float64)
        if latency.shape != (9,):
            raise ValueError("safe NPZ latency vector shape changed")
        promotion = np.asarray(archive["promotion_eligible"])
        if promotion.shape != () or promotion.dtype != np.uint8:
            raise ValueError("safe NPZ promotion flag shape/type changed")
        max_segments = np.asarray(archive["max_segments_per_glyph"])
        if max_segments.shape != () or max_segments.dtype != np.int32:
            raise ValueError("safe NPZ max-segments field shape/type changed")
        cache = {
            "schema_version": _npz_scalar_text(
                archive["schema_version"], "schema_version"
            ),
            "protocol_sha256": _npz_scalar_text(
                archive["protocol_sha256"], "protocol_sha256"
            ),
            "scoring_profile": _npz_scalar_text(
                archive["scoring_profile"], "scoring_profile"
            ),
            "b1_source_semantics": _npz_scalar_text(
                archive["b1_source_semantics"], "b1_source_semantics"
            ),
            "row_ids": np.asarray(archive["row_ids"]).astype(str).tolist(),
            "image_paths": np.asarray(archive["image_paths"]).astype(str).tolist(),
            "pixel_sha256": np.asarray(archive["pixel_sha256"]).astype(str).tolist(),
            "page_groups": np.asarray(archive["page_groups"]).astype(str).tolist(),
            "splits": np.asarray(archive["splits"]).astype(str).tolist(),
            "truth_labels": [
                np.flatnonzero(truth[index]).astype(int).tolist()
                for index in range(len(truth))
            ],
            "b1_glyph_scores": np.asarray(
                archive["b1_glyph_scores"], dtype=np.float32
            ),
            "legacy_logits": np.asarray(archive["legacy_logits"], dtype=np.float32),
            "segment_counts": np.asarray(archive["segment_counts"], dtype=np.int32),
            "fallback_modes": np.asarray(archive["fallback_modes"]).astype(str).tolist(),
            "latency": {
                "b1_ms_per_glyph": float(latency[0]),
                "legacy_ms_per_glyph": float(latency[1]),
                "serial_fusion_ms_per_glyph": float(latency[2]),
                "whole_glyph_b1_fallback_count": int(latency[3]),
                "whole_glyph_b1_fallback_rate": float(latency[4]),
                "whole_glyph_b1_fallback_classify_ms_total": float(latency[5]),
                "whole_glyph_b1_fallback_classify_ms_mean": float(latency[6]),
                "b1_direct_batch_size": int(latency[7]),
                "b1_direct_batches": int(latency[8]),
            },
            "promotion_eligible": bool(promotion.item()),
            "max_segments_per_glyph": int(max_segments.item()),
        }
    return cache


def publish_prepared_bundle(
    cache: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    input_binding: Mapping[str, str],
    profile: str,
) -> None:
    target_cache, target_manifest = prepared_paths(profile)
    target_dir = target_cache.parent
    assert_safe_output_dir(target_dir)
    if target_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact bundle: {target_dir}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    staging = PREPARED_DIR / f".{profile}-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    cache_path = staging / target_cache.name
    provenance_path = staging / target_manifest.name
    _write_npz_cache(cache_path, cache)
    manifest = build_prepared_manifest(
        cache_path, rows, input_binding, cache["fallback_modes"], profile
    )
    provenance_path.write_text(
        json.dumps(_normalise(manifest), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        os.rename(staging, target_dir)
    except FileExistsError as error:
        raise RuntimeError(f"immutable artifact bundle already exists: {target_dir}") from error


def validate_prepared_cache(cache: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "scoring_profile",
        "b1_source_semantics",
        "row_ids",
        "image_paths",
        "page_groups",
        "splits",
        "truth_labels",
        "b1_glyph_scores",
        "legacy_logits",
        "segment_counts",
        "fallback_modes",
        "latency",
        "pixel_sha256",
        "protocol_sha256",
        "promotion_eligible",
        "max_segments_per_glyph",
    }
    missing = required - set(cache)
    if missing:
        raise ValueError(f"prepared cache missing fields: {sorted(missing)}")
    profile = str(cache["scoring_profile"])
    if profile not in SCORING_PROFILES:
        raise ValueError("prepared cache scoring profile changed")
    if cache["schema_version"] != CACHE_SCHEMA:
        raise ValueError("prepared cache schema changed")
    lengths = {
        len(cache[key])
        for key in (
            "row_ids",
            "image_paths",
            "pixel_sha256",
            "page_groups",
            "splits",
            "truth_labels",
            "segment_counts",
            "fallback_modes",
        )
    }
    if len(lengths) != 1:
        raise ValueError("prepared cache row metadata lengths differ")
    row_count = lengths.pop()
    b1 = np.asarray(cache["b1_glyph_scores"])
    legacy = np.asarray(cache["legacy_logits"])
    if b1.shape != (row_count, CLASS_COUNT):
        raise ValueError("prepared B1 score matrix shape changed")
    if legacy.shape != (row_count, LEGACY_CLASS_COUNT):
        raise ValueError("prepared legacy logit matrix shape changed")
    if not np.isfinite(b1).all() or not np.isfinite(legacy).all():
        raise ValueError("prepared score cache contains non-finite values")
    if len(set(str(value) for value in cache["row_ids"])) != row_count:
        raise ValueError("prepared row IDs are not unique")
    if set(cache["splits"]) != {"train", "dev"}:
        raise ValueError("prepared cache must retain train/dev rows")
    if any(not labels for labels in cache["truth_labels"]):
        raise ValueError("prepared cache contains unresolved glyph truth")
    modes = [str(mode) for mode in cache["fallback_modes"]]
    if set(modes) - {
        B1_PRIMARY_MODE,
        B1_FALLBACK_MODE,
        B1_CACHED_MODE,
        B1_DIRECT_MODE,
    }:
        raise ValueError("prepared cache contains an invalid fallback mode")
    for count, mode in zip(cache["segment_counts"], modes, strict=True):
        count = int(count)
        if (mode == B1_FALLBACK_MODE and count != 0) or (
            mode == B1_PRIMARY_MODE and count <= 0
        ) or (
            mode == B1_CACHED_MODE and count != 0
        ) or (
            mode == B1_DIRECT_MODE and count != 0
        ):
            raise ValueError("prepared cache segment count/fallback mode mismatch")
    latency = cache["latency"]
    expected_latency = {
        "b1_ms_per_glyph",
        "legacy_ms_per_glyph",
        "serial_fusion_ms_per_glyph",
        "whole_glyph_b1_fallback_count",
        "whole_glyph_b1_fallback_rate",
        "whole_glyph_b1_fallback_classify_ms_total",
        "whole_glyph_b1_fallback_classify_ms_mean",
        "b1_direct_batch_size",
        "b1_direct_batches",
    }
    if not isinstance(latency, Mapping) or set(latency) != expected_latency:
        raise ValueError("prepared cache latency fields changed")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in latency.values()):
        raise ValueError("prepared cache latency values are invalid")
    fallback_count = modes.count(B1_FALLBACK_MODE)
    if int(latency["whole_glyph_b1_fallback_count"]) != fallback_count:
        raise ValueError("prepared cache fallback count/latency mismatch")
    if not math.isclose(
        float(latency["whole_glyph_b1_fallback_rate"]),
        fallback_count / row_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("prepared cache fallback rate/rows mismatch")
    expected_direct_batch_size = (
        DIRECT_B1_BATCH_SIZE
        if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
        else 0
    )
    expected_direct_batches = (
        math.ceil(row_count / DIRECT_B1_BATCH_SIZE)
        if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
        else 0
    )
    if int(latency["b1_direct_batch_size"]) != expected_direct_batch_size:
        raise ValueError("prepared cache direct B1 batch size changed")
    if int(latency["b1_direct_batches"]) != expected_direct_batches:
        raise ValueError("prepared cache direct B1 batch count changed")
    if cache["protocol_sha256"] != semantic_hash(frozen_protocol(profile)):
        raise ValueError("prepared cache protocol binding changed")
    expected_semantics = {
        PROFILE_WHOLE_GLYPH_DIRECT_B1: (
            "CodexClassifier.classify_batch raw cosine scores on whole glyph JPEG; "
            "exact runtime preprocessing and backbone included"
        ),
        PROFILE_WHOLE_GLYPH_CACHED_B1: (
            "B1 projection+prototypes forward on pinned P4 base features; "
            "backbone extraction excluded; research diagnostic only"
        ),
        PROFILE_MOBILE_SAM_B1: (
            "MobileSAM segmentation plus B1 backbone/projection/prototypes"
        ),
    }[profile]
    if cache["b1_source_semantics"] != expected_semantics:
        raise ValueError("prepared cache B1 source/latency semantics changed")
    if cache["promotion_eligible"] is not False:
        raise ValueError("prepared research cache unexpectedly allows promotion")
    expected_max_segments = (
        MAX_SEGMENTS_PER_GLYPH if profile == PROFILE_MOBILE_SAM_B1 else 0
    )
    if cache["max_segments_per_glyph"] != expected_max_segments:
        raise ValueError("prepared cache max-segments inference binding changed")


def prepare(
    *,
    device: str,
    batch_size: int,
    apply: bool,
    profile: str = DEFAULT_SCORING_PROFILE,
) -> dict[str, Any]:
    if profile not in SCORING_PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    rows, taxonomy, audit = preflight(profile)
    cache_path, _ = prepared_paths(profile)
    plan = {
        "action": "prepare real run_50 plus frozen-profile B1 glyph score cache",
        "scoring_profile": profile,
        "apply_required": True,
        "rows": len(rows),
        "device": device,
        "run50_batch_size": batch_size,
        "direct_b1_batch_size": (
            DIRECT_B1_BATCH_SIZE
            if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1
            else None
        ),
        "max_segments_per_glyph": (
            MAX_SEGMENTS_PER_GLYPH if profile == PROFILE_MOBILE_SAM_B1 else None
        ),
        "cache_path": str(cache_path),
        "preflight": audit,
        "promotion_eligible": False,
    }
    if not apply:
        return {**plan, "status": "DRY_RUN"}
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch = _torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA prepare requested but CUDA is unavailable")
    run50 = _load_run50_model(device, taxonomy.legacy_names)
    legacy_logits, legacy_latency = score_run50(rows, run50, device, batch_size)
    if profile == PROFILE_WHOLE_GLYPH_DIRECT_B1:
        classifier = _load_b1_classifier(device)
        if tuple(classifier.class_names) != taxonomy.runtime_names:
            raise ValueError("loaded B1 class order differs from pinned taxonomy")
        b1_scores, direct_timing = score_direct_b1_whole_glyph(rows, classifier)
        segment_counts = [0] * len(rows)
        fallback_modes = [B1_DIRECT_MODE] * len(rows)
        b1_timing = {
            **direct_timing,
            "whole_glyph_b1_fallback_count": 0,
            "whole_glyph_b1_fallback_rate": 0.0,
            "whole_glyph_b1_fallback_classify_ms_total": 0.0,
            "whole_glyph_b1_fallback_classify_ms_mean": 0.0,
        }
        b1_source_semantics = (
            "CodexClassifier.classify_batch raw cosine scores on whole glyph JPEG; "
            "exact runtime preprocessing and backbone included"
        )
    elif profile == PROFILE_WHOLE_GLYPH_CACHED_B1:
        base_features, feature_audit = load_pinned_base_features(rows)
        b1_scores, cached_timing = score_cached_b1_base_features(
            base_features, taxonomy, device
        )
        segment_counts = [0] * len(rows)
        fallback_modes = [B1_CACHED_MODE] * len(rows)
        b1_timing = {
            "b1_ms_per_glyph": cached_timing["b1_ms_per_glyph"],
            "b1_direct_batch_size": 0,
            "b1_direct_batches": 0,
            "whole_glyph_b1_fallback_count": 0,
            "whole_glyph_b1_fallback_rate": 0.0,
            "whole_glyph_b1_fallback_classify_ms_total": 0.0,
            "whole_glyph_b1_fallback_classify_ms_mean": 0.0,
        }
        b1_source_semantics = (
            "B1 projection+prototypes forward on pinned P4 base features; "
            "backbone extraction excluded; research diagnostic only"
        )
        if feature_audit != audit.get("whole_glyph_cached_b1"):
            raise ValueError("P4 feature cache audit changed between preflight and scoring")
    else:
        classifier, segmenter = _load_b1_and_segmenter(device)
        if tuple(classifier.class_names) != taxonomy.runtime_names:
            raise ValueError("loaded B1 class order differs from pinned taxonomy")
        b1_scores, segment_counts, fallback_modes, b1_timing = (
            score_b1_glyph_aggregate(
                rows, classifier, segmenter, MAX_SEGMENTS_PER_GLYPH
            )
        )
        b1_timing["b1_direct_batch_size"] = 0
        b1_timing["b1_direct_batches"] = 0
        b1_source_semantics = (
            "MobileSAM segmentation plus B1 backbone/projection/prototypes"
        )
    cache = {
        "schema_version": CACHE_SCHEMA,
        "protocol_sha256": semantic_hash(frozen_protocol(profile)),
        "scoring_profile": profile,
        "b1_source_semantics": b1_source_semantics,
        "row_ids": [row["row_id"] for row in rows],
        "image_paths": [row["image_path"] for row in rows],
        "pixel_sha256": [row["pixel_sha256"] for row in rows],
        "page_groups": [row["page_group"] for row in rows],
        "splits": [row["split"] for row in rows],
        "truth_labels": [row["truth_labels"] for row in rows],
        "b1_glyph_scores": b1_scores,
        "legacy_logits": legacy_logits,
        "segment_counts": segment_counts,
        "fallback_modes": fallback_modes,
        "latency": {
            **b1_timing,
            "legacy_ms_per_glyph": legacy_latency,
            "serial_fusion_ms_per_glyph": b1_timing["b1_ms_per_glyph"]
            + legacy_latency,
        },
        "promotion_eligible": False,
        "max_segments_per_glyph": (
            MAX_SEGMENTS_PER_GLYPH if profile == PROFILE_MOBILE_SAM_B1 else 0
        ),
    }
    validate_prepared_cache(cache)
    publish_prepared_bundle(
        cache, rows, observed_cache_input_binding(audit, profile), profile
    )
    return {**plan, "status": "PREPARED", "latency": cache["latency"]}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_fusion_variants(
    b1_scores: np.ndarray,
    legacy_logits: np.ndarray,
    taxonomy: Taxonomy,
    *,
    prior_weight: float = PRIOR_WEIGHT,
    cascade_threshold: float = CASCADE_THRESHOLD,
) -> dict[str, np.ndarray]:
    b1 = np.asarray(b1_scores)
    legacy_raw = np.asarray(legacy_logits, dtype=np.float64)
    if b1.ndim != 2 or b1.shape[1] != CLASS_COUNT or b1.dtype.kind != "f":
        raise ValueError("B1 scores must be N x 286")
    if legacy_raw.shape != (len(b1), LEGACY_CLASS_COUNT):
        raise ValueError("legacy logits must be N x 33")
    legacy_runtime = np.full(b1.shape, -1e9, dtype=np.float64)
    shared_prior = np.zeros(b1.shape, dtype=np.float64)
    shared_legacy_columns: list[int] = []
    for legacy_index, runtime_index in enumerate(taxonomy.legacy_to_runtime):
        if runtime_index is None:
            continue
        legacy_runtime[:, runtime_index] = legacy_raw[:, legacy_index]
        shared_prior[:, runtime_index] = legacy_raw[:, legacy_index]
        shared_legacy_columns.append(legacy_index)
    prior = np.array(b1, copy=True)
    prior[:, taxonomy.shared_runtime_indices] = (
        np.asarray(b1[:, taxonomy.shared_runtime_indices], dtype=np.float64)
        + prior_weight * shared_prior[:, taxonomy.shared_runtime_indices]
    )
    confidence = _sigmoid(legacy_raw[:, shared_legacy_columns]).max(axis=1)
    cascade = np.array(b1, copy=True)
    active_indices = np.flatnonzero(confidence >= cascade_threshold)
    cascade[np.ix_(active_indices, taxonomy.shared_runtime_indices)] = prior[
        np.ix_(active_indices, taxonomy.shared_runtime_indices)
    ]
    for candidate in (prior, cascade):
        if not np.array_equal(
            candidate[:, taxonomy.b1_only_indices], b1[:, taxonomy.b1_only_indices]
        ):
            raise AssertionError("fusion changed a protected B1-only class")
    return {
        "b1": np.array(b1, copy=True),
        "legacy": legacy_runtime,
        "prior_global": prior,
        "cascade": cascade,
    }


def _average_precision(scores: np.ndarray, positives: set[int]) -> float:
    if not positives:
        raise ValueError("average precision requires at least one positive")
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if int(index) in positives:
            hits += 1
            total += hits / rank
    return total / len(positives)


def _binary_average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    positives = int(truth.sum())
    if positives == 0:
        raise ValueError("binary AP requires a positive class")
    order = np.argsort(-scores, kind="stable")
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if truth[index]:
            hits += 1
            total += hits / rank
    return total / positives


def evaluate_scores(
    scores: np.ndarray,
    truth_labels: Sequence[Sequence[int]],
    page_groups: Sequence[str],
) -> dict[str, Any]:
    logits = np.asarray(scores, dtype=np.float64)
    if logits.ndim != 2 or len(logits) != len(truth_labels) or len(logits) != len(page_groups):
        raise ValueError("score/truth/group row counts differ")
    class_count = logits.shape[1]
    if class_count <= 0:
        raise ValueError("score matrix has no classes")
    truth = np.zeros_like(logits, dtype=bool)
    sample_ap: list[float] = []
    recall5: list[float] = []
    set_f1: list[float] = []
    probabilities = _sigmoid(logits)
    per_group_values: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row_index, labels in enumerate(truth_labels):
        positives = {int(label) for label in labels}
        if not positives or min(positives) < 0 or max(positives) >= class_count:
            raise ValueError("truth labels are empty or outside score taxonomy")
        truth[row_index, list(positives)] = True
        ap = _average_precision(logits[row_index], positives)
        top5 = set(np.argsort(-logits[row_index], kind="stable")[:5].tolist())
        recall = len(top5 & positives) / len(positives)
        predicted = set(np.flatnonzero(probabilities[row_index] >= SET_THRESHOLD).tolist())
        tp = len(predicted & positives)
        denominator = 2 * tp + len(predicted - positives) + len(positives - predicted)
        f1 = 2 * tp / denominator if denominator else 1.0
        sample_ap.append(ap)
        recall5.append(recall)
        set_f1.append(f1)
        per_group_values[str(page_groups[row_index])].append((ap, recall, f1))
    macro_values = [
        _binary_average_precision(logits[:, index], truth[:, index])
        for index in range(class_count)
        if truth[:, index].any()
    ]
    flat_probabilities = probabilities.reshape(-1)
    flat_truth = truth.astype(np.float64).reshape(-1)
    bins = np.minimum((flat_probabilities * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(flat_probabilities[mask].mean()) - float(flat_truth[mask].mean())
            )
    per_planche = {
        group: {
            "sample_ap": float(np.mean([value[0] for value in values])),
            "recall_at_5": float(np.mean([value[1] for value in values])),
            "set_f1": float(np.mean([value[2] for value in values])),
            "glyphs": len(values),
        }
        for group, values in sorted(per_group_values.items())
    }
    return {
        "glyphs": {
            "macro_ap": float(np.mean(macro_values)),
            "sample_ap": float(np.mean(sample_ap)),
            "recall_at_5": float(np.mean(recall5)),
            "set_f1": float(np.mean(set_f1)),
            "glyphs": len(logits),
            "evaluable_classes": len(macro_values),
        },
        "calibration": {
            "ece_10": ece,
            "multilabel_brier": float(np.mean((probabilities - truth) ** 2)),
            "set_threshold": SET_THRESHOLD,
        },
        "per_planche": per_planche,
    }


def evaluate_b1_only_ranking(
    scores: np.ndarray,
    truth_labels: Sequence[Sequence[int]],
    b1_only_indices: Sequence[int],
) -> dict[str, Any]:
    logits = np.asarray(scores)
    protected = tuple(int(index) for index in b1_only_indices)
    if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT:
        raise ValueError("B1-only ranking scores must be N x 286")
    if len(logits) != len(truth_labels) or len(protected) != B1_ONLY_CLASS_COUNT:
        raise ValueError("B1-only ranking input shape/taxonomy changed")
    protected_set = set(protected)
    sample_ap: list[float] = []
    recall5: list[float] = []
    top1_hits: list[float] = []
    for row_index, labels in enumerate(truth_labels):
        positives = {int(label) for label in labels} & protected_set
        if not positives:
            continue
        row = logits[row_index]
        order = np.argsort(-row, kind="stable")
        sample_ap.append(_average_precision(row, positives))
        top5 = set(order[:5].tolist())
        recall5.append(len(top5 & positives) / len(positives))
        top1_hits.append(float(int(order[0]) in positives))
    if not sample_ap:
        raise ValueError("dev benchmark has no glyph truth containing a B1-only class")
    return {
        "conditional_glyphs": len(sample_ap),
        "truth_scope": "B1-only positives",
        "ranking_universe": CLASS_COUNT,
        "sample_ap": float(np.mean(sample_ap)),
        "recall_at_5": float(np.mean(recall5)),
        "top1_hit_rate": float(np.mean(top1_hits)),
    }


def _truth_matrix(
    truth_labels: Sequence[Sequence[int]], class_count: int
) -> np.ndarray:
    truth = np.zeros((len(truth_labels), class_count), dtype=np.float64)
    for row_index, labels in enumerate(truth_labels):
        positives = {int(label) for label in labels}
        if not positives or min(positives) < 0 or max(positives) >= class_count:
            raise ValueError("truth labels are empty or outside score taxonomy")
        truth[row_index, list(positives)] = 1.0
    return truth


def _binary_cross_entropy(logits: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    if values.shape != targets.shape or values.size == 0:
        raise ValueError("calibration logits/truth shapes differ or are empty")
    losses = np.maximum(values, 0.0) - values * targets + np.log1p(
        np.exp(-np.abs(values))
    )
    return float(losses.mean())


def _mean_sample_ap(
    scores: np.ndarray, truth_labels: Sequence[Sequence[int]]
) -> float:
    if len(scores) != len(truth_labels) or not len(scores):
        raise ValueError("fusion selection requires non-empty aligned train rows")
    return float(
        np.mean(
            [
                _average_precision(scores[index], {int(label) for label in labels})
                for index, labels in enumerate(truth_labels)
            ]
        )
    )


def fit_train_parameters(
    b1_train: np.ndarray,
    legacy_train: np.ndarray,
    truth_labels: Sequence[Sequence[int]],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Fit frozen calibration/fusion grids using train glyphs only."""

    b1 = np.asarray(b1_train, dtype=np.float64)
    legacy = np.asarray(legacy_train, dtype=np.float64)
    if b1.shape != (len(truth_labels), CLASS_COUNT):
        raise ValueError("train B1 score matrix shape changed")
    if legacy.shape != (len(truth_labels), LEGACY_CLASS_COUNT):
        raise ValueError("train legacy logit matrix shape changed")
    truth = _truth_matrix(truth_labels, CLASS_COUNT)
    shared_runtime_indices = [
        int(runtime_index)
        for runtime_index in taxonomy.legacy_to_runtime
        if runtime_index is not None
    ]

    b1_candidates = [
        (
            _binary_cross_entropy(
                b1[:, shared_runtime_indices] / temperature,
                truth[:, shared_runtime_indices],
            ),
            temperature,
        )
        for temperature in TEMPERATURE_GRID
    ]
    b1_loss, b1_temperature = min(b1_candidates, key=lambda item: (item[0], item[1]))
    prior_candidates: list[tuple[float, float]] = []
    for weight in PRIOR_WEIGHT_GRID:
        variants = build_fusion_variants(
            b1,
            legacy,
            taxonomy,
            prior_weight=weight,
            cascade_threshold=CASCADE_THRESHOLD,
        )
        prior_candidates.append(
            (_mean_sample_ap(variants["prior_global"], truth_labels), weight)
        )
    prior_ap, prior_weight = max(
        prior_candidates, key=lambda item: (item[0], -item[1])
    )

    cascade_candidates: list[tuple[float, float]] = []
    for threshold in CASCADE_THRESHOLD_GRID:
        variants = build_fusion_variants(
            b1,
            legacy,
            taxonomy,
            prior_weight=prior_weight,
            cascade_threshold=threshold,
        )
        cascade_candidates.append(
            (_mean_sample_ap(variants["cascade"], truth_labels), threshold)
        )
    cascade_ap, cascade_threshold = max(
        cascade_candidates, key=lambda item: (item[0], -item[1])
    )
    return {
        "b1_temperature": float(b1_temperature),
        "prior_weight": float(prior_weight),
        "cascade_threshold": float(cascade_threshold),
        "train_objectives": {
            "b1_binary_cross_entropy": b1_loss,
            "prior_global_sample_ap": prior_ap,
            "cascade_sample_ap": cascade_ap,
        },
    }


def build_train_selected_variants(
    b1_scores: np.ndarray,
    legacy_logits: np.ndarray,
    splits: Sequence[str],
    truth_labels: Sequence[Sequence[int]],
    taxonomy: Taxonomy,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select on train rows, then apply the frozen transform to every row."""

    if not (
        len(b1_scores) == len(legacy_logits) == len(splits) == len(truth_labels)
    ):
        raise ValueError("calibration input row counts differ")
    if set(splits) != {"train", "dev"}:
        raise ValueError("calibration requires explicit train/dev rows")
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    train_truth = [truth_labels[index] for index in train_indices]
    parameters = fit_train_parameters(
        np.asarray(b1_scores)[train_indices],
        np.asarray(legacy_logits)[train_indices],
        train_truth,
        taxonomy,
    )
    calibrated_b1 = np.asarray(b1_scores, dtype=np.float64).copy()
    calibrated_b1[:, taxonomy.shared_runtime_indices] /= parameters[
        "b1_temperature"
    ]
    raw_variants = build_fusion_variants(
        np.asarray(b1_scores),
        np.asarray(legacy_logits),
        taxonomy,
        prior_weight=parameters["prior_weight"],
        cascade_threshold=parameters["cascade_threshold"],
    )
    variants = {
        "b1": raw_variants["b1"],
        "b1_calibrated": calibrated_b1,
        "legacy": raw_variants["legacy"],
        "prior_global": raw_variants["prior_global"],
        "cascade": raw_variants["cascade"],
    }
    report = {
        "selection_split": "train",
        "train_glyphs": len(train_indices),
        "parameters": parameters,
    }
    return variants, report


def build_oracle_scores(
    variants: Mapping[str, np.ndarray], truth_labels: Sequence[Sequence[int]]
) -> tuple[np.ndarray, list[str]]:
    allowed = ("b1", "legacy", "prior_global", "cascade")
    if set(variants) != set(allowed):
        raise ValueError("oracle requires exactly the four non-oracle variants")
    matrices = {name: np.asarray(variants[name]) for name in allowed}
    shape = matrices["b1"].shape
    if any(matrix.shape != shape for matrix in matrices.values()) or len(truth_labels) != shape[0]:
        raise ValueError("oracle input shapes differ")
    output = np.empty_like(matrices["b1"])
    sources: list[str] = []
    for row_index, labels in enumerate(truth_labels):
        positives = {int(label) for label in labels}
        winner = max(
            allowed,
            key=lambda name: (
                _average_precision(matrices[name][row_index], positives),
                -allowed.index(name),
            ),
        )
        output[row_index] = matrices[winner][row_index]
        sources.append(winner)
    return output, sources


def paired_planche_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    groups = sorted(set(candidate) & set(baseline))
    if set(candidate) != set(baseline) or not groups:
        raise ValueError("paired planche bootstrap requires identical non-empty groups")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    deltas = np.asarray([float(candidate[group]) - float(baseline[group]) for group in groups])
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(axis=1)
    return {
        "method": "paired bootstrap over whole planche groups",
        "groups": len(groups),
        "iterations": iterations,
        "seed": seed,
        "mean": float(deltas.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def final_gate(
    candidate_name: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    latency_ms_per_glyph: float,
    candidate_b1_only: Mapping[str, Any],
    baseline_b1_only: Mapping[str, Any],
    b1_only_logits_identical: bool,
) -> dict[str, Any]:
    if candidate_name not in {"prior_global", "cascade"}:
        raise ValueError("only frozen fusion candidates may enter the sanity gate")
    descriptive_glyph_gain = float(candidate["glyphs"]["sample_ap"]) - float(
        baseline["glyphs"]["sample_ap"]
    )
    mean_planche_gain = float(bootstrap["mean"])
    passes = {
        "mean_planche_sample_ap_gain_at_least_0_01": mean_planche_gain >= 0.01,
        "paired_planche_ci_lower_positive": float(bootstrap["lower_95"]) > 0.0,
        "latency_within_250ms_per_glyph": float(latency_ms_per_glyph)
        <= LATENCY_LIMIT_MS,
        "b1_only_logits_bit_identical": bool(b1_only_logits_identical),
        "b1_only_sample_ap_no_degradation": float(candidate_b1_only["sample_ap"])
        >= float(baseline_b1_only["sample_ap"]),
        "b1_only_recall_at_5_no_degradation": float(
            candidate_b1_only["recall_at_5"]
        )
        >= float(baseline_b1_only["recall_at_5"]),
        "b1_only_top1_hit_rate_no_degradation": float(
            candidate_b1_only["top1_hit_rate"]
        )
        >= float(baseline_b1_only["top1_hit_rate"]),
    }
    return {
        "candidate": candidate_name,
        "mean_planche_sample_ap_gain": mean_planche_gain,
        "glyph_weighted_sample_ap_gain_descriptive": descriptive_glyph_gain,
        "paired_planche_bootstrap": dict(bootstrap),
        "b1_only_ranking": {
            "candidate": dict(candidate_b1_only),
            "raw_b1_baseline": dict(baseline_b1_only),
        },
        "latency_ms_per_glyph": float(latency_ms_per_glyph),
        "passes": passes,
        "sanity_pass": all(passes.values()),
        "verdict": "INSUFFICIENT_FOR_ELEMENT_GATE",
        "fresh_expanded_element_benchmark_required": True,
        "automatic_global_training_allowed": False,
        "promotion_eligible": False,
    }


def _jsonable_metrics(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metrics(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_prepared_cache(
    expected_rows: Sequence[Mapping[str, Any]],
    observed_inputs: Mapping[str, str],
    *,
    cache_path: Path | None = None,
    manifest_path: Path | None = None,
    profile: str = DEFAULT_SCORING_PROFILE,
) -> dict[str, Any]:
    default_cache_path, default_manifest_path = prepared_paths(profile)
    cache_path = cache_path or default_cache_path
    manifest_path = manifest_path or default_manifest_path
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "immutable prepared cache is absent; review verify, then run prepare --apply"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("prepared cache manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("prepared cache manifest is not an object")
    validate_prepared_manifest_before_tensor_load(
        manifest, cache_path, expected_rows, observed_inputs, profile
    )
    cache = _load_npz_cache(cache_path)
    validate_prepared_cache(cache)
    expected = benchmark_binding(expected_rows)
    observed = benchmark_binding(
        [
            {
                "row_id": cache["row_ids"][index],
                "pixel_sha256": cache["pixel_sha256"][index],
                "split": cache["splits"][index],
                "page_group": cache["page_groups"][index],
                "truth_labels": cache["truth_labels"][index],
            }
            for index in range(len(cache["row_ids"]))
        ]
    )
    if observed != expected:
        raise ValueError("loaded cache benchmark metadata differs from manifest binding")
    if manifest["fallback_binding"] != fallback_binding(
        expected_rows, cache["fallback_modes"]
    ):
        raise ValueError("loaded cache fallback metadata differs from manifest binding")
    return cache


def run_all(profile: str = DEFAULT_SCORING_PROFILE) -> dict[str, Any]:
    assert_safe_output_dir(REPORT_DIR)
    rows, taxonomy, audit = preflight(profile)
    cache_path, _ = prepared_paths(profile)
    results_path, final_path = result_paths(profile)
    cache = load_prepared_cache(
        rows, observed_cache_input_binding(audit, profile), profile=profile
    )
    variants, calibration = build_train_selected_variants(
        np.asarray(cache["b1_glyph_scores"]),
        np.asarray(cache["legacy_logits"]),
        list(cache["splits"]),
        list(cache["truth_labels"]),
        taxonomy,
    )
    splits = list(cache["splits"])
    dev_indices = [index for index, split in enumerate(splits) if split == "dev"]
    if not dev_indices:
        raise ValueError("prepared cache contains no strict dev glyphs")
    truths = [cache["truth_labels"][index] for index in dev_indices]
    groups = [cache["page_groups"][index] for index in dev_indices]
    dev_variants = {name: scores[dev_indices] for name, scores in variants.items()}
    oracle_inputs = {
        name: dev_variants[name]
        for name in ("b1", "legacy", "prior_global", "cascade")
    }
    oracle_scores, oracle_sources = build_oracle_scores(oracle_inputs, truths)
    all_dev = {**dev_variants, "oracle": oracle_scores}
    metrics = {
        name: evaluate_scores(scores, truths, groups) for name, scores in all_dev.items()
    }
    for name, scores in all_dev.items():
        metrics[name]["b1_only_ranking"] = evaluate_b1_only_ranking(
            scores, truths, taxonomy.b1_only_indices
        )
    gates: dict[str, Any] = {}
    fusion_latency = float(cache["latency"]["serial_fusion_ms_per_glyph"])
    for name in ("prior_global", "cascade"):
        bootstrap = paired_planche_bootstrap(
            {
                group: values["sample_ap"]
                for group, values in metrics[name]["per_planche"].items()
            },
            {
                group: values["sample_ap"]
                for group, values in metrics["b1"]["per_planche"].items()
            },
        )
        gates[name] = final_gate(
            name,
            metrics[name],
            metrics["b1"],
            bootstrap,
            latency_ms_per_glyph=fusion_latency,
            candidate_b1_only=metrics[name]["b1_only_ranking"],
            baseline_b1_only=metrics["b1"]["b1_only_ranking"],
            b1_only_logits_identical=np.array_equal(
                dev_variants[name][:, taxonomy.b1_only_indices],
                dev_variants["b1"][:, taxonomy.b1_only_indices],
            ),
        )
    results = {
        "schema_version": RESULTS_SCHEMA,
        "scoring_profile": profile,
        "protocol": frozen_protocol(profile),
        "cache_sha256": sha256_file(cache_path),
        "dev_glyphs": len(dev_indices),
        "dev_planches": len(set(groups)),
        "train_only_calibration_and_selection": calibration,
        "metrics": metrics,
        "latency": cache["latency"],
        "latency_semantics": cache["b1_source_semantics"],
        "b1_zero_crop_fallback": {
            "policy": B1_FALLBACK_POLICY,
            "mode": B1_FALLBACK_MODE,
            "count": cache["fallback_modes"].count(B1_FALLBACK_MODE),
            "row_ids": [
                cache["row_ids"][index]
                for index, mode in enumerate(cache["fallback_modes"])
                if mode == B1_FALLBACK_MODE
            ],
            "adaptive_exclusion": False,
        },
        "oracle_source_counts": dict(Counter(oracle_sources)),
        "gates": gates,
        "verdict": "INSUFFICIENT_FOR_ELEMENT_GATE",
        "fresh_expanded_element_benchmark_required": True,
        "automatic_global_training_allowed": False,
        "promotion_eligible": False,
    }
    write_once_json(results_path, _jsonable_metrics(results))
    sanity_winners = [name for name, gate in gates.items() if gate["sanity_pass"]]
    final = {
        "schema_version": "autoresearch-legacy-fusion-final.v1",
        "scoring_profile": profile,
        "results_sha256": sha256_file(results_path),
        "sanity_winners": sanity_winners,
        "verdict": "INSUFFICIENT_FOR_ELEMENT_GATE",
        "reason": (
            "P4 is a small glyph relation benchmark after strict run_50 pixel exclusion; "
            "it cannot establish an element-level non-regression or promotion gate."
        ),
        "required_next_evidence": "larger fresh provenance-disjoint element benchmark",
        "automatic_global_training_allowed": False,
        "promotion_eligible": False,
    }
    write_once_json(final_path, final)
    return final


def verify(
    *,
    include_prepared: bool = True,
    profile: str = DEFAULT_SCORING_PROFILE,
) -> dict[str, Any]:
    rows, taxonomy, audit = preflight(profile)
    cache_path, _ = prepared_paths(profile)
    result: dict[str, Any] = {
        "status": "INPUTS_VERIFIED",
        "scoring_profile": profile,
        "strict_rows_after_exclusion": len(rows),
        "shared_classes": len(taxonomy.shared_names),
        "b1_only_classes": len(taxonomy.b1_only_names),
        "audit": audit,
        "prepared_cache": "absent",
        "promotion_eligible": False,
    }
    if include_prepared and cache_path.exists():
        cache = load_prepared_cache(
            rows, observed_cache_input_binding(audit, profile), profile=profile
        )
        result["prepared_cache"] = "verified"
        result["prepared_rows"] = len(cache["row_ids"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only run_50 + B1 Tlachia glyph fusion spike"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="hash and validate inputs/cache")
    verify_parser.add_argument("--inputs-only", action="store_true")
    verify_parser.add_argument(
        "--profile", choices=SCORING_PROFILES, default=DEFAULT_SCORING_PROFILE
    )
    prepare_parser = sub.add_parser("prepare", help="prepare immutable real-model score cache")
    prepare_parser.add_argument("--device", default="cuda")
    prepare_parser.add_argument("--batch-size", type=int, default=16)
    prepare_parser.add_argument(
        "--profile", choices=SCORING_PROFILES, default=DEFAULT_SCORING_PROFILE
    )
    prepare_parser.add_argument(
        "--apply",
        action="store_true",
        help="explicitly run the selected B1 profile plus run_50 and write the cache",
    )
    run_parser = sub.add_parser("run-all", help="evaluate frozen variants from prepared cache")
    run_parser.add_argument(
        "--profile", choices=SCORING_PROFILES, default=DEFAULT_SCORING_PROFILE
    )
    protocol_parser = sub.add_parser("protocol", help="print the frozen protocol")
    protocol_parser.add_argument(
        "--profile", choices=SCORING_PROFILES, default=DEFAULT_SCORING_PROFILE
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        output = verify(
            include_prepared=not args.inputs_only, profile=args.profile
        )
    elif args.command == "prepare":
        output = prepare(
            device=args.device,
            batch_size=args.batch_size,
            apply=args.apply,
            profile=args.profile,
        )
    elif args.command == "run-all":
        output = run_all(args.profile)
    else:
        output = frozen_protocol(args.profile)
    print(json.dumps(_jsonable_metrics(output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

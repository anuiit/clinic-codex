#!/usr/bin/env python3
"""Freeze, download and verify a bounded expanded Tlachia glyph benchmark.

This script deliberately stops before model inference.  It produces the fresh,
page-disjoint corpus required by ``autoresearch_legacy_fusion.py`` without
opening business-final, annotation, or locked-test inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from codex_pipeline.data.tlachia.catalog import CachedTransport
from codex_pipeline.data.tlachia.core import (
    PolicyError,
    artifact,
    read_json,
    verify_artifact,
    write_json,
)
from codex_pipeline.data.tlachia.media import (
    MEDIA_MANIFEST_SCHEMA,
    MEDIA_PLAN_SCHEMA,
    download_media_plan,
)


API = REPO / "reports/tlachia-p3-20260813/catalog/api-global-elements-v1.json"
JOIN = REPO / "reports/tlachia-p3-20260813/catalog/join-global-elements-v1.json"
ZIP = REPO / "reports/tlachia-p0-p2-20260813/catalog/zip-all-v2.json"
MAPPING = REPO / "reports/tlachia-p3-20260813/mapping/global-runtime-business-approved-v2.json"
RUNTIME_CONFIG = REPO / "backend/codex_model/config.json"
RUN50_LABELS = Path("/mnt/f/CODEX/AI_clinic_class/app/backend/labels.json")
P4_PLAN = REPO / "reports/p4-multiview-20260813/plans/pilot-v1.json"

SOURCE_SHA256 = {
    API: "74ffd60efcef960f13d16d73e0003569ef2ea1b305c11c99c323b6d851a9909f",
    JOIN: "9f7d92f3fa7f71e8afec2a86a760234329332eecbf670f05e43d7f503cd46784",
    ZIP: "c749f1c03ead30e304e555ea8a52537aca62c32eb383b3881d7ffd6a67ce05bb",
    MAPPING: "a17d3bc7b159d5eac06cf0af8ee482ea26e3264e895c4db0c6a3690f079fd1a3",
    RUNTIME_CONFIG: "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
    RUN50_LABELS: "6231a929786aa50b15a6610311355364cbebc707476ebe9391068b5e465470be",
    P4_PLAN: "4db90b3f2829d55100d049acfa5f104f1b4883dc23ee180d6587aa83b7564fe8",
}

TARGET_FILES = 2_000
MIN_GROUPS = 200
DEV_GROUPS = 50
MAX_PER_GROUP = 10
LIMITS = {
    "max_files": 2_000,
    "max_groups": 400,
    "max_compressed_bytes": 80_000_000,
    "max_uncompressed_bytes": 100_000_000,
    "max_requests": 4_000,
    "max_pixel_count": 1_000_000_000,
}
FORBIDDEN = ("annotations", "business-final", "business_final", "locked-test", "locked_test")
ALLOWED_MAPPING = frozenset({"accepted_exact", "approved_manual"})
SPLIT_NAMESPACE = "legacy-fusion-expanded-v1"
P4_FROZEN_DEV_GROUPS = frozenset({
    "tlachia:codex:44:planche:MC2",
    "tlachia:codex:1:planche:X.050",
    "tlachia:codex:18:planche:385_08r",
    "tlachia:codex:48:planche:K58_B",
    "tlachia:codex:43:planche:MT_14r",
    "tlachia:codex:36:planche:387_816v",
})


def _load_fusion_module() -> Any:
    """Load the existing, pinned B1/run_50 scorer without changing its P4 flow."""
    path = REPO / "scripts/autoresearch_legacy_fusion.py"
    spec = importlib.util.spec_from_file_location("legacy_fusion_expanded_scorer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def verify_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in SOURCE_SHA256.items():
        lowered = str(path).replace("\\", "/").lower()
        if any(token in lowered for token in FORBIDDEN):
            raise PolicyError(f"forbidden benchmark input: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PolicyError(f"source SHA-256 drift: {path}: {actual}")
        observed[str(path.resolve())] = actual
    return observed


def _load_p4_module() -> Any:
    path = REPO / "scripts/tlachia_p4_multiview.py"
    spec = importlib.util.spec_from_file_location("tlachia_p4_expanded_import", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strict_glyph_nodes(api: dict[str, Any], mapping: dict[str, Any]) -> set[str]:
    mapped: dict[str, str] = {}
    for row in mapping["payload"]["mappings"]:
        for node in row.get("node_keys", []):
            status = str(row.get("status"))
            old = mapped.setdefault(str(node), status)
            if old != status:
                raise PolicyError(f"mapping status collision for {node}")
    children: dict[str, list[str]] = defaultdict(list)
    for rel in api["payload"]["relations"]:
        if rel.get("type") == "contains" and ":element:" in str(rel.get("child", "")):
            children[str(rel["parent"])].append(str(rel["child"]))
    return {
        parent
        for parent, nodes in children.items()
        if nodes and all(mapped.get(node) in ALLOWED_MAPPING for node in nodes)
    }


def _taxonomy() -> tuple[set[str], set[str]]:
    runtime = set(json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))["class_names"])
    raw = json.loads(RUN50_LABELS.read_text(encoding="utf-8"))
    labels = raw.get("labels", raw) if isinstance(raw, dict) else raw
    legacy = {str(name) for name in labels}
    return runtime, runtime & legacy


def _candidate_glyphs() -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    sources = verify_sources()
    api, join, zips, mapping = map(read_json, (API, JOIN, ZIP, MAPPING))
    p4 = _load_p4_module()
    strict_nodes = _strict_glyph_nodes(api, mapping)
    candidates = p4._candidate_entries(
        api_catalog=api, join_catalog=join, zip_catalog=zips, mapping=mapping
    )
    glyphs: list[dict[str, Any]] = []
    for row in candidates:
        if row.get("roles") != ["glyph_occurrence"]:
            continue
        if not row.get("node_keys") or not all(node in strict_nodes for node in row["node_keys"]):
            continue
        if not row.get("candidate_runtime_classes"):
            continue
        text = json.dumps(row, ensure_ascii=False).lower()
        if any(token in text for token in FORBIDDEN):
            raise PolicyError("forbidden token reached expanded candidate inventory")
        glyphs.append(row)
    glyphs.sort(key=lambda r: (r["source_group"], r["archive_name"], r["entry_path"]))
    runtime, shared = _taxonomy()
    if {name for row in glyphs for name in row["candidate_runtime_classes"]} != runtime:
        raise PolicyError("strict glyph inventory does not cover the runtime taxonomy")
    return glyphs, sources, shared


def select_entries(
    glyphs: list[dict[str, Any]], shared: set[str], *, target: int = TARGET_FILES
) -> list[dict[str, Any]]:
    if target != TARGET_FILES:
        raise PolicyError("expanded benchmark size is frozen at 2000")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in glyphs:
        by_group[row["source_group"]].append(row)
    ranked_groups = sorted(
        by_group,
        key=lambda g: (
            -len({c for r in by_group[g] for c in r["candidate_runtime_classes"]} & shared),
            -len({c for r in by_group[g] for c in r["candidate_runtime_classes"]}),
            stable(SPLIT_NAMESPACE, "group", g),
        ),
    )
    chosen: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    group_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    def add(row: dict[str, Any]) -> bool:
        key = (row["archive_name"], row["entry_path"])
        group = row["source_group"]
        if (
            key in used
            or group_counts[group] >= MAX_PER_GROUP
            or len(chosen) >= target
            or (group not in group_counts and len(group_counts) >= LIMITS["max_groups"])
        ):
            return False
        chosen.append(row)
        used.add(key)
        group_counts[group] += 1
        class_counts.update(row["candidate_runtime_classes"])
        return True

    # Preserve the frozen P4 dev planches before general page diversity.
    seeded_groups = [g for g in sorted(P4_FROZEN_DEV_GROUPS) if g in by_group]
    for group in seeded_groups:
        rows = sorted(
            by_group[group],
            key=lambda r: (
                -len(set(r["candidate_runtime_classes"]) & shared),
                stable(SPLIT_NAMESPACE, "p4-seed", r["archive_name"], r["entry_path"]),
            ),
        )
        add(rows[0])

    # Establish the remaining page diversity.
    for group in ranked_groups:
        if len(group_counts) >= MIN_GROUPS:
            break
        if group in group_counts:
            continue
        rows = sorted(
            by_group[group],
            key=lambda r: (
                -len(set(r["candidate_runtime_classes"]) & shared),
                stable(SPLIT_NAMESPACE, r["archive_name"], r["entry_path"]),
            ),
        )
        add(rows[0])

    # Guarantee all 286 classes before density filling.
    runtime, _ = _taxonomy()
    while set(class_counts) != runtime:
        missing = runtime - set(class_counts)
        options = [
            r for r in glyphs
            if set(r["candidate_runtime_classes"]) & missing
            and (r["archive_name"], r["entry_path"]) not in used
            and group_counts[r["source_group"]] < MAX_PER_GROUP
        ]
        if not options:
            raise PolicyError(f"cannot cover runtime classes: {sorted(missing)}")
        options.sort(key=lambda r: (
            -len(set(r["candidate_runtime_classes"]) & missing),
            -len(set(r["candidate_runtime_classes"]) & shared),
            class_counts[min(r["candidate_runtime_classes"], key=lambda c: class_counts[c])],
            stable(SPLIT_NAMESPACE, "cover", r["archive_name"], r["entry_path"]),
        ))
        add(options[0])

    # Fill in one deterministic inverse-global-support pass.  Re-sorting the
    # complete 26k+ strict inventory after every addition would be quadratic.
    inventory_support: Counter[str] = Counter(
        c for row in glyphs for c in row["candidate_runtime_classes"]
    )
    remaining = sorted(
        (r for r in glyphs if (r["archive_name"], r["entry_path"]) not in used),
        key=lambda r: (
            min(inventory_support[c] for c in set(r["candidate_runtime_classes"]) & shared)
            if set(r["candidate_runtime_classes"]) & shared else 10**9,
            min(inventory_support[c] for c in r["candidate_runtime_classes"]),
            stable(SPLIT_NAMESPACE, "fill", r["archive_name"], r["entry_path"]),
        ),
    )
    for row in remaining:
        add(row)
        if len(chosen) == target:
            break
    if len(chosen) != target:
        raise PolicyError("not enough bounded glyphs to fill expanded benchmark")
    return sorted(chosen, key=lambda r: (r["source_group"], r["archive_name"], r["entry_path"]))


def _dev_groups(entries: list[dict[str, Any]]) -> set[str]:
    groups = sorted({row["source_group"] for row in entries})
    pinned = P4_FROZEN_DEV_GROUPS & set(groups)
    ranked = sorted(set(groups) - pinned, key=lambda g: stable(SPLIT_NAMESPACE, "dev", g))
    selected = set(sorted(pinned, key=lambda g: stable(SPLIT_NAMESPACE, "p4", g))[:DEV_GROUPS])
    selected.update(ranked[: DEV_GROUPS - len(selected)])
    if len(selected) != DEV_GROUPS:
        raise PolicyError("cannot allocate 50 whole-planche dev groups")
    return selected


def build_plan() -> dict[str, Any]:
    glyphs, sources, shared = _candidate_glyphs()
    entries = select_entries(glyphs, shared)
    dev = _dev_groups(entries)
    for row in entries:
        row["benchmark_split"] = "dev" if row["source_group"] in dev else "train"
    counts = Counter()
    for row in entries:
        counts["files"] += 1
        counts["compressed_bytes"] += int(row["entry"]["compressed_size"])
        counts["uncompressed_bytes"] += int(row["entry"]["uncompressed_size"])
        counts["planned_requests"] += 2
    groups = {row["source_group"] for row in entries}
    classes = {c for row in entries for c in row["candidate_runtime_classes"]}
    if len(entries) != TARGET_FILES or len(groups) < MIN_GROUPS or len(classes) != 286:
        raise PolicyError("expanded selection gates failed")
    if len(groups) > LIMITS["max_groups"]:
        raise PolicyError("expanded selection exceeds source-group budget")
    budget_keys = {
        "files": "max_files",
        "compressed_bytes": "max_compressed_bytes",
        "uncompressed_bytes": "max_uncompressed_bytes",
        "planned_requests": "max_requests",
    }
    for key, limit_key in budget_keys.items():
        if counts[key] > LIMITS[limit_key]:
            raise PolicyError(f"expanded plan exceeds {key} budget")
    counts.update({
        "source_groups": len(groups),
        "train_groups": len(groups - dev),
        "dev_groups": len(dev),
        "runtime_classes_evidenced": len(classes),
        "shared_runtime_classes": len(shared),
        "strict_inventory_glyphs": len(glyphs),
    })
    return artifact(MEDIA_PLAN_SCHEMA, {
        "purpose": "multiview_self_supervised",
        "research_subpurpose": "legacy_fusion_expanded_glyph_benchmark",
        "source_file_sha256": sources,
        "policy": {
            "selection": "page diversity, full taxonomy coverage, inverse-frequency shared-class fill",
            "target_files": TARGET_FILES,
            "minimum_groups": MIN_GROUPS,
            "max_per_group": MAX_PER_GROUP,
            "dev_groups": DEV_GROUPS,
            "split_namespace": SPLIT_NAMESPACE,
            "strict_all_descendants_approved": True,
            "whole_zip_fallback": False,
            "promotion_allowed": False,
            "business_or_locked_input_allowed": False,
        },
        "limits": LIMITS,
        "counts": dict(counts),
        "entries": entries,
    })


def validate_expanded_plan(plan: dict[str, Any]) -> None:
    """Rebuild the frozen plan and require exact semantic identity."""
    verify_artifact(plan, MEDIA_PLAN_SCHEMA)
    payload = plan["payload"]
    if payload.get("research_subpurpose") != "legacy_fusion_expanded_glyph_benchmark":
        raise PolicyError("not an expanded legacy-fusion benchmark plan")
    if payload.get("policy", {}).get("whole_zip_fallback") is not False:
        raise PolicyError("whole ZIP fallback is forbidden")
    if payload.get("policy", {}).get("promotion_allowed") is not False:
        raise PolicyError("expanded research plan cannot be promotable")
    expected = build_plan()
    if plan.get("payload_sha256") != expected.get("payload_sha256"):
        raise PolicyError("provided plan differs from the deterministic frozen plan")


def verify_download(plan: dict[str, Any], manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    verify_artifact(plan, MEDIA_PLAN_SCHEMA)
    verify_artifact(manifest, MEDIA_MANIFEST_SCHEMA)
    if manifest["payload"].get("plan_payload_sha256") != plan["payload_sha256"]:
        raise PolicyError("download manifest does not bind expanded plan")
    files = manifest["payload"].get("files", [])
    if len(files) != TARGET_FILES:
        raise PolicyError("expanded download is incomplete")
    if Path(manifest["payload"]["output_root"]).resolve() != root.resolve():
        raise PolicyError("expanded output root drift")
    return artifact("legacy-fusion-expanded-verification.v1", {
        "plan_payload_sha256": plan["payload_sha256"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "files": len(files),
        "download_complete": True,
        "promotion_eligible": False,
    })


def load_expanded_rows(
    plan: dict[str, Any], manifest: dict[str, Any], root: Path, taxonomy: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind the bounded download to strict glyph supervision before scoring."""
    validate_expanded_plan(plan)
    verify_download(plan, manifest, root)
    entries = plan["payload"].get("entries")
    files = manifest["payload"].get("files")
    if not isinstance(entries, list) or not isinstance(files, list):
        raise PolicyError("expanded plan/manifest entries are missing")
    files_by_key = {(row.get("archive_name"), row.get("entry_path")): row for row in files}
    if len(files_by_key) != len(entries):
        raise PolicyError("expanded manifest physical keys are not unique")
    runtime_index = {str(name): index for index, name in enumerate(taxonomy.runtime_names)}
    fusion = _load_fusion_module()
    rows: list[dict[str, Any]] = []
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        key = (entry.get("archive_name"), entry.get("entry_path"))
        downloaded = files_by_key.pop(key, None)
        if downloaded is None:
            raise PolicyError("expanded plan/download physical keys differ")
        group = str(entry.get("source_group", ""))
        split = str(entry.get("benchmark_split", ""))
        if not group or split not in {"train", "dev"}:
            raise PolicyError("expanded glyph lacks its frozen whole-planche split")
        for field in (
            "physical_source_group", "candidate_runtime_classes", "node_keys",
            "mapping_ids", "mapping_statuses",
        ):
            if downloaded.get(field) != entry.get(field):
                raise PolicyError(f"expanded download drifted {field}")
        if entry.get("physical_source_group") != group or entry.get("split_group") != group:
            raise PolicyError("expanded glyph group binding drifted")
        names = [str(name) for name in entry.get("candidate_runtime_classes", ())]
        statuses = {str(status) for status in entry.get("mapping_statuses", ())}
        if (
            entry.get("roles") != ["glyph_occurrence"]
            or entry.get("multilabel_evidence_kind") != "explicit_descendant_multilabel_evidence"
            or not names
            or not statuses <= ALLOWED_MAPPING
            or not entry.get("mapping_ids")
            or set(names) - set(runtime_index)
        ):
            raise PolicyError("expanded glyph relation is not strictly evaluable")
        path = (root / str(downloaded.get("output_path", ""))).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise PolicyError("expanded glyph path escapes output root") from error
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise PolicyError("expanded direct B1 input is not a JPEG glyph")
        if not path.is_file() or sha256_file(path) != downloaded.get("byte_sha256"):
            raise PolicyError("expanded glyph byte identity differs from manifest")
        pixel_sha = fusion._pixel_identity(path)
        if pixel_sha != downloaded.get("image", {}).get("pixel_sha256"):
            raise PolicyError("expanded glyph pixel identity differs from manifest")
        pixel_splits[pixel_sha].add(split)
        rows.append({
            "row_id": "expanded-glyph-" + stable(*key, pixel_sha)[:24],
            "image_path": str(path), "pixel_sha256": pixel_sha,
            "dhash64": fusion._dhash64(path), "page_group": group, "split": split,
            "truth_labels": sorted(runtime_index[name] for name in set(names)),
            "candidate_runtime_classes": names, "strict_metric_eligible": True,
        })
    if files_by_key:
        raise PolicyError("expanded manifest contains unplanned files")
    if any(len(splits) > 1 for splits in pixel_splits.values()):
        raise PolicyError("pixel-identical glyph crosses the frozen page split")
    seen_pixels, seen_dhashes, run50_inventory = fusion.load_run50_seen_pixels()
    kept, leakage = fusion.exclude_leakage(rows, seen_pixels, seen_dhashes)
    if not kept or {row["split"] for row in kept} != {"train", "dev"}:
        raise PolicyError("run_50 overlap exclusion left no valid train/dev benchmark")
    planned_groups = {
        split: {row["page_group"] for row in rows if row["split"] == split}
        for split in ("train", "dev")
    }
    retained_groups = {
        split: {row["page_group"] for row in kept if row["split"] == split}
        for split in ("train", "dev")
    }
    if len(retained_groups["dev"]) > DEV_GROUPS:
        raise PolicyError("run_50 overlap audit produced too many dev groups")
    replacements = sorted(
        retained_groups["train"], key=lambda group: stable("expanded-clean-dev-v1", group)
    )[: DEV_GROUPS - len(retained_groups["dev"])]
    if len(retained_groups["dev"]) + len(replacements) != DEV_GROUPS:
        raise PolicyError("run_50 overlap exclusion cannot restore the frozen dev group count")
    replacement_set = set(replacements)
    for row in kept:
        if row["page_group"] in replacement_set:
            row["split"] = "dev"
    final_groups = {
        split: {row["page_group"] for row in kept if row["split"] == split}
        for split in ("train", "dev")
    }
    if len(final_groups["dev"]) != DEV_GROUPS or not final_groups["train"]:
        raise PolicyError("clean page-disjoint split is incomplete")
    retained_binding = [
        {
            "row_id": row["row_id"],
            "pixel_sha256": row["pixel_sha256"],
            "page_group": row["page_group"],
            "split": row["split"],
        }
        for row in sorted(kept, key=lambda item: item["row_id"])
    ]
    return kept, {
        "download_rows": len(rows),
        "strict_rows_after_run50_exclusion": len(kept),
        "run50_training_inventory": run50_inventory,
        "leakage": leakage,
        "clean_split_rule": "preserve surviving dev groups, then stable-hash train replacements",
        "removed_frozen_groups": {
            split: sorted(planned_groups[split] - retained_groups[split])
            for split in ("train", "dev")
        },
        "replacement_dev_groups": replacements,
        "final_groups": {split: len(groups) for split, groups in final_groups.items()},
        "retained_rows_binding_sha256": fusion.semantic_hash(retained_binding),
        "retained_split_rows": dict(Counter(row["split"] for row in kept)),
        "retained_unique_pixels": len({row["pixel_sha256"] for row in kept}),
        "promotion_eligible": False,
    }


def rerank_shared_in_place(
    baseline: np.ndarray, candidate: np.ndarray, shared_indices: tuple[int, ...]
) -> np.ndarray:
    """Reorder shared labels without moving any B1-only rank position."""
    if baseline.shape != candidate.shape or baseline.ndim != 2:
        raise ValueError("shared reranker score shapes differ")
    shared = np.asarray(shared_indices, dtype=np.int64)
    result = np.array(baseline, copy=True)
    slots = np.sort(baseline[:, shared], axis=1)[:, ::-1]
    order = np.argsort(-candidate[:, shared], axis=1, kind="stable")
    for row in range(len(result)):
        result[row, shared[order[row]]] = slots[row]
    return result


def evaluate_expanded(
    plan: dict[str, Any], manifest: dict[str, Any], root: Path, *, device: str,
    batch_size: int, apply: bool,
) -> dict[str, Any]:
    """One bounded, direct-B1 evaluation; no training or promotion side effects."""
    fusion = _load_fusion_module()
    model_input_hashes = fusion.verify_static_hashes(
        fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1
    )
    taxonomy = fusion.load_taxonomy_from_pins()
    rows, audit = load_expanded_rows(plan, manifest, root, taxonomy)
    dry_run = {
        "action": "score direct whole-glyph B1 and run_50 on expanded benchmark",
        "rows": len(rows), "device": device, "run50_batch_size": batch_size,
        "scoring_profile": fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1,
        "model_input_hashes": model_input_hashes,
        "audit": audit, "promotion_eligible": False,
    }
    if not apply:
        return {**dry_run, "status": "DRY_RUN"}
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    model = fusion._load_run50_model(device, taxonomy.legacy_names)
    legacy_logits, legacy_latency = fusion.score_run50(rows, model, device, batch_size)
    classifier = fusion._load_b1_classifier(device)
    if tuple(classifier.class_names) != taxonomy.runtime_names:
        raise ValueError("loaded B1 class order differs from pinned taxonomy")
    b1_scores, b1_timing = fusion.score_direct_b1_whole_glyph(rows, classifier)
    variants, calibration = fusion.build_train_selected_variants(
        b1_scores, legacy_logits, [row["split"] for row in rows],
        [row["truth_labels"] for row in rows], taxonomy,
    )
    variants["prior_protected"] = rerank_shared_in_place(
        variants["b1"], variants["prior_global"], taxonomy.shared_runtime_indices
    )
    variants["cascade_protected"] = rerank_shared_in_place(
        variants["b1"], variants["cascade"], taxonomy.shared_runtime_indices
    )
    dev = [index for index, row in enumerate(rows) if row["split"] == "dev"]
    truths = [rows[index]["truth_labels"] for index in dev]
    groups = [rows[index]["page_group"] for index in dev]
    dev_variants = {name: scores[dev] for name, scores in variants.items()}
    oracle, oracle_sources = fusion.build_oracle_scores(
        {name: dev_variants[name] for name in ("b1", "legacy", "prior_global", "cascade")},
        truths,
    )
    all_variants = {**dev_variants, "oracle": oracle}
    metrics = {name: fusion.evaluate_scores(scores, truths, groups) for name, scores in all_variants.items()}
    for name, scores in all_variants.items():
        metrics[name]["b1_only_ranking"] = fusion.evaluate_b1_only_ranking(
            scores, truths, taxonomy.b1_only_indices
        )
    latency = float(b1_timing["b1_ms_per_glyph"]) + legacy_latency
    gates: dict[str, Any] = {}
    for name in ("prior_global", "cascade", "prior_protected", "cascade_protected"):
        bootstrap = fusion.paired_planche_bootstrap(
            {group: value["sample_ap"] for group, value in metrics[name]["per_planche"].items()},
            {group: value["sample_ap"] for group, value in metrics["b1"]["per_planche"].items()},
        )
        gate_name = {
            "prior_protected": "prior_global",
            "cascade_protected": "cascade",
        }.get(name, name)
        gate = fusion.final_gate(
            gate_name, metrics[name], metrics["b1"], bootstrap,
            latency_ms_per_glyph=latency,
            candidate_b1_only=metrics[name]["b1_only_ranking"],
            baseline_b1_only=metrics["b1"]["b1_only_ranking"],
            b1_only_logits_identical=fusion.np.array_equal(
                dev_variants[name][:, taxonomy.b1_only_indices],
                dev_variants["b1"][:, taxonomy.b1_only_indices],
            ),
        )
        gate["candidate"] = name
        gates[name] = gate
    return {
        **dry_run, "status": "EVALUATED", "dev_glyphs": len(dev),
        "dev_planches": len(set(groups)), "train_only_calibration_and_selection": calibration,
        "metrics": metrics, "oracle_source_counts": dict(Counter(oracle_sources)),
        "calibration_interpretation": (
            "descriptive sigmoid transform of cosine/sigmoid scores; not calibrated probability"
        ),
        "latency": {**b1_timing, "legacy_ms_per_glyph": legacy_latency,
                    "serial_fusion_ms_per_glyph": latency},
        "gates": gates,
        "light_model_iteration_gate_passed": any(
            gate["sanity_pass"] for gate in gates.values()
        ),
        "automatic_global_training_allowed": False,
        "promotion_eligible": False,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--output", type=Path)
    d = sub.add_parser("download")
    d.add_argument("--plan", type=Path, required=True)
    d.add_argument("--capture-dir", type=Path, required=True)
    d.add_argument("--output-root", type=Path, required=True)
    d.add_argument("--manifest-output", type=Path, required=True)
    d.add_argument("--apply", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--output-root", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate", help="score direct B1 + run_50 on the bounded glyph benchmark")
    e.add_argument("--plan", type=Path, required=True)
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--output-root", type=Path, required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--apply", action="store_true")
    e.add_argument("--output", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        result = build_plan()
        if args.output:
            write_json(args.output, result, write_once=True)
    elif args.command == "download":
        if not args.apply:
            raise PolicyError("download requires explicit --apply")
        plan = read_json(args.plan)
        validate_expanded_plan(plan)
        transport = CachedTransport(
            args.capture_dir, max_requests=int(plan["payload"]["limits"]["max_requests"])
        )
        result = download_media_plan(plan, transport=transport, output_root=args.output_root)
        write_json(args.manifest_output, result, write_once=True)
    elif args.command == "verify":
        result = verify_download(read_json(args.plan), read_json(args.manifest), args.output_root)
        write_json(args.output, result, write_once=True)
    else:
        result = evaluate_expanded(
            read_json(args.plan), read_json(args.manifest), args.output_root,
            device=args.device, batch_size=args.batch_size, apply=args.apply,
        )
        if args.output:
            fusion = _load_fusion_module()
            fusion.write_once_json(args.output, fusion._jsonable_metrics(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

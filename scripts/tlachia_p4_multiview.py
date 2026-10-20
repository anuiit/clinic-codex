#!/usr/bin/env python3
"""Build and verify a bounded Tlachia multi-view SSL acquisition pilot.

The media remain page/glyph/zone observations.  Runtime-class descendants are
provenance evidence only; they are deliberately never converted into crop
single-label supervision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.data.tlachia.catalog import CachedTransport  # noqa: E402
from codex_pipeline.data.tlachia.core import (  # noqa: E402
    IntegrityError,
    PolicyError,
    TlachiaError,
    artifact,
    read_json,
    semantic_hash,
    sha256_file,
    verify_artifact,
    write_json,
)
from codex_pipeline.data.tlachia.media import (  # noqa: E402
    MEDIA_MANIFEST_SCHEMA,
    MEDIA_PLAN_SCHEMA,
    download_media_plan,
)
from codex_pipeline.data.tlachia.p3 import (  # noqa: E402
    MAPPING_SCHEMA,
    TRAINABLE_MAPPING_STATUSES,
)


DEFAULT_API = REPO_ROOT / "reports/tlachia-p3-20260813/catalog/api-global-elements-v1.json"
DEFAULT_JOIN = REPO_ROOT / "reports/tlachia-p3-20260813/catalog/join-global-elements-v1.json"
DEFAULT_ZIP = REPO_ROOT / "reports/tlachia-p0-p2-20260813/catalog/zip-all-v2.json"
DEFAULT_MAPPING = (
    REPO_ROOT
    / "reports/tlachia-p3-20260813/mapping/global-runtime-business-approved-v2.json"
)
ALLOWED_ROLES = ("glyph_occurrence", "zone", "planche")
NODE_KIND = {"glyph_occurrence": "glyph", "zone": "zone", "planche": "planche"}
VERIFY_SCHEMA = "tlachia-multiview-verification.v1"


def _positive(value: int, label: str) -> int:
    if value <= 0:
        raise PolicyError(f"{label} must be positive")
    return value


def _stable(value: object) -> str:
    return semantic_hash({"selection_key": value})


def _indexes(
    api_catalog: dict[str, Any],
    join_catalog: dict[str, Any],
    zip_catalog: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    verify_artifact(api_catalog, "tlachia-api-catalog.v1")
    verify_artifact(join_catalog, "tlachia-api-zip-join.v1")
    verify_artifact(zip_catalog, "tlachia-zip-catalog.v1")
    verify_artifact(mapping, MAPPING_SCHEMA)
    if join_catalog["payload"]["api_payload_sha256"] != api_catalog["payload_sha256"]:
        raise IntegrityError("join catalog does not bind the API catalog")
    if join_catalog["payload"]["zip_payload_sha256"] != zip_catalog["payload_sha256"]:
        raise IntegrityError("join catalog does not bind the ZIP catalog")
    if mapping["payload"]["api_catalog_payload_sha256"] != api_catalog["payload_sha256"]:
        raise IntegrityError("mapping does not bind the API catalog")

    nodes: dict[str, dict[str, Any]] = {}
    for kind, rows in api_catalog["payload"].get("nodes", {}).items():
        for row in rows:
            key = row["key"]
            if key in nodes:
                raise IntegrityError(f"duplicate API node: {key}")
            nodes[key] = {**row, "_kind": kind}
    children: dict[str, list[str]] = defaultdict(list)
    for relation in api_catalog["payload"].get("relations", []):
        if relation.get("type") != "contains":
            continue
        parent, child = relation["parent"], relation["child"]
        if parent not in nodes or child not in nodes:
            raise IntegrityError("API containment relation references an unknown node")
        children[parent].append(child)
    for rows in children.values():
        rows.sort()

    archives: dict[str, dict[str, Any]] = {}
    for archive in zip_catalog["payload"].get("archives", []):
        name = archive["archive_name"]
        if name in archives:
            raise IntegrityError(f"duplicate ZIP archive: {name}")
        entry_index: dict[str, dict[str, Any]] = {}
        for entry in archive.get("entries", []):
            path = entry["path"]
            if path in entry_index:
                raise IntegrityError(f"duplicate ZIP entry: {name}:{path}")
            entry_index[path] = entry
        archives[name] = {**archive, "_entries": entry_index}
    return nodes, children, archives


def _descendant_mapping_evidence(
    *,
    nodes: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
    mapping: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    by_element: dict[str, dict[str, Any]] = {}
    for row in mapping["payload"].get("mappings", []):
        if row.get("status") not in TRAINABLE_MAPPING_STATUSES:
            continue
        runtime_class = row.get("candidate_runtime_class_name")
        if not runtime_class:
            continue
        for node_key in row.get("node_keys", []):
            if node_key in by_element:
                raise IntegrityError(f"accepted element belongs to multiple mappings: {node_key}")
            by_element[node_key] = row

    memo: dict[str, dict[str, set[str]]] = {}
    visiting: set[str] = set()

    def visit(node_key: str) -> dict[str, set[str]]:
        if node_key in memo:
            return memo[node_key]
        if node_key in visiting:
            raise IntegrityError("cycle in API containment graph")
        visiting.add(node_key)
        classes: set[str] = set()
        mapping_ids: set[str] = set()
        mapping_statuses: set[str] = set()
        row = by_element.get(node_key)
        if row is not None:
            classes.add(str(row["candidate_runtime_class_name"]))
            mapping_ids.add(str(row["mapping_id"]))
            mapping_statuses.add(str(row["status"]))
        for child in children.get(node_key, []):
            child_evidence = visit(child)
            classes.update(child_evidence["classes"])
            mapping_ids.update(child_evidence["mapping_ids"])
            mapping_statuses.update(child_evidence["mapping_statuses"])
        visiting.remove(node_key)
        memo[node_key] = {
            "classes": classes,
            "mapping_ids": mapping_ids,
            "mapping_statuses": mapping_statuses,
        }
        return memo[node_key]

    result: dict[str, dict[str, list[str]]] = {}
    for node_key in nodes:
        evidence = visit(node_key)
        result[node_key] = {key: sorted(values) for key, values in evidence.items()}
    return result


def _candidate_entries(
    *,
    api_catalog: dict[str, Any],
    join_catalog: dict[str, Any],
    zip_catalog: dict[str, Any],
    mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes, children, archives = _indexes(api_catalog, join_catalog, zip_catalog, mapping)
    evidence = _descendant_mapping_evidence(
        nodes=nodes, children=children, mapping=mapping
    )
    physical: dict[tuple[str, str], dict[str, Any]] = {}
    for relation in join_catalog["payload"].get("relations", []):
        role = relation.get("role")
        if role not in ALLOWED_ROLES:
            continue
        node_key = relation["node"]
        node = nodes.get(node_key)
        if node is None or node["_kind"] != NODE_KIND[role]:
            raise IntegrityError(f"{role} join does not bind an exact API node: {node_key}")
        archive_name, entry_path = relation["archive"], relation["entry_path"]
        archive = archives.get(archive_name)
        if archive is None or archive.get("status") != "available":
            raise IntegrityError(f"joined archive is unavailable: {archive_name}")
        entry = archive["_entries"].get(entry_path)
        if entry is None:
            raise IntegrityError(f"joined ZIP entry is absent: {archive_name}:{entry_path}")
        if entry["crc32"] != relation["entry_crc32"]:
            raise IntegrityError("join CRC differs from the exact ZIP central-directory entry")
        source_group = node.get("source_group")
        expected_group = (
            f"tlachia:codex:{node['codex_id']}:planche:{node.get('planche_cote') or node.get('cote')}"
        )
        if source_group != expected_group:
            raise IntegrityError(f"non-canonical codex+planche source_group: {node_key}")
        key = (archive_name, entry_path)
        item = physical.setdefault(
            key,
            {
                "archive_name": archive_name,
                "archive_url": archive["url"],
                "archive_etag": archive.get("etag"),
                "archive_content_length": archive["content_length"],
                "archive_central_directory_sha256": archive["central_directory_sha256"],
                "entry_path": entry_path,
                "entry": entry,
                "roles": [],
                "node_keys": [],
                "source_groups": [],
                "candidate_runtime_classes": [],
                "mapping_ids": [],
                "mapping_statuses": [],
                "codex_ids": [],
            },
        )
        item["roles"].append(role)
        item["node_keys"].append(node_key)
        item["source_groups"].append(source_group)
        item["codex_ids"].append(str(node["codex_id"]))
        item["candidate_runtime_classes"].extend(evidence[node_key]["classes"])
        item["mapping_ids"].extend(evidence[node_key]["mapping_ids"])
        item["mapping_statuses"].extend(evidence[node_key]["mapping_statuses"])

    result: list[dict[str, Any]] = []
    for item in physical.values():
        for key in (
            "roles",
            "node_keys",
            "source_groups",
            "candidate_runtime_classes",
            "mapping_ids",
            "mapping_statuses",
            "codex_ids",
        ):
            item[key] = sorted(set(item[key]))
        # A single ZIP member can be referenced by several logical occurrences.
        # It is downloaded once and deterministically owned by one canonical
        # planche group; all occurrence groups remain recorded as evidence.
        occurrence_groups = item.pop("source_groups")
        source_group = min(occurrence_groups, key=_stable)
        item.update(
            {
                "source_group": source_group,
                "split_group": source_group,
                "physical_source_group": source_group,
                "occurrence_source_groups": occurrence_groups,
                "supervision_kind": "unlabeled_ssl",
                "multilabel_evidence_kind": (
                    "explicit_descendant_multilabel_evidence"
                    if "glyph_occurrence" in item["roles"]
                    and item["candidate_runtime_classes"]
                    else None
                ),
                "single_label_class": None,
            }
        )
        result.append(item)
    return sorted(result, key=lambda row: (row["archive_name"], row["entry_path"]))


def _fits(
    counts: Counter[str], item: dict[str, Any], limits: dict[str, int]
) -> bool:
    return (
        counts["files"] + 1 <= limits["max_files"]
        and counts["compressed_bytes"] + int(item["entry"]["compressed_size"])
        <= limits["max_compressed_bytes"]
        and counts["uncompressed_bytes"] + int(item["entry"]["uncompressed_size"])
        <= limits["max_uncompressed_bytes"]
        and counts["planned_requests"] + 2 <= limits["max_requests"]
    )


def build_multiview_plan(
    *,
    api_catalog: dict[str, Any],
    join_catalog: dict[str, Any],
    zip_catalog: dict[str, Any],
    mapping: dict[str, Any],
    max_files: int,
    max_groups: int,
    max_compressed_bytes: int,
    max_uncompressed_bytes: int,
    max_requests: int,
    max_pixel_count: int,
    max_glyphs_per_group: int = 8,
    max_zones_per_group: int = 2,
    max_planches_per_group: int = 1,
    min_source_groups: int = 20,
    min_runtime_classes: int = 100,
) -> dict[str, Any]:
    limits = {
        "max_files": _positive(max_files, "max-files"),
        "max_groups": _positive(max_groups, "max-groups"),
        "max_compressed_bytes": _positive(max_compressed_bytes, "max-compressed-bytes"),
        "max_uncompressed_bytes": _positive(max_uncompressed_bytes, "max-uncompressed-bytes"),
        "max_requests": _positive(max_requests, "max-requests"),
        "max_pixel_count": _positive(max_pixel_count, "max-pixel-count"),
    }
    role_caps = {
        "glyph_occurrence": _positive(max_glyphs_per_group, "max-glyphs-per-group"),
        "zone": _positive(max_zones_per_group, "max-zones-per-group"),
        "planche": _positive(max_planches_per_group, "max-planches-per-group"),
    }
    min_source_groups = _positive(min_source_groups, "min-source-groups")
    min_runtime_classes = _positive(min_runtime_classes, "min-runtime-classes")
    if limits["max_groups"] < min_source_groups:
        raise PolicyError("max-groups is below the minimum source-group gate")

    candidates = _candidate_entries(
        api_catalog=api_catalog,
        join_catalog=join_catalog,
        zip_catalog=zip_catalog,
        mapping=mapping,
    )
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_group[item["source_group"]].append(item)
    available_classes = sorted(
        {name for item in candidates for name in item["candidate_runtime_classes"]}
    )
    required_classes = min_runtime_classes if len(available_classes) >= min_runtime_classes else 0

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    covered: set[str] = set()
    totals: Counter[str] = Counter()
    remaining = set(by_group)
    while remaining and len(selected_groups) < limits["max_groups"]:
        ranked_groups = sorted(
            remaining,
            key=lambda group: (
                -len(
                    {
                        name
                        for item in by_group[group]
                        for name in item["candidate_runtime_classes"]
                        if name not in covered
                    }
                ),
                _stable(group),
            ),
        )
        added_group = False
        for group in ranked_groups:
            role_counts: Counter[str] = Counter()
            group_added: list[dict[str, Any]] = []
            ranked_items = sorted(
                by_group[group],
                key=lambda item: (
                    -len(set(item["candidate_runtime_classes"]) - covered),
                    _stable([item["archive_name"], item["entry_path"]]),
                ),
            )
            for item in ranked_items:
                if any(role_counts[role] >= role_caps[role] for role in item["roles"]):
                    continue
                if not _fits(totals, item, limits):
                    continue
                group_added.append(item)
                for role in item["roles"]:
                    role_counts[role] += 1
                totals["files"] += 1
                totals["compressed_bytes"] += int(item["entry"]["compressed_size"])
                totals["uncompressed_bytes"] += int(item["entry"]["uncompressed_size"])
                totals["planned_requests"] += 2
            remaining.remove(group)
            if group_added:
                selected.extend(group_added)
                selected_groups.add(group)
                for item in group_added:
                    covered.update(item["candidate_runtime_classes"])
                added_group = True
                break
        if not added_group:
            break

    if len(selected_groups) < min_source_groups:
        raise PolicyError(
            f"pilot selected {len(selected_groups)} source groups; {min_source_groups} required"
        )
    if required_classes and len(covered) < required_classes:
        raise PolicyError(
            f"pilot covers {len(covered)} runtime classes; {required_classes} required"
        )
    if not selected:
        raise PolicyError("multi-view plan selected no entries")
    selected.sort(key=lambda item: (item["source_group"], item["archive_name"], item["entry_path"]))
    role_counts = Counter(role for item in selected for role in item["roles"])
    counts = {
        **dict(totals),
        "source_groups": len(selected_groups),
        "runtime_classes_evidenced": len(covered),
        "available_runtime_classes_evidenced": len(available_classes),
        "occurrences": sum(len(item["node_keys"]) for item in selected),
        "roles": {role: role_counts[role] for role in ALLOWED_ROLES},
    }
    return artifact(
        MEDIA_PLAN_SCHEMA,
        {
            "purpose": "multiview_self_supervised",
            "api_payload_sha256": api_catalog["payload_sha256"],
            "join_payload_sha256": join_catalog["payload_sha256"],
            "zip_payload_sha256": zip_catalog["payload_sha256"],
            "mapping_payload_sha256": mapping["payload_sha256"],
            "policy": {
                "whole_zip_fallback": False,
                "supported_compression": [0, 8],
                "require_etag": True,
                "require_http_206": True,
                "physical_unit": "exact archive and ZIP entry",
                "selection": "group greedy max-new-runtime-class evidence then stable hash",
                "single_label_supervision": False,
                "runtime_mapping_use": "descendant evidence or auxiliary multilabel only",
            },
            "limits": limits,
            "role_caps_per_group": role_caps,
            "gates": {
                "min_source_groups": min_source_groups,
                "min_runtime_classes_if_available": min_runtime_classes,
                "runtime_class_gate_available": bool(required_classes),
                "promotion_allowed": False,
            },
            "counts": counts,
            "entries": selected,
        },
    )


def _pixel_identity(path: Path) -> tuple[str, int]:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest(), rgb.width * rgb.height


def verify_multiview_download(
    plan: dict[str, Any], manifest: dict[str, Any], *, output_root: Path
) -> dict[str, Any]:
    verify_artifact(plan, MEDIA_PLAN_SCHEMA)
    verify_artifact(manifest, MEDIA_MANIFEST_SCHEMA)
    pp, mp = plan["payload"], manifest["payload"]
    if pp.get("purpose") != "multiview_self_supervised":
        raise PolicyError("not a multi-view self-supervised plan")
    if mp.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise IntegrityError("manifest does not bind the reviewed plan")
    if mp.get("purpose") != pp["purpose"]:
        raise IntegrityError("manifest purpose drift")
    if Path(mp["output_root"]).resolve() != output_root.resolve():
        raise IntegrityError("manifest output root differs from the explicit verification root")
    if pp.get("policy", {}).get("whole_zip_fallback") is not False:
        raise PolicyError("whole-ZIP fallback must remain disabled")
    if pp.get("gates", {}).get("promotion_allowed") is not False:
        raise PolicyError("pilot artifacts cannot be promoted")

    plan_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in pp.get("entries", []):
        key = (item["archive_name"], item["entry_path"])
        if key in plan_by_key:
            raise IntegrityError("duplicate physical key in plan")
        if not set(item.get("roles", [])) <= set(ALLOWED_ROLES):
            raise PolicyError("plan contains a forbidden media role")
        if item.get("supervision_kind") != "unlabeled_ssl" or item.get("single_label_class") is not None:
            raise PolicyError("multi-view media must not carry single-label crop supervision")
        if item.get("source_group") != item.get("split_group"):
            raise IntegrityError("split_group must equal canonical source_group")
        plan_by_key[key] = item
    manifest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mp.get("files", []):
        key = (row["archive_name"], row["entry_path"])
        if key in manifest_by_key:
            raise IntegrityError("duplicate physical key in manifest")
        manifest_by_key[key] = row
    if set(plan_by_key) != set(manifest_by_key):
        raise IntegrityError("downloaded physical keys differ from the plan")

    total_pixels = 0
    pixels_by_hash: dict[str, set[str]] = defaultdict(set)
    for key, item in plan_by_key.items():
        row = manifest_by_key[key]
        if row.get("entry_crc32") != item["entry"]["crc32"]:
            raise IntegrityError("downloaded CRC evidence differs from plan")
        if row.get("compressed_size") != item["entry"]["compressed_size"] or row.get(
            "uncompressed_size"
        ) != item["entry"]["uncompressed_size"]:
            raise IntegrityError("downloaded ZIP sizes differ from plan")
        path = output_root / row["output_path"]
        if not path.is_file() or sha256_file(path) != row.get("byte_sha256"):
            raise IntegrityError("downloaded file SHA-256 verification failed")
        pixel_sha, pixels = _pixel_identity(path)
        if pixel_sha != row.get("image", {}).get("pixel_sha256"):
            raise IntegrityError("downloaded pixel SHA-256 verification failed")
        if pixels != row.get("image", {}).get("pixel_count"):
            raise IntegrityError("downloaded pixel count verification failed")
        total_pixels += pixels
        pixels_by_hash[pixel_sha].add(item["source_group"])
    cross_group = {
        digest: sorted(groups) for digest, groups in pixels_by_hash.items() if len(groups) > 1
    }
    if cross_group:
        raise PolicyError("pixel-identical media crosses planche source groups")

    counts = pp["counts"]
    limits = pp["limits"]
    for key, limit_key in (
        ("files", "max_files"),
        ("compressed_bytes", "max_compressed_bytes"),
        ("uncompressed_bytes", "max_uncompressed_bytes"),
        ("planned_requests", "max_requests"),
    ):
        if int(counts[key]) > int(limits[limit_key]):
            raise PolicyError(f"frozen {key} exceeds budget")
    if total_pixels > int(limits["max_pixel_count"]):
        raise PolicyError("verified pixels exceed budget")
    groups = {item["source_group"] for item in plan_by_key.values()}
    classes = {
        name for item in plan_by_key.values() for name in item["candidate_runtime_classes"]
    }
    gates = pp["gates"]
    if len(groups) < int(gates["min_source_groups"]):
        raise PolicyError("source-group gate failed")
    if gates["runtime_class_gate_available"] and len(classes) < int(
        gates["min_runtime_classes_if_available"]
    ):
        raise PolicyError("runtime-class evidence gate failed")
    per_group: dict[str, Counter[str]] = defaultdict(Counter)
    for item in plan_by_key.values():
        for role in item["roles"]:
            per_group[item["source_group"]][role] += 1
    for role_counts in per_group.values():
        for role, cap in pp["role_caps_per_group"].items():
            if role_counts[role] > int(cap):
                raise PolicyError("per-source-group role cap failed")
    if not mp.get("gates", {}).get("download_complete"):
        raise IntegrityError("downloader completion gate failed")
    if not mp.get("gates", {}).get("training_usable_without_snapshot_quarantine"):
        raise IntegrityError("downloader integrity gate failed")
    if int(mp.get("counts", {}).get("downloaded_files", -1)) != len(plan_by_key):
        raise IntegrityError("manifest downloaded-file count drift")
    expected_bytes = sum(
        int(item["entry"]["uncompressed_size"]) for item in plan_by_key.values()
    )
    if int(mp.get("counts", {}).get("downloaded_bytes", -1)) != expected_bytes:
        raise IntegrityError("manifest downloaded-byte count drift")
    if int(mp.get("counts", {}).get("decoded_pixels", -1)) != total_pixels:
        raise IntegrityError("manifest decoded-pixel count drift")
    if int(mp.get("counts", {}).get("http_range_requests", -1)) != 2 * len(plan_by_key):
        raise IntegrityError("manifest bounded-request count drift")
    return artifact(
        VERIFY_SCHEMA,
        {
            "plan_payload_sha256": plan["payload_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "counts": {
                "files": len(plan_by_key),
                "source_groups": len(groups),
                "runtime_classes_evidenced": len(classes),
                "decoded_pixels": total_pixels,
                "cross_group_pixel_duplicates": 0,
            },
            "gates": {
                "plan_manifest_binding": True,
                "hashes_crc_and_budgets_verified": True,
                "group_and_class_coverage_verified": True,
                "single_label_supervision_absent": True,
                "cross_group_pixel_duplicates_absent": True,
                "promotion_allowed": False,
            },
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tlachia P4 bounded multi-view SSL pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="freeze a deterministic bounded pilot plan")
    plan.add_argument("--api-catalog", type=Path, default=DEFAULT_API)
    plan.add_argument("--join-catalog", type=Path, default=DEFAULT_JOIN)
    plan.add_argument("--zip-catalog", type=Path, default=DEFAULT_ZIP)
    plan.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    plan.add_argument("--max-files", type=int, default=800)
    plan.add_argument("--max-groups", type=int, default=100)
    plan.add_argument("--max-compressed-bytes", type=int, default=1_500_000_000)
    plan.add_argument("--max-uncompressed-bytes", type=int, default=2_000_000_000)
    plan.add_argument("--max-requests", type=int, default=1600)
    plan.add_argument("--max-pixel-count", type=int, default=1_000_000_000)
    plan.add_argument("--max-glyphs-per-group", type=int, default=8)
    plan.add_argument("--max-zones-per-group", type=int, default=2)
    plan.add_argument("--max-planches-per-group", type=int, default=1)
    plan.add_argument("--min-source-groups", type=int, default=20)
    plan.add_argument("--min-runtime-classes", type=int, default=100)
    plan.add_argument("--output", type=Path, required=True)

    download = sub.add_parser("download", help="execute an immutable reviewed plan")
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--capture-dir", type=Path, required=True)
    download.add_argument("--output-root", type=Path, required=True)
    download.add_argument("--manifest-output", type=Path, required=True)
    download.add_argument("--offline", action="store_true")
    download.add_argument("--apply", action="store_true")

    verify = sub.add_parser("verify", help="verify the exact completed pilot")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_multiview_plan(
                api_catalog=read_json(args.api_catalog),
                join_catalog=read_json(args.join_catalog),
                zip_catalog=read_json(args.zip_catalog),
                mapping=read_json(args.mapping),
                max_files=args.max_files,
                max_groups=args.max_groups,
                max_compressed_bytes=args.max_compressed_bytes,
                max_uncompressed_bytes=args.max_uncompressed_bytes,
                max_requests=args.max_requests,
                max_pixel_count=args.max_pixel_count,
                max_glyphs_per_group=args.max_glyphs_per_group,
                max_zones_per_group=args.max_zones_per_group,
                max_planches_per_group=args.max_planches_per_group,
                min_source_groups=args.min_source_groups,
                min_runtime_classes=args.min_runtime_classes,
            )
            write_json(args.output, result, write_once=True)
        elif args.command == "download":
            if not args.apply:
                raise PolicyError("download requires explicit --apply after plan review")
            plan = read_json(args.plan)
            verify_artifact(plan, MEDIA_PLAN_SCHEMA)
            if plan["payload"].get("purpose") != "multiview_self_supervised":
                raise PolicyError("download accepts only a multi-view SSL plan")
            transport = CachedTransport(
                args.capture_dir,
                offline=args.offline,
                max_requests=int(plan["payload"]["limits"]["max_requests"]),
            )
            result = download_media_plan(plan, transport=transport, output_root=args.output_root)
            write_json(args.manifest_output, result, write_once=True)
        elif args.command == "verify":
            result = verify_multiview_download(
                read_json(args.plan), read_json(args.manifest), output_root=args.output_root
            )
            write_json(args.output, result, write_once=True)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (TlachiaError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

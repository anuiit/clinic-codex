from __future__ import annotations

import binascii
import csv
import hashlib
import io
import os
import struct
import uuid
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

from .catalog import CachedTransport
from .core import IntegrityError, PolicyError, artifact, sha256_bytes, verify_artifact
from .p3 import MAPPING_SCHEMA, TRAINABLE_MAPPING_STATUSES


MEDIA_PLAN_SCHEMA = "tlachia-media-download-plan.v1"
MEDIA_MANIFEST_SCHEMA = "tlachia-media-download-manifest.v1"
PHYSICAL_REVIEW_SCHEMA = "tlachia-physical-mapping-review.v1"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")
MAX_IMAGE_PIXELS = 80_000_000


def build_physical_mapping_review(plan: dict[str, Any]) -> dict[str, Any]:
    """Extract archive-scoped theme references that cannot carry one class safely."""
    verify_artifact(plan, MEDIA_PLAN_SCHEMA)
    conflicts: list[dict[str, Any]] = []
    for item in plan["payload"].get("entries", []):
        classes = item.get("candidate_runtime_classes", [])
        pending_statuses = sorted(
            status
            for status in item.get("mapping_statuses", [])
            if status not in TRAINABLE_MAPPING_STATUSES
        )
        if len(classes) == 1 and not pending_statuses:
            continue
        if pending_statuses:
            reason = "pending_mapping_review_for_physical_reference"
        elif not classes:
            reason = "no_approved_runtime_class"
        else:
            reason = "multiple_runtime_classes_for_one_physical_reference"
        conflicts.append(
            {
                "archive_name": item["archive_name"],
                "entry_path": item["entry_path"],
                "reason": reason,
                "candidate_runtime_classes": classes,
                "mapping_ids": item["mapping_ids"],
                "mapping_statuses": item["mapping_statuses"],
                "codex_ids": item["codex_ids"],
                "occurrence_count": len(item["node_keys"]),
            }
        )
    conflicts.sort(key=lambda row: (row["reason"], row["archive_name"], row["entry_path"]))
    return artifact(
        PHYSICAL_REVIEW_SCHEMA,
        {
            "media_plan_payload_sha256": plan["payload_sha256"],
            "counts": {
                "physical_references": len(conflicts),
                "missing_class_references": sum(
                    row["reason"] == "no_approved_runtime_class" for row in conflicts
                ),
                "multiple_class_references": sum(
                    row["reason"] == "multiple_runtime_classes_for_one_physical_reference"
                    for row in conflicts
                ),
                "pending_review_references": sum(
                    row["reason"] == "pending_mapping_review_for_physical_reference"
                    for row in conflicts
                ),
                "occurrences": sum(row["occurrence_count"] for row in conflicts),
            },
            "conflicts": conflicts,
        },
    )


def write_physical_mapping_review_csv(review: dict[str, Any], path: Path) -> None:
    verify_artifact(review, PHYSICAL_REVIEW_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "archive_name",
        "entry_path",
        "reason",
        "candidate_runtime_classes",
        "mapping_ids",
        "mapping_statuses",
        "codex_ids",
        "occurrence_count",
        "business_status",
        "business_note",
        "business_reviewer",
        "business_reviewed_at",
        "business_ticket",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in review["payload"]["conflicts"]:
            writer.writerow(
                {
                    **{
                        key: ";".join(row[key]) if isinstance(row[key], list) else row[key]
                        for key in fieldnames[:8]
                    },
                    "business_status": "",
                    "business_note": "",
                    "business_reviewer": "",
                    "business_reviewed_at": "",
                    "business_ticket": "",
                }
            )


MAX_IMAGE_DIMENSION = 10_000


def _archive_index(zip_catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verify_artifact(zip_catalog, "tlachia-zip-catalog.v1")
    return {
        archive["archive_name"]: archive
        for archive in zip_catalog["payload"].get("archives", [])
    }


def _entry_index(archive: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entry in archive.get("entries", []):
        path = entry["path"]
        if path in index:
            raise IntegrityError(f"duplicate ZIP entry path: {archive['archive_name']}:{path}")
        index[path] = entry
    return index


def build_media_plan(
    *,
    api_catalog: dict[str, Any],
    join_catalog: dict[str, Any],
    zip_catalog: dict[str, Any],
    mapping: dict[str, Any],
    purpose: str,
    codex_ids: set[str] | None = None,
    max_files: int,
    max_compressed_bytes: int,
    max_uncompressed_bytes: int,
    max_requests: int,
    max_pixel_count: int,
) -> dict[str, Any]:
    verify_artifact(api_catalog, "tlachia-api-catalog.v1")
    verify_artifact(join_catalog, "tlachia-api-zip-join.v1")
    verify_artifact(mapping, MAPPING_SCHEMA)
    if purpose not in {"review_evidence", "training_source"}:
        raise PolicyError("purpose must be review_evidence or training_source")
    if join_catalog["payload"]["api_payload_sha256"] != api_catalog["payload_sha256"]:
        raise IntegrityError("join catalog does not bind the API catalog")
    if join_catalog["payload"]["zip_payload_sha256"] != zip_catalog["payload_sha256"]:
        raise IntegrityError("join catalog does not bind the ZIP catalog")
    if mapping["payload"]["api_catalog_payload_sha256"] != api_catalog["payload_sha256"]:
        raise IntegrityError("mapping does not bind the API catalog")

    mapping_by_node: dict[str, dict[str, Any]] = {}
    for row in mapping["payload"]["mappings"]:
        for node_key in row["node_keys"]:
            if node_key in mapping_by_node:
                raise IntegrityError(f"element node belongs to multiple mappings: {node_key}")
            mapping_by_node[node_key] = row

    nodes = {
        node["key"]: node
        for node in api_catalog["payload"].get("nodes", {}).get("element", [])
    }
    archives = _archive_index(zip_catalog)
    archive_entries = {
        name: _entry_index(archive) for name, archive in archives.items()
    }
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_statuses: dict[str, int] = defaultdict(int)

    for relation in join_catalog["payload"].get("relations", []):
        if relation.get("role") != "element_theme_reference":
            continue
        node_key = relation["node"]
        node = nodes.get(node_key)
        row = mapping_by_node.get(node_key)
        if node is None or row is None:
            raise IntegrityError(f"unbound element relation: {node_key}")
        codex_id = str(node["codex_id"])
        if codex_ids is not None and codex_id not in codex_ids:
            continue
        mapping_status = row["status"]
        if purpose == "training_source" and mapping_status not in TRAINABLE_MAPPING_STATUSES:
            skipped_statuses[mapping_status] += 1
            # A pending decision must still taint the physical ZIP member.  A
            # reviewed rejection/quarantine, however, is an explicit exclusion
            # and therefore does not participate in the physical unit.
            if mapping_status in {"rejected_manual", "quarantined_manual"}:
                continue
        archive_name = relation["archive"]
        entry_path = relation["entry_path"]
        archive = archives.get(archive_name)
        if archive is None or archive.get("status") != "available":
            raise IntegrityError(f"selected media archive unavailable: {archive_name}")
        entry = archive_entries[archive_name].get(entry_path)
        if entry is None:
            raise IntegrityError(f"selected ZIP entry missing: {archive_name}:{entry_path}")
        if entry["crc32"] != relation["entry_crc32"]:
            raise IntegrityError("join CRC differs from ZIP central directory")
        key = (archive_name, entry_path)
        item = selected.setdefault(
            key,
            {
                "archive_name": archive_name,
                "archive_url": archive["url"],
                "archive_etag": archive.get("etag"),
                "archive_content_length": archive["content_length"],
                "archive_central_directory_sha256": archive["central_directory_sha256"],
                "entry_path": entry_path,
                "entry": entry,
                "mapping_ids": [],
                "node_keys": [],
                "occurrence_source_groups": [],
                "candidate_runtime_classes": [],
                "mapping_statuses": [],
                "codex_ids": [],
            },
        )
        item["mapping_ids"].append(row["mapping_id"])
        item["node_keys"].append(node_key)
        item["occurrence_source_groups"].append(node["source_group"])
        if (
            mapping_status in TRAINABLE_MAPPING_STATUSES
            and row.get("candidate_runtime_class_name")
        ):
            item["candidate_runtime_classes"].append(row["candidate_runtime_class_name"])
        item["mapping_statuses"].append(mapping_status)
        item["codex_ids"].append(codex_id)

    entries: list[dict[str, Any]] = []
    physical_quarantine: list[dict[str, Any]] = []
    for item in selected.values():
        for key in (
            "mapping_ids",
            "node_keys",
            "occurrence_source_groups",
            "candidate_runtime_classes",
            "mapping_statuses",
            "codex_ids",
        ):
            item[key] = sorted(set(item[key]))
        pending_statuses = sorted(
            status
            for status in item["mapping_statuses"]
            if status not in TRAINABLE_MAPPING_STATUSES
        )
        if purpose == "training_source" and (
            pending_statuses or len(item["candidate_runtime_classes"]) != 1
        ):
            if pending_statuses:
                reason = "pending_mapping_review_for_physical_reference"
            elif len(item["candidate_runtime_classes"]) > 1:
                reason = "multiple_runtime_classes_for_one_physical_reference"
            else:
                reason = "missing_runtime_class_for_physical_reference"
            physical_quarantine.append(
                {
                    "archive_name": item["archive_name"],
                    "entry_path": item["entry_path"],
                    "candidate_runtime_classes": item["candidate_runtime_classes"],
                    "mapping_ids": item["mapping_ids"],
                    "mapping_statuses": item["mapping_statuses"],
                    "occurrence_count": len(item["node_keys"]),
                    "reason": reason,
                }
            )
            continue
        item["physical_source_group"] = (
            f"tlachia:archive:{item['archive_name']}:theme-reference:"
            f"{item['entry_path'].casefold()}"
        )
        entries.append(item)
    entries.sort(key=lambda item: (item["archive_name"], item["entry_path"]))

    counts = {
        "files": len(entries),
        "compressed_bytes": sum(item["entry"]["compressed_size"] for item in entries),
        "uncompressed_bytes": sum(item["entry"]["uncompressed_size"] for item in entries),
        "planned_requests": 2 * len(entries),
        "occurrences": sum(len(item["node_keys"]) for item in entries),
        "runtime_classes": len(
            {name for item in entries for name in item["candidate_runtime_classes"]}
        ),
        "physical_quarantine": len(physical_quarantine),
        "quarantined_occurrences": sum(
            item["occurrence_count"] for item in physical_quarantine
        ),
    }
    limits = {
        "max_files": max_files,
        "max_compressed_bytes": max_compressed_bytes,
        "max_uncompressed_bytes": max_uncompressed_bytes,
        "max_requests": max_requests,
        "max_pixel_count": max_pixel_count,
    }
    comparisons = {
        "files": max_files,
        "compressed_bytes": max_compressed_bytes,
        "uncompressed_bytes": max_uncompressed_bytes,
        "planned_requests": max_requests,
    }
    for count_key, maximum in comparisons.items():
        if counts[count_key] > maximum:
            raise PolicyError(f"planned {count_key} exceeds explicit budget")
    if not entries:
        raise PolicyError("media plan selected no entries")
    return artifact(
        MEDIA_PLAN_SCHEMA,
        {
            "purpose": purpose,
            "api_payload_sha256": api_catalog["payload_sha256"],
            "join_payload_sha256": join_catalog["payload_sha256"],
            "zip_payload_sha256": zip_catalog["payload_sha256"],
            "mapping_payload_sha256": mapping["payload_sha256"],
            "codex_ids": sorted(codex_ids) if codex_ids is not None else None,
            "policy": {
                "whole_zip_fallback": False,
                "supported_compression": [0, 8],
                "require_etag": True,
                "require_http_206": True,
                "pixel_hash": "sha256(RGB:{width}x{height}: + exif-transposed RGB bytes)",
                "physical_unit": "archive-scoped theme reference",
            },
            "limits": limits,
            "counts": counts,
            "skipped_mapping_statuses": dict(sorted(skipped_statuses.items())),
            "physical_quarantine": sorted(
                physical_quarantine,
                key=lambda row: (row["archive_name"], row["entry_path"]),
            ),
            "entries": entries,
        },
    )


def _range(
    transport: CachedTransport,
    *,
    url: str,
    start: int,
    end: int,
    etag: str,
    archive_size: int,
) -> bytes:
    if end < start:
        raise IntegrityError("invalid HTTP byte range")
    response = transport.request(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}", "If-Match": etag},
        max_bytes=end - start + 1,
        allowed_statuses=(206,),
    )
    if response.status != 206:
        raise IntegrityError("ZIP member request did not return HTTP 206")
    if response.headers.get("etag") != etag:
        raise IntegrityError("archive ETag drifted during bounded download")
    expected_range = f"bytes {start}-{end}/{archive_size}"
    content_range = response.headers.get("content-range", "")
    if content_range != expected_range:
        raise IntegrityError(f"unexpected Content-Range: {content_range!r}")
    content_length = response.headers.get("content-length")
    if content_length is not None and content_length != str(end - start + 1):
        raise IntegrityError("bounded range returned an unexpected Content-Length")
    if len(response.body) != end - start + 1:
        raise IntegrityError("bounded range returned an unexpected byte count")
    return response.body


def _decode_member(
    transport: CachedTransport, item: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    entry = item["entry"]
    archive_size = int(item["archive_content_length"])
    etag = item.get("archive_etag")
    if not isinstance(etag, str) or not etag:
        raise PolicyError("download requires a captured archive ETag")
    offset = int(entry["local_header_offset"])
    fixed = _range(
        transport,
        url=item["archive_url"],
        start=offset,
        end=offset + 29,
        etag=etag,
        archive_size=archive_size,
    )
    (
        signature,
        _version,
        flags,
        compression,
        _mtime,
        _mdate,
        local_crc,
        local_compressed_size,
        local_uncompressed_size,
        name_length,
        extra_length,
    ) = LOCAL_FILE_HEADER.unpack(fixed)
    if signature != LOCAL_FILE_SIGNATURE:
        raise IntegrityError("invalid local ZIP member signature")
    if flags != int(entry["flag_bits"]):
        raise IntegrityError("local ZIP flags differ from central directory")
    if flags & 0x1 or flags & 0x8:
        raise PolicyError("encrypted/data-descriptor ZIP members are forbidden")
    if compression != int(entry["compression"]) or compression not in {0, 8}:
        raise PolicyError("unsupported or drifted ZIP compression method")
    expected_crc = int(entry["crc32"], 16)
    if (
        local_crc != expected_crc
        or local_compressed_size != int(entry["compressed_size"])
        or local_uncompressed_size != int(entry["uncompressed_size"])
    ):
        raise IntegrityError("local ZIP CRC/sizes differ from central directory")
    compressed_size = int(entry["compressed_size"])
    suffix_start = offset + 30
    suffix_end = suffix_start + name_length + extra_length + compressed_size - 1
    suffix = _range(
        transport,
        url=item["archive_url"],
        start=suffix_start,
        end=suffix_end,
        etag=etag,
        archive_size=archive_size,
    )
    raw_name = suffix[:name_length]
    if sha256_bytes(raw_name) != entry["raw_name_sha256"]:
        raise IntegrityError("local ZIP member name differs from central directory")
    compressed = suffix[name_length + extra_length :]
    if len(compressed) != compressed_size:
        raise IntegrityError("compressed member byte count mismatch")
    expected_uncompressed_size = int(entry["uncompressed_size"])
    try:
        if compression == 0:
            data = compressed
        else:
            decompressor = zlib.decompressobj(-15)
            data = decompressor.decompress(compressed, expected_uncompressed_size + 1)
            if len(data) > expected_uncompressed_size or decompressor.unconsumed_tail:
                raise IntegrityError("ZIP member exceeded its frozen uncompressed size")
            remaining_budget = expected_uncompressed_size + 1 - len(data)
            data += decompressor.flush(remaining_budget)
            if (
                len(data) > expected_uncompressed_size
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise IntegrityError("ZIP deflate stream exceeds or violates its frozen bound")
    except zlib.error as error:
        raise IntegrityError("raw deflate decompression failed") from error
    if len(data) != expected_uncompressed_size:
        raise IntegrityError("uncompressed member byte count mismatch")
    crc32 = f"{binascii.crc32(data) & 0xFFFFFFFF:08x}"
    if crc32 != entry["crc32"]:
        raise IntegrityError("ZIP member CRC32 mismatch")
    return data, {"local_header_bytes": 30 + name_length + extra_length, "crc32": crc32}


def _image_identity(data: bytes, *, max_pixel_count: int) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            width, height = rgb.size
            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > min(MAX_IMAGE_PIXELS, max_pixel_count)
            ):
                raise PolicyError("decoded image exceeds the explicit safety budget")
            digest = hashlib.sha256()
            digest.update(f"RGB:{width}x{height}:".encode("ascii"))
            digest.update(rgb.tobytes())
            return {
                "width": width,
                "height": height,
                "mode": "RGB",
                "pixel_count": width * height,
                "pixel_sha256": digest.hexdigest(),
            }
    except PolicyError:
        raise
    except Exception as error:
        raise IntegrityError("downloaded member is not a valid bounded image") from error


def _safe_output_path(root: Path, archive_name: str, entry_path: str) -> Path:
    pure = PurePosixPath(entry_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PolicyError(f"unsafe ZIP entry path: {entry_path!r}")
    target = root / "files" / Path(archive_name).stem
    for part in pure.parts:
        target /= part
    resolved_root = root.resolve()
    resolved_parent = target.parent.resolve()
    if resolved_root != resolved_parent and resolved_root not in resolved_parent.parents:
        raise PolicyError("ZIP member output escapes destination")
    return target


def _write_binary_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if sha256_bytes(path.read_bytes()) != sha256_bytes(data):
            raise IntegrityError(f"refusing to replace a different downloaded file: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def download_media_plan(
    plan: dict[str, Any],
    *,
    transport: CachedTransport,
    output_root: Path,
) -> dict[str, Any]:
    verify_artifact(plan, MEDIA_PLAN_SCHEMA)
    payload = plan["payload"]
    if transport.max_requests < payload["counts"]["planned_requests"]:
        raise PolicyError("transport request budget is below the frozen plan")
    downloaded: list[dict[str, Any]] = []
    total_pixels = 0
    for item in payload["entries"]:
        data, zip_evidence = _decode_member(transport, item)
        image = _image_identity(data, max_pixel_count=payload["limits"]["max_pixel_count"])
        total_pixels += image["pixel_count"]
        if total_pixels > payload["limits"]["max_pixel_count"]:
            raise PolicyError("aggregate decoded pixels exceed the frozen plan budget")
        target = _safe_output_path(output_root, item["archive_name"], item["entry_path"])
        _write_binary_once(target, data)
        downloaded.append(
            {
                "archive_name": item["archive_name"],
                "archive_url": item["archive_url"],
                "archive_etag": item["archive_etag"],
                "archive_content_length": item["archive_content_length"],
                "archive_central_directory_sha256": item[
                    "archive_central_directory_sha256"
                ],
                "entry_path": item["entry_path"],
                "entry_crc32": item["entry"]["crc32"],
                "compressed_size": item["entry"]["compressed_size"],
                "uncompressed_size": item["entry"]["uncompressed_size"],
                "byte_sha256": sha256_bytes(data),
                "output_path": target.relative_to(output_root).as_posix(),
                "image": image,
                "physical_source_group": item["physical_source_group"],
                "mapping_ids": item["mapping_ids"],
                "node_keys": item["node_keys"],
                "candidate_runtime_classes": item["candidate_runtime_classes"],
                "mapping_statuses": item["mapping_statuses"],
                "occurrence_source_groups": item["occurrence_source_groups"],
                "zip_evidence": zip_evidence,
            }
        )

    by_pixel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bytes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in downloaded:
        by_pixel[row["image"]["pixel_sha256"]].append(row)
        by_bytes[row["byte_sha256"]].append(row)
    pixel_conflicts: list[dict[str, Any]] = []
    for pixel_sha, rows in by_pixel.items():
        classes = sorted({name for row in rows for name in row["candidate_runtime_classes"]})
        if len(classes) > 1:
            pixel_conflicts.append(
                {
                    "pixel_sha256": pixel_sha,
                    "runtime_classes": classes,
                    "entries": sorted(
                        f"{row['archive_name']}:{row['entry_path']}" for row in rows
                    ),
                }
            )
    return artifact(
        MEDIA_MANIFEST_SCHEMA,
        {
            "plan_payload_sha256": plan["payload_sha256"],
            "purpose": payload["purpose"],
            "output_root": output_root.resolve().as_posix(),
            "counts": {
                "downloaded_files": len(downloaded),
                "downloaded_bytes": sum(row["uncompressed_size"] for row in downloaded),
                "decoded_pixels": total_pixels,
                "unique_byte_sha256": len(by_bytes),
                "unique_pixel_sha256": len(by_pixel),
                "pixel_conflicts": len(pixel_conflicts),
                "http_range_requests": 2 * len(downloaded),
                "capture_replay_invariant": True,
            },
            "pixel_conflicts": pixel_conflicts,
            "gates": {
                "download_complete": len(downloaded) == payload["counts"]["files"],
                "training_source_pixel_conflicts_absent": not pixel_conflicts,
                "training_usable_without_snapshot_quarantine": (
                    payload["purpose"] != "training_source" or not pixel_conflicts
                ),
            },
            "files": downloaded,
        },
    )

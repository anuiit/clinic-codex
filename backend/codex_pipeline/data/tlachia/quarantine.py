from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .core import (
    IntegrityError,
    artifact,
    read_json,
    sha256_file,
    verify_artifact,
    write_json,
)


SCHEMA_VERSION = "tlachia-business-quarantine.v1"


def build_business_quarantine(
    business_manifest_path: Path,
    *,
    mapping_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(business_manifest_path)
    if manifest.get("schema_version") != "business-test-set.v1":
        raise IntegrityError("business quarantine requires business-test-set.v1")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise IntegrityError("business manifest contains no rows")
    if any(row.get("dataset_split") != "locked_test" for row in rows):
        raise IntegrityError("every business row must remain locked_test")

    mappings: dict[str, dict[str, Any]] = {}
    mapping_sha = None
    if mapping_path is not None:
        mapping_payload = read_json(mapping_path)
        raw_mappings = mapping_payload.get("mappings")
        if not isinstance(raw_mappings, list):
            raise IntegrityError("quarantine mapping file requires mappings[]")
        mapping_sha = sha256_file(mapping_path)
        for mapping in raw_mappings:
            analysis_id = str(mapping.get("analysis_id") or "")
            if not analysis_id or analysis_id in mappings:
                raise IntegrityError("mapping analysis_id is missing or duplicated")
            mappings[analysis_id] = mapping

    pages: dict[str, dict[str, Any]] = {}
    rows_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        required = (
            "row_id",
            "analysis_id",
            "page_sha256",
            "page_pixel_sha256",
            "crop_pixel_sha256",
            "bbox",
            "source_group",
        )
        if any(raw.get(field) is None for field in required):
            raise IntegrityError(f"business row lacks quarantine identity: {raw!r}")
        analysis_id = str(raw["analysis_id"])
        row = {field: raw[field] for field in required}
        rows_by_page[analysis_id].append(row)
        page = pages.setdefault(
            analysis_id,
            {
                "analysis_id": analysis_id,
                "page_sha256": raw["page_sha256"],
                "page_pixel_sha256": raw["page_pixel_sha256"],
                "source_group": raw["source_group"],
                "test_only": True,
                "quarantine_reason": "business_final_test",
            },
        )
        if any(
            page[field] != raw[field]
            for field in ("page_sha256", "page_pixel_sha256", "source_group")
        ):
            raise IntegrityError(f"inconsistent page identity: {analysis_id}")

    for analysis_id, page in pages.items():
        mapping = mappings.get(analysis_id)
        if mapping is None:
            page.update(
                {
                    "mapping_status": "unmapped",
                    "codex_id": None,
                    "planche_cote": None,
                    "mapping_evidence": None,
                }
            )
        else:
            if mapping.get("page_pixel_sha256") != page["page_pixel_sha256"]:
                raise IntegrityError(f"mapping page hash mismatch: {analysis_id}")
            status = mapping.get("mapping_status")
            if status not in {"mapped", "unmapped", "ambiguous"}:
                raise IntegrityError(f"invalid mapping status: {analysis_id}:{status}")
            page.update(
                {
                    "mapping_status": status,
                    "codex_id": mapping.get("codex_id") if status == "mapped" else None,
                    "planche_cote": (
                        mapping.get("planche_cote") if status == "mapped" else None
                    ),
                    "mapping_evidence": mapping.get("evidence"),
                }
            )
            if status == "mapped" and (
                not page["codex_id"]
                or not page["planche_cote"]
                or not page["mapping_evidence"]
            ):
                raise IntegrityError(f"mapped page lacks identity/evidence: {analysis_id}")

    canonical_pages = []
    for analysis_id in sorted(pages):
        page = pages[analysis_id]
        page["rows"] = sorted(rows_by_page[analysis_id], key=lambda row: row["row_id"])
        canonical_pages.append(page)
    payload = {
        "business_test": {
            "test_set_id": manifest.get("test_set_id"),
            "content_sha256": manifest.get("content_sha256"),
            "manifest_sha256": sha256_file(business_manifest_path),
            "row_count": len(rows),
            "page_count": len(canonical_pages),
        },
        "mapping_manifest_sha256": mapping_sha,
        "pages": canonical_pages,
        "test_only": True,
        "blocks_ml_export": True,
        "unresolved_page_count": sum(
            page["mapping_status"] != "mapped" for page in canonical_pages
        ),
        "enforcement": "verify_candidate_export must pass before any ML export",
    }
    return artifact(SCHEMA_VERSION, payload)


def write_quarantine(path: Path, value: dict[str, Any]) -> None:
    write_json(path, value, write_once=True)


def verify_candidate_export(
    quarantine: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_locked_test: bool = False,
) -> None:
    verify_artifact(quarantine, SCHEMA_VERSION)
    if candidate.get("schema_version") != "training-snapshot.v2":
        raise IntegrityError("candidate export must use training-snapshot.v2")
    payload = quarantine.get("payload")
    rows = candidate.get("rows")
    if not isinstance(payload, dict) or not isinstance(rows, list):
        raise IntegrityError("invalid quarantine or candidate export")
    forbidden_row_ids = {
        row["row_id"] for page in payload["pages"] for row in page["rows"]
    }
    forbidden_pages = {page["page_pixel_sha256"] for page in payload["pages"]}
    forbidden_groups = {page["source_group"] for page in payload["pages"]}
    forbidden_crops = {
        row["crop_pixel_sha256"] for page in payload["pages"] for row in page["rows"]
    }
    for row in rows:
        required = (
            "row_id",
            "dataset_split",
            "source_group",
            "source_pixel_sha256",
            "output_sha256",
        )
        if not isinstance(row, dict) or any(row.get(field) is None for field in required):
            raise IntegrityError("candidate row lacks mandatory quarantine identity")
        if row["row_id"] in forbidden_row_ids:
            raise IntegrityError("candidate contains a quarantined business row")
        identities = {row["source_pixel_sha256"], row["output_sha256"]}
        if identities & forbidden_pages:
            raise IntegrityError("candidate contains a quarantined business page")
        if identities & forbidden_crops:
            raise IntegrityError("candidate contains a quarantined business crop")
        if row["source_group"] in forbidden_groups:
            raise IntegrityError("candidate contains a quarantined business source group")
        if row["dataset_split"] == "locked_test" and not allow_locked_test:
            raise IntegrityError("candidate export contains locked_test data")

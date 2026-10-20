from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core import (
    IntegrityError,
    PolicyError,
    artifact,
    read_json,
    semantic_hash,
    verify_artifact,
)


MAPPING_SCHEMA = "tlachia-runtime-mapping.v1"
DECISIONS_SCHEMA = "tlachia-mapping-decisions.v1"
TRAINABLE_MAPPING_STATUSES = {"accepted_exact", "approved_manual"}
REVIEWABLE_STATUSES = {"needs_business_review"}
_SUFFIXED_CLASS_RE = re.compile(r"^(?P<stem>.+)_\d+$")


def _nfc(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def load_runtime_class_order(config_path: Path) -> tuple[list[str], str]:
    config = read_json(config_path)
    class_names = config.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise IntegrityError("runtime config must contain a non-empty class_names list")
    names = [_nfc(name) for name in class_names]
    if len(names) != 286 or len(set(names)) != len(names):
        raise IntegrityError("runtime taxonomy must contain exactly 286 unique classes")
    if int(config.get("num_classes", -1)) != len(names):
        raise IntegrityError("runtime num_classes does not match class_names")
    return names, semantic_hash(names)


def _runtime_families(class_names: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for name in class_names:
        match = _SUFFIXED_CLASS_RE.fullmatch(name)
        families[match.group("stem") if match else name].append(name)
    return {key: sorted(values) for key, values in families.items()}


def _candidate_from_raw(raw: dict[str, Any], runtime_by_name: dict[str, int]) -> str | None:
    # Values/meanings are only review hints.  They never auto-approve a mapping.
    for field in ("valeur", "meaning", "meaning2"):
        candidate = _nfc(raw.get(field))
        if candidate in runtime_by_name:
            return candidate
    return None


def build_runtime_mapping(
    api_catalog: dict[str, Any],
    *,
    runtime_config_path: Path,
) -> dict[str, Any]:
    verify_artifact(api_catalog, "tlachia-api-catalog.v1")
    api_payload = api_catalog["payload"]
    if not api_payload.get("include_elements"):
        raise PolicyError("an element-complete API catalog is required for P3 mapping")

    class_names, class_order_sha256 = load_runtime_class_order(runtime_config_path)
    runtime_by_name = {name: index for index, name in enumerate(class_names)}
    families = _runtime_families(class_names)
    elements = api_payload.get("nodes", {}).get("element")
    if not isinstance(elements, list) or not elements:
        raise IntegrityError("API catalog contains no element nodes")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in elements:
        if not isinstance(node, dict):
            raise IntegrityError("element node must be an object")
        label = _nfc(node.get("element"))
        theme = _nfc(node.get("theme"))
        if not label or not theme:
            raise IntegrityError("element node lacks label or theme")
        grouped[(label, theme)].append(node)

    mappings: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, Any]] = {
        name: {
            "runtime_class_label": index,
            "legacy_class_index": index + 1,
            "runtime_class_name": name,
            "mapping_ids": [],
            "occurrence_count": 0,
            "codex_count": 0,
            "themes": [],
        }
        for index, name in enumerate(class_names)
    }

    for (label, theme), nodes in sorted(grouped.items()):
        codex_ids = sorted({_nfc(node.get("codex_id")) for node in nodes})
        candidate: str | None = label if label in runtime_by_name else None
        match_method = "exact" if candidate is not None else "unmapped"
        review_reason: str | None = None
        status = "needs_business_review"

        if candidate is not None:
            match = _SUFFIXED_CLASS_RE.fullmatch(candidate)
            family_key = match.group("stem") if match else candidate
            family = families.get(family_key, [candidate])
            if len(family) == 1 and match is None:
                status = "accepted_exact"
            else:
                review_reason = "runtime_suffix_family_requires_semantic_review"
        else:
            hint_candidates = {
                value
                for value in (
                    _candidate_from_raw(node.get("raw", {}), runtime_by_name)
                    for node in nodes
                )
                if value is not None
            }
            if len(hint_candidates) == 1:
                candidate = next(iter(hint_candidates))
                match_method = "raw_field_hint"
                review_reason = "non_label_field_candidate_requires_review"
            elif len(hint_candidates) > 1:
                review_reason = "conflicting_raw_field_candidates"
            else:
                review_reason = "no_runtime_candidate"

        mapping_identity = {
            "tlachia_element_label": label,
            "theme": theme,
        }
        mapping_id = "tlm-" + semantic_hash(mapping_identity)[:20]
        row = {
            "mapping_id": mapping_id,
            "tlachia_element_label": label,
            "theme": theme,
            "match_method": match_method,
            "status": status,
            "review_reason": review_reason,
            "candidate_runtime_class_name": candidate,
            "runtime_class_label": (
                runtime_by_name[candidate] if candidate in runtime_by_name else None
            ),
            "legacy_class_index": (
                runtime_by_name[candidate] + 1 if candidate in runtime_by_name else None
            ),
            "occurrence_count": len(nodes),
            "codex_ids": codex_ids,
            "node_keys": sorted({_nfc(node.get("key")) for node in nodes}),
            "source_groups": sorted(
                {_nfc(node.get("source_group")) for node in nodes}
            ),
            "evidence": [
                {
                    "codex_id": _nfc(node.get("codex_id")),
                    "codex_code": _nfc(node.get("codex_code")),
                    "glyph_cote": _nfc(node.get("glyph_cote")),
                    "element_id": _nfc(node.get("element_id")),
                    "source_group": _nfc(node.get("source_group")),
                }
                for node in sorted(nodes, key=lambda item: _nfc(item.get("key")))
            ],
            "review": None,
        }
        mappings.append(row)
        if candidate in by_class:
            class_row = by_class[candidate]
            class_row["mapping_ids"].append(mapping_id)
            class_row["occurrence_count"] += len(nodes)
            class_row["themes"].append(theme)
            class_row.setdefault("_codex_ids", set()).update(codex_ids)

    runtime_classes: list[dict[str, Any]] = []
    for name in class_names:
        row = by_class[name]
        row["mapping_ids"] = sorted(row["mapping_ids"])
        row["themes"] = sorted(set(row["themes"]))
        row["codex_count"] = len(row.pop("_codex_ids", set()))
        runtime_classes.append(row)

    status_counts: dict[str, int] = defaultdict(int)
    for row in mappings:
        status_counts[row["status"]] += 1
    covered = {row["candidate_runtime_class_name"] for row in mappings if row["candidate_runtime_class_name"]}
    trainable = {
        row["candidate_runtime_class_name"]
        for row in mappings
        if row["status"] in TRAINABLE_MAPPING_STATUSES
    }
    payload = {
        "api_catalog_payload_sha256": api_catalog["payload_sha256"],
        "runtime_config_path": runtime_config_path.as_posix(),
        "runtime_config_sha256": hashlib.sha256(runtime_config_path.read_bytes()).hexdigest(),
        "class_order_sha256": class_order_sha256,
        "policy": {
            "automatic_acceptance": "NFC exact label only, excluding runtime suffix families",
            "manual_review_required_for": [
                "runtime suffix families",
                "non-label-field candidates",
                "unmapped labels",
            ],
            "training_statuses": sorted(TRAINABLE_MAPPING_STATUSES),
            "runtime_class_label_base": 0,
            "legacy_class_index_base": 1,
        },
        "counts": {
            "runtime_classes": len(class_names),
            "mapping_rows": len(mappings),
            "element_occurrences": len(elements),
            "candidate_runtime_classes": len(covered),
            "trainable_runtime_classes": len(trainable),
            "review_rows": sum(row["status"] in REVIEWABLE_STATUSES for row in mappings),
            "statuses": dict(sorted(status_counts.items())),
        },
        "runtime_classes": runtime_classes,
        "mappings": mappings,
    }
    return artifact(MAPPING_SCHEMA, payload)


def apply_business_decisions(
    mapping_artifact: dict[str, Any], decisions_artifact: dict[str, Any]
) -> dict[str, Any]:
    verify_artifact(mapping_artifact, MAPPING_SCHEMA)
    verify_artifact(decisions_artifact, DECISIONS_SCHEMA)
    decisions = decisions_artifact["payload"]
    if decisions.get("mapping_payload_sha256") != mapping_artifact["payload_sha256"]:
        raise IntegrityError("business decisions bind a different mapping proposal")
    reviewer = _nfc(decisions.get("reviewer"))
    reviewed_at = _nfc(decisions.get("reviewed_at"))
    if not reviewer or not reviewed_at:
        raise IntegrityError("business decisions require reviewer and reviewed_at")

    payload = dict(mapping_artifact["payload"])
    rows = [dict(row) for row in payload["mappings"]]
    by_id = {row["mapping_id"]: row for row in rows}
    seen: set[str] = set()
    runtime_names = {row["runtime_class_name"] for row in payload["runtime_classes"]}
    for decision in decisions.get("decisions", []):
        mapping_id = _nfc(decision.get("mapping_id"))
        if mapping_id in seen or mapping_id not in by_id:
            raise IntegrityError(f"unknown or duplicate mapping decision: {mapping_id}")
        seen.add(mapping_id)
        status = _nfc(decision.get("status"))
        if status not in {"approved", "rejected", "quarantined"}:
            raise IntegrityError(f"invalid business decision status: {status}")
        row = by_id[mapping_id]
        if row["status"] not in REVIEWABLE_STATUSES:
            raise PolicyError("business decisions may only resolve reviewable mappings")
        target = _nfc(decision.get("runtime_class_name")) or None
        if status == "approved" and target not in runtime_names:
            raise IntegrityError("approved decision must target an exact runtime class")
        if status != "approved" and target is not None:
            raise IntegrityError("rejected/quarantined decisions cannot target a class")
        row["candidate_runtime_class_name"] = target
        if target is not None:
            class_index = next(
                item["runtime_class_label"]
                for item in payload["runtime_classes"]
                if item["runtime_class_name"] == target
            )
            row["runtime_class_label"] = class_index
            row["legacy_class_index"] = class_index + 1
        else:
            row["runtime_class_label"] = None
            row["legacy_class_index"] = None
        row["status"] = {
            "approved": "approved_manual",
            "rejected": "rejected_manual",
            "quarantined": "quarantined_manual",
        }[status]
        row["match_method"] = "manual_business_review"
        row["review"] = {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "ticket": _nfc(decision.get("ticket")) or None,
            "note": _nfc(decision.get("note")) or None,
            "decision_payload_sha256": decisions_artifact["payload_sha256"],
        }

    payload["mapping_proposal_payload_sha256"] = mapping_artifact["payload_sha256"]
    payload["business_decisions_payload_sha256"] = decisions_artifact["payload_sha256"]
    payload["mappings"] = rows
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[row["status"]] += 1
    payload["counts"] = dict(payload["counts"])
    payload["counts"]["statuses"] = dict(sorted(statuses.items()))
    payload["counts"]["review_rows"] = sum(
        row["status"] in REVIEWABLE_STATUSES for row in rows
    )
    payload["counts"]["trainable_runtime_classes"] = len(
        {
            row["candidate_runtime_class_name"]
            for row in rows
            if row["status"] in TRAINABLE_MAPPING_STATUSES
        }
    )
    aggregates = {
        row["runtime_class_name"]: {
            **row,
            "mapping_ids": [],
            "occurrence_count": 0,
            "codex_count": 0,
            "themes": [],
            "_codex_ids": set(),
        }
        for row in payload["runtime_classes"]
    }
    for row in rows:
        target = row.get("candidate_runtime_class_name")
        if target not in aggregates or row["status"] not in TRAINABLE_MAPPING_STATUSES:
            continue
        aggregate = aggregates[target]
        aggregate["mapping_ids"].append(row["mapping_id"])
        aggregate["occurrence_count"] += int(row["occurrence_count"])
        aggregate["themes"].append(row["theme"])
        aggregate["_codex_ids"].update(row["codex_ids"])
    rebuilt_runtime_classes: list[dict[str, Any]] = []
    for original in payload["runtime_classes"]:
        aggregate = aggregates[original["runtime_class_name"]]
        aggregate["mapping_ids"] = sorted(set(aggregate["mapping_ids"]))
        aggregate["themes"] = sorted(set(aggregate["themes"]))
        aggregate["codex_count"] = len(aggregate.pop("_codex_ids"))
        rebuilt_runtime_classes.append(aggregate)
    payload["runtime_classes"] = rebuilt_runtime_classes
    return artifact(MAPPING_SCHEMA, payload)


def write_review_csv(mapping_artifact: dict[str, Any], path: Path) -> None:
    verify_artifact(mapping_artifact, MAPPING_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mapping_id",
        "tlachia_element_label",
        "theme",
        "occurrence_count",
        "codex_ids",
        "candidate_runtime_class_name",
        "review_reason",
        "business_status",
        "business_runtime_class_name",
        "business_note",
        "business_reviewer",
        "business_reviewed_at",
        "business_ticket",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping_artifact["payload"]["mappings"]:
            if row["status"] not in REVIEWABLE_STATUSES:
                continue
            writer.writerow(
                {
                    "mapping_id": row["mapping_id"],
                    "tlachia_element_label": row["tlachia_element_label"],
                    "theme": row["theme"],
                    "occurrence_count": row["occurrence_count"],
                    "codex_ids": ";".join(row["codex_ids"]),
                    "candidate_runtime_class_name": row["candidate_runtime_class_name"] or "",
                    "review_reason": row["review_reason"] or "",
                    "business_status": "",
                    "business_runtime_class_name": "",
                    "business_note": "",
                    "business_reviewer": "",
                    "business_reviewed_at": "",
                    "business_ticket": "",
                }
            )


def decisions_from_review_csv(path: Path, mapping_payload_sha256: str) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    reviewer_values: set[str] = set()
    reviewed_at_values: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = _nfc(row.get("business_status"))
            if not status:
                continue
            reviewer_values.add(_nfc(row.get("business_reviewer")))
            reviewed_at_values.add(_nfc(row.get("business_reviewed_at")))
            decisions.append(
                {
                    "mapping_id": _nfc(row.get("mapping_id")),
                    "status": status,
                    "runtime_class_name": _nfc(row.get("business_runtime_class_name")) or None,
                    "note": _nfc(row.get("business_note")) or None,
                    "ticket": _nfc(row.get("business_ticket")) or None,
                }
            )
    if len(reviewer_values) != 1 or "" in reviewer_values:
        raise IntegrityError("review CSV must contain one non-empty reviewer")
    if len(reviewed_at_values) != 1 or "" in reviewed_at_values:
        raise IntegrityError("review CSV must contain one non-empty reviewed_at")
    return artifact(
        DECISIONS_SCHEMA,
        {
            "mapping_payload_sha256": mapping_payload_sha256,
            "reviewer": next(iter(reviewer_values)),
            "reviewed_at": next(iter(reviewed_at_values)),
            "decisions": sorted(decisions, key=lambda item: item["mapping_id"]),
        },
    )


def iter_trainable_mappings(mapping_artifact: dict[str, Any]) -> Iterable[dict[str, Any]]:
    verify_artifact(mapping_artifact, MAPPING_SCHEMA)
    for row in mapping_artifact["payload"]["mappings"]:
        if row["status"] in TRAINABLE_MAPPING_STATUSES:
            yield row

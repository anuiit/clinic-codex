#!/usr/bin/env python3
"""Metadata-only, resumable Tlachia API/ZIP catalogue builder.

This command deliberately cannot download or extract image payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.codex_pipeline.data.tlachia.catalog import (  # noqa: E402
    CachedTransport,
    inventory_api,
    inventory_zips,
    join_catalogs,
    merge_api_catalogs,
)
from backend.codex_pipeline.data.tlachia.core import (  # noqa: E402
    IntegrityError,
    PolicyError,
    TlachiaError,
    artifact,
    read_json,
    sha256_file,
    write_json,
)
from backend.codex_pipeline.data.tlachia.quarantine import (  # noqa: E402
    build_business_quarantine,
    verify_candidate_export,
    write_quarantine,
)


DEFAULT_STORE = ROOT / "reports" / "tlachia-p0-p2-20260813"
DEFAULT_BUSINESS = (
    ROOT
    / "backend"
    / "training_corpus"
    / "evaluation"
    / "business-final-ea447440c1ae1d12e7b4"
    / "test_manifest.json"
)
DEFAULT_SNAPSHOT = (
    ROOT
    / "backend"
    / "training_corpus"
    / "snapshots"
    / "snapshot-471c74f801b749a9ad6e"
    / "snapshot_manifest.json"
)
DEFAULT_RUNTIME_CONFIG = ROOT / "backend" / "codex_model" / "config.json"


def _emit(value: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _transport(args: argparse.Namespace, capture: str) -> CachedTransport:
    return CachedTransport(
        args.store / "raw" / capture,
        offline=args.offline,
        max_requests=args.max_requests,
        timeout=args.timeout,
        min_interval_seconds=args.min_interval,
    )


def command_preflight(args: argparse.Namespace) -> dict[str, Any]:
    quarantine = build_business_quarantine(
        args.business_manifest,
        mapping_path=args.business_mapping,
    )
    quarantine_path = args.store / "contracts" / "business-quarantine.json"
    write_quarantine(quarantine_path, quarantine)
    for path in (args.training_snapshot, args.runtime_config):
        if not path.is_file():
            raise IntegrityError(f"required preflight input is missing: {path}")
    training_snapshot = read_json(args.training_snapshot)
    # The immutable training snapshot legitimately carries its own diagnostic
    # locked_test rows. This gate proves only that none belongs to the separately
    # frozen business-final quarantine; downstream ML exports remain stricter.
    verify_candidate_export(quarantine, training_snapshot, allow_locked_test=True)
    payload = {
        "mode": "metadata_only",
        "bulk_pixels_allowed": False,
        "business_quarantine_path": str(quarantine_path),
        "business_quarantine_payload_sha256": quarantine["payload_sha256"],
        "business_manifest": {
            "path": str(args.business_manifest),
            "sha256": sha256_file(args.business_manifest),
        },
        "training_snapshot": {
            "path": str(args.training_snapshot),
            "sha256": sha256_file(args.training_snapshot),
        },
        "runtime_config": {
            "path": str(args.runtime_config),
            "sha256": sha256_file(args.runtime_config),
        },
        "gates": {
            "business_rows_test_only": True,
            "training_snapshot_quarantine_verified": True,
            "candidate_export_verifier_required": True,
            "tlachia_train_import_enabled": False,
        },
    }
    result = artifact("tlachia-p0-preflight.v1", payload)
    # v2 records active verification of the training snapshot against the
    # business quarantine; keep the older v1 evidence immutable.
    write_json(args.store / "contracts" / "preflight-v2.json", result, write_once=True)
    return result


def command_verify_quarantine(args: argparse.Namespace) -> dict[str, Any]:
    quarantine = read_json(args.quarantine)
    candidate = read_json(args.candidate)
    verify_candidate_export(quarantine, candidate)
    return {"status": "PASS", "candidate": str(args.candidate)}


def command_inventory_api(args: argparse.Namespace) -> dict[str, Any]:
    transport = _transport(args, args.capture)
    result = inventory_api(
        transport,
        codex_ids=set(args.codex_id) if args.codex_id else None,
        include_elements=args.include_elements,
    )
    write_json(args.output, result, write_once=args.write_once)
    return result


def command_inventory_zips(args: argparse.Namespace) -> dict[str, Any]:
    transport = _transport(args, args.capture)
    result = inventory_zips(
        transport,
        archive_names=set(args.archive) if args.archive else None,
    )
    write_json(args.output, result, write_once=args.write_once)
    return result


def command_join(args: argparse.Namespace) -> dict[str, Any]:
    result = join_catalogs(read_json(args.api_catalog), read_json(args.zip_catalog))
    write_json(args.output, result, write_once=args.write_once)
    if args.fail_on_anomaly and result["payload"]["counts"]["anomalies"]:
        raise IntegrityError("API/ZIP join contains unresolved anomalies")
    return result


def command_merge_api(args: argparse.Namespace) -> dict[str, Any]:
    result = merge_api_catalogs([read_json(path) for path in args.api_catalog])
    write_json(args.output, result, write_once=args.write_once)
    return result


def _network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--capture", default="capture-v1")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-requests", type=int, default=5_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--write-once", action="store_true", default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--quiet", action="store_true", help="write artifacts without echoing them"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--store", type=Path, default=DEFAULT_STORE)
    preflight.add_argument("--business-manifest", type=Path, default=DEFAULT_BUSINESS)
    preflight.add_argument("--business-mapping", type=Path)
    preflight.add_argument("--training-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    preflight.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    preflight.set_defaults(handler=command_preflight)

    verify = subparsers.add_parser("verify-quarantine")
    verify.add_argument("--quarantine", type=Path, required=True)
    verify.add_argument("--candidate", type=Path, required=True)
    verify.set_defaults(handler=command_verify_quarantine)

    api = subparsers.add_parser("inventory-api")
    _network_options(api)
    api.add_argument("--codex-id", action="append")
    api.add_argument("--include-elements", action="store_true")
    api.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STORE / "catalog" / "api-catalog.json",
    )
    api.set_defaults(handler=command_inventory_api)

    zips = subparsers.add_parser("inventory-zips")
    _network_options(zips)
    zips.add_argument("--archive", action="append")
    zips.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STORE / "catalog" / "zip-catalog.json",
    )
    zips.set_defaults(handler=command_inventory_zips)

    join = subparsers.add_parser("join")
    join.add_argument("--api-catalog", type=Path, required=True)
    join.add_argument("--zip-catalog", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--write-once", action="store_true", default=True)
    join.add_argument("--fail-on-anomaly", action="store_true")
    join.set_defaults(handler=command_join)

    merge_api = subparsers.add_parser("merge-api")
    merge_api.add_argument("--api-catalog", type=Path, action="append", required=True)
    merge_api.add_argument("--output", type=Path, required=True)
    merge_api.add_argument("--write-once", action="store_true", default=True)
    merge_api.set_defaults(handler=command_merge_api)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (TlachiaError, OSError, ValueError) as error:
        _emit({"status": "ERROR", "error": str(error)}, as_json=args.json)
        return 2
    if not args.quiet:
        _emit(result, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

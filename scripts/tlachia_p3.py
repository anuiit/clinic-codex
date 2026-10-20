#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.data.tlachia.catalog import CachedTransport  # noqa: E402
from codex_pipeline.data.tlachia.core import (  # noqa: E402
    TlachiaError,
    read_json,
    write_json,
)
from codex_pipeline.data.tlachia.media import (  # noqa: E402
    build_physical_mapping_review,
    build_media_plan,
    download_media_plan,
    write_physical_mapping_review_csv,
)
from codex_pipeline.data.tlachia.p3 import (  # noqa: E402
    apply_business_decisions,
    build_runtime_mapping,
    decisions_from_review_csv,
    write_review_csv,
)
from codex_pipeline.data.tlachia.simulation import (  # noqa: E402
    simulate_child_snapshot,
    simulate_standalone_snapshot,
)


DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "backend/codex_model/config.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3 mapping, bounded media acquisition, and snapshot simulation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mapping = sub.add_parser("mapping", help="build Tlachia -> runtime 286 mapping")
    mapping.add_argument("--api-catalog", type=Path, required=True)
    mapping.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    mapping.add_argument("--output", type=Path, required=True)
    mapping.add_argument("--review-csv", type=Path, required=True)

    review = sub.add_parser("apply-review", help="apply business-reviewed CSV decisions")
    review.add_argument("--mapping", type=Path, required=True)
    review.add_argument("--review-csv", type=Path, required=True)
    review.add_argument("--decisions-output", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)

    plan = sub.add_parser("plan-download", help="freeze a bounded ZIP-member plan")
    plan.add_argument("--api-catalog", type=Path, required=True)
    plan.add_argument("--join-catalog", type=Path, required=True)
    plan.add_argument("--zip-catalog", type=Path, required=True)
    plan.add_argument("--mapping", type=Path, required=True)
    plan.add_argument(
        "--purpose", choices=("review_evidence", "training_source"), required=True
    )
    plan.add_argument("--codex-id", action="append", default=[])
    plan.add_argument("--max-files", type=int, required=True)
    plan.add_argument("--max-compressed-bytes", type=int, required=True)
    plan.add_argument("--max-uncompressed-bytes", type=int, required=True)
    plan.add_argument("--max-requests", type=int, required=True)
    plan.add_argument("--max-pixel-count", type=int, required=True)
    plan.add_argument("--output", type=Path, required=True)

    physical_review = sub.add_parser(
        "physical-review", help="extract physical theme-reference mapping conflicts"
    )
    physical_review.add_argument("--plan", type=Path, required=True)
    physical_review.add_argument("--output", type=Path, required=True)
    physical_review.add_argument("--review-csv", type=Path, required=True)

    download = sub.add_parser("download", help="execute an immutable bounded plan")
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--capture-dir", type=Path, required=True)
    download.add_argument("--output-root", type=Path, required=True)
    download.add_argument("--manifest-output", type=Path, required=True)
    download.add_argument("--offline", action="store_true")
    download.add_argument("--apply", action="store_true")

    simulate = sub.add_parser("simulate-snapshot", help="simulate append-only child")
    simulate.add_argument("--parent-manifest", type=Path, required=True)
    simulate.add_argument("--mapping", type=Path, required=True)
    simulate.add_argument("--media-manifest", type=Path, required=True)
    simulate.add_argument("--business-manifest", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)

    standalone = sub.add_parser(
        "simulate-standalone",
        help="simulate a clean-room Tlachia-only snapshot",
    )
    standalone.add_argument("--mapping", type=Path, required=True)
    standalone.add_argument("--media-manifest", type=Path, required=True)
    standalone.add_argument("--business-manifest", type=Path, required=True)
    standalone.add_argument("--output", type=Path, required=True)
    standalone.add_argument(
        "--split-salt", default="clinic-codex-tlachia-standalone-v1"
    )
    standalone.add_argument("--dev-fraction", type=float, default=0.1)
    standalone.add_argument("--test-fraction", type=float, default=0.1)
    standalone.add_argument("--near-duplicate-hamming-threshold", type=int, default=4)
    standalone.add_argument("--minimum-train-rows-per-class", type=int, default=2)
    return parser


def _positive(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "mapping":
            result = build_runtime_mapping(
                read_json(args.api_catalog), runtime_config_path=args.runtime_config
            )
            write_json(args.output, result, write_once=True)
            write_review_csv(result, args.review_csv)
        elif args.command == "apply-review":
            mapping = read_json(args.mapping)
            decisions = decisions_from_review_csv(
                args.review_csv, mapping["payload_sha256"]
            )
            result = apply_business_decisions(mapping, decisions)
            write_json(args.decisions_output, decisions, write_once=True)
            write_json(args.output, result, write_once=True)
        elif args.command == "plan-download":
            result = build_media_plan(
                api_catalog=read_json(args.api_catalog),
                join_catalog=read_json(args.join_catalog),
                zip_catalog=read_json(args.zip_catalog),
                mapping=read_json(args.mapping),
                purpose=args.purpose,
                codex_ids=set(args.codex_id) or None,
                max_files=_positive(args.max_files, "max-files"),
                max_compressed_bytes=_positive(
                    args.max_compressed_bytes, "max-compressed-bytes"
                ),
                max_uncompressed_bytes=_positive(
                    args.max_uncompressed_bytes, "max-uncompressed-bytes"
                ),
                max_requests=_positive(args.max_requests, "max-requests"),
                max_pixel_count=_positive(args.max_pixel_count, "max-pixel-count"),
            )
            write_json(args.output, result, write_once=True)
        elif args.command == "download":
            if not args.apply:
                raise ValueError("download requires explicit --apply after plan review")
            plan = read_json(args.plan)
            request_budget = int(plan["payload"]["limits"]["max_requests"])
            transport = CachedTransport(
                args.capture_dir,
                offline=args.offline,
                max_requests=request_budget,
            )
            result = download_media_plan(
                plan, transport=transport, output_root=args.output_root
            )
            write_json(args.manifest_output, result, write_once=True)
        elif args.command == "physical-review":
            result = build_physical_mapping_review(read_json(args.plan))
            write_json(args.output, result, write_once=True)
            write_physical_mapping_review_csv(result, args.review_csv)
        elif args.command == "simulate-snapshot":
            result = simulate_child_snapshot(
                parent_manifest_path=args.parent_manifest,
                mapping=read_json(args.mapping),
                media_manifest=read_json(args.media_manifest),
                media_manifest_path=args.media_manifest,
                business_manifest_path=args.business_manifest,
            )
            write_json(args.output, result, write_once=True)
        elif args.command == "simulate-standalone":
            result = simulate_standalone_snapshot(
                mapping=read_json(args.mapping),
                media_manifest=read_json(args.media_manifest),
                media_manifest_path=args.media_manifest,
                business_manifest_path=args.business_manifest,
                split_salt=args.split_salt,
                dev_fraction=args.dev_fraction,
                test_fraction=args.test_fraction,
                near_duplicate_hamming_threshold=(
                    args.near_duplicate_hamming_threshold
                ),
                minimum_train_rows_per_class=args.minimum_train_rows_per_class,
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

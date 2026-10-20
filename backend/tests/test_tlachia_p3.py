from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from codex_pipeline.data.tlachia.catalog import CachedTransport, HttpResponse, merge_api_catalogs
from codex_pipeline.data.tlachia.core import IntegrityError, PolicyError, artifact
from codex_pipeline.data.tlachia.media import (
    build_media_plan,
    build_physical_mapping_review,
    download_media_plan,
)
from codex_pipeline.data.tlachia.p3 import (
    apply_business_decisions,
    build_runtime_mapping,
)
from codex_pipeline.data.tlachia.simulation import (
    _assign_splits_atomically,
    simulate_child_snapshot,
    simulate_standalone_snapshot,
)


def _runtime_config(tmp_path: Path) -> Path:
    names = ["atl", "xihuitl_1", "xihuitl_2"] + [f"class-{i:03d}" for i in range(283)]
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"num_classes": 286, "class_names": names}), encoding="utf-8"
    )
    return path


def _element(key: str, label: str, theme: str, element_id: str) -> dict:
    return {
        "key": key,
        "codex_id": "42",
        "codex_code": "391",
        "glyph_cote": "391_A_001a",
        "planche_cote": "391",
        "source_group": "tlachia:codex:42:planche:391",
        "element_id": element_id,
        "element": label,
        "theme": theme,
        "raw": {"valeur": "atl" if label == "?" else label},
    }


def _api_catalog() -> dict:
    nodes = [
        _element("node-atl-1", "atl", "04.05.01", "1"),
        _element("node-atl-2", "atl", "04.05.01", "2"),
        _element("node-xihuitl", "xihuitl_1", "04.04.03", "3"),
        _element("node-unknown", "?", "05.99.99", "4"),
    ]
    return artifact(
        "tlachia-api-catalog.v1",
        {
            "include_elements": True,
            "nodes": {"element": nodes},
            "relations": [],
            "counts": {"element": len(nodes)},
            "selected_codex_ids": ["42"],
            "endpoint": "fixture",
            "response_body_sha256": [],
            "anomalies": [],
        },
    )


def test_mapping_separates_exact_suffix_and_manual_hint(tmp_path: Path) -> None:
    result = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    rows = {row["tlachia_element_label"]: row for row in result["payload"]["mappings"]}
    assert len(result["payload"]["runtime_classes"]) == 286
    assert rows["atl"]["status"] == "accepted_exact"
    assert rows["atl"]["runtime_class_label"] == 0
    assert rows["atl"]["legacy_class_index"] == 1
    assert rows["xihuitl_1"]["status"] == "needs_business_review"
    assert rows["?"]["match_method"] == "raw_field_hint"
    assert rows["?"]["candidate_runtime_class_name"] == "atl"


def test_business_decisions_are_hash_bound(tmp_path: Path) -> None:
    mapping = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    suffix = next(
        row for row in mapping["payload"]["mappings"] if row["tlachia_element_label"] == "xihuitl_1"
    )
    decisions = artifact(
        "tlachia-mapping-decisions.v1",
        {
            "mapping_payload_sha256": mapping["payload_sha256"],
            "reviewer": "metier",
            "reviewed_at": "2026-08-13T00:00:00Z",
            "decisions": [
                {
                    "mapping_id": suffix["mapping_id"],
                    "status": "approved",
                    "runtime_class_name": "xihuitl_1",
                }
            ],
        },
    )
    reviewed = apply_business_decisions(mapping, decisions)
    row = next(
        row for row in reviewed["payload"]["mappings"] if row["mapping_id"] == suffix["mapping_id"]
    )
    assert row["status"] == "approved_manual"
    aggregate = next(
        item
        for item in reviewed["payload"]["runtime_classes"]
        if item["runtime_class_name"] == "xihuitl_1"
    )
    assert suffix["mapping_id"] in aggregate["mapping_ids"]
    tampered = json.loads(json.dumps(decisions))
    tampered["payload"]["mapping_payload_sha256"] = "0" * 64
    tampered["payload_sha256"] = __import__(
        "codex_pipeline.data.tlachia.core", fromlist=["semantic_hash"]
    ).semantic_hash(tampered["payload"])
    with pytest.raises(IntegrityError, match="different mapping"):
        apply_business_decisions(mapping, tampered)


def test_merge_api_catalogs_requires_disjoint_complete_batches() -> None:
    first = _api_catalog()
    first["payload"]["selected_codex_ids"] = ["42"]
    first["payload_sha256"] = __import__(
        "codex_pipeline.data.tlachia.core", fromlist=["semantic_hash"]
    ).semantic_hash(first["payload"])
    second = artifact(
        "tlachia-api-catalog.v1",
        {
            "include_elements": True,
            "nodes": {"codex": [{"key": "cen:codex:43"}], "planche": [], "zone": [], "glyph": [], "element": []},
            "relations": [],
            "counts": {"codex": 1},
            "selected_codex_ids": ["43"],
            "endpoint": "fixture",
            "response_body_sha256": ["b" * 64],
            "anomalies": [],
        },
    )
    merged = merge_api_catalogs([first, second])
    assert merged["payload"]["selected_codex_ids"] == ["42", "43"]
    assert merged["payload"]["counts"]["source_catalogs"] == 2
    with pytest.raises(IntegrityError, match="overlap"):
        merge_api_catalogs([first, first])


def _zip_fixture(tmp_path: Path) -> tuple[bytes, dict, str]:
    image_path = tmp_path / "image.bmp"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image_path)
    zip_path = tmp_path / "t_391.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(image_path, "04_05_01.bmp")
    zip_bytes = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as archive:
        info = archive.getinfo("04_05_01.bmp")
        entry = {
            "path": info.filename,
            "normalized_path": info.filename.casefold(),
            "raw_name_sha256": __import__("hashlib").sha256(info.filename.encode()).hexdigest(),
            "flag_bits": info.flag_bits,
            "compression": info.compress_type,
            "crc32": f"{info.CRC:08x}",
            "compressed_size": info.compress_size,
            "uncompressed_size": info.file_size,
            "local_header_offset": info.header_offset,
        }
    return zip_bytes, entry, '"fixture-etag"'


def _plan_fixture(tmp_path: Path) -> tuple[dict, bytes, str]:
    api = _api_catalog()
    mapping = build_runtime_mapping(api, runtime_config_path=_runtime_config(tmp_path))
    zip_bytes, entry, etag = _zip_fixture(tmp_path)
    zip_catalog = artifact(
        "tlachia-zip-catalog.v1",
        {
            "archives": [
                {
                    "archive_name": "t_391.zip",
                    "url": "https://sup-infor.com/Marc/Images/t_391.zip",
                    "status": "available",
                    "etag": etag,
                    "content_length": len(zip_bytes),
                    "central_directory_sha256": "f" * 64,
                    "entries": [entry],
                }
            ],
            "counts": {"available": 1, "entries": 1},
        },
    )
    join = artifact(
        "tlachia-api-zip-join.v1",
        {
            "api_payload_sha256": api["payload_sha256"],
            "zip_payload_sha256": zip_catalog["payload_sha256"],
            "relations": [
                {
                    "node": node,
                    "role": "element_theme_reference",
                    "archive": "t_391.zip",
                    "entry_path": "04_05_01.bmp",
                    "entry_crc32": entry["crc32"],
                }
                for node in ("node-atl-1", "node-atl-2")
            ],
        },
    )
    plan = build_media_plan(
        api_catalog=api,
        join_catalog=join,
        zip_catalog=zip_catalog,
        mapping=mapping,
        purpose="training_source",
        codex_ids={"42"},
        max_files=1,
        max_compressed_bytes=10_000,
        max_uncompressed_bytes=10_000,
        max_requests=2,
        max_pixel_count=100,
    )
    return plan, zip_bytes, etag


def _physical_unit_plan_fixture(tmp_path: Path, mapping: dict) -> dict:
    api = _api_catalog()
    zip_path = tmp_path / "physical-units.zip"
    for index, color in enumerate(((10, 20, 30), (30, 20, 10)), start=1):
        Image.new("RGB", (3, 2), color).save(tmp_path / f"image-{index}.bmp")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(tmp_path / "image-1.bmp", "04_05_01.bmp")
        archive.write(tmp_path / "image-2.bmp", "04_05_02.bmp")
    with zipfile.ZipFile(zip_path) as archive:
        entries = []
        for name in ("04_05_01.bmp", "04_05_02.bmp"):
            info = archive.getinfo(name)
            entries.append(
                {
                    "path": info.filename,
                    "normalized_path": info.filename.casefold(),
                    "raw_name_sha256": __import__("hashlib")
                    .sha256(info.filename.encode())
                    .hexdigest(),
                    "flag_bits": info.flag_bits,
                    "compression": info.compress_type,
                    "crc32": f"{info.CRC:08x}",
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "local_header_offset": info.header_offset,
                }
            )
    zip_bytes = zip_path.read_bytes()
    zip_catalog = artifact(
        "tlachia-zip-catalog.v1",
        {
            "archives": [
                {
                    "archive_name": "t_391.zip",
                    "url": "https://sup-infor.com/Marc/Images/t_391.zip",
                    "status": "available",
                    "etag": '"fixture-etag"',
                    "content_length": len(zip_bytes),
                    "central_directory_sha256": "f" * 64,
                    "entries": entries,
                }
            ]
        },
    )
    crc_by_path = {entry["path"]: entry["crc32"] for entry in entries}
    join = artifact(
        "tlachia-api-zip-join.v1",
        {
            "api_payload_sha256": api["payload_sha256"],
            "zip_payload_sha256": zip_catalog["payload_sha256"],
            "relations": [
                {
                    "node": node,
                    "role": "element_theme_reference",
                    "archive": "t_391.zip",
                    "entry_path": path,
                    "entry_crc32": crc_by_path[path],
                }
                for node, path in (
                    ("node-atl-1", "04_05_01.bmp"),
                    ("node-atl-2", "04_05_02.bmp"),
                    ("node-xihuitl", "04_05_02.bmp"),
                )
            ],
        },
    )
    return build_media_plan(
        api_catalog=api,
        join_catalog=join,
        zip_catalog=zip_catalog,
        mapping=mapping,
        purpose="training_source",
        max_files=2,
        max_compressed_bytes=20_000,
        max_uncompressed_bytes=20_000,
        max_requests=4,
        max_pixel_count=100,
    )


def test_plan_deduplicates_occurrences_and_enforces_budget(tmp_path: Path) -> None:
    plan, _, _ = _plan_fixture(tmp_path)
    assert plan["payload"]["counts"]["files"] == 1
    assert plan["payload"]["counts"]["occurrences"] == 2

    api = _api_catalog()
    mapping = build_runtime_mapping(api, runtime_config_path=_runtime_config(tmp_path))
    _, entry, etag = _zip_fixture(tmp_path)
    zips = artifact(
        "tlachia-zip-catalog.v1",
        {"archives": [{"archive_name": "t_391.zip", "url": "https://sup-infor.com/Marc/Images/t_391.zip", "status": "available", "etag": etag, "content_length": 1_000, "central_directory_sha256": "f" * 64, "entries": [entry]}]},
    )
    join = artifact(
        "tlachia-api-zip-join.v1",
        {"api_payload_sha256": api["payload_sha256"], "zip_payload_sha256": zips["payload_sha256"], "relations": [{"node": "node-atl-1", "role": "element_theme_reference", "archive": "t_391.zip", "entry_path": "04_05_01.bmp", "entry_crc32": entry["crc32"]}]},
    )
    with pytest.raises(PolicyError, match="compressed_bytes"):
        build_media_plan(
            api_catalog=api, join_catalog=join, zip_catalog=zips, mapping=mapping,
            purpose="training_source", max_files=1, max_compressed_bytes=1,
            max_uncompressed_bytes=10_000, max_requests=2, max_pixel_count=100,
        )


def test_training_plan_quarantines_accepted_plus_pending_physical_unit(
    tmp_path: Path,
) -> None:
    mapping = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    plan = _physical_unit_plan_fixture(tmp_path, mapping)
    assert plan["payload"]["counts"]["files"] == 1
    assert plan["payload"]["counts"]["physical_quarantine"] == 1
    quarantine = plan["payload"]["physical_quarantine"][0]
    assert quarantine["entry_path"] == "04_05_02.bmp"
    assert quarantine["reason"] == "pending_mapping_review_for_physical_reference"


def test_training_plan_keeps_accepted_unit_after_pending_alias_is_rejected(
    tmp_path: Path,
) -> None:
    mapping = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    pending = next(
        row
        for row in mapping["payload"]["mappings"]
        if row["tlachia_element_label"] == "xihuitl_1"
    )
    decisions = artifact(
        "tlachia-mapping-decisions.v1",
        {
            "mapping_payload_sha256": mapping["payload_sha256"],
            "reviewer": "metier",
            "reviewed_at": "2026-08-13T00:00:00Z",
            "decisions": [
                {
                    "mapping_id": pending["mapping_id"],
                    "status": "rejected",
                    "runtime_class_name": None,
                }
            ],
        },
    )
    reviewed = apply_business_decisions(mapping, decisions)
    plan = _physical_unit_plan_fixture(tmp_path, reviewed)
    assert plan["payload"]["counts"]["files"] == 2
    assert plan["payload"]["counts"]["physical_quarantine"] == 0


def test_physical_review_extracts_missing_class_reference(tmp_path: Path) -> None:
    plan, _, _ = _plan_fixture(tmp_path)
    drifted = json.loads(json.dumps(plan))
    drifted["payload"]["entries"][0]["candidate_runtime_classes"] = []
    drifted["payload_sha256"] = __import__(
        "codex_pipeline.data.tlachia.core", fromlist=["semantic_hash"]
    ).semantic_hash(drifted["payload"])
    review = build_physical_mapping_review(drifted)
    assert review["payload"]["counts"]["missing_class_references"] == 1
    assert review["payload"]["counts"]["multiple_class_references"] == 0
    assert review["payload"]["conflicts"][0]["reason"] == "no_approved_runtime_class"


def test_physical_review_quarantines_pending_alias_on_accepted_image(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan_fixture(tmp_path)
    drifted = json.loads(json.dumps(plan))
    drifted["payload"]["entries"][0]["mapping_statuses"].append(
        "needs_business_review"
    )
    drifted["payload_sha256"] = __import__(
        "codex_pipeline.data.tlachia.core", fromlist=["semantic_hash"]
    ).semantic_hash(drifted["payload"])
    review = build_physical_mapping_review(drifted)
    assert review["payload"]["counts"]["pending_review_references"] == 1
    assert review["payload"]["conflicts"][0]["reason"] == (
        "pending_mapping_review_for_physical_reference"
    )


def test_bounded_download_verifies_range_crc_and_pixel_hash(tmp_path: Path) -> None:
    plan, zip_bytes, etag = _plan_fixture(tmp_path)

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        assert method == "GET"
        assert headers["if-match"] == etag
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["range"])
        assert match is not None
        start, end = map(int, match.groups())
        payload = zip_bytes[start : end + 1]
        return HttpResponse(
            206,
            {
                "etag": etag,
                "content-range": f"bytes {start}-{end}/{len(zip_bytes)}",
                "content-length": str(len(payload)),
            },
            payload,
            url,
        )

    transport = CachedTransport(
        tmp_path / "capture", max_requests=2, min_interval_seconds=0, sender=sender
    )
    manifest = download_media_plan(plan, transport=transport, output_root=tmp_path / "out")
    assert manifest["payload"]["counts"]["downloaded_files"] == 1
    assert manifest["payload"]["counts"]["http_range_requests"] == 2
    assert manifest["payload"]["counts"]["capture_replay_invariant"] is True
    assert manifest["payload"]["output_root"] == (tmp_path / "out").resolve().as_posix()
    row = manifest["payload"]["files"][0]
    assert row["entry_crc32"] == plan["payload"]["entries"][0]["entry"]["crc32"]
    assert row["image"]["width"] == 3
    assert (tmp_path / "out" / row["output_path"]).is_file()


def test_bounded_download_rejects_content_range_total_drift(tmp_path: Path) -> None:
    plan, zip_bytes, etag = _plan_fixture(tmp_path)

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        start, end = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["range"]).groups()
        )
        payload = zip_bytes[start : end + 1]
        return HttpResponse(
            206,
            {
                "etag": etag,
                "content-range": f"bytes {start}-{end}/{len(zip_bytes) + 1}",
                "content-length": str(len(payload)),
            },
            payload,
            url,
        )

    with pytest.raises(IntegrityError, match="unexpected Content-Range"):
        download_media_plan(
            plan,
            transport=CachedTransport(
                tmp_path / "capture-range-drift",
                max_requests=2,
                min_interval_seconds=0,
                sender=sender,
            ),
            output_root=tmp_path / "out-range-drift",
        )


def test_bounded_download_rejects_local_header_size_drift(tmp_path: Path) -> None:
    plan, zip_bytes, etag = _plan_fixture(tmp_path)
    drifted = bytearray(zip_bytes)
    local_offset = int(plan["payload"]["entries"][0]["entry"]["local_header_offset"])
    compressed_size_offset = local_offset + 18
    original_size = int.from_bytes(
        drifted[compressed_size_offset : compressed_size_offset + 4], "little"
    )
    drifted[compressed_size_offset : compressed_size_offset + 4] = (
        original_size + 1
    ).to_bytes(4, "little")

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["range"])
        assert match is not None
        start, end = map(int, match.groups())
        payload = bytes(drifted[start : end + 1])
        return HttpResponse(
            206,
            {
                "etag": etag,
                "content-range": f"bytes {start}-{end}/{len(drifted)}",
                "content-length": str(len(payload)),
            },
            payload,
            url,
        )

    with pytest.raises(IntegrityError, match="CRC/sizes differ"):
        download_media_plan(
            plan,
            transport=CachedTransport(
                tmp_path / "capture-drift",
                max_requests=2,
                min_interval_seconds=0,
                sender=sender,
            ),
            output_root=tmp_path / "out-drift",
        )


def test_simulation_quarantines_near_duplicate_and_blocks_training(tmp_path: Path) -> None:
    plan, zip_bytes, etag = _plan_fixture(tmp_path)

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["range"]).groups())
        return HttpResponse(
            206,
            {
                "etag": etag,
                "content-range": f"bytes {start}-{end}/{len(zip_bytes)}",
                "content-length": str(end - start + 1),
            },
            zip_bytes[start : end + 1],
            url,
        )

    media_root = tmp_path / "media"
    media = download_media_plan(
        plan,
        transport=CachedTransport(
            tmp_path / "capture-sim", max_requests=2, min_interval_seconds=0, sender=sender
        ),
        output_root=media_root,
    )
    media_path = media_root / "manifest.json"
    media_path.write_text(json.dumps(media), encoding="utf-8")

    parent_root = tmp_path / "parent"
    parent_image = parent_root / "Elements" / "parent.bmp"
    parent_image.parent.mkdir(parents=True)
    Image.new("RGB", (3, 2), (0, 0, 0)).save(parent_image)
    import hashlib

    digest = hashlib.sha256()
    digest.update(b"RGB:3x2:")
    digest.update(Image.new("RGB", (3, 2), (0, 0, 0)).tobytes())
    parent = {
        "schema_version": "training-snapshot.v2",
        "snapshot_id": "parent",
        "content_sha256": "a" * 64,
        "ready_for_training": True,
        "row_count": 1,
        "rows": [
            {
                "row_id": "parent-row",
                "class_name": "atl",
                "class_label": 0,
                "source_pixel_sha256": digest.hexdigest(),
                "output_path": "Elements/parent.bmp",
            }
        ],
        "class_order": ["atl"],
        "split_counts": {"train": 1, "dev": 0, "locked_test": 0},
        "policy": {
            "split_salt": "fixture",
            "requested_dev_fraction": 0.1,
            "requested_test_fraction": 0.1,
        },
    }
    parent_path = parent_root / "snapshot_manifest.json"
    mapping = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    parent["class_order"] = [
        row["runtime_class_name"] for row in mapping["payload"]["runtime_classes"]
    ]
    parent["class_order_sha256"] = mapping["payload"]["class_order_sha256"]
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    business_root = tmp_path / "business"
    business_crop = business_root / "Elements" / "crop.bmp"
    business_page = business_root / "page.bmp"
    business_crop.parent.mkdir(parents=True)
    Image.new("RGB", (3, 2), (255, 255, 255)).save(business_crop)
    Image.new("RGB", (4, 3), (255, 255, 255)).save(business_page)
    business_path = business_root / "test_manifest.json"
    business_semantic = {
        "schema_version": "business-test-set.v1",
        "rows": [
                    {
                        "row_id": "business-row",
                        "class_name": "atl",
                        "output_path": "Elements/crop.bmp",
                        "page_path": str(business_page),
                        "crop_pixel_sha256": "c" * 64,
                        "page_pixel_sha256": "d" * 64,
                    }
                ],
    }
    business_content_sha = hashlib.sha256(
        json.dumps(
            business_semantic, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    business_document = {
        **business_semantic,
        "test_set_id": "business",
        "content_sha256": business_content_sha,
    }
    business_path.write_text(
        json.dumps(business_document),
        encoding="utf-8",
    )
    result = simulate_child_snapshot(
        parent_manifest_path=parent_path,
        mapping=mapping,
        media_manifest=media,
        media_manifest_path=media_path,
        business_manifest_path=business_path,
        near_duplicate_hamming_threshold=4,
    )
    assert result["payload"]["counts"]["near_duplicate_matches"] == 1
    assert result["payload"]["counts"]["net_additions"] == 0
    assert any(
        row["reason"] == "business_final_perceptual_near_duplicate"
        for row in result["payload"]["quarantine"]
    )
    assert any(
        match["kind"] == "business_final"
        for evidence in result["payload"]["near_duplicate_evidence"]
        for match in evidence["matches"]
    )
    assert not result["payload"]["gates"]["training_allowed"]
    assert "business_mapping_review_incomplete" in result["payload"]["gates"]["training_blockers"]

    reordered = json.loads(json.dumps(mapping))
    reordered["payload"]["runtime_classes"][0], reordered["payload"]["runtime_classes"][1] = (
        reordered["payload"]["runtime_classes"][1],
        reordered["payload"]["runtime_classes"][0],
    )
    reordered["payload"]["class_order_sha256"] = "f" * 64
    reordered["payload_sha256"] = __import__(
        "codex_pipeline.data.tlachia.core", fromlist=["semantic_hash"]
    ).semantic_hash(reordered["payload"])
    with pytest.raises(IntegrityError, match="taxonomy order"):
        simulate_child_snapshot(
            parent_manifest_path=parent_path,
            mapping=reordered,
            media_manifest=media,
            media_manifest_path=media_path,
            business_manifest_path=business_path,
        )


def test_standalone_simulation_does_not_require_or_deduplicate_against_parent(
    tmp_path: Path,
) -> None:
    plan, zip_bytes, etag = _plan_fixture(tmp_path)

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        start, end = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["range"]).groups()
        )
        payload = zip_bytes[start : end + 1]
        return HttpResponse(
            206,
            {
                "etag": etag,
                "content-range": f"bytes {start}-{end}/{len(zip_bytes)}",
                "content-length": str(len(payload)),
            },
            payload,
            url,
        )

    media_root = tmp_path / "standalone-media"
    media = download_media_plan(
        plan,
        transport=CachedTransport(
            tmp_path / "standalone-capture",
            max_requests=2,
            min_interval_seconds=0,
            sender=sender,
        ),
        output_root=media_root,
    )
    media_path = media_root / "manifest.json"
    media_path.write_text(json.dumps(media), encoding="utf-8")
    business_root = tmp_path / "standalone-business"
    business_crop = business_root / "crop.bmp"
    business_page = business_root / "page.bmp"
    business_root.mkdir()
    crop_image = Image.new("RGB", (9, 8))
    page_image = Image.new("RGB", (9, 8))
    for y in range(8):
        for x in range(9):
            crop_image.putpixel((x, y), ((8 - x) * 25, y * 10, 20))
            page_image.putpixel((x, y), ((8 - x) * 20, y * 12, 40))
    crop_image.save(business_crop)
    page_image.save(business_page)
    import hashlib

    def pixel_sha(image: Image.Image) -> str:
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.convert("RGB").tobytes())
        return digest.hexdigest()
    business_path = business_root / "test_manifest.json"
    business_semantic = {
        "schema_version": "business-test-set.v1",
        "rows": [
            {
                "row_id": "business-row",
                "class_name": "class-000",
                "output_path": "crop.bmp",
                "page_path": str(business_page),
                "crop_pixel_sha256": pixel_sha(crop_image),
                "page_pixel_sha256": pixel_sha(page_image),
                "output_sha256": hashlib.sha256(business_crop.read_bytes()).hexdigest(),
                "page_sha256": hashlib.sha256(business_page.read_bytes()).hexdigest(),
            }
        ],
    }
    business_content_sha = hashlib.sha256(
        json.dumps(
            business_semantic, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    business_document = {
        **business_semantic,
        "test_set_id": "business",
        "content_sha256": business_content_sha,
    }
    business_path.write_text(json.dumps(business_document), encoding="utf-8")
    mapping = build_runtime_mapping(
        _api_catalog(), runtime_config_path=_runtime_config(tmp_path)
    )
    with pytest.raises(IntegrityError, match="frozen authority"):
        simulate_standalone_snapshot(
            mapping=mapping,
            media_manifest=media,
            media_manifest_path=media_path,
            business_manifest_path=business_path,
            near_duplicate_hamming_threshold=0,
        )
    result = simulate_standalone_snapshot(
        mapping=mapping,
        media_manifest=media,
        media_manifest_path=media_path,
        business_manifest_path=business_path,
        near_duplicate_hamming_threshold=0,
        expected_business_test_id="business",
        expected_business_content_sha256=business_content_sha,
    )
    assert result["payload"]["mode"] == "standalone_clean_room_rebuild"
    assert result["payload"]["counts"]["retained_rows"] == 1
    assert result["payload"]["counts"]["covered_runtime_classes"] == 1
    assert result["payload"]["counts"]["missing_runtime_classes"] == 285
    assert "runtime_class_coverage_incomplete" in result["payload"]["gates"][
        "training_blockers"
    ]
    assert result["payload"]["policy"]["prior_training_dataset_consulted"] is False

    mutated_path = business_root / "mutated-test-manifest.json"
    mutated = json.loads(json.dumps(business_document))
    mutated["rows"][0]["class_name"] = "mutated"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(IntegrityError, match="semantic content hash"):
        simulate_standalone_snapshot(
            mapping=mapping,
            media_manifest=media,
            media_manifest_path=media_path,
            business_manifest_path=mutated_path,
            near_duplicate_hamming_threshold=0,
            expected_business_test_id="business",
            expected_business_content_sha256=business_content_sha,
        )


def test_standalone_train_coverage_moves_whole_source_group() -> None:
    rows = [
        {"row_id": "a", "class_name": "atl", "source_group": "near-component"},
        {"row_id": "b", "class_name": "atl", "source_group": "near-component"},
    ]
    conflicts = _assign_splits_atomically(
        rows,
        split_salt="atomic-test",
        dev_fraction=0,
        test_fraction=0.99,
        minimum_train_rows_per_class=2,
    )
    assert conflicts == []
    assert {row["proposed_split"] for row in rows} == {"train"}


def test_standalone_rejects_non_finite_split_fraction() -> None:
    with pytest.raises(PolicyError, match="invalid standalone split policy"):
        simulate_standalone_snapshot(
            mapping={},
            media_manifest={},
            media_manifest_path=Path("unused"),
            business_manifest_path=Path("unused"),
            dev_fraction=float("nan"),
        )

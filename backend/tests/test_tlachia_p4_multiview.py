from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (REPO_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codex_pipeline.data.tlachia.core import IntegrityError, PolicyError, artifact
from codex_pipeline.data.tlachia.media import MEDIA_MANIFEST_SCHEMA
from scripts.tlachia_p4_multiview import (
    _parser,
    build_multiview_plan,
    verify_multiview_download,
)


def _fixture(groups: int = 20, classes_per_group: int = 5):
    nodes = {"codex": [], "element": [], "glyph": [], "planche": [], "zone": []}
    api_relations = []
    join_relations = []
    mappings = []
    entries = []
    for group_index in range(groups):
        codex_id = "1"
        cote = f"P{group_index:02d}"
        source_group = f"tlachia:codex:{codex_id}:planche:{cote}"
        planche_key = f"cen:codex:{codex_id}:planche:{cote}"
        nodes["planche"].append(
            {
                "key": planche_key,
                "codex_id": codex_id,
                "cote": cote,
                "source_group": source_group,
                "archive_name": "fixture.zip",
            }
        )
        entry_path = f"{cote}.png"
        crc = f"{group_index + 1:08x}"
        join_relations.append(
            {
                "role": "planche",
                "node": planche_key,
                "archive": "fixture.zip",
                "entry_path": entry_path,
                "entry_crc32": crc,
            }
        )
        entries.append(
            {
                "path": entry_path,
                "crc32": crc,
                "compressed_size": 10,
                "uncompressed_size": 11,
                "compression": 8,
                "flag_bits": 0,
                "local_header_offset": group_index * 100,
                "raw_name_sha256": "a" * 64,
            }
        )
        for class_index in range(classes_per_group):
            ordinal = group_index * classes_per_group + class_index
            element_key = f"cen:element:{ordinal}"
            nodes["element"].append({"key": element_key})
            api_relations.append(
                {"type": "contains", "parent": planche_key, "child": element_key}
            )
            mappings.append(
                {
                    "mapping_id": f"mapping-{ordinal}",
                    "node_keys": [element_key],
                    "status": "accepted_exact",
                    "candidate_runtime_class_name": f"class-{ordinal:03d}",
                }
            )
    api = artifact(
        "tlachia-api-catalog.v1", {"nodes": nodes, "relations": api_relations}
    )
    zip_catalog = artifact(
        "tlachia-zip-catalog.v1",
        {
            "archives": [
                {
                    "archive_name": "fixture.zip",
                    "status": "available",
                    "url": "https://sup-infor.com/Marc/Images/fixture.zip",
                    "etag": '"fixture"',
                    "content_length": 10_000,
                    "central_directory_sha256": "b" * 64,
                    "entries": entries,
                }
            ]
        },
    )
    join = artifact(
        "tlachia-api-zip-join.v1",
        {
            "api_payload_sha256": api["payload_sha256"],
            "zip_payload_sha256": zip_catalog["payload_sha256"],
            "relations": join_relations,
        },
    )
    mapping = artifact(
        "tlachia-runtime-mapping.v1",
        {
            "api_catalog_payload_sha256": api["payload_sha256"],
            "mappings": mappings,
        },
    )
    return api, join, zip_catalog, mapping


def _plan(**overrides):
    api, join, zip_catalog, mapping = _fixture()
    kwargs = {
        "api_catalog": api,
        "join_catalog": join,
        "zip_catalog": zip_catalog,
        "mapping": mapping,
        "max_files": 30,
        "max_groups": 20,
        "max_compressed_bytes": 1_000,
        "max_uncompressed_bytes": 1_000,
        "max_requests": 60,
        "max_pixel_count": 1_000_000,
    }
    kwargs.update(overrides)
    return build_multiview_plan(**kwargs)


def test_plan_exact_join_crc_and_deterministic_selection() -> None:
    first = _plan()
    second = _plan()
    assert first == second
    assert first["payload"]["purpose"] == "multiview_self_supervised"
    assert first["payload"]["counts"]["source_groups"] == 20
    assert first["payload"]["counts"]["runtime_classes_evidenced"] == 100
    assert first["payload"]["counts"]["files"] == 20
    assert first["payload"]["policy"]["whole_zip_fallback"] is False


def test_plan_rejects_join_crc_drift() -> None:
    api, join, zip_catalog, mapping = _fixture()
    join = artifact(
        "tlachia-api-zip-join.v1",
        {**join["payload"], "relations": [dict(row) for row in join["payload"]["relations"]]},
    )
    join["payload"]["relations"][0]["entry_crc32"] = "ffffffff"
    join = artifact("tlachia-api-zip-join.v1", join["payload"])
    with pytest.raises(IntegrityError, match="CRC"):
        build_multiview_plan(
            api_catalog=api,
            join_catalog=join,
            zip_catalog=zip_catalog,
            mapping=mapping,
            max_files=30,
            max_groups=20,
            max_compressed_bytes=1_000,
            max_uncompressed_bytes=1_000,
            max_requests=60,
            max_pixel_count=1_000_000,
        )


def test_plan_fails_closed_when_file_budget_cannot_meet_group_gate() -> None:
    with pytest.raises(PolicyError, match="source groups"):
        _plan(max_files=19)


def test_plan_enforces_per_group_role_cap() -> None:
    plan = _plan(max_planches_per_group=1)
    counts = {}
    for row in plan["payload"]["entries"]:
        counts[row["source_group"]] = counts.get(row["source_group"], 0) + 1
    assert set(counts.values()) == {1}


def test_plan_never_turns_descendant_evidence_into_single_label_truth() -> None:
    plan = _plan()
    assert all(row["supervision_kind"] == "unlabeled_ssl" for row in plan["payload"]["entries"])
    assert all(row["single_label_class"] is None for row in plan["payload"]["entries"])
    assert all(row["candidate_runtime_classes"] for row in plan["payload"]["entries"])
    assert plan["payload"]["gates"]["promotion_allowed"] is False


def test_verify_rejects_pixel_duplicate_across_source_groups(tmp_path: Path) -> None:
    plan = _plan()
    files = []
    image = Image.new("RGB", (3, 2), (7, 8, 9))
    pixel = hashlib.sha256(b"RGB:3x2:" + image.tobytes()).hexdigest()
    for index, item in enumerate(plan["payload"]["entries"]):
        relative = Path("files") / f"{index}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        files.append(
            {
                "archive_name": item["archive_name"],
                "entry_path": item["entry_path"],
                "entry_crc32": item["entry"]["crc32"],
                "compressed_size": item["entry"]["compressed_size"],
                "uncompressed_size": item["entry"]["uncompressed_size"],
                "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_path": relative.as_posix(),
                "image": {"pixel_sha256": pixel, "pixel_count": 6},
            }
        )
    manifest = artifact(
        MEDIA_MANIFEST_SCHEMA,
        {
            "plan_payload_sha256": plan["payload_sha256"],
            "purpose": "multiview_self_supervised",
            "output_root": tmp_path.resolve().as_posix(),
            "counts": {
                "downloaded_files": len(files),
                "downloaded_bytes": 11 * len(files),
                "decoded_pixels": 6 * len(files),
                "http_range_requests": 2 * len(files),
            },
            "gates": {
                "download_complete": True,
                "training_usable_without_snapshot_quarantine": True,
            },
            "files": files,
        },
    )
    with pytest.raises(PolicyError, match="pixel-identical"):
        verify_multiview_download(plan, manifest, output_root=tmp_path)


def test_cli_requires_explicit_download_paths_and_apply_flag_is_parseable() -> None:
    parsed = _parser().parse_args(
        [
            "download",
            "--plan",
            "plan.json",
            "--capture-dir",
            "capture",
            "--output-root",
            "media",
            "--manifest-output",
            "manifest.json",
            "--apply",
        ]
    )
    assert parsed.command == "download"
    assert parsed.apply is True

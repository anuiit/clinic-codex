from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from backend.codex_pipeline.data.tlachia.catalog import (
    API_URL,
    ZIP_LISTING_URL,
    CachedTransport,
    HttpResponse,
    _find_eocd,
    inventory_api,
    inventory_zips,
    join_catalogs,
)
from backend.codex_pipeline.data.tlachia.core import (
    IntegrityError,
    PolicyError,
    artifact,
    canonical_json_bytes,
    normalise_theme,
    verify_artifact,
    write_json,
)
from backend.codex_pipeline.data.tlachia.quarantine import (
    build_business_quarantine,
    verify_candidate_export,
)


def _response(body: object, url: str = API_URL) -> HttpResponse:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return HttpResponse(200, {"content-type": "application/json"}, raw, url)


def _api_sender(log: list[dict[str, object]]):
    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        del headers, timeout, max_bytes
        assert method == "POST"
        payload = json.loads(body)
        log.append(payload)
        key = (payload["type"], str(payload["id"]), payload["ext"])
        fixtures = {
            ("codex", "0", ""): {
                "codex": [
                    {
                        "id": "42",
                        "code": "391",
                        "dir": "/Home/2/CEN/Codex/t_391/",
                        "title": "sample",
                    }
                ]
            },
            ("planche", "42", ""): {
                "planches": [{"codexId": "42", "id": "1146", "cote": "391"}]
            },
            ("zone", "42", "391"): {
                "zones": [{"codexId": "42", "id": "1706", "cote": "391_A"}]
            },
            ("glyphe", "42", "391_A"): {
                "glyphes": [
                    {
                        "codexId": "42",
                        "id": "19412",
                        "cote": "391_A_001a",
                        "lecture": "tepetl",
                    }
                ],
                "personnes": [],
            },
            ("element", "42", "391_A_001a"): {
                "elements": [
                    {
                        "codexId": "42",
                        "id": "32783",
                        "cote": "391_A_001a",
                        "theme": "04.03.08",
                        "element": "tepetl",
                    }
                ]
            },
        }
        return _response(fixtures[key], url)

    return sender


def _zip_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("391.jpg", b"planche")
        archive.writestr("391_A.jpg", b"zone")
        archive.writestr("391_A_001a.jpg", b"jpg")
        archive.writestr("04_03_08.bmp", b"bmp")
        archive.writestr("391_A_00.INI", b"legacy")
    return output.getvalue()


def _zip_sender(zip_payload: bytes, log: list[tuple[str, str]]):
    listing = b'<html><a href="t_391.zip">t_391.zip</a></html>'
    archive_url = "https://sup-infor.com/Marc/Images/t_391.zip"

    def sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        del body, timeout, max_bytes
        log.append((method, url))
        if url == ZIP_LISTING_URL:
            return HttpResponse(200, {"content-type": "text/html"}, listing, url)
        assert url == archive_url
        if method == "HEAD":
            return HttpResponse(
                200,
                {
                    "content-length": str(len(zip_payload)),
                    "accept-ranges": "bytes",
                    "etag": '"sample"',
                },
                b"",
                url,
            )
        assert method == "GET"
        value = headers["range"].removeprefix("bytes=")
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        payload = zip_payload[start : end + 1]
        return HttpResponse(
            206,
            {
                "content-range": f"bytes {start}-{start + len(payload) - 1}/{len(zip_payload)}"
            },
            payload,
            url,
        )

    return sender


def test_canonical_artifact_is_nfc_and_rejects_float() -> None:
    left = artifact("sample.v1", {"label": "e\u0301"})
    right = artifact("sample.v1", {"label": "é"})
    assert left == right
    verify_artifact(left, "sample.v1")
    with pytest.raises(TypeError, match="floats are forbidden"):
        canonical_json_bytes({"score": 0.1})


def test_transport_cache_replay_uses_zero_network(tmp_path: Path) -> None:
    log: list[dict[str, object]] = []
    online = CachedTransport(
        tmp_path,
        sender=_api_sender(log),
        min_interval_seconds=0,
    )
    result, _ = online.post_json({"mode": "codex", "type": "codex", "id": 0, "ext": "", "lang": "es"})
    assert result["codex"][0]["code"] == "391"
    assert len(log) == 1

    def forbidden(*args, **kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("offline replay touched the network")

    offline = CachedTransport(tmp_path, offline=True, sender=forbidden)
    replay, _ = offline.post_json(
        {"mode": "codex", "type": "codex", "id": 0, "ext": "", "lang": "es"}
    )
    assert replay == result
    assert offline.network_requests == 0
    assert offline.cache_hits == 1


def test_transport_refuses_hosts_and_non_206_ranges(tmp_path: Path) -> None:
    transport = CachedTransport(tmp_path, sender=lambda *args: _response({}))
    with pytest.raises(PolicyError, match="allowlist"):
        transport.request("GET", "https://example.com/Marc/Images/")

    def bad_range(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        del method, headers, body, timeout, max_bytes
        return HttpResponse(200, {}, b"whole archive", url)

    transport = CachedTransport(tmp_path / "range", sender=bad_range)
    with pytest.raises(PolicyError, match="HTTP 206"):
        transport.request(
            "GET",
            "https://sup-infor.com/Marc/Images/t_391.zip",
            headers={"range": "bytes=0-9"},
            allowed_statuses=(200, 206),
        )


def test_api_inventory_builds_explicit_hierarchy_and_replays(tmp_path: Path) -> None:
    log: list[dict[str, object]] = []
    transport = CachedTransport(
        tmp_path,
        sender=_api_sender(log),
        min_interval_seconds=0,
    )
    catalog = inventory_api(transport, codex_ids={"42"}, include_elements=True)
    verify_artifact(catalog, "tlachia-api-catalog.v1")
    assert catalog["payload"]["counts"] == {
        "codex": 1,
        "planche": 1,
        "zone": 1,
        "glyph": 1,
        "element": 1,
    }
    glyph = catalog["payload"]["nodes"]["glyph"][0]
    assert glyph["cote"] == "391_A_001a"
    assert glyph["source_group"] == "tlachia:codex:42:planche:391"
    assert len(log) == 5

    replay = inventory_api(
        CachedTransport(tmp_path, offline=True),
        codex_ids={"42"},
        include_elements=True,
    )
    assert replay == catalog


def test_api_inventory_rejects_cross_codex_and_element_cote(tmp_path: Path) -> None:
    def bad_codex_sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        response = _api_sender([])(method, url, headers, body, timeout, max_bytes)
        payload = json.loads(body)
        if payload["type"] == "planche":
            value = json.loads(response.body)
            value["planches"][0]["codexId"] = "41"
            return _response(value, url)
        return response

    with pytest.raises(IntegrityError, match="codexId does not match"):
        inventory_api(
            CachedTransport(tmp_path / "codex", sender=bad_codex_sender, min_interval_seconds=0),
            codex_ids={"42"},
        )

    def bad_cote_sender(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        response = _api_sender([])(method, url, headers, body, timeout, max_bytes)
        payload = json.loads(body)
        if payload["type"] == "element":
            value = json.loads(response.body)
            value["elements"][0]["cote"] = "391_A_wrong"
            return _response(value, url)
        return response

    with pytest.raises(IntegrityError, match="element cote"):
        inventory_api(
            CachedTransport(tmp_path / "cote", sender=bad_cote_sender, min_interval_seconds=0),
            codex_ids={"42"},
        )


def test_zip_inventory_reads_only_listing_head_and_ranges(tmp_path: Path) -> None:
    payload = _zip_bytes()
    log: list[tuple[str, str]] = []
    catalog = inventory_zips(
        CachedTransport(
            tmp_path,
            sender=_zip_sender(payload, log),
            min_interval_seconds=0,
        )
    )
    verify_artifact(catalog, "tlachia-zip-catalog.v1")
    archive = catalog["payload"]["archives"][0]
    assert archive["entry_count"] == 5
    assert {row["path"] for row in archive["entries"]} == {
        "391_A_001a.jpg",
        "391.jpg",
        "391_A.jpg",
        "04_03_08.bmp",
        "391_A_00.INI",
    }
    assert [method for method, _ in log] == ["GET", "HEAD", "GET"]
    assert len(payload) > sum(
        len(path.read_bytes())
        for path in (tmp_path / "requests").glob("*/body.bin")
        if "t_391" in path.as_posix()
    ) or archive["central_directory_size"] < len(payload)
    assert inventory_zips(CachedTransport(tmp_path, offline=True)) == catalog


def test_zip_eocd_ignores_signature_inside_comment() -> None:
    payload = _zip_bytes()
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    comment = b"false:" + b"PK\x05\x06" + b"tail"
    modified = payload[: eocd + 20] + len(comment).to_bytes(2, "little") + comment
    assert _find_eocd(modified) == eocd


def test_zip_inventory_rejects_range_total_drift(tmp_path: Path) -> None:
    payload = _zip_bytes()
    sender = _zip_sender(payload, [])

    def drifting(method, url, headers, body, timeout, max_bytes):  # noqa: ANN001
        response = sender(method, url, headers, body, timeout, max_bytes)
        if method == "GET" and url.endswith(".zip"):
            start, end, _ = response.headers["content-range"].split(" ", 1)[1].replace("/", "-").split("-")
            return HttpResponse(
                response.status,
                {"content-range": f"bytes {start}-{end}/{len(payload) + 1}"},
                response.body,
                response.url,
            )
        return response

    with pytest.raises(IntegrityError, match="Content-Range"):
        inventory_zips(
            CachedTransport(tmp_path, sender=drifting, min_interval_seconds=0)
        )


def test_join_is_exact_and_reports_missing_archive(tmp_path: Path) -> None:
    api = inventory_api(
        CachedTransport(
            tmp_path / "api",
            sender=_api_sender([]),
            min_interval_seconds=0,
        ),
        codex_ids={"42"},
        include_elements=True,
    )
    zips = inventory_zips(
        CachedTransport(
            tmp_path / "zips",
            sender=_zip_sender(_zip_bytes(), []),
            min_interval_seconds=0,
        )
    )
    joined = join_catalogs(api, zips)
    verify_artifact(joined, "tlachia-api-zip-join.v1")
    assert joined["payload"]["counts"] == {
        "relations": 4,
        "anomalies": 0,
        "missing": 0,
        "ambiguous": 0,
    }
    assert {row["entry_path"] for row in joined["payload"]["relations"]} == {
        "391_A_001a.jpg",
        "391.jpg",
        "391_A.jpg",
        "04_03_08.bmp",
    }
    assert normalise_theme("04.03.08") == "04_03_08.bmp"

    api["payload"]["nodes"]["codex"][0]["code"] = "028"
    for kind in ("codex", "planche", "zone", "glyph", "element"):
        api["payload"]["nodes"][kind][0]["archive_name"] = "t_028.zip"
    api = artifact("tlachia-api-catalog.v1", api["payload"])
    missing = join_catalogs(api, zips)
    assert missing["payload"]["counts"]["missing"] == 4
    assert {row["archive"] for row in missing["payload"]["anomalies"]} == {"t_028.zip"}


def test_business_quarantine_binds_existing_hashes_and_blocks_export(
    tmp_path: Path,
) -> None:
    business = {
        "schema_version": "business-test-set.v1",
        "test_set_id": "business-final-test",
        "content_sha256": "content",
        "rows": [
            {
                "row_id": "business:analysis:0",
                "analysis_id": "analysis",
                "page_sha256": "a" * 64,
                "page_pixel_sha256": "b" * 64,
                "crop_pixel_sha256": "c" * 64,
                "bbox": [1, 2, 3, 4],
                "source_group": "business-page:" + "b" * 64,
                "dataset_split": "locked_test",
            }
        ],
    }
    path = tmp_path / "business.json"
    path.write_text(json.dumps(business), encoding="utf-8")
    quarantine = build_business_quarantine(path)
    verify_artifact(quarantine, "tlachia-business-quarantine.v1")
    page = quarantine["payload"]["pages"][0]
    assert page["mapping_status"] == "unmapped"
    assert page["test_only"] is True
    assert page["page_pixel_sha256"] == "b" * 64

    verify_candidate_export(quarantine, {"schema_version": "training-snapshot.v2", "rows": []})
    with pytest.raises(IntegrityError, match="training-snapshot.v2"):
        verify_candidate_export(quarantine, {"rows": []})
    with pytest.raises(IntegrityError, match="mandatory quarantine identity"):
        verify_candidate_export(
            quarantine,
            {"schema_version": "training-snapshot.v2", "rows": [{}]},
        )
    with pytest.raises(IntegrityError, match="quarantined business row"):
        verify_candidate_export(
            quarantine,
            {
                "schema_version": "training-snapshot.v2",
                "rows": [{
                    "row_id": "business:analysis:0",
                    "dataset_split": "train",
                    "source_group": "safe",
                    "source_pixel_sha256": "d" * 64,
                    "output_sha256": "e" * 64,
                }],
            },
        )
    with pytest.raises(IntegrityError, match="quarantined business page"):
        verify_candidate_export(
            quarantine,
            {
                "schema_version": "training-snapshot.v2",
                "rows": [{
                    "row_id": "other",
                    "dataset_split": "train",
                    "source_group": "safe",
                    "source_pixel_sha256": "b" * 64,
                    "output_sha256": "e" * 64,
                }],
            },
        )


def test_write_once_publication_is_atomic_under_competing_writers(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(value: int) -> None:
        barrier.wait()
        try:
            write_json(path, {"value": value}, write_once=True)
            outcomes.append("written")
        except IntegrityError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["rejected", "written"]
    assert json.loads(path.read_text()) in ({"value": 1}, {"value": 2})


def test_business_mapping_requires_exact_page_hash_and_evidence(tmp_path: Path) -> None:
    business = {
        "schema_version": "business-test-set.v1",
        "test_set_id": "business-final-test",
        "content_sha256": "content",
        "rows": [
            {
                "row_id": "row",
                "analysis_id": "analysis",
                "page_sha256": "a" * 64,
                "page_pixel_sha256": "b" * 64,
                "crop_pixel_sha256": "c" * 64,
                "bbox": [1, 2, 3, 4],
                "source_group": "business-page:" + "b" * 64,
                "dataset_split": "locked_test",
            }
        ],
    }
    manifest = tmp_path / "business.json"
    manifest.write_text(json.dumps(business), encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "mappings": [
                    {
                        "analysis_id": "analysis",
                        "page_pixel_sha256": "wrong",
                        "mapping_status": "mapped",
                        "codex_id": "42",
                        "planche_cote": "391",
                        "evidence": {"reviewer": "business"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="hash mismatch"):
        build_business_quarantine(manifest, mapping_path=mapping)

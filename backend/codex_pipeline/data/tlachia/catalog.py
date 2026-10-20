from __future__ import annotations

import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from .core import (
    IntegrityError,
    POLICY_VERSION,
    PolicyError,
    artifact,
    canonical_json_bytes,
    codex_key,
    element_key,
    glyph_key,
    normalise_media_name,
    normalise_theme,
    planche_key,
    read_json,
    semantic_hash,
    sha256_bytes,
    source_group,
    verify_artifact,
    write_json,
    zone_key,
)


API_URL = "https://cen2.sup-infor.com/Systeme/getCEN.json"
ZIP_LISTING_URL = "https://sup-infor.com/Marc/Images/"
ALLOWED_TARGETS = {
    ("https", "cen2.sup-infor.com", "/Systeme/getCEN.json"),
}
ZIP_PATH_RE = re.compile(r"^/Marc/Images/[A-Za-z0-9_.-]+(?:\.zip)?$")
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
MAX_EOCD_WINDOW = 65_557
ZIP64_SENTINELS = {0xFFFF, 0xFFFFFFFF}


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str


Sender = Callable[[str, str, dict[str, str], bytes | None, float, int], HttpResponse]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    target = (parsed.scheme, (parsed.hostname or "").lower(), parsed.path)
    if target in ALLOWED_TARGETS and not parsed.query and not parsed.fragment:
        return
    if (
        parsed.scheme == "https"
        and (parsed.hostname or "").lower() == "sup-infor.com"
        and (parsed.path == "/Marc/Images/" or ZIP_PATH_RE.fullmatch(parsed.path))
        and not parsed.query
        and not parsed.fragment
    ):
        return
    raise PolicyError(f"URL outside the Tlachia metadata allowlist: {url}")


def _live_sender(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
    max_bytes: int,
) -> HttpResponse:
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    status = int(response.status)
    response_headers = {key.lower(): value for key, value in response.headers.items()}
    response_url = response.geturl()
    if response_url != url:
        raise PolicyError(f"redirects are forbidden: {url} -> {response_url}")
    payload = b"" if method == "HEAD" else response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise PolicyError(f"response exceeded byte budget ({max_bytes}): {url}")
    return HttpResponse(status, response_headers, payload, response_url)


class CachedTransport:
    """Fail-closed HTTP capture with deterministic replay and request budgets."""

    def __init__(
        self,
        store: Path,
        *,
        offline: bool = False,
        max_requests: int = 5_000,
        timeout: float = 30.0,
        min_interval_seconds: float = 0.2,
        sender: Sender | None = None,
    ) -> None:
        self.store = store
        self.offline = offline
        self.max_requests = max_requests
        self.timeout = timeout
        self.min_interval_seconds = min_interval_seconds
        self.sender = sender or _live_sender
        self.network_requests = 0
        self.cache_hits = 0
        self._last_network_at = 0.0

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        max_bytes: int = 5 * 1024 * 1024,
        allowed_statuses: Iterable[int] = (200,),
    ) -> HttpResponse:
        method = method.upper()
        _validate_url(url)
        canonical_headers = {
            key.lower(): value.strip() for key, value in (headers or {}).items()
        }
        descriptor = {
            "policy_version": POLICY_VERSION,
            "method": method,
            "url": url,
            "headers": canonical_headers,
            "body_sha256": sha256_bytes(body or b""),
            "max_bytes": max_bytes,
        }
        request_key = semantic_hash(descriptor)
        request_dir = self.store / "requests" / request_key
        meta_path = request_dir / "response.json"
        body_path = request_dir / "body.bin"
        if meta_path.is_file() and body_path.is_file():
            meta = read_json(meta_path)
            cached_body = body_path.read_bytes()
            if meta.get("request") != descriptor:
                raise IntegrityError(f"cache request descriptor drift: {request_key}")
            if meta.get("body_sha256") != sha256_bytes(cached_body):
                raise IntegrityError(f"cache body checksum drift: {request_key}")
            self.cache_hits += 1
            response = HttpResponse(
                int(meta["status"]),
                dict(meta["headers"]),
                cached_body,
                str(meta["url"]),
            )
            self._check_response(response, allowed_statuses, canonical_headers)
            return response
        if self.offline:
            raise PolicyError(f"offline replay cache miss: {request_key}")
        if self.network_requests >= self.max_requests:
            raise PolicyError(f"network request budget exhausted: {self.max_requests}")
        lock_dir = self.store / "locks" / request_key
        lock_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_dir.mkdir()
        except FileExistsError as error:
            raise PolicyError(
                f"request capture is already in progress or needs review: {request_key}"
            ) from error
        try:
            elapsed = time.monotonic() - self._last_network_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            response = self.sender(
                method,
                url,
                canonical_headers,
                body,
                self.timeout,
                max_bytes,
            )
            self._last_network_at = time.monotonic()
            self.network_requests += 1
            self._check_response(response, allowed_statuses, canonical_headers)
            request_dir.mkdir(parents=True, exist_ok=True)
            temporary_body = request_dir / "body.bin.tmp"
            temporary_body.write_bytes(response.body)
            temporary_body.replace(body_path)
            write_json(
                meta_path,
                {
                    "schema_version": "tlachia-http-capture.v1",
                    "request": descriptor,
                    "status": response.status,
                    "headers": response.headers,
                    "url": response.url,
                    "body_sha256": sha256_bytes(response.body),
                    "body_size": len(response.body),
                },
                write_once=True,
            )
            return response
        finally:
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    @staticmethod
    def _check_response(
        response: HttpResponse,
        allowed_statuses: Iterable[int],
        request_headers: dict[str, str],
    ) -> None:
        if response.status not in set(allowed_statuses):
            raise PolicyError(f"unexpected HTTP status {response.status}: {response.url}")
        if "range" in request_headers and response.status != 206:
            raise PolicyError("range request did not receive HTTP 206")

    def post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        raw = canonical_json_bytes(payload)
        response = self.request(
            "POST",
            API_URL,
            headers={"content-type": "application/json", "accept": "application/json"},
            body=raw,
        )
        try:
            value = json.loads(response.body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("Tlachia API returned non-JSON content") from error
        if not isinstance(value, dict):
            raise IntegrityError("Tlachia API response root must be an object")
        return value, sha256_bytes(response.body)


def _require_list(response: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = response.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise IntegrityError(f"API field {key!r} must be a list of objects")
    return value


def _api_payload(type_name: str, codex_id: object, ext: object = "") -> dict[str, Any]:
    return {
        "mode": "codex",
        "type": type_name,
        "id": codex_id,
        "ext": ext,
        "lang": "es",
    }


def _archive_name_from_codex(codex: dict[str, Any]) -> str:
    directory = str(codex.get("dir") or "").replace("\\", "/").rstrip("/")
    dirname = PurePosixPath(directory).name
    if not re.fullmatch(r"t_[A-Za-z0-9_.-]+", dirname):
        raise IntegrityError(f"codex directory cannot identify ZIP archive: {directory!r}")
    return f"{dirname}.zip"


def _require_codex_identity(row: dict[str, Any], codex_id: str, kind: str) -> None:
    if str(row.get("codexId") or "") != codex_id:
        raise IntegrityError(
            f"{kind} row codexId does not match request: "
            f"{row.get('codexId')!r} != {codex_id!r}"
        )


def inventory_api(
    transport: CachedTransport,
    *,
    codex_ids: set[str] | None = None,
    include_elements: bool = True,
) -> dict[str, Any]:
    root, root_sha = transport.post_json(_api_payload("codex", 0))
    codices = _require_list(root, "codex")
    if codex_ids is not None:
        codices = [row for row in codices if str(row.get("id")) in codex_ids]
        missing = sorted(codex_ids - {str(row.get("id")) for row in codices})
        if missing:
            raise IntegrityError(f"requested codex IDs absent from API: {missing}")

    nodes: dict[str, list[dict[str, Any]]] = {
        "codex": [],
        "planche": [],
        "zone": [],
        "glyph": [],
        "element": [],
    }
    relations: list[dict[str, str]] = []
    response_hashes = [root_sha]
    seen: dict[str, set[str]] = {key: set() for key in nodes}
    node_lookup: dict[str, dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []

    def append_node(kind: str, key: str, value: dict[str, Any]) -> bool:
        if key in seen[kind]:
            if kind not in {"glyph", "element"}:
                raise IntegrityError(f"duplicate canonical API key: {key}")
            existing = node_lookup[key]
            essential = (
                ("codex_id", "codex_code", "glyph_id", "cote", "glyph_type")
                if kind == "glyph"
                else ("codex_id", "glyph_cote", "element_id", "theme", "element")
            )
            if any(existing.get(field) != value.get(field) for field in essential):
                differences = {
                    field: [existing.get(field), value.get(field)]
                    for field in essential
                    if existing.get(field) != value.get(field)
                }
                raise IntegrityError(
                    f"conflicting semantic {kind} occurrence: {key}: {differences}"
                )
            if kind == "glyph":
                planche_cotes = existing.setdefault(
                    "planche_cotes", [existing["planche_cote"]]
                )
                if value["planche_cote"] not in planche_cotes:
                    planche_cotes.append(value["planche_cote"])
                    planche_cotes.sort(key=lambda item: (item.casefold(), item))
                zone_cotes = existing.setdefault("zone_cotes", [existing["zone_cote"]])
                if value["zone_cote"] not in zone_cotes:
                    zone_cotes.append(value["zone_cote"])
                    zone_cotes.sort()
            variants = existing.get("raw_variants")
            if variants is None:
                variants = [existing.pop("raw")]
                existing["raw_variants"] = variants
            if value["raw"] not in variants:
                variants.append(value["raw"])
                variants.sort(key=semantic_hash)
            anomalies.append({"type": f"duplicate_semantic_{kind}_row", "node": key})
            return False
        seen[kind].add(key)
        node = {"key": key, **value}
        nodes[kind].append(node)
        node_lookup[key] = node
        return True

    for codex in sorted(codices, key=lambda row: str(row.get("id"))):
        codex_id = str(codex.get("id") or "")
        code = str(codex.get("code") or "")
        if not codex_id or not code:
            raise IntegrityError(f"codex lacks id/code: {codex!r}")
        archive_name = _archive_name_from_codex(codex)
        c_key = codex_key(codex_id)
        append_node(
            "codex",
            c_key,
            {
                "codex_id": codex_id,
                "code": code,
                "archive_name": archive_name,
                "raw": codex,
            },
        )
        planche_response, response_sha = transport.post_json(
            _api_payload("planche", codex_id)
        )
        response_hashes.append(response_sha)
        for planche in _require_list(planche_response, "planches"):
            _require_codex_identity(planche, codex_id, "planche")
            planche_cote = str(planche.get("cote") or "")
            p_key = planche_key(codex_id, planche_cote)
            append_node(
                "planche",
                p_key,
                {
                    "codex_id": codex_id,
                    "codex_code": code,
                    "archive_name": archive_name,
                    "cote": planche_cote,
                    "source_group": source_group(codex_id, planche_cote),
                    "raw": planche,
                },
            )
            relations.append({"type": "contains", "parent": c_key, "child": p_key})
            zone_response, response_sha = transport.post_json(
                _api_payload("zone", codex_id, planche_cote)
            )
            response_hashes.append(response_sha)
            for zone in _require_list(zone_response, "zones"):
                _require_codex_identity(zone, codex_id, "zone")
                zone_cote = str(zone.get("cote") or "")
                z_key = zone_key(codex_id, planche_cote, zone_cote)
                append_node(
                    "zone",
                    z_key,
                    {
                        "codex_id": codex_id,
                        "codex_code": code,
                        "archive_name": archive_name,
                        "planche_cote": planche_cote,
                        "cote": zone_cote,
                        "source_group": source_group(codex_id, planche_cote),
                        "raw": zone,
                    },
                )
                relations.append({"type": "contains", "parent": p_key, "child": z_key})
                glyph_response, response_sha = transport.post_json(
                    _api_payload("glyphe", codex_id, zone_cote)
                )
                response_hashes.append(response_sha)
                glyph_rows = [
                    ("glyph", item) for item in _require_list(glyph_response, "glyphes")
                ] + [
                    ("person", item)
                    for item in _require_list(glyph_response, "personnes")
                ]
                for glyph_type, glyph in glyph_rows:
                    _require_codex_identity(glyph, codex_id, glyph_type)
                    glyph_id = str(glyph.get("id") or "")
                    glyph_cote = str(glyph.get("cote") or "")
                    g_key = glyph_key(codex_id, glyph_id, glyph_cote)
                    append_node(
                        "glyph",
                        g_key,
                        {
                            "codex_id": codex_id,
                            "codex_code": code,
                            "archive_name": archive_name,
                            "planche_cote": planche_cote,
                            "zone_cote": zone_cote,
                            "glyph_id": glyph_id,
                            "cote": glyph_cote,
                            "glyph_type": glyph_type,
                            "source_group": source_group(codex_id, planche_cote),
                            "raw": glyph,
                        },
                    )
                    relations.append(
                        {"type": "contains", "parent": z_key, "child": g_key}
                    )
                    if not include_elements:
                        continue
                    element_response, response_sha = transport.post_json(
                        _api_payload("element", codex_id, glyph_cote)
                    )
                    response_hashes.append(response_sha)
                    for element in _require_list(element_response, "elements"):
                        _require_codex_identity(element, codex_id, "element")
                        if str(element.get("cote") or "") != glyph_cote:
                            raise IntegrityError(
                                "element cote does not match requested glyph: "
                                f"{element.get('cote')!r} != {glyph_cote!r}"
                            )
                        element_id = str(element.get("id") or "")
                        e_key = element_key(
                            codex_id,
                            glyph_cote,
                            element_id,
                            element.get("theme"),
                        )
                        append_node(
                            "element",
                            e_key,
                            {
                                "codex_id": codex_id,
                                "codex_code": code,
                                "archive_name": archive_name,
                                "planche_cote": planche_cote,
                                "zone_cote": zone_cote,
                                "glyph_cote": glyph_cote,
                                "element_id": element_id,
                                "theme": element.get("theme"),
                                "element": element.get("element"),
                                "source_group": source_group(codex_id, planche_cote),
                                "raw": element,
                            },
                        )
                        relations.append(
                            {"type": "contains", "parent": g_key, "child": e_key}
                        )

    for values in nodes.values():
        values.sort(key=lambda row: row["key"])
    relations = [
        {"type": rel_type, "parent": parent, "child": child}
        for parent, child, rel_type in sorted(
            {(row["parent"], row["child"], row["type"]) for row in relations}
        )
    ]
    payload = {
        "endpoint": API_URL,
        "include_elements": include_elements,
        "selected_codex_ids": sorted(codex_ids) if codex_ids else None,
        "nodes": nodes,
        "relations": relations,
        "anomalies": sorted(
            anomalies, key=lambda row: (row["type"], row.get("node", ""))
        ),
        "counts": {kind: len(values) for kind, values in nodes.items()},
        "response_body_sha256": sorted(response_hashes),
    }
    return artifact("tlachia-api-catalog.v1", payload)


class _ZipLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href and href.casefold().endswith(".zip"):
            self.links.append(href)


def parse_zip_listing(body: bytes, base_url: str = ZIP_LISTING_URL) -> list[str]:
    parser = _ZipLinkParser()
    parser.feed(body.decode("iso-8859-1"))
    urls = sorted({urllib.parse.urljoin(base_url, link) for link in parser.links})
    for url in urls:
        _validate_url(url)
    return urls


def _content_range(headers: dict[str, str]) -> tuple[int, int, int]:
    value = headers.get("content-range", "")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
    if not match:
        raise IntegrityError(f"invalid Content-Range: {value!r}")
    start, end, total = (int(match.group(index)) for index in range(1, 4))
    if end < start or total <= end:
        raise IntegrityError(f"incoherent Content-Range: {value!r}")
    return start, end, total


def _validate_range_response(
    response: HttpResponse,
    *,
    requested_start: int,
    requested_end: int,
    total_size: int,
    expected_etag: str | None,
) -> None:
    start, end, total = _content_range(response.headers)
    if (start, end, total) != (requested_start, requested_end, total_size):
        raise IntegrityError("ZIP Content-Range does not match the requested archive")
    if len(response.body) != requested_end - requested_start + 1:
        raise IntegrityError("ZIP range body length does not match Content-Range")
    response_etag = response.headers.get("etag")
    if expected_etag is not None and response_etag not in {None, expected_etag}:
        raise IntegrityError("ZIP ETag changed between HEAD and Range request")


def _find_eocd(data: bytes) -> int:
    position = len(data)
    while True:
        position = data.rfind(EOCD_SIGNATURE, 0, position)
        if position < 0:
            raise IntegrityError("ZIP EOCD not found")
        if position + 22 <= len(data):
            comment_length = struct.unpack_from("<H", data, position + 20)[0]
            if position + 22 + comment_length == len(data):
                return position


def _decode_zip_name(raw: bytes, flag_bits: int) -> tuple[str, str]:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw.decode(encoding), encoding
    except UnicodeDecodeError as error:
        raise IntegrityError(f"invalid ZIP filename encoding ({encoding})") from error


def parse_central_directory(data: bytes, *, expected_entries: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    position = 0
    while position < len(data):
        if data[position : position + 4] != CENTRAL_SIGNATURE:
            raise IntegrityError(f"invalid central directory signature at {position}")
        if position + 46 > len(data):
            raise IntegrityError("truncated central directory record")
        fields = struct.unpack_from("<4s6H3L5H2L", data, position)
        flag_bits = fields[3]
        compression = fields[4]
        crc32 = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        local_offset = fields[16]
        record_end = position + 46 + name_length + extra_length + comment_length
        if record_end > len(data):
            raise IntegrityError("truncated ZIP central directory variable fields")
        raw_name = data[position + 46 : position + 46 + name_length]
        filename, encoding = _decode_zip_name(raw_name, flag_bits)
        entries.append(
            {
                "path": filename.replace("\\", "/"),
                "normalized_path": normalise_media_name(filename),
                "raw_name_sha256": sha256_bytes(raw_name),
                "filename_encoding": encoding,
                "flag_bits": flag_bits,
                "compression": compression,
                "crc32": f"{crc32:08x}",
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_offset,
            }
        )
        position = record_end
    if len(entries) != expected_entries:
        raise IntegrityError(
            f"central directory count mismatch: {len(entries)} != {expected_entries}"
        )
    return entries


def inventory_zip_archive(
    transport: CachedTransport,
    url: str,
    *,
    max_central_directory_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    head = transport.request("HEAD", url, allowed_statuses=(200, 404))
    if head.status == 404:
        return {"url": url, "status": "missing", "entries": [], "anomalies": ["404"]}
    try:
        size = int(head.headers["content-length"])
    except (KeyError, ValueError) as error:
        raise IntegrityError(f"archive Content-Length missing/invalid: {url}") from error
    if size <= 0:
        raise IntegrityError(f"archive has invalid size: {url}")
    tail_start = max(0, size - MAX_EOCD_WINDOW)
    etag = head.headers.get("etag")
    tail_headers = {"range": f"bytes={tail_start}-{size - 1}"}
    if etag:
        tail_headers["if-match"] = etag
    tail = transport.request(
        "GET",
        url,
        headers=tail_headers,
        max_bytes=MAX_EOCD_WINDOW,
        allowed_statuses=(206,),
    )
    _validate_range_response(
        tail,
        requested_start=tail_start,
        requested_end=size - 1,
        total_size=size,
        expected_etag=etag,
    )
    eocd_at = _find_eocd(tail.body)
    eocd = struct.unpack_from("<4s4H2LH", tail.body, eocd_at)
    disk, central_disk, disk_entries, total_entries = eocd[1:5]
    central_size, central_offset, comment_length = eocd[5:8]
    if disk != 0 or central_disk != 0 or disk_entries != total_entries:
        raise PolicyError("multi-disk ZIP archives are unsupported")
    if any(
        value in ZIP64_SENTINELS
        for value in (disk_entries, total_entries, central_size, central_offset)
    ):
        raise PolicyError("ZIP64 archives require an explicit reviewed implementation")
    if central_size > max_central_directory_bytes:
        raise PolicyError("central directory exceeds metadata byte budget")
    if eocd_at + 22 + comment_length != len(tail.body):
        raise IntegrityError("ZIP EOCD does not terminate at archive boundary")
    eocd_absolute = tail_start + eocd_at
    if central_offset < 0 or central_offset + central_size > eocd_absolute:
        raise IntegrityError("central directory lies outside ZIP bounds")
    tail_end = tail_start + len(tail.body)
    if central_offset >= tail_start and central_offset + central_size <= tail_end:
        start = central_offset - tail_start
        central = tail.body[start : start + central_size]
    else:
        central_headers = {
            "range": f"bytes={central_offset}-{central_offset + central_size - 1}"
        }
        if etag:
            central_headers["if-match"] = etag
        central_response = transport.request(
            "GET",
            url,
            headers=central_headers,
            max_bytes=max_central_directory_bytes,
            allowed_statuses=(206,),
        )
        _validate_range_response(
            central_response,
            requested_start=central_offset,
            requested_end=central_offset + central_size - 1,
            total_size=size,
            expected_etag=etag,
        )
        central = central_response.body
    entries = parse_central_directory(central, expected_entries=total_entries)
    normalized: dict[str, list[str]] = {}
    for entry in entries:
        normalized.setdefault(entry["normalized_path"], []).append(entry["path"])
    collisions = [
        {"normalized_path": key, "paths": sorted(paths)}
        for key, paths in normalized.items()
        if len(paths) > 1
    ]
    return {
        "url": url,
        "archive_name": PurePosixPath(urllib.parse.urlsplit(url).path).name,
        "status": "available",
        "content_length": size,
        "etag": etag,
        "last_modified": head.headers.get("last-modified"),
        "entry_count": total_entries,
        "central_directory_size": central_size,
        "central_directory_sha256": sha256_bytes(central),
        "entries": entries,
        "anomalies": ["normalized_filename_collision"] if collisions else [],
        "filename_collisions": collisions,
    }


def inventory_zips(
    transport: CachedTransport,
    *,
    listing_url: str = ZIP_LISTING_URL,
    archive_names: set[str] | None = None,
) -> dict[str, Any]:
    listing = transport.request("GET", listing_url, max_bytes=2 * 1024 * 1024)
    urls = parse_zip_listing(listing.body, listing_url)
    if archive_names is not None:
        urls = [
            url
            for url in urls
            if PurePosixPath(urllib.parse.urlsplit(url).path).name in archive_names
        ]
    archives = [inventory_zip_archive(transport, url) for url in urls]
    payload = {
        "listing_url": listing_url,
        "listing_body_sha256": sha256_bytes(listing.body),
        "listing_parser_version": "apache-index-anchors.v1",
        "archives": archives,
        "counts": {
            "listed": len(urls),
            "available": sum(row["status"] == "available" for row in archives),
            "missing": sum(row["status"] == "missing" for row in archives),
            "entries": sum(int(row.get("entry_count", 0)) for row in archives),
        },
    }
    return artifact("tlachia-zip-catalog.v1", payload)


def _basename_index(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        basename = PurePosixPath(entry["normalized_path"]).name
        result.setdefault(basename, []).append(entry)
    return result


def join_catalogs(api_catalog: dict[str, Any], zip_catalog: dict[str, Any]) -> dict[str, Any]:
    verify_artifact(api_catalog, "tlachia-api-catalog.v1")
    verify_artifact(zip_catalog, "tlachia-zip-catalog.v1")
    api_payload = api_catalog["payload"]
    zip_payload = zip_catalog["payload"]
    archives = {
        row.get("archive_name", "").casefold(): row
        for row in zip_payload["archives"]
        if row.get("status") == "available"
    }
    relations: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    indexes = {
        name: _basename_index(archive["entries"]) for name, archive in archives.items()
    }

    def match(
        node_key: str,
        archive_name: str,
        expected: str,
        role: str,
        *,
        exact_alternates: tuple[str, ...] = (),
    ) -> None:
        archive = archives.get(archive_name.casefold())
        if archive is None:
            anomalies.append(
                {"type": "archive_missing", "node": node_key, "archive": archive_name}
            )
            return
        expected_variants = (expected, *exact_alternates)
        matches_by_path: dict[str, dict[str, Any]] = {}
        for variant in expected_variants:
            for row in indexes[archive_name.casefold()].get(
                normalise_media_name(variant), []
            ):
                matches_by_path[row["path"]] = row
        matches = list(matches_by_path.values())
        if len(matches) != 1:
            anomalies.append(
                {
                    "type": "media_missing" if not matches else "media_ambiguous",
                    "node": node_key,
                    "archive": archive_name,
                    "expected": expected,
                    "exact_alternates": list(exact_alternates),
                    "matches": [row["path"] for row in matches],
                }
            )
            return
        relations.append(
            {
                "node": node_key,
                "role": role,
                "archive": archive_name,
                "entry_path": matches[0]["path"],
                "entry_crc32": matches[0]["crc32"],
                "filename_serialization": (
                    "exact_primary"
                    if not exact_alternates
                    else (
                        "canonical_theme_underscore"
                        if normalise_media_name(matches[0]["path"])
                        == normalise_media_name(expected)
                        else "exact_dotted_theme_alias"
                    )
                ),
            }
        )

    for planche in api_payload["nodes"]["planche"]:
        match(
            planche["key"],
            planche["archive_name"],
            f"{planche['cote']}.jpg",
            "planche",
        )
    for zone in api_payload["nodes"]["zone"]:
        match(
            zone["key"],
            zone["archive_name"],
            f"{zone['cote']}.jpg",
            "zone",
        )
    for glyph in api_payload["nodes"]["glyph"]:
        match(
            glyph["key"],
            glyph["archive_name"],
            f"{glyph['cote']}.jpg",
            "glyph_occurrence",
        )
    for element in api_payload["nodes"]["element"]:
        try:
            expected = normalise_theme(element["theme"])
        except IntegrityError:
            anomalies.append(
                {"type": "invalid_theme", "node": element["key"], "theme": element["theme"]}
            )
            continue
        match(
            element["key"],
            element["archive_name"],
            expected,
            "element_theme_reference",
            exact_alternates=(f"{element['theme']}.bmp",),
        )
    relations.sort(key=lambda row: (row["node"], row["role"], row["entry_path"]))
    anomalies.sort(key=lambda row: (row["type"], row.get("node", "")))
    payload = {
        "api_payload_sha256": api_catalog["payload_sha256"],
        "zip_payload_sha256": zip_catalog["payload_sha256"],
        "relations": relations,
        "anomalies": anomalies,
        "counts": {
            "relations": len(relations),
            "anomalies": len(anomalies),
            "missing": sum(row["type"].endswith("missing") for row in anomalies),
            "ambiguous": sum(row["type"].endswith("ambiguous") for row in anomalies),
        },
        "authority": {"semantics": "Tlachia API", "pixels": "ZIP central directory"},
        "fuzzy_matching": False,
    }
    return artifact("tlachia-api-zip-join.v1", payload)


def merge_api_catalogs(catalogs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge disjoint, element-complete codex batches without weakening provenance."""
    if not catalogs:
        raise IntegrityError("at least one API catalog is required")
    payloads: list[dict[str, Any]] = []
    for catalog in catalogs:
        verify_artifact(catalog, "tlachia-api-catalog.v1")
        payload = catalog["payload"]
        if not payload.get("include_elements"):
            raise PolicyError("only element-complete API catalogs can be merged for P3")
        payloads.append(payload)
    endpoints = {payload.get("endpoint") for payload in payloads}
    if len(endpoints) != 1:
        raise IntegrityError("API batch endpoints differ")
    selected_sets = [set(map(str, payload.get("selected_codex_ids", []))) for payload in payloads]
    seen_codex: set[str] = set()
    for selected in selected_sets:
        overlap = seen_codex & selected
        if overlap:
            raise IntegrityError(f"API batches overlap on codex IDs: {sorted(overlap)}")
        seen_codex.update(selected)

    node_types = ("codex", "planche", "zone", "glyph", "element")
    nodes: dict[str, list[dict[str, Any]]] = {node_type: [] for node_type in node_types}
    for node_type in node_types:
        by_key: dict[str, dict[str, Any]] = {}
        for payload in payloads:
            for node in payload.get("nodes", {}).get(node_type, []):
                key = str(node.get("key", ""))
                if not key:
                    raise IntegrityError(f"{node_type} node lacks a key")
                existing = by_key.get(key)
                if existing is not None and existing != node:
                    raise IntegrityError(f"conflicting merged API node: {key}")
                by_key[key] = node
        nodes[node_type] = sorted(by_key.values(), key=lambda row: row["key"])

    relations_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in payloads:
        for relation in payload.get("relations", []):
            key = (relation["type"], relation["parent"], relation["child"])
            existing = relations_by_key.get(key)
            if existing is not None and existing != relation:
                raise IntegrityError(f"conflicting merged API relation: {key}")
            relations_by_key[key] = relation
    relations = sorted(
        relations_by_key.values(), key=lambda row: (row["type"], row["parent"], row["child"])
    )
    anomalies = sorted(
        [row for payload in payloads for row in payload.get("anomalies", [])],
        key=lambda row: (row.get("type", ""), row.get("node", ""), semantic_hash(row)),
    )
    merged_payload = {
        "endpoint": next(iter(endpoints)),
        "include_elements": True,
        "selected_codex_ids": sorted(seen_codex, key=lambda value: (len(value), value)),
        "nodes": nodes,
        "relations": relations,
        "anomalies": anomalies,
        "response_body_sha256": sorted(
            {
                value
                for payload in payloads
                for value in payload.get("response_body_sha256", [])
            }
        ),
        "source_catalog_payload_sha256": sorted(
            catalog["payload_sha256"] for catalog in catalogs
        ),
        "counts": {
            **{node_type: len(nodes[node_type]) for node_type in node_types},
            "relations": len(relations),
            "anomalies": len(anomalies),
            "source_catalogs": len(catalogs),
        },
    }
    return artifact("tlachia-api-catalog.v1", merged_payload)

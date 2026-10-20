from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any


CANONICALIZATION_VERSION = "canonical-json-nfc-v1"
HASH_ALGORITHM = "sha256"
POLICY_VERSION = "tlachia-metadata-only.v1"
THEME_RE = re.compile(r"^\d{2}(?:\.\d{2})+$")


class TlachiaError(RuntimeError):
    """Base error for policy or integrity failures."""


class PolicyError(TlachiaError):
    """Hard stop: the requested action violates the capture policy."""


class IntegrityError(TlachiaError):
    """Hard stop: cached or derived evidence is inconsistent."""


class AnomalyError(TlachiaError):
    """Completed metadata operation containing blocking anomalies."""


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise TypeError("floats are forbidden in canonical hashed payloads")
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalise(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def artifact(schema_version: str, payload: dict[str, Any]) -> dict[str, Any]:
    canonical = _normalise(payload)
    return {
        "schema_version": schema_version,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "payload": canonical,
        "payload_sha256": semantic_hash(canonical),
    }


def verify_artifact(value: dict[str, Any], expected_schema: str | None = None) -> None:
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise IntegrityError(
            f"unexpected schema: {value.get('schema_version')!r} != {expected_schema!r}"
        )
    if value.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise IntegrityError("unsupported canonicalization version")
    if value.get("hash_algorithm") != HASH_ALGORITHM:
        raise IntegrityError("unsupported hash algorithm")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise IntegrityError("artifact payload must be an object")
    if value.get("payload_sha256") != semantic_hash(payload):
        raise IntegrityError("artifact semantic hash mismatch")


def write_json(path: Path, value: dict[str, Any], *, write_once: bool = False) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(encoded)
    if not write_once:
        os.replace(temporary, path)
        return
    try:
        # A hard-link publishes the fully written temporary file atomically while
        # preserving O_EXCL semantics for immutable evidence artifacts.
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise IntegrityError(f"refusing to overwrite immutable artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON root must be an object: {path}")
    return value


def _component(value: object, label: str) -> str:
    if value is None:
        raise IntegrityError(f"missing canonical identity component: {label}")
    text = unicodedata.normalize("NFC", str(value).strip())
    if not text or ":" in text:
        raise IntegrityError(f"invalid canonical identity component {label}: {value!r}")
    return text


def codex_key(codex_id: object) -> str:
    return f"cen:codex:{_component(codex_id, 'codex_id')}"


def planche_key(codex_id: object, cote: object) -> str:
    return f"{codex_key(codex_id)}:planche:{_component(cote, 'planche_cote')}"


def zone_key(codex_id: object, planche_cote: object, cote: object) -> str:
    return f"{planche_key(codex_id, planche_cote)}:zone:{_component(cote, 'zone_cote')}"


def glyph_key(codex_id: object, api_id: object, cote: object) -> str:
    return (
        f"{codex_key(codex_id)}:glyph:{_component(api_id, 'glyph_id')}:"
        f"{_component(cote, 'glyph_cote')}"
    )


def element_key(
    codex_id: object,
    glyph_cote: object,
    api_id: object,
    theme: object,
) -> str:
    return (
        f"{codex_key(codex_id)}:glyph-cote:{_component(glyph_cote, 'glyph_cote')}:"
        f"element:{_component(api_id, 'element_id')}:"
        f"theme:{_component(theme, 'theme')}"
    )


def source_group(codex_id: object, planche_cote: object) -> str:
    return (
        f"tlachia:codex:{_component(codex_id, 'codex_id')}:"
        f"planche:{_component(planche_cote, 'planche_cote')}"
    )


def normalise_theme(theme: object) -> str:
    text = unicodedata.normalize("NFC", str(theme).strip())
    if not THEME_RE.fullmatch(text):
        raise IntegrityError(f"theme code is not canonical: {theme!r}")
    return text.replace(".", "_") + ".bmp"


def normalise_media_name(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


def validate_tlachia_provenance(row: dict[str, Any]) -> None:
    if row.get("source_kind") != "tlachia":
        return
    expected = source_group(row.get("codex_id"), row.get("planche_cote"))
    if row.get("source_group") != expected:
        raise PolicyError("Tlachia source_group must derive from codex_id + planche_cote")
    if expected.startswith(("source-path:", "image-pixels:")):
        raise PolicyError("path/pixel fallback groups are forbidden for Tlachia")

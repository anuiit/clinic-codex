"""Local admin review state for submitted annotations.

Submitted images and metadata remain immutable. Operator decisions and
corrections live in SQLite; corrected crops are derived immutable files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

try:
    from backend.services.annotation_storage import (
        clamp_bbox,
        normalize_bbox_to_int_pixels,
        sanitize_class_name,
        sanitize_note,
    )
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from services.annotation_storage import (  # type: ignore
        clamp_bbox,
        normalize_bbox_to_int_pixels,
        sanitize_class_name,
        sanitize_note,
    )


REVIEW_STATUSES = {"pending", "approved", "rejected"}
TRAINABLE_STATUS = "approved"
DATASET_SPLITS = {"train", "val", "test", "excluded"}
TRAINABLE_DATASET_SPLITS = {"train", "val", "test"}
SPLIT_REASONS = {
    "trainable_hash_80_10_10",
    "pending_review",
    "rejected_review",
    "missing_crop",
    "stale_decision",
}
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "review-index.json"
LOCAL_ONLY_WARNING = (
    "Local/dev-only annotation review endpoint. It is not production-secured; "
    "only use it with locally bound backend/frontend services."
)

_SAFE_ANALYSIS_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_UNSET = object()


class AnnotationReviewError(Exception):
    """Base class for annotation review errors."""


class AnnotationReviewValidationError(AnnotationReviewError, ValueError):
    """Raised for invalid review mutation input."""


class AnnotationReviewNotFoundError(AnnotationReviewError, LookupError):
    """Raised when a canonical submitted annotation cannot be found."""


class AnnotationReviewConflictError(AnnotationReviewError):
    """A decision changed or a legacy store requires an explicit import."""


class AnnotationReviewMigrationRequiredError(AnnotationReviewConflictError):
    """Existing JSON decisions must be imported before any review write."""


def review_key(analysis_id: str, index: int) -> str:
    return f"{analysis_id}:{index}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "decisions": {}}


def _diagnostic(
    code: str,
    message: str,
    *,
    analysis_id: str | None = None,
    index: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"code": code, "message": message}
    if analysis_id is not None:
        data["analysis_id"] = analysis_id
    if index is not None:
        data["index"] = index
    if key is not None:
        data["key"] = key
    return data


def _validate_analysis_id(analysis_id: str) -> None:
    if not isinstance(analysis_id, str) or not analysis_id:
        raise AnnotationReviewValidationError("analysis_id must be a non-empty string")
    if not _SAFE_ANALYSIS_ID.match(analysis_id):
        raise AnnotationReviewValidationError(
            "analysis_id must be alphanumeric/dash/underscore only"
        )


def _validate_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AnnotationReviewValidationError("index must be a non-negative integer")


def _validate_status(status: str) -> None:
    if status not in REVIEW_STATUSES:
        allowed = ", ".join(sorted(REVIEW_STATUSES))
        raise AnnotationReviewValidationError(f"status must be one of: {allowed}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_under_analysis_dir(path_value: object, analysis_dir: Path, fallback: Path) -> Path:
    if isinstance(path_value, str) and path_value:
        candidate = Path(path_value)
        if candidate.is_absolute() and candidate.exists() and _is_relative_to(candidate, analysis_dir):
            return candidate
        if not candidate.is_absolute():
            relative = analysis_dir / candidate
            if relative.exists() and _is_relative_to(relative, analysis_dir):
                return relative
    return fallback


def _path_payload(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    payload: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        stat = path.stat()
        payload["size"] = stat.st_size
        payload["mtime_ns"] = stat.st_mtime_ns
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(
    *,
    analysis_id: str,
    uploaded_at: str | None,
    annotation: dict[str, Any],
    crop_path: Path,
) -> str:
    payload = {
        "analysis_id": analysis_id,
        "index": annotation.get("index"),
        "uploaded_at": uploaded_at,
        "class_name": annotation.get("class_name"),
        "bbox": annotation.get("bbox"),
        "crop": _path_payload(crop_path),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _dataset_split_for_key(key: str) -> str:
    """Return the deterministic 80/10/10 split for an immutable review key."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _excluded_split_reason(
    *,
    review_status: str,
    crop_exists: bool,
    stale_decision: bool,
) -> str:
    if stale_decision:
        return "stale_decision"
    if not crop_exists:
        return "missing_crop"
    if review_status == "rejected":
        return "rejected_review"
    return "pending_review"


def _split_payload_for_element(
    *,
    key: str,
    review_status: str,
    crop_exists: bool,
    stale_decision: bool,
    decision: dict[str, Any] | None = None,
) -> dict[str, str]:
    if review_status == TRAINABLE_STATUS and crop_exists and not stale_decision:
        persisted_split = (decision or {}).get("dataset_split")
        dataset_split = (
            persisted_split
            if persisted_split in TRAINABLE_DATASET_SPLITS
            else _dataset_split_for_key(key)
        )
        return {
            "dataset_split": dataset_split,
            "split_reason": "trainable_hash_80_10_10",
        }
    return {
        "dataset_split": "excluded",
        "split_reason": _excluded_split_reason(
            review_status=review_status,
            crop_exists=crop_exists,
            stale_decision=stale_decision,
        ),
    }


class AnnotationReviewStore:
    """Transactional review decisions over immutable submitted annotations."""

    def __init__(self, annotations_dir: Path, manifest_path: Path | None = None):
        self.annotations_dir = Path(annotations_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else self.annotations_dir / MANIFEST_FILENAME
        self.db_path = self.annotations_dir / "review-state.sqlite3"

    def ensure_database(self) -> None:
        """Create an empty database, but never silently import a legacy manifest."""
        if not self.db_path.exists() and self.manifest_path.exists():
            raise AnnotationReviewMigrationRequiredError(
                'legacy review-index.json needs explicit import: run scripts/migrate_annotation_reviews.py '
                f'--annotations-dir "{self.annotations_dir}" --apply'
            )
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=10)) as connection, connection:
            self._create_schema(connection)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS review_decisions (key TEXT PRIMARY KEY, revision INTEGER NOT NULL, decision_json TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS review_history (key TEXT NOT NULL, revision INTEGER NOT NULL, action TEXT NOT NULL, state_json TEXT NOT NULL, PRIMARY KEY (key, revision))")
        connection.execute("CREATE TABLE IF NOT EXISTS review_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @contextmanager
    def _database(self, *, write: bool = False):
        self.ensure_database()
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def export_review_manifest(self) -> dict[str, Any]:
        """Compatibility snapshot for retraining; SQLite remains authoritative."""
        return self._load_manifest()

    def review_manifest_sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.export_review_manifest(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _check_revision(current: int, expected: int | None) -> None:
        if expected is None:
            return  # Direct Python callers retain the pre-API convenience contract.
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise AnnotationReviewValidationError("expected_revision must be a non-negative integer")
        if expected != current:
            raise AnnotationReviewConflictError(f"stale annotation revision: current is {current}")

    def _current_element(self, connection: sqlite3.Connection, analysis_id: str, index: int) -> tuple[dict[str, Any], int]:
        key = review_key(analysis_id, index)
        row = connection.execute("SELECT revision, decision_json FROM review_decisions WHERE key = ?", (key,)).fetchone()
        decision = json.loads(row["decision_json"]) if row else None
        analysis_dir = self.annotations_dir / analysis_id
        if not analysis_dir.is_dir() or not _is_relative_to(analysis_dir, self.annotations_dir):
            raise AnnotationReviewNotFoundError(f"annotation element not found: {key}")
        analysis = self._read_analysis(analysis_dir, {key: decision} if decision else {}, [], set(), only_index=index)
        element = self._find_element([analysis] if analysis else [], analysis_id, index)
        if element is None:
            raise AnnotationReviewNotFoundError(f"annotation element not found: {key}")
        return element, int(row["revision"]) if row else 0

    @staticmethod
    def _write_decision(
        connection: sqlite3.Connection, element: dict[str, Any], revision: int,
        status: str, reviewer_id: str | None, action: str,
    ) -> None:
        key = element["key"]
        split = _split_payload_for_element(
            key=key, review_status=status, crop_exists=Path(element["crop_path"]).is_file(),
            stale_decision=False,
        )
        crop = Path(element["crop_path"])
        decision = {
            "analysis_id": element["analysis_id"], "index": element["index"],
            "revision": revision, "status": status, "reviewed_at": utc_now_iso(),
            "reviewed_by": reviewer_id,
            "source_fingerprint": element["base_source_fingerprint"],
            "class_name": element["class_name"], "bbox": element["bbox"],
            "note": element.get("note"), "crop_path": element["crop_path"],
            "crop_sha256": _file_sha256(crop) if crop.is_file() else None,
            **split,
        }
        encoded = json.dumps(decision, sort_keys=True, ensure_ascii=False)
        connection.execute(
            "INSERT INTO review_decisions(key, revision, decision_json) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET revision=excluded.revision, decision_json=excluded.decision_json",
            (key, revision, encoded),
        )
        connection.execute(
            "INSERT INTO review_history(key, revision, action, state_json) VALUES (?, ?, ?, ?)",
            (key, revision, action, encoded),
        )

    def _mutation_result(self, analysis_id: str, index: int, *, include_counts: bool = False) -> dict[str, Any]:
        queue = self.list_queue()
        result = {
            "status": "ok", "local_only": True, "warning": LOCAL_ONLY_WARNING,
            "element": self._find_element(queue["analyses"], analysis_id, index),
        }
        if include_counts:
            result["counts"] = queue["counts"]
        return result

    def _write_crop(self, analysis_id: str, index: int, image: Image.Image, bbox: list[int]) -> Path:
        elements_dir = self.annotations_dir / analysis_id / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)
        name = f"review-{index}-{uuid.uuid4().hex}.png"
        final = elements_dir / name
        staged = elements_dir / f".{name}.tmp"
        x, y, w, h = bbox
        try:
            image.crop((x, y, x + w, y + h)).save(staged, format="PNG")
            # ponytail: a crash before DB commit can leave an unreferenced derived crop.
            # It never changes a decision; add orphan cleanup only if disk usage warrants it.
            os.replace(staged, final)
        finally:
            staged.unlink(missing_ok=True)
        return final

    def history_for(self, analysis_id: str, index: int) -> dict[str, Any]:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        with self._database() as connection:
            element, revision = self._current_element(connection, analysis_id, index)
            original = self._read_analysis(self.annotations_dir / analysis_id, {}, [], set())
            initial = self._find_element([original] if original else [], analysis_id, index)
            rows = connection.execute(
                "SELECT revision, action, state_json FROM review_history WHERE key = ? ORDER BY revision",
                (review_key(analysis_id, index),),
            ).fetchall()
        history = [{
            "revision": 0, "action": "submitted", "status": "pending",
            "class_name": initial["class_name"], "bbox": initial["bbox"],
            "note": initial.get("note"), "crop_path": initial["crop_path"],
            "source_fingerprint": initial["base_source_fingerprint"],
        }]
        history.extend({"revision": row["revision"], "action": row["action"], **json.loads(row["state_json"])} for row in rows)
        return {"status": "ok", "analysis_id": analysis_id, "index": index, "revision": revision, "history": history}

    def restore_element(
        self, analysis_id: str, index: int, *, target_revision: int,
        expected_revision: int | None = None, reviewer_id: str | None = None,
        allow_self_review: bool = False,
    ) -> dict[str, Any]:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        if isinstance(target_revision, bool) or not isinstance(target_revision, int) or target_revision < 0:
            raise AnnotationReviewValidationError("target_revision must be a non-negative integer")
        if not allow_self_review:
            self._assert_not_submitter(analysis_id, reviewer_id)
        with self._database(write=True) as connection:
            element, revision = self._current_element(connection, analysis_id, index)
            self._check_revision(revision, expected_revision)
            if target_revision == 0:
                original = self._read_analysis(self.annotations_dir / analysis_id, {}, [], set())
                target = self._find_element([original] if original else [], analysis_id, index)
            else:
                row = connection.execute(
                    "SELECT state_json FROM review_history WHERE key = ? AND revision = ?",
                    (review_key(analysis_id, index), target_revision),
                ).fetchone()
                target = json.loads(row["state_json"]) if row else None
            if target is None:
                raise AnnotationReviewNotFoundError(f"annotation revision not found: {analysis_id}:{index}@{target_revision}")
            if target.get("source_fingerprint", target.get("base_source_fingerprint")) != element["base_source_fingerprint"]:
                raise AnnotationReviewConflictError("submitted source changed; cannot restore this revision")
            analysis_dir = self.annotations_dir / analysis_id
            crop = Path(target["crop_path"]) if target.get("crop_path") else analysis_dir / "elements" / f"{index}.png"
            if not crop.is_absolute():
                crop = analysis_dir / crop
            if not _is_relative_to(crop, analysis_dir):
                raise AnnotationReviewConflictError("saved crop path is unsafe; cannot restore")
            if not crop.is_file():
                raise AnnotationReviewConflictError("saved crop is missing; cannot restore this revision")
            if target.get("crop_sha256") and _file_sha256(crop) != target["crop_sha256"]:
                raise AnnotationReviewConflictError("saved crop changed; cannot restore this revision")
            updated = {**element, "class_name": target["class_name"], "bbox": target["bbox"], "note": target.get("note"), "crop_path": str(crop)}
            self._write_decision(connection, updated, revision + 1, "pending", reviewer_id, "restore")
        return self._mutation_result(analysis_id, index, include_counts=True)

    def list_queue(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        decisions = manifest["decisions"]
        review_mode = "sqlite" if self.db_path.is_file() else "legacy_readonly" if self.manifest_path.is_file() else "empty"
        analyses: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        counts = {"total": 0, "pending": 0, "approved": 0, "rejected": 0, "trainable": 0}

        if not self.annotations_dir.exists():
            self.annotations_dir.mkdir(parents=True, exist_ok=True)

        for analysis_dir in sorted(self.annotations_dir.iterdir(), key=lambda p: p.name):
            if not analysis_dir.is_dir() or analysis_dir.name.startswith("."):
                continue
            analysis = self._read_analysis(analysis_dir, decisions, diagnostics, seen_keys)
            if analysis is None:
                continue
            analyses.append(analysis)
            for element in analysis["elements"]:
                counts["total"] += 1
                counts[element["review_status"]] += 1
                if element["trainable"]:
                    counts["trainable"] += 1

        for key, decision in sorted(decisions.items()):
            if key not in seen_keys:
                diagnostics.append(
                    _diagnostic(
                        "orphan_decision",
                        "Review decision has no matching canonical annotation and is not trainable.",
                        analysis_id=decision.get("analysis_id"),
                        index=decision.get("index"),
                        key=key,
                    )
                )

        return {
            "status": "ok",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "local_only": True,
            "warning": LOCAL_ONLY_WARNING,
            "counts": counts,
            "review_store": {
                "mode": review_mode,
                "legacy_decisions": len(decisions) if review_mode == "legacy_readonly" else 0,
            },
            "analyses": analyses,
            "diagnostics": diagnostics,
        }

    def set_status(
        self, analysis_id: str, index: int, status: str, *,
        reviewer_id: str | None = None, allow_self_review: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        _validate_status(status)
        if not allow_self_review:
            self._assert_not_submitter(analysis_id, reviewer_id)
        with self._database(write=True) as connection:
            element, revision = self._current_element(connection, analysis_id, index)
            self._check_revision(revision, expected_revision)
            self._write_decision(connection, element, revision + 1, status, reviewer_id, "review")
        return self._mutation_result(analysis_id, index)

    def modify_element(
        self, analysis_id: str, index: int, *,
        class_name: str, bbox: Sequence[int | float], status: str = "pending",
        note: str | None | object = _UNSET,
        reviewer_id: str | None = None, allow_self_review: bool = False,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        _validate_status(status)
        if not allow_self_review:
            self._assert_not_submitter(analysis_id, reviewer_id)
        if not isinstance(class_name, str):
            raise AnnotationReviewValidationError("class_name must be a string")
        try:
            clean_class = sanitize_class_name(class_name)
            integer_bbox = normalize_bbox_to_int_pixels(bbox)
            clean_note = sanitize_note(note) if note is not _UNSET else _UNSET
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnnotationReviewValidationError(str(exc)) from exc
        with self._database(write=True) as connection:
            element, revision = self._current_element(connection, analysis_id, index)
            self._check_revision(revision, expected_revision)
            image_path = self.image_path_for(analysis_id)
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                try:
                    box = list(clamp_bbox(integer_bbox, *source.size))
                except ValueError as exc:
                    raise AnnotationReviewValidationError(str(exc)) from exc
                crop_path = element["crop_path"]
                if box != element["bbox"]:
                    crop_path = str(self._write_crop(analysis_id, index, source, box))
            updated = {**element, "class_name": clean_class, "bbox": box, "crop_path": crop_path}
            if clean_note is not _UNSET:
                updated["note"] = clean_note
            self._write_decision(connection, updated, revision + 1, status, reviewer_id, "modify")
        return self._mutation_result(analysis_id, index, include_counts=True)

    def iter_approved_annotations(self) -> Iterable[dict[str, Any]]:
        """Yield canonical annotations that are exactly approved and trainable."""
        # Training export contract: the free-text research `note` field is
        # deliberately NOT yielded here. Notes are annotator memos, not
        # training data; keep them out of the classifier dataset.
        queue = self.list_queue()
        for analysis in queue["analyses"]:
            for element in analysis["elements"]:
                if element["trainable"]:
                    yield {
                        "analysis_id": element["analysis_id"],
                        "index": element["index"],
                        "class_name": element["class_name"],
                        "bbox": element["bbox"],
                        "crop_path": element["crop_path"],
                        "image_path": analysis["image_path"],
                        "image_name": analysis.get("image_name"),
                        "uploaded_at": analysis.get("uploaded_at"),
                        "source_fingerprint": element["source_fingerprint"],
                        "revision": element["revision"],
                        "dataset_split": element["dataset_split"],
                    }

    def image_path_for(self, analysis_id: str) -> Path:
        _validate_analysis_id(analysis_id)
        analysis_dir = self.annotations_dir / analysis_id
        path = analysis_dir / "image.png"
        if _is_relative_to(analysis_dir, self.annotations_dir) and path.is_file() and _is_relative_to(path, analysis_dir):
            try:
                metadata = json.loads((analysis_dir / "metadata.json").read_text(encoding="utf-8"))
                if metadata.get("analysis_id") == analysis_id:
                    return path
            except (OSError, ValueError, AttributeError):
                pass
        raise AnnotationReviewNotFoundError(f"annotation image not found: {analysis_id}")

    def crop_path_for(self, analysis_id: str, index: int) -> Path:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        analysis_dir = self.annotations_dir / analysis_id
        if not analysis_dir.is_dir() or not _is_relative_to(analysis_dir, self.annotations_dir):
            raise AnnotationReviewNotFoundError(f"annotation crop not found: {analysis_id}:{index}")
        key = review_key(analysis_id, index)
        if self.db_path.exists():
            with closing(sqlite3.connect(self.db_path, timeout=10)) as connection, connection:
                row = connection.execute("SELECT decision_json FROM review_decisions WHERE key = ?", (key,)).fetchone()
            decision = json.loads(row[0]) if row else None
        else:
            decision = self._load_manifest()["decisions"].get(key)
        analysis = self._read_analysis(analysis_dir, {key: decision} if decision else {}, [], set(), only_index=index)
        element = self._find_element([analysis] if analysis else [], analysis_id, index)
        if element is not None and element.get("crop_exists") and _is_relative_to(Path(element["crop_path"]), analysis_dir):
            return Path(element["crop_path"])
        raise AnnotationReviewNotFoundError(f"annotation crop not found: {analysis_id}:{index}")

    def _load_manifest(self) -> dict[str, Any]:
        if self.db_path.exists():
            with closing(sqlite3.connect(self.db_path, timeout=10)) as connection, connection:
                decisions = {
                    key: json.loads(raw) for key, raw in connection.execute(
                        "SELECT key, decision_json FROM review_decisions ORDER BY key"
                    )
                }
            return {"schema_version": MANIFEST_SCHEMA_VERSION, "decisions": decisions}
        if not self.manifest_path.exists():
            return _empty_manifest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("decisions"), dict):
            raise AnnotationReviewValidationError("review manifest must contain a decisions object")
        return {"schema_version": manifest.get("schema_version", MANIFEST_SCHEMA_VERSION), "decisions": manifest["decisions"]}

    def preview_legacy_import(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"status": "no_legacy_manifest", "decisions": 0}
        raw = self.manifest_path.read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(manifest.get("decisions"), dict):
            raise AnnotationReviewValidationError("legacy review-index.json has an unsupported schema")
        for key, decision in manifest["decisions"].items():
            if not isinstance(key, str) or not isinstance(decision, dict):
                raise AnnotationReviewValidationError("legacy decision key/value is invalid")
            if key != review_key(decision.get("analysis_id"), decision.get("index")):
                raise AnnotationReviewValidationError(f"legacy decision key mismatch: {key}")
            _validate_analysis_id(decision["analysis_id"])
            _validate_index(decision["index"])
            _validate_status(decision.get("status"))
            try:
                fingerprint = decision.get("source_fingerprint")
                if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                    raise ValueError("invalid fingerprint")
                name = decision.get("class_name")
                if not isinstance(name, str) or sanitize_class_name(name) != name:
                    raise ValueError("invalid class name")
                bbox = decision.get("bbox")
                if (not isinstance(bbox, list) or len(bbox) != 4
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
                        or min(bbox[:2]) < 0 or min(bbox[2:]) <= 0):
                    raise ValueError("invalid bbox")
                if "note" in decision:
                    sanitize_note(decision["note"])
                if decision.get("dataset_split", "excluded") not in DATASET_SPLITS:
                    raise ValueError("invalid dataset split")
                if not isinstance(decision.get("reviewed_at"), str) or (
                        decision.get("reviewed_by") is not None and not isinstance(decision["reviewed_by"], str)):
                    raise ValueError("invalid review attribution")
                crop = decision.get("crop_path")
                if crop is not None:
                    if not isinstance(crop, str) or not crop:
                        raise ValueError("invalid crop path")
                    analysis_dir = self.annotations_dir / decision["analysis_id"]
                    path = Path(crop)
                    if not _is_relative_to(path if path.is_absolute() else analysis_dir / path, analysis_dir):
                        raise ValueError("crop escapes analysis")
            except (TypeError, ValueError) as exc:
                raise AnnotationReviewValidationError(f"invalid legacy decision {key}: {exc}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        return {"status": "ready", "decisions": len(manifest["decisions"]), "sha256": digest}

    def import_legacy_manifest(self) -> dict[str, Any]:
        preview = self.preview_legacy_import()
        if preview["status"] != "ready":
            return preview
        if self.db_path.exists():
            with closing(sqlite3.connect(self.db_path, timeout=10)) as connection, connection:
                row = connection.execute("SELECT value FROM review_meta WHERE key = 'legacy_sha256'").fetchone()
            if row and row[0] == preview["sha256"]:
                return {**preview, "status": "already_imported"}
            raise AnnotationReviewConflictError("review database already exists with different state; refusing import")
        raw = self.manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != preview["sha256"]:
            raise AnnotationReviewConflictError("legacy manifest changed during preview; retry")
        backup_dir = self.annotations_dir / ".review-backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"review-index-{preview['sha256']}.json"
        try:
            with backup.open("xb") as handle:
                handle.write(raw)
        except FileExistsError:
            pass
        if hashlib.sha256(backup.read_bytes()).hexdigest() != preview["sha256"]:
            raise AnnotationReviewConflictError("legacy backup checksum mismatch; refusing import")
        manifest = json.loads(raw)
        staged = self.annotations_dir / f".review-import-{uuid.uuid4().hex}.sqlite3"
        try:
            with closing(sqlite3.connect(staged, timeout=10)) as connection, connection:
                self._create_schema(connection)
                for key, old in manifest["decisions"].items():
                    decision = {**old, "revision": 1}
                    encoded = json.dumps(decision, sort_keys=True, ensure_ascii=False)
                    connection.execute("INSERT INTO review_decisions VALUES (?, ?, ?)", (key, 1, encoded))
                    connection.execute("INSERT INTO review_history VALUES (?, ?, ?, ?)", (key, 1, "legacy_import", encoded))
                connection.execute("INSERT INTO review_meta VALUES ('legacy_sha256', ?)", (preview["sha256"],))
            if hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != preview["sha256"]:
                raise AnnotationReviewConflictError("legacy manifest changed during import; retry")
            os.link(staged, self.db_path)  # Atomic no-clobber publish on the same local filesystem.
        except FileExistsError as exc:
            raise AnnotationReviewConflictError("another review database appeared during import") from exc
        finally:
            staged.unlink(missing_ok=True)
        return {**preview, "status": "imported", "backup": str(backup), "db_path": str(self.db_path)}

    def _read_analysis(
        self,
        analysis_dir: Path,
        decisions: dict[str, Any],
        diagnostics: list[dict[str, Any]],
        seen_keys: set[str],
        *,
        only_index: int | None = None,
    ) -> dict[str, Any] | None:
        metadata_path = analysis_dir / "metadata.json"
        if not metadata_path.is_file():
            diagnostics.append(
                _diagnostic(
                    "missing_metadata",
                    "Annotation folder has no metadata.json and cannot be reviewed.",
                    analysis_id=analysis_dir.name,
                )
            )
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    f"metadata.json is invalid JSON: {exc}",
                    analysis_id=analysis_dir.name,
                )
            )
            return None

        if not isinstance(metadata, dict):
            diagnostics.append(_diagnostic(
                "invalid_metadata", "metadata.json must be an object.",
                analysis_id=analysis_dir.name,
            ))
            return None

        analysis_id = str(metadata.get("analysis_id") or analysis_dir.name)
        if not _SAFE_ANALYSIS_ID.match(analysis_id) or analysis_id != analysis_dir.name:
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    "metadata.json analysis_id must be alphanumeric/dash/underscore only.",
                    analysis_id=analysis_dir.name,
                )
            )
            return None
        uploaded_at = metadata.get("uploaded_at")
        submitted_by = metadata.get("submitted_by")
        image_path = analysis_dir / "image.png"
        annotations = metadata.get("annotations", [])
        if not isinstance(annotations, list):
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    "metadata.json annotations must be a list.",
                    analysis_id=analysis_id,
                )
            )
            return None

        elements: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                diagnostics.append(
                    _diagnostic(
                        "invalid_annotation",
                        f"annotations[{position}] must be an object.",
                        analysis_id=analysis_id,
                    )
                )
                continue
            index = annotation.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                diagnostics.append(
                    _diagnostic(
                        "invalid_annotation",
                        f"annotations[{position}].index must be a non-negative integer.",
                        analysis_id=analysis_id,
                    )
                )
                continue
            if index in seen_indexes:
                diagnostics.append(
                    _diagnostic(
                        "duplicate_index",
                        "Duplicate annotation index is ignored for review/training.",
                        analysis_id=analysis_id,
                        index=index,
                        key=review_key(analysis_id, index),
                    )
                )
                continue
            seen_indexes.add(index)
            if only_index is not None and index != only_index:
                continue

            key = review_key(analysis_id, index)
            seen_keys.add(key)
            fallback_crop = analysis_dir / "elements" / f"{index}.png"
            crop_path = _resolve_under_analysis_dir(
                annotation.get("crop_path"),
                analysis_dir,
                fallback_crop,
            )
            base_fingerprint = _source_fingerprint(
                analysis_id=analysis_id,
                uploaded_at=uploaded_at if isinstance(uploaded_at, str) else None,
                annotation=annotation,
                crop_path=crop_path,
            )
            decision = decisions.get(key) if isinstance(decisions.get(key), dict) else None
            review_status = "pending"
            stale_decision = False
            effective_annotation = annotation
            if decision:
                decision_status = decision.get("status")
                if decision_status in REVIEW_STATUSES:
                    if decision.get("source_fingerprint") == base_fingerprint:
                        review_status = decision_status
                        effective_annotation = {
                            **annotation,
                            "class_name": decision.get("class_name", annotation.get("class_name")),
                            "bbox": decision.get("bbox", annotation.get("bbox")),
                            "note": decision.get("note", annotation.get("note")),
                        }
                        if decision.get("crop_path"):
                            candidate = Path(decision["crop_path"])
                            if not candidate.is_absolute():
                                candidate = analysis_dir / candidate
                            if _is_relative_to(candidate, analysis_dir):
                                crop_path = candidate
                            else:
                                stale_decision = True
                                diagnostics.append(_diagnostic(
                                    "unsafe_crop_path", "Review crop path escapes its analysis folder.",
                                    analysis_id=analysis_id, index=index, key=key,
                                ))
                    else:
                        stale_decision = True
                        diagnostics.append(
                            _diagnostic(
                                "stale_decision",
                                "Review decision source fingerprint no longer matches canonical annotation; treating as pending and not trainable.",
                                analysis_id=analysis_id,
                                index=index,
                                key=key,
                            )
                        )

            fingerprint = base_fingerprint if not decision else _source_fingerprint(
                analysis_id=analysis_id,
                uploaded_at=uploaded_at if isinstance(uploaded_at, str) else None,
                annotation=effective_annotation,
                crop_path=crop_path,
            )
            crop_exists = crop_path.is_file()
            if crop_exists and decision and decision.get("crop_sha256") and _file_sha256(crop_path) != decision["crop_sha256"]:
                stale_decision = True
                diagnostics.append(_diagnostic(
                    "stale_crop", "Reviewed crop changed since its decision; element is not trainable.",
                    analysis_id=analysis_id, index=index, key=key,
                ))
            if not crop_exists:
                diagnostics.append(
                    _diagnostic(
                        "missing_crop",
                        "Canonical crop file is missing; element is not trainable.",
                        analysis_id=analysis_id,
                        index=index,
                        key=key,
                    )
                )

            trainable = review_status == TRAINABLE_STATUS and crop_exists and not stale_decision
            split_payload = _split_payload_for_element(
                key=key,
                review_status=review_status,
                crop_exists=crop_exists,
                stale_decision=stale_decision,
                decision=decision,
            )
            elements.append(
                {
                    "key": key,
                    "analysis_id": analysis_id,
                    "index": index,
                    "class_name": effective_annotation.get("class_name", ""),
                    "bbox": effective_annotation.get("bbox", []),
                    "note": effective_annotation.get("note") if isinstance(effective_annotation.get("note"), str) else None,
                    "crop_path": str(crop_path),
                    "crop_url": f"/admin/annotations/{analysis_id}/{index}/crop",
                    "crop_exists": crop_exists,
                    "review_status": review_status,
                    "trainable": trainable,
                    "source_fingerprint": fingerprint,
                    "base_source_fingerprint": base_fingerprint,
                    "revision": decision.get("revision", 0) if decision else 0,
                    "stale_decision": stale_decision,
                    **split_payload,
                }
            )

        return {
            "analysis_id": analysis_id,
            "image_name": metadata.get("image_name") if isinstance(metadata.get("image_name"), str) else None,
            "uploaded_at": uploaded_at,
            "image_path": str(image_path),
            "image_url": f"/admin/annotations/{analysis_id}/image",
            "image_exists": image_path.is_file(),
            "elements": elements,
        }

    def _assert_not_submitter(self, analysis_id: str, reviewer_id: str | None) -> None:
        """Legacy submissions without an author remain reviewable; known authors do not self-review."""
        if reviewer_id is None:
            return
        metadata_path = self.annotations_dir / analysis_id / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AnnotationReviewNotFoundError(f"annotation metadata not found: {analysis_id}") from exc
        except json.JSONDecodeError as exc:
            raise AnnotationReviewValidationError(f"metadata.json is invalid JSON: {exc}") from exc
        if metadata.get("submitted_by") == reviewer_id:
            raise AnnotationReviewValidationError("self-review is not permitted")

    @staticmethod
    def _find_element(
        analyses: list[dict[str, Any]],
        analysis_id: str,
        index: int,
    ) -> dict[str, Any] | None:
        for analysis in analyses:
            if analysis.get("analysis_id") != analysis_id:
                continue
            for element in analysis.get("elements", []):
                if element.get("index") == index:
                    return element
        return None

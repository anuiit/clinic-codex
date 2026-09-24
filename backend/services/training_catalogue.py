"""Confirmed training classes. Runtime taxonomy stays untouched until explicit deployment."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path

from backend.services.annotation_storage import sanitize_class_name


class CatalogueConflict(ValueError):
    pass


class CatalogueStaleConflict(CatalogueConflict):
    pass


class CatalogueSimilarConflict(CatalogueConflict):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def base_names(prior_path: Path) -> dict[int, str]:
    import torch
    from backend.codex_pipeline.scripts.export_model import _prototype_contract
    prior = torch.load(prior_path, map_location="cpu", weights_only=True)
    return _prototype_contract(prior, label="base taxonomy")[1]


def _confirmed(db_path: Path, connection=None) -> dict[int, str]:
    if connection is None:
        if not db_path.is_file():
            return {}
        with closing(sqlite3.connect(db_path)) as connection:
            return _confirmed(db_path, connection)
    table = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_classes'").fetchone()
    if not table:
        return {}
    return dict(connection.execute("SELECT class_label, class_name FROM training_classes ORDER BY class_label"))


def taxonomy(store, prior_path: Path, connection=None) -> dict:
    base = base_names(prior_path)
    added = _confirmed(store.db_path, connection)
    if set(base).intersection(added) or set(base.values()).intersection(added.values()):
        raise ValueError("confirmed classes conflict with the installed base taxonomy")
    if added and min(added) <= max(base, default=-1):
        raise ValueError("new class IDs must preserve the historical numeric contract")
    names = {**base, **added}
    return {"base_names": base, "new_names": added, "names": names,
            "revision": digest(sorted(names.items()))}


def catalogue(store, prior_path: Path) -> dict:
    state = taxonomy(store, prior_path)
    by_name = {name: label for label, name in state["names"].items()}
    counts = {}
    trainable = Counter()
    for analysis in store.list_queue()["analyses"]:
        for element in analysis["elements"]:
            name = element["class_name"]
            counts.setdefault(name, Counter())[element["review_status"]] += 1
            if element["trainable"]:
                trainable[name] += 1
    active = set(state["base_names"].values())
    rows = []
    for name in sorted(set(by_name) | set(counts), key=str.casefold):
        rows.append({"class_name": name, "class_label": by_name.get(name),
                     "status": "active" if name in active else "candidate" if name in by_name else "unconfirmed",
                     "counts": {status: counts.get(name, {}).get(status, 0) for status in ("pending", "approved", "rejected")},
                     "trainable_count": trainable[name] if name in by_name else 0})
    return {"revision": state["revision"], "classes": rows}


def confirm_class(store, prior_path: Path, name: object, expected_revision: object, actor_id=None) -> dict:
    from datetime import datetime, timezone
    if not isinstance(name, str):
        raise ValueError("class_name must be a string")
    name = sanitize_class_name(unicodedata.normalize("NFC", name))
    if len(name) > 120 or any(unicodedata.category(char).startswith("C") for char in name):
        raise ValueError("class_name must be at most 120 characters without control characters")
    store.ensure_database()
    with closing(sqlite3.connect(store.db_path, timeout=30)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        state = taxonomy(store, prior_path, connection)
        if expected_revision != state["revision"]:
            raise CatalogueStaleConflict("Le catalogue a changé. Actualisez avant de confirmer la classe.")
        normalized = {unicodedata.normalize("NFC", value).casefold(): value for value in state["names"].values()}
        if name.casefold() in normalized:
            if normalized[name.casefold()] != name:
                raise CatalogueSimilarConflict("Une classe similaire existe déjà : " + normalized[name.casefold()])
            return {"class_name": name, "class_label": next(key for key, value in state["names"].items() if value == name)}
        connection.execute("""CREATE TABLE IF NOT EXISTS training_classes (
            class_label INTEGER PRIMARY KEY, class_name TEXT NOT NULL UNIQUE,
            confirmed_by TEXT, confirmed_at TEXT NOT NULL)""")
        label = max(state["names"], default=-1) + 1
        connection.execute("INSERT INTO training_classes VALUES (?, ?, ?, ?)",
                           (label, name, actor_id, datetime.now(timezone.utc).isoformat()))
    return {"class_name": name, "class_label": label}


def validate_local_base(prior_path: Path, base_model_dir: Path | None = None):
    """Same fixed-prior contract for the admin preflight and the CLI."""
    import torch
    from backend.codex_pipeline.scripts.export_model import _prototype_contract
    base_model_dir = base_model_dir or prior_path.parent.parent / "codex_model"
    prior = torch.load(prior_path, map_location="cpu", weights_only=True)
    ordered, names = _prototype_contract(prior, label="shipped base")
    active = torch.load(base_model_dir / "weights/prototypes.pt", map_location="cpu", weights_only=True)
    active_labels, active_names = _prototype_contract(active, label="active base")
    if active_labels != ordered or active_names != names or not torch.equal(active["prototypes"], prior["prototypes"]):
        raise ValueError("installed prototypes differ from the shipped prior; use the advanced snapshot workflow")
    projection = torch.load(base_model_dir / "weights/projection.pt", map_location="cpu", weights_only=True)
    if projection.keys() != prior["model_state_dict"].keys() or any(
            not torch.equal(projection[key], prior["model_state_dict"][key]) for key in projection):
        raise ValueError("installed projection differs from the shipped prior")
    config = json.loads((base_model_dir / "config.json").read_text(encoding="utf-8"))
    if (config["backbone"] != "dinov2_vits14" or config["image_size"] != 224
            or config["hidden_dim"] != prior["hidden_dim"] or config["embedding_dim"] != prior["embedding_dim"]
            or config["class_names"] != [names[label] for label in ordered]
            or config["num_classes"] != len(names)):
        raise ValueError("installed configuration is incompatible with the shipped prior")
    return prior, active, projection, config


def training_data_state(store, prior_path: Path) -> dict:
    validate_local_base(prior_path)
    state = taxonomy(store, prior_path)
    records = list(store.iter_approved_annotations())
    known = set(state["names"].values())
    unknown = sorted({row["class_name"] for row in records} - known)
    semantic = [{key: row.get(key) for key in ("analysis_id", "index", "class_name", "bbox", "source_fingerprint")}
                for row in records]
    backend_root = prior_path.parent.parent
    model_paths = [prior_path, backend_root / "codex_model/config.json",
                   backend_root / "codex_model/weights/projection.pt", backend_root / "codex_model/weights/prototypes.pt"]
    model_revision = digest([(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in model_paths])
    state.update({"model_revision": model_revision, "records": records, "errors": [
        "Classe à confirmer dans Classes : " + name for name in unknown
    ], "data_revision": digest({"taxonomy": state["revision"], "model": model_revision, "annotations": sorted(semantic, key=lambda row: (row["analysis_id"], row["index"]))})})
    return state

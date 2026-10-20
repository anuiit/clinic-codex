"""Small checks for new-class confirmation, stable IDs and honest comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from backend.services.annotation_review import AnnotationReviewStore
from backend.services.annotation_storage import save_annotation
from backend.services.training_catalogue import CatalogueSimilarConflict, CatalogueStaleConflict, catalogue, confirm_class, taxonomy, training_data_state
from backend.codex_pipeline.scripts.export_model import _preserve_base_numeric_contract
from scripts import retrain_local as local
from scripts.compare_local_models import candidate_exposure, capture_samples, metrics


def prior():
    return {"prototypes": torch.tensor([[1., 0.], [0., 1.]]), "class_labels": torch.tensor([2, 8]),
            "class_names": {2: "atl", 8: "calli"},
            "class_meta": {2: {"count": 2, "variance": .2}, 8: {"count": 1}}}


def test_new_prototype_preserves_sparse_contract_and_export_is_opt_in(tmp_path):
    original = prior()
    vector = torch.tensor([[.6, .8]])
    with pytest.raises(ValueError):
        local.merge_prior(original, vector, [9])
    merged = local.merge_prior(original, vector, [9], {9: "tzapotl"})
    assert merged["class_labels"].tolist() == [2, 8, 9]
    assert merged["class_names"] == {2: "atl", 8: "calli", 9: "tzapotl"}
    assert torch.equal(merged["prototypes"][:2], original["prototypes"])
    assert torch.equal(local.merge_prior(original, vector, [9], {9: "tzapotl"})["prototypes"], merged["prototypes"])
    assert merged["class_meta"][9]["count"] == 1
    base = tmp_path / "base"
    (base / "weights").mkdir(parents=True)
    torch.save(original, base / "weights/prototypes.pt")
    before = (base / "weights/prototypes.pt").read_bytes()
    with pytest.raises(ValueError, match="taxonomies differ"):
        _preserve_base_numeric_contract(merged, base_model_dir=base)
    exported, contract = _preserve_base_numeric_contract(merged, base_model_dir=base, allow_new_classes=True)
    assert contract["preserved"]
    assert exported["class_labels"].tolist() == [2, 8, 9]
    assert (base / "weights/prototypes.pt").read_bytes() == before
    with pytest.raises(ValueError):
        local.merge_prior(original, vector, [5], {5: "tzapotl"})


def test_two_confirmed_dataset_classes_join_one_candidate(tmp_path):
    original = prior()
    prior_path = tmp_path / "backend/prototypes/prototypes.pt"
    prior_path.parent.mkdir(parents=True)
    torch.save(original, prior_path)
    store = AnnotationReviewStore(tmp_path / "annotations")
    for index, (name, color) in enumerate((("tzapotl", "red"), ("teocomitl", "blue"))):
        save_annotation(f"page-{index}", Image.new("RGB", (20, 20), color),
                        [{"index": 0, "class_name": name, "bbox": [0, 0, 20, 20]}],
                        base_dir=store.annotations_dir, elements_dir=tmp_path / "unused")
        store.set_status(f"page-{index}", 0, "approved")
        confirm_class(store, prior_path, name, taxonomy(store, prior_path)["revision"])
    names = taxonomy(store, prior_path)["names"]
    snapshot = local.capture_approvals(store, tmp_path / "snapshot", names)
    train = [row for row in snapshot["rows"] if row["dataset_split"] == "train"]
    assert {row["class_label"] for row in train} == {9, 10}
    vectors = {9: torch.tensor([.8, .6]), 10: torch.tensor([-.6, .8])}
    merged = local.merge_prior(original, torch.stack([vectors[row["class_label"]] for row in train]),
                               [row["class_label"] for row in train], {9: names[9], 10: names[10]})
    assert merged["class_names"] == {2: "atl", 8: "calli", 9: "tzapotl", 10: "teocomitl"}
    assert torch.equal(merged["prototypes"][:2], original["prototypes"])
    assert all(merged["class_meta"][label]["count"] == 1 for label in (9, 10))
    base = tmp_path / "base/weights"
    base.mkdir(parents=True)
    torch.save(original, base / "prototypes.pt")
    exported, _ = _preserve_base_numeric_contract(merged, base_model_dir=base.parent, allow_new_classes=True)
    assert exported["class_labels"].tolist() == [2, 8, 9, 10]


def test_catalogue_confirmation_revision_and_model_data_revision(tmp_path):
    backend = tmp_path / "backend"
    prior_path = backend / "prototypes/prototypes.pt"
    prior_path.parent.mkdir(parents=True)
    data = {**prior(), "model_state_dict": {"weight": torch.eye(2)}, "hidden_dim": 2, "embedding_dim": 2}
    torch.save(data, prior_path)
    weights = backend / "codex_model/weights"
    weights.mkdir(parents=True)
    torch.save(prior(), weights / "prototypes.pt")
    torch.save(data["model_state_dict"], weights / "projection.pt")
    (backend / "codex_model/config.json").write_text(json.dumps({
        "backbone": "dinov2_vits14", "image_size": 224, "hidden_dim": 2, "embedding_dim": 2,
        "class_names": ["atl", "calli"], "num_classes": 2}))
    store = AnnotationReviewStore(backend / "annotations")
    save_annotation("new-class", Image.new("RGB", (20, 20), "red"),
                    [{"index": 0, "class_name": "tzapotl", "bbox": [0, 0, 20, 20]}],
                    base_dir=store.annotations_dir, elements_dir=tmp_path / "unused", image_name="page originale.png")
    store.set_status("new-class", 0, "approved")
    blocked = training_data_state(store, prior_path)
    assert blocked["errors"] == ["Classe à confirmer dans Classes : tzapotl"]
    unconfirmed = next(row for row in catalogue(store, prior_path)["classes"] if row["class_name"] == "tzapotl")
    assert unconfirmed["status"] == "unconfirmed" and unconfirmed["trainable_count"] == 0
    old_revision = taxonomy(store, prior_path)["revision"]
    assert confirm_class(store, prior_path, "tzapotl", old_revision)["class_label"] == 9
    with pytest.raises(CatalogueStaleConflict):
        confirm_class(store, prior_path, "different", old_revision)
    revision = taxonomy(store, prior_path)["revision"]
    with pytest.raises(CatalogueSimilarConflict):
        confirm_class(store, prior_path, "Tzapotl", revision)
    state = training_data_state(store, prior_path)
    assert not state["errors"] and state["data_revision"] != blocked["data_revision"]
    torch.save({"weight": torch.ones(2, 2)}, backend / "codex_model/weights/projection.pt")
    with pytest.raises(ValueError, match="projection differs"):
        training_data_state(store, prior_path)


def test_snapshot_holds_out_sources_and_comparison_keeps_names_and_bytes(tmp_path):
    store = AnnotationReviewStore(tmp_path / "annotations")
    for page in range(4):
        image = Image.new("RGB", (24, 24), (page * 50, 30, 150))
        save_annotation(f"page-{page}", image,
                        [{"index": 0, "class_name": "atl", "bbox": [0, 0, 12, 24]},
                         {"index": 1, "class_name": "atl", "bbox": [12, 0, 12, 24]}],
                        base_dir=store.annotations_dir, elements_dir=tmp_path / "unused",
                        image_name=f"source-{page}.png")
        for index in (0, 1):
            store.set_status(f"page-{page}", index, "approved")
    annotations = tmp_path / "candidate/annotations"
    snapshot = local.capture_approvals(store, annotations, {2: "atl", 8: "calli"})
    assert snapshot["holdout_count"] > 0
    assert snapshot["duplicate_count"] == 4
    for group in {row["source_group"] for row in snapshot["rows"]}:
        assert len({row["dataset_split"] for row in snapshot["rows"] if row["source_group"] == group}) == 1
    run = tmp_path / "comparison"
    rows = capture_samples(SimpleNamespace(analysis_id=None), tmp_path / "candidate/runtime", run)
    assert len(rows) == 4
    for row, source in zip(rows, snapshot["rows"]):
        assert (run / row["crop_file"]).read_bytes() == (annotations / source["snapshot_crop"]).read_bytes()
        assert row["image_name"].startswith("source-")
        assert row["scope"] in ("train", "locked_test")
    live = capture_samples(SimpleNamespace(analysis_id="page-0", annotations_dir=store.annotations_dir,
                                           review_manifest=store.manifest_path), Path("unused"), tmp_path / "ad-hoc")
    assert len(live) == 2 and all(row["scope"] == "ad_hoc" for row in live)


def test_comparison_separates_new_classes_from_common_and_rejections():
    def result(name, rejected=False):
        return {"class_name": name, "rejected": rejected, "top_k": [{"class_name": name}]}
    rows = [
        {"expected_class": "atl", "base_supported": True, "base": result("calli"),
         "candidate": result("atl"), "outcome": "gain"},
        {"expected_class": "tzapotl", "base_supported": False, "base": result("atl"),
         "candidate": result("tzapotl"), "outcome": "new_class"},
        {"expected_class": None, "base_supported": False, "base": result("atl"),
         "candidate": result("atl", True), "outcome": "disagreement"},
    ]
    report = metrics(rows)
    assert report["common"]["support"] == 1 and report["common"]["gains"] == 1
    assert report["new_classes"] == {"support": 1, "candidate_correct": 1}
    assert report["coverage"]["candidate"] == pytest.approx(2/3)


def test_comparison_marks_candidate_page_exposure_by_exact_snapshot_hash(tmp_path):
    runtime = tmp_path / "candidate/runtime"
    manifest = runtime.parent / "annotations/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "schema_version": "local-approved-snapshot.v2",
        "rows": [{"source_image_sha256": "seen-page", "dataset_split": "train"}],
    }), encoding="utf-8")
    rows = [
        {"scope": "ad_hoc", "source_image_sha256": "seen-page"},
        {"scope": "ad_hoc", "source_image_sha256": "new-page"},
        {"scope": "locked_test", "source_image_sha256": "held-out"},
    ]
    candidate_exposure(rows, runtime)
    assert [row["candidate_exposure"] for row in rows] == ["train", "unseen_exact", "locked_test"]

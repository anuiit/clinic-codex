import io
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass

from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app
from backend.app.services.container import DefaultServices
from backend.services.annotation_review import AnnotationReviewStore
from backend.services.annotation_storage import save_annotation
from backend.services.approved_corrections import approved_page_corrections


def save(store, image, name="atl", analysis="page", bbox=None):
    save_annotation(analysis, image, [{"index": 0, "class_name": name, "bbox": bbox or [1, 1, 3, 3]}],
                    base_dir=store.annotations_dir, elements_dir=store.annotations_dir)
    store.set_status(analysis, 0, "approved")


def test_exact_pixels_replay_revocation_and_changed_page(tmp_path):
    store = AnnotationReviewStore(tmp_path / "annotations")
    image = Image.new("RGB", (8, 6), (120, 130, 140))
    save(store, image)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=0)
    assert approved_page_corrections(store, Image.open(io.BytesIO(encoded.getvalue())))["elements"][0]["class_name"] == "atl"
    changed = image.copy()
    changed.putpixel((7, 5), (1, 2, 3))
    assert not approved_page_corrections(store, changed)["elements"]
    # A changed canonical page cannot borrow an old approval, even outside its crop.
    changed.save(store.annotations_dir / "page" / "image.png")
    assert not approved_page_corrections(store, changed)["elements"]
    image.save(store.annotations_dir / "page" / "image.png")
    store.set_status("page", 0, "rejected")
    assert not approved_page_corrections(store, image)["elements"]


def test_conflicts_deduplication_and_legacy_crop_verification(tmp_path):
    store = AnnotationReviewStore(tmp_path / "annotations")
    image = Image.new("RGB", (8, 6), (120, 130, 140))
    save(store, image, analysis="first")
    save(store, image, analysis="duplicate")
    assert len(approved_page_corrections(store, image)["elements"]) == 1
    save(store, image, name="tetl", analysis="conflict")
    result = approved_page_corrections(store, image)
    assert result["conflicts"] == [[1, 1, 3, 3]] and not result["elements"]
    store.set_status("conflict", 0, "rejected")
    store.set_status("duplicate", 0, "rejected")
    # Legacy records lack the global page digest; still verify the actual crop pixels.
    with closing(sqlite3.connect(store.db_path)) as connection, connection:
        (raw,) = connection.execute("SELECT decision_json FROM review_decisions WHERE key = 'first:0'").fetchone()
        decision = json.loads(raw)
        decision.pop("source_image_pixel_sha256")
        connection.execute("UPDATE review_decisions SET decision_json = ? WHERE key = 'first:0'", (json.dumps(decision),))
    assert len(approved_page_corrections(store, image)["elements"]) == 1
    Image.new("RGB", (3, 3), "red").save(store.annotations_dir / "first" / "elements" / "0.png")
    store.set_status("first", 0, "approved")
    assert not approved_page_corrections(store, image)["elements"]


@dataclass
class Proposal:
    bbox: list
    crop: object


def test_segment_replays_missed_corrections_keeps_other_regions_and_raw_mode(tmp_path, monkeypatch):
    settings = Settings(testing=True, backend_root=tmp_path)
    services = DefaultServices(settings)
    image = Image.new("RGB", (8, 6), (120, 130, 140))
    save(services.annotation_review_store(), image)
    provenance = {"artifact_id": "actual-loaded-model", "model_version": "v1", "version_id": "runtime-id", "sha256": {}}
    classified_sizes = []
    def classify(images):
        classified_sizes.append(len(images))
        return [{"class_name": "wrong", "confidence": .6, "rejected": False, "top_k": [],
                 "model_provenance": provenance} for _ in images]
    monkeypatch.setattr(services, "classify_batch", classify)
    monkeypatch.setattr(services, "segmenter_provenance", lambda: {"name": "MobileSAM", "sha256": "loaded-sam", "points_per_side": 16})
    proposals = [Proposal([1, 1, 3, 3], image), Proposal([5, 1, 2, 2], image)]
    monkeypatch.setattr(services, "segment_page", lambda _: proposals)
    app = create_app(settings=settings, services=services)
    def post(client, **fields):
        data = io.BytesIO(); image.save(data, format="PNG"); data.seek(0)
        return client.post("/segment", data={"image": (data, "page.png"), **fields})
    with app.test_client() as client:
        response = post(client)
        assert response.status_code == 200
        body = response.get_json()
        assert [e["class_name"] for e in body["elements"]] == ["atl", "wrong"]
        assert classified_sizes == [1]
        assert body["elements"][0]["source"] == "approved_correction"
        assert body["elements"][0]["confidence_kind"] == "human_approval"
        assert body["elements"][0]["review"]["decision_id"] == "page:0"
        assert body["provenance"]["classifier_models"] == [provenance]
        assert body["provenance"]["segmenter"]["sha256"] == "loaded-sam"
        # A manual box is restored even when SAM never proposes it.
        proposals.pop(0)
        assert post(client).get_json()["elements"][0]["bbox"] == [1, 1, 3, 3]
        raw = post(client, use_corrections="false").get_json()
        assert raw["provenance"]["correction_count"] == 0
        assert all(e["source"] == "model" for e in raw["elements"])
        assert post(client, use_corrections="bad").status_code == 400
        save(services.annotation_review_store(), image, name="tetl", analysis="conflict")
        conflict = post(client)
        assert conflict.status_code == 409
        assert conflict.get_json()["error"]["code"] == "APPROVED_CORRECTION_CONFLICT"

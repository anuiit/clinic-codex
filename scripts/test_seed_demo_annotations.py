from backend.services.annotation_review import AnnotationReviewStore
from scripts.seed_demo_annotations import seed


def test_seed_is_idempotent_and_never_approves(tmp_path):
    seed(tmp_path)
    seed(tmp_path)

    queue = AnnotationReviewStore(tmp_path).list_queue()
    assert queue["counts"] == {
        "total": 6, "pending": 6, "approved": 0, "rejected": 0, "trainable": 0,
    }
    assert all(analysis["image_name"] for analysis in queue["analyses"])

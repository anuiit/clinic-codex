from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request
from backend.security.auth import current_user, require_csrf, require_permission
from backend.security.local_guard import require_local_request
from backend.services.annotation_review import AnnotationReviewConflictError, AnnotationReviewMigrationRequiredError
from backend.services.training_catalogue import CatalogueConflict, CatalogueSimilarConflict, CatalogueStaleConflict, catalogue, confirm_class, taxonomy

bp = Blueprint("classes", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _catalogue_inputs():
    services = _services()
    return services.annotation_review_store(), services.settings.backend_root / "prototypes/prototypes.pt"


@bp.get("/classes")
def get_classes():
    config = _services().load_classes()
    return jsonify({"num_classes": config["num_classes"], "class_names": config["class_names"]})


@bp.get("/annotation-classes")
@require_permission("analysis.submit")
def get_annotation_classes():
    try:
        store, prior = _catalogue_inputs()
        names = list(taxonomy(store, prior)["names"].values())
        return jsonify({"num_classes": len(names), "class_names": names})
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 503


@bp.get("/admin/classes")
@require_local_request
@require_permission("annotation.queue.read")
def get_admin_classes():
    try:
        return jsonify(catalogue(*_catalogue_inputs()))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 503


@bp.post("/admin/classes")
@require_local_request
@require_permission("annotation.review")
@require_csrf
def confirm_admin_class():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid JSON object"}), 400
    try:
        store, prior = _catalogue_inputs()
        actor = current_user()
        confirm_class(store, prior, data.get("class_name"), data.get("expected_revision"),
                      actor_id=actor["id"] if actor else None)
        return jsonify(catalogue(store, prior)), 201
    except AnnotationReviewMigrationRequiredError as exc:
        return jsonify({"error": str(exc), "error_code": "REVIEW_MIGRATION_REQUIRED"}), 409
    except CatalogueStaleConflict as exc:
        return jsonify({"error": str(exc), "error_code": "CATALOGUE_STALE"}), 409
    except CatalogueSimilarConflict as exc:
        return jsonify({"error": str(exc), "error_code": "CLASS_NAME_SIMILAR"}), 409
    except (CatalogueConflict, AnnotationReviewConflictError) as exc:
        return jsonify({"error": str(exc)}), 409
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": str(exc)}), 503

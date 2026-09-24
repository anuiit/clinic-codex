from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.training_jobs import (
    AdminTrainingConflictError,
    AdminTrainingForbiddenError,
    AdminTrainingValidationError,
    RequestLaunchContext,
)
from backend.security.auth import current_user, require_csrf, require_permission
from backend.security.local_guard import request_launch_context, require_local_request

bp = Blueprint("admin_training", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "error": message}), status_code


def _request_context() -> RequestLaunchContext:
    return request_launch_context()


@bp.get("/admin/training/summary")
@require_local_request
@require_permission("training.read")
def get_admin_training_summary():
    return jsonify(_services().admin_training_summary(_request_context())), 200


@bp.get("/admin/training/jobs/latest")
@require_local_request
@require_permission("training.read")
def get_latest_admin_training_job():
    job = _services().latest_admin_training_job()
    if job is None:
        return jsonify({"status": "ok", "local_only": True, "job": None}), 200
    return jsonify({"status": "ok", "local_only": True, "job": job}), 200


@bp.get("/admin/training/jobs/<run_id>")
@require_local_request
@require_permission("training.read")
def get_admin_training_job(run_id: str):
    job = _services().get_admin_training_job(run_id)
    if job is None:
        return _error(f"training job not found: {run_id}", 404)
    return jsonify({"status": "ok", "local_only": True, "job": job}), 200


@bp.post("/admin/training/jobs")
@require_local_request
@require_permission("training.run")
@require_csrf
def start_admin_training_job():
    data = request.get_json(silent=True)
    if data is None:
        return _error("invalid JSON", 400)
    try:
        actor = current_user()
        job = _services().start_admin_training_job(data, _request_context(), **({"actor_id": actor["id"]} if actor else {}))
    except AdminTrainingForbiddenError as exc:
        return _error(str(exc), 403)
    except AdminTrainingValidationError as exc:
        return _error(str(exc), 400)
    except AdminTrainingConflictError as exc:
        return _error(str(exc), 409)
    return jsonify({"status": "ok", "local_only": True, "job": job}), 202


@bp.get("/admin/training/models")
@require_local_request
@require_permission("training.read")
def get_comparable_models():
    return jsonify(_services().admin_training_service().models())


@bp.post("/admin/training/comparisons")
@require_local_request
@require_permission("training.run")
@require_csrf
def start_model_comparison():
    try:
        actor = current_user()
        job = _services().admin_training_service().start_comparison(
            request.get_json(silent=True), _request_context(), actor_id=actor["id"] if actor else None)
    except AdminTrainingForbiddenError as exc:
        return _error(str(exc), 403)
    except AdminTrainingValidationError as exc:
        return _error(str(exc), 400)
    except AdminTrainingConflictError as exc:
        return _error(str(exc), 409)
    return jsonify({"status": "ok", "local_only": True, "job": job}), 202


@bp.get("/admin/training/jobs/<run_id>/samples/<sample_id>")
@bp.get("/admin/training/jobs/<run_id>/pages/<sample_id>", defaults={"source": True})
@require_local_request
@require_permission("training.read")
def comparison_image(run_id: str, sample_id: str, source=False):
    from backend.app.routes.admin_annotations import _private_media
    path = _services().admin_training_service().comparison_media(run_id, sample_id, source=source)
    if path is None:
        return _error("comparison sample not found", 404)
    return _private_media(path)

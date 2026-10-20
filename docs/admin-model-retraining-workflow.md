# Local annotation and retraining workflow

Updated: 2026-09-17

The standard local mode combines the **shipped model base and all current approved annotations**. It needs no private corpus or manual snapshot. The installer downloads fixed MobileSAM and DINOv2 assets, enables local training, and permits the initial local administrator to review their own annotations.

1. Upload and analyze an image, open its annotation editor, correct boxes and labels, and mark the desired elements ready.
2. Send the annotations, then open **Admin → Review** and approve them.
3. In **Classes**, explicitly confirm any new label before learning it.
4. In **Entraîner**, click **Vérifier la préparation**, then **Créer un candidat**.
5. Open **Comparer les modèles** to inspect identical examples, source pages and metrics.

Each run captures current approved crops and review decisions automatically. Exact duplicate images count once; conflicting labels for identical images are rejected. Stale decisions and missing crops are excluded. Repeating the same approvals does not count their contribution twice.

The backbone and projection stay frozen. The update adapts existing prototypes and adds explicitly confirmed new classes with stable numeric IDs. It does not train MobileSAM. Every run rebuilds from the immutable shipped prior plus current approvals, so reruns do not compound previous candidates.

Candidates are stored under `backend/model_registry/versions/<version_id>/`, with provenance and checksums. They are **not activated** and promotion remains blocked. Source pages are reserved before fitting where enough examples exist. Reports separate training fit, reserved-page tests and new-class performance; independence from historical base training remains unknown. A tiny or absent holdout cannot establish generalization.

## Local permissions

The installer adds `ENABLE_ADMIN_TRAINING_JOBS=true` and `ALLOW_LOCAL_ADMIN_SELF_REVIEW=true` to the ignored `backend/.env`, preserving existing settings. Self-review requires the initial administrator, a loopback-bound backend and a local request. Session, role and CSRF checks still apply; the reviewer identity is recorded. Other accounts and configurations retain independent-review requirements.

## Existing annotations

Canonical folders under `backend/annotations/` appear in Review; their annotations need not be recreated. Missing or changed source files require repair and another review. New labels must be explicitly confirmed in Classes. Existing JSON decisions require the [backed-up SQLite import](retraining-stability-delivery.md); never recreate or delete the existing annotations to migrate.

Corrections and decisions carry an expected revision; stale concurrent edits return HTTP 409. Dataset → Ouvrir dans le triage → Remettre à vérifier removes an example from future runs. History → Restaurer restores its evidence as pending, requiring another approval. Already-created candidates remain immutable historical snapshots.

## Installation and recovery

Follow [INSTALL.md](../INSTALL.md). Re-running the installer preserves credentials and annotations. MobileSAM is downloaded through a temporary file, verified against a fixed SHA-256, then installed atomically. A failed download preserves the existing checkpoint. The startup smoke checks assets through `/ready`.

If a run fails, inspect its error/logs in Training, correct the issue, and retry. Current approvals are captured again.

## Advanced corpus mode

Setting `ADMIN_TRAINING_SNAPSHOT_DIR` explicitly selects the existing cumulative-corpus workflow. It requires a prepared snapshot and rejects stale evidence. Leave it unset for the standard workflow. A `MODEL_DIR` override blocks the admin launch to avoid incompatible model packages.

## Verification

Fast checks: backend pytest, frontend lint/test/build and Playwright.
`CLINIC_LOCAL_RETRAIN_E2E=1` enables real CPU inference and authenticated dry/full runs using bundled images.
`CLINIC_LIVE_E2E=1` enables the browser journey against a freshly installed backend. Run it in a disposable checkout: it creates an account, annotation and candidate.

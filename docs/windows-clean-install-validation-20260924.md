# Fresh Windows installation validation — 24 September 2026

Tested GitHub `main` commit: `95a72e0f888d4fe48788b65afd0f36f142c2d57a`.

## User workflow

1. Clone the repository into a new directory on a native Windows disk.
2. Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1`: successful with Python 3.11, a new virtual environment, `npm ci`, base-model export, and the official MobileSAM/DINOv2 setup.
3. Extract a copy of an existing legacy annotation archive into `backend\annotations`.
4. Start `scripts\run-dev.ps1` with unused ports. Leave authentication enabled and create the first account through the normal UI.
5. Verify the legacy-import warning and disabled class-confirmation actions before migration.
6. Stop the application and run the documented preview/apply commands below.
7. Restart, sign in, and confirm the nine new classes through their buttons and confirmation dialogs.
8. Check Dataset, run a dry preparation, train a real CPU candidate, reload the page, and compare both models on a full page with 11 regions.

Commands run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe scripts\repair_missing_annotation_crops.py --annotations-dir backend\annotations
.\backend\.venv\Scripts\python.exe scripts\repair_missing_annotation_crops.py --annotations-dir backend\annotations --apply
.\backend\.venv\Scripts\python.exe scripts\migrate_annotation_reviews.py --annotations-dir backend\annotations
.\backend\.venv\Scripts\python.exe scripts\migrate_annotation_reviews.py --annotations-dir backend\annotations --apply
```

Repeating the migration with `--apply` returned `already_imported`.

## Observed results

- 21 source images; 491 imported decisions: 475 approved, 16 rejected, none pending. All 475 approved annotations were usable; no diagnostic errors remained.
- Five missing derived crops reconstructed; six decimal bounding boxes normalized in imported decisions; no quarantined bounding boxes.
- All 536 original archive files remained byte-identical after import and training. The source archive was unchanged.
- Nine new classes confirmed and included in the candidate.
- Real candidate `20260924T190815Z-95a72e0f-71ca013c` completed successfully and remained visible after reload. No activation was performed; protected active-model files remained unchanged during training.
- Training used 466 unique crops: 420 fitting samples and 46 held-out samples. Seven duplicates and two conflicting annotations were excluded without altering their sources.
- Fitting accuracy: base 204/420, candidate 224/420. Held-out accuracy: 21/46 for both. This run showed no held-out improvement.
- Comparison displayed two full 867 × 1273 images with 11 regions: one correction and no regressions on this exploratory page. This is not an independent generalization result.
- No JavaScript errors were observed. HTTP 401 responses were limited to unauthenticated session checks before login.

## Scope and limitations

This test used native Edge automation against the real Windows backend: no mocked API responses, manual SQLite modifications, authentication bypass, or application-code fixes. A wrong selector in the test driver was corrected after the class-confirmation step; this was a test-driver correction, not an application change.

The official installer marked `backend/codex_model/config.json` modified only because of Windows line endings; `git diff --ignore-space-at-eol --exit-code` confirmed unchanged content.

Legacy review migration remains an explicit, documented operation while the application is stopped, not an automatic startup action. Successful import of this archive does not guarantee that every malformed legacy archive can be imported without diagnostics. The results above validate this Windows CPU workflow, not every operating system, accelerator, or model-quality outcome.

The original archive, copied annotations, generated candidate, screenshots, raw browser results, and test credentials remain local and are not published with this report.

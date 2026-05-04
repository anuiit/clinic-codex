# Learnings

## [2026-05-04] Session Init
- Git repo has NO commits yet on master; all files are untracked.
- Root `.gitignore` is empty (0 lines).
- `codex-frontend/README.md` is still the default Vite template.
- Frontend API base URL hardcoded to `http://localhost:5000` in `codex-frontend/src/services/api.ts`.
- No `.github/workflows/` CI files exist.
- Known noise: `frontend_integration_fix/__MACOSX/`, `**/__pycache__/`, `.DS_Store`.
- Backend Python deps: torch, torchvision, numpy, Pillow, albumentations, flask, mobile-sam, segment-anything, scipy, tqdm, pyyaml.
- Sample glyph: `frontend_integration_fix/frontend_integration/data/glyphs_sample/atl-glyph/026r_a_07-2.jpg`
- Model artifacts: `codex_model/weights/prototypes.pt`, `codex_model/weights/projection.pt`, `prototypes/prototypes.pt`, `mobile_sam.pt`
- Frontend commands: `npm install`, `npm run dev`, `npm run build`, `npm run lint`
- Backend demo entrypoint: `frontend_integration_fix/frontend_integration/examples/flask_api.py`
- Backend API endpoints: `/classify`, `/classify-batch`, `/segment`, `/classes`
- Root inventory classification completed: `codex-frontend/` = keep; `frontend_integration_fix/` = keep-for-now; `package-lock.json` and `presentation_codex.pdf` = remove; `sam_glyph_test/` and `.sisyphus/` = non-canonical reference/process artifacts.
- Artifact policy direction: real `.pt` weights stay keep-for-now as external assets pending a proper asset-management path; sample integration data stays keep-for-now when used for demos/tests; experimental SAM output bundle is better archived externally.
- Confirmed removable noise artifacts include `__MACOSX/`, `.DS_Store`, `__pycache__/`, and AppleDouble `._*.pt` sidecars.

## [2026-05-04] Smoke baseline created
- Added smoke-check definitions for frontend and backend at `.sisyphus/evidence/task-2-frontend-smoke-definition.txt` and `.sisyphus/evidence/task-2-backend-smoke-definition.txt`.
- Frontend baseline covers deterministic install (npm ci), build (npm run build -> tsc + vite), and lint (npm run lint -> eslint) — all must exit 0.
- Backend baseline covers running `python examples/flask_api.py`, using the sample glyph `data/glyphs_sample/atl-glyph/026r_a_07-2.jpg`, and a curl POST to `/segment` expecting HTTP 200 and JSON with keys `num_elements`, `image_size`, and `elements` (elements is an array).

---

## [2026-05-04] Ignore rules added

- Root .gitignore was populated with OS artifacts, Python, Node, build outputs, and editor files.
- Evidence saved to `.sisyphus/evidence/task-3-ignore-rules.txt` and `.sisyphus/evidence/task-3-ignore-safety.txt`.
- Safety checks: confirmed `codex-frontend/` and `frontend_integration_fix/frontend_integration/` remain tracked; sample data path `frontend_integration_fix/frontend_integration/data/` is explicitly not ignored.
- Replaced generic Vite README with project-specific documentation for Codex Glyph Analyzer.
- Documented prerequisites, installation, environment variables, and relationship with the backend API.
- Verified removal of template-specific phrases (HMR, Vite template, etc.) to ensure clean project documentation.
## [2026-05-04] Root README Creation
Created a comprehensive root README.md that consolidates frontend and backend information.
- Included verified smoke check commands from evidence files.
- Documented model artifact requirements and caveats.
- Linked to sub-READMEs for deeper technical details.
- Ensured no placeholders or unverified instructions were included.
- [2026-05-04] Make API base URL env-driven via VITE_API_BASE_URL. Updated codex-frontend/src/services/api.ts to use import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000' and added codex-frontend/.env.example. Verified build attempt; vite binary permission error in this environment prevented successful build here. Evidence files created at .sisyphus/evidence/task-6-default-api-url.txt and task-6-env-override.txt
### 2026-05-04: Backend Documentation Cleanup
- Documented `examples/flask_api.py` as the supported demo entrypoint.
- Explicitly listed required model artifacts (`prototypes.pt`, `projection.pt`, `mobile_sam.pt`) and their locations.
- Documented external asset policy: weights are treated as external, updateable assets.
- Clarified that the current package is for demonstration/development scope only.

## Task 8 - Noise Artifact Removal (2026-05-04)
- No `__MACOSX/` at repo root; it only existed inside `frontend_integration_fix/` (left in place per scope)
- `presentation_codex.pdf` and root `package-lock.json` deleted successfully
- 2 `.DS_Store` files found and deleted (both in `frontend_integration_fix/`)
- 14 `__pycache__/` directories deleted (all in `frontend_integration_fix/`, including `.venv` and `.venv2` site-packages)
- `sam_glyph_test/` archived to `_archive/sam_glyph_test/`; `_archive/` dir created fresh
- Root is now clean: `.git`, `.gitignore`, `.omc`, `.sisyphus`, `README.md`, `_archive`, `codex-frontend`, `frontend_integration_fix`

- Normalized local dependency and output policy. Confirmed .gitignore covers node_modules, dist, __pycache__, .DS_Store, and .env. Added *.pt to root .gitignore. Documented commit policy in root README.md.

## Task 10 - CI Workflow Created (2026-05-04)
- Created `.github/workflows/smoke.yml` with 3 jobs: `frontend-smoke`, `frontend-lint`, `backend-import`.
- Triggers: push and pull_request on main and master branches.
- `frontend-smoke`: Node 18, `npm ci && npm run build` in `codex-frontend/` — must pass.
- `frontend-lint`: Node 18, `npm ci && npm run lint` in `codex-frontend/` — `continue-on-error: true` (pre-existing lint errors).
- `backend-import`: Python 3.10, `pip install -r requirements.txt` then `python -c "from codex_model import CodexClassifier; print('import ok')"` in `frontend_integration_fix/frontend_integration/` — import-only, no model weights needed.
- Full `/segment` curl check excluded from CI (requires `mobile_sam.pt` weights not available in CI).
- Evidence at `.sisyphus/evidence/task-10-ci-workflow.txt`.

## Task-11: Post-Cleanup Smoke Check Results (2026-05-04)

All three smoke checks passed against the cleaned repository:

| Check      | Result | Notes                                      |
|------------|--------|--------------------------------------------|
| Build      | PASS   | vite build: 1787 modules, exit 0           |
| Lint       | PASS*  | 3 pre-existing issues only, no regressions |
| TypeScript | PASS   | tsc --noEmit: exit 0, no type errors       |

Pre-existing lint errors confirmed unchanged:
- AnnotationPage.tsx:22 react-hooks/set-state-in-effect
- DetailPage.tsx:16 react-hooks/set-state-in-effect
- DetailPage.tsx:71 react-hooks/exhaustive-deps (warning)

Evidence files: .sisyphus/evidence/task-11-*.txt

## [2026-05-04] T12 — Documentation Audit
- All 40 doc references across README.md, codex-frontend/README.md, frontend_integration_fix/frontend_integration/README.md, and .github/workflows/smoke.yml verified PASS.
- No broken references found. No fixes were needed.
- sam_glyph_test/ (moved to _archive/) is NOT referenced in any doc — clean.
- presentation_codex.pdf (deleted) is NOT referenced in any doc — clean.
- Root package-lock.json (deleted) is NOT referenced in any doc — clean.
- codex-frontend/package-lock.json exists and is correctly used in smoke.yml cache-dependency-path.
- All model artifact paths (codex_model/weights/, prototypes/, mobile_sam.pt) exist under frontend_integration_fix/frontend_integration/.
- Sample data paths (data/glyphs_sample/atl-glyph/026r_a_07-2.jpg, data/elements_sample/0015-cacahuatl/03_04_22-27.bmp) verified present.

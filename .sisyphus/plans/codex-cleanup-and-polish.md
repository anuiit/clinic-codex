# Clinic Codex — Cleanup, Polish, Bug Fixes & Embedding Prep

## TL;DR

> **Quick Summary**: Restructure the repo to a classic `backend/` + `frontend/` layout, switch to stable env-driven ports (7117/7118), produce a dual-audience README with one-command install scripts for non-technical researchers, fix three confirmed annotation UX bugs (coord mismatch under zoom, narrow handle hit-test, no pointer capture), reconcile a backend preprocessing inconsistency, and document the embedding-similarity readiness for when the dataset arrives.
>
> **Deliverables**:
> - Renamed repo: `backend/` (from `frontend_integration_fix/frontend_integration/`) + `frontend/` (from `codex-frontend/`)
> - Archived `_legacy/EXISTING_CODE/` and `_legacy/sam_glyph_test/`
> - `dev-scripts/` with patch artifacts + README
> - Env-driven config: backend on `7117`, frontend on `7118`, `.env.example` files for both
> - One-command launchers: `scripts/install.sh` / `install.ps1`, `scripts/run-dev.sh`, `scripts/download-weights.sh`
> - Rewritten `README.md` (technical + non-technical sections) + `INSTALL.md` (researcher-friendly)
> - Three concrete annotation bug fixes in `frontend/src/pages/AnnotationPage.tsx` (pointer events, scan-all handle hit-test, getScreenCTM coord mapping)
> - Vitest scaffolding (frontend) + pytest scaffolding (backend) with one example test each
> - Reconciled image preprocessing (gray vs white pad)
> - `EMBEDDING-READINESS.md` documenting the inventory + plug-in plan
> - Updated `.github/workflows/smoke.yml` for new paths
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Wave 0 (WIP commit + branch) → Wave 1 (rename + CI fix) → Wave 2 (config + bug fixes + docs in parallel) → Wave 3 (install scripts + tests) → Final Verification

---

## Context

### Original Request
- "De clean un peu le repo, rename les dossiers, remettre les choses au bon endroit."
- "Utiliser des ports stables et généralement non utilisés en développement en production pour éviter les conflits."
- "Avoir un readme clair, avec une partie technique et une partie non-technique."
- "Un guide d'installation clair de l'application pour pouvoir l'utiliser pour des gens non-techniques. Simplifier l'installation au maximum."
- "Fix les petits bugs, annotation box pas toujours précis dans la sélection, déplacement / agrandissement de la zone plus fiable (des fois cliquer ne prend pas en compte le mouvement, demande de cliquer une deuxième fois pour sélectionner...)"
- "Préparation d'affichage des images proches en termes d'embeddings, je te donnerai le dataset complet plus tard. Commence à réfléchir."
- "Si tu vois autre chose n'hésite pas à le mentionner."

### Interview Summary

**Confirmed Decisions**:
- Repo target: `backend/` + `frontend/` (classique)
- Ports: backend `7117`, frontend `7118` (env-driven, palindromic, IANA-libre)
- Install audience: chercheurs/Nahuatl specialists, dev/testing, Mac/Linux primary, **no Docker**
- Tests: Vitest (frontend) + pytest (backend), critical zones only, plus per-task agent QA
- Embedding work: **documentation only now**, infra wait for dataset
- Patch scripts: move to `dev-scripts/` with README
- Additional improvements: **essentials only** (env vars, run-dev, weights downloader, .gitignore, logs, preprocessing fix). Skip i18n + pre-commit hooks
- Logistics: feature branch `cleanup/repo-restructure`, commit WIP first

**Research Findings (verified by exploration)**:
- `EXISTING_CODE/` is a duplicate of `frontend_integration_fix/frontend_integration/` (must diff to confirm before archiving)
- Backend Flask hardcoded to `0.0.0.0:5000` in `examples/flask_api.py` — needs env-driven `PORT` + `HOST`
- Vite uses default `5173` — must add explicit `port: 7118` in `vite.config.ts`
- Frontend `services/api.ts` already env-driven (`VITE_API_BASE_URL`) ✓
- AnnotationPage.tsx bug locations: `getHitHandle` L89-97, `mousedown` L99-144, `mousemove` L146-233, `mouseup` L235-258, SVG wrapper L354-396
- Embedding pipeline: DINOv2-ViT-S/14 → 384d backbone → projection → 128d L2-normalized; 286 prototypes; cosine via `torch.mm`; `/similar` and `/trust` already exist with prototype-mode
- **Preprocessing inconsistency**: `CodexClassifier._preprocess_image` pads gray (128), `InferenceEngine.preprocess` pads white (255) — must reconcile
- CI workflow `.github/workflows/smoke.yml` references `codex-frontend/` paths — must update
- WIP exists on master (workspace UI redesign, i18n folder, modified annotation files)

### Self-Review Findings (Metis-style gap analysis)
**Identified Gaps (addressed in plan)**:
- Git history preservation: use `git mv` exclusively for renames, single commit per move
- Pre-flight: verify no Python imports of `frontend_integration_fix.frontend_integration` before flatten
- CI workflow update: must change in same wave as rename to keep CI green
- `EXISTING_CODE/` diff verification: confirm true duplicate before archiving
- Model weights paths: explicit verification post-rename, document expected locations
- Port collision pre-check: launcher must `lsof -i :PORT` before bind
- Windows install parity: `install.ps1` mirror of `install.sh`
- `.sisyphus/` boulder/state preservation: add to "Must NOT touch" list
- WIP commit ordering: commit WIP on master FIRST, then create branch
- CORS production: env-driven `CORS_ORIGINS` list, not hardcoded
- AnnotationPage handle visual size (8) vs hit area (12) harmonization
- RAF throttling on pointermove for jank prevention

---

## Work Objectives

### Core Objective
Transform Clinic Codex from a research-prototype-shaped repo into a maintainable, easy-to-install project with reliable annotation UX, while preparing the foundation for the upcoming embedding-similarity feature — all without breaking the running app or losing git history.

### Concrete Deliverables
- `backend/` directory (renamed + flattened from `frontend_integration_fix/frontend_integration/`)
- `frontend/` directory (renamed from `codex-frontend/`)
- `_legacy/EXISTING_CODE/`, `_legacy/sam_glyph_test/` (archived)
- `dev-scripts/{patch.js, patch2.js, patch_workspace.js, patch_workspace.sh, README.md}`
- `backend/.env.example`, `frontend/.env.example`
- `scripts/install.sh`, `scripts/install.ps1`, `scripts/run-dev.sh`, `scripts/download-weights.sh`
- Rewritten `README.md` (dual-audience)
- New `INSTALL.md` (non-technical, step-by-step)
- New `EMBEDDING-READINESS.md` (inventory + plug-in plan)
- Modified `frontend/src/pages/AnnotationPage.tsx` (3 bug fixes + 2 polish)
- New `frontend/vite.config.ts` with `port: 7118`
- Modified `backend/examples/flask_api.py` with env-driven `PORT`/`HOST`/`CORS_ORIGINS`
- Modified `backend/codex_pipeline/preprocess.py` (or equivalent) — unified pad color
- New `frontend/vitest.config.ts` + 1 example test
- New `backend/tests/conftest.py` + 1 example pytest test
- Updated `.github/workflows/smoke.yml` (new paths)
- Updated `.gitignore` (dist/, evidence/)

### Definition of Done
- [ ] `git status` clean on branch `cleanup/repo-restructure`
- [ ] `cd frontend && npm run build` exits 0
- [ ] `cd frontend && npm test` exits 0 (≥1 passing test)
- [ ] `cd backend && pytest` exits 0 (≥1 passing test)
- [ ] `PORT=7117 python backend/examples/flask_api.py &` then `curl http://localhost:7117/health` returns 200
- [ ] `cd frontend && npm run dev` binds 7118; browser loads UI
- [ ] Annotation: drag empty area creates box; click box-handle directly resizes (no pre-click); drag works with mouse leaving SVG bounds
- [ ] `bash scripts/install.sh` on a fresh clone succeeds end-to-end
- [ ] `.github/workflows/smoke.yml` passes on push (verify locally with `act` or trust workflow logic update)
- [ ] No file in repo references the old paths `codex-frontend/` or `frontend_integration_fix/`
- [ ] `EXISTING_CODE/` diff vs new `backend/` documented before archive
- [ ] All evidence files saved under `.sisyphus/evidence/codex-cleanup/`

### Must Have
- Git history preserved via `git mv` for every rename (verify with `git log --follow`)
- Backend reads `PORT`, `HOST`, `CORS_ORIGINS`, `MODEL_DIR` from env (defaults: 7117, 0.0.0.0, http://localhost:7118, ./models)
- Frontend Vite binds 7118 + reads `VITE_API_BASE_URL` (default http://localhost:7117)
- `install.sh` checks: Python ≥ 3.10, Node ≥ 18, ports free, weights present (or auto-download)
- `run-dev.sh` launches backend + frontend together with port-collision check
- README has explicit "For researchers (no terminal experience needed)" section pointing to INSTALL.md
- AnnotationPage: pointer events + setPointerCapture + getScreenCTM coord mapping + scan-all handle hit-test
- Image preprocessing: single source of truth for pad color (white=255), both classifier and engine use it
- CI smoke workflow updated and green on the new paths
- `_legacy/` directory has top-level `README.md` explaining what's archived and why

### Must NOT Have (Guardrails)
- No deletion of `EXISTING_CODE/` without diff confirmation against new `backend/`
- No hardcoded ports anywhere — `5000`, `5173`, `8000`, `3000` must NOT appear in source after this work
- No `cp` or manual move for renames — only `git mv` (history preservation)
- No touching of `.sisyphus/state/`, `.sisyphus/boulder.json`, `.sisyphus/notepads/` (active session tracking)
- No deletion of model weight files (`.pt`, `.pth`, `.safetensors`)
- No new dependencies beyond Vitest, @testing-library/react, pytest (NO FAISS, NO concurrently NPM, NO Docker)
- No instance-level embedding code (script, API extension, gallery UI) — documentation only per user choice
- No completion of i18n WIP — leave `frontend/src/i18n/text.ts` as-is
- No pre-commit hooks setup
- No production deployment artifacts (Dockerfile, nginx.conf, systemd units)
- No mass refactoring beyond the listed bug fixes in AnnotationPage.tsx
- No "while we're at it" cleanups in unrelated files
- No commit on `master` after Wave 0 — all work on `cleanup/repo-restructure`

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (neither vitest nor pytest currently configured)
- **Automated tests**: YES (tests-after, critical zones only)
- **Frameworks**: Vitest (frontend) + pytest (backend)
- **TDD**: Not strict; bug fixes get a regression test alongside the fix

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/codex-cleanup/task-{N}-{slug}.{ext}`.

- **Repo structure**: `Bash` — `ls`, `find`, `git log --follow` to verify history
- **Backend API**: `Bash` — `curl http://localhost:7117/health`, response assertions
- **Frontend dev server**: `Bash` — `curl http://localhost:7118/` returns HTML
- **Annotation UI bugs**: `Playwright` (playwright skill) — open AnnotationPage, drag/resize, screenshot
- **Install scripts**: `Bash` — run `install.sh` in a temp clone, assert exit 0
- **CI workflow**: `Bash` — `yamllint .github/workflows/smoke.yml` + grep for old paths

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Foundation — sequential, blocks everything):
└── Task 1: Commit current WIP on master + create feature branch [quick]

Wave 1 (Structure — sequential within wave, blocks Wave 2):
├── Task 2: Verify EXISTING_CODE is duplicate, then archive [quick]
├── Task 3: git mv codex-frontend/ → frontend/ [quick]
├── Task 4: git mv backend flatten (frontend_integration_fix/frontend_integration/ → backend/) [quick]
├── Task 5: Move dev-scripts + archive sam_glyph_test [quick]
└── Task 6: Update CI workflow paths [quick]

Wave 2 (Config + Code in PARALLEL, after Wave 1):
├── Task 7: Backend env-driven config (PORT/HOST/CORS) [unspecified-high]
├── Task 8: Frontend Vite port + .env.example [quick]
├── Task 9: Annotation bug fix #1 — pointer events + setPointerCapture [unspecified-high]
├── Task 10: Annotation bug fix #2 — scan-all handle hit-test [unspecified-high]
├── Task 11: Annotation bug fix #3 — getScreenCTM coord mapping [unspecified-high]
├── Task 12: Annotation polish — handle size harmonization + RAF throttle [quick]
├── Task 13: Reconcile image preprocessing (gray vs white pad) [unspecified-high]
└── Task 14: Update .gitignore (dist, evidence) [quick]

Wave 3 (Install + Docs + Tests in PARALLEL, after Wave 2):
├── Task 15: scripts/install.sh + install.ps1 [unspecified-high]
├── Task 16: scripts/run-dev.sh + scripts/download-weights.sh [unspecified-high]
├── Task 17: Vitest scaffolding + 1 example test [quick]
├── Task 18: pytest scaffolding + 1 example test [quick]
├── Task 19: Rewrite README.md (dual-audience) [writing]
├── Task 20: Write INSTALL.md (non-technical) [writing]
└── Task 21: Write EMBEDDING-READINESS.md [writing]

Wave FINAL (4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high + playwright)
└── Task F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: T1 → T3 → T7 → T9 → T15 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 8 (Wave 2)
```

### Dependency Matrix

- **T1**: blocks all | depends on: nothing
- **T2-T6**: depend on T1 | block Wave 2
- **T7-T14**: depend on T3+T4 (need new dirs) | block Wave 3
- **T15-T21**: depend on Wave 2 | block Final
- **F1-F4**: depend on Wave 3 complete | block user signoff

### Agent Dispatch Summary

- **Wave 0**: 1 task — T1 → `quick`
- **Wave 1**: 5 tasks — T2-T6 → `quick`
- **Wave 2**: 8 tasks — T7,T9,T10,T11,T13 → `unspecified-high`; T8,T12,T14 → `quick`
- **Wave 3**: 7 tasks — T15,T16 → `unspecified-high`; T17,T18 → `quick`; T19,T20,T21 → `writing`
- **FINAL**: 4 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high` + `playwright`, F4 → `deep`

---

## TODOs

- [x] 1. Commit current WIP on master + create feature branch

  **What to do**:
  - On `master`: `git add -A` to stage all current modifications and untracked files (excluding `.sisyphus/evidence/` per .gitignore once added — but for this commit, include them so nothing is lost; they will be ignored going forward)
  - Commit with message: `chore(wip): commit work-in-progress before restructure`
  - Create branch: `git checkout -b cleanup/repo-restructure`
  - Push branch (no upstream needed yet, just track locally is fine)

  **Must NOT do**:
  - Do NOT touch `.sisyphus/state/`, `.sisyphus/boulder.json`, `.sisyphus/notepads/` (they may have uncommitted state changes — leave them alone or stash separately)
  - Do NOT run `git stash` (loses context)
  - Do NOT continue any further work on `master`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single git operation sequence, no logic
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit hygiene
  - **Skills Evaluated but Omitted**:
    - None — pure git operation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Wave 0
  - **Blocks**: All other tasks
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `git status` output (see latest `git status --short`) — current WIP state

  **WHY Each Reference Matters**:
  - Need full WIP snapshot to know what's being committed and that nothing is lost

  **Acceptance Criteria**:
  - [ ] `git log -1 --format=%s` returns `chore(wip): commit work-in-progress before restructure`
  - [ ] `git branch --show-current` returns `cleanup/repo-restructure`
  - [ ] `git status` is clean (no modified, no untracked except `.sisyphus/evidence/`)

  **QA Scenarios**:

  ```
  Scenario: WIP committed and branch created
    Tool: Bash
    Preconditions: On master with modifications shown in git status
    Steps:
      1. Run `git log -1 --format='%s|%an'` → assert subject contains "chore(wip)"
      2. Run `git branch --show-current` → assert equals "cleanup/repo-restructure"
      3. Run `git status --porcelain | grep -v '^??.*\.sisyphus/evidence' | wc -l` → assert == 0
    Expected Result: All three assertions pass
    Failure Indicators: Wrong branch, dirty state, or wrong commit message
    Evidence: .sisyphus/evidence/codex-cleanup/task-1-branch-state.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-branch-state.txt (output of `git log -3 --oneline && git branch --show-current && git status`)

  **Commit**: Already part of task itself
  - Message: `chore(wip): commit work-in-progress before restructure`
  - Files: all currently modified
  - Pre-commit: none

- [x] 2. Verify EXISTING_CODE is duplicate, then archive to _legacy/

  **What to do**:
  - Run `diff -rq EXISTING_CODE/ frontend_integration_fix/frontend_integration/` and capture output
  - If only differences are `.pyc`, `__pycache__`, or generated files: confirmed duplicate → proceed
  - If meaningful differences exist: STOP and surface them in evidence file, do not archive
  - On confirmation: `mkdir -p _legacy && git mv EXISTING_CODE _legacy/EXISTING_CODE`
  - Create `_legacy/README.md` explaining: "Archived on YYYY-MM-DD. Contents: EXISTING_CODE/ was a duplicate of the active backend (now `backend/`). Kept for reference only. Do not import from this directory."

  **Must NOT do**:
  - Do NOT delete EXISTING_CODE — only archive
  - Do NOT proceed if diff shows real code differences
  - Do NOT use `mv` or `cp` — only `git mv` for history

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical diff + git mv
  - **Skills**: [`git-master`]
    - `git-master`: Preserve history with git mv

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T3, T4, T5 — different paths)
  - **Parallel Group**: Wave 1
  - **Blocks**: F1 (compliance check), F4 (scope check)
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `EXISTING_CODE/` and `frontend_integration_fix/frontend_integration/` — to diff

  **WHY Each Reference Matters**:
  - Must confirm true duplicate before archiving — losing live code would be catastrophic

  **Acceptance Criteria**:
  - [ ] `_legacy/EXISTING_CODE/` exists
  - [ ] `EXISTING_CODE/` (top-level) does not exist
  - [ ] `_legacy/README.md` exists with archive rationale
  - [ ] Diff evidence file captured
  - [ ] `git log --follow _legacy/EXISTING_CODE/codex_model/` shows pre-rename history

  **QA Scenarios**:

  ```
  Scenario: Duplicate confirmed and archived with history
    Tool: Bash
    Preconditions: T1 complete, on cleanup/repo-restructure branch
    Steps:
      1. Run `test -d _legacy/EXISTING_CODE && test ! -d EXISTING_CODE` → exit 0
      2. Run `git log --follow --oneline _legacy/EXISTING_CODE/ | wc -l` → assert > 0
      3. Run `cat _legacy/README.md | grep -q "EXISTING_CODE"` → exit 0
    Expected Result: Directory moved, README in place, history preserved
    Failure Indicators: Either dir missing/extra, no history, no README
    Evidence: .sisyphus/evidence/codex-cleanup/task-2-diff.txt + task-2-archive-state.txt

  Scenario: Diff shows non-trivial differences (negative path)
    Tool: Bash
    Preconditions: Hypothetical — if diff revealed real code differences
    Steps:
      1. Diff output captured to evidence
      2. Task halts, raises blocker for user
    Expected Result: User informed, no archive performed
    Evidence: .sisyphus/evidence/codex-cleanup/task-2-diff-blocker.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-diff.txt (full diff output)
  - [ ] task-2-archive-state.txt (`ls _legacy/` + `git log --follow` excerpt)

  **Commit**: YES
  - Message: `chore(legacy): archive EXISTING_CODE as confirmed duplicate`
  - Files: `_legacy/EXISTING_CODE/`, `_legacy/README.md`
  - Pre-commit: diff verification passes

- [x] 3. git mv codex-frontend/ → frontend/

  **What to do**:
  - `git mv codex-frontend frontend` (single atomic move)
  - Update any in-repo references: search `grep -r "codex-frontend" . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.sisyphus` and replace with `frontend` in source files (NOT in archived legacy dirs, NOT in .sisyphus old artifacts)
  - Sanity: `cat frontend/package.json | head -3` to confirm package metadata intact

  **Must NOT do**:
  - Do NOT update references in `_legacy/`, `.sisyphus/notepads/`, or git history
  - Do NOT change `package.json` `name` field unless it currently says `codex-frontend` (then update to `frontend`)
  - Do NOT touch `node_modules/` or `dist/` (will be cleaned by T14)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single git mv + targeted text replacement
  - **Skills**: [`git-master`]
    - `git-master`: Atomic rename with history preservation

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T4, T5)
  - **Parallel Group**: Wave 1
  - **Blocks**: T6 (CI update), all Wave 2 frontend tasks
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `codex-frontend/` — current frontend directory

  **External References**:
  - `git mv` docs: https://git-scm.com/docs/git-mv

  **WHY Each Reference Matters**:
  - `git mv` is the ONLY way to preserve `git log --follow` for renamed files

  **Acceptance Criteria**:
  - [ ] `frontend/` exists with package.json, src/, public/
  - [ ] `codex-frontend/` does not exist
  - [ ] `git log --follow frontend/package.json | wc -l` > 5 (history preserved)
  - [ ] `grep -r "codex-frontend" . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.sisyphus --exclude-dir=_legacy --exclude-dir=dist` returns ONLY false-positive matches in archived/state files

  **QA Scenarios**:

  ```
  Scenario: Frontend renamed with history intact
    Tool: Bash
    Preconditions: T1 complete
    Steps:
      1. `test -d frontend && test ! -d codex-frontend` → exit 0
      2. `test -f frontend/package.json && test -f frontend/src/App.tsx` → exit 0
      3. `git log --follow --oneline frontend/package.json | wc -l` → assert > 5
      4. Source-tree grep for "codex-frontend" outside excluded dirs returns 0
    Expected Result: All assertions pass
    Failure Indicators: Missing dir, lost history, lingering references
    Evidence: .sisyphus/evidence/codex-cleanup/task-3-frontend-rename.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-frontend-rename.txt (`ls frontend/`, `git log --follow`, grep results)

  **Commit**: YES
  - Message: `chore(repo): rename codex-frontend to frontend`
  - Files: all of `frontend/` (renamed) + any source files updated to drop "codex-frontend" references
  - Pre-commit: history check passes

- [x] 4. git mv backend flatten — frontend_integration_fix/frontend_integration/ → backend/

  **What to do**:
  - Pre-flight: `grep -r "frontend_integration_fix\|from frontend_integration\|import frontend_integration" . --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.sisyphus --exclude-dir=_legacy` to find Python imports — capture to evidence
  - Single atomic move: `git mv frontend_integration_fix/frontend_integration backend`
  - Then: `rmdir frontend_integration_fix` (the now-empty wrapper)
  - Update any imports found in pre-flight (replace `frontend_integration_fix.frontend_integration` → `backend` if any exist)
  - Verify: `cat backend/examples/flask_api.py | head -5` (sanity)

  **Must NOT do**:
  - Do NOT delete `frontend_integration_fix/` if it has any other content besides `frontend_integration/`
  - Do NOT use anything other than `git mv`
  - Do NOT touch model weight files (`.pt`, `.pth`, `.safetensors` inside)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical move with pre-flight check
  - **Skills**: [`git-master`]
    - `git-master`: Preserve history through nested move

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T5)
  - **Parallel Group**: Wave 1
  - **Blocks**: T6, all Wave 2 backend tasks
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `frontend_integration_fix/frontend_integration/` — current backend
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py` — entry point

  **WHY Each Reference Matters**:
  - The double-nesting is a historical artifact; flattening simplifies imports and paths

  **Acceptance Criteria**:
  - [ ] `backend/` exists with `examples/flask_api.py`, `codex_model/`, `codex_pipeline/`, etc.
  - [ ] `frontend_integration_fix/` does not exist
  - [ ] `git log --follow backend/examples/flask_api.py | wc -l` > 3
  - [ ] No import of `frontend_integration_fix` anywhere in active source
  - [ ] All `.pt` model files still present in `backend/`

  **QA Scenarios**:

  ```
  Scenario: Backend flattened with history and weights intact
    Tool: Bash
    Preconditions: T1 complete
    Steps:
      1. `test -d backend && test ! -d frontend_integration_fix` → exit 0
      2. `test -f backend/examples/flask_api.py` → exit 0
      3. `find backend -name "*.pt" | wc -l` → compare to pre-rename count (saved in evidence)
      4. `git log --follow --oneline backend/examples/flask_api.py | wc -l` → assert > 3
      5. `grep -r "frontend_integration_fix" --include="*.py" backend/ frontend/ scripts/ 2>/dev/null | wc -l` → 0
    Expected Result: All checks pass
    Failure Indicators: Missing weights, lost history, dangling imports
    Evidence: .sisyphus/evidence/codex-cleanup/task-4-backend-flatten.txt
  ```

  **Evidence to Capture**:
  - [ ] task-4-backend-flatten.txt (pre-flight grep, ls backend/, find .pt count, git log)

  **Commit**: YES
  - Message: `chore(repo): flatten and rename to backend`
  - Files: all of `backend/` + any updated import paths
  - Pre-commit: pre-flight grep + history check pass

- [x] 5. Move dev-scripts + archive sam_glyph_test

  **What to do**:
  - `mkdir dev-scripts`
  - `git mv patch.js patch2.js patch_workspace.js patch_workspace.sh dev-scripts/`
  - Create `dev-scripts/README.md` with:
    - Purpose: "One-off patch scripts used historically to migrate frontend code. Kept for reference and reproducibility."
    - For each script: 1-line description (read first 5 lines of each to infer purpose)
    - Note: "These are not part of the install/build flow."
  - `mkdir -p _legacy && git mv _archive/sam_glyph_test _legacy/sam_glyph_test`
  - If `_archive/` is now empty: `rmdir _archive`

  **Must NOT do**:
  - Do NOT delete the patch scripts (user wants them moved with explanation)
  - Do NOT execute the patch scripts to figure out what they do — only read first lines
  - Do NOT touch `.sisyphus/`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File organization
  - **Skills**: [`git-master`]
    - `git-master`: Preserve history

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T4)
  - **Parallel Group**: Wave 1
  - **Blocks**: F4
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `patch.js`, `patch2.js`, `patch_workspace.js`, `patch_workspace.sh` (root)
  - `_archive/sam_glyph_test/` (existing archive)

  **WHY Each Reference Matters**:
  - Reduces root-level noise; consolidates legacy in one place

  **Acceptance Criteria**:
  - [ ] `dev-scripts/` contains all 4 patch files + README.md
  - [ ] No `patch*.js` or `patch_workspace.sh` at repo root
  - [ ] `_legacy/sam_glyph_test/` exists, `_archive/` either gone or documented
  - [ ] `dev-scripts/README.md` is non-empty and explains each script

  **QA Scenarios**:

  ```
  Scenario: Dev-scripts consolidated with documentation
    Tool: Bash
    Preconditions: T1 complete
    Steps:
      1. `ls dev-scripts/ | wc -l` → assert >= 5 (4 scripts + README)
      2. `test -f dev-scripts/README.md && [ $(wc -l < dev-scripts/README.md) -gt 5 ]` → exit 0
      3. `ls patch*.js patch_workspace.sh 2>/dev/null | wc -l` → 0
      4. `test -d _legacy/sam_glyph_test` → exit 0
    Expected Result: All assertions pass
    Failure Indicators: Files still at root, missing README
    Evidence: .sisyphus/evidence/codex-cleanup/task-5-dev-scripts.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-dev-scripts.txt (`ls dev-scripts/`, `cat dev-scripts/README.md`, `ls _legacy/`)

  **Commit**: YES
  - Message: `chore(repo): consolidate dev-scripts and archive sam_glyph_test`
  - Files: `dev-scripts/`, `_legacy/sam_glyph_test/`
  - Pre-commit: presence checks

- [x] 6. Update CI workflow paths

  **What to do**:
  - Edit `.github/workflows/smoke.yml`: replace all `codex-frontend` with `frontend`, all `frontend_integration_fix/frontend_integration` with `backend`
  - Re-read the workflow to verify it still makes sense (cache paths, working-directory, etc.)
  - Local validation: `python -c "import yaml; yaml.safe_load(open('.github/workflows/smoke.yml'))"` (no parse errors)
  - Optional: run `act` if installed; otherwise trust grep + parse

  **Must NOT do**:
  - Do NOT add new CI jobs in this task (scope creep)
  - Do NOT change Node/Python versions
  - Do NOT touch other workflows if they exist

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Targeted text replacement in YAML
  - **Skills**: []
    - No special skill needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T3 + T4 completing)
  - **Parallel Group**: Wave 1 (last in wave)
  - **Blocks**: F2
  - **Blocked By**: T3, T4

  **References**:

  **Pattern References**:
  - `.github/workflows/smoke.yml` — current CI definition

  **WHY Each Reference Matters**:
  - CI must stay green after rename or PRs will fail

  **Acceptance Criteria**:
  - [ ] `grep -E "codex-frontend|frontend_integration_fix" .github/workflows/smoke.yml | wc -l` → 0
  - [ ] YAML parses without error
  - [ ] All `working-directory:` values are `frontend` or `backend` (or absent)

  **QA Scenarios**:

  ```
  Scenario: CI workflow updated and valid
    Tool: Bash
    Preconditions: T3 + T4 complete
    Steps:
      1. `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/smoke.yml'))"` → exit 0
      2. `! grep -E "codex-frontend|frontend_integration_fix" .github/workflows/smoke.yml`
      3. `grep -c "working-directory: frontend" .github/workflows/smoke.yml` → assert >= 1
    Expected Result: All checks pass
    Failure Indicators: Old paths remain or YAML invalid
    Evidence: .sisyphus/evidence/codex-cleanup/task-6-ci.txt
  ```

  **Evidence to Capture**:
  - [ ] task-6-ci.txt (full smoke.yml after edit + grep results)

  **Commit**: YES
  - Message: `ci(smoke): update workflow paths post-restructure`
  - Files: `.github/workflows/smoke.yml`
  - Pre-commit: yaml parse + grep checks

- [x] 7. Backend env-driven config (PORT/HOST/CORS/MODEL_DIR)

  **What to do**:
  - Edit `backend/examples/flask_api.py`: read `PORT` (default 7117), `HOST` (default `0.0.0.0`), `CORS_ORIGINS` (comma-separated, default `http://localhost:7118`), `MODEL_DIR` (default `./models` or current path) from env via `os.environ.get`
  - Use these values in `app.run(host=HOST, port=int(PORT))` and CORS config (e.g. `flask_cors.CORS(app, origins=CORS_ORIGINS.split(","))`)
  - At startup, log: `[startup] backend listening on {host}:{port}, CORS allowing {origins}, model dir {model_dir}`
  - Add startup pre-flight check: if `MODEL_DIR` doesn't exist or required `.pt` files missing → log clear error with remediation hint and exit non-zero
  - Create `backend/.env.example` with all four vars + comments explaining each

  **Must NOT do**:
  - Do NOT hardcode 5000, 7117, or any port elsewhere — env-only with defaults
  - Do NOT change route handlers or business logic
  - Do NOT add new dependencies (flask-cors should already be present; if not, document but skip)
  - Do NOT include real secrets in `.env.example`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Touches config + startup + error handling, needs care
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T8-T14, different files)
  - **Parallel Group**: Wave 2
  - **Blocks**: F1, F3 (must run backend), T15/T16 (install scripts reference these vars)
  - **Blocked By**: T4

  **References**:

  **Pattern References**:
  - `backend/examples/flask_api.py` — Flask entry point (was `frontend_integration_fix/frontend_integration/examples/flask_api.py`)

  **External References**:
  - Flask docs: https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.run
  - flask-cors: https://flask-cors.readthedocs.io/

  **WHY Each Reference Matters**:
  - flask_api.py currently hardcodes `0.0.0.0:5000` — must become env-driven
  - CORS docs needed to validate the origins-list pattern

  **Acceptance Criteria**:
  - [ ] `PORT=7117 python backend/examples/flask_api.py &` then `curl -sf http://localhost:7117/health` returns 200
  - [ ] `PORT=9999 python backend/examples/flask_api.py &` binds 9999 (env override works)
  - [ ] Missing `MODEL_DIR` → process exits non-zero with clear error
  - [ ] `backend/.env.example` exists with `PORT=7117`, `HOST=0.0.0.0`, `CORS_ORIGINS=http://localhost:7118`, `MODEL_DIR=...`
  - [ ] `! grep -nE ":5000|port=5000" backend/examples/flask_api.py`

  **QA Scenarios**:

  ```
  Scenario: Default port binding
    Tool: Bash
    Preconditions: T4 complete, deps installed
    Steps:
      1. Free port 7117: `lsof -ti:7117 | xargs -r kill -9`
      2. Run `python backend/examples/flask_api.py &` (no env), capture pid
      3. Wait 5s, run `curl -sf http://localhost:7117/health`
      4. Assert response status 200, body contains "ok" or "healthy"
      5. Kill pid
    Expected Result: Backend binds 7117, /health returns 200
    Failure Indicators: Wrong port, /health 404 or 500
    Evidence: .sisyphus/evidence/codex-cleanup/task-7-default-port.txt

  Scenario: Env override
    Tool: Bash
    Preconditions: same
    Steps:
      1. `PORT=9999 python backend/examples/flask_api.py &`
      2. `curl -sf http://localhost:9999/health` → assert 200
      3. `curl -sf http://localhost:7117/health` → assert connection refused
      4. Kill
    Expected Result: PORT env wins
    Evidence: .sisyphus/evidence/codex-cleanup/task-7-env-override.txt

  Scenario: Missing model dir error (negative path)
    Tool: Bash
    Preconditions: same
    Steps:
      1. `MODEL_DIR=/nonexistent python backend/examples/flask_api.py 2>&1 | tee task-7-missing-model.txt`
      2. Assert exit code != 0
      3. Assert output contains "MODEL_DIR" and remediation hint
    Expected Result: Process exits with helpful error
    Evidence: .sisyphus/evidence/codex-cleanup/task-7-missing-model.txt
  ```

  **Evidence to Capture**:
  - [ ] task-7-default-port.txt, task-7-env-override.txt, task-7-missing-model.txt

  **Commit**: YES
  - Message: `feat(backend): env-driven port/host/CORS config`
  - Files: `backend/examples/flask_api.py`, `backend/.env.example`
  - Pre-commit: all 3 QA scenarios pass

- [x] 8. Frontend Vite port 7118 + .env.example

  **What to do**:
  - Edit `frontend/vite.config.ts` to add `server: { port: 7118, strictPort: false }` to defineConfig
  - Create `frontend/.env.example` with `VITE_API_BASE_URL=http://localhost:7117` (and any other VITE_ vars currently used; check `frontend/src/services/api.ts` for `import.meta.env.*` references)
  - Verify: `cd frontend && npm run dev` (background) then `curl -sf http://localhost:7118/` returns HTML

  **Must NOT do**:
  - Do NOT use `strictPort: true` (researchers may have port conflicts; allow Vite to fall back)
  - Do NOT change React/Vite versions
  - Do NOT modify `package.json` scripts beyond what's necessary

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Two small file edits

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7, T9-T14)
  - **Parallel Group**: Wave 2
  - **Blocks**: F3, T15/T16
  - **Blocked By**: T3

  **References**:

  **Pattern References**:
  - `frontend/vite.config.ts` — current Vite config
  - `frontend/src/services/api.ts` — uses `import.meta.env.VITE_API_BASE_URL`

  **External References**:
  - Vite server config: https://vitejs.dev/config/server-options.html

  **WHY Each Reference Matters**:
  - Vite default 5173 conflicts with other dev tools; explicit port = predictability
  - `.env.example` needs to mirror all env vars the frontend reads

  **Acceptance Criteria**:
  - [ ] `cd frontend && npm run dev` binds 7118 (or fallback if 7118 busy and `strictPort: false`)
  - [ ] `frontend/.env.example` includes `VITE_API_BASE_URL=http://localhost:7117`
  - [ ] `cat frontend/vite.config.ts | grep -E "port:\s*7118"` returns match

  **QA Scenarios**:

  ```
  Scenario: Frontend binds 7118
    Tool: Bash
    Preconditions: T3 complete, npm ci done
    Steps:
      1. `lsof -ti:7118 | xargs -r kill -9`
      2. `cd frontend && npm run dev > /tmp/vite.log 2>&1 &`
      3. Wait 8s, run `curl -sf http://localhost:7118/` → assert returns HTML containing `<div id="root">`
      4. Kill background
    Expected Result: Vite serves on 7118
    Failure Indicators: Connection refused on 7118, or HTML doesn't have root div
    Evidence: .sisyphus/evidence/codex-cleanup/task-8-vite-port.txt
  ```

  **Evidence to Capture**:
  - [ ] task-8-vite-port.txt (vite.log + curl output)

  **Commit**: YES
  - Message: `feat(frontend): bind to port 7118 + env example`
  - Files: `frontend/vite.config.ts`, `frontend/.env.example`
  - Pre-commit: QA scenario passes

- [x] 9. Annotation bug fix #1 — pointer events + setPointerCapture

  **What to do**:
  - In `frontend/src/pages/AnnotationPage.tsx`: migrate `handleSvgMouseDown` → `handleSvgPointerDown`, `handleSvgMouseMove` → `handleSvgPointerMove`, `handleSvgMouseUp` → `handleSvgPointerUp`
  - Change SVG element handlers from `onMouseDown/Move/Up` to `onPointerDown/Move/Up`
  - On pointerdown: call `e.currentTarget.setPointerCapture(e.pointerId)` to capture the pointer (fixes mouseup-outside-SVG losing drag state)
  - On pointerup: call `e.currentTarget.releasePointerCapture(e.pointerId)`
  - Type: `React.PointerEvent<SVGSVGElement>` instead of MouseEvent
  - Add a regression test in `frontend/src/pages/__tests__/AnnotationPage.test.tsx` (Vitest + React Testing Library) that simulates pointerdown → pointermove (outside SVG bounds) → pointerup, asserts drag completes

  **Must NOT do**:
  - Do NOT change the underlying drag/resize logic — only event plumbing
  - Do NOT introduce a third-party drag library
  - Do NOT remove existing keyboard accessibility if any
  - Do NOT touch handle hit-test logic (that's T10)
  - Do NOT touch coord conversion (that's T11)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: React event refactor with regression test
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: PARTIAL — T9, T10, T11, T12 all touch AnnotationPage.tsx; do them sequentially OR coordinate via clear merge protocol. Recommended: do T9 first, then T10, T11, T12 each as separate commits
  - **Parallel Group**: Wave 2 (sequenced within file)
  - **Blocks**: F3
  - **Blocked By**: T3

  **References**:

  **Pattern References**:
  - `frontend/src/pages/AnnotationPage.tsx:99-258` — current mouse handlers
  - `frontend/src/pages/AnnotationPage.tsx:354-396` — SVG wrapper element

  **External References**:
  - PointerEvent + setPointerCapture: https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
  - React pointer events: https://react.dev/reference/react-dom/components/common#pointer-events

  **WHY Each Reference Matters**:
  - Pointer events unify mouse/touch/pen and pointer capture is the standard fix for "mouseup outside element loses drag"

  **Acceptance Criteria**:
  - [ ] All `MouseEvent` types in AnnotationPage.tsx event handlers replaced with `PointerEvent`
  - [ ] `setPointerCapture` called in pointerdown
  - [ ] `releasePointerCapture` called in pointerup
  - [ ] `npm test -- AnnotationPage` passes (≥1 regression test for pointer-outside-SVG case)
  - [ ] No TypeScript errors: `cd frontend && npx tsc --noEmit`

  **QA Scenarios**:

  ```
  Scenario: Drag completes when pointer leaves SVG bounds
    Tool: Playwright (playwright skill)
    Preconditions: T8 done, dev server running on 7118, AnnotationPage accessible at /annotate/<some-id>
    Steps:
      1. Navigate to annotation page with a sample image loaded
      2. Locate SVG element via selector `svg[data-testid="annotation-svg"]` (add testid if missing in same task)
      3. `page.mouse.move(x_inside, y_inside)`
      4. `page.mouse.down()` — start drag
      5. `page.mouse.move(x_outside_svg, y_outside_svg, { steps: 10 })` — drag outside
      6. `page.mouse.up()`
      7. Assert: a new annotation box exists in DOM (`svg rect[data-role="annotation"]` count increased by 1)
    Expected Result: Box created end-to-end, no stale drag state
    Failure Indicators: No box created, or box stays as preview ("ghost") indicating drag never finalized
    Evidence: .sisyphus/evidence/codex-cleanup/task-9-pointer-outside.png

  Scenario: Vitest regression test green
    Tool: Bash
    Preconditions: T17 done OR scaffold inline here
    Steps:
      1. `cd frontend && npm test -- AnnotationPage --run 2>&1 | tee task-9-vitest.txt`
      2. Assert exit 0 and "passed" present
    Evidence: .sisyphus/evidence/codex-cleanup/task-9-vitest.txt
  ```

  **Evidence to Capture**:
  - [ ] task-9-pointer-outside.png, task-9-vitest.txt

  **Commit**: YES
  - Message: `fix(annotation): use pointer events with setPointerCapture`
  - Files: `frontend/src/pages/AnnotationPage.tsx`, `frontend/src/pages/__tests__/AnnotationPage.test.tsx` (new)
  - Pre-commit: tsc + test pass

- [x] 10. Annotation bug fix #2 — scan-all handle hit-test

  **What to do**:
  - In `frontend/src/pages/AnnotationPage.tsx` `handleSvgPointerDown` (post-T9): BEFORE checking which annotation is under cursor for body-hit, iterate over ALL annotations and check if cursor is on any of their resize handles
  - If a handle hit is found: focus that annotation AND start resize drag in one event (no second click required)
  - The `getHitHandle` helper (currently L89-97) takes a single annotation; either: (a) keep it as-is and call it in a loop, or (b) create `findHandleHitAcrossAll(annotations, cursor)` returning `{ annotationId, handle } | null`
  - Add regression test: render two annotations, click directly on second annotation's NW handle (where second is not currently focused) — assert second becomes focused AND resize starts in same event

  **Must NOT do**:
  - Do NOT change handle visual rendering (T12 handles size harmonization)
  - Do NOT change the bbox-hit fallback logic (clicking inside body still focuses)
  - Do NOT exceed O(N) iteration (no fancy spatial index)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Subtle interaction logic + test

  **Parallelization**:
  - **Can Run In Parallel**: NO — same file as T9, T11, T12
  - **Parallel Group**: Wave 2 (sequenced after T9)
  - **Blocks**: F3
  - **Blocked By**: T9

  **References**:

  **Pattern References**:
  - `frontend/src/pages/AnnotationPage.tsx:89-97` — current `getHitHandle`
  - `frontend/src/pages/AnnotationPage.tsx:99-144` — handleSvgMouseDown (will be PointerDown post-T9)

  **WHY Each Reference Matters**:
  - The bug is structural: handle hit-test only runs against the focused annotation, so other annotations need a focus-click before their handles respond

  **Acceptance Criteria**:
  - [ ] Pointerdown directly on a non-focused annotation's handle → that annotation becomes focused AND resize begins in same event
  - [ ] Body click (no handle) still focuses the clicked annotation
  - [ ] Vitest regression test green
  - [ ] No TS errors

  **QA Scenarios**:

  ```
  Scenario: Direct handle click on non-focused annotation works first try
    Tool: Playwright
    Preconditions: T9 done, dev server up
    Steps:
      1. Navigate to AnnotationPage with 2 sample annotations created
      2. Click annotation 1 to focus it
      3. Locate annotation 2's NW handle by selector `[data-annotation-id="2"] [data-handle="nw"]` (add testids if missing)
      4. `page.mouse.move(handle.x, handle.y); page.mouse.down(); page.mouse.move(handle.x - 30, handle.y - 30); page.mouse.up()`
      5. Assert annotation 2 is now focused AND its bbox shrunk/grew (NW corner moved)
      6. Assert NO state where "first click did nothing, second click resized" — verify by checking annotation 2 was not focused at step 3 start
    Expected Result: Single-action focus + resize
    Failure Indicators: Annotation 2 only focused but not resized, or no change
    Evidence: .sisyphus/evidence/codex-cleanup/task-10-handle-direct.png
  ```

  **Evidence to Capture**:
  - [ ] task-10-handle-direct.png

  **Commit**: YES
  - Message: `fix(annotation): scan all elements for handle hit-test`
  - Files: `frontend/src/pages/AnnotationPage.tsx`, test file
  - Pre-commit: tsc + test pass

- [x] 11. Annotation bug fix #3 — getScreenCTM coord mapping

  **What to do**:
  - In `frontend/src/pages/AnnotationPage.tsx` pointer handlers: replace all `e.nativeEvent.offsetX/offsetY` with a coord conversion using `svgEl.getScreenCTM().inverse()` applied to `e.clientX, e.clientY`
  - Helper signature: `function clientToSvg(svg: SVGSVGElement, clientX: number, clientY: number): { x: number; y: number }`
  - Use this helper in pointerdown, pointermove, AND in `getHitHandle` calls so all coords are in SVG-viewBox space regardless of CSS zoom transform on the wrapper (currently L354-366)
  - Regression test: simulate scenario where wrapper has `transform: scale(1.5)` applied; pointerdown at clientX=Y; assert internal coords match expected SVG-space value (within 1px)

  **Must NOT do**:
  - Do NOT remove the zoom feature
  - Do NOT use bounding-rect math (`svg.getBoundingClientRect()`) — getScreenCTM is the correct approach for transforms
  - Do NOT cache CTM across events (CTM may change if zoom changes mid-interaction)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Coord-math correctness, easy to get wrong

  **Parallelization**:
  - **Can Run In Parallel**: NO — same file
  - **Parallel Group**: Wave 2 (after T10)
  - **Blocks**: F3
  - **Blocked By**: T10

  **References**:

  **Pattern References**:
  - `frontend/src/pages/AnnotationPage.tsx:354-396` — SVG with `transform: scale(zoom)` wrapper
  - `frontend/src/pages/AnnotationPage.tsx:99-258` — handlers using offsetX/Y

  **External References**:
  - getScreenCTM: https://developer.mozilla.org/en-US/docs/Web/API/SVGGraphicsElement/getScreenCTM
  - SVGPoint matrixTransform pattern: https://developer.mozilla.org/en-US/docs/Web/API/SVGPoint/matrixTransform

  **WHY Each Reference Matters**:
  - `offsetX/Y` is unreliable under CSS transforms — getScreenCTM().inverse() is the canonical fix

  **Acceptance Criteria**:
  - [ ] No `offsetX` or `offsetY` references in AnnotationPage event handlers
  - [ ] Helper `clientToSvg` defined and used consistently
  - [ ] Vitest test simulating zoomed scenario passes (within 1px tolerance)
  - [ ] No TS errors

  **QA Scenarios**:

  ```
  Scenario: Annotation accuracy under zoom
    Tool: Playwright
    Preconditions: T10 done
    Steps:
      1. Navigate to AnnotationPage
      2. Apply zoom (2x) via existing zoom UI control
      3. Click at a known point on the visible image (e.g. center of a glyph)
      4. Drag to create annotation box
      5. Read back annotation bbox from state (e.g. via `data-bbox` attribute or DOM inspection)
      6. Assert bbox center is within 5px of intended target in image-space (not screen-space)
    Expected Result: Annotation lands precisely where clicked
    Failure Indicators: Annotation offset/scaled by zoom factor
    Evidence: .sisyphus/evidence/codex-cleanup/task-11-zoom-accuracy.png
  ```

  **Evidence to Capture**:
  - [ ] task-11-zoom-accuracy.png

  **Commit**: YES
  - Message: `fix(annotation): map client coords via getScreenCTM`
  - Files: `frontend/src/pages/AnnotationPage.tsx`, test file
  - Pre-commit: tsc + test pass

- [x] 12. Annotation polish — handle size harmonization + RAF throttle

  **What to do**:
  - In `frontend/src/pages/AnnotationPage.tsx`: harmonize handle visual size and hit area. Currently visual is ~8x8 and hit-test radius is 12. Either: (a) increase visual to match hit area (`width=12 height=12`), or (b) keep visual 8 but make hit area an invisible larger rect (12x12) sibling. Choose (a) for simplicity unless visually too chunky.
  - Add RAF throttling on pointermove: wrap the move handler body in `requestAnimationFrame` so updates batch to display refresh; cancel pending RAF on subsequent move
  - Pattern: `const rafRef = useRef<number | null>(null); const onPointerMove = (e) => { if (rafRef.current) cancelAnimationFrame(rafRef.current); rafRef.current = requestAnimationFrame(() => { /* body */ }); }`
  - Cleanup RAF on pointerup and on unmount

  **Must NOT do**:
  - Do NOT change handle positions (still 4 corners + 4 sides if currently so)
  - Do NOT introduce `lodash.throttle` or third-party throttle utilities
  - Do NOT skip RAF cleanup (memory leak)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small polish changes

  **Parallelization**:
  - **Can Run In Parallel**: NO — same file
  - **Parallel Group**: Wave 2 (after T11)
  - **Blocks**: F3
  - **Blocked By**: T11

  **References**:

  **Pattern References**:
  - `frontend/src/pages/AnnotationPage.tsx:89-97` — `getHitHandle` (uses radius 12)
  - `frontend/src/pages/AnnotationPage.tsx` handle render section (search for `<rect` with handle classes/data)

  **External References**:
  - requestAnimationFrame throttle: https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame

  **WHY Each Reference Matters**:
  - Visual/hit mismatch causes "I clicked the handle but nothing happened" feeling
  - RAF throttling smooths drag on slower devices

  **Acceptance Criteria**:
  - [ ] Handle visual size matches hit area (or hit area is intentionally larger via invisible rect)
  - [ ] `requestAnimationFrame` used in pointermove
  - [ ] `cancelAnimationFrame` called on subsequent move + on cleanup
  - [ ] Manual smoothness check: drag feels smooth at 60fps

  **QA Scenarios**:

  ```
  Scenario: Handle visually matches click target
    Tool: Playwright
    Steps:
      1. Navigate to AnnotationPage with annotation present
      2. Take screenshot, inspect handle: width and height attributes both 12 (or both equal value)
      3. Click 5px from handle center → resize starts (within hit area)
      4. Click 20px from handle center → resize does NOT start (outside hit area)
    Expected Result: Visual and hit area aligned
    Evidence: .sisyphus/evidence/codex-cleanup/task-12-handle-size.png

  Scenario: RAF throttling active
    Tool: Bash + Playwright
    Steps:
      1. In test, monkey-patch requestAnimationFrame to count calls
      2. Simulate 100 rapid pointermove events
      3. Assert RAF call count <= 100 AND state updates count <= number of frames (typically much less than 100)
    Expected Result: Throttling reduces render frequency
    Evidence: .sisyphus/evidence/codex-cleanup/task-12-raf.txt
  ```

  **Evidence to Capture**:
  - [ ] task-12-handle-size.png, task-12-raf.txt

  **Commit**: YES
  - Message: `polish(annotation): harmonize handle hit area + RAF throttle`
  - Files: `frontend/src/pages/AnnotationPage.tsx`
  - Pre-commit: tsc + tests pass

- [x] 13. Reconcile image preprocessing (gray vs white pad)

  **What to do**:
  - Locate `CodexClassifier._preprocess_image` and `InferenceEngine.preprocess` in `backend/` (paths likely `backend/codex_model/` and `backend/codex_pipeline/`)
  - Identify the pad color discrepancy: classifier uses `(128,128,128)` gray, engine uses `(255,255,255)` white
  - Decide on **white (255)** as canonical (matches the projection/prototype precompute path, which is what `/similar` and `/trust` use)
  - Update the classifier to use white pad
  - Add a brief comment in both functions: `# Pad color: white (255). Must stay consistent with projection-head training data and engine preprocess.`
  - Add a pytest unit test (`backend/tests/test_preprocess_consistency.py`) that:
    - Creates a small dummy PIL image
    - Calls both preprocessing functions
    - Asserts the resulting tensors are equal (or assert pad pixels are white in both)

  **Must NOT do**:
  - Do NOT retrain or re-export model weights
  - Do NOT change image resize dimensions, normalization, or color mode
  - Do NOT change any other preprocessing step (only pad color)
  - Do NOT modify training scripts (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: ML pipeline correctness, easy to subtly break

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7-T12, T14)
  - **Parallel Group**: Wave 2
  - **Blocks**: F2
  - **Blocked By**: T4

  **References**:

  **Pattern References**:
  - `backend/codex_model/` — classifier implementation (search for `_preprocess_image`)
  - `backend/codex_pipeline/` (or `inference.py`) — InferenceEngine

  **External References**:
  - PIL ImageOps.pad: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.pad

  **WHY Each Reference Matters**:
  - Inconsistent pad color = subtly different feature vectors = bad similarity rankings; this is a latent bug

  **Acceptance Criteria**:
  - [ ] Both preprocessing functions use `(255, 255, 255)` pad color
  - [ ] Comment present in both
  - [ ] `cd backend && pytest tests/test_preprocess_consistency.py` exits 0
  - [ ] No other preprocessing parameters changed (verify via diff)

  **QA Scenarios**:

  ```
  Scenario: Pad consistency unit test passes
    Tool: Bash
    Preconditions: T4 + T18 done (pytest scaffolded)
    Steps:
      1. `cd backend && pytest tests/test_preprocess_consistency.py -v 2>&1 | tee task-13-pytest.txt`
      2. Assert exit 0
      3. Assert "passed" in output
    Evidence: .sisyphus/evidence/codex-cleanup/task-13-pytest.txt

  Scenario: End-to-end /similar still works after change
    Tool: Bash (curl)
    Preconditions: T7 done, backend running on 7117
    Steps:
      1. Pick a sample image path from existing test fixtures (or use /classes endpoint)
      2. POST a sample image to /similar
      3. Assert 200 response with non-empty results array
    Expected Result: Similarity endpoint still functional
    Evidence: .sisyphus/evidence/codex-cleanup/task-13-similar-smoke.txt
  ```

  **Evidence to Capture**:
  - [ ] task-13-pytest.txt, task-13-similar-smoke.txt

  **Commit**: YES
  - Message: `fix(backend): unify image padding color across pipelines`
  - Files: classifier file + engine file + new pytest
  - Pre-commit: pytest + smoke pass

- [x] 14. Update .gitignore (dist, evidence)

  **What to do**:
  - Append to `.gitignore`:
    - `frontend/dist/`
    - `frontend/node_modules/` (if not already)
    - `backend/__pycache__/` (if not already)
    - `backend/*.egg-info/`
    - `.sisyphus/evidence/`
    - `.env` (NOT `.env.example` — that one is committed)
  - Untrack already-committed `frontend/dist/`: `git rm -r --cached frontend/dist 2>/dev/null || true`
  - Verify: `git ls-files | grep -E "frontend/dist/|\.sisyphus/evidence/" | wc -l` → 0

  **Must NOT do**:
  - Do NOT commit `.env` (only `.env.example`)
  - Do NOT untrack model weights
  - Do NOT untrack `.sisyphus/plans/`, `.sisyphus/notepads/`, `.sisyphus/state/` — those are project state
  - Do NOT use `**/dist` or other broad globs that might catch unrelated dist dirs in `_legacy/`

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: T3

  **References**:

  **Pattern References**:
  - `.gitignore` (current)
  - `frontend/dist/` (currently tracked, must untrack)

  **WHY Each Reference Matters**:
  - Dist artifacts cause bloated diffs and merge conflicts

  **Acceptance Criteria**:
  - [ ] `.gitignore` includes the listed entries
  - [ ] `git ls-files | grep "frontend/dist/"` returns 0 lines
  - [ ] `.env.example` files still tracked
  - [ ] Model weights still tracked (or LFS-tracked, whichever was the case)

  **QA Scenarios**:

  ```
  Scenario: Ignored paths are no longer tracked
    Tool: Bash
    Steps:
      1. `git ls-files | grep -E "frontend/dist/|\.sisyphus/evidence/" | wc -l` → assert 0
      2. `git ls-files | grep "\.env\.example" | wc -l` → assert >= 2
      3. `git check-ignore frontend/dist/index.html` → exit 0 (ignored)
    Evidence: .sisyphus/evidence/codex-cleanup/task-14-gitignore.txt
  ```

  **Evidence to Capture**:
  - [ ] task-14-gitignore.txt

  **Commit**: YES
  - Message: `chore(git): ignore dist and evidence dirs`
  - Files: `.gitignore` + tracked-removal of `frontend/dist/*`
  - Pre-commit: assertions pass

- [ ] 15. `scripts/run-dev.sh` — one-command dev launcher

  **What to do**:
  - Create `scripts/run-dev.sh` (executable, `chmod +x`)
  - Behavior:
    - Source `backend/.env` and `frontend/.env` if they exist (else use `.env.example` as fallback with warning)
    - Pre-flight checks: `python3 --version` >= 3.10, `node --version` >= 20, ports 7117 and 7118 are free (`lsof -ti:7117` empty)
    - If port busy: print error with `lsof -ti:7117 | xargs kill -9` hint and exit 1
    - Activate Python venv if `backend/.venv/` exists; otherwise warn
    - Launch backend: `(cd backend && python examples/flask_api.py) &` capture pid → `BACKEND_PID`
    - Wait for backend health: poll `curl -sf http://localhost:7117/health` up to 30s; abort + kill backend if timeout
    - Launch frontend: `(cd frontend && npm run dev) &` capture pid → `FRONTEND_PID`
    - Wait for frontend: poll `curl -sf http://localhost:7118/` up to 20s
    - Print clear banner: `Backend: http://localhost:7117 | Frontend: http://localhost:7118 | Logs: ./logs/`
    - Tee both stdout/stderr to `logs/backend.log` and `logs/frontend.log`
    - Trap SIGINT/SIGTERM/EXIT: kill both pids cleanly
  - Create `logs/.gitkeep` so directory exists; ensure `logs/` is in `.gitignore` (T14)

  **Must NOT do**:
  - Do NOT install dependencies (that's `install.sh`)
  - Do NOT use Docker
  - Do NOT auto-open browser (researchers may be on headless ssh)
  - Do NOT silently overwrite log files — append with date prefix or rotate

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Bash scripting with traps, port checks, health polling — needs care

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T16-T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: F3 (used by manual QA), T19 (README references this)
  - **Blocked By**: T7, T8, T14

  **References**:

  **Pattern References**:
  - `backend/.env.example` (created in T7)
  - `frontend/.env.example` (created in T8)

  **External References**:
  - Bash trap idioms: https://www.gnu.org/software/bash/manual/html_node/Signals.html
  - lsof port check: `lsof -ti:PORT`

  **WHY Each Reference Matters**:
  - One-command dev start is the #1 quality-of-life win for researchers

  **Acceptance Criteria**:
  - [ ] `bash scripts/run-dev.sh` from clean state launches both services and prints banner within 60s
  - [ ] Ctrl+C cleanly kills both pids (no orphans: `lsof -ti:7117` and `lsof -ti:7118` empty within 5s)
  - [ ] Port-busy scenario shows clear error and non-zero exit
  - [ ] Logs written to `logs/backend.log` and `logs/frontend.log`

  **QA Scenarios**:

  ```
  Scenario: Happy path launch + clean shutdown
    Tool: interactive_bash (tmux)
    Preconditions: T7+T8+T14 done, deps installed
    Steps:
      1. Create tmux session, run `bash scripts/run-dev.sh`
      2. Wait 60s, capture pane: assert banner "Backend: http://localhost:7117" and "Frontend: http://localhost:7118" present
      3. From another shell: `curl -sf http://localhost:7117/health` → 200 AND `curl -sf http://localhost:7118/` → HTML
      4. Send Ctrl+C to tmux pane (`send-keys C-c`)
      5. Wait 5s, assert `lsof -ti:7117` and `lsof -ti:7118` both empty
    Expected Result: Clean lifecycle
    Failure Indicators: Orphan processes, missing banner, health check 500
    Evidence: .sisyphus/evidence/codex-cleanup/task-15-launch.txt + tmux capture

  Scenario: Port already in use (negative)
    Tool: Bash
    Steps:
      1. `python3 -m http.server 7117 &` (occupy port)
      2. `bash scripts/run-dev.sh 2>&1 | tee task-15-port-busy.txt; echo $?`
      3. Assert exit code != 0 AND output contains "7117" + "in use" or similar + remediation hint
      4. Cleanup: kill the http.server
    Expected Result: Graceful failure with actionable message
    Evidence: .sisyphus/evidence/codex-cleanup/task-15-port-busy.txt
  ```

  **Evidence to Capture**:
  - [ ] task-15-launch.txt, task-15-port-busy.txt

  **Commit**: YES
  - Message: `feat(scripts): add run-dev.sh one-command launcher`
  - Files: `scripts/run-dev.sh`, `logs/.gitkeep`
  - Pre-commit: both QA scenarios pass

- [ ] 16. `scripts/install.sh` + `scripts/install.ps1` + `scripts/download-weights.sh`

  **What to do**:
  - **`scripts/install.sh`** (Mac/Linux, executable):
    - Detect OS (`uname -s`)
    - Check & install prereqs: Python 3.10+, Node 20+ (instruct user with brew/apt commands; do NOT auto-install)
    - Create Python venv: `python3 -m venv backend/.venv`
    - Install backend deps: `backend/.venv/bin/pip install -r backend/requirements.txt`
    - Install frontend deps: `cd frontend && npm ci` (or `npm install` if no lockfile yet)
    - Copy `backend/.env.example → backend/.env` if not exists; same for frontend
    - Call `scripts/download-weights.sh` (skip if `MODEL_DIR` already populated)
    - Print success summary + "Next: `bash scripts/run-dev.sh`"
  - **`scripts/install.ps1`** (Windows): minimal mirror — `python -m venv`, `pip install`, `npm ci`, env file copy, weight download. Note in README that Windows is best-effort.
  - **`scripts/download-weights.sh`**:
    - Reads `MODEL_DIR` from `backend/.env` (default `backend/models/`)
    - Checks if required `.pt` files exist; if yes, exit 0 with "weights present" message
    - If missing: download from a placeholder URL (TODO marker `# TODO: replace with real download URL`) — user will provide later
    - Verify SHA256 if checksum file present
    - For now (since weights URL not provided): print clear message "Weights expected at $MODEL_DIR but not found. Place model files manually or update download URL in scripts/download-weights.sh."
  - All scripts: `set -euo pipefail` at top, clear stdout, exit codes meaningful

  **Must NOT do**:
  - Do NOT auto-install Python/Node (security/permissions concern; just instruct)
  - Do NOT use sudo
  - Do NOT hardcode the weights URL until user provides it (use TODO marker)
  - Do NOT skip `set -euo pipefail`
  - Do NOT delete existing `.env` files

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-platform install scripting

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T15, T17-T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: T19, T20 (docs reference these)
  - **Blocked By**: T7, T8

  **References**:

  **Pattern References**:
  - `backend/requirements.txt` (verify path post-T4)
  - `frontend/package.json` (post-T3)
  - `backend/.env.example`, `frontend/.env.example` (T7, T8)

  **External References**:
  - venv: https://docs.python.org/3/library/venv.html
  - `npm ci` vs `npm install`: https://docs.npmjs.com/cli/v10/commands/npm-ci

  **WHY Each Reference Matters**:
  - install.sh is the entry point for non-technical researchers; clarity > cleverness

  **Acceptance Criteria**:
  - [ ] `bash scripts/install.sh` from clean clone (no .venv, no node_modules) completes successfully
  - [ ] Both `.env` files exist after install (copied from .example)
  - [ ] `backend/.venv/bin/python --version` works
  - [ ] `cd frontend && node_modules/.bin/vite --version` works
  - [ ] `scripts/download-weights.sh` exits 0 if weights present, prints clear message if not
  - [ ] All scripts have `set -euo pipefail`
  - [ ] `scripts/install.ps1` exists (not necessarily QA-tested on Windows)

  **QA Scenarios**:

  ```
  Scenario: Fresh install on Linux
    Tool: Bash
    Preconditions: T7+T8 done. Use a temp dir clone or stash venv/node_modules
    Steps:
      1. Stash: `mv backend/.venv /tmp/stash-venv 2>/dev/null; mv frontend/node_modules /tmp/stash-nm 2>/dev/null`
      2. `bash scripts/install.sh 2>&1 | tee task-16-install.txt`
      3. Assert exit 0
      4. Assert `backend/.venv/bin/python --version` succeeds
      5. Assert `frontend/node_modules` exists and is non-empty
      6. Assert `backend/.env` and `frontend/.env` exist
      7. Restore stash if needed
    Expected Result: Idempotent install
    Evidence: .sisyphus/evidence/codex-cleanup/task-16-install.txt

  Scenario: download-weights handles missing weights gracefully
    Tool: Bash
    Steps:
      1. `MODEL_DIR=/tmp/empty bash scripts/download-weights.sh 2>&1 | tee task-16-weights.txt`
      2. Assert exit code: 0 with clear "manual placement needed" message OR documented non-zero with remediation
    Evidence: .sisyphus/evidence/codex-cleanup/task-16-weights.txt
  ```

  **Evidence to Capture**:
  - [ ] task-16-install.txt, task-16-weights.txt

  **Commit**: YES
  - Message: `feat(scripts): add install.sh, install.ps1, download-weights.sh`
  - Files: `scripts/install.sh`, `scripts/install.ps1`, `scripts/download-weights.sh`
  - Pre-commit: install QA scenario passes

- [ ] 17. Frontend test scaffold (Vitest + RTL)

  **What to do**:
  - Add to `frontend/package.json` devDependencies: `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`, `jsdom`, `@vitest/ui` (optional)
  - Add scripts: `"test": "vitest run"`, `"test:watch": "vitest"`, `"test:ui": "vitest --ui"`
  - Create `frontend/vitest.config.ts` (or extend `vite.config.ts`) with `test: { environment: 'jsdom', setupFiles: ['./src/test/setup.ts'], globals: true }`
  - Create `frontend/src/test/setup.ts` with `import '@testing-library/jest-dom'`
  - Create one smoke test `frontend/src/test/smoke.test.ts`: `expect(1+1).toBe(2)` to verify wiring
  - Run `npm test` → assert exit 0

  **Must NOT do**:
  - Do NOT migrate any existing code to test framework yet (only scaffold)
  - Do NOT add e2e (Playwright) here — that's project-level QA via the playwright skill
  - Do NOT touch existing components

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Scaffold + smoke test

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T15, T16, T18-T21) — but T9-T12 tests depend on this; if T9-T12 ran without scaffold, they may have inlined setup. Recommend: this task starts in Wave 3 but the test files from T9-T12 already exist and will pass once scaffold lands.
  - **Parallel Group**: Wave 3
  - **Blocks**: F2 (CI runs tests)
  - **Blocked By**: T3

  **References**:

  **Pattern References**:
  - `frontend/package.json` (current scripts)
  - `frontend/vite.config.ts` (post-T8)

  **External References**:
  - Vitest: https://vitest.dev/guide/
  - RTL: https://testing-library.com/docs/react-testing-library/intro/

  **WHY Each Reference Matters**:
  - Vitest integrates seamlessly with Vite; jsdom env required for component tests

  **Acceptance Criteria**:
  - [ ] `cd frontend && npm test` exits 0
  - [ ] Smoke test runs and passes
  - [ ] `vitest.config.ts` configured for jsdom
  - [ ] `@testing-library/jest-dom` matchers available

  **QA Scenarios**:

  ```
  Scenario: Test scaffold smoke
    Tool: Bash
    Steps:
      1. `cd frontend && npm test 2>&1 | tee task-17-vitest.txt`
      2. Assert exit 0
      3. Assert "1 passed" or similar
    Evidence: .sisyphus/evidence/codex-cleanup/task-17-vitest.txt
  ```

  **Evidence to Capture**:
  - [ ] task-17-vitest.txt

  **Commit**: YES
  - Message: `chore(frontend): scaffold Vitest + RTL`
  - Files: `frontend/package.json`, `frontend/package-lock.json`, `frontend/vitest.config.ts`, `frontend/src/test/setup.ts`, `frontend/src/test/smoke.test.ts`
  - Pre-commit: QA passes

- [ ] 18. Backend test scaffold (pytest)

  **What to do**:
  - Add `backend/requirements-dev.txt` with `pytest>=8`, `pytest-cov` (optional)
  - Create `backend/tests/__init__.py` (empty) and `backend/tests/test_smoke.py` with `def test_smoke(): assert 1+1 == 2`
  - Create `backend/pytest.ini` or `backend/pyproject.toml` `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `python_files = "test_*.py"`
  - Document in install.sh: `pip install -r backend/requirements-dev.txt` (if dev mode flag passed)
  - Run `cd backend && pytest` → assert exit 0

  **Must NOT do**:
  - Do NOT migrate existing code to tests (only scaffold)
  - Do NOT add tox/nox
  - Do NOT touch backend code

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F2, T13 (uses pytest)
  - **Blocked By**: T4

  **References**:

  **Pattern References**:
  - `backend/requirements.txt` (post-T4)

  **External References**:
  - pytest: https://docs.pytest.org/en/stable/

  **WHY Each Reference Matters**:
  - pytest is Python community standard; minimal setup

  **Acceptance Criteria**:
  - [ ] `cd backend && pytest` exits 0
  - [ ] At least the smoke test runs

  **QA Scenarios**:

  ```
  Scenario: Pytest smoke
    Tool: Bash
    Steps:
      1. `cd backend && pip install -r requirements-dev.txt && pytest -v 2>&1 | tee task-18-pytest.txt`
      2. Assert exit 0
      3. Assert "1 passed"
    Evidence: .sisyphus/evidence/codex-cleanup/task-18-pytest.txt
  ```

  **Evidence to Capture**:
  - [ ] task-18-pytest.txt

  **Commit**: YES
  - Message: `chore(backend): scaffold pytest`
  - Files: `backend/requirements-dev.txt`, `backend/tests/__init__.py`, `backend/tests/test_smoke.py`, `backend/pytest.ini` (or pyproject section)
  - Pre-commit: QA passes

- [ ] 19. README rewrite — dual-audience

  **What to do**:
  - Rewrite root `README.md` with two clear sections:
    - **Section A: Pour les chercheurs / utilisateurs** (French primary, English secondary OK)
      - What is Clinic Codex (1 paragraph, plain language, no jargon)
      - Screenshots/GIF of the app (link to `docs/screenshots/` placeholder if not yet captured)
      - Quick start: "Suivez le guide d'installation: [INSTALL.md](./INSTALL.md)"
      - How to use: workspace → upload → annotate → similar glyphs (high-level workflow, 4-5 bullets)
      - Where to get help: contact info placeholder, GitHub issues
    - **Section B: Pour les développeurs / techniciens** (English OK, French OK)
      - Architecture: backend (Flask + DINOv2) ↔ frontend (Vite + React)
      - Tech stack table
      - Project structure: tree of `backend/`, `frontend/`, `scripts/`, `docs/`, `_legacy/`, `dev-scripts/`
      - Dev quickstart: `bash scripts/install.sh && bash scripts/run-dev.sh`
      - Ports: backend 7117, frontend 7118
      - Running tests: `cd frontend && npm test` and `cd backend && pytest`
      - Contributing: branch convention, commit style
      - Pointer to `EMBEDDING-READINESS.md` (T21) and `INSTALL.md` (T20)
  - Add badges (optional): CI status, license
  - Keep total length ≤ 400 lines (linkable, not monolithic)
  - Cross-link `INSTALL.md`, `EMBEDDING-READINESS.md`, `dev-scripts/README.md`

  **Must NOT do**:
  - Do NOT include detailed install steps (those go in INSTALL.md)
  - Do NOT include API docs (separate file later if needed)
  - Do NOT use marketing language; stay factual and academic
  - Do NOT delete any existing useful content without checking — review current README first

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation, dual audience requires careful tone
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T15-T18, T20, T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: F1
  - **Blocked By**: T15, T16 (references run-dev.sh and install.sh)

  **References**:

  **Pattern References**:
  - Current `README.md` (preserve any unique technical content worth keeping)
  - Project structure post-renames (T3, T4, T5)

  **External References**:
  - README best practices: https://www.makeareadme.com/

  **WHY Each Reference Matters**:
  - Current README may have institutional context (Nahuatl scholarship pointers) worth preserving

  **Acceptance Criteria**:
  - [ ] Two sections present and clearly delineated
  - [ ] Project tree reflects new structure (`backend/`, `frontend/`, `scripts/`, `_legacy/`, `dev-scripts/`)
  - [ ] Ports 7117/7118 mentioned
  - [ ] Links to INSTALL.md and EMBEDDING-READINESS.md present (even if those files come from T20/T21 in same wave; ordering: ensure T19 commits last in Wave 3 docs, OR use forward links that will resolve)
  - [ ] Markdown lints clean (no broken links): `npx markdown-link-check README.md` (optional)

  **QA Scenarios**:

  ```
  Scenario: README structure check
    Tool: Bash
    Steps:
      1. `grep -E "^## " README.md` → assert sections for both audiences present
      2. `grep -E "7117|7118" README.md` → both ports referenced
      3. `grep -E "INSTALL\.md|EMBEDDING-READINESS\.md" README.md` → links present
    Evidence: .sisyphus/evidence/codex-cleanup/task-19-readme.txt
  ```

  **Evidence to Capture**:
  - [ ] task-19-readme.txt

  **Commit**: YES
  - Message: `docs: rewrite README with researcher + developer sections`
  - Files: `README.md`
  - Pre-commit: structure check passes

- [ ] 20. INSTALL.md — non-technical install guide

  **What to do**:
  - Create root `INSTALL.md` (French primary)
  - Audience: researchers with zero command-line experience
  - Structure:
    1. **Prérequis** — install Python 3.10+ and Node 20+ (links to official installers, screenshots placeholders for Mac/Linux/Windows)
    2. **Téléchargement du projet** — `git clone` OR download ZIP from GitHub (with screenshots)
    3. **Installation automatique** — open Terminal, navigate to folder, run `bash scripts/install.sh` (Mac/Linux) or `powershell scripts/install.ps1` (Windows)
    4. **Placement des modèles** — where to put `.pt` weight files (`backend/models/`), with placeholder for download link
    5. **Lancement** — `bash scripts/run-dev.sh`, then open http://localhost:7118 in browser
    6. **Dépannage** — common issues:
       - "Port 7117 déjà utilisé" → kill command
       - "Python introuvable" → reinstall + PATH
       - "npm: command not found" → reinstall Node
       - "Modèles manquants" → place .pt files in MODEL_DIR
    7. **Arrêt** — Ctrl+C in terminal
    8. **Mise à jour** — `git pull && bash scripts/install.sh`
  - Use numbered steps, code blocks, callouts
  - Keep ≤ 200 lines

  **Must NOT do**:
  - Do NOT assume CLI fluency
  - Do NOT include dev/contributor info (that's README section B)
  - Do NOT use Docker

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Non-technical audience, requires plain-language clarity

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F1
  - **Blocked By**: T15, T16

  **References**:

  **External References**:
  - Python install: https://www.python.org/downloads/
  - Node install: https://nodejs.org/

  **WHY Each Reference Matters**:
  - Researchers will follow this verbatim; precision matters

  **Acceptance Criteria**:
  - [ ] All 8 numbered sections present
  - [ ] References `scripts/install.sh`, `scripts/run-dev.sh`
  - [ ] Mentions `MODEL_DIR` and where to place weights
  - [ ] Troubleshooting section present
  - [ ] French primary language

  **QA Scenarios**:

  ```
  Scenario: Sections + references
    Tool: Bash
    Steps:
      1. `grep -cE "^##? " INSTALL.md` → assert >= 8
      2. `grep -E "scripts/install\.sh|scripts/run-dev\.sh" INSTALL.md` → both present
      3. `grep -iE "modèle|MODEL_DIR" INSTALL.md` → present
      4. `grep -iE "dépannage|troubleshoot" INSTALL.md` → present
    Evidence: .sisyphus/evidence/codex-cleanup/task-20-install-md.txt
  ```

  **Evidence to Capture**:
  - [ ] task-20-install-md.txt

  **Commit**: YES
  - Message: `docs: add non-technical INSTALL.md guide`
  - Files: `INSTALL.md`
  - Pre-commit: structure check passes

- [ ] 21. EMBEDDING-READINESS.md — similarity infrastructure design

  **What to do**:
  - Create root `EMBEDDING-READINESS.md` (English or French — engineer audience)
  - Document current state and proposed instance-level similarity infra (no code, doc only):
    1. **État actuel** — DINOv2-ViT-S/14 backbone (384d) + projection head (128d L2-normalized); 286 prototype classes; cosine similarity already working in prototype mode via `/similar` and `/trust`
    2. **Limitation actuelle** — similarity is class-level (prototype) not instance-level (per-image)
    3. **Cible** — instance-level: given a glyph image, return top-K most similar individual images from the dataset
    4. **Infra needed** (when dataset arrives):
       - Embedding extraction script: `backend/scripts/extract_embeddings.py` — iterates dataset, runs encoder + projection, saves `(image_id, 128d_vector)` to disk
       - Storage format: numpy `.npy` array (N×128) + JSON manifest (`{"index": [{"id": "...", "path": "...", "metadata": {...}}, ...]}`); for ≤100k images this fits in memory
       - For >100k: FAISS or hnswlib index (recommended `hnswlib` for L2-normalized cosine — `space='cosine'`)
       - API endpoints: `/embed/extract` (admin, batch), `/search/similar?image_id=X&k=20` (query)
       - Frontend integration: similarity panel in AnnotationPage / WorkspacePage showing top-K thumbnails
    5. **Décisions à prendre quand le dataset arrive**:
       - In-memory vs FAISS/hnswlib (depends on dataset size)
       - Storage location (filesystem vs database)
       - Re-extraction trigger (on dataset update)
    6. **Estimation effort** (rough): 1-2 days for in-memory MVP, +2-3 days for FAISS/hnswlib + API endpoints
    7. **Références code existant** — `backend/codex_pipeline/inference.py` (preprocess + encode), `/similar` route in `backend/examples/flask_api.py`
  - This file is a planning artifact, not user-facing

  **Must NOT do**:
  - Do NOT write any actual extraction code yet (waiting for dataset)
  - Do NOT install FAISS/hnswlib yet
  - Do NOT modify backend code

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Technical doc for future engineering work

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F1
  - **Blocked By**: T4

  **References**:

  **Pattern References**:
  - `backend/codex_pipeline/inference.py` (encoder + projection)
  - `backend/examples/flask_api.py` (`/similar` route)
  - `backend/codex_model/` (classifier preprocessing)

  **External References**:
  - hnswlib: https://github.com/nmslib/hnswlib
  - FAISS: https://github.com/facebookresearch/faiss

  **WHY Each Reference Matters**:
  - Captures architectural decisions while context is fresh; avoids re-discovery when dataset arrives

  **Acceptance Criteria**:
  - [ ] All 7 sections present
  - [ ] References existing backend files with paths
  - [ ] Lists concrete decisions to make later
  - [ ] No code/dependency changes

  **QA Scenarios**:

  ```
  Scenario: Doc completeness
    Tool: Bash
    Steps:
      1. `grep -cE "^##? " EMBEDDING-READINESS.md` → assert >= 7
      2. `grep -E "DINOv2|prototype|hnswlib|FAISS" EMBEDDING-READINESS.md` → all referenced
      3. `grep -E "/similar|/trust" EMBEDDING-READINESS.md` → existing endpoints documented
    Evidence: .sisyphus/evidence/codex-cleanup/task-21-embedding-doc.txt
  ```

  **Evidence to Capture**:
  - [ ] task-21-embedding-doc.txt

  **Commit**: YES
  - Message: `docs: add EMBEDDING-READINESS.md for instance-level similarity`
  - Files: `EMBEDDING-READINESS.md`
  - Pre-commit: structure check passes

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read this plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Specifically: `grep -r "5000\|5173\|codex-frontend\|frontend_integration_fix" .` must return zero results in source files. Verify all evidence files exist in `.sisyphus/evidence/codex-cleanup/`.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `cd frontend && npm run build && npm test && npm run lint` + `cd backend && pytest`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports, AI slop (excessive comments, generic names like `data/result/item/temp`, over-abstraction).
  Output: `Build [PASS/FAIL] | Tests [N pass/N fail] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill)
  Start from clean state. Run `bash scripts/install.sh` in fresh checkout. Run `bash scripts/run-dev.sh`. Open `http://localhost:7118/` in Playwright. Execute EVERY annotation QA scenario from tasks T9-T12. Test annotation: create box, resize from each handle, drag with mouse leaving SVG, zoom-then-annotate. Save evidence to `.sisyphus/evidence/codex-cleanup/final-qa/`.
  Output: `Install [PASS/FAIL] | Run-dev [PASS/FAIL] | Annotation scenarios [N/N pass] | Edge cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (`git log/diff` on `cleanup/repo-restructure`). Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance especially: no FAISS, no instance-level embedding code, no i18n changes, no Docker artifacts, no touching `.sisyphus/state/`. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

One commit per logical task on branch `cleanup/repo-restructure`. Conventional commits. Each rename = single commit (preserves `git log --follow`).

- **T1**: `chore(wip): commit work-in-progress before restructure` — all current modified files
- **T2**: `chore(legacy): archive EXISTING_CODE as confirmed duplicate` — `_legacy/EXISTING_CODE/`
- **T3**: `chore(repo): rename codex-frontend to frontend` — `git mv`
- **T4**: `chore(repo): flatten and rename to backend` — `git mv`
- **T5**: `chore(repo): consolidate dev-scripts and archive sam_glyph_test` — `dev-scripts/`, `_legacy/sam_glyph_test/`
- **T6**: `ci(smoke): update workflow paths post-restructure` — `.github/workflows/smoke.yml`
- **T7**: `feat(backend): env-driven port/host/CORS config` — `backend/examples/flask_api.py`, `backend/.env.example`
- **T8**: `feat(frontend): bind to port 7118 + env example` — `frontend/vite.config.ts`, `frontend/.env.example`
- **T9**: `fix(annotation): use pointer events with setPointerCapture` — `frontend/src/pages/AnnotationPage.tsx`
- **T10**: `fix(annotation): scan all elements for handle hit-test` — `frontend/src/pages/AnnotationPage.tsx`
- **T11**: `fix(annotation): map client coords via getScreenCTM` — `frontend/src/pages/AnnotationPage.tsx`
- **T12**: `polish(annotation): harmonize handle hit area + RAF throttle` — `frontend/src/pages/AnnotationPage.tsx`
- **T13**: `fix(backend): unify image padding color across pipelines` — `backend/codex_*/`
- **T14**: `chore(git): ignore dist and evidence dirs` — `.gitignore`
- **T15**: `feat(install): bash + powershell installers` — `scripts/install.sh`, `scripts/install.ps1`
- **T16**: `feat(scripts): run-dev launcher and weights downloader` — `scripts/run-dev.sh`, `scripts/download-weights.sh`
- **T17**: `test(frontend): vitest scaffolding + smoke test` — `frontend/vitest.config.ts`, `frontend/src/**/*.test.tsx`
- **T18**: `test(backend): pytest scaffolding + smoke test` — `backend/tests/`
- **T19**: `docs(readme): dual-audience rewrite` — `README.md`
- **T20**: `docs(install): non-technical install guide` — `INSTALL.md`
- **T21**: `docs(embedding): readiness inventory and plug-in plan` — `EMBEDDING-READINESS.md`

Pre-commit per task: relevant test command from Definition of Done.

---

## Success Criteria

### Verification Commands
```bash
# Structure
test -d frontend && test -d backend && test -d _legacy/EXISTING_CODE && test -d dev-scripts
test ! -d codex-frontend && test ! -d frontend_integration_fix
git log --follow frontend/package.json | head -5  # history preserved

# Ports
PORT=7117 python backend/examples/flask_api.py &
sleep 3 && curl -sf http://localhost:7117/health | grep -q ok
pkill -f flask_api

# Build + tests
cd frontend && npm ci && npm run build && npm test -- --run
cd backend && pip install -r requirements.txt && pytest

# CI
grep -r "codex-frontend\|frontend_integration_fix" .github/ && exit 1 || true

# Install script
bash scripts/install.sh --dry-run

# No forbidden artifacts
! grep -rE "from frontend_integration_fix|import frontend_integration_fix" backend/
! grep -rE ":5000|:5173" frontend/src backend/
```

### Final Checklist
- [ ] All 21 tasks completed and committed
- [ ] All 4 final verification agents APPROVED
- [ ] User explicitly said "okay" after reviewing F1-F4 reports
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All evidence files captured in `.sisyphus/evidence/codex-cleanup/`
- [ ] Branch `cleanup/repo-restructure` ready to merge

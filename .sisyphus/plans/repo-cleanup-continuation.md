# Repository Cleanup Continuation Plan

## TL;DR

> **Quick Summary**: Clean and normalize the repository around the already-built Codex demo so the project has a clear canonical structure, trustworthy documentation, safer git hygiene, and repeatable smoke checks before any deeper product work continues.
>
> **Deliverables**:
> - Canonical repo inventory and keep/archive/remove policy
> - Root README plus corrected frontend README
> - Root `.gitignore` and cleanup of generated/OS noise
> - Env-driven frontend API configuration
> - Minimal smoke-test and CI baseline for repo integrity
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: T1 inventory → T2 smoke baseline → T8 cleanup execution → T10 CI verification

---

## Context

### Original Request
Get the current progress of the project, then turn that understanding into a continuation plan.

### Interview Summary
**Key Discussions**:
- The repository is a Codex glyph / Nahuatl element analyzer with a Python ML backend/integration package and a React/Vite frontend.
- Core prototype/demo behavior appears substantially implemented already.
- The user wants the continuation plan to prioritize **repo cleanup first**.

**Research Findings**:
- Frontend entrypoint: `codex-frontend/src/App.tsx`
- Backend demo API entrypoint: `frontend_integration_fix/frontend_integration/examples/flask_api.py`
- Frontend API base URL is hardcoded in `codex-frontend/src/services/api.ts`
- Root `.gitignore` is currently empty.
- `codex-frontend/README.md` is still the default Vite template.
- No `.github/workflows/*` files were found.
- Mac/archive noise and generated artifacts exist or are strongly implied (`__MACOSX`, `node_modules`, `__pycache__`, etc.).

**Resolved Decisions**:
- Keep current root-level project materials for now unless later proven redundant.
- Treat `frontend_integration_fix/` as a backend/AI developer base, not an immediate removal target.
- Treat root `package-lock.json` as likely stray from prior npm usage unless later proven canonical.
- `presentation_codex.pdf` may be removed if not needed.
- Treat model artifacts as external assets, while preserving future model-update workflows from user feedback.
- Use a **full local demo smoke** as the required verification target for this cleanup phase.

### Metis Review
**Identified Gaps** (addressed in this draft plan):
- Need explicit guardrails to keep cleanup from turning into feature work or deployment work.
- Need a keep/archive/remove policy before deleting or moving large assets.
- Need concrete smoke-check acceptance criteria before making structural changes.
- Need atomic commit slicing so cleanup does not become one giant unreviewable change.
- Initial placeholders for canonical directories, artifact retention, and smoke scope have now been resolved by user decisions.

---

## Work Objectives

### Core Objective
Normalize the repository so contributors can understand what is canonical, run the project with documented commands, and continue development from a clean and reviewable baseline.

### Concrete Deliverables
- A documented canonical top-level repository structure
- A real root README covering frontend + backend usage
- A project-specific `codex-frontend/README.md`
- Root `.gitignore` covering generated and OS-specific artifacts
- Full local demo smoke workflow for frontend + backend
- One CI workflow running those smoke checks
- Cleanup of disposable root-level and nested noise according to an approved inventory

### Definition of Done
- [ ] Canonical top-level repo tree is documented and matched by the filesystem state
- [ ] Root README includes exact run commands for frontend and backend
- [ ] Full local demo smoke passes using documented commands and sample glyph input
- [ ] Generated and OS-specific artifacts are ignored and removed from canonical source tree
- [ ] CI workflow exists and runs the agreed smoke checks successfully

### Must Have
- Preserve current demo/prototype functionality while cleaning the repo
- Use TDD-oriented sequencing: define smoke checks first, then cleanup, then prove nothing regressed
- Use atomic commits throughout
- Keep cleanup work separate from product feature work

### Must NOT Have (Guardrails)
- No feature additions, UI redesign, ML-model changes, or backend rewrites in this phase
- No broad packaging/deployment work (Docker, hosting, release engineering) in this phase
- No deletion or relocation of model/data artifacts until the external-asset policy is implemented safely
- No one-shot “big bang” cleanup commit
- No path renames that break scripts without paired smoke verification

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: NO formal test framework discovered
- **Automated tests**: YES (TDD-oriented full local demo smoke first)
- **Framework**: frontend commands + backend demo API run + sample glyph request/response verification + CI workflow
- **If TDD**: Each relevant cleanup task defines the full local demo smoke/assertion first, then applies the cleanup, then re-runs verification

### QA Policy
Every task includes agent-executed QA scenarios with evidence captured in `.sisyphus/evidence/`.

- **Frontend**: Bash for `npm`/build/lint checks; Playwright only if a UI task requires runtime verification
- **Backend/API**: Bash for Python import/smoke commands and `curl` if a lightweight API run is included
- **Repo hygiene**: Bash + Grep/Glob to verify ignored/noise files are absent or excluded
- **Docs**: Read + Bash command verification against documented commands

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately - inventory + verification foundations):
├── Task 1: Canonical inventory and retention policy map [deep]
├── Task 2: Define smoke-check baseline before cleanup [quick]
├── Task 3: Expand `.gitignore` for generated/noise artifacts [quick]
├── Task 4: Replace frontend template README [writing]
├── Task 5: Draft root README tying frontend + backend together [writing]
└── Task 6: Externalize frontend API base URL configuration [quick]

Wave 2 (After Wave 1 - cleanup execution):
├── Task 7: Document backend/demo artifact expectations [writing]
├── Task 8: Remove/archive agreed non-canonical root and nested noise [unspecified-high]
├── Task 9: Normalize local dependency/output policy (`node_modules`, caches, lockfiles) [unspecified-high]
└── Task 10: Add CI workflow for smoke checks [quick]

Wave 3 (After Wave 2 - prove and reconcile):
├── Task 11: Re-run smoke checks against cleaned structure [quick]
├── Task 12: Audit docs/paths/commands against actual cleaned repo [deep]
└── Task 13: Prepare atomic commit sequence and staging boundaries [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA / smoke execution (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: T1 → T2 → T8 → T10 → T11 → T12 → T13 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 6

### Dependency Matrix

- **1**: None → 8, 9, 12
- **2**: None → 8, 10, 11, 12, 13
- **3**: None → 8, 9, 11
- **4**: None → 12, 13
- **5**: None → 12, 13
- **6**: None → 11, 12
- **7**: 1 → 12
- **8**: 1, 2, 3 → 11, 12, 13
- **9**: 1, 3 → 11, 12, 13
- **10**: 2 → 11, 13
- **11**: 2, 3, 6, 8, 9, 10 → 12, 13
- **12**: 1, 2, 4, 5, 6, 7, 8, 11 → 13
- **13**: 2, 4, 5, 8, 9, 10, 11, 12 → FINAL

### Agent Dispatch Summary

- **Wave 1**: T1 → `deep`, T2 → `quick`, T3 → `quick`, T4 → `writing`, T5 → `writing`, T6 → `quick`
- **Wave 2**: T7 → `writing`, T8 → `unspecified-high`, T9 → `unspecified-high`, T10 → `quick`
- **Wave 3**: T11 → `quick`, T12 → `deep`, T13 → `quick`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Create a canonical inventory and retention policy map

  **What to do**:
  - Inspect the top-level repo contents and classify each root-level file/directory as canonical product code, generated artifact, imported/archive material, documentation asset, or undecided.
  - Produce a keep/archive/remove table that specifically addresses `codex-frontend/`, `frontend_integration_fix/`, `sam_glyph_test/`, root `package-lock.json`, `presentation_codex.pdf`, and any nested `__MACOSX` content.
  - Record whether path moves are allowed in this cleanup phase or whether only removals/ignores/docs are allowed.
  - Encode the user decisions: keep current project materials for now, treat `frontend_integration_fix/` as a backend/AI-dev base, treat root `package-lock.json` as likely stray, and allow removal of `presentation_codex.pdf` if nonessential.

  **Must NOT do**:
  - Do not delete or move anything yet.
  - Do not infer artifact retention policy without explicitly documenting the assumption or decision.

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires repo-wide judgment across structure, intent, and future maintainability.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `git-master`: Commit work is not started yet.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6)
  - **Blocks**: 8, 9, 12
  - **Blocked By**: None

  **References**:
  - `.sisyphus/drafts/project-status-review.md` - Existing repository-status synthesis and current assumptions.
  - `.gitignore` - Currently empty; shows repo hygiene has not been established yet.
  - `codex-frontend/package.json` - Confirms frontend is a real app, not just scaffolding.
  - `frontend_integration_fix/frontend_integration/README.md` - Confirms nested integration package is substantive and likely canonical.

  **Acceptance Criteria**:
  - [ ] A written inventory exists listing every root-level item and its status: keep/archive/remove/undecided.
  - [ ] The plan for `*.pt`, sample assets, nested archive artifacts, and stray root files is explicit.
  - [ ] `frontend_integration_fix/` is marked keep-for-now unless later proven redundant.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Root inventory completed
    Tool: Bash (python or shell listing) + Read
    Preconditions: Repository present at /home/sina/omxpro/clinic-codex
    Steps:
      1. Enumerate root-level files/directories.
      2. Compare each root item against the written inventory artifact.
      3. Assert every root item has exactly one disposition: keep/archive/remove/undecided.
    Expected Result: No unclassified root-level items remain.
    Failure Indicators: Missing root item, duplicate classification, or undocumented disposition.
    Evidence: .sisyphus/evidence/task-1-root-inventory.txt

  Scenario: Sensitive artifact policy explicitly documented
    Tool: Read
    Preconditions: Inventory artifact exists.
    Steps:
      1. Read the inventory/policy section covering model weights, sample data, and root PDF assets.
      2. Assert each asset class has a stated retention policy.
    Expected Result: Weight/sample/document artifacts are not left implicit.
    Evidence: .sisyphus/evidence/task-1-artifact-policy.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-1-root-inventory.txt`
  - [ ] `task-1-artifact-policy.txt`

  **Commit**: NO

- [x] 2. Define smoke-check baseline before cleanup

  **What to do**:
  - Define the exact full local demo workflow for this cleanup phase.
  - Specify exact commands for frontend install/run, backend install/run, and a sample glyph demo request through the documented local stack.
  - Use existing sample glyph images to make the smoke deterministic and reproducible.

  **Must NOT do**:
  - Do not expand into broad test infrastructure or model accuracy benchmarking.
  - Do not rely on manual testing instructions.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Narrow task centered on selecting concrete commands and pass/fail checks.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `test-engineer`: Overkill for smoke-baseline definition at this stage.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 8, 10, 11, 12, 13
  - **Blocked By**: None

  **References**:
  - `codex-frontend/package.json` - Existing frontend `dev`, `build`, and `lint` commands.
  - `frontend_integration_fix/frontend_integration/README.md` - Existing backend run commands and sample curl usage.
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py` - Confirms the demo API surface that smoke checks should preserve.
  - `frontend_integration_fix/frontend_integration/data/glyphs_sample/atl-glyph/026r_a_07-2.jpg` - Concrete sample glyph path available in repo for full demo smoke.

  **Acceptance Criteria**:
  - [ ] Frontend demo commands are specified exactly.
  - [ ] Backend API run command(s) are specified exactly.
  - [ ] A concrete sample image path and expected API/UI success conditions are specified.
  - [ ] Expected success condition is binary (exit code 0 / expected response shape / visible UI state).

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Full local frontend demo baseline documented
    Tool: Read
    Preconditions: Smoke-check section exists in repo docs or helper script.
    Steps:
      1. Read the smoke-check definition.
      2. Assert it includes exact frontend install/run commands with no placeholders.
    Expected Result: Frontend demo path is executable without interpretation.
    Failure Indicators: Placeholder text, ambiguous commands, or missing expected outcomes.
    Evidence: .sisyphus/evidence/task-2-frontend-smoke-definition.txt

  Scenario: Backend API and sample glyph smoke baseline documented
    Tool: Read
    Preconditions: Smoke-check section exists.
    Steps:
      1. Read the backend smoke definition.
      2. Assert it names the exact API entrypoint, sample glyph path, request path, and expected success condition.
    Expected Result: End-to-end backend demo smoke is concrete and automatable.
    Evidence: .sisyphus/evidence/task-2-backend-smoke-definition.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-2-frontend-smoke-definition.txt`
  - [ ] `task-2-backend-smoke-definition.txt`

  **Commit**: YES
  - Message: `test(repo): add cleanup smoke baseline`
  - Files: `README.md`, helper smoke docs/scripts, CI references if introduced here
  - Pre-commit: chosen smoke commands

- [x] 3. Expand root `.gitignore` for generated and OS-specific artifacts

  **What to do**:
  - Add ignore rules for `node_modules/`, Python caches/virtualenvs, macOS archive noise, editor/temp files, and build outputs.
  - Ensure ignore rules are scoped so canonical source/assets are not accidentally hidden.
  - Align ignore rules with the inventory from Task 1.

  **Must NOT do**:
  - Do not ignore real source directories or intentionally tracked assets without explicit decision.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file hygiene task with straightforward verification.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `git-master`: Actual commit slicing happens later.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 8, 9, 11
  - **Blocked By**: None

  **References**:
  - `.gitignore` - Empty starting point.
  - `codex-frontend/node_modules/` - Evidence that generated frontend dependencies currently exist in-tree.
  - `frontend_integration_fix/__MACOSX/` - Evidence of archive noise needing ignore/remove coverage.
  - `frontend_integration_fix/frontend_integration/codex_pipeline/scripts/__pycache__/` - Evidence of generated Python cache directories.

  **Acceptance Criteria**:
  - [ ] Root `.gitignore` contains rules for major generated/noise categories present in the repo.
  - [ ] No canonical source directory is accidentally ignored.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Ignore file covers known generated artifacts
    Tool: Read
    Preconditions: Root .gitignore updated.
    Steps:
      1. Read .gitignore.
      2. Assert it contains patterns for node_modules, __pycache__, .DS_Store, __MACOSX, and virtualenv directories.
    Expected Result: All known generated/noise families are covered.
    Failure Indicators: Missing pattern for observed artifact category.
    Evidence: .sisyphus/evidence/task-3-ignore-rules.txt

  Scenario: Ignore rules do not hide canonical product roots
    Tool: Read
    Preconditions: Root .gitignore updated.
    Steps:
      1. Read .gitignore.
      2. Assert `codex-frontend/` and `frontend_integration_fix/frontend_integration/` are not ignored wholesale.
    Expected Result: Ignore rules target generated content, not source roots.
    Evidence: .sisyphus/evidence/task-3-ignore-safety.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-3-ignore-rules.txt`
  - [ ] `task-3-ignore-safety.txt`

  **Commit**: YES
  - Message: `chore(repo): ignore generated and OS-specific artifacts`
  - Files: `.gitignore`
  - Pre-commit: grep/readback validation of ignore rules

- [x] 4. Replace the frontend template README with project-specific documentation

  **What to do**:
  - Replace the Vite template README in `codex-frontend/README.md` with documentation for the actual frontend app.
  - Cover purpose, development commands, environment variables, and relationship to the backend API.
  - Keep it aligned with the root README so they do not drift.

  **Must NOT do**:
  - Do not leave generic template guidance in place.
  - Do not document commands that are not verified by smoke checks.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation-focused deliverable.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `writer`: Native writing category is sufficient.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 12, 13
  - **Blocked By**: None

  **References**:
  - `codex-frontend/README.md` - Current template to replace.
  - `codex-frontend/package.json` - Actual available frontend commands.
  - `codex-frontend/src/App.tsx` - Routes and app purpose.
  - `codex-frontend/src/services/api.ts` - Backend dependency and env-config target.

  **Acceptance Criteria**:
  - [ ] The frontend README no longer contains generic Vite template content.
  - [ ] It documents exact frontend commands and backend dependency.
  - [ ] It mentions env-based API configuration once Task 6 is complete.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Template content removed
    Tool: Grep
    Preconditions: codex-frontend/README.md updated.
    Steps:
      1. Search the frontend README for template phrases like "React + TypeScript + Vite" and generic plugin guidance.
      2. Assert those template sections are absent.
    Expected Result: Template scaffold text is gone.
    Failure Indicators: Default Vite template content remains.
    Evidence: .sisyphus/evidence/task-4-template-removal.txt

  Scenario: Frontend README contains runnable project commands
    Tool: Read
    Preconditions: README updated.
    Steps:
      1. Read the frontend README.
      2. Assert it includes exact install/dev/build/lint commands and a note on the backend API dependency.
    Expected Result: README is project-specific and actionable.
    Evidence: .sisyphus/evidence/task-4-frontend-readme.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-4-template-removal.txt`
  - [ ] `task-4-frontend-readme.txt`

  **Commit**: YES
  - Message: `docs(frontend): replace template readme`
  - Files: `codex-frontend/README.md`
  - Pre-commit: README content verification

- [x] 5. Add a root README that defines the project and how to run it

  **What to do**:
  - Create a top-level README describing the project, canonical structure, frontend/backend relationship, prerequisites, smoke commands, and known artifact requirements.
  - Include a short repository map and explicit "current supported workflow" statement.
  - Link to deeper docs rather than duplicating too much detail.

  **Must NOT do**:
  - Do not write aspirational or unverified instructions.
  - Do not omit backend artifact/setup caveats.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Root documentation artifact.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `writer`: Not necessary beyond writing category.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 12, 13
  - **Blocked By**: None

  **References**:
  - `.sisyphus/drafts/project-status-review.md` - Current overall understanding of repo purpose and status.
  - `frontend_integration_fix/frontend_integration/README.md` - Backend/integration quick-start source.
  - `codex-frontend/package.json` - Frontend command source.

  **Acceptance Criteria**:
  - [ ] Root README exists.
  - [ ] It explains what the project is, what directories matter, and how to run frontend + backend.
  - [ ] It includes verified smoke commands and caveats about model artifacts/checkpoints.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Root README exists with canonical repo overview
    Tool: Read
    Preconditions: README.md created at repo root.
    Steps:
      1. Read the root README.
      2. Assert it contains project summary, canonical top-level structure, frontend/backend run sections, and links to key subdirectories.
    Expected Result: New contributors can understand the repo from the root README alone.
    Evidence: .sisyphus/evidence/task-5-root-readme.txt

  Scenario: Root README commands are concrete
    Tool: Grep
    Preconditions: Root README exists.
    Steps:
      1. Search the root README for placeholder markers like TODO/TBD/[fill in].
      2. Assert no unresolved placeholders remain in setup/run sections.
    Expected Result: Documentation is actionable and complete enough for smoke use.
    Evidence: .sisyphus/evidence/task-5-readme-completeness.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-5-root-readme.txt`
  - [ ] `task-5-readme-completeness.txt`

  **Commit**: YES
  - Message: `docs(repo): add root project readme`
  - Files: `README.md`
  - Pre-commit: README verification

- [x] 6. Externalize the frontend API base URL

  **What to do**:
  - Replace hardcoded `http://localhost:5000` usage with env-based configuration suitable for local default plus override.
  - Document the env variable in frontend/root docs.
  - Preserve current local-dev behavior by default.

  **Must NOT do**:
  - Do not change the API contract or route names.
  - Do not refactor unrelated frontend components.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small, focused source/config change.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `visual-engineering`: No UI work involved.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 11, 12
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/services/api.ts` - Current hardcoded base URL.
  - `codex-frontend/package.json` - Confirms Vite environment context.

  **Acceptance Criteria**:
  - [ ] Frontend API base URL is env-driven.
  - [ ] Default local value preserves current behavior without additional source edits.
  - [ ] Documentation includes the env variable name and usage.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Default local API URL preserved
    Tool: Read
    Preconditions: Frontend config updated.
    Steps:
      1. Read the API configuration file.
      2. Assert it resolves to localhost:5000 when no env override is set.
    Expected Result: Local development behavior remains unchanged by default.
    Evidence: .sisyphus/evidence/task-6-default-api-url.txt

  Scenario: Env override path exists
    Tool: Read
    Preconditions: Frontend config and docs updated.
    Steps:
      1. Read the API configuration and docs.
      2. Assert both define the same env variable and show it can override the default.
    Expected Result: API endpoint is configurable without source edits.
    Evidence: .sisyphus/evidence/task-6-env-override.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-6-default-api-url.txt`
  - [ ] `task-6-env-override.txt`

  **Commit**: YES
  - Message: `refactor(frontend): externalize api base url`
  - Files: `codex-frontend/src/services/api.ts`, env example/docs
  - Pre-commit: frontend smoke commands

- [x] 7. Document backend/demo artifact expectations and supported runtime path

  **What to do**:
  - Clarify whether `frontend_integration_fix/frontend_integration/examples/flask_api.py` is the supported demo backend entrypoint for now.
  - Document required artifacts/checkpoints and where they are expected.
  - Make explicit that the supported cleanup-phase workflow is a full local demo, not merely import smoke.
  - Note that model assets should move toward external-asset handling while preserving updateability from user feedback workflows.

  **Must NOT do**:
  - Do not redesign backend packaging or replace the Flask example with a new service.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Clarification/documentation task.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `document-specialist`: Internal repo docs are sufficient.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 10)
  - **Blocks**: 12
  - **Blocked By**: 1

  **References**:
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py` - Current backend demo/API entrypoint.
  - `frontend_integration_fix/frontend_integration/README.md` - Current quick-start and artifact notes.
  - `frontend_integration_fix/frontend_integration/codex_model/weights/prototypes.pt` - Existing model artifact.
  - `frontend_integration_fix/frontend_integration/codex_model/weights/projection.pt` - Existing projection artifact.
  - `frontend_integration_fix/frontend_integration/mobile_sam.pt` - Existing checkpoint in repo.

  **Acceptance Criteria**:
  - [ ] Supported backend runtime path is named explicitly in docs.
  - [ ] Artifact/checkpoint expectations are documented accurately.
  - [ ] External-asset policy direction is documented without breaking current local demo assumptions.
  - [ ] README instructions do not imply unsupported production guarantees.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Supported backend path clearly documented
    Tool: Read
    Preconditions: Docs updated.
    Steps:
      1. Read root/integration docs.
      2. Assert one current backend runtime path is named as the supported cleanup-phase workflow.
    Expected Result: Contributors do not have to guess which backend entrypoint to use.
    Evidence: .sisyphus/evidence/task-7-supported-backend.txt

  Scenario: Artifact expectations documented without ambiguity
    Tool: Read
    Preconditions: Docs updated.
    Steps:
      1. Read artifact/setup notes.
      2. Assert required model files/checkpoints and their expected locations are listed.
    Expected Result: Runtime prerequisites are explicit.
    Evidence: .sisyphus/evidence/task-7-artifact-docs.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-7-supported-backend.txt`
  - [ ] `task-7-artifact-docs.txt`

  **Commit**: YES
  - Message: `docs(backend): clarify supported demo runtime`
  - Files: root/integration docs
  - Pre-commit: docs verification

- [x] 8. Remove or archive agreed non-canonical root and nested noise

  **What to do**:
  - Execute the keep/archive/remove decisions from Task 1.
  - Remove recursively generated and archive noise such as `__MACOSX`, `.DS_Store`, `__pycache__`, and other explicitly non-canonical artifacts.
  - Clean root-level clutter only after verifying it is classified as remove/archive.
  - Keep `frontend_integration_fix/` for now unless its redundancy is demonstrated during inventory review.
  - Treat `presentation_codex.pdf` and root `package-lock.json` as removable if inventory confirms they are non-canonical.

  **Must NOT do**:
  - Do not delete undecided or policy-protected artifacts.
  - Do not move/rename canonical directories without explicit approval in the inventory.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-location cleanup with risk of deleting the wrong material.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `ai-slop-cleaner`: This is repository cleanup, not code simplification.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 11, 12, 13
  - **Blocked By**: 1, 2, 3

  **References**:
  - Task 1 inventory artifact - Source of truth for keep/archive/remove decisions.
  - `frontend_integration_fix/__MACOSX/` - Known nested archive-noise subtree.
  - `frontend_integration_fix/frontend_integration/codex_pipeline/scripts/__pycache__/` - Known generated cache subtree.
  - Root git status results - Evidence of multiple untracked root-level items needing disposition.

  **Acceptance Criteria**:
  - [ ] All explicitly removable noise classified in Task 1 is removed or archived.
  - [ ] No canonical source or policy-protected assets are deleted.
  - [ ] Post-cleanup filesystem matches documented canonical inventory.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Known noise families removed from canonical tree
    Tool: Glob
    Preconditions: Cleanup execution completed.
    Steps:
      1. Search for `**/__MACOSX/**`, `**/__pycache__/**`, and `**/.DS_Store` within the repo.
      2. Assert no matches remain in canonical source areas.
    Expected Result: Archive/cached noise is gone from the working tree or isolated per policy.
    Failure Indicators: Residual noise directories/files remain in source paths.
    Evidence: .sisyphus/evidence/task-8-noise-removal.txt

  Scenario: Canonical roots preserved
    Tool: Glob
    Preconditions: Cleanup completed.
    Steps:
      1. Assert `codex-frontend/` and `frontend_integration_fix/frontend_integration/` still exist if marked canonical.
      2. Assert removed root items correspond only to Task 1 dispositions.
    Expected Result: Cleanup is precise, not destructive.
    Evidence: .sisyphus/evidence/task-8-canonical-preservation.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-8-noise-removal.txt`
  - [ ] `task-8-canonical-preservation.txt`

  **Commit**: YES
  - Message: `chore(repo): remove non-canonical artifacts`
  - Files: removed/archive targets per inventory
  - Pre-commit: smoke baseline + filesystem validation

- [x] 9. Normalize local dependency and output policy

  **What to do**:
  - Confirm the user-guided assumption that root `package-lock.json` is stray from prior npm usage unless evidence shows otherwise.
  - Remove tracked local-install outputs from canonical source tree where appropriate and ensure they are ignored.
  - Clarify which lockfiles belong in the repo and at which level.

  **Must NOT do**:
  - Do not delete lockfiles that correspond to a canonical package without documenting why.
  - Do not re-run dependency installs as part of cleanup unless required by smoke verification.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-cutting repo hygiene touching package-manager outputs and conventions.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `git-master`: Commit slicing handled in Task 13.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 11, 12, 13
  - **Blocked By**: 1, 3

  **References**:
  - `codex-frontend/package.json` - Confirms canonical frontend package root.
  - Root `package-lock.json` - Needs classification.
  - `codex-frontend/node_modules/` - Known local install output currently present.

  **Acceptance Criteria**:
  - [ ] Canonical lockfile policy is documented.
  - [ ] Disposable install outputs are not left in canonical source tree.
  - [ ] Ignore rules support the chosen policy.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Lockfile policy documented and reflected in tree
    Tool: Read + Glob
    Preconditions: Dependency/output policy task complete.
    Steps:
      1. Read the repo docs or policy note.
      2. Verify actual lockfile locations match the stated policy.
    Expected Result: There is no ambiguity about which lockfiles belong.
    Evidence: .sisyphus/evidence/task-9-lockfile-policy.txt

  Scenario: Disposable dependency outputs are absent or ignored
    Tool: Glob + Read
    Preconditions: Cleanup complete.
    Steps:
      1. Search for in-repo `node_modules/` under canonical source roots.
      2. Verify `.gitignore` covers them.
    Expected Result: Local install outputs are treated as disposable.
    Evidence: .sisyphus/evidence/task-9-dependency-outputs.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-9-lockfile-policy.txt`
  - [ ] `task-9-dependency-outputs.txt`

  **Commit**: YES
  - Message: `chore(repo): normalize dependency outputs`
  - Files: lockfiles, `.gitignore`, removed disposable outputs
  - Pre-commit: smoke baseline + tree validation

- [x] 10. Add CI workflow for repository smoke checks

  **What to do**:
  - Add a minimal CI workflow under `.github/workflows/` that runs the smoke checks chosen in Task 2.
  - Keep scope minimal: prove repo integrity with the full local demo baseline as far as feasible, not full deployment readiness.
  - Ensure the workflow matches documented commands exactly.

  **Must NOT do**:
  - Do not add expansive matrix builds or heavyweight ML evaluation in this phase.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small config/task scoped to one workflow.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `document-specialist`: No external docs lookup required yet.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 11, 13
  - **Blocked By**: 2

  **References**:
  - `.github/workflows/` - Currently absent, confirming CI needs to be created.
  - Task 2 smoke-check definition - Source of truth for workflow commands.

  **Acceptance Criteria**:
  - [ ] At least one CI workflow file exists.
  - [ ] It runs the same smoke commands documented for the repo.
  - [ ] It avoids heavyweight model-evaluation steps beyond the agreed demo smoke.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: CI workflow file exists and references smoke checks
    Tool: Read
    Preconditions: Workflow created.
    Steps:
      1. Read the workflow YAML.
      2. Assert it contains the chosen smoke commands and triggers on push/PR (or documented equivalent).
    Expected Result: CI is real and aligned with docs.
    Evidence: .sisyphus/evidence/task-10-ci-workflow.txt

  Scenario: CI scope remains minimal
    Tool: Read
    Preconditions: Workflow created.
    Steps:
      1. Read workflow steps.
      2. Assert it does not run broad deployment or heavy ML evaluation steps outside the smoke baseline.
    Expected Result: CI matches cleanup-phase scope.
    Evidence: .sisyphus/evidence/task-10-ci-scope.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-10-ci-workflow.txt`
  - [ ] `task-10-ci-scope.txt`

  **Commit**: YES
  - Message: `ci(repo): run repository smoke checks`
  - Files: `.github/workflows/*`
  - Pre-commit: workflow lint/readback + smoke baseline

- [x] 11. Re-run smoke checks against the cleaned repository

  **What to do**:
  - Execute the full local demo smoke commands defined earlier after cleanup/config/doc changes land.
  - Capture outputs as evidence and compare against the baseline.
  - Resolve any cleanup-caused regressions before proceeding.

  **Must NOT do**:
  - Do not mark cleanup complete if smoke commands fail.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused verification run.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `verifier`: Final verification wave will cover broader approval.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 13 after dependencies satisfied)
  - **Blocks**: 12, 13
  - **Blocked By**: 2, 3, 6, 8, 9, 10

  **References**:
  - Task 2 smoke baseline - Source of truth for commands.
  - `codex-frontend/package.json` - Frontend smoke command source.
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py` - Backend smoke target.

  **Acceptance Criteria**:
  - [ ] Frontend local demo commands pass.
  - [ ] Backend API demo commands pass.
  - [ ] Sample glyph request succeeds against the local stack.
  - [ ] Evidence is saved for frontend, backend, and sample demo request.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Frontend local demo smoke passes after cleanup
    Tool: Bash
    Preconditions: Frontend dependencies installed per project policy.
    Steps:
      1. Run the documented frontend install/build/run smoke commands.
      2. Assert exit code 0 for each required command.
    Expected Result: Cleanup did not break the frontend baseline.
    Evidence: .sisyphus/evidence/task-11-frontend-smoke.txt

  Scenario: Backend API demo smoke passes after cleanup
    Tool: Bash
    Preconditions: Python environment available per project policy.
    Steps:
      1. Run the documented backend API command(s).
      2. Submit the chosen sample glyph image to the documented endpoint.
      3. Assert expected response structure or exit code 0.
    Expected Result: Cleanup did not break the supported backend path.
    Evidence: .sisyphus/evidence/task-11-backend-smoke.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-11-frontend-smoke.txt`
  - [ ] `task-11-backend-smoke.txt`

  **Commit**: NO

- [x] 12. Audit docs, paths, and commands against the actual cleaned repo

  **What to do**:
  - Compare all documentation, env guidance, and path references against the final cleaned structure.
  - Verify no README or config still references deleted noise, wrong root paths, or old hardcoded assumptions.
  - Reconcile any drift introduced by cleanup.

  **Must NOT do**:
  - Do not leave stale path references or placeholder docs.

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Cross-file consistency audit across docs/config/structure.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `writer`: This task is primarily audit, not authoring.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 13
  - **Blocked By**: 1, 2, 4, 5, 6, 7, 8, 11

  **References**:
  - `README.md` - Root project docs.
  - `codex-frontend/README.md` - Frontend docs.
  - `codex-frontend/src/services/api.ts` - Env-based config and path expectations.
  - Task 1 inventory - Canonical structure source of truth.

  **Acceptance Criteria**:
  - [ ] No stale path references remain in docs/config.
  - [ ] README commands match the actual cleaned repo structure.
  - [ ] Backend/frontend docs agree on supported workflow.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Docs reference only real current paths
    Tool: Grep + Glob
    Preconditions: Cleanup and docs updates complete.
    Steps:
      1. Search docs/config for root paths and filenames.
      2. Assert each referenced path exists in the cleaned repo.
    Expected Result: No broken doc references remain.
    Evidence: .sisyphus/evidence/task-12-path-audit.txt

  Scenario: Docs and config agree on frontend/backend workflow
    Tool: Read
    Preconditions: Audit underway.
    Steps:
      1. Read root README, frontend README, and API config docs.
      2. Assert they describe the same supported local workflow and env configuration.
    Expected Result: Contributor guidance is internally consistent.
    Evidence: .sisyphus/evidence/task-12-workflow-consistency.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-12-path-audit.txt`
  - [ ] `task-12-workflow-consistency.txt`

  **Commit**: NO

- [x] 13. Prepare the atomic commit sequence and staging boundaries

  **What to do**:
  - Group completed cleanup work into small, reviewable commits matching the commit strategy.
  - Ensure each commit has a coherent purpose and relevant smoke verification.
  - Avoid bundling documentation, ignore rules, config refactors, CI, and deletion-heavy cleanup into one commit unless tightly coupled.

  **Must NOT do**:
  - Do not create one giant cleanup commit.
  - Do not commit failing states between slices.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small planning/packaging task around staging boundaries.
  - **Skills**: [`git-master`]
    - `git-master`: Needed for disciplined atomic commit slicing.
  - **Skills Evaluated but Omitted**:
    - `omc-reference`: Already loaded at session level.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: FINAL
  - **Blocked By**: 2, 4, 5, 8, 9, 10, 11, 12

  **References**:
  - Commit Strategy section in this plan - Intended atomic slices.
  - Task outputs from 2, 3, 4, 5, 6, 8, 9, 10 - Actual material to partition.

  **Acceptance Criteria**:
  - [ ] Every planned commit has a coherent scope and message.
  - [ ] Each commit references the validation it must pass before creation.
  - [ ] No commit mixes unrelated cleanup domains unnecessarily.

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Commit slices are atomic
    Tool: Read
    Preconditions: Commit staging plan written.
    Steps:
      1. Read the commit slice list.
      2. Assert each slice has a message, included file groups, and associated verification command(s).
    Expected Result: Commit plan is executable and reviewable.
    Evidence: .sisyphus/evidence/task-13-commit-slices.txt

  Scenario: No unrelated work is bundled
    Tool: Read
    Preconditions: Commit staging plan written.
    Steps:
      1. Review each planned commit scope.
      2. Assert docs, CI, ignore rules, config refactor, and deletions are separated unless explicitly justified.
    Expected Result: Cleanup history will be understandable.
    Evidence: .sisyphus/evidence/task-13-commit-boundaries.txt
  ```

  **Evidence to Capture:**
  - [ ] `task-13-commit-slices.txt`
  - [ ] `task-13-commit-boundaries.txt`

  **Commit**: NO

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify the canonical tree, docs, ignore rules, env config, smoke checks, CI workflow, and cleanup outputs all exist and match the plan. Confirm evidence files exist in `.sisyphus/evidence/`.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run relevant lint/build/smoke commands and review changed files for accidental feature creep, bad ignore rules, placeholder docs, or unsafe cleanup changes.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Smoke [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Execute every smoke scenario from the tasks, including README commands and backend/frontend baseline checks. Save evidence under `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  Compare each task’s scope against actual diff/content. Reject if cleanup drifted into feature work, deployment work, or undocumented structure changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- `test(repo): add cleanup smoke baseline`
- `docs(repo): add canonical project documentation`
- `chore(repo): ignore generated and OS-specific artifacts`
- `refactor(frontend): externalize API base url`
- `chore(repo): remove non-canonical artifacts`
- `ci(repo): run repository smoke checks`

---

## Success Criteria

### Verification Commands
```bash
cd codex-frontend && npm install && npm run build && npm run lint
python frontend_integration_fix/frontend_integration/examples/flask_api.py
curl -X POST -F "image=@frontend_integration_fix/frontend_integration/data/glyphs_sample/atl-glyph/026r_a_07-2.jpg" http://localhost:5000/segment
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Smoke checks pass
- [ ] CI workflow added and valid

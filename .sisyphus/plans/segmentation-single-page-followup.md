# Segmentation Single-Page Workflow Follow-up

## TL;DR

> **Quick Summary**: Replace the current multi-page upload/history/detail flow with one unified frontend screen that supports upload, current analysis inspection, and saved-history browsing in a single place, while fixing the detected-element click-selection bug.
>
> **Deliverables**:
> - Single-page upload + current analysis + history workflow
> - Stable detected-element selection behavior when switching between items
> - Preserved separate annotation route and storage/API contracts
> - Manual QA validation of the merged workflow
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 4 → Task 8 → Task 10 → Final Verification

---

## Context

### Original Request
- “Instead of having Upload and summary, everything should be on a single page.”
- “Clicked elements is not correctly selected but previously selected element is correctly disabled.”

### Interview Summary
**Key Discussions**:
- Merge target is specifically **Upload + analysis result/detail**.
- User also wants **history/dashboard merged** into that same single-page workflow.
- **Annotation remains separate**.
- **Manual QA only** for this follow-up.

**Research Findings**:
- Current routes:
  - `/` → `UploadPage`
  - `/dashboard` → `DashboardPage`
  - `/analysis/:id` → `DetailPage`
  - `/annotate/:id` → `AnnotationPage`
- `UploadPage` owns file selection, preview, `segmentGlyph`, `saveAnalysis`, then navigation.
- `DashboardPage` owns history list/filter/delete.
- `DetailPage` owns inspection UI state (`hoveredIdx`, `focusedIdx`, `overlayMode`, expanded items).
- `AnnotationPage` owns correction/save flow via `getClasses()` and `updateAnnotations()`.
- Frontend has build/lint/CI but **no test runner**.

### Gap Review
**Addressed Planning Gaps**:
- Locked scope to a unified page for upload + current analysis + history only.
- Kept annotation isolated as a separate route to avoid unnecessary coupling.
- Kept existing storage/API contracts as a guardrail to minimize regression risk.
- Chose manual QA only because no frontend test infrastructure exists today.

---

## Work Objectives

### Core Objective
Refactor the frontend into a single-page analyst workspace that combines upload, current analysis inspection, and history browsing, while making detected-element switching reliable and predictable.

### Concrete Deliverables
- A single-page shell that replaces separate upload/history/detail navigation for the main workflow
- Reused or extracted upload controls integrated into the merged screen
- Embedded history list/selection within the merged screen
- Embedded current-analysis inspection workspace within the merged screen
- Fixed clicked-element selection behavior when switching between proposals

### Definition of Done
- [ ] A user can upload an image and inspect results without navigating to a separate detail page
- [ ] A user can access saved analysis history from the same page
- [ ] Clicking detected element A and then B leaves B correctly selected/focused
- [ ] Annotation remains accessible via its separate route
- [ ] Manual QA confirms the merged flow works end-to-end with real saved records

### Must Have
- Upload, history, and current analysis all visible/reachable within one page
- Current storage behavior (`saveAnalysis`, `getHistory`, `getAnalysisById`, `updateAnnotations`) preserved unless a small compatibility wrapper is needed
- Existing inspection features from `DetailPage` retained in the merged experience
- Separate annotation page preserved

### Must NOT Have (Guardrails)
- No model retraining, threshold tuning, or segmentation parameter changes
- No backend API redesign
- No annotation embedding into the merged page in this follow-up
- No frontend test-runner setup in this follow-up
- No accidental removal of saved-history management features

---

## Verification Strategy

> **Manual QA only** for this follow-up. Existing automated safety net remains build + lint.

### Test Decision
- **Infrastructure exists**: Partial (build/lint/CI only)
- **Automated tests**: None for this work
- **Framework**: Manual browser QA + `npm run build` + `npm run lint`

### QA Policy
Every task must include explicit agent-executed QA scenarios. Final verification must include manual browser validation of upload, history switching, current-analysis inspection, and annotation handoff.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Flow architecture + shared state foundation):
├── Task 1: Audit and reshape main route/shell for single-page workflow [deep]
├── Task 2: Extract/upload integration strategy from UploadPage into reusable unit [quick]
├── Task 3: Map history data + current-record selection model for unified page [quick]

Wave 2 (Single-page assembly):
├── Task 4: Embed upload area and analysis creation flow into the merged page [deep]
├── Task 5: Embed history list/selection into the merged page [visual-engineering]
├── Task 6: Rehost detail inspection workspace inside the merged page [deep]
├── Task 7: Preserve separate annotation handoff and route compatibility [quick]

Wave 3 (Bug fix + workflow polish):
├── Task 8: Fix detected-element switching/focus-selection bug [deep]
├── Task 9: Improve merged-page information hierarchy and empty/loading states [visual-engineering]
├── Task 10: Manual browser QA pass and issue cleanup [unspecified-high]

Wave FINAL:
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: Real manual QA [unspecified-high]
└── Task F4: Scope fidelity check [deep]
```

### Dependency Matrix
- **1**: - - 4, 5, 6
- **2**: - - 4, 6
- **3**: - - 5, 6, 7
- **4**: 1, 2 - 8, 9, 10
- **5**: 1, 3 - 9, 10
- **6**: 1, 2, 3 - 8, 9, 10
- **7**: 3 - 10
- **8**: 4, 6 - 9, 10
- **9**: 4, 5, 6, 8 - 10
- **10**: 4, 5, 6, 7, 8, 9 - F1-F4

### Agent Dispatch Summary
- **Wave 1**: T1 → `deep`, T2/T3 → `quick`
- **Wave 2**: T4 → `deep`, T5 → `visual-engineering`, T6 → `deep`, T7 → `quick`
- **Wave 3**: T8 → `deep`, T9 → `visual-engineering`, T10 → `unspecified-high`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Audit and reshape the main route/shell for a single-page workflow

  **What to do**:
  - Decide which route becomes the new primary merged screen.
  - Refactor top-level page composition so upload, history, and current analysis can coexist on one route.
  - Preserve direct navigation support for annotation and any necessary deep links.

  **Must NOT do**:
  - Do not break `/annotate/:id`.
  - Do not redesign backend contracts.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/App.tsx` - current route ownership and nav links
  - `codex-frontend/src/pages/UploadPage.tsx` - existing upload-only route responsibilities
  - `codex-frontend/src/pages/DashboardPage.tsx` - existing history-only route responsibilities
  - `codex-frontend/src/pages/DetailPage.tsx` - current inspection-only route responsibilities

  **Acceptance Criteria**:
  - [ ] A clear single primary route/page architecture is implemented for upload + history + current analysis
  - [ ] Annotation route remains intact

  **QA Scenarios**:
  ```
  Scenario: Primary route exposes merged workflow shell
    Tool: Playwright
    Preconditions: Frontend running
    Steps:
      1. Open the new primary page URL
      2. Verify upload area, history area, and current-analysis area are all reachable on the same page
      3. Verify no forced navigation is required to inspect the current analysis area
    Expected Result: Main workflow exists within one page shell
    Evidence: .sisyphus/evidence/task-1-single-shell.png

  Scenario: Annotation route still exists
    Tool: Playwright
    Preconditions: Frontend running with known record id
    Steps:
      1. Open `/annotate/<known-id>`
      2. Verify annotation page loads
    Expected Result: Separate annotation workflow remains functional
    Evidence: .sisyphus/evidence/task-1-annotation-route.png
  ```

- [x] 2. Extract or rehome upload behavior into a reusable merged-page unit

  **What to do**:
  - Move file selection, preview, analyze action, and save-first flow out of a route-only structure.
  - Preserve `segmentGlyph` + `saveAnalysis` behavior exactly.
  - Ensure current-analysis state is updated immediately after successful analyze.

  **Must NOT do**:
  - Do not change saved `AnalysisRecord` shape.
  - Do not duplicate upload logic in two places.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: 4, 6
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/UploadPage.tsx` - source of file-picker/analyze/save behavior
  - `codex-frontend/src/services/api.ts` - `segmentGlyph`
  - `codex-frontend/src/services/storage.ts` - `saveAnalysis`
  - `codex-frontend/src/types/index.ts` - `AnalysisRecord`

  **Acceptance Criteria**:
  - [ ] Upload/analyze behavior is reusable inside the merged page
  - [ ] Successful analyze still persists a valid record

  **QA Scenarios**:
  ```
  Scenario: Upload and analyze from merged page
    Tool: Playwright
    Preconditions: Frontend + backend running; sample glyph image available
    Steps:
      1. Open merged page
      2. Upload a sample glyph image
      3. Trigger analysis
      4. Verify a result appears in the current-analysis area without redirecting to a separate detail page
    Expected Result: Analyze flow works inside merged page
    Evidence: .sisyphus/evidence/task-2-upload-analyze.png

  Scenario: Failed analyze shows readable error state
    Tool: Playwright
    Preconditions: Frontend running, backend unavailable
    Steps:
      1. Upload a sample image
      2. Trigger analysis
      3. Verify visible error feedback appears in-page
    Expected Result: Upload flow fails gracefully within merged page
    Evidence: .sisyphus/evidence/task-2-upload-error.png
  ```

- [x] 3. Define unified history/current-record selection model

  **What to do**:
  - Establish how saved history and the currently inspected record are selected within one page.
  - Preserve delete/history refresh behavior.
  - Ensure newly uploaded analysis becomes the active record predictably.

  **Must NOT do**:
  - Do not lose persisted history.
  - Do not require page navigation to switch active records.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: 5, 6, 7
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/DashboardPage.tsx` - history display/filter/delete behavior
  - `codex-frontend/src/services/storage.ts` - `getHistory`, `getAnalysisById`, `deleteAnalysis`
  - `codex-frontend/src/pages/DetailPage.tsx` - current-record consumption pattern

  **Acceptance Criteria**:
  - [ ] A single current-record selection model exists for the merged page
  - [ ] History operations still work

  **QA Scenarios**:
  ```
  Scenario: Select saved record from in-page history
    Tool: Playwright
    Preconditions: At least two saved records exist
    Steps:
      1. Open merged page
      2. Click saved record A in history
      3. Verify current-analysis area shows A
      4. Click saved record B
      5. Verify current-analysis area switches to B without route navigation
    Expected Result: In-page history switching works reliably
    Evidence: .sisyphus/evidence/task-3-history-switch.png

  Scenario: Delete history item updates page correctly
    Tool: Playwright
    Preconditions: At least one saved record exists
    Steps:
      1. Delete a history item
      2. Verify it disappears from history
      3. Verify current-analysis state remains valid or falls back cleanly
    Expected Result: Delete flow remains functional in merged page
    Evidence: .sisyphus/evidence/task-3-history-delete.png
  ```

- [x] 4. Embed upload area and analysis creation flow into the merged page

  **What to do**:
  - Render upload controls directly in the merged page shell.
  - Keep preview/loading/error behavior coherent with current upload experience.
  - Ensure creating a new analysis updates merged-page state immediately.

  **Must NOT do**:
  - Do not regress drag-and-drop/file-pick behavior.
  - Do not force users through the old upload-only route.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: 8, 9, 10
  - **Blocked By**: 1, 2

  **References**:
  - `codex-frontend/src/pages/UploadPage.tsx`
  - `codex-frontend/src/services/api.ts`
  - `codex-frontend/src/services/storage.ts`

  **Acceptance Criteria**:
  - [ ] Upload controls live in the merged page
  - [ ] Analyze result opens into the current-analysis panel on the same page

  **QA Scenarios**:
  ```
  Scenario: New analysis becomes current record immediately
    Tool: Playwright
    Preconditions: Frontend + backend running
    Steps:
      1. Open merged page
      2. Upload and analyze a sample image
      3. Verify the current-analysis panel updates to the new record
    Expected Result: Same-page analysis creation works end-to-end
    Evidence: .sisyphus/evidence/task-4-current-record.png

  Scenario: Loading state is visible during analyze
    Tool: Playwright
    Preconditions: Frontend + backend running
    Steps:
      1. Upload an image
      2. Trigger analysis
      3. Verify loading state appears until response completes
    Expected Result: User gets visible in-page feedback during analyze
    Evidence: .sisyphus/evidence/task-4-loading.png
  ```

- [x] 5. Embed history list and browsing controls into the merged page

  **What to do**:
  - Bring saved-history browsing into the single-page layout.
  - Preserve filtering/empty-state/delete affordances where appropriate.
  - Make history visually secondary to current analysis, but always accessible.

  **Must NOT do**:
  - Do not hide history behind a separate route-only requirement.
  - Do not remove empty-state guidance.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocks**: 9, 10
  - **Blocked By**: 1, 3

  **References**:
  - `codex-frontend/src/pages/DashboardPage.tsx`
  - `codex-frontend/src/services/storage.ts`

  **Acceptance Criteria**:
  - [ ] History is accessible on the merged page
  - [ ] Empty and populated states remain understandable

  **QA Scenarios**:
  ```
  Scenario: History list is visible on same page as current analysis
    Tool: Playwright
    Preconditions: Saved records exist
    Steps:
      1. Open merged page
      2. Verify history section and current-analysis section are both present
    Expected Result: History no longer requires a separate dashboard route for primary use
    Evidence: .sisyphus/evidence/task-5-history-present.png

  Scenario: Empty history state remains helpful
    Tool: Playwright
    Preconditions: No saved records in localStorage
    Steps:
      1. Open merged page
      2. Verify helpful empty-history messaging appears
    Expected Result: Single-page flow handles empty history gracefully
    Evidence: .sisyphus/evidence/task-5-history-empty.png
  ```

- [x] 6. Rehost the current inspection workspace inside the merged page

  **What to do**:
  - Move or compose the existing analyst inspection UI into the merged page without losing current capabilities.
  - Preserve overlay, proposal cards, crop previews, stats, and view controls.
  - Ensure current record changes correctly rerender inspection content.

  **Must NOT do**:
  - Do not remove inspection features already built.
  - Do not break record loading for saved records.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocks**: 8, 9, 10
  - **Blocked By**: 1, 2, 3

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx`
  - `codex-frontend/src/types/index.ts`
  - `codex-frontend/src/services/storage.ts`

  **Acceptance Criteria**:
  - [ ] Current inspection workspace is available inside merged page
  - [ ] Switching current records updates overlay/cards correctly

  **QA Scenarios**:
  ```
  Scenario: Inspection features survive the merge
    Tool: Playwright
    Preconditions: Saved record with detected elements exists
    Steps:
      1. Open merged page
      2. Select a saved record
      3. Verify overlay, proposal cards, crop previews, and controls all render
    Expected Result: DetailPage capabilities remain intact inside merged page
    Evidence: .sisyphus/evidence/task-6-inspection-merged.png

  Scenario: Record switch rerenders correct analysis
    Tool: Playwright
    Preconditions: Two different saved records exist
    Steps:
      1. Select record A
      2. Note visible image/proposal count
      3. Select record B
      4. Verify image/proposal data updates to B
    Expected Result: Merged page tracks current record reliably
    Evidence: .sisyphus/evidence/task-6-record-rerender.png
  ```

- [x] 7. Preserve separate annotation handoff and route compatibility

  **What to do**:
  - Keep annotation as a separate route and preserve all handoff links from the merged page.
  - Ensure merged-page current record feeds correct annotation ids.

  **Must NOT do**:
  - Do not embed annotation UI into this follow-up.
  - Do not break existing save/return flow.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: 10
  - **Blocked By**: 3

  **References**:
  - `codex-frontend/src/pages/AnnotationPage.tsx`
  - `codex-frontend/src/pages/DetailPage.tsx`
  - `codex-frontend/src/App.tsx`

  **Acceptance Criteria**:
  - [ ] Annotation links still resolve correctly from merged page context
  - [ ] Save-and-return flow remains valid

  **QA Scenarios**:
  ```
  Scenario: Open annotation from merged page
    Tool: Playwright
    Preconditions: Current record with rejected element exists
    Steps:
      1. Open merged page
      2. Open annotation from a rejected/current proposal action
      3. Verify annotation page loads for the same record id
    Expected Result: Separate annotation workflow remains reachable
    Evidence: .sisyphus/evidence/task-7-annotate-open.png

  Scenario: Save annotation returns to valid analysis context
    Tool: Playwright
    Preconditions: Annotation page open for known record
    Steps:
      1. Save an annotation change
      2. Verify return flow lands on a valid analysis context
    Expected Result: Existing annotation save flow is preserved
    Evidence: .sisyphus/evidence/task-7-annotate-return.png
  ```

- [x] 8. Fix detected-element switching and focus-selection behavior

  **What to do**:
  - Trace the current selection/focus logic for clicking proposal cards or overlay-related items.
  - Fix the state transition so clicking B after A deselects A and correctly selects/focuses B.
  - Ensure expanded state, focused state, and hover state do not conflict.

  **Must NOT do**:
  - Do not introduce a regression where no element can remain selected.
  - Do not break keyboard-focus or blur behavior while fixing click behavior.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10 after dependencies)
  - **Blocks**: 9, 10
  - **Blocked By**: 4, 6

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current `focusedIdx`, `hoveredIdx`, `expandedItems`, click/blur handlers
  - Any merged-page successor file/component containing rehosted inspection UI

  **Acceptance Criteria**:
  - [ ] Clicking a new detected element always leaves the newly clicked element selected
  - [ ] Previously selected element is deselected cleanly
  - [ ] No mismatch between selected card and highlighted/focused overlay state

  **QA Scenarios**:
  ```
  Scenario: Switching selected proposals works correctly
    Tool: Playwright
    Preconditions: Current record with at least 3 detected elements exists
    Steps:
      1. Open merged page and current analysis
      2. Click proposal 0 and verify it is visibly selected/focused
      3. Click proposal 1
      4. Verify proposal 0 is deselected and proposal 1 is now visibly selected/focused
      5. Click proposal 2 and verify selection transfers again
    Expected Result: Selection transfers to the newly clicked element every time
    Evidence: .sisyphus/evidence/task-8-selection-switch.png

  Scenario: Hover does not override clicked selection incorrectly
    Tool: Playwright
    Preconditions: Current record with multiple elements exists
    Steps:
      1. Click proposal 1 to select it
      2. Hover proposal 2 without clicking
      3. Verify proposal 1 remains the selected/focused element while proposal 2 shows hover-only feedback
    Expected Result: Hover and click states remain distinct and stable
    Evidence: .sisyphus/evidence/task-8-hover-vs-click.png
  ```

- [x] 9. Improve merged-page hierarchy, loading states, and empty states

  **What to do**:
  - Refine layout so upload, history, and current analysis feel coherent on one page.
  - Add or preserve clear empty/loading states for no history, no current record, in-progress analyze, and no detected elements.
  - Keep the analyst workflow visually readable despite more content on one page.

  **Must NOT do**:
  - Do not overload the page into an unreadable three-column cluttered layout.
  - Do not remove useful context labels/help text.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 10
  - **Blocked By**: 4, 5, 6, 8

  **References**:
  - `codex-frontend/src/pages/DashboardPage.tsx`
  - `codex-frontend/src/pages/UploadPage.tsx`
  - `codex-frontend/src/pages/DetailPage.tsx`

  **Acceptance Criteria**:
  - [ ] Single-page layout remains readable and understandable
  - [ ] Empty/loading states exist for all primary merged-page modes

  **QA Scenarios**:
  ```
  Scenario: Empty page states are clear
    Tool: Playwright
    Preconditions: No saved history and no current record
    Steps:
      1. Open merged page
      2. Verify clear guidance for upload and empty history/current analysis states
    Expected Result: User understands what to do first
    Evidence: .sisyphus/evidence/task-9-empty-states.png

  Scenario: Loaded page remains visually readable
    Tool: Playwright
    Preconditions: At least one saved record and one current analysis exist
    Steps:
      1. Open merged page
      2. Verify upload, history, and current analysis can all be understood without confusion
    Expected Result: Information hierarchy supports analyst workflow on one screen
    Evidence: .sisyphus/evidence/task-9-readability.png
  ```

- [x] 10. Run a manual browser QA pass and resolve merged-workflow issues

  **What to do**:
  - Launch frontend and backend and manually test the merged workflow end-to-end.
  - Resolve any integration issues found in upload/history/current-analysis switching and annotation handoff.

  **Must NOT do**:
  - Do not mark complete without real browser validation.
  - Do not ignore selection-state edge cases.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`playwright`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1-F4
  - **Blocked By**: 4, 5, 6, 7, 8, 9

  **References**:
  - `codex-frontend/src/pages/UploadPage.tsx`
  - `codex-frontend/src/pages/DashboardPage.tsx`
  - `codex-frontend/src/pages/DetailPage.tsx`
  - `codex-frontend/src/pages/AnnotationPage.tsx`
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py`

  **Acceptance Criteria**:
  - [ ] Real browser QA executed against merged page
  - [ ] Upload, history selection, current-analysis inspection, and annotation handoff all validated

  **QA Scenarios**:
  ```
  Scenario: End-to-end merged workflow
    Tool: Playwright
    Preconditions: Frontend + backend running; sample image available
    Steps:
      1. Open merged page
      2. Upload a sample image and wait for analysis
      3. Verify new result becomes current analysis
      4. Select a different saved record from history
      5. Verify current analysis switches correctly
      6. Click between multiple detected elements and verify correct selection transfer
      7. Open annotation route from current analysis
    Expected Result: Entire merged workflow works without route-hopping for primary use
    Evidence: .sisyphus/evidence/task-10-end-to-end.png

  Scenario: Refresh persistence remains valid
    Tool: Playwright
    Preconditions: At least one saved record exists
    Steps:
      1. Open merged page and select a saved record
      2. Refresh the browser
      3. Verify history still loads and current-analysis state falls back predictably
    Expected Result: Single-page flow remains stable across reloads
    Evidence: .sisyphus/evidence/task-10-refresh-persistence.png
  ```

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify merged page contains upload + history + current analysis, selection bug is fixed, annotation remains separate, and no prohibited scope expansion occurred.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run build/lint, inspect merged-page state ownership, event handling, selection logic, and any refactoring quality issues.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute all major merged-workflow scenarios in the browser and capture evidence.

- [x] F4. **Scope Fidelity Check** — `deep`
  Confirm only the requested merge + selection-fix scope was delivered, with annotation still separate and no backend/model drift.

---

## Commit Strategy

- One atomic commit for merged workflow refactor
- One follow-up commit only if manual QA reveals integration fixes that are cleaner isolated

---

## Success Criteria

### Verification Commands
```bash
cd codex-frontend && npm run build
cd codex-frontend && npm run lint
```

### Final Checklist
- [ ] Upload, history, and current analysis live in one primary page
- [ ] Switching clicked detected elements works correctly
- [ ] Annotation remains a separate route
- [ ] Manual QA completed
- [ ] No model/backend contract changes introduced

# Segmentation UI Analyst Workflow Upgrade

## TL;DR

> **Quick Summary**: Upgrade the segmentation review experience from a static, messy result page into a focused analyst workflow centered on a single glyph. Keep model behavior unchanged, but make proposals easier to inspect through synchronized overlay interactions, proposal cards, and clearer focus states.
>
> **Deliverables**:
> - Interactive single-glyph inspection workspace
> - Synchronized overlay + proposal panel interactions
> - Clear proposal focus states, crop previews, and overlay view modes
> - Manual QA validation of the full inspection flow
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 5 → Task 8 → Final Verification

---

## Context

### Original Request
Improve the user interface for segmentation/proposal display because the current experience is too messy. Keep model thresholds/parameters unchanged. Make the experience more interactive and easier to inspect.

### Interview Summary
**Key Discussions**:
- The current issue is UX clarity, not model tuning.
- The classifier appears to work on known element samples; segmentation review is the bottleneck.
- The user wants to optimize for **single glyph clarity first**.
- The user chose **manual QA only** for this phase.
- The user chose an **analyst workflow upgrade** rather than a lightweight polish pass.

**Research Findings**:
- `DetailPage.tsx` already renders an image and a canvas overlay for bounding boxes.
- The overlay canvas is currently `pointer-events-none`, so boxes are not directly interactive.
- A proposal/details list already exists, but it is not synchronized with overlay focus.
- `AnnotationPage.tsx` already has per-proposal suggestion/correction patterns.
- `DashboardPage.tsx` already has a reusable card/gallery browsing pattern.
- No frontend automated test infrastructure currently exists.

### Gap Review
**Resolved Planning Gaps**:
- Locked scope to a single-glyph inspection workflow, not multi-image comparison.
- Locked verification strategy to manual QA only.
- Locked model behavior as out of scope.

---

## Work Objectives

### Core Objective
Transform the segmentation detail experience into a cleaner analyst-oriented inspection workflow that helps users understand, inspect, and navigate detected proposals for a single glyph image.

### Concrete Deliverables
- A redesigned detail/inspection experience for a single analysis record
- Interactive linking between proposal list items and overlay boxes
- Proposal crop previews and clearer per-proposal metadata presentation
- Overlay visibility/focus controls that reduce visual clutter

### Definition of Done
- [ ] A user can inspect one glyph and clearly understand which proposal card corresponds to which region on the image
- [ ] Hover/focus interactions visibly connect overlay boxes and proposal detail cards
- [ ] The UI supports a focused inspection mode without changing segmentation thresholds or model outputs
- [ ] The full flow is manually QA’d in the browser against real segmentation results

### Must Have
- Single glyph clarity takes priority over batch/multi-image workflows
- Overlay and proposal panel are synchronized
- Proposal metadata is easier to inspect than in the current UI
- Existing annotation workflow remains reachable

### Must NOT Have (Guardrails)
- No model retraining, threshold tuning, or segmentation parameter changes
- No backend API redesign
- No test infrastructure setup in this phase
- No scope expansion into multi-image comparison unless explicitly deferred/follow-up

---

## Verification Strategy

> **Manual QA only** for this phase. No frontend test infrastructure will be added.

### Test Decision
- **Infrastructure exists**: Limited (lint/build only, no frontend test runner)
- **Automated tests**: None for this work
- **Framework**: Manual browser QA

### QA Policy
Every implementation task still needs explicit manual verification scenarios. The final workflow must be checked end-to-end in the running frontend against the local backend.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Interaction architecture + display foundation):
├── Task 1: Audit and refactor DetailPage layout structure [deep]
├── Task 2: Define inspection state model for focus/hover/view modes [quick]
├── Task 3: Extract overlay rendering responsibilities into reusable units [quick]

Wave 2 (Analyst workflow UI features):
├── Task 4: Implement synchronized overlay/list hover + selection behavior [deep]
├── Task 5: Build richer proposal panel cards with crop previews and metadata [visual-engineering]
├── Task 6: Add overlay visibility and focus view controls [visual-engineering]
├── Task 7: Preserve and improve annotation handoff from inspection workflow [quick]

Wave 3 (Workflow polish + validation prep):
├── Task 8: Improve detail-page information hierarchy and empty/error states [visual-engineering]
├── Task 9: Validate storage/types integration and route continuity [quick]
├── Task 10: Manual browser QA pass and issue cleanup [unspecified-high]

Wave FINAL:
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: Real manual QA [unspecified-high]
└── Task F4: Scope fidelity check [deep]
```

### Dependency Matrix
- **1**: - - 4, 5, 8
- **2**: - - 4, 6, 9
- **3**: - - 4, 6
- **4**: 1, 2, 3 - 8, 10
- **5**: 1 - 8, 10
- **6**: 2, 3 - 8, 10
- **7**: 1 - 9, 10
- **8**: 4, 5, 6 - 10
- **9**: 2, 7 - 10
- **10**: 4, 5, 6, 8, 9 - F1-F4

### Agent Dispatch Summary
- **Wave 1**: T1 → `deep`, T2/T3 → `quick`
- **Wave 2**: T4 → `deep`, T5/T6 → `visual-engineering`, T7 → `quick`
- **Wave 3**: T8 → `visual-engineering`, T9 → `quick`, T10 → `unspecified-high`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Audit and refactor the detail-page layout structure for analyst inspection

  **What to do**:
  - Rework `codex-frontend/src/pages/DetailPage.tsx` so the page layout clearly separates image inspection from proposal inspection.
  - Preserve existing data flow from `services/storage.ts` and route loading in `App.tsx`, but restructure the page into a more deliberate analyst workspace.
  - Make room for a stable image pane and a proposal-inspection pane without yet implementing all interactions.

  **Must NOT do**:
  - Do not alter backend APIs or result payload shapes.
  - Do not introduce multi-image comparison in this task.

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: page structure changes affect interaction flow, state ownership, and layout composition.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: useful later, but this task is more about information architecture than visual polish.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: 4, 5, 8
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current inspection page, overlay, list layout, and state shape.
  - `codex-frontend/src/App.tsx` - route structure and page responsibilities.
  - `codex-frontend/src/types/index.ts` - shape of `AnalysisRecord`, `SegmentResult`, and `DetectedElement` that the layout must respect.

  **Acceptance Criteria**:
  - [ ] `DetailPage.tsx` is reorganized into a clearly readable inspection workspace structure.
  - [ ] Existing analysis loading still works for `/analysis/:id`.
  - [ ] No backend contract changes are required.

  **QA Scenarios**:
  ```
  Scenario: Detail page loads into a clearer two-zone inspection layout
    Tool: Playwright
    Preconditions: Frontend and backend running; at least one saved analysis record exists
    Steps:
      1. Open `http://localhost:5173/analysis/<known-id>`
      2. Verify the page visibly separates the image inspection region from the proposal information region
      3. Verify the image and proposal list are simultaneously visible without relying on expanded rows only
    Expected Result: The page presents a stable inspection workspace rather than a cramped mixed layout
    Failure Indicators: Image and proposal data still compete in one cluttered column; important detail remains hidden until multiple clicks
    Evidence: .sisyphus/evidence/task-1-detail-layout.png

  Scenario: Missing analysis record still fails gracefully
    Tool: Playwright
    Preconditions: Frontend running
    Steps:
      1. Open `http://localhost:5173/analysis/non-existent-id`
      2. Verify the page shows a readable fallback state
    Expected Result: The page handles missing data without layout breakage
    Evidence: .sisyphus/evidence/task-1-missing-analysis.png
  ```

- [x] 2. Define inspection state model for hover, focus, and view modes

  **What to do**:
  - Introduce explicit UI state for proposal hover, selected/focused proposal, and overlay display mode.
  - Keep the state local to the appropriate inspection page/components unless a broader abstraction is clearly justified.
  - Ensure this state can drive both overlay rendering and proposal-panel presentation.

  **Must NOT do**:
  - Do not add global app state for a page-local concern.
  - Do not encode focus state into stored analysis data.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: localized state modeling and prop flow cleanup.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: 4, 6, 9
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - existing expanded-items state pattern.
  - `codex-frontend/src/pages/AnnotationPage.tsx` - example of local page-level interaction state.

  **Acceptance Criteria**:
  - [ ] Hovered proposal, focused proposal, and overlay mode are modeled explicitly.
  - [ ] State can be consumed by both image-side and list-side UI.

  **QA Scenarios**:
  ```
  Scenario: Hover and focus states are independent and stable
    Tool: Playwright
    Preconditions: Detail page with proposals exists
    Steps:
      1. Hover one proposal item
      2. Click a different proposal item to focus it
      3. Move pointer away
    Expected Result: Hover state clears while focused state remains locked
    Evidence: .sisyphus/evidence/task-2-focus-state.txt

  Scenario: Overlay mode can change without corrupting selection state
    Tool: Playwright
    Preconditions: Detail page loaded with a focused proposal
    Steps:
      1. Switch overlay mode between available options
      2. Verify the same focused proposal remains selected
    Expected Result: UI mode changes do not reset the user’s inspection target
    Evidence: .sisyphus/evidence/task-2-overlay-mode.txt
  ```

- [x] 3. Extract overlay rendering responsibilities into reusable units

  **What to do**:
  - Refactor overlay drawing logic out of the monolithic detail-page render path into a clearer reusable unit or helper.
  - Preserve existing bbox scaling correctness while making later hover/focus styling easier to implement.
  - Keep the implementation simple and tied to single-image inspection.

  **Must NOT do**:
  - Do not introduce premature abstraction for future multi-image workflows.
  - Do not change bbox semantics or stored data structures.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: localized extraction/refactor with narrow scope.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: 4, 6
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current canvas drawing logic.
  - `codex-frontend/src/types/index.ts` - bbox data shape used for rendering.

  **Acceptance Criteria**:
  - [ ] Overlay drawing responsibilities are isolated enough to support focused/highlight states cleanly.
  - [ ] Existing bounding boxes still align with the image.

  **QA Scenarios**:
  ```

- [x] 4. Implement synchronized overlay and proposal-panel hover/focus behavior

  **What to do**:
  - Link proposal cards and overlay boxes so they respond to each other.
  - Support hover preview and click-to-lock focus.
  - Ensure image-side and list-side interactions reinforce each other instead of competing.

  **Must NOT do**:
  - Do not require users to rely only on the list or only on the image.
  - Do not create unstable focus behavior where hover overrides click-lock unexpectedly.

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: interaction logic spans state, overlay rendering, and proposal list behavior.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: 8, 10
  - **Blocked By**: 1, 2, 3

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current static overlay/list split.
  - `codex-frontend/src/types/index.ts` - proposal indices and bbox metadata.

  **Acceptance Criteria**:
  - [ ] Hovering a proposal card visually highlights the corresponding region.
  - [ ] Clicking a proposal locks focus until another proposal is selected or focus is cleared.
  - [ ] If overlay interaction is enabled, clicking a box can also focus the corresponding proposal entry.

  **QA Scenarios**:
  ```
  Scenario: Hovering a proposal card highlights the correct box
    Tool: Playwright
    Preconditions: Detail page loaded with 3+ proposals
    Steps:
      1. Hover the second proposal card in the right panel
      2. Observe the image overlay
    Expected Result: Only the corresponding proposal box becomes visually emphasized
    Failure Indicators: No highlight appears, wrong box is highlighted, or all boxes remain equally noisy
    Evidence: .sisyphus/evidence/task-4-hover-highlight.png

  Scenario: Clicking a proposal locks focus across pointer movement
    Tool: Playwright
    Preconditions: Detail page loaded
    Steps:
      1. Click one proposal card
      2. Move the pointer over another card without clicking
      3. Move the pointer out of the panel entirely
    Expected Result: The clicked proposal remains the focused one until changed explicitly
    Evidence: .sisyphus/evidence/task-4-click-focus.png
  ```

- [x] 5. Build richer proposal cards with crop previews and clearer metadata hierarchy

  **What to do**:
  - Turn the current proposal list into inspection cards that show cropped proposal previews, label/confidence prominence, and clearer top-k access.
  - Use the existing bbox/result data to derive a more analyst-friendly presentation.
  - Preserve annotation-awareness (show corrected labels if applicable).

  **Must NOT do**:
  - Do not overload cards with too much low-priority metadata at rest.
  - Do not hide important states like `rejected` or annotation corrections.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: this is primarily presentation quality, hierarchy, and inspection ergonomics.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocks**: 8, 10
  - **Blocked By**: 1

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current element list UI and top-k expansion pattern.
  - `codex-frontend/src/pages/AnnotationPage.tsx` - corrected-label and suggestion presentation patterns.
  - `codex-frontend/src/pages/DashboardPage.tsx` - existing card layout pattern.

  **Acceptance Criteria**:
  - [ ] Each proposal card includes a visual crop preview.
  - [ ] Primary prediction and confidence are readable at a glance.
  - [ ] Rejected/corrected states are visually obvious.

  **QA Scenarios**:
  ```
  Scenario: Proposal cards show enough information without expansion
    Tool: Playwright
    Preconditions: Detail page loaded with multiple proposals
    Steps:
      1. Open the proposal panel
      2. Inspect at least three proposal cards without expanding any
    Expected Result: User can understand each card’s region, class, and confidence at a glance
    Evidence: .sisyphus/evidence/task-5-proposal-cards.png

  Scenario: Corrected/rejected states remain visible in the new card design
    Tool: Playwright
    Preconditions: Analysis record with at least one rejected or annotated proposal
    Steps:
      1. Open the detail page
      2. Inspect the affected proposal cards
    Expected Result: The card clearly indicates rejected and/or corrected status
    Evidence: .sisyphus/evidence/task-5-proposal-status.png
  ```

- [x] 6. Add overlay visibility and focus view controls

  **What to do**:
  - Add lightweight controls so users can reduce clutter during inspection.
  - Support at least: all proposals visible, focused proposal only, and overlay hidden.
  - Ensure these controls integrate with the focus/hover model without confusion.

  **Must NOT do**:
  - Do not add excessive display modes that complicate the workflow.
  - Do not bury these controls where analysts won’t discover them.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: compact control design and clean interaction feedback.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocks**: 8, 10
  - **Blocked By**: 2, 3

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current always-on overlay behavior.

  **Acceptance Criteria**:
  - [ ] Users can switch between all-boxes, focused-only, and hidden overlay modes.
  - [ ] Mode changes do not break proposal focus or card interactions.

  **QA Scenarios**:
  ```
  Scenario: Overlay mode controls reduce clutter without breaking context
    Tool: Playwright
    Preconditions: Detail page loaded and at least one proposal focused
    Steps:
      1. Switch between all, focused-only, and hidden modes
      2. Inspect whether the proposal panel still indicates the active selection
    Expected Result: The user can declutter the image while preserving inspection context
    Evidence: .sisyphus/evidence/task-6-overlay-modes.png

  Scenario: Hidden overlay mode still leaves the workflow usable
    Tool: Playwright
    Preconditions: Detail page loaded
    Steps:
      1. Hide the overlay
      2. Use the proposal panel to inspect and focus proposals
    Expected Result: The user still has a coherent inspection experience even with overlays off
    Evidence: .sisyphus/evidence/task-6-overlay-hidden.png
  ```

- [x] 7. Preserve and improve annotation handoff from the inspection workflow

  **What to do**:
  - Make the path from inspection to correction more obvious and contextual.
  - Ensure focused/rejected proposals can flow naturally into the existing annotation page.
  - Preserve current storage behavior while improving the workflow affordance.

  **Must NOT do**:
  - Do not redesign annotation storage in this phase.
  - Do not require backend support for annotation persistence.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: this is mostly route/action continuity using existing structures.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: 9, 10
  - **Blocked By**: 1

  **References**:
  - `codex-frontend/src/pages/AnnotationPage.tsx` - current correction flow.
  - `codex-frontend/src/services/storage.ts` - local annotation persistence.
  - `codex-frontend/src/pages/DetailPage.tsx` - existing annotate CTA.

  **Acceptance Criteria**:
  - [ ] Annotation entry remains available from the inspection page.
  - [ ] The handoff is clearer for low-confidence/rejected proposals.
  - [ ] Existing annotation save/return behavior still works.

  **QA Scenarios**:
  ```

- [x] 8. Improve information hierarchy, empty states, and analyst readability across the inspection page

  **What to do**:
  - Polish the inspection workflow so users can quickly understand what happened in a segmentation run.
  - Improve the readability of summary information, rejected-state messaging, and low/no-proposal outcomes.
  - Make the page resilient when analyses contain sparse or confusing results.

  **Must NOT do**:
  - Do not turn this into a full dashboard redesign.
  - Do not introduce batch/multi-image compare UX in this task.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: this is about clarity, hierarchy, and visual communication quality.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10)
  - **Blocks**: 10
  - **Blocked By**: 4, 5, 6

  **References**:
  - `codex-frontend/src/pages/DetailPage.tsx` - current cluttered information presentation.
  - `codex-frontend/src/pages/DashboardPage.tsx` - examples of concise metadata badges.
  - `codex-frontend/src/pages/AnnotationPage.tsx` - rejected-state messaging patterns.

  **Acceptance Criteria**:
  - [ ] The page communicates proposal count, rejected count, and inspection state cleanly.
  - [ ] Empty, low-confidence, or sparse-result analyses are handled gracefully.
  - [ ] The user can tell what to do next when results are weak or incomplete.

  **QA Scenarios**:
  ```
  Scenario: Sparse segmentation results remain understandable
    Tool: Playwright
    Preconditions: Analysis record with 0-1 detected proposals
    Steps:
      1. Open the detail page
      2. Inspect the summary and empty/sparse-state messaging
    Expected Result: The page explains the situation clearly and does not feel broken or empty
    Evidence: .sisyphus/evidence/task-8-sparse-state.png

  Scenario: Analysts can identify next steps for weak results
    Tool: Playwright
    Preconditions: Analysis with rejected or low-confidence proposals
    Steps:
      1. Open the detail page
      2. Inspect the summary and workflow hints
    Expected Result: The UI makes it obvious whether to inspect, annotate, or move on
    Evidence: .sisyphus/evidence/task-8-next-step.png
  ```

- [x] 9. Validate storage, typing, and route continuity for the upgraded workflow

  **What to do**:
  - Ensure the upgraded inspection flow still works with existing `AnalysisRecord` storage and route assumptions.
  - Confirm annotation updates, history navigation, and detail loading remain compatible.
  - Keep the data contract stable unless a clearly safe local UI extension is necessary.

  **Must NOT do**:
  - Do not redesign the persistence model.
  - Do not add backend state synchronization.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: targeted compatibility checks across existing services and routes.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 10)
  - **Blocks**: 10
  - **Blocked By**: 2, 7

  **References**:
  - `codex-frontend/src/services/storage.ts` - local analysis/annotation persistence.
  - `codex-frontend/src/types/index.ts` - record/result shapes.
  - `codex-frontend/src/App.tsx` - route continuity.
  - `codex-frontend/src/pages/DashboardPage.tsx` - history entrypoint into detail inspection.

  **Acceptance Criteria**:
  - [ ] Existing saved analyses still load into the upgraded detail experience.
  - [ ] Annotation updates remain visible after returning to detail view.
  - [ ] Dashboard → Detail → Annotation → Detail flow still works.

  **QA Scenarios**:
  ```
  Scenario: Existing saved records still render correctly
    Tool: Playwright
    Preconditions: Local storage contains one or more analysis records created before the UI upgrade
    Steps:
      1. Open the dashboard
      2. Navigate into an older analysis record
    Expected Result: Existing data renders without migration failures or broken layout assumptions
    Evidence: .sisyphus/evidence/task-9-existing-record.png

  Scenario: Full route continuity survives the workflow upgrade
    Tool: Playwright
    Preconditions: Frontend running with existing local history
    Steps:
      1. Open dashboard
      2. Open detail page
      3. Move into annotation page
      4. Save and return
      5. Navigate back to dashboard
    Expected Result: Navigation remains coherent across all existing routes
    Evidence: .sisyphus/evidence/task-9-route-continuity.txt
  ```

- [x] 10. Run a manual browser QA pass and resolve inspection-workflow issues

  **What to do**:
  - Perform the agreed manual QA against the implemented analyst workflow.
  - Validate the core inspection interactions using real segmentation results.
  - Fix any usability or state bugs discovered during the manual pass.

  **Must NOT do**:
  - Do not add test infrastructure in response to QA findings.
  - Do not use QA as a reason to drift into threshold/model changes.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: integrated browser verification and iteration across the upgraded workflow.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9)
  - **Blocks**: F1-F4
  - **Blocked By**: 4, 5, 6, 8, 9

  **References**:
  - `codex-frontend/src/pages/UploadPage.tsx`
  - `codex-frontend/src/pages/DetailPage.tsx`
  - `codex-frontend/src/pages/AnnotationPage.tsx`
  - `codex-frontend/src/services/api.ts`
  - `codex-frontend/src/services/storage.ts`

  **Acceptance Criteria**:
  - [ ] The full upload → inspection → annotation handoff flow is manually verified in the browser.
  - [ ] Hover/focus interactions behave consistently.
  - [ ] The upgraded page is meaningfully less cluttered than before.

  **QA Scenarios**:
  ```
  Scenario: Full analyst workflow works on a real uploaded glyph
    Tool: Playwright
    Preconditions: Frontend and backend running locally
    Steps:
      1. Open `http://localhost:5173`
      2. Upload a known glyph image
      3. Run analysis
      4. Inspect the resulting detail page using hover, click-focus, and overlay mode controls
      5. Move into annotation if applicable and return
    Expected Result: The upgraded flow supports a coherent analyst review loop end-to-end
    Evidence: .sisyphus/evidence/task-10-full-workflow.png

  Scenario: Inspection remains usable on a second analysis without page-state leakage
    Tool: Playwright
    Preconditions: At least two saved analyses exist
    Steps:
      1. Inspect one analysis and focus a proposal
      2. Return to dashboard
      3. Open a different analysis
    Expected Result: The second analysis starts with clean UI state and no stale focus/overlay mode leakage
    Evidence: .sisyphus/evidence/task-10-state-isolation.txt
  ```
  Scenario: Analyst can move from a problematic proposal to annotation quickly
    Tool: Playwright
    Preconditions: Detail page with at least one rejected proposal
    Steps:
      1. Focus a rejected proposal
      2. Trigger the annotation workflow
      3. Confirm the annotation page loads the relevant analysis record
    Expected Result: The user can move from inspection to correction without losing context
    Evidence: .sisyphus/evidence/task-7-annotation-handoff.png

  Scenario: Saving annotations still returns to detail inspection cleanly
    Tool: Playwright
    Preconditions: Annotation page open for an analysis record
    Steps:
      1. Change one annotation
      2. Save
      3. Return to detail inspection
    Expected Result: The updated label is reflected in the inspection workflow
    Evidence: .sisyphus/evidence/task-7-annotation-return.png
  ```
  Scenario: Bounding boxes remain aligned after overlay refactor
    Tool: Playwright
    Preconditions: Detail page with known proposals loaded
    Steps:
      1. Open the detail page
      2. Compare rendered boxes against visible regions in the image
      3. Resize the browser window and re-check alignment
    Expected Result: Boxes remain correctly positioned after refactor and responsive resize
    Evidence: .sisyphus/evidence/task-3-overlay-alignment.png

  Scenario: Overlay rendering does not disappear when proposal count is zero
    Tool: Playwright
    Preconditions: Analysis record with zero proposals or mocked equivalent state
    Steps:
      1. Open the detail page
      2. Verify the page remains stable and overlay area does not crash
    Expected Result: Empty overlay state renders safely
    Evidence: .sisyphus/evidence/task-3-empty-overlay.png
  ```

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify the implemented UI matches the single-glyph analyst workflow scope, preserves annotation access, and does not change model behavior.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Review changed frontend files for state-management clarity, overlay correctness, accidental regressions, and lint/build health.

- [x] F3. **Real Manual QA** — `unspecified-high`
  Exercise the upload → detail inspection → annotation handoff flow in the browser with real segmentation results and capture evidence.

- [x] F4. **Scope Fidelity Check** — `deep`
  Confirm the work stayed focused on single-glyph clarity and did not expand into threshold tuning, backend redesign, or multi-image comparison.

---

## Commit Strategy

- `refactor(detail): restructure segmentation inspection layout`
- `feat(detail): add synchronized proposal focus interactions`
- `feat(detail): add proposal preview cards and overlay modes`
- `refactor(annotation): preserve analyst workflow handoff`

---

## Success Criteria

### Verification Commands
```bash
cd codex-frontend && npm run build
cd codex-frontend && npm run lint
cd frontend_integration_fix/frontend_integration && uv run examples/flask_api.py
```

### Final Checklist
- [ ] Single glyph inspection workflow is cleaner and easier to understand
- [ ] Overlay/list synchronization works in both hover and focused states
- [ ] Proposal information is easier to inspect than in the current UI
- [ ] Annotation path remains functional
- [ ] No model or threshold logic changed

# Workspace UI Redesign

## TL;DR

> **Quick Summary**: Redesign the glyph-analysis workspace into an image-first production UI: compact chrome, collapsible history, clickable segmentation regions, zoomable canvas, inline annotation for any detection, and a stronger full annotation editor.
>
> **Deliverables**:
> - Image-centered workspace on `/`
> - Collapsible history sidebar
> - Clickable/zoomable segmentation canvas with synchronized detail panel
> - Inline relabel/status editing for any region regardless of confidence
> - Improved `/annotate/:id` full editor with context handoff
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 6 → Task 9 → Task 12 → Task 16 → Final Verification

---

## Context

### Original Request
- “Use visual engineering agent to have a better UI.”
- “Analyzed image should be the central point, if possible with the possibility to zoom.”
- “Segmentation squares should be clickable with panel appearing, as well as sidebar with more informations.”
- “There are too much useless text information that are obvious, things should be more compact and more thoughtfully placed.”
- “Annotation is not really working, even if percentage is high enough, annotating should be possible.”
- “Make it as state of the art as possible while keeping it very user friendly.”
- “KPIs are simply useless. We are targeting a highly usable app for anyone to use. This is not a demo, it will be a production app, it should feel like it.”

### Interview Summary
**Key Discussions**:
- Annotation should support **inline relabel/status editing** in the main workspace.
- A **separate full editor route** must still exist.
- History should become a **collapsible sidebar**.
- Zoom should support **buttons + mouse wheel + fit/reset**.
- Priority is **desktop-only**.
- Verification stays **manual QA only**.

**Research Findings**:
- `WorkspacePage.tsx` currently mixes upload, history, KPI cards, overlay view, and proposal cards in a card-heavy layout.
- `AnnotationPage.tsx` currently blocks editing when no rejected elements exist and only targets rejected detections.
- `storage.ts` already supports persisted annotations by record id.
- `api.ts` already exposes `getClasses()` for annotation choices.

### Oracle Gap Review
**Identified Gaps** (addressed in this plan):
- Selection model was under-specified → default to **single-select** region interaction.
- Unsaved inline-edit behavior was ambiguous → default to **immediate persisted save with visible status**, avoiding draft divergence.
- Full-editor handoff context was unclear → open `/annotate/:id?element=<idx>` when a region is focused.
- Dense/overlapping region handling needed explicit treatment → add zoom bounds, hit-area rules, and fallback list-first disambiguation.
- History default state was not locked → default to **collapsed when a record is active**, accessible by explicit toggle.

---

## Work Objectives

### Core Objective
Transform the current merged workspace into a production-grade analysis experience where the analyzed image is the primary surface, supporting direct on-canvas inspection and fast correction workflows without clutter.

### Concrete Deliverables
- Compact application chrome with no demo-style hero copy or KPI cards
- Image-first analysis stage with zoom, fit, and interactive segmentation boxes
- Collapsible history rail that does not dominate the screen
- Compact region details panel synchronized with the selected box
- Inline annotation controls for all detections, not only rejected ones
- Improved full annotation page that works for every detection and preserves context from the workspace

### Definition of Done
- [ ] The first visible workspace state is dominated by the analyzed image rather than stats or instructional copy
- [ ] Users can click a segmentation box on the image and see the corresponding detail panel state update
- [ ] Users can zoom in/out, fit, and reset the image without losing overlay alignment
- [ ] Users can relabel or change status for any detection regardless of confidence level
- [ ] `/annotate/:id` remains available as a richer full editor and can open from the currently focused region
- [ ] Upload, history switching, delete, and annotation persistence still work end-to-end

### Must Have
- Image-first layout
- Collapsible history sidebar
- Clickable overlay regions
- Synchronized region panel + sidebar behavior
- Inline edit support for any detection
- Full editor route preserved and improved
- Compact copy and reduced visual noise

### Must NOT Have (Guardrails)
- No backend API redesign
- No model retraining, threshold tuning, or inference logic changes
- No full geometry editing inline in the workspace
- No broad app-wide redesign outside the analysis workspace + annotation entry points
- No mobile-first/responsive overhaul in this phase
- No new heavy UI framework unless existing utilities cannot reasonably deliver the required UX

---

## Verification Strategy

> **Manual QA only** for this redesign. Existing automated safety net remains build + lint.

### Test Decision
- **Infrastructure exists**: Partial (build/lint only)
- **Automated tests**: None for this work
- **Framework**: Manual browser QA + `npm run build` + `npm run lint`

### QA Policy
Every task must include agent-executed browser QA scenarios. Final verification must cover empty, normal, dense, overlapping, and annotation-handoff flows.

- **Frontend/UI**: Use Playwright for interaction verification and screenshots
- **Persistence**: Use browser reload + storage-backed flows to verify saved state
- **Evidence**: Save screenshots or terminal outputs to `.sisyphus/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation + layout contract):
├── Task 1: Replace hero/KPI shell with compact production chrome [visual-engineering]
├── Task 2: Convert history into a collapsible secondary rail [visual-engineering]
├── Task 3: Define interaction state model for selection, zoom, and focused-region context [deep]
├── Task 4: Normalize copy/empty/error language across workspace surfaces [writing]
└── Task 5: Unlock annotation data flow for all detections [deep]

Wave 2 (Core workspace interaction):
├── Task 6: Build image-first canvas stage with floating zoom/overlay controls [visual-engineering]
├── Task 7: Make overlay boxes directly clickable and synchronized with the detail panel [deep]
├── Task 8: Replace bulky proposal cards with compact contextual region panel [visual-engineering]
├── Task 9: Add inline relabel/status editing with immediate persistence feedback [deep]
├── Task 10: Refine history switching/search/toggle behavior around active analysis [quick]
└── Task 11: Add baseline keyboard/focus navigation for region review [quick]

Wave 3 (Full editor + production polish):
├── Task 12: Redesign the full annotation page for all detections, not just rejected ones [visual-engineering]
├── Task 13: Preserve focused-region handoff and return flow between workspace and full editor [quick]
├── Task 14: Polish visual density, pane balance, and compact production styling [visual-engineering]
├── Task 15: Handle dense/tiny/overlapping-region edge cases in the workspace [deep]
└── Task 16: Manual browser QA pass and issue cleanup [unspecified-high]

Wave 4 (Cleanup + verification prep):
├── Task 17: Final frontend hygiene cleanup [quick]

Wave FINAL:
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: Real manual QA [unspecified-high]
└── Task F4: Scope fidelity check [deep]
```

### Dependency Matrix
- **1**: - - 6, 8, 14
- **2**: - - 10, 14
- **3**: - - 6, 7, 9, 11, 13, 15
- **4**: - - 14, 16
- **5**: - - 9, 12, 13, 16
- **6**: 1, 3 - 7, 8, 14, 15, 16
- **7**: 3, 6 - 8, 9, 11, 15, 16
- **8**: 1, 6, 7 - 9, 14, 16
- **9**: 3, 5, 7, 8 - 12, 13, 16
- **10**: 2 - 16
- **11**: 3, 7 - 16
- **12**: 5, 9 - 13, 16
- **13**: 3, 5, 9, 12 - 16
- **14**: 1, 2, 4, 6, 8 - 16
- **15**: 3, 6, 7 - 16
- **16**: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 - 17
- **17**: 9, 12, 16 - F1-F4

### Agent Dispatch Summary
- **Wave 1**: T1/T2 → `visual-engineering`, T3/T5 → `deep`, T4 → `writing`
- **Wave 2**: T6/T8 → `visual-engineering`, T7/T9 → `deep`, T10/T11 → `quick`
- **Wave 3**: T12/T14 → `visual-engineering`, T13 → `quick`, T15 → `deep`, T16 → `unspecified-high`
- **Wave 4**: T17 → `quick`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Replace the current hero/KPI shell with compact production chrome

  **What to do**:
  - Remove the current hero-style intro copy and KPI/stat cards from the top of `WorkspacePage`.
  - Reduce upload controls to a compact top action area that does not compete with the analysis canvas.
  - Keep upload, error, and loading affordances discoverable without occupying the first screen with marketing/demo styling.

  **Must NOT do**:
  - Do not remove upload capability.
  - Do not add product-marketing copy or dashboard-style KPI blocks back in another form.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: 6, 8, 14
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:319-429` - current hero/upload bench and KPI-heavy top shell to replace
  - `codex-frontend/src/App.tsx:25-36` - app-level shell and available chrome area

  **Acceptance Criteria**:
  - [ ] First viewport no longer shows KPI cards like “Stored runs”, “Current file”, or “Selection” as primary content
  - [ ] Upload remains available from the main workspace without pushing the analysis stage below the fold

  **QA Scenarios**:
  ```
  Scenario: Compact first-load workspace
    Tool: Playwright
    Preconditions: Frontend running, no current analysis selected
    Steps:
      1. Open `/`
      2. Capture the first viewport without scrolling
      3. Verify the page shows compact app chrome and upload action, not a hero banner plus KPI cards
    Expected Result: The UI reads like a production tool, not a marketing/dashboard screen
    Evidence: .sisyphus/evidence/task-1-first-load.png

  Scenario: Upload action remains discoverable
    Tool: Playwright
    Preconditions: Frontend running
    Steps:
      1. Open `/`
      2. Verify a visible control exists to choose or drop an image
      3. Click it and confirm the file picker affordance is wired
    Expected Result: Upload remains obvious without consuming the whole page
    Evidence: .sisyphus/evidence/task-1-upload-entry.png
  ```

- [x] 2. Convert history into a collapsible secondary rail

  **What to do**:
  - Turn the current permanent history sidebar into a toggleable/collapsible rail.
  - Default the rail to a secondary state when a record is active.
  - Preserve selection, delete, filter, and empty-state behavior.

  **Must NOT do**:
  - Do not remove history browsing.
  - Do not break delete or filter behavior.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 10, 14
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:431-527` - current always-open history sidebar
  - `codex-frontend/src/services/storage.ts:11-40` - history retrieval/delete behavior that must stay intact

  **Acceptance Criteria**:
  - [ ] History can be expanded/collapsed without losing the active record
  - [ ] Filtering, selecting, and deleting history entries still works

  **QA Scenarios**:
  ```
  Scenario: Collapse and reopen history rail
    Tool: Playwright
    Preconditions: At least two saved records exist
    Steps:
      1. Open `/`
      2. Collapse the history rail
      3. Verify the canvas/detail area gains horizontal space
      4. Reopen the history rail
    Expected Result: History becomes secondary without disappearing from the workflow
    Evidence: .sisyphus/evidence/task-2-history-collapse.png

  Scenario: History actions still work
    Tool: Playwright
    Preconditions: At least one saved record exists
    Steps:
      1. Open the history rail
      2. Filter by part of an image name
      3. Select a record
      4. Delete a different record
    Expected Result: Filter, select, and delete all still function
    Evidence: .sisyphus/evidence/task-2-history-actions.png
  ```

- [x] 3. Define the workspace interaction state model for selection, zoom, and focused-region context

  **What to do**:
  - Establish single-select region behavior as the workspace contract.
  - Add state ownership for zoom level, pan position if needed, focused region, and editor handoff context.
  - Ensure region selection survives expected UI transitions and resets predictably when the active record changes.

  **Must NOT do**:
  - Do not introduce multi-select scope.
  - Do not create ambiguous state between overlay focus and panel focus.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 6, 7, 9, 11, 13, 15
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:107-124` - current inspection state reset logic
  - `codex-frontend/src/pages/WorkspacePage.tsx:126-145` - current record selection/sync behavior
  - `codex-frontend/src/pages/WorkspacePage.tsx:664-676` - current focus/hover click behavior

  **Acceptance Criteria**:
  - [ ] Single-select interaction rules are implemented consistently across canvas and panel
  - [ ] Focused-region context can be handed to the full annotation route

  **QA Scenarios**:
  ```
  Scenario: Selection resets cleanly on record switch
    Tool: Playwright
    Preconditions: Two saved records exist with multiple regions
    Steps:
      1. Open record A and select region 1
      2. Switch to record B
      3. Verify prior selection from A is cleared
    Expected Result: No stale selection leaks across records
    Evidence: .sisyphus/evidence/task-3-selection-reset.png

  Scenario: Focused region can drive editor handoff context
    Tool: Playwright
    Preconditions: Current record has multiple regions
    Steps:
      1. Select region 2
      2. Open the full editor
      3. Verify the route/context preserves the focused element intent
    Expected Result: Workspace and editor share clear focused-region context
    Evidence: .sisyphus/evidence/task-3-editor-context.png
  ```

- [x] 4. Normalize copy, empty states, and error language across the analysis flow

  **What to do**:
  - Remove redundant instructional paragraphs and obvious explanatory text.
  - Replace noisy labels with compact, production-appropriate copy.
  - Keep empty/error states helpful but brief.

  **Must NOT do**:
  - Do not leave walls of text in the main workspace.
  - Do not remove important error context.

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 14, 16
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:329-332` - verbose intro copy to remove
  - `codex-frontend/src/pages/WorkspacePage.tsx:551-553` - inspection prose to tighten
  - `codex-frontend/src/pages/WorkspacePage.tsx:603-605` - proposal panel prose to tighten
  - `codex-frontend/src/pages/AnnotationPage.tsx:139-141` - verbose annotation instructions to simplify

  **Acceptance Criteria**:
  - [ ] Obvious instructional copy is removed from primary surfaces
  - [ ] Empty/error states remain understandable in one or two compact sentences

  **QA Scenarios**:
  ```
  Scenario: Empty workspace copy is concise
    Tool: Playwright
    Preconditions: No active record
    Steps:
      1. Open `/`
      2. Inspect empty-state messaging
    Expected Result: The workspace communicates next actions briefly without tutorial-like paragraphs
    Evidence: .sisyphus/evidence/task-4-empty-copy.png

  Scenario: Error copy remains actionable
    Tool: Playwright
    Preconditions: Backend unavailable
    Steps:
      1. Start an analysis request
      2. Inspect the visible error message
    Expected Result: Error copy is short, clear, and action-oriented
    Evidence: .sisyphus/evidence/task-4-error-copy.png
  ```

- [x] 5. Unlock annotation data flow so any detection can be edited

  **What to do**:
  - Remove the rejected-only annotation assumption from the workflow.
  - Ensure both workspace inline editing and full editor can work with any detection.
  - Preserve existing annotation persistence contract using `updateAnnotations`.

  **Must NOT do**:
  - Do not change backend APIs.
  - Do not require a detection to be rejected before it can be annotated.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 9, 12, 13, 16
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/pages/AnnotationPage.tsx:31-37` - current rejected-only initialization behavior
  - `codex-frontend/src/pages/AnnotationPage.tsx:72-90` - current early-return block when nothing is rejected
  - `codex-frontend/src/services/storage.ts:24-35` - persisted annotation update path
  - `codex-frontend/src/types/index.ts:29-35` - annotation storage shape

  **Acceptance Criteria**:
  - [ ] Users can annotate any detection regardless of confidence/rejected state
  - [ ] Existing persisted annotation contract still works

  **QA Scenarios**:
  ```
  Scenario: High-confidence region can still be edited
    Tool: Playwright
    Preconditions: Record exists with at least one non-rejected region
    Steps:
      1. Open the record
      2. Attempt to edit a high-confidence region
      3. Save and reload the record
    Expected Result: The annotation persists even though the region was not rejected
    Evidence: .sisyphus/evidence/task-5-high-confidence-edit.png

  Scenario: Rejected-region flow still works
    Tool: Playwright
    Preconditions: Record exists with at least one rejected region
    Steps:
      1. Edit a rejected region
      2. Save the change
      3. Reload the record
    Expected Result: Existing rejected-region correction still functions
    Evidence: .sisyphus/evidence/task-5-rejected-edit.png
  ```

- [x] 6. Build the image-first canvas stage with floating zoom and overlay controls

  **What to do**:
  - Rework the central inspection area so the analyzed image is the dominant visual surface.
  - Add zoom in, zoom out, and fit/reset controls with clear visible placement over or adjacent to the canvas.
  - Keep overlay mode controls compact and close to the image stage.

  **Must NOT do**:
  - Do not leave the image boxed into a secondary-feeling card.
  - Do not break overlay alignment while zooming.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10, 11)
  - **Blocks**: 7, 8, 14, 15, 16
  - **Blocked By**: 1, 3

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:547-597` - current static image overlay section
  - `codex-frontend/src/pages/WorkspacePage.tsx:30-89` - overlay drawing logic that must remain aligned

  **Acceptance Criteria**:
  - [ ] The analyzed image is the largest and clearest element of the active workspace
  - [ ] Zoom controls support buttons plus wheel and include a fit/reset behavior

  **QA Scenarios**:
  ```
  Scenario: Zoom controls work on the canvas
    Tool: Playwright
    Preconditions: Active record loaded
    Steps:
      1. Open the canvas stage
      2. Click zoom in twice
      3. Use mouse wheel to zoom out
      4. Click fit/reset
    Expected Result: The image scale changes predictably and returns to fit view
    Evidence: .sisyphus/evidence/task-6-zoom-controls.png

  Scenario: Overlay remains aligned while zooming
    Tool: Playwright
    Preconditions: Active record with visible boxes
    Steps:
      1. Capture a box position at fit view
      2. Zoom in and inspect the same region
      3. Verify the box still surrounds the same target area
    Expected Result: Overlay alignment is preserved at different zoom levels
    Evidence: .sisyphus/evidence/task-6-overlay-alignment.png
  ```

- [x] 7. Make overlay boxes directly clickable and synchronized with the detail panel

  **What to do**:
  - Add direct click/hover interaction on the overlay boxes.
  - Synchronize canvas selection with the detail panel selection state.
  - Ensure clicking a box opens or focuses the corresponding region details.

  **Must NOT do**:
  - Do not keep the canvas non-interactive.
  - Do not create mismatch between selected box and selected panel item.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 8, 9, 11, 15, 16
  - **Blocked By**: 3, 6

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:587-596` - current image/canvas structure where pointer events are disabled
  - `codex-frontend/src/pages/WorkspacePage.tsx:641-747` - current list-based region selection state
  - `codex-frontend/src/types/index.ts:14-21` - bbox geometry source

  **Acceptance Criteria**:
  - [ ] Users can click a segmentation region directly on the image
  - [ ] Canvas selection and panel selection always match

  **QA Scenarios**:
  ```
  Scenario: Clicking a box selects the matching region
    Tool: Playwright
    Preconditions: Active record with multiple regions
    Steps:
      1. Click a visible segmentation box on the canvas
      2. Verify the matching region becomes selected in the panel
    Expected Result: Canvas click drives panel focus
    Evidence: .sisyphus/evidence/task-7-box-click.png

  Scenario: Panel selection updates canvas selection
    Tool: Playwright
    Preconditions: Active record with multiple regions
    Steps:
      1. Click a region in the panel
      2. Verify the matching box is highlighted on the image
    Expected Result: Selection is bi-directionally synchronized
    Evidence: .sisyphus/evidence/task-7-panel-sync.png
  ```

- [x] 8. Replace bulky proposal cards with a compact contextual region panel

  **What to do**:
  - Redesign the current right-side proposal card stack into a denser region details panel.
  - Surface only the most relevant metadata by default.
  - Keep alternative predictions and region metadata accessible without forcing large expanded cards.

  **Must NOT do**:
  - Do not keep the current card-heavy list as-is.
  - Do not hide critical region information behind impossible-to-find interactions.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 9, 14, 16
  - **Blocked By**: 1, 6, 7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:600-750` - current detected-elements card list to redesign
  - `codex-frontend/src/pages/WorkspacePage.tsx:222-250` - current stats usage to demote/remove from primary UI

  **Acceptance Criteria**:
  - [ ] Region details are more compact than the current expanded-card design
  - [ ] Users can still access label, confidence, alternatives, and edit actions for the selected region

  **QA Scenarios**:
  ```
  Scenario: Dense region panel remains readable
    Tool: Playwright
    Preconditions: Record with many detections
    Steps:
      1. Open a dense record
      2. Review the region panel without expanding many cards
    Expected Result: The panel remains scannable and compact
    Evidence: .sisyphus/evidence/task-8-compact-panel.png

  Scenario: Selected region details are easy to find
    Tool: Playwright
    Preconditions: Active record loaded
    Steps:
      1. Select a region on the canvas
      2. Verify its detail state is clearly visible in the panel
    Expected Result: Contextual details appear without excessive scrolling or expansion
    Evidence: .sisyphus/evidence/task-8-selected-details.png
  ```

- [x] 9. Add inline relabel/status editing with immediate persistence feedback

  **What to do**:
  - Add inline controls to relabel or change status for the selected region directly in the workspace.
  - Persist edits immediately using existing storage-backed annotation flow.
  - Show visible saved/error feedback without modal interruption.

  **Must NOT do**:
  - Do not require users to leave the workspace for simple relabel/status edits.
  - Do not create hidden unsaved draft state.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 12, 13, 16
  - **Blocked By**: 3, 5, 7, 8

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:641-747` - current region card actions where inline edit controls should replace “Annotate” dependence
  - `codex-frontend/src/pages/AnnotationPage.tsx:159-184` - existing class-selection and suggestion patterns to reuse
  - `codex-frontend/src/services/storage.ts:24-35` - immediate persistence path
  - `codex-frontend/src/services/api.ts:20-22` - classes source

  **Acceptance Criteria**:
  - [ ] Selected regions can be relabeled/status-edited inline regardless of confidence level
  - [ ] Successful edits persist immediately and survive refresh
  - [ ] Save failures are surfaced clearly in the workspace

  **QA Scenarios**:
  ```
  Scenario: Inline relabel persists immediately
    Tool: Playwright
    Preconditions: Active record loaded and classes available
    Steps:
      1. Select a region
      2. Change its label inline
      3. Refresh the page
      4. Reopen the same record
    Expected Result: The edited label persists after reload
    Evidence: .sisyphus/evidence/task-9-inline-persist.png

  Scenario: Inline save failure is visible
    Tool: Playwright
    Preconditions: Simulate storage save failure or invalid record state
    Steps:
      1. Trigger an inline edit on a stale/deleted record
      2. Observe the save result
    Expected Result: The user sees a clear non-silent error state
    Evidence: .sisyphus/evidence/task-9-inline-save-error.png
  ```

- [x] 10. Refine history switching, search, and toggle behavior around the active analysis

  **What to do**:
  - Ensure history toggle/search/select flows work smoothly in the more compact layout.
  - Keep the active record obvious even when history is collapsed.
  - Preserve fast switching between recent analyses.

  **Must NOT do**:
  - Do not make history discoverability worse.
  - Do not lose active-record feedback when the sidebar is closed.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 16
  - **Blocked By**: 2

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:205-220` - current history filtering logic
  - `codex-frontend/src/pages/WorkspacePage.tsx:464-525` - current history list item behavior

  **Acceptance Criteria**:
  - [ ] Users can still search, switch, and delete records after the sidebar redesign
  - [ ] Active-record context remains visible when history is collapsed

  **QA Scenarios**:
  ```
  Scenario: Search and switch within compact history
    Tool: Playwright
    Preconditions: Several saved records exist
    Steps:
      1. Open history
      2. Filter to one record
      3. Select it
      4. Collapse history
    Expected Result: The active record remains understandable after selection
    Evidence: .sisyphus/evidence/task-10-history-switch.png

  Scenario: Delete while another record is active
    Tool: Playwright
    Preconditions: At least two records exist
    Steps:
      1. Activate record A
      2. Delete record B from history
    Expected Result: Record A remains active and the list updates correctly
    Evidence: .sisyphus/evidence/task-10-history-delete.png
  ```

- [x] 11. Add baseline keyboard and focus navigation for region review

  **What to do**:
  - Support predictable keyboard focus for the region list/panel.
  - Add basic next/previous selection and enter/escape style behaviors where appropriate.
  - Ensure keyboard interaction does not conflict with mouse-driven selection.

  **Must NOT do**:
  - Do not expand into a full hotkey platform.
  - Do not break existing pointer interactions.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: 16
  - **Blocked By**: 3, 7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:668-677` - current focusable region item pattern
  - `codex-frontend/src/pages/WorkspacePage.tsx:477-485` - current keyboard handling for history items

  **Acceptance Criteria**:
  - [ ] Users can move region focus without relying only on the mouse
  - [ ] Keyboard focus and selected region remain synchronized

  **QA Scenarios**:
  ```
  Scenario: Keyboard navigation selects regions
    Tool: Playwright
    Preconditions: Active record with multiple regions
    Steps:
      1. Focus the region panel
      2. Use keyboard navigation to move to the next region
      3. Confirm the canvas selection updates
    Expected Result: Keyboard-driven review works predictably
    Evidence: .sisyphus/evidence/task-11-keyboard-nav.png

  Scenario: Escape/blur does not break selection state
    Tool: Playwright
    Preconditions: Region selected
    Steps:
      1. Select a region
      2. Move focus away / press escape if implemented
      3. Verify the workspace remains stable
    Expected Result: Focus handling is predictable and non-destructive
    Evidence: .sisyphus/evidence/task-11-focus-stability.png
  ```

- [x] 12. Redesign the full annotation page so it works for all detections

  **What to do**:
  - Update `AnnotationPage` from a rejected-only repair screen into a full editor for any detection.
  - Preserve the richer editing context of the dedicated route.
  - Keep class loading, suggestion selection, and save behavior reliable.

  **Must NOT do**:
  - Do not leave the page blocked when no rejected elements exist.
  - Do not remove the dedicated editor route.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 13, 14, 15, 16 after dependencies)
  - **Blocks**: 13, 16
  - **Blocked By**: 5, 9

  **References**:
  - `codex-frontend/src/pages/AnnotationPage.tsx:72-90` - current rejected-only block to replace
  - `codex-frontend/src/pages/AnnotationPage.tsx:107-194` - current editor layout to redesign
  - `codex-frontend/src/services/api.ts:20-22` - classes lookup
  - `codex-frontend/src/services/storage.ts:24-35` - save path

  **Acceptance Criteria**:
  - [ ] Full annotation page loads for records even when nothing is rejected
  - [ ] Users can edit any detection from the dedicated page
  - [ ] Save and return flow still works

  **QA Scenarios**:
  ```
  Scenario: Full editor opens for a fully confident record
    Tool: Playwright
    Preconditions: Record with no rejected detections exists
    Steps:
      1. Open `/annotate/<id>`
      2. Verify the page loads an editor rather than a “nothing to annotate” block
    Expected Result: Full editor works regardless of rejected-state
    Evidence: .sisyphus/evidence/task-12-full-editor-all.png

  Scenario: Full editor save persists changes
    Tool: Playwright
    Preconditions: Record loaded in full editor
    Steps:
      1. Change a label
      2. Save
      3. Return to workspace or reload editor
    Expected Result: Edited value persists
    Evidence: .sisyphus/evidence/task-12-full-editor-save.png
  ```

- [x] 13. Preserve focused-region handoff and return flow between workspace and full editor

  **What to do**:
  - Open the full editor from the currently focused region when available.
  - Preserve return navigation so users can come back to the relevant analysis context.
  - Keep deep-link behavior compatible with existing route structure.

  **Must NOT do**:
  - Do not lose the user’s place unnecessarily when switching routes.
  - Do not break `/annotate/:id` route compatibility.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 16
  - **Blocked By**: 3, 5, 9, 12

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:313-317` - current route handoff trigger
  - `codex-frontend/src/App.tsx:31-34` - route contract to preserve
  - `codex-frontend/src/pages/AnnotationPage.tsx:111-113` - current return-link behavior

  **Acceptance Criteria**:
  - [ ] Full editor can open from a focused region context
  - [ ] Return flow restores a valid analysis context in the workspace

  **QA Scenarios**:
  ```
  Scenario: Focused region opens in full editor context
    Tool: Playwright
    Preconditions: Active record with selected region
    Steps:
      1. Select region 3 in workspace
      2. Open the full editor
      3. Verify the editor highlights or lands with context for region 3
    Expected Result: Region handoff context is preserved
    Evidence: .sisyphus/evidence/task-13-editor-handoff.png

  Scenario: Return to workspace preserves analysis context
    Tool: Playwright
    Preconditions: Full editor open for a known record
    Steps:
      1. Use the return/back action
      2. Verify the workspace opens the same analysis record
    Expected Result: Route return is stable and predictable
    Evidence: .sisyphus/evidence/task-13-return-flow.png
  ```

- [x] 14. Polish pane balance, compact styling, and production visual density

  **What to do**:
  - Reduce heavy borders/radii/padding and remove “box-in-box” feel.
  - Balance canvas, history rail, and region panel proportions for desktop usage.
  - Apply a more mature production-tool visual hierarchy.

  **Must NOT do**:
  - Do not create another card-heavy dashboard aesthetic.
  - Do not bury actions in the name of minimalism.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 16
  - **Blocked By**: 1, 2, 4, 6, 8

  **References**:
  - `codex-frontend/src/App.tsx:25-29` - current app shell spacing
  - `codex-frontend/src/pages/WorkspacePage.tsx:321-322` - heavy hero shell styling to demote
  - `codex-frontend/src/pages/WorkspacePage.tsx:432,548,600` - major pane containers with heavy card treatment
  - `codex-frontend/src/pages/AnnotationPage.tsx:108-126` - editor page chrome to align visually with the redesigned workspace

  **Acceptance Criteria**:
  - [ ] Primary desktop layout feels compact and production-oriented
  - [ ] The image remains visually dominant after spacing/styling polish

  **QA Scenarios**:
  ```
  Scenario: Desktop workspace feels compact and image-first
    Tool: Playwright
    Preconditions: Active record loaded on desktop viewport
    Steps:
      1. Open `/`
      2. Capture the full workspace view
      3. Compare visual dominance of canvas vs side chrome
    Expected Result: The canvas is clearly the focal point and chrome is secondary
    Evidence: .sisyphus/evidence/task-14-desktop-balance.png

  Scenario: Actions remain discoverable after visual compression
    Tool: Playwright
    Preconditions: Active record loaded
    Steps:
      1. Try to find history toggle, annotate action, and zoom controls without scrolling
    Expected Result: Important actions remain visible and easy to use
    Evidence: .sisyphus/evidence/task-14-action-discovery.png
  ```

- [x] 15. Handle dense, tiny, and overlapping-region edge cases in the workspace

  **What to do**:
  - Improve selection usability for tiny or overlapping boxes.
  - Add reasonable zoom bounds and region-selection fallback behavior.
  - Ensure the region panel can still serve as the disambiguation path when direct canvas selection is hard.

  **Must NOT do**:
  - Do not expand into geometry editing or advanced minimap scope.
  - Do not leave tiny/overlapping regions effectively impossible to inspect.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: 16
  - **Blocked By**: 3, 6, 7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:30-89` - bbox drawing behavior that influences hit testing
  - `codex-frontend/src/pages/WorkspacePage.tsx:641-747` - selected/hovered region panel behavior
  - `codex-frontend/src/types/index.ts:14-21` - box geometry source

  **Acceptance Criteria**:
  - [ ] Small and overlapping regions remain inspectable with zoom or panel-based disambiguation
  - [ ] Zoom bounds prevent unusable over- or under-scaling

  **QA Scenarios**:
  ```
  Scenario: Tiny region can still be reviewed
    Tool: Playwright
    Preconditions: Record with small boxes exists
    Steps:
      1. Zoom in on a tiny region
      2. Select it via canvas or panel
      3. Verify details are readable
    Expected Result: Tiny regions are practical to inspect
    Evidence: .sisyphus/evidence/task-15-tiny-region.png

  Scenario: Overlapping region fallback remains usable
    Tool: Playwright
    Preconditions: Record with overlapping or dense boxes exists
    Steps:
      1. Attempt canvas selection in a dense area
      2. Use the panel to select the intended region
    Expected Result: The user still has a reliable disambiguation path
    Evidence: .sisyphus/evidence/task-15-overlap-fallback.png
  ```

- [x] 16. Run a manual browser QA pass and resolve production-workspace issues

  **What to do**:
  - Launch frontend and backend and exercise the redesigned workspace end-to-end.
  - Fix any regression found in upload, analysis viewing, zoom, history switching, inline editing, or full-editor handoff.
  - Confirm the UI feels production-ready on desktop, not demo-like.

  **Must NOT do**:
  - Do not mark complete without real browser validation.
  - Do not ignore edge-case regressions discovered during the QA pass.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`playwright`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1-F4
  - **Blocked By**: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx`
  - `codex-frontend/src/pages/AnnotationPage.tsx`
  - `codex-frontend/src/services/storage.ts`
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py`

  **Acceptance Criteria**:
  - [ ] Real browser QA executed against the redesigned workspace
  - [ ] Upload, zoom, region selection, inline editing, history switching, and full-editor handoff all validated

  **QA Scenarios**:
  ```
  Scenario: End-to-end production workspace flow
    Tool: Playwright
    Preconditions: Frontend + backend running; sample image available
    Steps:
      1. Open `/`
      2. Upload a glyph image and analyze it
      3. Verify the image stage becomes the central focus
      4. Zoom in, click a region, and edit its label inline
      5. Open the full editor from the focused region
      6. Return to workspace and switch to another saved history item
    Expected Result: The primary production workflow works without friction
    Evidence: .sisyphus/evidence/task-16-end-to-end.png

  Scenario: Refresh persistence and save stability
    Tool: Playwright
    Preconditions: Existing saved record with annotation edits
    Steps:
      1. Open a saved record
      2. Confirm prior edits are visible
      3. Refresh the browser
      4. Verify history, active record, and saved edits remain valid
    Expected Result: Persisted state remains stable across reloads
    Evidence: .sisyphus/evidence/task-16-refresh-stability.png
  ```

- [x] 17. Apply final frontend hygiene cleanup from verification feedback

  **What to do**:
  - **T17.1** — Remove dead `panOffset` state from `WorkspacePage.tsx` (declared at line ~125, reset at ~146-151 and ~781, never applied to rendering). Add a comment if removal is partial.
  - **T17.2** — Replace silent catch around `getClasses()` in `WorkspacePage.tsx` (line ~183-189) with `console.warn` or minimal observable handling consistent with existing UI patterns.
  - **T17.3** — Remove redundant state update in `handleInlineRelabel` (`WorkspacePage.tsx` line ~459-491): it sets `records`/`currentRecord` directly AND calls `syncRecords` which sets them again. Keep the single source of truth.
  - **T17.4** — Fix `resetInspectionState` dependency array (`WorkspacePage.tsx` line ~138-153) to prevent unnecessary re-renders. Use functional state updates or ref-based patterns.
  - **T17.5** — Remove `console.error` from `AnnotationPage.tsx` `loadData()` (line ~46-52). The UI already shows `classesError` state; no need for console noise.
  - **T17.6** — Add explicit `type="button"` on **all buttons** in `AnnotationPage.tsx` that lack it (save button at line ~118-125, suggestion buttons at line ~216-223).
  - **T17.7** — Fix `saving` state bug in `AnnotationPage.tsx` `handleSave` (line ~94-107): call `setSaving(false)` on the success path before `navigate('/')`.

  **Must NOT do**:
  - Do NOT refactor shared state architecture beyond these 7 targeted fixes.
  - Do NOT add new notifications, retry flows, or broader UX changes.
  - Do NOT change backend APIs, model behavior, or route structure.
  - Do NOT run formatting-only changes (prettier/eslint --fix) unless a line is already touched for a real fix.
  - Do NOT fix silent catches or missing `type="button"` outside the two specified files.
  - Do NOT touch CSS/styling.

  **Guardrails**:
  - Make the SMALLEST possible change per issue (≤5 lines preferred, ≤10 lines max).
  - Run `tsc --noEmit` after EACH individual fix.
  - Run app smoke test after EACH fix: WorkspacePage (upload → analyze → click overlay → relabel → save) and AnnotationPage (load → edit → save → navigate back).
  - STOP and escalate if any fix requires >10 lines or breaks `tsc --noEmit` / smoke test.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: 9, 12, 16

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx` - inline relabel flow, class loading path, and unused interaction state to simplify without changing UX.
  - `codex-frontend/src/pages/AnnotationPage.tsx` - save lifecycle, class-load error handling, and button semantics to clean up.
  - Final verification findings (F2) - source of the six minor cleanup items.

  **Acceptance Criteria**:
  - [ ] T17.1: `panOffset` state removed from `WorkspacePage.tsx`; `grep 'panOffset'` returns only comments or zero matches.
  - [ ] T17.2: Silent catch in `WorkspacePage.tsx` (~line 183-189) replaced with `console.warn` or minimal observable handling.
  - [ ] T17.3: `handleInlineRelabel` no longer performs redundant state updates before calling `syncRecords`.
  - [ ] T17.4: `resetInspectionState` dependency array fixed to prevent unnecessary re-renders.
  - [ ] T17.5: `console.error` removed from `AnnotationPage.tsx` `loadData()` (~line 46-52).
  - [ ] T17.6: All buttons in `AnnotationPage.tsx` missing `type="button"` now have it (save + suggestion buttons).
  - [ ] T17.7: `handleSave` in `AnnotationPage.tsx` calls `setSaving(false)` on success path before `navigate('/')`.
  - [ ] `tsc --noEmit` passes with zero errors after all fixes.
  - [ ] `npm run lint` passes on modified files with zero new warnings.
  - [ ] Bundle size does not increase (or increases by <0.1KB).

  **QA Scenarios**:
  ```
  Scenario: Inline relabel remains stable after cleanup
    Tool: Playwright
    Preconditions: Active record loaded with at least one region
    Steps:
      1. Select a region in the workspace
      2. Change its label inline
      3. Refresh the page and reopen the same record
    Expected Result: The updated label persists and no duplicate-refresh regression appears
    Evidence: .sisyphus/evidence/task-17-inline-relabel-stability.png

  Scenario: Annotation save flow resets cleanly
    Tool: Playwright
    Preconditions: Open a valid annotation editor record
    Steps:
      1. Change a label in `/annotate/:id`
      2. Click Save
      3. Observe navigation and button lifecycle
    Expected Result: Save completes cleanly, returns to workspace, and the button is not left in a stuck loading state
    Evidence: .sisyphus/evidence/task-17-annotation-save-flow.png
  ```

---

## Final Verification Wave (Post-T17)

> **Note**: F1-F4 are manual verification checklists, not executable test suites. Each must be completed by an agent with access to the codebase and browser.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read `WorkspacePage.tsx` and `AnnotationPage.tsx`. Verify:
  - Image is the dominant workspace surface (canvas > side chrome)
  - History sidebar is collapsible with toggle control
  - Overlay canvas is clickable and drives panel focus
  - Region panel is compact (single-row unselected, expanded focused)
  - Inline relabel works for any detection regardless of confidence
  - Full editor route `/annotate/:id` is preserved and improved
  - No KPI cards or hero copy in primary viewport
  Return: `PASS` or `FAIL` with specific line references for any gap.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run these commands and inspect results:
  ```bash
  cd codex-frontend && npm run build
  cd codex-frontend && npm run lint
  cd codex-frontend && npx tsc --noEmit
  ```
  Also read both files for:
  - `as any`, `@ts-ignore`, empty catches (outside the one intentionally fixed in T17.2)
  - Leftover TODO/FIXME/HACK
  - Interaction state ownership (zoom, focusedIdx, historyOpen in one place)
  - Overlay hit-testing correctness
  - Persistence logic correctness (`updateAnnotations` return value checked)
  Return: `PASS` or `FAIL` with command outputs and file:line references.

- [x] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill)
  Launch frontend dev server. Use Playwright to verify:
  - `/` loads with compact chrome, no console errors (CORS expected if backend off)
  - History rail collapses/expands
  - `/annotate/fake-id` shows graceful "not found" state
  - Upload file input is present and wired
  - Zoom controls exist when image loaded
  - Region panel is scrollable and compact
  Return: `PASS` or `FAIL` with screenshots saved to `.sisyphus/evidence/final-qa/`.

- [x] F4. **Scope Fidelity Check** — `deep`
  Read these files and confirm NO scope violations:
  - `codex-frontend/src/pages/WorkspacePage.tsx` — frontend only, no backend changes
  - `codex-frontend/src/pages/AnnotationPage.tsx` — frontend only
  - `codex-frontend/src/App.tsx` — routes unchanged except legacy redirects
  - `codex-frontend/src/services/api.ts` — no new endpoints
  - `codex-frontend/src/services/storage.ts` — no new storage methods
  - `codex-frontend/package.json` — no new dependencies
  Reject if: backend API redesign, model/threshold changes, inline geometry editing, app-wide redesign outside workspace/annotation, new heavy UI framework, or new dependencies added.
  Return: `PASS` or `FAIL` with specific violations found.

---

## Commit Strategy

- Commit 1: compact shell + history rail + copy cleanup
- Commit 2: image stage + clickable overlay + compact region panel
- Commit 3: inline annotation + full editor improvements + handoff flow
- Commit 4: QA-driven fixes and polish only if needed
- Commit 5 (T17): Atomic hygiene fixes — one commit per micro-fix:
  - `T17.1: Remove dead panOffset state`
  - `T17.2: Replace silent getClasses catch with console.warn`
  - `T17.3: Remove redundant state update in handleInlineRelabel`
  - `T17.4: Fix resetInspectionState dependency array`
  - `T17.5: Remove console.error from AnnotationPage loadData`
  - `T17.6: Add type="button" to AnnotationPage buttons`
  - `T17.7: Fix saving state bug in AnnotationPage handleSave`

---

## Success Criteria

### Verification Commands
```bash
cd codex-frontend && npm run build
cd codex-frontend && npm run lint
```

### Final Checklist
- [ ] The analyzed image is the dominant workspace surface
- [ ] Segmentation regions are clickable on-canvas and synchronize with the panel
- [ ] Zoom, fit, and reset work without overlay drift
- [ ] KPI-style summary cards and verbose copy are removed from primary UX
- [ ] Any detection can be annotated inline or in the full editor route
- [ ] History remains available but secondary
- [ ] No backend/model contract changes were introduced

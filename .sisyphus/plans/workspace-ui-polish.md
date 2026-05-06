# Workspace UI Polish — Quick Fixes

## TL;DR

> **Quick Summary**: Suite de fixes et polish rapides sur le workspace existant : suppression du header, normalisation des images, overlay synchronisé au zoom, crop zoomé de la région sélectionnée, drag & drop amélioré, panel d'annotation inline.
>
> **Deliverables**:
> - Header supprimé, workspace pleine hauteur
> - Images analysées à taille normalisée
> - Overlay segmentation suit le zoom correctement
> - Sidebar "History" sans label redondant
> - Panneau de détail montre un crop zoomé de la région sélectionnée
> - Drag & drop sur toute la page + modal d'import
> - Annotation inline sur la même page (panel glissant ou modal)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (header) → Task 3 (zoom overlay) → Task 5 (crop zoom) → Task 7 (inline annotate)

---

## Context

### Original Request (user feedback)
- "Enlève le header il sert à rien."
- "Toutes les images devraient être affichées à la même taille, peu importe la taille originale de l'image."
- "Les segmentations ne s'updatent pas correctement lors du changement de zoom."
- "La zone sélectionnée devrait afficher une portion zoomée de la segmentation pour bien comprendre ce que l'on voit."
- "Il y a de place place, utilise là à bon escient."
- "Le bouton 'Annotate current record' devrait faire apparaître un panel sur la même page, plus simple pour annoter avec les informations en direct."
- "Enlève 'History' dans la barre d'historique, c'est pas utile."
- "Le drop d'image devrait être plus simple pour l'utilisateur. Glisser une image devrait marcher plus naturellement que juste sur la zone dédiée actuellement."
- "Le bouton d'importation devient un panel modal, comme une pop up centrée en foreground, pour glisser ou cliquer pour importer et qui affiche des informations sur l'image importée en détails."

### Current State
- Workspace redesign (T1-T17) is complete and verified.
- `WorkspacePage.tsx` ~988 lines, `AnnotationPage.tsx` ~246 lines.
- Overlay canvas uses CSS `transform: scale(zoom)` on a wrapper div.
- Canvas hit-testing uses `offsetX/offsetY` in canvas coordinate space.
- Region panel shows a small 48px crop thumbnail when expanded.
- History sidebar has a label "ANALYSIS HISTORY" at the top.
- Upload is via a dropzone area only.
- Top nav bar shows "Codex Analyzer" with amber styling.

---

## Work Objectives

### Core Objective
Polish the existing workspace to feel more seamless, remove visual clutter, fix zoom overlay alignment, and bring annotation closer to the analysis context.

### Concrete Deliverables
- Full-height workspace without top nav header
- Consistent image display sizing
- Correct overlay-to-zoom synchronization
- Cleaner history sidebar (no redundant label)
- Zoomed crop preview of selected region
- Global drag-and-drop with modal import preview
- Inline annotation panel on workspace page

### Must Have
- Header removed entirely
- Image size normalization
- Zoom overlay fix
- History label removed
- Selected region zoomed crop
- Global drag & drop
- Inline annotation panel

### Must NOT Have (Guardrails)
- No backend API redesign
- No model retraining or threshold changes
- No mobile-first overhaul
- No new heavy UI framework

---

## Verification Strategy

- **Build/lint/tsc**: After each task
- **Manual QA**: Playwright smoke tests for drag-drop, modal, zoom overlay
- **Evidence**: Screenshots to `.sisyphus/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Visual cleanup + fixes):
├── Task 1: Remove top nav header and make workspace full-height [quick]
├── Task 2: Normalize image display size regardless of original dimensions [quick]
├── Task 3: Fix overlay canvas synchronization with zoom transforms [deep]
├── Task 4: Remove "History" label from sidebar, clean sidebar chrome [quick]
└── Task 5: Add zoomed crop preview of selected region in detail panel [visual-engineering]

Wave 2 (Interaction improvements):
├── Task 6: Implement global drag-and-drop + modal import with preview [deep]
└── Task 7: Create inline annotation panel on workspace page [visual-engineering]
```

---

## TODOs

- [x] 1. Remove top nav header and make workspace full-height

  **What to do**:
  - Remove the `<nav>` element in `App.tsx` that shows "Codex Analyzer".
  - Adjust `WorkspacePage.tsx` root container to use `h-screen` instead of `h-[calc(100vh-2rem)]`.
  - Ensure no vertical scroll appears from removed header space.
  - Move any critical nav actions (if any) into the workspace chrome.

  **Must NOT do**:
  - Do not break the route structure.
  - Do not remove the `BrowserRouter` wrapper.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] No top nav bar visible on any route.
  - [ ] Workspace uses full viewport height without scroll.

  **QA Scenarios**:
  ```
  Scenario: Full-height workspace without header
    Tool: Playwright
    Steps:
      1. Open `/`
      2. Verify no nav bar at top
      3. Verify workspace fills entire viewport height
    Evidence: .sisyphus/evidence/plan-a-task-1-no-header.png
  ```

- [x] 2. Normalize image display size regardless of original dimensions

  **What to do**:
  - In `WorkspacePage.tsx`, wrap the analyzed image in a container with fixed aspect ratio or max dimensions.
  - Use `object-fit: contain` or CSS constraints to ensure all images render at the same display size.
  - The canvas overlay must scale to match the normalized image size, not the original.

  **Must NOT do**:
  - Do not resize the actual image data (keep original for analysis).
  - Do not break the overlay alignment.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] Images of different original sizes display at the same dimensions.
  - [ ] Overlay boxes still align correctly.

  **QA Scenarios**:
  ```
  Scenario: Consistent image size
    Tool: Playwright
    Steps:
      1. Analyze a small image
      2. Analyze a large image
      3. Verify both display at the same size in the workspace
    Evidence: .sisyphus/evidence/plan-a-task-2-size-normalize.png
  ```

- [x] 3. Fix overlay canvas synchronization with zoom transforms

  **What to do**:
  - Currently the canvas and image are both inside a CSS-transformed wrapper (`transform: scale(zoom)`).
  - The overlay drawing logic (`drawOverlay`) uses the image's original dimensions to scale bboxes.
  - When zoomed, the canvas coordinate space scales with the wrapper, but `drawOverlay` computes against the unscaled image size — this causes misalignment.
  - Fix: either (a) draw overlay in unscaled space and let CSS zoom handle it, or (b) compute overlay in the zoomed coordinate space.
  - Recommended approach (a): keep canvas at natural image size, draw bboxes at natural scale, and apply the same CSS `scale(zoom)` to the canvas wrapper. The canvas and image must share the exact same transform origin and scale.

  **Must NOT do**:
  - Do not change the zoom interaction model (buttons + wheel).
  - Do not break hit-testing.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] At zoom = 1, overlay boxes align perfectly.
  - [ ] At zoom = 2, overlay boxes still align perfectly.
  - [ ] Hit-testing works correctly at all zoom levels.

  **QA Scenarios**:
  ```
  Scenario: Overlay alignment at zoom levels
    Tool: Playwright
    Steps:
      1. Load a record with visible boxes
      2. Capture box positions at zoom=1
      3. Zoom to 2x
      4. Verify boxes still surround the same regions
    Evidence: .sisyphus/evidence/plan-a-task-3-zoom-align.png
  ```

- [x] 4. Remove "History" label from sidebar, clean sidebar chrome

  **What to do**:
  - Remove the "ANALYSIS HISTORY" text label from the history sidebar header.
  - Keep the item count badge (e.g., "ITEMS 3") if present.
  - Consider making the sidebar more like a ChatGPT-style slim rail when collapsed — icon-only with tooltips.
  - When expanded, show records without the header label.

  **Must NOT do**:
  - Do not remove the history functionality.
  - Do not break collapse/expand.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] No "History" or "ANALYSIS HISTORY" text in sidebar.
  - [ ] Collapsed rail is clean and minimal.

  **QA Scenarios**:
  ```
  Scenario: Clean sidebar without History label
    Tool: Playwright
    Steps:
      1. Open `/` with history items
      2. Verify no "History" label in sidebar
      3. Verify collapsed rail is icon-only
    Evidence: .sisyphus/evidence/plan-a-task-4-clean-sidebar.png
  ```

- [x] 5. Add zoomed crop preview of selected region in detail panel

  **What to do**:
  - When a region is selected (`focusedIdx`), extract the bbox coordinates.
  - Render a zoomed-in crop of that region in the detail panel (e.g., 2x or 3x zoom).
  - Use the existing crop canvas mechanism or a new `<canvas>` element.
  - Show the crop prominently so the user can clearly see what the region contains.
  - Update dynamically when `focusedIdx` changes.

  **Must NOT do**:
  - Do not add a full image editor.
  - Do not change the backend.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] Selected region shows a zoomed crop in the detail panel.
  - [ ] Crop updates when a different region is selected.
  - [ ] Crop is clearly visible and large enough to inspect details.

  **QA Scenarios**:
  ```
  Scenario: Zoomed crop of selected region
    Tool: Playwright
    Steps:
      1. Load a record
      2. Select a region
      3. Verify detail panel shows a zoomed crop
      4. Select another region
      5. Verify crop updates
    Evidence: .sisyphus/evidence/plan-a-task-5-zoomed-crop.png
  ```

- [x] 6. Implement global drag-and-drop + modal import with preview

  **What to do**:
  - Add a global `dragover`/`drop` listener on the workspace page (or a full-page overlay).
  - When a file is dragged over the page, show a visual feedback overlay.
  - On drop, open a centered modal (not the current inline dropzone) showing:
    - Image preview
    - File name, size, dimensions
    - "Analyze" button to start processing
    - "Cancel" button to close
  - The existing dropzone can be replaced by a simple "Import" button that opens the same modal.
  - The modal should handle both drag-and-drop and file picker.

  **Must NOT do**:
  - Do not change the backend upload endpoint.
  - Do not break the existing upload flow.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] Dragging a file over the page shows visual feedback.
  - [ ] Dropping opens a modal with image preview and details.
  - [ ] Clicking "Import" opens the same modal.
  - [ ] "Analyze" starts processing and closes the modal.

  **QA Scenarios**:
  ```
  Scenario: Global drag and drop with modal
    Tool: Playwright
    Steps:
      1. Open `/`
      2. Drag an image over the page
      3. Verify visual feedback appears
      4. Drop the image
      5. Verify modal opens with preview and details
      6. Click Analyze
      7. Verify processing starts and modal closes
    Evidence: .sisyphus/evidence/plan-a-task-6-drag-modal.png
  ```

- [x] 7. Create inline annotation panel on workspace page

  **What to do**:
  - Replace the "Open in full editor" button behavior.
  - Instead of navigating to `/annotate/:id`, open a panel (slide-in from right or modal) on the workspace page.
  - The panel should show:
    - The selected region's zoomed crop (reuse Task 5)
    - Class dropdown (reuse `getClasses`)
    - Suggestion buttons (top_k)
    - Save button that calls `updateAnnotations`
    - Close button
  - This keeps the user in context with the full image visible.
  - Alternatively, keep `/annotate/:id` but enhance it with zoomed crop + custom segmentation selection — user preference was "garder ça sur la page actuelle serait peut-etre mieux".

  **Must NOT do**:
  - Do not remove `/annotate/:id` route (keep as fallback).
  - Do not change backend APIs.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Acceptance Criteria**:
  - [ ] Clicking "Annotate" opens an inline panel without leaving the workspace.
  - [ ] Panel shows zoomed crop, class dropdown, suggestions, save.
  - [ ] Saving persists and updates the workspace state.
  - [ ] Closing the panel returns to normal workspace view.

  **QA Scenarios**:
  ```
  Scenario: Inline annotation panel
    Tool: Playwright
    Steps:
      1. Load a record
      2. Select a region
      3. Click "Annotate"
      4. Verify inline panel opens with crop + controls
      5. Change class and save
      6. Verify panel closes and workspace updates
    Evidence: .sisyphus/evidence/plan-a-task-7-inline-annotate.png
  ```

---

## Commit Strategy

- Commit 1: Header removal + full-height workspace
- Commit 2: Image size normalization + zoom overlay fix
- Commit 3: Sidebar cleanup + zoomed crop preview
- Commit 4: Drag-drop modal + inline annotation panel

---

## Success Criteria

### Verification Commands
```bash
cd codex-frontend && npm run build
cd codex-frontend && npm run lint
cd codex-frontend && npx tsc --noEmit
```

### Final Checklist
- [x] No top header on any route
- [x] Images display at consistent size
- [x] Overlay boxes align correctly at all zoom levels
- [x] History sidebar has no redundant label
- [x] Selected region shows zoomed crop in panel
- [x] Global drag-and-drop works with modal preview
- [x] Inline annotation panel works on workspace page
- [x] No backend/model changes

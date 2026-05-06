# Workspace Context & Trust — Layout Hardening + Advanced Features

## TL;DR

> **Quick Summary**: Two-step plan. **Step 1** hardens workspace layout (zero-scroll on 1920×1080 + 2K, image fills canvas, zoom-stable overlay, full-editor Annotate page, ChatGPT-style thumbnail-rail history sidebar). **Step 2** adds visual-similarity context and detailed model-trust transparency (research-first: R1–R5 then implementation).
>
> **Deliverables**:
> - Step 1: Zero-scroll workspace, full-bleed image, zoom-stable overlay, thumbnail-rail history, full-editor Annotate page, unified sidebar style
> - Step 2: Similar-images panel, trust-factor panel (prototype distances, entropy), prototype gallery
>
> **Estimated Effort**: XL (Step 1 = Medium, Step 2 = Large)
> **Parallel Execution**: YES — Step 1 has 2 waves of parallel tasks; Step 2 implementation parallelizes after research
> **Critical Path**: S1.5 (layout shell) → S1.1 (image fill) → S1.2 (zoom overlay) → S1.6 (zoom no-scroll) → S1.4 (Annotate editor) → R1–R5 → Step 2 waves

---

## Context

### Original Request (user feedback after Plan A)

**Layout & scaling complaints:**
- "Les images dans le modal d'import sont correctement scalées mais pas correctement lorsqu'elles apparaissent dans le workspace, elles restent toutes petites."
- "Il faut aussi fix et scale les segmentations en fonction du zoom."
- "Fais en sorte que toute la page tienne sans scroll sur du 1920x1080 et du 2K."
- "Lorsque l'utilisateur zoom/unzoom sur l'image du workspace la page ne doit pas scroll. La page ne doit jamais scroll."
- "Modifie les sidebars pour fit le style."

**History sidebar UX:**
- "Dans la sidebar il faut afficher tout l'historique même lorsqu'elle est contractée, et cliquer sur une image permettrait de directement afficher l'analyse au lieu de devoir d'abord agrandir la sidebar d'historique pour cliquer, ça fait gagner du temps."

**Annotate page:**
- "Il faut pouvoir ajouter/modifier/retirer des zones dans la page annotate. Elle doit être le plus complète possible avec des prévisualisations et une modification poussée."

**Original Plan B scope (preserved as Step 2):**
- "There should also be a way to visualize close embedded images from the analyzed image."
- "Anything that could help the user have more context, and the context of the models themselves, is welcome."
- "The trust factor of the predictions is a great factor, but we can go further."

### What We Did So Far (Recap)

**Plan `workspace-ui-redesign` (T1–T17 + F1–F4)**: COMPLETE
- Compact chrome, collapsible history, image-first canvas with zoom (0.25–4x), clickable overlay, compact region panel, inline relabel, keyboard nav, AnnotationPage v1 (relabel-only), `?element=<idx>` handoff, visual polish, dense/tiny/overlap edge cases, browser QA.
- T17 cleanup (7 atomic micro-fixes) applied and verified by Oracle + Playwright.

**Plan `workspace-ui-polish` (T1–T7)**: COMPLETE
- T1: nav header removed, `<App>` → `h-screen` root.
- T2: image container `max-w-[800px] max-h-[600px]` + `<img>` `object-contain` (⚠️ this is what's now too small — Step 1 will fix).
- T3: canvas at original-image resolution, draw in original space, CSS `scale(zoom)` on wrapper (⚠️ user reports still broken — Step 1 S1.2 will redo).
- T4: removed sidebar "History"/"Saved runs" labels.
- T5: zoomed crop preview (2.5x) of focused region in detail panel.
- T6: global drag-and-drop overlay + centered Import modal with file preview.
- T7: inline annotation slide-in panel (right side, 400px) with class dropdown, top-k suggestions, save handler.

### Current Pain Points (from this round)
1. Workspace image stays tiny (T2's `max-w-[800px]` is too restrictive on big screens).
2. Segmentation overlay still desynchronizes at non-1.0 zoom levels (T3 fix incomplete).
3. History sidebar: collapsed state hides items entirely; user must expand → click → wait. Slow.
4. Annotate page only allows relabel + delete; no add/move/resize.
5. Page scrolls on 1920×1080 (some panels overflow vertically).
6. Zooming the image makes the entire page scroll.
7. Two sidebars (left history, right region detail) have inconsistent styling.

---

## Work Objectives

### Core Objective
**Step 1**: Make the workspace feel like a desktop application — zero page scroll, image fills available canvas, overlay always aligned with zoom, history always one click away, Annotate page is a real editor.

**Step 2**: Add model-introspection features (visual similarity + detailed trust metrics) to make analysis decisions transparent.

### Concrete Deliverables

**Step 1 (Frontend-only):**
- Workspace at 1920×1080: zero scroll, image fills central area
- Workspace at 2560×1440: zero scroll, image fills central area, sidebars expand proportionally
- Image zoom (0.25–4x) never causes page scroll; overlay boxes track zoom precisely
- History sidebar: 64px collapsed rail with vertical thumbnail stack; clicking thumbnail loads analysis directly
- Annotate page: drag to create new box, drag existing box to move, drag corner handles to resize, Delete key to remove, dropdown to relabel; live zoomed crop preview per region; save persists changes
- Both sidebars use unified style tokens

**Step 2 (Research-first, then frontend + backend):**
- R1–R5 research artifacts in `.sisyphus/research/`
- New backend endpoints: `POST /similar`, `POST /trust`, `GET /prototypes`
- Workspace Visual Context panel (similar dataset images for analyzed image / selected region)
- Region-detail Trust Factor panel (distance to predicted prototype, distance to nearest competing prototype, entropy, top-k breakdown)
- Prototype gallery view

### Definition of Done

**Step 1:**
- [ ] At 1920×1080 viewport, no scrollbar visible on workspace, dashboard, or annotate routes
- [ ] At 2560×1440 viewport, no scrollbar visible; image area is visibly larger than at 1920×1080
- [ ] Image zoom 0.25x and 4x both keep page non-scrollable
- [ ] Overlay boxes match image content at zoom 0.5x, 1x, 2x, 4x (Playwright pixel diff acceptable)
- [ ] Collapsed history sidebar shows ≥5 thumbnails simultaneously; clicking one loads its analysis
- [ ] Annotate page: can draw a new box, move it, resize it, delete it, relabel it, save changes; refresh shows persisted state

**Step 2:**
- [ ] R1–R5 research artifacts written and reviewed
- [ ] Three new backend endpoints implemented and curl-tested
- [ ] Visual Context panel shows ≥3 similar images
- [ ] Trust Factor panel shows ≥4 metrics (predicted-proto distance, nearest-competing distance, entropy, top-k)
- [ ] No model retraining performed

### Must Have
- Zero scroll at both target resolutions (1920×1080, 2560×1440)
- Image fills available canvas area (not capped at 800px)
- Overlay synchronizes with zoom transforms perfectly
- Annotate page = full editor (add/move/resize/delete/relabel)
- Thumbnail-rail history sidebar (always visible)
- Step 1 completes BEFORE R1–R5 starts

### Must NOT Have (Guardrails)
- No new UI framework or DnD library (Konva, React-DnD, Fabric.js — FORBIDDEN)
- No mobile-specific code (desktop-only)
- No backend changes in Step 1
- No model retraining in Step 2
- No frontend test infrastructure (manual QA only)
- No changes to inference pipeline, segmentation params, or model thresholds
- No edits to `frontend_integration_fix/frontend_integration/` in Step 1
- No `pip install` (use `uv` for Python)
- No formatting-only churn

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO frontend test framework
- **Automated tests**: NONE (manual QA only)
- **Framework**: N/A
- **Agent-Executed QA**: Playwright (UI scenarios) + Bash (build/lint/tsc) + curl (Step 2 backend)

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/context-trust/task-{N}-{slug}.{ext}`.
- **Frontend**: Playwright at viewports 1920×1080 AND 2560×1440 — navigate, interact, assert DOM/CSS, screenshot
- **Backend (Step 2 only)**: curl POST/GET, parse JSON, assert fields and status

---

## Execution Strategy

### Parallel Execution Waves

```
STEP 1: Layout Hardening (Frontend-Only, NO Backend)

Wave 1 (Foundation — START IMMEDIATELY):
├── S1.5  Zero-scroll layout shell (h-screen + overflow-hidden) [quick]
└── S1.7  Sidebar style tokens (unified border/bg/header) [quick]

Wave 2 (Layout-Dependent — after Wave 1):
├── S1.1  Workspace image fills canvas (depends: S1.5) [visual-engineering]
├── S1.3  History thumbnail-rail sidebar (depends: S1.5, S1.7) [visual-engineering]
└── S1.6  Zoom-no-scroll containment (depends: S1.5) [deep]

Wave 3 (Image-Dependent — after Wave 2):
├── S1.2  Segmentation overlay zoom sync (depends: S1.1, S1.6) [deep]
└── S1.4  Annotate page full editor (depends: S1.5, S1.7) [visual-engineering]

Wave STEP-1-VERIFY (after all S1.*):
└── F-S1  Playwright dual-resolution + zoom + editor QA [unspecified-high]

────────────────────────────────────────

STEP 2: Context & Trust (RESEARCH-FIRST)

Wave 4 (Research — sequential per question, parallel reads):
├── R1  Embedding extraction architecture (deep)
├── R2  Prototype distance feasibility (deep)
├── R3  PoC similarity search (deep)
├── R4  API contract design (unspecified-high, depends: R1, R2, R3)
└── R5  Synthesize + replan (oracle, depends: R1–R4)

⚠️  HUMAN GATE after R5: present findings, get user okay before Wave 5.

Wave 5 (Backend — after R5 approval):
├── S2.B1  POST /similar endpoint (deep)
├── S2.B2  POST /trust endpoint (deep)
└── S2.B3  GET /prototypes endpoint (unspecified-high)

Wave 6 (Frontend — parallel with Wave 5 once R4 contracts frozen):
├── S2.F1  Visual Context panel (visual-engineering, depends: S2.B1)
├── S2.F2  Trust Factor panel (visual-engineering, depends: S2.B2)
└── S2.F3  Prototype gallery (visual-engineering, depends: S2.B3)

Wave FINAL:
├── F1  Plan compliance audit (oracle)
├── F2  Code quality review (unspecified-high)
├── F3  Real manual QA (unspecified-high + playwright)
└── F4  Scope fidelity check (deep)

Critical Path: S1.5 → S1.1 → S1.2 → S1.4 → F-S1 → R1 → R5 → S2.B1 → S2.F1 → F1–F4
Parallel Speedup: ~50% vs sequential
Max Concurrent: 3 (Wave 2, Wave 3, Wave 5+6 overlap)
```

### Dependency Matrix (abbreviated)

- **S1.5, S1.7**: no deps → unblock S1.1, S1.3, S1.6
- **S1.1**: S1.5 → unblocks S1.2
- **S1.6**: S1.5 → unblocks S1.2
- **S1.2**: S1.1, S1.6 → unblocks F-S1
- **S1.3**: S1.5, S1.7 → unblocks F-S1
- **S1.4**: S1.5, S1.7 → unblocks F-S1
- **F-S1**: all S1.* → unblocks R1
- **R1, R2, R3**: F-S1 → unblock R4
- **R4**: R1, R2, R3 → unblocks R5
- **R5**: R1–R4 → HUMAN GATE → unblocks S2.B*, S2.F*
- **S2.B1**: R5 → unblocks S2.F1
- **S2.B2**: R5 → unblocks S2.F2
- **S2.B3**: R5 → unblocks S2.F3
- **F1–F4**: all S2.* → final

### Agent Dispatch Summary

- **Wave 1**: 2 — S1.5 → quick, S1.7 → quick
- **Wave 2**: 3 — S1.1 → visual-engineering, S1.3 → visual-engineering, S1.6 → deep
- **Wave 3**: 2 — S1.2 → deep, S1.4 → visual-engineering
- **Wave STEP-1-VERIFY**: 1 — F-S1 → unspecified-high (+ playwright)
- **Wave 4**: 5 — R1, R2, R3 → deep; R4 → unspecified-high; R5 → oracle
- **Wave 5**: 3 — S2.B1, S2.B2 → deep; S2.B3 → unspecified-high
- **Wave 6**: 3 — S2.F1, S2.F2, S2.F3 → visual-engineering
- **Wave FINAL**: 4 — F1 → oracle, F2 → unspecified-high, F3 → unspecified-high (+ playwright), F4 → deep

---

## TODOs

### STEP 1 — Layout Hardening (Frontend Only)

- [x] S1.5. Zero-scroll layout shell

  **What to do**:
  - In `codex-frontend/src/index.css`, add `html, body, #root { height: 100%; overflow: hidden; }` to top of file (or merge into existing reset).
  - In `codex-frontend/src/App.tsx`, ensure root wrapper is `<div className="h-screen w-screen overflow-hidden flex flex-col bg-stone-950 text-stone-100">`.
  - In `codex-frontend/src/pages/WorkspacePage.tsx`, ensure root is `<div className="flex h-full w-full overflow-hidden">` with three flex children: history sidebar, main canvas area, region detail panel — all with `overflow-hidden` and internal scroll only where needed.
  - In `codex-frontend/src/pages/AnnotationPage.tsx`, same `h-full w-full overflow-hidden flex` shell.
  - Audit all `min-h-screen`, `min-h-full`, and large fixed `min-h-*` values in changed files — replace with `h-full` where they cause overflow.

  **Must NOT do**:
  - No edits to backend
  - No new CSS framework or utility lib
  - No refactor of unrelated layout (only changes needed for zero-scroll)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical CSS/className changes, no design judgment.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.7)
  - **Parallel Group**: Wave 1
  - **Blocks**: S1.1, S1.3, S1.6
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/index.css` — global styles entry point
  - `codex-frontend/src/App.tsx` — root layout wrapper
  - `codex-frontend/src/pages/WorkspacePage.tsx:1-50` — current workspace shell
  - `codex-frontend/src/pages/AnnotationPage.tsx:1-50` — current annotate shell
  - Tailwind docs: `https://tailwindcss.com/docs/overflow` — `overflow-hidden`, `overflow-y-auto`

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] `npm run lint` passes
  - [ ] `npx tsc --noEmit` passes
  - [ ] Playwright @ 1920×1080 on `/`: `document.documentElement.scrollHeight === window.innerHeight` (no vertical overflow)
  - [ ] Playwright @ 2560×1440 on `/`: same assertion
  - [ ] Playwright on `/annotate/<seeded-id>`: same assertion at both resolutions

  **QA Scenarios**:
  ```
  Scenario: Zero scroll at 1920x1080 on workspace
    Tool: Playwright
    Preconditions: Backend running, frontend at localhost:5173
    Steps:
      1. Set viewport to 1920x1080
      2. Navigate to http://localhost:5173/
      3. Wait for `.workspace-root` (or equivalent) to be visible
      4. Evaluate: document.documentElement.scrollHeight, window.innerHeight, document.documentElement.scrollWidth, window.innerWidth
      5. Assert scrollHeight <= innerHeight AND scrollWidth <= innerWidth
    Expected Result: Both dimensions fit; no scrollbar present
    Failure Indicators: scrollHeight > innerHeight, visible scrollbar
    Evidence: .sisyphus/evidence/context-trust/task-S1.5-zero-scroll-1920.png

  Scenario: Zero scroll at 2560x1440 on workspace
    Tool: Playwright
    Preconditions: Same as above
    Steps:
      1. Set viewport to 2560x1440
      2. Navigate to http://localhost:5173/
      3. Same assertion as above
    Expected Result: No scrollbar; image area visibly larger
    Evidence: .sisyphus/evidence/context-trust/task-S1.5-zero-scroll-2560.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.5-zero-scroll-1920.png
  - [ ] task-S1.5-zero-scroll-2560.png

  **Commit**: NO (groups with S1.7 → Commit 1)

- [x] S1.7. Unified sidebar style tokens

  **What to do**:
  - Define shared CSS classes (or Tailwind component pattern) for sidebar sections in `codex-frontend/src/index.css`:
    - `.sidebar-shell` — `bg-stone-950 border-stone-800 flex flex-col`
    - `.sidebar-header` — `px-3 py-2 border-b border-stone-800 text-xs uppercase tracking-wider text-stone-400`
    - `.sidebar-body` — `flex-1 overflow-y-auto`
  - Apply these classes to:
    - History sidebar in `WorkspacePage.tsx` (left side)
    - Region detail panel in `WorkspacePage.tsx` (right side)
    - Annotation panel in `WorkspacePage.tsx` (T7 slide-in panel — match style)
    - Annotation page sidebars (if any) in `AnnotationPage.tsx`
  - Verify all sidebars now share identical border color, background, and header treatment.

  **Must NOT do**:
  - No content changes (only style classes)
  - No layout/structural changes (S1.5 owns layout)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical class application, no design choices (tokens given).
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.5)
  - **Parallel Group**: Wave 1
  - **Blocks**: S1.3, S1.4
  - **Blocked By**: None

  **References**:
  - `codex-frontend/src/index.css` — add new utility classes here
  - `codex-frontend/src/pages/WorkspacePage.tsx` — apply to history sidebar (~line 900+) and region panel
  - Existing color palette: `bg-stone-950`, `border-stone-800`, `text-stone-100`, `text-stone-400`

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] grep `bg-stone-950` AND `border-stone-800` appears on every sidebar wrapper
  - [ ] Visual diff Playwright: history sidebar left edge color == region panel right edge color

  **QA Scenarios**:
  ```
  Scenario: Sidebar color consistency
    Tool: Playwright
    Preconditions: Frontend running
    Steps:
      1. Navigate to http://localhost:5173/
      2. Get computed style of history sidebar's right border: getComputedStyle(historySidebar).borderRightColor
      3. Get computed style of region panel's left border: getComputedStyle(regionPanel).borderLeftColor
      4. Assert both equal (both should be stone-800)
      5. Get bg color of both — assert both equal stone-950
    Expected Result: Identical border + bg colors
    Evidence: .sisyphus/evidence/context-trust/task-S1.7-sidebar-style.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.7-sidebar-style.png

  **Commit**: YES (Commit 1: groups with S1.5)
  - Message: `feat(workspace): zero-scroll layout shell + unified sidebar tokens`
  - Files: `codex-frontend/src/index.css`, `codex-frontend/src/App.tsx`, `codex-frontend/src/pages/WorkspacePage.tsx`, `codex-frontend/src/pages/AnnotationPage.tsx`
  - Pre-commit: `cd codex-frontend && npm run build && npm run lint`

- [x] S1.1. Workspace image fills available canvas

  **What to do**:
  - Locate the image container in `WorkspacePage.tsx` (Plan A T2 set `max-w-[800px] max-h-[600px]` on the wrapper and `object-contain` on `<img>`).
  - Replace the wrapper sizing with: `flex-1 flex items-center justify-center overflow-hidden p-4` (container fills the central flex space).
  - On the `<img>` itself: `max-w-full max-h-full w-auto h-auto object-contain` so it scales to fill the container while preserving aspect ratio.
  - Remove `max-w-[800px]` and `max-h-[600px]` constraints entirely.
  - Ensure the canvas overlay element (positioned absolute over `<img>`) has matching dimensions: same `max-w-full max-h-full` strategy and is wrapped in a sized parent that uses the rendered image bounding rect via `imageRef.current.getBoundingClientRect()` for canvas placement.
  - Verify on small screens (1280×720) it still doesn't overflow (S1.5 handles this via flex shrink).

  **Must NOT do**:
  - No fixed pixel dimensions on image container
  - No removal of zoom logic
  - No change to overlay drawing logic itself (S1.2 owns that)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Visual sizing decisions, image-fit balance, must look right at multiple resolutions.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.3, S1.6)
  - **Parallel Group**: Wave 2
  - **Blocks**: S1.2
  - **Blocked By**: S1.5

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx` — image container (search for `max-w-[800px]` from Plan A T2)
  - `codex-frontend/src/pages/WorkspacePage.tsx:imageRef` — image ref used for canvas sync
  - Tailwind: `https://tailwindcss.com/docs/object-fit` — `object-contain` preserves aspect ratio

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] No occurrence of `max-w-[800px]` or `max-h-[600px]` on workspace image
  - [ ] Playwright @ 1920×1080: image rendered width > 1000px (was ~800px before)
  - [ ] Playwright @ 2560×1440: image rendered width > 1400px

  **QA Scenarios**:
  ```
  Scenario: Image fills canvas at 1920x1080
    Tool: Playwright
    Preconditions: Backend running, sample image uploaded
    Steps:
      1. Set viewport 1920x1080
      2. Navigate to /, upload sample image, wait for analysis
      3. Get rendered image bounding rect: imageRef.getBoundingClientRect()
      4. Assert rect.width >= 1000 AND rect.height >= 600
      5. Assert image is centered (rect.left > 200 AND rect.right < viewport.width - 200)
    Expected Result: Image is significantly larger than 800px wide
    Evidence: .sisyphus/evidence/context-trust/task-S1.1-image-fill-1920.png

  Scenario: Image fills canvas at 2560x1440
    Tool: Playwright
    Preconditions: Same
    Steps:
      1. Set viewport 2560x1440
      2. Same flow
      3. Assert rect.width >= 1400
    Evidence: .sisyphus/evidence/context-trust/task-S1.1-image-fill-2560.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.1-image-fill-1920.png
  - [ ] task-S1.1-image-fill-2560.png

  **Commit**: NO (groups with S1.6 → Commit 2)

- [x] S1.6. Zoom never triggers page scroll

  **What to do**:
  - Wrap the image+canvas+zoom-transform area in an `overflow-hidden` container (the same container from S1.1 should already be `overflow-hidden`).
  - The zoom transform must apply to a child element: `<div style={{ transform: \`scale(\${zoom})\`, transformOrigin: 'center center' }}>`.
  - When zoom > 1, the scaled child overflows its parent — the parent's `overflow-hidden` must clip it (no propagation up to body/html which already have `overflow: hidden` from S1.5).
  - Add CSS `will-change: transform` on the zoomed child for smooth performance.
  - Test at zoom 0.25, 1, 2, 4 — at each level, verify `document.documentElement.scrollWidth === window.innerWidth` and `scrollHeight === window.innerHeight`.

  **Must NOT do**:
  - No removal of zoom feature
  - No change to zoom range (keep 0.25–4)
  - No change to overlay logic (S1.2 owns)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CSS containment + transform interaction needs careful reasoning about layout vs paint.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.1, S1.3)
  - **Parallel Group**: Wave 2
  - **Blocks**: S1.2
  - **Blocked By**: S1.5

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx` — `zoom` state and CSS transform application
  - MDN: `https://developer.mozilla.org/en-US/docs/Web/CSS/transform` — transforms don't trigger reflow but can cause visual overflow
  - MDN: `https://developer.mozilla.org/en-US/docs/Web/CSS/overflow` — `overflow: hidden` on parent clips transformed children

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] At zoom 0.25, 1, 2, 4 → no page scroll
  - [ ] Image visually scales (transform applied to inner child, not outer container)
  - [ ] Overlay positions still correct (S1.2 will further verify)

  **QA Scenarios**:
  ```
  Scenario: Zoom 4x doesn't scroll page
    Tool: Playwright
    Preconditions: Image loaded at zoom 1.0
    Steps:
      1. Set viewport 1920x1080
      2. Click zoom-in button 6 times (1.0 → 4.0 in 0.5 steps, or use direct setter)
      3. Wait 500ms for transform
      4. Evaluate: document.documentElement.scrollWidth === window.innerWidth
      5. Evaluate: document.documentElement.scrollHeight === window.innerHeight
    Expected Result: Both true; image is visibly zoomed but page does not scroll
    Evidence: .sisyphus/evidence/context-trust/task-S1.6-zoom-4x.png

  Scenario: Zoom 0.25x doesn't scroll page
    Tool: Playwright
    Steps:
      1. Click zoom-out to reach 0.25
      2. Same assertions
    Evidence: .sisyphus/evidence/context-trust/task-S1.6-zoom-025x.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.6-zoom-4x.png
  - [ ] task-S1.6-zoom-025x.png

  **Commit**: YES (Commit 2: groups with S1.1)
  - Message: `feat(workspace): full-bleed image with zoom-safe containment`
  - Files: `codex-frontend/src/pages/WorkspacePage.tsx`
  - Pre-commit: `cd codex-frontend && npm run build && npm run lint`

- [x] S1.3. History thumbnail-rail sidebar (always-visible)

  **What to do**:
  - Refactor history sidebar in `WorkspacePage.tsx` to have two visual modes controlled by `historyOpen`:
    - **Collapsed (default, ~64px wide)**: vertical stack of thumbnail buttons. Each thumbnail is 48×48px, shows the analyzed image (use `record.imageDataUrl`), `border-stone-700`. Hover → tooltip with `record.imageName` + timestamp. Click → directly call `selectRecord(record)` (no need to expand first).
    - **Expanded (~280px wide)**: current expanded view with thumbnails + filename + timestamp + delete button.
  - Toggle button at top of sidebar to switch between modes (chevron icon).
  - Preserve all existing functionality: select, delete, expand-on-detail.
  - Apply `.sidebar-shell`, `.sidebar-header`, `.sidebar-body` classes from S1.7.
  - Ensure body uses `overflow-y-auto` for scrolling when many records.

  **Must NOT do**:
  - No removal of expand/collapse toggle
  - No change to record data model
  - No change to `selectRecord` logic itself
  - No more than 64px in collapsed mode

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Visual decisions on thumbnail size, spacing, hover state, tooltip positioning.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.1, S1.6)
  - **Parallel Group**: Wave 2
  - **Blocks**: F-S1
  - **Blocked By**: S1.5, S1.7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx` — current history sidebar (search for `historyOpen`, `selectRecord`)
  - `codex-frontend/src/types/index.ts` — `AnalysisRecord` shape (`id`, `imageName`, `imageDataUrl`, `timestamp`, `result`, `annotations`)
  - ChatGPT sidebar reference: thin rail with stacked items, click-to-load
  - Tailwind tooltip pattern: `group` + `group-hover:opacity-100` + absolute positioned tooltip

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] Collapsed sidebar width ≤ 80px (target 64px)
  - [ ] Collapsed sidebar shows all records as thumbnails
  - [ ] Click on collapsed thumbnail loads analysis (without expanding sidebar)
  - [ ] Hover shows tooltip with filename
  - [ ] Toggle button switches between modes

  **QA Scenarios**:
  ```
  Scenario: Click thumbnail in collapsed mode loads analysis
    Tool: Playwright
    Preconditions: ≥2 records in history, sidebar collapsed
    Steps:
      1. Confirm sidebar width ≤ 80px (getBoundingClientRect)
      2. Count visible thumbnail buttons — assert ≥2
      3. Click second thumbnail
      4. Assert main canvas now shows that record's image (compare imageRef.src)
      5. Assert sidebar still collapsed (didn't auto-expand)
    Expected Result: Direct load, no expansion required
    Evidence: .sisyphus/evidence/context-trust/task-S1.3-collapsed-click.png

  Scenario: Tooltip shows filename on hover
    Tool: Playwright
    Steps:
      1. Hover over a thumbnail
      2. Wait for tooltip (200ms)
      3. Assert tooltip element visible and contains record.imageName
    Evidence: .sisyphus/evidence/context-trust/task-S1.3-tooltip.png

  Scenario: Many records — sidebar scrolls internally
    Tool: Playwright
    Preconditions: Seed 20 records
    Steps:
      1. Confirm sidebar overflow-y is auto/scroll
      2. Scroll inside sidebar — assert page itself does not scroll
    Evidence: .sisyphus/evidence/context-trust/task-S1.3-internal-scroll.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.3-collapsed-click.png
  - [ ] task-S1.3-tooltip.png
  - [ ] task-S1.3-internal-scroll.png

  **Commit**: YES (Commit 4)
  - Message: `feat(workspace): thumbnail-rail history sidebar`
  - Files: `codex-frontend/src/pages/WorkspacePage.tsx`
  - Pre-commit: `cd codex-frontend && npm run build && npm run lint`

- [x] S1.2. Segmentation overlay zoom synchronization (re-fix)

  **What to do**:
  - **Discard Plan A T3 approach if it doesn't work cleanly with new image-fill (S1.1).** Treat this as fresh.
  - The canvas overlay must always exactly cover the rendered `<img>` regardless of zoom/object-fit:
    - Use a `useLayoutEffect` that reads `imageRef.current.getBoundingClientRect()` (which already accounts for `object-contain` letterboxing) and sets canvas inline `style.width/height` and `width/height` attributes accordingly.
    - Re-run on: window resize, image load, zoom change, focusedIdx change.
    - When zoom is applied via parent CSS transform (S1.6), the canvas inside the transformed wrapper is also transformed — so canvas dimensions can stay in image-original space, and bbox draw coords stay in original space too. Hit-testing converts client coords back to original space via the same transform inverse.
  - Verify alignment by drawing a test pattern at zoom 0.25, 1, 2, 4 and pixel-comparing overlay edges to known image features.
  - Add a small debug aid (only in dev): when `?debug-overlay=1` query param, draw a 2px red border around each bbox and a crosshair at center.

  **Must NOT do**:
  - No change to bbox data format (`[x, y, w, h]` in original image coords)
  - No change to zoom range
  - No edits to image fill logic (S1.1 owns)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Coordinate-system math, requires reasoning about CSS transforms vs canvas pixel space.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (must run after S1.1 + S1.6 land)
  - **Parallel Group**: Wave 3 (with S1.4)
  - **Blocks**: F-S1
  - **Blocked By**: S1.1, S1.6

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx:drawOverlay` — current overlay drawing logic (Plan A T3)
  - `codex-frontend/src/pages/WorkspacePage.tsx:getElementIndexFromCanvasPoint` — hit-testing logic
  - MDN: `https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement` — width/height attr vs CSS size distinction
  - MDN: `https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect` — accounts for transforms

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] At zoom 0.5x, 1x, 2x, 4x: overlay bbox edges align with image features (visual Playwright check)
  - [ ] Hit-testing accurate at all zoom levels (click on visible bbox triggers selection)
  - [ ] No console errors during zoom/pan

  **QA Scenarios**:
  ```
  Scenario: Overlay aligned at multiple zoom levels
    Tool: Playwright
    Preconditions: Sample image with known bbox at, say, x=100,y=200,w=50,h=80 in original coords
    Steps:
      1. Load image at zoom 1.0
      2. Screenshot overlay region — save baseline
      3. Set zoom 2.0, take screenshot — verify overlay scaled proportionally with image
      4. Set zoom 0.5, take screenshot — verify overlay scaled down too
      5. Set zoom 4.0, take screenshot
      6. For each zoom level, click at expected on-screen bbox center → assert correct element index selected
    Expected Result: Visual alignment + correct hit-testing at every zoom level
    Evidence: .sisyphus/evidence/context-trust/task-S1.2-overlay-zoom-{05,10,20,40}.png

  Scenario: Hit-testing at zoom 4x
    Tool: Playwright
    Steps:
      1. Zoom to 4x
      2. Calculate on-screen position of bbox[0] center using current transform
      3. Click that position
      4. Assert focusedIdx === 0
    Evidence: .sisyphus/evidence/context-trust/task-S1.2-hit-test-4x.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.2-overlay-zoom-05.png, -10.png, -20.png, -40.png
  - [ ] task-S1.2-hit-test-4x.png

  **Commit**: YES (Commit 3)
  - Message: `feat(workspace): zoom-synced segmentation overlay`
  - Files: `codex-frontend/src/pages/WorkspacePage.tsx`
  - Pre-commit: `cd codex-frontend && npm run build && npm run lint`

- [x] S1.4. Annotate page = full editor (add/move/resize/delete/relabel + live crop preview)

  **What to do**:
  - In `codex-frontend/src/pages/AnnotationPage.tsx`, build a real region editor on top of the existing image+canvas:
    - **Add new box**: When a "Draw mode" toggle is ON, mousedown on canvas starts a drag; mousemove draws a rubber-band rectangle; mouseup commits a new bbox `[x, y, w, h]` in original image coords. Push to `record.result.elements` with default `class_name="unknown"`, `confidence=1.0`, `top_k=[]`, `rejected=false`.
    - **Move existing box**: When NOT in draw mode, mousedown inside an existing bbox starts a move drag; mousemove updates that bbox's `x, y` (clamped to image bounds); mouseup commits.
    - **Resize existing box**: Render 4 corner handles (8×8px squares) on the focused bbox; mousedown on a handle starts a resize drag affecting that corner; mousemove updates `w, h` (and `x, y` for top-left/bottom-left handles); mouseup commits.
    - **Delete**: When a bbox is focused (clicked), pressing `Delete` or `Backspace` removes it from `record.result.elements`.
    - **Relabel**: existing dropdown stays — extended to use updated element list.
    - **Live zoomed crop preview**: For the focused bbox, render a 200×200px canvas in the side panel showing a 3x crop of that region (similar to Plan A T5 in workspace).
  - All edits update local state; "Save" button persists via `updateAnnotations(recordId, annotations)` AND also updates `record.result.elements` in storage (extend `storage.ts` if needed to persist the elements array, not just annotations).
  - Render bboxes with: focused = blue, others = stone-500. Handles only on focused.
  - Toolbar at top: [Draw mode toggle] [Save] [Cancel/back] [zoom -/+] [reset].

  **Must NOT do**:
  - No new dependency (no React-DnD, Konva, Fabric — raw mouse events only)
  - No changes to backend API
  - No changes to `DetectedElement` type signature (extend storage if needed for persistence, but type stays compatible)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Heavy interaction design, mouse event choreography, visual handles, this is the central UX surface.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S1.2)
  - **Parallel Group**: Wave 3
  - **Blocks**: F-S1
  - **Blocked By**: S1.5, S1.7

  **References**:
  - `codex-frontend/src/pages/AnnotationPage.tsx` — current relabel-only editor
  - `codex-frontend/src/types/index.ts` — `DetectedElement { bbox, class_name, confidence, rejected, top_k }`
  - `codex-frontend/src/services/storage.ts` — `updateAnnotations`, `getAnalysisById` (may need new `updateElements` function)
  - `codex-frontend/src/pages/WorkspacePage.tsx:zoomedCropRef` — pattern for crop canvas (Plan A T5)
  - MDN MouseEvent: `https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent` — `clientX/Y`, `offsetX/Y`
  - Pattern: rubber-band selection — track mousedown origin, mousemove current, draw rect (min, max, abs)

  **Acceptance Criteria**:
  - [ ] `npm run build` passes
  - [ ] Draw mode: can create new bbox by drag
  - [ ] Move: drag existing bbox repositions it
  - [ ] Resize: drag corner handle resizes bbox
  - [ ] Delete: focused bbox + Delete key removes it
  - [ ] Relabel: dropdown changes class_name
  - [ ] Save persists changes; reload shows persisted state
  - [ ] Live crop preview updates when focusing different bbox

  **QA Scenarios**:
  ```
  Scenario: Draw new bbox
    Tool: Playwright
    Preconditions: On /annotate/<id>, draw mode ON
    Steps:
      1. Note initial element count: N
      2. Mousedown at (200, 300) on canvas
      3. Mousemove to (350, 450)
      4. Mouseup
      5. Wait 200ms
      6. Assert element count is N+1
      7. Assert new element bbox approximately [200, 300, 150, 150] (in image-original coords after transform inverse)
    Evidence: .sisyphus/evidence/context-trust/task-S1.4-draw-new.png

  Scenario: Resize existing bbox via corner handle
    Tool: Playwright
    Steps:
      1. Click an existing bbox to focus it
      2. Locate bottom-right handle position
      3. Mousedown on handle
      4. Mousemove +30px right, +30px down
      5. Mouseup
      6. Assert bbox w and h both increased by ~30
    Evidence: .sisyphus/evidence/context-trust/task-S1.4-resize.png

  Scenario: Delete focused bbox
    Tool: Playwright
    Steps:
      1. Note element count: N
      2. Click bbox to focus
      3. Press Delete key
      4. Assert element count is N-1
    Evidence: .sisyphus/evidence/context-trust/task-S1.4-delete.png

  Scenario: Save persists changes
    Tool: Playwright
    Steps:
      1. Make changes (draw new + delete one)
      2. Click Save
      3. Reload page
      4. Assert changes persisted (count + classes match)
    Evidence: .sisyphus/evidence/context-trust/task-S1.4-persist.png

  Scenario: Live crop preview updates on focus
    Tool: Playwright
    Steps:
      1. Click bbox A → assert crop canvas pixels match region A (sample 5 pixels)
      2. Click bbox B → assert crop canvas pixels changed and now match region B
    Evidence: .sisyphus/evidence/context-trust/task-S1.4-crop-preview.png
  ```

  **Evidence to Capture**:
  - [ ] task-S1.4-draw-new.png, -resize.png, -delete.png, -persist.png, -crop-preview.png

  **Commit**: YES (Commit 5)
  - Message: `feat(annotate): full editor with add/move/resize/delete/relabel`
  - Files: `codex-frontend/src/pages/AnnotationPage.tsx`, `codex-frontend/src/services/storage.ts` (if extended)
  - Pre-commit: `cd codex-frontend && npm run build && npm run lint && npx tsc --noEmit`

- [x] F-S1. Step 1 verification gate (Playwright dual-resolution + integration QA)

  **What to do**:
  - Run all S1.* QA scenarios end-to-end at BOTH 1920×1080 AND 2560×1440.
  - Test cross-task interactions:
    - Zoom to 4x → click thumbnail in collapsed sidebar → assert no scroll, image swapped, zoom reset
    - Annotate page: draw new bbox → save → return to workspace → assert new bbox visible in overlay
    - Resize browser between 1920×1080 and 2560×1440 → assert layout adapts without reload, no scroll
  - Save all evidence to `.sisyphus/evidence/context-trust/step1-verify/`.
  - Block Step 2 kickoff until ALL pass.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-task integration QA requires breadth + judgment.
  - **Skills**: `[playwright]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (gates Step 2)
  - **Blocks**: R1
  - **Blocked By**: S1.1, S1.2, S1.3, S1.4, S1.5, S1.6, S1.7

  **Acceptance Criteria**:
  - [ ] All S1.* scenarios PASS at 1920×1080
  - [ ] All S1.* scenarios PASS at 2560×1440
  - [ ] Cross-task scenarios PASS
  - [ ] No regressions in Plan A features (drag-drop modal, inline panel, focused-region crop)
  - [ ] Evidence saved to `.sisyphus/evidence/context-trust/step1-verify/`

  **Commit**: NO (verification only)

---

### Step 2 — Visual Context + Trust Factor (research-first)

- [x] R1. Research: visual similarity / nearest-neighbor for CV embeddings

  **What to do**:
  - Survey approaches to visualize "close embedded images" given an analyzed image's regions and the model's embedding space.
  - Evaluate: cosine k-NN over CLIP embeddings, FAISS index, dataset prototypes, contrastive embedding viz.
  - Identify what the existing model in `frontend_integration_fix/` exposes (embeddings? logits? feature maps?). DO NOT modify model.
  - Output a written report `.sisyphus/research/context-trust/R1-similarity.md` with: 3 candidate approaches, pros/cons, integration cost, recommended pick.

  **Must NOT do**:
  - Modify model weights, training, or thresholds.
  - Implement code (research only).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-domain research synthesis (CV + UX + integration constraints).
  - **Skills**: `[]` (web search built-in via librarian sub-delegation)

  **Parallelization**:
  - **Can Run In Parallel**: NO (R1 first to set similarity strategy that R2/R3 build on)
  - **Blocks**: R2, R5
  - **Blocked By**: F-S1

  **References**:
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py` — current API surface
  - `frontend_integration_fix/frontend_integration/` (full module) — what model exposes

  **Acceptance Criteria**:
  - [ ] `.sisyphus/research/context-trust/R1-similarity.md` exists
  - [ ] Documents 3+ candidate approaches with concrete pros/cons
  - [ ] Identifies what model currently exposes (embeddings or not)
  - [ ] Recommends ONE approach with justification

  **QA Scenarios**: N/A (research artifact)
  **Evidence**: `.sisyphus/research/context-trust/R1-similarity.md`
  **Commit**: YES (Commit 6) — `docs(research): R1 visual similarity strategy`

- [x] R2. Research: model trust / confidence visualization patterns

  **What to do**:
  - Survey state-of-the-art for surfacing model trust to expert users beyond a single confidence number.
  - Evaluate: top-k probabilities, calibration curves, per-region uncertainty, OOD detection signals, gradient-based saliency, feature contribution heatmaps.
  - Map each to data the current model exposes (`top_k` already present in `DetectedElement`).
  - Output `.sisyphus/research/context-trust/R2-trust.md` with: 4+ trust signals, data requirements per signal, UX patterns from medical/scientific imaging tools.

  **Must NOT do**:
  - Propose changes that require model retraining.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with R3, R4)
  - **Blocks**: R5
  - **Blocked By**: R1

  **References**:
  - `codex-frontend/src/types/index.ts` — `DetectedElement.top_k`, `confidence`
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py`

  **Acceptance Criteria**:
  - [ ] `.sisyphus/research/context-trust/R2-trust.md` exists
  - [ ] 4+ distinct trust signals documented
  - [ ] Each signal mapped to existing-or-required model output
  - [ ] References at least 2 medical/scientific imaging UX precedents

  **QA Scenarios**: N/A
  **Evidence**: `.sisyphus/research/context-trust/R2-trust.md`
  **Commit**: YES (Commit 7) — `docs(research): R2 trust visualization patterns`

- [x] R3. Research: prototype/exemplar gallery patterns for class context

  **What to do**:
  - Survey UX patterns for "show the user representative training examples for predicted class X".
  - Evaluate: per-class prototype grids, hard-negatives, confusion-class examples, distance-ranked exemplars.
  - Determine what's available from current model/dataset (read `frontend_integration_fix/` for any prototype indices, dataset paths, class catalogs).
  - Output `.sisyphus/research/context-trust/R3-prototypes.md`.

  **Must NOT do**:
  - Re-train or re-embed datasets.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with R2, R4)
  - **Blocks**: R5
  - **Blocked By**: R1

  **References**:
  - `frontend_integration_fix/frontend_integration/`

  **Acceptance Criteria**:
  - [ ] `.sisyphus/research/context-trust/R3-prototypes.md` exists
  - [ ] Documents 3+ gallery patterns
  - [ ] Lists what dataset/prototype assets are reachable from current backend

  **QA Scenarios**: N/A
  **Evidence**: `.sisyphus/research/context-trust/R3-prototypes.md`
  **Commit**: YES (Commit 8) — `docs(research): R3 prototype gallery patterns`

- [x] R4. Research: backend API extension surface (read-only audit)

  **What to do**:
  - Audit `frontend_integration_fix/frontend_integration/` for: existing endpoints, available model methods, embedding extraction feasibility, response shapes.
  - Document what NEW endpoints would be needed for R1+R2+R3 picks: signature, request/response JSON, expected latency.
  - Output `.sisyphus/research/context-trust/R4-api.md` with proposed `/similar`, `/trust`, `/prototypes` contracts (request/response shapes).

  **Must NOT do**:
  - Modify backend code (research only).
  - Touch model parameters or thresholds.

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires reading Python backend carefully to design clean contracts.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with R2, R3)
  - **Blocks**: R5
  - **Blocked By**: R1

  **References**:
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py`
  - Full `frontend_integration_fix/frontend_integration/` directory

  **Acceptance Criteria**:
  - [ ] `.sisyphus/research/context-trust/R4-api.md` exists
  - [ ] Proposes 3 endpoint contracts (`/similar`, `/trust`, `/prototypes`) with request/response JSON
  - [ ] Identifies any model gaps blocking implementation
  - [ ] Latency estimate per endpoint

  **QA Scenarios**: N/A
  **Evidence**: `.sisyphus/research/context-trust/R4-api.md`
  **Commit**: YES (Commit 9) — `docs(research): R4 API contracts`

- [x] R5. Research synthesis + HUMAN GATE

  **What to do**:
  - Synthesize R1–R4 into ONE document `.sisyphus/research/context-trust/R5-synthesis.md`.
  - Recommend final scope for Step 2 implementation: which signals ship, which deferred.
  - List implementation tasks for Step 2 with effort estimates.
  - **STOP** and present to user. Wait for explicit approval before S2.B1 begins.

  **Must NOT do**:
  - Start any implementation without user approval.

  **Recommended Agent Profile**:
  - **Category**: `oracle`
    - Reason: High-stakes synthesis decision; needs strongest reasoning.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (HUMAN GATE)
  - **Blocks**: S2.B1, S2.B2, S2.B3, S2.F1, S2.F2, S2.F3
  - **Blocked By**: R1, R2, R3, R4

  **Acceptance Criteria**:
  - [ ] `.sisyphus/research/context-trust/R5-synthesis.md` exists
  - [ ] Final Step 2 scope explicitly defined (IN/OUT)
  - [ ] User has explicitly approved scope (recorded in plan or follow-up)
  - [ ] Implementation task list with effort estimates

  **QA Scenarios**: N/A
  **Evidence**: `.sisyphus/research/context-trust/R5-synthesis.md` + user approval
  **Commit**: YES (Commit 10) — `docs(research): R5 synthesis + approved scope`

---

- [ ] S2.B1. Backend `/similar` endpoint

  **What to do**:
  - Implement `/similar` per R4 contract in `frontend_integration_fix/frontend_integration/examples/flask_api.py`.
  - Accept `{ image_b64, region_bbox }`, return `{ neighbors: [{ image_url, distance, class, source }] }`.
  - Use embedding/index strategy from R5.
  - Use `uv` for any Python deps. NEVER `pip`.

  **Must NOT do**:
  - Modify model weights or thresholds.
  - Use `pip` (use `uv add` only).

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.B2, S2.B3)
  - **Blocks**: S2.F1
  - **Blocked By**: R5

  **References**:
  - `.sisyphus/research/context-trust/R4-api.md` (contract)
  - `.sisyphus/research/context-trust/R5-synthesis.md` (final scope)
  - `frontend_integration_fix/frontend_integration/examples/flask_api.py`

  **Acceptance Criteria**:
  - [ ] `/similar` endpoint live; `curl -X POST localhost:5000/similar -d '...'` returns valid JSON matching R4 contract
  - [ ] Latency < target from R4
  - [ ] Backend launches via `cd frontend_integration_fix/frontend_integration && uv run examples/flask_api.py`

  **QA Scenarios**:
  ```
  Scenario: Similar returns k=5 neighbors for valid region
    Tool: Bash (curl)
    Steps:
      1. uv run examples/flask_api.py & ; sleep 3
      2. curl -X POST localhost:5000/similar -H 'Content-Type: application/json' -d @.sisyphus/evidence/context-trust/fixtures/similar-req.json
      3. Assert response has neighbors array of length 5
    Evidence: .sisyphus/evidence/context-trust/task-S2B1-similar-happy.json

  Scenario: Similar errors gracefully on bad bbox
    Tool: Bash (curl)
    Steps:
      1. curl -X POST localhost:5000/similar -d '{"image_b64":"...","region_bbox":[-1,-1,0,0]}'
      2. Assert 400 status with error message
    Evidence: .sisyphus/evidence/context-trust/task-S2B1-similar-error.json
  ```

  **Commit**: YES (Commit 11) — `feat(api): /similar endpoint`

- [ ] S2.B2. Backend `/trust` endpoint

  **What to do**:
  - Implement `/trust` per R4 contract.
  - Accept `{ image_b64, region_bbox, predicted_class }`, return trust signals (top-k probs, calibration, OOD score, etc. per R5 scope).

  **Must NOT do**:
  - Modify model weights or thresholds.
  - Use `pip`.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.B1, S2.B3)
  - **Blocks**: S2.F2
  - **Blocked By**: R5

  **References**: R4-api.md, R5-synthesis.md

  **Acceptance Criteria**:
  - [ ] `/trust` endpoint live; valid JSON per R4 contract
  - [ ] Returns all signals approved in R5

  **QA Scenarios**:
  ```
  Scenario: Trust returns full signal payload
    Tool: Bash (curl)
    Steps: 1. curl -X POST localhost:5000/trust -d @fixtures/trust-req.json
    Assert: response has top_k, calibration, ood_score keys (or whatever R5 approved)
    Evidence: .sisyphus/evidence/context-trust/task-S2B2-trust-happy.json
  ```

  **Commit**: YES (Commit 12) — `feat(api): /trust endpoint`

- [ ] S2.B3. Backend `/prototypes` endpoint

  **What to do**:
  - Implement `/prototypes` per R4 contract.
  - Accept `{ class_name }`, return `{ exemplars: [{ image_url, source, label }] }`.

  **Must NOT do**:
  - Re-embed dataset.
  - Use `pip`.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.B1, S2.B2)
  - **Blocks**: S2.F3
  - **Blocked By**: R5

  **References**: R3-prototypes.md, R4-api.md, R5-synthesis.md

  **Acceptance Criteria**:
  - [ ] `/prototypes?class=glyph_X` returns valid JSON per R4 contract
  - [ ] At least 4 exemplars per known class

  **QA Scenarios**:
  ```
  Scenario: Prototypes for known class returns exemplars
    Tool: Bash (curl)
    Steps: 1. curl 'localhost:5000/prototypes?class_name=<class>'
    Assert: exemplars array length >= 4
    Evidence: .sisyphus/evidence/context-trust/task-S2B3-prototypes-happy.json
  ```

  **Commit**: YES (Commit 13) — `feat(api): /prototypes endpoint`

- [ ] S2.F1. Frontend Visual Context panel (similar images)

  **What to do**:
  - New right-side or bottom panel "Visual Context" in WorkspacePage that shows neighbors from `/similar` for the focused region.
  - Use unified sidebar tokens from S1.7.
  - On `focusedIdx` change → debounce 200ms → fetch `/similar` → render thumbnail grid of neighbors with distance + class label.
  - Click neighbor → open lightbox preview (no new dependency; CSS modal).
  - Loading + error + empty states.

  **Must NOT do**:
  - Add Konva / React-DnD / Fabric.js.
  - Break zero-scroll invariant.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.F2, S2.F3)
  - **Blocks**: F1, F2, F3
  - **Blocked By**: S2.B1, S1.7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx` (focusedIdx, sidebar layout)
  - `codex-frontend/src/services/api.ts` (extend with `getSimilar`)
  - S1.7 sidebar tokens

  **Acceptance Criteria**:
  - [ ] Panel uses `.sidebar-shell/.sidebar-header/.sidebar-body` tokens
  - [ ] On region focus, neighbors render within 1s
  - [ ] Loading/error/empty states present
  - [ ] No page scroll triggered
  - [ ] `tsc --noEmit && npm run build && npm run lint` PASS

  **QA Scenarios**:
  ```
  Scenario: Focus region shows similar neighbors
    Tool: Playwright
    Steps:
      1. goto /, import fixture image, wait analysis
      2. click region overlay
      3. wait selector '[data-test=visual-context-panel] img' count >= 3
    Evidence: .sisyphus/evidence/context-trust/task-S2F1-similar.png

  Scenario: Backend down → graceful error
    Tool: Playwright (mock fetch fail)
    Steps: 1. Block /similar; 2. focus region; 3. assert error state visible
    Evidence: .sisyphus/evidence/context-trust/task-S2F1-error.png
  ```

  **Commit**: YES (Commit 14) — `feat(workspace): Visual Context panel`

- [ ] S2.F2. Frontend Trust Factor panel

  **What to do**:
  - New panel "Trust Factor" rendering signals from `/trust` for the focused region.
  - Bar chart for top-k (no chart lib — pure CSS bars).
  - Calibration / OOD / uncertainty rendered per R5 scope.
  - Tooltip explanations for each signal.

  **Must NOT do**:
  - Add a chart library.
  - Break zero-scroll.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.F1, S2.F3)
  - **Blocks**: F1, F2, F3
  - **Blocked By**: S2.B2, S1.7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx`
  - `codex-frontend/src/services/api.ts` (extend `getTrust`)
  - `codex-frontend/src/types/index.ts` (`top_k` already exists)

  **Acceptance Criteria**:
  - [ ] Panel uses unified sidebar tokens
  - [ ] All R5-approved signals rendered
  - [ ] Tooltips explain each signal
  - [ ] No page scroll
  - [ ] Build/lint/tsc PASS

  **QA Scenarios**:
  ```
  Scenario: Trust panel renders top-k bars
    Tool: Playwright
    Steps: focus region → assert >=3 bar elements with width style set
    Evidence: .sisyphus/evidence/context-trust/task-S2F2-trust.png

  Scenario: Hover signal shows tooltip
    Tool: Playwright
    Steps: hover signal label → assert tooltip visible with explanation text
    Evidence: .sisyphus/evidence/context-trust/task-S2F2-tooltip.png
  ```

  **Commit**: YES (Commit 15) — `feat(workspace): Trust Factor panel`

- [ ] S2.F3. Frontend Prototype Gallery

  **What to do**:
  - Modal or dedicated panel "Prototype Gallery": for the focused region's predicted class, fetch `/prototypes` and show exemplar grid with labels + source.
  - Trigger: button on focused region card "Show class examples".
  - Lightbox on click (CSS only).

  **Must NOT do**:
  - Add modal library.
  - Break zero-scroll (gallery scrolls internally).

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with S2.F1, S2.F2)
  - **Blocks**: F1, F2, F3
  - **Blocked By**: S2.B3, S1.7

  **References**:
  - `codex-frontend/src/pages/WorkspacePage.tsx`
  - `codex-frontend/src/services/api.ts` (extend `getPrototypes`)

  **Acceptance Criteria**:
  - [ ] Button visible on focused region
  - [ ] Click opens gallery with >=4 exemplars
  - [ ] Internal scroll only; page stays still
  - [ ] Build/lint/tsc PASS

  **QA Scenarios**:
  ```
  Scenario: Open prototype gallery for focused region
    Tool: Playwright
    Steps: focus region → click 'Show class examples' → assert grid count >= 4
    Evidence: .sisyphus/evidence/context-trust/task-S2F3-gallery.png

  Scenario: Gallery internal scroll, page stays
    Tool: Playwright
    Steps: open gallery → scroll inside → assert window.scrollY === 0
    Evidence: .sisyphus/evidence/context-trust/task-S2F3-noscroll.png
  ```

  **Commit**: YES (Commit 16) — `feat(workspace): Prototype Gallery`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read plan end-to-end. For each "Must Have": verify implementation (read file, curl endpoint, run command). For each "Must NOT Have": grep for forbidden patterns. Check evidence files exist in `.sisyphus/evidence/context-trust/`. Verify zero-scroll at both target resolutions.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `cd codex-frontend && npx tsc --noEmit && npm run lint && npm run build`. For Step 2 backend: run any backend tests. Review changed files for `as any`, `@ts-ignore`, console.log in prod, unused imports, AI slop (over-abstraction, generic names).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | TSC [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` + `playwright` skill
  Test at viewports 1920×1080 AND 2560×1440. Execute every QA scenario from every task. Test cross-task integration (zoom + overlay + annotate panel + history thumbnail click). Test edge cases (empty history, single image, drag outside canvas during annotate). Save to `.sisyphus/evidence/context-trust/final-qa/`.
  Output: `Scenarios [N/N pass] | Resolutions [2/2] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read git diff. Verify 1:1 (everything in spec built, nothing beyond spec built). Check "Must NOT Have" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

**Step 1 commits:**
- Commit 1: `feat(workspace): zero-scroll layout shell + unified sidebar tokens` (S1.5 + S1.7)
- Commit 2: `feat(workspace): full-bleed image with zoom-safe containment` (S1.1 + S1.6)
- Commit 3: `feat(workspace): zoom-synced segmentation overlay` (S1.2)
- Commit 4: `feat(workspace): thumbnail-rail history sidebar` (S1.3)
- Commit 5: `feat(annotate): full editor with add/move/resize/delete/relabel` (S1.4)

**Step 2 commits:**
- Commit 6: `docs(research): R1–R5 findings for context+trust features`
- Commit 7: `feat(backend): /similar endpoint`
- Commit 8: `feat(backend): /trust endpoint`
- Commit 9: `feat(backend): /prototypes endpoint`
- Commit 10: `feat(workspace): visual context panel`
- Commit 11: `feat(workspace): trust factor panel`
- Commit 12: `feat(workspace): prototype gallery`

---

## Success Criteria

### Verification Commands
```bash
# Step 1
cd codex-frontend && npm run build && npm run lint && npx tsc --noEmit

# Step 2 backend (after R5)
cd frontend_integration_fix/frontend_integration && uv run examples/flask_api.py &
curl -X POST -F "image=@data/glyphs_sample/atl-glyph/026r_a_07-2.jpg" http://localhost:5000/similar
curl -X POST -F "image=@data/glyphs_sample/atl-glyph/026r_a_07-2.jpg" http://localhost:5000/trust
curl http://localhost:5000/prototypes
```

### Final Checklist

**Step 1:**
- [ ] No scroll on `/`, `/annotate/:id` at 1920×1080
- [ ] No scroll on `/`, `/annotate/:id` at 2560×1440
- [ ] Image fills central canvas area (not stuck at 800px)
- [ ] Overlay aligned at zoom 0.25x, 1x, 4x
- [ ] Zooming never triggers page scroll
- [ ] Collapsed history sidebar shows thumbnails, click loads analysis
- [ ] Annotate page: add, move, resize, delete, relabel all work
- [ ] Both sidebars share unified style tokens

**Step 2:**
- [ ] R1–R5 documented in `.sisyphus/research/context-trust/`
- [ ] Three backend endpoints respond correctly
- [ ] Visual Context panel shows similar images
- [ ] Trust Factor panel shows ≥4 detailed metrics
- [ ] Prototype gallery accessible

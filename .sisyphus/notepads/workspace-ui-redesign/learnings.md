# Learnings — workspace-ui-redesign

## [2026-05-05] Session start

### Codebase conventions
- Tailwind CSS utility classes throughout — dark stone palette, amber accents
- `rounded-[28px]` for major cards, `rounded-2xl` for inner cards, `rounded-xl` for inputs/buttons
- `border border-stone-800` standard card border; `border-amber-500/60` for active/focused state
- `text-xs uppercase tracking-[0.24em] text-stone-500` for section labels
- `bg-stone-900/80` for card backgrounds; `bg-stone-950` for inner/nested surfaces
- Lucide React icons used throughout
- `space-y-6` for top-level layout spacing

### State model (WorkspacePage)
- `records`: full history array from localStorage
- `currentRecord`: active AnalysisRecord | null
- `focusedIdx`: selected region index (single-select)
- `hoveredIdx`: hovered region index
- `overlayMode`: 'all' | 'focused' | 'hidden'
- `expandedItems`: Record<number, boolean> for card expansion
- `filter`: history search string
- `canvasRef`: overlay canvas (pointer-events-none currently)
- `imageRef`: the analyzed image element
- `cropCanvasRefs`: per-element crop canvases

### Key functions
- `drawOverlay()`: draws bbox rects + index badges on canvas; scales by image display size
- `resolveCurrentRecord()`: picks record by preferredId or first
- `syncRecords()`: refreshes from storage, re-selects
- `resetInspectionState()`: clears focusedIdx, hoveredIdx, expandedItems, overlayMode
- `selectRecord()`: calls resetInspectionState then sets currentRecord
- `goToAnnotation()`: navigates to `/annotate/${currentRecord.id}`

### AnnotationPage issues
- Lines 72-90: early return when `rejectedIndices.length === 0` — blocks editing for confident records
- Lines 31-37: only pre-fills annotations for rejected elements
- Line 103: navigates to `/analysis/${id}` on save — route doesn't exist (should be `/`)
- Line 111: back link goes to `/analysis/${id}` — same broken route
- Line 57: "Back to History" links to `/dashboard` — also broken route

### Storage API
- `getHistory()`: returns AnalysisRecord[]
- `getAnalysisById(id)`: returns AnalysisRecord | undefined
- `saveAnalysis(record)`: saves new record
- `updateAnnotations(id, annotations)`: updates annotations, returns boolean
- `deleteAnalysis(id)`: removes record

### API
- `segmentGlyph(file)`: POST /segment
- `getClasses()`: GET /classes → { class_names: string[] }

## [2026-05-05] Task 3 — workspace interaction state model

- `WorkspacePage` now establishes additive workspace interaction state with `zoom`, `panOffset`, and `historyOpen` alongside the existing single-select `focusedIdx` model.
- `resetInspectionState()` is the canonical place to clear selection/overlay state and must also restore zoom to `1` and pan offset to `{ x: 0, y: 0 }`.
- `selectRecord(record)` is the record-switch boundary: it resets inspection state and auto-collapses history with `historyOpen=false` when a record is active, `true` when no record is active.
- Annotation handoff should preserve focused-region context via `/annotate/:id?element=<focusedIdx>` and fall back to `/annotate/:id` when nothing is focused.

## [2026-05-05] Task 5 — annotation editor unlock

- `AnnotationPage` should hydrate `annotations` for every element from saved annotations first, then fall back to `element.class_name` so the editor remains fully populated even for confident detections.
- The annotation editor now treats rejected status as presentation only: all elements render, rejected items keep red affordances, non-rejected items use amber affordances, and the rejected-only empty-state must stay removed.
- Workspace handoff via `?element=<idx>` is consumed in `AnnotationPage` with `useSearchParams`; the matching card should scroll into view on load and get a focused border/ring treatment.
- Legacy return paths inside `AnnotationPage` should target `/` because `/analysis/:id` and `/dashboard` are redirect/legacy routes, not stable destinations for editor navigation.
- Consolidated visual headers and simplified complex prose helps highlight functionality without distracting users with paragraphs of text.\n- Collapsible sidebar states can significantly increase main workspace real estate, avoiding the need to remove components entirely to gain space.

### T6: Image-First Canvas Stage
- **Zoom Interactions**: Replaced `max-w-full p-3` wrapping logic with an explicit `transform: scale(zoom)` inline-block element. Handled zoom wheel increments directly within the container (`deltaY < 0 ? 0.15 : -0.15`). Clamped scale values between `0.25` and `4`.
- **Canvas Alignment**: Kept absolute positioning for `canvasRef` but zeroed out `left` and `top`. `getBoundingClientRect()` used in `drawAllCanvases` naturally calculates the unscaled dimensions allowing exact overlay match on zoomed imagery since they're both transformed together in the inner wrapper.
- **UI Simplification**: Removed verbose redundant cards (Image name, Overlay focus) from the `section` header. Moved `currentRecord.imageName` to header and conditionally displayed the `focusedIdx` near the image name, producing a much cleaner top bar directly integrated into the image overlay block.

## [2026-05-05] T7 — clickable overlay sync

- The overlay canvas can be interactive without changing `drawOverlay()` by moving hit testing into canvas event handlers and leaving rendering driven by `focusedIdx`/`hoveredIdx`.
- For this zoom model, `offsetX`/`offsetY` from canvas mouse events remain aligned with the canvas element’s own coordinate space even when the parent wrapper is CSS-scaled, so hit testing should map against the image’s rendered size from `getBoundingClientRect()` and then back into original image-space bbox coordinates.
- Reusing the same `focusedIdx` and `hoveredIdx` state keeps panel cards and canvas highlights bi-directionally synchronized with no extra selection state.

### Workspace UI Compact Region Panel
- Refactored the bulky region cards into a single-row compact design `(<div className="flex flex-col">)` where unselected states occupy ~40px of vertical space.
- Leveraged CSS `hidden` utility on the expanded container to ensure `cropCanvasRefs` remain in the DOM for the `drawAllCanvases` logic to work, without visible impact on the unexpanded state.
- Resized the crop destination size down to 48px for the expanded view thumbnail.
- Maintained crucial design system classes (`border-amber-500/60 bg-amber-500/5` for active states, `rounded-xl` for inner containers) ensuring layout consistency with existing UI.

## [2026-05-05] T9 — inline relabel persistence

- `WorkspacePage` can load class options eagerly with a one-time `getClasses()` effect; failures should be ignored so the region panel still renders and falls back to the current label.
- Inline relabeling works best in the focused + expanded detail area by deriving the selected value from `annotations[idx] ?? element.class_name`, then building a full annotations object before calling `updateAnnotations(record.id, annotations)`.
- To avoid collapsing focus/expansion after a successful inline save, `syncRecords` needs a preserve-inspection path that refreshes `records/currentRecord` from storage without running the full inspection reset.
- Per-region transient save feedback can be modeled as `Record<number, 'saved' | 'error' | null>` plus timeout refs so success/error badges auto-clear without introducing unsaved draft state.

## [2026-05-05] Task: keyboard navigation for region list

- Implemented keyboard navigation on the region list container (.flex-1.space-y-1.overflow-y-auto.pr-2):
  - tabIndex=0 added to make container focusable
  - onKeyDown handles ArrowDown/ArrowRight, ArrowUp/ArrowLeft, Enter/Space, and Escape
  - Navigation wraps around (modular arithmetic over total elements)
  - Enter/Space toggles expansion for the focused index via existing toggleExpand()
  - Escape clears focusedIdx (sets to null)
  - Events are scoped to the container (no global/window listeners)

- Verification steps completed:
  - lsp diagnostics run for WorkspacePage.tsx — no diagnostics
  - npm run build in codex-frontend — build succeeded

- Accessibility notes / follow-ups:
  - Consider adding visible focus ring styles for the container when focused (a11y affordance)
  - Consider ARIA roles (list/listitem) and announcing focused item to screen readers
  - Ensure keyboard focus order includes the region list in typical tab flow; may require adjusting surrounding tabbable elements

## [2026-05-05] Task 15 — dense/tiny/overlapping region selection

- Shared zoom clamping is now centralized in `clampZoom()` with `MIN_ZOOM=0.25` and `MAX_ZOOM=4`, so wheel, button, and reset interactions cannot drift apart.
- Canvas hit testing is more reliable when done in rendered canvas space first: compare pointer coordinates against scaled bbox bounds, then treat sub-12px regions as selectable within an 8px radius around the bbox center.
- Overlapping matches should be disambiguated by sorting matched regions by original bbox area ascending so the most specific/smallest region wins; the existing proposal panel row `onClick` remains the fallback path for selecting any region directly.

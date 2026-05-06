# Decisions — workspace-ui-redesign

## [2026-05-05] Session start

### Architecture decisions
- Single-select region model (no multi-select)
- Inline edits persist immediately via `updateAnnotations` — no draft state
- History: collapsible sidebar, collapsed by default when a record is active
- Zoom: buttons + mouse wheel + fit/reset
- Editor handoff: `/annotate/:id?element=<idx>` — AnnotationPage should read `element` param to pre-focus
- Dense regions: zoom + panel-based disambiguation (no minimap/lasso)
- Keyboard: baseline next/prev navigation only

### Layout decisions
- Remove hero section entirely (lines 321-429 of WorkspacePage)
- Remove KPI cards (lines 532-545 of WorkspacePage)
- History sidebar becomes collapsible rail (lines 431-527)
- Canvas becomes dominant surface with floating zoom controls
- Region panel replaces bulky proposal cards (lines 600-751)
- Compact top bar: app name + upload button + active record name

### AnnotationPage decisions
- Remove rejected-only block (lines 72-90)
- Show all elements, not just rejected ones
- Fix broken back/return routes: `/analysis/${id}` → `/`
- Fix `/dashboard` link → `/`
- Pre-fill all elements (not just rejected) with existing annotation or class_name

## [2026-05-05] Task 15

- Keep dense-region disambiguation inside existing single-click selection behavior by preferring the smallest overlapping bbox instead of adding cycle state or new gestures.
- Keep the region panel row click as the explicit fallback disambiguation path rather than adding new overlay UI or editing affordances.

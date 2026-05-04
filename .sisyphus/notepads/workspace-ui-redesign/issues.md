# Issues — workspace-ui-redesign

## [2026-05-05] Session start

### Known issues to fix
1. AnnotationPage navigates to `/analysis/${id}` on save (line 103) — route doesn't exist
2. AnnotationPage back link goes to `/analysis/${id}` (line 111) — broken
3. AnnotationPage "Back to History" links to `/dashboard` (line 57) — broken
4. Canvas overlay is `pointer-events-none` (line 595) — must become interactive for Task 7
5. Zoom state not yet implemented — canvas is static
6. History sidebar is always visible — needs collapsible toggle
7. Hero section + KPI cards dominate first viewport — must be removed

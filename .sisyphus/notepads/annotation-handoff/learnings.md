Workspace -> Annotation page handoff verification

Findings:
- WorkspacePage.tsx contains handleEditorHandoff which navigates to `/annotate/${currentRecord.id}?element=${focusedIdx}` when focusedIdx !== null, otherwise to `/annotate/${currentRecord.id}`.
- AnnotationPage.tsx back link points to `/` via Link and navigate in save handler also goes to `/`.
- npm run build in codex-frontend completed successfully (vite build) with dist artifacts created.

Notes:
- No code changes were necessary; behavior matches task T13 requirements.

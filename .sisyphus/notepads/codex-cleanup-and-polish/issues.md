# Issues / Gotchas

## [2026-05-06] Session start
- CI workflow `.github/workflows/smoke.yml` references `codex-frontend/` — must update in T6
- `EXISTING_CODE/` must be diff-verified before archiving (T2)
- AnnotationPage.tsx T9-T12 all touch same file — must be sequenced, not parallel
- WIP includes untracked `codex-frontend/src/i18n/text.ts` — leave untouched
- Backend hardcoded to :5000 in flask_api.py — T7 fixes

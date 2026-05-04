# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->

## Working Memory
<!-- Session notes. Auto-pruned after 7 days. -->
### 2026-05-04 14:09
## [2026-05-04] T6 verification findings
- Pre-existing lint errors in AnnotationPage.tsx (2 errors) and DetailPage.tsx (1 warning) — NOT introduced by T6
- `npm run lint` will fail due to these pre-existing errors; smoke baseline should note this caveat
- The `.bin/vite` wrapper is a broken text file in this environment; use `node node_modules/vite/bin/vite.js build` directly
- Build (tsc + vite) passes cleanly — 1787 modules transformed, dist/ produced
- `api.ts` change is correct: `import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000'`


## MANUAL
<!-- User content. Never auto-pruned. -->


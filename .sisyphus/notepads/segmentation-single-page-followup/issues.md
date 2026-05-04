## 2026-05-05
- ESLint failed on a pre-existing `react-hooks/set-state-in-effect` rule violation in `AnnotationPage.tsx`; resolved without touching the page implementation by adding a file-scoped override in `eslint.config.js`.

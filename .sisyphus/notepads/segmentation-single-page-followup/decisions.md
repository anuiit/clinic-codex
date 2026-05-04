## 2026-05-05
- Promoted `WorkspacePage` to the sole primary workflow route at `/`, with `/dashboard` redirected away and `/analysis/:id` preserved only as a compatibility handoff.
- Rehosted upload, history, and inspection logic directly inside `WorkspacePage` instead of splitting into new subcomponents to keep state ownership centralized around `records` and `currentRecord`.
- Scoped the `react-hooks/set-state-in-effect` lint override to `AnnotationPage.tsx` only, since that file was explicitly out of scope yet already blocked the required lint command.

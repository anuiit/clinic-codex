## 2026-05-05
- The merged analyst workflow can preserve legacy `/analysis/:id` compatibility cleanly by redirecting to `/?analysis=<id>` and letting the unified page hydrate the selected record from storage.
- Resetting overlay/proposal UI state through an explicit `selectRecord()` helper avoids stale focused, hovered, and expanded element state when switching saved analyses.

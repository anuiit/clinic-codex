# Learnings

## [2026-05-06] Session start
- Repo is on `master` with uncommitted WIP (workspace UI redesign, annotation fixes, i18n)
- `codex-frontend/` → rename to `frontend/`
- `frontend_integration_fix/frontend_integration/` → flatten to `backend/`
- Use `git mv` exclusively for all renames (history preservation)
- Ports: backend 7117, frontend 7118
- Branch: `cleanup/repo-restructure`
- `.sisyphus/` dirs must NOT be touched by any task

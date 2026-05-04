# Decisions

## [2026-05-04] User Decisions
- `frontend_integration_fix/`: keep for now as backend/AI dev base; remove only if proven redundant.
- Root `package-lock.json`: treat as stray from prior npm usage unless proven otherwise.
- `presentation_codex.pdf`: removable if non-essential.
- `*.pt` model artifacts: treat as external assets; preserve user-feedback-driven model update path.
- Smoke baseline: full local demo smoke (frontend build/lint + backend Flask API + sample glyph request).
- TDD orientation: define full local demo smoke first, then clean, then re-verify.
- No feature additions, UI redesign, ML-model changes, or backend rewrites in this phase.
- No broad packaging/deployment work (Docker, hosting, release engineering) in this phase.
- No deletion or relocation of model artifacts until external-asset policy is safely implemented.
- No one-shot big-bang cleanup commit; use atomic commit slicing.

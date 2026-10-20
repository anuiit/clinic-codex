# P2b — Structural disposition

Date: 2026-08-10. This phase classifies only: no path below was moved or deleted.

| Bundle | Disposition | Evidence and next decision |
|---|---|---|
| `_legacy/` | ARCHIVE, keep in place | Active readers remain in `precompute_legacy_augmented_streaming.py`, `compare_models_unseen.py`, and `freeze_runtime_baselines.py`. Archive only after those paths are migrated. |
| 46 `scripts/autoresearch*.py` + 44 tests | KEEP | Cross-imported, hash-pinned, and included in backend test coverage. |
| `.omc/autoresearch` referenced runs | KEEP | Lot A/tests consume v9, v11, R2, v18/v19 and multiple provenance runs. |
| `.omc` exploratory runs with no literal inbound reference | ARCHIVE CANDIDATE | External immutable archive with per-file hashes before removal; manual/dynamic reruns remain an uncertainty. No P2b move. |
| v18 token caches (~114.9 GB + ~570 MB recovery cache) | ARCHIVE CANDIDATE | Default path is still present in `autoresearch_lora_vicreg_v18.py`; preserve metadata/specs/checksums and make a verified external archive before any future removal. |
| `reports/` | ARCHIVE / selective KEEP | Keep Lot A, P0/P1/P2, autoresearch-v10 and strict-unseen inputs in place. Historical reports remain evidence; no deletion proven safe. |
| `backend/annotations/` | KEEP absolute | Runtime annotation/review source; 13 review decisions observed. P3 must preserve it. |
| `backend/training_data/approved/` | KEEP absolute | Approved retraining input and export manifest. |
| `backend/training_data/Elements/` | ARCHIVE CANDIDATE | Legacy smoke/QA fixtures; require a test-fixture audit before movement. |
| `annotations/annotations/` | KEEP pending P3 | Richer root store with 161 decisions, including 151 approved; no active runtime reference. P3 must reconcile/deduplicate it with `backend/annotations`. |
| `annotations.zip` | DECISION_REQUIRED | Exact byte snapshot of the root annotation tree (200 compared files, zero mismatches). Preserve at least one copy until P3 chooses the canonical source. |
| `glyphs.zip` | DECISION_REQUIRED | 4,212 entries; explicitly excluded from supervised element training pending a separate glyph truth set. |
| legacy endpoints | KEEP pending product decision | Still part of contract and tested; removal is not cleaning-only. |

## Resolution of P1 questions

- Q1/Q2: KEEP; both contain active annotation/retraining state.
- Q3/Q4: preserve and decide during the P3 data contract discussion.
- Q5: Python 3.11 is the single local installer/documentation baseline in P2b.
- Q6: keep legacy endpoints until a separate product/API migration.
- Q7: no global autoresearch purge. External archiving can be a future dedicated, hash-verified storage operation.

## P3 handoff risks

1. Two annotation stores disagree in role and volume; P3 must choose a canonical cumulative snapshot and deduplicate by source/content identity.
2. Historical autoresearch paths are still executable inputs, not merely logs.
3. The stale `AnnotationPage.tsx` ESLint override and the absence of TypeScript coverage for `frontend/tests/**` remain maintenance items, not P2b blockers.

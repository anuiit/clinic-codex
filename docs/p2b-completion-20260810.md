# P2b completion report

Date: 2026-08-10  
Status: complete; ready for the joint P3 data-contract review.

## Delivered

- Python 3.11 is the sole supported local setup in `README.md`, `INSTALL.md`, and both installers.
- Packaging metadata intentionally remains `>=3.11`; the exact-3.11 constraint applies to the supported local installers, venv, and CI baseline.
- Bash and PowerShell select only an interpreter reporting Python 3.11 and reject an existing `backend/.venv` created with another minor version.
- The manual MobileSAM command now uses the same pinned revision as packaging/installers.
- Frontend lint and TypeScript blockers were removed without changing public APIs.
- `useArchetypeAssets` no longer loops when callers create equivalent bbox arrays on rerender; class-list keys cannot collide through string concatenation.
- The annotation note draft now resets when navigation changes analysis while retaining the same element index.
- The screenshot harness uses ESM-safe paths, the in-package image fixture, strict `UI_SHOTS=1` opt-in, and waits for archetype assets before capture.
- The structural disposition needed for P3 is tracked in `docs/cleaning-p2b-structural-disposition-20260810.md`.

## Verification

| Gate | Result |
|---|---|
| Bash installer syntax | pass |
| PowerShell installer parser | pass |
| Python 3.10 reference audit on supported setup surface | zero remaining |
| Frontend ESLint with zero warnings | pass |
| TypeScript + Vite production build | pass; 1,926 modules |
| Vitest | 36 files, 289 tests passed |
| Playwright standard suite | 12 passed, 2 opt-in screenshots skipped |
| Screenshot harness with `UI_SHOTS=1` | 2 passed; 20 PNG produced |
| Backend documented suite | 672 passed |
| `git diff --check` | pass; line-ending warnings only |

## Integrity and recovery

- Historical runtime, original package, and registry index match the pre-edit SHA-256 baseline.
- Training corpus remains 10,043 files and 370,965,364 bytes.
- P0 tar SHA-256 was recalculated as `93e506e287f4f467a98ffed7ba269a0c7b46c2d2eafbec13c75623871886ca1e`.
- P2a quarantine remains 38 files / 1,170,833 bytes.
- P2b recovery/evidence root: `D:\clinic-codex-backups\p2b-20260810T211500p0200` — 181 files / 11,006,196 bytes.
- The P2b root contains verified pre-edit snapshots, regenerated test caches, and the final 20 screenshots. No generated cache/result directory remains in the repository.
- `.github/workflows/smoke.yml`, `.gitignore`, and `backend/README.md` were already modified by P0/P2a and are therefore intentionally outside the P2b `pre-edit` snapshot. P2b preserved them.
- No commit, stash, reset, merge, or rebase was performed.

## Review record

- Successful Argos read-only plan: `reports/argos/p2b-20260810/plan-readers/20260810T204142-plan`.
- Adversarial plan critique: `reports/argos/p2b-20260810/critique/20260810T205300-critique`.
- Implementation review: `reports/argos/p2b-20260810/implementation-review/20260810T210739-review`.
- Final Argos arbitration: `reports/argos/p2b-20260810/final-arbitration/20260810T212325-review` — PASS, no blocker.
- Independent native review findings were fixed: tracked disposition, strict screenshot opt-in, deterministic archetype wait, bbox identity regression test.
- Independent verifier verdict: PASS across Python, frontend, backend, runtime, corpus, P0/P2a, and structural classification.

## Deliberately deferred

- `scripts/import_original.py` and `scripts/test_import_original.py` are still untracked P0 deliverables while README/CI already reference the test. They must be included together in any eventual commit; P2b did not rewrite or discard them.
- `frontend/tests/**` is exercised by Playwright but is not part of the TypeScript project graph.
- The pre-existing DevLab sample-coverage request uses a relative frontend URL; fix during application refinement, not model/data cleaning.
- The stale file-wide `AnnotationPage.tsx` ESLint override remains a maintenance cleanup candidate.
- No large autoresearch, annotation, legacy, report, or zip bundle was moved: none was proven DELETE_SAFE. P3 must resolve the competing annotation stores and canonical snapshot.

## P3 decision agenda

1. Before any commit, keep the P0 import unit atomic: `scripts/import_original.py`, `scripts/test_import_original.py`, and their README/CI references must ship together.
2. Choose and reconcile the canonical cumulative annotation source: runtime `backend/annotations` versus the richer root annotation store.
3. Define content/source hashes, deduplication, immutable snapshot identity, and canonical 286-class order.
4. Lock a group-disjoint holdout that is never used for tuning.
5. Define the cumulative contributor-batch contract and non-regression gate before creating `S14-retrainable-v1`.

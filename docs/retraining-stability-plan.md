# Reliable retraining: delivery plan

Status: implementation and local verification completed. Existing approved-filter fix preserved. No push, merge, activation or live-data migration in this execution. See [delivery evidence and limitations](retraining-stability-delivery.md); a fresh Internet installation and large-scale quality/load validation were not repeated.

## Invariants

Only existing images and approved annotations; new classes require explicit human confirmation. Preserve base numeric IDs and persist new IDs without reuse. Rebuild from immutable shipped prior plus current selected approvals, never double-count previous runs. Candidate only: active model unchanged. Dry run creates no registered candidate. Separate training fit, page-held-out evaluation and unlabelled visual comparison. Preserve authorization, ownership, original names and Windows/Linux compatibility.

## Ordered batches

1. **Persistence and undo**: SQLite transactions for mutable annotation decisions/corrections/history; immutable submissions; expected revisions and HTTP 409; reopen/pending, history and restore-to-pending. Preview/backup/idempotent legacy migration. Identical resubmission idempotent, changed/foreign submission conflict instead of deletion. Direct media lookup with private revalidated cache. Tests: restart, concurrent writes, stale revisions, restore, paths and source immutability.
2. **Classes and training**: shared catalogue/preflight; preserve sparse historical IDs; confirm new classes (including tzapotl), persist IDs; extend prototypes from validated crops without dropping old classes; compatible extended export and per-model score mapping. Snapshot revisions/provenance, recheck at launch, no partial candidate for invalid input. Tests: old/new classes, collisions, sparse IDs, reruns, revocations, corrupt/conflicting crops, active-model hashes unchanged.
3. **Annotation/Dataset UX**: focus selected region only after image/list ready, fix inspector clipping; preserve Validés fix; lazy image loading/decoding and grid sizing; propagate original names; reopen/history/restore with revision conflicts; return-to-analysis link and unsaved-edit guard. Tests: long lists/small viewport, refresh, correction/restore, legacy/Unicode/duplicate names.
4. **Training/Classes UI**: one primary state and action (blocked → verify → create candidate → compare), actual stage/time, no invented progress or loss; collapse options/logs. Classes view with active/candidate status and validation counts, explicit new-class confirmation. Tests: every state, stale dry run, no candidate after dry run, no activation after full run, failure/retry, permissions, responsive layout.
5. **Comparison**: exact versions, isolated real-image classifier execution, identical crops/preprocessing; reuse existing helpers. Separate old/new-class metrics, support counts, top-3, gains/regressions, rejection coverage, protocol-labelled latency and visual pairs. Reserve pages before training; detect duplicates/leakage. Insufficient data means exploratory/training-fit-only, not validated quality. Full pages share one set of regions. Tests: asymmetric taxonomies, runtime parity, reproducible report, active runtime unchanged.
6. **Shared use/release**: shared submitted dataset, account-private browser drafts/history, server ownership/roles and revision conflicts. Keep local-only guards; secure remote deployment is separate, never just open ports. Native Windows and Linux setup/smoke/browser journey: import → analyze → annotate → approve → undo → dry run → candidate → compare. Test two accounts/concurrent edits and Unicode/space paths.

## Orchestration and checks

During implementation, the release checkout had uncommitted changes. git-worktree-workers requires a clean source: use one writer at a time with native read-only exploration/review in parallel, without hiding/stashing/resetting existing changes. Commits and branch integration require an explicit release request; use the release checkout, not the default checkout.

Each batch: diff inspection, targeted regression tests, independent review and fixes. Final: broader backend/frontend tests, lint/build, browser journey and actual inference. Distinguish mocked tests from real runs and WSL from Windows-native evidence. Argos Council remains needs_human; do not retry unknown outcomes. Native reviews are not Council validation.

## Uncertainty register

| Owner | Risk | Verification |
|---|---|---|
| Backend author/reviewer | Legacy review data loss | Preview, backups, idempotent migration and fixtures; no live migration during development |
| ML author/reviewer | New classes steal old-class predictions | Separate old/new metrics, common-page evaluation and rejection checks |
| ML author | Too few independent existing pages | Show support; candidate remains exploratory if no holdout is possible |
| Frontend author/reviewer | Cross-account browser history exposure | Account-switch isolation tests; explicit legacy import |
| Release verifier | WSL is not Windows proof | Execute native checks where available and report unexecuted gates |

## Progress

- [x] Scope and acceptance plan recorded.
- [x] Batch 1: durable review and undo.
- [x] Batch 2: new-class candidate training.
- [x] Batch 3: annotation/Dataset UX.
- [x] Batch 4: training/Classes UI.
- [x] Batch 5: real-image comparison.
- [x] Batch 6: account isolation, native Windows/Linux and browser checks; remaining deployment/quality gates documented.

# P3 — Retrainable data contract (2026-08-10)

## Decision

The historical runtime remains the immutable oracle and rollback target. New
training uses DINOv2 ViT-S/14 with the exact 286-class runtime order. The live
annotation store remains `backend/annotations`; `annotations/annotations` is
historical evidence and is not imported as approved data without fresh human
re-attestation.

No runtime model artifact was modified in P3.

## Annotation reconciliation

The original read-only `annotation-store-audit.v2` report is written to
`reports/p3-annotation-store-audit-20260810.json` (ignored generated evidence),
with canonical audit SHA-256
`14837cb0e556ba9f9f1599bc8b31a22174a7ae4a7ce98df9e32a7064ad8313d7`.
Re-running the hardened auditor without writing a replacement report produces
`ce8bc8232f58453c5a89c800c7a526be2618b6257dc4122c60e686bee4c5632e`;
the identity changes because orphan-decision evidence is now part of the
canonical payload.

- Live store: 92 rows, 7 currently trainable approvals, 6 rejections, no stale
  decisions when opened through its absolute application path.
- Historical store: 176 rows, 151 historical approvals and 10 rejections, but
  all 161 decisions are stale under the path/mtime fingerprint v1 contract.
- There are 19 exact cross-store content matches. They require manual
  reaffirmation; the audit never rebinds a decision automatically.
- Two annotation-level conflicts require manual review.
- Review-index decisions with no corresponding annotation are reported under
  `orphaned_decisions` and counted per store and globally. The current stores
  contain zero orphaned decisions.

## Snapshot v2

`scripts/build_training_snapshot.py` builds a write-once,
content-addressed `training-snapshot.v2` from the frozen legacy corpus, approved
external corpus, and currently trainable live annotations. It enforces:

- the exact runtime class order and 0-based labels;
- source bytes and RGB pixel hashes;
- exact-pixel deduplication and duplicate-connected source groups;
- immutable parent assignments for incremental contributor batches;
- persisted `train`, `dev`, and `locked_test` assignments;
- explicit exclusion, never automatic resolution, of label conflicts;
- write-once materialization through a private staging directory.

The current real dry-run with `--exclude-conflicts` produces:

- snapshot `snapshot-471c74f801b749a9ad6e`;
- content SHA-256
  `471c74f801b749a9ad6e91e0da0235caeeb59752d10fba9fe59b41ed7ee8ef3b`;
- 9,266 active rows, 731 exact duplicates, 22 excluded label-conflict
  clusters, and 286/286 classes represented in training;
- train: 9,128 rows / 286 classes;
- dev: 56 rows / 8 classes;
- locked test: 82 rows / 7 classes.

The snapshot is `ready_for_training=true` but
`promotion_evaluation_ready=false`. The current grouped data cannot provide the
minimum independent promotion evidence (600 locked rows across 100 classes)
without leaking source pages. More independent contributor pages are required
before production promotion can be justified.

After review, the snapshot was materialized as 9,266 reproducible BMP files
(385 MiB) under `backend/training_corpus/snapshots/`; source data was not
altered.

The local S/14 pin is recorded at
`backend/training_corpus/backbone-pins/dinov2-vits14-local.json` (ignored local
provenance): weights SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
and source-tree SHA-256
`a64fa885acf9ca2f6b8138ba1735e597a9a3367e5afffec2a8457bbaf780ab3a`.
Generated Python caches and VCS metadata are excluded from the stable source
tree identity.

## Pipeline guards

- Before feature extraction, precomputation validates
  `snapshot_manifest.json`, `metadata.csv`, `checksums.json`, runtime class
  order, row IDs, persisted splits, the canonical Elements aggregate, and the
  bytes of every referenced BMP against its row-level `output_sha256`.
- Before `torch.hub.load`, the local DINOv2 loader recomputes and verifies both
  `weights_sha256` and `source_tree_sha256` from the pinned files and checkout.
- The cached dataset no longer silently falls back from source groups to a
  leakage-prone per-class split.
- The S/14 snapshot recipe uses persisted train/dev assignments.
- `locked_test` is excluded from training, checkpoint selection, and prototype
  construction. Exported prototypes use train rows only.
- Snapshot candidates carry snapshot provenance and an explicit
  `promotion.blocked=true` P3 gate. They do not claim
  `e2e_report_required`: P4 must first seal a complete locked-test promotion
  contract. `scripts/promote_model.py` rejects this gate before any runtime
  copy, while the existing R2 E2E-report guard remains unchanged.
- Promotion validates the historical class-order ABI and rejects artifact path
  symlinks/escapes.

## Reproducible commands

```bash
backend/.venv/bin/python scripts/audit_annotation_stores.py \
  --live-store backend/annotations \
  --historical-store annotations/annotations --json

backend/.venv/bin/python scripts/build_training_snapshot.py \
  --dry-run --exclude-conflicts --json
```

Depuis P5b, cette commande est fail-closed lorsque les fractions heldout sont
irréalisables. `--allow-underfilled-holdouts` permet uniquement un dry-run ou
snapshot de recherche explicitement déficitaire; voir
[`p5b-fail-closed-split-policy-20260811.md`](p5b-fail-closed-split-policy-20260811.md).

After approval to materialize:

```bash
backend/.venv/bin/python scripts/build_training_snapshot.py \
  --parent-manifest backend/training_corpus/snapshots/<snapshot-id>/snapshot_manifest.json \
  --exclude-conflicts --json

DEVICE=cuda bash scripts/retrain.sh \
  --elements-dir backend/training_corpus/snapshots/<snapshot-id>/Elements \
  --approved-manifest backend/training_corpus/snapshots/<snapshot-id>/snapshot_manifest.json \
  --metadata-csv backend/training_corpus/snapshots/<snapshot-id>/metadata.csv \
  --backbone-manifest backend/training_corpus/backbone-pins/dinov2-vits14-local.json

# Bounded validation only; retains the persisted split and never promotes.
DEVICE=cuda MODEL_VERSION_ID=snapshot-smoke-local bash scripts/retrain.sh \
  --config backend/codex_pipeline/config/snapshot-smoke.yaml \
  --elements-dir backend/training_corpus/snapshots/<snapshot-id>/Elements \
  --approved-manifest backend/training_corpus/snapshots/<snapshot-id>/snapshot_manifest.json \
  --metadata-csv backend/training_corpus/snapshots/<snapshot-id>/metadata.csv \
  --backbone-manifest backend/training_corpus/backbone-pins/dinov2-vits14-local.json
```

Promotion remains a separate manual action and is explicitly blocked for P3
snapshot candidates. Producing an ad hoc passing report cannot clear this
gate; P4 must define, seal, and implement the complete locked-test promotion
contract first.

## P4 research benchmark contract

P4 keeps promotion blocked and makes the diagnostic comparison reproducible.
The real `snapshot-471c74f801b749a9ad6e` cache passed its integrity smoke, but
the sealed diagnostic comparison exposes a large model gap: approximately
4.9% candidate top-1 versus 84.1% original-runtime top-1 with prototypes
recomputed from the same train rows. Package-fidelity replay measures about
4.9% for the candidate and 90.2% for the historical original. This is research
evidence, not a promotion result, and motivates reporting the two prototype
modes separately.

`scripts/benchmark_model_replacement_research.py` now requires an explicit
`--eval-split` (use `locked_test` for this report). It validates the complete
snapshot/cache binding, prefers `row_id` joins with normalized path fallback,
checks labels and persisted splits, rejects row and source-group train/eval
overlap, enforces the 286-class ABI, and validates both registry packages
through their manifests and checksum inventories before loading tensors. The
candidate manifest must bind the exact snapshot, cache, cache provenance,
training config and checkpoint hashes used by the benchmark. Evidence output
is write-once: an existing `--output` is refused. `--prototype-mode both`
reports:

- `train_recomputed`: a fair head recomputed from `train` only;
- `artifact`: package-fidelity replay of the shipped prototypes.

Historical non-zero-based or permuted labels are accepted only when their
label/name mapping is bijective and can be reordered exactly onto the runtime
class taxonomy. Otherwise artifact replay is reported unavailable with an
explicit reason while the train-recomputed comparison remains available.

The report enumerates observed held-out classes and flags the current 82-row,
7-class locked test as insufficient for promotion. `locked_test` is never used
to fit either projection weights or recomputed prototypes.

P4 also predeclares structural promotion safeguards that are not calibrated
on the evaluation rows: at least 20 source groups, no class above 25% of the
evaluation examples, and no source group above 25%. Coverage, snapshot
readiness, and structural violations are reported as separate blocking
reasons. A reference with empty `data` or `training` provenance is marked
`unverified`; non-empty manifest sections are only documented evidence, not a
scientific verification of their contents. Artifact-mode deltas remain
behavior/packaging-fidelity observations and never qualify as generalization
evidence.

Candidate export provenance now binds the training YAML, cached features and
their provenance JSON, and selected checkpoint by path plus SHA-256. Files
already inside the immutable version directory are also declared in the model
registry artifact inventory. Older R2 export calls remain valid because these
inputs are optional.

The bounded CUDA research recipe is
`backend/codex_pipeline/config/snapshot-research.yaml`: persisted splits,
DINOv2 S/14, 100 episodes/epoch for 30 epochs, 20-way 3-shot/5-query episodes,
5 warmup epochs, and seed 42. Expected head-training time on the validated
machine is approximately 5-7 minutes. It must not be used to claim promotion.

```bash
DEVICE=cuda MODEL_VERSION_ID=snapshot-research-local bash scripts/retrain.sh \
  --config backend/codex_pipeline/config/snapshot-research.yaml \
  --elements-dir backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/Elements \
  --approved-manifest backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/snapshot_manifest.json \
  --metadata-csv backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/metadata.csv \
  --backbone-manifest backend/training_corpus/backbone-pins/dinov2-vits14-local.json

backend/.venv/bin/python scripts/benchmark_model_replacement_research.py \
  --features backend/model_registry/versions/<candidate-id>/training_data/precomputed/features.pt \
  --split-manifest backend/training_corpus/snapshots/snapshot-471c74f801b749a9ad6e/snapshot_manifest.json \
  --runtime-dir backend/model_registry/versions/<registered-original-id> \
  --candidate-dir backend/model_registry/versions/<candidate-id> \
  --eval-split locked_test --prototype-mode both \
  --output reports/p4-model-replacement-research.json
```

## P4 preregistered historical warm-start (B0/B1)

The next bounded comparison is fixed before execution and remains
research/non-promotion. The historical raw projection state dict has exactly
the four `ProjectionHead` tensors and SHA-256
`0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210`.
Training now accepts that raw state dict (or a checkpoint containing
`model_state_dict`) only after exact key, shape, and finiteness validation. Its
resolved path, file hash, and source format are bound into every checkpoint,
the training manifest, and candidate export provenance.

The preregistered arms are:

- **B0 anchor:** `snapshot-anchor.yaml`, descriptive global dev evaluation
  only (`epoch=-1`), with no optimizer or scheduler. `best.pt` and `latest.pt`
  are byte-identical aliases of the historical initialization.
- **B1 warm-start:** `snapshot-warmstart.yaml`, exactly 3 epochs × 100
  episodes, 20-way, 3-shot, 5-query, learning rate `1e-5`, weight decay
  `1e-4`, no warmup, seed 42. Final selection is fixed to `latest`; dev must
  not be used to tune this recipe.

The fixed policy lives at `training.checkpoint_selection`. Both launchers read
it with safe YAML parsing: an omitted CLI option adopts the recipe value (or
`best` for older configs), while a contradictory CLI override is rejected.
Training checkpoints also persist optimizer, cosine-scheduler, and warmup-
scheduler state where applicable so a resume cannot silently restart the
learning-rate schedule.

These arms address the locked random-initialization research result (8.54%
top-1) while retaining the two historical reference views (84.15%
train-recomputed and 90.24% artifact/package replay). Those values motivate
the experiment; they are not selection thresholds. `locked_test` remains
excluded from fitting, hyperparameter choice, and checkpoint selection.

```bash
# B0: no fitting. The init file is validated and hashed by train.py.
DEVICE=cuda MODEL_VERSION_ID=snapshot-anchor-local bash scripts/retrain.sh \
  --config backend/codex_pipeline/config/snapshot-anchor.yaml \
  --elements-dir backend/training_corpus/snapshots/<snapshot-id>/Elements \
  --approved-manifest backend/training_corpus/snapshots/<snapshot-id>/snapshot_manifest.json \
  --metadata-csv backend/training_corpus/snapshots/<snapshot-id>/metadata.csv \
  --backbone-manifest backend/training_corpus/backbone-pins/dinov2-vits14-local.json \
  --init-projection backend/codex_model/weights/projection.pt \
  --eval-only --checkpoint-selection best

# B1: fixed final-epoch selection; no dev-based best-checkpoint selection.
DEVICE=cuda MODEL_VERSION_ID=snapshot-warmstart-local bash scripts/retrain.sh \
  --config backend/codex_pipeline/config/snapshot-warmstart.yaml \
  --elements-dir backend/training_corpus/snapshots/<snapshot-id>/Elements \
  --approved-manifest backend/training_corpus/snapshots/<snapshot-id>/snapshot_manifest.json \
  --metadata-csv backend/training_corpus/snapshots/<snapshot-id>/metadata.csv \
  --backbone-manifest backend/training_corpus/backbone-pins/dinov2-vits14-local.json \
  --init-projection backend/codex_model/weights/projection.pt \
  --checkpoint-selection latest
```

Neither command mutates `backend/codex_model`; each writes an immutable
candidate and leaves promotion as a separate, still-blocked action.

The preregistered runs are now complete. B0 is exactly tied with the
historical head under the comparable train-recomputed protocol (84.15% top-1,
91.46% top-3). B1 produces the same held-out metrics and is stopped. The
historical shipped prototypes replay at 90.24% top-1, but their training-data
provenance is unavailable and the 82-row evaluation is 87.8% `chimalli`.
Accordingly, B0 is retained as the auditable retrainable baseline, while the
historical runtime remains the product oracle and rollback. The complete P4
decision and causal analysis are recorded in
`docs/p4-s14-retrainable-baseline-20260811.md`.

## Council and independent analysis

The clean read-only Argos Council session is `adv_20260810T214742_6a231897`
(Opus and GLM 5.2). Its synthesis is stored at
`C:\Users\anmou\.argos\workspace\.omx\argos-council\codex-subagent-panel\turns\006\synthesis.md`.
The council and two native read-only scouts converged on the same priority:
freeze the data/split/class-order contract before spending compute on a new
model. The main disagreement was sequencing provenance commits; no commit was
created because none was requested.

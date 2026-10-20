# P4 — S/14 retrainable baseline (2026-08-11)

## Decision

The historical DINOv2 ViT-S/14 runtime remains active, immutable, and available
as the product oracle and rollback target. The scientific retrainable baseline
is B0: the historical projection loaded exactly, zero gradient updates, and
prototypes deterministically recomputed from the immutable snapshot's `train`
rows.

B1 is stopped. Three fixed low-learning-rate epochs did not improve B0. No
blend or shrinkage coefficient will be selected from the current `dev` or
`locked_test` partitions.

No candidate was promoted and `backend/codex_model` was not modified.

## Executed candidates

| Candidate | Projection | Training | Prototype source | Locked top-1 | Locked top-3 |
| --- | --- | --- | --- | ---: | ---: |
| Historical runtime | historical | historical/unknown | shipped historical artifact | 90.24% | 95.12% |
| Historical fair protocol | historical | none | current snapshot `train` | 84.15% | 91.46% |
| B0 `20260811-s14-snapshot-anchor-b0` | byte-identical historical | zero update | current snapshot `train` | 84.15% | 91.46% |
| B1 `20260811-s14-snapshot-warmstart-b1` | historical warm-start | 3 x 100 episodes, LR `1e-5` | current snapshot `train` | 84.15% | 91.46% |

B0 and the historical runtime have exactly zero delta when both use prototypes
recomputed from the same train rows. B0's exported projection is byte-identical
to the historical projection. B1's finite projection drift has squared L2 norm
`0.0075883189329`, but its held-out predictions do not improve.

The reports are generated evidence under `reports/`:

- `p4-s14-anchor-b0-locked-test.json`;
- `p4-s14-warmstart-b1-locked-test.json`;
- `p4-s14-research-r1-locked-test.json` for the rejected random-head run.

All three candidates remain registry candidates with
`promotion.blocked=true`.

## Causal diagnosis

Backbone, class mapping, projection, prototype aggregation, and tensor
normalization are not the observed cause of the remaining difference.

- The legacy and canonical taxonomies contain the same 286 unique names; the
  historical artifact is reordered bijectively by name.
- B0 uses the exact historical projection SHA-256 `0ad6ce348eaf44c...`.
- B0's packaged prototypes and train-recomputed prototypes agree to numerical
  precision.
- All five historical-correct/B0-wrong changes are `chimalli`; there are no
  gains in the opposite direction.
- `chimalli` has only 2 train examples but 72 locked-test examples. Its
  historical/B0 prototype cosine is `0.806`, while the other six observed
  locked classes range from `0.967` to `0.998`.
- B0 and B1 prototypes have mean cosine `0.99990`, so further head fine-tuning
  is not the useful next lever.

The historical manifest contains no training-data provenance (`data: {}` and
`training.command: import_current_runtime`). Its shipped prototypes may encode
legitimate lost training data, overlap with today's holdout, or both. The
available artifacts cannot distinguish those hypotheses. They therefore
remain useful for product fidelity, but they are not imported as the source of
truth for the reproducible training pipeline.

## Evaluation limitation

The current partitions are valid for leakage checks but not for model
selection or promotion:

- `locked_test`: 82 examples, 7 classes, 4 source groups; `chimalli` is 72/82
  examples (87.8%);
- `dev`: 56 examples, 8 classes, also dominated by a rare class with only two
  train supports;
- promotion contract minimum: 600 examples and 100 classes.

Consequently, the historical artifact's +6.10 percentage-point top-1 result
is a fidelity observation, not attributable evidence of better
generalization. It must not be used to choose a blend coefficient or training
recipe.

## Next production gate

The next work is on the evaluation instrument, not a larger backbone or more
head epochs:

1. report reference-model provenance explicitly and mark the historical
   reference as unverified for generalization claims;
2. distinguish shipped-artifact fidelity from comparable train-only model
   evaluation;
3. reject promotion evidence with fewer than 20 source groups, with any class
   above 25% of examples, or with any source group above 25%; these safeguards
   are predeclared conservative limits, not thresholds fitted on `dev` or
   `locked_test`;
4. acquire new independent source pages for rare classes;
5. if a single 600/100 group-disjoint holdout is impossible with the corpus,
   preregister repeated grouped cross-validation for model selection and keep
   the locked test as a final one-shot sanity check.

Until those gates pass, the historical runtime can continue to serve the
application, but the replacement model is not production-ready.

The benchmark now emits the reference manifest's `data` and `training`
sections, labels empty evidence `unverified`, and treats merely non-empty
sections as documented rather than scientifically verified. Artifact-mode
deltas are always marked as behavior/packaging fidelity and are never eligible
as generalization evidence. Structural and coverage failures are reported as
separate blocking reasons.

## Council

The read-only Argos Council was re-seated as
`adv_20260811T003335_9fcb766a`, with artifacts under
`C:\Users\anmou\.argos\sessions\adv_20260811T003335_9fcb766a`.
Opus completed and recommended B0 as the only fully auditable baseline.
GLM-5.2 ran through OpenCode `--pure` with the `argos-reader` agent but timed
out after 1,500 seconds without returning content; it was not retried.

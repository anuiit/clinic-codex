# S/14 autoresearch v3 — train-only row CV (2026-08-12)

> **Superseded evidence:** this campaign used the historical projection SHA `0ad6ce34…` rather than the B1 runtime projection SHA `52507c66…`. Its scores are preserved for audit only and must not be described as B1 results. The corrected replay is V5.

This is a research-only, in-page row-stratified experiment. It does not estimate source-page generalization and it does not promote or package a model.

- Frozen specifications: 47; completed screening variants: 47.
- Physical sealed cache: 9128 train rows, 286 classes; no image paths or non-train rows.
- Screening: deterministic stratified 3-fold CV, seed 17. The top eight advance regardless of the preregistered screening gate.
- Confirmation: deterministic stratified 5-fold CV; paired bootstrap resamples class deltas.
- Final multi-seed confirmation (17/42/73) runs only when at least one candidate clears the outer gate.
- Verdict: `PLATEAU_NO_RESEARCH_CHALLENGER`.

| # | Method | Macro top-1 | Low support | High support | Screening gate | Status |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | control-b1-ncm | 0.9608 | 0.9828 | 0.9576 | control | COMPLETED |
| 1 | raw-ncm | 0.3875 | 0.6640 | 0.3476 | false | COMPLETED |
| 2 | simpleshot-b1 | 0.9607 | 0.9828 | 0.9575 | false | COMPLETED |
| 3 | simpleshot-raw | 0.3907 | 0.6971 | 0.3466 | false | COMPLETED |
| 4 | power-b1-p025 | 0.9593 | 0.9828 | 0.9559 | false | COMPLETED |
| 5 | power-b1-p050 | 0.9611 | 0.9828 | 0.9580 | false | COMPLETED |
| 6 | power-b1-p075 | 0.9618 | 0.9828 | 0.9587 | false | COMPLETED |
| 7 | power-raw-p025 | 0.4014 | 0.6825 | 0.3609 | false | COMPLETED |
| 8 | power-raw-p050 | 0.3937 | 0.6710 | 0.3538 | false | COMPLETED |
| 9 | power-raw-p075 | 0.3910 | 0.6687 | 0.3510 | false | COMPLETED |
| 10 | zscore-b1 | 0.9613 | 0.9828 | 0.9582 | false | COMPLETED |
| 11 | zscore-raw | 0.3961 | 0.7110 | 0.3508 | false | COMPLETED |
| 12 | pca-whiten-b1-e001 | 0.9596 | 0.9828 | 0.9563 | false | COMPLETED |
| 13 | pca-whiten-b1-e010 | 0.9609 | 0.9828 | 0.9578 | false | COMPLETED |
| 14 | pca-whiten-raw-e001 | 0.4981 | 0.8006 | 0.4545 | false | COMPLETED |
| 15 | pca-diag-b1-e001 | 0.9613 | 0.9828 | 0.9582 | false | COMPLETED |
| 16 | pca-diag-b1-e010 | 0.9607 | 0.9828 | 0.9575 | false | COMPLETED |
| 17 | coord-median-b1 | 0.9586 | 0.9643 | 0.9578 | false | COMPLETED |
| 18 | geomedian-b1 | 0.9612 | 0.9828 | 0.9581 | false | COMPLETED |
| 19 | trimmed-b1-p010 | 0.9611 | 0.9828 | 0.9580 | false | COMPLETED |
| 20 | trimmed-b1-p020 | 0.9612 | 0.9828 | 0.9580 | false | COMPLETED |
| 21 | tukey-biweight-b1-c4685 | 0.9586 | 0.9643 | 0.9578 | false | COMPLETED |
| 22 | centroid-shrink-b1-l025 | 0.9608 | 0.9828 | 0.9576 | false | COMPLETED |
| 23 | centroid-shrink-b1-l1 | 0.9608 | 0.9828 | 0.9576 | false | COMPLETED |
| 24 | centroid-shrink-b1-l4 | 0.9608 | 0.9828 | 0.9576 | false | COMPLETED |
| 25 | centroid-shrink-b1-l8 | 0.9608 | 0.9828 | 0.9576 | false | COMPLETED |
| 26 | ridge-cos-b1-l00001 | 0.9101 | 0.9292 | 0.9074 | false | COMPLETED |
| 27 | ridge-cos-b1-l0001 | 0.9192 | 0.9315 | 0.9174 | false | COMPLETED |
| 28 | ridge-cos-b1-l001 | 0.9220 | 0.9084 | 0.9239 | false | COMPLETED |
| 29 | ridge-cos-b1-l01 | 0.9133 | 0.9041 | 0.9146 | false | COMPLETED |
| 30 | ridge-cos-b1-l1 | 0.9226 | 0.9620 | 0.9170 | false | COMPLETED |
| 31 | ridge-cos-b1-l10 | 0.9568 | 0.9782 | 0.9537 | false | COMPLETED |
| 32 | ridge-cos-b1-l100 | 0.9606 | 0.9828 | 0.9574 | false | COMPLETED |
| 33 | knn-b1-k1 | 0.9663 | 0.9828 | 0.9639 | true | COMPLETED |
| 34 | knn-b1-k3 | 0.9450 | 0.8810 | 0.9542 | false | COMPLETED |
| 35 | knn-b1-k5 | 0.9336 | 0.7860 | 0.9548 | false | COMPLETED |
| 36 | knn-b1-k9 | 0.8865 | 0.4038 | 0.9560 | false | COMPLETED |
| 37 | knn-b1-k15 | 0.8396 | 0.1445 | 0.9396 | false | COMPLETED |
| 38 | knn-k15-beta2 | 0.9098 | 0.6005 | 0.9543 | false | COMPLETED |
| 39 | knn-k15-beta5 | 0.9454 | 0.8654 | 0.9569 | false | COMPLETED |
| 40 | knn-k15-beta10 | 0.9634 | 0.9782 | 0.9612 | false | COMPLETED |
| 41 | mahal-global-diag-s01 | 0.9614 | 0.9828 | 0.9583 | false | COMPLETED |
| 42 | mahal-global-diag-s05 | 0.9613 | 0.9828 | 0.9583 | false | COMPLETED |
| 43 | fecam-diag-s05 | 0.9559 | 0.9788 | 0.9526 | false | COMPLETED |
| 44 | relu-ridge-m256 | 0.9612 | 0.9828 | 0.9581 | false | COMPLETED |
| 45 | relu-ridge-m512 | 0.9615 | 0.9828 | 0.9585 | false | COMPLETED |
| 46 | relu-ridge-m1024 | 0.9627 | 0.9828 | 0.9598 | false | COMPLETED |

Conceptual method families: nearest-class means and SimpleShot centering; power normalization; robust location estimators; shrinkage; ridge regression; cosine nearest neighbours; diagonal covariance classifiers; deterministic random-feature ridge. Implementations are original to this harness.

## Outer confirmation

The matched five-fold control reached `0.962336` macro top-1. Only cosine 1-NN retained a positive class-paired bootstrap interval: `0.968688`, a gain of `0.006353` with 95% interval `[0.003283, 0.009534]`. It did not reach the preregistered `+0.01` promotion threshold. The other seven finalists had smaller gains or intervals crossing zero. Consequently no multi-seed confirmation was triggered and B1 NCM remains the research incumbent.

This high row-CV score must not be confused with new-page performance: rows from the same page population occur on both sides of these folds. The experiment is useful for rejecting classifier heads, not for claiming production generalization.

## Conceptual references

- Wang et al., [A Simple Baseline for Few-shot Image Classification](https://arxiv.org/abs/1911.04623) (SimpleShot centering and normalization).
- Mensink et al., [Metric Learning for Large Scale Image Classification](https://doi.org/10.1007/978-3-642-33709-3_28) (nearest class means).
- Goswami et al., [FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning](https://arxiv.org/abs/2210.01076) (covariance-aware class scoring).
- Rahimi and Recht, [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) (deterministic random-feature candidates).

Machine-readable evidence: `reports/autoresearch-s14-v3-20260812/results.json`.

# S/14 autoresearch v5 — true B1 replay (2026-08-12)

V5 replaces the invalid B1 interpretation of V3/V4. It fail-closes on the actual B1 runtime projection path and SHA-256 before loading any projection tensor.

- True B1 projection SHA-256: `52507c66ae568380c150a4ff94da78c41f4d07848c46cb366ba5a60c9f716d03`.
- Frozen matrix: 77 variants (47 V3 + 30 adaptive V4), all screened on the physical train-only cache.
- Outer: top 10 plus the NCM and cosine 1-NN baselines.
- Confirmation: top 2 outer challengers plus both baselines, five folds on fresh seeds 42/73/101; selection seed 17 is excluded.
- Balanced logloss is descriptive only and is never used to gate unlike scoring methods.
- This is in-page row CV and is not evidence of new-page or business-set generalization.
- The winning cache/fallback method is explicitly `runtime_compatible=false`; this result authorizes neither packaging nor production use.
- Verdict: `TRUE_B1_RESEARCH_CHALLENGER`.

## Multi-seed confirmation

| Method | Macro top-1 | Macro top-3 | Micro top-1 | Balanced logloss | Gain vs B1 NCM | CI95 vs B1 NCM | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v3-control-b1-ncm | 0.962082 | 0.977936 | 0.943178 | 0.357458 | +0.000000 | [+0.000000, +0.000000] | false |
| v4-baseline-cosine-k1 | 0.969869 | 0.970102 | 0.958772 | 0.314163 | +0.007787 | [+0.004843, +0.011121] | false |
| v4-margin-fallback-002 | 0.971119 | 0.977134 | 0.959064 | 0.243985 | +0.009037 | [+0.006488, +0.011835] | true |
| v4-margin-fallback-005 | 0.970180 | 0.979484 | 0.957055 | 0.230963 | +0.008098 | [+0.005901, +0.010509] | true |

The historical-projection V3/V4 reports remain preserved solely as an audit trail; their method rankings motivated V5 but their claimed B1 metrics are superseded.

Machine-readable evidence: `reports/autoresearch-s14-v5-true-b1-20260812/results.json`.

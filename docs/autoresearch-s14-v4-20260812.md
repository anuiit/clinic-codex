# S/14 autoresearch v4 — adaptive neighbourhood phase (2026-08-12)

> **Superseded evidence:** this follow-up inherited the historical projection SHA `0ad6ce34…`, not the B1 runtime projection SHA `52507c66…`. Its scores are preserved for audit only and its B1 interpretation is invalid. The corrected replay is V5.

V4 is a separate, preregistered follow-up to the positive cosine 1-NN signal in V3. It reads only the physical train-only cache and remains an in-page row-CV experiment; it does not estimate new-page generalization.

- Frozen variants: 30; screening completed: 30.
- Outer finalists: 5, each compared with B1 NCM and cosine 1-NN.
- Multi-seed winner: `margin-fallback-002` on seeds 17/42/73.
- Verdict: `PLATEAU_NO_V4_CHALLENGER`.

| Screening # | Variant | Macro top-1 | Low support | High support |
| ---: | --- | ---: | ---: | ---: |
| 0 | baseline-b1-ncm | 0.960750 | 0.982804 | 0.957575 |
| 1 | baseline-cosine-k1 | 0.966307 | 0.982804 | 0.963932 |
| 2 | euclidean-k1 | 0.966307 | 0.982804 | 0.963932 |
| 3 | centered-cosine-k1 | 0.966175 | 0.982804 | 0.963780 |
| 4 | power050-cosine-k1 | 0.963361 | 0.978175 | 0.961228 |
| 5 | power075-cosine-k1 | 0.964956 | 0.978175 | 0.963053 |
| 6 | zscore-cosine-k1 | 0.966384 | 0.982804 | 0.964020 |
| 7 | cosine-k2 | 0.947846 | 0.908730 | 0.953479 |
| 8 | cosine-k3 | 0.944957 | 0.880952 | 0.954174 |
| 9 | cosine-k4 | 0.934689 | 0.811508 | 0.952427 |
| 10 | cosine-k5 | 0.933572 | 0.786045 | 0.954816 |
| 11 | cosine-k7 | 0.915728 | 0.640212 | 0.955403 |
| 12 | cosine-k9 | 0.886482 | 0.403770 | 0.955992 |
| 13 | cosine-k15 | 0.839554 | 0.144511 | 0.939640 |
| 14 | kernel-k15-beta2 | 0.909781 | 0.600529 | 0.954313 |
| 15 | kernel-k15-beta5 | 0.945395 | 0.865410 | 0.956913 |
| 16 | kernel-k15-beta10 | 0.963377 | 0.978175 | 0.961247 |
| 17 | kernel-k15-beta20 | 0.966242 | 0.982804 | 0.963857 |
| 18 | hybrid-k1-alpha025 | 0.966553 | 0.982804 | 0.964213 |
| 19 | hybrid-k1-alpha050 | 0.966307 | 0.982804 | 0.963932 |
| 20 | hybrid-k1-alpha075 | 0.966307 | 0.982804 | 0.963932 |
| 21 | balanced-k31-beta5-a050 | 0.960144 | 0.982804 | 0.956881 |
| 22 | balanced-k31-beta5-a100 | 0.959543 | 0.982804 | 0.956194 |
| 23 | balanced-k31-beta10-a050 | 0.963094 | 0.982804 | 0.960256 |
| 24 | balanced-k31-beta10-a100 | 0.962348 | 0.982804 | 0.959402 |
| 25 | adaptive-k-1-3 | 0.957777 | 0.982804 | 0.954174 |
| 26 | adaptive-k-1-3-5 | 0.958299 | 0.982804 | 0.954770 |
| 27 | margin-fallback-002 | 0.967680 | 0.982804 | 0.965502 |
| 28 | margin-fallback-005 | 0.966790 | 0.982804 | 0.964484 |
| 29 | margin-fallback-010 | 0.964870 | 0.982804 | 0.962287 |

## Multi-seed confirmation

- `baseline-b1-ncm`: macro top-1 `0.961702`, macro top-3 `0.976874`, micro top-1 `0.942521`, balanced logloss `0.363119`.
- `baseline-cosine-k1`: macro top-1 `0.969286`, macro top-3 `0.969519`, micro top-1 `0.956946`, balanced logloss `0.319995`.
- `margin-fallback-002`: macro top-1 `0.971008`, macro top-3 `0.976431`, micro top-1 `0.958333`, balanced logloss `0.248538`.

Candidate bootstrap versus NCM: `{'mean': 0.009306142953962874, 'lower_95': 0.0067369065944487055, 'upper_95': 0.012062865917298986}`.
Candidate bootstrap versus 1-NN: `{'mean': 0.0017217849495329352, 'lower_95': 0.0005537000060979333, 'upper_95': 0.002974475210670969}`.

The fallback is consistently better than both baselines and its paired-class intervals exclude zero. It nevertheless misses the frozen final requirement of a mean macro gain of at least `0.002` over 1-NN: its three-seed gain is `0.001722`. It is therefore recorded as a promising research result, not a challenger eligible for packaging. Relative to B1 NCM, it gains `0.009306` macro top-1, `0.015812` micro top-1, and improves balanced logloss by `0.114582`; macro top-3 is effectively preserved (`-0.000444`).

Conceptual basis: cosine nearest-neighbour cache models, class-normalized kernel voting, prototype/cache score interpolation, and confidence-based fallback. Implementations are original to this harness. Relevant foundations include [SimpleShot](https://arxiv.org/abs/1911.04623) and [Tip-Adapter](https://arxiv.org/abs/2207.09519).

Machine-readable evidence: `reports/autoresearch-s14-v4-20260812/results.json`.

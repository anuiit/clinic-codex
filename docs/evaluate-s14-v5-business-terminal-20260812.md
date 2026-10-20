# S/14 v5 terminal business evaluator

This evaluator is the frozen, one-shot bridge between true-B1 train-only research and the immutable business test. Authoring and preflight do not accept or open business paths. The terminal `evaluate` command is reserved until independent review.

It emits exactly four methods in one process and one validated business-cache load:

1. packaged B1 artifact prototypes;
2. train-recomputed B1 nearest-class means;
3. cosine 1-NN, with top-3 ranked by each class's nearest cached example;
4. margin fallback: use the cache ranking when the two nearest training examples have a cosine gap of at least `0.02`, otherwise use B1 NCM.

The report contains per-row predictions, micro/macro top-1 and top-3, per-class and per-page metrics, and page-cluster bootstrap deltas against packaged B1 and recomputed B1 NCM. It is globally write-once and explicitly ineligible for tuning, automatic promotion, or runtime use. The cache methods require the full training embedding cache and are marked `runtime_compatible=false`.

The execution contract is frozen to CPU with batch size 4096. There is no CLI override for the spec, device, batch size, ledger, or output. The only accepted spec is the exact file `reports/evaluate-s14-v5-business-spec-20260812/frozen-spec.json`, whose file SHA-256 is `ef3d5e0e8bc0d46cafdf111fcdbcbbd8ea6cdf30dc7d04550e4832e55c191362` and whose self-hash is `51725972aec0f8fe987d85c005624394c098d25bfbf9212897cbedd44e2a2dc1`.

## Safe review sequence

Run only preflight during review:

```bash
backend/.venv/bin/python3 scripts/evaluate_s14_v5_business.py preflight
```

Preflight validates the exact B1 package/projection, the sealed 9,128-row train cache against its B1 source rows, provenance hashes, evaluator dependencies, and V5 evidence. It writes the frozen spec at `reports/evaluate-s14-v5-business-spec-20260812/frozen-spec.json` without opening business data.

After review, a human or primary agent may run `evaluate` exactly once, supplying the independently known business content SHA-256 plus the three business-file SHA-256 values. Before any business file is opened, the evaluator atomically creates `reports/evaluate-s14-v5-business-terminal-20260812/terminal-ledger.json`, keyed by those four hashes and the frozen specification hash. It then transitions from `reserved` to `started`, writes only `reports/evaluate-s14-v5-business-terminal-20260812/result.json`, and finally records `completed` plus the report hash.

Any existing ledger—whether `reserved`, `started`, or `completed`—blocks every later invocation, even one proposing different inputs. A crash therefore remains visibly reserved or started and is never retried automatically. Business results must not alter this evaluator, its margin, methods, or spec.

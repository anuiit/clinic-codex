# S14 development-only fixed sweep (2026-08-12)

This v2 evidence is research-only: all 11 runs and every evaluation use a separately sealed physical cache containing only persisted train and dev rows. No locked rows are present in that cache, and no registry/runtime artifacts are modified.

- Source cache SHA-256: `1e1ee289b2a091811b26724edd1472b62b881356ecb50ce05d0cbefe25c067d3`
- Source cache provenance SHA-256: `bd91d10f144d20dd9cca6885ac55bb52e4f59ec3e84bb4c084edeb01c4eec2d8`
- Sealed train+dev cache SHA-256: `168f7809e79439d4da90abe42348cad9a67b1abfb2fc253694d085089f264608`
- Initial projection SHA-256: `0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210`
- Subprocess environment pins `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `source_group_metric_redundant=true`: development data has eight classes and eight source groups with one class per group.
- Fixed protocol: 11 independent runs, same initial projection and seed; each candidate changes one factor from control.
- Selection: `J = .75 macro_class + .25 macro_source_group`; KEEP requires `J > control + .01` and no per-class accuracy drop greater than .20.

| Run | Changed factor | Macro class | Macro group | Micro | Balanced logloss | J | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | control | 0.7790 | 0.7790 | 0.5714 | 1.2810 | 0.7790 | CONTROL |
| lr-3e-6 | learning_rate | 0.7790 | 0.7790 | 0.5714 | 1.2882 | 0.7790 | DISCARD |
| lr-3e-5 | learning_rate | 0.7790 | 0.7790 | 0.5714 | 1.2652 | 0.7790 | DISCARD |
| epochs-1 | epochs | 0.7790 | 0.7790 | 0.5714 | 1.2855 | 0.7790 | DISCARD |
| epochs-6 | epochs | 0.7790 | 0.7790 | 0.5714 | 1.2752 | 0.7790 | DISCARD |
| wd-0 | weight_decay | 0.7790 | 0.7790 | 0.5714 | 1.2810 | 0.7790 | DISCARD |
| wd-1e-3 | weight_decay | 0.7790 | 0.7790 | 0.5714 | 1.2810 | 0.7790 | DISCARD |
| temp-0.05 | temperature | 0.7790 | 0.7790 | 0.5714 | 1.6823 | 0.7790 | DISCARD |
| temp-0.2 | temperature | 0.7790 | 0.7790 | 0.5714 | 2.4928 | 0.7790 | DISCARD |
| noise-0.01 | noise_std | 0.7790 | 0.7790 | 0.5714 | 1.2810 | 0.7790 | DISCARD |
| mixup-0.25 | mixup_prob | 0.7790 | 0.7790 | 0.5714 | 1.2741 | 0.7790 | DISCARD |

Machine-readable v2 evidence: `reports/autoresearch-s14-20260812-v2/results.json`.

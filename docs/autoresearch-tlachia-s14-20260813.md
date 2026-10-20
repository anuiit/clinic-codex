# Tlachia S/14 dev-only autoresearch (2026-08-13)

Research-only sweep on the clean-room Tlachia snapshot. The sealed cache contains train+dev only; locked_test and the business-final set are never inputs to training or selection.

- Snapshot manifest SHA-256: `5671aec6e9b603fa803535712a0cde15ac0265c75d4f7b6257d477ccc4406531`
- Sealed cache SHA-256: `0bd4a9bb6332459fa6dbb04441809c2e8e568d26bf4bb76c87b584c33815a40c`
- Historical initializer SHA-256: `0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210`
- Five classes have one train image (`cemixtlapalli`, `cozcapetlatl`, `cáliz`, `quimilli`, `tlamamallahuiztli`). In total, 86 classes have fewer than the 8 distinct train rows required by k=3/q=5; they remain outside episodic head updates but retain train-only prototypes.
- Promotion eligible: false.

| Run | Factor | Macro class | Macro group | Micro | J | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| control | control | 0.9325 | 0.9312 | 0.9312 | 0.9321 | CONTROL |
| lr-3e-6 | learning_rate | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| lr-3e-5 | learning_rate | 0.9341 | 0.9333 | 0.9333 | 0.9339 | DISCARD |
| epochs-1 | epochs | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| epochs-6 | epochs | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| wd-0 | weight_decay | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| wd-1e-3 | weight_decay | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| temp-0.05 | temperature | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| temp-0.2 | temperature | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| noise-0.01 | noise_std | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |
| mixup-0.25 | mixup_prob | 0.9325 | 0.9312 | 0.9312 | 0.9321 | DISCARD |

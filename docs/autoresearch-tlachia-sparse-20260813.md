# Tlachia sparse-class autoresearch (2026-08-13)

Research only. Inputs are the preregistered sealed train+dev cache and historical projection initializer; business-final, annotations and locked-test are excluded physically and by path guard.

| Screening run | Sparse macro (primary) | Global macro | Micro |
| --- | ---: | ---: | ---: |
| control | 0.9685 | 0.9325 | 0.9312 |
| sparse-slots-2 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-4 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-6 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-8 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-10 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-14 | 0.9685 | 0.9300 | 0.9290 |
| sparse-slots-20 | 0.9685 | 0.9292 | 0.9269 |

## Final gates

| Candidate | Pass | Sparse gain | Bootstrap lower 95% |
| --- | --- | ---: | ---: |
| sparse-slots-2 | False | 0.0000 | 0.0000 |
| sparse-slots-4 | False | 0.0000 | 0.0000 |

Promotion eligible: **false**. This campaign cannot alter registry or runtime artifacts.

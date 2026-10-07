# EXP-010 — V0 aggregation variants (v1b sum-of-scores)

Date: 2026-09-25. Separate from EXP-011 (Class-2 proxy). Not a leaderboard submission.
Scripts: `research/scripts/exp010_score_benchmarks.py`, `research/scripts/exp010_eval_aggregation.py`.
Results: `results/exp010_v0_aggregation/exp010_report.json`.

## Question
V0 scores each (query spectrum, candidate spectrum) pair, then takes the max per molecule. Does a different
**aggregation of the same V0 scores** rank the true molecule higher? No change to candidate generation or scoring.

## Variants
| ID | Candidate adducts | Per spectrum | Across a molecule_id's spectra |
|---|---|---|---|
| max_all (V0) | all | max | max |
| max_same (v1a) | query adduct only | max | max |
| sum_all | all | max | sum |
| sum_same (v1b) | query adduct only | max | sum |

## Benchmarks (labelled, local)
- **P1 Class-1 proxy**: 1,184 `enveda-np-examples` spectra / 250 molecules (same pipeline as the hidden test).
  The np-examples library is removed; the same molecules stay available via public libraries.
- **P2 visible test**: 1,213 spectra / 400 molecules; the 1,213 byte-identical train copies are removed.
  106/400 molecules have no other train spectra, so R@25 is capped around 0.73.

## Results (MRR@25; ΔMRR vs V0 with paired bootstrap 95% CI, 2,000 resamples)
| Variant | P1 MRR | P1 R@1 | P1 ΔMRR [CI] | P1 better/worse | P2 MRR | P2 R@1 | P2 ΔMRR [CI] | P2 better/worse |
|---|---|---|---|---|---|---|---|---|
| max_all (V0) | 0.862 | 0.800 | — | — | 0.321 | 0.237 | — | — |
| max_same | 0.887 | 0.824 | +0.025 [+0.011, +0.041] | 20 / 0 | 0.353 | 0.265 | +0.032 [+0.021, +0.043] | 59 / 0 |
| sum_all | 0.912 | 0.872 | +0.050 [+0.025, +0.076] | 29 / 8 | 0.342 | 0.265 | +0.021 [+0.001, +0.039] | 46 / 32 |
| **sum_same (v1b)** | **0.913** | 0.868 | **+0.051 [+0.027, +0.078]** | 33 / 7 | **0.361** | **0.285** | **+0.040 [+0.017, +0.061]** | 68 / 27 |

No variant produced a zero-candidate molecule on either benchmark. R@25 is unchanged, because aggregation
reorders candidates but cannot add them.

## Interpretation
- FACT: on two independent labelled benchmarks, sum_same beats V0, with the CI excluding 0 on both. max_same never
  worsens any molecule.
- INFERENCE: the gain only applies where V0 already has the true molecule (Class 1). If the hidden Class-1
  share is ~14% (the 0.122 LB vs 0.862 proxy estimate), the expected LB effect is about +0.005–0.007. It is small,
  because the LB gap is a Class-2/3 coverage problem (EXP-011).
- RISK (not measured locally): the same-adduct filter could empty candidate lists for the hidden test's water-loss
  adducts (`[M-H2O+H]+`, `[M-2H2O+H]+`, `[M-H2O-H]-`) when the library labels those spectra differently.
  Such molecules would receive the placeholder. Counts are recorded in the notebook's run report.

## Implementation (not pushed, not submitted)
`casmi_v0_kaggle.py` gained `AGGREGATION` (`max_all` default = V0; the V0 parity gate is enforced only for it).
`build_notebook.py --variant sum_same` writes `research/kaggle_v0b/` (kernel id `casmi-v0b-sum-same-adduct`).
Local checks: V0 still byte-identical (`4cbddddd…`) after the change. The v1b local run passes all validation
checks, reorders 395/400 rows on the visible test and changes 0 top-1 predictions (exact copies are present there).

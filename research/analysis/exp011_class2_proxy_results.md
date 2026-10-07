# EXP-011 — Class-2 proxy: results

Date: 2026-09-25. Design (fixed before outcomes): `research/analysis/exp011_class2_proxy_design.md`.
Code: `research/scripts/exp011_class2_proxy.py`, `research/scripts/exp011_pool_projection.py`.
Outputs: `results/exp011_class2_proxy/` (manifest, report.json, controls.json, b3_by_pool_size.csv, pool_projection.json).
Smoke run (integrity only): `results/exp011_class2_proxy_smoke/`. Not an LB experiment.

## Setup actually executed
- Targets: T1 = 250 `enveda-np-examples` molecules (1,184 query spectra, same pipeline as the hidden test);
  T2 = 1,000 random `enveda-180` molecules (3,617 query spectra).
- Held out from all training and evidence: 1,265 molecules (targets + parent-group salt/charge forms +
  tautomer aliases; **14 targets had an alias under another InChIKey14**). DB-only decoys: 27,454 molecules.
- Training: 744,492 spectra (+39,275 validation, molecule-disjoint) from 10 test adducts, ≤3 per (molecule, library).
- Universe: 275,810 train structures (targets included). Pools: neutral mass from adduct, ±5 ppm.

## Integrity (all zero / as expected)
| Check | Result |
|---|---|
| K1 training spectra from held-out or DB-only molecules | 0 |
| K2a query rids in training / K2b byte-identical query spectra in training | 0 / 0 |
| K3 targets missing from universe | 0 |
| K6 validation molecules held out | 0 |
| Oracle control (prediction = true fingerprint) | MRR 0.9996, R@1 0.9992 (n=1,243) |
| Permutation control, each molecule given another's prediction: B3 T1 / T2 | 0.587 → 0.342 / 0.500 → 0.139 (chance 0.362 / 0.142) |
| Permutation control B4 T1 / T2 | 0.535 → 0.328 / 0.433 → 0.150 |
| Memorisation: score − target score, non-target candidates in training vs DB-only (B3 / B4) | −0.120 vs −0.123 / −0.080 vs −0.080 |

The signal disappears under permutation and does not favour molecules seen in training.

## Candidate generation
| | T1 np (n=250) | T2 tims (n=1,000) |
|---|---|---|
| Pool recall 3 / 5 / 10 / 20 ppm | 0.872 / **0.972** / 0.996 / 0.996 | 0.864 / **1.000** / 1.000 / 1.000 |
| Pool size at 5 ppm, p10 / p50 / p90 | 2 / 9 / 44 | 8 / 50.5 / 182 |

## Ranking (5 ppm pools; unconditional = over all targets; ties → expected rank)
T1 `np` (test-like natural products):

| Ranker | MRR | MRR@25 | R@1 | R@10 | R@50 | R@100 | cond. MRR | median rank |
|---|---|---|---|---|---|---|---|---|
| B0 chance | 0.351 | 0.349 | 0.190 | 0.705 | 0.937 | 0.965 | 0.362 | 5.5 |
| B1 mass error | 0.385 | 0.383 | 0.212 | 0.763 | 0.935 | 0.965 | 0.396 | 4.0 |
| B2 fingerprint prior | 0.284 | 0.280 | 0.134 | 0.644 | 0.912 | 0.956 | 0.292 | 7.0 |
| **B3 spectrum kNN** | **0.570** | **0.570** | **0.390** | **0.888** | 0.960 | 0.968 | **0.587** | **2.0** |
| B4 MLP | 0.520 | 0.519 | 0.328 | 0.860 | 0.956 | 0.968 | 0.535 | 2.0 |

T2 `tims`:

| Ranker | MRR | MRR@25 | R@1 | R@10 | R@50 | R@100 | median rank |
|---|---|---|---|---|---|---|---|
| B0 chance | 0.142 | 0.134 | 0.050 | 0.337 | 0.753 | 0.895 | 25.75 |
| B1 mass error | 0.154 | 0.147 | 0.054 | 0.364 | 0.772 | 0.901 | 20.5 |
| B2 fingerprint prior | 0.152 | 0.143 | 0.055 | 0.358 | 0.769 | 0.908 | 19.0 |
| **B3 spectrum kNN** | **0.500** | **0.497** | **0.336** | **0.794** | **0.941** | **0.981** | **2.0** |
| B4 MLP | 0.433 | 0.430 | 0.298 | 0.711 | 0.923 | 0.971 | 4.0 |

Paired ΔMRR, bootstrap 95% CI: T1 B3−B0 +0.225 [+0.185, +0.269], B3−B2 +0.295 [+0.246, +0.340],
B4−B0 +0.173 [+0.134, +0.215]; T2 B3−B0 +0.357 [+0.335, +0.379], B4−B0 +0.291 [+0.270, +0.315].
Median Tanimoto(top-1, truth): B3 0.708 (T1) / 0.417 (T2), so wrong top-1s are often close analogues.

## Dependence on pool size (measured, B3; `b3_by_pool_size.csv`)
| T2 pool bucket | n | median pool | chance MRR | B3 MRR | B3 R@1 | B3 R@10 |
|---|---|---|---|---|---|---|
| 11–25 | 177 | 17 | 0.205 | 0.633 | 0.463 | 0.944 |
| 26–50 | 200 | 38 | 0.116 | 0.517 | 0.350 | 0.795 |
| 51–100 | 232 | 69 | 0.071 | 0.447 | 0.276 | 0.772 |
| 101–250 | 216 | 149 | 0.038 | 0.322 | 0.157 | 0.657 |
| >250 | 52 | 338 | 0.019 | 0.194 | 0.038 | 0.462 |

The target's median percentile rank stays around 3–5% of the pool. Pool size is confounded with mass and chemistry.
An analytical projection to larger pools (independent-outranking model) **failed its self-check**: it
under-predicted measured MRR by 0.12–0.21 from 9-candidate subsamples, so no projected numbers are reported.

## Compute (local, 8 logical CPUs, 15.6 GB RAM, no GPU)
build 30 min (mostly the tautomer alias search) · features 6 min · pools + B0–B2 2.4 min · B3 kNN 9 min
(4,801 queries × 784k training spectra) · B4 training 42 min (4 epochs; validation BCE still falling:
0.0701 → 0.0677 → 0.0667 → 0.0658, so not converged).

## Interpretation (pre-registered reading applied)
- FACT: on both populations, spectrum-based rankers (B3, B4) beat every spectrum-free baseline, with CIs well
  clear of 0, and the lift vanishes under permutation. By the pre-registered rule, **usable spectral signal for
  Class 2 exists**.
- FACT: a simple nearest-neighbour fingerprint transfer (B3) beats this first MLP (B4). B4 is under-trained,
  so the ordering between them is not settled.
- LIMITATION: the universe is train structures (pools median 9 for T1). Real Class-2 candidates come from COCONUT or
  PubChem, whose 5 ppm pools will be larger. MRR falls with pool size (0.63 at ~17 → 0.19 at ~340 on T2),
  so the absolute numbers here are optimistic. The size of the real pools, and whether hidden Class-2 molecules are in
  COCONUT at all, are UNKNOWN.
- LIMITATION: T1 is 250 deliberately common natural products. The hidden Class-2 molecules are by definition rarer.

## What this justifies next (proposal, not executed)
1. Obtain COCONUT (public, allowed external data) and measure, without any model: 5 ppm pool sizes for the test
   neutral masses, and what fraction of the T1 molecules and of train natural products are in COCONUT
   (reachability of Class 2).
2. Re-run B3/B4 with the COCONUT universe, which gives realistic pools.
3. Evaluate a hybrid on a mixed proxy: V0 (library) when a strong library hit exists, else Class-2 retrieval.
   The switching rule must be learned on proxy data, not on the LB.

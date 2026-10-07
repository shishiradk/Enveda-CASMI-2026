# EXP-013 report — why natural-product Class-2 ranking "collapsed"

Date: 2026-09-26. Design (locked, with one pre-result amendment for D's runtime):
`research/analysis/exp013_np_ranking_forensics_design.md`. Code: `research/scripts/exp013_np_forensics.py`.
Outputs: `results/exp013_np_forensics/` (A_targets.parquet, A_candidates.parquet, AB_report.json, C_report.json,
C_rr_per_target.parquet, D_report.json, D_*.pt, exp012_rank_corrected/). EXP-012 outputs untouched (hash-verified).
Population: 250 held-out enveda-np-examples natural products (243 with the truth in their COCONUT 5 ppm pool).

## Answer
**The dominant reason for the fall from MRR 0.587 to 0.244 was a bug in EXP-012's ranking stage, not the science.**
`coconut_fp_packed.npy` holds one fingerprint per InChIKey14 *including* the 4 RDKit parse failures. EXP-012 dropped
those 4 rows from the candidate table and then indexed the unfiltered fingerprint array with the filtered row numbers.
Every candidate after position 63,700 (≈86% of COCONUT) was scored with a neighbouring molecule's fingerprint.
When fingerprints were recomputed from SMILES for 300 random candidates, 42/300 were correct with EXP-012 indexing and
300/300 with EXP-013's. Corrected rerun (script fixed, written to `exp012_rank_corrected/`) and EXP-013's
independent implementation agree: **kNN MRR 0.502** on the same 243 targets and pools.

| COCONUT pools, 243 NP targets (corrected) | chance | mass | **kNN (B3)** | MLP (B4) |
|---|---|---|---|---|
| MRR | 0.184 | 0.200 | **0.502** | 0.393 |
| kNN R@1 / R@10 / R@50 | | | 0.340 / 0.798 / 0.963 | |

The genuine proxy → COCONUT loss is 0.587 → 0.502, **−0.085 [−0.120, −0.054]** (paired, same targets). It comes from
larger, isomer-rich pools (median 9 → 36 candidates). Affected EXP-012 numbers: only B3/B4 ranking. Pool sizes,
coverage and the chance/mass baselines did not use fingerprints and are unaffected.

## A — Error taxonomy (aggregate, all targets; thresholds fixed in advance)
161/243 targets are not ranked first; 7/250 (2.8%) fail at candidate generation.

| Category | top-1 errors (n=161) | wrong candidates in top-10 (n=2,009) | wrong candidates above truth (n=1,776) | pool base rate (n=14,861) |
|---|---|---|---|---|
| 1 same-formula near-isomer (Tc ≥ 0.5) | **37.9%** | 27.6% | 13.2% | 8.3% |
| 2 same-formula distant isomer (Tc < 0.5) | **57.1%** | 65.8% | 82.7% | 86.3% |
| 3 same scaffold, other formula | 0% | 0% | 0% | 0% |
| 4 similar, different formula | 0% | <0.1% | 0% | <0.1% |
| 5 unrelated (Tc < 0.3) | 5.0% | 6.5% | 4.2% | 5.4% |
| 7 stereochemical | 0 by construction (InChIKey14 collapses stereoisomers) | | | |

**95% of top-1 errors are constitutional isomers with the target's formula.** Near-isomers are enriched in the top-1
(38% vs an 8% base rate), so the scorer prefers chemically closer molecules but cannot finish the job.

## B — Nearest-structure analysis (Morgan r2 / 2048, Tanimoto; evaluation only)
- **Fingerprint oracle** (prediction = true fingerprint) in the same pools: **MRR 0.996, R@1 0.992**. The candidate
  structures are separable at this representation. The error lies in the spectrum → structure prediction.
- Median Tc(truth, wrong top-1) 0.33 (IQR 0.18–0.62), but that top-1 sits at the 90th percentile of pool similarity
  to the truth. Median best wrong pool member Tc 0.62; mean pool Tc 0.20.
- 95.5% of targets have ≥1 same-formula wrong candidate; 38% have one with Tc ≥ 0.70; 10.7% have one with Tc ≥ 0.85.
- Ambiguity does not drive failure: targets with a near-duplicate wrong candidate (Tc ≥ 0.85, n=26) score kNN MRR
  0.583 vs 0.492 for the rest.

## C — Reference-library test (kNN rule unchanged; R_full reproduces EXP-011 B3 exactly, max |Δ| = 0.0)
| Reference set | spectra | % enveda-180 | MRR | R@1 | R@10 | Δ vs full [95% CI] |
|---|---|---|---|---|---|---|
| R_full | 744,492 | 60% | 0.502 | 0.340 | 0.798 | — |
| **R_noE180** | 298,095 | 0% | **0.543** | 0.383 | 0.831 | **+0.041 [+0.019, +0.065]** |
| R_NPlib | 177,824 | 0% | 0.517 | 0.358 | 0.827 | +0.015 [−0.013, +0.045] |
| R_COCONUTmol | 126,133 | 0% | 0.528 | 0.370 | 0.823 | +0.026 [−0.003, +0.057] |
| R_COCONUTmol 50% | 62,968 | 0% | 0.496 | 0.348 | 0.794 | −0.006 |
| R_COCONUTmol 25% | 31,363 | 0% | 0.451 | 0.300 | 0.790 | −0.051 |
| R_sizecontrol (random, = R_COCONUTmol size) | 126,135 | 60% | 0.433 | 0.288 | 0.712 | −0.069 |

Pool coverage 97.2% and median size 36 are unchanged by construction. Same size, different chemistry:
R_COCONUTmol − R_sizecontrol = **+0.095 [+0.043, +0.144]**. More NP references help (dose-response 0.451 → 0.496 → 0.528).
Reference chemistry matters, but it is worth +0.04 to +0.10, which does not explain the isomer errors.

## D — Hard-negative two-tower ranker (small, fixed settings; 3 epochs each)
Leakage: 0 training molecules and 0 negatives among the 28,752 excluded target / alias / parent / decoy keys;
validation molecules disjoint from training; negative universe 697,535 structures.

| Model | MRR | R@1 | R@10 | R@50 | Δ vs kNN [95% CI] | same-formula targets (n=232) MRR |
|---|---|---|---|---|---|---|
| kNN (unchanged) | **0.502** | 0.340 | 0.798 | 0.963 | — | 0.484 |
| two-tower, hard negatives | 0.475 | 0.325 | 0.811 | 0.979 | −0.027 [−0.080, +0.024] | 0.453 |
| two-tower, random negatives | 0.425 | 0.292 | 0.733 | 0.971 | −0.077 [−0.133, −0.024] | 0.404 |

Hard vs random negatives: **+0.050 [+0.008, +0.090]**. Hard negatives help the learned ranker, but it does not beat kNN.
By the kNN's own top-1 category, the two-tower does better where kNN failed (distant-isomer failures 0.385 vs 0.18) and
worse where kNN succeeded (0.681 vs 1.0). This split is **selection-biased** (categories are defined by kNN's errors,
so regression to the mean favours the other model). It only suggests complementarity; it does not demonstrate it.

## Decision gate
- A (reference chemistry is the main problem): **no**. It is a real but secondary effect (+0.04 to +0.10).
- **B (pools fine; the spectrum → structure mapping fails on isomers): yes.** 97.2% coverage, median pool 36, oracle
  0.996, and 95% of top-1 errors are same-formula isomers.
- C (hard-negative ranking substantially improves isomer discrimination): **not yet**. It is better than random
  negatives, not better than kNN.
- D / E (irreducible ambiguity): **no**. The oracle is near-perfect and near-duplicate targets are not harder.

## Validity notes
- The same 243 targets have now been evaluated across EXP-011 to EXP-013. Every threshold was fixed in advance, but any
  future selection among variants needs a fresh natural-product holdout to avoid overfitting the proxy.
- These natural products are deliberately common; hidden Class-2 molecules may be rarer and less covered.
- The EXP-012 report text still states the buggy 0.244. It is left untouched as instructed; this report is the erratum.

## Strongest evidence for the next architectural change, and the next experiment
The evidence points to **the spectrum representation used to discriminate same-formula isomers**. Not model size, not
candidate generation, not ambiguity. The unchanged kNN, with a better reference set, is still the best scorer (0.543).
Proposed EXP-014, on a **fresh** held-out NP set (e.g. public-library COCONUT molecules held out in the EXP-011 manner,
plus the existing 243 as a secondary check), with all choices fixed in advance:
1. kNN representation ablation with the kNN rule fixed: fragment-only vs fragment + neutral-loss, bin width, intensity
   transform, and ModifiedCosine instead of binned cosine for neighbour search. All use R_noE180 references.
2. Pre-registered fusion (equal-weight rank average) of kNN and the hard-negative two-tower, to test complementarity
   without the selection bias above.
Build the hybrid V0 + Class-2 architecture only after that, with its switching rule learned on a mixed C1/C2 proxy, not
on the leaderboard.

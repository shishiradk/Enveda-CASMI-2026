# EXP-014 report — fragmentation-derived features for Class-2 ranking (v2)

Date: 2026-09-29. Identity: fragmentation-derived features / fragmentation-aware ranking. Independent of EXP-015
(not used for any choice). No hybrid, no Kaggle submission.
Design: `exp014_fragmentation_features_design.md` (v1) → **v1 INVALID**; amendment locked before v2 outcomes:
`exp014_fragmentation_features_v2_design.md`. Code: `research/scripts/exp014_fragmentation_features_v2.py`.
Outputs: `results/exp014_fragfeats_v2/{prep,frag,feats,train}.json, queries.pkl, fragments.pkl, pairs.parquet, models.pkl}`,
`research/analysis/exp014/v2/{eval_summary.json, per_target_results.csv}`. v1 files left untouched.

## 0. Why v1 was invalid (audit)
- **Leak**: v1 `build_ref_index` used `split .fillna("train")`, so every spectrum outside EXP-011's training.parquet,
  including all held-out spectra, became reference evidence. 1,245 targets, 1,259 H molecules, 246 truth aliases and
  11,595 DB-only decoys (132,540 spectra) were eligible. All 4,801 target query spectra were eligible. In 4/2,500 target
  groups a truth alias carried held-out spectra. v1 smoke outputs in `research/analysis/exp014/` are contaminated.
- **Design mismatch**: v1 spectral features compared the query with the candidate's *own* reference spectra, which a
  Class-2 truth never has (truth always feature-zero at evaluation). The v1 VAL comparison gave that model the truth's
  spectra but not the kNN.
- Minor, no effect: v1 `rid_spectra` sorted m/z without reordering intensities. The stored m/z are already sorted in
  100% of 20,000 sampled spectra, so there was no practical impact. v2 sorts both together.

## 1. Population
| | T1 natural products (primary) | T2 drug-like timsTOF |
|---|---|---|
| Targets / ranked (truth in 5 ppm pool) | 250 / 243 (97.2%) | 1,000 / 1,000 |
| Universe | COCONUT 2026-09 | COCONUT + train structures |
| Median pool | 37 | 64 |
| Same-formula alternatives present | 232 / 243 | 966 / 1,000 |
Training: 2,000 query spectra (EXP-011 split=train molecules). Validation: 400 (split=val). Pools: COCONUT+train at 5 ppm,
with 16,818 held-out/decoy/alias candidates removed. 168,449 candidate structures fragmented (0 failures);
624,433 query–candidate pairs, 0 NaN.

## 2. Method (v2)
In-silico fragmentation of each candidate structure (1- and 2-bond cleavages, implicit H on atoms; median 80 fragment
masses). Ion hypotheses ±H rearrangements, tolerance max(0.01 Da, 20 ppm); top-30 query peaks below precursor − 2 Da.
Features: fraction / intensity / count of query peaks explained, fraction explained by single cleavages, log fragment
count, plus a 5-feature mass/adduct block. **No reference spectra, no `has_spectra`**. Primary model LightGBM (v1
hyperparameters). Secondary: LR, fragmentation-only LGB, mass-only LGB. Molecule score = mean over its query spectra.

## 3. Results (tie-aware ranks; paired bootstrap 95% CI, 2,000 resamples, seed 20260929)
**T1 natural products, COCONUT pools (n = 243)**
| Scorer | MRR | MRR@25 | R@1 | R@10 | same-formula MRR | isomer err (registered, strict) | isomer err (tie-aware, post-hoc) |
|---|---|---|---|---|---|---|---|
| chance | 0.184 | | | | | | |
| B1 mass error | 0.200 | 0.194 | 0.088 | 0.443 | 0.164 | 0.123 | 0.496 |
| LGB mass-only | 0.188 | 0.181 | 0.076 | 0.426 | 0.158 | 0.000 | 0.500 |
| **E14 LGB (all)** | **0.345** | 0.340 | 0.215 | 0.631 | 0.321 | 0.224 | 0.249 |
| E14 LGB fragment-only | 0.348 | 0.344 | 0.219 | 0.618 | 0.325 | 0.236 | 0.260 |
| E14 LR (secondary) | 0.378 | 0.374 | 0.239 | 0.660 | 0.351 | 0.216 | 0.232 |
| kNN (EXP-011 B3) | 0.502 | 0.501 | 0.340 | 0.798 | 0.484 | 0.178 | 0.179 |

Paired deltas (T1): E14 − B1 MRR **+0.145 [+0.105, +0.187]**; E14 − mass-only LGB +0.157 [+0.119, +0.195];
fragment-only − chance +0.164 [+0.127, +0.202]; E14 − kNN MRR **−0.157 [−0.212, −0.102]**, R@1 −0.124 [−0.189, −0.055].
Isomer error: registered strict E14 − B1 +0.101 [+0.057, +0.144]; **post-hoc tie-aware E14 − B1 −0.247 [−0.288, −0.206]**;
tie-aware E14 − kNN +0.070 [+0.028, +0.112].

**T2 drug-like timsTOF, COCONUT+train pools (n = 1,000)**
| Scorer | MRR | R@1 | R@10 | same-formula MRR | isomer err strict / tie-aware |
|---|---|---|---|---|---|
| B1 mass error | 0.118 | 0.031 | 0.299 | 0.109 | 0.177 / 0.505 |
| **E14 LGB (all)** | **0.750** | **0.627** | **0.949** | 0.743 | 0.037 / 0.040 |
| kNN | 0.492 | 0.330 | 0.787 | 0.482 | 0.142 / 0.142 |
E14 − kNN MRR **+0.258 [+0.228, +0.286]**, R@1 +0.297 [+0.260, +0.334]; tie-aware isomer error −0.101 [−0.116, −0.086].

## 4. Same-formula / isomer analysis and transitions (correct ⇔ tie-aware P(rank = 1) ≥ 0.5)
| | kNN correct / E14 wrong | kNN wrong / E14 correct | both correct | both wrong |
|---|---|---|---|---|
| T1 | 52 | 23 | 31 | 137 |
| T2 | 70 | 383 | 260 | 287 |

Stratified by the kNN's top-1 error category. The categories are defined by kNN errors, so this is selection-biased: it
suggests where each helps and does not estimate size.
| T1 category (n) | kNN MRR | E14 MRR | B1 MRR | E14 better / worse than kNN |
|---|---|---|---|---|
| kNN correct (79) | 1.0 | 0.480 | 0.246 | 0 / 54 |
| distant isomer (92) | 0.180 | 0.281 | 0.139 | 49 / 39 |
| near isomer (61) | 0.355 | 0.192 | 0.151 | 10 / 47 |
| unrelated (8) | 0.221 | 0.664 | 0.533 | 7 / 1 |
| no competitor (3) | 1.0 | 1.0 | 1.0 | — |
On T2, E14 is better than kNN on 391/459 distant-isomer and 66/135 near-isomer kNN failures.

## 5. Complementarity (descriptive only; no fusion was pre-registered for EXP-014)
T1: 23 targets solved at rank 1 only by E14 and 52 only by kNN. E14 helps where the kNN's top-1 is a distant isomer or
unrelated, and is clearly worse on near-isomer confusions. T2: 383 solved only by E14 vs 70 only by kNN.

## 6. Leakage audit (v2): PASS
Train/val query molecules in exclusions (targets, EXP-011 aliases, DB-only decoys, EXP-012 truth aliases, target parent
groups; 28,752 keys): 0 / 0. Target-parent matches: 0 / 0. Val ∩ train molecules: 0. Query rids outside their EXP-011
split: 0 / 0. Held-out candidates left in train/val pools after removal: 0. No reference spectra are used anywhere.
Target spectra are used only at evaluation. COCONUT fingerprints (kNN) are parse_ok-indexed. EXP-015 is untouched and unused.

## 7. Runtime / resources (local, 8 logical CPUs, 15.6 GB)
prep 52 s · fragmentation 875 s (6 processes) · pair features 128 s · training 13 s · evaluation ≈ 60 s → ≈ 19 min.
Peak RAM well under 4 GB. No GPU.

## 8. Gate
- Registered PASS condition 1 (E14 − B1 ΔMRR CI > 0): **met** (+0.145 [+0.105, +0.187]).
- Registered PASS condition 2 (E14 strict isomer error significantly below B1): **not met** (+0.101). This criterion was
  mis-specified: mass-only scorers tie every same-formula isomer with the truth, so their strict error is ≈ 0 by
  construction (mass-only LGB 0.000) and no scorer can beat it. With ties counted as 0.5 (post-hoc correction, same
  convention as all rank metrics here) the condition would be met (−0.247 [−0.288, −0.206]).
- **EXP-014 gate: INCONCLUSIVE** by the pre-registered rule (exactly one condition met). Substantively, the evidence that
  fragmentation features carry structural, isomer-discriminating information beyond mass is strong on both populations.
  Confirming it formally needs a fresh population with the corrected, tie-aware criterion registered in advance.

## 9. What EXP-014 establishes on its own (not ranked against EXP-015)
- FACT: structure-derived in-silico fragmentation features, computable for spectrum-less Class-2 candidates, rank the
  truth far above mass alone on natural products (+0.145 MRR) and drug-like molecules (+0.633).
- FACT: on natural products they are weaker than the spectrum kNN (−0.157 MRR) and worse on near-isomer confusions. On
  drug-like timsTOF targets they are much stronger than the kNN (+0.258 MRR).
- OBSERVATION (descriptive, selection-biased split): errors partly differ from the kNN's (T1: 23 E14-only solves; T2: 383).
- CAVEATS: training queries are mostly drug-like (EXP-011 train split, ~60% enveda-180), which may explain the NP/drug-like
  gap. 2-bond cleavage does not model rearrangements. T1 is the 243-target set also used in EXP-012/013/015.

## 10. Implication for the next experiment (no hybrid built)
A pre-registered confirmation on a fresh natural-product population (one not used in EXP-011 to EXP-015), with the tie-aware
isomer criterion fixed in advance, of (a) the fragmentation model alone vs mass and kNN, and (b) a fixed-rule
combination of fragmentation and kNN scores. The complementarity seen here was descriptive only. Also worth isolating:
whether adding natural-product training queries closes the NP/drug-like gap, before any model scaling.

# EXP-014 v2 — structure-derived fragmentation features (design amendment, locked before any v2 outcome)

Date: 2026-09-29. Supersedes the invalid v1 run of `exp014_fragmentation_features_design.md` (identity unchanged:
fragmentation-derived features / fragmentation-aware ranking). v1 files are left untouched and marked INVALID.
EXP-015 is not used for any choice here. No Kaggle submission; no hybrid.

## Why v1 is invalid (audit 2026-09-27)
1. **Leak**: `build_ref_index` set `split = training.parquet split .fillna("train")`. Every train spectrum outside EXP-011's
   training.parquet (all held-out target, alias and decoy spectra) became eligible reference evidence. 1,245 targets,
   1,259 EXP-011 H molecules, 246 truth aliases and 11,595 DB-only decoys had 132,540 held-out spectra available.
   All 4,801 target query spectra were eligible references. In 4/2,500 target groups a truth alias carried held-out spectra.
2. **Design mismatch**: all v1 spectral features compare the query with the *candidate's own reference spectra*.
   A Class-2 truth has none, so the truth is always feature-zero at evaluation, while training positives always had
   spectra. The v1 VAL comparison gave the fragmentation model the truth's spectra but not the kNN. It was not paired-fair.
   The v1 smoke outputs in `research/analysis/exp014/` are contaminated by (1) and not interpretable.

## v2 question (unchanged)
Do fragmentation-derived structural features carry information that the spectrum-kNN representation does not?

## Features (computed from the query spectrum and the candidate STRUCTURE only; no reference spectra anywhere)
In-silico fragmentation (MetFrag-style, RDKit graph, implicit hydrogens kept on their atoms):
- Fragments = connected components after cleaving 1 bond, or 2 bonds (all bond pairs) when the molecule has ≤ 60 bonds
  (otherwise 1 bond only). The intact molecule is included. Fragment neutral mass F = Σ atomic masses incl. implicit H.
- Ion hypotheses: positive mode m/z = F + 1.007276 + h·1.007825, h ∈ {−2, −1, 0, +1}; negative mode m/z = F − 1.007276 +
  h·1.007825, h ∈ {−1, 0, +1, +2}. Match tolerance = max(0.01 Da, 20 ppm).
- Query peaks: top-30 by intensity among peaks with m/z < precursor − 2 Da (precursor-region peaks excluded).
Pair features (fixed): `frac_explained` (explained / used peaks), `int_explained` (explained intensity / used
intensity), `n_explained`, `frac_explained_1cut` (1-cleavage fragments only), `n_fragments_log`
(log10 of distinct fragment ions; a candidate-size control), plus the v1 mass block (`mass_diff_ppm`,
`mass_diff_mDa`, `adduct_err_ppm`, `best_adduct_err_ppm`, `adduct_is_best`).
No spectrum-to-spectrum feature and no `has_spectra` feature (it would identify train molecules vs COCONUT-only).

## Data (reuses the v1 build's query tables; its reference pickles are NOT used)
- Train / val queries: v1 `queries_train.parquet` (2,000 molecules, EXP-011 split=train) and `queries_val.parquet`
  (400, split=val); one query spectrum each (a training.parquet row). Pools: COCONUT+train ±5 ppm with every
  molecule in EXP-011 H (targets + aliases), EXP-012 truth aliases and DB-only decoys D **removed**
  (held-out structures never appear, even as negatives). The positive is kept.
- Targets: T1 np (COCONUT pools) and T2 tims (COCONUT+train pools), molecule-level pools at ±5 ppm of the median
  neutral mass, truth = key or EXP-012 alias. Target spectra are used only at evaluation.
- Leakage gates (zero tolerance): train/val query molecules ∉ H ∪ D ∪ aliases (also by parent key);
  no H/D/alias molecule in train/val pools; val molecules disjoint from train molecules; features use no reference spectra.

## Models
Primary (fixed): LightGBM, the same hyperparameters as v1 (200 trees, lr 0.05, 63 leaves, balanced, subsample 0.8,
colsample 0.8, seed 20260925), on all v2 features. Molecule score = mean predicted probability over its query spectra.
Secondary (reported, not selected on targets): LogisticRegression on the same features, and a fragmentation-only LightGBM
(no mass block).

## Comparators and analysis (all on identical target pools)
kNN = EXP-011 B3 predicted-fingerprint cosine (`pred_B3.npz`, parse_ok-indexed COCONUT fingerprints); B1 = mass error;
chance. Metrics: MRR@25, MRR, R@1, R@10 (tie-aware). Paired bootstrap 95% CIs (2,000 resamples, seed 20260929) vs kNN.
Same-formula subset; isomer error rate (share of same-formula wrong candidates scored above truth); stratification by
the kNN's top-1 error category (EXP-013 taxonomy: near isomer Tc ≥ 0.5, distant isomer, unrelated Tc < 0.3,
Morgan r2/2048). Transitions: kNN-correct/E14-wrong, kNN-wrong/E14-correct, both, neither (correct ⇔ tie-aware
P(rank = 1) ≥ 0.5). Complementarity is reported only descriptively (overlap counts). No fusion model: none was
pre-registered for EXP-014.

## Gate
Evaluated on T1 (natural products, COCONUT pools), the primary population:
- **PASS**: E14 − B1 ΔMRR CI > 0 **and** E14 isomer error rate significantly below B1's (CI of the paired difference < 0).
  Fragmentation then carries isomer-discriminating structural information beyond mass. The kNN-wrong/E14-correct count
  and the E14 − kNN ΔMRR are reported to show whether that information is *absent from the kNN* (complementary) or
  already captured.
- **FAIL**: E14 − B1 ΔMRR CI includes 0 or is < 0, i.e. no information beyond mass.
- **INCONCLUSIVE**: exactly one of the two PASS conditions holds, or any leakage gate fails.

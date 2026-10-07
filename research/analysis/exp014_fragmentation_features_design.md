# EXP-014 — Fragmentation-Aware Candidate Scoring (design, locked BEFORE outcomes)

Experiment: explicit mass-spectrometry feature engineering per (query spectrum, candidate molecule) pair,
learned with simple classifiers, tested against the EXP-011/012 spectrum-kNN (B3) baseline. Completely
independent of EXP-013 (no reproduction, no same error taxonomy) and of the V0/v1b pipeline. No Kaggle submit.

Decision rule: does explicit fragmentation-aware feature engineering provide **information the current
spectrum-kNN representation is missing**? Positive ⇒ smallest feature+model combo that beats B3; negative ⇒
recommended next representation.

## Scoping law (from readme of EXP-012)
Ranking happens over COCONUT 2026-09 InChIKey14 candidates inside a ±5 ppm mass window around the query
neutral mass (median over the query's spectra). Truth = query molecule InChIKey14 (plus parent/tautomer aliases
per EXP-012).

## Critical structural fact (drives everything)
COCONUT candidate *molecules* have NO mass spectra. Spectrum-spectrum features (Modified Cosine, fragment
overlap, neutral losses, intensity stats…) are only computable when the **candidate molecule itself has
reference spectra in the training library**. This makes candidates a mixture of:

- **spectral candidates** (train-split molecules, 229 631, all with AD10 spectra; AD10 covers 100 % of
  training.parquet) → full 19-dim feature vector;
- **cold candidates** (COCONUT-only) → mass/adduct features + `has_spectra=0` sentinel, spectral block zero.

Training pairs therefore must come from a setting where positives (and at least some negatives) are spectral,
so the classifier learns the "matching fragment evidence ⇒ positive" signal; evaluation on the real target
pools is done on both universe types so the real-data answer (spectral evidence is sparse in COCONUT pools) is
measured honestly.

## Leakage-safe data flow (reuses EXP-011/012)
- **Train pairs**: sample molecules from `training.parquet` `split=='train'` (they are ∉ H∪D∪val; targets are
  not in training.parquet, aliases/parents clean per EXP-012 L2 checks = 0). Per molecule, hold out 1 query
  spectrum; the molecule's *other* spectra are its candidate-reference evidence (positives always have ref
  evidence, query≠ref by construction).
- **Val pairs**: same construction from `split=='val'` (12 085 mols, all present in fp_index). Used for model
  selection, honest held-out MRR, and the ablation ladder.
- **Target eval**: T1_np + T2_tims targets, EXP-012 pools (`window(U.mass, M, 5)`), universe
  `COCONUT` (T1_np) and `COCONUT+train` (T2_tims), truth via `target_aliases.json`. Baselines computed
  identically: B0 chance, B1 |mass error|, B3 spectrum-kNN predicted-fingerprint cosine
  (`pred_B3.npz` + `coconut_fp_packed.npy` masked with `parse_ok`, per the EXP-012/13 bugfix note).
- Leakage guards asserted: no query molecule shares ik/parent/tautomer with val/targets; val molecules not in
  train queries; target queries only at eval.

## Feature vector (19 dims; maps 1:1 to the 12 requested categories)
Mass + adduct block (always):
1. `mass_diff_ppm` = 1e6·(M_q − M_c)/M_c                      (1. precursor mass diff)
2. `mass_diff_mDa`                                            (1.)
3. `adduct_err_ppm` = |precursor − (M_c + ad[query])| in ppm  (12. adduct-aware)
4. `best_adduct_err_ppm` = min over AD10                       (12.)
5. `is_best_adduct` = ad[query] == argmin                      (12.)
6. `has_spectra` + `n_ref_spectra`                            (guard)
Spectral block (zero when cold):
7. `mod_cos` = max over ≤3 refs of Modified Cosine (tolerance 0.1 Da, mz_power 0, intensity_power 1,
   numpy mirror validated against matchms)                     (3. Modified Cosine)
8. `frag_overlap_frac` = fraction of query top-100 fragment peaks within 0.1 Da of any ref peak
                                                                 (2. fragment m/z overlap, 4. matched peak count)
9. `int_weighted_matched_frac` = Σ matched query intensity / Σ query intensity   (5. intensity-weighted)
10. `top10_overlap_frac` (top-N similarity)                   (11. top-N peak similarity)
11. `frag_cos_binned` = cosine of 1 Da-binned fragment histograms (query vs ref) (7. fragment mass dist.)
12. `n_peaks_q`, `n_peaks_ref`, `peak_ratio`                  (8. peak-count stats)
13. `int_total_ratio`, `top5_frac_q`, `top5_frac_ref`         (9. intensity statistics)
14. `entropy_q`, `entropy_ref` (Shannon over normalized intensity)  (10. spectrum entropy)
15. `nl_overlap_frac`, `nl_int_frac` (top-20 neutral losses, 0.1 Da)  (6. neutral-loss features)

## Models
A. LogisticRegression (Scaler, class_weight balanced)
B. ExtraTrees / RandomForest (balanced)
C. LightGBM (scale_pos_weight) and XGBoost (if autotuned within budget)
D. small MLP (sklearn MLP, (64,32) relu, early stop)

Positive:negative ≈ 1:≤40 (mass-nearest = hardest) negatives per query spectrum. Failure mode fast: if
negatives in target pools are nearly all cold, report model degrades to mass-only and B3 wins on universe
coverage — that is the central factual finding either way.

## Evaluation & report
- Metrics (same rank machinery as EXP-011/012 `expected_rank_stats`): MRR@25, R@1, R@10 per molecule (query
  spectra aggregated by mean score), candidate coverage, by pool-size bins, by pop (T1_np natural-product vs
  T2_tims drug-like), hard negatives (≥2 pool members within 10 mDa), train/inference wall time.
- Ablation ladder: kNN-only (B3) → +mass → +peak-overlap → +intensity → +neutral-loss → +Modified Cosine
  → all-features; on val (full spectrum features possible) and on target pools (cold-dominated).
- Outputs: `results/exp014_fragfeats/*.parquet|json`, machine-readable copy in `research/analysis/exp014/`,
  final report `research/analysis/exp014_fragmentation_features_report.md`.

Seed 20260925 for all draws (E.AD10, E.UNIVERSE, E.FP, pred_B3 all EXP-011 artifacts reused read-only).
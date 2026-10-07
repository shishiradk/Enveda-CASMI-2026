# EXP-016 — Pre-registered confirmation of EXP-014 fragmentation features on a fresh natural-product population

Date: 2026-09-30 (locked before any EXP-016 outcome). Follow-up to EXP-014 v2 (gate INCONCLUSIVE because its registered
isomer criterion was degenerate). Script: `research/scripts/exp016_fragmentation_confirmation.py`.
Outputs: `results/exp016/`, `research/analysis/exp016/`, report `research/analysis/exp016_fragmentation_confirmation_report.md`.
EXP-014/015 outputs are read-only. The EXP-015 representation findings are **not** used: the kNN here is the unchanged
EXP-011 B3 rule used in EXP-014. No hybrid, no Kaggle submission.

## Population (fresh)
All molecules in EXP-011's DB-only set D that are in COCONUT 2026-09, have ≥ 1 spectrum in the 10 test adducts, and were
**not** sampled by EXP-015 (never an evaluation target in EXP-011 to EXP-015; excluded from EXP-014's training/val pools
as negatives). No subsampling. Queries: up to 4 random spectra per molecule (seed 20260930). Exclusions, as in EXP-015:
train-universe alias (parent group or tautomer-canonical InChIKey14) in the kNN reference set or in the fragmentation
model's training molecules; query spectra byte-identical to a kNN reference spectrum.
Candidates: COCONUT ±5 ppm of the molecule's median neutral mass; truth = key or COCONUT alias (parent/tautomer).

## Scorers (all frozen before evaluation)
- **B1** mass error; **chance**.
- **kNN**: EXP-011 B3 unchanged: EXP-011 features (0.1 Da, fragment + neutral loss, √I, top-150), references = EXP-011
  training split = train (744,492 spectra), same polarity, top-20, fingerprint prediction, cosine to COCONUT fingerprints.
- **FRAG**: EXP-014 v2 LightGBM (all features), frozen (`results/exp014_fragfeats_v2/models.pkl`); in-silico
  fragmentation and pair features exactly as EXP-014 v2.
- **FRAG-NP**: identical features and hyperparameters, retrained on 2,000 training queries drawn only from EXP-011
  split=train molecules that are in COCONUT (natural-product-like), with ≥ 2 test-adduct spectra (query = lowest-M
  spectrum, v1/v2 construction). Pools: COCONUT+train ±5 ppm minus the EXP-014 v2 exclusion set and this population.
- **COMBO**: fixed rule, score = −(0.5·rank_kNN + 0.5·rank_FRAG) (average ranks within the pool). Weight not tuned.

## Metrics
MRR, MRR@25, R@1, R@10 (tie-aware). **Isomer error rate (tie-aware)** = per target, mean over same-formula wrong
candidates of [score > truth] + 0.5·[score = truth] (registered here as the primary isomer metric). Same-formula subset;
transitions; paired bootstrap 95% CIs (2,000 resamples, seed 20260930).

## Pre-registered hypotheses / gates
- **H1 (confirms EXP-014)**: FRAG − B1 ΔMRR CI > 0 **and** FRAG − B1 tie-aware isomer error CI < 0 → PASS; neither → FAIL;
  one → INCONCLUSIVE.
- **H2 (complementarity)**: COMBO − kNN ΔMRR CI > 0 → complementarity confirmed; CI ∋ 0 → not confirmed; < 0 → harmful.
- **H3 (training composition)**: FRAG-NP − FRAG ΔMRR CI > 0 → natural-product training closes part of the gap.
Leakage gates (zero tolerance, else INVALID): no target / alias / parent in kNN references or in FRAG/FRAG-NP training
queries or pools; no target overlap with EXP-011 to EXP-015 evaluation sets; target spectra used only at evaluation.

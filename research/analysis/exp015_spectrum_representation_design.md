# EXP-015 — Spectrum representation ablation + kNN/two-tower complementarity (design, locked before outcomes)

Date: 2026-09-26. Script: `research/scripts/exp015_spectrum_representation.py`. Outputs: `research/analysis/exp015/`
(small tables), `results/exp015/` (caches). Report: `research/analysis/exp015_spectrum_representation_report.md`.
Separate from EXP-014 (fragmentation features), which keeps its own ID, code and folders. No Kaggle submission.
Corrected context: R_noE180 + current kNN = MRR 0.543 on the old 243 targets. The EXP-012 value 0.244 was a bug and is not used.

## Question
Does changing the MS/MS representation improve discrimination of same-formula natural-product isomers when candidate
pools, reference chemistry and the kNN rule are fixed, on a fresh holdout?

## Populations
- **PRIMARY (fresh)**: molecules from EXP-011's DB-only set D (spectra removed from every EXP-011 training/reference set
  and from EXP-013's two-tower training and negatives; never an evaluation target) that are in COCONUT 2026-09 and have
  ≥ 1 spectrum in the 10 test adducts. 1,000 molecules sampled with seed 20260927. Queries: up to 4 random spectra per
  molecule (seed 20260927). CAVEAT: D ∩ COCONUT has only 10 molecules with timsTOF spectra, so the queries are mostly
  public-library spectra (GNPS, RIKEN, MoNA, …), not the hidden test's instrument.
- **DEV / SECONDARY**: the 243 EXP-013 targets (enveda-np-examples, timsTOF). Used only to select levels for the
  combination stage and as a regression check. Never used to report the primary result.
- Exclusions from PRIMARY: any EXP-011/012/013 target; any target whose train-universe alias (salt/charge parent group
  or RDKit tautomer-canonical InChIKey14 among same-formula structures within ±2 mDa) appears in R_noE180 or in the
  two-tower's training molecules; any query spectrum byte-identical to an R_noE180 spectrum (target dropped if none remain).

## Fixed components
- **Reference library R_noE180**: EXP-011 training rows with split = train and ingest_lib ≠ enveda-180 (298,095 spectra).
  Identical for every variant and both populations; sha256 of the sorted rids is recorded.
- **Candidate generation**: COCONUT 2026-09 InChIKey14 structures within ±5 ppm of the molecule's median neutral mass
  (precursor_mz − Δadduct, EXP-011 table). Correct = the target's key or a COCONUT alias (key / parent / tautomer).
  Fingerprints are indexed with the parse_ok filter (the EXP-012 bug is not reproduced).
- **kNN rule** (EXP-011 B3): same-polarity references, top-20 by similarity, weights = max(sim, 0) + 1e-6,
  prediction = weighted mean of neighbour Morgan r2/2048 bits, molecule = mean over its spectra, candidate score =
  cosine(prediction, candidate fingerprint). Ties → expected rank.

## Variants (one factor changed at a time; current = fragment + neutral loss, 0.1 Da, √intensity, top-150 peaks)
| ID | change |
|---|---|
| B0 / V1 | current representation (must reproduce EXP-011 features and the 0.543 DEV result) |
| V2 | fragment-only (neutral-loss block removed). The current representation already includes neutral losses. |
| V3a / V3b | bin width 0.01 Da (finer) / 0.5 Da (coarser) |
| V4a / V4b | intensity linear / log1p(1000·I) (current = √I) |
| V5 | ModifiedCosineGreedy (tol 0.1, mz_power 0, intensity_power 1; V0's scorer) on raw peaks re-scores the top-500 references by current binned cosine (same polarity); top-20 by ModCos, weights = ModCos. The prefilter is needed for tractability. |
Neutral loss = precursor_mz − fragment m/z, kept if > 0.5 Da and < 500 Da, same bin width as fragments, placed in a
separate block. Duplicates within a bin are summed. Rows are L2-normalised. Top-150 peaks are selected by raw intensity.

## Combination stage (levels selected on DEV only)
For each factor (neutral loss on/off; bin; intensity) pick the level with the highest DEV MRR (tie → current).
C1 = best NL + best intensity · C2 = best NL + best bin + best intensity · C3 = C2 + ModCos re-scoring.
All combinations are evaluated once on PRIMARY. Every V and C result is reported; nothing is selected on PRIMARY.

## Metrics (every variant, both populations)
MRR, MRR@25, R@1/5/10/50 (tie-aware). Same-formula subset = targets with ≥ 1 same-formula wrong candidate
(RDKit CalcMolFormula), reporting MRR / R@1 / R@10. **Isomer error rate** = per target, the share of same-formula wrong
candidates scored above the truth (mean over targets). Transitions vs B0 (correct ⇔ tie-aware P(rank = 1) ≥ 0.5):
wrong→correct, correct→wrong, both correct, both wrong. Rank deltas are stratified by B0's top-1 error category
(EXP-013 taxonomy: near-isomer Tc ≥ 0.5 / distant isomer / unrelated Tc < 0.3; Morgan r2/2048 Tanimoto).
Paired bootstrap 95% CIs (2,000 resamples, seed 20260927) for ΔMRR and ΔR@1 vs B0.
Fingerprint oracle on PRIMARY (diagnostic only).

## Complementarity (secondary, pre-registered)
kNN = B0; two-tower = EXP-013 `D_hardneg.pt`, unchanged (no retraining: its training and negatives excluded D).
Fusion score = 0.5 · rank_kNN + 0.5 · rank_two-tower (average ranks within the pool; lower is better). Weight not tuned.
Report MRR, R@1, R@10, unique-correct counts, fusion fixes and regressions vs kNN, all on PRIMARY.

## Gates (a failure ⇒ INVALID)
0 target / alias / parent overlap with R_noE180 and with two-tower training · 0 query spectra in the references (rid or
hash) · 0 overlap with earlier evaluation targets · identical pools and query sets across variants (hashes) ·
deterministic B0 (recomputed twice, identical) · no PRIMARY-based selection.

## Decision gates
A: a representation materially improves same-formula discrimination (ΔMRR CI > 0 and a lower isomer error rate) → learned
encoder on that representation. B: representations weak but fusion shows complementary errors → hybrid / fusion scorer.
C: neither → reconsider the spectrum → structure mapping (fragmentation-aware modelling).

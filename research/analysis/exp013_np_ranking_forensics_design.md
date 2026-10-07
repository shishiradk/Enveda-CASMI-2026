# EXP-013 — Forensic diagnosis of the natural-product Class-2 ranking failure (design, locked before outcomes)

Date: 2026-09-26. Separate outputs: `results/exp013_np_forensics/`. EXP-011/012 outputs are read-only inputs.
Not an LB exercise; no submission.

## Fixed inputs
- Targets: the 243 EXP-011 T1 natural-product molecules whose true structure survives the COCONUT 5 ppm pool
  (EXP-012), plus the 7 that do not, for the candidate-generation category. Queries: their 1,184 np-examples spectra.
- Pools: COCONUT 2026-09, InChIKey14-keyed, ±5 ppm around the molecule's median neutral mass (EXP-012 exactly).
  A candidate is correct if it is the target's key or an alias (parent / tautomer; EXP-012 target_aliases.json).
- Scorer "kNN": EXP-011 B3, unchanged (binned fragment + neutral-loss features, cosine, same polarity, k=20,
  similarity-weighted mean Morgan fingerprint, cosine to the candidate fingerprint, molecule = mean over its spectra).
- Structure similarity (evaluation only, never a scorer input): Tanimoto on Morgan radius-2, 2048-bit
  fingerprints (RDKit), denoted Tc. Formula: RDKit `CalcMolFormula` of both SMILES. Scaffold: RDKit Bemis–Murcko
  scaffold (canonical SMILES); generic scaffold = `MakeScaffoldGeneric`.

## A — Error taxonomy (thresholds fixed now)
Each wrong candidate ranked above the truth is assigned the FIRST matching category:
1. **same-formula near-isomer**: same formula and Tc ≥ 0.50
2. **same-formula distant isomer**: same formula and Tc < 0.50 (a constitutional isomer with different chemistry)
3. **same scaffold**: same Murcko scaffold, different formula
4. **similar, different formula**: different formula, Tc ≥ 0.50
5. **unrelated**: different formula, Tc < 0.30, different scaffold
6. **other / moderate**: everything else (different formula, 0.30 ≤ Tc < 0.50, different scaffold)
7. **stereochemical**: 0 by construction (stereoisomers share an InChIKey14 and collapse to one candidate)
8. **candidate generation**: target not in its 5 ppm pool (per target, not per candidate)
Reported for top-1 errors (targets with rank > 1) and for all wrong candidates inside the top 10. The same categories
are computed for the whole pool as a **base rate**, so enrichment (scorer preference vs pool composition) is visible.

## B — Nearest-structure analysis
Per target: Tc(truth, top-1), mean Tc(truth, top-10 wrong), max Tc(truth, any pool member), mean Tc(truth, pool),
number of pool members with Tc ≥ 0.70 / ≥ 0.85, and the percentile of Tc(top-1) within the pool.
**Fingerprint oracle**: rank with prediction = the true fingerprint in the same COCONUT pools. If this oracle is high,
the fingerprint representation can separate the candidates and the problem is the prediction. If it is low, the
candidates are indistinguishable at this representation (ambiguity, decision E).

## C — Reference-library test (kNN unchanged, only the reference set changes)
All reference sets are subsets of the EXP-011 training split (already free of targets and aliases):
- R_full: all 744,492 training spectra (= B3)
- R_noE180: all libraries except enveda-180
- R_NPlib: libraries riken, gnps, massbank, mona, spectraverse, msdial, masaryk (NP-rich by description; excludes
  enveda-180, pluskal_ms2, drug_plus)
- R_COCONUTmol: training spectra whose molecule is in COCONUT (a structure-based natural-product definition)
- Dose-response: R_COCONUTmol molecule-subsamples at 25% / 50% (seed 20260926)
- Size control: random molecule-subsample of R_full with the same spectrum count as R_COCONUTmol
  (separates chemistry from reference size)
Metrics per set: MRR, MRR@25, R@1, R@10, pool coverage and size (unchanged by construction), lift over chance with
paired bootstrap 95% CI (2,000 resamples, seed 20260926).

## D — Hard-negative ranker (only after A–C)
Two-tower model, small and fixed before evaluation:
- spectrum tower: EmbeddingBag(20,000 features → 512, sum, √intensity weights) + adduct embedding → ReLU → Linear 512
- structure tower: EmbeddingBag(2,048 Morgan bits → 512) → ReLU → Linear 512
- L2-normalised; score = cosine / τ, τ = 0.1; softmax cross-entropy over 1 positive + 31 negatives
- Adam lr 1e-3, batch 256, ≤ 3 epochs, early stop on validation-molecule loss (EXP-011 val split, molecule-disjoint)
- Negatives per training spectrum, drawn from the negative universe
  (COCONUT ∪ train structures) minus H ∪ D (targets, aliases, DB-only decoys):
  8 top-Tc among same-mass (5 ppm) candidates, 8 random same-mass, 8 random from 5–50 ppm (close mass),
  7 uniform random. Missing slots are filled with uniform random negatives.
- Ablation with identical settings: 31 uniform random negatives (isolates the effect of hard negatives).
Evaluation: identical COCONUT T1 pools and metrics, molecule score = mean spectrum embedding · candidate embedding,
by A-category of the kNN's top-1 error, and on same-formula-confusion targets. Compared with the unchanged kNN.

## Leakage (checked in every stage, zero tolerance)
No target spectra in any reference or training set; no target / alias / parent-key molecule in training or in negatives;
molecule-level splits; COCONUT used only as a structure database; nothing selected on the LB.

## Amendment (2026-09-26, before any D result existed)
D's first run was stopped at 500 batches, before any validation or evaluation output: it ran about 1.6 s/batch, which projected to about 7.5 h.
Changes: popcount via `np.bitwise_count` (numerically identical, verified) and each epoch = 250,000 randomly sampled
training spectra (still ≤ 3 epochs, early stop on validation-molecule loss). Identical for the hard- and random-negative models.

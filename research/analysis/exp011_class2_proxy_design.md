# EXP-011 — Class-2 proxy: can spectra rank a known structure that has no spectra?

Date: 2026-09-25. Status: DESIGN, fixed before any EXP-011 outcome was computed. Not an LB optimisation.
Separate from EXP-010 (V0 aggregation variants).

## Question
Official Class 2 = "known structure (PubChem/COCONUT), no public spectra". V0 scores 0 there by
construction (candidates must have library spectra). Is there enough signal, using only spectra of
*other* molecules, to rank the true structure inside a mass-matched structure pool well enough to
justify scaling a structure-retrieval pipeline?

## Construction
- **Structure universe U**: all 275,810 train structures (`results/exp008_structure_universe.parquet`:
  RDKit ExactMolWt, parent key, SMILES). Targets stay in U. LIMITATION: the real Class-2 universe
  (COCONUT ~0.7M, PubChem ~10^8) is larger and different, so pools here are smaller → optimistic;
  pool-size sensitivity is reported so results can be rescaled.
- **Targets** (seeded 20260925):
  - T1 `np`: all 250 `enveda-np-examples` molecules; queries = their np-examples spectra (same
    acquisition/processing pipeline as the hidden test). Main population.
  - T2 `tims`: 1,000 random `enveda-180` molecules; queries = up to 4 random enveda-180 spectra each.
    Larger n, but drug-like chemistry (secondary population).
  - Only the 10 hidden-test adducts are used as queries.
- **Held-out set H** = targets ∪ their parent-key group (salt/charge forms) ∪ tautomer aliases
  (RDKit tautomer-canonical InChIKey14 equal to the target's, searched within ±2 mDa).
  **Every spectrum of every molecule in H is removed from training and from any library** (all libraries).
- **DB-only decoys D**: a random 10% of the remaining molecules are also removed from training but kept
  in U, so pools contain spectrum-less non-targets as in reality. Used for a memorisation-bias check.
- **Training spectra**: molecules ∉ H ∪ D, 10 test adducts, ≤3 random spectra per (inchikey14,
  ingest_lib), top-150 peaks. 5% of training molecules form a molecule-disjoint validation split.

## Candidate generation (exact)
Per spectrum `M = precursor_mz − Δ(adduct)`, with Δ from the EXP-002 table plus
`[M-H2O+H]+ −17.003289`, `[M-2H2O+H]+ −35.013854`, `[M-H2O-H]- −19.017841`. The molecule's
`M` is the median over its query spectra. Pool = {c ∈ U : |mass(c) − M| ≤ tol·M}. Primary tol = 5 ppm;
3/10/20 ppm reported for recall/size. The candidate count and target-in-pool are known before ranking.

## Rankers (fixed; molecule-level; ties → expected rank)
| ID | Uses spectrum? | Score of candidate c |
|---|---|---|
| B0 chance | no | uniform random order (exact expectation) |
| B1 mass error | no | −|mass(c) − M| |
| B2 fingerprint prior | no | cosine(p̄, fp(c)); p̄ = Morgan-bit frequency of training molecules with mass within ±5 Da of M |
| B3 spectrum kNN | yes | cosine(p̂, fp(c)); p̂ = similarity-weighted mean fp of the 20 nearest training spectra (same polarity; cosine on binned features) |
| B4 learned MLP | yes | cosine(p̂, fp(c)); p̂ = MLP(spectrum) → 2048 Morgan-bit probabilities |

Fingerprints: Morgan r=2, 2048 bits (RDKit). Spectrum features: 0.1 Da bins of fragment m/z
(0–1500) and neutral loss precursor−fragment (0–500), √intensity, L2-normalised, + adduct one-hot.
Multi-spectrum molecules: p̂ averaged over the molecule's spectra. Secondary score (reported, not
primary): Bernoulli log-likelihood with p clipped to [0.01, 0.99].
B4 architecture: sparse input → 1024 (ReLU, dropout 0.2) → 1024 → 2048 sigmoid, BCE, Adam 1e-3,
batch 512, ≤4 epochs, early stop on validation-molecule BCE.

## Metrics
Pool size (molecules), pool recall (target ∈ pool) at 3/5/10/20 ppm; per ranker: MRR (full list),
MRR@25 (official form), R@1/10/50/100, all unconditional and conditional on target ∈ pool; paired
bootstrap 95% CI (1,000 resamples, seed 20260926) for B3/B4 − B2 and − B0; Tanimoto(top-1, target);
wall-clock per stage.

## Leakage checks (zero tolerance; run automatically)
K1 no training spectrum from H ∪ D · K2 no query rid in training and no byte-identical
spectrum between queries and training · K3 targets ∈ U · K4 scorer inputs only ms2 arrays,
precursor_mz, adduct · K5 alias search done and aliases in H · K6 MLP validation molecules ∉ H.

## Smoke → full
Smoke: T1 all + T2 100, MLP on 50k spectra × 1 epoch — checks only, numbers not interpreted.
Full: as above. Decision reading (pre-registered): B3/B4 materially above B2 (CI excludes 0) on T1 →
spectral signal for Class 2 exists → scale (COCONUT universe + stronger model). B3/B4 ≈ B2 →
no usable signal with this representation; do not scale this approach as-is.

# EXP-012 report — COCONUT pools, Class-2 coverage, realistic-pool ranking

Date: 2026-09-26. Design locked beforehand: `research/analysis/exp012_coconut_pools_design.md`.
Code: `research/scripts/exp012_coconut_pools.py`. Outputs: `results/exp012_coconut/`
(build.json, pools.json, coverage.json, target_aliases.json, rank.json, rank_ci.json, per-target parquet).
No model trained or tuned; no LB used; V0/v1b Class-1 branch untouched.

## 1. Data and representation
- COCONUT 2.0 release **2026-09**, `coconut_csv_lite-09-2026.zip` (199,089,109 bytes, sha256 `75ca12f5…9614e`),
  from coconut.s3.uni-jena.de (linked on coconut.naturalproducts.net/download), downloaded 2026-09-25.
  **License CC0**; citation: Chandrasekhar et al., COCONUT 2.0, NAR 2024, gkae1063.
- 738,827 entries collapse to **479,721 InChIKey14** candidates (metric granularity). 479,717 parse in RDKit. Representative =
  lowest COCONUT identifier; its `canonical_smiles` is the SMILES that would be submitted. RDKit ExactMolWt vs CSV mass:
  median |Δ| 2.4e-6 Da, 13 entries > 1 mDa. Morgan r2/2048 fingerprints; parent key; 5,806 charged entries.
- Universes: COCONUT; COCONUT ∪ train structures (EXP-008 universe).

## 2. Pool sizes (neutral mass from adduct; molecule M = median over its spectra)
The hidden-test masses cannot be observed (the local test.parquet is a train subsample, 245–460 Da). The test-like
population is the 250 enveda-np-examples molecules.

| Query set, universe, 5 ppm | min | median | mean | p90 | p95 | p99 | max | 0 | 1–10 | 11–100 | 101–1k | >1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| np, COCONUT | 0 | 36 | 60.5 | 144 | 205 | 302 | 385 | 1.2% | 21.2% | 55.6% | 22.0% | 0% |
| tims, COCONUT | 0 | 10 | 19.3 | 49 | 65 | 106 | 479 | 6.6% | 45.2% | 46.8% | 1.4% | 0% |
| np, COCONUT+train | — | 48 | — | 179 | — | 306 | 388 | 0.8% | | | | 0% |
| visible, COCONUT+train | — | 71.5 | — | 228 | — | 500 | 539 | 0% | | | | 0% |

Tolerance effect (np, COCONUT): median pool 0 / 14.5 / 26.5 / **36** / 40 / 52.5 and zero-pool 60% / 22% / 7.2% /
**1.2%** / 0% / 0% at 1 / 2 / 3 / **5** / 10 / 20 ppm. Pools grow slowly beyond 5 ppm; below 3 ppm they lose the target.

## 3. Coverage of the Class-2 proxy targets
| | T1 np (n=250) | T2 tims (n=1,000) |
|---|---|---|
| In COCONUT by InChIKey14 | **99.6%** | 0.0% (drug-like screening compounds) |
| … incl. parent / tautomer aliases | 99.6% (0 alias-only) | 0.0% |
| Survives mass filter 1/2/3/**5**/10/20 ppm | 28.4 / 67.6 / 87.2 / **97.2** / 99.6 / 99.6% | 0% |
| Candidate-gen recall@1/10/100/1k, mass-error order only | 4.0 / 37.6 / 90.8 / 97.2% | — |

## 4. Ranking in realistic pools (EXP-011 B3 kNN and B4 MLP predictions, unchanged)
Leakage re-verified: 0 training spectra of targets or their COCONUT aliases; 0 target InChIKey14 and 0 target parent keys
among training molecules (molecule-level split); COCONUT carries no spectra; no retraining. The models saw only
spectra of molecules outside H ∪ D (744,492 training spectra; EXP-011 K1–K6 all 0).

| Universe / targets (covered, ranked) | pool median | chance MRR | mass MRR | **B3 MRR** | B3 MRR@25 | B3 R@1 | B3 R@10 | B3 R@100 | B4 MRR |
|---|---|---|---|---|---|---|---|---|---|
| EXP-011 train universe, T1 (243) | 9 | 0.362 | 0.396 | **0.587** | 0.570 | 0.401 | 0.888 | 0.968 | 0.535 |
| **COCONUT, T1 (243)** | **37** | 0.184 | 0.200 | **0.244** | 0.239 | 0.148 | 0.494 | 0.922 | 0.233 |
| COCONUT+train, T1 (244) | 50 | 0.150 | 0.165 | 0.190 | 0.183 | 0.111 | 0.373 | 0.889 | 0.178 |
| COCONUT+train, T2 (1,000) | 64 | 0.111 | 0.118 | **0.496** | 0.494 | 0.333 | 0.793 | 0.979 | 0.427 |

Paired bootstrap 95% CIs (2,000 resamples):
- COCONUT T1: B3 − chance +0.060 [+0.024, +0.096]; B3 − mass +0.044 [+0.004, +0.084]; B4 − chance +0.049 [+0.013, +0.085].
- Same T1 targets, train universe → COCONUT: B3 **−0.341 [−0.401, −0.281]**; B4 −0.300 [−0.357, −0.245].
- COCONUT+train T2: B3 − chance +0.385 [+0.363, +0.408]; train universe → COCONUT+train: −0.004 (unchanged).

B3 MRR on COCONUT T1 by pool size: 0.50 (1–10 candidates, n=49), 0.21 (11–100, n=139), 0.11 (101–1k, n=55).

## 5. What this establishes
- FACT: COCONUT covers the test-like natural products almost completely (99.6%; 97.2% after the 5 ppm filter).
- FACT: realistic pools are manageable: median 36, p99 302, max 385, none above 1k at 5 ppm.
- FACT: in those pools the current scorers keep only a small signal on natural products: +0.06 MRR over chance,
  CI excludes 0. EXP-011's 0.587 was mostly an artefact of the small, chemically mixed train-universe pool.
- FACT: the same scorers remain strong on drug-like targets even in the enlarged universe (0.496).
  INFERENCE: the scorer discriminates well among drug-like structures, the chemistry dominating its training data
  (enveda-180 is ~45% of train rows), but not among natural-product isomers.
- OBSERVATION: even uninformed ordering of a COCONUT 5 ppm pool gives MRR@25 ≈ 0.18 on covered natural products,
  where V0 scores 0 by construction. This is a design input for the hybrid, not an optimisation target here.

## 6. Decision gate
Coverage high and pools manageable, but ranking largely collapses → **C: improve the spectrum representation / scorer**
before building the hybrid.

## 7. Caveats
- T1 is 250 deliberately common natural products. Hidden Class-2 molecules have no public spectra and may be rarer,
  and Class 2 also includes PubChem-only structures. COCONUT coverage of the hidden set is therefore an upper
  estimate, and PubChem coverage was not measured.
- Hidden-test masses are unobservable locally; pool statistics use test-like proxies.
- B3/B4 were built in EXP-011 and deliberately not retrained. The B4 MLP was not converged.

## 8. Recommended next experiment (EXP-013, small, diagnostic first)
Question: why does ranking collapse on natural-product isomers: training-data domain, or representation?
1. Hardness diagnostic (no model): Tanimoto of each target to its COCONUT pool members, and the score margin to the
   top wrong candidate.
2. Domain test (cheap, reuses features): B3 with the kNN reference restricted to natural-product-rich libraries
   (all except enveda-180), on the same COCONUT T1 pools.
3. Scorer test (still small): train the fingerprint MLP with a candidate-ranking loss using same-mass COCONUT isomers
   as hard negatives. Evaluate on the same held-out T1 pools with the same leakage rules. Report the lift over
   chance and over B3.
Proceed to the hybrid V0 + Class-2 architecture only if the scorer's lift on COCONUT natural-product pools becomes
substantial.

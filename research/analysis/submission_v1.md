# CASMI 2026 Submission — V1 (kNN fingerprint retrieval over COCONUT + train structures)

Date: 2026-09-30. Code: `research/kaggle_v1/` (`casmi_v1_kaggle.py`, `build_assets.py`, `proxy_eval.py`,
`build_notebook.py`). Kaggle: notebook `shishiradhikari11/casmi-v1-hybrid-knn` (private), dataset
`shishiradhikari11/casmi-v1-assets` (private).

## Why V1
V0 (0.122) and V0b (0.124) only retrieve molecules that already have train spectra. EXP-010 put V0 at ~0.86–0.91 on a
Class-1 proxy, so the leaderboard implies roughly 14% Class-1 molecules in the hidden test. The rest is Class 2 (structure
in PubChem/COCONUT, no public spectra) or Class 3. EXP-011 to EXP-015 established a kNN fingerprint ranker over COCONUT
pools for Class 2 (EXP-015 PRIMARY: 0.01 Da bins significantly better than 0.1 Da).

## Algorithm (locked)
- **Universe**: COCONUT 2026-09 (CC0, 479,717 parse-ok InChIKey14) + 249,671 train-only structures = 729,388 structures,
  Morgan r2/2048 fingerprints (EXP-012 fingerprints reused; 300/300 recompute spot check).
- **References**: train spectra in the 10 test adducts, enveda-180 excluded (EXP-013 R_noE180: +0.041 on NP targets),
  ≤3 per (InChIKey14, ingest_lib), fixed hash order. ~353k spectra / 92k molecules on Kaggle.
- **Features**: top-150 peaks, fragment + neutral-loss bins at 0.01 Da, sqrt intensity, L2-normalised (EXP-015 V3a).
- **kNN**: top-20 cosine neighbours of the same polarity; similarity-weighted mean neighbour fingerprint per spectrum;
  mean over the molecule's spectra.
- **Candidates**: universe within ±5 ppm of the molecule's neutral mass (median over spectra of precursor − adduct shift).
  Score = cosine(predicted fp, candidate fp); top 25; SMILES from the universe row.
- **No library branch** (`tau = None`, see below). Placeholder `C` only if a molecule has no candidates.

## Local mixed proxy (`proxy_eval.py`)
250 enveda-np-examples molecules / 1,184 spectra (timsTOF natural products), molecule-level MRR@25.
S1 "Class 1": only the np-examples library is removed. S2 "Class 2": every spectrum of the target and its EXP-012
aliases is removed from library and references, and its train-only structure leaves the universe (leak gates: 0 / 0).
Caveat: this population has been used since EXP-010. Only one choice (fusion rule + intensity) was made on it.

| Rule | S1 | S2 | mix 15/85 | mix 50/50 |
|---|---|---|---|---|
| V0b library only | 0.915 | 0.000 | 0.137 | 0.458 |
| library ≥ 0.9 first, then kNN | 0.908 | 0.431 | 0.503 | 0.670 |
| library ≥ 0.7 first, then kNN | 0.910 | 0.300 | 0.391 | 0.605 |
| kNN + 0.05·library max (soft) | 0.893 | 0.494 | 0.554 | 0.693 |
| **kNN only (V1)** | 0.856 | **0.573** | **0.615** | **0.715** |
| kNN only, log intensity | 0.862 | 0.566 | 0.610 | 0.714 |
| kNN only, 0.1 Da bins | 0.864 | 0.502 | 0.556 | 0.683 |

Paired deltas vs V0b (2,000 bootstrap): S1 kNN − V0b −0.059 [−0.094, −0.024]; S2 +0.573 [+0.524, +0.620].
kNN pool coverage 97.6% (S1) / 97.2% (S2); 2/250 molecules have an empty 5 ppm window.

**Decision**: kNN only. Every library gate or boost cost far more on S2 (the library confidently returns wrong
same-formula relatives when the truth has no spectra) than it gained on S1. kNN only is best for any Class-1 share
up to 50%. Log intensity gave no gain here, so the pre-registered EXP-015 winner (0.01 Da, sqrt) is kept.

## Expected effect (hypothesis, not a measurement)
If ~14% of the hidden molecules are Class 1 and a large share of the rest are COCONUT-reachable Class 2, V1 should
move well above V0b's 0.124. The size depends on the hidden Class-2 share, on how much of it COCONUT (not PubChem)
covers, and on how representative the 250 np-examples molecules are.

## Leaderboard result (2026-09-30)
Public LB **0.237** (V0 0.122, V0b 0.124): notebook version 1, submitted by the user.
Rough reading (INFERENCE, assumes the public split resembles the whole test): if ~14% of molecules are Class 1 and V1
scores ~0.86 on them (~0.12 of the LB), the other ~86% contribute ~0.12, i.e. ~0.14 per molecule, against 0.57 on the
proxy Class 2. So only about a quarter of the non-Class-1 molecules behave like the proxy: the rest are likely Class 3,
PubChem-only Class 2 (not in COCONUT), or harder than the 250 np-examples molecules.

## Local run on the visible test
400/400 molecules ranked (median pool 72), 0 placeholders, validation passed, 52 s total. The visible test consists of
enveda-180 train spectra, which V1 excludes from its references, so its score is not informative.

## Known limitations / next steps
- Class 3 (not in PubChem) is unreachable; PubChem-only Class 2 structures are unreachable (COCONUT only).
- Candidates are only ranked by fingerprint cosine; EXP-014 fragmentation features and ModCos re-ranking are not used.
- Proxy S1 shows the library helps Class-1 molecules by ~0.06; a confidence gate that detects true Class 1 (e.g.
  near-identical spectrum match of the same adduct *and* precursor within ppm) might recover it without hurting S2.

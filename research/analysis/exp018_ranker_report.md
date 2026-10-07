# EXP-018 — learned candidate ranker (LightGBM lambdarank) over kNN-derived features

Date: 2026-10-01. Code: `research/exp018/exp018_ranker.py`. Outputs: `results/exp018/`
(`targets.parquet`, `feats_*.parquet`, `ranker_*.txt`, `train_*.json`).

## Design
One shared reference set R (V1 construction, 353,408 spectra). Each query molecule masks its own spectra:
**C2-type** masks every R row of the target and removes its train-only universe row. **C1-type** masks only the rows
from its own library. Byte-identical spectra are masked in both.

| Population | Train | Val | Notes |
|---|---|---|---|
| NP libraries, C2-type | 2,715 | 917 | gnps/riken/mona/massbank/msdial/pluskal molecules |
| NP libraries, C1-type | 285 | 83 | molecules with spectra in ≥ 2 libraries |
| **C1x timsTOF** | 900 | 300 | enveda-180 query vs other-instrument references (the test's Class-1 geometry) |

Final tests, untouched: S1 / S2 (V1 np-examples constructions).

## FACT: realistic Class-2 coverage of COCONUT + train structures is ~30%
Truth in the 5 ppm pool: **0.29–0.30** for held-out natural products (C2-type, NP libraries), versus 0.98 for the
np-examples proxy (S1/S2). This independently matches the LB decomposition in `submission_v1.md` and the COCONUT
library coverage (GNPS 37%, Pluskal 17%). **Coverage, not ranking, is the dominant loss.**

## Results (MRR@25; Δ vs V1 kNN with paired bootstrap 95% CI)

| Variant | Val (1,268) | Val C1x timsTOF | Val C2 | S1 | S2 |
|---|---|---|---|---|---|
| V1 kNN | 0.356 | 0.930 | 0.117 | 0.865 | 0.597 |
| ranker, all features | +0.017 [+0.010, +0.025] | 0.952 | 0.135 | **−0.329** | +0.063 [+0.029, +0.097] |
| no count features | +0.018 [+0.011, +0.026] | 0.954 | 0.133 | −0.213 | +0.039 |
| no counts, no source flags | +0.011 [+0.005, +0.017] | 0.947 | 0.125 | **−0.063** [−0.093, −0.033] | +0.018 [−0.007, +0.045] |

(The "all features" row was trained before the C1x queries were added; with C1x it was S1 −0.329 as shown, and before
C1x −0.377.)

## Diagnosis of the S1 failure
- **Popularity confound:** np-examples truths have ~12 reference spectra (many libraries); in C2 training, candidates
  with many references are always wrong, so `n_ref`/`nb_vote` learned "popular = wrong".
- **Source artefact:** C1x truths are train-only structures, S1 truths are COCONUT structures; C2 training teaches
  "COCONUT + own library match + top kNN = wrong analog" (a relative with spectra pulls the kNN prediction to itself).
  Source flags encode how the queries were constructed, not chemistry.
- After removing both, a residual −0.063 on S1 remains: with features that are mostly transforms of one kNN score,
  the ranker has little independent signal to combine and keeps fitting construction artefacts.

## Decision
Do **not** ship the ranker now. V1 kNN stays the core score. Revisit the ranker only when it has independent signals
(trained fingerprint network, fragmentation features from EXP-016, forward-model scores, PubChem priors), and
require **S1 non-inferiority** (Δ CI lower bound > −0.01) as a gate, besides gains on val / S2 / S3.
Priority now: coverage (PubChem through a gate, EXP-017 S3 proxy).

---

# EXP-019 addendum — trained spectrum → fingerprint MLP (CPU), 2026-10-01
Code: `research/exp019/exp019_fpnet.py`; outputs `results/exp019/`. 327,420 training spectra / 86,658 molecules
(every evaluation target, alias and parent group excluded), 0.1 Da bins + adduct embedding, MLP 20k→1024→1024→2048,
BCE, 10 epochs on CPU (val BCE 0.0718 → 0.0615; the run died in epoch 11, best checkpoint kept).

| Set | kNN (V1) | FPNet | z(kNN)+z(FPNet) | fusion − kNN [95% CI] |
|---|---|---|---|---|
| val (1,268) | 0.356 | 0.167 | 0.334 | −0.022 [−0.031, −0.015] |
| val C1x timsTOF | 0.930 | 0.300 | 0.870 | |
| val C2 | 0.117 | 0.098 | 0.114 | |
| S1 | 0.865 | 0.458 | 0.800 | −0.065 [−0.092, −0.041] |
| S2 | 0.597 | 0.454 | 0.566 | −0.031 [−0.062, −0.001] |

Decision: not used. A CPU-trained small MLP is far weaker than the kNN and fusion hurts. Revisit only with GPU training
(0.01 Da input, larger model, more epochs, contrastive or hard-negative loss).

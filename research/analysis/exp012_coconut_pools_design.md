# EXP-012 — COCONUT candidate pools, coverage and realistic-pool ranking (design, locked before outcomes)

Date: 2026-09-25. Gated experiment; no new model; V0/v1b Class-1 branch untouched; no LB use.

## Data
COCONUT 2.0, release **2026-09** (`coconut_csv_lite-09-2026.zip`, 199,089,109 bytes, sha256
`75ca12f57045c7be66d63f04238a673da1595d993e092c400bf23913ae09614e`), downloaded 2026-09-25 from
https://coconut.s3.uni-jena.de/prod/downloads/2026-09/ (listed on coconut.naturalproducts.net/download).
License: CC0 ("free use, modification, and distribution without any restrictions"). Cite COCONUT 2.0,
Chandrasekhar et al., NAR 2024, gkae1063. Stored under `external/coconut/`.

## Candidate structure DB (downstream representation)
- Key = InChIKey14 (first block of `standard_inchi_key`): the competition's match granularity
  (connectivity, no stereo). Entries sharing a key collapse to one candidate. The representative is the lowest
  `identifier`; its `canonical_smiles` is the SMILES that would be submitted.
- Per candidate: RDKit parse of that SMILES, RDKit `ExactMolWt` (primary mass, same function as the train
  universe; the CSV `exact_molecular_weight` is used only as a cross-check), Morgan r=2 2048-bit fingerprint
  (same as EXP-011), parent key (LargestFragment → Uncharger → InChIKey14), formal charge.
- Universes: **U_C** = COCONUT; **U_CT** = COCONUT ∪ train structures (EXP-008 universe), keyed by InChIKey14.

## Query sets (the hidden-test masses are not observable locally)
- `visible`: local test.parquet, 400 molecules. It is a train subsample (masses 245–460 Da), so it is reported
  but is not the hidden distribution.
- `np`: 250 enveda-np-examples molecules (same pipeline as the hidden test) = EXP-011 T1.
- `tims`: 1,000 EXP-011 T2 molecules.
Neutral mass per spectrum = precursor_mz − Δ(adduct) (EXP-011 table, 10 test adducts); molecule M = median.

## Measurements
1. Pool size at 1/2/3/5/10/20 ppm (5 ppm primary) for U_C and U_CT: min/median/mean/p90/p95/p99/max,
   % zero, % in [1,10], (10,100], (100,1k], (1k,10k], >10k.
2. Coverage on np and tims targets: target in COCONUT by (a) InChIKey14, (b) parent key, (c) tautomer-canonical
   InChIKey14 among same-formula COCONUT entries (the metric's equivalence); target survives the 5 ppm filter;
   candidate-generation recall@1/10/100/1k when candidates are ordered by |mass error| only.
3. Ranking in realistic pools: EXP-011 B3 (kNN) and B4 (MLP) predicted fingerprints are reused unchanged. Those
   models were built with every target molecule (plus parent/tautomer aliases) removed from all training
   spectra (EXP-011 K1–K6). Ranks in U_C and U_CT 5 ppm pools vs chance (B0) and mass error (B1).
   Paired comparison with the EXP-011 train-universe result on the same targets.
Correct candidate = candidate whose InChIKey14 equals the target's or belongs to its alias set (parent /
tautomer). Ties → expected rank.

## Leakage (re-verified here, zero tolerance)
L1 no training spectrum of any target/alias (re-read EXP-011 training.parquet); L2 molecule-level split: target
InChIKey14s ∩ training-molecule InChIKey14s = ∅, also by parent key; L3 COCONUT contains no spectra (structure-only);
L4 no model retraining and no tuning on these results.

## Decision gate (as specified by the user)
A coverage poor → broader structure universe · B coverage high but pools enormous → candidate generation /
compression first · C coverage high, pools manageable, ranking collapses → better spectrum representation /
scorer · D coverage and ranking strong → hybrid V0 + Class-2.

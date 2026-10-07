# Hidden-test class mix from our leaderboard scores (calibrated proxy)

Date: 2026-10-03.

Scripts:
- `research/scripts/c3_census.py`
- `research/scripts/c3_split_eval.py`
- `research/scripts/c3_visible_eval.py`
- `research/scripts/c3_calibrate.py`

Outputs: `results/c3/` (`census.json`, `split_e1.json`, `visible_eval.json`, `calibration.json`).

## 1. Training-molecule census

FACT, stored inchikey14 against the raw InChIKey14 of PubChem (100.9M CIDs) and COCONUT. Tautomers are not
reconciled, so "neither" is an upper bound.

| Set | Molecules | In COCONUT | PubChem only | Neither (real Class 3) |
|---|---|---|---|---|
| all train | 275,810 | 26,139 | 229,916 | 19,755 |
| enveda-180 (timsTOF) | 182,941 | 51 | 177,650 | 5,240 |
| enveda-np-examples | 250 | 249 | 1 | 0 |
| gnps | 45,750 | 16,938 | 18,701 | 10,111 |

The old bench S3 scenario (300 molecules, GNPS-type spectra) splits into 213 PubChem-only and 87 neither.

## 2. Bucket scores of our submissions

FACT, MRR@25. Pipelines:
- E1: honest ranker (`clean_kf`) on S1/S2, as-submitted `blend` on S3.
- V2: our kNN + PubChem engine.
- E2: fusion of E1 and V2.

The visible set is the 400 test.parquet molecules (all exact enveda-180 duplicates).

| Pipeline | Visible | S1 | S2 | PubChem-only | Neither |
|---|---|---|---|---|---|
| E1 | 1.000 | 0.863 | 0.645 | 0.188 | 0.012 |
| V2 | 0.270 | 0.855 | 0.541 | 0.280 | 0.011 |
| E2 | 0.963 | 0.892 | 0.681 | 0.226 | 0.012 |

E2 lost on the leaderboard (0.350 against E1's 0.353) because fusing in V2 pushed 29 correct first answers down
on the visible set (0.963). V2 excludes enveda-180 from its references, which is why it scores 0.270 on the visible
set.

## 3. Fitted mix

INFERENCE from 6 leaderboard readings: E1, V2, E2, E3, E3b, E4. Each reading is about 130 molecules with noise of
about ±0.016.

`LB ≈ 0.146·Visible + 0.21·S1 + 0.00·S2 + 0.11·PubChemOnly + (≈0.53 that scores ≈0 today)`

- **Fit quality:** leave-one-out RMSE 0.014, inside leaderboard noise. All six submissions are predicted within
  0.007 except V2 (predicted 0.219, observed 0.251).
- **Library hits (Visible + S1), ≈36% of the test:** already near 0.9-1.0, so headroom is about +0.03 at most.
- **COCONUT-listed, no spectra (S2), ≈0%:** isomer re-ranking of COCONUT lists cannot move the leaderboard. This
  agrees with E3b, E3c and E4.
- **PubChem-only, ≈9-13%:** E1 scores 0.19 and V2 0.28. Every +0.1 here is worth about +0.011 on the leaderboard.
- **The rest, ≈50%, scores ≈0 for every pipeline:** Class 3 plus truths no pipeline lists. Its split between Class
  3 and "unreachable" cannot be identified, because every submission scores about 0.01 on Class 3.

## 4. Decision rules from now on

1. **Proxy for every candidate change:** `0.146·V + 0.21·S1 + 0.11·PC`, with the Class-3 score reported separately
   as upside.
2. **Library protection:** never ship a change that lowers the visible-set score. Every library-hit molecule lost
   costs about 0.146/400 per molecule.
3. **Where the levers are:**
   - **(a) Gated fusion** that keeps E1 when the library match is strong and adds PubChem candidates otherwise:
     predicted +0.01-0.02.
   - **(b) PubChem-only ranking with our own nets (ho1, CFT):** up to about +0.04 if the bucket reaches 0.5.
   - **(c) Class-3 generation:** the only bucket large enough to go beyond about 0.42.

## 5. Addendum, 2026-10-03: E5 falsified the "Visible = exact duplicate" assumption

- **What happened:** E5 v2 (gated fusion) scored **0.287**, against a predicted 0.370. See DEC-008 in
  `research/decision_log.md`.
- **What it means:** the Visible bucket must be re-modelled with re-measured enveda-180 spectra before any gate is
  trusted.
- **Status of the proxy:** valid only for changes that do not depend on the library-similarity distribution of
  hidden library hits, until it is refitted with E5 as a seventh point.

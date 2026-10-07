# EXP-020 — PubChem popularity prior on top of V2

Date: 2026-10-01. Code: `research/kaggle_v3/proxy_pop.py` (stages `pop`, `coco`, `eval`). Outputs: `results/kaggle_v3_proxy/`
(`variants.csv` = full grid with CIs, `per_mol.parquet` = per-molecule RR for every variant, `truth_info.csv`, `coco_pop.csv`,
`pop_S12.parquet`, `pop_S3.parquet`). Nothing under kaggle_v0/v1/v2 was modified; nothing was submitted.

## 1. Recommendation

```
score(c) = cosine(pred_fp, fp_c) + 0.05 * [c is a universe row] + 0.015 * log1p(n_pmid_c) * [c is a PubChem-only row]
```

`n_pmid_c` = sum of `n_pmid` over the CIDs that share the candidate's InChIKey14 inside the molecule's +/-5 ppm window
(the same grouping `pubchem_windows` already does for `n_cid`). Universe rows get no popularity term.

| MRR@25 | S1 (n=250) | S2 (n=250) | S3 (n=300) |
|---|---|---|---|
| V2 (bonus 0.05, no prior) | 0.8531 | 0.5376 | 0.1984 |
| recommended | 0.8524 | 0.5366 | 0.2418 |
| difference [95% paired bootstrap CI] | -0.0007 [-0.002, 0.000] | -0.0011 [-0.002, -0.000] | +0.043 [+0.027, +0.062] |
| S3 if the truth had the PubChem minimum (1 SID, 0 PMID) | | | -0.003 [-0.007, -0.001] |
| S3 if the truth had COCONUT-typical counts | | | +0.009 [+0.004, +0.014] |

The S3 gain of +0.043 is an upper bound, not an estimate for the hidden test (section 4). The honest range for
PubChem-only truths is about -0.003 to +0.04. S1 non-inferiority holds (CI lower bound -0.002 > -0.01).

More aggressive options, same form, measured:
- `0.02 * log1p(n_pmid)`, PubChem-only rows: S1 -0.001, S2 -0.005 [-0.011, -0.001], S3 +0.059 [+0.038, +0.082]; S3 with minimum-count truth -0.006.
- `0.005 * (log1p(n_sid) + log1p(n_pmid))`, PubChem-only rows: S1 -0.001, S2 -0.007 [-0.012, -0.003], S3 +0.045 [+0.028, +0.063];
  COCONUT-typical truth +0.015; minimum-count truth -0.010 [-0.020, -0.003].

Not recommended: applying the prior to all rows (section 3.2).

## 2. Design

- Caches from `proxy_eval2.py run` (`results/kaggle_v2_proxy/cache_S{1,2,3}.pkl`): per molecule a list of
  `(ik, raw cosine, src, n_cid)`; `src` 'u' = universe (COCONUT + train), 'p' = PubChem-only. No n_sid / n_pmid there.
- Popularity (`pop` stage): molecule neutral mass recomputed as in V2 (median of precursor - adduct shift over the 10 adducts),
  one DuckDB range join per query file against `external/pubchem/pubchem_rows_pop.parquet` (100,955,275 rows; built by
  `research/kaggle_v3/build_pop.py`), grouped by (molecule, ik): `n_cid`, sum and max of `n_sid`, sum and max of `n_pmid`.
  125 s for S1/S2 (same queries), 317 s for S3. Check: every PubChem-only candidate in the caches received a popularity row
  (0 missing in S1, S2, S3), so the windows match V2. Universe candidates are looked up the same way (0 if not in PubChem
  inside the window).
- Variants (1,953 per scenario): feature in {log1p(n_sid), log1p(n_pmid), their sum, the same with max instead of sum over
  CIDs, log1p(n_cid), [n_pmid > 0]} x {raw, z-scored within the pool} x {all rows, PubChem-only rows} x tier bonus
  {0, 0.05, 0.1} x w in {0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1}. Tie-break as in V2 (score desc, ik asc).
- Baseline reproduced exactly: bonus 0 / 0.05 / 0.1 give S1 0.8378 / 0.8531 / 0.8536, S2 0.4512 / 0.5376 / 0.5653,
  S3 0.2236 / 0.1984 / 0.1761 (identical to `results/kaggle_v2_proxy/eval.json`).
- CIs: paired bootstrap over molecules, 5,000 resamples, percentile 95% interval of the MRR difference against
  V2 (bonus 0.05, w = 0). Molecules whose truth is not among the candidates count as 0 in both arms.
- Counterfactual modes for the bias question (section 4): the truth's counts are replaced before scoring.
  `cfmin`: 1 CID, 1 SID, 0 PMID (only if the truth is in PubChem). `cfcoco`: counts of a random COCONUT structure without
  train spectra (PubChem-present ones only when the truth is a PubChem-only row). `cfpool`: counts of a random PubChem
  candidate of the same window. The last two are means over 5 random draws.

## 3. Results (difference in MRR@25 vs V2; tier bonus 0.05, raw scale)

### 3.1 Prior on PubChem-only rows

For S1/S2 the truth is a universe row, so its own popularity is not used: the S1/S2 columns are free of the truth-popularity
bias (they measure only how many popular PubChem decoys overtake universe truths).

| feature | w | S1 [CI] | S2 [CI] | S3 actual [CI] | S3 cfcoco | S3 cfmin | S3 cfpool |
|---|---|---|---|---|---|---|---|
| log1p(n_pmid) | 0.005 | +0.000 [0.000, 0.000] | -0.000 [-0.001, 0.000] | +0.024 [+0.012, ...] | +0.004 | -0.001 | +0.000 |
| log1p(n_pmid) | 0.010 | -0.000 [-0.001, ...] | -0.001 [-0.002, ...] | +0.033 [+0.019, ...] | +0.008 | -0.001 | -0.000 |
| **log1p(n_pmid)** | **0.015** | **-0.001 [-0.002, 0.000]** | **-0.001 [-0.002, -0.000]** | **+0.043 [+0.027, +0.062]** | **+0.009** | **-0.003** | **-0.002** |
| log1p(n_pmid) | 0.020 | -0.001 [-0.003, 0.000] | -0.005 [-0.011, -0.001] | +0.059 [+0.038, +0.082] | +0.010 | -0.006 | -0.005 |
| log1p(n_pmid) | 0.030 | -0.002 [-0.004, ...] | -0.018 [-0.029, ...] | +0.071 [+0.048, ...] | +0.011 | -0.011 | -0.008 |
| log1p(n_pmid) | 0.040 | -0.006 [-0.013, ...] | -0.032 [-0.046, ...] | +0.083 [+0.057, ...] | +0.016 | -0.013 | -0.010 |
| log1p(n_sid) | 0.005 | -0.000 [-0.001, ...] | -0.004 [-0.006, ...] | +0.033 [+0.018, ...] | +0.013 | -0.010 | -0.004 |
| log1p(n_sid) | 0.010 | -0.002 [-0.004, -0.001] | -0.019 [-0.029, -0.010] | +0.055 [+0.037, +0.076] | +0.022 | -0.019 | -0.006 |
| log1p(n_sid) | 0.020 | -0.007 [-0.015, ...] | -0.081 [-0.106, ...] | +0.107 [+0.080, ...] | +0.042 | -0.041 | -0.019 |
| sid + pmid | 0.005 | -0.001 [-0.002, 0.000] | -0.007 [-0.012, -0.003] | +0.045 [+0.028, +0.063] | +0.015 | -0.010 | -0.004 |
| sid + pmid | 0.010 | -0.002 [-0.005, -0.001] | -0.023 [-0.034, -0.014] | +0.079 [+0.056, +0.104] | +0.029 | -0.022 | -0.008 |
| sid + pmid | 0.020 | -0.014 [-0.024, -0.005] | -0.103 [-0.131, -0.077] | +0.146 [+0.113, +0.181] | +0.046 | -0.044 | -0.023 |
| sid + pmid | 0.040 | -0.158 [-0.192, ...] | -0.272 [-0.315, ...] | +0.189 [+0.149, ...] | +0.035 | -0.090 | -0.061 |
| [n_pmid > 0] | 0.020 | -0.001 | -0.002 | +0.024 [+0.014, ...] | +0.009 | -0.002 | -0.001 |
| log1p(n_cid) | 0.020 | -0.003 | -0.023 | +0.017 [+0.007, ...] | | -0.003 | |

("..." = only the lower bound was tabulated; both bounds are in `variants.csv`.)

Reading: on the proxy the S3 gain keeps growing to +0.19, but past w ~ 0.015-0.02 it is paid for by S2 (popular PubChem
decoys overtake COCONUT truths), and the gain under COCONUT-typical truth popularity saturates near +0.05 at best.

### 3.2 Prior on all rows (universe rows included) — large on the proxy, but an artefact

| feature | w | S1 actual | S2 actual | S3 actual | S1 cfcoco | S2 cfcoco | S1 cfmin | S2 cfmin | S3 cfmin |
|---|---|---|---|---|---|---|---|---|---|
| sid + pmid | 0.0025 | +0.028 | +0.076 | +0.023 | -0.026 | -0.027 | -0.037 | -0.043 | -0.010 |
| sid + pmid | 0.010 | +0.057 [+0.036, +0.080] | +0.214 [+0.175, +0.253] | +0.063 [+0.040, +0.087] | -0.085 [-0.109, -0.062] | -0.076 [-0.102, -0.053] | -0.126 | -0.132 | -0.037 |
| sid + pmid | 0.030 | +0.055 | +0.292 | +0.111 | -0.237 | -0.204 | -0.311 | -0.326 | -0.086 |
| log1p(n_pmid) | 0.0025 | +0.017 | +0.051 | +0.010 | -0.013 | -0.015 | -0.015 | -0.019 | -0.002 |
| log1p(n_pmid) | 0.010 | +0.046 | +0.170 | +0.029 | -0.045 | -0.040 | -0.051 | -0.049 | -0.006 |

S2 would go from 0.538 to 0.75-0.83. This is the bias of section 4, not signal we can count on: the sign flips as soon as
the truth has COCONUT-typical counts.

### 3.3 Other axes

- z-scoring within the pool: no advantage. sid + pmid, z, PubChem-only rows, w = 0.005: S1 -0.000, S2 -0.003 [-0.008, -0.001],
  S3 +0.057 [+0.038, +0.078], but cfmin -0.018 [-0.029, -0.009] (worse low-popularity behaviour than raw at equal gain).
  z-scored n_pmid is worse than raw (S2 -0.031 at w = 0.01) because 97% of a pool has n_pmid = 0, so z-values are huge.
- max instead of sum over CIDs: same within +/-0.005 everywhere.
- log1p(n_cid): weak (S3 at most +0.055 at w = 0.1 with S2 -0.30), consistent with the earlier EXP-017 n_cid test.
- Tier bonus 0 with a PubChem-only prior: S2 collapses (-0.09 to -0.14); the bonus stays necessary.
- Tier bonus 0.1 with a PubChem-only prior (vs V2 at 0.05): sid + pmid w = 0.01 gives S1 +0.000, S2 +0.016, S3 +0.053,
  S3 cfcoco +0.004; n_pmid w = 0.04 gives S1 -0.001, S2 +0.008, S3 +0.058, S3 cfcoco -0.005. Positive in all three scenarios
  on the proxy, but the S3 part again relies on truth popularity (bonus 0.1 alone costs S3 -0.022). Not recommended as the
  default; listed as an option.

## 4. Bias analysis: proxy truths are far more popular than plausible hidden-test truths

Measured popularity (sum over the CIDs of the InChIKey14 inside the window):

| population | n | in PubChem | n_sid quartiles (25 / 50 / 75%) | n_pmid quartiles | share with n_pmid > 0 | n_sid <= 3 |
|---|---|---|---|---|---|---|
| S1 / S2 truths (np-examples) | 244 | 96.7% | 249 / 462 / 850 | 88 / 880 / 5,166 | 96.3% | 3.3% |
| S3 truths that are candidates (gnps / riken / mona / massbank / msdial) | 202 | 99.5% | 4 / 28 / 78 | 0 / 0 / 5 | 38.6% | 21.3% |
| random COCONUT structures without train spectra | 4,000 | 91.7% | 3 / 8 / 16 | 0 / 0 / 0 | 20.7% | 30.9% |
| candidates of the same window (pool median) | | | median 2 | | 2.8-4.4% | |

- The np-examples truths sit at the median 100th percentile of their own pool for sid + pmid (10th percentile of the truths:
  99.6th pool percentile). They are textbook compounds. That alone explains the +0.2 to +0.3 on S2 with an all-rows prior.
- S3 truths: median 98th pool percentile (25th percentile of the truths: 84th). Still far above a random PubChem compound,
  and above COCONUT structures without spectra (median 28 SIDs vs 8; 39% vs 21% with PubMed links).
- In S3, base MRR does not depend on truth popularity (Spearman -0.05 with n_sid, +0.04 with n_pmid), so the prior's gain is
  carried by the popularity of the truths, not by an easier subset.

S3 gain by truth popularity (PubChem-only truths; difference vs V2):

| truth n_sid | n | V2 MRR | pmid 0.015 (rec.) | pmid 0.02 | sid+pmid 0.005 | sid+pmid 0.01 | sid+pmid 0.02 |
|---|---|---|---|---|---|---|---|
| 1 | 25 | 0.445 | 0.000 | -0.020 | -0.026 | -0.051 | -0.069 |
| 2-3 | 16 | 0.464 | 0.000 | -0.004 | +0.005 | +0.011 | +0.024 |
| 4-10 | 23 | 0.187 | -0.000 | -0.001 | +0.009 | +0.016 | +0.015 |
| 11-30 | 37 | 0.299 | +0.014 | +0.047 | +0.035 | +0.081 | +0.237 |
| 31-100 | 56 | 0.181 | +0.043 | +0.055 | +0.071 | +0.119 | +0.234 |
| > 100 | 42 | 0.297 | +0.239 | +0.320 | +0.205 | +0.355 | +0.549 |
| truth has n_pmid = 0 | 122 | 0.284 | -0.001 [-0.003, -0.000] | -0.006 [-0.016, -0.001] | +0.014 [-0.004, +0.031] | +0.024 [-0.001, +0.048] | +0.077 |
| truth has n_pmid > 0 | 77 | 0.284 | +0.171 | +0.240 | +0.154 | +0.271 | +0.448 |
| truth n_sid <= 3 | 41 | | 0.000 (no molecule changed) | -0.014 [-0.039, 0.000] | -0.014 [-0.048, +0.005] | -0.027 [-0.080, +0.008] | -0.033 |

(98 further S3 truths are not among the candidates at all and score 0 in every variant; 3 are universe rows.)

More than half of the recommended variant's S3 gain comes from the 42 truths with more than 100 substance records. For
truths with at most 3 substance records it changes nothing (0 of 41 molecules moved); for the 122 truths without PubMed
links it costs 0.001 (6 molecules slightly worse, none better). The n_sid-based variants hurt the n_sid = 1 stratum
(-0.03 to -0.07) and that is the stratum a newly described natural product most plausibly falls in.

Why n_pmid rather than n_sid for a conservative prior: 97% of the candidates in a window have n_pmid = 0, so a truth with no
literature loses only to the ~3% of decoys that have some; n_sid is >= 1 for every PubChem compound and spreads the whole
pool, so an unpopular truth is overtaken by many more decoys.

## 5. What is measured and what is assumed

Measured: all numbers above, on the three proxy scenarios, from cached V2 cosines; the popularity distributions.

Assumed / not known:
- The popularity of hidden-test truths. Class 2 is "known structure, no public spectra" (data page), so its truths are in
  PubChem or COCONUT but have no library spectra; I assume they look more like the random COCONUT sample (or lower) than
  like the proxy truths. That is why `cfcoco` / `cfmin` are used as the planning cases. If the hidden Class 2 molecules
  resemble np-examples in popularity, an all-rows prior would be worth far more than the recommendation; the proxy cannot
  tell, only a leaderboard probe could.
- `cfcoco` / `cfmin` / `cfpool` replace the truth's counts only; decoy counts are real. They are scenarios, not estimates.
- The share of hidden molecules that are PubChem-only vs in COCONUT is unknown; the recommendation was chosen so that both
  S1 and S2 lose about 0.001.

Caveats:
- S3 has 300 molecules, 202 with the truth among the candidates; the weight was picked on the same molecules (a 9-point grid
  and a deliberately conservative choice, but no held-out confirmation).
- `CID-PMID.gz` was used as downloaded; which PubMed link types it contains was not checked.
- Not implemented in the submission pipeline: V3 needs `pubchem_rows_pop.parquet` (2.76 GB, vs 2.72 GB) attached as the
  Kaggle dataset and `sum(n_pmid)` added to the window query of `pubchem_windows`. Runtime impact not measured (one extra
  int32 column in the same range join).
- Pools capped at 20,000 PubChem rows (1 molecule in S1/S2, 3 in S3) keep the V2 cap by n_cid; unchanged here.

## 6. What did not work

- Prior on all rows: +0.06 / +0.21 / +0.06 on the proxy at w = 0.01, but -0.08 / -0.08 with COCONUT-typical truths and
  -0.13 / -0.13 with minimum-count truths. Rejected as a default because of the bias.
- z-scoring within the pool: no gain over raw, worse for unpopular truths; z-scored n_pmid is unstable.
- n_cid: little signal.
- Dropping the tier bonus in favour of the prior: S2 falls by 0.09 or more.

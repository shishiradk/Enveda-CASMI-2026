# C3 generator tournament on the full C3NP bench (2026-10-05)

**Executive summary**
1. FACT: both generators ran on all 550 C3NP molecules with 0 errors and a mass-valid share of 1.0. MRR@25: mmp_edit 0.403 (99.9% CI 0.341-0.468), biotransform 0.492 (0.426-0.556).
2. FACT: the best combination is a round-robin union with biotransform first (`rr_bt`): MRR@25 0.508 (0.443-0.570), hit@25 0.653. It beats biotransform alone by +0.016 (paired 95% CI +0.007 to +0.025), and by +0.030 on molecules without a 1-atom neighbour.
3. FACT: re-ranking the union by zlog fails (0.253). zlog+RRF reaches 0.191 against 0.166 on molecules without a 1-atom neighbour, but that edge is not significant (CI −0.001 to +0.051), and it loses −0.094 overall.
4. FACT: the score is mostly congener-driven. On `rr_bt`, molecules with a ±CH2/±O neighbour in the seed sources (nb1, 58% of the bench) score 0.757; the other 42% score 0.166. With those neighbours stripped (`--strip edit1`), the score falls to 0.399.
5. FACT: in-sample rows (0-59, 300-359) score slightly higher than the rest: `rr_bt` 0.539 vs 0.499, biotransform 0.529 vs 0.482, mmp_edit 0.413 vs 0.400.
6. FACT (proxy): inserting 5 C3 candidates at final ranks 4-8 of the E6 list costs about 0.0037 LB, via the calibration weights. Ranks 6-10 cost 0.0015 and ranks 3-5 cost 0.0079.
7. INFERENCE: E7 LB gain from `rr_bt` at ranks 4-8 is about +0.007 (floor) / +0.019 (no near congener) / +0.039 (25% of hidden C3 have one) / +0.050 (stripped bench as the forecast). After E6's observed 2x shortfall, best guess is +0.01 to +0.03, so about 0.37-0.39 LB. This is not enough for the 0.432 prize line on its own.
8. FACT: runtime is about 4.4 s (mmp) + 4.7 s (biotransform) CPU per molecule, with a worst case of 17.8 s. INFERENCE: 400 test molecules take 15-30 min on Kaggle's 4 CPUs.
9. FACT: new assets total about 690 MB, dominated by `pool_fp.npy` at 617 MB. Every asset comes from train.parquet structures, COCONUT, RDKit or our own code. No Ahmed, FRIGID or NIST input is used (grep of file paths in both generators and `assets.py`).
10. Recommendation: E7 = E6 + `rr_bt` top-5 at ranks 4-8, ungated, after metric-key de-duplication against the E6 list. Before shipping, fix biotransform's wall-clock guard (fired 6/550, and 18/550 under load) and port the context builder.

## 1. What was run (all FACT)
- Generation: `research/scratch_wf/c3_tournament/run_chunks.py {mmp_edit|biotransform} [--strip edit1]`, run as 5 chunks of 110 molecules. Each chunk was `research/c3gen/<gen>.py --c3np --n 110 --offset O --workers 2`. Outputs are in `results/c3gen/full/`. Free RAM stayed between 3.9 and 4.8 GB. Two runs went concurrently (4 workers), and I started nothing else.
- Combinations (`combine.py [--suffix _strip_edit1]`; union de-duplicated on plain InChIKey14, capped at 200): `rr_bt` / `rr_mmp` alternate the two lists (biotransform / mmp_edit first); `rrf` = Σ 1/(10+rank); `zlog` = z-scored full 6930-bit fp @ zlog (`biotransform._zfull_many`); `zlog_rrf` = z(zlog) + 10·rrf (weight not tuned).
- Scoring: `research/scripts/c3np_eval.py <json> --name <x>`. Tables come from `analyze.py`, the LB simulation from `displace.py` + `sim_lb.py`, all in `research/scratch_wf/c3_tournament/`. JSON outputs: `results/c3gen/full/{tournament_tables,displace,sim_lb}.json`.
- Subset sizes: npex 250 (rows 0-249); s3pc 215 and s3none 85, interleaved in rows 250-549. In-sample rows (0-59, 300-359) total 120 (npex 60, s3pc 42, s3none 18). All other rows are "rest" (n=430); biotransform's held-out smoke slices (rows 120-179, 420-479) count as rest.

## 2. Verifier / review outcomes carried in (from `research/scratch_wf/c3_resume_done_reports.md` + review notes)
| Item | Outcome |
|---|---|
| C3NP bench leak checks | Clean. Baselines score exactly 0, and no alias was found among Tc=1 matches. 4 minor metric issues (stereo leniency ≤4 molecules, empty entries, >25 lists, string input). |
| Forbidden alias (myricetin `BLVASQLGJVHTQN`) | Minor. It is only in `universe.parquet`, which no generator reads. Add it to `forbidden.parquet` before any generator seeds from the universe. |
| `mid` = truth ik14 passed into `generate()` | Minor. Neither generator reads it. Deployment must pass an opaque id and `forbidden=frozenset()`. |
| Row-order assumption in the task | Wrong. s3pc and s3none are interleaved. This tournament slices by the `subset` column. |
| mmp_edit non-determinism under load | Fixed (work budgets; guard flag). In this run the guard fired 0/550 in both modes. |
| Bench realism (1-atom edits = 75% of the reciprocal-rank sum) | Addressed by nb1 strata and `--strip`. Used in §5. |
| Bench rules at min_support=1 | Still open (`_clean_rules`, mmp_edit.py:525). Realised impact was 0 per the review. |
| biotransform wall-clock guard | Still open. FACT: the time guard fired on 6/550 molecules (default run) and 18/550 (strip run, which ran beside the evaluator). Those rows depend on machine load. |

## 3. Single generators, full bench (FACT)
| Generator | Subset | MRR@25 | hit@1 | hit@25 | hit@200 | MRR in-sample / rest |
|---|---|---|---|---|---|---|
| mmp_edit | npex | 0.465 | 0.404 | 0.656 | 0.736 | 0.461 / 0.466 |
| | s3pc | 0.277 | 0.247 | 0.400 | 0.433 | 0.293 / 0.273 |
| | s3none | 0.540 | 0.494 | 0.671 | 0.706 | 0.531 / 0.543 |
| | **all** | **0.403** | 0.356 | 0.558 | 0.613 | 0.413 / 0.400 |
| biotransform | npex | 0.718 | 0.672 | 0.824 | 0.868 | 0.734 / 0.713 |
| | s3pc | 0.241 | 0.205 | 0.326 | 0.353 | 0.266 / 0.235 |
| | s3none | 0.462 | 0.424 | 0.553 | 0.624 | 0.463 / 0.462 |
| | **all** | **0.492** | 0.451 | 0.587 | 0.629 | 0.529 / 0.482 |
- **Strip edit1** (the truth's ±CH2/±O/±CH2O congeners removed from analogs, window and seeds; all MRR@25):
  | Generator | all | npex | s3pc | s3none |
  |---|---|---|---|---|
  | mmp_edit | 0.289 | 0.274 | 0.232 | 0.477 |
  | biotransform | 0.374 | 0.556 | 0.180 | 0.330 |
- **Complementarity:** both generators hit@25 on 47.6% of molecules, mmp_edit alone on 8.2%, biotransform alone on 11.1%. An oracle that takes the better of the two per molecule would score 0.556. Each list has 200 entries; on average 27.5 are shared, and the median union holds 364 structures.
- **Runtime per molecule** (2 workers, with a second 2-worker run alongside):
  | Generator | Mean per chunk | Max | Guards |
  |---|---|---|---|
  | mmp_edit | 3.1-5.6 s (about 4.4) | 13.7 s | fired 0 times |
  | biotransform | 3.3-6.4 s (about 4.7) | 17.8 s | time guard fired 6 times (default run) |
  About 380 MB per process.

## 4. Combinations (FACT; MRR@25; nb1-no = molecules without a 1-atom neighbour, n=232)
| Combo | all | npex | s3pc | s3none | nb1-no | hit@25 all | rest (oos) | strip-edit1 all | strip s3pc | strip nb1-no |
|---|---|---|---|---|---|---|---|---|---|---|
| **rr_bt** | **0.508** | 0.717 | 0.263 | 0.512 | 0.166 | 0.653 | 0.499 | **0.399** | 0.214 | 0.169 |
| rr_mmp | 0.459 | 0.579 | 0.284 | 0.550 | 0.168 | 0.651 | 0.455 | 0.351 | 0.236 | 0.168 |
| rrf | 0.466 | 0.627 | 0.268 | 0.498 | 0.147 | 0.653 | 0.464 | 0.338 | 0.215 | 0.147 |
| zlog | 0.253 | 0.223 | 0.238 | 0.383 | 0.157 | 0.469 | 0.247 | 0.252 | 0.237 | 0.159 |
| zlog_rrf | 0.414 | 0.495 | 0.282 | 0.513 | **0.191** | 0.587 | 0.406 | 0.349 | **0.259** | **0.191** |
- **Paired bootstrap** (5,000 resamples; difference, 95% CI):
  | Comparison | Rows | Difference | 95% CI |
  |---|---|---|---|
  | rr_bt − biotransform | all | +0.016 | +0.007 to +0.025 |
  | rr_bt − biotransform | nb1-no | +0.030 | +0.017 to +0.046 |
  | rr_bt − mmp_edit | all | +0.105 | +0.076 to +0.135 |
  | zlog_rrf − rr_bt | all | −0.094 | −0.123 to −0.065 |
  | zlog_rrf − rr_bt | nb1-no | +0.025 | −0.001 to +0.051 |
- INFERENCE: zlog helps only where no close seed exists, and it destroys the congener ranking where one does. A seed-similarity-dependent mix (zlog weight rising as the best seed Tc falls) is the obvious next re-ranker. It has not been built.

## 5. Realism discount (how much carries to real Class 3)
- **MRR by best seed-source Tanimoto band** (FACT; n per band in brackets):
  | Generator | <0.5 (n=18) | 0.5-0.7 (n=93) | 0.7-0.85 (n=233) | ≥0.85 (n=206) |
  |---|---|---|---|---|
  | mmp_edit | 0.000 | 0.190 | 0.420 | 0.515 |
  | biotransform | 0.006 | 0.237 | 0.497 | 0.644 |
- **nb1 split for `rr_bt`** (FACT): yes 0.757 (n=318), no 0.166 (n=232).
- **nb1 share by subset** (FACT): npex 0.856, s3none 0.588, s3pc 0.251.
- **Evaluator projection** p·MRR(yes) + (1−p)·MRR(no) for `rr_bt` (FACT arithmetic; p is not measured): p=0.10 → 0.225, p=0.25 → 0.314, p=0.50 → 0.462.
- INFERENCE: real Class-3 molecules are absent from both PubChem and COCONUT, so unlike bench truths (known NPs from library collections) they are less likely to have a database congener one OH or CH3 away. Planning range p ≈ 0.1-0.25 (s3pc, 0.25, is the closest measured analogue), plus a novelty haircut f ≈ 0.5 for timsTOF context quality and truly novel scaffolds.
  - Planning numbers for standalone `rr_bt` on hidden Class 3: realistic MRR@25 ≈ 0.10-0.20 (central 0.15); optimistic 0.31 (p=0.25, f=1); floor 0.08.

## 6. LB effect under the append-only rule (DEC-008/009)
- **Model (INFERENCE):** E6 list = engine top-1, then engine[1:] alternating with the PubChem channel. The C3 list fills a fixed block of positions P and E6 entries shift down. Cost = Σ_b w_b · ΔMRR_b on the bench buckets with calibration weights SV 0.146, S1 0.21, PC 0.111 (S2 weight 0; C3 engine MRR 0.012, ignored). Gain = (Class-3 share) × C3-list MRR mapped to the positions in P.
  - Displacement uses the measured E6 bench ranks (FACT). The rank-1 counts are SV 342/400, S1 200/250, PC 37/213; 11-25 holds SV 1, S1 6, PC 23.
- **Cost and Class-3 gain by position block** (FACT for both columns; `sim_lb.py`):
  | Positions P | LB cost | C3 gain: all | strip-edit1 | nb1-yes | nb1-no |
  |---|---|---|---|---|---|
  | 3-5 | 0.0079 | 0.173 | 0.136 | 0.259 | 0.057 |
  | **4-8** | **0.0037** | 0.134 | 0.108 | 0.199 | 0.046 |
  | 6-10 | 0.0015 | 0.091 | 0.074 | 0.134 | 0.032 |
  | 11-20 | 0.0007 | 0.053 | 0.044 | 0.076 | 0.021 |
- **Net LB delta, Class-3 share 0.5** (INFERENCE; ±10% for shares of 0.45-0.55):
  | P | floor (nb1-no, f=0.5, share 0.45) | nb1-no | p=0.25 | strip-edit1 bench |
  |---|---|---|---|---|
  | 3-5 | +0.005 | +0.021 | +0.046 | +0.060 |
  | **4-8** | **+0.007** | **+0.019** | **+0.039** | **+0.050** |
  | 6-10 | +0.006 | +0.015 | +0.027 | +0.035 |
  | 11-20 | +0.004 | +0.010 | +0.017 | +0.021 |
- INFERENCE: ranks 4-8 has the best floor and costs half as much as 3-5, which wins only if the C3 hit rate is high. The proxy's cost model has an LOO RMSE of 0.039 (DEC-008 addendum), so cost estimates around 0.004 are below its resolution. E6's +0.005 measured against +0.010 predicted suggests the realised gain falls about 2x below these figures.

## 7. Kaggle feasibility
- **Runtime** (FACT local, INFERENCE Kaggle): about 9.1 CPU-s per molecule for both generators (worst 13.7 s mmp, 17.8 s biotransform); round-robin needs no extra scoring, zlog variants add about 0.9 s wall per molecule with 2 workers. 400 molecules × 9.1 s / 4 processes ≈ 15-30 min. E6 used 7,623 s, and its channel budget formula implies about 9 h of wall time, so more than 5 h remain.
- **Context per test molecule** (INFERENCE, to verify in the E6 notebook): zlog (E6 nets, already shipped); the window (E6 PubChem ±5 ppm + COCONUT window scored by zlog, already computed in E6); `analogs` (library spectral analogs with sim and dmass; the bench used ho1 `ana` records, so the engine's analog list must be exposed).
- **Assets** (FACT sizes):
  | Asset | Size | Origin | Licence status |
  |---|---|---|---|
  | `train_pkg/data/pool_fp.npy` (+`pool_mass` 5.7 MB, `pool_key` 10 MB, `pool_smiles.txt` 45 MB, offsets 5.7 MB) | 617 MB | train.parquet structures ∪ COCONUT (711,626) | train-derived: host D/742193 allows models trained on train.parquet; COCONUT: host D/743234 "COCONUT's own license is sufficient" (CC BY 4.0 per dataset notices, so attribute) |
  | `mmp_rules.parquet` (deploy with `load_rules(min_freq=3)`, not the bench-clean file) | 8.7 MB | mined from 150k train ∪ COCONUT | same as above |
  | `biotx_pool_np.npy` NP mask (or `np_pool.parquet` 27.6 MB) | 0.7 MB | NP score on pool | ours (MIT); scorer provenance to confirm (INFERENCE: RDKit Contrib, BSD) |
  | `fp_bits.npy` | 56 KB | ours | MIT |
  | zlog nets | 0 new | `shishiradhikari11/casmi-e6-pcnets` | MIT, train.parquet only |
  | code (`mmp_edit.py`, `biotransform.py`, `assets.py`) | <0.2 MB | ours | MIT; RDKit BSD-3 |
  - Total new: about 690 MB. Residual risk (FACT, competition_intel §3): the Rules page still lists the Competition Data as CC BY-NC 4.0, while the host says train-derived models and artifacts are fine. This is the same risk E6 already carries.

## 8. E7 recommendation
1. **Ship:**
   - E7 = E6 list with ranks 1-3 untouched. Ranks 4-8 take the first 5 `rr_bt` candidates not already in the E6 list (metric-key dedup); E6 ranks 4+ follow, truncated at 25.
   - No gate. Lesson 1 of DEC-008: library-similarity gates misfire on re-measured hits, and the ungated cost is only about 0.004.
   - Expected LB +0.007 to +0.05, central +0.01 to +0.03 after the E6-style 2x haircut (INFERENCE).
   - If a second slot is available, submit ranks 6-10 as the low-risk twin. The pair measures the cost/gain slope on the real LB.
2. **Before shipping:**
   - Replace biotransform's wall-clock guard with work budgets, as was done for mmp_edit.
   - Deploy-mode switches: full rules (min_freq 3), opaque mid, `forbidden=∅`, and the full pool without the 585 held-key exclusions.
   - Build the Kaggle context (analog list export).
   - Check parity on 20 bench molecules against `results/c3gen/full/*.json`, then run the visible-set check (top-3 unchanged on 400/400).
3. **Build next, in order of value:**
   1. A seed-Tc-aware re-ranker over the union: rr positions, z(zlog), best seed Tc, rule frequency. Fit it with nb1-stratified CV, on the rest rows only. Target: the nb1-no stratum, currently 0.166-0.191.
   2. Estimate p (the nb1 share of real Class 3) from a time-split of COCONUT or PubChem NPs: newer entries scored against an older pool. This is the largest uncertainty in §6.
   3. Fix the remaining minors (`_clean_rules` min_support=3, the myricetin alias).
   4. The C3 lever alone does not reach 0.432. The rest of the gap must come from Class-2 ranking (same-formula isomers in the PubChem channel).

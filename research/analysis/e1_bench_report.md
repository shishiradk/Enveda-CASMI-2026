# Local bench of the E1 engine on the leak-controlled proxy scenarios

Date: 2026-10-02. Code: `research/bench/`. Outputs: `results/bench/` (run tag `e1`).
This run was started by one agent (S2 finished, then the process was killed) and finished by a second one
(S1, S3, the Kaggle parity check, the K-fold ranker refit, all evaluation). Section 8 lists what the second
pass found wrong or misleading in the first pass.

## 1. Bottom line

| | S1 (Class 1) | S2 (Class 2, in COCONUT) | S3 (known structure, PubChem only) |
|---|---|---|---|
| E1 as submitted, MRR@25 | 0.905 [0.875, 0.933] | 0.695 [0.651, 0.740] | 0.136 [0.100, 0.174] |
| E1 best honest estimate | **0.863 [0.826, 0.897]** | **0.647 [0.600, 0.694]** (0.620 with E1's shipped pool only) | **0.11 - 0.14** |
| optimism of the as-submitted number | +0.042 (ranker rows) | +0.049 (ranker rows) +0.03 (pool) | at most +0.03 (fingerprint nets) |
| our V2 engine alone | 0.855 [0.816, 0.892] | 0.543 [0.491, 0.595] | 0.202 [0.164, 0.242] |
| where E1 loses | ranking among isomers | ranking among isomers | coverage (85% of truths not in its pool) |

- The local engine reproduces the Kaggle run to within ranker-fit noise (section 5).
- Fusion with our lists helps in all three scenarios; the gain is larger on the honest E1 scores than on the
  leaky ones (section 4).
- Absolute levels do not map onto the leaderboard (E1 0.353, ours 0.251) for any class mix; use the bench for
  paired comparisons only (section 7).

## 2. Design

`bench.py` imports the notebook's own engine modules (`extract_engine.py` writes `pv.py`, `pv_fp.py`,
`casmi_engine.py`, `eng_runner.py` from `research/kaggle_e1/casmi_e1.ipynb` into `research/bench/eng/`, with one
patch: the input root is read from `CASMI_ROOTS`). It replays the `eng_runner.py` call sequence (same constants,
asserted against the runner text) but loads the 2.54 M-spectrum library once and masks rows in memory per scenario.
Every per-candidate channel and score is saved, so rankers, blends, fusion and the error analysis are computed
offline in seconds.

| scenario | queries | library rows masked | pool |
|---|---|---|---|
| S1 | 250 enveda-np-examples molecules, 1,184 timsTOF spectra | 1,184 (every enveda-np-examples row) | unchanged; all 250 targets keep spectra from other libraries |
| S2 | same | 56,808 (all rows of the 285 held keys: targets, EXP-012 aliases, parents) | a held structure stays only if COCONUT / ChEBI / LIPID MAPS has it |
| S3 | 300 natural products not in COCONUT, 829 public-library spectra | 2,490 (all rows of the 300 held keys) | 257 train-only held structures removed; 46 truths remain through ChEBI / LIPID MAPS |
| S3i | same as S3 | 2,490 | oracle: the true structures put back without spectra |

Naming caveat: S3 is "structure known to PubChem but outside COCONUT", i.e. the part of Class 2 that E1's pool
does not cover. A true Class 3 molecule (not in PubChem) is unreachable for both engines and scores 0.

Scoring: a candidate is correct if its tautomer-canonical InChIKey14 (`casmi_engine.canon_key`, local RDKit) equals
the truth's, or its raw InChIKey14 is in the proxy alias set. Lists are built as in `eng_runner.py`
(top 80 by blend score, de-duplicated on the metric key, 40 kept); MRR is taken at 25. CIs are 10,000-resample
bootstraps over molecules; differences are paired.

## 3. Leak handling

### 3a. Library leak (clean)

Masked rows get `row_p = -1`, so they leave the exact-mass index and the analog representatives. Checks:

- Mask sizes equal independent DuckDB counts on `train.parquet`: 1,184 / 56,808 / 2,490. The engine's index sizes
  agree (2,539,608 minus the mask = 2,538,424 / 2,482,800 / 2,537,118).
- `bench.py` asserts that no held key keeps a library row and no held train-only structure stays in the pool.
- Share of in-window truths with a non-zero library similarity: S1 1.00, S2 0.00.
- S2 was run before `bench.py` was last edited (the smoke run reports 16 structures removed, the full run 0).
  Recomputing the mask with the current rule gives 0 removed and no candidate to drop, so `S2.pkl` is consistent
  with the current code. The 20 smoke molecules have identical ranks in the full run.
- Truth keys: 0 of 1,100 molecule-scenario pairs change when only the strict tautomer-canonical rule is used.
  1,957 of 1,957 SMILES from the Kaggle lists get the identical key locally (RDKit 2026.03.6 vs 2026.03.3 on Kaggle).

Pool provenance (optimistic for S2 by about 0.03): 12 of the 250 S2 truths are in the bench pool only as
train-origin structures. They are kept because our COCONUT copy contains them, but the candidate files E1 ships
(prvsiyan COCONUT + ChEBI/LIPID MAPS) do not, so on Kaggle a Class-2 molecule like them is unreachable. Zeroing
them gives 0.664 for the as-submitted blend and 0.620 for the honest one. Real S2 pool coverage of E1 is 95.2%, not 100%.

### 3b. Model leak

**Ranker rows (leak confirmed, measured).** Both ranker training sets contain rows simulated from the 250 S1/S2
molecules, in sorted-InChIKey order: `rank_train.npz` groups 0-249, and megayak's `sim_rank_rows_nofp.npz`
groups 2g and 2g+1 of query set `np`. The top-analog similarity stored in those rows equals the bench's own value
within 1e-3 for 84% / 89% of the 500 molecule-scenario pairs.

`bench_cvrank.py` refits both rankers 5-fold, held out by molecule (the author's "held out by query group" CV),
and scores each molecule with the rankers that never saw its rows:

| | S1 | S2 |
|---|---|---|
| blend (as submitted) | 0.905 | 0.695 |
| clean_kf (both rankers query-held-out) | 0.863 [0.826, 0.897] | 0.647 [0.600, 0.694] |
| clean_kf - blend | -0.042 [-0.060, -0.026] | -0.049 [-0.068, -0.030] |
| pv_kf - pv (prvsiyan ranker alone) | -0.052 [-0.072, -0.034] | -0.059 [-0.079, -0.039] |

The first agent's `clean` / `pv_cv` / `ours_cv` sets drop all 250 molecules' rows at once. That also removes every
timsTOF natural-product training query, and S1 collapses to 0.640. This is a domain-shift artefact, not a leak
estimate; do not use those three sets.

**Fingerprint nets on S1/S2 (clean).** Three independent pieces of evidence that the public nets never trained on
the 250 molecules:
- the author's notebook states that all 250 validation structures are in the model's held-out split;
- megayak's README reports 0.493 for the public net on this population, against 0.76-0.82 on libraries it had seen;
- measured here: FP-only MRR 0.486 with the public nets vs 0.480 with megayak's nets, which held these molecules
  out by construction (difference +0.005 [-0.018, +0.029]).

The residual is early stopping on a validation split that contains them, which the author estimates as negligible.

**Fingerprint nets on S3 (leak confirmed, bounded).** The S3 targets come from gnps/riken/mona/massbank/msdial,
and both net pairs trained on nearly all of them. FP-only MRR on S3i is 0.841 (public) and 0.850 (megayak) against
0.486 on S2: memorisation.
- S3i (oracle pool) 0.856 is not usable as an absolute number. With the FP features zeroed it is 0.742
  [0.697, 0.785]; the honest value lies between 0.74 and 0.86.
- S3 proper is bounded by coverage, so the leak can only reorder the 46 reachable truths: 0.111 (FP zeroed,
  prvsiyan ranker) to 0.136-0.140.

Not quantified: whether some S3 molecules are among the 569 "obscure" groups of `rank_train.npz` or the 2,000
non-np groups of megayak's rows (no identities shipped; expected overlap is a handful). The author's training
folds are not public: `kaggle kernels list --user prvsiyan` shows only the baseline notebook, whose embedded
`train_fp.py` reads an `is_val` flag from an unpublished `spec.npz`.

## 4. Fusion with our V2 lists (weighted RRF, K = 3)

Our lists are rebuilt from `results/kaggle_v2_proxy/cache_S*.pkl` as `v2_lists.py` emits them: cosine + 0.05 for
universe rows, sorted by (-score, key), metric-key de-duplication, 40 kept. `bench_eval.fuse` is the E2 notebook
cell. The fusion sweep was already implemented by the first agent and checked against both sources.

Differences versus BETA 0, paired, 95% CI:

| BETA | S1, as submitted | S2, as submitted | S3 | S1, honest (clean_kf) | S2, honest (clean_kf) |
|---|---|---|---|---|---|
| 0 (MRR) | 0.905 | 0.695 | 0.152 | 0.863 | 0.647 |
| 0.2 | +0.001 [-0.000, +0.003] | +0.005 [+0.002, +0.009] | +0.004 [+0.002, +0.005] | +0.003 [+0.001, +0.005] | +0.008 [+0.005, +0.013] |
| 0.4 (production) | +0.013 [+0.003, +0.024] | +0.028 [+0.009, +0.049] | +0.014 [+0.010, +0.019] | +0.029 [+0.015, +0.044] | +0.035 [+0.016, +0.056] |
| 0.6 | +0.024 [+0.009, +0.040] | +0.038 [+0.012, +0.065] | +0.021 [+0.011, +0.030] | +0.047 [+0.028, +0.067] | +0.039 [+0.013, +0.066] |
| 1.0 | +0.029 [+0.011, +0.048] | +0.006 [-0.027, +0.040] | +0.041 [+0.023, +0.060] | +0.054 [+0.031, +0.077] | +0.032 [-0.004, +0.069] |

- BETA 0 is not "E1 alone" on S3: the E2 cell appends our candidates with score 0 behind a short E1 list, which
  gives 0.152 against 0.136 for E1 alone (+0.016 [+0.009, +0.026]). Against E1 alone, BETA 0.4 is +0.030
  [+0.022, +0.041] on S3. On S1/S2 the two baselines are identical.
- BETA 0.4 is significantly positive everywhere. 0.6 is at least as good in every column; 1.0 loses the gain on
  S2. The 0.4 vs 0.6 difference is within the CIs.
- On S3 the fused 0.166 stays below our engine alone (0.202): with weight 0.4 our rank-1 candidate lands around
  rank 7 behind E1's confident wrong isomers. A gate on E1's confidence would recover more than a larger BETA.
- S3i fusion is negative (-0.078 at 0.4) but is not a valid scenario: E1 has the oracle pool and leaked nets while
  our lists do not have the oracle pool.

## 5. Reproduction of the Kaggle run

`bench_parity.py` runs the unmasked engine on the first 40 test molecules and compares with
`results/bench/kaggle_e1_output/eng_lists.json`. The seed-split column compares two disjoint halves of the local
ranker ensemble, as a scale for fit noise.

| | local vs Kaggle | local half A vs half B |
|---|---|---|
| same rank 1 | 40/40 | 40/40 |
| same top-3 order | 77.5% | 62.5% |
| same top-5 order | 45% | 22.5% |
| top-25 Jaccard | 0.911 | 0.848 |
| mean rank shift of the reference's rank-5 / rank-10 | 0.45 / 1.2 | 0.78 / 1.8 |
| identical top-25 order | 0/40 | 0/40 |

List lengths match for 40/40, the pool (772,653 structures) and representative counts match the Kaggle log, and
the canonical keys are identical. The order below rank 1 differs, by less than what half the seeds produce.
Read: same engine, differences at the level of ranker-fit noise (CPU vs CUDA nets, RDKit 2026.03.6 vs .3,
scikit-learn build). Only 40 molecules were compared, all of which have a stable rank 1.

## 6. Error analysis (as-submitted blend; share of molecules, MRR lost in brackets)

| truth is ... | S1 | S2 | S3 | S3i (oracle pool) |
|---|---|---|---|---|
| absent from the pool | 0.000 | 0.000 | 0.847 (0.847) | 0.000 |
| in the pool, not in the top 40 | 0.004 (0.004) | 0.008 (0.008) | 0.000 | 0.067 (0.067) |
| rank 26-40 | 0.000 | 0.004 (0.004) | 0.000 | 0.000 |
| rank 2-25 | 0.144 (0.091) | 0.436 (0.293) | 0.030 (0.017) | 0.130 (0.077) |
| rank 1 | 0.852 | 0.552 | 0.123 | 0.803 |
| rank-1 candidate has the truth's formula (truth listed, not first) | 35/36 | 105/110 | 6/9 | 33/39 |

With the honest rankers (clean_kf) the same table reads, for S1: 0.012 / 0.008 / 0.180 / 0.800 and for S2:
0.020 / 0.036 / 0.436 / 0.508 (not top 40 / 26-40 / 2-25 / rank 1), same-formula winner 45/47 and 112/118.

- **Class 1 (S1): ranking.** Coverage is 100% and the truth is in the top 25 for 98-99%. Almost all loss is the
  truth sitting at rank 2-25 behind a same-formula isomer.
- **Class 2 in COCONUT (S2): ranking.** 96% of the loss is the truth at rank 2-25, and the winner is a same-formula
  isomer in 95% of those cases. Coverage costs about 0.03 (the 12 truths missing from the shipped pool). This is
  the population where a same-formula re-ranker (E3) has the most room: up to 0.29 MRR.
- **Known structures outside COCONUT (S3): coverage.** 85% of the truths are not in E1's pool. When the structure
  is supplied (S3i) the engine ranks it well, though that number is leak-inflated. The 6.7% "in pool, not in
  top 40" on S3i are precursor/adduct problems of public-library queries (water-loss adducts, wrong precursor),
  not a property of timsTOF test data.

## 7. Limitations

- **Absolute levels are too high.** No class mix reproduces both leaderboard scores from these numbers: if every
  test molecule were Class 1 or Class 2 in COCONUT, the honest bench would predict at least 0.62 for E1. The
  proxies are popular, well-studied molecules. The bench also compresses the E1 vs ours gap (S2: 0.647 vs 0.543;
  leaderboard 0.353 vs 0.251), so it may overstate what our lists add: the fusion gains and the BETA optimum
  should be read as upper-leaning.
- S1 and S2 are the same 250 molecules, so their errors are correlated; n = 250 / 300 gives CIs of about +-0.03-0.05.
- S3 queries are public-library spectra on many instruments (3 timsTOF spectra of 829), 54% without collision energy.
- The K-fold refit covers the ranker rows only. It does not undo choices the author tuned on these molecules
  (class prior, constants); he reports those local preferences did not transfer to the leaderboard.
- The parity check covers 40 unmasked test molecules; masked scenarios cannot be compared with Kaggle.
- Process: during the run the side jobs (evaluation with 2 canonicalisation workers, rescoring) ran beside the
  3-thread engine, so the "3 workers / ~6 GB" budget was exceeded for a few minutes at a time (peak about 6 GB).

## 8. Problems found in the first pass

1. **S3 ran with a dead fingerprint channel.** A NULL `collision_energy_ev` makes `parse_ce` return NaN, so 234 of
   300 S3 molecules had NaN logits (S3 0.113 instead of 0.136). Fixed for the saved run by `bench_fixfp.py`
   (recomputed logits are bit-identical for the 65 unaffected molecules) and at the source in `bench.py run`.
   Kaggle is unaffected: every test row has a collision energy. One S3 molecule (BUIKOOKPWXVMOG) gets no
   candidates at all and scores 0.
2. **`clean`, `pv_cv`, `ours_cv` are not leak estimates** (section 3b). Replaced by `clean_kf`.
3. **BETA 0 is not E1 alone** when E1's list is short (section 4).
4. **S2 pool coverage is optimistic** by 12 of 250 molecules (section 3a).
5. Misleading but harmless: the comment in `bench.py` that the shipped COCONUT file excludes train structures is
   wrong (240 of the 256 held keys present in the pool come from that file); the S3 log line "0 train structures
   out of the pool" describes the oracle mask (the strict mask removes 257); scenario `T` sets `--limit` for
   everything after it, so it must be last in `--scen`.
6. `bench.py` was edited after the S2 process had started; S2 was verified against the current rule (section 3a).
7. No parity check had been run; the `smoke/` directory is a 20-molecule S2 smoke test, not a Kaggle comparison.

## 9. How to run

```
python research/bench/extract_engine.py                                   # only when the notebook's engine changes
python research/bench/bench.py run --tag e1 --scen S1,S2,S3 --workers 3    # --scen resumes per scenario
python research/bench/bench_cvrank.py e1 3                                 # honest rankers for S1/S2 (22 min)
python research/bench/bench.py eval --tag e1 [--score clean_kf]            # MRR, paired CIs, fusion sweep
python research/bench/bench_report.py e1 blend                             # five-way error table, audits
python research/bench/bench_parity.py 40 3                                 # Kaggle reproduction (6 min)
python research/bench/bench.py rescore --tag e1 --scen S1,S2,S3            # new rankers/blends, no library needed
```

Detached on Windows (survives a session restart):

```
Start-Process python -ArgumentList "-u","research/bench/bench.py","run","--tag","e1","--scen","S1,S3" `
  -WorkingDirectory D:\Enveda-CASMI-2026 -RedirectStandardOutput results\bench\run.log `
  -RedirectStandardError results\bench\run.err -WindowStyle Hidden -PassThru
```

Timing on 3 threads: start-up 3 min with the pool cache (17 min without), S1 56 min, S2 85 min (first scenario, includes 11 min of key canonicalisation that is now cached), S3 22 min.
Peak memory 5.2 GB. Another engine variant: `--eng-dir DIR --tag NAME`. `rescore` after `bench_cvrank.py`
overwrites `S1.pkl` / `S2.pkl` without the `*_kf` sets; rerun `bench_cvrank.py` (rankers are cached).

Files:

| path | content |
|---|---|
| `results/bench/e1/S1.pkl`, `S2.pkl`, `S3.pkl`, `S3i.pkl` | per-molecule candidates, all score sets, metric keys |
| `results/bench/e1/recs_*.pkl` | channel records for `rescore` (`recs_S3_nanfp.pkl`: S3 before the fix) |
| `results/bench/e1/eval_blend.md`, `eval_clean_kf.md`, `report_blend.md`, `report_clean_kf.md` | printed tables (the S3 columns of the `clean_kf` files are empty by construction) |
| `results/bench/e1/eval_*.json`, `report_*.json`, `per_molecule_*.csv` | the same as data |
| `results/bench/parity/parity2.json`, `lists_T.json` | Kaggle reproduction |
| `results/bench/run_e1.log`, `run_e1_resume.log`, `run_cvrank.log`, `run_parity.log` | run logs |

# v4r simulation driver + ranker v0 (phase B, pilot)

Our code, MIT. Written against `research/v4n_rebuild/REBUILD_SPEC.md` sections 3.9, 4, 6 and 9 (V5). It uses the
phase-A engine in `../engine/` unchanged: no engine file was modified. The reference code in `ahmed_ref/` was read
only; nothing was copied from it.

Labels: **FACT** = measured on 2026-10-08 on this PC, with the command given. **INFERENCE** = deduced.
Every run on this PC shared the CPU with the COCONUT `featurize` build (8 workers on 4 cores / 8 threads). Free RAM
was 1–3 GB. All timings below are therefore pessimistic.

## Files

| File | What it does |
|---|---|
| `common.py` | Paths, regime list, class mix (c1 0.20 / c2 0.55 / c3 0.25), RSS helpers, `mrr_and_top1` (MRR@25 and top-1; ties broken by row order) |
| `build_queries.py` | Truth set → query table. One query per truth score key and regime: c1, c2, c3 (REBUILD_SPEC 4) |
| `simulate.py` | Process-pool driver. Runs the engine with the masks, labels the rows and writes parquet shards. Resumable and shardable across machines. `--test` mode runs the visible test |
| `ranker.py` | Ranker v0: LightGBM lambdarank with 4 seeds averaged, 5-fold CV on F0..F4, regime weights, f·z baseline, feature importance, V5c probes, all-data fit |
| `v5.py` | The V5 checks a, b and c, an SV evaluation of ranker v0, and runtime statistics |

## How the simulation works (`build_queries.py`)

**Truth unit.** A truth is one **score key** (tautomer-canonical InChIKey14). It must have a pool row.

**Closure rule (leakage).** A key is used only if *every* sid sharing it is in the bank's hold-out set:
- for the pilot, the ho2 inchikey14 list;
- for the full run, `in_hoR`.

**Fold.** The fold is the `split_v4r` fold of the sid that owns the pool row. Every query of a key therefore falls in
one fold, so CV is grouped by score key.

**Query library ℓ.** enveda-180 if the key has usable spectra there, otherwise the library holding most of them.

**Query spectra.**
- A random subset of the key's spectra in ℓ.
- Only spectra with |precursor error| ≤ 10 ppm are used. Reason (FACT): enveda-180, the test instrument, is 100 % within
  10 ppm, while riken is only 73.5 %. If no spectrum qualifies, all spectra of the key are used.
- Test adducts are preferred.
- The subset size is drawn from the visible test distribution of spectra per molecule: 1: 13.8 %, 2: 24.3 %,
  3: 24.8 %, 4: 26.5 %, 5+: 10.8 %.
- Query spectra are L1-cleaned library spectra, from `Library.query_spectrum`.
- Target = median neutral mass of the chosen spectra.

**Regimes.**

| Regime | `exclude` (hidden spectra) | `exclude_sid` / `exclude_lib` | `drop_pid` | Eligible if |
|---|---|---|---|---|
| c1 | every spectrum of every key sid in ℓ | sid0 / ℓ | −1 | the key has non-empty spectra in another library |
| c2 | every spectrum of every key sid | sid0 / −1 | −1 | always |
| c3 | as c2 | sid0 / −1 | the truth's pool row, looked up by key at run time | always |

**Label.** `y = (cand_key == truth key)`. A generated hit in c3 counts.

**Order.** Queries are shuffled with a seed, so any prefix of chunks is a random sample.

## Running it

### Pilot (done)

```
python research/v4n_rebuild/sim/build_queries.py --run pilot_ho2 --truths ho2 --max-mass 470
python research/v4n_rebuild/sim/simulate.py --run pilot_ho2 --workers 4 --threads 1 --chunk 20
python research/v4n_rebuild/sim/simulate.py --run test400 --test --workers 4 --chunk 10        # unmasked visible test
python research/v4n_rebuild/sim/simulate.py --run testSV --test --workers 3 --chunk 10 \
       --exclude-rows results/bench/sv_dup_rows.npy --truth results/bench/truth_SV.parquet   # SV protocol (V5b)
LGB_THREADS=4 python research/v4n_rebuild/sim/ranker.py --run pilot_ho2                       # 4 seeds, 500 rounds
python research/v4n_rebuild/sim/v5.py --run pilot_ho2                                        # V5 a/b/c + SV eval
```

### Full R-A run (later)

```
python research/v4n_rebuild/engine/build_tables.py project --bits <R-A ckpt>        # after the pool tail is assembled
python research/v4n_rebuild/sim/build_queries.py --run hoR_RA --truths hoR          # 46,920 queries (preview below)
python research/v4n_rebuild/sim/simulate.py --run hoR_RA --ckpt <R-A ckpt> --workers 4 [--shard-of i/n]
python research/v4n_rebuild/sim/simulate.py --run testSV_RA --test --ckpt <R-A ckpt> --exclude-rows ... --truth ...
python research/v4n_rebuild/sim/ranker.py --run hoR_RA ; python research/v4n_rebuild/sim/v5.py --run hoR_RA --test-run testSV_RA
```

**Resuming.** A chunk is written atomically: `.tmp`, then `os.replace`. A rerun skips finished chunks.

**Splitting the work.** `--shard-of i/n` assigns chunk c to this process when c % n == i. Run the n parts on different
machines or Kaggle sessions, then copy the `rows/` and `qstats/` folders together.

## Outputs (`results/v4n/sim/<run>/`)

- `queries.parquet`: one row per query. Columns: qid, key, regime, fold, stratum, sid0, pid, qlib, spec_idx,
  mask_idx, target, exclude_lib, drop.
- `rows/rows_<c>.parquet`: one row per candidate. Columns: qid, regime, key, fold, cand_key, cand_pid (−1 =
  generated), cand_src (0 train / 1 COCONUT / −1 generated), y, and 86 `f_*` columns in `engine.FEATURES` order.
- `qstats/q_<c>.parquet`: one row per query. Columns: n_query, q_npeaks, lib_max, top_sim, n_cand, n_gen, has_pos,
  fz_rank, sec, n_pos.
- `cv_report.json`, `cv_pred.parquet`, `feature_importance.csv`, `ranker_v0.pkl`, `v5_report.json`, logs.

`ranker_v0.pkl` uses our format: `dict(features, params, rounds, boosters=[model strings], mix)`. The pilot model was
trained on `cft_ho2` features and on ≤ 470 Da ho2 truths only. **It is a smoke artefact, not a submission ranker.**

## Pilot results (FACT)

### Query table (`build_queries.json`)
- 1,152 ho2 sids in HO_R → 1,149 keys in the pool.
- 16 keys dropped by the closure rule (a tautomer sid outside the ho2 list).
- 105 keys dropped by the mass cap (> 470 Da).
- **1,028 truth keys → 2,349 queries:** c1 293, c2 1,028, c3 1,028.
- Query libraries: enveda-180 603, gnps 203, riken 91, pluskal 88, other 43.
- Mean of 2.72 spectra per query; the cap is the number of spectra available.

### Rows (`simulate_stdout.log`)
- **290,163 candidate rows** with 1,398 positives. 74 MB of parquet, about 33 KB per query. 0 failed queries.
- Truth among the candidates:

  | Regime | Truth in candidates |
  |---|---|
  | c1 | 99.0 % |
  | c2 | 99.3 % |
  | c3 | 8.5 % (generation) |

- Median candidates per query: 96. Mean generated candidates: 9.1.

### Runtime (FACT, under the contention described at the top)
- **1.91 s per query wall** with 4 workers (2,349 queries in 75 min).
- Per query per worker: mean 7.5 s, median 7.0 s, p90 10.3 s, maximum 103 s (a generation-heavy query).
- Visible test: 7.2 s per molecule per worker, so 400 molecules in about 16 min with 4 workers.
- Memory per worker: 1.07–1.33 GB private, about 2.0 GB peak working set including the shared memory maps. That is
  after `_slim()`, which turns the adduct and instrument string columns into codes and drops unused SMILES columns.

### Ranker v0 CV (`cv_report.json`)

5 folds (F0..F4, key-grouped). 4 seeds averaged, 500 rounds, lr 0.03, 63 leaves. Weights: each regime contributes its
mix share, and every structure counts once per regime. Queries with no positive are excluded from training and count
as 0 in evaluation.

| Regime | n | f·z-only MRR@25 / top-1 | **ranker v0 MRR@25 / top-1** | ranker MRR when the truth is a candidate |
|---|---|---|---|---|
| c1 | 293 | 0.629 / 0.495 | **0.864 / 0.802** | 0.873 |
| c2 | 1,028 | 0.775 / 0.672 | **0.924 / 0.893** | 0.930 |
| c3 | 1,028 | 0.057 / 0.046 | **0.056 / 0.043** | 0.662 (87 queries) |
| mix 0.20 / 0.55 / 0.25 | — | 0.566 / 0.480 | **0.695 / 0.662** | — |

- Rounds 100 / 200 / 300 / 500 give mix MRR 0.687 / 0.692 / 0.694 / 0.695, so there is no overfitting at 500 rounds.
- Per fold, c2 ranges from 0.896 to 0.947 and c1 from 0.788 to 0.883.
- CV took 70 min under contention.

**Feature importance** (gain share, `feature_importance.csv`):

| Feature | Gain share |
|---|---|
| fz_single_gap | 0.284 |
| fz_gap | 0.272 |
| fz_merged_gap | 0.103 |
| fp_ll_gap | 0.063 |
| fz_softmax | 0.028 |
| ap_rank | 0.026 |
| gen_rank_fz | 0.015 |
| lib_nspec | 0.014 |
| fz_rank | 0.014 |
| ap_gap | 0.012 |

No feature has zero gain.

**Where the gain comes from** (FACT, analysis of `cv_pred.parquet`):
- In c2, 90.6 % of train-origin decoys have `lib_hit = 1`: their own spectra are in the window and do not match. The
  masked truth has `lib_hit = 0`, like every COCONUT candidate.
- The ranker learns "this train candidate has library spectra and they don't match, so it is not the truth".
- That cuts the share of train-origin candidates among the top-5 decoys from 0.793 (f·z) to 0.699. Their base share is
  0.650.
- In c1, 87 % of the top-1 gains have a truth analog with Tanimoto ≥ 0.95. These are the visible other-library
  spectra.

### V5 (`v5_report.json`)

**(a) Per-regime CV MRR against the spec ranges.**

| Regime | Spec range | Measured | Verdict |
|---|---|---|---|
| c1 | ≥ 0.85 | 0.864 | PASS |
| c2 | 0.60–0.73 | 0.924 | **FAIL (above the range)** |
| c3 | > 0 | 0.056 (0.662 where generation finds the truth) | PASS |

- INFERENCE on c2: the spec range comes from prvsiyan's engine, so it is not a target for this pipeline.
- Here f·z alone already gives 0.775: `cft_ho2` is strong on these ≤ 470 Da bench truths.
- The ranker adds the "non-matching library spectra" rule, which is legitimate for real class-2 molecules, whose truth
  has no train spectra.
- The pilot c2 number is still optimistic. The ho2 truths are bench molecules ≤ 470 Da, and there are only 1,028
  of them. Treat 0.92 as an upper bound until the HO_R / R-A run.

**(b) KS distance between simulated queries (mix-weighted) and the 400 visible test molecules.**
- The visible test spectra are exact enveda-180 train rows. Unmasked, lib_max is 1.0 for 99 % of them (FACT,
  `test400`). That makes the unmasked run useless for comparison.
- So V5b uses the **SV protocol**: only the exact duplicate spectra are masked (`results/bench/sv_dup_rows.npy`).

| Statistic | Test median | Sim median | KS (mix) | Verdict |
|---|---|---|---|---|
| n_query | 3 | 3 | 0.089 | PASS |
| q_npeaks | 255 | 48 | **0.647** | **FAIL** |
| lib_max | 0.787 | 0.543 | **0.311** | **FAIL** |
| top_sim | 0.767 | 0.739 | 0.120 | PASS |
| n_cand | 112.5 | 96 | 0.120 | PASS |

- **q_npeaks.** Test spectra are raw, while simulated queries are L1-cleaned (0.1 % floor). Cleaning the test spectra
  with the same rule gives a median of 59 and **KS 0.160**.
  - **Action:** clean test spectra with `spectra.clean(.., 0.001, 512, 2.0)` before the engine, offline = online.
  - Alternatively, drop `q_npeaks`.
- **lib_max.** SV is class-1 by construction: all 400 truths keep their other enveda-180 spectra. The hidden test is a
  class mix, so this KS is not decisive. Even c1 alone differs (KS 0.452): c1 hides the whole query library, while SV
  keeps same-library spectra.
  - INFERENCE: add a fourth regime, "c1s" (hide only the query spectra, keep the other spectra of the same library),
    to cover SV-like molecules.

**(c) Provenance probe** (c2, CV predictions):
- Train-origin decoys in the top 5:
  - ranker 0.699; f·z 0.793; base share 0.650.
  - The ranker **demotes** train-origin decoys (−0.094 relative to f·z). It has not learned "train-style ⇒ truth".
- An explicit candidate-origin feature (`cand_src`, 1 seed) raises CV MRR:

  | Regime | Gain from `cand_src` |
  |---|---|
  | c2 | **+0.044** |
  | c1 | +0.036 |
  | c3 | +0.005 |

  - The simulation contains a provenance shortcut: the truth is always train-origin, and is the only train-origin
    candidate without library evidence.
  - At test, class-2 truths are COCONUT-origin, so the shortcut would hurt.
- **Verdict: FAIL on the shortcut criterion (> 0.02).**
  - The 86 current features do not expose origin directly, which the decoy-demotion direction shows.
  - **Rule:** never add origin-correlated features to the ranker (`src`, `n_train_sids`, train membership, popularity
    derived from train), unless the simulation also draws COCONUT-origin truths.

**SV evaluation of ranker v0** (400 visible test molecules, duplicates masked, truth = score key of `truth_SV`):

| Ordering | MRR@25 | top-1 |
|---|---|---|
| Ranker v0 | 0.930 | 0.878 |
| f·z only | 0.906 | 0.838 |

The truth is a candidate for 100 % of the molecules. Caveat: `cft_ho2` was trained on all 400 SV structures and on
their exact spectra (0 SV keys are in ho2), so both numbers are leaky. Only the direction (ranker ≥ f·z) is
informative.

## Findings to fix before the full run

1. **`split_v4r` is not closed under the score key** (FACT):
   - 377 non-HO_R sids share 350 HO_R keys;
   - 24 HO_R keys span more than one fold.
   - The driver drops these 350 keys (closure rule) and takes the fold from the pool-row sid. However, R-A trained on
     the current split will have seen spectra of those keys.
   - **Fix:** close HO_R under the score key before training R-A.
2. **Pool tail.**
   - Pilot truths are ≤ 470 Da.
   - The full run needs `featurize` (running) → `assemble` → `project --bits <R-A>`.
   - **Do not run `assemble` while a simulation is running.** On Windows the memory maps are locked, and pool row ids
     change.
   - Rows store keys, and c3 looks `drop_pid` up by key, so old shards stay interpretable. Do not mix shards from
     different pools in one ranker run.
3. **Test-spectrum cleaning** for q_npeaks: see V5b.
4. **RDKit version.** Tables were built with 2026.03.6. Generated-candidate keys on Kaggle must use the same version,
   or V0 must be re-checked there.
5. **Optional:** the regime "c1s" (see V5b), and a mass-stratified check of c2 MRR once truths go above 470 Da.

## Full HO_R run: size and wall-time estimate

**Size.** `build_queries.py --run hoR_preview --truths hoR` (FACT):
- 21,821 keys are in the pool; 350 are dropped by the closure rule; 3 have no spectra.
- **21,468 keys → 46,920 queries:** c2 21,468, c3 21,468, c1 3,984.
- Rows: about 125 per query, so about 5.9 M rows and about 1.6 GB of parquet (INFERENCE from the pilot ratio). The
  > 480 Da tail adds candidates.

**Wall time** (INFERENCE; the pilot measured 1.91 s per query wall with 4 workers under contention; I assume heavier
targets cost ×1.3):

| Where | Throughput | Total |
|---|---|---|
| This PC while `featurize` runs | ~1.9–2.5 s/q | **≈ 25–33 h** |
| This PC idle, 4 workers (RAM allows 4–5 at 1.3 GB each) | ~1.0–1.3 s/q | **≈ 13–18 h** |
| Kaggle CPU session (4 vCPU, 30 GB RAM), 4 workers, assuming 4–6 s per query per worker | 1.0–1.5 s/q | **≈ 17–25 h of session time** |

- On Kaggle that is **2–3 sessions of 12 h**. Run them in parallel with `--shard-of 0/3`, `1/3`, `2/3` for about 6–8 h
  wall each, then merge the `rows/` and `qstats/` folders.
- Ranker CV on about 5.9 M rows (20× the pilot): the pilot CV took 70 min under contention. Expect several hours on this
  PC (INFERENCE). Fewer seeds in CV (1–2) and 4 seeds only for the final fit cut that by 2–4×.

## Can the driver run as a Kaggle CPU kernel? Yes (INFERENCE; not tried)

It is pure CPU (numba, RDKit, torch CPU, pandas/pyarrow) and needs about 1.3 GB of RAM per worker. Build
`queries.parquet` locally, then upload these files:

**Code** (keep the relative layout, because `engine/model.py` finds `cft_model` at `../../train_cft`):
- `research/v4n_rebuild/engine/{__init__,chem,derive,engine,fingerprint,fragments,model,spectra,tables}.py`
- `research/v4n_rebuild/sim/{common,simulate}.py` (plus `ranker.py` if the ranker is trained there)
- `research/train_cft/cft_model.py`

**Tables** (`results/v4n/tables/`, about 2.6 GB now and about 3 GB after the tail):
- `spec_meta.parquet` (64 MB), `spec_off.npy` (20 MB), `spec_mz.npy` (533 MB), `spec_it.npy` (533 MB)
- `train_structs.parquet` (33 MB)
- `pool_meta.parquet` (36 MB, more after the tail)
- `fp_bits.npy`, `pool_fp.npy` (767 MB) and `train_fp_sel.npy` (386 MB), projected for the R-A bits. The raw
  `*_fp_raw.npy` files are not needed once projected.
- `pool_frag_off.npy` and `pool_frag_mass.npy` (203 MB)

**Other inputs:**
- Run inputs: `results/v4n/sim/<run>/queries.parquet` (about 20 MB) and the R-A checkpoint (about 205 MB).
- Not needed on Kaggle: `split_v4r`, `train.parquet`. `test.parquet` is needed only for `--test`, plus
  `sv_dup_rows.npy` and `truth_SV.parquet` for SV.
- RDKit wheel: the same version as the tables, see point 4 above.
- Point the tables at the dataset mount and the output at `/kaggle/working` with the environment variables
  `V4R_TABLES` and `V4R_SIM_OUT` (read in `common.py`), or with `--tables`.

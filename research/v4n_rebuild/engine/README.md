# v4r engine: phase A of the minimal rebuild (MVR build-order steps 1–3)

This is our own code, MIT-licensed. It is a v1-equivalent evidence engine plus its table builders, written against
`research/v4n_rebuild/REBUILD_SPEC.md` (sections 3.0–3.7, 4, 6 and 9).

The reference "v1 engine" (`ahmed_ref/code/casmi`, non-commercial) was read only to learn its behaviour. It was run
locally as a **test oracle** in `parity.py` (V3), on our tables. Its code, weights and tables are not copied and do not
ship. Its numba and bytecode caches were redirected to the scratch directory, so nothing was written into `ahmed_ref/`.

Labels: **FACT** = measured, with the command that measured it. **INFERENCE** = deduced.
All numbers below come from 2026-10-08 runs on this PC: i7-1165G7, 4 cores / 8 threads, often throttled to 1.0 GHz;
RDKit 2026.03.6, numba 0.67, torch 2.14 CPU.

## Module map

| File | What it does |
|---|---|
| `chem.py` | Generic adduct parser and neutral mass; standardiser (largest fragment, uncharge, no stereo, None if still charged); **score key** = `MolToInchiKey(TautomerEnumerator().Canonicalize(mol))[:14]` (same definition as `research/bench/eng/casmi_engine.canon_key`); `canonical()` = one Canonicalize call that returns both the key and the tautomer-canonical molecule; formula helpers |
| `fingerprint.py` | cfp1 raw layout, 16,384 bits: ecfp4 4096, ecfp6 4096, fcfp4 2048, path 2048, apair 2048, tors 2048 (same definition as `train_cft/cft_fp.py`). `select()` projects a fingerprint to a model's bit list |
| `fragments.py` | Fragments from one- and two-bond cleavages (CSR BFS, rounded to 1e-3 Da, unique, float32); peak-explanation kernel; ion offsets; subformula enumeration (cached per formula); the F and S feature functions |
| `spectra.py` | numba kernels: library cleaning; entropy weights; entropy, cosine and mass-shifted hybrid similarity; batched library scoring; spectrum merging |
| `derive.py` | Class-3 grammar: 45 reaction SMARTS with their formula change, applied as single steps and unordered pairs, with the v1 budget and caps |
| `model.py` | **Pluggable FP-model interface** (`nbits`, `fp_bits`, `prepare()`, `logits_el(items)`) and `CFTBank`, the adapter for one or more `research/train_cft` checkpoints (outputs averaged) |
| `tables.py` | Memory-mapped readers: `Library` (L1 + P2) and `Pool` (P4/P5/P6). Fingerprints are served in the bank's bit layout |
| `engine.py` | `Engine.run(spectra, target, exclude, exclude_sid, exclude_lib, drop_pid)` returns `(Cands, X[n, 86], info)`. Channels L, A, M, F, S and G; the 86 live v1 features (the 8 `xs_*` dropped); simulation masks |
| `build_tables.py` | Table builders: `featurize` (resumable), `assemble`, `project`, `lib` |
| `run_queries.py` | Runs the engine on a parquet of queries |
| `validate.py` | V0, V1, V2 checks. Output: `results/v4n/validation/V*.json` |
| `parity.py` | V3: our engine against the v1 oracle. Output: `results/v4n/parity/` |

## Tables (`results/v4n/tables/`, 5.6 GB including the 1.6 GB `_chunks/` resume store)

| ID | Files | Content |
|---|---|---|
| P2 | `train_structs.parquet` | 275,810 sids, in inchikey14 order. Columns: `smiles` (standardised; raw-charged fallback for 554 structures), `smiles_tc`, `key`, `mass`, `formula`, `n_heavy`, `n_spec`, `pid` (pool row of the sid's key), `charged` |
| P4 | `pool_meta.parquet` | Mass-ascending. Columns: `key` (unique), `smiles` (standardised, used as output), `smiles_tc`, `formula`, `n_heavy`, `src` (0 = train, 1 = COCONUT), `train_sid`, `n_train_sids` |
| P3 / P5 | `train_fp_raw.npy`, `pool_fp_raw.npy` | Raw cfp1 fingerprints, packed, 2,048 bytes per row. These serve **any** model's bit selection: cft_ho2 uses 11,184 bits, the R models 11,171 |
| P3' / P5' | `fp_bits.npy`, `pool_fp.npy`, `train_fp_sel.npy` | Copies in the model layout for the current bank (`project`; currently cft_ho2, 1,398 bytes per row) |
| P6 | `pool_frag_off.npy`, `pool_frag_mass.npy` | Depth-2 fragment CSR, float32, 90.2 masses per structure |
| L1 | `spec_meta.parquet`, `spec_off.npy`, `spec_mz.npy`, `spec_it.npy` | All 2,539,608 spectra of train.parquet. Peaks above precursor + 2 Da dropped, 0.1 % floor, top 512, m/z-sorted, base peak = 1. 133.2 M peaks |

Fingerprints and fragments are always computed from the **tautomer-canonical** molecule. This applies to the pool,
train structures and generated candidates (REBUILD_SPEC 3.0). Keys come from the standardised SMILES; that is also the
SMILES we would submit.

The pool reuses the 436,389 cleaned COCONUT SMILES of our existing 712,199-row pool
(`results/train_pkg/data_full`). Every derived column was recomputed.

### Build commands

```
python research/v4n_rebuild/engine/build_tables.py lib                       # L1: 139 s
python research/v4n_rebuild/engine/build_tables.py featurize --workers 8 [--coco-max-mass M]   # resumable
python research/v4n_rebuild/engine/build_tables.py assemble                  # P2/P4/P3/P5/P6 from finished chunks: 45 s
python research/v4n_rebuild/engine/build_tables.py project --bits models/cft_ho2_akriti/cft_ho2.pt   # model layout: 6 min
```

**KNOWN GAP: the pool is partial.**
- `featurize` was run with `--coco-max-mass 480`. The current pool is all 273,636 neutral train keys plus the 275,220
  COCONUT structures of ≤ 480.16 Da: **548,856 rows**, not about 712k.
- Cause (FACT): RDKit `TautomerEnumerator.Canonicalize`, which the metric key requires, costs 0.1–4 s per heavy
  COCONUT molecule. Single-thread measurements: 0.50 s mean at COCONUT index 335k, 4.1 s at 436k. With throttling, the
  first build attempt reached 50 min per 2,000-molecule chunk per worker.
- The remaining 80 chunks (158k COCONUT structures, 480–1,300 Da) are an estimated 4–8 h on this PC (INFERENCE).
- To finish: run `featurize --workers 8` with no mass limit (it resumes), then `assemble` and `project`.
- Impact:
  - (INFERENCE) Nothing for the visible test: its neutral masses are all ≤ 439.1 Da.
  - Simulated queries with targets above 480 Da lack their COCONUT decoys. Restrict them, or finish the tail, before
    phase B.

### Running the engine on queries

```
python research/v4n_rebuild/engine/run_queries.py --queries test.parquet --out results/v4n/runs/x.pkl \
       [--ckpt models/cft_ho2_akriti/cft_ho2.pt] [--limit N] [--no-generate] [--truth results/bench/truth.parquet --qset S12]
```

The `test.parquet` column layout is expected, one molecule per `molecule_id`. Target = median neutral mass of the
molecule's spectra. The output pickle holds, per molecule: `pid`, `key`, `smiles`, `formula`, `X` (86 features in
`engine.FEATURES` order), `info`, `sec`.

In code:

```python
E = Engine(Library(d), Pool(d), CFTBank([ckpt]), EngineCfg())
C, X, info = E.run(spectra, target, exclude=mask, exclude_sid=s, exclude_lib=-1, drop_pid=p)
```

`drop_pid` removes the pool row from the window but keeps it generatable. That is the class-3 simulation exception of
REBUILD_SPEC 4. For simulation, `Library.query_spectrum(i)` turns a library spectrum into a query spectrum.

## Validation results

### V0: chemistry. Command: `validate.py v0`

PASS (FACT, `results/v4n/validation/V0.json`).
- **Score key:** 2,000 / 2,000 equal to the bench metric-key function. The 2,000 SMILES are 1,200 train, 500 COCONUT
  and 300 bench truths.
- **`canonical()`** gives the same key on 2,000 / 2,000.
- **Idempotence:** the key of the tautomer-canonical SMILES equals the original key on only 1,998 / 2,000. So we never
  submit `smiles_tc`.
- **Neutral mass:** for all ten test adducts, checked against RDKit ion masses (`[H+]`, `[Na+]`, `[Cl-]`, …), the
  maximum error is 1.1e-12 Da (criterion < 1e-6).
- **Adduct parsing:** 120 of the 121 adduct strings in train and test are parsed. The exception is `[Cat]2+`
  (15 spectra).
- **Against the v1 hand table:** the 43 shared adducts agree to 5.1e-9 Da.

### V1: structure tables. Command: `validate.py v1`

PASS, except the pool size (FACT, `V1.json`).

| Check | Result |
|---|---|
| P2 rows | 275,810; sid dense and in key order; 0 failed; 554 charged (kept with raw-charged SMILES, not in the pool) |
| Pool | 548,856 rows (see the known gap); mass sorted; keys unique |
| Train keys in pool | 99.80 % of all train keys; 100 % of neutral train keys |
| Consistency | `P2.pid` and `pool.train_sid` consistent; 1,540 keys are shared by more than one sid (tautomer duplicates) |
| Spot check, 500 random pool rows | 0 key, 0 fingerprint, 0 fragment mismatches against recomputation |
| Kept bits (cft_ho2) | 11,184: ecfp4 1650, ecfp6 3840, fcfp4 908, path 2048, apair 1808, tors 930 |
| Schema oracle | The v1 `Pool` and `Library` readers load our files (548,856 rows, nbits 11,184; 2,539,608 spectra) |

### V2: spectrum cache. Command: `validate.py v2`

PASS (FACT, `V2.json`).
- 2,539,608 spectra, 0 with `sid = -1`, and every sid's inchikey14 matches P2.
- 133,228,108 peaks. An independent DuckDB recount of the cleaning rule gives the same total, with **0 spectra
  differing** (criterion within 1 %).
- 11,053 spectra are empty after cleaning; 17,235 are at the 512-peak cap.
- 95.1 % of spectra have a neutral mass within 10 ppm of the labelled structure mass.

### Kernel unit parity against v1 (FACT, ad-hoc scripts run during the build)

| Component | Test | Result |
|---|---|---|
| Similarity kernels (`weigh`, entropy, cosine, shifted) | 1,000 random library pairs × 3 shifts | max difference 3.9e-15 |
| Batched scoring | 2,000 spectra | max difference 1.3e-15 |
| `merge` | — | 0 / 1,000 mismatches |
| Fragments | 300 pool structures | 300 / 300 bit-identical |
| Subformula masses | 200 formulas | 200 / 200 identical |
| `derive` product sets | 60 parents × 13 deltas | 780 / 780 identical |

### V3: engine parity against the v1 oracle. Commands: `parity.py ours | oracle [--equalise] | compare`

**Setup.**
- Both engines ran on our tables with the same bank: `cft_ho2` through `CFTBank`.
- Queries: 193 bench S12 molecules, which is every S12 molecule with a target ≤ 470 Da (pool coverage).
- Every query uses the S2 mask: all spectra of every sid with the truth key are hidden, and so is the truth's analog
  representative. Every third query is run as c3 (truth pool row dropped). Generation is on.
- Harness adaptations, so both engines get identical inputs:
  - CFT peak prep;
  - our adduct index;
  - our tautomer-canonical fingerprint and fragment functions for generated candidates;
  - `ce_n = 1`;
  - our engine run with `merged_nm_cap = 8` and `gen_true_steps = False`.

**Results (FACT, `results/v4n/parity/report_*.json`).**
- **Candidate sets** (pool rows and generated keys): identical in **193 / 193** queries; 0 generated keys unique to
  either side.
- **Analog lists:** identical in 191 / 193. The other 2 differ only in the order of analogs with equal similarity
  (to 5 decimals).
- **Truth in candidates:** c2 100 % (129 queries); c3 40.6 % (26 of 64, recovered by generation). Identical for both
  engines.
- **Features:** after excluding the 45 queries with adduct-count ties (see 1 below) and equalising float noise (see 2),
  **17,316 / 17,330 rows equal to 1e-5 relative** (148 queries). The only deviating feature is `best_tan_np` in 1 query.
  It comes from the tie order of the analogs in 3 below.

**Every mismatch is explained:**
1. **Merged-view adduct (45 queries, all `fz*`, `fp_*`, `el_*` and dependent features).**
   - When the most frequent adduct of a polarity is tied, v1 picks `max(set(adducts), key=count)`. That depends on the
     string-hash seed, so v1 itself changed between two oracle runs: 7 queries differed in run 1 and 45 in run 2.
   - Ours is deterministic: ties go to the earlier adduct in `chem.ADDUCT_LIST`. This is a deliberate difference.
2. **Near-zero direct-search scores (raw oracle: `lib_hit` 60, `lib_top3` 76, `lib_max_rank` 54 queries).**
   - v1 keeps rounding noise such as 2.4e-15 as a "hit". That flips `lib_hit`, puts near-0 entries into `lib_top3`
     means and reorders rank ties.
   - Ours clamps scores ≤ 1e-6 to 0, a deliberate fix. With the same clamp applied to the oracle (`--equalise`) these
     differences vanish.
   - Before the clamp, our first run differed from the raw oracle in only 20 / 12 / 6 queries on these features.
3. **Ordering of exact-tie analogs (2 queries).** This is float-level noise in the hybrid similarity between two kernels
   that agree to 1e-15.

**Runtime per query (FACT, our engine, 193 queries, 6 numba threads, CFT on CPU, PC shared and throttled).**
- Median **1.5 s**; mean 3.4 s; p90 4.2 s.
- The maximum was 81 s, in generation-heavy queries with 150 generated candidates; tautomer canonicalisation of the
  products dominates.
- The v1 oracle on the same inputs: median 1.6 s, mean 5.6 s.
- Typical candidate counts: median 84; 18 generated on average.

Sanity check (FACT): the f·z-only ranking gives MRR@25 0.497 on c2 and 0.193 on c3. This is no ranker, just the
`cft_ho2` score.

## Deliberate differences from v1 (all recorded)

1. **Fingerprints:** cfp1, with the bit selection of the bank model, computed on the tautomer-canonical form
   (REBUILD_SPEC 3.0). v1 used a 10,226-bit selection with MACCS, on the standardised form.
2. **Peak prep for the FP model:** the bank's own (CFT: 128 peaks, precursor + 1.5 Da, window-diversified), with an
   instrument token.
3. **Model-view semantics:**
   - single views have `nm = 1`;
   - merged views have `nm` = number of spectra, capped at 16 (`merged_nm_cap`), where v1 used CE-count semantics;
   - the merged-view adduct is chosen deterministically.
4. **`gen_steps` = real combo length (1 or 2).** v1 always wrote 1. Flag: `gen_true_steps`.
5. **Direct-search similarity ≤ 1e-6 → 0** (noise clamp).
6. **Grammar:** 45 transforms. The duplicated `dehexose2` is dropped; it can only consume budget.
7. **Adducts:** parsed from the string instead of a hand table (agrees to 5e-9 Da). The adduct index list is ours and is
   recomputed from the string at load.
8. **Generated candidates:** fragments are cast to float32 like the pool table (REBUILD_SPEC subtlety 7). Fingerprints
   and fragments come from the tautomer-canonical form.
9. **Unchanged v1 semantics:**
   - library evidence is attributed through `pool.train_sid` only, so other sids sharing the key (1,540 keys) do not
     contribute;
   - the analog representative is chosen on the unmasked library;
   - the 10 ppm window, with a 30 ppm fallback.

## Known gaps before phase B (ranker simulation)

1. **Pool tail.** COCONUT above 480 Da (158k structures) still needs `featurize` (estimated 4–8 h CPU) followed by
   `assemble` and `project`. Until then, limit simulated queries to targets ≤ 470 Da.
2. **Bank.** `cft_ho2` saw almost all of HO_R in training; its hold-out is only the 1,185 ho2 keys.
   - The R-A checkpoint must replace it for real simulation rows. It uses a different bit list (11,171), so run
     `project --bits <R-A ckpt>`.
   - The raw tables need no rebuild.
3. **Simulation driver.** Not written yet. That means:
   - building query spectra from `Library.query_spectrum`;
   - the c1, c2 and c3 masks of REBUILD_SPEC 4, with the mask covering every sid that shares the truth key;
   - subsampling the number of spectra;
   - weights.
4. **Runtime tail.** Generation-heavy queries take 30–80 s. Caching canonical keys per product SMILES would cut this.
   So would running simulation shards as separate processes (RSS per process not measured).
5. **RDKit version.** Tables were built with RDKit 2026.03.6 on this PC; the competition wheel is 2026.03.3. Keys and
   tautomer forms must be re-checked on Kaggle (V0 there) before shipping.
6. **Families.** Not started: fe_v4 (frag2, analog_struct, fragnet, derivation priors) and DreamsFP.

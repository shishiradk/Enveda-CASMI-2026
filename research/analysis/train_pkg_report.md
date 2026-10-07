# Warm-up training package: held-out retrain of the two fingerprint networks

Date 2026-10-02. Code `research/train_pkg/`, prepared data `results/train_pkg/data/` (1.61 GB, also the staged
upload folder), smoke evidence `results/train_pkg/smoke_out/export/`. Nothing was uploaded or pushed.

## 1. The public method (source)

Source: public Kaggle notebook `prvsiyan/analog-propagation-casmi-2026-baseline` (pulled with
`kaggle kernels pull`). The training script is the string `TRAIN_SCRIPT_SRC` in code cell 1 (written out as
`train_fingerprint_model.py`, "not executed here; needs a GPU and ~2 h"); the model is the `fpmodel` cell.
megayak has no separate training script (its dataset README only reports a retrain of the same architecture).

Licence: **not determined.** `kaggle kernels pull -m` metadata and the Kaggle MCP `get_notebook_info` response
contain no licence field, and the web page is rendered client-side. (Kaggle shows the licence on the notebook
page; someone should read it there.) The weights dataset `casmi26-fp-models-v2` is CC0-1.0, the COCONUT candidate
dataset CC BY 4.0 (both from `kaggle datasets metadata`).

| Item | Public recipe |
|---|---|
| Network | FPNet: log-spaced sinusoidal m/z and neutral-loss embeddings + intensity -> Linear; a global token (precursor embedding, CE/100, mode, log1p(prec)/10, adduct and instrument embeddings); 6 pre-norm transformer blocks, d 512, 8 heads, dropout 0.1; head on [CLS, mean] -> 2048 -> 6,930 logits. 36.0 M parameters |
| Input | `prep_peaks`: peaks <= precursor + 1.5, >= 0.1 % of base peak, at most 128 (8 per 50 Da window first), m/z ascending, sqrt(i / max) |
| Target | ECFP4(4096) + ECFP6(4096) + RDKitFP(2048, maxPath 6) + MACCS(167), the 6,930 bits of `fp_bits.npy` |
| Loss | BCE on the bits + 1.0 x softmax cross-entropy of `f.z` over the truth and 63 decoys from the +-10 ppm mass window of the pool (COCONUT + training structures; random decoys when the window has no isomer); score centred per example and divided by sqrt(nbits), in fp32 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.01, betas (0.9, 0.98), gradient clip 1.0, fp16 autocast + GradScaler |
| Schedule | linear warm-up 2,000 steps, cosine to 2 % of lr at step 80,000; batch 256 spectra sampled uniformly |
| Augmentation | peak dropout (rate U(0, 0.30) per spectrum), intensity x exp(N(0, 0.25)), m/z x (1 + N(0, 5e-6)) |
| Merged model | `--merge_p 0.6`: 60 % of inputs are 2-4 spectra of the same structure concatenated, peaks closer than 0.005 m/z reduced to the stronger, top-N kept. Single model: merge_p 0 |
| Selection | validation every 2,000 steps on 2,048 random held-out spectra; keep the best hard-negative top-1 |
| Public files | `fp_single_s2.pt` step 20,000, `fp_merged_m1.pt` step 24,000; dict keys `model, step, nbits, d, layers` (87 tensors, fp32, no optimiser state) |

Not published by the author: the script that builds `spec.npz` (spectrum arrays and the `is_val` flag), the
validation fraction, the seeds, the exact command lines (total steps actually run) for s2 and m1.

## 2. What we reproduce, and how

- **Adapted (copied, kept identical for load compatibility):** `fp_model.py` = `ADDUCT_LIST`, `INSTR_LIST`,
  `instr_family`, `prep_peaks`, `SinEmb`, `Block`, `FPNet` (one cosmetic change in `prep_peaks`: a set is built
  once instead of per loop iteration; same result).
- **Re-implemented:** data preparation (`prep_data.py`), batch building, negative sampling, loss, schedule,
  training loop, checkpoint/resume, export (`train_fp.py`), checks (`check_result.py`, `compat_test.py`).
  The merge rule and the fingerprint recipe follow the public code line by line.

Deliberate differences from the public script:

| Difference | Reason |
|---|---|
| Proxy molecules (and their tautomers) removed from spectra AND from the decoy pool | the purpose of the job |
| Validation split: 1 molecule in 50 by md5(InChIKey14) | the author's split is unpublished |
| Fixed 4,096 validation spectra with fixed decoys, every 1,000 steps (public: 2,048 fresh random, every 2,000) | less noise in best-checkpoint choice; resumable |
| Stop at step 40,000, or 10,000 steps without improvement (lr schedule still 80,000 long) | public best steps were 20k/24k and the author reports decay afterwards; saves GPU hours |
| Seeds: single 2, merged 1; model initialised after seeding; one RNG stream per step | author's seeds unknown (file names suggest 2 and 1: a guess); makes resume order-independent |
| Decoys drawn with a vectorised formula, m/z sorts use a stable sort | same distribution; speed and determinism |
| Pool kept bit-packed on the GPU (0.6 GB) and unpacked per batch | public keeps 4.9 GB unpacked on the GPU, too much for 8 GB VRAM |
| Batch building in numba in background threads | public Python loops would starve the GPU |
| Collision energy = mean of abs(values), missing -> 25.0; precursor rounded to float32 before the peak filter | matches the E1 engine (`casmi_engine.parse_ce`, float32 precursors) |
| Export names `fp_single_ho1.pt`, `fp_merged_ho1.pt` | cannot be confused with the public files; `merged` in the name is what the engine keys on |

Inherited quirks kept on purpose: merging at training time works on already prepared (sqrt, 128-capped) peaks and
keeps the metadata of one spectrum, while the engine merges raw peaks, uses the median precursor and CE 25;
spectra are sampled uniformly, so heavily measured molecules dominate; merge peers may mix adducts.

## 3. Data (`results/train_pkg/data/`, built by `prep_data.py` in 335 s, DuckDB only)

| | spectra | molecules (InChIKey14) |
|---|---|---|
| train.parquet | 2,539,608 | 275,810 |
| removed: in the two held-out key files (585 keys, 552 present in train) | 59,298 | 552 |
| removed: tautomers of held molecules (7,703 same-formula molecules checked, same tautomer-canonical key) | 529 | 17 |
| removed: no peak left after `prep_peaks` | 10,736 | 13 (molecules left without spectra) |
| **kept** | **2,469,045** | **275,228** |
| train split | 2,417,823 | 269,723 |
| validation split | 51,222 | 5,505 |

- Checked after the build: 0 held keys in the pool, 0 held molecules with a target, 0 held spectra kept.
  All 1,184 enveda-np-examples spectra are gone (all 250 molecules are held).
- 105.1 M peaks, mean 42.6 per spectrum, 9.4 % of spectra at the 128 cap. Unknown adduct 2,152; CE missing 322,066.
- Pool: 711,626 structures = 436,385 COCONUT (4 removed as held) + 275,241 training structures, RDKit failures 0.
  The shipped COCONUT file contains no training key, so every target fingerprint is computed here with RDKit
  2026.03.6. 4,992 of 5,000 sampled pool fingerprints are bit-identical to the bench engine's pool cache; the
  other 8 were not investigated in detail (largest mass difference 1.00728, one proton, which suggests keys that
  occur in the bench pool under another structure form).
- Files (plain `.npy`, memory-mappable): `spec_off/mz/it` (CSR peaks, 420 MB each for mz and it), `spec_prec/ad/
  ins/ce/mode/lib/row/mol/split`, `mol_key/pool/split/fold`, `mol_smiles.tsv`, `pool_fp` (bit-packed, 617 MB),
  `pool_mass/key/src`, `pool_smiles.txt`, `fp_bits.npy`, `meta.json` (all counts). `spec_row` is the row number in
  train.parquet; `mol_fold` is a spare 5-fold id. These are reusable for the later contrastive model (tokens,
  SMILES, split are already there).
- RAM: training loads about 1.6 GB (or uses `--mmap`).

## 4. Smoke test (this machine, CPU, torch 2.14.0+cpu, Python 3.14)

`train_fp.py --smoke` = real architecture (36.0 M parameters), 2,000 training molecules (18,809 spectra), batch 32,
60 steps per model, validation every 20 steps, checkpoint every 12 s.

1. Run 1 with `--stop_after 25`: stopped at step 25 of `single`, exit code 2, checkpoint + partial export written.
2. Run 2, same command: `[single] RESUMED from step 25 (best 0.0312 at 20)`, finished `single` (60 steps), trained
   `merged` (60 steps), exported both files, exit code 0. Wall time 2 min 16 s + 8 min 44 s.
3. Loss fell as expected (single: BCE 0.693 -> 0.489, contrastive 4.14 -> 2.87; chance top-1 is 1/64 = 0.016,
   validation top-1 0.03-0.06 after 60 steps: a pipeline test, not a trained model).
4. `check_result.py`: both files OK, `RESULT: ALL OK`. `run_train.bat` and `check_result.bat` were run through
   `cmd` with a temporary venv (start, the exit-code-2 message, summary file). `setup_windows.bat` was NOT run in
   full (3 GB download); only its Python-detection block was run. `send_results.bat` and `smoke_test.bat` were not run.
5. Batch building at batch 256: 2.0 ms (single) and 7.2 ms (merged) per batch on one thread.
6. Pipeline sanity with real weights: the PUBLIC networks, scored through our validation code on 256 of our
   validation spectra, give hard-negative top-1 0.789 (single) and 0.836 (merged). The author reports 0.45-0.51 on
   his own held-out molecules, so our validation molecules were very likely in his training set: the same
   memorisation the bench found. megayak's README reports 0.436 / 0.476 for a leak-free retrain of this
   architecture: that is the range to expect from this job.

Not tested: any GPU run, mixed precision, VRAM use at batch 256, resume on GPU, the Kaggle notebook, exact
bit-for-bit equality of a resumed run with an uninterrupted one.

## 5. Compatibility proof (`compat_test.py`, output in `smoke_out/export/compat_test.json`)

- `research/bench/eng/pv_fp.load_fp_models` (the engine's loader, unmodified) loaded the two exported files:
  1 single + 1 merged, nbits 6930.
- Checkpoint structure equals the public files: same top-level keys (`d, layers, model, nbits, step`), same 87
  tensor names, shapes and dtypes, same Python types of the scalars.
- `pv_fp.molecule_logits` on raw train.parquet spectra of 40 validation molecules (222 spectra): finite 6,930-vector
  for every molecule.
- Featurisation parity: the engine's logits from RAW peaks and our network on the PREPARED arrays differ by
  0.0 (max abs) over those spectra, so the prepared data is exactly what the engine feeds the network.

## 6. Time estimate (GUESS, not measured on any GPU)

CPU: 0.21 step/s at batch 32 (4 threads), about 0.026 step/s at batch 256. One epoch = 9,445 steps at batch 256.
Guess for a T4 with mixed precision: 4-8 step/s, so **20-40 min per epoch**, 1.4-2.8 h per model if it runs to
step 40,000, **3-6 h for both** (less if the patience stop triggers). An RTX 5050 should be similar or somewhat
faster. The author's "~2 h on a T4" is unverified and may refer to one model.

## 7. Open risks

1. No GPU run has been done; time, VRAM at batch 256 and AMP stability are unverified. `--bs 128` is the fallback,
   but it changes the recipe (the lr is not rescaled).
2. PyTorch build for the RTX 5050: `torch==2.11.0` from the `cu128` index exists for Windows, Python 3.10-3.14
   (verified on download.pytorch.org). That CUDA 12.8 builds are what 50-series cards need is from memory; the
   official install page could not be read (JavaScript). `check_env.py` runs a real GPU operation, so a wrong
   build is detected at setup. Fallback: `torch==2.13.0`, `cu130`.
3. `fp_bits.npy` was selected by the author from bit frequencies over all training structures, held ones included
   (negligible, and required for engine compatibility).
4. Tautomer exclusion covers training molecules only where the held molecule is in train (552 of 585 keys);
   COCONUT entries that are tautomers of a held molecule stay in the decoy pool.
5. The validation molecules are not the proxy molecules; the real test of the new networks is the bench (S2/S3i),
   to be run after the files come back. Best-step selection uses our validation split only.
6. The ranker rows (`rank_train.npz`, megayak rows) were built with the public networks' features; swapping the
   networks without refitting the rankers changes the feature distribution.
7. Seeds and total steps of the public s2/m1 runs are unknown, so this is a reproduction of the method, not of
   the exact weights.
8. Kaggle: the notebook and `kernel-metadata.json` are untested there; the dataset mount path is found by glob.
   The package contains competition training data, so the dataset must stay private and team-only.
9. Notebook licence unknown (section 1) for the adapted `fp_model.py` code.

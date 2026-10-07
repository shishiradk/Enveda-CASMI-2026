# CFT training package: our contrastive spectrum -> fingerprint transformer

Date 2026-10-03. Code `research/train_cft/`, new data and staged upload folder `results/train_cft/data/`
(2.13 GB, 37 files), smoke evidence `results/train_cft/smoke_out/export/`. Nothing was uploaded, pushed or
submitted. No existing file was modified. No GPU run exists yet: every time, memory and quality statement about
the GPU is a guess and is labelled as such.

## 1. What the model is for

Rank the true structure among all structures of the same mass. The pool window (COCONUT + training structures)
has a median of 92 candidates; a PubChem window has a median of **about 11,000** (measured here, section 3).
The model must therefore be judged on PubChem-like candidates, not only on the pool. That requirement drove
three choices that go beyond the public description: PubChem decoys in the negative pool, the PubChem form of
the truth as an alternative positive, and a PubChem validation set.

## 2. Design

Legend: **P** = taken from the public description (`v4n_method_spec.md` section 2, ideas only, no code read or
copied); **W** = same as our warm-up package; **O** = our own choice.

| Item | Choice | Source | Reason |
|---|---|---|---|
| Peaks | The prepared arrays as they are: peaks <= precursor + 1.5, >= 0.1 % of base peak, at most 128, m/z ascending | W | No rebuild; 9.4 % of spectra reach the cap. The public model uses 160 peaks; not worth re-preparing 2.5 M spectra |
| Peak token | Linear over [Fourier(m/z), Fourier(precursor - m/z), s, s^2], s = sqrt(relative intensity) | P | Set of peaks, no binning |
| Fourier features | d/2 wavelengths log-spaced 0.01 .. 2000 Da, sin and cos; angle computed in float64 | P (form), O (range, float64) | float32 loses phase at 0.01-Da wavelengths for m/z 1000 |
| Global token | Linear over [Fourier(precursor), Fourier(CE) x CE-known, CE-known flag, n_merged / 4, log1p(precursor) / 10, polarity] + adduct embedding + instrument embedding | P, instrument: W | Test is all timsTOF, training is 47 % timsTOF: the instrument token lets the model separate them |
| Encoder | Pre-LayerNorm transformer, GELU, dropout 0.1, key padding mask; pooled = [global token, mean of peak tokens] | P | |
| Presets | `default`: 8 layers, width 512, 8 heads, head 2048 = **51.3 M parameters**. `small`: 4 layers, width 256, 4 heads, head 1024 = 15.5 M. (`tiny` for CPU tests only) | P (default), O | default matches the publicly described size |
| Outputs | Logits z over 11,170 fingerprint bits; second head: log1p counts of C, H, N, O, P, S, F, Cl, Br, I | P | Candidate score = f . z |
| Fingerprint "cfp1" | ECFP4 4096, ECFP6 4096, FCFP4 2048, RDKit path (max 6) 2048, atom pair 2048, topological torsion 2048; bits with frequency in [0.005, 0.995] on 100,000 structures (60 % pool, 40 % PubChem) -> **11,170 bits** | P (blocks, rule), O (no MACCS, sample includes PubChem) | See 2.1 |
| Loss | cross-entropy over (truth + 63 decoys) of (f.z - mean) / sqrt(nbits) x exp(learned scalar) + 1.0 x BCE(z, truth bits) + 0.2 x smooth-L1(element counts) | P (contrastive, K = 63), **weights are our guess** | See 2.2 |
| Decoys | From the +-10 ppm window of the pool, never the truth's own key, without repetition: first up to 50 % same-formula structures, then the rest of the window, then fillers from the 3,000 pool rows on each side of the window | P (window), O (same-formula quota, near-mass fillers) | Isomers are the main error class; random far-mass fillers (public description) teach little |
| Decoy pool | Existing pool (711,626) + 400,000 PubChem structures drawn from the windows of the training molecules + 14,662 PubChem-form duplicates | O | Median training window in PubChem is 10,792 structures; the pool alone has none of them |
| Positive | The training SMILES; with probability 0.5 the PubChem SMILES of the same InChIKey14 when its fingerprint differs | O | At inference a PubChem-only truth arrives in PubChem's representation. Only 5.8 % of molecules differ (2.4 below) |
| Balancing | Half of a batch uniform over spectra, half uniform over (molecule, polarity) | P | Spectra per molecule: median 6, maximum 1,328 |
| Merged inputs | 30 % of inputs are 2-4 spectra of the same molecule **and adduct**, pooled, peaks closer than 0.005 Da collapsed to the stronger, 128 strongest kept | P (merging), O (same adduct, 0.3) | With mixed adducts "precursor - m/z" is wrong for part of the peaks |
| Augmentation | Peak dropout rate U(0, 0.3) with at least 3 peaks kept; intensity x exp(N(0, 0.3)); m/z and precursor x (1 + N(0, 3 ppm)); CE hidden with probability 0.1 | P, values close to the description | |
| Optimiser | AdamW 3e-4, betas (0.9, 0.98), weight decay 0.01 (not on biases, norms, embeddings), 2,000 warm-up steps, cosine to 5 % at step 40,000, batch 256, gradient clip 1.0, EMA 0.999 | **our guess** (not public) | Same family as the warm-up recipe; EMA is mentioned in the public material for another model |
| Stopping | 40,000 steps, or 12,000 steps without a better validation MRR; best checkpoint (EMA or raw, whichever validates better) is exported | O | |
| Validation | Section 5 | P (window MRR), O (query definition, PubChem set) | |
| Inference | `CFTScorer.score(spectra, candidate fingerprints)`: each spectrum is a "single" view; spectra sharing an adduct also give one merged view; molecule logits = mean of single views, averaged 50/50 with the mean of merged views when there are any; at most 16 spectra | P (50/50 rule), O (per adduct) | Training-time validation calls the same functions |

### 2.1 Fingerprint decision

The existing 6,930-bit selection would keep compatibility with our candidate caches, but:
- measured on all 5,505 validation molecules, 57 of them (1.0 %) have at least one same-mass pool candidate with a
  **bit-identical** 6,930-bit fingerprint (244 candidates); with cfp1 the number is **0**, also among 188,739
  PubChem candidates;
- the strongest public result uses a multi-block fingerprint of this kind;
- the warm-up job already trains the 6,930-bit variant, so using it again would give no new information.

MACCS was dropped: it is 42 % of the fingerprint time (1.9 of 4.6 ms per molecule) for 139 bits, and inference on
PubChem windows is fingerprint-bound. cfp1 costs about 2.7 ms per structure. Candidates must be fingerprinted with
`cft_fp.Fingerprinter(bits)`; the bit list is stored inside the exported model.
Whether cfp1 beats the 6,930 bits for ranking is **not measured**; `eval_cft.py` gives the comparison against the
warm-up networks on identical candidates once both are trained.

### 2.2 Loss weights (unknown publicly)

One CPU experiment (`results/train_cft/ablation_w_bce/`): `small` preset, 2,000 training molecules, 300 steps,
batch 32, validation on 400 molecules.

| BCE weight | train contrastive loss at step 300 | train BCE | best pool MRR | pcv MRR at step 300 |
|---|---|---|---|---|
| 0 | 1.89 | 1.65 (uncalibrated) | 0.117 | 0.105 |
| 1 | 2.16 | 0.48 | 0.119 | 0.105 |
| 10 | 2.95 | 0.23 | 0.119 | 0.095 |

Reading: a large BCE weight slows the contrastive fit; validation differences are inside noise (400 molecules, and
2,000 training molecules are memorised quickly). Default set to **1.0**. This is weak evidence at toy scale; the
weight remains the main untested hyper-parameter (`--w_bce`).

### 2.3 What is deliberately not done

- No instrument-specific fine-tuning, no second (per-polarity) model, no MS1 / formula input.
- No tautomer canonicalisation of structures (too slow for PubChem windows; the alternative-positive trick is the
  substitute).
- The warm-up code is not imported by the model, the trainer or the inference API. `eval_cft.py` imports
  `fp_model.py` only to run the warm-up baseline.

## 3. Data additions (`research/train_cft/prep_cft.py`, 40 min, 3 workers, DuckDB for all parquet reads)

Input: the warm-up data (unchanged) + `external/pubchem/pubchem_rows_pop.parquet` (100,955,275 rows, sorted by
mass, 150-1170 Da) + the two held-key files.

| File(s) | Content | Size |
|---|---|---|
| `cfp_bits.npy` | 11,170 selected bit positions (ECFP4 1655, ECFP6 3826, FCFP4 904, path 2048, atom pair 1808, torsion 929) | 0.1 MB |
| `cpool_fp.npy` | 1,126,288 x 1,397 bytes, bit-packed cfp1, pool sorted by mass | 1,573 MB |
| `cpool_mass/key/kid/form/elem/src/orig.npy`, `cpool_smiles.txt`, `cft_formulas.txt` | mass, InChIKey14, key id, formula id (163,183 formulas), element counts, source (0 COCONUT, 1 training, 2 PubChem decoy, 3 PubChem form), row in the old pool | 118 MB |
| `mol_cpool.npy`, `mol_alt.npy` | molecule -> pool row of its training-form / PubChem-form structure | 2 MB |
| `pcv_*.npy`, `pcv_smiles.txt` | PubChem validation: 500 validation molecules, 188,739 sampled same-mass PubChem structures (median 384 per molecule) in cfp1 and in the 6,930-bit fingerprint, fingerprint of the truth from its PubChem SMILES, full window sizes | 438 MB |
| `cft_meta.json` | all counts and formats | |

Facts measured during the build:
1. PubChem +-10 ppm window of the training molecules: 10th / 50th / 90th percentile 2,046 / 10,792 / 20,362
   structures. 267,599 of 269,736 training molecules have a window.
2. 252,866 of 275,241 molecules (92 %) have their InChIKey14 in PubChem.
3. PubChem decoys: 2 drawn per training molecule -> 513,498 new keys -> 400,000 kept (disk budget). No RDKit failure.
4. PubChem form vs training form: identical cfp1 fingerprint for 238,204 molecules; 14,662 (5.8 %) differ and were
   added as alternative positives (14,348 training, 314 validation). The representation gap is small.
5. Held-out proxy keys (585 + 17 tautomer keys) are excluded from the PubChem decoys and from the PubChem
   validation candidates; they were already absent from the old pool.
6. A training batch (measured): 41 % of decoys are PubChem structures, 52 % share the truth's formula, 86 % are
   inside the +-10 ppm window (the rest are near-mass fillers), 28 % of inputs are merged.
7. Validation windows (all 5,505 molecules): 1,361,354 candidates, median 187 per molecule (10th / 90th percentile
   22 / 540), 54 % same formula, 51 % PubChem decoys.
8. 1.5 % of the PubChem validation candidates are also training decoys (same fingerprint).

No per-molecule window index is stored: a window is two binary searches on `cpool_mass`.

## 4. Smoke test (this machine, CPU, torch 2.14.0+cpu, Python 3.14)

`train_cft.py --smoke --preset small` = real code path, 15.5 M parameters, 2,000 training molecules (18,809
spectra), batch 32, 80 steps, validation every 20 steps on 100 pool-window molecules + 40 PubChem-validation
molecules.

1. Run 1 with `--stop_after 30`: stopped at step 30, **exit code 2**, checkpoint and partial export written.
2. Run 2, same command: `RESUMED from step 30 (best MRR 0.0785 at 20)`, finished at step 80, exported
   `cft_small.pt` (62 MB), **exit code 0**.
3. Loss: contrastive 4.147 (chance = ln 64 = 4.159) -> 3.223; BCE 0.693 -> 0.441; element loss 0.625 -> 0.045;
   in-batch top-1 0.019 -> 0.294. Validation pool MRR 0.078 -> 0.094 (chance 0.066): a pipeline test, not a
   trained model.
4. `eval_cft.py --smoke`: exit code 0, `RESULT: ALL OK`; printed CFT, the prior baseline and the warm-up baseline
   (the 60-step smoke networks in `results/train_pkg/smoke_out/export`) on the same candidates. The inference API
   and the training-time validation path differ by 1.4e-6 in the logits.
5. `parity_test.py`: `CFTScorer.prepare` on RAW `train.parquet` rows (DuckDB) reproduces the prepared arrays for
   5,000 of 5,000 spectra (1,373 of them at the 128-peak cap): peak count, values (max difference 0.0), adduct,
   collision energy, precursor. `CFTScorer.score` on raw spectra agrees with the validation code (1.6e-7 relative).
6. `default` preset: 8 steps on CPU, 51.3 M parameters, exported file 205.5 MB, loss finite and falling.
7. `run_train_cft.bat` and `check_result_cft.bat` were run through `cmd` in a hard-linked copy of the two data
   folders with a temporary venv (`tiny` preset): both ended with exit code 0 and the expected messages.
8. Batch building at batch 256: 31 ms per batch in RAM on one thread (189 ms with `--mmap` and a cold cache).

The GPU path differs from the tested CPU path only in `device` and in `torch.autocast` + `GradScaler` being active.

After the smoke run two edits were made to `train_cft.py`: an array copy in the metric code (removes a PyTorch
warning) and the helper call path in the `.bat` files. `eval_cft.py` and both `.bat` files were re-run after them;
the training smoke was not.

Not tested: any GPU run, mixed precision, the Kaggle notebook, `send_results_cft.bat`, `smoke_test_cft.bat`,
bit-for-bit equality of a resumed run with an uninterrupted one.

## 5. How to judge a trained model

`check_result_cft.bat` / `eval_cft.py` prints, for all 5,505 validation molecules (query = up to 6 spectra from one
library, timsTOF `enveda-180` when the molecule has it, combined by the inference API):

| Row | Candidates | Use |
|---|---|---|
| `pool` | all same-mass structures of the CFT pool (median 187) | selection metric during training |
| `pool_orig` | the same without PubChem decoys (median about 90) | comparable to public window-MRR numbers and to the warm-up networks |
| `pool_isomers` | same-formula candidates only | the "wrong isomer first" failure |
| `pcv` | up to 384 random same-mass PubChem structures, truth in its PubChem form | **the number for admitting PubChem candidates**; also printed as an estimate for the full window (rank scaled by window size / sample size) |
| `pcv_truth_train_form` | as `pcv`, truth as the training SMILES | size of the representation gap |

Each row has MRR@25, top-1, median rank and the MRR of a random order. Baselines on identical candidates:
`prior` (no spectrum) on every row, and the warm-up networks on `pool_orig` and `pcv` when `fp_single*.pt` is found
(`out\export` of the warm-up job, or `--baseline <folder>`).

Decision guide:
- The model works if `pool_orig` MRR is in the range the public material reports for this kind of model on unseen
  structures (0.47-0.58; those numbers use other splits and query definitions, so this is an order of magnitude,
  not a target) **and** it beats the warm-up networks on `pool_orig` and on `pcv`.
- PubChem admission is worth building only if the full-window `pcv` estimate is clearly above what our current
  scorers reach (proxy MRR about 0.2 when the truth is only in PubChem, measured with a different protocol). No
  threshold has been validated.
- If `pool` is good and `pcv` is poor, the model separates natural-product-like decoys but not PubChem ones: train
  with more PubChem decoys (rebuild with a larger `--max_decoys`) before changing the architecture.
- The validation molecules are not the proxy molecules; the bench remains the final test.

Known bias of the metric: PubChem candidates that are tautomers of the truth count as wrong (the competition
metric would count them as right), so `pcv` is slightly pessimistic. Validation structures are in the pool and are
seen as decoys during training, which is also slightly pessimistic.

## 6. Time and memory: GUESSES (nothing measured on a GPU)

| Quantity | Guess | Basis |
|---|---|---|
| T4, `default`, batch 256, mixed precision | 2-3.5 steps/s -> 280-500 s per 1,000 steps -> **3-6 h for 40,000 steps**, plus about 30 min of validation | scaled from typical BERT-base throughput on a T4 by layers x width^2 |
| RTX 5050, same settings | 1.5-2x the T4 speed | unverified |
| VRAM, `default`, batch 256 | 5-6.5 GB | activations about 4 GB, weights + optimiser + EMA about 1 GB, candidate bits about 0.4 GB |
| VRAM, `small` | 2-3 GB; speed 3-4x `default` | |
| Host RAM | about 3 GB (`--mmap`: about 1 GB) | array sizes |
| Disk for outputs | checkpoint about 0.8 GB, best + export 0.2 GB each | |

If VRAM is short the script exits with code 3 and says to use `--bs 128` (the learning rate is not rescaled).
On Windows a nearly full GPU can also become very slow instead of failing; the log prints `vram` and `step/s`
every 200 steps, and the run sheet asks for these two numbers after 15 minutes.

CPU reference: `small` 0.49 step/s and `default` 0.11 step/s at batch 32 on 3 threads.

## 7. Open risks

1. **Loss weights, learning rate, schedule, step count are guesses.** The public values are unknown. Only the BCE
   weight has any evidence (2.2), at toy scale. If the first run's validation curve is still rising at step 40,000,
   rerun with `--steps 60000` (a resume with a changed schedule is allowed and logged as a warning).
2. No GPU run: AMP stability, VRAM at batch 256 and speed are unverified.
3. cfp1 versus the 6,930-bit fingerprint is undecided until both models exist. cfp1 is incompatible with existing
   candidate caches: PubChem and pool candidates must be fingerprinted again (about 2.7 ms each).
4. The selection metric (`pool`) is dominated by pool-like structures; `pcv` has 500 molecules (standard error of
   an MRR about 0.02) and a sample of each window, so its full-window figure is an extrapolation.
5. 400,000 PubChem decoys are about 0.4 % of PubChem. Each training window sees a few of its roughly 10,000
   PubChem neighbours. This may be too few; the cap was set by the disk budget.
6. The peak-preparation rule of the prepared arrays (at most 8 peaks per 50-Da window among the 128 kept) came from
   the warm-up's adapted public code, whose licence is undetermined. `cft_model.prepare_peaks` is our own
   implementation of that rule. If the rule itself must not be reused, the spectra have to be prepared again with
   a plain top-N rule and the model retrained.
7. Pool structures from COCONUT came through a public Kaggle dataset labelled CC BY 4.0; PubChem structures are
   public domain. Attribution requirements for a released solution have not been reviewed here.
8. Tautomers of held-out proxy molecules inside PubChem were excluded only by key (602 keys), not by a tautomer
   search over the 400,000 decoys.
9. Kaggle: the notebook is untested there; dataset paths are found by glob. Both datasets contain competition
   training data and must stay private.
10. A single model is trained. The public evidence for a second model (other seed or view) is mixed; `--seed 2
    --name default_s2` trains one without overwriting the first.

## 8. Commands

```
# training (Windows: run_train_cft.bat; Kaggle: research/train_cft/kaggle/casmi_cft_train.ipynb)
python train_cft.py --data <casmi-train-pkg folder> <casmi-train-cft folder> --out out_cft
python train_cft.py ... --preset small          # the 15.5 M model
python train_cft.py ... --smoke                 # CPU/GPU pipeline test
# evaluation with baselines
python eval_cft.py --data <both folders> --out out_cft [--baseline <folder with fp_single*.pt>]
# data build and staging (already done here)
python research/train_cft/prep_cft.py
python research/train_cft/make_upload_cft.py
# upload, after review (not done):
kaggle datasets create -p results/train_cft/data
```

Exit codes of `train_cft.py`: 0 finished, 2 not finished (run again), 3 GPU out of memory.
Outputs in `<out>/export`: `cft_<name>.pt` (weights + model config + adduct / instrument lists + fingerprint bit
list + validation numbers), `train_result_<name>.json`, `train_log.txt`, `metrics.jsonl`, and after evaluation
`eval_cft.json`, `check_summary_cft.txt`.

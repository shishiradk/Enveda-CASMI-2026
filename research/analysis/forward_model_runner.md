# Forward-model re-scorer (ICEBERG / GLACIER) — our own implementation

Date: 2026-10-01/02. Author: research agent. Nothing was uploaded or submitted to Kaggle; no existing project file was modified.

Labels: **[M]** measured here, **[S]** read from a source (URL / file given), **[A]** assumption or estimate.

Disk used under `external/forward_model/`: 2.6 GB (reference environment 2.1 GB, checkpoints 350 MB, upstream clone 81 MB, pure-Python deps 10 MB, MassSpecGym metadata 4 MB). The reference environment can be deleted once the equivalence check is no longer needed.

## 1. What exists now

| Path | What |
|---|---|
| `research/forward_model/ice_runner.py` | Standalone scorer (subprocess, CPU or CUDA, wall-clock budget, partial output). `--model iceberg` (default) or `--model glacier`. |
| `research/forward_model/fm_env.py` | Import bootstrap: puts our shims and the unmodified upstream `ms_pred` on `sys.path`. |
| `research/forward_model/shims/` | Our own pure-torch stand-ins: `dgl` (graph container, batch, node_subgraph, reorder_graph, update_all/apply_edges, sum_nodes, AvgPooling, pad/pack), `torch_scatter` (scatter, add, min, max, softmax), `pytorch_lightning` (inference-only `LightningModule`), `pygmtools` (placeholder). About 450 lines, written from the DGL / torch_scatter public API semantics, not from the public Kaggle packaging. |
| `research/forward_model/strip_ckpt.py` | Lightning checkpoint → `{state_dict, hyper_parameters, provenance}` (weights unchanged, loadable with `weights_only=True`). |
| `research/forward_model/build_validation.py`, `eval_validation.py`, `run_shards.py`, `merge_partial.py`, `smoke_test.py` | Validation set builder, evaluator, local CPU shard driver. |
| `external/forward_model/ms-pred/` | Upstream clone, commit `708148c2a8eb0c32120644436fefd2fe90cd2214` (2026-09-24). |
| `external/forward_model/ckpt/` | Downloaded checkpoints + stripped copies. |
| `external/forward_model/refenv/` | Python 3.11 reference environment (real DGL 2.2.1 + torch_scatter + RDKit 2025.03.6) used for the equivalence check and for validation. |
| `results/forward_model/` | Validation inputs, truth, scores, evaluation JSONs. |

We did **not** download, open or read `ahmedberatozer/casmi26-iceberg` / `casmi26-glacier`. Behaviour (top-N, same-formula groups, covered adducts, CE handling) was taken from our own prose notes in `v4n_method_spec.md`.

## 2. Sources and licences

| Item | Location | Licence | Size |
|---|---|---|---|
| ms-pred code | https://github.com/coleygroup/ms-pred (branch `main`) | MIT, "Copyright (c) 2023 Samuel Goldman" (`LICENSE` in repo; GitHub API `spdx_id: MIT`) **[S]** | clone 81 MB; `src/ms_pred` 3.4 MB |
| ICEBERG 2.1 MassSpecGym weights | Dropbox folder linked from the upstream README ("open-source MassSpecGym-trained weights"): `https://www.dropbox.com/scl/fo/kwm35ih8tlfnshfrcq8ot/AOeS4M0_v9MhqeEZys9sCRQ?rlkey=f1n6pbzx94g1k2el2wcbmee61&st=dcl9pbyp&dl=0` | No licence file inside the download **[M]**. README calls them "open-source"; repo is MIT **[S]**. | `gen/best.ckpt` 41.6 MB, `inten_contr/best.ckpt` 40.6 MB (stripped: 25.4 + 16.6 MB) |
| GLACIER MassSpecGym weights | Dropbox folder from the same README line: `https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui&st=k8km33p3&e=1&dl=0` | same situation | `best.ckpt` 181.6 MB (stripped 60.5 MB) |
| MassSpecGym (training data of both) | https://huggingface.co/datasets/roman-bushuiev/MassSpecGym | MIT (HF card `license: mit`) **[S]** | metadata extract 4 MB kept; the 261 MB TSV was deleted |

sha256 **[M]**: gen `77d78adf…43f16a`, inten `d2785ff3…bdaa7fa`, glacier `5a47cecc…d7db11` (full values in the stripped files' `provenance`).

No Zenodo or Hugging Face copy of the weights was found; the Dropbox links in the README are the only upstream location we located **[M]**. NIST'20/23 weights exist but are only sent by e-mail against a NIST licence **[S: README]** — we did not request or use them.

### Training data: MassSpecGym only?

- **[M]** The checkpoints record their own training paths: `results/dag_msg_all_iceberg/split_rnd1/version_0/best.ckpt` (gen), `results/dag_inten_msg_all_iceberg/split_1_rnd1/version_1/best.ckpt` (intensity), `results/joint_train_msg/split_rnd1/version_5/best.ckpt` (GLACIER). These match the upstream configs `configs/iceberg/msg_all/*.yaml` (`dataset-name: msg_all_iceberg`) and `configs/glacier/joint_train_msg.yaml` (`dataset-name: msg`).
- **[S]** README: `msg_all` = all MassSpecGym entries; ICEBERG is first trained on entries with a known collision energy, then used to impute missing energies, then retrained on everything. The contrastive fine-tune stage uses PubChem structures as decoys (structures only, no spectra).
- **[S]** MassSpecGym is built from MoNA, MassBank and GNPS. NIST is not part of it.
- **[A]** We cannot prove from the weights alone that no other data touched them; the evidence is the path names inside the checkpoints, the README statement and the host's explicit permission for MassSpecGym-trained ICEBERG 2.1 and GLACIER.
- Residual licence point: the weights carry no licence file of their own. Keep the README quote and the MIT notice with whatever we ship.

### What the checkpoints support

- **[M]** MassSpecGym 1.5 (231,104 spectra, 28,929 molecules by InChIKey first block): adducts `[M+H]+` (195,237 spectra) and `[M+Na]+` (35,867) only; instruments Orbitrap (172,058), QTOF (53,823), unknown (5,223); collision energy known for 121,746 spectra, mostly 10–60 eV.
- **[S]** The model has embedding slots for 14 adduct classes (incl. `[M+K]+`, `[M+NH4]+`, `[M-H]-`, `[M+Cl]-`, `[M+CHO2]-`) and 4 instrument tokens (`Orbitrap`, `QTOF`, `IT-FT`, `Unknown`), but with MassSpecGym training only `[M+H]+` / `[M+Na]+` and Orbitrap / QTOF were ever seen. Negative mode will run but is untrained → we score positive `[M+H]+` / `[M+Na]+` only.
- Collision energy is a required input (eV, continuous). Elements: the upstream `VALID_ELEMENTS` list; molecules with other elements or several fragments are skipped.
- **[M]** Test set coverage: 366 of 400 test molecules have at least one `[M+H]+` spectrum (959 of 1,213 spectra); `[M+Na]+` 22 spectra / 16 molecules. All test spectra are timsTOF with CE 20, 40, 60 or the merged 20/40/60.

## 3. Getting inference to run

- **[M]** Local main Python is 3.14 (torch 2.14 CPU, RDKit 2026.03.6): no DGL, torch_scatter, pytorch_lightning, pygmtools, einops, h5py. Python 3.13 and 3.11 (uv) are also installed.
- Chosen route: keep upstream `ms_pred` unmodified and supply pure-torch shims for the four binary / heavy dependencies. Soft placeholders are registered for cosmetic imports (`seaborn`, `cairosvg`, `multiprocess`, `pathos`, `h5py`) only when those packages are missing.
- **[M] Equivalence check.** A Python 3.11 reference environment with real DGL 2.2.1, torch_scatter 2.1.2 and pytorch_lightning 2.6 was built. 40 random universe molecules (250–650 Da) × (`[M+H]+` at 20/40/60 eV, `[M+Na]+` at 30 eV) = 160 predictions: cosine between shim and real-DGL merged spectra ≥ 0.99999999995 for all 80 spectra; largest intensity difference in the 8-molecule smoke test 4e-7. The shims reproduce upstream numerically.
- **[M] RDKit version matters a lot.** Same shims and weights under Python 3.14 / torch 2.14 / RDKit 2026.03.6 versus Python 3.11 / torch 2.3 / RDKit 2025.03.6, 40 molecules: 31 of 80 merged spectra differ (cosine < 0.999), 14 of 80 fall below 0.95, the worst is 0.17. In the 8-molecule smoke test 6 spectra agreed to 1e-7 across the two environments and 1 dropped to 0.953, which points at RDKit (kekulisation / tautomer canonicalisation giving different input graphs) rather than torch **[A: torch and RDKit were not varied separately]**. Upstream pins `rdkit==2025.03.*`. All validation below used RDKit 2025.03.6; RDKit 2026.03 was not validated and should not be used with these weights.
- GLACIER also needs `LinSATNet` (MIT, pure Python, 0.1.3) and nothing else beyond the same shims **[M]**.

Example (ICEBERG, `[M+H]+`, QTOF, 30 eV) **[M]**: aspirin → 121.028 (0.62), 77.039, 65.039, 163.039; caffeine → 195.088, 138.066, 123.043, 110.071; naringenin-like flavone → 271.060, 153.018, 119.049.

## 4. Runner behaviour (`ice_runner.py`)

Input: JSON `{molecule_id: {"smiles": [...], "adduct": ..., "spectra": [{"mzs", "intensities", "adduct", "precursor_mz", "collision_energies"}]}}`; spectra may instead come from `--spectra-parquet test.parquet` (columns `molecule_id, ms2_mzs, ms2_normalized_intensities, adduct, precursor_mz, collision_energy_ev`). Output: `{"scores": {id: [float|null]}, "cosine": {...}, "meta": {...}}`, aligned with the input candidate order.

- Covered spectrum = adduct in `--adducts` (default `[M+H]+,[M+Na]+`). A molecule without one gets all `null`.
- Per covered spectrum the candidate is predicted at each listed collision energy (rounded to 5 eV, minimum 5; `20,40,60` if unknown), instrument token `QTOF`. Per-energy predictions are scaled to max 1, summed, merged at 4 decimals, top 100 peaks kept.
- Candidate normalisation follows upstream: stereo removed, InChI round trip with upstream's tautomer canonicaliser (ICEBERG); plain RDKit canonical SMILES (GLACIER).
- Matching: one-to-one, greedy by intensity product, tolerance max(0.01 Da, 20 ppm). `scores` = unweighted spectral-entropy similarity, `cosine` = square-root-intensity cosine; both averaged over the molecule's covered spectra.
- Predictions are cached by (canonical SMILES, adduct, CE), so the 20 / 40 / 60 eV and merged spectra of one molecule cost three predictions per candidate, not six.
- Budget: `--budget-sec` is counted from process start; the loop stops `--safety-sec` early. Molecules are processed in input order, so the caller sorts by importance. An all-`null` output file is written at start, refreshed every `--save-every-sec`, and written atomically at the end.
- Degradation **[M, tested]**: invalid SMILES, salts, unsupported elements → `null` with a reason count in `meta.reasons`; a failing batch is split until the offending candidate is isolated; CUDA OOM halves the batch size; missing checkpoint → exit code 0, `meta.status = "failed"`, all `null`; budget exhausted → `meta.status = "out_of_time"`, finished molecules keep their scores.
- Not included: the fusion step (z-score mix inside same-formula groups, slot-preserving). That is a few lines in the notebook and depends on our ranker's score scale.

## 5. Validation

Set-up **[M]**:
- `train.parquet` read with DuckDB. Eligible: 167,196 molecules (InChIKey first block) with `ingest_lib = 'enveda-180'`, `[M+H]+`, positive mode; neutral mass 200–700 Da filter.
- **Main set**: 200 random molecules (seed 20261001), mass 241–433 Da (median 328). Candidates = truth + 10 most similar isomers (Morgan-2 Tanimoto, from a random 800-row subset of the isomer pool) + 10 random isomers; identical molecular formula checked with RDKit; de-duplicated on InChIKey first block; 4,196 candidates, 20.98 per molecule; decoys 3,886 from PubChem, 110 from `universe.parquet`. Mean Tanimoto to truth: similar 0.34, random 0.16.
- **Hard set**: 60 other random molecules (seed 7); truth + the 10 nearest isomers out of the whole isomer window (median pool 3,610 isomers); mean Tanimoto 0.41, nearest decoy 0.55 on average.
- Observed spectra: one per distinct CE setting (20, 40, 60, merged), up to 4 per molecule (116 molecules with 4, 72 with 3, 12 with 1–2).
- The runner never sees the truth index; candidate order is shuffled; ties get the mid-rank. Random baseline = exact expectation for a random order of the same candidate list. CI = percentile bootstrap over molecules, 5,000 resamples.
- Environment: Python 3.11, torch 2.3.0 CPU, RDKit 2025.03.6, our shims, stripped checkpoints.

### Results — rank of the true structure by forward-model score alone **[M]**

All candidates were scored (4,196 / 4,196 main, 659 / 659 hard; no failures). Score = entropy similarity unless stated.

**Main set** (200 molecules, 21 candidates each; random baseline MRR 0.174, top-1 0.048):

| Model | MRR [95% CI] | top-1 [95% CI] | top-3 | MRR − random [95% CI] |
|---|---|---|---|---|
| ICEBERG, entropy | **0.886** [0.853, 0.917] | 0.800 [0.745, 0.855] | 0.975 | +0.712 [0.679, 0.743] |
| ICEBERG, cosine | 0.890 [0.858, 0.921] | 0.810 [0.755, 0.860] | 0.970 | +0.717 [0.684, 0.748] |
| GLACIER, entropy | 0.883 [0.849, 0.917] | 0.810 [0.755, 0.865] | 0.950 | +0.709 [0.675, 0.743] |
| GLACIER, cosine | 0.878 [0.844, 0.913] | 0.800 [0.745, 0.855] | 0.955 | +0.705 [0.670, 0.739] |
| z(ICEBERG) + z(GLACIER) | 0.896 [0.866, 0.926] | 0.815 [0.760, 0.870] | 0.975 | +0.722 [0.692, 0.752] |

Sub-rankings inside the main set, ICEBERG entropy: truth vs the 10 similar decoys only — MRR 0.887 [0.854, 0.918], top-1 0.800 (random 0.275 / 0.091); truth vs the 10 random decoys only — MRR 0.995 [0.988, 1.000], top-1 0.990. Random isomers are almost never confused with the truth; all the errors come from the similar ones.

**Hard set** (60 molecules, truth + 10 nearest isomers; random baseline MRR 0.275, top-1 0.091):

| Model | MRR [95% CI] | top-1 [95% CI] | top-3 | MRR − random [95% CI] |
|---|---|---|---|---|
| ICEBERG, entropy | **0.750** [0.668, 0.827] | 0.600 [0.483, 0.717] | 0.867 | +0.475 [0.393, 0.552] |
| ICEBERG, cosine | 0.752 [0.670, 0.829] | 0.600 [0.467, 0.717] | 0.867 | +0.477 [0.396, 0.554] |
| GLACIER, entropy | 0.804 [0.725, 0.876] | 0.683 [0.567, 0.800] | 0.900 | +0.529 [0.450, 0.602] |
| GLACIER, cosine | 0.810 [0.732, 0.881] | 0.683 [0.567, 0.800] | 0.917 | +0.536 [0.457, 0.606] |
| z(ICEBERG) + z(GLACIER) | 0.785 [0.706, 0.859] | 0.650 [0.533, 0.767] | 0.883 | +0.510 [0.432, 0.584] |

Reading: both models carry a large, clearly non-random signal on unseen timsTOF molecules. The signal shrinks as decoys get closer to the truth (MRR 0.886 → 0.750 for ICEBERG). Differences between ICEBERG and GLACIER, and between entropy and cosine, are inside the confidence intervals; the z-sum is not better than the best single model on the hard set.


### Leak check

- **[M]** MassSpecGym 1.5 contains 28,929 molecules (InChIKey first block). Of the 167,196 enveda-180 `[M+H]+` molecules only **21** are in MassSpecGym (0.013%). None of the 200 + 60 validation truths is in MassSpecGym (any fold). The numbers above are therefore leak-free with respect to the forward models' training molecules.
- For comparison, 28,697 of the 95,157 molecules of the *other* train libraries are in MassSpecGym: a validation built from GNPS / MassBank / MoNA rows would have been heavily leaked.
- **[A]** The hidden test set is timsTOF like enveda-180, so the same low overlap probably holds there, but we cannot measure it.

### What the validation does not show

- Decoys are PubChem isomers picked by fingerprint similarity. The real task is to re-order candidates our ranker already scored close together, which are harder than these decoys. Publicly reported gain of this step on the leaderboard was about +0.008 (ICEBERG alone) to +0.02 (ICEBERG + GLACIER, weight 1.0) **[S: competition_intel.md]** — far smaller than the stand-alone numbers suggest.
- enveda-180 chemistry (241–433 Da here) may differ from the hidden test molecules.
- No fusion with our ranker was tested; λ and the top-N are untuned for our score scale.

## 6. Timing

CPU, this machine (8 logical cores, torch 2.3.0 CPU, molecules of 241–433 Da, batch 16) **[M]**:

| Run | Predictions | Speed |
|---|---|---|
| ICEBERG, one process, 8 threads, nothing else running (3 molecules) | 189 | 1.00 prediction/s → **3.0 s per candidate** (three collision energies) |
| GLACIER, same | 189 | 3.12 predictions/s → **0.96 s per candidate** |
| ICEBERG, main set, 4 processes × 2 threads | 12,399 | 5,812 s wall → 2.13 predictions/s in total, 1.39 s wall per candidate |
| GLACIER, main set, 4 processes × 2 threads | 12,399 | 1,216 s wall → 10.2 predictions/s in total, 0.29 s wall per candidate |

Model load + imports: 16–30 s per process. Our shims were slightly faster than real DGL on CPU (ICEBERG 0.75 vs 0.63 predictions/s, GLACIER 3.4 vs 2.7, same machine, 160 predictions).

Kaggle estimate for 400 molecules × 60 candidates:
- Worst case = 366 covered molecules × 60 candidates × 3 energies ≈ 66,000 predictions per model. In practice only candidates that share a formula with another top-60 candidate are sent, so the real number is lower **[A]**.
- **[S]** The public notebook reports 25–32 ICEBERG predictions/s on a T4 and 25–30 min for 400 × 60. At that rate the worst case is 35–45 min for ICEBERG. **[A]** We expect the same order from our runner (same upstream model code, tensor-only shims) but have **not measured any GPU number**.
- **[A]** GLACIER: 3.1× faster than ICEBERG on CPU here **[M]** and about 8× faster on GPU in the GLACIER paper **[S]**; its featurisation is CPU-bound, so expect roughly 10–20 min on a T4 notebook.
- CPU-only Kaggle (4 vCPU) is not viable for ICEBERG at this size (66,000 predictions at ≤ 1/s ≈ 18 h or more). GLACIER on CPU would need about 6 h for the worst case; it only fits with a smaller top-N.


## 7. Kaggle packaging plan (not executed)

One offline dataset, all MIT / BSD:

| File | Licence | Size | Note |
|---|---|---|---|
| `ms_pred/` (upstream `src/ms_pred`, unmodified, with upstream `LICENSE`) | MIT | 3.4 MB | only `common`, `iceberg`, `glacier`, `graphormer`, `magma`, `nn_utils` are imported |
| `shims/`, `fm_env.py`, `ice_runner.py` | ours (MIT) | < 100 kB | |
| `iceberg_gen.pt`, `iceberg_inten.pt` (stripped) | upstream weights, MIT repo | 25.4 + 16.6 MB | provenance block inside |
| `glacier.pt` (stripped) | same | 60.5 MB | optional |
| `rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl` | BSD-3-Clause | 36.1 MB | installed with `pip install --no-index --no-deps --target /kaggle/working/rdkit2025` and passed as `--prepend-path`; keeps the notebook's own RDKit untouched |
| `LinSATNet-0.1.3-py3-none-any.whl` | MIT | 12 kB | GLACIER only |
| `einops-0.8.2-py3-none-any.whl` | MIT | small | safety copy; normally preinstalled |

No DGL, torch_scatter, torch_sparse, pytorch_lightning, pygmtools, ray, pathos or h5py wheel is needed: the shims and soft placeholders cover them, and they depend only on `torch`, `numpy`, `pandas`, `tqdm`, `platformdirs`, `packaging`, `matplotlib` **[A: all preinstalled in the Kaggle image — to be confirmed in a first dry run]**.

Open items before it runs on Kaggle:
1. Dry-run on a T4 notebook (internet off): import check, 100 predictions, measure predictions per second. Our shims are plain tensor ops and were only executed on CPU **[M]**; CUDA correctness is expected but **untested**.
2. Kaggle's torch build **[A: newer than 2.3]**: the shims ran here on torch 2.3.0 and 2.14.0 CPU **[M]**.
3. Write the fusion step in the notebook: z(ranker) + λ·z(forward) inside same-formula groups of the top-N, slot-preserving, touching a group only when ≥ 2 members were scored; tune λ and N on a hold-out that uses our own ranker scores.
4. Order molecules by ascending library-match confidence so the budget is spent where it matters.
5. Ship the MIT notice, the README quote about the weights and the checkpoint sha256 with the dataset.

## 8. GLACIER

- **Feasible and working [M].** `ice_runner.py --model glacier --glacier-ckpt glacier.pt` runs the upstream `ms_pred.glacier.joint_model.JointModel.predict_mol` with the same shims; the only extra dependency is `LinSATNet` (MIT, pure Python).
- **[M]** Shim vs real DGL: 80 merged spectra (160 predictions), minimum cosine 1.0000.
- Source, licence and training data: see section 2 (same README line, same Dropbox hosting, `dataset-name: msg`, checkpoint path `results/joint_train_msg/split_rnd1/version_5`). The path inside the published checkpoint is the joint-training run, not the contrastive fine-tune that upstream also has a config for **[M]**.
- Same coverage as ICEBERG: `[M+H]+` / `[M+Na]+`, Orbitrap / QTOF tokens, collision energy required.
- Validation and timing: tables above. On the hard set GLACIER is at least as good as ICEBERG (MRR 0.804 vs 0.750, overlapping intervals) at about one third of the CPU cost.
- GLACIER candidate normalisation in the runner follows the upstream prediction script (RDKit canonical SMILES, no InChI round trip). RDKit-version sensitivity was measured for ICEBERG only; pin RDKit 2025.03.x for both.


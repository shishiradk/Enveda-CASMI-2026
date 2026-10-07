# v4n method specification (clean-room description for an independent rebuild)

Compiled 2026-10-01. Research only: no project code changed, nothing pushed or submitted to Kaggle.

**Purpose.** Describe how Ahmed Berat Özer's "v4n" stack works, in our own words, so that we can write an independent implementation and train our own models on `train.parquet`. No code is copied here; only short identifiers and constants are quoted.

**Labels.** FACT = read directly in the named file. INFERENCE = my reading, estimate or reconstruction. "LB" = public leaderboard as stated by a notebook author. Public-LB noise is about ±0.006–0.02 (FACT, `competition_intel.md` §2.1), so single-step deltas below that are not reliable.

**Sources read** (all downloads in the session scratch directory, about 1 MB in total):
- Fork notebook `shishiradhikari11/casmi26-v4n-engine-fusion-union-lb-0-399` (17 cells; "cell N" below refers to it).
- Ahmed's notebooks `casmi26-v2-baseline`, `casmi26-v3-inference`, `casmi26-v4n-inference`, `casmi26-v4q-inference` (he has no training notebook: FACT, `kaggle kernels list --user ahmedberatozer`).
- `ahmedberatozer/casmi26-v4b-models`: `MANIFEST.json`, `LICENSE-NOTICE.txt`, all of `code/casmi/*.py` and `code/fe_v4/**/*.py`, `res/frag2/caps.txt`.
- `casmi26-iceberg` and `casmi26-glacier`: README, MANIFEST, `fuse.py`, `gl_fuse.py`, `ice_runner.py` (header).
- `casmi26-v2-pool/fp_bits.npy` (41 KB) and the file listings of all seven datasets.
- `research/analysis/competition_intel.md` and `fork_licence_rebuild_plan.md`.

**Not inspected.**
- No weight file was opened (`fpnet_*.pt`, `fe_*.pt`, `ranker_*.pkl`, fragnet `stageB.txt`). A download of the two ranker pickles and of `pool_meta.parquet` / `train_structs.parquet` was blocked by the session's permission system, so the ranker's stored parameters and the pool's source counts are taken from code defaults, the MANIFEST and file sizes, and are marked as such.
- The multi-GB pool and PubChem arrays were not downloaded (by design).

---

## 0. One-page summary

- v4n is a **retrieve-and-rerank** system. For each molecule it takes every structure within ±10 ppm of the neutral mass from a fixed pool (training structures ∪ COCONUT, 710,701 entries), adds up to 150 generated derivatives of the best spectral analogs, computes about 160 features per candidate, and sorts by a LightGBM LambdaRank score.
- The features come from five evidence channels: library match (L), analog propagation (A), a spectrum→fingerprint transformer "FPNet" (M), in-silico fragmentation (F) and sub-formula explanation (S), plus generation flags (G).
- Three post-steps follow: (1) ICEBERG and GLACIER forward models re-order candidates **inside same-formula groups only**; (2) a separate PubChem-only list is interleaved at fixed slots when the library match is weak; (3) in the 0.399 fork, the result is rank-fused with a second engine's list and re-ordered once more by the forward models.
- The LB history says the value is in the base engine (0.354), the forward models (+0.022) and the fusion (+0.015). The elaborate `fe_v4` feature families and the DreaMS-based model did not improve the LB (0.358 → 0.354).

---

## 1. End-to-end data flow at inference

### 1.1 Query construction (same for every molecule)

| Step | What happens | Label and source |
|---|---|---|
| Neutral mass | Per spectrum `M = (mz·|z| − δ_adduct) / n_M`, with the electron mass included in δ. The molecule's target mass is the **median** over its spectra. | FACT: `chem.neutral_mass`; cell 11 |
| Spectrum record | Each spectrum becomes `(mz, intensity, mode ±1, adduct, precursor m/z, mean CE, ce_n)`. `ce_n` is the number of values in the row's `collision_energy_ev` list (1 if empty). | FACT: cell 11 |
| Cap | Only the first 16 spectra of a molecule are used. | FACT: `EngineCfg.max_query_spectra`, `engine.py` line 243 |
| Adduct table | 59 adducts, including dimers and trimers; the ten test adducts come first in the index, and an `<unk>` entry is last. | FACT: `chem.ADDUCTS`, `TEST_ADDUCTS` |

### 1.2 Candidate generation

| Source | Rule | Label and source |
|---|---|---|
| Pool window | All pool entries with mass within ±10 ppm of the target; if that is empty, ±30 ppm. **No cap** on the number of candidates. | FACT: `Engine.candidates`, `engine.py` lines 210–214 |
| Library hits | Not a separate source. A training structure is already in the pool; library evidence only adds features to that pool row (via `train_sid`). | FACT: `engine.py` lines 264–305 |
| Analog-derived | Only when `generate=True`: derivatives of the top analogs whose mass matches the target (§4). They get `pid = −1`. | FACT: `Engine.generate` |
| PubChem | Never enters the candidate set. It is a separate list merged afterwards (§5). | FACT: cells 8 and 14 |

### 1.3 Scoring and the base list

1. The engine builds a 94-column feature matrix (§2.4); the `fe_v4` wrapper appends 66 more columns.
2. Score = mean of 4 LightGBM boosters. Candidates are sorted by score, de-duplicated on the metric key (tautomer-canonical InChIKey14), and the **top 60** are kept (`TOPN = 60`) with their scores and formulas. (FACT: cell 11.)
3. The engine also reports `lib_max` = the best library entropy similarity of any candidate. The gate in step 6 uses it. (FACT: `engine.py` line 463.)

### 1.4 What happens per class of molecule

The system never predicts the class. The same pipeline runs for all molecules; the classes differ only in which evidence is non-zero. (INFERENCE from the code structure; the class shares are hidden.)

| Class | Where the true structure can come from | Evidence that carries it | Post-steps that apply |
|---|---|---|---|
| 1: has library spectra | Pool row with `train_sid ≥ 0` | `lib_*` features (direct entropy match, same polarity, ±10 ppm on the labelled structure's mass). The analog channel also fires with shift 0. | If `lib_max ≥ 0.9` the PubChem merge is skipped. ICE/GLACIER still run, but these molecules are processed last. |
| 2: known structure, no spectra | Pool row from COCONUT (or a training structure whose spectra do not match), or the PubChem-only list | FPNet `f·z`, analog propagation, fragmentation, sub-formula | Forward-model re-order inside formula groups; PubChem slots when `lib_max < 0.9` |
| 3: not in PubChem | Only a generated derivative (`is_gen = 1`), if one of ≤ 6 analogs is one or two listed transformations away | Same features plus `gen_*` flags; the ranker decides how high generated rows go | Same as class 2. PubChem slots cannot help and cost positions. |

### 1.5 Order of operations in the notebook

| # | Step | Detail | Source |
|---|---|---|---|
| 1 | Second engine (other authors) | Subprocess; produces a ranked list per molecule (`ENG`). Not covered here. | cell 5 |
| 2 | PubChem-only channel | Subprocess with FPNet A+B; writes per molecule: up to 25 SMILES, their `f·z`, their keys, `best_pool_fz`, target. Timeout 4 h. | cell 8 |
| 3 | Spectrum cache | Cleans all of `train.parquet` at run time (drop peaks above precursor + 2 Da, floor 0.1% of base peak, top 512, base peak = 1). | cell 9; `build.build_spec_cache` |
| 4 | Base lists | Engine + ranker → top 60 per molecule. | cells 10–11 |
| 5 | ICEBERG, then GLACIER | Input: candidates that share a formula with at least one other candidate within the top 60, plus the second engine's top 40 not already there, plus the PubChem list when `lib_max < 0.9`. Molecules are sorted by ascending `lib_max`, so weak-library molecules are scored first if the time budget runs out. Budgets 5400 s and 4000 s in the fork (2700 / 2400 s in Ahmed's own v4n). | cell 13; `fuse.ice_candidates` |
| 6 | Re-rank base list | Inside each formula group among the top 60: sort by `z(ranker) + 1.0·z(ICE) + 1.0·z(GL)`, put members back into the group's own positions. Keep top 25. | cell 13; `fuse.rerank`, `gl_fuse.rerank_multi` |
| 7 | Re-rank PubChem list | Same rule on the 25 PubChem candidates with `f·z` as base score. The gate keeps using the **original** top `f·z`. | cell 13 |
| 8 | Gated merge | See §5. Output is the v4n submission. | cell 14 |
| 9 | Fusion (fork only) | Weighted reciprocal-rank fusion on the metric key: `1/(3 + r_v4n) + 0.6/(3 + r_engine)`; keep 40; then the step-6 rule again over all 40 with the fused score as base; keep 25. | cell 16 |

**Forward-model re-rank rule in detail** (FACT: `fuse.py`, `gl_fuse.py`):
- `z` is computed within the formula group with the sample standard deviation (ddof = 1). A missing score, a group with fewer than 2 scored members, or zero spread gives z = 0.
- A group is touched only if at least one forward model scored ≥ 2 of its members.
- The permutation is **slot-preserving**: a candidate can only swap places with its own isomers. Candidates of other formulas never move.
- With two forward models, fused values are rounded to a 1e-9 grid so that exact ties keep the current order.

**Forward-model score** (FACT: `ice_runner.py` header, iceberg/glacier READMEs):
- Only positive-mode `[M+H]+` and `[M+Na]+` spectra are "covered". A molecule with no covered spectrum gets no forward score and keeps the ranker order.
- For each covered spectrum the model predicts the candidate's spectrum at that adduct, an instrument token (`Orbitrap` or `QTOF`), and the CE rounded to 5 eV (minimum 5). If CE is unknown, predictions at 20, 40 and 60 eV are merged.
- Prediction post-processing: drop zero intensities, merge identical m/z (4 decimals; GLACIER by max), keep top 100, scale to max 1.
- Score = mean over covered spectra of the entropy similarity between query and prediction (0.01 Da / 20 ppm, same kernel as the library channel).
- Models: ICEBERG 2.1 `msg_all` and GLACIER, both MassSpecGym checkpoints from ms-pred (MIT), pinned to RDKit 2025.03.6 in a separate process because RDKit 2026.03 kekulises differently and changes predictions.
- Stated cost: about 25–32 predictions/s on a T4; about 25–30 min for 400 molecules at 60 candidates.

---

## 2. Learned models

### 2.1 Spectrum featurisation shared by all FP models

| Item | Value | Source |
|---|---|---|
| Representation | **A set of peak tokens. No m/z binning.** | FACT: `fpnet.FPNet.forward` |
| Peak filter | Keep `0 < mz ≤ precursor + 2 Da`; scale to base peak 1; drop below 0.001; keep the 160 most intense; sort by m/z | FACT: `fpnet.prep_peaks` |
| Intensity transform | Square root of relative intensity; the token receives `[√I, I]` through a linear layer | FACT |
| Neutral loss | `max(precursor_mz − mz, 0)` per peak, embedded separately. It is the loss from the **precursor m/z**, not from the neutral mass. | FACT |
| m/z and loss embedding | Sinusoidal, `d/2` wavelengths log-spaced from 10⁻² to 10^3.3; sin and cos concatenated | FACT: `SinEmb` |
| Peak token | Linear(3d → d) of [m/z embedding, loss embedding, intensity embedding] | FACT |
| Global token (first position) | Linear over [precursor embedding (d), CE embedding (d/4, wavelengths 1 to 10^2.2 on `CE/10 + 1`), CE-known flag, `n_merged/4`, mode ±1, `log1p(precursor)/10`] plus a learned adduct embedding | FACT |
| CE unknown | CE = 0 and flag = 0 | FACT: `engine.py` lines 342–345 |
| `n_merged` | `min(ce_n, 8)` for one spectrum; for a merged view the sum of `max(1, ce_n)` over the merged spectra, capped at 8 | FACT |

**Multi-spectrum handling** (FACT: `engine.py` lines 336–366, `spectra.merge_spectra`):
- *Single view:* every spectrum is run on its own; logits are averaged over spectra.
- *Merged view:* one merged spectrum per polarity. Each spectrum is scaled to base peak 1; all peaks are pooled and sorted; peaks closer than 0.005 Da are collapsed keeping the more intense one. Precursor = median, CE = mean of known CEs, adduct = the most frequent one.
- *Final logits:* `z = 0.5·(mean single logits + mean merged logits)`. The element head is averaged the same way.

### 2.2 FPNet (models A, B and full1)

| Item | Value | Source |
|---|---|---|
| Purpose | Map spectra to logits `z` over fingerprint bits so that the dot product `f·z` with a candidate's 0/1 fingerprint ranks candidates | FACT: module docstring |
| Encoder | 8 pre-LayerNorm transformer blocks, width 512, 8 heads, feed-forward 2048 with GELU, dropout 0.1, padding mask on keys | FACT: `FPNet.__init__` defaults; the checkpoint stores `d`, `layers`, `heads` |
| Pooling | Concatenate the global-token state and the masked mean of the peak states (1024 values) | FACT |
| Fingerprint head | Linear(1024 → 2048), GELU, dropout, Linear(2048 → 10,226) | FACT; 10,226 from `fp_bits.npy` |
| Element head | Linear(1024 → 256), GELU, Linear(256 → 10); target is `log1p` of the counts of C, H, N, O, P, S, F, Cl, Br, I | FACT: `traindata.Data.elem` |
| Size | About 50 M parameters; checkpoint 199 MB in fp32 | INFERENCE from the layer sizes; FACT for the file size |
| Precision | fp16 autocast on T4, bf16 on newer GPUs | FACT: `autocast_dtype` |

**Output fingerprint** (FACT: `chem.FP_BLOCKS`, `structs.select_bits`, `fp_bits.npy`):
- Raw layout, 16,551 bits: ECFP4 (Morgan radius 2, 4096) ‖ ECFP6 (radius 3, 4096) ‖ FCFP4 (feature invariants, radius 2, 2048) ‖ RDKit path fingerprint (max path 6, 2048) ‖ atom pair (2048) ‖ topological torsion (2048) ‖ MACCS (167).
- Kept bits: those whose frequency over a 200,000-structure sample lies in [0.005, 0.995] → **10,226 bits**.
- Kept per block (computed from `fp_bits.npy`): ECFP4 1464, ECFP6 3278, FCFP4 792, RDKit 2048, atom pair 1678, torsion 827, MACCS 139.
- Structures are standardised first: largest fragment, uncharged, stereo removed.

**Use of the output in scoring** (FACT: `engine.py` lines 364–393 unless noted):
- Main score `f·z` (the module docstring calls it the Bayes log-likelihood up to a candidate-independent constant).
- Derived columns: z-score, rank, gap to best, value divided by √(bits set), a softmax share with temperature `√nbits / 4`, the group entropy of that softmax, the top-2 gap; the same for the single-only and merged-only logits.
- From `p = sigmoid(z)`: full Bernoulli log-likelihood, cosine with `p`, Tanimoto with the thresholded prediction.
- Element head: absolute error between predicted and candidate `log1p` element counts (C, N, O, sum, rank).
- `f·z` also selects which candidates get fragmentation features when there are more than 600 (line 421), and drives the PubChem channel and its gate (§5).

**Training objective** (partly FACT, partly INFERENCE):
- FACT: the docstring says "mass-window contrastive ranking loss … hard negatives drawn from the candidate pool within ±10 ppm of the true mass".
- FACT (`traindata.Data`): each example has 1 positive + **63 negatives** sampled from the pool's ±10 ppm window around the true mass. If the window has ≤ 1 entry, all negatives are random pool entries; if it has fewer than 8, half are random.
- FACT: half of a batch is sampled uniformly over spectra, half uniformly over (structure, polarity) groups.
- FACT: with probability `merge_p` an example is a merge of 2–4 spectra of the same structure and polarity.
- FACT: augmentation = random peak dropout (rate drawn from 0–0.3, at least 3 peaks kept), log-normal intensity jitter (σ 0.3), m/z jitter (σ 4 ppm), CE flag dropped with probability 0.1.
- FACT: the default hold-out folds are `np`, `nplib`, `syn`, `plusk`, `twin`; only structures present in the pool are used.
- FACT: validation metric = MRR@25 of the truth inside its ±10 ppm pool window by `f·z`, per spectrum or per merged group (`window_mrr`).
- INFERENCE: loss = softmax cross-entropy over the 64 `f·z` scores, plus a regression loss on the element head, possibly plus a bit-wise BCE term. The loop, the loss weights, batch size, learning rate, schedule and step count are not public.

**The three FPNet checkpoints.**

| Name | Where used | What is known |
|---|---|---|
| A (`fpnet_0.pt`, `fe_A.pt`) | Original engine bank (v3.6, v4b–v4l); PubChem channel; `fe_v4` model views | FACT (MANIFEST): source `work/fpnet_a.pt`. FACT (intel §2.3): 0.934 MRR on training structures vs 0.581 on the unseen np-examples panel → it was trained with the hold-out folds excluded. |
| B (`fpnet_1.pt`, `fe_B.pt`) | PubChem channel (averaged with A); `fe_v4` model views | FACT: source `work/fpnet_b.pt`, same class and size. How B differs from A (seed, `merge_p`, steps) is **not determinable**. |
| full1 (`fpnet_full1.pt`) | Engine bank in v4m/v4n; falls back to A if absent | FACT (dataset description): "trained on ALL train folds incl. np-examples, same architecture as … FPNet A (drop-in)". FACT (intel): 0.928 on the np panel because it has seen it. |

Note (INFERENCE): in v4n the ranker was trained on features produced with model A (hold-out-clean), but at inference it receives full1's features. full1 is sharper on structures it was trained on, which include every pool entry that is a training structure. The author's own LB numbers do not show a clear gain (v4m 0.378 vs v4l 0.380; v4n 0.384).

### 2.3 DreamsFP (model D, `fe_D.pt`)

- FACT (`dreams_model.py`): a PyTorch port of the DreaMS backbone (7 layers, width 1024, 8 heads, Fourier m/z features, Graphormer-style bias) with the same fingerprint and element heads as FPNet. Input: precursor token plus the top 100 peaks by intensity, linear relative intensity; negative-mode m/z shifted by two proton masses. Conditioning (adduct, polarity, CE, merged count) is zero-initialised and added to the precursor token.
- FACT (MANIFEST): fine-tuned checkpoint `pilot_d1_step20k_ema` (20k steps, EMA weights). 486 MB.
- FACT (`model_views.py`): used **only** as feature columns: `fzD_gap`, `fzD_rank`, `fzD_z`, the A+B+D ensemble gap and rank, and the spread / worst rank across A, B, D.
- Evidence of value: none positive. v4d (DreamsFP as engine bank) was reverted by its author without a number (intel §2.2); DreaMS as an analog channel ties or loses in two independent CV reports (intel §2.3).

### 2.4 The LightGBM ranker

| Item | Value | Source |
|---|---|---|
| Objective | `lambdarank`, binary relevance (`label_gain = [0, 1]`), truncation level 30 | FACT: `ranker.DEFAULT_PARAMS` |
| Defaults in code | learning rate 0.03, 63 leaves, min 40 rows per leaf, feature fraction 0.7, bagging 0.8 every round, L2 = 1 | FACT (code defaults; the values stored in the pickle were not inspected) |
| Bagging | 4 boosters with seeds 0–3, predictions averaged | FACT: `RankerBag.fit`; MANIFEST `n_boosters: 4` |
| Rounds | 500 for `ranker_v4b` (code default 600) | FACT: MANIFEST |
| Features | 160 = 94 engine + 66 `fe_v4` | FACT: MANIFEST |
| Group | One simulated query = one group; the label marks the true structure | FACT: `ranker.mrr_at`, `cv_report` |
| CV | 5 folds grouped by structure id; MRR reported per `regime` and `fold` | FACT: `cv_report` |
| Row weights | Supported (`w`); the values used are not public | FACT / unknown |

**The 94 engine features by family** (FACT: `engine.FEATURES` and `Engine.run`).

| Family | Count | Columns (meaning) |
|---|---|---|
| L library | 10 | best entropy similarity over query spectra and reference spectra; mean and top-3 mean of per-query best; log count of matching reference spectra; best cosine; rank; gap to query best; query best; hit flag; best similarity restricted to the same adduct |
| A analog | 16 | `ap` = max over analogs of Tanimoto × sim⁴; `a1` = same with sim¹; best Tanimoto; Tanimoto to the top analog; sim⁴-weighted mean Tanimoto; log sum; rank, gap, query max; top analog similarity; the same four restricted to natural-product libraries; number of analogs with sim > 0.5; best Tanimoto × sim among analogs with zero mass shift |
| M model | 17 | `f·z` and its z, rank, gap, normalised value, top flag, softmax share, group entropy, top-2 gap; single-only and merged-only `f·z` with gaps; cosine, Tanimoto, log-likelihood and its gap |
| Element | 5 | abs error for C, N, O; summed error; rank |
| F fragmentation | 9 | max and mean explained intensity share, explained count share, share of the top 10 peaks, strict (±1 H) share, rank, gap, z, log number of fragments |
| S sub-formula | 8 | explained intensity, count, top-10 share, rank, gap; share of candidates with the same formula; number of distinct formulas; flag "formula explains the most" |
| Priors / context | 11 | mass error in ppm and its rank; heavy atoms; bits set; log candidates; number of query spectra, positive, negative; target mass; log mean peak count |
| Interactions | 5 | `lib_max × (1 − f·z rank)`; `ap × (1 − f·z rank)`; `f·z` rank of the library-best and of the analog-best candidate; `fr_int × (1 − f·z rank)` |
| G generation | 6 | generated flag; parent similarity; steps (always 1 in the shipped code); Tanimoto to parent; number generated in the query; `f·z` rank among generated |
| X cross-encoder | 8 | `xs_*`. **All zero in v4n**: no cross-encoder is passed to the engine (FACT: cell 10). `xenc.py` is shipped but unused; intel §2.3 records that prvsiyan's cross-encoder attempts all failed. |

Several columns are query-level constants (`lib_grp_max`, `ap_grp_max`, `top_sim`, `n_cand`, `n_query`, `target_mass`, …). They cannot rank within a query by themselves; in tree models they let the ranker switch behaviour by regime, for example "trust the library when `lib_grp_max` is high". (INFERENCE.)

**The 66 `fe_v4` features** (FACT: MANIFEST `family_features`; meanings from the module docstrings).

| Family | Count | What it measures |
|---|---|---|
| `derivation` | 19 | For generated candidates and for pool candidates that the generator also produced: number of transformation combos, minimum steps, a "found-rate" prior of the transform and of the attachment site (tables built from COCONUT parents), rank among sibling positional isomers, parent similarity split by zero-mass-delta (isomerisation) vs real derivation, ranks of these |
| `frag2` | 13 | Discriminative fragmentation: explained intensity weighted by how few candidates explain each peak, depth-1 vs depth-2 mix, coverage of the candidate's own one-bond cleavages, comparisons against the `f·z` leader, support for diagnostic neutral losses given 22 substructure capabilities |
| `analog_struct` | 13 | Relations between candidate and top-15 analogs on the element graph (ring-system Jaccard, scaffold equality, substructure containment both ways, formula difference being an exact 0–2-step transform delta) |
| `model_views` | 7 | Agreement of models A, B and D (§2.3) |
| `fragnet` | 14 | A count-based log-likelihood-ratio model of which fragment classes are observed for true structures vs same-formula decoys, followed by a small LightGBM (31 inputs, 15 leaves, 300 rounds); output as score, ranks and gaps |

Evidence for the families:
- Offline (FACT, decision logs inside `fams/*.py`): each family gives +0.003 to +0.008 overall MRR on the author's simulation, with larger gains on class-3-type queries (+0.02 to +0.10) and confidence intervals that often include zero for class 2.
- LB (FACT, intel §2.2): v3.6 → v4b was 0.358 → 0.354.

### 2.5 Non-learned channels in detail

**L, library match** (FACT: `library.SearchCfg`, `spectra.py`, `engine.py` lines 261–305).
- Query peaks: floor 0.2% of base peak, top 256, **no precursor cut**. Reference peaks come from the cleaned cache and get the same preparation.
- Weights: `p = I / ΣI` (power 1). If the spectral entropy `S < 3`, `p ← p^(0.25 + 0.25·S)`, renormalised.
- Similarity: `1 − (2·S_AB − S_A − S_B) / ln 4`, peaks matched within `max(0.01 Da, 20 ppm)`.
- Reference window: reference spectra whose **labelled structure mass** is within ±10 ppm of the target, same polarity. The reference precursor m/z and adduct are not used for the window, so a reference with a different adduct still matches.
- Each reference score is credited to the candidate that is the same training structure.

**A, analog propagation** (FACT: `Engine._reps`, `Engine.analogs`, `spectra.hybrid_sim`).
- Reference set: one representative spectrum per (structure, polarity, library): the one with the most cleaned peaks.
- Search window: structure mass within ±200 Da of the target; same polarity.
- Query: the merged spectrum per polarity.
- Similarity: hybrid entropy similarity. A reference peak may match a query peak directly or after adding `target − reference mass`; each query peak is used once; direct matches are assigned first.
- Keep the best score per structure; the top 100 structures are the analogs.
- Propagation: Tanimoto between each candidate and each analog on the 10,226 selected bits, weighted by `sim⁴`.
- A zero-shift analog is a same-mass structure, so a class-1 truth also shows up here.

**F, fragmentation** (FACT: `frag.py`).
- Fragments: masses of all connected components after removing any 1 or 2 bonds of the heavy-atom graph (ring bonds included), implicit hydrogens included, rounded to 0.001 Da, de-duplicated; the parent mass is included. Precomputed for the pool; computed on the fly for generated candidates.
- Peak explained if `|peak − (fragment + ion offset + s·H)| ≤ max(0.01 Da, 15 ppm)` for an integer `s` in −2…2 (strict variant −1…1).
- Ion offsets: +proton in positive mode, plus the metal or ammonium cation if the adduct contains Na, K or NH4; −proton in negative mode, plus chloride or formate variants.
- Peak weight = √(relative intensity). Query peaks: floor 0.2%, top 128.
- Five numbers per candidate: max and mean over spectra of the explained weight share, max explained count share, max share of the 10 strongest peaks, max strict share.
- If there are more than 600 candidates, only the top 600 by `f·z` are evaluated; the rest get zeros.

**S, sub-formula** (FACT: `frag.subformula_masses`, `subformula_features`).
- Enumerate all sub-formulas of the candidate's formula over C, H, N, O, P, S and halogens with `H ≤ 2C + N + 2P + 2` and a loose ring-double-bond bound; match peaks at `max(0.005 Da, 10 ppm)` with no hydrogen shift.
- Computed once per distinct formula in the window, so it separates formulas, not isomers.

---

## 3. The candidate pool

| Property | Value | Label and source |
|---|---|---|
| Sources | Structures of `train.parquet` ∪ COCONUT | FACT: `pool.py` docstring; dataset description |
| Size | 710,701 structures | FACT: fragnet log in `fams/fragnet.py`; confirmed by file sizes (`pool_fp.npy` = 128 + 710,701 × 1279 bytes) |
| Training structures | 275,810 | INFERENCE from `train_fp_sel.npy` size (128 + 275,810 × 1279 bytes) |
| Overlap | 25,746 keys occur in both sources; for 2,576 of them the two SMILES strings differ (tautomer style) | FACT: decision logs in `analog_struct.py`, `fragnet.py` |
| COCONUT entries | About 460,600 unique keys (710,701 − 275,810 + 25,746) | INFERENCE; prvsiyan reports 462,028 for COCONUT 2.0 (intel §1.2) |
| Columns | mass, key (metric key), smiles, formula, heavy atoms, `src`, `train_sid` | FACT: `pool.Pool` |
| Standardisation | Largest fragment, uncharged (structures that stay charged are dropped), stereo removed, canonical SMILES | FACT: `chem.standardize_smiles`; pool SMILES are confirmed to be its output in the `derivation.py` log |
| De-duplication key | Tautomer-canonical InChIKey14, computed exactly as the metric does (RDKit 2026.03.3) | FACT: `chem.score_key` |
| Mass index | Rows sorted by exact mass; a window is two binary searches | FACT: `Pool.window`; the PubChem runner asserts the sort |
| Window | ±10 ppm of the target mass, fallback ±30 ppm when empty | FACT |
| Fingerprints | Bit-packed, 1279 bytes per structure (909 MB) | FACT |
| Fragments | CSR arrays: offsets plus about 89 M float32 masses (about 126 per structure on average) | FACT for the files; INFERENCE for the count (357 MB / 4 bytes) |
| Training-structure fingerprints | Separate packed array indexed by structure id, used for analogs and generation parents | FACT: `Engine.__init__` |
| Element filter, mass range | Not determinable without the table. Sub-formula and element features cover only C, H, N, O, P, S, F, Cl, Br, I; other elements give zero features. | unknown / FACT |

Note (INFERENCE): which training SMILES represents a key shared by both sources is not visible. The author's logs show he treated "train-style vs COCONUT-style SMILES" as a leak risk (a ranker could learn provenance from fingerprint style) and designed the `fe_v4` families to avoid it. For our rebuild the simplest safe choice is to canonicalise every pool structure through one tautomer-canonical form before fingerprinting.

---

## 4. Class-3 generation ("moderate generation", `EngineCfg(generate=True)`)

All FACT from `derive.py`, `Engine.generate` and `EngineCfg` unless marked.

**Idea.** A class-3 molecule is assumed to be a close biosynthetic relative of a library compound. Take the best spectral analogs, compute the mass difference to the target, and apply only those chemical edits whose exact mass change equals that difference.

**Parents.** Analogs with hybrid similarity ≥ 0.3, at most the top 6.

**Transformations.** 46 named single-step edits written as reaction SMARTS, each with an exact element delta:
- Additions and redox edits (32): hydroxylation (aromatic CH, aliphatic CH/CH2/CH3); O-, N- and C-methylation; aromatic methoxylation; O- and N-acetylation; malonylation; O-glycosylation with hexose, pentose, deoxyhexose, glucuronic acid; C-hexosylation; sulfation; C- and O-prenylation; O-ethylation; hydrogenation of C=C and C=O; dehydrogenation; alcohol → ketone; aromatic carboxylation; CH3 → COOH; aromatic amination; chlorination; O-acylation with coumaroyl, feruloyl, caffeoyl, galloyl, benzoyl.
- Removals (14 names, hexose loss listed twice): demethylation (O, N); dehydroxylation (aromatic, aliphatic); deacetylation; loss of hexose, pentose, deoxyhexose; demalonylation; desulfation; deprenylation; decarboxylation; demethoxylation.

**Matching.** All single edits and all unordered pairs (with repetition) are precomputed with their mass change. A combo is used if its mass change is within the tolerance of `target − parent mass`. Tolerance = `max(0.004 Da, target × 10 ppm)`.

**Isomerisation.** When the parent has the same mass as the target, remove-then-add pairs with zero net change are applied, which yields positional isomers of the parent.

**Application limits.**
- Each edit is applied at every matching site through RDKit reactions.
- First step: at most 32 products per molecule; second step: at most 16 products for each of at most 24 intermediates.
- At most 800 reaction applications per parent.
- At most 60 products per parent and **150 per query**.
- Maximum depth: 2 steps.

**Acceptance of a product.** It must standardise to an uncharged structure, have a metric key that is not already in the ±10 ppm pool window and not already generated, and have a mass within the tolerance. Parents are visited in descending similarity, so the list fills from the best analog first.

**Scoring.** Generated candidates are appended to the pool window and go through exactly the same feature computation (fingerprint and fragments computed on the fly). They differ only in the six G columns: flag, parent similarity, steps, Tanimoto to parent, number generated, `f·z` rank among generated. Library and analog features apply as for any candidate.

**Limiting their rank.** There is no explicit cap or reserved slot. The ranker learns from simulated class-3 queries how high a generated row may go. The `derivation` family of `fe_v4` adds site and transform priors for the same purpose.

**Simulation hook.** `Engine.run(..., drop_pid=...)` removes the true structure from the pool so that only generation can recover it; `exclude` and `exclude_sid` hide its spectra and its analog representative. This is how class-3 ranker rows must have been produced. (FACT for the hook; INFERENCE for the use.)

**Evidence.**
- Offline (FACT, `derivation.py` log): 13% of class-2 truths are "derivable"; the `derivation` features improve class-3-type queries by +0.07 to +0.10 MRR in the author's simulation.
- LB: no isolated number for generation on vs off in this stack. Other teams: targeted derivative enumeration 0.337 → 0.335 (prvsiyan); generated candidates "has not yet paid off on the leaderboard" (Udam, rank 6) (intel §2.2).

---

## 5. The PubChem-only channel

All FACT from cell 8 (embedded `probe_core2` and `pc_runner`) and cell 14 unless marked.

**Table.** `casmi26-pubchem-tier`: 105,875,489 structures (from the array sizes), stereo stripped, organic CHNOPS + halogens, 150–1250 Da, sorted by mass; three arrays (mass, byte offsets, concatenated SMILES; 7.2 GB).

**Screen, per molecule.**
1. Logits `z` from FPNet **A+B averaged** (not full1), built exactly as in §2.1.
2. Window: all PubChem entries within ±10 ppm of the target; no subsampling.
3. Pass 1: compute only the 4096-bit ECFP4 of every window entry and score it with the 1464 selected ECFP4 bits of `z`. Keep the top `PC_N1` = 5000 (1000 before v4i).
4. Pass 2: full 10,226-bit fingerprints for the survivors; full `f·z`; sort descending.
5. Walk down the order, compute the metric key, skip keys already seen and **skip every key that is in the pool**. Stop at 25.
6. Also record `best_pool_fz` = the highest `f·z` (same `z`) over the pool window (±10 ppm, fallback 30).

The channel therefore proposes only structures the main engine cannot see.

**Gate** (constants `LIB_TAU = 0.9`, `REL_TH = 600`):

| Condition | Action |
|---|---|
| No PubChem list | Base list unchanged |
| `lib_max ≥ 0.9` | Base list unchanged ("untouched") |
| otherwise, `rel = top PubChem f·z − best_pool_fz > 600` | "Aggressive": PubChem candidates take positions 2, 4, 6, 8, 10 |
| otherwise | "Gentle": PubChem candidates take positions 4, 8, 12, 16, 20 |

- Position 1 always stays with the base list. At most 5 PubChem candidates are inserted (more only if the base list runs out). Base candidates fill the other positions in order, so the tail of the base top-25 is pushed out.
- `f·z` is a sum of logits over roughly 100–300 set bits, so 600 is a margin in raw logit units specific to these models. It must be re-tuned for any retrained FPNet. (INFERENCE.)
- Origin of the thresholds: "Gate chosen on out-of-fold simulation (positive under all mixtures)" (FACT, `casmi26-v3-inference` header). The simulation is not public.

**Popularity prior.** There is **none** in Ahmed's stack or in the 0.399 fork (FACT: no popularity term in cells 8 and 14). The popularity prior (`log1p(SID count) + log1p(PMID count)`, weight 0.25 on the PubChem list, 0.15 on the main list) belongs to wangpenghua / dmitriigluzdov's variants (intel §1.2).

**Evidence.**
- The PubChem-only list alone scores 0.037 as a probe submission (FACT, v3 header).
- Adding the gated channel: 0.354 → 0.358 (inside noise).
- Pass-1 depth 1000 → 5000: no stated change (0.373).
- Popularity prior + deeper screen in dmitrii's fork: 0.373 → 0.386; his follow-up promotion rule: 0.386 → 0.374 (intel §2.2).
- Unfiltered PubChem expansion in a different stack: 0.335 → 0.205 (prvsiyan). This is why the channel is kept outside the candidate pool and behind fixed slots.

---

## 6. What is not recoverable, and how to reconstruct it

| # | Gap | What the public material does tell us | Most plausible reconstruction (INFERENCE) | How to validate locally |
|---|---|---|---|---|
| 1 | FPNet training loop (loss, optimiser, schedule, batch size, steps, `merge_p`) | Sampler, negatives, augmentation, validation metric (§2.2); checkpoint saves `step`; the DreamsFP run used 20k steps with EMA | AdamW, lr about 2–3e-4 with warm-up and cosine decay, batch 128–256, 30–40k steps, EMA of weights, `merge_p` about 0.3. Loss = cross-entropy over the 64 `f·z` scores + 0.1–0.5 × BCE on bits + small L1/L2 on element counts. prvsiyan's public 6-layer model with the same idea trains in "~2 h" on a GPU at 24–36k steps. | `window_mrr`-style metric on a held-out structure fold, per spectrum and merged. Reference points: prvsiyan's channel alone 0.468 class-2 MRR in simulation; Ahmed's A 0.581 on np-examples; megayak 0.49 on np-examples (intel §2.3). A short 2k-step run on the T4 gives the time per step before committing quota. |
| 2 | `split.parquet` (fold names `np`, `nplib`, `syn`, `plusk`, `twin`) | Names only; library index list in `build.LIBS`; natural-product libraries listed in `engine.NP_LIBS` | `np` = structures of `enveda-np-examples`; `nplib` = a sample of structures from the NP libraries; `syn` = a sample from the synthetic/drug libraries (`enveda-180`, `drug_plus`); `plusk` = a sample from `pluskal_ms2`; `twin` = structures that have a same-formula or near-duplicate partner. All held out by structure. | The exact split does not need to be reproduced. Requirement: every structure used as a simulated query for the ranker must be unseen by the FPNet that produces its features. Check: FPNet MRR on the held-out fold must be clearly below its MRR on training structures (A: 0.58 vs 0.93). |
| 3 | Ranker row simulation ("sim1x"; "sim10_s2" for the v6 engine) | Engine hooks `exclude`, `exclude_sid`, `exclude_lib`, `drop_pid`; meta columns `gid, sid, fold, regime`; regime names in logs: `c1`, `c2`, `c2L`, C2FAM, C3FAM; about 18–24k query groups in the v6 simulation; structure-grouped 5-fold CV; "LB estimate" from a class mixture | For each held-out structure pick one library's spectra of one polarity as the query (1–16 spectra). **c1:** hide only the query spectra themselves (and that library's representative, via `exclude_lib`); other spectra of the structure stay visible. **c2:** hide all spectra of the structure; keep it in the pool. **c3:** as c2 and also remove it from the pool (`drop_pid`), so only generation can recover it; queries where generation fails have no positive row. Label = 1 for the truth key. Weight regimes to an assumed class mix. `c2L` is not determinable (possibly class 2 with a weaker library context). | (a) megayak's public calibration `LB ≈ 0.162·c1 + 0.220·c2` (intel §2.3) as a sanity check of regime MRRs. (b) Compare per-regime MRR with public numbers: class-2 simulation 0.61–0.73 for the four-channel ranker (prvsiyan). (c) One LB submission of engine + ranker alone; the author's value is 0.354. |
| 4 | Query sampling details (how many spectra per simulated query, instrument mix) | Test: 1–16 spectra, median 3, all timsTOF (intel §3.5); fragnet training sets weight NP libraries ×2 and timsTOF ×2 | Sample the number of spectra from the test distribution; over-weight timsTOF and NP-library queries by about 2 | Compare feature distributions (`n_query`, `q_npeaks`, `lib_max`, `top_sim`) between simulated queries and the visible dummy test |
| 5 | Ranker parameters actually stored in `ranker_v4b.pkl` and `ranker_0.pkl` | Code defaults, 500 rounds, 4 boosters | Use the defaults in §2.4 | Structure-grouped CV; the public record says deeper models and "honest-CV" tuning hurt on the LB (0.335 → 0.321 / 0.330, intel §2.2), so do not tune hard |
| 6 | Pool build script | Readers, standardiser, fingerprint, bit selection, fragment enumerator are all described in §2–3 | Standardise → metric key → de-duplicate on key → mass sort → fingerprints → bit selection on a 200k sample → fragment CSR | Row count close to 710,701 with COCONUT 2.0; on a sample, recall of training truths in the ±10 ppm window should be about 0.99 (prvsiyan reports 0.992 with no cap) |
| 7 | Gate thresholds (`0.9`, `600`, slot tables) origin | "Out-of-fold simulation (positive under all mixtures)" | Simulate PubChem-only truths: take held-out structures, remove them from the pool, check where they land in the PubChem list; sweep thresholds under several class mixes | Same simulation. Expect a small effect: the probe is worth 0.037 LB in total |
| 8 | `fe_v4` builders (prior tables, fragnet counts, capability tables, `build_v4v1.py`) | Consumers and long decision logs | Do not rebuild (§7) | n/a |
| 9 | DreamsFP fine-tune script | Architecture port; 20k steps; EMA; layer-wise parameter groups | Do not rebuild | n/a |
| 10 | Difference between FPNet A and B | Same class, two files | A second seed or a different `merge_p` | Train two seeds; the public evidence for a two-model pair is +0.019 LB in family A (per-spectrum + merged models), and a third model hurt (0.335 → 0.329) |
| 11 | Forward-model panel that justified λ and top-N | README: gains "+0.027 syn / +0.013 nplib / +0.055 np"; N = 60 keeps about 97% of the gain; λ 0.5 recommended, later 1.0 on LB | Reuse the public constants | Our own simulation on positive-mode `[M+H]+` queries; LB: λ 0.5 → 1.0 gave +0.007, λ 2.0 gave −0.034 |

---

## 7. Prioritised rebuild list

Effort is for one person writing new code from this description. Compute: local = 8 cores / 15.6 GB RAM, about 13 GB free disk; GPU = Kaggle T4, about 30 h per week. Effort and compute are INFERENCE; LB evidence is FACT from `competition_intel.md` §2.2 unless marked.

| # | Component | LB evidence | Effort | Compute | Rebuild? |
|---|---|---|---|---|---|
| 1 | Chemistry core: adduct table, neutral mass, standardiser, metric key, 7-block fingerprint + bit selection | Required by everything | 1–2 days | CPU minutes | **Yes** |
| 2 | Pool = train ∪ COCONUT, mass-sorted, packed fingerprints, depth-2 fragment masses | Required. No-cap ±10 ppm window is supported by 0.282 → 0.311 when a cap of 80 was removed (family A) | 2–3 days | CPU: fingerprints about 1–2 h on 8 cores; fragments several hours. Output about 1.3 GB, so build on Kaggle CPU or free local disk first | **Yes** |
| 3 | Spectrum cache + library channel (entropy similarity, entropy weighting) | Library only: 0.151 (family A); a library-twin-only probe 0.275 (D/742088) | 1–2 days | CPU minutes with numba; RAM is the constraint (2.5 M spectra) | **Yes** |
| 4 | Analog propagation (representatives, ±200 Da hybrid search, sim⁴ × Tanimoto) | 0.151 → 0.233 with a calibrated ranker (family A); "+0.082" in the intel summary | 2 days | CPU; about 0.5–2 s per query | **Yes** |
| 5 | FPNet, one model trained on all folds (deployment) and one trained with hold-out folds (to produce honest ranker rows) | FPNet channel 0.266 → 0.299; second view 0.299 → 0.335 (family A). full1 vs A in this stack: 0.378–0.384 vs 0.380, inside noise | 3–4 days of code | GPU: about 50 M parameters; estimate 4–8 T4 hours per model at 30–40k steps (unmeasured: time a 2k-step run first). Two models fit in one week of quota. The 909 MB pool fingerprints must sit on the GPU for negative sampling (fits a 16 GB T4) | **Yes** |
| 6 | Fragmentation (MetFrag-lite) + sub-formula features | 0.245 → 0.266 (family A); a tuned fragmentation scorer gave 0.341 → 0.336 elsewhere | 2 days | CPU; about 14 ms per candidate reported for a similar implementation | **Yes** (cheap once the pool fragments exist) |
| 7 | Ranker simulation + LightGBM LambdaRank on about 86 features (the 94 minus the unused cross-encoder block) | The whole v1 pipeline = 0.354. Seed spread of rankers on the LB is about ±0.017 (megayak) | 4–6 days; highest risk | CPU: several thousand simulated queries × about 1–3 s each = hours on 8 cores, plus GPU minutes for FPNet logits; LightGBM training minutes | **Yes** |
| 8 | Class-3 derivation (46 transforms, ≤ 2 steps, ≤ 6 parents, ≤ 150 products) | No isolated number in this stack. Elsewhere: 0.337 → 0.335 (prvsiyan); "has not yet paid off" (Udam). Offline: only the class-3 regime gains | 2 days | CPU; RDKit reactions, bounded by the 800-application budget | **Yes, but last among the engine parts.** It is the only route to class-3 truths, and its cost is small; validate with a `drop_pid` simulation before trusting it |
| 9 | ICEBERG + GLACIER same-formula re-scoring (public MIT code and MassSpecGym weights; our own runner and slot-preserving re-rank) | 0.358 → 0.366 (ICEBERG) → 0.373 (GLACIER) → 0.380 (λ 0.5 → 1.0). λ 2.0: 0.386 → 0.352. Top-N 60 → 100: no change | 3–5 days (dependency shims for DGL / torch-scatter offline, RDKit 2025.03 pin, subprocess isolation, time budget) | No training. Inference about 25–32 predictions/s on T4; budgets 45–90 min each inside the 9 h notebook limit | **Yes.** Largest measured gain per unit of work, and licence-clean at source |
| 10 | Gated PubChem-only channel (two-pass screen, pool exclusion, slot merge) | 0.354 → 0.358; probe 0.037; N1 1000 → 5000 no change | 2–3 days including our own mass-sorted PubChem table | CPU; table about 7 GB (build on Kaggle; local disk is too small to hold it next to the pool). Needs FPNet logits only | **Optional, second wave.** The gain is inside noise; the popularity-prior variant has better evidence (0.373 → 0.386) but is another author's idea and was followed by a −0.012 step |
| 11 | Second FPNet (B) | Only used by the PubChem channel and `fe_v4` views. Family A: two-model pair +0.019, third model −0.006 | 0 extra code | One more GPU run | **Only if #10 is built** or as a seed ensemble |
| 12 | Fusion with a second, differently built engine (RRF, α = 0.6, K = 3) + forward re-rank on the union | 0.384 → 0.399; 0.366 → 0.382 → 0.386 on the older base; "+0.006" quoted in #4 | 1 day for the fusion itself | none | **Yes if we have a second engine.** Our current pipeline added nothing as a third list (0.399 → 0.399, task brief); the gain depends on the second list being strong and different |
| 13 | `fe_v4` families (derivation, frag2, analog_struct, fragnet) and their tables | **Negative:** 0.358 → 0.354 | 1–2 weeks | CPU hours for tables | **No** |
| 14 | DreamsFP (model D) and `model_views` | No positive evidence; v4d reverted; DreaMS ties or loses in CV elsewhere | 1 week | 486 MB model; many GPU hours | **No** |
| 15 | Cross-encoder (`xenc`) | Unused in v4n; failed for prvsiyan | — | — | **No** |

### Recommended minimal subset

Build items **1–7 and 9** first, then **8**, then decide on 10 and 12 from our own simulation.

- Expected result (INFERENCE): the author's equivalent configuration (v1 engine + FPNet A + generation) scored 0.354, and the forward models added about +0.022. A re-implementation with our own training will not match this exactly; the public-LB noise (±0.006–0.02) is of the same size as most of the individual steps.
- Total effort (INFERENCE): about 3–4 working weeks for 1–7 + 9, plus 2 days for 8. GPU: one week of T4 quota for two FPNet runs, with a timing run first.
- Order that de-risks early: 1 → 2 → 5 (start the GPU run as soon as the pool exists) → 3, 4, 6 in parallel with training → 7 → one LB submission (target: near 0.35) → 9 → 8.
- Things to decide by our own measurement, not by copying constants: the `600` gate margin (model-specific), the class mix used to weight ranker rows, and the forward-model weight (public evidence supports 1.0 and rejects 2.0).

---

## 8. Open points

1. How FPNet A and B differ.
2. The exact loss and schedule of the FPNet runs, and their T4 training time.
3. The meaning of regime `c2L` and of fold `twin`.
4. The class mixture and row weights used for the ranker and for the gate.
5. The parameters stored inside the two ranker pickles (not inspected; code defaults assumed).
6. Pool composition by source, element filter and mass range (table not inspected).
7. Whether generation is a net gain on the LB in this stack (no isolated measurement exists).
8. Licence of the notebook code itself: the fork's header says the v4n notebook code is "Apache-2.0" while the datasets that hold the `casmi` package say "non-commercial, with attribution" (FACT, fork cell 0 and dataset descriptions). This spec deliberately contains no copied code so that the question does not affect our implementation.

# v4n rebuild specification: every artifact, how it was made, how we rebuild it

Written 2026-10-07 for DEC-014. Research only: no existing file was changed, nothing was trained or downloaded.
Companion documents: `research/analysis/v4n_method_spec.md` (method in prose, written 2026-10-01 before the code was
read line by line) and `research/analysis/fork_licence_rebuild_plan.md`. This spec supersedes both where they differ,
because it is based on a full read of the reference code.

## 0. Conventions

**Labels.**
- **FACT**: read in a file. Code citations use `file:line` inside `research/v4n_rebuild/ahmed_ref/` (`code/casmi/...`,
  `code/fe_v4/...`, `notebooks/...`), or name a function.
- **INFERENCE**: deduced from how something is consumed (shapes, dtypes, keys, names, logs).
- **HYPOTHESIS**: a plausible guess that a run must confirm.

**Clean-room rule.** We read his code to learn what each artifact is. We write our own code and train our own
artifacts. We ship none of his weights, pools, tables or ranker. His code may run locally as a **test oracle** on
OUR artifacts (section 9, V3). It must never enter a Kaggle dataset or notebook. That local use is non-commercial
research use; label it in the decision log.

**Allowed inputs** (host ruling plus team rule, see memory `goal-prize-eligible`):
- `train.parquet`: 2,539,608 spectra and 275,810 InChIKey14s (FACT, read today).
- COCONUT (CC BY 4.0).
- PubChem.
- enveda-np-examples. It is already inside `train.parquet` as `ingest_lib == 'enveda-np-examples'`, 1,184 spectra (FACT).
- DreaMS SSL weights (MIT / CC BY 4.0).
- MassSpecGym ICEBERG/GLACIER checkpoints (MIT).
- RDKit (BSD).

Not allowed: NIST/METLIN-trained weights, any `ahmedberatozer/*` file, prvsiyan's BIO table (CC BY-NC-SA), and
other teams' popularity arrays (`pc_lsid.npy`, `pool_lsid.npy`, `pool_lpmid.npy`). We build our own popularity arrays.

**Short names.**
- His "A" is `fpnet_0.pt` = `fe_A.pt`. The two files are identical (FACT: same sha256 `edccf479…` in `MANIFEST.json`).
- "B" = `fe_B.pt`. "full1" = `fpnet_full1.pt`. "D" = `fe_D.pt` (DreamsFP).
- "Engine" = `code/casmi/engine.py:Engine`. "V1FE" = `code/fe_v4/v1engine.py:V1FE` (engine plus feature families).

---

## 1. Executive summary

1. **What v4n is** (FACT: `notebooks/casmi26-v4l-inference.ipynb`, sovereign cell 2 and cells 10–13). Per molecule:
   - The v1 Engine retrieves every pool structure within ±10 ppm (pool = train ∪ COCONUT, 710,701 rows).
   - It appends up to 150 generated derivatives and computes 94 features (8 of them, `xs_*`, are always 0).
   - V1FE adds 66 family features.
   - A 4-booster LightGBM lambdarank (`ranker_v4b.pkl`, 160 features) sorts the candidates; the top 60 are kept.
   - ICEBERG and GLACIER re-order the list inside each same-formula group (λ = 1.0 each).
   - A gated PubChem-only list is interleaved at fixed slots when `lib_max < 0.9`.
   - v4n differs from v4l in one way: the engine's FPNet bank is `fpnet_full1.pt` instead of A.
2. **Nothing that was trained ships with its training code.**
   - Present: architectures, data loaders, feature code and table *consumers*.
   - Absent: every *producer script*. That covers the pool build, the split, the FPNet and DreamsFP training loops,
     the ranker simulation (`simulate_v1ctx.py` / "sim1x"), `build_v4v1.py`, `res_derivation.py`, `res_frag2.py`,
     `res_analog_struct.py` and `res_fragnet.py`.
   - Every recipe below is reconstructed. Each one is labelled.
3. **The value is not in his artifacts.** LB history, all FACT from version headers and `competition_intel.md` §2.2:
   - v1 engine + FPNet A + ranker: 0.354.
   - + gated PubChem: 0.358.
   - fe_v4 families + new ranker (v4b): 0.354.
   - + ICEBERG: 0.366.
   - + GLACIER: 0.373.
   - λ = 1.0: 0.380.
   - + full1: 0.384.
   - The fused public notebooks add popularity (+0.010 / +0.004) and a second engine (+0.015).
   - The 0.42 "sovereign" notebook contains ineligible parts.
   - **Conclusion (INFERENCE):** the parts worth rebuilding first are the base engine, the FPNet, the ranker simulation
     and the post-steps we already own. The fe_v4 families and DreamsFP come last.
4. **Our assets cover a large part** (FACT, inventory 2026-10-07):
   - CFT nets (`research/train_cft/`) are architecturally almost the same as his FPNet: d512, 8 layers, 51.3 M
     parameters, contrastive loss over 63 decoys + BCE + element head.
   - We already have a pool of 712,199 structures, our own ICEBERG/GLACIER runner, a PubChem table of 100.96 M rows
     with popularity counts, the local bench, and E6 (LB 0.360) as a fusion partner.
5. **MVR (section 6).**
   - v1-equivalent engine (86 live features) + our CFT-style FPNet trained with a large structure hold-out + our own
     simulation-trained ranker + class-3 generation.
   - Plugged into our existing post-steps (forward models, PubChem, popularity), then rank-fused with E6.
   - Expected 0.37–0.40 public (HYPOTHESIS).
   - About 3 weeks; about 45 T4-h + about 60 CPU-h.
   - The full rebuild (fe_v4 + D) adds about 2 weeks, about 25 GPU-h and about 40 CPU-h, for an unproven gain.

---

## 2. The v4n inference path: artifact inventory

FACT unless marked. "Loaded by" is the first consumer on the v4n path.

| ID | Artifact (his dataset) | Size | Loaded by | Used in v4n? |
|---|---|---|---|---|
| L1 | spectrum cache `spec_meta.parquet`, `spec_off/mz/it.npy` (built at run time) | ~1 GB (INFERENCE) | `casmi/build.py:build_spec_cache` → `casmi/library.py:Library.__init__` (lines 30–54) | yes |
| P1 | `fp_bits.npy` (v2-pool) | 41 KB, int32 ×10,226 | `casmi/pool.py:26` | yes |
| P2 | `train_structs.parquet` (v2-pool) | ~20 MB (INFERENCE) | `library.py:36`, `build.py:52` | yes |
| P3 | `train_fp_sel.npy` (v2-pool) | 353 MB, uint8 (275,810 × 1,279) | notebook cell 11; `Engine.__init__` | yes |
| P4 | `pool_meta.parquet` (v2-pool) | ? | `pool.py:17–24` | yes |
| P5 | `pool_fp.npy` (v2-pool) | 909 MB, uint8 (710,701 × 1,279) | `pool.py:25` | yes |
| P6 | `pool_frag_off.npy` + `pool_frag_mass.npy` (v2-pool) | 5.7 MB + 357 MB | `pool.py:30–32` | yes |
| M1 | `fpnet_full1.pt` (fpnet-full1) | 199 MB | sovereign cell 11: engine `ModelBank` | yes (engine bank) |
| M2 | `fpnet_0.pt` = `fe_A.pt` (v4b-models) | 199 MB | `v1engine.py:78` slot A; also the engine fallback | yes (model-view slot A) |
| M3 | `fe_B.pt` (v4b-models) | 199 MB | `v1engine.py:78` slot B | yes |
| M4 | `fe_D.pt` (v4b-models) | 486 MB | `v1engine.py:78` slot D via `fpnet6.model_from_ckpt` | yes |
| M5 | v3-models `fpnet_0.pt`, `fpnet_1.pt` | 2 × 199 MB | PubChem runner (`pc_runner`, `assert len(models)==2`) | yes (PubChem z) |
| R1 | `ranker_v4b.pkl` (v4b-models) | 14 MB | v4l cell 3 `rank_score` | yes |
| D1 | `fe_v4/res/derivation/prior_tables.pkl` | 20 MB | `fams/derivation.py:Priors` (273–282) | yes |
| D2 | `fe_v4/res/derivation/parity_fixture.pkl` | small | `derivation.py:self_test` (613–626), asserted in v4l cell 3 | yes (deployment check) |
| G1 | `fe_v4/res/frag2/caps.txt` | 149 B | `frag2.py:prepare` (119–120) | yes |
| G2 | `fe_v4/res/frag2/pool_cap.npy` | ~15.6 MB uint8 (710,701 × 22) (INFERENCE) | `frag2.py:118` | yes |
| G3 | `fe_v4/res/frag2/pool_f1_off.npy`, `pool_f1_mass.npy` | ? | `frag2.py:121–122` | yes |
| S* | `fe_v4/res/analog_struct/{pool,train}_{hash,num,fvec,elsmi_buf,elsmi_off,coresmi_buf,coresmi_off}.npy` (14 files) | ~100–200 MB (INFERENCE) | `analog_struct.py:prepare` (349–366) | yes |
| N1 | `fe_v4/res/fragnet/rec_{ok,aoff,atom,boff,bond}.npy` | 150 MB (FACT: fragnet log) | `fragnet.py:PoolRecords` (723–741) | yes |
| N2 | `fe_v4/res/fragnet/stageA.npz` | small | `fragnet.py:StageA` (532–567), `prepare` (816–823) | yes |
| N3 | `fe_v4/res/fragnet/stageB.txt` | 545 KB, LightGBM, 300 trees, 31 features | `fragnet.py:prepare` (821–822) | yes |
| N4 | `fe_v4/res/fragnet/{countsA.npz, featB.npz, sets.pkl}` | sets 9 MB | **not loaded**: only sha-checked by the MANIFEST loop in v4l cell 0 | no (training intermediates) |
| X1 | PubChem tier `pc_mass.npy`, `pc_off.npy`, `pc_smiles.npy` | ~7 GB | `probe_core2.init_worker` (v4l cell 1) | yes |
| X2 | `casmi26-iceberg` (ice_runner, fuse, MassSpecGym ckpt, wheels) | ? | v4l cell 5 | yes |
| X3 | `casmi26-glacier` (gl_runner, gl_fuse, ckpt) | ? | v4l cell 5 | yes |
| X4 | RDKit 2026.03.3 wheel | 37 MB | cell 0 `pip install` | yes |
| X5 | code `casmi/*`, `fe_v4/*` | — | everything | yes. We write our own code |

**On disk but not on the v4n path** (FACT):
- `xenc.py` and `graphs.py`: the engine is built without an `xscorer`, so `xs_*` = 0 (v4l cell 3).
- `chemfeat.py`: no family imports it (grep).
- `structs.py`: producer-side helper.
- `traindata.py`: training only.
- `ranker_0.pkl` of v3-models: only used to locate the directory.
- Sovereign-only extras: `pc_lsid.npy`, `pool_lsid.npy`, `pool_lpmid.npy`, `dreams_np_emb.npy`, and the prvsiyan engine.

**Hidden subtleties found in the code** (each matters for the rebuild):

1. **The ranker was trained on features from model A but receives full1 features in v4n.**
   - FACT: the v4b ranker was built "on the v1-engine simulation sim1x" (v4l header), and A is the hold-out model
     (`traindata.py:11` default `holdout_folds`). full1 is used only from v4m onwards (sovereign cell 2).
   - INFERENCE: the train/serve mismatch is deliberate; it is worth +0.004 LB, inside noise.
2. **The family code uses the engine's f·z.**
   - `frag2` (top 600) and `fragnet` (top 400) choose their candidate subsets by the engine's `X[:, 'fz']`, i.e. the
     engine-bank f·z (FACT `frag2.py:505–510`, `fragnet.py:873–877`).
   - In v4n that is full1. So family features depend on the engine bank too.
3. **The DreamsFP checkpoint was trained in bf16.**
   - FACT, `fpnet6.py:426–431`: "pilot_d1 trained with --attn_bf16". T4 inference runs it in fp16 with fp32 attention.
   - INFERENCE: it was trained on an Ampere-or-newer GPU. On a T4 we cannot use bf16 (section 3.8).
4. **The derivation family uses a different transform table from generation.**
   - Its priors use the v6.3 grammar `derive6.py`: 71 transforms, 2,627 one- and two-step combos.
   - The engine's generation uses `casmi/derive.py`: 46 transforms, 1,127 combos. `dehexose2` exists only in v1.
   - FACT: counted with `research/scratch_wf/v4n/count.py`; `derivation.py:196–203`.
5. **Two different `_rank_norm` functions.**
   - Engine features use a stable-argsort rank (`engine.py:50–52`). It depends on candidate order when values tie.
   - Families use an average-tie rank (`ctxlib.py:324–343`).
   - We must reproduce both behaviours, or retrain the ranker on our own definitions. We retrain, so the only rule is:
     identical code offline and online.
6. **`n_merged` is computed differently in two places.**
   - Engine items: `n_merged = min(int(ce_n), 8)` (`engine.py:345`).
   - Family model views: `min(max(1, ce_n), 8)`, with a deterministic merged-adduct tie-break by train adduct counts
     (`v1engine.py:39–70`).
   - Same rule as point 5: offline must equal online.
7. **Fragment masses are rounded to 1e-3 Da before the float32 cast.**
   - FACT: fragment masses are rounded (`frag.py:fragment_masses`), and the pool table is stored as float32
     (`frag2.py:10`, `_gen_frags` 383–402).
   - Generated candidates must get the same rounding and cast, so pool and generated rows are comparable.

---

## 3. Per-artifact specification

Each artifact has four parts:
- (1) what it is and how inference uses it;
- (2) how it was produced;
- (3) our rebuild recipe;
- (4) whether one of our assets can substitute.

Compute figures are INFERENCE unless a log is cited. "8c" = this 8-core PC.

### 3.0 Shared chemistry core (no artifact, but every artifact depends on it)

FACT, `casmi/chem.py`.

**Adducts and masses.**
- 59 adducts plus `<unk>`. The ten test adducts come first (`chem.py:46–111`).
- Neutral mass formula: `(mz·z − δ)/n`, with the electron correction inside δ (`chem.py:118–127`).

**Standardiser** (`chem.py:230–246`).
- Keep the largest fragment, uncharge, remove stereo, write canonical SMILES.
- Return **None if the molecule stays charged**.

**Score key** (`chem.py:203–214`).
- Tautomer-canonical InChIKey14, as the metric computes it, under RDKit 2026.03.3.

**Raw fingerprint** (`chem.py:264–298`).
- Blocks, in order: ECFP4 4096 | ECFP6 4096 | FCFP4 2048 | RDKit path (maxPath 6) 2048 | atom pair 2048 |
  torsion 2048 | MACCS 167.
- Total 16,551 bits.
- Generators: `rdFingerprintGenerator` defaults with the stated `fpSize`. FCFP4 uses `GetMorganFeatureAtomInvGen`.

**Our code.**
- `vrb/chem.py`: our own adduct table, neutral mass, standardiser, key and fingerprint.
- We already have equivalent code:
  - `research/train_cft/cft_fp.py` builds the cfp1 layout = the same blocks **without MACCS**.
  - `research/train_pkg/prep_data.py` builds the prvsiyan layout.

**Decision needed: which bit layout.**

| Option | Bit set | For | Against |
|---|---|---|---|
| **cfp1 (recommended)** | 11,170 bits from 6 blocks, no MACCS (FACT `train_cft_report.md` §2) | Our CFT trainer, data and nets already use it. Every consumer is agnostic to `nbits`: FACT `pool.py:27`, `engine.py` uses `P.nbits`, `fpnet.FPNet(nbits)` | `model_views` (BLOCKS, `model_views.py:6–13`) and `analog_struct` `bt_*` map bits to blocks by the raw layout, so we must give our own block table. MACCS is lost (139 kept bits in his set, INFERENCE: minor) |
| His 10,226-bit rule | `[0.005, 0.995]` frequency over a 200k sample, 7 blocks | Literal reproduction | All CFT data must be re-made |

**Leakage / provenance precaution (INFERENCE from his logs, `analog_struct.py:50–57`, `fragnet.py:5–15`).**
- 2,576 of 25,746 keys present in both sources have different SMILES in train vs COCONUT. The cause is tautomer
  style, e.g. imidic acid `C(O)=N` vs amide.
- Full-fp Tanimoto between the two styles is only 0.70.
- Risk: the ranker learns "train-style fingerprint ⇒ truth", because simulated truths are train structures while
  hidden class-2 truths are COCONUT-style.
- Fix: run `TautomerEnumerator().Canonicalize` before **every** fingerprint, for the pool, train structures, generated
  candidates, PubChem candidates and FPNet targets.
- Cost: 10–15 ms per molecule (FACT `fragnet.py:7`), so about 3 CPU-h for 0.74 M structures and about 25 min on 8c.
- **This is a deliberate deviation from his pool.** HYPOTHESIS: it is neutral or positive. Validate with V5c.

### 3.1 L1 — spectrum cache (built at inference)

1. **What and how used (FACT).**
   - Format: CSR peak arrays of every `train.parquet` spectrum (`build.py:48–93`).
   - Cleaning: peaks above precursor + 2 Da dropped; floor 0.1 % of base peak; top 512; m/z sorted; base peak = 1.
   - Metadata columns: `sid` (inchikey14 → `train_structs.sid`), `lib` (index into `LIBS`, 11 libraries), `instr`,
     `adduct_ix`, `mode`, `nm`, `n_clean`, `ce_mean/min/max/n`, plus the remaining parquet columns.
   - Uses: `Library` uses it for direct search (`engine.py:261–305`), analog representatives (`engine.py:114–124`)
     and analog search (`engine.py:127–171`). `Library.smass` = **mass of the labelled structure**, not the
     precursor-derived mass (`library.py:47–50`).
2. **How produced (FACT).** `build.py:build_spec_cache` is present. It runs in the notebook at every inference
   (v4l cell 2).
3. **Rebuild.**
   - Write `vrb/build_lib.py` with the same semantics.
   - Inputs: `train.parquet` and our P2.
   - Cost: minutes on the Kaggle CPU (v4l prints it). RAM is about 2–3 GB.
   - Leakage: none at inference. For simulation the masks live in the engine (section 4).
4. **Substitute.** The E1/E6 engine already builds an equivalent cache (`research/bench/eng/`, FACT inventory). Its
   cleaning parameters differ: precursor + 1.5 Da, 128 peaks (FACT `train_cft_report.md` §2). Reuse the loader, but
   rebuild with the v1 parameters, because library search uses top-256 of the cleaned list (`library.py:15–24`).
   Cost: 0.

### 3.2 P2 — `train_structs.parquet`

1. **What and how used.**
   - FACT: columns include `sid`, `inchikey14`, `mass`, `key`, `smiles` (`library.py:36`, `build.py:52`,
     `traindata.py:22`).
   - FACT: 275,810 rows, from the `train_fp_sel.npy` size (128 + 275,810 × 1,279 bytes) and from today's count of
     unique `inchikey14` in `train.parquet` (275,810).
   - Uses:
     - `sid` indexes P3, `Library.struct_*` and analog lists;
     - `struct_mass[sid]` gives the library window mass;
     - `struct_smiles[sid]` gives the generation parent (`engine.py:181`) and the derivation and analog_struct
       lookups.
2. **How produced: absent.**
   - INFERENCE: `structs.process_smiles` (present, `structs.py:34–53`) over one SMILES per `inchikey14`, with mass,
     formula, `n_heavy`, `key` and `rawkey`.
   - `sid` = dense 0..275,809. The order is unknown and irrelevant.
   - HYPOTHESIS: charged structures that fail `standardize_smiles` keep their raw SMILES here (the library needs every
     sid) but are missing from the pool. FACT: `traindata.py:42` filters `sid2pid >= 0`, so some sids have no pool row.
3. **Rebuild.**
   - Group `train.parquet` by `inchikey14`. Take the most frequent `normalized_smiles`.
   - Standardise, with a fallback to the raw SMILES when charged. Tautomer-canonicalise (3.0).
   - Compute mass with `ExactMolWt` of the stored SMILES, formula, `n_heavy`, score key and raw key.
   - Store `sid` in key order.
   - Cost: ~1 CPU-h, 10 min on 8c. Disk 25 MB.
   - Check that no spectrum gets `sid = -1`. FACT: `build.py:84` silently maps a missing key to −1, and
     `library.py:48` would then index `struct_mass[-1]`.
4. **Substitute.** `research/train_pkg/prep_data.py` already maps ik14 → molecules: 275,797 molecules after ho
   removal (FACT inventory). Reuse its logic. Cost: low.

### 3.3 P1 — `fp_bits.npy`

1. **What and how used (FACT).**
   - int32 indices into the raw layout. 10,226 bits (`pool.py:26–27`).
   - `nbits = len(bits)` defines the width of every fingerprint, the FPNet output and the analog Tanimoto.
   - Generated candidates are projected with `fp[self.pool.bits]` (`engine.py:201`).
2. **How produced: present as a function.**
   - `structs.select_bits(fp_packed, lo=0.005, hi=0.995, sample=200000, seed=0)` (`structs.py:56–68`).
   - The input set is unknown. INFERENCE: the pool.
3. **Rebuild.**
   - With cfp1: reuse `results/train_cft/data/cfp_bits.npy` (11,170 bits).
   - Otherwise: our own selection over 200k random pool structures.
   - Cost: minutes. Leakage: none (structure-only statistic).
4. **Substitute.** `cfp_bits.npy`. Cost 0.

### 3.4 P3 — `train_fp_sel.npy`

1. **What and how used (FACT).**
   - `np.packbits` of the selected bits per sid, shape (275,810, ceil(nbits/8)).
   - Used for analog Tanimoto (`engine.py:314`), generation parent fingerprints (`engine.py:182`) and
     `QueryView.an_fp` (`ctxlib.py:445–451`).
2. **How produced: absent.** INFERENCE: `chem.struct_record` per P2 SMILES.
3. **Rebuild.**
   - Fingerprint the tautomer-canonical P2 SMILES. Pack MSB-first, as `np.packbits` and `fpnet.unpack_gpu` expect.
   - Cost: ~0.5 CPU-h. Disk 390 MB with cfp1 (1,397 bytes per row).
   - Precaution: for keys present in both sources, use the **same** SMILES as the pool row, so analog Tanimoto and
     candidate fingerprints agree.
4. **Substitute.** None stored. The CFT data has fingerprints per pool row (`cpool_*`). Train sids can be mapped
   through `mol_cpool` (FACT file list), but rebuilding is simpler.

### 3.5 P4 / P5 — `pool_meta.parquet` + `pool_fp.npy`

1. **What and how used (FACT).**
   - Columns: `mass` (float64, **ascending**; `pc_runner` asserts it), `key`, `smiles`, `formula`, `n_heavy`,
     `src` (int8), `train_sid` (int64, −1 = not a train structure) (`pool.py:17–24`).
   - Window: `searchsorted` at ±10 ppm with a 30 ppm fallback (`pool.py:38–42`, `engine.py:210–216`). No cap.
   - `train_sid` links library evidence to a candidate (`engine.py:264`).
   - `key` drives de-duplication of the output and the PubChem exclusion set (`pool_keys`).
   - Size: 710,701 rows (FACT: `fragnet.py:29` "710,701"; `pool_fp.npy` size).
   - The derivation log states that pool SMILES are `standardize_smiles` output (11,123 / 11,123 checked, FACT
     `derivation.py:75`).
2. **How produced: absent.** INFERENCE:
   - Union of P2 SMILES and COCONUT SMILES (`pool.py:1` docstring "train structures U COCONUT").
   - Standardise and drop None (charged). De-duplicate on the score key. Sort by mass. Fingerprint, select bits, pack.
   - Source split: ~275k train + ~460k COCONUT − 25,746 overlap (FACT overlap count from `analog_struct.py:53`).
   - `src` codes are not determinable. HYPOTHESIS: 0 = train, 1 = COCONUT. The engine never reads `src` (FACT grep);
     only provenance diagnostics do.
3. **Rebuild.**
   - Inputs: P2 + COCONUT. Our CSV: `external/coconut/coconut_csv_lite-09-2026.csv`, 436,389 usable after cleaning
     (FACT inventory).
   - Steps:
     - standardise;
     - tautomer-canonicalise;
     - compute the score key;
     - de-duplicate (prefer the train row when a key is in both, and set `train_sid`);
     - sort by mass;
     - compute fingerprints with the chosen layout;
     - pack.
   - Code: `vrb/build_pool.py`.
   - Cost: fingerprints and keys for ~0.74 M molecules ≈ 4–6 CPU-h ≈ 45 min on 8c, or one Kaggle CPU session.
   - Disk: meta ~60 MB; fp ~1.0 GB (cfp1: 712k × 1,397 B).
   - Leakage: the pool deliberately contains hold-out structures (the hidden test also has its truths in the pool).
     The simulation removes them per query where the regime demands it (section 4).
4. **Substitute.**
   - `results/train_pkg/data*/pool_{fp,mass,key,src}.npy`: 712,199 rows, prvsiyan 6,930-bit layout, mass-sorted.
   - `results/train_cft/data/cpool_*`: 1,126,288 rows (INFERENCE from the 9.0 MB `cpool_mass.npy`). It includes
     ~400k PubChem decoys that must be filtered out (`cpool_src`).
   - Substitution cost: a format adapter (parquet meta) plus a `train_sid` column. About half a day.
   - Two changes are still needed: tautomer canonicalisation, and `formula` / `n_heavy` columns if missing.

### 3.6 P6 — `pool_frag_off.npy` / `pool_frag_mass.npy`

1. **What and how used (FACT).**
   - CSR of sorted unique fragment masses (rounded to 1e-3 Da, float32) per pool row.
   - `Pool.frags` (`pool.py:47–50`) → `frag.frag_features` (`engine.py:414–430`) = the 9 F columns. Also frag2 depth 2
     (`frag2.py:405–411`).
   - Only the top 600 candidates by f·z are evaluated (`engine.py:421–422`).
2. **How produced: present as a function.**
   - `frag.fragments_for_smiles(smiles, max_depth=2, max_out=200000)` (`frag.py:122–129`), cast to float32 (INFERENCE
     from `frag2.py:10`, `_gen_frags`).
   - Size: 357 MB / 4 B = 89 M masses ≈ 126 per structure.
   - `frag2.frag_masses_fast` (`frag2.py:161–219`) is a 12× faster bit-exact twin (FACT `frag2.py:21–23`).
3. **Rebuild.**
   - Our own CSR-adjacency fragmenter (one- and two-bond cuts, component masses with implicit H, round, unique, sort),
     run over the pool SMILES.
   - Cost: ~2–4 CPU-h, multiprocessing. Disk ~370 MB.
   - Use the **same function at run time** for generated candidates.
4. **Substitute.** The E1 engine has a MetFrag-lite channel (FACT inventory, `research/bench/eng/`). Its fragment
   definition differs. Reuse only if it is the same depth-2 enumeration. Recommendation: rebuild (cheap).

### 3.7 M2 / M3 / M1 — FPNet A, B, full1 (and M5, the PubChem pair)

1. **What and how used (FACT).**
   - Architecture `fpnet.FPNet` (`fpnet.py:75–132`):
     - 8 pre-LN blocks, d512, 8 heads, GELU FFN 2048, dropout 0.1;
     - peak token = Linear(3d → d) of [SinEmb(m/z), SinEmb(precursor − m/z), Linear([√I, I])];
     - global token = Linear over [SinEmb(precursor), SinEmb_{d/4}(CE/10 + 1), CE-known, n_merged/4, mode,
       log1p(precursor)/10] + adduct embedding (60);
     - pooled = [CLS, masked mean];
     - head = 1024 → 2048 → nbits; element head = 1024 → 256 → 10 (log1p counts).
   - About 49.6 M parameters, fp32: 199 MB (my count matches the file size).
   - Peak prep (`fpnet.py:24–40`): ≤ precursor + 2 Da, floor 1e-3, top 160 by intensity, m/z sorted, √intensity.
   - Engine use (`engine.py:336–393`):
     - per-spectrum items and one merged item per polarity;
     - `z = 0.5·(mean single + mean merged)`;
     - features fz, fp_ll, fp_cos, fp_tan and el_*.
   - `ModelBank` averages several models (`fpnet.py:187–208`).
   - Checkpoint dict: `model`, `step`, `nbits`, `d`, `layers`, `heads` (`fpnet.py:172–184`).
   - Roles:
     - A = engine bank v3.6–v4l, model-view slot A, PubChem z (as v3 `fpnet_0`, INFERENCE);
     - B = slot B and PubChem z (v3 `fpnet_1`, INFERENCE: same file as `fe_B`);
     - full1 = engine bank in v4n.
2. **How produced: partially present.**
   - Present:
     - data access `traindata.Data` (`traindata.py:10–129`);
     - hold-out folds `('np','nplib','syn','plusk','twin')` (line 11);
     - only structures present in the pool are used (line 42);
     - batch = half uniform over spectra, half uniform over (sid, polarity) groups (97–102);
     - `merge_p` merges of 2–4 peers (57–67);
     - augmentation: peak dropout U(0, 0.3), intensity ×exp(N(0, 0.3)), m/z 4 ppm, CE-known dropped with p 0.1
       (83–92);
     - negatives: K = 63 from the ±10 ppm pool window, random if the window has ≤ 1 entry, half random if < 8
       (104–121);
     - batch returns `(inp, pos pid, cand = [pos, negs])` (123–129);
     - evaluation `window_mrr` (132–174): MRR@25 of the truth inside the ±10 ppm pool window, per spectrum or merged,
       bf16 autocast;
     - element targets: log1p counts of the pool formula (30–37).
   - **Absent:** the loop, loss, optimiser, schedule, steps, batch size, `merge_p`, and `split.parquet` (`sid`, `fold`;
     line 21).
   - INFERENCE on the loss:
     - the docstring says "Score … f.z (Bayes log-likelihood up to a constant)" (`fpnet.py:3–5`), so z are
       calibrated bit log-odds, which implies a **BCE term**;
     - "mass-window contrastive ranking loss", so a **softmax CE over the 64 f·z scores**;
     - plus an element regression term.
     - Our CFT trainer implements exactly that (FACT `train_cft_report.md` §2).
   - INFERENCE: A and B differ by seed and/or `merge_p` (not determinable).
   - FACT: full1 = "trained on ALL train folds incl. np-examples" (dataset description quoted in
     `v4n_method_spec.md` §2.2).
   - FACT: reference quality of A from our intel: window MRR 0.934 on training structures vs 0.581 on the
     np-examples panel.
3. **Rebuild (our code: reuse `research/train_cft/` with these changes).**
   - (a) **Split `split_v4r.parquet` (new; replaces the unknown `split.parquet`).** By score key, tautomers together.
     Hold-out set `HO_R` ≈ 22k structures (~8 %):
     - all ho2 keys (1,185: bench S1/S2/S3/S4 truths; FACT inventory);
     - all enveda-np-examples structures;
     - a stratified ~7 % random sample of the remaining structures, by "primary library" (the library holding most
       of the structure's spectra), with enveda-180 over-weighted ×2, because the test is all timsTOF.
     - Folds `F0..F4` inside `HO_R` (md5 of the key) for ranker CV.
     - INFERENCE for his fold names: `np` = np-examples, `nplib` = NP libraries, `syn` = drug/synthetic libraries,
       `plusk` = pluskal_ms2, `twin` = a class-1-like group. Keep his naming as a `stratum` column so per-stratum
       reports are comparable to his logs (`np ALL`, `nplib C2FAM`).
   - (b) **FPNet-R-A**: CFT `default` preset, trained on all structures not in `HO_R`. Loss as CFT (CE + 1.0 BCE +
     0.2 element). 40k steps, bs 256, EMA 0.999.
   - (c) **FPNet-R-B**: same with seed + 1 and `merge_p` 0.6 (our prvsiyan-style "merged" variant; FACT inventory).
   - (d) **FPNet-F-A** (full, deployment variant = full1 analogue): no hold-out.
   - (e) Optional **FPNet-F-B**.
   - Input differences to accept (they are not his, but they are ours and proven):
     - 128 peaks instead of 160;
     - precursor + 1.5 Da;
     - an instrument token (FACT `train_cft_report.md` §2).
     - The engine's item builder must then call **our** peak prep. Offline equals online, so that is fine.
   - Leakage precautions:
     - (i) `HO_R` structures, all their tautomer keys and all their spectra are excluded from R-A/R-B training **and**
       from the R-model validation split;
     - (ii) PubChem decoys are allowed (structure only);
     - (iii) never evaluate F models on bench S1/S2 or `HO_R`: they have seen them.
   - Compute (FACT anchor: CFT-ho2 took 423 min on Akriti's account; the prvsiyan trainer takes ~4.2 h at 40k steps):
     **~7 T4-h per model** → 14 T4-h for R-A + R-B, plus 7–14 for F.
   - Disk: 205 MB per checkpoint.
4. **Substitute.**
   - `models/cft_ho2_akriti/cft_ho2.pt` is architecturally an FPNet-A twin. FACT: pool MRR 0.820, PubChem-val MRR
     0.846, top-1 0.776.
   - Its hold-out is only 1,185 keys, too few for ranker simulation (~1.2k queries vs his ~18–24k groups, FACT
     `fragnet.py:74`).
   - **Use it now** as the engine bank for engine smoke tests and for the 1,185-query pilot simulation. Replace it with
     R-A for the real ranker.
   - Cost: 0 now, 7 T4-h later.
   - The prvsiyan 6,930-bit nets (`models/fp_*`) can serve as a second, *different* model view (B slot) if the pool
     carries both layouts. Not recommended: two bit layouts double the pool.

### 3.8 M4 — `fe_D.pt` (DreamsFP)

1. **What and how used (FACT).**
   - DreaMS backbone port (`dreams_model.py:302–386`):
     - 7 layers, d1024, 8 heads, Fourier m/z features 11,994 → 5-layer MLP → 980, plus ff_peak 44;
     - Graphormer key bias; "faithful" padding semantics.
   - Plus zero-initialised conditioning: adduct, polarity, CE MLP added to the precursor token.
   - Plus FP head 2048 → 2048 → 10,226 and element head (`dreams_model.py:494–578`).
   - Input: top-100 peaks, linear relative intensity, negative-mode m/z + 2 proton masses (`fp_to_dreams_spec`,
     444–463).
   - About 121.6 M parameters, fp32: 486 MB (my count matches).
   - Checkpoint: `arch='dreams_fp'`, `fcfg`, `bcfg`, `model` (589–601).
   - Inference precision: fp16 with fp32 attention and Fourier path on T4 (`fpnet6.py:426–446`).
   - Used **only** by `model_views`: `fzD_*`, `fzABD_*`, `rank_spread`, `rank_worst` (`model_views.py:20–32`).
2. **How produced: partially present.**
   - Present: architecture, SSL-checkpoint loader with argument assertions (391–420), `build_dreams_fp` (581–586),
     layer-wise LR groups `param_groups` (534–550), gradient checkpointing and Fourier checkpointing.
   - Absent: `work/analysis/dreams_ft/train_dreams_fp.py`.
   - FACT from names: run `s2/pilot_d1_step20k_ema` = 20k steps, EMA weights; `--attn_bf16`.
   - INFERENCE:
     - same data and loss as FPNet (`traindata.Data` batches, since `forward` accepts the FPNet collate signature,
       526–578);
     - hold-out folds excluded (it feeds hold-out-clean ranker rows).
3. **Rebuild.**
   - Inputs: DreaMS `ssl_model.ckpt` (HF `roman-bushuiev/DreaMS`, MIT; pre-trained on GeMS/MassIVE only, FACT
     `dreams_model.py:10`) plus our L1 / pool / `HO_R`.
   - Code: our own port of the backbone from the **official MIT DreaMS repo** (not his port) + our FP head + the CFT
     loss.
   - Training: 20k steps, layer-wise LR decay, EMA, bf16.
   - **Run on the RTX 5050 laptop** (Blackwell, bf16, 8 GB, gradient checkpointing). The T4 has no bf16, and the
     Fourier phases reach 3e7 rad (FACT `fpnet6.py:426–429`), which fp16 cannot hold.
   - Compute: HYPOTHESIS 12–20 GPU-h on the 5050, or ~15–25 T4-h with fp32 Fourier/attention.
   - Disk: 0.5 GB for the checkpoint plus 1–2 GB for the SSL checkpoint.
   - Leakage: train on non-`HO_R` structures only.
4. **Substitute.**
   - None trained. A cheaper stand-in for the "third view": FPNet-R-B, or the prvsiyan 6,930-bit net.
   - Without D, `model_views` loses 5 of its 7 kept columns (`fzD_*`, `fzABD_*` become A+B only).
   - Cost of dropping D: unknown, probably ≤ 0.002 (HYPOTHESIS; v4d "DreamsFP engine" was reverted by its author).

### 3.9 R1 — `ranker_v4b.pkl`

1. **What and how used (FACT).**
   - Pickle dict `features` (160 names), `params`, `rounds` (500), `boosters` (4 LightGBM model strings) — the
     `RankerBag.save` format (`ranker.py:60–64`; MANIFEST).
   - Inference: `X` (94 engine) ‖ `F` (66 families) reordered by name; the mean of the 4 boosters' predictions
     (v4l cell 3).
   - Defaults in code (`ranker.py:31–35`): lambdarank, lr 0.03, 63 leaves, min 40 per leaf, feature fraction 0.7,
     bagging 0.8/1, L2 1, truncation 30, label_gain [0, 1].
2. **How produced: partially present.**
   - Present: `RankerBag.fit` (seeds 0–3), `cv_report` (structure-grouped 5-fold, MRR per `(regime, fold)`),
     `mrr_at`.
   - Absent: the row simulation ("sim1x", `simulate_v1ctx.py`) and `build_v4v1.py`.
   - What the code still proves about the simulation:
     - FACT: `V1FE.run(spectra, target, **sim)` forwards `exclude`, `exclude_sid`, `exclude_lib`, `drop_pid` to
       `Engine.run` (`v1engine.py:144–148`, `engine.py:236–252`). So training rows were made by the same engine
       with masks.
     - FACT: `ctxlib.view_from_record` rebuilds queries from **library spectra** (`spectra_from_library(res.L, qidx)`,
       `ctxlib.py:525`). Simulated query spectra are train spectra.
     - FACT: regime names `c1`, `c2`, `c2L`, `C2FAM`, `C3FAM` and strata `np`, `nplib` in the evaluation logs.
     - FACT: about 18,465 groups in one evaluation, 23,748 queries in `sim10_s2` (`fragnet.py:61–74`).
   - Section 4 gives the reconstructed recipe.
3. **Rebuild.**
   - Section 4 (simulation) → rows → `RankerBag` with defaults, 500 rounds, 4 seeds.
   - First on 86 engine features (MVR), later + families.
   - Compute:
     - simulation ~20k queries × ~2–4 s = 15–25 CPU-h per pass (2–3 h on 8c, RAM-bound to ~4–6 processes on
       15.6 GB; or 3–4 Kaggle CPU sessions);
     - FPNet logits on CPU at ~0.1 s per query, or batched on GPU;
     - LightGBM ~10–20 min × 4 seeds. Disk: rows ~2–4 GB (float32, ~150 candidates × 160 features × 20k).
4. **Substitute.** Our megayak/pv rankers (E1) were trained on other engines' features. Not transferable. Rebuild.

### 3.10 D1 / D2 — derivation priors and parity fixture

1. **What and how used (FACT).**
   - `prior_tables.pkl` = dict:
     - `glob` (global found rate);
     - `tf` {transform → (s, n)};
     - `env` {r ∈ 1..3 → {(transform, Morgan env id at radius r) → (s, n)}};
     - `meta` (`derivation.py:273–282`).
   - `p_transform = (s + 2·glob)/(n + 2)`. `p_site` backs off r = 1 → 2 → 3 while n ≥ 20 (`MIN_N`, `A = 2`)
     (284–297).
   - `Priors.group` enumerates all products of a 1–2-step combo on the parent (32 first-step, 16 second-step,
     24 intermediates; heavy-atom check), scores `gen_site_logfr`, and computes sibling ranks (299–342).
   - Env ids = **unfolded** Morgan radius-3 sparse-count bit ids of the reaction-site atom (`atom_envs`, 212–228).
   - The family computes 36 columns (`NAMES`, 366–370). The ranker keeps 19 (MANIFEST).
   - D2: 200 cases `(parent smiles, combo, product) → features + full group`. `self_test` must return 0 mismatches,
     or the RDKit build differs (613–626; v4l cell 3 asserts it).
2. **How produced: absent (`res_derivation.py`).** Docstring FACT (`derivation.py:151–155`):
   - "tables built from random COCONUT parents with every non-train-fold key removed from parents and from 'found'
     → holdout-clean; the target is pool existence of the product".
   - INFERENCE, procedure:
     - sample N COCONUT parents not in hold-out;
     - for each of the 71 `derive6` transforms, `enumerate_products` with the site atom;
     - standardise the product;
     - found = 1 if its score key is in (pool keys − hold-out keys);
     - accumulate per transform and per (transform, env_r) → (found, total);
     - `glob` = overall found rate.
   - N is unknown. HYPOTHESIS: 30–100k parents.
3. **Rebuild.**
   - Our `vrb/derive_priors.py` with our own transform grammar (start from the 46-transform v1 list plus the 25 v6.3
     additions, written as our own SMARTS).
   - Inputs: COCONUT + our pool keys + `HO_R`.
   - Precaution: drop `HO_R` keys from the parents and from the "found" set.
   - Compute: products ~ parents × 71 × ≤ 32. Score key 10–15 ms each, so **~10–30 CPU-h** (the biggest family cost).
     Use a Kaggle CPU kernel or the night PC.
   - Our own D2: 200 random cases. Re-run on Kaggle with RDKit 2026.03.3.
   - **Pin RDKit 2026.03.3 everywhere**, because env ids and reaction order are version-dependent (FACT `derivation.py:107–109`).
4. **Substitute.** None (`research/c3gen/` generates derivatives but has no site priors, INFERENCE). Value: his log
   shows overall +0.008 and class 3 +0.07–0.10 offline (FACT `derivation.py:92–103`); the LB effect is unknown.

### 3.11 G1–G3 — frag2 tables

1. **What and how used (FACT).**
   - `caps.txt` = the 22 capability names, asserted equal to `fragcaps.CAPS` (`frag2.py:119–120`).
   - `pool_cap.npy` uint8 (n_pool, 22) = substructure match counts per capability SMARTS (`fragcaps.caps_of_smiles`;
     table `CAP_SMARTS`).
   - `pool_f1_{off,mass}` = depth-1 fragment CSR, the same function as P6 with `max_depth = 1`, float32
     (`frag2.py:405–411`).
   - The family computes 31 columns. The ranker keeps 13 (`RECOMMENDED`, `frag2.py:113–114`). Depth-2 fragments come
     from P6.
2. **How produced: present as functions.** `caps_of_smiles`, `frag_masses_fast`. The driver `res_frag2.py` is absent
   but trivial.
3. **Rebuild.**
   - `vrb/fams/frag2_tables.py`. Our own capability SMARTS list, with the same chemistry ideas (sugars, acyls,
     OMe/NMe, sulfate/phosphate) written by us.
   - Cost: ~1–2 CPU-h. Disk ~200 MB.
4. **Substitute.** None. Cheap.

### 3.12 S* — analog_struct tables

1. **What and how used (FACT).**
   - Per pool row (by pid) and per train structure (by sid):
     - `hash` int64 (n, 8) = element-graph core hash, generic core hash, up to 6 ring-system hashes (blake2b 62-bit);
     - `num` int16 (n, 8) = ring count, ring systems, core atoms, ring N/O/S, largest ring system, heavy atoms;
     - `fvec` int16 (n, 10) = element counts;
     - `elsmi` / `coresmi` = packed canonical element-graph SMILES and core SMILES (`analog_struct.py:106–108`,
       245–305, 349–366).
   - Used against the top 15 analogs. 33 columns are computed; the ranker keeps 13 (docstring line 79–80).
2. **How produced: present as functions** (`graph_all`, `formula_vec`, `pack_strings`). The driver is absent. FACT
   cost: train 94 s, pool 239 s (`analog_struct.py:51`).
3. **Rebuild.** Our own element-graph code (2-core, Tarjan bridges, ring systems, canonical subgraph SMILES). Cost
   < 1 CPU-h. Disk ~150 MB. The bit-exact hash only has to match **our** online code.
4. **Substitute.** None. Cheap.

### 3.13 N1–N4 — fragnet tables and models

1. **What and how used (FACT).**
   - **N1, `rec_*`**: style-invariant records (element, canonical mobile-H count, atom kind; bonds with ring type) per
     pool row (`mol_record` 136–266, `pack_record` 275–285).
   - **N2, `stageA.npz`**: per chain (`A` = mode, ft → c1 → c2; `H` = mode, ft → size → frac) and level: sorted keys
     plus an LLR table (keys × 6 match classes), α = 80 Dirichlet back-off (`fit_chain` 504–529, `CHAINS` 744–751,
     `STAGEA_CHAINS` 769).
   - **N3, `stageB.txt`**: LightGBM lambdarank over 31 stage-B features (`BNAMES` 771). The file header lists the
     same 31 names, 300 trees.
   - Inference:
     - top 400 candidates by engine f·z;
     - enumerate annotated fragments;
     - match against peaks (6 match classes);
     - stage-A aggregates per spectrum;
     - stage-B score per spectrum;
     - 14 output columns (`compute` 864–916).
   - **N4** (`sets.pkl`, `countsA.npz`, `featB.npz`) are training intermediates and not loaded.
2. **How produced: partially present.**
   - Present: `fit_chain`, `model_from_counts`, `save_stageA`, `stageB_features`, `agg_cands`.
   - Absent: `res_fragnet.py --step sets / countA / fitB`.
   - Procedure, FACT from the log (`fragnet.py:29–47`):
     - 32,000 **train-fold** structures × 1 library spectrum (NP libraries ×2, timsTOF ×2 sampling weight);
     - decoys: same formula ≤ 15 + 15, other formula ±10 ppm ≤ 3 + 3; source 60 % train / 40 % COCONUT, reweighted
       50/50;
     - parts A 16,144 / B 12,751 / V 3,105;
     - stage A = class counts on part A;
     - stage B = LightGBM (15 leaves, min leaf 200, lr 0.03, 300 rounds) on part B with within-candidate-group
       relevance;
     - V within-formula MRR 0.7712.
3. **Rebuild.**
   - `vrb/fams/fragnet_train.py`. Queries from **non-`HO_R`** structures only, so stage B is hold-out-clean for ranker
     rows.
   - Cost: records ~0.9 ms per molecule ≈ 11 min; sets + counts + fit ≈ 2–4 CPU-h. Disk ~170 MB.
4. **Substitute.** None. Our CFT/E1 have no learned fragment scorer.

### 3.14 X1 — PubChem tier and the gate

1. **What and how used (FACT, v4l cell 1).**
   - Mass-sorted `pc_mass` (float64), byte offsets `pc_off`, an ASCII SMILES buffer. 105.9 M rows (FACT in
     `v4n_method_spec.md` §5).
   - Pass 1: the ECFP4 part of z on the 4,096-bit ECFP4 block. Top `PC_N1` = 5000.
   - Pass 2: full fingerprint, f·z.
   - Skip pool keys. Keep the top 25.
   - `best_pool_fz` uses the same z.
   - Gate: `LIB_TAU = 0.9`, `REL_TH = 600`; slots [2, 4, 6, 8, 10] or [4, 8, 12, 16, 20] (v4l cell 6).
   - z = mean of v3 `fpnet_0` + `fpnet_1` (M5).
2. **How produced: absent.** INFERENCE: a PubChem CID-SMILES filter (CHNOPS + halogens, neutral, 150–1250 Da), stereo
   stripped, mass-sorted.
3. **Rebuild.**
   - Use **our** `external/pubchem/pubchem_rows.parquet`: 100,955,275 rows (ik, cid, smiles, mass), mass-sorted
     (FACT inventory). Convert it to the three-array layout (~7 GB) on a Kaggle CPU kernel.
   - **Re-tune `REL_TH`.** It is in raw logit units of his models (INFERENCE). Recipe: simulate PubChem-only truths
     (`HO_R` queries whose key is removed from the pool, truth in the PubChem window) and sweep the threshold and slot
     tables under several class mixes. "Positive under all mixtures" is his stated criterion (FACT, v3 header).
   - Compute: ~2 CPU-h of sweep after the simulation; PubChem fingerprints on the fly cost ~1 ms per structure
     (E6: 2.68 M structures in 2,452 s, FACT).
4. **Substitute.**
   - Our E6 PubChem channel (append-only, fp@zlog, 5 ppm) and our popularity arrays `pubchem_rows_pop.parquet`
     (n_sid, n_pmid) and `results/kaggle_e4/pop_lookup/` (91.9 % pool coverage).
   - Recommended: keep our channel and swap only its z for the R/F FPNets. Cost about 1 day.

### 3.15 X2 / X3 — ICEBERG and GLACIER

1. **What and how used (FACT, v4l cell 5).**
   - Input: candidates of the top-60 list in formula groups, plus PubChem lists when `lib_max < 0.9`.
   - Molecules ordered by ascending `lib_max`. Budgets 2700 s / 2400 s.
   - Re-rank `z(ranker) + 1.0·z(ICE) + 1.0·z(GL)` inside formula groups, slot-preserving.
   - The PubChem list is re-ranked too; the gate keeps the original top f·z.
2. **How produced.** His packages wrap MassSpecGym checkpoints (MIT). Not read by us (FACT
   `forward_model_runner.md:23`).
3. **Rebuild.** Already done: `research/forward_model/ice_runner.py` with `external/forward_model/ckpt/{iceberg,glacier}`.
   Cost ~26 ms per candidate on a Kaggle GPU (FACT inventory).
4. **Substitute.**
   - **Ours (cost 0).** Note: our bench prefers smaller weights (λ_ice 0.25 / λ_gl 0.1: +0.042 mean S1/S2 vs λ 1/1:
     S1 −0.057; FACT `results/bench/fm/analysis.txt`). His LB says 0.5 → 1.0 was +0.007.
   - Decide λ on our bench **with the new base list**, then confirm on the LB.

### 3.16 X4 — RDKit

- FACT: the v4 notebooks install the competition's RDKit wheel (2026.03.3). Our public dataset
  `shishiradhikari11/casmi-rdkit2026-cp313` exists (decision_log DEC-013).
- The forward models need 2025.3.6 in a separate process (FACT, sovereign markdown).
- Every table (P*, D1, S*, N1) must be built with 2026.03.3, because hashes, keys and env ids are version-dependent.

---

## 4. The ranker simulation (the central missing piece)

**What the code fixes (FACT).**
- The `Engine.run` masks (`engine.py:236–252`):
  - `exclude`: a boolean mask over library spectra, applied to direct search (`library.py:57–67`) and analog
    representatives (`engine.py:149–150`);
  - `exclude_sid` + `exclude_lib`: hides the analog representative of (sid, lib) or of sid entirely (143–148);
  - `drop_pid`: removes the truth from the pool window, but **lets generation produce it**. The window keys passed as
    `exclude_keys` omit `drop_pid` (248–251).
- Representatives are chosen on the **unmasked** library and then filtered (`_reps`, 114–124). So hiding the
  representative removes that (sid, polarity, lib) group from the analog channel even if other spectra of the group
  stay visible.
- Learned components on his side were all fit on **train folds**: FPNet A/B (hold-out folds,
  `traindata.py:11, 41–43`), fragnet ("train-fold structures"), derivation priors ("every non-train-fold key
  removed"). Evaluation is reported by fold (`np`, `nplib`) and regime.
- **INFERENCE, strongly supported:** simulated queries are hold-out-fold structures, featurised with hold-out-clean
  models.

**Our recipe (INFERENCE / HYPOTHESIS where marked).** For each structure s in `HO_R` with library spectra, build up to
three queries:

| Regime | Query spectra | `exclude` | `exclude_sid` / `exclude_lib` | `drop_pid` | Eligible if |
|---|---|---|---|---|---|
| c1 (class 1) | all spectra of s in one library ℓ (prefer enveda-180), both polarities, subsampled to the test count distribution (1–16, median 3; FACT intel §3.5) | every spectrum of (s, ℓ) | s / ℓ | −1 | s has spectra in ≥ 1 other library |
| c2 (class 2) | same | every spectrum of s | s / −1 | −1 | always |
| c3 (class 3) | same | every spectrum of s | s / −1 | pid(s) | always (rows without a positive are kept only for evaluation) |
| c2L (HYPOTHESIS) | as c2, but s is a COCONUT-overlap key whose pool row uses COCONUT-style SMILES | as c2 | as c2 | −1 | key in both sources |

The c2L row is a guess at his `c2L`. Its purpose is to stress the provenance gap; it becomes moot if 3.0's tautomer
canonicalisation is adopted.

**Label.** `y = 1` iff the candidate's score key equals the truth's score key. This includes a generated candidate
that hits the truth in c3.

**Weights.**
- Per-regime weights to a target class mix. HYPOTHESIS: c1 0.20 / c2 0.55 / c3 0.25 (our class-share notes: 17–30 %
  class 1, `competition_intel.md:612`).
- Within a regime, each structure counts once. Down-weight multi-query structures.
- His `C2FAM_wt` shows he used weights (FACT name only).

**Leakage checklist.**
1. FPNet-R-A, R-B and D never saw s (structure and tautomers).
2. The fragnet stage B and the derivation priors are fit without `HO_R`.
3. In c2/c3, no spectrum of s is visible, not even through a tautomer sid. Mask every sid that shares s's score key.
4. Analog representatives of s are hidden (`exclude_sid`).
5. Ranker CV folds are grouped by score key (`cv_report` groups by `sid`, FACT `ranker.py:80–84`). Use the key, so
   tautomer sids fall into the same fold.
6. Query-level constants (`n_query`, `q_npeaks`) must follow the **test** distribution. Check this with V5.

**Volume.** ~22k `HO_R` structures → ~45–55k queries over three regimes. His evaluation used ~18–24k groups
(FACT). Start with 20k.

---

## 5. Dependency DAG and build order

```
train.parquet ─┬─> L1 spec cache (runtime + offline)
               └─> P2 train_structs ──┐
COCONUT ──────────────────────────────┼─> P4 pool_meta ─> P1 bits ─> P5 pool_fp, P3 train_fp
                                      │        └─> P6 frags, G2/G3 frag2, S* analog_struct, N1 fragnet rec
split_v4r (HO_R) <── P2 + L1 meta     │
L1 + P4/P5 + HO_R ──> M R-A, R-B (GPU) ; M F-A(/F-B) (GPU) ; M4 D (laptop, + DreaMS SSL)
COCONUT + P4 keys + HO_R ──> D1 priors (CPU) ──> D2 fixture
L1 + P2/P4 + N1 + HO_R ──> N2 stageA ──> N3 stageB
engine code + L1 + P* + R-A/R-B + generation ──> SIM rows v0 (86 feats) ──> ranker v0
SIM rows v0 + families(G*, S*, N*, D1, D/A/B views) ──> rows v1 (160) ──> ranker v1
PubChem (ours) + R-A/R-B z + SIM ──> gate REL_TH
final notebook = engine + ranker + [families] + our ICE/GL + our PubChem/popularity + RRF(E6)
```

**Build order (critical path in bold).**
1. **Chemistry core + P2 + split_v4r** (day 1–2).
2. **P4/P5/P3** (day 2–3) → **start the R-A / R-B GPU runs immediately** (day 3).
3. In parallel, P6 + L1 builder + engine re-implementation + generation (days 3–8). Parity oracle V3.
4. **Simulation driver + ranker v0** (days 8–12), first with `cft_ho2` (pilot), then with R-A as soon as it exists.
5. **LB submission 1:** engine + ranker v0 + our post-steps (day ~13).
6. Families tables G*, S*, N*, D1 (CPU, days 8–16) and D on the laptop (days 5–12) → rows v1 → ranker v1 → bench
   A/B → **LB submission 2**.
7. Gate re-tune, λ re-tune, RRF with E6, popularity → submissions 3–5.

---

## 6. Minimal viable rebuild (MVR)

**Contribution estimate from the LB history** (FACT values; their attribution is INFERENCE).

| Component | LB step | Delta | Already owned by us? |
|---|---|---|---|
| v1 engine (L, A, M, F, S + generation) + FPNet A + ranker | — | **0.354 base** | no: rebuild |
| gated PubChem | 0.354 → 0.358 | +0.004 | yes (E6 channel) |
| fe_v4 families + DreamsFP + new ranker | 0.358 → 0.354 | −0.004 (noise) | no |
| ICEBERG | 0.358 → 0.366 | +0.008 | yes |
| GLACIER | 0.366 → 0.373 | +0.007 | yes |
| λ 0.5 → 1.0 | 0.373 → 0.380 | +0.007 | parameter |
| full1 engine bank | 0.380 → 0.384 | +0.004 (v4m alone −0.002) | one more GPU run |
| pool popularity re-rank μ 0.15 (sovereign) | — | +0.010 | yes (E4 arrays) |
| PubChem popularity (sovereign) | — | +0.004 | yes |
| RRF fusion with a second engine | 0.384 → 0.399 | +0.015 | yes (E6 = second engine) |

The base engine is ~90 % of v4n's score, and the families show no LB evidence. So the MVR is:

1. **Engine v1-equivalent** (our code): L, A, M, F, S, priors and interactions. 86 live features: drop the 8 `xs_*`.
   Generation with our 46-transform grammar.
2. **FPNet-R-A** as the engine bank (CFT recipe, `HO_R` hold-out). `cft_ho2` until it exists.
3. **Ranker v0**: 4-seed lambdarank on simulation rows (section 4).
4. **Post-steps we own:** our ICE/GL runner, our PubChem channel (z = R-A + R-B), pool and PubChem popularity.
5. **RRF fusion with E6** on the metric key (`1/(3 + r_v4r) + 0.6/(3 + r_E6)`, constants FACT from the fork), then
   the forward-model re-rank on the union.

**Expected (HYPOTHESIS).**
- Engine + ranker alone: 0.33–0.35, because a re-implementation loses some of his tuning.
- With post-steps: 0.36–0.38.
- Fused with E6: 0.37–0.40.
- This does **not** reach the 0.448 prize line by itself. It gives a second strong, eligible engine. The remaining gap
  must come from new ideas on top (section 10).

**MVR cost.** ~3 weeks of work. GPU: 14 T4-h (R-A, R-B) + 7 (F-A) + ~12 for submission runs ≈ 35–45 T4-h. CPU ≈ 40–60
CPU-h. Disk ≈ 6 GB of new files on D: (37 GB free, FACT).

**Full rebuild beyond the MVR** (fe_v4 + D + full models): +2 weeks; +15–25 GPU-h (D) + 7 (F-B); +30–40 CPU-h
(D1 dominates); +1 GB disk.

---

## 7. Parallel job plan (3 Kaggle accounts × ~30 T4-h/week, 2 laptops, this PC)

| Slot | Week 1 | Week 2 | Week 3 |
|---|---|---|---|
| Kaggle U (user) | day 3: R-A (7 h); then F-A (7 h) | engine + ranker v0 submission run (~5 h) | submissions v1 / fusion (2 × 5 h) |
| Kaggle K (akriti, run by us) | day 3: R-B (7 h, seed + 1, `merge_p` 0.6) | F-B (7 h, optional) | spare / reruns |
| Kaggle 3rd account | day 1: 2k-step timing run of the CFT preset on the new pool (0.5 h); CPU kernels: pool fingerprints (P5), PubChem three-array table, D1 priors (CPU sessions do not use GPU quota) | CPU kernels: simulation shards (4 sessions × 4 cores) | bench / gate sweeps |
| Laptop RTX 5050 | day 5+: DreamsFP fine-tune (12–20 h wall, bf16) | — | — |
| Laptop RTX 3050 | P6 fragments + G*/S*/N1 tables (CPU); FPNet logits for simulation shards (GPU) | fragnet stage A/B | — |
| This PC (8c, CPU) | chemistry core, P2, split, P4 meta, engine code, parity tests | simulation (local shards), ranker training, bench | rows v1, ranker v1, λ/gate tuning |

---

## 8. (moved) Leakage summary

See section 4 (simulation) and the per-artifact items 3.7(3), 3.8(3), 3.10(3) and 3.13(3). One rule covers them all:
**every learned component that produces a ranker feature is fit without `HO_R`; simulated queries come only from
`HO_R`.**

---

## 9. Validation checkpoints (local bench and parity)

| # | Checkpoint | Pass criterion |
|---|---|---|
| V0 | Chemistry: score key vs the metric notebook on 2,000 SMILES; neutral mass of the ten test adducts | 100 % equal keys; mass error < 1e-6 Da |
| V1 | Structure tables: P2 rows = 275,810; pool ≈ 712k (his 710,701); mass sorted; keys unique; ≥ 99 % of train keys in the pool; per-block kept-bit counts listed. Schema oracle: his `Pool('work')` / `Library('work')` readers load our files without error (local only) | all hold |
| V2 | L1: 2,539,608 spectra; 0 spectra with `sid = -1`; peak counts within 1 % of an independent recount | all hold |
| V3 | **Engine parity:** his `engine.py` (local oracle) vs ours, on our tables with the same FPNet (`cft_ho2` via an adapter), on 200 bench S2 queries | candidate sets identical; feature matrices equal to 1e-5 except order-dependent ties in `*_rank` |
| V4 | FPNet R-A / R-B: `window_mrr` per spectrum and merged on `HO_R` (his metric definition) | references: his A 0.934 on train structures / 0.581 on the np panel (FACT intel); our `cft_ho2` pool MRR 0.820. Target ≥ 0.80 on `HO_R` merged |
| V5 | Simulation: (a) per-regime MRR of ranker v0 in 5-fold CV — c1 ≥ 0.85, c2 0.60–0.73 (prvsiyan range, FACT intel), c3 > 0 only where generation finds the truth; (b) KS distance of `n_query`, `q_npeaks`, `lib_max`, `top_sim`, `n_cand` between simulated queries and the 400 visible test molecules; (c) provenance probe: for c2 truths, rank of train-style vs COCONUT-style decoys | (a) in range; (b) KS < 0.15; (c) no gap > 0.02 |
| V6 | Bench: S1, S2, S3, S3i, SV with our scorer (`bench_eval.engine_list` / `rank_in` / `rr`); E1 reference S1 0.905 / S2 0.695 / S3 0.136 / S3i 0.856 (FACT). Proxy LB ≈ 0.146·SV + 0.21·S1 + 0.11·PubChemOnly (DEC-008). **R models only** (F models have seen S1/S2) | v0 within 0.03 of E1 on S1/S2; fused list beats E6 on the proxy |
| V7 | Families: permutation invariance (shuffle candidates → identical features), finite values, timing vs his logs (frag2 30–45 ms, fragnet 73 ms mean, analog_struct 92 ms cold, derivation 50 ms per query, FACT docstrings); within-formula MRR of fragnet on our V split vs his 0.7712 (fr_int 0.6465, frag2 mix 0.7022) | all hold; fragnet ≥ 0.74 |
| V8 | Derivation self-test on Kaggle (our D2) | 0 mismatches |
| V9 | Kaggle runtime: full hidden-size run on a T4 | < 7.5 h including ICE/GL budgets |
| V10 | `eng_lists.json` parity between consecutive live kernels (lesson of DEC-013) | identical except intended changes |

---

## 10. Risks and unknowns (ranked)

1. **Simulation fidelity (high impact, high likelihood).** The ranker *is* the engine's score, and its training rows
   come from a simulation whose exact rules are absent. Unknowns: query construction, c1 masking granularity, class
   mix, weights. Public evidence shows CV-tuned rankers that did not transfer (0.335 → 0.321 / 0.330, intel §2.2).
   Mitigation: V5b/c, bench SV, and early LB submission 1.
2. **Prize gap (certain).** Even a perfect rebuild ≈ v4n 0.384, and the fused version is probably ≈ 0.40. The prize
   line is ≥ 0.448. The rebuild is necessary but not sufficient. Budget time for new levers after week 3.
3. **FPNet quality and train/serve mismatch.** Our R models may be weaker than his A on hold-out (target 0.58 on the
   np panel). The full model as engine bank changes the feature distribution under a ranker trained on R features.
   Mitigation: deploy R-A first; test F-A as a separate variant.
4. **Provenance leak via SMILES style.** Train-style truths vs COCONUT-style test truths. Mitigation: tautomer
   canonicalisation (3.0) plus probe V5c.
5. **RDKit version coupling.** Keys, hashes, env ids and reaction enumeration are version-dependent (FACT). The
   forward models need 2025.3.6. Mitigation: build everything on 2026.03.3 and run self-tests on Kaggle.
6. **Runtime.** Engine + families + 3 models + ICE/GL on 400+ molecules within 9 h. His v4n fits (FACT: about
   3.5–4.5 h for the fork). Our Python may be slower. Mitigation: numba kernels and V9.
7. **Class-3 generation value unknown.** No isolated LB number. Mitigation: ablation on bench S3/S4 (`queries_S4C3`).
8. **DreamsFP training feasibility.** No bf16 on T4. The 8 GB laptop needs gradient checkpointing, and the laptop is
   often unavailable (memory note). Mitigation: D is optional.
9. **RAM for simulation** on 15.6 GB: library + pool mmaps with 4–6 workers. Mitigation: Kaggle CPU kernels.
10. **Licence trail.** COCONUT (CC BY 4.0) and DreaMS (CC BY 4.0 for the Zenodo weights) need attribution. PubChem
    needs nothing. Our own code is MIT. Record every input in `fork_licence_rebuild_plan.md`.

# Making the 0.399 fork prize-eligible: licence audit and rebuild plan

Compiled 2026-10-01. Research only: nothing was submitted, pushed or changed in project code. Downloads were small files only (about 59 MB, in the session scratch directory).

**Labels.** FACT = read directly from the named source. INFERENCE = my reading or estimate. "LB" = public leaderboard as stated by the notebook author; the public-LB noise is about ±0.006–0.02 (FACT, `competition_intel.md` §2.1), so single-step deltas below that are not reliable.

**Sources used.**
- Kaggle CLI: `kaggle datasets metadata` / `files` for all 13 datasets; single-file downloads of README, LICENSE-NOTICE, MANIFEST and the `casmi` Python files.
- Fork source pulled with `kaggle kernels pull shishiradhikari11/casmi26-v4n-engine-fusion-union-lb-0-399` (17 cells).
- Upstream notebooks pulled: `prvsiyan/analog-propagation-casmi-2026-baseline`, `megayak/casmi26-two-rankers-one-engine`, `megayak/casmi26-one-engine-nine-scores`, `ahmedberatozer/casmi26-v2-baseline`.
- Kaggle MCP: competition pages (rules), discussion topics 743236 and 742193 read in full.
- `research/analysis/competition_intel.md` (§2 ablations, §3 host rulings).
- Web: ChEBI "About" page, LIPID MAPS download page (via search), ms-pred README.

---

## 1. Verdict

**Feasible only partly.**

- The licence obstacle is mostly labels, not source data. Every source behind the stack is one the host has allowed (train.parquet, enveda-np-examples, COCONUT, PubChem, MassSpecGym-trained ICEBERG/GLACIER) or is CC BY 4.0 at source (ChEBI, LIPID MAPS). Nothing in the stack is NIST- or METLIN-derived as far as the shipped notices say.
- The one real blocker is that Ahmed Berat Özer's five model/pool datasets are **his** artifacts under **his** stated terms ("non-commercial, with attribution"). We cannot relabel another person's weights as MIT. Either he relabels them, or the host says rule 2.5.a.3 covers them, or we retrain.
- Retraining the Ahmed part is the expensive piece: the model classes and engine are public, but the training loops, the simulation that produced the ranker rows, the split file and the fe_v4 table builders are not. A rebuild is a re-implementation, and it will not reproduce 0.384 exactly.
- The second engine (prvsiyan/megayak two-ranker) is already clean (CC0 + CC BY) once the ChEBI/LIPID MAPS file is rebuilt from source or switched off.
- Separate from licences: 0.399 is where ranks 15–40 sit; prizes go to the top 5, currently ≥ 0.421 (FACT, `competition_intel.md` §0). Eligibility of this stack only matters if we also add about +0.02 or more on top of it.

**Cheapest path that is definitely eligible:** the two-ranker engine with CC0 assets (stated LB 0.341 without BIO), plus ICEBERG and GLACIER re-scoring, plus a PubChem table we build ourselves from our own `external/pubchem` download. No Ahmed model or pool is in it. Its score is unknown; see §6.

---

## 2. The train.parquet "CC BY-NC components" claim

| # | Statement | Label | Source |
|---|---|---|---|
| 1 | The Rules page lists "DATA ACCESS AND USE: Competition Use and Non-Commercial & Academic Research (NC) - CC BY-NC 4.0", and rule 2.4.a.1 says "You may access and use the Competition Data for non-commercial purposes only". | FACT | competition `rules` page |
| 2 | Host (David Healey, 2026-09-25): "The Competition Data in train.parquet is a compilation of open-source databases, and so models trained on it can be open-sourced. Sorry for the licensing confusion on that; the **test** data when it is released will be restricted to noncommercial use." | FACT | D/743236, read in full |
| 3 | In the same reply, to a participant who uses private datasets (PubChem index, COCONUT table, a reference library built from train.parquet, model weights) plus rebuild scripts: "the artifacts you describe seem acceptable open-source-derived solutions." | FACT | D/743236 |
| 4 | Host (2026-09-25): models "trained on this competitions train.parquet, which is compiled from open-source data" are fine; "enveda-np-examples we neglected to assign a specific license to but should also be considered open source". | FACT | D/742193, read in full |
| 5 | Winner licence is "Open Source - MIT"; rule 2.5.a.3: "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s)." | FACT | competition `rules` page |
| 6 | Welcome post: "the winning solution cannot include any licenses that would otherwise prohibit commercial use or redistribution." | FACT | D/741359, as quoted in `competition_intel.md` §3.1 |
| 7 | Foundational rules: code shared publicly for the competition is "deemed to have [been] licensed ... under an Open Source Initiative-approved license ... that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code." | FACT | competition `foundational-rules` page |

**Reading.**
- INFERENCE: statements 2–4 make Ahmed's *reason* for the non-commercial label moot. The host has said that train.parquet-derived models and libraries can be open-sourced, and that np-examples (used by `fpnet_full1`) counts as open source. The Rules page text (statement 1) has not been edited, so the written rules and the host's forum statements still disagree; the forum statements are the later and more specific ones.
- INFERENCE: what is **not** resolved is the label itself. The weights and tables are Ahmed's work, published under terms he chose. The host's ruling says he *could* release them openly; it does not release them for him. Statement 7 covers code shared in notebooks and the forum; whether it reaches weights uploaded as a Kaggle dataset is not stated anywhere.
- INFERENCE: statements 5 and 6 pull in different directions for third-party weights with a non-commercial label. No host statement addresses public Kaggle datasets from other participants.

**Question to post to the host (exact text):**

> Several public notebooks rely on Kaggle datasets uploaded by other participants that contain model weights, candidate pools and feature tables derived only from train.parquet, enveda-np-examples and COCONUT. Some uploaders labelled these datasets "non-commercial, with attribution" because the Rules page lists the Competition Data as CC BY-NC 4.0 (for example ahmedberatozer/casmi26-v4b-models, casmi26-v3-models, casmi26-v2-pool, casmi26-fpnet-full1). You clarified in discussion 743236 that models trained on train.parquet can be open-sourced.
> 1. If a winning solution uses such third-party weights unchanged, is it prize-eligible, given that the uploader's own label says non-commercial? Or must the uploader relabel the dataset, or the winner retrain the models from train.parquet with their own code?
> 2. Does rule 2.5.a.3 (no open-source grant needed for input data or pretrained models with an incompatible licence) apply to such participant-uploaded weights?
> 3. Is the same answer true for a candidate table built only from ChEBI and LIPID MAPS (both CC BY 4.0 at source) that its uploader labelled CC BY-NC-SA 4.0?

**Message to Ahmed (cheaper than any rebuild):** ask him to relabel the five datasets (MIT, Apache-2.0 or CC BY 4.0), citing D/743236. If he does, the rebuild in §5 is unnecessary.

---

## 3. Dataset audit

Licence status: **(a)** real and inherent in the source data; **(b)** the uploader's label on something we can regenerate from allowed sources; **(c)** not a problem.

| Dataset | Kaggle licence (FACT, dataset metadata) | Contents (FACT, file listing) | Derived from | Status |
|---|---|---|---|---|
| `prvsiyan/casmi26-fp-models-v2` | CC0-1.0 | `fp_single_s2.pt`, `fp_merged_m1.pt` (144 MB each) | FACT: trained on train.parquet with the script published in his notebook ("needs a GPU and ~2 h") | **(c)** |
| `prvsiyan/casmi26-ranker-features` | CC0-1.0 | `rank_train.npz` (9.5 MB) | FACT: simulated ranker rows from train.parquet + pool (his notebook). Numeric features only (INFERENCE) | **(c)** |
| `megayak/casmi26-simulated-ranker-rows` | CC0-1.0 | 3 row files, 2 leak-free FPNets, README | FACT (README): 2,250 simulated molecules from 7 train libraries, pool = COCONUT + train; "No structures, SMILES or spectra are included; only numeric features." | **(c)** |
| `prvsiyan/coconut-casmi26-candidates` | CC BY 4.0 | `coco_fp.npy` (378 MB), `coco_mass.npy`, `coco_meta.pkl`, `fp_bits.npy` | FACT (his notebook): COCONUT 2.0 `coconut_csv-09-2026.zip`, 462,028 unique InChIKey14. Host: "COCONUT's own license is sufficient" (D/743234) | **(c)** attribution required |
| `prvsiyan/chebi-lipidmaps-casmi26` | **CC BY-NC-SA 4.0** | `bio_fp.npy` (54 MB), `bio_mass.npy`, `bio_meta.pkl` | ChEBI + LIPID MAPS structures (title; no description, no build script). FACT: ChEBI data is "available under the Creative Commons License (CC BY 4.0)" (ebi.ac.uk/chebi About page). FACT (search result citing lipidmaps.org/databases/lmsd/download): the LMSD SDF is offered as CC BY 4.0 — re-check on the download page before use, I could not open the terms page directly | **(b)** the NC-SA label is the uploader's; the file as published must not be used |
| `ahmedberatozer/casmi26-v2-pool` | other; "non-commercial, with attribution" | `pool_fp.npy` (909 MB), `pool_frag_mass.npy` (357 MB), `pool_meta.parquet`, `train_fp_sel.npy` (353 MB), `train_structs.parquet`, offsets | FACT (description): competition training data + COCONUT (CC BY 4.0) | **(b)** but see §2: his label, his artifact |
| `ahmedberatozer/casmi26-v3-models` | other; same note | `casmi` package (15 files), `fpnet_0.pt`, `fpnet_1.pt` (199 MB each), `ranker_0.pkl` | same | **(b)** |
| `ahmedberatozer/casmi26-v4b-models` | other; same note | `casmi` + `fe_v4` code, `fe_A.pt`, `fe_B.pt`, `fpnet_0.pt`, `fe_D.pt` (486 MB), `ranker_v4b.pkl`, about 560 MB of `fe_v4/res` tables, MANIFEST, LICENSE-NOTICE | same, plus DreaMS: FACT (LICENSE-NOTICE.txt) "fe_D.pt is derived (fine-tuned ...) from DreaMS ... weights MIT / CC-BY-4.0, code MIT"; FACT (`dreams_model.py` header) "pretrained on GeMS/MassIVE only" | **(b)** for his parts; DreaMS base **(c)** |
| `ahmedberatozer/casmi26-fpnet-full1` | other; "non-commercial, with attribution" | `fpnet_full1.pt` (199 MB) | FACT (description): "trained on ALL train folds incl. np-examples, same architecture as ... FPNet A (drop-in)" | **(b)** |
| `ahmedberatozer/casmi26-iceberg` | other | ms-pred inference subset, `ckpt/gen.pt`, `ckpt/inten.pt`, runner, shims, 5 wheels | FACT (LICENSE-NOTICE.txt): ms-pred commit 708148c2a8eb, MIT; "ICEBERG msg_all (MassSpecGym-trained) weights published with ms-pred"; wheels BSD-3 / Apache-2.0 / MIT; shim "own code". FACT (ms-pred README): NIST-trained weights exist but are given only on proof of a NIST licence; the MassSpecGym ones are the public download. Host: MassSpecGym models released permissively are "fair game" (D/742991) | **(c)** for the third-party parts. Ahmed's own runner/fuse/shim code carries no licence text (INFERENCE: small, rewritable in 1–2 days if needed) |
| `ahmedberatozer/casmi26-glacier` | other | ms-pred subset, `ckpt/glacier.pt` (61 MB), runner, shims | FACT (LICENSE-NOTICE.txt): GLACIER MassSpecGym checkpoint, sha256 recorded; "The download carries no separate license file; used under the ms-pred repository's MIT License". Host: "MassSpecGym is allowed, so is GLACIER" (D/744338) | **(c)**, same caveat for the runner code |
| `ahmedberatozer/casmi26-pubchem-tier` | other | `pc_smiles.npy` (5.5 GB), `pc_mass.npy`, `pc_off.npy` | FACT (description): PubChem CID-SMILES + CID-Mass, stereo stripped, CHNOPS + halogens, 150–1250 Da. Host: "PubChem structures for retrieval are acceptable" (D/741857) | **(c)** as data; rebuild ourselves anyway so the label question disappears (we already hold `external/pubchem/CID-SMILES.gz`) |
| `metric/rdkit-2026-3-3-wheel` | unknown (empty description) | 4 RDKit 2026.3.3 wheels | INFERENCE: `metric` is the account that publishes the competition metric notebook; RDKit itself is BSD-3-Clause (FACT for 2025.3.6 per Ahmed's notice; same project) | **(c)** |

Nothing in the list is status (a).

---

## 4. Which component uses which dataset

All FACT from the fork source unless marked (cell numbers refer to the pulled fork notebook).

| Component | Where | Reads |
|---|---|---|
| RDKit install | cell 1 | `rdkit-*.whl` from `metric/rdkit-2026-3-3-wheel` |
| **Second engine** (two-ranker + BIO + AFIX), subprocess | cell 5 | `fp_*.pt` (prvsiyan fp-models-v2: s2 + m1), `rank_train.npz` (prvsiyan ranker, 8 HGB fitted at run time), `sim_rank_rows_nofp.npz` (megayak ranker B, fitted at run time), `coco_*.npy/pkl`, `fp_bits.npy`, `bio_*.npy/pkl` (ChEBI + LIPID MAPS), train.parquet. Blend `w_pv = 0.88` |
| Our own engine (the third list we added) | cell 6 | our `v2_lists.py` dataset |
| **PubChem-only channel**, subprocess | cell 8 | `pc_*.npy` (pubchem-tier); `fpnet_*.pt` from v3-models (FPNet A + B); `pool_fp.npy`, `pool_meta.parquet`, `fp_bits.npy` (v2-pool); `casmi` code from v3-models |
| Spectrum cache | cell 9 | `casmi.build.build_spec_cache` on train.parquet at run time; pool files symlinked from v2-pool |
| **v1 engine** (library, analog, class-3 derivation, `EngineCfg(generate=True)`) | cell 10 | `fpnet_full1.pt` as the engine FP bank (falls back to `fpnet_0.pt`); `train_fp_sel.npy`, pool files |
| **fe_v4 families** (derivation, frag2, analog_struct, model_views, fragnet) | cell 10, MANIFEST | `fe_A.pt`, `fe_B.pt`, `fe_D.pt` (DreamsFP) and the `fe_v4/res` tables |
| **LightGBM ranker** | cell 10 | `ranker_v4b.pkl`: 160 features, 4 boosters, 500 rounds (MANIFEST) |
| **ICEBERG** on v4n top-60 + engine top-40 (+ our top-20) | cell 13 | casmi26-iceberg (`ice_runner.py`, `fuse.py`, checkpoints, RDKit 2025.3.6 wheel) |
| **GLACIER** on the same input | cell 13 | casmi26-glacier, reusing the iceberg wheels |
| Gated PubChem merge | cell 14 | no dataset |
| RRF fusion + forward re-rank | cell 16 | no dataset |

**Is training code public?**

| Artifact | Training / build code | Label |
|---|---|---|
| prvsiyan FPNet s2 / m1 | Yes. `train_fingerprint_model.py` is embedded in his notebook (AdamW, 63 same-window decoys, "~2 h" on a T4). megayak retrained the same architecture on a T4 (20k and 26k steps) | FACT |
| prvsiyan `rank_train.npz` | The builder `make_ranker3.py` is named in the notebook but not included. Not needed: the file is CC0 | FACT |
| megayak simulated rows | The row builder is not in the notebook (rows are loaded from the dataset). Not needed: CC0 | FACT |
| COCONUT candidate table | Recipe written out in prvsiyan's notebook ("reproducible in ~10 minutes") | FACT |
| ChEBI + LIPID MAPS table | No recipe or version published | FACT |
| Ahmed pool (`pool_fp`, `pool_frag_*`, `train_fp_sel`) | `casmi/chem.py`, `frag.py`, `pool.py` (readers) are shipped; the script that builds the arrays is not | FACT |
| Ahmed FPNet A / B / full1 | Model class (`fpnet.py`) and a batch sampler with negatives (`traindata.py`) are shipped. The training loop is not ("scripts/03" is referenced in `build.py`), and `traindata.py` needs `work/split.parquet`, which is not shipped | FACT |
| DreamsFP `fe_D.pt` | Backbone port shipped (`dreams_model.py`); the fine-tune script `work/analysis/dreams_ft/train_d...` is referenced, not shipped | FACT |
| `ranker_v4b.pkl` / `ranker_0.pkl` | `ranker.py` (LightGBM lambdarank bag + grouped CV) is shipped. The simulation that produces the rows ("sim1x") and `work/analysis/fe_v4/build_v4v1.py` are not | FACT |
| fe_v4 `res` tables (`prior_tables.pkl`, fragnet `stageA.npz` / `stageB.txt` / `featB.npz`, frag2 caps) | Consumers shipped, builders not | FACT |
| Ahmed's other public notebooks | Only inference notebooks v2–v4q; no training notebook (`kaggle kernels list --user ahmedberatozer`) | FACT |
| ICEBERG / GLACIER | No training needed: public MassSpecGym checkpoints | FACT |

---

## 5. The plan

Effort and compute are INFERENCE unless a source is given. "Score if dropped" uses the author-stated LB deltas in `competition_intel.md` §2.2; each is a single public-LB reading on about 130 molecules.

| Component | Dataset | Status | Replacement | Training code available? | Effort and compute | Score impact if simply dropped |
|---|---|---|---|---|---|---|
| RDKit wheel | metric/rdkit-2026-3-3-wheel | c | keep | n/a | none | n/a (required) |
| Engine FPNet pair (s2 + m1) | prvsiyan/casmi26-fp-models-v2 | c | keep | yes | none | FPNet channel is worth 0.266 → 0.299 → 0.335 in family A; do not drop |
| Engine ranker A rows | prvsiyan/casmi26-ranker-features | c | keep | not needed | none | — |
| Engine ranker B rows | megayak/casmi26-simulated-ranker-rows | c | keep | not needed | none | blend vs prvsiyan ranker alone: 0.333 → 0.337, inside noise |
| COCONUT candidates | prvsiyan/coconut-casmi26-candidates | c | keep, with attribution | recipe public | none | not droppable |
| ChEBI + LIPID MAPS candidates (BIO) | prvsiyan/chebi-lipidmaps-casmi26 | b | rebuild from ChEBI SDF + LMSD SDF with the same 6,930-bit fingerprint (`fp_bits.npy`), publish with CC BY attribution; or set BIO off | no recipe; simple to write | 0.5–1 day, CPU only, under 1 GB disk | engine alone 0.341 → 0.350 with BIO (seyit); 0.299 → 0.295 in prvsiyan's stack; effect inside the fused 0.399 stack: **unknown** |
| PubChem table | ahmedberatozer/casmi26-pubchem-tier | c | rebuild from our `external/pubchem/CID-SMILES.gz` into the same three arrays (mass-sorted, stereo stripped, element and mass filter) | format readable from cell 8 code | 0.5–1 day; output about 7.2 GB, so build on Kaggle or free local disk first (14 GB free is tight) | whole PubChem channel: 0.354 → 0.358; N1 1000 → 5000: no stated change (0.373) |
| ICEBERG re-scorer | ahmedberatozer/casmi26-iceberg | c | keep; optionally re-download the checkpoints from ms-pred ourselves and keep the MIT notice | no training | 0–2 days (only if we rewrite the runner/shim) | 0.358 → 0.366 |
| GLACIER re-scorer | ahmedberatozer/casmi26-glacier | c | same | no training | same | 0.366 → 0.373; with weights 0.5 → 1.0 a further 0.373 → 0.380 |
| Candidate pool + fragment arrays | ahmedberatozer/casmi26-v2-pool | b | rebuild from train.parquet + our COCONUT CSV with the shipped `casmi/chem.py` and `frag.py` | builder not public; must be written and checked against his arrays | 2–4 days; CPU hours; about 1.7 GB output | not droppable while the v1 engine is used |
| FPNet A and B (PubChem channel, fe_v4 views) | casmi26-v3-models, casmi26-v4b-models | b | retrain with our own loop around the shipped `FPNet` class and `traindata.Data`, with our own split | loop and split not public | 2–3 days of code; GPU time **unknown** (the model is larger than prvsiyan's 2 h one: 199 MB vs 144 MB); budget 5–10 GPU h per model | FPNet channel itself is not droppable. Second model: +0.019 for the pair in family A; unknown here |
| FPNet full1 (engine bank) | casmi26-fpnet-full1 | b | the same retrain without holding out np-examples | same | one more run, same cost | v4l 0.380 vs v4n 0.384; v4m (full1 at λ 0.5) 0.378. Inside noise |
| DreamsFP `fe_D` | casmi26-v4b-models | b (base c) | drop together with fe_v4 | fine-tune script not public | not recommended; unknown GPU time | v4d (DreamsFP engine) was reverted by its author, no number |
| fe_v4 families + tables + `ranker_v4b` | casmi26-v4b-models | b | **drop**: fall back to the v3.6 engine (94 features) and train our own LightGBM ranker on it | simulation driver and table builders not public | dropping: 1–2 days to rewire. Rebuilding fe_v4 instead: 1–2 weeks, not recommended | v3.6 → v4b was 0.358 → 0.354 (NEG), so dropping fe_v4 costs nothing measurable |
| v1-engine ranker (`ranker_0.pkl`, 94 features) | casmi26-v3-models | b | write our own class-1 / class-2 / class-3 simulation over the engine, then fit with the shipped `ranker.py` | simulation not public | 4–7 days of code plus CPU days of simulation on 8 cores. This is the highest-risk step: the gate thresholds and class-3 behaviour were tuned on his simulation | not droppable |
| Engine code (`casmi`, `fe_v4`, runners) | inside the Ahmed datasets | b | reuse with attribution if he or the host confirms an open licence; otherwise this is a rewrite | n/a | **unknown**; a clean-room rewrite of `engine.py`, `derive.py`, `frag.py`, `library.py` is 1–2 weeks | not droppable for the v4n list |

### Recommended order of work

1. **Day 1, no compute.** Post the host question in §2 and ask Ahmed to relabel. Both answers decide whether steps 5–7 are needed at all.
2. **Days 1–2.** Rebuild the PubChem table from our own download and the ChEBI + LIPID MAPS table from the source SDFs. Publish both as our own datasets with source attribution. This removes two of the three non-Ahmed labels at low cost.
3. **Days 2–4.** Build the *definitely eligible* notebook: two-ranker engine (CC0 assets, rebuilt BIO) + our own engine as a second list + ICEBERG and GLACIER on the union in same-formula groups. Submit it once to get its real LB. This is the fallback whatever Ahmed and the host say.
4. **Decision point.** If Ahmed relabels or the host confirms eligibility of his datasets, stop here and spend the remaining weeks on score, not on rebuilds.
5. **Otherwise, weeks 2–3.** Rebuild the pool, then retrain FPNet A, B and full1 on Kaggle GPU (roughly one week of quota at 30 h).
6. **Weeks 3–5.** Write the simulation, train the 94-feature ranker, re-tune the PubChem gate, and compare against the unmodified fork on the LB with at least two submissions per variant.
7. **Do not rebuild** fe_v4, DreamsFP or `ranker_v4b`. The only LB evidence for them is negative.

### Is it realistic for one person on this hardware?

- INFERENCE: steps 1–3 are realistic: about one week, CPU only, small disk footprint.
- INFERENCE: steps 5–6 are realistic in calendar time (deadline 2026-12-14, about ten weeks away) but not in outcome. Three to five weeks of work would buy a *re-implementation* that will land somewhere near 0.36–0.39, with the difference from 0.399 inside or near the public-LB noise, and with no gain toward the 0.42+ needed for a prize.
- INFERENCE: local disk (about 14 GB free) rules out building the 7.2 GB PubChem table and the 1.7 GB pool locally at the same time; do those builds on Kaggle.
- INFERENCE: GPU quota is sufficient for three FPNet runs if each fits in one 9-hour session; per-model time on a T4 is not published for Ahmed's architecture and must be measured with a short run first.

---

## 6. What the definitely-eligible path is expected to score

- FACT: two-ranker engine W088 alone is quoted at 0.341; with BIO, 0.350 (seyit notebook header).
- FACT: on the Ahmed base, ICEBERG + GLACIER + weight 1.0 moved 0.358 → 0.380.
- INFERENCE: nobody has published the two-ranker engine with ICEBERG/GLACIER and without the v4n list. The forward-model gain may transfer only partly, because the two-ranker engine has no PubChem channel and no class-3 generation. A range of 0.35–0.37 is a guess, not a measurement. It needs one submission.
- FACT: our own pipeline scores 0.251 and added nothing as a third list in the fork (0.399 → 0.399, task brief).

---

## 7. Open questions

1. Will the host accept third-party participant datasets labelled "non-commercial" whose only inputs are train.parquet, np-examples and COCONUT? (Question text in §2.)
2. Will Ahmed relabel his five datasets?
3. Does the "deemed open-source" clause for publicly shared competition code extend to code and weights shared as Kaggle datasets rather than notebooks?
4. Which ChEBI and LMSD releases did prvsiyan use, and does a rebuild from current releases change the engine's 0.350?
5. The LIPID MAPS licence was confirmed only through a search result pointing at the LMSD download page; read that page directly before publishing a rebuilt table.
6. T4 training time for Ahmed's 8-layer FPNet: unpublished.
7. Whether rebuilt models would hold the 0.384 base score: cannot be known without doing it, and the public LB cannot resolve differences under about 0.02.
8. The Rules page still says CC BY-NC 4.0 for Competition Data and has not been edited to match the host's forum statement.

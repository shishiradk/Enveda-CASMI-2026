# Methods survey for CASMI 2026 (Kaggle): what to add to our pipeline

> **Correction 2026-10-04 (FACT, Zenodo API `records/19685145`):** the FRIGID pretrained checkpoints are licensed CC BY-NC 4.0, not Apache-2.0. Any row below that uses FRIGID weights is not prize-eligible; only the published recipe may be re-implemented and retrained on eligible data. The GitHub code licence was not verified.

Date: 2026-09-30. Our current public LB is 0.237 (kNN spectrum to Morgan fingerprint, cosine against COCONUT and train candidates).
Legend: **[M]** = measured fact with a source. **[I]** = my inference or estimate.

---

## 0. The main finding: we are well below the public state of the art, and that code is public

- **[M]** The public LB top is 0.464 (team "pikachu"). About 15 teams are at or above 0.399 (leaderboard pulled 2026-09-30 via the Kaggle API).
- **[M]** The public notebook *CASMI26 | v4n + Engine Fusion + Union* scores **0.399**. It runs offline on a T4 in about 3.5 to 4.5 h, and its code is Apache-2.0. https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399
  Its public version history:

  | Version | Change | Public LB |
  |---|---|---|
  | v4b | FPNet engine bank | 0.354 |
  | v3.6 | + moderate generation + gated PubChem channel | 0.358 |
  | v4f/v4g | + ICEBERG isomer re-scoring | 0.366 |
  | v4h | + GLACIER | (not reported) |
  | v4l | ICE/GL weight 1.0 | 0.380 |
  | v4n | + fpnet_full1 + PubChem depth 5000 | 0.384 |
  | fusion | + second retrieval engine, reciprocal-rank fusion | 0.382 → 0.386 on the v4g base |
  | union | + forward models on the fused union | 0.399 |

  Sources: https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference and https://www.kaggle.com/code/ahmedberatozer/casmi26-v4h-inference
- **[M]** The components are public Kaggle datasets:
  - `prvsiyan/casmi26-fp-models-v2`, `prvsiyan/coconut-casmi26-candidates`, `prvsiyan/chebi-lipidmaps-casmi26`
  - `ahmedberatozer/casmi26-iceberg` (ICEBERG 2.1, MassSpecGym msg_all checkpoint, MIT)
  - `ahmedberatozer/casmi26-glacier` (GLACIER MassSpecGym checkpoint, MIT)
  - `ahmedberatozer/casmi26-pubchem-tier`, `ahmedberatozer/casmi26-fpnet-full1`
- **[M]** Analog-propagation baseline (four channels plus FPNet plus HistGBM): 0.335 to 0.339. https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline and discussion 741745.

**[I]** Before building any of the model families below, we should rebase onto (or port) this stack. The gap of about 0.15 LB is larger than any single-model gain reported anywhere in the literature for this setting.

---

## 1. Rules that constrain the choice of models (host answers on the forum)

- **[M]** Anything trained on NIST is disqualified. The same goes for weights derived from METLIN or vendor data, even if the weights are MIT-licensed (discussion 744470, host David Healey, 2026-09-29).
- **[M]** MassSpecGym and models trained on it under permissive licences are allowed ("fair game", discussion 742991).
- **[M]** The stock CFM-ID 4 models are allowed, even though they were trained on METLIN (discussion 743774).
- **[M]** PubChem structures may be used for retrieval (discussion 741857).
- **[M]** Training on external or cloud compute is allowed (discussion 741876).
- **[I]** Avoid NPLIB1/CANOPUS checkpoints (MIST, JESTR, FRIGID and ms-pred NPLIB1 variants). A participant thread (742275) describes NPLIB1 as partly NIST-derived, so they carry eligibility risk. Use the MassSpecGym checkpoints only.

---

## 2. Spectrum → fingerprint / embedding models, and forward-model rerankers

### 2.1 Reliable MassSpecGym retrieval numbers

The "MassSpecGym in the Wild" audit ([arXiv 2606.19624](https://arxiv.org/abs/2606.19624), Table 3) found that most headline numbers are inflated. Its corrected numbers are the ones to trust.

**[M]** Formula-known ("bonus") track, 256 same-formula candidates, hit@1 / @5 / @20:

| Method | @1 | @5 | @20 | Notes |
|---|---|---|---|---|
| Random | 3.1 | 11.4 | 27.7 | |
| Fingerprint FFN | 5.1 | 14.7 | 32.0 | |
| Nearest neighbour | 9.6 | 22.3 | 39.9 | ≈ our kNN approach |
| MIST (MassSpecGym) | 9.6 | 22.1 | 41.1 | |
| JESTR | 11.8 | 33.5 | 61.5 | |
| MVP | 14.0 | 36.9 | 68.1 | |
| FLARE | 22.7 | 50.0 | 75.2 | 43.2 @1 on the mass track |
| ICEBERG forward-model retrieval | 36.2 | 60.7 | 78.8 | |
| FRIGID generative retrieval | 45.3 | 58.3 | 69.5 | |
| **MIST vFRIGID** (trained only on ICEBERG-simulated PubChem spectra, leak-free) | **53.8** | 65.3 | 74.7 | |

**[M]** Same paper, Table 1: MIST with no augmentation reaches 9.57% @1 and a spectrum→fingerprint Tanimoto of 0.262. Trained on ICEBERG (MassSpecGym-train) simulations of PubChem structures, it reaches **53.76% @1 and Tanimoto 0.581**. That is the leak-free setting.

**[M]** Inflated results, not usable:
- SpecBridge / DreaMS+ChemBERTa: 82 to 85% @1 falls to 6.7% after uniform RDKit canonicalisation.
- GLMR / ChemFormer two-stage: 64 to 68% falls to 7.6%.
- MIST vDiffMS checkpoint: batched-mask bug and suspected contamination.
- The DiffMS / MolForge numbers built on that checkpoint.

**[M]** A spectrum-blind "PubChem default order" baseline scores **49.98% @1**. This is the size of the popularity prior (see §4).

**[M]** GLACIER paper ([arXiv 2606.29161](https://arxiv.org/abs/2606.29161)), MassSpecGym same-formula track, up to 256 candidates, top-1 retrieval:
- ICEBERG 2.0: 44.4%
- GLACIER: 49.9%
- Mass track: 64.0% → 70.0%

Inference speed on an A5000: 1.3 ms per spectrum for GLACIER versus 10.8 ms for ICEBERG 2.0. ICEBERG 2.1 is about 3× faster than 2.0 according to the ms-pred README.

### 2.2 Per-method assessment

| Method | Weights / licence | Inputs | Runtime | Evidence in this competition | Verdict |
|---|---|---|---|---|---|
| **GLACIER** (ms-pred) | MassSpecGym checkpoint via Dropbox link in https://github.com/coleygroup/ms-pred (MIT); already packaged as Kaggle dataset `ahmedberatozer/casmi26-glacier` | SMILES, adduct, collision energy; all adduct types | ~1.3 ms/spectrum on A5000 [M]; T4 perhaps 3–5 ms [I] | [M] part of the 0.366 → 0.380 → 0.399 chain | **Use** |
| **ICEBERG 2.1** | MassSpecGym msg_all checkpoint (MIT); Kaggle `ahmedberatozer/casmi26-iceberg` | Same as GLACIER | Public notebook budgets 2700–5400 s for top-60 same-formula candidates × 400 molecules on a T4 | [M] ICEBERG re-scoring is in the 0.358 → 0.366 step. [M] A forum user measured "+0.0015 as a learned feature in the >100-candidate stratum" (742055), so the gain comes from z-score fusion inside same-formula groups rather than as a raw GBDT feature. Also: forward weight 2.0 everywhere → 0.352 (it destroys library matches) | **Use**, weight ≈1 inside same-formula groups only |
| MARASON / SCARF (ms-pred) | Code MIT; no public MassSpecGym checkpoints listed [M] | | | | Skip |
| **MIST encoder from FRIGID** | Zenodo record 19685145 (https://github.com/coleygroup/FRIGID). **Weights CC BY-NC 4.0 (corrected 2026-10-04): NOT prize-eligible** | Needs a chemical formula; works with the candidate's formula per candidate | Small transformer, fast on GPU | [M] 53.8% @1 on the MassSpecGym formula track. [M] One forum user reports "a little traction" (742055) | **Skip: ineligible weights** (was "try as a feature") |
| JESTR (HassounLab/JESTR1) | NPLIB1 and MassSpecGym weights | | | [M] 11.8% @1 | Weaker than MIST vFRIGID; skip |
| MVP | HassounLab | | | [M] 14.0% @1 | Skip |
| FLARE | bioRxiv 2026.01.27.702086; code with MassSpecGym | | | [M] 22.7% @1 (formula) | Weaker than the forward models; low priority |
| DreaMS | Code MIT; weights on HF `roman-bushuiev/DreaMS` / Zenodo 10997887. GitHub says MIT, but one Zenodo record is CC BY-NC-ND, so check before depending on it | Precursor m/z and peaks; 1024-d embedding | | [M] Forum (742055): AUC 0.88 at separating same-molecule from same-formula spectra in general, but on real blocked isomer pairs the truth was closer in only 20 of 42 cases, which is below chance | Useful only for analog search, not isomers |
| MS2DeepScore 2 (dual-mode) | Zenodo 16962640, CC BY 4.0; `pip install ms2deepscore` | Spectrum-to-spectrum, pos/neg | Fast on CPU | None | Optional analog-channel similarity. [I] Expected ≤ +0.005 |
| Spec2Vec | Apache-2.0, trainable on our 2.5M spectra | | | None | Superseded by MS2DeepScore; skip |
| CFM-ID 4 | LGPL; stock models allowed | | Slow: seconds per molecule on CPU [I] | Can't score thousands of candidates at runtime. Precomputed in-silico libraries (ISDB for COCONUT/LOTUS, CC BY) are an allowed alternative | Low priority |
| CSI:FingerID / SIRIUS structure search, MSNovelist | Fingerprint prediction runs on Bright Giant web services | | | | **Not usable offline** [I] |
| SpecBridge, GLMR | | | | Headline numbers are artifacts | Skip |

---

## 3. Formula-level constraints

- **[M]** Forum (743254, rank-5 team): when the truth is in the list but not first, the winner has the same formula about 98% of the time. On timsTOF, 99% of precursor masses fall within about 5 ppm, and widening the window from 8.5 to 10 ppm barely changes candidate counts.
- **[M]** A second team (742055): perfect same-formula separation would close 95.1% of the gap. Recall plus formula combined is worth only 0.013. A formula/subformula gate got weight 0.0 in 10 of 10 splits.
- Available tools:
  - **msbuddy** (Apache-2.0, `pip install msbuddy`, successor to BUDDY). https://github.com/Philipbear/msbuddy
  - **MIST-CF** (MIT). https://github.com/samgoldman97/mist-cf. A data-safe MassSpecGym version ships with MassSpecGym v1.5.
  - SIRIUS fragmentation trees run locally [I]. SIRIUS itself is not needed.
- **[M]** Accuracy (arXiv 2601.00941, MoNA): SIRIUS top-1 0.19 / top-5 0.44. MIST-CF pretrained top-1 0.40, and 0.74 after in-domain retraining. SIRIUS is best on rare adducts (water losses); retrained MIST-CF collapses there.
- **[I]** Formula prediction is not a lever for ranking. Its only use for us is pruning PubChem windows that contain several formulas (S, P or halogen alternatives), plus a formula-score feature for the ranker. Expected gain < +0.005. Effort 3–5 h. **Low priority.**

---

## 4. Very large PubChem pools: priors and gating

**[M] Evidence against blind expansion:**
- Blind PubChem expansion took the LB from 0.335 to 0.205 (prvsiyan, cited in 743254 and 742055).
- Another team saw +0.002 as a full source and +0.001 in tail slots (742088).
- The rank-5 team finds PubChem ∪ COCONUT net-positive, but only together with a strong same-formula ranker (743254 reply).

**[M] What works in the public 0.38+ notebooks: a gated PubChem-only channel.**
1. Score the PubChem ±10 ppm window with an FPNet ensemble in two passes (pass 1: ECFP4 partial dot product; keep the top N1 = 5000; pass 2: full fingerprint).
2. Keep PubChem-only structures that are not already in the pool.
3. Insert them only into fixed slots, and only when the best library match is below 0.9:
   - "aggressive" slots [2, 4, 6, 8, 10] when the PubChem top score exceeds the best pool score by a margin;
   - "gentle" slots [4, 8, 12, 16, 20] otherwise.
4. The PubChem-only truths contributed about 0.037 MRR in the probe (v4h notebook docstring).

**[M] Popularity priors:**
- The spectrum-blind PubChem default order reaches 50% @1 on MassSpecGym (2606.19624, Table 3).
- MetFrag 2.2 on 473 spectra with PubChem candidates: 30 top-1 hits from fragmentation alone, 336 (71%) with reference counts plus retention time. Source: [Ruttkies et al. 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9).
- Sources of counts:
  - PubChem FTP `Extras/CID-PMID.gz` and `CID-Patent.gz` for all CIDs;
  - PubChemLite (about 470k CIDs, with LiteratureCount / PatentCount / AnnoTypeCount; CC BY 4.0; Zenodo 13989963, monthly versions).

**[M] Caveat:** the forum reports that curated-database membership flags looked great on public holdouts and hurt on the LB, because holdout molecules are over-represented in curated databases (743254 #5). The test is Enveda natural products with no public spectra (Class 2).

**[I] Recommendation:**
- Add log(1 + PMID count), log(1 + patent count), log CID, "in COCONUT" and an NP-likeness score (RDKit `NP_Score`, public model) as ranker features.
- Train them only on the gated PubChem channel, with a Class-2-style holdout (InChIKey14 removed from every library) that is resampled to the test's 3.03 spectra per molecule.
- Expect a gain of +0.005 to +0.02, with a high risk of LB/CV disagreement; the prior is much weaker for novel NPs than for MassSpecGym/CASMI standards.
- Effort 6–10 h. Precomputing PubChem fingerprints for the 150–1200 Da window is the costly part (tens of millions of structures, done offline).

---

## 5. De novo generation for Class 3

- **[M]** MassSpecGym de novo top-1, formula-known:

  | Method | Top-1 |
  |---|---|
  | MADGEN | 1.31% |
  | DiffMS | 2.30% |
  | MBGen | 7.58% |
  | FRIGID-base / FRIGID scaled | 16.1% / 18.3% |

  FRIGID runtime: 6.6 s per spectrum on an RTX 6000 Ada. FRIGID's MassSpecGym checkpoints on Zenodo 19685145 are **CC BY-NC 4.0** (corrected 2026-10-04), so they are not prize-eligible.
- **[M]** The audit warns that de novo decoder gains are largely memorisation of near-analogs (Tanimoto ≥ 0.7) seen in pretraining.
- **[M]** On NPLIB1 (2601.00941), DiffMS pretrained with MIST-CF formulas reached only 0.052 top-1.
- **[M]** Forum experience:
  - "Adding generated candidates to the final list has not yet paid off on the leaderboard, even when only inserted for molecules predicted to be novel", and deletion-based novelty simulations overstate reach about 3× (743254 #6).
  - Derivative candidates (±O, ±CH₂, ±hexose, acetyl, ±2H on strong analogs): 0.337 → 0.335 (742055).
  - The public 0.358 → 0.399 chain keeps "moderate generation" (`EngineCfg(generate=True)`) with class-3 derivation priors, but its marginal gain is not reported separately.
- **[I] Expected value for us:**
  - Class 3 is about 39% of molecules by the forum's estimate (742088 comment).
  - Assume a realistic exact-connectivity top-25 hit rate of 3–8% on NPs and mean reciprocal rank of about 0.3 when hit. That gives 0.39 × 0.05 × 0.3 ≈ **+0.006**, minus the losses from displaced retrieval slots.
  - A FRIGID MassSpecGym run on about 150 low-confidence molecules on a T4 would take roughly 1–2 h.
  - **Low priority.** Only do this after sections 6.1–6.4, and only in tail slots (16–25) gated by low retrieval confidence.

---

## 6. Ranked recommendations for our pipeline

| # | Action | Expected LB gain for us | Effort | Kaggle-offline fit | Links / licence |
|---|---|---|---|---|---|
| 1 | **Rebase onto the public stack:** analog propagation + FPNet 6930-bit logits (score = f·z) + LightGBM ranker + gated PubChem channel. At minimum, reproduce the 0.384 v4n notebook as our baseline and keep our own components as extra features or engines | **+0.10 to +0.16** [M: public notebooks score 0.335–0.399 versus our 0.237] | 4–8 h to fork, verify and submit; 15–25 h to port into our repo with leak-free CV | T4, 3.5–4.5 h, fits | v4n and fusion notebooks (Apache-2.0); prvsiyan datasets |
| 2 | **Forward-model isomer re-scoring:** GLACIER + ICEBERG 2.1 MassSpecGym checkpoints, score = z(ranker) + 1.0·z(ICEBERG) + 1.0·z(GLACIER), inside same-formula groups of the top 60. [I] Improvement: precompute GLACIER spectra at 3 collision energies for the whole pool (≈1M structures × 2 polarities ≈ 1–3 GPU-h offline) so the forward score becomes a first-stage feature, with no runtime budget cap | [M] ≈ +0.02 to +0.03 inside the public chain (0.358 → 0.380). [I] Precomputation adds another +0.005 to +0.015 | 6–12 h (the wrappers already exist as Kaggle datasets) | GPU; fits | ms-pred (MIT); Kaggle `casmi26-iceberg`, `casmi26-glacier` |
| 3 | **Rank fusion of two independent engines** (weighted reciprocal-rank fusion, α = 0.6, K = 3), then forward-score the union. Our kNN/COCONUT engine can be the second list | [M] +0.016 (0.366 → 0.382) and +0.013 (0.386 → 0.399) | 2–4 h | CPU, trivial | Fusion notebook |
| 4 | **Train spectrum→fingerprint on simulated isomers** (the "MIST vFRIGID" recipe; forum 744343 "train a SMILES→MS/MS predictor too"): GLACIER/ICEBERG-simulate spectra for COCONUT + PubChem-window structures at the test's collision energies, mix with the 2.5M real spectra, and add a listwise loss over same-formula candidates. Separately, drop the FRIGID MassSpecGym MIST encoder in as a feature | [M] 9.6% → 53.8% @1 on the MassSpecGym formula track. [I] Our FPNet already trains on 2.5M spectra, so expect +0.01 to +0.04. [M] Forum warns that fingerprint models which improve offline often do nothing on the LB when validation leaks | 20–40 h + 1–2 GPU-days (external compute allowed) | Inference is cheap | FRIGID weights CC BY-NC 4.0 (Zenodo 19685145): recipe only, retrain ourselves |
| 5 | **Leak-free validation discipline** (it gates everything above): hold out by InChIKey14 across **all** libraries (require zero self-matches), resample holdout molecules to 3.03 spectra each, build a same-formula isomer panel, average GBDT over ≥ 4 seeds, and deduplicate by tautomer-canonical InChIKey14 (8% of the truths had tautomer duplicates). Never pad with junk SMILES (a padded submission scored 0.000) | Indirect. Prevents shipping changes below the ~0.016 LB noise floor | 4–6 h | n/a | Forum 742055, 742088, 743254 |
| 6 | PubChem popularity / NP priors as ranker features on the gated channel (§4) | [I] +0.005 to +0.02, high variance | 6–10 h | Offline precompute | PubChem FTP, PubChemLite (CC BY 4.0) |
| 7 | Formula scoring (msbuddy / MIST-CF) for PubChem pruning | [I] < +0.005 | 3–5 h | CPU | msbuddy (Apache-2.0), mist-cf (MIT) |
| 8 | De novo (FRIGID MassSpecGym) in gated tail slots | [I] 0 to +0.01; forum says it has not paid off yet | 10–20 h | T4, 1–2 h | FRIGID weights CC BY-NC 4.0: **ineligible** |

**Do not use:**
- NIST-, METLIN- or vendor-trained weights (disqualification).
- NPLIB1 checkpoints (eligibility risk).
- The MIST vDiffMS checkpoint (bug and suspected contamination).
- SpecBridge / GLMR-style claims (artifacts).
- DreaMS for isomer separation (below chance on blocked pairs).
- CSI:FingerID / MSNovelist (need the web service).
- Blind PubChem expansion without gating.

**Why the top of the LB is at 0.43 to 0.46:**
- **[I]** Likely a stronger same-formula ranker and/or better PubChem gating, since the class-3 ceiling makes pure retrieval top out near 0.61 (742088 comment).
- **[M]** Undisclosed. The host confirmed that non-prize teams need not disclose.

---

## 7. Past CASMI strategies

- **[M]** CASMI 2016: MetFrag with statistical fragment learning plus metadata beat the winner CSI:IOKR. Database boosting was needed to reach 93% ("Comprehensive comparison ... Database boosting is needed to achieve 93% accuracy", Blaženović et al. 2017).
- **[M]** CASMI 2022 (Fiehn lab results page): Dührkop (SIRIUS team) led formulas at 94% and adducts at 93%. The best 2D-structure rate was only 38% (Nikolic, 21 of 55). Top winners: Adamo Young, Adriano Rutz, Armando Alcazar. https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results
- **[I]** Lesson: metadata priors plus an in-silico fragmentation or forward model for the final isomer decision. This is the same structure as recommendations 2 and 6.

---

## Sources
- Kaggle discussions: 741745, 743254, 742055, 742088, 743974, 744343, 744470, 742991, 742275, 743774, 741857, 741876. All at https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>
- Notebooks: seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399; ahmedberatozer/casmi26-v4h-inference; prvsiyan/analog-propagation-casmi-2026-baseline
- MassSpecGym in the Wild: https://arxiv.org/abs/2606.19624
- GLACIER: https://arxiv.org/abs/2606.29161
- ms-pred: https://github.com/coleygroup/ms-pred
- FRIGID: https://github.com/coleygroup/FRIGID, https://arxiv.org/abs/2604.16648
- SpecBridge: https://arxiv.org/abs/2601.17204
- MassSpecGym: https://github.com/pluskal-lab/MassSpecGym
- DreaMS: https://github.com/pluskal-lab/DreaMS
- JESTR: https://github.com/HassounLab/JESTR1
- FLARE: https://www.biorxiv.org/content/10.64898/2026.01.27.702086v1
- MS2DeepScore: https://zenodo.org/records/16962640
- msbuddy: https://github.com/Philipbear/msbuddy
- MIST-CF: https://github.com/samgoldman97/mist-cf
- Formula/structure comparison: https://arxiv.org/abs/2601.00941
- De novo numbers: https://arxiv.org/abs/2602.01643 (MBGen, DiffMS, MADGEN)
- MetFrag relaunched: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9
- PubChemLite: https://zenodo.org/records/13989963
- CASMI 2022: https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results

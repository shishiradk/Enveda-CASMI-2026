Design: `fpga`, a fingerprint-guided GA for Class-3 candidates (research/c3gen/fpga.py, not built yet)

**Measured inputs to this design** (FACT unless marked; read-only probe at C:\Users\LENOVO\AppData\Local\Temp\claude\d--Enveda-CASMI-2026\b55166b4-4275-49ce-9a40-a16cd42aff87\scratchpad\c3gen_probe.py; truth removed from analogs and window)
- **S1 (250 NP timsTOF), Tanimoto (ECFP4) from truth to its best analog among the top 50:** median 0.62. Shares at or above 0.7 / 0.8 / 0.9: 38% / 21% / 8%. Top-1 analog alone: median 0.28.
- **Mass of that best analog:** only 10% are within 0.01 Da of the target. Median |Δm| is 26.5 Da, p90 146 Da. 43% share the truth's Murcko scaffold.
- **zlog inside the ±5 ppm window (S1, about 60 real database decoys):** full-fingerprint MRR 0.467, top-1 0.30. Where neighbours with Tanimoto ≥ 0.5 exist (177 of 250), on average 28% of them score above the truth.
- **S3 gives top-1 0.84 for the same measure.** INFERENCE: S3 is leaky or easy for the ho1 nets, so never tune on S3.
- **Cost per NP molecule (median 25 heavy atoms):**
  - full 6930-bit fingerprint: 16.5 ms, of which RDKitFP is 5.1 ms and MACCS 8.4 ms;
  - Morgan r2+r3 together: 0.7 ms;
  - tautomer InChIKey14: **138 ms**.
- **Morgan-only zlog (4743 of the 6930 bits) against full:** Spearman 0.77; in-window MRR 0.36 vs 0.47.
- **Consequence:** the GA must use a Morgan-only surrogate score, run the full fingerprint only on finalists, and never compute tautomer keys inside `generate`. Even 200 keys would take 28 s, over the 20 s budget.

**Algorithm** (`generate(ctx, k=200, forbidden)`; deterministic, RNG seeded by crc32(mid))
1. **Parse the target.**
   - M = ctx.target, tolerance 10 ppm.
   - Enumerate CHNOPS+Cl/Br formulas within 10 ppm, filtered by RDBE ≥ 0 and the Kind–Fiehn rules.
   - Rank them by sub-formula explanation of the MS2 peaks and by agreement with analog and window formulas. Keep the top 3 as "formula targets" for the mass-repair moves.
2. **Seeds.**
   - Window structures: the 40 best by fpscore.
   - Analogs: the 30 best by spectral similarity, plus the 30 best by zlog.
   - Drop any seed whose key is in `forbidden`. This filters inputs only, never outputs.
   - Keep about 60–100 seeds, deduplicated by canonical SMILES with stereo stripped.
3. **Mass repair of each seed.**
   - Δ = M − m(seed).
   - Look up single moves or pairs from the transform table (A5) and the MMP swap table (A2) whose dmass is within tolerance of Δ.
   - Apply each at every matching site, capped at 40 products per seed.
   - This step alone is the v4n "derivation" idea re-implemented clean-room on a larger rule set. It forms generation 0.
4. **GA loop.**
   - Population 200, 400 offspring per generation, 15 generations: about 6k evaluations.
   - Operators (weights):
     - (a) 30%: biotransform add or remove, using about 60 SMARTS we write ourselves (hydroxylation, O/N/C-methylation, glycosylation hexose/pentose/deoxyhexose/glucuronide, acylation, prenylation, sulfation, reduction/oxidation, decarboxylation and so on).
     - (b) 25%: MMP substituent swap from A2, weighted by its corpus count.
     - (c) 20%: mass-neutral positional moves, i.e. moving OH, OMe, Me, a sugar or a prenyl to another site of the same atom type, or a double-bond or ring-position shift.
     - (d) 10%: GB-GA-style atom/bond edits (aromatic C↔N, O↔NH, ring open/close), re-implemented.
     - (e) 15%: BRICS cut-and-join crossover between two parents, using A1 fragments.
   - A child that misses the mass gets one compensating repair move (step 3). If it is still off-mass it is dropped.
   - Validity gate: the child must sanitize, be neutral, be a single fragment, and contain only atom environments in A3 (seen at least twice in train+COCONUT). Otherwise it is rejected.
   - A tabu set of canonical SMILES prevents re-evaluating the same molecule.
   - All on-mass valid children go into an archive.
5. **Surrogate fitness.**
   - F = zM + 0.5·A + 0.3·NP + 0.2·E, where:
     - zM = Morgan-part fp@zlog, divided by √nbits and z-scored against the seed and window distribution (the engine's own normalisation);
     - A = Σ over seeds of spectral_sim × Tanimoto(child, seed), the analog-propagation term (the strongest single channel in the intel: within-isomer MRR 0.54 vs 0.49 for f·z);
     - NP = A4 log-ratio prior;
     - E = −(number of steps) + Σ log rule-frequency.
   - The weights are starting values. Fit them by a small grid on a scaffold-split half of C3NP and report on the other half. Tuning inflated gains about 2.8× for NeckBeard.
6. **Finalists.**
   - Take the archive's top 400 by F and compute the full 6930-bit fingerprint (about 6.6 s).
   - Re-score with the full zM.
   - Add MetFrag-lite explained intensity (1–2 bond cuts) for the top 250 (about 1.5 s).
   - Diversity cap: at most 6 outputs per (seed, site-family).
   - Return the top k as (smiles with stereo stripped, score, "seed=<ik14>|ops=…").
   - Estimated total is about 15 s per molecule on one core (INFERENCE; to be timed on the CLI smoke run).
7. **CLI evaluation.** Hits are found cheaply: compute the tautomer key only for outputs whose ECFP Tanimoto to the truth is ≥ 0.5. Nothing else can share the connectivity.

**Data assets** (results/c3gen/assets/; built offline; at most 3 workers, each under 2.5 GB; DuckDB memory_limit '2GB'; streamed in chunks of 20k molecules)
- **A1, BRICS fragment library.**
  - Source: about 0.77M structures (smiles from `pool_smiles.txt` / train_classes and coco_meta).
  - Fragments with count ≥ 5 and ≤ 15 heavy atoms.
  - Time: about 15–25 min.
- **A2, MMP swap table.**
  - RDKit rdMMPA single cuts with core ≥ 60% of heavy atoms, giving a (core, substituent) parquet.
  - DuckDB self-join on core, capped at 50 substituents per core. Keep pairs with count ≥ 3 and substituents ≤ 12 heavy atoms, indexed by dmass at 1 mDa.
  - Output: about 1–3M rules, about 100 MB.
  - Time: about 45–90 min. INFERENCE: rdMMPA speed on NPs is untimed, so time 5k molecules first.
- **A3, atom-environment whitelist.** Morgan r1 sparse ids with count ≥ 2. About 10 min, under 100 MB.
- **A4, NP prior.**
  - Log-ratio of fragment frequencies, COCONUT vs a 1M PubChem random sample.
  - The sample is pulled through DuckDB with projection on the smiles column only, never pandas.
  - About 30 min.
- **A5, transform SMARTS** (written clean-room) plus a frequency prior counted from A2.
- **Total:** about 2–2.5 h wall time and under 2 GB per process.

**Expected performance** (INFERENCE, anchored on the figures above)
- **Reachability.**
  - On S1, 21% of truths have an analog at Tanimoto ≥ 0.8 (about 1–2 edits) and 38% at ≥ 0.7 (about 2–4 edits).
  - True Class-3 molecules are novel by definition, so they are probably farther from the library than S1 truths.
  - Estimated share of truths inside the GA archive at all: 12–25%.
- **Ranking.**
  - Generated decoys are closer to the truth than database decoys. Positional isomers share most ECFP bits; same-formula isomers are 95% of the remaining gap in public analyses.
  - The truth's rank among thousands of archive members will therefore be worse than its in-window MRR of 0.47 among about 60 database decoys.
- **Published reference points:**
  - MassSpecGym de novo, formula known, top-1: MADGEN 1.3%, DiffMS 2.3%, MBGen 7.6%, FRIGID 16–18% (NC, so ineligible).
  - The audit attributes much of the de novo gain to memorising near-analogs (Tanimoto ≥ 0.7). That is the same regime this GA targets explicitly.
  - v4n derivation: 13% of class-2 truths can be derived (≤ 2 edits, ≤ 6 parents). It gave +0.07–0.10 MRR in the author's class-3 simulation.
  - Udam reports that deletion-based simulations overstate reach about 3×, and that generated candidates have "not yet paid off on the LB".
  - Derivative enumeration elsewhere moved 0.337 → 0.335.
- **Estimate on NP Class-3** (C3NP):
  - recall@200: 6–15%, central 10%;
  - hit@25: 3–7%;
  - standalone MRR@25: 0.015–0.04.
- **LB effect when appended:** with a Class-3 share of about 0.5, and generator rank g landing near final rank 3g under a three-way interleave, the gain is about **+0.003 to +0.012**.
- That is inside the public-LB noise of ±0.006–0.02. **This GA alone will not close the 0.072 gap to 0.432.** Its value is as one component combined with better same-formula ranking.

**Eligibility of inputs**
- Allowed:
  - train.parquet (host: models and derivatives OK);
  - COCONUT (host OK);
  - PubChem sample (host OK);
  - our ho1/full FPNets (trained on train.parquet, ours, MIT);
  - RDKit, including BRICS, rdMMPA and MolStandardize (BSD);
  - ChEBI and LIPID MAPS, if seeds come from the engine pool (CC BY 4.0).
- Transform SMARTS are written by us from textbook biotransformation chemistry.
- **Do not port** Ahmed's `derive.py` or his tables (NC-labelled).
- Re-implement the GB-GA operators from the paper rather than copying code, unless its licence is verified as MIT.
- No NIST and no FRIGID.

**Main failure modes and mitigations**
1. **zlog hacking.** A linear bit-weight objective rewards bit-stuffed, unnatural molecules. Mitigations: the √nbits normalisation, the analog-Tanimoto and NP terms, the A3 hard gate, and the full-fingerprint re-score.
2. **Isomer near-ties** bury the truth among positional siblings. Mitigations: the diversity cap, plus ICEBERG/GLACIER re-scoring of same-formula groups downstream (MIT/MassSpecGym).
3. **Unreachable truths.** The S1 median best-analog Tanimoto is only 0.62.
4. **Runtime.** The full fingerprint (16.5 ms) and tautomer keys (138 ms) force the two-stage design.
5. **Wrong neutral mass** from a mis-assigned adduct (dimers, water loss). Use only adduct-consistent targets.
6. **Over-tuning** on 250–300 molecules. Use a scaffold split; never tune on S3.
7. **Outputs that are actually PubChem structures.** These are harmless but redundant with E6; the merge deduplicates them.

**Combination with the existing pipeline (E7 = E6 + a generator lane; append-only, per the DEC-008 rule)**
- Final list: engine top-1, then engine[1:], the PubChem channel and the generator interleaved. Generator candidates enter **only at slots ≥ 4 (default 4, 7, 10, …)**, never displace an engine candidate, and are deduplicated by metric key against the list (only about 25 keys per molecule, roughly 3 s).
- Optional promotion (slots 2–3) only when (best generator full zlog − best database window zlog) exceeds a margin.
- Any gate must pass the SV re-measured-library check (≥ 0.92) before submission. Library-similarity gates are banned (the E5 lesson).
- Kaggle cost: about 15 s × 400 ≈ 1.7 h on CPU, inside the 9 h budget alongside E6's 2.1 h.

**Decision gates before any submission**
- **G1, after the CLI smoke run:** on C3NP and S1, the archive contains the truth (before ranking) for at least 10% of molecules. If it does not, stop; keep only step 3 (one-shot mass repair).
- **G2:** hit@25 ≥ 3% on the held-out C3NP half, with mass-valid outputs at 100%.
- **G3:** the E7 merge loses ≤ 0.003 on SV and S1 against E6, and the proxy gain is ≥ +0.005.
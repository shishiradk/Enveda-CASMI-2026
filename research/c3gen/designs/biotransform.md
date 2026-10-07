**Design: `biotx`, a Class-3 generator that applies hand-written biotransformation rules to library analogs and to mass-shifted database neighbours** (research/c3gen/biotx.py, not implemented)

**0. Measured reachability before any design work (FACT; script at C:/Users/LENOVO/AppData/Local/Temp/claude/d--Enveda-CASMI-2026/b55166b4-4275-49ce-9a40-a16cd42aff87/scratchpad/oracle2.py)**
- Setup: for each bench molecule I took the top 20 of `ana` from results/bench/ho1/recs_*.pkl, with truth keys removed. A molecule counts as "reachable" when one analog has a formula delta equal to one or two rule deltas, and the smaller structure is a substructure of the larger one (bond orders ignored). Rule set: OH, CH2, H2, H2O, hexose, deoxyhexose, pentose, glucuronic acid, acetyl, malonyl, coumaroyl, caffeoyl, feruloyl, galloyl, prenyl, SO3, each in both directions.
- S1 (enveda-np, n=250): 46% reachable (1 step 36%, 2 steps 10%). The reachable analog is in the top 5 for 35%.
- S3 (GNPS NP, n=300): 29% reachable (1 step 17%, 2 steps 13%). Top 5: 23%.
- Most frequent single steps: -CH2 / +CH2 (demethylation, methylation; 15+13 on S1, 13+13 on S3), ±OH, -hexose (10 on S1), ±H2, ±H2O. The **removal direction matters as much as addition**.
- Same formula as some top-20 analog (relocation isomers): 26% on S1, 19% on S3.
- INFERENCE: this is an upper bound on a non-Class-3 bench. Library congeners are over-represented (S1 holds compound series). True Class-3 molecules will have more distant analogs, so I expect real reachability of about 0.5–0.7× these figures.

**1. Algorithm, per molecule (deterministic, one core)**
1. **Seeds A, library analogs.** `ctx['analogs']` best-first, top 20, keeping those with sim ≥ 0.3. Their ik14 values are checked against `forbidden`; this filters inputs only.
2. **Seeds B, mass-shifted database neighbours.** For each of about 40 one-step rule deltas d, take pool + COCONUT structures within 10 ppm of target − d. Score them with packed fp @ zlog (memory-mapped `pool_fp` / `coco_fp`) and keep the top 3 per d, about 120 in total. Drop any with a tautomer key in `forbidden`. Because zlog predicts the unknown's fingerprint, a precursor that scores high is a plausible parent.
3. **Seeds C, window structures.** Same-mass entries from `ctx['window']` (top 10 by fpscore) are used only for zero-net relocation pairs: -X at one site, +X at another.
4. **Mass-first plan.** For each seed, Δ = target − ExactMolWt(seed). The precomputed delta table has about 40 single deltas and about 900 unordered pairs plus zero-net relocations. Keep only rule multisets whose delta is within 10 ppm of target. This prunes about 99% of the combinations a MyCompoundID-style blind expansion would produce.
5. **Enumerate** the matching rules at every symmetry-unique site (RDKit RunReactants plus hand-written graph edits for removals). Sanitize, drop stereo, dedupe on canonical SMILES. Caps: 400 products per seed and 6,000 per molecule. Two-step enumeration runs only on the top 8 seeds.
6. **Rules.** About 45 hand-written SMARTS:
   - +O: aromatic cH→cOH; aliphatic CH→COH; N-oxide; epoxide.
   - Methylation: O-, N- and aromatic C-methyl. Demethylation: O-CH3, N-CH3.
   - O-glycosylation at OH with hexose, deoxyhexose, pentose or glucuronic acid; aromatic C-glycosylation. One sugar per class is enough, because stereo-stripped InChIKey14 merges glucose, galactose and mannose.
   - O- and N-acylation: acetyl, malonyl, coumaroyl, caffeoyl, feruloyl, galloyl.
   - Prenyl: C-prenyl and O-prenyl, plus prenyl→2,2-dimethyl(dihydro)chromene macros.
   - Sulfation at OH.
   - ±H2: C=C, C=O/CHOH, aliphatic CH-CH. ±H2O: dehydration and hydration.
   - Ring closure: lactonization (-H2O), phenol + alkene → dihydrofuran/pyran (-H2 or 0), catechol-OMe → methylenedioxy (+C, a very common NP motif).
   - One generic **hydrolysis rule** (ester, amide, or anomeric C-O) that keeps either fragment. It covers deglycosylation and deacylation at once; the mass filter picks the fragment.
7. **Two-stage scoring.**
   - Stage 1: ECFP4-block partial zlog. This costs about 1 ms per molecule; the full recipe costs about 11 ms (FACT, measured with prep_data fp_and_mass). Keep the top 400.
   - Stage 2: full 6930-bit fp @ zlog.
   - Final score = zlog_full + a·log prior(rule path) + b·seed sim (spectral sim for A, normalised fp score for B) + c·shift localization + d·n_steps penalty.
8. **Shift localization (own implementation of the published ModiFinder idea).** Applies to seed-A analogs that have train spectra.
   - Load the analog's spectra by mapping ik14 → spec rows through the memory-mapped results/train_pkg/data spec_* arrays.
   - Query peaks matching analog peaks at +0 are "unshifted"; those matching at +Δ are "shifted".
   - Run a MetFrag-lite fragmentation of the seed once: break at most 2 bonds, at most 300 fragments.
   - Site score = share of shifted-peak fragments that contain the site + share of unshifted-peak fragments that do not.
   - This is the only signal that separates regioisomers, which zlog cannot. It is computed for the top 100 only.
9. **Output.**
   - Drop products whose key matches a `ctx['window']` key. Those are already in E6 or the engine, and a Class-3 truth is not in the database by definition. This is window dedupe, not use of `forbidden`.
   - Verify ExactMolWt within 10 ppm and return the top k=200 with provenance (for example "ana3:+Hex@O12").
   - The tautomer canonical key is computed only for final dedupe of the top 200. That step costs about 54 ms per molecule (FACT), about 11 s total, which is too slow. So dedupe uses the non-tautomer InChIKey14 (about 3 ms), and the downstream merge, which already dedupes on the metric key, finishes the job.

**2. Data assets (one-off builds, within the resource rules)**
- **Rules plus delta table:** hand-written, no build cost.
- **Mass-sorted index** over pool_mass + coco_mass: seconds, about 15 MB.
- **Pool tautomer keys:** results/c3gen/pool_tkey.npy, built with prep_data canon_key. 772k × 54 ms ÷ 3 workers ≈ 4 h, about 0.6 GB per worker, resumable in chunks. Needed so seed B can be checked against `forbidden`. Reuse results/bench/cache/canon.pkl where it already covers a structure.
- **ik14 → train spectrum rows map** (mol_key/spec_mol, memory-mapped): about 1 min, about 50 MB.
- **Rule priors:**
  - Mine train/COCONUT NP pairs (in_coco) linked by one rule: formula delta + containment, using the same test as section 0. Sample 30k seeds; about 1 h on 3 workers.
  - log prior = log frequency per rule and site type.
  - Fit a, b, c, d by grid search on S1 ∪ S3 (the C3NP dev split once it exists). Four parameters keep overfitting risk low.
- **Runtime estimate:** enumeration 1–3 s, stage-1 fp 3–6 s, stage-2 fp 4.4 s, localization 1–3 s; about 10–16 s per molecule. Resident memory under 1 GB (memory-mapped fingerprints). Total build time is about 5 h of CPU, mostly the tautomer keys.

**3. Expected performance on natural-product Class 3 (INFERENCE, with reasoning)**
- **Reachability:** 46% / 29% on the S1 / S3 proxies (FACT, section 0). Discounted for novelty, I expect 15–30% on C3NP.
- **Ranking given reachability:** a 1-step product competes with 20–200 sibling regioisomers; a 2-step product with up to thousands.
  - zlog alone cannot rank regioisomers (assumption). Expect the truth in the top 200 for about 70% of reachable 1-step cases and about 40% of 2-step cases.
  - Top 25: about 40% / 20%. Conditional MRR about 0.15.
- **Estimates on C3NP:** recall@200 ≈ 12–25%, hit@25 ≈ 6–12%, standalone MRR@25 ≈ 0.03–0.06.
- **Published reference points (quoted from memory, verify before citing externally):**
  - MassSpecGym de novo (Bushuiev et al., NeurIPS 2024): baselines at about 0% top-1.
  - DiffMS (2025): about 2.3% top-1 and 4.3% top-10 with the formula given.
  - So even 5% hit@25 would beat model-only de novo generation. That is plausible here because rules exploit near library neighbours.
  - MyCompoundID (Li et al., Anal. Chem. 2013): 76 reactions expanded 8,021 metabolites to about 376k (1 step) and about 10.6M (2 steps). Blind 2-step expansion explodes, which is why the mass-first plan (step 4) is required.
  - BioTransformer / MINE-style rule expansion is generally reported as high recall, low precision, which matches the ranking bottleneck above.
- **Leaderboard impact:**
  - Class 3 is about 50% of the test.
  - Appending the generator below the engine and E6 costs it a factor of 2–3 in reciprocal rank.
  - Expected gain: **+0.005 to +0.015**. That is a real step, but it does not close the 0.072 gap to 0.432 on its own. It needs to stack with other Class-3 generators and with the localization scorer.

**4. Eligibility of every input**
- **Clean:** RDKit (BSD); our own SMARTS (MIT, written from textbook chemistry); train.parquet structures and spectra (competition data); COCONUT (CC0); PubChem (public domain); zlog from our own ho1/full FPNets (trained on train).
- **Not used:** BioTransformer rule files (licence not verified), RetroRules/MINE rule sets, Ahmed's datasets, FRIGID, and anything trained on NIST.
- **Ideas from papers** (ModiFinder, MetFrag, MyCompoundID) are reimplemented, not copied.

**5. Main failure modes**
1. **Regioisomer ambiguity.** The truth gets generated but ranks 30–200. Mitigation: shift localization plus rule and site priors; measure hit@200 against hit@25 separately.
2. **New skeletons are unreachable.** Many real Class-3 NPs differ from any library analog by more than two decorations. This caps recall (about 70% of C3NP is expected out of reach).
3. **Spurious analogs.** DEC-009 found 38% spurious nonzero library matches in the PubChem-only bucket. Seed B is the hedge, because it does not depend on spectral analogs.
4. **Mass collisions.** Two-step deltas can coincide within 10 ppm, and 56–64% of top-20 analogs had a formula-level 2-step match, much of it chance. Mitigation: n_steps penalty, and 2-step only on the top 8 seeds.
5. **Chemistry failures.** Sanitization fails on sulfates, glucuronides and quaternary N; acid tautomers can change the key. Every product is sanitized; failures are dropped and counted.
6. **Bench optimism.** S1 congeneric series inflate reachability. Fit and judge on C3NP only; S1/S3 are for debugging.
7. **Runtime.** The 11 ms per full fingerprint dominates. The stage-1 prefilter is mandatory, and the caps are hard limits.

**6. How it combines with the existing pipeline (append-only, per the DEC-008 rule)**
- Candidates enter only after the engine's existing top-k and never displace engine or E6 entries.
- Proposed slotting: keep E6's m1_alt list as is. Generator candidates take every 3rd slot after the E6 stream reaches rank 10, plus all slots after the engine and E6 run out (25-guess cap). Dedupe on the metric key against everything above.
- Acceptance criteria: SV drop ≤ 0.002 and S1 drop ≤ 0.005, a positive C3NP gain, and runtime within the Kaggle budget (400 molecules × about 15 s ≈ 1.7 h on one core, or about 35 min on 3).
- No gating on library similarity (rejected in DEC-009).

**7. Evaluation order and kill criterion**
1. Run the reachability oracle on C3NP (no ranking): about 10 min.
2. CLI smoke test: `--bench S1|S3 --n 30` following the interface spec, with hit@1/@25/@200, mean seconds, and mass-valid share.
3. Full C3NP run, then the merge evaluation.
- **Kill** if C3NP recall@200 < 8% or merged C3NP MRR gain < +0.01.
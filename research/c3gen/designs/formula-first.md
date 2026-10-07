FORMULA-FIRST ISOMER-EDITING GENERATOR (`research/c3gen/ffiso.py`): design only, nothing implemented

**Bottom line (INFERENCE):** expect C3NP recall@200 of about 6–12% on S1-like molecules and 3–5% on S3-like ones, and roughly 2–4% on the real hidden Class 3. That is worth about +0.002 to +0.006 on the LB. It is a cheap, eligible piece of the Class-3 answer, not the fix for the 0.07 gap to 5th place. The upside is the measured result that the zlog-best same-formula database isomers are often near neighbours of the truth.

**0. Grounding measurement (FACT).** Script: `C:\Users\LENOVO\AppData\Local\Temp\claude\d--Enveda-CASMI-2026\b55166b4-4275-49ce-9a40-a16cd42aff87\scratchpad\c3_formula_probe2.py`. Output: `probe2.parquet` in the same folder. It ran for 108 s at about 1 GB RAM. Each S1/S3 truth was treated as Class 3: its key was removed from the engine pool (772,653 structures) and from its analog list, and the ±10 ppm window was examined. Morgan2 Tanimoto is written Tc.

| Measured on the truth-removed window | S1 (250) | S3 (300) |
|---|---|---|
| Truth formula still present among window structures | 96.8% | 74.0% |
| Truth formula ranked by window population: top-1 / top-3 | 83.6% / 93.2% | 46.7% / 67.0% |
| Number of formulas in the window (median) | 4 | 3 |
| Same-formula isomer with the same generic Murcko scaffold | 72.4% | 42.0% |
| Same-formula isomer with the same exact scaffold | 48.4% | 36.0% |
| Best isomer Tc ≥ 0.7 / ≥ 0.85 | 40.8% / 13.2% | 12.3% / 6.3% |
| Top-5 isomers by fp@zlog: max Tc ≥ 0.7 | 28.4% | 12.3% |
| Top-5 isomers by fp@zlog: same generic scaffold | 56.4% | 39.7% |
| Spectral analog has the truth formula: top-1 / any of top-10 | 12.4% / 23.6% | 10.3% / 16.3% |
| Spectral analogs: max Tc ≥ 0.7 within the top-10 | 29.2% | 32.7% |
| Analog mass difference equals 0 or one of 16 common transforms: top-1 / any of top-10 | 36% / 58% | 27% / 46% |
| Truth's fp@zlog rank among real same-formula isomers: top-1 / MRR | 28.9% / 0.465 | 81.1% / 0.878 |

- The S3 ranking result is suspiciously high (INFERENCE): S3 truths may come from heavily trained families. Use S1 as the primary dev set.
- Number of valid formulas in a window (CHNOPS + Cl/Br/F, Senior/RDBE/H:C rules, own enumerator): about 55 within 5 ppm at 348 Da and about 1,100 at 900 Da. Only 3–16 of them are CHNO-only.
- Reading (INFERENCE): parent selection should use the zlog-ranked same-formula window, not spectral analogs. That window carries a near-scaffold relative of the truth 2–4× more often than the analogs do. This is the main difference from the v4n `generate=True` generator, which is analog-only.

**1. Algorithm, per molecule (deterministic; ties broken by canonical SMILES)**
1. **Formula candidates F.** Take the union of:
   - (a) formulas of window structures;
   - (b) each analog formula plus or minus a transform delta (own list of about 40 single NP edits: ±O, ±CH2, ±H2, ±hexose, ±pentose, ±deoxyhexose, glucuronide, ±acetyl, malonyl, prenyl, sulfate, ±CO2, acyl groups), kept when the mass lands within 5 ppm;
   - (c) the enumerated formulas from step 0, used only as a fallback.
   
   A formula without a parent cannot be edited into a structure, so it is used only to veto other formulas.
2. **Formula score.** Sum these terms:
   - log of the population prior, P(formula | mass bin, element set, RDBE bin), from table T1;
   - spectral consistency: the intensity share of fragment peaks explainable as sub-formulas of F (adduct-aware, 10 ppm or 3 mDa), plus required-element checks for diagnostic neutral losses (hexose 162.0528 needs ≥ C6H10O5; HCl or SO3 losses need Cl or S);
   - an analog-agreement bonus: the zlog-weighted share of analogs with formula F, or F reachable by one transform.
   
   Keep the top 3 formulas, or more while cumulative softmax mass is below 0.9. FACT: the population prior alone already puts the truth formula in the top 3 for 93% (S1) / 67% (S3).
3. **Parents (at most 12).** Per kept formula:
   - the top 6 window structures by `fpscore` (the zlog score) and the window structures with the same generic scaffold as them;
   - analogs with sim ≥ 0.3 and formula F;
   - analogs one transform away from F, with the transform applied first (depth 1, plus one move at depth 2).
   
   Parents are weighted by zlog rank, analog similarity and formula probability.
4. **Formula-preserving moves.** These are own reaction SMARTS plus graph code, each applied at every allowed site:
   - M1: ring-walk of one substituent (OH, OMe, O-sugar, prenyl, Cl, Me, OAc) to another ring atom carrying an H, within the same ring system;
   - M2: swap the positions of two different substituents (OH↔OMe gives methyl migration);
   - M3: acyl migration between hydroxyls of the same sugar or cyclitol (the caffeoylquinic-acid type);
   - M4: move the glycosylation site between phenolic OHs, and change the interglycosidic linkage (1→2, 1→4, 1→6);
   - M5: prenyl plus ortho-OH ↔ 2,2-dimethylchromane or chromene (formula-preserving cyclisation), and shift of a double bond within a chain;
   - M6: O- vs C-methyl and O- vs C-glycoside interchange.
   
   Limits: depth ≤ 2 moves, at most 300 products per parent and 3,000 per molecule. A site is admitted only if its COCONUT site prior (table T2) is ≥ 1e-4.
5. **Validation.** Each product must sanitise, be uncharged, keep the target formula, and have ExactMolWt within 10 ppm. Dedupe by plain InChIKey14 first (about 1 ms each). Drop products whose key equals a parent's key or any window key; that is only dedupe against the engine list, because the window is already truth-free. The `forbidden` set filters only the database inputs: window, analogs and T2 rows. It never filters outputs.
6. **Score.** `s = z(fp@zlog)` within the molecule, `+ 0.3·log T2(site|scaffold) + 0.2·parent_weight − 0.1·depth`, plus optional cheap fragment coverage (1–2 bond-cut fragment masses explaining peaks). Weights are tuned on the C3NP S1 dev split only.
7. **Output.** Tautomer-canonical keys for the top 300 only, with `SetMaxTautomers(200)`; collapse duplicates; return the top k = 200 as `(smiles, score, "move|parent_ik14|formula")`.

**2. Data assets** (built under `results/c3gen/` with at most 3 workers, under 2.5 GB, DuckDB `memory_limit='2GB'`, projection only)
- **T1, formula prior.** Formula, element set, RDBE and mass of the pool: COCONUT `coco_meta.pkl` smiles plus `results/train_pkg/data/pool_smiles.txt` (711k rows). Streamed in 50k chunks. About 5 min with 3 workers, under 0.8 GB, output under 20 MB.
- **T2, site and substituent priors.** Per structure: Murcko scaffold, canonical attachment-atom ranks and the substituent at each. Counts are aggregated to P(substituent at rank-class | generic scaffold) and P(acyl or sugar on sugar position). About 20–30 min with 3 workers, under 1.5 GB in total, output about 50 MB.
  - **Leak rule:** T2 must be built with every bench, C3NP and forbidden key excluded, and with a per-split rebuild if needed. Otherwise the truth's own substitution pattern leaks into the prior.
- **Transform and move SMARTS.** About 60 hand-written lines, our own MIT code. The idea comes from the public v4n description; no NC code is copied.
- **Runtime estimate (INFERENCE):**
  - formula step under 0.3 s;
  - 3,000 products × (sanitise + InChIKey + `prep_data` fingerprint) at about 3 ms each ≈ 9 s;
  - tautomer keys for 300 ≈ 3–5 s (the probe showed uncapped tautomer canonicalisation is the bottleneck);
  - total ≈ 12–16 s per molecule with a hard 18 s cap;
  - full S1+S3 bench (550 molecules, 3 workers) ≈ 45 min at under 0.6 GB per worker.

**3. Expected recall@200 and MRR@25 on NP Class 3 (INFERENCE, derivation shown)**
- **Reach ceiling.** A same-scaffold isomer with Tc ≥ 0.7 exists for 36% of S1 and 12% of S3 molecules, and the top-5 zlog isomers carry Tc ≥ 0.7 for 28% / 12%. Perhaps 30–50% of those truths are exactly one or two M1–M6 moves from such a parent. That gives a simulated reach of about 9–14% (S1) / 4–6% (S3). The truth survives into the top 200 in about 70% of those cases.
- **C3NP recall@200.** About **6–12% (S1-like) and 3–5% (S3-like)**.
- **Real Class 3.** Forum D/743254 reports that deletion-based novelty simulations overstate reach by about 3×. The test's "hypothesised NPs / NP analogs" are also probably farther from known isomers. Expect about **2–4% recall@200**.
- **Ranking.** fp@zlog ranks the truth first among real isomers 29% of the time (S1, MRR 0.465). Generated positional isomers are harder decoys, so the MRR given a hit is about 0.25–0.35. C3NP S1 MRR@25 ≈ 0.7 × 9% × 0.3 ≈ **0.015–0.025**; S3 ≈ 0.006–0.012.
- **Published reference points** (formula-known MassSpecGym de novo top-1): MADGEN 1.31%, DiffMS 2.30%, MBGen 7.58%; FRIGID 16–18% uses NC weights and is ineligible. DiffMS on NPLIB1 reached 0.052. The v4n log says 13% of Class-2 truths are "derivable", with no isolated LB gain. Other teams' derivative candidates went 0.337 → 0.335 (D/742055). This design should land near the MADGEN/DiffMS level at CPU cost.
- **LB value.** 0.5 Class-3 share × about 3% hit within the slots used × reciprocal rank about 0.2 (generated candidates sit at slot 4 or lower) ≈ **+0.002 to +0.006**.

**4. Eligibility**

| Input | Licence | Status |
|---|---|---|
| COCONUT | CC0 / CC BY | OK |
| train.parquet structures | competition data | OK |
| PubChem window | public domain | OK |
| zlog | our own ho1/full FPNets, trained only on train.parquet | OK |
| RDKit | BSD | OK |
| T1/T2 tables and transform list | our own | OK |

Two items need action:
- **LIPID MAPS (open).** The engine pool includes prvsiyan's ChEBI + LIPID MAPS dataset. ChEBI is CC BY 4.0. LIPID MAPS terms have not been verified here, so either exclude its rows from T1/T2 or confirm the licence first.
- **Excluded inputs:** no Ahmed datasets or code, no FRIGID, no NIST-trained weights. ICEBERG/GLACIER MassSpecGym MIT weights are optional later as within-formula re-scorers.

**5. Main failure modes**
1. The truth formula is not in the window and no single transform reaches it: 3% (S1), 26% (S3).
2. Skeletal novelty, such as a new ring fusion, an unusual substituent or a rearranged core, is beyond one or two moves. This is dominant for real Class 3.
3. fp@zlog barely separates positional isomers, so hits land at ranks 30–200 and add no MRR. Forward-model re-scoring would be needed.
4. Combinatorial blow-up for glycosides above 700 Da with more than 10 OH sites. Use the caps and keep the site-prior cut first.
5. Tautomer canonicalisation cost and timeouts.
6. Implausible regiochemistry, such as OH on a bridgehead or enols, fills slots. T2 must be strict.
7. Proxy optimism: C3NP truths have published close isomers by construction. Also, if T2 is built without the forbidden exclusion, the bench is leaky.

**6. Combining with the pipeline** (append-only, extending the E6 m1_alt rule)
- Engine top-1 is always kept (or the top-2 when the top-1 library similarity is ≥ 0.9). Then slots 2–25 cycle E, P, E, G: engine[1:], the E6 PubChem channel, then the generator's top-6. A candidate already listed is skipped.
- No library-similarity gate; that was rejected in DEC-008/009.
- The generator's best candidate therefore sits at slot 5, and at most 6 engine slots are displaced, all at rank 8 or below.
- **Acceptance test:** SV drop ≤ 0.002, S1/S2 drop ≤ 0.004, PubChem-only unchanged within ±0.003, and C3NP up by at least 0.01. Only then one LB submission.
- **Later:** the ranker learns `gen_*` features from C3NP drop-truth simulations instead of fixed slots.
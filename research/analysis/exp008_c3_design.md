# EXP-008 — Pseudo-Class-3 design (first C3 experiment)

Date: 2026-09-24. Status: **DESIGN ONLY — not approved, not run.** No `src/` change, no
research-state edit, no DEC-008. Builds on `research/analysis/parallel_c3_candidate_map.md`
(option space) and the completed EXP-007 full run (`research/analysis/exp007_c2_full_run.md`).

Labels: FACT (measured / read from code), INTERPRETATION, UNKNOWN.

---

## 1. What EXP-007 established (input to this design)

### FACT (measured directly)
- C2(τ) dose–response on 490 reachable queries: MRR@25 0.221 (τ=.50) → 0.519 (τ=.95) vs C1 0.660.
- C1-matched pool-size control flat at 0.660–0.661 at every τ; true rank changed in 12/2940 control runs.
- Failure decomposition at τ=.50 *(as reported; not exact, see CAVEAT below)*: **A** candidate loss 171, **B** present but rank>25 72, present in top 25 247, no-op 33.
  At τ=.95: A 36, B 36.
- C1 by novelty stratum: S1 (all same-adduct evidence <0.50) MRR 0.340 / R@25 0.727 **with the truth in the pool
  every time** (candRec 1.000); S2 0.411; S4 0.821.
- *(New analysis of the existing jsonl, this design, no rescoring)* Of the 171 τ=.50 candidate-loss queries *(provisional, see CAVEAT)*,
  **124 still have train spectra of the truth, all of them cross-adduct only; 47 have no remaining spectra at all.**
  At τ=.95: 34 cross-adduct-only, 2 none.
- *(New, same source)* In ±0.01 Da pools the top-ranked wrong molecule has the truth's formula in 47.8% of C1
  queries (54.4% in τ=.50 candidate-loss rows). Tanimoto(top-wrong, truth) median 0.165; the best
  analogue anywhere in the top-25 wrong list has median Tc 0.273 and is ≥0.70 in only 5.8% of queries.

### FACT: EXP-007 construction issue found while preparing this design
- `research/scripts/exp007_audit_phase3.py` builds same-adduct "families" keyed on
  `(inchikey14, adduct, round(precursor_mz, 4))`. Same-adduct truth spectra whose precursor m/z differs
  in the 4th decimal are in the ±0.01 Da candidate window but are **neither counted as retained nor
  removed** by C2(τ).
- Data consistent with that: 96/110 "UR" (supposedly unreachable) queries have the truth in the C1 pool as a
  **same-adduct enveda-180** candidate (truth score median 0.921); and in C2 at τ=.50, **102/490 rows still
  have a truth score ≥ τ**, which the construction was meant to rule out (96 of them genuine k≥1 runs; 56/490 at τ=.95).
- **CAVEAT (binding for every use of EXP-007 in this design, added at review 2026-09-24):**
  The EXP-007 audit used exact precursor matching to 4 decimal places, while candidate generation used a ±0.01 Da
  precursor window. Some same-adduct spectra can therefore remain in the actual candidate pool without being counted
  as retained evidence by the audit, or removed by C2(τ). Consequently:
  - the τ=.50 decomposition **171 / 72 / 247** (candidate loss / rank>25 / top-25) is **not exact** and must be
    recomputed after the discrepancy is corrected;
  - the derived split **124 other-adduct-only / 47 no-evidence** is **provisional** and is not a final quantitative conclusion;
  - EXP-007's **magnitude** estimates (C2 MRR per τ, deltas, the UR stratum) **are affected** and must not be cited
    as exact. The direction of the effect is not in question, but the size is.
  - **Not invalidated** and kept separate from the affected audit decomposition: the C1-matched pool-size
    control (flat 0.660–0.661; 12/2940 rank changes). It removes guaranteed-wrong spectra and does not depend
    on the family keying. The same goes for the leakage checks (query group excluded), the smoke replay
    (390/390) and run-to-run determinism.
  - EXP-008 does not inherit the discrepancy: it excludes by `inchikey14` (plus parent-key salt/charge forms)
    over all libraries, never by metadata family. EXP-009 must not run until the family keying is corrected.

### INTERPRETATION
- **A (candidate-generation loss)** is the dominant C2 loss at deep novelty. On the provisional counts, most of it (124/171) is a
  *window artefact*: the molecule is known, but its only remaining evidence is at another adduct, outside
  the raw-m/z window. The rest (47/171) is genuine loss of coverage, so those queries are effectively C3.
- **B (ranking loss while reachable)** is real but secondary. It is concentrated in low-similarity evidence (S1/S2):
  when the only evidence is spectrally distant (<0.50–0.70), Modified Cosine ranks the truth poorly even when it is present.
- **C (pool effects)**: none measurable at k≤3 (flat control). EXP-002's pool-expansion damage came from
  *what* was added, not from how many.
- **D (spectral generalization)**: current scorer generalises weakly. MRR falls roughly linearly with the
  similarity of the best remaining evidence, and the top wrong candidates are mostly isomers that are structurally unrelated to the truth.

### UNKNOWN
- The true C2 magnitude once the family keying is fixed.
- Whether cross-adduct evidence (the 124) can be *ranked* once it is made reachable. EXP-002 recovered 12 cases,
  9 of them via shifted matching. Audit phase 2 found a cross-adduct pair median of 0.013. The evidence conflicts and n is small.
- The share of C3 molecules in the hidden test, and whether they exist in any structure DB we could ship. Neither can be
  measured locally: `test.parquet` is a train subsample (parallel map §0).

---

## 2. C3 objective and the construction that follows from it

A spectral-library ranker can only return molecules that have spectra (FACT, code). Under true C3 the
target has none, so **V0 / the C1 pipeline scores 0 on every C3 query by construction.** Pseudo-C3 therefore
needs two separate sources:

- **Structure universe** (candidates): all 275,810 train structures, built once, independent of any target.
  It contains the target, because a real C3 answer can only be returned if it is in the shipped structure DB.
- **Spectral evidence library**: train spectra with the **target removed entirely**. It supplies "known/learned
  spectral evidence" about *other* molecules.

This implements "known evidence → unknown molecule". It is not simply "delete the target and call it C3", because
the target stays reachable as a structure and the question is whether evidence from *other* molecules can point to it.

---

## 3. Option assessment

| Option | Scientific question | Legit test-time info | Leakage risk | Pool size | Diagnostic metric | Difficulty | Quick? |
|---|---|---|---|---|---|---|---|
| Formula-constrained retrieval | Does knowing the formula shrink the pool enough to matter? | **No formula field at test** (FACT); only inferred | Using train formula of truth = oracle | median 35 isomers vs 52 at 5 ppm (census) | pool size + truth-in-pool, oracle vs mass window | Low as oracle; high as real inference (SIRIUS-like) | Oracle: yes |
| Same-formula candidate retrieval | Can spectra rank truth among isomers? | Same as above | Same | ~35 | MRR vs chance H₂₅/n | Low | Yes (a filter on EXP-008 scores) |
| Structural analogue retrieval | Do spectral neighbours carry structural information about an absent target? | Query spectrum, adduct, library spectra and their structures | Target or connectivity duplicates left in library; "has spectra" used as a feature | pool ~52 (5 ppm); wide spectral search window | lift over chance; decoy-query control | Medium | Yes (smoke ~30–60 min) |
| Spectral embedding retrieval | Does a learned embedding beat Modified Cosine for analogues? | Same | Training set must exclude targets | Same | Same, vs the ModCos analogue arm | High (training) | No; wait for the analogue result |
| Oracle candidate pool | Is C3 limited by generation or by ranking? | n/a (oracle) | Must be labelled an upper bound | Same | ranker MRR in a guaranteed-truth pool | Low | Yes, **it is the neutral-mass structure pool itself** |
| Hybrid candidate generation | Library hits + structure pool combined | Legit | Same as analogue option | union | per-class MRR | Medium | After EXP-008 |

The oracle pool and neutral-mass structure pool coincide for pseudo-C3: the universe contains the target by
construction. So EXP-008 is simultaneously the reachability measurement (Q1) and the oracle-pool ranking test (Q2/Q3).

---

## 4. Minimal experiment sequence

| Step | Answers | Why this order (evidence) |
|---|---|---|
| **EXP-008a** pool census (no scoring, minutes) | Q1: is the unseen molecule reachable with legitimate candidate generation (neutral mass from adduct, ppm window, structure universe)? | EXP-002: neutral mass is the chemically right generator, with a sign-fixed adduct table. Census: timsTOF precursor error p99 4.4 ppm. EXP-007: A-loss dominates, so reachability is the first-order question. |
| **EXP-008b** smoke ranking (n=30) | Q2: can the current ranker distinguish it? Q3: does structural analogue information help? | By construction the current direct ranker cannot score a spectrum-less molecule (canary = 0). The only legitimate use of the current scorer under C3 is *indirect*, via neighbours. The same-mass pools lack analogues (best Tc median 0.27 in top-25), so analogue evidence must come from a wide-mass spectral search. |
| EXP-009 (conditional, C2) cross-adduct rescue on EXP-007's other-adduct-only queries (count to be recomputed) | Q2 for "known molecule, only other-adduct evidence" | Largest single recoverable C2 block. It needs its competitor set generated by neutral mass, not truth-only (else oracle). **Blocked until the family keying is corrected.** |

Learned embeddings or fingerprint prediction are considered only if EXP-008 shows no lift over chance.

---

## 5. EXP-008 specification

### 5.1 Population
- Source: the locked EXP-007 manifest (`results/exp007_c2_full_run_manifest.json`, 600 timsTOF query rids,
  seed 20260922). Each C3 query therefore has a C1 and C2 result to compare against per query.
- Eligibility:
  1. Query adduct is a **monomer** adduct in the EXP-002 §5 table (sign-fixed: `[M+Cl]-`, `[M+CH2O2-H]-` are net additions).
     Multimers are excluded from the smoke.
  2. The truth `normalized_smiles` parses in RDKit (census: 0 failures expected).
  3. Pool (5 ppm) contains ≥2 molecules.
  4. A decoy exists (§5.6).
- Stratification (for reporting only, computed from the truth structure *before* any scoring):
  NN-Tc = max Tanimoto (Morgan r=2, 2048 bits) of target to any molecule that still has a same-adduct spectrum
  in the evidence library. Tiers fixed now from the census quartiles: **T-low <0.60, T-mid [0.60,0.70), T-high ≥0.70**.
- Smoke: **n=30**, 10 per tier, seed **20260924**, sampled from eligible rids. The manifest is written before scoring.
- Full (only after approval): all eligible rids of the 600.

### 5.2 Exclusion / leakage rules
- **L1** Evidence library excludes every row with the target `inchikey14`, across all libraries. Assert 0 remaining.
- **L2** Connectivity duplicates: also remove spectra of molecules whose RDKit largest-fragment + neutralised
  InChIKey14 equals the target's (salt / charge forms). Report the count.
- **L3** Structure universe built once from all train molecules. No per-target edits. No feature derived from
  spectrum count or ingest metadata, because the target is the only pool member with zero spectra and such a feature would leak it.
- **L4** Scorer inputs: query spectrum, precursor_mz, adduct, library spectra with their structures, candidate
  structures. Never the target's inchikey14, SMILES, or formula. Formula is used only in the labelled oracle diagnostic.
- **L5** Tc(·, truth) is used only for stratification and evaluation.
- **L6 canary**: the direct C1 ranker (R2) must give RR = 0 for every target. Any non-zero value means the exclusion failed: stop.
- **L7** Decoy exclusion is symmetric with target exclusion (§5.6).
- **L8** W, K, ppm, and tiers are fixed in this document; they are not tuned on outcomes. Any change becomes a new version.

### 5.3 Candidate generation (Q1)
- Neutral mass per query: `M = (precursor_mz − Δ_adduct) / n` (n=1 for monomers).
- Structure universe masses: RDKit `ExactMolWt` of each train structure (as in the census).
- Pool: universe molecules with |M_c − M| ≤ **5 ppm** (primary; ≥ p99 timsTOF error). 10 ppm is recorded as a sensitivity arm.
- Diagnostic only (oracle, labelled): the pool restricted to the truth's formula.

### 5.4 Rankers
- **R0 chance**: exact per-query expectation `Σ_{r≤min(25,n)} (1/r)/n`.
- **R2 direct (current C1 ranker)**: score each pool molecule by max ModifiedCosineGreedy (tol 0.1, mz_power 0,
  intensity_power 1) over its own library spectra. Molecules with no spectra are unscored and rank last.
  Serves as the Q2 answer for C3 and as the L6 canary.
- **R1 analogue propagation (primary, current scorer used indirectly)**:
  1. Wide spectral search: the evidence library restricted to **same adduct, enveda-180**, one representative spectrum per
     molecule (merged-CE spectrum if present, else max `num_peaks`, tie → lowest rid; a label-free rule),
     precursor within **±150 Da** of the query. Score with ModifiedCosineGreedy (same params).
  2. Take the top **K=20** hit molecules h_k with scores s_k.
  3. Candidate score `S(c) = max_k s_k · Tc(c, h_k)`, where **R1-noself excludes h_k = c**. This is the clean Q3 test: all
     candidates are treated alike and none can score through its own spectra.
  4. Secondary variants: **R1-self** (h_k = c allowed), the realistic hybrid that shows how strongly spectrum-bearing
     wrong candidates beat the spectrum-less truth, and **R1-sum** `Σ_k s_k·Tc(c,h_k)`.

### 5.5 Baseline
R0 chance within the identical pool. The primary effect is **lift = RR(R1-noself) − RR(R0)**, paired per query.

### 5.6 Control
- **Decoy-query control**: for each target t, choose decoy d uniformly (seeded) among the other pool molecules
  that have a same-adduct spectrum. Replace the query spectrum with d's representative spectrum, remove all of
  d's spectra from the evidence library (as for t), keep the pool unchanged, and evaluate the rank of **t**.
  Expected ≈ chance. If decoy lift ≈ R1 lift, the signal comes from pool structure (the target is "central"),
  not from the spectrum.
- **Formula-oracle diagnostic**: R1-noself and R0 inside the truth-formula pool, showing the value of a perfect formula.

### 5.7 Metrics
Pool recall (5/10 ppm), pool size (molecules), MRR@25, R@1/5/10/25, per-query lift with paired bootstrap CI
(1000 resamples, seed 20260924), Tc(top-1, truth) vs mean Tc(pool, truth), max Tc(h_k, truth).
Everything is reported overall and by NN-Tc tier, alongside each rid's EXP-007 C1/C2 rank.

### 5.8 Expected failure categories (recorded per query)
- **F0** truth not in pool (mass or adduct error).
- **F1** no informative neighbours (all s_k < 0.30).
- **F2** neighbours found but structurally unrelated to truth (max Tc(h_k, truth) < 0.40): spectral/structural decoupling.
- **F3** related neighbours, but a pool isomer is structurally closer to them (isomer confusion).
- **F4** truth in top 25 but not rank 1.
- **F5** (R1-self only) displaced by spectrum-bearing wrong candidates scoring through self-hits.

### 5.9 Output files
- `research/scripts/exp008_c3_smoke.py`
- `results/exp008_c3_smoke_manifest.json` (written before scoring: rids, tiers, decoys, params, seeds)
- `results/exp008_c3_smoke_query_results.jsonl` (per query × ranker: pool ids/size, truth rank, RR, chance RR,
  top-25 with scores, h_k list with s_k and Tc to truth, failure category, L1/L2/L6 check fields)
- `results/exp008_c3_smoke.json` (aggregates, checks, runtime)
- `research/analysis/exp008_c3_smoke_test.md`

### 5.10 Reproducibility
Reuse EXP-007's rid reconstruction (`PRAGMA threads=1`, `row_number() OVER () − 1`, verify the two known triples).
Pin the seeds (population 20260924, decoys 20260924+1, bootstrap 20260924+2). Write the manifest before scoring.
The script must be deterministic: running it twice must produce a byte-identical jsonl (as in EXP-007 §13).
Record the matchms/RDKit versions.

### 5.11 Cost
- 8a: minutes (duckdb masses + RDKit exact masses, already computed in the census).
- 8b: roughly 20–40 k representative spectra in a ±150 Da same-adduct window at ~0.5 ms/pair gives ~10–20 s per query.
  With query and decoy per rid, 30 rids take ~15–25 min (INFERENCE from the submission-pipeline timing; to be measured).

### 5.12 Stopping criteria
- **Stop the smoke** if L1, L2 or the L6 canary fails; if pool recall at 5 ppm is < 0.95 (fix the mass/adduct code
  first); or if the median runtime is > 2 min/query (switch to a prefilter design before scaling).
- The smoke validates integrity only. At n=30 it is not powered for conclusions, and it does not authorise the full run.
- Full-run reading, fixed in advance:
  - R1-noself lift CI > 0 **and** decoy lift CI ∋ 0: spectral-library evidence carries transferable
    structural signal for unseen molecules (Q3 yes). Next, compare with learned representations.
  - R1 lift ≈ decoy lift: pool artefact, no spectral signal.
  - Both ≈ chance: the current scorer offers no C3 leverage, and C3 needs a learned structure scorer (fingerprint
    prediction). Training is justified only at that point.

# Parallel map: candidate-generation bottlenecks, oracle experiments, pseudo-C3 design space

Date: 2026-09-24. Status: **ANALYSIS / DESIGN ONLY.** Nothing here is approved. No C3 run, no `src/` change,
no edit to `CASMI_RESEARCH_STATE.md` / `decision_log.md`, no DEC-008. EXP-007 full run is in progress
(OpenCode) and was not touched.

Labels: FACT (measured / read from code), INFERENCE, HYPOTHESIS, DESIGN OPTION, UNKNOWN.
Counts: `research/analysis/parallel_c3_census.json` (single DuckDB pass + RDKit; methods recorded there).

---

## 0. Urgent finding (outside the brief, affects everything that uses `test.parquet`)

- **FACT:** all **1,213/1,213** local `test.parquet` spectra have an exact duplicate in train `enveda-180`
  (identical `ms2_mzs`, identical `ms2_normalized_intensities`, identical `precursor_mz`, same adduct).
  All 400 test `molecule_id`s map to exactly one `inchikey14` each. These molecules also have further
  train timsTOF spectra (median 3 extra; 106/400 have none beyond the test copies).
- **INFERENCE:** the local `test.parquet` is a sample drawn from train, not the scored hidden test. The observed
  public leaderboard (~0.33–0.40 MRR, state §15) is incompatible with the scored set being these spectra
  (exact lookup would score ~1.0). Most consistent explanation: code competition whose hidden test replaces
  this file at rerun. **UNKNOWN until checked against the official rules/data page.**
- **Consequences:**
  1. Test *schema* is still valid evidence of what fields exist at test time (INFERENCE that the hidden test
     shares the schema — standard for code competitions, not verified).
  2. Test *distributions* (adduct mix 79% [M+H]+, ~3 spectra/molecule, median ~230 peaks) are distributions
     of a train subsample, **not** evidence about the hidden test. `00_dataset_report.md` §"test" claims
     that use them need a caveat.
  3. `test.parquet` cannot be used as a class-mix proxy. (My 150-spectrum top-1 proxy returned 1.0 for every
     spectrum for exactly this reason; discarded.)
  4. Nothing in this project should exploit the duplication; it is flagged, not used.
- Proposed (not done): record as a research-state fact after the user reviews it.

---

## 1. Candidate-generation bottleneck map

### 1a. What generates candidates (FACT, from `research/scripts/exp007_smoke_test.py` + state §5)

`raw precursor_mz ± 0.01 Da` window over library **spectra** (no adduct filter) → matchms
`ModifiedCosineGreedy(tol 0.1, mz_power 0, intensity_power 1)` per spectrum → **molecule-level max** over
spectra → top 25. Candidates can only be molecules **that have a spectrum in the window**.

### 1b. Query-time information

| Available at test time (FACT, local schema) | NOT available (FACT) |
|---|---|
| `ms2_mzs`, `ms2_normalized_intensities`, `base_peak_intensity` | structure, `inchikey14`, `normalized_smiles` |
| `precursor_mz`, **`adduct`**, `ionization_mode`, `instrument_type` | **`molecular_formula`** |
| `collision_energy_orig/_ev/_units` | `precursor_error_ppm`, `ingest_lib`, `num_peaks` field (derivable) |
| `molecule_id` grouping → **1–9 spectra per molecule**, some multi-adduct | MS1 / isotope pattern (no field) |

INFERENCE: because `adduct` is given, neutral mass `M = precursor_mz − adduct_mass` is computable exactly
per query. Local check: neutral masses of spectra of the same `molecule_id` agree to ≤1.25 mDa (p99), so
the adduct labels are internally consistent (on the train-derived sample).

### 1c. Bottleneck table

| # | Bottleneck | Evidence | Established by | Confidence | Unknown | Cheapest next test |
|---|---|---|---|---|---|---|
| 1 | **Raw-m/z window misses cross-adduct evidence** | Mode B candidate recall .970: 11 absent + 1 zero-cand; the 12 absent = exactly the "only cross-adduct evidence" queries | EXP-001, DEC-005 | FACT (C1, n=400) | Behaviour under C2 novelty | Candidate-recall-only recompute of EXP-007 C2 pools under neutral-mass generation (no scoring) |
| 2 | **Fixing #1 costs ranking** | Variant B: recall 1.000 but MRR .651→.618, R@1 .535→.490; pool ×1.86 | EXP-002 | FACT | *Why* (pool size vs cross-adduct calibration vs lost accidental hits) | EXP-003 full run (designed, smoke-passed, never run) |
| 3 | Cross-adduct displacers | 54/92 displacers cross-adduct, 38/92 same-adduct; wrong same-adduct median .929 vs correct .791 | EXP-003 targeted | OBSERVATION (failure-selected) | Full-population share | EXP-003 arms B vs C |
| 4 | **Ranking, not generation, dominates C1** | 388/400 truth present; only 366 in top 25, R@1 .535 | EXP-001 | FACT | Whether that holds under C2/C3 | Already answered for C1; EXP-007 answers C2 |
| 5 | Evidence removal → candidate loss (C2) | Smoke τ=.50: candGen .70; **all 9 "close→lost" rows are truth-absent-from-pool**, 0 are present-but-ranked->25 | EXP-007 smoke (n=30) | OBSERVATION | Full-run share; whether truth still exists in library outside the raw window | EXP-007 full run + #1's test |
| 6 | Precursor tolerance too wide | Window ±0.01 Da ≈ 25 ppm at m/z 400; timsTOF abs error p99 4.4 ppm, p99.9 5.7 ppm | Census (FACT); effect on ranking untested | INFERENCE | Whether narrowing removes displacers or just shrinks pool | Re-rank existing EXP-001 Mode-B pools with ppm window (no new scoring: filter stored scores) |
| 7 | Isomer multiplicity | Median **35** same-formula train molecules per timsTOF molecule (p90 168); median 52 other molecules within 5 ppm | Census | FACT (counts) | How often displacers are isomers | Tag displacers in EXP-001/007 by formula-equal-to-truth |
| 8 | Max aggregation favours molecules with many spectra | Aggregation is max over spectra (code) | Code only | HYPOTHESIS | Any effect at all | Correlate displacer's #spectra-in-window with displacement on existing EXP-001 rows |
| 9 | Pool size per se | EXP-007 smoke C1-matched: 0/180 rank changes, but k≤3 → near-zero power; EXP-003 E arm never run | EXP-007 smoke | UNKNOWN (untested at useful k) | Everything | EXP-003 E arm; EXP-007 full C1-matched vs k |
| 10 | **Library coverage (C3)** | Candidates must have a spectrum in window → a molecule with no spectra can never be ranked | Code (logic) | FACT | Hidden-test C3 fraction | None cheap (hidden test unobservable; see §0) |
| 11 | Spectral similarity ↔ structure | Same-mass (±0.01 Da) best Tanimoto to a timsTOF molecule: median **.35**; overall NN Tanimoto median .66 | Census | FACT (structure side only) | Does spectral NN retrieve structural NN? | Analogue experiment AN-1 (§4) |

INFERENCE (important for reading EXP-007): in this pipeline "candidate recall" conflates *the structure is
known* with *a spectrum of it lies in the raw-m/z window*. A C2 "candidate failure" may be a window/adduct
artefact rather than an absence of knowledge about the molecule (bottleneck #1 re-appearing under novelty).

---

## 2. Oracle-candidate experiment map (DESIGN OPTIONS, not executed)

Fixed across all: EXP-007 population + τ grid, ModifiedCosineGreedy params, max aggregation, metrics
(candGen, R@1/5/10/25, MRR@25), leakage rules of EXP-007 §8.

| Oracle | What changes | Question | Supports **generation** limit if | Supports **ranking** limit if | Leakage risk / invalid if |
|---|---|---|---|---|---|
| **O-A** truth-evidence-present | C2 pool + truth molecule's *remaining* library spectra that lie outside the raw window (other adducts), found via neutral mass; nothing else added | Are C2 candidate losses recoverable evidence that the window misses? | Most C2 truth-absent queries regain truth AND rank ≤25 | Truth regained but ranks >25 or R@1 unchanged | Must not re-add removed (≥τ) spectra; invalid if any added spectrum is a removed rid or query-group member |
| **O-B** formula-isomer pool | Candidates = train molecules with truth's formula that have a spectrum (any adduct), scored with existing spectra | Given perfect formula, can spectra rank the truth among isomers? | — (pool is oracle-complete) | MRR stays low among ~35 isomers | Formula is oracle info (not at test time); must be labelled upper bound |
| **O-C** realistic neutral-mass pool | Candidates = molecules with neutral mass within measured ppm of query, all adducts | Realistic structure pool; is ranking within it the limit? | Truth often missing from pool | Truth present but poorly ranked | Tolerance must come from train ppm distribution, not tuned on outcomes |
| **O-D** truth + analogues | O-C pool restricted to truth + molecules with Tanimoto ≥ t to truth | Does structural neighbourhood confuse the scorer more than random isomers? | — | Truth displaced mainly by high-Tc analogues | Uses truth structure to build pool → oracle only |

Minimal first step: O-A on the EXP-007 population. It reuses EXP-007 pools and adds a few rids per
query, so it is cheap and directly separates the two readings of bottleneck #5.

---

## 3. Formula-constrained retrieval

- FACT: test schema has **no `molecular_formula`** and no MS1/isotope fields. Train has `molecular_formula`
  (118 inchikey14 have >1 formula string, i.e. mostly consistent).
- FACT: adduct + precursor give neutral mass; timsTOF error p99 4.4 ppm.
- FACT: median 35 isomers per timsTOF molecule; median 52 other molecules within 5 ppm.
  **INFERENCE:** once neutral mass is known to ~5 ppm, an exact formula narrows the structure pool only
  modestly (~52 → ~35 median). Formula is a weak *candidate* constraint here; most of the difficulty sits in
  ranking isomers. (Hypothesis until measured on pools with spectra.)

| Setting | Info source | Question answered | Legitimate at test time? |
|---|---|---|---|
| F-known | train `molecular_formula` of truth | Upper bound of formula filtering (O-B) | No (oracle) |
| F-inferred | formula from neutral mass (+ MS2 fragment consistency, e.g. SIRIUS-like) | Does inferred formula match truth often enough to beat a mass window? | Yes, if the inferrer uses no labels |
| F-unavailable | neutral-mass ppm window only | Realistic baseline pool (O-C) | Yes |

Cheapest decisive number (no scoring): for EXP-007 queries, |O-C pool| vs |O-B pool| and truth presence
in each. If they are similar, formula inference is low-value for candidate generation.

---

## 4. Structural-analogue retrieval

- FACT: `normalized_smiles` for all 275,810 train molecules parse in RDKit 2026.03.6 (0 failures); Morgan
  fingerprints, exact mass and Murcko scaffolds computed locally, no downloads.
- FACT (1000-molecule sample, seed 20260924): nearest-neighbour Tanimoto among train molecules:
  p10 .545 · p25 .596 · **median .661** · p75 .735 · p90 .791.
  Share with NN ≥ .85: **3.5%**; ≥ .70: **35.9%**; ≥ .50: 97.5%. Median count of molecules with Tc ≥ .5: 8.
- FACT: Murcko scaffolds: 62,526/183,191 timsTOF molecules (34%) have a scaffold that no other train
  molecule shares; median scaffold family size 3.
- FACT: best Tanimoto among other molecules within ±0.01 Da of the same molecule: median **.35**. Analogues
  are generally *not* at the same mass. Analogue retrieval therefore needs a mass-shift-tolerant search
  (Modified Cosine over a wide window), not the ±0.01 Da window.

**AN-1 (DESIGN OPTION):** remove all spectra of the target `inchikey14` (all libraries). Search a wide
neutral-mass window (e.g. ±X Da, X a design choice) with ModifiedCosine. Measure the Tanimoto between truth
and the top-k retrieved molecules against a **random-pool baseline** (same window, same k).
Question: do spectral neighbours carry structural information when the target is absent?
Supports usefulness if Tc@top-k is materially above the random-pool baseline. Invalid if the target's
connectivity survives under another inchikey14 (salts/charged forms). Check with an exact-structure
lookup before running.

---

## 5. Pseudo-C3 design space (DESIGN OPTIONS; not chosen)

Core constraint (FACT, §1c #10): a spectral-library ranker **cannot** return a molecule with no spectra.
A pseudo-C3 benchmark therefore needs two sources:
**structure universe** (candidates; may contain target) ≠ **spectral library** (evidence; target removed).
The first question it answers is whether any scorer beats the chance baseline inside a realistic structure
pool. It does not ask whether the library finds the target.

Common to all tiers: target selection from timsTOF molecules with ≥2 spectra (99.3% qualify); exclusion =
**all spectra of target inchikey14 in every library**. Candidate pool = train structures within query
neutral-mass ppm window, target included, identity hidden. Evaluation = MRR@25, R@k, plus Tanimoto of top-1
to truth. **Chance baseline** = H₂₅/n: ≈ .109 at n=35, ≈ .073 at n=52.

| Tier | Target selection | Extra exclusion | Allowed evidence | Frequency (sample NN Tc) | Interpretation |
|---|---|---|---|---|---|
| C3-Easy | NN Tc to a spectrum-bearing molecule ≥ a (e.g. .70) | none | all other spectra | ~36% at .70; only 3.5% at .85 | Analogue propagation ceiling |
| C3-Medium | NN Tc in [b, a) | none | all other spectra | bulk of population | Family-level inference |
| C3-Hard | any | remove spectra of all molecules with Tc ≥ b to target (neighbourhood restriction) | remaining spectra | constructible for all | Isomer ranking without neighbours |

Thresholds a, b are placeholders, to be set from the Tc distribution before any outcome is seen.
"Close analogues ≥ .85" is **rare (3.5%)**, so an Easy tier defined at .85 would be tiny.

Leakage controls: target structure must not enter any scorer feature. Remove duplicate connectivity
under different inchikey14. For Hard, remove neighbours from the *spectral library* only, not from the
structure universe (otherwise the pool changes too). Also check that the structure universe does not
embed library-order or ingest metadata that correlates with the target.

---

## 6. Approach hypothesis map (not ranked)

| Approach | Solves | Class | Current evidence for relevance | Must be true to beat baseline | Experiment first |
|---|---|---|---|---|---|
| Spectral-library retrieval | Match to known spectra | C1 (C2 partly) | EXP-001 .651 (C1) | — (baseline) | — |
| Learned spectral embeddings | Similarity robust to novelty | C2 | none yet | C2 ranking loss exists among truth-present queries | EXP-007 decomposition |
| Fingerprint prediction (CSI:FingerID-like) | Score structures without spectra | C3 | none; public claims only | Structure pools small enough, fingerprints predictable | Pseudo-C3 chance baseline + O-C |
| Formula-constrained search | Shrink pool | C2/C3 | isomers ≈ mass window (§3) | Formula inference narrows more than ppm window | |O-B| vs |O-C| count |
| Graph/molecular retrieval (analogues) | Use neighbours' spectra | C3 | NN Tc median .66 (structure only) | Spectral NN ⇒ structural NN | AN-1 |
| De novo generation | Structures absent from any DB | C3 beyond DB | none | Hidden test contains DB-absent molecules | Unknown class mix first |
| Hybrid retrieval + reranking | Fix ranking in rich pools | C1/C2 | EXP-002: richer pool hurt max-cosine | Reranker uses features beyond max cosine | EXP-003 full run |
| Ensembles | Complementary errors | all | none | Methods fail on different queries | Per-query overlap once ≥2 methods exist |
| Multi-spectrum query fusion | Use 1–9 spectra per molecule_id | all | EXP-001 add-on n=10 only (max-agg .675 vs best-single oracle .839) | Fusion beats best single spectrum | Fusion arm on Mode-B molecules with ≥2 spectra |

---

## 7. Decision tree after EXP-007 (conditions measurable; no numeric thresholds pre-set)

Define on genuine interventions (k ≥ 1) per τ:
`L_cand` = Σ ΔRR over queries where truth left the pool; `L_rank` = Σ ΔRR over queries where truth stayed
in the pool but dropped rank; `L_ctrl` = Σ ΔRR in C1-matched at the same k.

```
EXP-007 valid? (smoke semantics reproduced, 0 leakage) ── no ──► fix, rerun; nothing below applies
        │ yes
Is C2 MRR loss (k≥1) materially larger than bootstrap noise AND than L_ctrl?
        │
        ├─ yes ── L_cand materially larger than L_rank?
        │          ├─ yes ─► does truth still have library spectra outside the raw window?
        │          │          ├─ mostly yes ─► NEXT-A (window/adduct artefact; O-A)
        │          │          └─ mostly no  ─► genuine coverage loss; C2 ≈ C3-like for those; NEXT-C
        │          └─ no (L_rank dominates) ─► NEXT-B (ranking under novelty)
        │
        └─ no  ── is k / removed-evidence strength adequate (S1–S5 spread, k distribution)?
                   ├─ no  ─► intervention too weak; harder novelty (cross-adduct-only or all-same-adduct removal)
                   └─ yes ─► C2 not a bottleneck for this scorer; NEXT-C (pseudo-C3 feasibility)
Also: if C1-matched loss is comparable to C2 loss at matched k ─► pool-size confound; run EXP-003 E arm first.
```

---

## 8. Conditional next experiments (not approved)

| | NEXT-A | NEXT-B | NEXT-C |
|---|---|---|---|
| Trigger | L_cand ≫ L_rank and truth has out-of-window spectra | L_rank ≫ L_cand | Weak C2 effect with adequate k, or L_cand from genuine coverage loss |
| Question | Is C2 candidate loss an artefact of raw-m/z generation? | What displaces the truth once near-duplicates are gone? | Can anything beat chance in a realistic structure pool without target spectra? |
| Hypothesis | Neutral-mass generation restores truth; ranking then decides | Displacers are isomers / multi-spectrum molecules / cross-adduct | Analogue propagation > chance baseline |
| Population | EXP-007 600 × τ, k≥1 | Same | ~200 timsTOF molecules, stratified by NN Tc |
| Intervention | O-A (add only truth's out-of-window remaining spectra), then full neutral-mass pool | Tag displacers; arms: ppm window, aggregation variants (max vs count-normalised) on fixed pools | Pseudo-C3 (§5) with AN-1 scorer |
| Baseline | EXP-007 C2 | EXP-007 C2 | Random ranking in same pool (H₂₅/n) |
| Metrics | candGen, R@k, MRR, ΔRR by case | same + displacer type mix | MRR, R@k, Tc@1 vs truth |
| Leakage | Never re-add removed or group rids | Scorer features exclude identity | Exclude all target spectra; check duplicate connectivity |
| Interpretation | Recovery ⇒ fix generation before C3 | Specific displacer type ⇒ targeted ranking fix | ≈ chance ⇒ need a learned structure scorer (fingerprints) |
| Complexity | Low (reuse pools; recall step needs no scoring) | Low–medium (re-rank stored scores) | Medium (new pool builder + wide-window search) |

Precondition for all three: resolve §0, because the local test file cannot validate anything.

---

## 9. Minimum information needed from the EXP-007 full run

Per query × τ (C1, C2, C1-matched):
- k and removed rids
- truth present (bool)
- truth rank
- truth best score, and max sim of the evidence that remains after removal
- pool size in spectra **and in molecules**
- top-1 wrong molecule: score, adduct, whether formula equals truth, and its number of spectra in the window
- whether truth has library spectra outside the raw window (other adducts)
- stratum

Aggregates:
- L_cand, L_rank, L_ctrl per τ and per stratum, restricted to k≥1
- bootstrap CI on ΔMRR
- the transition matrix with case 3 split into "truth left pool" vs "present, rank >25"

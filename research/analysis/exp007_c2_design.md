# EXP-007 — Genuine Pseudo-Class-2 Experiment: Design

Date: 2026-09-22

Status: **DESIGN + DATASET AUDIT ONLY.** No production pipeline change, no smoke test,
no full run, no `experiments.md` update. Deliverables: this design doc,
`results/exp007_c2_design/design_spec.json` (machine-readable spec), a PROPOSED
DECISION (DEC-007) in `research/decision_log.md`, and a state update in
`research/CASMI_RESEARCH_STATE.md`. Do NOT proceed to C3, do NOT train models.

Builds on: `research/CASMI_RESEARCH_STATE.md`, `research/decision_log.md` (DEC-006,
DEC-005, DEC-004, DEC-003), `research/05_class1_to_class2_design.md`,
`research/06_matched_c1_control.md`, `research/analysis/exp004_c2_failure_analysis.md`.

---

## 1. Research question

> When a molecule is known/represented in the search library but the *query spectrum
> is genuinely unseen* — i.e. every retained spectrum of that molecule differs from
> the query by at least a data-driven similarity margin — how much does
> molecule-identification performance degrade, relative to the same query with its
> full retained same-molecule evidence available?

EXP-004 failed to answer this because it was not a distinct computation from EXP-001
Mode B (DEC-006): it reused the identical whole-group exclusion, candidate generation,
and scorer, so C1 and C2 were bit-identical. EXP-007 fixes that by making the novelty
filter a **protected, experiment-controlled intervention**: the C2 condition actually
*removes* retained evidence, so the C1-vs-C2 difference is a real experimental
manipulation, not a relabeling.

Primary secondary question: which failure taxonomies change as novelty increases —
candidate-generation ceiling (unreachable), ranking/close-loss, ranking/decisive-loss —
and does performance degrade smoothly with novelty or cliff at some threshold?

## 2. Why a new experiment is needed (premise, from DEC-006)

This section is restated from the decision log and prior analysis; nothing here is new.

- EXP-004 ("C2") was, by its own documented construction (§7 of
  `research/05_class1_to_class2_design.md`), EXP-001 Mode B re-run on the same 400
  queries minus 3 near-duplicates, with the *identical* exclusion, candidate rule, and
  scorer. The full query-level join
  (`research/analysis/exp004_c2_failure_analysis.md` §5) showed **0 rank transitions,
  0 n_candidates differences, 0 score differences, 100% shared failures** across the
  397-query shared population.
- Therefore EXP-004's near-equal MRR (0.6519 vs 0.6510) must NOT be cited as "C2
  doesn't matter" (DEC-006 §Decision a).
- Leveraging Mode B *as-is* as the operational pseudo-C2 benchmark is **permitted** by
  DEC-006 §Decision b, but the user's explicit directive for this task is to design a
  construction with an actual experimental manipulation of held-out spectral novelty.
  This design does that: **the C2/novelty restriction is a covariate that EXP-007
  chooses per query, and the retained-evidence set is sliced by it**, rather than
  whatever Mode B happened to retain.

## 3. Dataset audit (Phase 1–4; full numbers in audit scripts)

All raw numbers below are recomputed from `train.parquet` via DuckDB (DEC-001) by
three executable, seed-pinned scripts. Rows ↔ `rid` via
`ROW_NUMBER() OVER () - 1` under `PRAGMA threads=1`, verified against known
(rid, inchikey14, adduct) triples before use (reproducibility precondition).

| Script | Output | What it measures |
|---|---|---|
| `research/scripts/exp007_audit_phase1.py` | `results/exp007_audit_phase1.json` | Row/molecule/adduct/CE/nominal-spectrum census, full + timsTOF |
| `research/scripts/exp007_audit_phase2.py` | `results/exp007_audit_phase2.json` | Within-molecule pairwise ModifiedCosine distributions, by same/cross adduct and same/diff CE |
| `research/scripts/exp007_audit_phase3.py` | `results/exp007_audit_phase3.json` | Query-centric: 600 sampled timsTOF queries, retained same-adduct sibling counts + similarity |
| `research/scripts/exp007_design_sizing.py` | `results/exp007_design_sizing.json` | Strict eligibility (all retained same-adduct evidence below τ) + strata sizes |

### 3a. Population census (Phase 1)

- Full train: **2,539,608 rows, 275,810 molecules**; timsTOF subset: **1,154,969
  rows, 183,191 molecules**.
- Spectra per molecule (full): median 6, p90 14, max 1,468.
- Nominal spectra per molecule (adduct, precursor rounded to 4 dp, CE string): median
  5, max 244.
- Per molecule: adducts median 2; distinct collision-energy strings median 4; distinct
  ingest libraries median **1** (p90 2, max 9).
- Molecules with ≥2 spectra (full): 258,777; same-adduct ≥2 (full): 255,294.
- timsTOF-specific: ≥2 spectra 181,818; ≥2 same-adduct spectra **181,651**;
  ≥2 same-adduct *and* different-CE **181,650**; ≥2 same-adduct *nominal* spectra
  only **193** — i.e. nearly all same-adduct multi-spectra differ only by CE.

### 3b. Within-molecule pairwise ModifiedCosine (Phase 2, sample, seed 20260922)

Config byte-identical to EXP-001/004: `ModifiedCosineGreedy(tolerance=0.1,
mz_power=0.0, intensity_power=1.0)`.

| Group | n pairs | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| all within-molecule pairs | 10,987 | 0.003 | 0.193 | 0.794 | 0.977 |
| same-adduct | 4,119 | 0.501 | 0.795 | 0.969 | 0.998 |
| same-adduct, diff CE | 4,098 | 0.501 | 0.793 | 0.969 | 0.997 |
| same-adduct, same CE | 21 | 0.980 | 0.999 | 1.000 | 1.000 |
| cross-adduct | 6,868 | 0.000 | 0.013 | 0.194 | 0.793 |

Readings, as hypotheses to be used in design (not facts about causality):
- **Same-adduct, different-CE spectra are genuinely but *mildly* novel** (median 0.79;
  25% of pairs < 0.50).
- **Same-adduct, same-CE pairs are near-identical** (median 0.999) → the within-adduct
  novelty axis is essentially the CE axis.
- **Cross-adduct pairs are mostly unreachable-in-score** (median 0.013), consistent
  with the 12-query candidate-generation ceiling under Variant A; but p90 = 0.79 shows
  a long tail where shift-matching does work — never assume cross-adduct ⇒ low
  similarity.

Molecule-level: **293/300** (max) sampled molecules have ≥1 same-adduct pair < 0.90;
**281/300** have ≥1 same-adduct pair < 0.80 — so strict novelty strata are well
populated at the molecule level.

### 3c. Query-centric eligibility (Phase 3)

600 deterministic timsTOF spectra sampled as candidate queries (seed 20260922).
For each query, retained same-adduct siblings = same (molecule, adduct,
precursor-rounded, CE-string family) minus the query's own metadata group
(inchikey14, adduct, precursor_mz 4 dp, num_peaks) — the EXP-001 exclusion rule.

- 490/600 (81.7%) have ≥1 retained same-adduct sibling.
- Of those, retained-sibling count: 1 → 90; 2 → 181; 3 → 218; 4 → 1.
- 110/600 (18.3%) have **zero** retained same-adduct evidence → this is the
  cross-adduct-only / candidate-generation-ceiling population (mirror of EXP-004's 12
  on the EXP-001 population; here measured on the wider timsTOF pool).

### 3d. Strict eligibility sizing (Phase 4, design-specific)

Strict eligibility for "novelty cutoff τ": query has ≥1 retained same-adduct sibling
**and the maximum** similarity to any retained sibling is < τ (i.e. ALL retained
same-adduct evidence is below τ).

| τ | # eligible (of 600) | fraction |
|---|---:|---:|
| 0.50 | 33 | 0.055 |
| 0.60 | 72 | 0.120 |
| 0.70 | 120 | 0.200 |
| 0.75 | 150 | 0.250 |
| 0.80 | 175 | 0.292 |
| 0.85 | 203 | 0.338 |
| 0.90 | 253 | 0.422 |
| 0.95 | 313 | 0.522 |

max_sim among queries with retained same-adduct evidence: p10 0.543, p25 0.702,
median 0.889, p75 0.977, p90 0.998 → the population density is smooth across the
novelty axis; **no knife-edge threshold exists**, which justifies continuous
stratification rather than a binary "novel vs not".

## 4. Design constraints (binding, from prior decisions)

C1–C4 mirror the standing rules; do not violate without a DECISION.

1. **Variant A candidate generation** (raw precursor_mz ±0.01 Da, all-train library),
   NOT Variant B (DEC-004). EXP-007 is a *spectral-novelty* experiment; the 12-query
   cross-adduct ceiling is a documented limitation, not a new treatment to fix.
2. **Whole metadata-group exclusion** = `(inchikey14, adduct, precursor_mz, num_peaks)`
   (EXP-001). Preserved.
3. **Same scorer/config** byte-identical to EXP-001 (tolerance 0.1, mz_power 0.0,
   intensity_power 1.0, molecule-level max-score aggregation, top-25).
4. **Queries are timsTOF** (test is 100% timsTOF); eligibility audits sample timsTOF.
5. **Builder may know identity; retriever must not receive inchikey/inchikey14/SMILES**
   (documented in 05 §5 field table).
6. **No production pipeline (`src/`) change**, no `experiments.md` update yet.

## 5. Novelty definition and measurement (precise)

- **Novelty metric:** per-query, `max_sim = max` over retained same-adduct siblings of
  ModifiedCosine(query, sibling) using the same scorer/config as retrieval. Same-CE
  pairs would trivially hit ~1.0; that is the *near-duplicate* extreme the metric is
  designed to flag. Cross-adduct retained evidence is **excluded from the novelty
  metric's denominator but retained in the library** (see §7).
- **Why ModifiedCosine and not direct cosine:** clone of the §3b justification —
  direct cosine under-measures cross-adduct similarity; ModifiedCosine is the tool the
  project validated for cross-adduct work. (For same-adduct pairs they agree closely;
  the metric is applied well within the same-adduct regime.)
- **Thresholds are data-driven, not blind 0.90/0.95.** Because the max_sim density is
  smooth (3d), the design uses **continuous strata** (bins) rather than a binary
  novelty flag, exactly per the master prompt's caution against a single magic
  threshold.
- **The metric is a covariate, not a post-hoc outcome filter.** The audit (§3c–d)
  picks queries and the design then *slices retained evidence*; it does not
  select-on-success.

## 6. Candidate designs (three alternatives)

### Design A — Spectrum-level holdout with controlled novelty ceiling (RECOMMENDED)

- For each query, define the retained same-adduct evidence set `R`.
- **C1 (control):** library = full train minus query group (Mode B as-is).
- **C2(τ):** library = C1 library **minus every retained same-adduct sibling with
  ModifiedCosine(query, sibling) ≥ τ**. So the *only* same-molecule same-adduct
  evidence left has similarity < τ.
- Condition-pairing per query: same query, same candidate generation, same scorer,
  same seeds; the ONLY free variable is whether high-similarity retained evidence is
  physically present in the reference library. **That is the experimental
  intervention, and it is executed (removal), not merely audited** — fixing EXP-004's
  fatal flaw.
- Multiple τ levels per query (see §9 strata) → per-query dose–response: MRR@25 end
  states, margin shapes, and rank transitions read out against a graded novelty axis.

**Why clearest causal interpretation:** the intervention removes exactly one kind of
library content (same-adduct high-similarity retained evidence), holding everything
else fixed; any C1→C2 change is attributable to the removal. It is not a relabeling,
not a post-hoc stratification, not random candidate deletion, not an arbitrary
threshold — it is the *defined, data-driven novelty treatment*.

### Design B — Condition-held-out (whole CE-string held out)

- For each query, C2 removes **the whole collision-energy condition string** of the
  query from the library (all SameCe siblings), i.e. retained evidence = same molecule,
  same adduct, *different CE only*.
- Rationale: real-world "unseen spectrum" is often a new CE acquisition for an
  otherwise-known molecule.
- Weakness (documented, measured): same-adduct diff-CE pairs still reach ModifiedCosine
  ≥ 0.97 at p90 (§3b) — the "condition-held-out" removal does not by itself guarantee a
  *low* similarity ceiling; some diff-CE retained spectra are still quasi-duplicates of
  the query. So Design B must be combined with the §3d similarity check to define
  levels, otherwise it under-tests novelty.
- Verdict: **subsumed by Design A** (A with τ tuned per the same-adduct diff-CE
  distribution already implements "different-CE-only retained evidence"; it simply also
  enforces the maximum-similarity ceiling that B leaves loose). Keep as an optional
  *coarser* slice label on Design A strata (query-vs-retained CE match already recorded
  as a covariate in the audit).

### Design C — Leave-one-evidence-family-out (metabolic/degradation family)

- Hold out an entire *evidence family* for a molecule in C2 — e.g. one precursor-adduct
  (cross-adduct holdout) or one ingest-library family — leaving only other families.
- Rationale: family-level novelty (the molecule was re-measured under a different
  condition class).
- Weakness (documented, measured): cross-adduct retained evidence is **structurally
  unreachable under Variant A** (the 12-query ceiling, §3c: 110/600 queries have NO
  same-adduct retained evidence); ingest-library families are ~1 per molecule (§3a),
  so C leaves near-zero *reachable* evidence for most molecules — the intended "novel
  family" becomes an empty candidate pool, which is a C3-flavored (absent-molecule)
  construction, not a C2 one.
- Verdict: **rejected as the C2 construction**; the cross-adduct-only population
  (12/400 in the EXP-001 population; 18.3% of the wider timsTOF pool) is kept as an
  explicit, separately-reported diagnostic stratum (`unreachable`), not mixed into the
  C2 headline.

## 7. Recommended construction (Design A) — full specification

Per query `q` (timsTOF), with true molecule `M`:

- `qgroup(q)` = the query's whole metadata group — always excluded from the library.
- `R(q)` = all retained same-molecule **same-adduct** siblings (`(M, adduct, precursor
  4dp, CEfamily)` minus `qgroup`).
- `X(q)` = all other same-molecule spectra (cross-adduct, different precursor window) —
  retained in the library in ALL conditions (they are Mode B's normal evidence; under
  Variant A most are unreachable, which is the 12-query ceiling, documented).

Conditions, per query:

- **C1** — library = `train` ∖ `qgroup`. (Bit-identical to EXP-001/EXP-004 Mode B for
  queries shared with that population; used here as the "known, fully retained"
  control.)
- **C2(τ)** for τ in the design grid `{0.50, 0.60, 0.70, 0.80, 0.90, 0.95}` —
  library = `train` ∖ `qgroup` ∖ {r ∈ R(q) : sim(q, r) ≥ τ}.
  Nothing else changes. All queries are run in **both** C1 and every C2(τ) for which
  they are eligible (non-empty retained set under the cutoff).
  **Eligibility for C2(τ): `max_sim(q) < τ` and ≥1 retained same-adduct sibling**
  (§3d) — otherwise the condition is empty and the query simply is not run at that
  cutoff (reported in the "exhausted-eligibility" table, not silently dropped).

- **C1-matched (candidate-pool-size control):** EXP-003 demonstrated that candidate
  pool *size itself* affects ranking (DEC-004: pools grew 2.74x and MRR fell). Since
  C2(τ) removes `k` true-molecule spectra from the pool, pool size shrinks by `k` and
  could alone explain any MRR drop. Therefore for each C2(τ) run we ALSO run
  **C1-matched(τ)**: C1 library minus `k` **randomly-selected, same-window,
  guaranteed-wrong-molecule** spectra (selected from other molecules' spectra within
  the query's precursor window, so they are real candidates in C1), where `k = [C2(τ)
  removals]`. If MRR(C2(τ)) ≈ MRR(C1-matched(τ)), the novelty effect is not separable
  from pool-size noise; if MRR(C2(τ)) < MRR(C1-matched(τ)), the effect is specific to
  removing *the true molecule's similar evidence* rather than pool shrinkage. This
  directly answers "did we change the answer's evidence (novelty) or just the candidate
  count (size)?"

## 8. Leakage controls (checklist — must be verified in smoke test)

- [x] (design) query's whole metadata group excluded from library in all conditions
  (EXP-001 rule, unchanged).
- [x] (design) near-duplicate policy: queries with simultaneous same-adduct same-CE
  siblings that are near-identical (sim ≥ 0.95) to the query are the **most novel
  strata** under C2's removal; they are not excluded from the population but their
  retention is *by construction* limited — no hidden near-dup leak.
- [x] (design) molecule identity hidden from prediction path: candidate generation
  consumes only `ms2_mzs`, `ms2_normalized_intensities`, `precursor_mz` (+ identical
  metadata fields as EXP-001); inchikey/inchikey14/SMILES are used only to build the
  library exclusion sets and the ground-truth evaluator, never fed to scoring. Same
  field table as 05 §5; no new leakage surface.
- [x] (design) no molecule-level prototype/embedding caching (pipeline scores
  per-candidate spectra; no cached profile to leak query's own row into).
- [ ] (to verify in smoke) removal integrity: for every C2(τ) query, assert each
  removed rid is a retained same-adduct sibling with sim ≥ τ computed independently by
  the audit path, and that the remaining set is `R ∖ removed` with non-empty
  intersection at eligibility boundaries.
- [ ] (to verify in smoke) C1-matched wrongness: assert each removed filler spectrum's
  molecule ≠ query molecule.

## 9. Strata (difficulty stratification; continuous, data-driven)

No binary novelty flag. Strata on the max-sim axis (from §3d bins) plus two
non-novelty strata:

| Stratum | Definition | n in 600-sample | Role |
|---|---|---|---|
| S1 | max_sim ∈ [0, 0.50) | 33 | Deep novelty; expected hardest C2 range |
| S2 | [0.50, 0.70) | 87 | Genuine novelty |
| S3 | [0.70, 0.85) | 83 | Mild novelty (CE-driven) |
| S4 | [0.85, 0.95) | 110 | Near-margin novelty |
| S5 | [0.95, 1.0] | 177 | Near-duplicate retained evidence (novelty floor) |
| UR | no retained same-adduct evidence | 110 | unreachable/candidate-generation ceiling — reported separately |

C1 is run on **all** 490 same-adduct-retained queries (all strata S1–S5); C2(τ) and
C1-matched(τ) run on the eligibility-subsets per τ. Stratum S5's C2 removals are at
τ ≥ 0.95 — this is where near-duplicate evidence is actively removed, making the
"unseen spectrum" maximally effective.

Stratification variables recorded for every query (covariates, in the per-query
record): adduct, CE-string family cardinality, num_peaks, precursor_mz, max_sim,
argmax-sibling CE match (same/diff), n_retained, n_removed_at_tau, candidate_count.

## 10. Metrics (primary/secondary/diagnostics)

- **Primary: MRR@25** per condition (C1, each C2(τ), each C1-matched(τ)), reported on
  the shared eligible query set per τ so comparisons are like-for-like.
- **Secondary: Recall@1/5/10/25**, candidate-generation recall per condition.
- **Diagnostics (per query, all conditions):** rank, n_candidates, best_true_score,
  best_wrong_score, margin, top-100 candidate scores, stratum tag, retained-count,
  removed-count, max_sim, argmax-sibling CE match.
- **Uncertainty:** bootstrap 95% CIs on MRR@25 per condition (query-level resampling,
  per the 05 §9 precedent).

## 11. Interpretation matrix (pre-registered; per-query, per τ)

For each query, classify the C1→C2(τ) transition:

| Case | C1 | C2(τ) | Reading |
|---|---|---|---|
| 1 | rank 1 | rank 1 | Novelty harmless at this level |
| 2 | rank 1 | rank ≤ 25, >1 | Novelty degrades ranking but identifier survives |
| 3 | rank ≤ 25 | rank > 25 | Novelty causes failure (close→lost) |
| 4 | rank > 25 | (any) | Pre-existing failure; novelty not the driver |
| 5 | not in candidates | (n/a) | candidate-generation ceiling (recorded, not counted as novelty effect) |

Aggregate the matrix per τ and per stratum. Report:
- `P(regress at τ)` = share of S# queries moving rank-1→>1 or ≤25→>25;
- whether the C1→C2 MRR gap is explained by pool-size control (C1-matched) or is
  specific to true-evidence removal;
- monotonicity of MRR vs τ (the §13-across-strata smoothness check from EXP-004 is
  superseded here by an actual dose–response with a matched control).

## 12. Smoke-test protocol (proposed; NOT yet approved/run)

- ~28–32 queries, deterministic seed 20260922, stratified: pick 5–6 per S1–S5 and
  none from UR (UR handled in full run as a report-only stratum).
- For each: run C1, all eligible C2(τ), all eligible C1-matched(τ).
- Checks (all must pass before full run):
  1. **Population**: query set matches the audit's expected stratum sizes; rid
     reconstruction re-verified.
  2. **Removal integrity**: every removed rid is a retained same-adduct sibling with
     sim ≥ τ (independent recompute).
  3. **Leakage**: zero occurrences of query's inchikey14 in any scoring input;
     filler spectra in C1-matched verified wrong-molecule.
  4. **Reproducibility**: run the full smoke twice; byte-identical per-query results
     (rank, scores).
  5. **C1/C2 isolation**: C1 library ⊇ C2 library ⊇ (minus removals only); candidate
     sets differ exactly by removed rids.
  6. **Candidate-gen sanity**: candidate_count(C2(τ)) = candidate_count(C1) − k exactly
     where k = number of removed true-molecule spectra that were in C1's candidate pool;
     k-truth check via audit path.
  7. **Scoring sanity**: spot-check a few ModifiedCosine scores against an independent
     recompute; margins/rank transitivity over conditions.
- Output: `results/exp007_c2_design/smoke.json`.

## 13. Full-run protocol (proposed; NOT yet approved/run)

- Eligible population: all (approximately) 1,154,969-era timsTOF spectra that satisfy
  ≥1 retained same-adduct sibling — but for a *feasible* controlled benchmark, sample
  **600 queries deterministically** (seed 20260922, same as audit; the audit's 600
  may be reused as the population, re-verified, so audit and run share the exact query
  set — no new sample drift). All 600 run in C1; run C2(τ) per eligibility per τ
  (approximately 72–408 queries per τ, §3d); run C1-matched(τ) on a
  ≤60-query subsample (the pool-size control is about *mechanism*, not population
  coverage; subsample stratified).
- Memory-safe batched loading (single bulk fetch, per EXP-002/003 lesson); checkpoint
  after every 50 queries.
- Write: `results/exp007_c2_design/full_run.json`, `results/exp007_c2_design/run.log`.
- Approval gate: full run only after smoke passes every check in §12.

## 14. What success / failure means (evidence-based exit criteria)

- **Evidence that a C2 bottleneck exists:** MRR(C2(τ)) drops monotonically as τ
  decreases; the drop survives the C1-matched pool-size control (i.e. it is specific
  to removing the true molecule's similar evidence); the regress-rate Case-3 grows
  with novelty. Total runtime ~ likely 2–3x EXP-004's (C1 + several C2) — manageable.
- **Evidence that novelty is harmless at these levels:** flat MRR across τ,
  including with the matched control, for all strata — then the *reason* Mode B was
  already C2-like (median retained sim 0.055 in the EXP-004 audit) needs re-reading.
- **Not a valid outcome:** reporting a C1(control-removed)-only effect as "novelty
  effect" (that's pool-size noise — the C1-matched arm exists to catch this).

## 15. Expected eligible population for the full run (from §3d)

- 600-query deterministic sample: 490 same-adduct-retained (UR excluded from
  headline). C2(τ) eligible per τ: ~72 (0.60) to ~408 (0.95). At τ=0.70: ~120
  queries/nominal ~20% of the sampled pool — a statistically usable stratum at the
  full-sample scale; at τ=0.50~33 — thin but usable for a specific deep-novelty claim
  with bootstrap CIs.
- For the actual competition's needs (26 ranked SMILES etc.), MRR@25 of 0.60–0.90 at
  these strata plus the C1 0.65 baseline gives a calibrated view of the C2
  contribution envelope.

## 16. Threats to validity (pre-registered limitations)

1. **Cross-adduct novelty is untestable under Variant A.** The UR stratum (110/600;
   analog of 12/400) is structurally unreachable; this experiment cannot quantify
   cross-adduct C2 novelty without Variant B/D, which DEC-004 defers.
2. **Pool-size confound handled via control, not eliminated.** C1-matched(τ) cannot
   replicate the *full* pool-size trajectory of real C2 because removing true-molecule
   spectra also removes their scores; the control holds size constant while netting out
   content — the residual is the best available read, not a perfect counterfactual.
3. **Retained same-adduct evidence is sparse** (mostly 1–3 siblings, §3c), so per-query
   novelty levels come from few specific spectra; a handful of pathologies dominate
   some strata. Mitigations: bootstrap CIs, report per-query rids, keep strata
   disaggregated.
4. **Novelty axis ≠ chemical/common sense novelty.** ModifiedCosine similarity is the
   operational novelty measure; it is the right tool for *this* retrieval pipeline and
   is not a claim about structural chemistry distance.
5. **100% timsTOF** — no cross-instrument C2 is tested.
6. **Training-data queries** — this is a controlled benchmark on known molecules, not
   the hidden test set.

## 17. Reproducibility requirements (binding)

- Python/DuckDB only (DEC-001). No pandas/pyarrow parquet reads.
- Rid reconstruction: `ROW_NUMBER() OVER () - 1` under `PRAGMA threads=1`, verified
  against the two triples in `exp007_audit_phase2.py` (and 4 in the c2 analysis script)
  before any downstream number. Audits already passed this gate.
- Seeds: query sample 20260922 (audit + full run identical); smoke seed 20260922;
  bootstrap resampling seed local to each metric computation (documented).
- Scorer config constants fixed in one module, reused by audit + conditions
  (`ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)`).
- Every `results/exp007_c2_design/*` file emits a `generated_by` + `seed` + runtime
  header.

## 18. Explicitly NOT done yet (boundaries)

1. No smoke test run, no full EXP-007 run, no `experiments.md` entry.
2. No C3 design, no model training, no `src/` pipeline change.
3. No adoption of Variant B or adduct-curated candidate generation (DEC-004).
4. The UR (unreachable/cross-adduct-only) population (110/600; 12/400 analog) is
   reported, not fixed; whether Variant B in a *separate* experiment could recover it
   is explicitly out of scope here.

## 19. Verdict

**DESIGN READY FOR REVIEW.** The construction is a genuine experimental intervention
(fixed EX-004's fatal flaw), the novelty axis is continuous and data-driven (no blind
0.90/0.95 cutoff), pool-size artifact is controlled (C1-matched arm), population is
measured and sufficient for the planned strata, and all prior decisions (Variant A,
whole-group exclusion, same scorer) are preserved. Smoke test is proposed (§12), not
approved; the user decides afterwards.

---

Files:
- `results/exp007_c2_design/design_spec.json` (machine-readable spec below).
- Audit artifacts: `results/exp007_audit_phase1.json`, `results/exp007_audit_phase2.json`,
  `results/exp007_audit_phase3.json`, `results/exp007_design_sizing.json`.
- Proposed decision: `research/decision_log.md` DEC-007.
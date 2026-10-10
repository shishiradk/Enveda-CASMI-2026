# Decision Log — Enveda CASMI 2026

## DEC-001 — Use `duckdb` instead of `pandas.read_parquet`/`pyarrow.parquet.read_table` for all data access

Date: 2026-09-20

Decision:
Standardize all data loading on `duckdb` (see `src/data/loader.py`), querying
`train.parquet` / `test.parquet` directly by path.

Reason:
`pandas.read_parquet()` and `pyarrow.parquet.read_table()` both fail on these exact files in
this environment with `OSError: Repetition level histogram size mismatch` (pyarrow 19.0.0).
`duckdb` reads both files with no issue and was used for every query in
`research/00_dataset_report.md` and `research/01_dataset_recon.ipynb`.

Evidence:
Direct reproduction in this session; documented in `research/00_dataset_report.md` §1.

Alternatives considered:
Upgrading/downgrading pyarrow, using `fastparquet`. Not pursued yet since `duckdb` already
works and is fast on these file sizes; may revisit if the final Kaggle notebook environment
has a different pyarrow version that doesn't hit this bug and a pandas-native path is
preferred there.

Expected impact:
Unblocks all further data work. Low risk — duckdb output converts to pandas DataFrames
via `.fetchdf()` so downstream code is otherwise unaffected.

---

## DEC-002 — Treat `enveda-180` (timsTOF) as the primary hypothesized Class 1 matching library, pending a real experiment

Date: 2026-09-20

Decision:
Working hypothesis only (not yet acted on): prioritize the `enveda-180` library (timsTOF,
~183k molecules) as the most likely source of direct spectral matches for the test set,
rather than treating all 2.5M train spectra as equally relevant to Class 1 library search.

Reason:
`test.parquet` is 100% `instrument_type == 'timsTOF'`. Within train, `timsTOF` is
essentially synonymous with `enveda-180` (99.9% of timsTOF rows), and `enveda-180` is ~97%
disjoint (by molecule) from every other library in train (GNPS, MassBank, MoNA, RIKEN,
pluskal_ms2, spectraverse, msdial, drug_plus, masaryk). Test's adduct set and
collision-energy-string formatting also match `enveda-180`'s conventions more closely than
the other libraries'.

Evidence:
`research/00_dataset_report.md` §5, reproduced in `research/01_dataset_recon.ipynb` §4.

Alternatives considered:
Treating the full 2.5M-row train set as one undifferentiated library for Class 1 search.
Rejected as the default because it would dilute/slow retrieval with ~1.39M spectra from a
near-completely disjoint molecule population, without evidence that they help Class 1
matching specifically (they may still matter for Class 2/3).

Expected impact:
If confirmed, Class 1 baselines should search `enveda-180` (+ `enveda-np-examples`) first,
and the non-timsTOF libraries should be evaluated separately for their actual contribution
(if any) rather than assumed to help. **This decision must be validated by EXP-001 (proposed,
not yet run) before being relied upon** — it is currently a hypothesis derived from metadata
correlation, not a measured retrieval result.

**UPDATE 2026-09-20 (post EXP-001): hypothesis not confirmed as stated — superseded by DEC-003.**

---

## DEC-003 — Search the full train set for Class 1 retrieval, not just `enveda-180`; the precursor window is the real filter

Date: 2026-09-20

Decision:
Do not restrict Class 1 candidate generation to `enveda-180`/`enveda-np-examples`. Search
the full train set (library variant "C" in EXP-001). This reverses DEC-002's working
restriction.

Reason:
EXP-001 (`research/02_class1_coverage.md`) measured all three library variants head to
head under identical, leakage-safe conditions and found library choice made essentially
no difference: MRR@25 spread of 0.0008 (Mode B) to 0.0025 (Mode A) across variants — well
within sampling noise for n=200–400, and in Mode B the full-train variant was marginally
*better*, not worse, the opposite of DEC-002's predicted direction. The mechanism: the
0.01 Da precursor window is a far stronger filter than library identity — non-timsTOF
candidates mostly never enter the candidate pool for a timsTOF query regardless of which
library is nominally searched, because their reported precursor_mz rarely lands within
0.01 Da of the query's even for the same molecule. Restricting to `enveda-180` therefore
added complexity without a measurable benefit, and searching the full train set recovered
marginally more candidates in Mode B's candidate-generation recall (97.0% vs 96.5%) at no
measured cost.

Evidence:
EXP-001, `research/02_class1_coverage.md` §7–9, `research/experiments.md` EXP-001 entry.

Alternatives considered:
Keep restricting to `enveda-180` for speed. Rejected — the runtime difference between
library variants in EXP-001 was modest (fetch/score time driven mostly by candidate count,
which only grew from median ~400 to ~690 for Mode B, not a large multiple) and not worth
giving up the (small, but real) extra candidate-generation recall.

---

## DEC-004 — Do NOT adopt adduct-aware (neutral-mass) candidate generation as-is; ranking is the actual bottleneck it exposed

Date: 2026-09-21

Decision:
Reject Variant B (adduct-aware neutral-mass candidate generation, `research/03_exp002_design.md`)
as a drop-in replacement for the raw-precursor-mz candidate filter, at least paired with
the current molecule-level max-Modified-Cosine-score ranking. Do not proceed to a
ranking-side fix or a refined candidate-generation variant without separate review.

Reason:
EXP-002 (`research/03_exp002_results.md`) measured Variant B head-to-head against
Variant A on the identical, leakage-safe Mode B query set. Candidate-generation recall
reached a perfect 100% (from 97.0%), confirming the EXP-001-motivated hypothesis that
adduct-naivety was hiding real candidates, and matchms's Modified Cosine shift-matching
was confirmed (per-candidate, not assumed) to meaningfully contribute for 9 of the 12
recovered queries. However, final ranking metrics net declined: MRR@25 fell from 0.6510
to 0.6178 (-0.0332), Recall@1 fell from 0.535 to 0.490 (-4.5pp), and only Recall@25 saw
a marginal gain (+0.005). Of the 388 queries that already had the correct candidate
under Variant A, 84 got worse ranks, 30 lost rank-1 status, and 8 fell out of the
top-25 entirely — the larger, more heterogeneous candidate pool (median growth 2.74x
for these queries) degrades the simple max-score ranking more than the 12 recovered
candidates improve it. This is a clean instance of the general finding the master
prompt's interpretation rule anticipated: candidate generation was not the sole
bottleneck, and fixing it in isolation exposed that ranking does not scale gracefully
to a larger candidate pool.

Evidence:
EXP-002, `research/03_exp002_results.md` §5–9, `research/experiments.md` EXP-002 entry.

Alternatives considered:
(a) Adopt Variant B anyway on the strength of the candidate-generation and Recall@25
numbers alone. Rejected — this is exactly the "declare success merely because
candidate recall increases" trap the experiment was explicitly designed to avoid;
MRR@25 and Recall@1 are the metrics that matter and both declined.
(b) Immediately build a precision-focused refinement (Variant C's adduct-pair
curation, or a tighter neutral-mass tolerance) to shrink the noise growth. Not done
yet — untested whether it would preserve the 8 net top-25 gains while avoiding the
ranking cost; proposed as a follow-up, not executed.
(c) Immediately build a ranking-side fix (calibration, tie-break, pre-scoring
candidate cap). Not done — this touches the ranker, explicitly out of scope for
EXP-002 per instruction, and needs its own separate, reviewed experiment design.

Expected impact:
The Class-1 pipeline (EXP-001's Variant A / DEC-003 configuration) remains the current
baseline. No pipeline change is made as a result of EXP-002. The finding reframes the
next research question from "is candidate generation the bottleneck" (answered: partly,
and now fixed for those specific cases) to "how do we rank a larger, chemistry-correct
candidate pool without degrading queries that were already correct" — a ranking
question, not a candidate-generation question, and requires separate design and
approval before implementation.

Expected impact:
Simpler pipeline (one library, not a curated subset) with no measured downside. This does
not mean the non-timsTOF libraries are useless overall — they were not tested for Class 2
(learned/instrument-invariant similarity) or Class 3 (broadening structural/formula
evidence), where they may still matter; DEC-003 is scoped to Class 1 precursor-filtered
retrieval only.

---

## DEC-005 — Correct a verified bug in EXP-004's Scenario audit; confirm EXP-001 Mode B's leakage control via a matched control; still await approval for the EXP-004 C2 smoke test

Date: 2026-09-21

Decision:
(a) Correct `research/05_class1_to_class2_design.md`'s Scenario classification and the
0.90 near-duplicate threshold justification, both found to be wrong. (b) Merge the
CE-based Scenario A/B split into a single "same-adduct evidence retained" category
rather than invent an unreproducible exact-CE-match rule. (c) Run the required cheap
matched Class-1 row-exclusion control (EXP-005) before any decision on the EXP-004 C2
smoke test. (d) Do NOT treat this correction pass as approval for the EXP-004 smoke
test itself — that remains a separate, explicit decision.

Reason:
`research/05_class1_to_class2_design.md` claimed "108 queries have only cross-adduct
evidence" and that the 0.90 Modified-Cosine near-duplicate threshold was "the threshold
EXP-001 itself already used and validated." Both were checked against source data and
found false. The 0.90 threshold claim is contradicted directly by
`results/exp001_query_sets.json` (`near_dup_cosine_threshold: 0.95`, and measured on a
different quantity — direct Cosine on same-metadata-group siblings, not cross-group
Modified Cosine). The "108" claim is contradicted by a same-adduct/cross-adduct
classification bug in `results/exp004_novelty_audit.json`, found by directly verifying
one case against `train.parquet` (query rid 468783's only retained evidence, rid
468784, is provably the same adduct as the query, but the audit said otherwise) and
independently confirmed via EXP-001's own `candidate_gen_hit` ground truth: a true
"108 cross-adduct-only" population would structurally force ~108 Variant-A
candidate-absent failures (adduct changes shift precursor mass far outside the 0.01 Da
window), but EXP-001 measured exactly 12. `research/scripts/exp004_reaudit.py`
recomputes the classification directly from source data (with a verified, reproducible
`rid` reconstruction) and finds exactly 12 cross-adduct-only queries, matching EXP-001's
12 candidate-absent queries exactly, zero contradictions in either direction.

The CE-based Scenario A (n=15)/B (n=21) split rested on the same broken same-adduct
detection. A reproducible CE-match rule was implemented and run on the corrected
388-query same-adduct-retained population: only 3/388 (0.8%) match exactly. Too thin
for a standalone benchmark, and the underlying field (ramped multi-value CE strings) is
fragile to compare exactly without a data-justified tolerance, so A/B are merged rather
than kept as an unreliable hard split.

Separately, EXP-005 (`research/06_matched_c1_control.md`) tested whether EXP-001 Mode
B's whole-metadata-group exclusion is meaningfully different from a naive single-row
exclusion (the "Class 1 with the query spectrum renamed" risk that partly motivated
EXP-004's design). Result: bit-identical Recall@1/5/10/25 and MRR@25 between the two
conditions, because only 8/400 (2%) Mode-B queries have any metadata-group sibling to
exclude at all. This rules out that specific leakage mechanism as a confound in the
existing EXP-001/EXP-004 construction.

Evidence:
`research/scripts/exp004_reaudit.py`, `results/exp004_scenario_audit_v2.json`,
`research/scripts/exp005_matched_c1_control.py`, `results/exp005_matched_c1_n25.json`
(smoke), `results/exp005_matched_c1_n400.json` (full), `research/06_matched_c1_control.md`,
corrected `research/05_class1_to_class2_design.md` Sec.4/Sec.6/Sec.11/Verdict.
Independently corroborated by a separate, read-only audit produced in parallel
(`research/04_exp004_independent_audit.md`, `research/analysis/`), which recomputed
the same/cross-adduct census from source data via its own script and reached the
identical corrected count (12, exact rid-for-rid agreement, 0 disagreements against
`exp004_scenario_audit_v2.json`) and the identical recommendations (merge CE A/B,
fix the 0.90 threshold's provenance claim, keep Variant A). Two independently-written
recomputations agreeing exactly is stronger evidence than either alone.

Alternatives considered:
(a) Silently fix the numbers without flagging the original claims as wrong. Rejected —
this project's standing rule is to preserve and show corrections transparently (the
precedent already set in Sec.4 of the same design doc for the direct-vs-Modified-Cosine
methodology fix), not overwrite history.
(b) Invent a CE-tolerance rule to keep Scenario A/B as a hard split. Rejected — no
data-justified tolerance exists yet, and the instruction explicitly permitted merging
as the honest alternative.
(c) Treat EXP-005's clean result as automatic approval to run the EXP-004 C2 smoke
test. Rejected — explicitly out of scope for this correction pass per the user's own
staged instruction; approval is a separate, later decision.

Expected impact:
`research/05_class1_to_class2_design.md` is now internally consistent and verifiable
against source data. The EXP-004 C2 smoke test's main open precondition (Sec.11 item 1)
is unchanged in substance (still: use Variant A, not Variant B) but is now backed by a
precisely quantified 12-query gap instead of a vague, wrong "108" estimate. No `src/`
(production pipeline) file was changed by this work.

---

## DEC-006 — EXP-004, as constructed, is not an independent test of "hidden spectrum, retained evidence"; do not treat its C1-vs-C2 aggregate similarity as evidence that spectral novelty is harmless before fixing the construction

Date: 2026-09-22

Decision:
(a) Do not cite EXP-004's near-equal MRR@25 (0.6519 vs EXP-001's 0.6510) as
evidence that "C2 doesn't matter" or that the retrieval pipeline is robust to
spectral novelty. (b) Before designing pseudo-C3, first resolve a methodology
gap in EXP-004: either give it an actual experimental manipulation of held-out
spectral novelty (not just a post-hoc audit of whatever Mode B retains), or
explicitly adopt EXP-001 Mode B itself as this project's operational
pseudo-C2 benchmark and treat novelty stratification as a descriptive lens on
its existing failures, not a separate arm. (c) No production pipeline or
candidate-generation change is implied by this decision.

Reason:
A full query-level join of EXP-001/EXP-005's C1 per-query results (rank,
n_candidates, best_true_score/best_wrong_score/margin) against EXP-004's C2
per-query results (rank, n_candidates, top-100 candidate scores) on the
identical 397-query eligible population found **zero differences on every
measured quantity**: identical rank for all 397 queries, identical
`n_candidates` for all 397, identical true-molecule score for all 378 queries
where computable in both, and consequently zero rank transitions at every
threshold checked (rank≤25, rank≤10, rank=1), zero regressions, zero
improvements, and 100% (34/34) of failures shared between the two conditions.
This is not a coincidence of similar aggregate metrics — it is a direct
consequence of EXP-004's own documented construction
(`research/05_class1_to_class2_design.md` §7, and
`research/scripts/exp004_full_run.py`'s own docstring): EXP-004 explicitly
reuses EXP-001 Mode B's exact whole-metadata-group exclusion, exact Variant-A
candidate generation, and exact Modified-Cosine scorer, applied to 397 of
EXP-001's original 400 queries (dropping 3 near-duplicates). It is EXP-001's
own pipeline re-run on a near-duplicate-filtered subset, not a distinct
computation. The full derivation and every supporting number is in
`research/analysis/exp004_c2_failure_analysis.md`.

The master research prompt's headline "MRR delta ≈ +0.0009" was independently
verified to be **fully and exactly explained** by the change in denominator
(397 vs 400 queries) rather than by any experimental effect: the 3 dropped
queries' own reciprocal ranks (0.5, 1.0, 0.1; mean 0.533) are below the
400-query MRR of 0.651, so removing them mechanically raises the remaining
397-query average by precisely the observed delta (verified to 10+ significant
figures).

A secondary, exploratory analysis (novelty-stratified C2 performance by
`max_sim_modcos` tercile, `research/analysis/exp004_c2_failure_analysis.md`
§13) found no meaningful monotonic relationship between retained-evidence
similarity and ranking outcome (Spearman ≈ −0.02) and a non-monotonic pattern
across similarity strata — but this is a within-C1/C2 descriptive finding on
the same shared population, not evidence about a genuine C2 manipulation,
since no such manipulation was actually run.

Evidence:
`research/analysis/exp004_c2_failure_analysis.md` (full report),
`research/scripts/exp004_c2_analysis.py` (reproducible join + all statistics,
rid-reconstruction verified against 4 known triples before use),
`results/exp004_c2_analysis/summary.json`,
`results/exp004_c2_analysis/query_comparison.jsonl` (397-row per-query table).

Alternatives considered:
(a) Accept the near-equal aggregate MRR at face value as "C2 ≈ C1, novelty
doesn't matter" and proceed directly to pseudo-C3 design. Rejected — this is
exactly the failure mode the master research prompt warned against
("MRR barely changed" must not become "C2 doesn't matter"), and the query-level
join shows the aggregate similarity is not even suggestive evidence about
novelty's effect, since no distinct C2 condition was actually run.
(b) Treat the 12/397 candidate-generation misses or the ranking-failure margin
shapes as new findings specific to C2. Rejected — both are identical to C1's
already-published EXP-005 numbers (12-query reachability ledger, −0.07/−0.39
close/decisive margin medians), confirming inheritance rather than a new
mechanism.

Expected impact:
Pseudo-C3 design (`research/CASMI_RESEARCH_STATE.md` §12) should not assume
EXP-004 has already validated that the pipeline is robust to spectral novelty.
Before or alongside C3 design, a corrected pseudo-C2 construction (or an
explicit decision to use Mode B as-is) is needed so that a genuine "known
molecule, unseen spectrum" comparison exists to build C3 on, per this
project's standing rule not to design C3 purely from C1 in isolation.

---

## PROPOSED DEC-007 — EXP-007: a corrected pseudo-C2 methodology with a genuine spectral-novelty intervention (design approved for review; smoke/full run NOT yet approved)

Date: 2026-09-22

Status: **APPROVED for FULL 600-QUERY RUN (2026-09-23).** The user/research lead approved
DEC-007 for the EXP-007 full run via execution authorization (dated 2026-09-23,
"EXP-007 FULL RUN — EXECUTION AUTHORIZATION"). The approval authorizes the full
600-query run (n=600, seed 20260922, tau grid {0.50,0.60,0.70,0.80,0.90,0.95},
real-intervention semantics validated in the smoke test) plus its execution artifacts
(results/exp007_c2_full_run_manifest.json, results/exp007_c2_query_results.jsonl,
results/exp007_c2_full_run.json, research/analysis/exp007_c2_full_run.md). It does NOT
authorize C3, training, or any production pipeline (src/) change. History: 2026-09-23
earlier authorized ONLY the smoke test (n=30, seed 20260922, strata-balanced) also via
execution instruction; prior to that it was PROPOSED — pending user review — and
recorded that a new pseudo-C2 construction (EXP-007) had been designed to repair the
methodological gap formalized in DEC-006, with a dataset audit executed and three
candidate constructions evaluated.

Decision:
(a) Design a corrected pseudo-C2 benchmark in which the novelty restriction is an
executed, experiment-controlled intervention — not a post-hoc audit of whatever Mode B
happens to retain. (b) Measure novelty continuously (per-query max ModifiedCosine to the
retained same-adduct evidence) and strata from the population's own distribution rather
than a blind 0.90/0.95 cutoff. (c) Adopt the dataset-audit-grounded conclusions from
`research/analysis/exp007_c2_design.md`: Design A (spectrum-level holdout with a
controlled novelty ceiling) is recommended as the clearest-causal-interpretation
construction; Design B (whole-CE-string holdout) is subsumed by A; Design C
(leave-one-evidence-family-out) is rejected as a C2 construction because retained
evidence families are effectively absent (cross-adduct evidence is unreachable under
Variant A; ingest-library families are ~1 per molecule). (d) The experimental wiring is
C1 (Mode-B control) vs C2(tau) (library minus every retained same-adduct sibling with
ModifiedCosine >= tau, tau in {0.50,0.60,0.70,0.80,0.90,0.95}) plus a candidate-pool-size
control C1-matched(tau) to separate novelty from the EXP-003/DEC-004 pool-size effect.

Reason:
EXP-004's construction was not distinct from EXP-001 Mode B (DEC-006): identical
exclusion, candidate generation, and scorer produced bit-identical per-query results,
so it provided no evidence about spectral novelty's effect. The EXP-007 audit measures a
real, drillable population: 490/600 sampled timsTOF queries have retained same-adduct
evidence; max-sim density is smooth across the novelty axis (no knife-edge), and strict
eligibility for C2 at tau=0.70 is ~20% (120/600), at tau=0.95 ~52% (313/600) — enough for
strata-sized conclusions with bootstrap CIs. Within-molecule pairwise ModifiedCosine
(audit Phase 2) established that same-adduct diff-CE spectra are genuinely but mildly
novel (median 0.79, 25% below 0.50), same-adduct same-CE pairs are near-identical
(median 0.999), and cross-adduct pairs are mostly low (median 0.013) — a direct,
measurement-backed justification for the CE-as-novelty-axis framing, replacing the
earlier unsupported CE assumptions flagged in DEC-005.

Evidence:
`research/analysis/exp007_c2_design.md` (design + audit writeup),
`research/scripts/exp007_audit_phase1.py` + `results/exp007_audit_phase1.json` (census),
`research/scripts/exp007_audit_phase2.py` + `results/exp007_audit_phase2.json`
(pairwise ModifiedCosine distributions; rid reconstruction verified against 2 known
triples before scoring),
`research/scripts/exp007_audit_phase3.py` + `results/exp007_audit_phase3.json`
(query-centric eligibility, seed 20260922, 600 timsTOF queries),
`research/scripts/exp007_design_sizing.py` + `results/exp007_design_sizing.json`
(strict eligibility by tau, max-sim distribution),
`results/exp007_c2_design/design_spec.json` (machine-readable spec).

Alternatives considered:
(a) Adopt EXP-001 Mode B as-is as the operational pseudo-C2 benchmark (per DEC-006
option b) and stratify its existing failures descriptively. Rejected for this task —
the user's explicit directive is a construction with an actual manipulation of held-out
novelty; Mode B remains the standing fallback and the C1 control here is exactly
Mode B, so this alternative is preserved as the C1 arm rather than discarded.
(b) Use a single headline novelty cutoff. Rejected — the max-sim density is smooth
(no knife-edge), so continuous stratification is the honest representation.
(c) Add Variant-B candidate generation to test cross-adduct C2 novelty. Rejected —
DEC-004 defers Variant B until the ranking-side question is resolved; including it
would confound EXP-007's novelty treatment with the separate, unresolved
candidate-generation question. The unreachable population is reported as a diagnostic
stratum instead.

Expected impact:
If accepted, EXP-007's smoke (n=30, seed 20260922, strata-balanced) may be run to
validate the §12 checks; only with a passed smoke test may the full 600-query run
proceed. The C1/C2(tau)/C1-matched results give the first genuine, controlled read of
how much identification degrades when a known molecule's query spectrum is unseen —
directly preconditioning C3 design. No production pipeline (`src/`) file is touched by
this design stage.

Smoke-test eligibility clarification (user resolution, 2026-09-23):
DEC-007's C2(tau) eligibility rule text ("max_sim < tau AND >=1 retained sibling",
design_spec.json) contradicts its removal rule ("remove retained siblings with
sim >= tau"): under strict eligibility no sibling meets sim >= tau, making C2(tau)
vacuous (identical to C1). The design spec's own §15 sizing mixes strict (72@0.60)
and loose (408@0.95) counts. The user resolved the semantics for the smoke test as
the REAL-INTERVENTION RULE: run C2(tau) for every sampled query with >=1 retained
same-adduct sibling, removing ALL retained siblings with sim >= tau; k (removed count)
may be 0 or >=1 per (query, tau); k>=1 runs prove a genuine intervention (non-empty
C1-only evidence), k=0 runs are recorded as no-op diagnostics. Strata sizing per tau
uses the per-tau k counts.

---

## DEC-008 (2026-10-03): prize-eligible track; calibrated proxy; E5 gated fusion failed on the leaderboard

**Decision.**
- The goal is a top-5 prize, so every component must be prize-eligible.
- Changes are judged on the calibrated proxy (`research/analysis/c3_calibration_report.md`). The bench's S2 scenario
  alone is no longer used to justify a change.
- **E5 (gate top-1 lib < 0.7 -> RRF with V2 at BETA 2.0) is rejected.**

**Evidence.**
- **Calibration.** Six leaderboard readings fit `LB ≈ 0.146·Visible + 0.21·S1 + 0·S2 + 0.11·PubChemOnly`, with about 50%
  scoring 0. Leave-one-out RMSE is 0.014.
  Files: `results/c3/calibration.json`, `research/scripts/c3_*.py`.
- **E5 v2 scored 0.287 on the public leaderboard** (kernel `shishiradhikari11/casmi-e5-eligible-gated` v2, submission
  56794027). E3b scored 0.355. The proxy predicted 0.370.
- **The pre-submission check passed (visible MRR 1.000), but it could not detect the failure.** All 400 visible
  molecules are exact enveda-180 copies with top-1 library similarity ≥ 0.988, so the gate never fired on them.

**Diagnosis (INFERENCE, not yet measured).**
- Hidden library-hit molecules are re-measured spectra, so their top-1 library similarity is often below 0.7 and the
  gate fires on them.
- V2 has no enveda-180 references (visible MRR 0.270), and BETA 2.0 lets it outrank E1. Its answers then replace the
  correct E1 answers.
- If the gate fired on about 2/3 of the "Visible" bucket, the expected loss is about 0.146 × 0.73 × 2/3 ≈ 0.07. The
  observed loss is 0.068.
- Root cause: the calibration's "Visible" bucket was modelled by exact duplicates, which have unrealistically high
  library similarity.

**Lessons.**
1. Any gate on library similarity must be tested on a scenario with **re-measured** library hits: hold out one
   enveda-180 spectrum per molecule and keep its other spectra in the library. Exact-duplicate visible molecules
   prove nothing about gates.
2. A second engine must never get a higher weight than E1 where E1 has library support it lacks.

**Next.**
1. Build the re-measured enveda-180 scenario and refit the calibration with E5's 0.287 as a seventh point.
2. Re-design the fusion, for example: never fuse when the top-1 has any enveda-180 library support; BETA ≤ 0.4; or
   give V2 enveda-180 references.
3. Rewrite the S4 scenario builder with stage checkpoints. The opencode version (`research/scripts/c3_s4_build.py`)
   parsed all 100M PubChem SMILES with RDKit and was stopped after about 4 CPU-hours with no output.

**Running on Kaggle at this point.**
- CFT training: `shishiradhikari11/casmi-cft-train`.
- Full-data FPNet retrain: `akritirijal04/casmi-fp-train-full`, dataset `akritirijal04/casmi-train-pkg-full`. Its
  exported files will be named `*_ho1.pt`; rename them to `*_full.pt` on download.

**DEC-008 addendum (2026-10-03, evening): scenario SV measured; calibration refit.**
- **SV scenario:** `research/scripts/sv_prepare.py`, bench scenario "SV", `research/scripts/sv_gate_check.py`. The 400
  visible molecules are queried with their test spectra; only the 1,213 exact duplicate train rows are masked; 294
  molecules keep other enveda-180 spectra.
- **Scores on SV (FACT):** E1 0.921; V2 0.270; E2 (BETA 0.4, no gate) 0.897; E5 rule (gate 0.7, BETA 2.0) **0.671**;
  gate 0.7 with BETA 0.4: 0.912.
- **Top-1 library similarity on SV (FACT):** median 0.737, p25 0.0; 47% are below 0.7.
- **Refit with SV as the library-hit bucket and E5 as a 7th point (FACT):** leave-one-out RMSE 0.039, against 0.014
  before. E5 is predicted at 0.31-0.33 against an observed 0.287. The weights become unstable (PubChem-only 0-0.10,
  S2 0-0.19).
- **Reading (INFERENCE):**
  - The direction of E5's failure is confirmed: an over-weighted second engine displaces library answers.
  - Its size is larger than SV plus the bench predict.
  - The linear bucket model with 7 noisy readings is too coarse to rank fusion-type changes. Treat it as a rough
    guide only; the leaderboard is the referee.
- **Design rule for E6:** additions must be **append-only**. Candidates from a second channel may only enter below
  the engine's existing top-k, and never displace an engine candidate. On SV that is ≥ 0.92 by construction.

---

## DEC-009 (2026-10-04): E6 append-only PubChem channel submitted; S4 shows the test is synthetic chemistry

**E6 offline (FACT, `research/scripts/e6_pc_channel.py`, `e6_merge_eval.py`, `results/c3/e6_merge.json`).**
- Channel: PubChem structures within 5 ppm of the target mass, scored as fp @ zlog with the held-out ho1 nets, top 40
  after metric-key de-duplication. Alone it scores PubChem-only **0.407** (E1 0.215), SV 0.637, S1 0.252.
- Chosen rule: engine top-1 kept, then engine[1:] and the channel alternate. Scores: SV 0.9211 (−0.0003), S1 0.8606
  (−0.003), S2 0.634 (−0.013), PubChem-only **0.309** (+0.094). Proxy ≈ +0.010.
- Gating on top-1 library similarity (0.01–0.7) cut the PubChem-only gain to 0.25–0.29: 38% of PubChem-only
  molecules have a spurious nonzero library match. **Rejected.**
- The run crashed overnight on reading back its fingerprint chunks (pandas-3 object arrays), not on a power loss.
  The scoring stage now streams chunk by chunk (peak ≈ 0.5 GB).

**E6 on Kaggle (FACT).**
- Kernel `shishiradhikari11/casmi-e6-pubchem-channel` v1 uses dataset `shishiradhikari11/casmi-e6-pcnets` (MIT): our
  full-data nets, renamed `e6net_*`, so the engine's `fp_*.pt` glob ignores them.
- Runtime: 7,623 s in total, of which the channel took 2,452 s (400 molecules, 2.68M structures). The channel's time
  budget is min(5 h, 8.25 h − elapsed).
- Parity with the bench lists: 20/20 identical. Visible check: MRR 1.000, top-1 unchanged on 400/400.
- Submitted as 56815529.

**S4 (FACT, `research/analysis/c3_s4_scenarios.md`).**
- 300 Class-3 + 300 PubChem-only enveda-180 molecules, timsTOF, no dimer adducts. 116 of 5,112 Class-3 candidates were
  PubChem tautomers, found with an exact-mass window plus an ElementGraph pre-filter.
- NP-likeness > 0: S4C3 1.7%, S4PC 0.3%, **visible test 0.7%**, S12 86%, S3 40%. The mass and adduct mix of S4 also
  matches the test.

**INFERENCE.**
- The test is synthetic, screening-library chemistry.
- The S12/S3 benches are natural-product-heavy and are unrepresentative of it.
- Class-3 generation should target building-block / reaction enumeration, not natural-product scaffolds.
- S4 cannot judge learned scorers until a net is retrained with `held_S4` excluded.

**Next.**
1. E6's leaderboard score.
2. CFT vs ho1 as the channel scorer (`e6_cft_channel.py`).
3. Wire S4 into `bench.py`.
4. A held_S4-excluded FPNet retrain on akritirijal04's account.
5. Synthetic-chemistry Class-3 generation design.

**DEC-009 addendum (2026-10-04): CFT as the channel scorer (FACT, `research/scripts/e6_cft_channel.py`, `results/c3/e6_cft.json`).**
- Setting: S3 PubChem-only bucket (213 molecules); identical 5 ppm windows; CFT is leak-free here.

  | Scorer | Channel alone | m1_alt merge |
  |---|---|---|
  | CFT | 0.226 | 0.264 |
  | ho1 nets | **0.407** | **0.309** |
  | RRF of both (CFT weight 0.25–0.5) | 0.409 | 0.312 |

- The RRF gain is noise. **Decision: keep the FPNet channel; no CFT in E6.**
- INFERENCE: S3 spectra are GNPS spectra with no collision energy or instrument. CFT's own validation (mostly
  timsTOF) had it +0.02 over the same nets, so its weakness here may be domain shift. A timsTOF comparison needs a
  held_S4-excluded retrain of both nets.

**DEC-009 CORRECTION (2026-10-04, later the same day): the "test is synthetic" inference is WITHDRAWN.**
- **What the data page says (FACT, data-description page):**
  - The competition "focuses on molecules that resemble those found in natural samples from plants, mammals, or
    microbes".
  - enveda-180 is "synthetic drug-like screening compounds, a different region of chemical space from the test
    molecules".
  - test.parquet "is comprised of examples from the training dataset; it will be replaced by the hidden test set".
- **Why the inference was wrong:** the visible test's low NP-likeness (0.7%) reflects the placeholder file (enveda-180
  copies), not the hidden test. S4 matches the instrument and mass range, **not** the chemistry.
- **Consequences:**
  1. S4 is a timsTOF query set from a different chemical space. Use it only as a secondary instrument check, never to
     choose chemistry-dependent methods.
  2. The ho2 / CFT-ho2 retrains, motivated by S4, have lower value.
  3. Class-3 generation must target natural-product analogs. S1 (enveda-np-examples) is "the closest library to the
     test set" per the data page.
- **E6 leaderboard (FACT):** submission 56815529 scored **0.360** against E3b's 0.355. That is +0.005, where the proxy
  predicted +0.010. Append-only did not hurt.

---

## DEC-010 (2026-10-05): Class-3 generators measured on the C3NP bench; E7 = E6 + Class-3 block at ranks 4-8

Full write-up: `research/analysis/c3gen_tournament.md`. Bench and review reports: `research/scratch_wf/c3_resume_done_reports.md`;
workflow wf_36d8cec9-5ce journal.

**C3NP bench (FACT).**
- 550 known natural products: npex 250, s3pc 215, s3none 85. s3pc and s3none are interleaved in rows 250-549.
- Window and analog baselines score exactly 0.
- Leak checks are clean. The minor findings are the myricetin flavylium alias, present only in universe.parquet, and
  the truth ik14 used as `mid`.
- `c3np_eval.py` now reports nb1 strata (whether the truth has a ±CH2/±O neighbour in the seed sources).
- New `--strip edit1|tc070|tc085` modes remove the truth's congeners from all seed sources.

**Generators, MRR@25 on all 550 (FACT).**

| Generator | Plain | Strip edit1 |
|---|---|---|
| mmp_edit | 0.403 | 0.289 |
| biotransform | 0.492 | 0.374 |
| rr_bt (round-robin union, biotransform first) | **0.508** | 0.399 |

- rr_bt scores 0.757 on nb1-yes molecules and **0.166 on nb1-no**.
- Re-ranking by zlog fails overall (0.253) but helps nb1-no (0.191).
- Runtime is about 9 CPU-s per molecule for both generators.
- New assets are about 690 MB, dominated by pool_fp at 617 MB. All are train-derived, COCONUT or our own code.

**INFERENCE.**
- The bench score is congener-driven: 75% of the reciprocal-rank sum comes from 1-heavy-atom edits.
- Real Class 3 should have fewer close database congeners. Planning MRR on hidden Class 3 is about 0.10-0.20.
- Inserted at ranks 4-8, the cost is 0.0037 LB and the gain about +0.007 to +0.05, central +0.01 to +0.03.
- **This alone does not reach 0.432.** The remaining gap must come from Class-2 ranking and a seed-Tc-aware re-ranker.

**Decision.** Build E7 = E6 with ranks 1-3 kept, then the first 5 rr_bt candidates not already in the list at ranks
4-8, then E6 ranks 4 and up. It is ungated. Before shipping:
1. Replace biotransform's wall-clock guard with work budgets.
2. Use deploy mode: full rules at min_freq 3, an opaque mid, forbidden=∅, and the full pool.
3. Export the engine's analog list in the kernel.
4. Check parity on 20 bench molecules.
5. Run the visible check: top-3 unchanged on 400/400.

A low-risk twin at ranks 6-10 is optional.

**DEC-010 addendum (2026-10-05, evening): CFT-ho2 received from Akriti's laptop (FACT).**
- Files are in `models/cft_ho2_akriti/` (`cft_ho2.pt`, from her `archive.zip`). The run finished on the RTX 5050:
  34k steps, best at 22k, 423 min. Status `RESULT: ALL OK`.
- Validation:

  | | Pool MRR@25 | PubChem-val MRR | Full-window estimate |
  |---|---|---|---|
  | CFT-ho2 | 0.820 | 0.846 | 0.785 |
  | Earlier CFT (cft_kaggle) | 0.819 | 0.842 | 0.776 |

  It is as good as the earlier CFT.
- Use: S4 molecules are held out, so the CFT-vs-FPNet comparison on S4 timsTOF spectra (DEC-009 addendum) is now
  leak-free. That check is secondary only, because S4 is different chemistry from the hidden test.
- Next for Akriti's laptop: the 4-network ho1 ensemble (`research/train_pkg/ensemble/`, about 35–45 h).

**DEC-010 addendum (2026-10-06): E7 built; the PubChem re-ranker measured (opencode tasks A and B, checked by Claude).**

**E7** (report: `research/kaggle_e1/e7/E7_STATUS.md`).
- Work budgets fixed (FACT): the old W_GEN truncated 519 of 550 molecules, and npex rows 0-59 fell to 0.575. With
  the new budgets the score is 0.7335, deterministic across 3 runs.
- Dry run on the 400 visible molecules (FACT): top-3 identical 400/400; 25 unique metric keys per row; 399 rows
  with 5 insertions at ranks 4-8; 1 RDKit failure, which kept its E6 list; c3 stage 1,204 s with 2 workers.
- Dataset `casmi-e7-c3assets` is 656 MB. The kernel's dataset sources are E6's plus that one dataset.
- Expected (INFERENCE): +0.005 to +0.015 LB.

**PubChem re-ranker** (report: `research/analysis/c2_gap_diagnosis.md`).
- ho1+struct+pop lambdarank, no gate (FACT): proxy +0.0083, and no bucket gets worse.
- The gate adds +0.031 proxy but costs SV -0.030. **Rejected.**
- Popularity is strong on NP-like molecules (S2 pop-only MRR 0.79) and useless on enveda-180 (0.003).
- ho2 adds nothing over ho1 for zlog (CI includes 0).
- Expected (INFERENCE): about +0.004 LB.
- Before an E8 can ship: SV must be re-scored with the full-data nets, because the deploy nets differ from the
  held-out ones the re-ranker was fitted on. Shipping needs the whole popularity map, from a 2.6 GB source.

**INFERENCE.** E7 plus E8 is about 0.37-0.38 LB. Neither closes the gap to 0.432; the largest lever is still open.

**DEC-010 addendum (2026-10-06 afternoon): E8 go/no-go (TASK D) + C3 seed-Tc re-ranker (TASK F).**

**E8 go/no-go — NO-GO** (report: `research/analysis/e8_sv_gonogo.md`).
- The C2 re-ranker was fitted on ho1-net scores; the deploy nets are full-data nets, so score/score_gap/score_rk shift.
- SV is the only honest check (full nets never trained on it).
- (FACT) With full-net scores on SV: merged MRR 0.9209 vs. floor 0.9211 — fails by 0.0002 on one regressed molecule,
  across all three feature variants (full / rankov / rankz). CI lower bound ≥ −0.005 passes, but the floor does not.
- Decision: do not ship the C2 re-ranker as fitted. E8 = E7 is safe.
- Path to GO: refit on full-net scores on the bench PC+S2 channels, then re-run this SV check. The
  popularity/struct features remain plausible; only the score block must be net-consistent at train and deploy.
- Fixed two bugs in `e8_sv_gonogo.py` (swapped improved/regressed label; eager-default KeyError).

**C3 seed-Tc re-ranker — RECOMMEND** (report: `research/analysis/c3_seedtc_reranker.md`).
- (FACT) LGBMRanker lambdarank (5-fold grouped CV on 550 C3NP bench molecules):
  - default: MRR@25 0.5394 vs rr_bt 0.5077; nb1-no 0.2150 vs 0.1664; CI [+0.017, +0.047]
  - strip-edit1: 0.4418 ≥ 0.3893 floor; nb1-no 0.2150; CI [+0.028, +0.058]
- (FACT) 2-param hand rule fails: best cell 0.212 nb1-no, goal unreachable.
- (FACT) Top feature: seed_tc (436/1500 splits), then z_z, z, rank_rr.
- Fixed two build bugs: Tanimoto word-sum→popcount; union-slice off-by-one.
- Deploy: ~ms/mol (RDKit fp + dot + GBM); no truth use; mode-agnostic features.
- (INFERENCE) Expected LB gain: modest, no regression risk below rr_bt baseline.

**E9 plan (next ship after E7):**
- E9 = E7 + C3 seed-Tc re-ranker at ranks 4-8 (lambdarank re-orders the rr_bt union before insertion).
- Gate: dry run visible check identical to E7 (top-3 unchanged 400/400; 25 unique per row).
- E8 (C2 re-ranker) confirmed NO-GO even after full-net S2/S3 refit (0.9209 vs 0.9211 floor); closed.

**DEC-010 addendum (2026-10-06 evening): E7 executed on Kaggle and submitted (FACT).**
- Kernel `shishiradhikari11/casmi-e7-c3-channel` v2 finished with `KernelWorkerStatus.COMPLETE`.
- All pass criteria from `UPLOAD_STEPS.md` met on visible test (400 molecules):
  - Top-3 unchanged vs E6: 400 / 400
  - 25 unique per row: 400 / 400
  - Rows with C3 insertions at ranks 4-8: 399 / 400 (1 fallback molecule m_0d08be kept E6)
  - Work budget guards: `bt_guard = 0`, `mmp_guard = 0`
  - COCONUT availability: `coco_ok = True`
  - Total runtime: ~8317 s (E6 channel ~2337 s, E7 C3 channel ~1000 s).
- First submission attempt **56881046** encountered a transient Kaggle platform backend error (`Kaggle Error: A system error. Please try resubmitting to resolve the error`).
- Immediately resubmitted as **56883907** (Version 2) with description: `E7: E6 + Class-3 rr_bt block at ranks 4-8 (biotransform+mmp_edit), ungated. Prize-eligible. (retry 1)`.
- Both submissions later failed with `SubmissionStatus.ERROR` ("A system error"). Root cause identified in DEC-011.

---

## DEC-011 (2026-10-07): Resolution of Competition Submission Failures, Strong Open Licensing, and Public Dataset Unification

- **Diagnosis of Prior Submissions (56881046, 56883907) & Kernel v5 (FACT):**
  1. **Private Datasets in Code Competitions:** In Kaggle Code Competitions, the evaluation worker runs in a separate sandboxed container. Because all attached user datasets had default `isPrivate: True`, the evaluator container was denied read access, causing the generic platform failure "A system error".
  2. **Python 3.12 vs 3.13 Wheel Mismatch:** The external `metric/rdkit-2026-3-3-wheel` dataset was deleted by its author (404). Attempting to remove it left only a `cp313` wheel, while Kaggle runs Python 3.12 (`cp312`), crashing Cell 0 with `IndexError`.
  3. **Third-Party Dataset 404s:** `casmi26-fp-models-v2`, `casmi26-ranker-features`, `casmi26-simulated-ranker-rows`, and `coconut-casmi26-candidates` had been deleted by external users.

- **Actions Taken (FACT):**
  1. Created and uploaded public dataset `shishiradhikari11/casmi-rdkit2026-cp312` containing `rdkit-2026.3.3-cp312-cp312-manylinux_2_28_x86_64.whl` under BSD 3-Clause license.
  2. Updated all attached CASMI datasets to **PUBLIC** (`isPrivate: False`) with strict adherence to strong, open-source licensing:
     - `casmi-fp-models-v2`: `CC0-1.0`
     - `casmi-rank-features`: `CC0-1.0`
     - `casmi-sim-rows`: `CC0-1.0`
     - `casmi-v2-pubchem`: `CC0-1.0`
     - `casmi-coco-candidates`: `CC BY 4.0`
     - `casmi-bio-clean`: `CC BY 4.0`
     - `casmi-e7-c3assets`: `CC BY 4.0`
     - `casmi-fm-runner`: `MIT`
     - `casmi-e6-pcnets`: `MIT`
     - `casmi-rdkit2025-cp313`: `other` (BSD-3-Clause)
     - `casmi-rdkit2026-cp312`: `other` (BSD-3-Clause)
  3. Updated `research/kaggle_e1/e7/kernel-metadata.json` to mount all 11 verified public datasets.
  4. Pushed **Kernel version 6** (`shishiradhikari11/casmi-e7-c3-channel`), verified status: `KernelWorkerStatus.RUNNING`.


---

## DEC-012 (2026-10-07): FPNet seed ensemble benched — GO with 3–4 standard nets; E7 v6 passes; Binaya's Kaggle seeds were ho1, not full

**Ensemble bench (FACT, `research/scripts/ens_members.py` + `ens_eval.py`, `results/ens/ens_eval.json`).**
- Members: ho1 (base) and Akriti's laptop ensemble ho1s10 / ho1s20 / ho1s30 / ho1big (`models/fp_ens_ho1`, all `RESULT: ALL OK`).
  All trained on the ho1 package, so the S2/S3 PC+S2 buckets (459 molecules) are leak-free.
- Parity: re-scored ho1 matches `scores_5.0ppm.npy` (corr 1.0, max |diff| 0.003) and reproduces ens_window.json (PC 0.4155 / 0.3039, S2 0.2574 / 0.6319).

  | ALL (n=459) | window MRR | E6 sim | Δ window vs ho1 [CI] | Δ E6 sim vs ho1 [CI] |
  |---|---|---|---|---|
  | ho1 | 0.3308 | 0.4797 | | |
  | mean2 (ho1+s10) | 0.3490 | 0.4836 | +0.018 [+0.005, +0.032] | +0.004 [+0.001, +0.007] |
  | mean3 | 0.3552 | 0.4838 | +0.025 [+0.009, +0.040] | +0.004 [+0.001, +0.007] |
  | zmean4 (ho1,s10,s20,s30) | **0.3586** | **0.4840** | **+0.028 [+0.012, +0.044]** | **+0.004 [+0.001, +0.008]** |
  | mean5 (+big) | 0.3403 | 0.4819 | +0.010 [−0.006, +0.025] | +0.002 [−0.001, +0.006] |

- ho1big (d768, 8 layers, bs128) is clearly worse alone (window 0.236). **Exclude big nets.**
- Gains saturate: most of the benefit comes with 2–3 members. PC bucket E6 sim +0.009 [+0.002, +0.016].
- The earlier ho1+ho2 "ensemble" (DEC-010) was not significant; same-package seed averaging is.
- **Decision: GO for a seed-averaged FPNet channel in deploy**, using the full-data nets full (s0), fulls10, fulls20
  (all downloaded, `RESULT: ALL OK`, `models/fp_full*`), optionally a 4th (fulls50 from Binaya's laptop).
  Because score = fp @ zlog is linear, deploy = average the members' zlogs.
- **Risk (INFERENCE):** averaged zlogs have a smaller spread. Any downstream model fitted on single-net z features
  (the E9 C3 seed-Tc re-ranker: z, z_z; the closed E8 re-ranker) sees shifted inputs. That is the E8 failure mode.
  Re-fit or re-check those on ensemble scores before combining.
- Expected LB (INFERENCE): +0.002 to +0.005 from the PubChem channel alone.

**E7 v6 kernel (FACT, `research/kaggle_e1/e7/v6_output/`).** COMPLETE: top-3 unchanged 400/400, 25 unique 400/400,
C3 insertions 399 (m_0d08be RDKit Range Error → kept E6), bt_guard 0, mmp_guard 0, coco_ok True, E6 stage 2196 s,
C3 stage 980 s. **Not yet submitted** (needs the user).
- Correction to DEC-011 cause #2: the Kaggle image here runs **Python 3.13** (log: `/usr/local/lib/python3.13`, wheel
  used `casmi-rdkit2025-cp313`), not 3.12. The cause #1 claim (private datasets → "system error") is still a
  hypothesis; the v6 submission tests it. Deleted third-party datasets are an equally good explanation.

**Binaya's Kaggle seed kernels (FACT).** `binayaadhikari13/casmi-fp-train-full-s30/-s40` mount
`shishiradhikari11/casmi-train-pkg`, the **ho1** package (held_keys 585), not the full one (that is
`akritirijal04/casmi-train-pkg-full`, held_keys 0). Their outputs, named `fulls30/fulls40`, are ho1 nets.
s30 duplicates Akriti's finished ho1s30. Given the saturation above, more ho1 members add little and more full members
add about +0.002. **Recommendation: cancel both** (website only; the CLI has no cancel). The s30/s40
kernel-metadata now points at the full package, if they are ever re-pushed.
**Binaya's laptop** package was also seed 30 on ho1 → changed to full data, `--seed_offset 50 --tag fulls50`
(`research/train_pkg/casmi-binaya.zip` rebuilt, 883 MB, held_keys 0).

---

## DEC-013 (2026-10-07 afternoon): E7 v6 scored 0.320 because the notebook RDKit fell back to 2025.03.6; v7 pins 2026.03.3

**FACT.**
- Submission 56904156 (E7 v6) scored **0.320**, against E6's 0.360. E7's insertions at ranks 4-8 can cost at most ~0.004.
- E6 live kernel (`research/kaggle_e1/e6_live/out`) vs E7 v6 (`research/kaggle_e1/e7/v6_output`) on the visible test:
  engine lists identical **3/400**, top-1 identical 400/400, submission top-3 identical **207/400**.
- Same data on both sides (our ranker 399,790 x 51, prvsiyan ranker 142,762 x 31). The only input difference is the RDKit:
  E6 used `metric/rdkit-2026-3-3-wheel` cp313 → **2026.03.3**. v6 used `casmi-rdkit2025-cp313` → **2025.03.6**, because
  the replacement `casmi-rdkit2026-cp312` cannot install on the Python **3.13** image.
- DEC-011's claims that the `metric` wheel was deleted (404) and that Kaggle runs 3.12 are both wrong: the dataset lists
  cp310-cp313 wheels today.
- The in-kernel "top-3 unchanged vs E6" check compares against E6 lists computed in the same run, so it could not catch
  this. **Parity must be checked against the previous live kernel's output.**

**Fix (E7 v7, pushed 2026-10-07 ~16:30).**
- New public dataset `shishiradhikari11/casmi-rdkit2026-cp313`: the PyPI wheel, BSD-3 LICENSE.txt,
  sha256 9a21f96d…ceab3, same size as metric's (37,187,145 B).
- Cell 0 prefers `rdkit-2026.3.3-*` and **asserts** `rdkit.__version__ == '2026.03.3'`; `casmi-rdkit2026-cp312` dropped.
  The forward-model stage still pins its own 2025.3.6, as in E6.
- Gate before submitting v7: `eng_lists.json` identical to e6_live on ~400/400 and top-3 vs the e6_live submission ~400/400.

**Public notebook nursrijan/…-sovereign-zenith (0.42-0.44 family).** Ahmed v4n stack plus prvsiyan BIO
(CC BY-NC-SA) and Ahmed datasets that are still "non-commercial, with attribution" (checked today) → not prize-eligible.
Method ideas: GLACIER-only final re-rank (0.402 → 0.417), GLACIER on [M+H]+ only (+0.006), pool-popularity re-rank
mu 0.15 (+0.010). Our FM bench (`results/bench/fm/analysis.txt`) does **not** favour GLACIER-only (lam_ice 0, lam_gl 1.0:
+0.003). It favours smaller weights: (0.25, 0.1) +0.042 vs our (0.5, 0.5) +0.024, mean of S1 and S2. Their result also fits
"less forward-model weight". Candidate E7 variant after v7 parity: LAM_ICE 0.25, LAM_GL 0.1.

**DEC-013 CORRECTION (2026-10-07 evening): RDKit was NOT the cause. The engine ran with 0 FP nets.**
- (FACT) E7 v7 ran on RDKit 2026.03.3 and its engine lists still matched live E6 on only 3/400. v6 vs v7 lists were identical
  on 398/400, so the RDKit version barely matters.
- (FACT) Root cause: the engine runner keeps fp models with `'casmi26-fp-models-v2' in p`. DEC-011 re-hosted that dataset as
  `shishiradhikari11/casmi-fp-models-v2`, so the filter matched nothing. Log: `engine fp models []`, `fp models: 0 single + 0 merged`.
  E6 live had `1 single + 1 merged`. The engine lost its fingerprint channel → LB 0.320.
- Also (FACT): our `casmi-sim-rows` copy holds only `sim_rank_rows_nofp.npz`; megayak's original also has 2 FPNets and the
  fp16k/fppair row files. The engine reads only `sim_rank_rows_nofp.npz`, so this does not matter for E6/E7.
- Fix (v9, pushed 2026-10-07): the filter matches `'fp-models-v2'`, and the runner asserts exactly 2 nets. v8 was an accidental
  re-push of v7 (the patch failed its own assert). RDKit 2026 pin from v7 kept (harmless; matches E6).
- Lesson: any dataset re-host must be followed by a parity diff of `eng_lists.json` against the previous live kernel
  before submitting. An in-kernel self-comparison cannot catch input regressions.

**DEC-014 (2026-10-07): Rebuild Ahmed's v4n engine ourselves (no waiting on relabel).**
- User decision: re-implement and retrain v4n from train.parquet / COCONUT / PubChem / np-examples / DreaMS base, using
  his public code only as reference. His code is kept read-only in `research/v4n_rebuild/ahmed_ref/` (git-ignored, never shipped).
- Spec in progress: `research/v4n_rebuild/REBUILD_SPEC.md`.

**DEC-014 progress (2026-10-08).**
- (FACT) E7 v9 parity vs live E6: engine lists 400/400 identical, submission top-3 400/400, e6 lists 399/400 (one rank-8
  swap, m_5bfac3). Engine loads 1 single + 1 merged FP net. **v9 is the version to submit.**
- (FACT) HO_R split built: `results/v4n/split_v4r.parquet`, 21,927 train keys (296,998 spectra), enveda-180 weight 0.33
  (39% of HO_R). This deliberately departs from the spec's x2 up-weight, because of DEC-009: the test chemistry is NP-like.
- (FACT) FPNet R-A (CFT recipe, HO_R held out, seed 1, merge_p 0.3) is training on `binayaadhikari13/casmi-cft-hor-a`.
  R-B is staged for `akritirijal04/casmi-cft-hor-b` once her quota resets on 10 Oct (Akriti's quota was exhausted on 10-07).
- (FACT) Phase A engine (`research/v4n_rebuild/engine/`, ours, MIT): V0 keys 2000/2000 equal; V1 P2 275,810 rows, pool 548,856
  (COCONUT ≤ 480 Da only; tail build running); V2 2,539,608 spectra, 0 recount diffs. V3 parity vs his engine as a local
  oracle (148 non-tie queries): candidate sets 148/148, feature rows 17,316/17,330 equal; the 45 tie queries differ
  because his merged-view adduct choice depends on set order (his own two runs disagree). Median 1.5 s/query on CPU.
- Phase B (simulation driver + ranker v0 pilot on ho2 keys with cft_ho2) started.

**E7 v9 leaderboard (FACT, 2026-10-08):** 0.363 vs E6 0.360 (+0.003; DEC-010 expected +0.005 to +0.015; within LB noise). New best eligible score. The Class-3 block at ranks 4-8 does no harm and gains little.

**DEC-014 result (FACT, 2026-10-09):** v4n rebuilt engine e9-sub1 v4 (our engine + R-A + ranker_v0, no FM/PubChem) scored **0.341** on the public LB (his v1 engine + ranker: 0.354). Kaggle CPU 2.2 s/mol. The first hang (v2) came from workers crashing at init on a missing pool_fp_raw.npy, which multiprocessing respawns forever; the notebook now runs a smoke test first.

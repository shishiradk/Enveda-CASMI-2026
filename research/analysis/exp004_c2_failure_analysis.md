# EXP-004 C2 Failure Analysis

Date: 2026-09-22

Status: Post-hoc analysis of completed, immutable results. No new experiment was
run, no `src/` file was modified, no existing results file was overwritten.

Reproducibility: `research/scripts/exp004_c2_analysis.py` (verifies `rid`
reconstruction against 4 known triples before trusting anything, per this
project's established convention). Outputs: `results/exp004_c2_analysis/summary.json`,
`results/exp004_c2_analysis/query_comparison.jsonl` (397 rows, one per eligible
query), `results/exp004_c2_analysis/run_log.txt` (full console trace).

---

## 1. Research question

> Does hiding one spectrum of a known molecule while retaining other spectral
> evidence for that molecule materially change the molecule-identification
> problem under Variant A?

Secondary: if C1 and C2 performance is nearly identical, are the remaining
failures mostly the same underlying ranking/candidate-generation problems?

**Headline answer, stated precisely up front because it changes how every
later section should be read:** on this benchmark, as actually constructed and
executed, EXP-004 ("C2") did not run a different query-construction or
candidate-generation procedure than EXP-001 Mode B ("C1"). Both experiments
withhold exactly the same query spectrum's whole metadata group and search the
same library with the same candidate rule and the same scorer. The only
difference between the two experiments' populations is that EXP-004 additionally
drops 3 near-duplicate queries (302973, 519143, 944352) that EXP-001 included.
Restricted to the shared 397-query population, **C1 and C2 per-query results are
bit-identical**: identical rank for all 397 queries, identical `n_candidates`
for all 397, identical true-molecule score for all 378 queries where a score is
computable in both. This was verified directly, not assumed (Section 5).

Given that, the research question as posed cannot be answered from EXP-004
alone — EXP-004 characterizes (audits the novelty of) the evidence that Mode
B's exclusion mechanism already withholds; it does not apply an additional or
different withholding step on top of C1. See Section 15 for what this does and
does not mean for the underlying question, and Section 16 for what evidence a
real answer would require.

## 2. Experimental context

- EXP-001 Mode B (`research/02_class1_coverage.md`): 400-query benchmark;
  query's whole metadata group (`inchikey14`, `adduct`, `precursor_mz`,
  `num_peaks`) excluded from the search library; Variant A candidate generation
  (raw precursor m/z ± 0.01 Da, full train library); matchms 0.33.1
  `ModifiedCosineGreedy` (tolerance 0.1 Da, mz_power 0.0, intensity_power 1.0).
- EXP-004 (`research/05_class1_to_class2_design.md`, `research/scripts/exp004_full_run.py`):
  explicitly documented (design doc §7, and the full-run script's own docstring)
  as "EXP-001's Mode B construction, with the added post-hoc novelty filter...
  applied" — i.e. the same 400-query population minus the 3 near-duplicates,
  scored with the identical unchanged Variant-A pipeline.
- EXP-005 (`research/06_matched_c1_control.md`): a matched control that
  re-scored EXP-001's exact 400 queries and additionally persisted per-query
  `best_true_score`/`best_wrong_score`/`margin`, which EXP-001's own checkpoint
  never stored. Condition W of EXP-005 is bit-identical to EXP-001 Mode B
  (already validated in that experiment). This analysis uses EXP-005's
  condition W as the authoritative C1 source specifically because it is the
  only artifact that has per-query scores/margins for C1, not just ranks.

## 3. Authoritative inputs

| File | Role | n |
|---|---|---|
| `results/exp001_query_sets.json` | 400-query Mode-B population definition (rid, true_inchikey14, excluded groups) | 400 |
| `results/exp001_checkpoint.json` (`B\|C_all_train\|ModifiedCosine`) | EXP-001's own per-query diagnostics (rank capped to null above 25) | 400 |
| `results/exp005_matched_c1_n400.json` (`condition_W_group_exclusion`) | Authoritative C1 per-query rank (uncapped) + best_true_score/best_wrong_score/margin | 400 |
| `results/exp004_full_run.json` (`per_query_results`, `margin_rows`) | Authoritative C2 per-query rank (uncapped) + top-100 candidates + margin (where true molecule is within persisted top-100) | 397 / 378 |
| `results/exp004_novelty_audit.json` | Per-query Modified-Cosine similarity between held-out query and its most-similar retained same-molecule spectrum | 400 |
| `results/exp004_scenario_audit_v2.json` | Corrected (DEC-005) same-adduct / cross-adduct-only classification per query | 400 |
| `train.parquet` | Query metadata not persisted elsewhere: `precursor_mz`, `ionization_mode`, `collision_energy_ev`, `num_peaks`, `adduct` | joined by verified `rid` |

All schemas were inspected directly before writing any analysis code; no field
names were assumed.

## 4. Population verification

400 original Mode-B queries, minus 3 confirmed near-duplicate exclusions
(302973, 519143, 944352, exactly as documented) = 397 eligible. EXP-004's own
`per_query_results` rid set was checked for exact set-equality against this
397-query eligible set: **match confirmed, 0 discrepancies**. No population
repair was necessary — verification passed cleanly on the first check.

## 5. C1 vs C2 aggregate metrics — and the central finding

Metrics recomputed independently from the joined per-query data (not copied
from either experiment's self-reported summary), restricted to the identical
397-query population for both conditions:

| Metric | EXP-001 C1 (397-subset) | EXP-004 C2 (397) | Delta |
|---|---:|---:|---:|
| n | 397 | 397 | — |
| Candidate recall | 0.96977 | 0.96977 | 0.00000 |
| R@1 | 0.53652 | 0.53652 | 0.00000 |
| R@5 | 0.80605 | 0.80605 | 0.00000 |
| R@10 | 0.87154 | 0.87154 | 0.00000 |
| R@25 | 0.91436 | 0.91436 | 0.00000 |
| MRR@25 | 0.65192 | 0.65192 | **0.00000** |
| Median candidates | 687 | 687 | 0 |
| Mean candidates | 899.95 | 899.95 | 0.00 |

The recomputed C2 metrics were cross-checked against EXP-004's own
self-reported `A_overall_397` block in `results/exp004_full_run.json`: exact
match to 6 decimal places on every metric, confirming this analysis correctly
reproduces the authoritative published numbers before drawing any new
conclusion from them.

**Per-query check (not just aggregate):** of 397 queries, rank differs between
C1 and C2 for **0**; `n_candidates` differs for **0**; the true molecule's own
score (available for both in 378/397 cases) differs by more than 1e-9 for
**0**. C1 and C2 are the same computation on the same 397 rows.

**Why the master prompt's headline numbers (C1 MRR 0.6510 vs C2 MRR 0.6519,
Δ≈+0.0009) differ slightly:** that comparison is between EXP-001's **400**-query
aggregate and EXP-004's **397**-query aggregate — different denominators, not
different per-query outcomes. The 3 excluded queries' own reciprocal ranks in
the 400-population are 1/2 (rid 302973), 1/1 (rid 519143), 1/10 (rid 944352) —
sum 1.6, mean 0.533, below the population MRR of 0.651. Removing them
mechanically raises the mean of the remaining 397 by exactly
(400×0.6510265432280139 − 1.6) / 397 = 0.6519159125723061 — verified to match
EXP-004's reported MRR to 10+ significant figures. **The entire ≈+0.0009
aggregate delta is fully explained by dropping 3 below-average-scoring queries
from the denominator; zero of it comes from any experimental manipulation.**

Reachable-only (n=385, i.e. excluding the 12 structurally cross-adduct-only
queries) reproduces the same pattern: C1 MRR@25 = C2 MRR@25 = 0.67224 exactly.

## 6. Query-level transition matrix

Three "good" definitions, each is a strict superset/subset relationship, all
show the same pattern:

| Definition | C1 good → C2 good | C1 good → C2 not | C1 not → C2 good | C1 not → C2 not |
|---|---:|---:|---:|---:|
| rank ≤ 25 | 363 | 0 | 0 | 34 |
| rank ≤ 10 | 346 | 0 | 0 | 51 |
| rank = 1 | 213 | 0 | 0 | 184 |

All off-diagonal cells are exactly 0 for all three thresholds. There is no
rank movement of any kind between C1 and C2 anywhere in the 397-query
population — consistent with, and a stronger/more precise statement than,
Section 5's per-query identity check.

## 7. Candidate-generation comparison

| | C2 hit | C2 miss |
|---|---:|---:|
| **C1 hit** | 385 | 0 |
| **C1 miss** | 0 | 12 |

`c1_miss_rids == c2_miss_rids == cross_adduct_only_rids` exactly (all three
sets are identical, 12 elements, 0 symmetric difference). No candidate-generation
misses exist outside the known 12 structurally cross-adduct-only queries in
either condition — the instruction to "investigate explicitly" any miss outside
the known 12 found none to investigate.

The 12 structurally unreachable cross-adduct-only queries (identical in both
C1 and C2, since candidate generation is the same computation): rids 6296,
8769, 27493, 156672, 318845, 395356, 410447, 431909, 626670, 689395, 733288,
1029257 — the exact reachability ledger already published in
`research/06_matched_c1_control.md` §1. Median candidate count for this group:
81.5 (small, because their precursor windows mostly miss the true molecule's
spectra entirely — a structural artifact of adduct-driven mass shift, not
pool-size noise).

## 8. Ranking-only failures (candidate generated but not rank 1)

Following the same "close vs. decisive" split already established in
`research/06_matched_c1_control.md` §3 (close = found, rank 2–25; decisive =
found, rank > 25 — these two groups are disjoint and partition the
not-rank-1-but-found population; they do **not** nest, unlike a literal reading
of the brief's phrasing might suggest):

| Group | n | C1 median margin | C2 median margin | n with computable margin (C1 / C2) |
|---|---:|---:|---:|---|
| Close losses (rank 2–25) | 150 | −0.0728 | −0.0728 | 150 / 150 |
| Decisive losses (rank > 25) | 22 | −0.3909 | −0.2798* | 22 / 15 |

\* The apparent C1/C2 decisive-loss median difference is **not a real scoring
difference** — it is a truncation artifact. EXP-004 only persists the top-100
scored candidates per query; 7 of the 22 decisive-loss queries have their true
molecule ranked between 144 and 282 (rids 6959, 204510, 432363, 474736,
729359, 852894, 1070420 — exactly `exp004_full_run.json`'s own documented
`margin_anomalies` list), so C2's margin is undefined for those 7. Restricted
to the 15 queries where both C1 and C2 have a computable margin, both values
are identical (same underlying rank and score, as established in Section 5).
The C1 median (−0.3909, all 22 available) is the more complete number for this
group; it exactly reproduces EXP-005's own published decisive-loss margin
(−0.391) on the full 400-query population.

Close losses are near-misses (median margin only −0.07); decisive losses are
landslides (median margin −0.39) — the same shape already reported in EXP-005,
now confirmed to persist identically in C2 because C2 is the same computation.

## 9. C1/C2 shared failures

Using rank ≤ 25 as "success":

| Category | n | % of 397 | C1 MRR contribution | C2 MRR contribution |
|---|---:|---:|---:|---:|
| Both fail (rank > 25 or candidate-absent, both) | 34 | 8.56% | 0.000 | 0.000 |
| C1-only failure | 0 | 0.00% | — | — |
| C2-only failure | 0 | 0.00% | — | — |
| Both succeed (rank ≤ 25, both) | 363 | 91.44% | 258.811 | 258.811 |

**Are the same molecules difficult in both C1 and C2? Yes — quantified exactly:
100% of failures (34/34) are shared, 0% are unique to either condition.** This
isn't a correlation to be estimated; it follows deterministically from Section
5's finding that C1 and C2 are the same computation on the same data.

## 10. C2 regressions

Checked at three thresholds (C1 ≤25→C2>25, C1≤10→C2>10, C1=1→C2≠1): **0
regressions found at every threshold.** This is not a null result masked by
aggregation — Section 6's transition matrix shows the off-diagonal cell is
exactly 0 at every threshold, and Section 5 shows why: there is no
computational difference between C1 and C2 for any of the 397 queries to
produce a regression from. **C2 introduces no new failure mechanism, because
under this construction C2 is not a distinct experimental condition from C1.**

## 11. C2 improvements

Symmetric check (C1>25→C2≤25, C1>10→C2≤10, C1≠1→C2=1): **0 improvements found
at every threshold**, for the identical reason as Section 10.

## 12. Displacer analysis

Since C1 and C2 share every ranking outcome, the "C2 displacer" and "C1
displacer" for a given failing query are the same molecule at the same score.
Computed over the 165 non-rank-1 outcomes with a computable margin (both close
and decisive losses combined, 150 + 15 of the 22 decisive):

- Same-adduct-as-query displacer: 142/165 (86.1%)
- Cross-adduct displacer: 23/165 (13.9%)
- Displacer library origin: enveda-180 153/165 (92.7%), pluskal_ms2 7, riken 2,
  gnps 2, massbank 1.

**Comparison to EXP-003's reference statistics (54/92 cross-adduct, 38/92
same-adduct displacers) is explicitly not a like-for-like comparison and is
reported only as context, per the master prompt's own caution**: EXP-003's 92
queries were failure-selected specifically from EXP-002's adduct-aware
Variant-B run, where the candidate pool is deliberately expanded with
cross-adduct candidates. Here, under unchanged Variant-A raw-precursor
candidate generation, cross-adduct candidates can only appear when their
reported `precursor_mz` coincidentally falls within the same ±0.01 Da window
as the query despite a different adduct label — a much rarer event — which is
consistent with the much lower observed cross-adduct-displacer share (13.9% vs
59%) in this population.

**Whether a given displacer was "already present in C1" or "newly admitted":**
this question is answered trivially (always "already present," since C1 and C2
share the same candidate pool by construction) but is otherwise **not
independently verifiable from existing artifacts** for any hypothetical
different C2 construction, because EXP-001/EXP-005 only persisted the
aggregate `best_wrong_score`, never the wrong candidate's molecule identity.
This is flagged as a real limitation for future C2/C3 designs, not glossed
over.

## 13. Novelty-stratified analysis

`max_sim_modcos` distribution over the 397 eligible queries (recomputed
directly, matches the design doc's 400-query figures closely): min 0.000, p25
0.0094, median 0.0529, p75 0.1748, p90 0.3840, max 0.8485.

Bins defined from this population's own quartiles (per instruction, not chosen
a priori):

| Stratum | Range | n | Cand. recall | R@1 | R@5 | R@10 | R@25 | MRR@25 | Median rank (found) | Median candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low similarity (≤p25) | [0, 0.0094) | 99 | 0.970 | 0.505 | 0.808 | 0.869 | 0.929 | 0.635 | 1 | 434 |
| Moderate similarity (p25–p75) | [0.0094, 0.1748) | 198 | 0.965 | 0.571 | 0.818 | 0.909 | 0.934 | 0.677 | 1 | 716.5 |
| High similarity (≥p75) | [0.1748, 1.0] | 100 | 0.980 | 0.500 | 0.780 | 0.800 | 0.860 | 0.619 | 1 | 873.5 |

Spearman rank correlation between `max_sim_modcos` and C2 reciprocal rank over
all 397 queries: **−0.020** — statistically indistinguishable from no
monotonic relationship.

**Does performance degrade as retained spectral evidence becomes less
similar (more "novel")? No — and notably, the pattern is not even
monotonic in the expected direction.** The *highest*-similarity stratum (evidence
most similar to the held-out query) has the *worst* MRR@25 (0.619) and R@25
(0.86) of the three strata, and the moderate stratum outperforms both the low
and high similarity strata. This is Outcome C from the master prompt's
decision tree (mixed/non-monotonic), reported honestly rather than forced into
Outcome A or B. Given the near-zero correlation and the non-monotonic bin
pattern, the most defensible reading is that **within the range of novelty
values present in this population, similarity to retained evidence is not a
meaningful predictor of ranking outcome** — but this should not be
over-interpreted as a causal claim (see Limitations); candidate-pool size
differs substantially by stratum (434 vs 874 median candidates) and is a
plausible confound that this analysis does not control for.

## 14. Margin analysis

Win margins (rank = 1, n = 213): median +0.1277, mean +0.1882 — identical
between C1 and C2 (same underlying computation). Combined with Section 8's
close/decisive loss margins, this reproduces EXP-005's full four-way margin
shape (win / close-loss / decisive-loss / candidate-absent) exactly on the
397-query subset.

## 15. Interpretation

### FACTS
- C1 (EXP-001 Mode B, 397-query subset) and C2 (EXP-004 full run) produce
  bit-identical per-query rank, candidate count, and true-molecule score for
  all 397 eligible queries (Section 5).
- The only difference between the two experiments is which 397-of-400 (C2) vs.
  400-of-400 (C1) queries are included in the aggregate; the ≈+0.0009 MRR
  delta reported in the master prompt is entirely attributable to that
  denominator difference (Section 5).
- 100% of failures are shared between C1 and C2 (Section 9); 0 regressions and
  0 improvements exist at any rank threshold (Sections 10–11).
- 12/397 queries are structurally candidate-absent under Variant A
  (cross-adduct-only evidence), identical in both C1 and C2, fully explained
  and previously documented (Section 7).
- Ranking failures split into near-miss "close losses" (median margin −0.07,
  n=150) and landslide "decisive losses" (median margin −0.39, n=22) —
  identical in both conditions (Section 8).
- Modified-Cosine similarity between the held-out query and its most-similar
  retained same-molecule evidence shows no meaningful monotonic relationship
  with C2 ranking outcome (Spearman ≈ −0.02) and a non-monotonic pattern
  across similarity strata (Section 13).

### DECISION
EXP-004, as executed, does not provide evidence that can distinguish "C2" from
"C1" on this benchmark, because it is not a distinct computation from C1 on
the shared population — it is C1's own pipeline re-run on a near-duplicate-filtered
subset of the same 400 queries. **This is not evidence that hiding a spectrum
"doesn't matter"** — it is evidence that EXP-004's specific construction did
not test a scenario different from what EXP-001 Mode B already tests. The
already-completed novelty audit (`research/05_class1_to_class2_design.md` §4:
median retained-evidence similarity to the held-out query is 5.5%, genuinely
low) establishes that *some* meaningful spectral novelty already exists
*within* C1/EXP-004's shared population — but EXP-004 did not vary that
novelty as an experimental treatment; it only measured and stratified it
post-hoc (Section 13), and found no rank effect from doing so.

### HYPOTHESES (unproven)
- The non-monotonic novelty-stratum pattern (Section 13) may partly reflect a
  candidate-pool-size confound rather than a genuine novelty effect — untested
  here.
- Same+cross-adduct evidence outperforming same-adduct-only evidence (E vs. D
  in `results/exp004_full_run.json`'s own stratification, MRR 0.694 vs 0.638)
  may indicate that additional legitimate candidate diversity helps ranking
  under Variant A — this remains an observational correlation on this
  population, not a controlled ablation (already flagged in
  `research/CASMI_RESEARCH_STATE.md` §17).
- The 86%/14% same/cross-adduct displacer split (Section 12) suggests most
  ranking failures under Variant A are same-adduct confusions (chemically
  similar molecules at the same adduct out-scoring the true one), not an
  adduct-driven artifact — plausible given Variant A's candidate rule, not
  independently confirmed against a controlled counterfactual.

### LIMITATIONS
- Pseudo-C2 construction: EXP-004 is not, in its current form, an independent
  test of "hide one spectrum, keep the rest" beyond what EXP-001 Mode B
  already implements — see Decision above.
- Same underlying Mode-B population: all conclusions here are conditioned on
  EXP-001's original 400-query sample; no new sampling was performed.
- 100% timsTOF: no cross-instrument generalization is tested by this
  population.
- 12/397 structurally cross-adduct-only cases are a clean, understood
  candidate-generation ceiling, not a ranking phenomenon — should not be
  conflated with ranking failures in any future aggregate statistic.
- Novelty metric limitations: `max_sim_modcos` measures peak-level spectral
  similarity to the single most-similar retained spectrum; it is not a
  validated proxy for "true" chemical or instrumental novelty, and Section 13
  explicitly found no effect from it in this population — that finding is
  scoped to this metric and this population, not a general claim about
  spectral novelty.
- Possible retained spectral evidence: some retained same-molecule evidence
  may still be closer to the held-out query than a genuinely independent
  measurement would be (the high-similarity stratum reaches 0.85 Modified
  Cosine) — the near-duplicate filter only removes the most extreme 0.75% of
  cases.
- Controlled-benchmark vs. hidden test: this entire analysis operates on known
  training-data molecules under leakage controls; it does not directly
  measure performance on the actual hidden CASMI test set.
- Displacer continuity (already flagged in Section 12): cannot be assessed
  across hypothetically different C1/C2 conditions because C1's artifacts
  never persisted per-candidate molecule identity.

## 16. Decision recommendation

1. **Did C2 materially change retrieval difficulty?** Not measurably — and it
   could not have, because as constructed it is the same computation as C1 on
   (almost) the same population (Section 5).
2. **Did C2 create new failure modes?** No — 0 regressions, 0 improvements, 0
   condition-unique failures at any threshold (Sections 9–11).
3. **Are failures mostly inherited from C1?** All of them, not merely "mostly"
   — 100% of C2's 34 failures are the identical 34 queries that fail under C1
   (Section 9).
4. **Does novelty correlate with rank degradation?** No meaningful correlation
   was found (Spearman ≈ −0.02); the relationship across similarity strata is
   non-monotonic, not a clean "more novel → worse" trend (Section 13).
5. **What evidence is now required from C3?** Before any C3 design, this
   analysis implies EXP-004's approach needs a genuine methodological fix if
   the goal is to test "known molecule + unseen spectrum" as a condition
   distinct from C1: either (a) a query construction that holds out a
   *specific* spectrum while deliberately varying which/how-similar the
   retained evidence is (an actual experimental manipulation of novelty, not
   just a post-hoc audit of whatever Mode B happened to retain), or (b) an
   explicit acknowledgment that Mode B already *is* the operational
   pseudo-C2 benchmark for this project, and future novelty stratification
   should be treated as a descriptive lens on C1's existing failures (as done
   in Section 13) rather than as a separate experimental arm. Pseudo-C3 design
   should proceed only after this C2-methodology question is resolved, since
   C3 will need to build on a C2 story that is actually a controlled
   comparison, not a re-run of C1 on a filtered population.

---

Machine-readable outputs: `results/exp004_c2_analysis/summary.json`,
`results/exp004_c2_analysis/query_comparison.jsonl` (397 rows).

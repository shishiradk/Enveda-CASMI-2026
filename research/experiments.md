# Experiment Log — Enveda CASMI 2026

Every experiment gets an ID (`EXP-001`, `EXP-002`, ...), is never overwritten, and records
enough (code/config/data version/seed) to be re-run. See `research/00_dataset_report.md` for
the Phase 0/1 groundwork this experiment builds on.

Template for each entry:

```text
## EXP-XXX — <short title>

Date:
Class: 1 / 2 / 3
Hypothesis:
Method:
Parameters:
Dataset / split:
MRR@25:
Top-1 / Top-5 / Top-10:
Recall@25:
Runtime:
Memory:
Candidate count:
Result:
Interpretation:
Failure cases:
Next action:
```

---

## EXP-001 — Class 1 Spectral Library Coverage

Date: 2026-09-20

Class: 1

Hypothesis: If a molecule's spectrum genuinely exists in the library, precursor-filtered
+ Modified Cosine retrieval finds it in the top 25 most of the time; and (DEC-002)
restricting the search to `enveda-180` (timsTOF) performs at least as well as the full
2.54M-row train set for this.

Method: Two leakage-safe validation modes (Mode A: near-exact duplicate ceiling test,
peak-level-validated at cosine>=0.95; Mode B: same molecule via a genuinely different
metadata-group, whole group excluded) x three reference-library variants (enveda-180;
enveda-180+enveda-np-examples; all-train) x matchms 0.33.1 `ModifiedCosineGreedy`
rescoring on precursor-filtered candidates, molecule-level max-score aggregation. Full
methodology, leakage controls, and interpretation in `research/02_class1_coverage.md`.

Parameters: precursor tolerance 0.01 Da (adduct-naive), matchms peak tolerance 0.1 Da,
mz_power=0.0, intensity_power=1.0, near-dup validation threshold 0.95 direct-cosine,
seed=42, Mode A n=200, Mode B n=400, multi-spectrum add-on n=150x2.

Dataset / split: `train.parquet`, timsTOF (`enveda-180`/`enveda-np-examples`) molecules
only for query sampling, per `research/02_class1_coverage.md` §2-3.

MRR@25: Mode A 0.6629-0.6654 across libraries; Mode B 0.6502-0.6510 across libraries
(final, corrected run — see Stages below for earlier, superseded numbers)

Top-1 / Top-5 / Top-10: Mode B / all-train: 0.5350 / 0.8050 / 0.8725

Recall@25: Mode A ~0.960 (all libraries); Mode B ~0.9150 (all libraries)

Runtime: 2,517.3s (~42 min) for the full corrected run

Memory: peak RSS 3,857 MB (Mode B/all-train, the largest condition), bounded per
condition, final RSS 1,838 MB — see Stages below for the earlier OOM-killed attempt

Candidate count: median 232-300 (Mode A), median 400-690 (Mode B), depending on library

Result: Library choice barely matters (MRR@25 spreads of 0.0008-0.0025 across variants,
within sampling noise) — the DEC-002 hypothesis as literally stated ("enveda-180 beats
all-train") is not confirmed; the real driver is that the 0.01 Da precursor window
filters far more aggressively than library identity does. Modified Cosine and direct
Cosine are byte-identical on every condition in this pipeline, traced to the precursor
tolerance being tighter than matchms's own 0.1 Da shift-fallback threshold (confirmed
correct via a separate diagnostic showing the shift branch does activate on a real
cross-adduct pair, outside this pipeline). Multi-spectrum max-aggregation beats a
single-spectrum baseline on every metric (real, non-oracle comparison).

Interpretation: Class 1 (near-exact duplicate in library) is strongly solvable (~96%
top-25 coverage). The more realistic case (same molecule, different spectrum) reaches
~91.5% top-25 coverage, an ~8.5-point gap split roughly evenly between candidates never
being generated (2.8-3.5%) and correct candidates being ranked too low (5.5%). Full
interpretation in `research/02_class1_coverage.md` §14-15.

Failure cases: n=400 (Mode B/all-train): 366 rank 1-25 (91.5%), 22 found but ranked
below 25 (5.5%), 11 molecule absent from candidate pool (2.8%), 1 zero-candidate
failure (0.2%). Of 34 non-rank-1 failures, sub-categorization found 0 attributable to
adduct mismatch, 1 to an extreme spectrum (>2000 peaks), 21 unexplained ("other") —
flagged as unknown, needs direct case inspection.

Next action: EXP-002 candidate — adduct-aware (neutral-mass-based) candidate generation
to let genuinely cross-adduct same-molecule candidates into the pool, since the
precursor filter currently excludes them structurally and Modified Cosine's
shift-matching (confirmed working) never gets a chance to help inside the real
pipeline. Secondary: inspect the 21 "other" failure cases directly. **STOP — awaiting
review before either of these or any Class 2 work begins.**

### Stages (all preserved, not overwritten, per the requirement to distinguish them)

1. **Preliminary, leakage-uncontrolled** (`results/exp001_results.json`): the very
   first pass, direct cosine only, excluded only the single held-out row (not
   near-duplicate siblings) — some "different spectrum" claims in this run would
   actually have been near-exact matches in disguise. Variant A (timsTOF)
   Recall@25=0.9425, MRR@25=0.6808; Variant B (full train) Recall@25=0.9275,
   MRR@25=0.6677. Superseded — kept only as an early exploratory data point.
2. **Smoke test** (tiny N=15 Mode A/15 Mode B/10 multi-spec): used to validate the
   leakage-safe Mode A/B construction, the Modified Cosine toy self-check, and
   candidate/ranking correctness before committing to the expensive full run. Not a
   result in its own right — see the six-point evidence report given at the time.
3. **Partial run, killed by memory pressure** (`results/exp001_partial_run1_KILLED_preliminary.md`):
   full-scale (N=200/400/150) attempt using the corrected leakage-safe methodology,
   killed by Claude Code's own memory-pressure safeguard after 4/6 primary conditions
   completed. Those 4 conditions' numbers are close to (within ~1 point of) the final
   run below, differing due to non-deterministic row ordering in duckdb's multi-threaded
   `list()` aggregates (see `research/02_class1_coverage.md` §16 point 3) — not a
   methodology error. Preserved as PARTIAL/PRELIMINARY, not final.
4. **Final, corrected, complete run** (this entry; `results/exp001_checkpoint.json`,
   `results/exp001_query_sets.json`): memory-safe (per-condition fetch/release,
   incremental checkpointing), all 6 primary conditions + direct-cosine comparison +
   multi-spectrum add-on + failure analysis completed successfully. **This is the
   authoritative EXP-001 result.**

---

## EXP-002 — Adduct-Aware Candidate Generation

Date: 2026-09-21

Class: 1

Hypothesis: Class-1 retrieval loses recall because candidate generation compares
reported precursor m/z directly instead of accounting for adducts / neutral molecular
mass; adduct-aware candidate generation should recover the queries EXP-001 found
missing for this reason.

Method: Reused EXP-001's exact Mode B query set (n=400, unchanged, no re-sampling) and
EXP-001's exact Variant-A (baseline) scoring results. Built Variant B: candidate
generation by comparing adduct-derived neutral mass (`(precursor_mz - delta) / n`,
13-adduct table incl. multimers) instead of raw precursor_mz, tolerance 0.01 Da,
falling back to Variant A's rule for unsupported adducts (none occurred in this
sample). Identical scorer (matchms 0.33.1 `ModifiedCosineGreedy`, same parameters) for
both variants — the ranker was not touched. Full design in
`research/03_exp002_design.md`, full results in `research/03_exp002_results.md`.

Parameters: precursor tolerance 0.01 Da (Variant A), neutral-mass tolerance 0.01 Da
(Variant B), candidate-explosion safety cap 50,000/query (never triggered), batch
size 50 queries (memory-safe, checkpointed).

Dataset / split: identical to EXP-001's Mode B (`results/exp001_query_sets.json`), all
train (library C).

MRR@25: Variant A 0.6510 -> Variant B 0.6178 (delta -0.0332)

Top-1 / Top-5 / Top-10: Variant A 0.5350/0.8050/0.8725 -> Variant B
0.4900/0.7800/0.8550 (all declined)

Recall@25: Variant A 0.9150 -> Variant B 0.9200 (delta +0.0050, marginal)

Runtime: 467.9s (~7.8 min) for the scoring stage; candidate generation + classification
~41s

Memory: peak RSS 1,395 MB, bounded and released between batches, no OOM risk

Candidate count: median grew from ~690 (A) to ~1,224 (B) overall (2.74x for
already-succeeding queries specifically); max 5,020 (monomer) / 4,763 (multimer);
explosion cap never triggered

Result: candidate-generation recall reached a perfect 100% (from 97.0%, +0.03) --
all 12 of EXP-001's candidate-absence failures were recovered, and Modified Cosine's
shift-matching was confirmed (per-candidate, not assumed) to be doing real work for
9 of those 12. BUT final ranking metrics net declined: of the 388 queries that already
had the correct candidate under Variant A, 84 got worse ranks (vs 25 improved), 30
lost rank-1 status, and 8 fell out of the top-25 entirely despite being correctly
ranked before. The 8 net new top-25 successes from recovered candidates were
essentially cancelled out by these 8 newly-failed previously-correct queries, and rank
degradation among still-successful queries directly hurt Recall@1/5/10 and MRR@25.

Interpretation: candidate generation was not the whole bottleneck. Fixing it exposed
that the ranking function (simple molecule-level max Modified-Cosine score) does not
scale gracefully to a larger, more heterogeneous candidate pool -- more candidates
means more chances for a spuriously higher-scoring wrong molecule to outrank the
correct one. Also discovered and confirmed empirically (not assumed): Variant A's
raw-precursor rule was itself admitting many false-positive candidates (different
molecule, different adduct, coincidentally close raw precursor_mz -- e.g. an
[M-H]-/[M+H]+ pair of unrelated molecules landing within 0.01 Da by chance despite a
true same-molecule pair at those adducts differing by ~2.01 Da). Variant B correctly
excludes most of these, meaning candidate-pool "growth" numbers understate how much
the pool composition actually changed.

Failure cases: of the 12 previously-absent queries, 3 remain unranked in the top 25
even after being correctly recovered as candidates (rid 733288, 410447, 318845) --
for these specifically, ranking (not candidate generation) is now the confirmed
bottleneck.

Next action: NOT adopted as-is. Two follow-up directions identified, neither executed
pending review: (a) a precision-focused refinement (Variant C's pair-curation, or a
tighter neutral-mass tolerance) to capture the recall gain with less noise growth; (b)
a ranking-side fix (secondary tie-break, score calibration, or a pre-scoring candidate
cap) to make the ranking function robust to larger candidate pools -- explicitly a
ranking question, out of scope until separately approved. **STOP -- awaiting review.**

---

## EXP-005 — Matched Class-1 Row-Exclusion Control (prerequisite check for EXP-004)

Date: 2026-09-21

Class: 1 (control experiment, gating the EXP-004 Class-1->pseudo-Class-2 smoke test)

Hypothesis: EXP-001 Mode B / EXP-004's whole-metadata-group exclusion might still
allow a near-duplicate spectrum of the query to remain searchable via some OTHER path
than the query's own group, making the "genuinely withheld evidence" claim weaker than
believed. A matched row-only-exclusion control (same queries, same candidate
generation, same scorer) should reveal how much this specific mechanism matters.

Method: Reused EXP-001's exact 400 Mode-B query rids, Variant-A candidate generation
(raw precursor_mz +/- 0.01 Da, library = all train), and matchms 0.33.1
ModifiedCosineGreedy (tolerance=0.1, mz_power=0.0, intensity_power=1.0). Each query's
candidate pool is scored once; two conditions (L = row-only exclusion, W = whole-group
exclusion, identical to EXP-001 Mode B) are derived from the same scored pool so the
comparison is exactly paired. Smoke test (n=25) run and verified (rid reconstruction,
external agreement with EXP-001's stored per-query results, correct differential
leakage behavior) before the full n=400 run. Full design and results in
`research/06_matched_c1_control.md`. A separate corrected audit
(`research/scripts/exp004_reaudit.py`) also fixed a same-adduct/cross-adduct
classification bug in `results/exp004_novelty_audit.json` and produced a 12-query
reachability ledger (see that script and `research/06_matched_c1_control.md` Sec.1).

Parameters: identical to EXP-001 (precursor tolerance 0.01 Da, matchms peak tolerance
0.1 Da, mz_power=0.0, intensity_power=1.0), no new parameters introduced.

Dataset / split: identical to EXP-001's Mode B (`results/exp001_query_sets.json`), all
train (library C).

MRR@25: Condition L 0.6510, Condition W 0.6510 (bit-identical; both match EXP-001's
originally published Mode B/all-train result exactly, an external validation of this
freshly written pipeline)

Top-1 / Top-5 / Top-10: identical between L and W: 0.5350 / 0.8050 / 0.8725

Recall@25: L 0.9150, W 0.9150 (identical)

Runtime: ~175s total (48.7s table materialization + 126.7s scoring, n=400)

Memory: not separately profiled; single in-memory pandas table of train.parquet's
needed columns, same order of magnitude as EXP-001/002.

Candidate count: median ~690 (unchanged from EXP-001, same candidate-generation rule)

Result: row-exclusion and group-exclusion produce IDENTICAL aggregate metrics because
only 8/400 (2%) Mode-B queries have any metadata-group sibling to exclude in the first
place; of those 8, 7 show zero effect beyond one extra candidate in the pool, and the
8th shows a real but outcome-irrelevant score change (rank stayed 13th either way). The
12-query reachability ledger independently confirms the corrected EXP-004 audit: all 12
of EXP-001's Variant-A candidate-absent queries are exactly the queries with no
same-adduct evidence retained at all (previously mis-reported as 108 due to a bug now
fixed). Correct-vs-top-wrong margin analysis: rank-1 successes win by a comfortable
median +0.127; correct-but-not-rank-1 losses are typically close (median -0.072);
correct-but-outside-top-25 losses are decisive (median -0.391).

Interpretation: the leakage-mechanism concern that motivated EXP-004's design (Mode B
secretly remaining "Class 1 with the query spectrum renamed" via group-exclusion being
too weak) is not a real risk in this population -- ruled out with a matched,
bit-identical-result experiment, not just an assumption. The separate, already-measured
novelty axis (retained-evidence Modified-Cosine similarity, median 5.5%,
`research/05_class1_to_class2_design.md` Sec.4) remains the operative one. Ranking
failures, when candidates are present, are mostly close calls (median margin -0.072 for
the 152 non-rank-1-but-found queries), suggesting headroom for a calibration/tie-break
fix distinct from a candidate-generation fix.

Failure cases: 12/400 candidate-absent (fully explained, see ledger), 152/400 found but
not rank 1 (mostly close margin), 22/400 found but outside top 25 (decisive margin
loss), 214/400 rank-1 success.

Next action: this was a required prerequisite check before the EXP-004 C2 smoke test,
not the smoke test itself. No leakage confound found; recommend proceeding to the
EXP-004 smoke test is a decision for explicit user approval, not automatically granted
by this result. **STOP -- awaiting approval for the EXP-004 smoke test specifically.**

---

## EXP-004 — Class-1 -> Pseudo-Class-2 (smoke test + full 397-query run)

Date: 2026-09-21

Class: 1 -> pseudo-2 (query spectrum withheld; genuinely different same-molecule
evidence, where present, remains searchable)

Hypothesis: with the query's own spectrum (and its whole metadata group) withheld,
how much of EXP-001's Class-1 performance survives using only genuinely different
same-molecule evidence, and how much is a pure candidate-generation ceiling (no
same-adduct evidence reachable by Variant A at all) versus a ranking effect?

Method: population = EXP-001's 400 Mode-B queries minus 3 near-duplicate rids (302973,
944352, 519143) = 397 eligible (near-dup rule: max Modified-Cosine similarity >= 0.90
to retained same-molecule evidence, justified as this experiment's own data-driven
threshold, not inherited from EXP-001 -- see `research/05_class1_to_class2_design.md`
Sec.4). Candidate generation: EXP-001 Variant A, unchanged (raw precursor_mz +/- 0.01
Da, all-train library). Leakage control: whole metadata-group exclusion, reused
unchanged from `results/exp001_query_sets.json`. Scorer: matchms 0.33.1
ModifiedCosineGreedy (tolerance=0.1, mz_power=0.0, intensity_power=1.0), unchanged. No
Variant B, no CE split (per the corrected, merged Scenario stratification --
`research/05_class1_to_class2_design.md` Sec.6).

Two stages, both required before interpreting results (per this project's staged-
execution protocol):
1. **Smoke test** (n=28, seed=20260921, deterministically stratified: 8 cross-
   adduct-only + 10 same-adduct-only + 10 same+cross-adduct, sample manifest written
   before scoring) -- `research/scripts/exp004_smoke_test.py`,
   `results/exp004_smoke_test.json`. All validation checks passed: 0 leakage failures,
   0 reproducibility mismatches vs EXP-001's checkpoint, 0 reachability-classification
   surprises (all 8 sampled cross-adduct-only rids correctly showed
   `candidate_gen_hit=False`; all 20 sampled same-adduct-retained rids correctly showed
   `True`). A separate artifact-only margin sanity check on the persisted top-50
   candidates found 0 anomalies and confirmed the margin distribution's direction and
   rough magnitude were consistent with the EXP-005 baseline (not treated as a
   performance estimate, per instruction, given n=20).
2. **Full run** (all 397 eligible) -- `research/scripts/exp004_full_run.py`,
   `results/exp004_full_run.json`. Top-100 per-query candidate scores/metadata
   persisted for every query (rank, molecule inchikey14, score, adduct, origin
   library) so future margin/near-miss analysis never needs to re-run the scorer.

Parameters: identical to EXP-001 (precursor tolerance 0.01 Da, matchms peak tolerance
0.1 Da, mz_power=0.0, intensity_power=1.0); no new parameters.

Dataset / split: EXP-001's Mode-B query rids minus 3 near-duplicates (397/400), all
train (library C).

MRR@25: A (overall, n=397) 0.6519; B (reachable-only, n=385) 0.6722; C (same-adduct
retained, n=385) 0.6722 [identical to B by construction: reachability and same-adduct
evidence coincide exactly on this population]; D (same-adduct-only, n=149) 0.6379; E
(same+cross-adduct, n=236) 0.6939; F (cross-adduct-only, n=12) undefined (0%
candidate-gen recall, no ranking applies).

Top-1 / Top-5 / Top-10 (overall, n=397): 0.5365 / 0.8060 / 0.8715

Recall@25: A 0.9144; B/C 0.9429; D 0.9329; E 0.9492; F 0.0000

Runtime: 116.5s table materialization + 327.2s scoring (~7.4 min total, n=397)

Memory: not separately profiled; same order of magnitude as EXP-001/002/005.

Candidate count: median 687 (overall), 838 (same-adduct-only), 667.5 (same+cross),
81.5 (cross-adduct-only, small because these queries' precursor windows mostly miss
the true molecule's spectra entirely -- consistent with structural unreachability, not
a general candidate-pool-size effect)

Result: with the query's own metadata group withheld, overall Class-1-to-pseudo-C2
performance (MRR@25 0.6519, R@25 0.9144) closely tracks EXP-001's original Mode-B
result (MRR@25 0.6510, R@25 0.9150) -- consistent with EXP-005's finding that
whole-group exclusion introduces no material change versus what EXP-001 already
measured (this population overlaps EXP-001's 400 almost entirely, 397/400, scored
under the same unchanged pipeline). The 12 cross-adduct-only queries are a clean,
fully-expected 0% candidate-generation ceiling under Variant A, not a ranking failure
-- reconfirms the corrected reachability ledger (`research/06_matched_c1_control.md`
Sec.1) at full population scale rather than the 8-query smoke subset. Margin analysis
(n=378 with computable margin; 7 excluded because their true molecule ranked beyond the
persisted top-100, ranks 144-282, a top-K truncation artifact not a scoring bug):
rank-1 median margin +0.1277, correct-not-rank-1 median -0.0728, correct-outside-top-25
median -0.2798 -- all closely matching EXP-005's Class-1 margin baseline (+0.127 /
-0.072 / -0.391 respectively), i.e. the same close-loss-vs-decisive-loss shape found in
EXP-005 replicates under EXP-004's population and construction. Top-wrong-molecule
adduct is same-as-query in 313/378 (83%) of margin cases, cross-adduct in 65/378 (17%).

Interpretation: no major regime shift is observed moving from EXP-001's Mode-B
construction to EXP-004's near-duplicate-filtered population -- expected, since the two
populations overlap almost completely and use an identical pipeline; EXP-004's real
contribution here is the corrected, verified 12-query candidate-generation ceiling and
a full-population (not outcome-selected) margin distribution that replicates EXP-005's
shape at n=378 rather than n=20. Same-adduct-only queries (D, MRR 0.6379) perform worse
than same+cross-adduct queries (E, MRR 0.6939) -- the presence of additional
cross-adduct evidence in the library appears to help here (more candidate diversity)
rather than hurt, the opposite direction from EXP-002's Variant-B finding; this
population-level correlation is observational, not a controlled ablation of "does
cross-adduct evidence help," and should not be read as contradicting DEC-004.

Failure cases: 12/397 candidate-absent (cross-adduct-only, fully explained), 150/397
found but not rank 1 (mostly close-margin losses), 15/397 found but outside top 25
(decisive-margin losses, consistent with EXP-005's shape), 213/397 rank-1 success.

Next action: per explicit instruction, STOP after this run. No Class-3 work launched.
Full results reported for scientific review; no further experimental design decisions
made in this entry. **STOP -- awaiting review.**

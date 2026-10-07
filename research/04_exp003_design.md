# 04 — EXP-003 Design: What Causes EXP-002's Ranking Degradation?

Status: **DESIGN ONLY. Not executed at full scale.** Per instruction, this document
proposes EXP-003 and reports the required pre-design evidence (a targeted, real
re-analysis of the 92 queries whose rank got worse or who fell out of the top-25 in
EXP-002). No full ablation run, no decision-log entry, and no production pipeline
change has been made.

---

## 1. Research question

> Are Modified-Cosine scores globally comparable across same-adduct and cross-adduct
> candidates, and do cross-adduct candidates cause incorrect high-ranked candidates to
> displace correct same-adduct candidates? More generally: what actually causes the
> ranking degradation observed when moving from raw-precursor to adduct-aware
> (neutral-mass) candidate generation?

Framed as four non-exclusive hypotheses to distinguish with evidence, not assumption:

- **H1** — the problem is simply candidate-pool size (more candidates, more chances to lose).
- **H2** — cross-adduct candidates have poorly calibrated/comparable Modified-Cosine scores.
- **H3** — removing Variant-A's accidental false positives changes ranking dynamics.
- **H4** — no single dominant mechanism; multiple effects contribute.

## 2. Evidence from EXP-001 and EXP-002

EXP-001 established the baseline Class-1 pipeline (precursor-filtered Modified Cosine,
Mode B/all-train: MRR@25 0.6510). EXP-002 found that adduct-aware candidate generation
(Variant B) raised candidate-generation recall to a perfect 100% (from 97.0%) but
*lowered* MRR@25 to 0.6178 and Recall@1 to 0.490 (from 0.535). Of the 388 queries that
already had the correct candidate under Variant A, 84 got worse ranks, 30 lost rank-1
status, and 8 fell out of the top-25 entirely. EXP-002 also found, contrary to the
EXP-002 design's own assumption, that Variant B is **not** a superset of Variant A:
398/400 queries had some Variant-A candidates that Variant B correctly excluded,
because Variant A's raw-precursor rule was itself admitting false positives (unrelated
molecules at unrelated adducts, coincidentally close in raw precursor value). This
means Variant A → Variant B changes candidate pools in (at least) two directions at
once — additions (legitimate cross-adduct matches) and removals (false-positive
noise) — so EXP-002 alone cannot attribute the ranking degradation to either
mechanism. That is precisely EXP-003's job.

## 3. Candidate-level data available (and what had to be freshly computed)

EXP-002's checkpoint (`results/exp002_checkpoint.json`) persisted, per query: the
correct molecule's best score and rank under B, candidate counts, and (for newly
admitted true-molecule candidates only) a direct-vs-modified-cosine shift comparison.
It did **not** persist per-competing-molecule scores or adduct tags for the *wrong*
candidates — that data only existed transiently in memory during EXP-002's scoring
loop. To ground this design in real evidence rather than speculation, a **targeted
re-analysis** (not the full EXP-003 experiment) was run for exactly the 92 queries
identified as "worsened" (84) or "newly failed" (8) in EXP-002 (`results/exp003_inspection.json`):
full Variant-A and Variant-B candidate sets were recomputed, every candidate was
scored with the identical Modified Cosine call used throughout, and each result was
tagged same-adduct/cross-adduct (relative to the query) and old/new (present in
Variant A or newly admitted by Variant B).

### 3a. Real findings from the 92-query targeted re-analysis

- **90/92 (98%)** of these "went wrong" queries have the **correct** answer at the
  **same adduct** as the query — this is not primarily a story about a cross-adduct
  correct-candidate being poorly scored. The correct candidate's own adduct identity
  barely changed between A and B for these cases.
- The **displacing** (top wrong-scoring) candidate is cross-adduct in **54/92 (59%)**
  and same-adduct in **38/92 (41%)** — a majority but not an overwhelming one.
- The displacer was **newly admitted by Variant B** in **57/92 (62%)**, but was
  **already present under Variant A** in **35/92 (38%)**. This is the most
  important single finding motivating the ablation design below: over a third of the
  time, the specific candidate row that ends up outranking the correct answer under B
  was *already in the pool under A too* — meaning simply "a new candidate showed up"
  does not explain all of the degradation. Something about the overall pool
  composition changes the effective competition even for candidates that didn't
  change. This is not yet mechanistically explained by this targeted inspection alone
  (see §9) and is exactly what the full ablation is designed to isolate.
- Score quantiles (this 92-query subset): correct-same-adduct scores have median 0.791
  (wide spread, p25=0.590); top-wrong-same-adduct scores have median **0.929**;
  top-wrong-cross-adduct scores have median **0.939**. **Both** same-adduct and
  cross-adduct wrong candidates are highly competitive — this argues against H2 (a
  pure cross-adduct-miscalibration story) being the sole explanation, since
  same-adduct wrong candidates are just as often the problem.

These are preliminary observations from a non-random, outcome-selected subset (the 92
queries where something already went wrong) — they motivate the ablation design but
are not a substitute for it, since this subset cannot by construction speak to
category-1's 270 unchanged or 25 improved queries, nor to whether the same patterns
hold in a representative sample.

## 4. Score-comparability analysis plan

For the full 400-query run (not yet executed), collect, per query:
- Correct candidate's score, its adduct-class (same/cross), and whether its specific
  candidate row was in Variant A.
- The single highest-scoring wrong candidate overall, plus separately the highest-
  scoring wrong same-adduct and highest-scoring wrong cross-adduct candidate (as done
  for the 92-query subset in §3a).
- The margin: correct score − top-wrong score (signed; negative means correct lost).

Report full quantile distributions (not just means) for four populations, over **all**
400 queries this time (not just the worsened subset): (A) correct/same-adduct, (B)
correct/cross-adduct, (C) wrong/same-adduct, (D) wrong/cross-adduct. Compare medians,
IQRs, and the fraction of each population scoring above common thresholds (e.g. 0.7,
0.9) to assess whether a score of, say, 0.8 means the same thing across categories —
directly answering the "are scores globally comparable" half of the research question.

## 5. Candidate-set ablations

Four controlled candidate-generation rules, each isolating a specific causal
comparison. All four reuse the identical EXP-001/EXP-002 Mode B query set, leakage
exclusions, scorer, and aggregation rule — only the candidate *set* passed into
scoring differs.

**A. Variant-A baseline** (EXP-001/EXP-002's rule, reused unchanged). The reference
point every other arm is compared against.

**B. Variant-B (EXP-002's adduct-aware rule)**, reused unchanged from EXP-002's own
checkpoint where possible (no need to rescore what's already scored) — the "both
changes at once" arm.

**C. Variant-B minus cross-adduct candidates** (same-adduct-only subset of B's
candidate set — i.e. Variant B's candidates filtered to `adduct == query.adduct`).
**Answers:** does removing Variant-A's raw-precursor false positives *by itself*
(without adding any cross-adduct candidates) already change rankings? If C's metrics
are close to A's, false-positive removal (H3) has little effect in isolation. If C's
metrics are already worse than A's, false-positive removal alone is *part* of the
degradation mechanism, independent of adding cross-adduct candidates at all — a
direct test of H3 in isolation from H1/H2.

**D. Variant-A plus only the legitimate newly-recoverable cross-adduct candidates**
(Variant A's own candidate set, with only the specific cross-adduct candidates that
EXP-002's classification identified as "newly admitted under B" added — nothing
removed). **Answers:** does *adding* legitimate cross-adduct candidates, without
removing any of Variant A's false positives, by itself degrade ranking? If D's
metrics are close to A's, adding cross-adduct candidates in isolation is not harmful
(H2 not much supported) — the degradation would then come from false-positive
removal (C) or from the interaction/pool-size effect (H1) rather than the cross-adduct
candidates themselves. If D's metrics already show meaningful degradation, cross-adduct
candidate addition alone (H2/H1) is implicated, independent of any false-positive
removal.

Together, A/B/C/D form a 2×2-ish decomposition: C isolates "remove false positives
only," D isolates "add cross-adduct only," and B (=C+D's combined effect, though not
exactly additive since both changes interact) is the full EXP-002 change. Comparing
A→C, A→D, and A→B lets us read off how much of B's degradation each individual change
explains, and whether they combine additively (supporting H1 simply via combined pool
churn) or interact (supporting H4).

A fifth, smaller comparison is worth including cheaply: **pool-size-matched control**
— for category-1 queries only, an artificial candidate set of the same *size* as
Variant B's but built by randomly adding same-adduct-only "filler" candidates to
Variant A's set (not cross-adduct ones) up to B's candidate count. This directly tests
H1 (pure size) in isolation from adduct composition: if this pool-size-matched-but-
composition-unchanged set degrades ranking about as much as B does, size alone is
doing most of the work; if it does not, composition (H2/H3) matters more than raw
count.

## 6. Leakage controls

All four (five) variants filter from the *same* leakage-safe base: each query's own
`excluded_rids` (the whole metadata group, per EXP-001's Mode B construction) is
applied identically in every arm, since candidate generation for every variant queries
`train_index WHERE ... AND rid NOT IN (excluded_rids)`. Variant C and D are strict
subsets/supersets built by filtering/adding to already-leakage-safe sets (A and B),
so no new leakage path is introduced — verified by construction (a set operation on
two already-verified-safe sets cannot reintroduce an excluded rid). The pool-size-
matched control in §5 must ensure its "filler" candidates are also drawn with the
same exclusion applied (sampled from `train_index WHERE adduct=query.adduct AND rid
NOT IN excluded_rids`, disjoint from A's existing set) — this will be explicit in the
implementation, not assumed.

## 7. Metrics

Per variant (A, B, C, D, and the pool-size-matched control): candidate-generation
recall, Recall@1/5/10/25, MRR@25, median/mean/percentile candidate counts, runtime,
peak RSS — identical metric set to EXP-001/EXP-002 for direct comparability. Plus the
category-1-style decomposition from EXP-002 §6 (same rank / improved / worsened /
newly failed / newly succeeded), computed for each variant against the Variant-A
baseline, so degradation (or lack thereof) is visible query-by-query, not just in
aggregate.

## 8. Failure-case taxonomy

Extending EXP-001/EXP-002's categories with the adduct/origin tags now available:
1. Correct absent from candidate pool (candidate-generation failure).
2. Correct present, ranked 1–25 (success).
3. Correct present, ranked below 25, displaced by a **same-adduct, previously-present** wrong candidate.
4. Correct present, ranked below 25, displaced by a **same-adduct, newly-admitted** wrong candidate.
5. Correct present, ranked below 25, displaced by a **cross-adduct, previously-present** wrong candidate.
6. Correct present, ranked below 25, displaced by a **cross-adduct, newly-admitted** wrong candidate.
Categories 3–6 map directly onto the 2×2 (same/cross × old/new) breakdown already
computed for the 92-query subset in §3a, extended to all 400 queries and to every
variant, not just B.

## 9. Expected interpretations for H1–H4

- **If C (remove-false-positives-only) shows most of B's degradation, and D
  (add-cross-adduct-only) shows little:** H3 dominates — Variant A's own false
  positives were, unintuitively, helping more than hurting (perhaps by "absorbing"
  probability mass in the aggregation, or by coincidence in this scorer/dataset), and
  their removal is the main driver. This would be a genuinely surprising, high-value
  finding worth its own follow-up.
- **If D shows most of the degradation and C shows little:** H2 (or H1 restricted to
  cross-adduct additions specifically) dominates — legitimate cross-adduct candidates
  really are scored in a way that lets them wrongly outcompete correct same-adduct
  matches. This would directly support pursuing a cross-adduct-specific score
  correction/calibration as the next step.
- **If the pool-size-matched control (same-adduct filler, no composition change)
  degrades ranking about as much as B:** H1 dominates — raw pool size is the driver,
  regardless of what's added. This would support a ranking-side fix independent of
  candidate-generation chemistry (e.g. a pre-scoring shortlist/cap).
- **If none of C, D, or the size-matched control individually reproduces most of B's
  degradation, but B still shows it:** H4 — the effects interact and no single
  ablation isolates it; §3a's finding that 38% of displacers were already present
  under A (suggesting a compositional/interaction effect, not simple addition) makes
  this the currently most evidence-consistent expectation, though it is not
  prejudged and the full run may show otherwise.

## 10. Smoke-test plan

Before the full run: repeat the same style of targeted, real check already done in
§3a, but for a small (~20–30 query) *random* sample of category-1 queries (not
outcome-selected this time), scoring Variants A, C, and D for that sample only, and
confirming (a) leakage exclusions hold (no excluded rid appears in any variant's
candidate set — automatic check), (b) candidate-set relationships behave as designed
(C ⊆ B, D ⊇ A, D's added candidates are exactly the ones classified "newly admitted"
in EXP-002), and (c) no candidate-pool explosion beyond EXP-002's already-established
bounds (50,000/query cap, never triggered in EXP-002). Only after this smoke test
passes would the full 400-query, 4-5-variant run be proposed for execution.

## 11. Full-run configuration (proposed, NOT executed)

If approved: reuse the exact 400 Mode B queries and leakage exclusions unchanged;
compute candidate sets for A (reused from EXP-001/EXP-002 checkpoints where possible),
B (reused from EXP-002's checkpoint directly — no rescoring needed), C, D, and the
pool-size-matched control; score C, D, and the control with the identical
`ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)` call and
molecule-level max aggregation, using the same batched (BATCH_SIZE≈50), memory-safe,
incrementally-checkpointed architecture as `exp002_full.py` (fetch/score/release per
batch, resume-safe). Expected runtime: roughly comparable to EXP-002's own ~468s
scoring stage per additional variant scored (C and D; the control if included), so a
few-times-EXP-002's runtime in total, well within the "does not need to finish in one
uninterrupted process" tolerance already established. Output: `results/exp003_checkpoint.json`
(new artifact, EXP-001/EXP-002 outputs untouched) and, after review, a
`research/04_exp003_results.md` write-up following the same rigor as EXP-002's.

**No full run executed. No decision-log entry made. No production pipeline change.
Awaiting review.**

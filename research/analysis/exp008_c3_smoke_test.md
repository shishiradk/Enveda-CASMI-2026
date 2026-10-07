# EXP-008 — Pseudo-Class-3 smoke test: execution report

Date: 2026-09-24. Authorization: "EXP-008 SMOKE TEST — AUTHORIZED" (user, 2026-09-24).
Script: `research/scripts/exp008_c3_smoke.py` (sha256 `a49fa4a0a72a0109…`, identical to the preflight manifest).
Spec: `research/analysis/exp008_c3_smoke.md`. Design: `research/analysis/exp008_c3_design.md`.
Artifacts: `results/exp008_c3_census.json`, `results/exp008_c3_smoke_manifest.json` (unchanged),
`results/exp008_c3_smoke_query_results.jsonl` (30 rows), `results/exp008_c3_smoke.json`, `results/exp008_c3_smoke_log.txt`.
Not done: no `src/` change, no training, no EXP-009, no EXP-007 repair, no manifest change, no DEC-008, no research-state edit.
**The EXP-007 4-decimal vs ±0.01 Da caveat remains binding.** No EXP-007 decomposition number is used here.

## Verdict

- **Integrity: PASS.** All leakage checks are 0, the canary holds, and pool recall is 1.000.
- **Runtime stop criterion: BREACHED.** The median was 434 s/query against the 2 min limit; the total was 14,417 s.
  The run was allowed to finish because results are written only at the end. It must not be scaled in this form.
- n=30 (10 per tier) is a smoke test. Every number below is descriptive, with wide intervals.

## 1. EXP-008a pool census (all 425 monomer-adduct queries of the EXP-007 manifest)

| | value |
|---|---|
| target in 5 ppm structure pool | **425/425 (1.000)**; 10 ppm also 1.000; every adduct 1.000 |
| \|neutral-mass error\| p50 / p95 / p99 | 1.86 / 3.39 / 3.70 ppm |
| pool size (5 ppm) p10 / p50 / p90 | 9 / 47 / 170 structures (10 ppm median 72) |
| formula-oracle pool (5 ppm) median | 31 |
| chance MRR@25 in 5 ppm pool | 0.136 |

On the 30 smoke queries: recall 1.000, median pool 54 (T-low 36, T-mid 90, T-high 34).

## 2. Baseline B: existing direct spectral ranker

- The target received **0 direct spectra in 30/30 queries** (L6 canary = 0). The target sat in the bottom (−∞) tie group
  in 30/30. There were no unexpected non-zero results and no leakage failure.
- A median of 40 pool structures per query had direct evidence.
- MRR@25 = 0.038. This is entirely the expected RR from random tie-breaking inside the bottom tie group of
  spectrum-less candidates. It is not evidence for the target.

## 3. Baseline C: analogue propagation (R1)

| n | R@1 | R@5 | R@10 | R@25 | MRR@25 |
|---|---|---|---|---|---|
| 30 | 0.333 | 0.467 | 0.567 | 0.733 | **0.398** |

- Median candidate pool size was 54.
- Score distributions, as median [p25, p75]:
  - Top wide-search hit s_h: 0.973 [0.913, 0.986]. Spectrally close molecules exist in almost every query.
  - Max Tanimoto of those hits to the truth: 0.259 [0.218, 0.437].
  - Top-1 candidate score S: 0.359 [0.297, 0.456].
- Tanimoto of the top-1 candidate to the truth: median 0.254, against a pool mean of 0.171.
- Median evidence spectra scored per query: see the jsonl field `C_n_evidence_scored`.

## 4. Baseline A: chance

MRR@25 = 0.129 (the exact expectation within each pool). **Lift C − A = +0.269, bootstrap 95% CI [+0.142, +0.401].**

## 5. Decoy control

The query spectrum is replaced by a same-adduct spectrum of another molecule d from the same pool. d is removed from
evidence, and the pool and all other settings stay the same. The measure is the rank of the target t.

| | MRR@25 (target) | R@1 | lift vs chance (95% CI) |
|---|---|---|---|
| real query (C) | 0.398 | 0.333 | +0.269 [+0.142, +0.401] |
| decoy query | 0.082 | **0.000** | **−0.048 [−0.080, −0.018]** |

- Per query, C > decoy in 19, C < decoy in 6, and they tie in 5. Tanimoto(t, d) had median 0.13 (max 0.46), so the decoys
  are not structural near-copies of the target.
- **The analogue signal survives the control.** With the pool held fixed, a different spectrum removes all of the target's
  advantage: decoy R@1 = 0 and decoy MRR is below chance. So the lift comes from the target's own spectrum, not from
  pool structure. The decoy falls below chance because the decoy spectrum pulls d-like candidates up, which pushes t down.
- Secondary (recorded, not pre-registered as primary): under the decoy query, the decoy molecule d itself, which is also
  spectrum-less in that arm, reached MRR 0.776. This is a second, non-stratified C3-style observation in the same direction.

## 6. By structural-similarity tier (n=10 each)

| tier | pool median | R@1 | R@5 | R@10 | R@25 | MRR C | chance A | direct B | decoy | lift C−A (95% CI) | lift decoy−A (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <0.60 | 36 | 0.5 | 0.5 | 0.5 | 0.8 | 0.522 | 0.152 | 0.051 | 0.140 | +0.370 [+0.135, +0.609] | −0.012 [−0.068, +0.054] |
| 0.60–0.70 | 90 | 0.0 | 0.3 | 0.4 | 0.6 | 0.098 | 0.066 | 0.007 | 0.028 | +0.033 [−0.039, +0.105] | −0.038 [−0.070, −0.012] |
| ≥0.70 | 34 | 0.5 | 0.6 | 0.8 | 0.8 | 0.574 | 0.171 | 0.057 | 0.078 | +0.404 [+0.152, +0.646] | −0.093 [−0.153, −0.047] |

- The pattern is not monotone in the tier: the middle tier is weakest. The middle tier also has pools about 2.5× larger
  (median 90 vs 34–36), so tier and pool size are confounded at n=10. No tier-level conclusion is drawn.

## 7. Query-level outcomes (R1)

- Target absent from structure pool: **0/30**.
- Target present, rank >25: **8/30** (rids 291617, 1153823, 122956, 553571, 579238, 901199, 438093, 1003161).
- Target in top 25: **22/30**, of which rank 1: 10/30.
- The pre-registered failure labels are hierarchical and are applied before success:
  - F2 "neighbours unrelated" (max Tc(h,t) < 0.40): 21
  - SUCCESS_rank1: 8
  - F4: 1
  - Two rank-1 queries (774192, 930111) are labelled F2, because the target won with max Tc to hits of only 0.33 and 0.24.
    Their decoy RRs were 0.000 and 0.500 respectively. The 930111 pool has only 6 structures.
- Unexpected or leakage cases: **none.** L1 target evidence rows 0; L1 hits in target group 0; L7 decoy hits in excluded groups 0;
  L6 direct spectra for target 0.

## 8. What EXP-008 establishes (at smoke scale)

1. **Reachability (Q1): yes.** A legitimate neutral-mass structure pool (adduct table + 5 ppm) contains the unseen target for
   425/425 queries, with a median of 47 candidates.
2. **Current direct ranker (Q2): no, by construction.** It has zero evidence for an unseen target, so on true C3 the C1/V0
   pipeline can only score through tie-breaking.
3. **Complementary information (Q3): yes, provisionally.** Transferring validated Modified Cosine evidence from other molecules
   through structural similarity ranks the unseen target well above chance (+0.27 MRR, CI excludes 0). The signal disappears
   under a spectrum-swap control, so it is spectrum-specific rather than a pool artefact.

## 9. What remains unknown

- The effect size at scale: n=30, CIs ±0.13.
- Tier behaviour: confounded with pool size at n=10.
- How much depends on small pools. Several successes have pools ≤ 22.
- The runtime needed at scale, since the design breached its runtime rule.
- Robustness to the locked choices (K=20, ±150 Da, max aggregation, merged-CE representative), none of which were varied.
- Whether pseudo-C3 resembles hidden-test C3 in structure-DB coverage. The target is in the train structure universe by construction.
- The C3 share of the hidden test.
- EXP-007 magnitudes, which remain under the binding caveat.

## 10. Recommended next experiment

**EXP-008 full run on all 407 eligible queries, same locked definitions, after a performance-only change.**
The median of 434 s/query breached the stop rule. The cost has not been profiled. The likely candidates are the
~10⁵ Modified Cosine pairs per arm, with Spectrum objects rebuilt per pair; the change would be to cache them and
parallelise across queries. No change to eligibility,
scorer, K, window, tiers, or decoy rule. The output must reproduce the 30 smoke rows exactly as its regression gate.
Report lift against chance and against the decoy with pool size as a covariate. This requires separate authorization.

# EXP-007 — Genuine Pseudo-Class-2 Experiment: Full 600-Query Run Report

Date: 2026-09-24
Status: **COMPLETE — full run executed, all integrity checks pass, smoke replay exact.**
Approval: DEC-007 full-run authorization via user execution instruction (2026-09-23). Recorded in `research/decision_log.md`.

Scope: `results/exp007_c2_full_run_manifest.json`, `results/exp007_c2_query_results.jsonl`, `results/exp007_c2_full_run.json`, `results/exp007_c2_smoke_replay.json`, plus this report. No `src/` change, no C3, no training, no new decision record (DEC-008 was not created).

---

## 1. Population

- Population: **all 600 queries** whose `per_query` entries are locked in `results/exp007_audit_phase3.json` (n_query_sample=600, seed **20260922**). Manifest written to disk **before any scoring** and consumed by the run (no live resampling).
- Strata bounds (max_sim to a retained same-adduct sibling): S1 [0,0.50), S2 [0.50,0.70), S3 [0.70,0.85), S4 [0.85,0.95), S5 [0.95,1.0]; UR = `n_same_adduct_retained < 1` (unreachable).
- **Reconciliation (audit vs run)**: S1=33, S2=87, S3=83, S4=110, S5=177, UR=110, **total 600**. Recomputed `n_same_adduct_retained` and `max_sim` matched the audit for every rid (max drift ≤ 1e-9). UR recomputed count == manifest count (110).
- rid reconstruction re-verified every run against the two known triples (773739 / RCWXXMNGYWDMMO / [M+H]+, 733288 / QHALUOAFNBWZED / [2M+Na]+).
- Reachable population (≥1 retained same-adduct sibling, the C2(τ)/C1-matched(τ) population): **490**. UR (110) run in **C1 only** and reported as a diagnostic stratum, not mixed into the C2 headline (design §13).
- **Check 1 (population): PASS — 0 violations.**

## 2. Conditions executed (rows = 6480)

- **C1** (control): full train minus the query's whole metadata group `(inchikey14, adduct, precursor_mz 4dp, num_peaks)` — **600/600 queries**.
- **C2(τ)**, τ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}: C1 library **minus every retained same-adduct sibling with independently recomputed ModifiedCosine(query, sibling) ≥ τ**, run for every reachable query (490) at every τ → **2940 C2 runs**. k = number of genuinely removed rids; k==0 recorded as no-op diagnostic (C2 == C1); k≥1 is the genuine intervention. No artificial intervention forcing (real-intervention rule).
- **C1-matched(τ)**: C1 library minus `k` randomly selected same-window **guaranteed-wrong-molecule** spectra, `k =` the C2(τ) removal count (EXP-003/DEC-004 pool-size control) — 2940 runs on the 490 reachable queries.
- Row accounting: 600 (C1) + 2940 (C2) + 2940 (C1-matched) = **6480 rows**. UR queries carry only C1 rows.

## 3. Intervention proof (full scale)

- 2940 C2 runs: **1974 genuine interventions (k≥1), 966 no-ops (k=0)**.
- k distribution over genuine runs: k=1→914, k=2→803, k=3→257. Median k per τ: 0.50→2.0, 0.60→1.0, 0.70→1.0, 0.80→1.0, 0.90→0.0, 0.95→0.0.
- Genuine-intervention count per τ: 0.50→457, 0.60→418, 0.70→370, 0.80→315, 0.90→237, 0.95→177 (of 490 reachable).
- No-op pattern matches the design: S1 (all retained evidence < 0.50) never intervenes at any τ (k=0 for all 33 S1 queries, all τ); S5 is the last to lose removals as τ rises.
- C2 is a **real experimental manipulation**: 1974 executed removals across ≥1 evidence spectra, distributed over all six τ levels. (Fixes the EXP-004 vacuousness, DEC-006.)

## 4. Metrics (primary/secondary/diagnostics)

Shared populations: C1 on 600; C2(τ)/C1-matched(τ) on the 490 reachable queries (like-for-like denominators; no-ops included and flagged).

### 4a. C1 baseline (n=600)

| Stratum | n | candRec | R@1 | R@5 | R@10 | R@25 | MRR@25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 33 | 1.000 | 0.242 | 0.515 | 0.606 | 0.727 | 0.340 |
| S2 | 87 | 1.000 | 0.241 | 0.586 | 0.805 | 0.908 | 0.411 |
| S3 | 83 | 1.000 | 0.482 | 0.916 | 0.964 | 0.988 | 0.659 |
| S4 | 110 | 1.000 | 0.718 | 0.964 | 0.991 | 1.000 | 0.821 |
| S5 | 177 | 1.000 | 0.638 | 0.859 | 0.910 | 0.977 | 0.741 |
| UR | 110 | 0.936 | 0.500 | 0.782 | 0.827 | 0.873 | 0.612 |
| **All** | **600** | **0.988** | **0.527** | **0.813** | **0.885** | **0.940** | **0.651** [0.620,0.682] |

Monotone difficulty by novelty is confirmed at full scale: S4/S5 (loose novelty) strong; S1/S2 (deep novelty) weak even with full evidence. UR underperforms C1 overall and is C1-only (candidate ceiling 0.936).

### 4b. C2(τ) dose–response (n=490 per τ)

| τ | candRec | R@1 | R@5 | R@10 | R@25 | MRR@25 | CI95 | median cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 (reachable, n=490) | 1.000 | 0.533 | 0.820 | 0.882 | 0.955 | 0.660 | — | 667 |
| 0.50 | 0.651 | 0.151 | 0.304 | 0.392 | 0.504 | 0.221 | [0.189,0.251] | 746.5 |
| 0.60 | 0.735 | 0.165 | 0.361 | 0.465 | 0.586 | 0.253 | [0.222,0.283] | — |
| 0.70 | 0.820 | 0.196 | 0.447 | 0.571 | 0.692 | 0.307 | [0.276,0.339] | — |
| 0.80 | 0.857 | 0.243 | 0.539 | 0.659 | 0.753 | 0.370 | [0.333,0.403] | — |
| 0.90 | 0.894 | 0.329 | 0.622 | 0.724 | 0.810 | 0.456 | [0.417,0.493] | — |
| 0.95 | 0.927 | 0.388 | 0.692 | 0.784 | 0.853 | 0.519 | [0.480,0.556] | — |

The reachable-restricted C1 (n=490, MRR@25 0.660 — slightly higher than the all-600 C1 of 0.651 because UR rows are excluded) is the like-for-like C1 baseline used for every delta below.

- **STRICTLY monotone full-scale dose–response**: MRR@25 0.221 → 0.519 as τ increases (novelty loosens), recovering toward C1's 0.660. Candidate-generation recall follows the same shape (0.651 → 0.927).
- The deepest novelty (τ=0.50) removes the true molecule from the candidate pool for 171/490 queries (candRec 0.651) — the C2 candidate-gen ceiling, quantified separately from ranking loss in §6.

### 4c. C1-matched(τ) control (n=490 per τ)

| τ | candRec | R@1 | R@5 | R@25 | MRR@25 |
|---|---:|---:|---:|---:|---:|
| 0.50 | 1.000 | 0.533 | 0.822 | 0.955 | 0.660 [0.625,0.696] |
| 0.60 | 1.000 | 0.535 | 0.820 | 0.955 | 0.661 [0.626,0.698] |
| 0.70 | 1.000 | 0.533 | 0.820 | 0.955 | 0.660 [0.626,0.696] |
| 0.80 | 1.000 | 0.533 | 0.820 | 0.955 | 0.660 [0.625,0.696] |
| 0.90 | 1.000 | 0.533 | 0.820 | 0.955 | 0.660 [0.625,0.696] |
| 0.95 | 1.000 | 0.535 | 0.820 | 0.955 | 0.661 [0.626,0.697] |

- **Control is flat at every τ** despite candidate-set changes (removing 1–3 wrong-molecule spectra each) in 457/490 runs @0.50 and 177/490 @0.95. True-molecule rank changed in only **12/2940** control runs total (per τ: 3, 2, 3, 1, 0, 3); candidate-gen recall stays 1.000 everywhere.
- Matched-control MRR (0.660–0.661) is statistically indistinguishable from the reachable C1 baseline (0.660; CIs overlap), far above C2 at every τ. **The C2 degradation is specific to removing the true molecule's similar evidence**, not a pool-size artifact (design §14 evidence-that-a-C2-bottleneck-exists, condition 2, confirmed at full scale).

## 5. Deltas (C2−C1 and control−C1, like-for-like on the 490 reachable; C1 restricted to reachable)

| τ | Δ candRec (C2) | ΔR@1 | ΔR@5 | ΔR@25 | ΔMRR@25 | Δ candRec (ctrl) | ΔMRR@25 (ctrl) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | −0.349 | −0.382 | −0.516 | −0.451 | −0.438 | 0.000 | +0.000 |
| 0.60 | −0.265 | −0.367 | −0.459 | −0.369 | −0.406 | 0.000 | +0.001 |
| 0.70 | −0.180 | −0.337 | −0.373 | −0.263 | −0.352 | 0.000 | +0.000 |
| 0.80 | −0.143 | −0.290 | −0.282 | −0.202 | −0.289 | 0.000 | +0.000 |
| 0.90 | −0.106 | −0.204 | −0.198 | −0.145 | −0.203 | 0.000 | +0.000 |
| 0.95 | −0.073 | −0.145 | −0.129 | −0.102 | −0.141 | 0.000 | +0.001 |

Changed-only (k≥1) deltas at τ=0.50: Δ candRec −0.374, ΔR@1 −0.409, ΔR@5 −0.554, ΔMRR@25 −0.470 (control −0.000 / +0.000); at τ=0.95: Δ candRec −0.203, ΔR@1 −0.401, ΔR@5 −0.356, ΔMRR@25 −0.390 (control −0.000 / +0.003). No-op-only deltas are exactly 0 by construction (C2 == C1). Deltas are monotone in τ; control deltas are uniformly ~0 — the pool-size mechanism contributes nothing measurable.

## 6. Failure decomposition (candidate-gen loss vs ranking loss, per τ)

| τ | A: cand-loss (in C1, lost from C2) | B: present, rank >25 | C: present, rank 1–25 | D: no-op (k=0) | n C2 runs | n C1 rank1 before |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 171 | 72 | 247 | 33 | 490 | 261 |
| 0.60 | 130 | 73 | 287 | 72 | 490 | 261 |
| 0.70 | 88 | 63 | 339 | 120 | 490 | 261 |
| 0.80 | 70 | 51 | 369 | 175 | 490 | 261 |
| 0.90 | 52 | 41 | 397 | 253 | 490 | 261 |
| 0.95 | 36 | 36 | 418 | 313 | 490 | 261 |

- Candidate-loss (A) dominates at deep novelty (171 @0.50) and is nearly resolved by 0.95 — this is the **C2 candidate-generation ceiling**, not a ranking failure.
- Ranking loss (B, the "close→lost" category) is smaller and stays roughly constant across τ (72→36), i.e. where the true molecule survives the pool the rank mostly stays ≤25.

## 7. Transition matrix (per-query, per-τ, k≥1 only)

| Case | Reading | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | rank1→rank1 (harmless) | 66 | 64 | 67 | 66 | 55 | 42 |
| 2 | rank1→≤25,>1 (degrades) | 61 | 74 | 86 | 80 | 56 | 40 |
| 3 | ≤25→>25 (close→lost) | 197 | 154 | 106 | 82 | 60 | 42 |
| 4 | >25→any (pre-existing) | 133 | 126 | 111 | 87 | 66 | 53 |
| 5 | not-in-candidates | captured by candRec in §4b (A in §6); not a transition row | | | | | |
| k≥1 runs | | 457 | 418 | 370 | 315 | 237 | 177 |

- P(regress) = (case 2 + case 3) / k≥1: 0.565 @0.50 → 0.545 → 0.519 → 0.514 → 0.489 → 0.463 @0.95, strictly monotone decline with looser novelty.
- Case 3 (close→lost), the most load-bearing ranking failure, is 197 @0.50 and declines to 42 @0.95; case 4 (already outside 25 pre-intervention) is substantial at every τ — pre-existing ranking bottlenecks dominate at high τ.

## 8. Control power analysis

- Per τ: k distribution over reachable runs (e.g. 0.50: k=1→175, k=2→198, k=3→84, k=0→33; 0.95: k=1→101, k=2→58, k=3→18, k=0→313).
- k vs rank change (control, τ=0.50): k=1→1/175, k=2→1/198, k=3→1/84 (rank changes vs C1). At τ=0.95: k=1→2/101, k=2→1/58, k=3→0/18. **Total: 12/2940 control rank changes** (per τ: 3, 2, 3, 1, 0, 3) regardless of k — pool-size removal does not move the true molecule's rank.
- Best-evidence-removed analysis (C2): per τ the number of genuine C2 runs where the removed set contained the max-scoring true-molecule spectrum: 0.50→416, 0.60→380, 0.70→336, 0.80→286, 0.90→213, 0.95→157; "no effect because another true spectrum remained max": 0.50→41, 0.60→38, 0.70→34, 0.80→29, 0.90→24, 0.95→20.
- C1 true-rank histogram (reachable): 261 queries ranked 1 in C1; the bulk of the reachable population is already well-ranked before any intervention.

## 9. Query-level per-query output fields (jsonl, 6480 rows)

Per row (minimal audit set): `rid, inchikey14, adduct, precursor_mz, num_peaks, stratum, max_sim, n_retained_same_adduct, condition, tau, k_removed, removed_rids, filler_rids, no_op, n_candidates, candidate_gen_hit, c1_candidate_gen_hit, rank, c1_rank, reciprocal_rank, transition_case, true_score, best_wrong_score, best_wrong_molecule_inchikey14, molecule_above_true_rank, top_k, true_in_candidates_c1, true_in_candidates_c2, candidate_loss, rank_gt_25, rank_1_to_25, best_true_evidence_removed, score_unchanged`. C1 rows additionally record the UR/stratum diagnostic; every reachable row carries all three per-tau ranks and the wrong-molecule-above-true field. Every row preserves the full top-25 ranked list with scores for audit reconstruction.

## 10. Leakage controls (design §8)

- **Check 3 (leakage): PASS — 0 violations.** Query rid absent from every C1 pool; query's whole metadata group absent from every C1 pool; 0 filler spectra in C1-matched carried the true molecule's inchikey14 (every filler verified guaranteed-wrong at selection; the 12/2940 control rank invariance in §8 is an independent confirmation).
- Scorer inputs carry only `ms2_mzs`, `ms2_normalized_intensities`, `precursor_mz` (+ identical metadata fields as EXP-001); inchikey14 used exclusively for exclusion-set building and ground-truth evaluation, never fed to scoring. `passed_checks` = all six buckets; `failed_checks` = {}.

## 11. Checks (design §12, full-scale record)

| # | Check | Result |
|---|---|---|
| 1 | Population: 600/600 audit rids, strata reconciled to S1=33/S2=87/S3=83/S4=110/S5=177/UR=110; rid reconstruction re-verified | PASS (0) |
| 2 | Removal integrity: every removed rid is a retained same-adduct sibling with sim ≥ τ (independent recompute) | PASS (0) |
| 3 | Leakage: query inchikey absent from scoring input; C1-matched fillers wrong-molecule | PASS (0) |
| 4 | C1/C2 isolation: candidate sets differ exactly by removed rids | PASS (0) |
| 5 | Candidate-gen sanity: cand_count(C2) == cand_count(C1) − k exactly | PASS (0) |
| 6 | Scoring sanity: independent ModifiedCosine recompute on 3 surviving candidates per genuine C2(τ); retained-sibling pool score == recomputed sim | PASS (0) |
| 7 | Smoke replay: 30 shared rids × 13 condition rows = 390 rows compared field-by-field against `results/exp007_smoke_query_results.jsonl` | PASS (390/390 exact, 0 mismatches) |

`results/exp007_c2_full_run.json`: all check buckets empty, `failed_checks: {}`, `passed_checks`: all six.

## 12. Smoke replay (reproducibility gate)

- Full run reused the validated smoke-test machinery by import (identical functions in `research/scripts/exp007_full_run.py`).
- `research/scripts/exp007_smoke_replay.py` compared the full run's rows for the **30 smoke rids** against `results/exp007_smoke_query_results.jsonl` on: rid, tau, stratum, max_sim, n_retained, condition, k_removed, no_op, n_candidates, candidate_gen_hit, c1_candidate_gen_hit, rank, c1_rank, transition_case, true_score (where both present), and full top_k molecule/score lists.
- **Result: 390/390 rows exact, 0 mismatches.** `material_disagreement: false`.

## 13. Reproducibility / determinism

- The full run was executed twice end-to-end (`exp007_full_run.py`): the second run overwrote the first's `results/exp007_c2_query_results.jsonl`; **sha256 identical** (`F1E5…2982`, 6,480 lines, 40,226,204 bytes) — byte-level result equality across deterministic reruns (threads=1, fixed seed 20260922, fixed seeds for filler selection and bootstrap). No full rerun was unnecessary; the replay (§12) plus this run-pair equality constitute the requested reproducibility evidence.
- `results/exp007_c2_full_run.json` scientific content is identical across the two runs (only `runtime_seconds` and manifest `timestamp`/`environment` metadata vary — the manifest is intentionally time-stamped at write).
- Determinism sources: `PRAGMA threads=1`; `ROW_NUMBER() OVER () - 1` rid reconstruction; fixed `random.Random(SEED…` for fillers/bootstrap; `ModifiedCosineGreedy` deterministic.

## 14. Runtime / feasibility

- Full run wall time ≈ **1185 s** (~60 s parquet→rid materialization under threads=1 + ~1100 s scoring/audit recompute/bootstrap, 6480 rows). C1 candidate pool median 667, min 7, max 3117; C2:0.50 median 746.5, max 3116; no zero-candidate pools anywhere.

---

## Verdict

**COMPLETE.** The full 600-query EXP-007 C2 run executed all 6480 condition rows (600 C1 + 2940 C2 + 2940 control) with every design §12 check passing empty, the smoke replay exact (390/390), and the per-query jsonl byte-identical across two independent deterministic runs. The full-scale dose–response is strictly monotone (MRR@25 0.651 → 0.221 → 0.519 over the τ grid), matched-control MRR is flat at ~0.66 across τ, and the C2 degradation is attributable to real evidence removal (1974 executed removals) rather than pool size. Detailed scientific interpretation of these quantities is **not** part of this execution report and has not been performed.

## Explicitly NOT done

- No C3 design/run, no training, no `src/` change, no new decision record (DEC-008 not created), no modification of `research/CASMI_RESEARCH_STATE.md` or any EXP-001/EXP-002 code based on these results; `experiments.md` untouched.

## Files

- `results/exp007_c2_full_run_manifest.json` — 600-rid manifest + population reconciliation + locked config, written before scoring.
- `results/exp007_c2_query_results.jsonl` — 6480 per-query condition rows (all §9 fields).
- `results/exp007_c2_full_run.json` — checks, metrics, deltas, per-stratum, intervention proof, failure decomposition, control power, candidate-pool stats (`failed_checks: {}`).
- `results/exp007_c2_smoke_replay.json` — 390-row smoke replay, 0 mismatches.
- This report.
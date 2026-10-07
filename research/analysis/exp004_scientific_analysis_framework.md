# EXP-004 Scientific Analysis Framework (Pre-registered Interpretation Plan)

Status: **PREPARED IN ADVANCE OF EXP-004.** Independent, read-only. No smoke
test run, no experiment launched, no `src/` or experiment-code modification, no
interference with Claude Code's execution of EXP-004. This document fixes the
**interpretation contract**: what conclusions EXP-004's eventual pseudo-Class-2
smoke/full results do and do not license, so that results are read against a
pre-committed standard rather than post-hoc narratives.

Sources read for this framework (current on-disk state):
`research/CASMI_RESEARCH_STATE.md`,
`research/05_class1_to_class2_design.md`,
`research/06_matched_c1_control.md`, `research/decision_log.md` (DEC-001…
DEC-005), `research/experiments.md` (EXP-001/002/005),
`results/exp001_checkpoint.json`, `results/exp002_checkpoint.json`,
`results/exp003_inspection.json`, `results/exp004_scenario_audit_v2.json`,
`results/exp005_matched_c1_n400.json`. Cross-checked against the independent
read-only verification in `research/04_exp004_independent_audit.md` and
`research/analysis/exp004_independent_verify.py` (which agrees rid-for-rid with
`exp004_scenario_audit_v2.json`).

---

## 1. Research state (fixed points this framework relies on)

- **EXP-001 Mode B (Variant A, all-train, ModifiedCosineGreedy) is the
  authoritative Class-1 baseline** — and will be EXP-004's *unchanged* candidate
  generation and scorer (design §11/§13, DEC-005).
- **The "108 cross-adduct-only" claim is dead** — corrected to **12**, verified
  two independent ways (source-data recomputation; exact set-equality with
  EXP-001's 12 `candidate_gen_hit=False` queries).
- **The 0.90 novelty threshold is now self-justified** (≈99.25th percentile of
  this population's own Modified-Cosine similarity), no longer mis-attributed to
  EXP-001 (whose validated threshold was 0.95 direct Cosine on same-metadata-group
  siblings).
- **The CE-based Scenario A/B split (15/21) is abandoned** — merged into one
  "same-adduct evidence retained" category; CE similarity (3/388 exact matches,
  0.8%) is descriptive only.
- **EXP-005 (matched C1 control) is complete**: row-exclusion vs whole-group
  exclusion are bit-identical on every headline metric (only 8/400 queries have a
  group sibling to exclude). The specific leakage mechanism (a near-duplicate
  secretly remaining searchable) is ruled out as a confound.
- **EXP-004's C2 smoke test has NOT been run** and is not part of this work.

**Verified evidence categories for the 400-query EXP-001 Mode B population**
(same-adduct = same `inchikey14` + same adduct among retained spectra; from
`exp004_scenario_audit_v2.json`, `exp004_independent_verify_out.json`):

| Evidence category | n | Meaning for EXP-004 |
|---|---|---|
| Same-adduct retained | 388 | The molecule has at least one retained spectrum at the *query's* adduct |
| Same-adduct-only | 149 | All retained evidence is at the query's adduct |
| Same + cross | 239 | Retained evidence at mixture of query-adduct and other adducts |
| Cross-only | 12 | **No retained same-adduct spectrum at all** → Variant-A unreachable (structural, all 12 are EXP-001's candidate-absent queries) |
| Near-duplicates excluded by 0.90 filter | 3 | Not in EXP-004's 397-query population |

---

## 2. Baselines (established, verified against artifacts)

| Metric | EXP-001 Variant A (all-train) | EXP-002 Variant B | EXP-005 matched C1 control (L row / W group) |
|---|---|---|---|
| candidate recall | 0.970 | 1.000 | 0.970 / 0.970 |
| Recall@1 | 0.535 | 0.490 | 0.5350 / 0.5350 |
| Recall@5 | 0.805 | 0.780 | 0.8050 / 0.8050 |
| Recall@10 | 0.8725 | 0.855 | 0.8725 / 0.8725 |
| Recall@25 | 0.915 | 0.920 | 0.9150 / 0.9150 |
| MRR@25 | **0.6510** | 0.6178 | **0.6510 / 0.6510** |

Margins (correct `best_true_score` − `best_wrong_score`), EXP-005 condition W,
full n=400 (persisted per-query in `exp005_matched_c1_n400.json`):

| C1 control outcome bucket | n | median margin | mean margin | % negative |
|---|---|---|---|---|
| Rank 1 (success) | 214 | **+0.127** | +0.187 | 0 |
| Rank 2–25 (found, not top) | 152 | **−0.072** | −0.129 | 100 |
| Found, ranked below 25 | 22 | **−0.391** | −0.331 | 100 |
| Candidate-absent (the 12) | 12 | n/a (no margin) | — | — |

EXP-003 (92 outcome-selected queries; **failure-selected, not a population
sample**): 90/92 correct at same adduct; displacer is cross-adduct-vs-query in
54/92, newly admitted by Variant B in 57/92 (overlapping, non-identical sets);
same-adduct wrong candidates score as high as cross-adduct wrong candidates
(wrong same-adduct median 0.929, wrong cross-adduct median 0.939).

---

## 3. What EXP-004 should test (the exact question)

**Primary scientific question:**

> Holding the query's entire metadata group out of the library — while its
> molecule remains represented elsewhere (the EXP-001 Mode-B construction, plus
> the 0.90-novelty filter and the corrected evidence categories) — how much does
> end-to-end pseudo-Class-2 performance change relative to the matched Class-1
> control, and **why**?

EXP-004 must decompose any observed delta into two independent mechanisms:

1. **Candidate-generation failure** — the correct molecule is absent from the
   ranked pool. Under the pre-committed Variant-A rule this is *predicted* to be
   exactly the 12 cross-only rids (structural; adduct-shifted precursor mass ≫
   the ±0.01 Da window). Anything beyond that 12 is a new, unexplained finding.
2. **Ranking / discriminability failure** — the correct molecule is in the pool
   but outranked by a wrong molecule. This is the mechanism suspected to dominate
   (EXP-002's Variant B: pool grows, MRR falls; EXP-001's own failure split of
   ~2.8–3.5% candidate-absent vs ~5.5% ranking).

Pre-registered expectation to test (not to assume): under Variant A in EXP-004,
candidate recall should reproduce ≈0.97 (the same 12 structural absents) and any
MRR/R@1 drop would therefore be classified **by presumption as a ranking/discrim
inability phenomenon**, to be corroborated by the margin and top-wrong analyses
(§6–§7) rather than asserted.

**Statistics to report per query in both control and test (paired):**
`candidate_gen_hit`, `rank`, `n_candidates`, `best_true_score`, `best_wrong_score`,
`margin`, **`top_wrong_adduct`**, **`top_wrong_origin`** (was this wrong molecule a
Variant-A candidate / newly admitted under the compared rule), plus
evidence-category (same-only / same+cross / cross-only) as a stratification key.

---

## 4. The important comparisons

**C1 matched control vs pseudo-C2 — paired, same rids (397 retained).**
`exp005_matched_c1_n400.json` condition W *is* the C1 control on the exact same
query rids; EXP-004 is the pseudo-C2 treatment.

Headline comparison metrics (all must be reported, in this order of importance):

1. **MRR@25** — primary CASMI metric (rank-weighted; R@25 alone is misleading
   because rank 25 ≈ 0.04).
2. **Recall@1 / 5 / 10 / 25** — where the damage lands.
3. **Candidate recall + candidate count (median, distribution; L vs W vs EXP-004
   pool sizes)** — separates "can't find" from "won't rank".
4. **Correct-vs-wrong score margin** (§7) and its distribution shift vs control.
5. **Top-wrong adduct** (cross-adduct vs query-adduct vs correct-adduct).
6. **Top-wrong candidate origin** (already-in-Variant-A vs newly-admitted) —
   directly tests the EXP-003-motivated hypothesis that *newly admitted* decoys,
   not merely cross-adduct ones, drive displacement.
7. Stratified versions of everything: same-adduct-only (149) / same+cross (239)
   / cross-only (12, candidate-gen analysis only).

**What the matched nature licenses:** difference-in-differences per query
(EXP-004 rank − control rank), a paired bootstrap CI on MRR@25, and "net wins"
(count of queries improved minus query worsened), avoiding the trap of reading
only aggregate-delta.

---

## 5. No fake collision-energy story (corrected stratification)

- The A(15)/B(21)/C(364) split is **invalid** and must not appear in EXP-004
  interpretation.
- Evidence categories are exclusively the **adduct-based** ones above:
  same-adduct-only (149), same+cross (239), cross-only (12).
- The cross-only 12 are **Variant-A-unreachable by construction** — they appear
  as candidate-generation findings, not as a ranking/CE result.
- Collision energy is recorded **descriptively** (e.g. share of retained
  same-adduct spectra sharing a CE value; the 0.8% exact-CE-match rate) and must
  not be used as a causal explanation for any observed degradation. Observed
  score differences are facts; attributing them to "different collision
  energies cause different fragmentation" is an **unproven hypothesis** in this
  data and is explicitly off-limits as a conclusion (state file §10/§16).

---

## 6. Interpretation logic (pre-committed table)

Each row is: **Observation → Possible interpretations → What it DOES establish →
What it DOES NOT establish.** EXP-004 results license only the "DOES establish"
cells.

| Observation | Possible interpretation | Establishes | Does NOT establish |
|---|---|---|---|
| **A. Candidate recall stays ≈0.97, MRR drops vs 0.6510** | Ranking degrades on genuinely-different evidence while retrieval still works; larger/looser evidence yields overconfident wrong molecules | The C2 degradation (if any) is overwhelmingly a ranking/discriminability phenomenon in this pipeline; the 12 structural gaps are the entire candgen delta | That a better ranker fixes it; that ranking is *the* general bottleneck for C3; that the max-modified-cosine scorer is "broken" per se |
| **B. Candidate recall drops substantially (<0.95)** | The novelty filter or evidence pool makes more molecules unreachable than the 12 | A structural C2 ceiling below EXP-001's measured 3% gap — must enumerate the new absent queries rids and re-run the reachability logic | That ranking is fine; causality for why new cases are absent without per-case inspection |
| **C. R@25 stays high, R@1/MRR falls** | Correct molecules present but outranked — "found, not top" pattern (C1: 152/400 near-miss at median margin −0.072) magnified | C2 difficulty concentrates on ranking the top-1; degradation is margin/threshold-shaped, so calibration/tie-break headroom is quantifiable | That top-25=25 is a usable result (MRR@25 reward at rank ≥2 is small); that candidate generation is healthy merely because R@25 is high |
| **D. Same-adduct-only (149) much worse than same+cross (239)** | Same-adduct retained evidence is genuinely dissimilar (design §4: even same-adduct median sim ≈3–5%); cross-adduct evidence sometimes *rescues* via shift matching | Adduct alignment of the retained evidence is a real difficulty modulator; evidence category is a meaningful stratification axis | That collision energy caused it; that any specific adduct is hard in general (the 12 cross-only are a fixed special set) |
| **E. Same-adduct-only performs similarly to same+cross** | Adduct mismatch is not the dominant C2 difficulty; within-adduct spectral differences and pool effects dominate | Difficulty is spread across all evidence types; C2 is not primarily a "cross-adduct" story | That spectral novelty is uniform; that the adduct axis can be ignored for C3 |
| **F. Cross-adduct candidates frequently top-wrong** | Cross-adduct scoring is miscalibrated (overconfident), or the strongest cross-adduct wrongs are chemically similar molecules | Score calibration by adduct is a candidate defect locus; margin analysis will show near-miss vs landslide shape | That removing cross-adduct candidates helps (EXP-002 Variant B composition change — needs its own ablation, not conclusion); that "cross-adduct" is a single failure class |
| **G. Correct score stays high, wrong scores rise** | Failure is relative (discriminability), not absolute: correct evidence fine, decoys inflate | The problem is margin/relative ranking, not coverage; an absolute-score threshold would not fix it | That a threshold/tie-break of any fixed value is optimal; that correct evidence is "high quality" absent paired scoring |
| **H. Correct score itself collapses (best_true_score drops a lot vs the same query in control)** | The self-measurement was the dominant evidence; remaining spectra are genuinely weak/different | The "hidden evidence" axis is real; Modified-Cosine similarity has a C2-relevant scale problem; spectral novelty can be severe | That the molecule is absent (check candgen hit); that embeddings/fingerprints would do better (untested); that "novel" = "harder for all comparators" |
| **I. Margin is the main degradation (margins shift strongly negative vs control)** | Near-miss rank degradation is systemic (C1: median −0.072 in rank2–25); a calibration/tie-break change could flip a meaningful share | Quantifies headroom for a ranking-side intervention and bounds how much is structural | That the headroom is reachable without a controlled ranking experiment (DEC-004 precedent); that margin alone locates the fix |
| **J. Candidate count changes dramatically (explode/shrink vs EXP-001/control)** | Pool-statistics change via filter/evidence differences; EXP-002 showed bigger pools hurt the current ranker (84 worsened at 2.74× growth) | Reports whether counts differ at all; if similar counts, composition (adduct mix) — not size — is the operative change | That a count change *causes* performance change (confound: composition); that stable counts imply stable performance |

**Global guardrail:** no cell in "Establishes" may be upgraded to a causal claim
("X *causes* Y", "removing Z *fixes* W") unless EXP-004 itself contained the
controlled intervention to test it. Absent that, findings license **hypotheses
for the next designed experiment**, not conclusions (see §6 of the interpretation
discipline in `CASMI_RESEARCH_STATE.md §1`, DEC-004).

---

## 7. Margin analysis (definitions and planned analyses)

**Definition:** `margin = best_true_score − best_wrong_score` (Modified-Cosine
scores, molecule-level max aggregation), per query.

- **Positive margin** → correct molecule beats the strongest wrong molecule
  (EXP-004 "wins" this query).
- **Negative margin** → a wrong molecule is the top match.
- `margin > 0` is *necessary but not sufficient* for rank 1 (ties/stochastic
  ordering); `margin < 0` guarantees rank ≥ 2 (modulo composition).

Baseline already established (EXP-005, condition W, n=400, §2 table): wins are
comfortable (median +0.127), rank 2–25 losses are close (median −0.072, 152/400),
sub-25 losses are decisive (median −0.391, 22/400).

**Analyses to run when top-25 (molecule, score) lists are persisted by EXP-004
(they are NOT persisted by EXP-001/002 — EXP-005 persisted only the top wrong
score, not its identity):**

1. Paired margin shift: EXP-004 margin vs control margin for the same rid
   (distribution of `Δmargin`); decompose into `Δbest_true` and `Δbest_wrong`.
2. Margin × evidence category (same-only / same+cross) — is the loss shared or
   concentrated?
3. Margin × top-wrong adduct and × top-wrong origin (already-in-Variant-A vs
   newly-admitted) — tests the EXP-003 displacement hypothesis on the full
   population instead of the 92 selected queries.
4. Failure classification with explicit thresholds:
   - **win:** margin > 0 (model-bucket: correct wins)
   - **near-miss:** −0.1 ≤ margin ≤ 0 (calibration/tie-break sensitive)
   - **decisive loss:** margin < −0.1 to −0.4 → discrete
   - **landslide:** margin < −0.4 (different molecule fundamentally outranks)
   - **unreachable:** candgen miss (the 12)
   Compute the share of the C1→C2 delta that is *near-miss* (recoverable in
   principle by a better ranker) vs *landslide* (requires better evidence or a
   different comparator).

---

## 8. C1 → C2 → C3 research bridge

Regime definitions:
- **C1:** known molecule + known/retrievable spectral evidence (measured:
  MRR@25 ≈ 0.651 Mode B).
- **C2:** known molecule + unseen spectrum (EXP-004 = controlled pseudo-C2).
- **C3:** unseen molecule + unseen spectrum (pseudo-C3 to be designed **only
  after** C1+C2 evidence; explicit prohibition in state §4/§12).

EXP-004's failure modes, in order of importance for C3 design (do NOT design the
C3 model yet — this is which questions C2 evidence should answer):

| C2 failure mode (EXP-004 observation filter) | Why it matters for C3 | C3-relevant question it should answer |
|---|---|---|
| **Candidate-generation gap** (B, and the 12 cross-only) | In C3 there is *no* same-molecule evidence at all, so candidate generation is the entire game, not a 3% edge. The 12 cross-only rids are the preview: precursor-window-only retrieval fails when the "self-evidence" absent | Which candidate-generation strategies (adduct-aware neutral mass, formula-window, library-agnostic broadened windows) have principled recall ceilings worth building |
| **Spectral-similarity calibration** (A, C, G, I — near-miss margins) | The ranker is max-Modified-Cosine; if it systematically over-scores decoys on distant evidence, C3's *only* comparator signal is similarly mis-scaled | Does similarity need scale/calibration fixes, embedding-based reranking, or a fundamentally different evidence notion for cross-condition comparison |
| **Molecular-family confusion** (F — top-wrong identity) | If top-wrong molecules are structural analogues, analogue/family-based C3 retrieval has signal; if they are random decoys, C3 needs discriminative (formula/fragment/fingerprint) features | What is the chemical relationship between query and top-wrong (nearest spuriously-scored molecule) — measured, per failure class |
| **Formula-constrained retrieval** (B, H — unreachable/weak cases) | EXP-001/002/004 never used formula; C3-hard tiers (target absent, neighbourhood restricted) will force formula/neutral-mass constraints | Does restricting to the true formula class recover candidates/rankings the precursor window misses (the 12 cross-only + beyond) |
| **Cross-adduct robustness** (D, F) | Test spectra plausibly arrive at adducts/multimers unseen together; multi-adduct/neutral-loss evidence is exactly the C3 data-wrangling problem at scale | Does Modified-Cosine shift-matching plus multi-adduct evidence scale, or is a learned adduct-robust representation needed |
| **Fingerprint/embedding ranking** (H, I — score collapse) | If correct-score collapse is systematic, a learned representation is motivated as a *replacement* ranker on top of generated candidates | What is the residual MRR after a scoring fixedpoint — the candidate pool's true ceiling before any learned component is worth its complexity |
| **De novo structure inference** (C, H — only if the above all fail) | If even with correct candidates in the pool ranking fails decisively, C3's hardest tier may require structure generation, not retrieval | Only pursued once the retrieval/ranking ceiling from EXP-004 + a designed oracle-candidate experiment (state §14) is measured |

Rule: EXP-004 results **motivate** the bullet above them in this table; they do
**not** by themselves justify implementing any C3 model. The C3 design must come
after a written C2 evidence read-out against §6's table.

---

## 9. Deliverables this framework pre-registers

On EXP-004 smoke/full results:

1. Cited paired table: MRR@25, R@1/5/10/25, candidate recall, candidate-count
   median, vs the EXP-005 control (§2), stratified by evidence category (§5).
2. Decomposition: candidate-generation vs ranking, cross-checked by the 12-rid
   ledger (candgen) and by margin analysis (ranking).
3. Margin read-out per §7 and top-wrong adduct/origin read-out per §4.6.
4. A §6 interpretation, cell-by-cell, marking which observations occurred and
   the exact cells they license.
5. No causal claim unsupported by a within-experiment control.
6. A C1→C2→C3 handoff note (§8) listing which C3 design questions EXP-004's
   evidence now answers and which remain open.

Failure of EXP-004's smoke test to pass its own stated verification checks
(query exclusion, novelty-filter exactness, paired sanity) is a **verification
failure**, not a scientific result, and this framework's interpretation logic
does not apply to it.

---

## WAITING FOR EXP-004 SMOKE RESULTS
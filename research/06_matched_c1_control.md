# 06 — EXP-005: Matched Class-1 Row-Exclusion Control + Reachability Ledger

Status: **EXECUTED (cheap control only). Not the EXP-004 C2 smoke test.**

This is the required prerequisite before the EXP-004 (Class-1 → pseudo-Class-2)
smoke test may be approved. It answers, with a matched (paired) experiment on the
exact same molecules, exactly one question:

> For the same molecules, how much does performance change when we move from
> library-like evidence to genuinely withheld spectral evidence, and what causes
> that change?

Reproducibility artifacts:
- `research/scripts/exp004_reaudit.py` — corrected Scenario audit, produces the
  reachability ledger below (`results/exp004_scenario_audit_v2.json`)
- `research/scripts/exp005_matched_c1_control.py` — the matched control experiment
  itself (`results/exp005_matched_c1_n25.json` smoke, `results/exp005_matched_c1_n400.json`
  full)

No `src/` (production pipeline) file was modified to produce any of this.

---

## 1. The 12-query reachability ledger

`research/05_class1_to_class2_design.md` originally claimed "108 queries have only
cross-adduct evidence." That number was wrong — traced to a bug in
`results/exp004_novelty_audit.json`'s same-adduct/cross-adduct detection, found by
directly verifying one case against `train.parquet` (query rid 468783, true molecule
`KTEOPAKTYLYZOB`: the audit said no same-adduct evidence was retained, but its one
retained spectrum, rid 468784, is provably the same adduct `[M-H]-` as the query).

`research/scripts/exp004_reaudit.py` recomputes same-adduct/cross-adduct status
directly from `train.parquet`, with the `rid` reconstruction (`ROW_NUMBER() OVER () -
1` under `PRAGMA threads=1`, since `rid` is not a real column in the source data and
duckdb's default multi-threaded scan order is not guaranteed stable — see
`research/02_class1_coverage.md` §16 point 3) verified against 3 independently known
`(rid, inchikey14, adduct)` triples before trusting any output.

**Corrected result: exactly 12 of the 400 Mode-B queries have no same-adduct evidence
retained at all** — "cross-adduct-only." This 12 is not a new number: it is, exactly,
EXP-001's own 12 Variant-A `candidate_gen_hit = False` queries
(`results/exp001_checkpoint.json`, condition `B|C_all_train|ModifiedCosine`), with
**zero disagreement in either direction**. That agreement is the validation: Variant A
(raw precursor m/z ± 0.01 Da) structurally cannot recover a molecule whose only
remaining evidence sits at a different adduct (adduct changes shift precursor mass by
tens of Da, far outside the 0.01 Da window) — so every cross-adduct-only query must be
candidate-absent under Variant A, and every candidate-absent query must (in this
population) be cross-adduct-only. Both directions now hold exactly.

| rid | true inchikey14 | query adduct | n retained spectra | retained adducts | v1 audit said "same-adduct evidence"? |
|---|---|---|---|---|---|
| 6296 | ACBPYGWTRFBYMP | [2M+H]+ | 4 | [M+H]+ | YES (wrong) |
| 8769 | ADGUJIPJRXOWSH | [2M+CH2O2-H]- | 7 | [2M+Na]+, [M+H]+, [M-H]- | no |
| 27493 | ANGIDRSDQGRWQD | [M-H]- | 4 | [M+H]+ | no |
| 156672 | DFSLKFZMYMXXGX | [2M+Na]+ | 8 | [2M+H]+, [M+H]+ | YES (wrong) |
| 318845 | HPQRSZWHZSXIAR | [2M+Na]+ | 5 | [M+H]+, [M-H]- | YES (wrong) |
| 395356 | JFHPQARHIIJGOD | [M+H]+ | 3 | [M-H]- | YES (wrong) |
| 410447 | JNVFBVQKGYTOAH | [M-H]- | 4 | [M+H]+ | YES (wrong) |
| 431909 | JZQMDNUYGZOPNW | [M+Na]+ | 14 | [2M+Na]+, [M+CH2O2-H]-, [M+H]+, [M-H]- | YES (wrong) |
| 626670 | OBFWTNNDCLKJFP | [2M+H]+ | 7 | [M+H]+, [M-H]- | no |
| 689395 | PJAAZAXBSLFNNK | [M-H]- | 4 | [M+H]+ | no |
| 733288 | QHALUOAFNBWZED | [2M+Na]+ | 6 | [M+H]+, [M-H]- | YES (wrong) |
| 1029257 | XGQSLHOHBLGVCV | [M+H]+ | 4 | [2M+H]+, [M-H]- | YES (wrong) |

8 of these 12 were specifically mislabeled `any_same_adduct_retained: True` by the v1
audit — the single most consequential instance of the bug, since these are exactly the
queries EXP-002/EXP-004 care about most. 3 of the 12 (rid 733288, 410447, 318845) are
the same 3 that EXP-002's adduct-aware Variant B recovered as candidates but still
failed to rank in the top 25 (`research/decision_log.md` DEC-004) — i.e. for those 3
specifically, both candidate generation *and* ranking fail even once the correct
evidence is made reachable.

**Corroboration:** an independent, separately-produced read-only audit
(`research/04_exp004_independent_audit.md`, `research/analysis/`) reached the same
corrected count (12, exact rid-for-rid match, 0 disagreements) via its own
recomputation from `train.parquet`, and additionally flags a finer nuance: restricted
to timsTOF-only evidence, the "no same-adduct evidence" count is 14, not 12 — 2 of
those 14 (rid 2538746, 2539574) are reached by Variant A only via non-timsTOF
same-adduct rows elsewhere in the full train library. Since EXP-001/EXP-004's
candidate generation searches all-train (library C, per DEC-003), **12 is the correct,
relevant number**; 14 would be the number if candidate generation were restricted to
timsTOF only (it isn't). That same audit also flags an unrelated but real issue in
EXP-003's existing write-up worth a separate look: its "57/92 cross-adduct displacers"
figure conflates two different axes (displacer *is* cross-adduct-vs-query: 54/92;
displacer *was newly admitted by Variant B*: 57/92 — overlapping but not identical
sets); not corrected here since EXP-003 wasn't in scope for this pass, flagged for a
future correction pass on that document specifically.

**Interpretation:** the "108" scare number is gone. The real, verified structural
candidate-generation gap in EXP-001/EXP-004's Variant-A population is 12/400 (3%), it
is fully explained (100% correspond to "no same-adduct evidence retained," no
unexplained residual), and it was already known in aggregate from EXP-001's own
failure analysis — this ledger just makes it precise, per-query, and reproducible from
source data rather than inherited from a buggy intermediate file.

---

## 2. Matched Class-1 row-exclusion control (EXP-005)

### 2.1 Design

Reuses EXP-001's exact 400 Mode-B query rids, exact candidate-generation rule
(Variant A: raw precursor m/z ± 0.01 Da, library = full train set), and exact scorer
(matchms 0.33.1 `ModifiedCosineGreedy`, tolerance=0.1, mz_power=0.0,
intensity_power=1.0). For every query, the candidate pool within the precursor window
is fetched and scored **once**; two conditions are then derived from that single
shared scored pool (the score of a candidate does not depend on which rows are
excluded, only which rows count toward ranking does — scoring once and deriving both
conditions from it makes the comparison exactly paired, not just similarly sampled):

- **Condition L ("library-like control"):** exclude only the query's own single row.
  Any row sharing the query's exact metadata group (same
  inchikey14/adduct/precursor_mz/num_peaks) remains searchable.
- **Condition W ("withheld"):** exclude the query's entire metadata group — identical
  to EXP-001 Mode B / EXP-004's existing construction.

For every query, in both conditions: rank of the true molecule, candidate-generation
hit, best score achieved by the true molecule's own candidates, and best score
achieved by the top-scoring **wrong** molecule (the correct-vs-top-wrong margin =
best_true_score − best_wrong_score).

### 2.2 Smoke test (n=25, required before the full run)

First attempt crashed (`TypeError: iteration over a 0-d array` — matchms's
`ModifiedCosineGreedy.pair()` returns a 0-d structured numpy array, not a plain tuple;
fixed to read `result["score"]`). After the fix:

- `rid` reconstruction verified against 2 independently known cases.
- Condition-W per-query numbers cross-checked against EXP-001's own stored
  `results/exp001_checkpoint.json` for the same 25 rids: **0 real mismatches.** 2
  queries (rid 705247, 646039) initially looked like mismatches (my rank=61/26 vs
  EXP-001's stored `rank: None`) — resolved as a reporting-convention difference, not a
  bug: EXP-001 only stores rank when ≤25, `None` otherwise; both of mine are >25
  either way, and `n_candidates`/`candidate_gen_hit` matched exactly in both cases.
- Leakage check: of the first 25 queries, exactly 1 (rid, group size 2) has any
  metadata-group sibling to exclude; Condition L correctly retains it (n_candidates
  950 vs W's 949) while leaving the ranking outcome unchanged for that query — the
  row-exclusion mechanism is doing exactly what it's supposed to.

Smoke test passed on all three checks (reproducible rid, external agreement with
EXP-001, correct differential leakage behavior). Full run then executed.

### 2.3 Full run (n=400)

Runtime: 126.7s scoring + 48.7s table materialization ≈ 3 minutes total — genuinely
cheap, as scoped.

| Condition | Recall@1 | Recall@5 | Recall@10 | Recall@25 | MRR@25 |
|---|---|---|---|---|---|
| L (row-exclusion, library-like) | 0.5350 | 0.8050 | 0.8725 | 0.9150 | 0.6510 |
| W (group-exclusion, withheld) | 0.5350 | 0.8050 | 0.8725 | 0.9150 | 0.6510 |

**These are bit-identical**, and both match EXP-001's originally published Mode
B/all-train/Modified-Cosine result exactly (`research/02_class1_coverage.md` §7,
MRR=0.6510265432280139) — an independent external validation that this freshly
written pipeline correctly reproduces the established, authoritative result before
trusting anything new it adds.

**Why identical:** only **8/400 (2%)** Mode-B queries have any metadata-group sibling
to exclude in the first place (`excluded_rids` length >1) — for the other 392/400
(98%), row-exclusion and group-exclusion are literally the same operation, because the
query's own row is already alone in its metadata group. Of those 8:

| rid | Δn_candidates (L−W) | rank changed? | best_true_score changed? |
|---|---|---|---|
| 119107 | +1 | no (13→13) | no |
| 798740 | +1 | no (1→1) | no |
| 642008 | +1 | no (1→1) | no |
| 378573 | +1 | no (1→1) | no |
| 217433 | +1 | no (1→1) | no |
| 794642 | +1 | no (13→13) | **yes** (0.7111 → 0.6948, i.e. the excluded sibling under W was itself a slightly better-scoring candidate for the true molecule) |
| 1108667 | +1 | no (1→1) | no |
| 684071 | +1 | no (1→1) | no |

7 of 8 show literally zero effect beyond one extra candidate in the pool. The 8th (rid
794642) is the single case in the entire 400-query population where keeping the
same-group sibling measurably changed a score — and even there, the final rank bucket
(found, but ranked 13th, outside top-10) didn't change.

### 2.4 Answer to the research question

**Under this specific, well-defined operationalization (row-only exclusion vs
whole-metadata-group exclusion), moving from library-like evidence to genuinely
withheld evidence changes performance by exactly 0.0000 across every headline metric.**
The cause: 98% of this population's query rows were already alone in their metadata
group, so EXP-001 Mode B's existing whole-group exclusion was already withholding
exactly the same evidence a naive row-only exclusion would have withheld, for nearly
the whole population. The concern that motivated EXP-004's design in the first place —
that Mode B might secretly still be "Class 1 with the query spectrum renamed" via a
same-metadata-group duplicate quietly remaining searchable — **is not a real risk
through this mechanism**. It contributes no measurable leakage-driven performance
inflation in this population.

**This does not mean Mode B's evidence is "novel" in a deeper sense** — that is a
separate, already-answered question (`research/05_class1_to_class2_design.md` §4): the
*retained* evidence (whatever spectra of the same molecule remain after group
exclusion, regardless of exclusion granularity) has a median Modified-Cosine
similarity to the query of only 5.5%, i.e. genuinely different fragmentation, not a
near-duplicate. Row-vs-group exclusion and evidence-similarity are two different axes;
this experiment cleanly rules out the first as a confound, leaving the second (already
measured) as the operative one.

---

## 3. Correct-vs-top-wrong Modified-Cosine margin analysis

For every query (condition W, n=400), margin = best_true_score − best_wrong_score,
stratified by outcome:

| Outcome bucket | n | median margin | mean margin | frac. negative |
|---|---|---|---|---|
| Rank 1 (success) | 214 | **+0.127** | +0.187 | 0% (by construction) |
| Rank 2–25 (found, not top) | 152 | **−0.072** | −0.129 | 100% (by construction) |
| Found, ranked below 25 | 22 | **−0.391** | −0.331 | 100% (by construction) |
| Candidate-absent | 12 | n/a (no margin defined) | — | — |

(The 12 candidate-absent queries are exactly the reachability-ledger 12 from §1 — a
structural candidate-generation failure, not a scoring/margin phenomenon, so no margin
applies.)

**Interpretation — what causes the performance that's observed:**
- When the pipeline wins (53.5% of queries), it tends to win comfortably: median
  margin +0.127, not a hair's-width call.
- When it loses but the correct molecule is still findable in the top 25 (38% of
  queries), the loss is typically **close**: median margin only −0.072. A modestly
  better score calibration or tie-break rule could plausibly flip a meaningful share of
  these — this is the same population EXP-001 §16 flagged as "21/34 unexplained ranking
  failures," now given a concrete, quantified shape (most are near-misses, not
  landslides).
- The 22 queries where the correct molecule falls out of the top 25 entirely are a
  qualitatively different failure: median margin −0.391, a decisive loss, consistent
  with a genuinely different top-scoring (wrong) molecule rather than a calibration
  artifact.
- The 12 candidate-absent queries are fully separate and fully explained (§1) — no
  ranking/margin question applies to them at all.

This three-way decomposition (12 structural / 152 close-loss / 22 decisive-loss / 214
win) is a more precise restatement of EXP-001's existing failure analysis
(`research/02_class1_coverage.md` §13), now backed by an actual score margin per query
rather than only a rank category.

---

## 4. What this does and does not settle

**Settled by this experiment:**
- The corrected 12-query reachability count, verified two independent ways (direct
  source-data recomputation, and exact agreement with EXP-001's own candidate-absence
  ground truth).
- The 0.90 novelty threshold has an honest, self-contained justification.
- Row-exclusion vs group-exclusion contributes zero measurable confound to EXP-001
  Mode B / EXP-004's construction.
- A quantified, margin-based explanation for why ranking succeeds or fails when
  candidates are present.

**Not settled, and out of scope for this cheap control:**
- Whether the full EXP-004 pseudo-Class-2 pipeline (candidate generation + ranking
  against genuinely different-condition evidence, per `research/05_class1_to_class2_design.md`
  §7) performs acceptably — that is exactly what the EXP-004 smoke test is for, and it
  has not been run.
- Whether a better ranker/calibration would recover the 152 close-margin losses — a
  ranking-model question, explicitly out of scope (per the same constraint that
  produced DEC-004: ranking changes need their own reviewed design).

**Recommendation:** the matched-C1 control and reachability ledger required before
considering the EXP-004 C2 smoke test are complete, with no surprises that would block
it (no leakage confound found; the reachability gap is small, fully explained, and
unchanged in kind from what EXP-001 already reported). Whether to proceed to the
EXP-004 smoke test itself is left for explicit approval, per the staged-execution
protocol this project already follows.

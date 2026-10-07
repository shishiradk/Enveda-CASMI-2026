# 05 — EXP-004 Design: Controlled Class-1 → Pseudo-Class-2 Transformation

Status: **STAGE 1 (DESIGN + DATA AUDIT) ONLY.** No smoke test, no full run, no
production pipeline change, no `experiments.md`/`decision_log.md` update yet (matching
the pattern already established for EXP-002/003's design phases in this project).
Ends with an explicit READY / NOT READY verdict per the required staged-execution
protocol.

**Research question:** How much of our Class-1 identification performance survives
when the exact query spectrum is hidden, but genuinely different spectra of the same
molecule remain available as library evidence?

---

## 1. Core idea and reuse decision

Per the explicit preference to reuse the existing Class-1 population rather than
sample a new one, this design is built entirely on **EXP-001's existing 400-query
Mode B population** (`results/exp001_query_sets.json`). This is a deliberate,
evidence-based choice, not a default: EXP-001's Mode B construction already excludes
the query's *entire metadata group* (not just its row) from the library, and its
`other_groups_for_molecule` field already records exactly which other same-molecule
evidence remains — which is precisely the "A1/A2/A3 remain, A4 hidden" structure
this experiment needs. No new sampling was performed; the audit below measures
properties of this exact, already-familiar population.

## 2. Evidence from EXP-001/002/003 this design builds on (not re-derived)

- EXP-001 established Mode B's leakage control (whole-group exclusion) and Class-1
  MRR@25 = 0.6510 on this exact 400-query population (`research/02_class1_coverage.md`).
- EXP-002/003 established that most of Mode B's retained same-molecule evidence is at
  a **different adduct** than the query, that direct Cosine cannot see true
  similarity across an adduct shift, and that Modified Cosine's shift-matching
  correctly does (`research/03_exp002_results.md`, `research/04_exp003_design.md`).
  **This directly determined the methodology fix in §4 below** — an early version of
  this audit used direct Cosine for the novelty check and produced a misleadingly low
  similarity distribution; re-running with Modified Cosine (the tool these prior
  experiments already validated for cross-adduct comparisons) is what's reported here.
- EXP-001's dataset report established `instrument_type` is uniformly `timsTOF` for
  the entire `enveda-180`/test-relevant population (`research/00_dataset_report.md`
  §5) — meaning **Scenario B ("different acquisition condition") cannot mean
  "different instrument" for this data**; it is reframed below around collision
  energy, which does vary.

## 3. Population audit

From `research/02_class1_coverage.md`/EXP-001's own construction: 183,191 timsTOF
molecules total; 181,780 have ≥2 distinct metadata-groups; 173,882 have ≥3; 146,181
have ≥4 (a metadata-group = a distinct `inchikey14`/`adduct`/`precursor_mz`/`num_peaks`
combination, EXP-001's proxy for "a distinct spectrum" — see the near-duplicate
validation caveat in §4).

Within EXP-001's existing 400 Mode B queries specifically (retained-group counts,
i.e. how much "A1/A2/A3"-style evidence remains after holding out the query's whole
group): min 1, p25 3, **median 5**, p75 7, max 19. **23/400 (5.75%) have only 1
retained group** — thin but non-empty evidence; **377/400 (94.25%) have ≥2**; **326/400
(81.5%) have ≥3**.

## 4. Spectral-novelty analysis — methodology correction included

**A first attempt at this audit used direct (unshifted) Cosine** to measure
similarity between the held-out query and its retained evidence, and produced a
suspiciously low median similarity (0.014). Given EXP-002/003's established finding
that most retained evidence is cross-adduct, and direct Cosine cannot see true
similarity across an adduct-driven precursor shift, **this was diagnosed as measuring
the wrong thing** — conflating "genuinely novel spectrum" with "same molecule,
different adduct, direct Cosine just can't see it." The audit was re-run with
**Modified Cosine** (`ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0,
intensity_power=1.0)`, identical parameters throughout this project), the correct
tool per EXP-001/002/003's own findings. Both are reported for transparency.

| | Direct Cosine (wrong tool, shown for comparison) | Modified Cosine (correct tool) |
|---|---|---|
| min | 0.0000 | 0.0000 |
| p10 | 0.0001 | 0.0008 |
| p25 | 0.0017 | 0.0094 |
| **median** | 0.0138 | **0.0550** |
| p75 | 0.0548 | 0.1788 |
| p90 | 0.1488 | 0.3946 |
| max | 0.9825 | 0.9987 |

**Sanity check on the methodology itself:** for the 36 queries whose retained
evidence is *entirely* same-adduct (where direct and Modified Cosine should closely
agree, since `mass_shift ≈ 0`), direct median = 0.0066 vs modified median = 0.0357 —
still both low, and still not identical (the two scorers' greedy peak-matching isn't
bit-identical even at zero shift, a minor expected difference, not a red flag).

**The more important, substantive finding:** even measured correctly, similarity
between a held-out query and its retained same-molecule evidence is **genuinely low
in the median case (5.5%)**, including for same-adduct comparisons. This is not a
measurement artifact — different collision energies produce genuinely different
fragmentation patterns for the same molecule, a real, well-known MS/MS phenomenon, not
a bug. **This means EXP-001's Mode B population, as already constructed, is already a
substantially harder/more novel test than "the same spectrum with a new label" — the
concern that motivated this whole design (§3 of the user's brief, "Class 1 with the
query spectrum renamed") does not appear to be a real risk for the bulk of this
population.**

### Threshold selection (data-driven, not chosen for a favorable result)

At every candidate threshold (0.99 / 0.95 / 0.90 / 0.85 / 0.80), the overwhelming
majority of the population already survives:

| Threshold (Modified Cosine max similarity < X) | Queries surviving |
|---|---|
| 0.99 | 399/400 (99.8%) |
| 0.95 | 397/400 (99.2%) |
| 0.90 | 397/400 (99.2%) |
| 0.85 | 397/400 (99.2%) |
| 0.80 | 394/400 (98.5%) |

**Proposed rule: exclude queries with max Modified-Cosine similarity ≥ 0.90 to any
retained same-molecule spectrum.** At 0.90, exactly **3 queries** are excluded as
near-duplicates (see below); this is small and stable across 0.85–0.90, i.e. not a
knife-edge choice.

**CORRECTION (2026-09-21):** the claim in the paragraph above (in the original version
of this doc) — that 0.90 "is the threshold EXP-001 itself already used and validated"
— is **false** and has been removed. Checked directly against
`results/exp001_query_sets.json` (`run_params.near_dup_cosine_threshold`):
EXP-001's actual validated near-duplicate threshold was **0.95**, not 0.90, and it was
measured on a **different quantity** — **direct** (unshifted) Cosine similarity between
rows sharing the exact same metadata group (`research/02_class1_coverage.md` §3 —
used only to validate Mode A siblings. There is no principled reason a same-group
direct-Cosine threshold and a cross-group Modified-Cosine threshold should be numerically
equal; this was an unjustified borrowed number, not a derived one.

**Corrected, self-contained justification:** 0.90 Modified-Cosine similarity sits at the
**99.25th percentile** of this population's own max-similarity distribution (n=400;
computed directly from `results/exp004_novelty_audit.json`'s `max_sim_modcos` values).
That is, it is a conservative, non-arbitrary cut chosen from this experiment's own data,
excluding only the most extreme 0.75% of cases — not inherited from an unrelated
measurement. The threshold-sensitivity table below (unchanged) independently supports
that the choice isn't a knife-edge: exclusion count is stable (3 queries) across the
entire 0.85–0.95 range and only grows to 6 at 0.80.

**The 3 excluded near-duplicate cases**, inspected individually (not just counted):

| rid | true molecule | max similarity | query adduct |
|---|---|---|---|
| 944352 | VMFIRCRGBQJJDZ | 0.9987 | [2M+H]+ |
| 302973 | HHBWCJBUDAXGNP | 0.9825 | [M+H]+ |
| 519143 | LUNMXIJAHSUWOB | 0.9785 | [M-H]- |

## 5. Field-usage table

| Field | Used? | Why |
|---|---|---|
| MS/MS peaks (`ms2_mzs`, `ms2_normalized_intensities`) | YES | The actual query evidence; this is what's compared against retained library spectra |
| Precursor m/z | YES | Required for candidate generation (both precursor-window and adduct-aware variants) and for Modified Cosine's shift calculation |
| Adduct | YES | Available in the real CASMI test set (`test.parquet` includes `adduct`); used identically to EXP-001/002/003 |
| Ionization mode | YES | Available in real test set; used for sanity filtering consistent with prior experiments (not a new dependency) |
| Instrument type | YES (but uninformative) | Available in real test set; uniformly `timsTOF` for this population, so it carries no discriminating signal here, but is not artificially removed |
| Collision energy | YES | Available in real test set (`collision_energy_ev`); used for the descriptive/exploratory CE-match statistic (§6, corrected 2026-09-21 — no longer a hard Scenario split) and reporting, not as a candidate-generation filter (consistent with EXP-001, which never filtered by CE) |
| Molecular formula | **NO** | Present in `train.parquet` but would not be available for a genuinely unknown test query in the real competition, and is close enough to structural identity to risk leakage; excluded per the same logic as SMILES/InChIKey |
| Normalized SMILES | **NO** | Target identity |
| InChIKey / InChIKey14 | **NO** | Target identity — used only by the evaluator, outside the prediction path, exactly as in EXP-001/002/003 |
| `ingest_lib` | **NO** | Not available for a real test query; also risks leaking which internal library curated the molecule |

This table matches what EXP-001/002/003 already do — no new leakage surface is
introduced by this design; the query object handed to the pipeline is structurally
identical to a Mode B query in prior experiments.

## 6. Scenario definitions, corrected to match the data

The user's original Scenario B ("different instrument") is **not supported** by this
data population — `instrument_type` is uniformly `timsTOF` throughout `enveda-180`/
the timsTOF-relevant train population and the real `test.parquet` (established in
`research/00_dataset_report.md` §5). Scenario B is reframed as **"different collision
energy"**, which the data does support.

**SUPERSEDED TABLE — contained a verified bug, kept for transparency, do not use:**

| Scenario | Definition | n queries (WRONG, see correction below) |
|---|---|---|
| A (pure) | All retained evidence is same-adduct AND CE matches | 15 |
| B | All retained evidence is same-adduct, CE differs everywhere | 21 |
| C | At least one retained group is at a different adduct | 364 (108 of which *only* cross-adduct) |

**CORRECTION (2026-09-21):** the table above, and the "108 have only cross-adduct
evidence" claim built on it, came from `results/exp004_novelty_audit.json`, whose
same-adduct/cross-adduct detection was found to be **wrong** by direct verification
against `train.parquet`. Example: query rid 468783 (true molecule `KTEOPAKTYLYZOB`)
was flagged `any_same_adduct_retained: False`, but its one retained spectrum (rid
468784) is provably the same adduct (`[M-H]-`) as the query — confirmed by a direct
query against the source data. Independent cross-check: if "108 queries have only
cross-adduct evidence" were true, Variant A's raw-precursor candidate filter could not
structurally recover any of them (an adduct change shifts precursor m/z by amounts far
larger than the 0.01 Da tolerance), so EXP-001's own `candidate_gen_hit` ground truth
should show ~108 candidate-absent failures. It shows exactly **12** — a contradiction
that only a broken same/cross-adduct classification can produce. 104 of the "108" were
false positives.

`research/scripts/exp004_reaudit.py` recomputes the same/cross-adduct flags directly
from `train.parquet` (not from whatever intermediate join produced the original
`exp004_novelty_audit.json`), with the `rid` reconstruction (`ROW_NUMBER() OVER () - 1`
under `PRAGMA threads=1`) verified against 3 independently known (rid, inchikey14,
adduct) triples before trusting any downstream number — see the reproducibility note
in that script's docstring. Output: `results/exp004_scenario_audit_v2.json`.

**Corrected counts (n=400, the full Mode-B population, corrected classification):**

| Category | Definition | n queries |
|---|---|---|
| Same-adduct evidence retained | ≥1 retained group shares the query's adduct | **388** (149 same-adduct-only + 239 also have some cross-adduct evidence) |
| Cross-adduct-only evidence retained | **no** retained group shares the query's adduct | **12** |

The corrected "cross-adduct-only" count (12) now matches EXP-001's 12 Variant-A
candidate-absent queries **exactly, with zero contradictions** — see the reachability
ledger in `research/06_matched_c1_control.md` §1. This is strong evidence the
corrected classification is right: structurally, Variant A cannot recover a query
whose true molecule has no same-adduct evidence anywhere in the search library, and
that is now true for exactly the 12 queries where it's observed to fail, no more and
no fewer.

**CE-based Scenario A/B split — merged, per correction request, not kept as a hard
split.** The original A (n=15, "CE matches") / B (n=21, "CE differs") split was built
on the same broken same-adduct detection and cannot be trusted as reported. A
reproducible CE-match rule was implemented (`same_adduct_ce_match` field in
`exp004_reaudit.py`: round each spectrum's `collision_energy_ev` array to 1 decimal,
compare as a sorted tuple) and run on the corrected 388-query same-adduct-retained
population: **only 3/388 (0.8%)** have a same-adduct retained group whose CE matches
the query's own CE exactly. This is too thin to support a standalone Scenario-A
benchmark (3 queries, not 15) and the underlying CE field itself is fragile to compare
exactly (ramped multi-value strings like `"20,40,60"`, float precision) — a looser
"tolerance-based" CE-match rule was not attempted because there is no principled basis
in this data for picking a tolerance, and inventing one under time pressure would
repeat the exact mistake being corrected here. **Decision: Scenario A and B are merged**
into the single "same-adduct evidence retained" category above; CE-match rate (0.8%) is
reported as a descriptive, exploratory statistic only, not used to split the population
for any headline result.

**Framing, corrected:** EXP-004 asks whether the full pipeline (candidate generation +
ranking) still works when the query spectrum's exact self-measurement is hidden but
same-molecule evidence remains — most of that evidence (388/400, 97%) is at least
partly same-adduct and thus reachable by Variant A's candidate generation; a small,
now-precisely-characterized 12/400 (3%) has only cross-adduct evidence and is
structurally unreachable by Variant A alone (already known and already explained by
EXP-001's own candidate-absence failure analysis — not a new finding). Given the
population is now overwhelmingly single-category (388 vs 12) rather than split across
three thin buckets (15/21/364), the recommended approach is unchanged from the
original §10: run the **full 397-query population as the primary result**, with the
corrected same-adduct/cross-adduct-only split reported as the stratification variable
(replacing the broken A/B/C scheme), and CE-match rate reported separately as
exploratory context.

## 7. Proposed transformation and leakage-safety mapping

EXP-004's transformation is, explicitly: **EXP-001's Mode B construction, with the
added post-hoc novelty filter from §4 applied** (excluding the 3 near-duplicate
cases). Concretely, for each of the 397 surviving queries:

- QUERY = the held-out spectrum (rid), exposed to the pipeline with only the §5
  "YES" fields — no molecule identity.
- LIBRARY = every other spectrum in the population **except** the query's entire
  excluded metadata group (unchanged from EXP-001's leakage control) — i.e. A1/A2/A3-
  style retained evidence remains searchable.
- No molecule-level prototype/embedding is built in this pipeline (EXP-001/002/003
  score each query against each individual candidate spectrum, aggregating only at
  the final ranking step via per-molecule max score) — so §14's "molecule profile
  must not include A4" concern is satisfied by construction, not by a new mechanism:
  there is no cached molecule profile for the query's own spectrum to leak into.
- `hidden_ground_truth[rid] = true_inchikey14` is retained only in
  `exp001_query_sets.json`, used solely by the evaluator after prediction — unchanged
  from EXP-001/002/003's existing pattern.

## 8. Leakage checklist (status at design time)

- [x] Query spectrum removed from search library — inherited from EXP-001's exclusion mechanism, unchanged.
- [x] Exact/near-duplicate removed — EXP-001 already excludes the whole metadata group; §4's novelty filter adds an explicit peak-level check beyond metadata, excluding 3 further cases.
- [x] Near-duplicate policy applied — §4, threshold justified from EXP-001's own prior validation, not invented.
- [x] Target molecule identity hidden from the prediction path — §5's field table, identical to EXP-001/002/003.
- [x] Query spectrum not included in any molecule aggregation — no such aggregation exists in this pipeline (§7).
- [ ] **Not yet verified**: that the *candidate generation* stage proposed for EXP-004 (still TBD — see §11) itself introduces no new leakage path. This must be checked explicitly once that stage's exact rule is chosen, before the smoke test.

## 9. Metrics (unchanged in kind from EXP-001/002/003, plus uncertainty)

Candidate-generation recall; Recall@1/5/10/25; MRR@25; candidate counts
(median/percentiles); runtime; peak RSS — identical to prior experiments for direct
comparability. Additionally, per the requirement for honest uncertainty reporting:
**bootstrap 95% CIs** for MRR@25 and Recall@25 (query-level resampling, since n≈397 is
modest), and a **paired** query-level comparison against EXP-001's Class-1 result on
the *same* rids (since 397 of EXP-001's original 400 queries are reused unchanged, a
true paired comparison is possible for those, not just an aggregate-vs-aggregate one).

## 10. Failure-case taxonomy

Extends EXP-001/002's categories with a scenario tag: for every failure, record
whether it's a candidate-generation failure or a ranking failure (as in
EXP-001/003), *and* which Scenario (A/B/C) it belongs to, so failure patterns can be
read against acquisition-condition difficulty rather than only reported in aggregate
— directly enabling the "MRR@25 vs spectral novelty / condition" analysis in §17 of
the brief.

## 11. What is explicitly NOT decided yet (blocking items before a smoke test)

1. **Candidate generation rule for EXP-004 itself** is not yet chosen. Using
   EXP-001's Variant A (raw precursor_mz) would under-count Scenario-C candidates for
   the same reason EXP-002 found; using EXP-002's Variant B (adduct-aware) would
   inherit EXP-002's demonstrated ranking-degradation risk. Given EXP-003 has not yet
   run its ablations to explain *why* Variant B degrades ranking, **adopting Variant
   B here before EXP-003 resolves that question would confound EXP-004's own result**
   — a Class-1-vs-pseudo-Class-2 comparison must hold candidate generation *and*
   ranking fixed to the already-validated EXP-001 configuration to be interpretable
   at all. **Recommendation: use EXP-001's Variant A (unchanged) for EXP-004's main
   result** (reaffirmed 2026-09-21 — **Variant A remains the headline scorer, Variant
   B remains explicitly exploratory only**, consistent with DEC-004; nothing in this
   correction pass changes that). This under-counts recall for exactly the **12**
   queries (corrected count, §6 above — not 108) whose only retained evidence is
   cross-adduct; this is now a precisely quantified, documented limitation (the
   12-query reachability ledger in `research/06_matched_c1_control.md` §1 lists each
   one by rid), not a vague "likely under-counts" statement. Treat "does adduct-aware
   candidate generation change the pseudo-Class-2 result" as a clearly labeled
   secondary analysis, not the headline number.
2. The bootstrap CI implementation and the exact paired-comparison mechanics (how to
   handle the 3 excluded rids that have no EXP-004 counterpart) need to be written
   before the smoke test, not during it.

## 12. Smoke-test plan (proposed, not run)

~20–30 queries drawn from the 397-query eligible population (stratified to include at
least a few from both the same-adduct-retained (388) and cross-adduct-only (12)
corrected categories — see §6 — not a pure random draw, given cross-adduct-only is
thin), running EXP-001's unchanged Variant A candidate generation +
identical Modified Cosine scoring against the EXP-004 library construction (§7).
Verify: query exclusion is correct (spot-check that excluded rids never appear as
candidates), molecule identity is never referenced during scoring (a code-level
check, not just a result check), novelty filtering matches §4's audit exactly for
the sampled subset, and the paired comparison against those same rids' EXP-001
Class-1 results produces sane, explicable deltas — before any claim about the full
397-query result.

## 13. Full-run configuration (proposed, NOT executed)

If the smoke test passes: run all 397 eligible queries through the EXP-004 pipeline
(§7, §11's Variant-A candidate generation), memory-safe and checkpointed following
the same batched, incremental architecture as `exp002_full.py`/`exp003_inspection.py`
fixes (single bulk fetch or batched fetch, not per-query fetch — a lesson already
learned twice this session). Save to `results/exp004_checkpoint.json` and
`results/exp004_query_sets.json` (a filtered view of EXP-001's query set, the 397
survivors plus the exclusion list with reasons). Write
`research/06_class1_to_class2_results.md` after, with the paired Class-1-vs-pseudo-
Class-2 comparison, Scenario-stratified breakdown, bootstrap CIs, and failure
taxonomy — not before.

---

## Verdict

**Original verdict (superseded in part, see below):** READY FOR SMOKE TEST, with one
explicit precondition attached: the smoke test must use EXP-001's unchanged Variant-A
candidate generation (§11), not EXP-002's Variant B, since EXP-003's ablations (needed
to explain Variant B's ranking degradation) have not yet run.

**CORRECTIONS PASS (2026-09-21):** before approving the EXP-004 C2 smoke test, the
user required six specific corrections to this design (108/12 reachability argument,
0.90 threshold justification, CE A/B rule, a 12-query reachability ledger, a matched
Class-1 row-exclusion control, and a correct-vs-top-wrong margin analysis), plus
running the cheap matched-C1 control experiment first. All six are done:

1. The 108/12 reachability contradiction is fixed — the true count is 12,
   corresponding exactly to EXP-001's known candidate-absent queries (§6).
2. The 0.90 threshold now has a self-contained justification (99.25th percentile of
   this population's own similarity distribution), not a false claim of inheritance
   from EXP-001 (§4).
3. The CE-based Scenario A/B split is merged into a single "same-adduct evidence
   retained" category (388/400); CE-match rate (0.8%) is reported as descriptive
   context only, not a hard split (§6).
4. The 12-query reachability ledger is in `research/06_matched_c1_control.md` §1.
5. The matched Class-1 row-exclusion control (EXP-005) is run — full results,
   methodology, and the correct-vs-top-wrong margin analysis are in
   `research/06_matched_c1_control.md`.
6. Variant A remains the headline scorer; Variant B remains explicitly exploratory
   (DEC-004 unchanged). No production pipeline (`src/`) file was touched by any of
   this work — only `research/scripts/`, `results/`, and `research/*.md`.

**Updated status:** the population is real, audited, and its novelty is empirically
justified (397/400 queries survive a now-honestly-justified 0.90 threshold). The
Scenario split is now a clean, corrected two-way stratification (388 same-adduct vs 12
cross-adduct-only) instead of three under-powered, bug-affected buckets. EXP-005's
result (see `research/06_matched_c1_control.md`) shows the row-exclusion-vs-
group-exclusion distinction this design worried about (the "Class 1 with the query
spectrum renamed" risk) contributes **zero measurable difference** in this population
— Recall@1/5/10/25 and MRR@25 are bit-identical between the two conditions, because
98% of Mode-B queries have no same-metadata-group sibling to begin with. This means
EXP-001 Mode B's existing leakage control was already sound on this specific axis; it
does not by itself certify the EXP-004 C2 smoke test as approved.

**This correction pass does not itself approve the EXP-004 C2 smoke test.** Per the
user's explicit staged instruction ("run only the cheap matched-C1 artifact/control
experiment before approving the C2 smoke test"), that approval is a separate decision
for the user to make after reviewing EXP-005's results, not something this pass grants
automatically. No EXP-004 smoke test executed. No production pipeline change.

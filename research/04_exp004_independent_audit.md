# 04 — EXP-004 Independent Verification & Research Audit (Q1–Q6)

Status: **INDEPENDENT AUDIT COMPLETE.** Read-only verification of every claim
EXP-004's revised design relies on, re-derived from persisted artifacts plus
`train.parquet`, with no production pipeline change, no experiment re-run, and no
smoke test. One material correction is upstream of the smoke-test decision (see
§3 and §11): the same/cross-adduct and Scenario A/B/C census EXP-004's design
sections §2/§4/§6 are built on contains **112 wrong per-query flags** in
`results/exp004_novelty_audit.json` (v1), inflating "cross-adduct-only" from
**12** to **108**. The corrected census is reproduced independently and agrees
exactly with the separate re-audit `results/exp004_scenario_audit_v2.json`.

---

## 1. Files inspected

| Artifact | Role in this audit |
|---|---|
| `results/exp001_query_sets.json` | 400 mode_b queries, `excluded_rids`, `other_groups_for_molecule`, `run_params` (`near_dup_cosine_threshold: 0.95`, `precursor_tol_da: 0.01`, `seed: 42`), 200 mode_a queries |
| `results/exp001_checkpoint.json` | Variant A per-condition metrics + `per_query_diagnostics` (rank/hit/adduct; **no scores**) + failure taxonomy |
| `results/exp002_checkpoint.json` | `final_metrics` A/B + `delta`, `classification_summary` (388/12), `per_query_results` (`best_true_molecule_score_b` only, no per-candidate scores), `explosion_flags` |
| `results/exp003_inspection.json` | 92 outcome-selected queries: `correct_score`, `top_wrong_overall/same_adduct/cross_adduct`, `was_in_variant_a`, ranks |
| `results/exp004_novelty_audit.json` | **v1 audit** (400 entries; adduct flags + sim stats). Confirmed internally consistent with the design doc's tables but **flag portion corrupted (112 errors)** |
| `results/exp004_scenario_audit_v2.json` | v2 re-audit (`research/scripts/exp004_reaudit.py`). **Matches my independent recomputation exactly (0 disagreements)** |
| `research/scripts/exp004_reaudit.py` | v2 generator; `PRAGMA threads=1`, rid reconstruction verified vs 3 known triples |
| `train.parquet` | rid reconstruction (`ROW_NUMBER() OVER () - 1`, `PRAGMA threads=1`, verified vs 3 triples), adduct/CE/row-level retention |
| `research/05_class1_to_class2_design.md` | The design under audit (§2/§4/§6 census claims) |
| `research/02_class1_coverage.md`, `research/03_exp002_results.md`, `research/CASMI_RESEARCH_STATE.md` | Threshold provenance + Mode B construction provenance |

My own read-only analysis scripts (no pipeline touched):
- `research/analysis/exp004_independent_verify.py` → `research/analysis/exp004_independent_verify_out.json`
- `research/analysis/exp004_margin_analysis.py` → `research/analysis/exp004_margin_analysis_out.json`

---

## 2. Claims verified (reproduced from artifacts)

All of the following reproduce exactly from the persisted artifacts:

1. **Population and construction.** 400 Mode-B queries; whole-metadata-group
   exclusion (`excluded_rids`: 392×1, 8×2 — every len-2 pair verified to be
   same-`(inchikey14, adduct, precursor_mz, num_peaks)` siblings); retained
   groups per query min 1 / p25 3 / median 5 / p75 7 / max 19; 23/400 have 1,
   377/400 ≥2, 326/400 ≥3.
2. **Novelty filter (Q2).** Survivors at `max_sim_modcos < t` over the 400:
   `t=0.99 → 399`, `0.95 → 397`, `0.90 → 397`, `0.85 → 397`, `0.80 → 394` — exact
   match to design §4's table. The 3 excluded at ≥0.90 are rids **302973**
   (0.9825, [M+H]+), **944352** (0.9987, [2M+H]+), **519143** (0.9785, [M-H]-),
   individually confirmed.
3. **Variant A (Q1/Q3).** On all 400 (condition `B|C_all_train|ModifiedCosine`,
   `scorer ModifiedCosineGreedy`): candidate-gen recall **0.97** (388 hit, **12
   absent**), Recall@1 0.535 / R5 0.805 / R10 0.8725 / R25 0.915 /
   **MRR@25 0.6510265**; median candidates 689.5, mean 898.765, 1
   zero-candidate query. Failure taxonomy: 1 candidate-generation failure + 11
   molecule-absent-from-library; of the 388 reached, 366 correct at rank≤25, 22
   correct below rank 25.
4. **Variant B (Q3).** Recall@1 0.49 / R5 0.78 / R10 0.855 / R25 0.92 /
   **MRR@25 0.6178025**; candidate-gen recall 1.0 (all 12 previously-absent now
   admitted). Delta B−A: R1 −0.045, R5 −0.025, R10 −0.0175, R25 +0.005,
   MRR −0.033224, candgen +0.03. Classification **388 already-candidate-under-A
   / 12 newly-admitted-under-B**; `explosion_flags: []`; 50k cap never hit.
5. **EXP-003 (Q4).** 92/92 inspected (outcome-selected). Correct found at the
   same adduct: **90/92**; correct candidate present in Variant A: **90/92**.
   Displacer (top wrong) vs query adduct: **cross-adduct 54, same-adduct 38**.
   Displacer source: **newly admitted by B 57, already in Variant A 35**.
6. **Similarity distribution (design §4, Q2).** Reproduced up to percentile-
   method noise: per-query max direct median 0.0137 / modcos 0.0544
   (design: 0.0138 / 0.0550), p75 0.0548 / 0.1783, p90 0.1488 / 0.3943,
   max 0.9825 / 0.9987.
7. **Labels used by EXP-002.** EXP-002 re-used EXP-001's mode_b query set
   unchanged and Variant A scores from the same checkpoint — the paired A-vs-B
   comparison is internally valid.

---

## 3. Claims corrected (the material findings)

1. **The v1 novelty audit's adduct flags are wrong for 112/400 queries.** My
   independent row-level recomputation (all retained rows of the true molecule,
   threads=1, rid-reconstruction sanity-checked) disagrees with
   `exp004_novelty_audit.json`'s `any_same_adduct_retained` on **112 queries**,
   and agrees with `exp004_scenario_audit_v2.json` on **all 400** (0 disagreements).
   The v1 bug is an incorrect group→adduct join (group numbering is not stable
   across duckdb runs — the caveat already documented in `02_class1_coverage.md`
   §16 point 3), not a scoring error: similarity values are computed over
   genuinely-retained spectra and survive.

2. **The corrected census** (row-level, all train rows of the molecule outside
   the excluded group):

   | Count | v1 (design §2/§6) | Corrected (mine = v2) |
   |---|---|---|
   | have same-adduct retained evidence | 292 | **388** |
   | have cross-adduct retained evidence | 364 | **251** |
   | all retained evidence same-adduct | 36 | **149** |
   | **only cross-adduct evidence (no same at all)** | **108** | **12** |

3. **Q1 headline answers change.**
   - "How many of the 400 have NO same-adduct retained evidence in Variant A?"
     → **12, not 108.**
   - "How many of those are exactly Variant A's candidate-absent queries?"
     → **all 12.** `cross_adduct_only` rids == EXP-001 `candidate_gen_hit=False`
     rids, exactly (set equality, 12/12). No "reached despite cross-only"
     contradiction survives once the flags are corrected (v1-era "104/108
     reached" was an artifact of the bug — v2's contradiction check: 0 reached
     despite cross-only, 12 not-reached as expected).
   - The 12 nohit rids are the queries Variant A structurally cannot see
     (adduct shift ⇒ precursor-mass shift ≫ the ±0.01 Da window). EXP-002's 12
     `newly_admitted_under_B` are the same 12.

4. **Design §6 Scenario A/B/C table (A15 / B21 / C364) is broken twice over.**
   (a) It is computed from v1's wrong adduct flags; (b) the A-vs-B split uses a
   collision-energy rule that is not operationalized anywhere
   (`exp004_novelty_audit.json` persists only a boolean
   `any_different_ce_retained`). On the corrected all-same-adduct set (n=149),
   a reasonable "query CE appears among same-adduct retained rows" rule yields
   **A=2, B=147** — not 15/21. Scenario A/B as published cannot be reproduced
   and should not gate design decisions (§9, §11).

5. **Design §4's sanity-check subset (n=36 "entirely same-adduct") is the
   wrong subset.** Corrected subset is 149 (overlap with the old 36: only 31).
   Re-derived on the corrected subset the qualitative conclusion is unchanged —
   even same-adduct, retained similarity is low (per-query max median direct
   0.0075, modcos 0.0335) — so the "genuinely novel spectrum" conclusion
   survives, but the published numbers were computed on a buggy subset.

6. **Q4's "57 cross-adduct displacers" conflates two axes.** Decomposed:
   57/92 displacers were *newly admitted by Variant B*; 54/92 were
   *cross-adduct vs the query*; these are different subsets (39/92 displacers
   share the *correct* molecule's adduct). The correct "cross-adduct displacer"
   figure is **54/92, and 90/92 correct hits were still same-adduct**.

---

## 4. Twelve-query reachability ledger

Rids, query adduct, retained evidence, and status. All twelve are
cross-adduct-only (no same-adduct retained spectrum anywhere in the train
library), which is why Variant A's ±0.01 Da precursor filter misses them;
EXP-002's adduct-aware Variant B admits all twelve.

| rid | true molecule | query adduct | retained rows | retained adducts | v1 said same-adduct? |
|---|---|---|---|---|---|
| 6296 | ACBPYGWTRFBYMP | [2M+H]+ | 4 | [M+H]+ | YES (wrong) |
| 8769 | ADGUJIPJRXOWSH | [2M+CH2O2-H]- | 7 | [2M+Na]+, [M+H]+, [M-H]- | no |
| 27493 | ANGIDRSDQGRWQD | [M-H]- | 4 | [M+H]+ | no |
| 156672 | DFSLKZFMYMXXGX | [2M+Na]+ | 8 | [2M+H]+, [M+H]+ | YES (wrong) |
| 318845 | HPQRSZWHZSXIAR | [2M+Na]+ | 5 | [M+H]+, [M-H]- | YES (wrong) |
| 395356 | JFHPQARHIIJGOD | [M+H]+ | 3 | [M-H]- | YES (wrong) |
| 410447 | JNVFBVQKGYTOAH | [M-H]- | 4 | [M+H]+ | YES (wrong) |
| 431909 | JZQMDNUYGZOPNW | [M+Na]+ | 14 | [2M+Na]+, [M+CH2O2-H]-, [M+H]+, [M-H]- | YES (wrong) |
| 626670 | OBFWTNNDCLKJFP | [2M+H]+ | 7 | [M+H]+, [M-H]- | no |
| 689395 | PJAAZAXBSLFNNK | [M-H]- | 4 | [M+H]+ | no |
| 733288 | QHALUOAFNBWZED | [2M+Na]+ | 6 | [M+H]+, [M-H]- | YES (wrong) |
| 1029257 | XGQSLHOHBLGVCV | [M+H]+ | 4 | [2M+H]+, [M-H]- | YES (wrong) |

Notes:
- 8/12 were mislabeled "has same-adduct evidence" by v1 — the single most
  consequential bug class (they are the queries EXP-002/004 most care about).
- Two additional queries (2538746, 2539574) have **no same-adduct timsTOF
  evidence** but are reached by Variant A through non-timsTOF same-adduct rows,
  which is why the timsTOF-only "no same-adduct" count is 14 while the
  library-wide count is 12. EXP-001's candidate generation searches all-train,
  so the relevant number is **12**.
- All 12 appear in the failed/targeted inspection sets (EXP-003's 92; EXP-002's
  `known_11_report` covers 11, with rid 733288 at [2M+Na]+, shift 2.54 Da, i.e.
  a two-monomer shift).

---

## 5. Threshold verification (Q2)

- **EXP-001's actual near-duplicate criterion: direct (unshifted) Cosine
  ≥ 0.95** — `run_params.near_dup_cosine_threshold: 0.95`, applied to Mode A
  siblings (min validated sibling similarity 0.9514 over all 200 mode_a
  queries; mode_a validation in exp001_full_v2 uses direct `CosineGreedy`).
- **EXP-004's proposal: max Modified-Cosine ≥ 0.90 to any retained
  same-molecule spectrum** → excludes exactly 3. The survivor counts (§2.2)
  reproduce exactly.
- **What is wrong:** the design's justification — "the threshold EXP-001 itself
  already used and validated" — is a mis-citation. EXP-001 validated **0.95
  direct cosine**, not **0.90 Modified Cosine**. The coincidence that both pick
  the same 3 rids is incidental (the 3 are ≥0.98 sim, so any threshold in
  0.85–0.95 catches them).
- **What it means:** the *number 3* is data-derived and robust; the *provenance
  claim* is inaccurate. If EXP-004 keeps 0.90, it must be justified as its own
  data-driven choice (stable across 0.85–0.95), not attributed to EXP-001.

---

## 6. Variant A vs B verification (Q3)

Verified against `exp002_checkpoint.final_metrics` verbatim (n=400 each):

| Metric | A (EXP-001 cfg) | B (adduct-aware) | Δ (B−A) |
|---|---|---|---|
| Candidate-gen recall | 0.97 | 1.00 | +0.03 |
| Recall@1 | 0.535 | 0.490 | −0.045 |
| Recall@5 | 0.805 | 0.780 | −0.025 |
| Recall@10 | 0.8725 | 0.855 | −0.0175 |
| Recall@25 | 0.915 | 0.920 | +0.005 |
| MRR@25 | 0.6510 | 0.6178 | **−0.0332** |

- `per_query_results` classification: 388 `1_already_candidate_under_A` / 12
  `2_newly_admitted_under_B`; `explosion_flags: []` (50k cap never hit) — no
  candidate explosion confounder.
- Delta direction (ranking degrades on recall@1/5/10 and MRR even as
  candidate-gen recall rises) is confirmed. VERDICT: as published, correct.

---

## 7. Cross-adduct displacement (Q4)

From `exp003_inspection.json` (92 outcome-selected queries — a *selected* set,
not a random sample; interpret as "among the rank-degradation cases"):

| Count | Value |
|---|---|
| correct molecule found at same adduct | 90/92 |
| correct candidate present in Variant A | 90/92 |
| top-wrong displacer **cross-adduct** vs query | **54/92** |
| top-wrong displacer same-adduct vs query | 38/92 |
| displacer **newly admitted by Variant B** | **57/92** |
| displacer already ranked under Variant A | 35/92 |
| displacer shares the *correct* molecule's adduct | 39/92 |

Correct figures for the record: "54/92 cross-adduct" and "57/92 newly-admitted-by-B"
axes are different and both naturally emerged; the earlier "57 = cross-adduct"
phrasing is a conflation. The EXP-003 severity claim ("adduct-aware admission
introduces high-scoring cross-adduct decoys that displace correct same-adduct
hits") is **directionally supported but sized at 54/92, not 57/92**.

---

## 8. Margin analysis (Q5)

**Feasibility — resolved.** The full-397 margin distribution **cannot be
extracted from persisted artifacts**:

- `exp001_checkpoint.json` `per_query_diagnostics`: rank/hit/adduct only — **no
  scores**.
- `exp002_checkpoint.json` `per_query_results`: only `best_true_molecule_score_b`
  (correct molecule's best score under B) — **no wrong-molecule / per-candidate
  scores**, and no persisted candidate lists for a re-score.
- Only `exp003_inspection.json` (the 92 selected queries) stores
  `correct_score` + `top_wrong_overall.score`.

What IS computable (read-only script `research/analysis/exp004_margin_analysis.py`
+ `_out.json`), margin = correct − top-wrong, all on the 92-query subset:

| Stat | same-adduct displacer (38) | cross-adduct displacer (54) | newly-in-B (57) | already-in-A (35) |
|---|---|---|---|---|
| median margin | −0.071 | −0.053 | −0.045 | −0.118 |
| p25 / p75 | −0.256 / −0.004 | −0.219 / −0.009 | −0.207 / −0.004 | −0.286 / −0.005 |
| min / max | −0.637 / −1e-4 | −0.595 / −1e-4 | −0.595 / −1e-4 | −0.637 / −1e-4 |

92/92 margins are negative (this subset is, by construction, where a wrong
molecule outscored the correct one; median gap **−0.063**, i.e. the decoy beats
the true hit by ~6 sim points). For the full 397/EXP-004 result, margins are
only obtainable from a future run that persists per-query top-k (molecule,
score) — this should be added to EXP-004's checkpoint format.

---

## 9. Collision-energy audit / Scenario A-B reproducibility (Q6)

- `collision_energy_ev` is a `DOUBLE[]` array column: 2,202,165 / 2,539,608
  rows non-null (**86.7% present; 13.3% NULL**); values like `[40.0]`, `[60.0]`,
  `[20.0, 40.0, 60.0]`.
- **The A=15 / B=21 split is NOT reproducible from artifacts.**
  - `exp004_novelty_audit.json` stores only a boolean
    `any_different_ce_retained` (n=263) — there is no per-group CE, no "closest
    retained group" distance, and no record of which CE rule produced 15/21.
  - The rule as written ("closest retained group shares a CE value") is not
    implementable from saved data and, on the *corrected* all-same-adduct set
    (149, not 36), any reasonable implementation gives a very different split
    (my "query CE present among same-adduct retained rows" rule → A=2, B=147).
- **Recommendation (Q6):** collapse Scenario A and B into a single
  "all retained evidence is same-adduct" stratum (**n = 149**, corrected) for
  EXP-004 reporting, with CE variation reported descriptively (e.g. share of
  retained rows sharing a CE value) rather than as a standalone experimental
  condition — matching the design's own small-N warning but removing dependence
  on an unreproducible split. Only if a leading wanted a *new* A/B split should
  the exact CE rule be defined, applied to the corrected census, and the audit
  re-run (results/exp004_scenario_audit_v2.json already carries
  `same_adduct_ce_match` per query to make that cheap).

---

## 10. Remaining uncertainties

1. **Similarity values in v1** (`max/median/min_sim_direct/modcos`) appear sound
   (computed over genuinely-retained spectra) but were produced by the same
   code family that mis-assigned adduct flags. The 3 near-dups and the survivor
   table are individually verified; the full per-query sim distribution has not
   been byte-reproduced. If EXP-004's novelty filter must be exact, re-derive
   the sims using the corrected retained-set logic.
2. **The v1 join bug's full footprint** is bounded by the flag disagreement set
   (112 rids, listed in `exp004_independent_verify_out.json`); no v1 output
   other than adduct/CE flags appears affected, but any downstream consumer of
   v1's flags (design §2/§6 narrative, conflict-Analysis in `CASMI_RESEARCH_STATE`)
   must be re-surveyed after the flags are swapped to the corrected values.
3. **The 8/12 ledger rids v1 mislabeled** as having same-adduct evidence are the
   highest-risk class for downstream use; each is individually confirmed here as
   cross-adduct-only.
4. EXP-003's 92 inspections are outcome-selected; margin and displacement
   percentages must not be read as population-wide rates for the 397.
5. `<5% sim` under same-adduct comparisons (design §4) may partly reflect
   collision-energy variation; the split of that effect between CE and np.lib's
   duplicate-label phenomena is not quantified.

---

## 11. Recommendation

1. **Apply the corrected census before the smoke test** (owned by Claude Code
   per the handoff): swap v1 flags for the corrected row-level flags in design
   §2/§4/§6 and `CASMI_RESEARCH_STATE.md`; the headline fact changes from "108
   queries with only cross-adduct evidence" to "**12**, identical to the 12
   Variant-A candidate-absent queries". The design's motivation ("cross-adduct
   exposure is large and dominates") must be restated accordingly — exposure
   exists for 251/400 (63%), but only 12 queries have **no** same-adduct
   evidence at all.
2. **Keep the 0.90-Modified-Cosine novelty filter** (the 3-exclusion result is
   robust) but fix its provenance claim (EXP-001's own threshold is 0.95 direct
   cosine; 0.90 is a new data-driven pick, stable 0.85–0.95).
3. **Drop the A/B experimental split**; report one "all-same-adduct" stratum
   (n=149 corrected). Do not block on a reproducible CE definition.
4. **Keep EXP-001's Variant A as EXP-004's candidate generation** (design §11),
   unchanged — confirmed the correct choice: Variant A reaches 388/400 and its
   12 failures are exactly the structurally-unreachable cross-adduct-only set.
5. **Add per-query top-k (molecule, score) persistence** to EXP-004's checkpoint
   format so margins and near-miss families are computable after the run (Q5).
6. EXP-004's paired comparison (397 rids vs EXP-001) and its strict exclusion
   of the query's whole metadata group are valid and unchanged; no leakage
   finding beyond the flag bug.

**Verdict for the audit question asked:** the four science decisions EXP-004
rests on (population, novelty filter, Variant-A reuse, paired evaluation) are
verified-consistent after the census correction; the one decision that must be
re-made against corrected numbers is the **Scenario A/B/C stratification and the
cross-adduct-exposure narrative**.
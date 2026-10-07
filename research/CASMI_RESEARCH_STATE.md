# CASMI 2026 — MASTER RESEARCH STATE

Last consolidated: 2026-09-22

## Purpose
Persistent handoff document for continuing the Enveda CASMI 2026 research across ChatGPT chats and coding agents. Read this before proposing experiments, changing direction, or interpreting results.

## 1. Research philosophy

The goal is not merely a good public leaderboard score. The goal is to understand the competition deeply enough that model choices are evidence-driven.

Golden rule:
> Do not ask first "What model should we use?" Ask "What must be true for a model to win this competition, and what experiment can establish that?"

Every major experiment needs:
1. precise research question;
2. hypothesis;
3. leakage controls;
4. measurable baseline;
5. defined intervention;
6. MRR@25 and R@1/5/10/25 where applicable;
7. failure analysis;
8. evidence-based decision.

Always distinguish:
- FACT = measured/documented.
- DECISION = deliberate choice based on evidence.
- HYPOTHESIS = plausible but unproven.
- IDEA = possible future direction.
- ILLUSTRATIVE = example only, not measured.

Never turn a hypothesis into a fact.

## 2. Competition

Enveda CASMI 2026 — Molecule ID From Mass Spectra.

Task: Given LC-MS/MS spectra, predict natural-product structures / SMILES.

Submission: Up to 25 ranked SMILES per molecule_id.

Evaluation: MRR@25. Approximate contribution:
- rank 1 = 1.00
- rank 2 = 0.50
- rank 3 = 0.333
- rank 5 = 0.20
- rank 10 = 0.10
- rank 25 = 0.04
- absent from top 25 = 0

Matching canonicalizes predicted/answer SMILES through RDKit tautomer canonicalization and uses InChIKey14. Connectivity matters; stereochemistry/tautomer representation does not determine the match.

MRR@25 is not ordinary accuracy.

## 3. Dataset / technical facts

Train:
- ~2.54M spectra / 2,539,608 rows.
- ~275.8k molecules.
- train.parquet ~2.82 GB.
- test.parquet ~4.6 MB, 1,213 rows.
- Test is 100% timsTOF.
- enveda-180 is the corresponding timsTOF train library and ~97% disjoint by molecule from other libraries.

Libraries:
- enveda-180: 1,153,785
- pluskal_ms2: 527,581
- riken: 347,171
- gnps: 220,849
- massbank: 101,727
- mona: 92,416
- spectraverse: 50,933
- msdial: 40,765
- drug_plus: 2,545
- enveda-np-examples: 1,184
- masaryk: 652

Important train fields:
ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks, base_peak_intensity, collision_energy_ev, collision_energy_orig, collision_energy_orig_units.

Test has spectral fields plus molecule_id/spectrum_id, but structure identity is hidden.

Peak statistics:
- train median ~42 peaks overall.
- timsTOF train median ~138.
- test median ~230.
- intensities approximately [6.6e-8, 1].

Data-loading issue:
pandas/pyarrow 19 fails with `OSError: Repetition level histogram size mismatch`.
Use src/data/loader.py / DuckDB.

## 4. Class framework

Conceptual regimes:
- Class 1: molecule / relevant spectral evidence represented in library; spectral-library retrieval.
- Class 2: molecule known/represented, but query spectrum is novel/different; spectral generalization.
- Class 3: target molecule absent from training/relevant library; formula-constrained retrieval, analogue inference, learned embeddings, de novo/fingerprint approaches.

Critical refinement:
Pseudo-Class-3 should NOT be designed merely by taking Class-1 data and deleting the target molecule. It should be designed after understanding both C1 and C2.

Clean mental model:
- C1 = known molecule + known/retrievable spectral evidence.
- C2 = known molecule + unseen spectrum.
- C3 = unseen molecule + unseen spectrum.

Pseudo-C3 should incorporate lessons from C1+C2 and can have chemical-neighbourhood tiers:
- Easy: target absent, close analogues present.
- Medium: target absent, family represented.
- Hard: target absent, close chemical neighbourhood restricted.

These tiers are design ideas, not executed.

## 5. EXP-001 — Class-1 baseline

Purpose: leakage-safe library-search ceiling / controlled Class-1 retrieval.

Mode A:
- near-exact duplicate ceiling test;
- peak-level validation direct cosine >=0.95;
- exact/near-identical sibling intentionally allowed.

Mode B:
- realistic same-molecule/different-spectrum test;
- query's whole metadata group `(inchikey14, adduct, precursor_mz, num_peaks)` excluded;
- valid same-molecule candidates can come from different groups.

Config:
- Mode A n=200.
- Mode B n=400.
- multi-spectrum add-on n=150x2.
- precursor tolerance ±0.01 Da.
- matchms peak tolerance 0.1 Da.
- mz_power=0.0.
- intensity_power=1.0.
- seed=42.
- candidate generation -> matchms ModifiedCosineGreedy -> molecule-level max-score aggregation -> top 25.

Authoritative results:

Mode A / enveda-180:
- cand recall 1.000
- R@1 .565
- R@5 .795
- R@10 .855
- R@25 .960
- MRR .665

Mode A / all-train:
- cand recall 1.000
- R@1 .560
- R@5 .795
- R@10 .855
- R@25 .960
- MRR .663

Mode B / enveda-180:
- cand recall .965
- R@1 .530
- R@5 .803
- R@10 .875
- R@25 .915
- MRR .650

Mode B / all-train:
- cand recall .970
- R@1 .535
- R@5 .805
- R@10 .8725
- R@25 .915
- MRR .6510

Mode B decomposition:
- 366/400 ranked 1-25.
- 22 below 25.
- 11 absent from candidate pool.
- 1 zero-candidate failure.

Important interpretation:
R@25 91.5% means the correct molecule is often present. MRR .651 is mainly a ranking-quality measurement, not 65% accuracy.

## 6. Modified Cosine facts

matchms 0.33.1 source copied to research/_matchms_src_ref/.

- Modified Cosine = direct peak matches + shifted matches based on precursor mass difference / neutral loss.
- tolerance 0.1 Da.
- mz_power 0.0.
- intensity_power 1.0.
- mass_shift = precursor_mz_ref - precursor_mz_query.
- greedy assignment sorts by intensity-product contribution.
- Hungarian uses global assignment.
- score = matched weight-product sum / product of norms.

Critical quirk:
If abs(mass_shift) <= tolerance, Modified Cosine falls back to ordinary Cosine.

Therefore EXP-001's ±0.01 Da precursor candidate filter made Modified Cosine effectively direct Cosine for those candidates.

A real cross-adduct diagnostic ([M+H]+ vs [M+Na]+, ~22 Da apart) showed shifted matching works: 9 matched peaks vs direct Cosine 6.

In EXP-002, shifted matching contributed for 9/12 recovered cases.

## 7. EXP-002 — adduct-aware candidate generation

Question:
Can chemically correct neutral-mass candidate generation recover true molecules missed by raw precursor-m/z filtering?

Variants:
- A = raw precursor m/z ±0.01 Da baseline.
- B = adduct-aware neutral-mass conversion using 13-adduct table including multimers, tolerance 0.01 Da; unsupported adducts fall back to A.
- C = adduct-pair-curated deferred.
- D = brute-force ±50 Da diagnostic only.

A chemistry sign bug for [M+Cl]- and [M+CH2O2-H]- was caught and fixed; these are net additions, unlike [M-H]-.

Smoke:
- 25 Mode-B queries.
- candidate pools grew ~1.86x average (778 -> 1447).
- no pathological explosion; multimer [2M+Na]+ could grow ~13x.

Full results:
- candidate recall 1.000 vs .970.
- R@1 .490 vs .535.
- R@5 .780 vs .805.
- R@10 .855 vs .8725.
- R@25 .920 vs .915.
- MRR .6178 vs .6510.

Other findings:
- 398/400 queries had some Variant-A candidates excluded by B.
- This is not a bug: raw-m/z can admit chemically incorrect candidates with coincidentally similar reported m/z.
- Variant B removes these and adds chemically valid cross-adduct candidates.
- Median candidate count ~690 -> ~1224.
- For already-successful queries pool grew ~2.74x.
- Max ~5,020 monomer / ~4,763 multimer.
- 50,000 cap never triggered.
- Of 388 queries where correct candidate was already present under A: 84 worsened, 25 improved, 30 lost rank 1, 8 fell out of top25.
- 12 previously absent cases recovered; 3 still failed top25 (rids 733288, 410447, 318845).
- 8 new top25 successes were essentially cancelled by 8 previous successes falling out.
- Shifted Modified Cosine contributed for 9/12 recovered cases.

Decision DEC-004:
Do NOT adopt Variant B as-is with current max-Modified-Cosine ranker.
Current research/production Class-1 baseline remains EXP-001 Variant A.
Do not change candidate generation until ranking-side causes are better understood.

## 8. EXP-003 — ranking analysis/design

Targeted 92 problematic queries:
- 84 worsened.
- 8 newly failed.

Initial implementation rescanned full file 92x and was killed; corrected to one bulk fetch and rerun ~110 sec.

Findings:
- 90/92 (98%) wrong-outcome queries have correct answer at same adduct as query.
- cross-adduct displacer: 54/92 (59%).
- same-adduct displacer: 38/92 (41%).
- displacer newly admitted by B: 57/92 (62%).
- already present under A: 35/92 (38%).

Outcome-selected score medians:
- correct same-adduct .791 (p25 .590)
- wrong same-adduct .929
- wrong cross-adduct .939

This argues against cross-adduct score calibration being the sole cause because same-adduct wrong candidates also score highly.

Caveat:
92 cases are failure-selected; they motivate hypotheses but do not replace full-population evidence.

Design:
A = Variant-A baseline.
B = full Variant-B.
C = Variant-B minus cross-adduct candidates.
D = Variant-A plus legitimate newly-recoverable cross-adduct candidates.
E = pool-size-matched control adding same-adduct filler candidates to A.

Metrics:
candidate recall, R@1/5/10/25, MRR, candidate counts, runtime, RSS, query-level categories.

Hypotheses:
H1 pool-size effect.
H2 cross-adduct score calibration.
H3 removal of Variant-A accidental false positives.
H4 multiple effects.

Smoke test PASSED:
- 25-query fixed seed 20260921.
- full-400 A/B reconstruction matched EXP-001/002.
- rank_A and rank_B matched prior results.
- best_true_score_b identical.
- per-candidate scores byte-identical across B/C/D.
- leakage zero A/B/C/D/E.
- D set matched EXP-002 shift analysis on 14 sampled queries.
- all 12 recovered category-2 cases validated at set level.
- E matched |B| where possible.
- 45/400 have |B| < |A|, so E=A there.
- cap 50,000; max sampled B 4,123.
- smoke ~16 min.

Design limitations:
1. Existing EXP-002 checkpoint persisted only true-molecule newly admitted rids, so current D adds only true-molecule candidates. It cannot demonstrate wrong-candidate harm under max aggregation. A real wrong-candidate D arm requires persisting all newly admitted candidates.
2. B does not always contain A: 45/400 have |B| < |A|, so E cannot always size-match by adding.
3. For n=2 multimers, B neutral-mass window corresponds approximately to ±0.02 Da precursor, so C can exceed A∩same-adduct.

At last state, full 400-query EXP-003 was not completed.

## 9. EXP-004 — Class-1 -> pseudo-Class-2

Strategy:
Use known-label training data to construct controlled pseudo-C2/C3 benchmarks before moving to real C2/C3.

Golden rule:
Hide the answer, not the evidence.

Model must not see target normalized_smiles, inchikey, inchikey14, or molecule identity. It may see metadata genuinely available at competition test time, such as precursor m/z, adduct, ionization mode, instrument, collision energy, and formula only if formula is actually available at test time.

Pseudo-C2:
Known molecule, hidden/unseen spectrum.

Example:
- X-A/B/C available.
- X-D hidden.
- model must infer X without being given identity.

Novelty must be explicitly defined. Different metadata does not automatically mean spectral novelty.

Potential strata:
A = same instrument/adduct/CE but genuinely different spectrum.
B = same adduct/different CE.
C = cross-adduct.
Different instrument would be useful, but current 400-query audit is 100% timsTOF, so it is unsupported in this population.

## 10. EXP-004 Stage-1 audit — latest

Claude Code performed design/data audit only.

It reused exact EXP-001 400-query Mode-B population.

It initially used direct Cosine for novelty, then caught that cross-adduct evidence makes direct Cosine inappropriate and reran using Modified Cosine.

Reported:
- direct-Cosine novelty median ~1.4% was misleading due to cross-adduct evidence.
- Modified-Cosine novelty median ~5.5%.
- reused 0.90 near-duplicate cutoff.
- only 3/400 excluded as near duplicates.
- instrument type 100% timsTOF.
- same adduct/same CE: 15.
- same adduct/different CE: 21.
- cross-adduct: 364.
- A/B are too small for standalone conclusions; proposed as strata within 397 retained queries.

Design file:
`research/05_class1_to_class2_design.md`

Verdict (original):
READY FOR SMOKE TEST, with precondition:
Use EXP-001 unchanged Variant-A candidate generation, NOT EXP-002 Variant B, because EXP-003 has not resolved why B degrades ranking.

Critical caution:
Do not accept the causal statement "median Modified Cosine 0.055 means different collision energies cause genuinely different fragmentation" as proven. The score is observed; the causal interpretation is not established.

**CORRECTED 2026-09-21 (see `research/decision_log.md` DEC-005, `research/06_matched_c1_control.md`):** the original "108 queries have only cross-adduct evidence" claim was a bug (found by direct source-data verification and confirmed via EXP-001's own candidate-absence ground truth). Corrected count: **12**, exactly matching EXP-001's 12 Variant-A candidate-absent queries. The claim that 0.90 "was inherited from EXP-001" was also wrong — EXP-001's actual validated threshold was 0.95, on a different quantity (direct Cosine, same-metadata-group); 0.90 now has an honest, self-contained justification (99.25th percentile of this population's own similarity distribution). The CE-based Scenario A/B split (15/21) was built on the same bug and is merged into one "same-adduct evidence retained" category (388/400); CE-match rate (0.8%, only 3/388) is reported as descriptive/exploratory only, not a hard split. A required cheap matched control (EXP-005) also ran: row-only exclusion vs whole-metadata-group exclusion produce bit-identical Recall/MRR (only 8/400 queries have any group sibling to exclude at all), ruling out that specific leakage mechanism as a confound in EXP-001 Mode B / EXP-004's construction. **The EXP-004 C2 smoke test itself has still NOT been run and is NOT approved by this correction pass** — that remains a separate decision.

## 11. Important question: is EXP-001 Mode B already pseudo-C2?

It is arguably a primitive pseudo-C2-like benchmark because:
- it excludes the query's whole metadata group;
- it searches for the same molecule through other spectra/groups.

But it is not automatically a clean C2 benchmark because:
- it does not explicitly require a scientifically meaningful novelty level;
- retained evidence may still be very similar;
- cross-adduct evidence dominates the current audit;
- multiple difficulty sources are mixed.

Therefore:
Use EXP-001 Mode B as important evidence/baseline, but do not blindly relabel it as the final pseudo-C2 benchmark.

## 12. Pseudo-C3

Do NOT define pseudo-C3 purely from C1.

Design it after understanding C1 + C2.

For target X:
C2:
- X-A/B/C available.
- X-D hidden.

C3:
- all X spectra/evidence removed.
- target identity hidden.
- related molecules can remain depending on difficulty tier.

C3 should model:
unseen molecule + unseen spectrum.

Potential tiers:
- Easy: target absent, close analogues present.
- Medium: target absent, family represented.
- Hard: target absent, close chemical neighbourhood restricted.

Do not implement until C2 benchmark/results are understood.

## 13. Broader strategic research map

Do not rush from C1 -> C2 -> C3 into one model.

Phase 0: competition understanding.
Phase 1: Class-1 forensic analysis.
Phase 2: clean pseudo-C2.
Phase 3: pseudo-C3.
Phase 4: bottleneck experiments.
Phase 5: model families.
Phase 6: ensemble + leaderboard validation.

Potential bottleneck experiments:

1. Candidate-generation ceiling.
2. Oracle candidate pool.
3. Formula constraints.
4. Adduct knowledge.
5. Learned spectral embeddings.
6. Spectral similarity vs structural similarity / analogue retrieval.
7. External data coverage and downstream impact.
8. Fragment / neutral-loss interpretation.
9. Spectrum-domain transformation / augmentation.
10. Ensemble complementarity.

Potential modeling paths are hypotheses, not commitments:
- improved spectral similarity / learned reranker;
- candidate generation + ranking co-design;
- formula-first retrieval;
- fragment interpretation;
- molecular analogue propagation;
- spectrum-domain transformation;
- external spectral/chemical data;
- ensemble of retrieval + learned + structure/formula reasoning.

## 14. Oracle experiments

Use oracle experiments to locate bottlenecks before expensive modeling.

Oracle candidate pool:
Guarantee correct molecule is in candidate pool. If performance becomes high, candidate generation matters; if not, ranking/representation matters.

Other possible controlled oracles:
- true formula.
- true adduct.
- structural family where scientifically valid.

Avoid unrealistic leakage.

## 15. Leaderboard interpretation

Current public leaderboard has been observed around ~0.33-0.40 MRR depending on snapshot/time.

Do not directly compare this to EXP-001 .65 as if they are the same task.

EXP-001 is controlled training-data evaluation. Real hidden test mixes difficulty regimes and the public leaderboard can be partial and changing.

The gap is a clue, not proof of how much C2/C3 contributes.

## 16. Do not do these yet

- Do not blindly adopt EXP-002 Variant B.
- Do not treat 0.90 Modified-Cosine as universal C2 novelty.
- Do not claim low Modified-Cosine proves collision energy caused spectral differences.
- Do not call all EXP-001 Mode-B cases clean Class 2.
- Do not create pseudo-C3 merely by deleting target molecules from C1.
- Do not choose a neural architecture because public Kaggle code uses it.
- Do not optimize leaderboard before understanding failure modes.
- Do not assume external data automatically helps.
- Do not conflate candidate recall with ranking quality.
- Do not treat outcome-selected failure subsets as unbiased full-population evidence.
- Do not change the production pipeline during exploratory experiments without a recorded decision.

## 17. Current status and immediate next step

Established:
- Class-1 baseline measured.
- EXP-001 authoritative.
- EXP-002 completed and rejected as drop-in candidate-generation replacement.
- EXP-003 targeted analysis and smoke test completed; full run not yet completed.
- Pseudo-C2 concept established.
- EXP-004 design adversarially reviewed 2026-09-21: found and fixed a real bug (the
  "108 cross-adduct-only" Scenario count; corrected to 12, exactly matching EXP-001's
  known candidate-absent queries) and a false justification (the 0.90 threshold was
  claimed to be inherited from EXP-001; it wasn't — corrected to a self-contained
  percentile-based justification). CE-based Scenario A/B split merged (too thin, same
  underlying bug, no reproducible tolerance rule justified by the data).
- EXP-005 (matched Class-1 row-exclusion control) run and complete: row-only exclusion
  vs whole-metadata-group exclusion produce bit-identical Recall/MRR on EXP-001's
  400-query Mode-B population, because only 8/400 queries have any metadata-group
  sibling to exclude at all — rules out that specific leakage mechanism as a confound.
  See `research/06_matched_c1_control.md`, `research/decision_log.md` DEC-005.
- EXP-004's C2 smoke test (n=28, seed=20260921, stratified) ran and passed all
  validation checks (0 leakage failures, 0 reproducibility mismatches vs EXP-001, 0
  reachability-classification surprises); a separate artifact-only margin sanity check
  on the persisted top-50 candidates found 0 anomalies. Approved in principle by the
  user after review.
- EXP-004's full 397-query run is COMPLETE. `research/scripts/exp004_full_run.py`,
  `results/exp004_full_run.json`. Headline: MRR@25 0.6519 overall (n=397), 0.6722
  reachable-only (n=385), closely tracking EXP-001's original Mode-B result (0.6510).
  The 12 cross-adduct-only queries are a clean 0% candidate-generation ceiling under
  Variant A (not a ranking failure), reconfirmed at full scale. Margin analysis (n=378)
  replicates EXP-005's Class-1 baseline shape almost exactly: rank-1 median +0.128,
  correct-not-rank-1 median -0.073, correct-outside-top-25 median -0.280 (vs EXP-005's
  +0.127 / -0.072 / -0.391). Full breakdown (A-G) in `research/experiments.md` EXP-004
  entry. Reported for scientific review; per explicit instruction, no Class-3 work
  launched and no further interpretation/decision made yet.

- EXP-004 C2 failure analysis (2026-09-22) is COMPLETE:
  `research/analysis/exp004_c2_failure_analysis.md`, `research/scripts/exp004_c2_analysis.py`,
  `results/exp004_c2_analysis/`. A full query-level join of C1 (EXP-001/EXP-005) vs C2
  (EXP-004) on the shared 397-query population found **zero differences on every
  measured quantity** — identical rank, n_candidates, and true-molecule score for every
  query, hence zero rank transitions, zero regressions, zero improvements, and 100%
  (34/34) of failures shared between conditions. This is a direct, documented
  consequence of EXP-004's own construction (it explicitly reuses EXP-001 Mode B's exact
  exclusion/candidate-generation/scorer on 397 of the same 400 queries, per
  `research/05_class1_to_class2_design.md` §7 and `exp004_full_run.py`'s docstring) — it
  is EXP-001's pipeline re-run on a near-duplicate-filtered subset, not a distinct
  computation. The master prompt's headline "MRR delta ≈ +0.0009" was verified to be
  fully explained by the 400→397 denominator change (the 3 dropped near-dup queries'
  own reciprocal ranks average below the population MRR), not by any experimental
  effect. See DEC-006. **Do not cite EXP-004's near-equal aggregate MRR as evidence that
  spectral novelty is harmless — no distinct C2 condition was actually tested.** A
  secondary novelty-stratified analysis found no meaningful rank-vs-similarity
  correlation (Spearman ≈ −0.02) and a non-monotonic pattern across similarity strata,
reported as a descriptive finding on this shared population, not evidence about a real
   novelty manipulation.

- EXP-007 pseudo-C2 redesign (2026-09-22) DESIGNED (not run):
  `research/analysis/exp007_c2_design.md`, `results/exp007_c2_design/design_spec.json`,
  proposed `PROPOSED DEC-007` in `research/decision_log.md`. Dataset audit executed
  across four scripts:
  - Phase 1 census (`results/exp007_audit_phase1.json`): full train 2,539,608
    rows / 275,810 molecules; timsTOF 1,154,969 rows / 183,191 molecules; timsTOF
    181,651 molecules with >=2 same-adduct spectra; 181,650 with same-adduct
    different-CE; only 193 with same-adduct >=2 *nominal* spectra (i.e. within-adduct
    multi-spectra differ essentially only by CE).
  - Phase 2 pairwise ModifiedCosine (`results/exp007_audit_phase2.json`): same-adduct
    diff-CE pairs median 0.79 (p25 0.50 — genuine but mild novelty), same-adduct
    same-CE pairs median 0.999 (near-identical), cross-adduct pairs median 0.013
    (unreachable-in-score, p90 0.79 tail); 281/300 sampled molecules have >=1 same-
    adduct pair < 0.80.
  - Phase 3 query-centric (`results/exp007_audit_phase3.json`): of 600 seeded timsTOF
    queries, 490/600 (81.7%) have >=1 retained same-adduct sibling; 110/600 have none
    (cross-adduct-only ceiling, mirror of EXP-004's 12/400); retained-sibling counts
    1/2/3/4 -> 90/181/218/1.
  - Phase 4 design sizing (`results/exp007_design_sizing.json`): strict C2 eligibility
    (ALL retained same-adduct evidence below tau) — tau=0.50: 33/600 (5.5%),
    tau=0.70: 120/600 (20.0%), tau=0.90: 253/600 (42.2%), tau=0.95: 313/600 (52.2%);
    max-sim distribution smooth (p10 0.54, p25 0.70, median 0.89) — no knife-edge
    threshold, justifying continuous stratification.
  Construction (Design A recommended; B subsumed; C rejected as C2): C1 = Mode B
  control; C2(tau) = C1 library minus every retained same-adduct sibling with
  ModifiedCosine(query,sibling) >= tau, tau in {0.50,0.60,0.70,0.80,0.90,0.95};
  C1-matched(tau) = pool-size control removing the same number k of wrong-molecule
  spectra (isolates novelty from EXP-003's pool-size effect). Verdict in design doc:
  READY FOR REVIEW; smoke (n=30, seed 20260922) and full run (600 queries) NOT yet
  approved.

Immediate next:
1. Resolve EXP-004's methodology gap before designing C3: DEC-006 options are
   (a) genuine novelty intervention — now satisfied by EXP-007's design — or
   (b) explicitly adopt Mode B as operational pseudo-C2. **Pending user decision on
   whether to approve EXP-007 (PROPOSED DEC-007); if approved, smoke test runs first,
   then the full run.**
2. Decide what (if anything) EXP-004's population-level observation implies for C2 more
   broadly — e.g. whether same+cross-adduct evidence (E, MRR 0.6939) genuinely
   outperforming same-adduct-only evidence (D, MRR 0.6379) deserves a dedicated,
   controlled follow-up (still only an observational correlation, not a controlled
   ablation — flagged, not concluded).
3. Only after a genuine C2 comparison exists (EXP-007 full run, or an explicit
   Mode-B-as-pseudo-C2 adoption): design C3 using BOTH C1 and C2 knowledge.
4. Only after C1/C2/C3 understanding should major architecture selection happen.

## 18. New-chat handoff instruction

At the start of a new CASMI chat, say:

"Read `research/CASMI_RESEARCH_STATE.md` first. Treat it as the persistent state of the CASMI 2026 research. Do not restart the reasoning from scratch. Distinguish established facts, decisions, hypotheses, and ideas. Before proposing a new experiment, check whether an existing experiment or decision already answers the question. Preserve leakage controls and comparability with EXP-001 unless there is an explicit scientific reason to change them."

For coding agents:
"Read `research/CASMI_RESEARCH_STATE.md` plus the specific experiment/design/result files relevant to the task. Do not make pipeline changes until the research question and design are validated. Report evidence, uncertainty, and decision implications separately."

## 19. One-sentence state

We have a strong leakage-safe Class-1 library-search baseline (~0.651 MRR / 0.915 R@25 in Mode B), learned that simply expanding adduct-aware candidates can hurt ranking, corrected two verified errors in the pseudo-Class-2 design doc (a 108->12 reachability bug, a false 0.90-threshold provenance claim), ran the required matched Class-1 row-exclusion control (EXP-005, clean result, no leakage confound found), completed EXP-004's full 397-query pseudo-Class-2 run, and then ran a rigorous query-level C1-vs-C2 failure analysis (2026-09-22) that discovered EXP-004, as constructed, is literally the same computation as EXP-001 Mode B on 397 of the same 400 queries (zero per-query differences in rank/candidates/scores, zero regressions/improvements, 100% shared failures) — so its near-equal aggregate MRR must NOT be read as evidence that spectral novelty is harmless (DEC-006); a corrected pseudo-C2 experiment (EXP-007) has now been designed with a genuine, executed spectral-novelty intervention (Design A spectrum-level holdout, C1 vs C2(tau) vs C1-matched pool-size control), backed by a four-phase dataset audit showing the population is drillable (81.7% of sampled timsTOF queries retain same-adduct evidence; smooth novelty axis, strict C2 eligibility 20% at tau=0.70, 52% at tau=0.95) — DESIGN READY FOR REVIEW (PROPOSED DEC-007), smoke/full run not yet approved, no pipeline change, C3 still blocked pending the genuine C2 comparison (either EXP-007 if approved, or an explicit Mode-B-as-pseudo-C2 adoption).

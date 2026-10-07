# EXP-015 → Next-Experiment Proposal (no execution)

Scope: frozen proposal only. **No EXP-016 run is launched here.** The exact
design decision waits for the EXP-014 result; this document states what EXP-015
actually established, what remains unknown, and what a minimal, discriminative
EXP-016 should measure.

---

## A. What EXP-015 established (frozen)

1. Isomer-resolution quality is *factorizable and small*: single-level changes
   move MRR by −0.08…+0.05 on a fresh 954-target holdout (838 ranked), with
   tight paired CIs at n=838.
2. **Bin width 0.01 Da is a robust win**: +0.047 MRR / +0.057 R@1, iso-error
   −2.1 pp (CI excludes 0); log-intensity adds +0.031; coarser binning (0.5 Da)
   and linear intensity are significantly worse.
3. ModCos rerank (kNN prefilter k=20) is a genuine +0.035 MRR, orthogonal to
   binning (A+B combine to ~+0.08 in C2).
4. **Neutral-loss features are load-bearing for isomer resolution** while nearly
   neutral for overall MRR: removing them was the DEV winner (+0.028) yet on
   PRIMARY the MRR gain does not transfer and iso-error worsens +1.6 pp (CI
   0.001–0.030). EXP-013's 243 targets overfit NL absence.
5. The EXP-013 two-tower is weaker than the kNN alone (ΔMRR −0.026), but its
   correct-set overlaps only ~partially: fusion gains are plausible, not proven
   (CIs straddle zero). Observed fusion iso-error 0.199→0.151 is a hypothesis.
6. Oracle ceiling: 94% R@1 achievable *if* the exact class fingerprint were
   recovered; the error budget is in spectral-feature retrieval, not pool
   geometry (coverage 87.8%, median pool 28.5).

## B. What remains uncertain

- Whether the bin/log gains replicate on *other* pool geometries and on a
  genuinely novel target draw (this holdout is the first fresh one; single draw).
- Whether NL features matter *enough* to pay for bigger feature grams, and
  whether the NL→iso-error direction holds under a different fragment-ion model.
- Whether Fusion-50/50 gains (or iso-error drop) are real: CIs cross zero and
  the two-tower itself is unvalidated on this population (frozen EXP-013
  weights, embeddings re-derived but tower never retrained post-corpus change).
- Magnitudes of the by-category (distant/near/unrelated) deltas have no CIs.
- Full-coverage behavior: 116/954 uncovered targets are outside the mass-window
  coverage and contribute nothing to the above MRRs.

## C. What EXP-014 must establish before EXP-016 is chosen

1. The frozen two-tower's *per-sample reproducibility framework* (does its
   embedding drift on this corpus? what is its per-sample variance?) — gates any
   fusion arm.
2. Whether EXP-014 produces a second independent typed holdout draw (same R
   filter, fresh seed) usable as PRIMARY-2 / final-eval — otherwise the only
   untouched evaluation target is the 2,513−1,000 remainder ≈1,513 eligible
   molecules (lower coverage projection possible).
3. A decision on *which tenant of evaluation is contractually fixed*: MRR@all vs
   R@1 vs isomer-error — EXP-015 shows these can score in different directions,
   so EXP-016 arms must be pre-registered on ONE primary metric (suggest R@1 as
   primary, MRR and iso-error as secondary, matching CASMI's target-pool
   ranking).

## D. Candidate EXP-016 arcs (pick ≤2; all pre-registered on the fixed population)

- **X-spectrum-resolution sweep**: bin ∈ {0.005, 0.01, 0.02, 0.05} Da × inten ∈
  {sqrt, log} on the SAME ranked set. Q: is 0.01 Da an interior optimum in MRR
  and in iso-error? Statistically stronger than the single A/B that EXP-015 ran,
  and cheap (one feature recompute + one run).
- **X-isomer-budget audit**: restrict to the 746 same-formula pools; report
  iso-error with paired CI for {B0, 0.01Da, ModCos} and *require* iso-error CI
  to decide the NL question made in §A4. Q: which representation actually
  resolves near-isomers (ratio near/distant by category)?
- **X-complementarity test for fusion**: pre-commit a non-inferiority test
  (ΔMRR ≥ −0.010 must be excluded; then decide superiority) for
  fusion_50_50 vs kNN-B0 on the same ranked set, now with EXP-014-level
  reproducibility. Q: is the observed +0.0155 MRR real?
- **X-fragmentation-feature contribution**: kNN with fragment-only features vs
  separate NL-only block on the iso-error metric (isolate fragments vs NL vs
  joint). Q: which feature family carries the isomer signal found in §A4?
- **X-cost window**: rank-first prefilter k_pre ∈ {200, 500, 1000} at k=20 —
  isolates how much of the ModCos gain comes from the deeper prefilter vs the
  rerank itself.

Each arm keeps the frozen R, evaluation code, and metric definitions; the only
parameter touched is the one under test. No arm touches EXP-013/014.

## E. Why a fresh evaluation population is needed

- This corpus is now *partially consumed*: EXP-015 scored 10 variants on PRIMARY
  (DEV-selection already reused the EXP-013 243 targets). Any further
  within-PRIMARY tuning in EXP-016 inherits selection entanglement — §A4 (V2)
  is the concrete failure mode.
- Paired-bootstrap CIs weight each query once; they do not capture the *draw-to-
  draw* variance of the target sample. A second independent draw (EXP-014 or a
  fresh seed on the remaining ~1,513 eligible) is the only way to report
  selection-corrected expected gains.
- All EXP-016 arms above are pre-registered so they can legally run on the same
  ranked set; anything exploratory must move to the untouched remainder.

---

**Next action**: receive EXP-014, then choose at most two arcs from §D and
pre-register metrics + population in `research/analysis/exp016_preanalysis.md`
before any run.
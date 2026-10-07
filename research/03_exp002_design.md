# 03 — EXP-002 Design: Adduct-Aware Candidate Generation

Status: **DESIGN ONLY. Not executed at full scale.** This document proposes EXP-002 and
reports the required pre-design evidence (failure-case inspection, a manual-verification
correction, and a bounded smoke test of the proposed rule). No full run, no decision-log
entry, and no claim of success has been made. Per instruction, this is evidence-gathering,
not a result.

---

## 1. Research question

> How much Class-1 retrieval recall is lost because candidate generation compares reported
> precursor m/z directly instead of accounting for adducts / neutral molecular mass?

## 2. Evidence from EXP-001 motivating it

`research/02_class1_coverage.md` found that in the primary Mode B / all-train condition
(n=400), 11 queries (2.75%) failed because the true molecule had **zero** candidates in
the ±0.01 Da precursor window, despite the molecule having other spectra elsewhere in
train. A separate, standalone diagnostic in EXP-001 confirmed matchms's
`ModifiedCosineGreedy` correctly activates its shifted-peak-matching logic on a genuine
cross-adduct pair ([M+H]+ vs [M+Na]+, 22 Da apart) — the scorer works; the candidate
generator simply never gives it a chance, because the 0.01 Da precursor filter
structurally excludes any candidate whose adduct differs enough to shift the observed
precursor m/z by more than that.

## 3. Failure-case analysis

### 3a. The 11 candidate-generation-absence failures (Mode B / all-train)

For every one of the 11 failures, I queried `train.parquet` directly for all other spectra
of the true molecule (outside the query's leakage-excluded metadata group) and computed
each side's adduct-implied neutral mass using a standard adduct mass-delta table (§5),
explicitly handling multimer (`2M`) adducts by dividing by the multimer count.

| rid | true molecule | query adduct | best matching ref. adduct | \|ΔM\| (Da) | ppm | label |
|---|---|---|---|---|---|---|
| 733288 | QHALUOAFNBWZED | [2M+Na]+ | [M-H]- | 0.00079 | 2.54 | confirmed cross-adduct |
| 410447 | JNVFBVQKGYTOAH | [M-H]- | [M+H]+ | 0.00025 | 0.76 | confirmed cross-adduct |
| 689395 | PJAAZAXBSLFNNK | [M-H]- | [M+H]+ | 0.00035 | 0.83 | confirmed cross-adduct |
| 626670 | OBFWTNNDCLKJFP | [2M+H]+ | [M-H]- | 0.00001 | 0.06 | confirmed cross-adduct |
| 318845 | HPQRSZWHZSXIAR | [2M+Na]+ | [M-H]- | 0.00119 | 3.06 | confirmed cross-adduct |
| 8769 | ADGUJIPJRXOWSH | [2M+CH2O2-H]- | [2M+Na]+ | 0.00016 | 0.41 | confirmed cross-adduct |
| 156672 | DFSLKFZMYMXXGX | [2M+Na]+ | [M+H]+ | 0.00058 | 1.42 | confirmed cross-adduct |
| 431909 | JZQMDNUYGZOPNW | [M+Na]+ | [M+CH2O2-H]- | 0.00008 | 0.23 | confirmed cross-adduct |
| 27493 | ANGIDRSDQGRWQD | [M-H]- | [M+H]+ | 0.00195 | 5.89 | confirmed cross-adduct |
| 6296 | ACBPYGWTRFBYMP | [2M+H]+ | [M+H]+ | 0.00054 | 1.93 | confirmed cross-adduct |
| 1029257 | XGQSLHOHBLGVCV | [M+H]+ | [M-H]- | 0.00095 | 2.70 | confirmed cross-adduct |

**Result: 11/11 (100%) are confirmed cross-adduct**, none "unresolved" and none
"same-adduct precursor mismatch." All implied neutral-mass differences are 0.06–5.89 ppm
(max 0.00195 Da at m/z ~330) — far tighter than any reasonable candidate-generation
tolerance, and consistent with ordinary instrument mass-accuracy noise, not a chemical
mismatch. This is strong, clean, load-bearing evidence for the adduct-aware approach.

**A bug was caught during this verification and is worth recording explicitly**: my first
pass at the mass-delta table had the wrong sign for `[M+Cl]-` and `[M+CH2O2-H]-`,
treating them like simple deprotonation (`[M-H]-`, a subtraction) when they are actually
net *additions* (a heavier group is added, then a proton is lost, but the group's mass
exceeds the proton's by far). This produced a nonsensical ~45 Da "mismatch" for rid 8769
on the first attempt. Manually verifying a real example caught it immediately — this is
exactly why step 4 of the required workflow ("verify candidate sets manually for several
cross-adduct examples") matters, and it directly informs the corrected formula in §5.

### 3b. The 22 ranking failures (correct candidate found, ranked below top 25)

Using data already captured in `results/exp001_checkpoint.json` (no new queries needed):
for **all 22** of these cases, `adduct_match` was `True` — the query's own adduct was
already present among the true molecule's candidates in the pool. Candidate generation
was not the problem for any of these; the correct spectrum, at the correct adduct, was
already being compared by Modified Cosine and still scored too low relative to competing
wrong candidates.

**Label: 22/22 "candidate present but ranking failure."** This is **out of scope for
EXP-002** — an adduct-aware candidate rule cannot help these, since candidate generation
was never the bottleneck here. This is flagged as a distinct future ranking-quality
question, not folded into EXP-002.

## 4. Candidate-generation variants

### Variant A — current baseline (control, unchanged)
Direct comparison: `|candidate.precursor_mz - query.precursor_mz| <= 0.01 Da`. No adduct
awareness. This is exactly EXP-001's rule, kept as the control arm.

### Variant B — adduct-aware (neutral-mass conversion)
Convert every row (query and candidate) independently to an estimated neutral mass using
its own recorded adduct, then compare neutral masses within a tolerance.

- Formula: `neutral_mass = (precursor_mz - delta) / n`, where `(delta, n)` come from the
  adduct table in §5, `n` being the multimer count (1 for monomer, 2 for `2M` adducts).
- Tolerance: **0.01 Da on neutral mass** (same absolute size as the baseline's precursor
  tolerance, chosen for direct comparability; see §4d for why this isn't tightened to
  ppm-based yet).
- Unsupported adducts: any adduct not in the §5 table (e.g. `[M-H2O+H]+`, `[M-2H2O+H]+`,
  bare `[M]+`, `[M+Br]-`) is **not converted**; that row falls back to the Variant A rule
  (direct precursor_mz comparison) rather than being silently dropped or wrongly
  converted. Coverage of the modeled adducts is 98.4% of train rows (measured directly).
- Candidate-pool control: comparing on a single derived scalar (neutral mass) with the
  same absolute tolerance as before keeps the filter mechanism identical in shape to the
  baseline — it is not an unbounded search, just a different (chemically correct) axis to
  filter on.

### Variant C — adduct-pair-aware (curated subset of B)
Mathematically the same conversion as B, but candidate inclusion is restricted to an
explicit, reviewed allowlist of `(query_adduct, candidate_adduct)` pairs — built from the
adducts actually observed in `test.parquet` and `enveda-180` (§5's 13-adduct table gives
`C(13,2) + 13 = 91` possible pairs; the allowlist starts as "all pairs among the 13
modeled adducts" and can be narrowed later if a specific pair turns out to be unreliable).
The distinction from B is auditability and the ability to tighten trust per pair (e.g. a
stricter tolerance for well-established pairs like `[M+H]+`↔`[M-H]-`, or exclusion of a
pair found to cause false positives) without touching the underlying mechanism. C is a
strict subset of B's candidate set by construction (C ⊆ B), never a superset.

### Variant D — broader raw precursor window (diagnostic only, not a proposed solution)
Widen Variant A's window from ±0.01 Da to ±50 Da (chosen to structurally bound the
largest single-adduct-swap mass delta in the §5 table, ~59 Da for the acetic-acid
adduct). This variant exists **only** to quantify the candidate-pool cost of catching the
same cross-adduct cases via brute force instead of chemistry — expected to be far more
expensive (many more false-positive candidates admitted) than B/C for the same recall
gain, which is the point of running it as a comparison. It is explicitly **not** proposed
for production, per the instruction not to "fix" this by loosening the filter arbitrarily.

### 4d. Why 0.01 Da and not ppm-based, for now
All 11 confirmed cross-adduct cases had mass differences under 6 ppm (max 0.00195 Da).
Using the same 0.01 Da absolute tolerance as the baseline keeps the two rules
directly comparable (isolating "adduct-aware vs not" as the only variable) and is
already ~5x more generous than anything observed. A ppm-scaled tolerance is a reasonable
future refinement (absolute Da tolerances are inconsistent in relative terms across a
wide precursor-mass range) but changing two things at once (adduct-awareness and
tolerance scaling) would confound EXP-002's isolation of candidate generation — left for
a later experiment if 0.01 Da absolute proves inadequate at scale.

## 5. Exact mass/adduct formulas

`precursor_mz = n × neutral_mass + delta` (equivalently `neutral_mass = (precursor_mz - delta) / n`)

| Adduct | delta (Da) | n (multimer) | Basis |
|---|---|---|---|
| `[M+H]+` | +1.007276 | 1 | proton mass |
| `[M-H]-` | −1.007276 | 1 | loss of a proton |
| `[M+Na]+` | +22.989221 | 1 | Na atom − electron |
| `[M+K]+` | +38.963158 | 1 | K atom − electron |
| `[M+NH4]+` | +18.033823 | 1 | NH4 − electron |
| `[M+Cl]-` | +34.969402 | 1 | Cl atom + electron |
| `[M+CH2O2-H]-` | +44.998203 | 1 | formic acid (46.005479) − proton |
| `[M+C2H4O2-H]-` | +59.013853 | 1 | acetic acid (60.021129) − proton |
| `[2M+H]+` | +1.007276 | 2 | dimer, proton |
| `[2M-H]-` | −1.007276 | 2 | dimer, deprotonation |
| `[2M+Na]+` | +22.989221 | 2 | dimer, Na |
| `[2M+CH2O2-H]-` | +44.998203 | 2 | dimer, formic acid |
| `[2M+C2H4O2-H]-` | +59.013853 | 2 | dimer, acetic acid |

Sign convention (the bug caught in §3a): **every adduct that nets an addition to the
neutral mass gets a positive delta**, including the "`+X-H`" style ones, because the
added group's mass always exceeds the lost proton's by a wide margin. Only genuine
deprotonation (`[M-H]-`, `[2M-H]-`) gets a negative delta. This table covers 98.4% of
train rows and all 7 distinct adducts present in `test.parquet`; the remaining ~1.6%
(rarer adducts like `[M-H2O+H]+`, `[M]+`, `[M+Br]-`) fall back to Variant A.

## 6. Leakage controls (unchanged from EXP-001)

Every candidate-generation variant reuses the **exact same** Mode B query set and
exclusion sets from `results/exp001_query_sets.json` — the same 400 query rids, the same
whole-metadata-group exclusion per query (not just the query row). No new near-duplicate
risk is introduced: broadening the mass-comparison axis does not touch which rids are
excluded, only which of the *remaining* (already-leakage-safe) rows are proposed as
candidates. This was verified directly in §3a/§7: the recovered candidates for all 11
known failures are real, distinct spectra (different rids, different metadata groups),
not the excluded query or its siblings.

## 7. Smoke-test results (already run, per the required pre-design check)

**Check 1 — the 11 known failures, explicit recovery test** (SQL-computed, not just the
manual mass-diff estimate in §3a): **11/11 recovered** by Variant B with a 0.01 Da neutral-mass
tolerance. Candidate counts for these 11 grew from a median of ~105 (baseline) to a
median of ~424 (Variant B) — more candidates, but not explosive (max observed: 5,020).

**Check 2 — pool-size behavior on 25 random Mode B queries** (not the known failures,
a fresh random sample, seed=7): mean candidate count grew from 778 (baseline) to 1,447
(Variant B), a ~1.86x average increase; max grew from 2,606 to 3,746. Growth was
adduct-dependent: monomer-adduct queries (`[M+H]+`, `[M-H]-`) grew modestly (1.2–1.6x),
while multimer queries (`[2M+Na]+`, `[2M+H]+`) grew more sharply (3.7–13x), since they
now correctly match against the much larger monomer-adduct population sharing the same
neutral mass. **No pathological explosion observed** (thousands, not millions, of
candidates; runtime for 25 queries including the neutral-mass computation itself was
2.7 seconds). The multimer growth pattern is worth watching at full scale (n=400) and is
called out explicitly in §8.

Both checks used the exact adduct-mass table in §5 and were computed via a single
server-side SQL `CASE` expression (not a per-row Python loop, which was tried first and
was too slow to be practical — noted here since the production EXP-002 implementation
must do the same).

## 8. Expected failure modes

1. **Multimer candidate-pool growth.** As seen in §7, `2M`-adduct queries could see
   double-digit-multiple candidate growth at full scale. If this makes Modified Cosine
   scoring too slow, a mitigation (not yet needed, not yet built) would be Variant C's
   pair-curation to only admit multimer↔monomer pairs that have actually shown recovery
   value, rather than all of them uniformly.
2. **False-positive adduct coincidences.** A 0.01 Da neutral-mass tolerance could
   occasionally admit a *different* molecule with an accidentally similar neutral mass
   under a different adduct. This is a candidate-generation precision cost, not
   (necessarily) a final-ranking cost, since Modified Cosine still has to actually match
   peaks — but candidate-generation Recall@25 alone won't reveal it; final MRR@25 must be
   watched for any degradation, not just improvement.
3. **The 1.6% adduct-uncovered rows** silently fall back to Variant A behavior for those
   specific candidates — expected to be a small, acceptable gap, but should be reported
   as its own statistic in the full run (how many queries' true molecule was among the
   uncovered 1.6%, if any).
4. **Ranking, not candidate generation, may still dominate.** §3b already shows 22/400
   failures are unrelated to candidate generation. Even a perfect candidate-generation
   fix cannot close that gap — EXP-002's expected ceiling is roughly the 11/400 (2.75%)
   candidate-absence failures, not the full 8.5% Mode B gap.

## 9. Smoke-test plan (completed; documented for reproducibility)

Already executed as part of this design (§7), using:
- `results/exp001_query_sets.json`'s existing Mode B queries (same leakage-safe
  construction as EXP-001, no new sampling).
- A single duckdb `train_index` table with one added `neutral_mass` column computed via
  a SQL `CASE` expression over the §5 table (fast: ~1s for 2.5M rows).
- 25 randomly-sampled Mode B queries (seed=7) for general pool-size behavior, plus all
  11 known EXP-001 candidate-absence failures for targeted recovery verification.
- No scoring (Modified Cosine) was run in this smoke test — it measured candidate
  generation only, per the instruction to isolate that stage before touching ranking.

## 10. Exact full EXP-002 configuration (proposed, NOT executed)

If approved, the full run would:

1. Reuse `results/exp001_query_sets.json`'s Mode B queries unchanged (n=400, same seed,
   same leakage exclusions) — **no re-sampling**, so results are directly comparable to
   EXP-001's Mode B numbers query-for-query.
2. Run candidate generation for Variants A (control), B (adduct-aware), and D (broad
   window, diagnostic only) against library C (all-train, per DEC-003). Variant C is
   deferred unless B shows a need for pair-level curation (e.g. if a specific pair proves
   noisy in the B results) — avoiding building an untested extra variant speculatively.
3. For each variant, run the **identical** EXP-001 scoring stage: matchms 0.33.1
   `ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)`, same
   molecule-level max-score aggregation, same top-25 truncation. The ranker itself is not
   touched.
4. Report, per variant: candidate-generation Recall@25, final Recall@1/5/10/25, MRR@25,
   median + full distribution of candidate counts, runtime, peak memory (reusing
   `exp001_full_v2.py`'s per-condition fetch/release/checkpoint pattern to avoid repeating
   the memory-pressure incident from EXP-001).
5. Decompose every query into: `previously absent -> now generated -> final rank`
   (a 3-way join between EXP-001's Mode B/all-train diagnostics and the new variant's
   diagnostics), so candidate-generation gains and ranking outcomes are never conflated.
6. Expected runtime: comparable to or somewhat longer than EXP-001's Mode B/all-train
   condition alone (candidate pools are larger per §7, roughly 1.9x on average, so
   scoring time should scale similarly) — a single-digit number of minutes for Variant B
   given EXP-001's Mode B/all-train condition took ~8 minutes; Variant D is expected to
   be substantially slower given its much larger candidate pools and is run mainly for
   the cost comparison, not for its own recall number.

**This configuration is proposed only. No full run has been executed. Awaiting review
before proceeding.**

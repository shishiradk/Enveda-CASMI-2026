# 02 — Class 1 Spectral-Library Retrieval Coverage (EXP-001)

Status: **COMPLETE.** This is the final, corrected, leakage-safe, memory-safe EXP-001
report. Earlier attempts (a leakage-uncontrolled preliminary run, a smoke test, and a
partial run killed by memory pressure) are preserved separately for provenance and
referenced below, but are **not** the result this report stands on.

Reproducibility artifacts:
- `results/exp001_checkpoint.json` — full per-condition results + per-query diagnostics
- `results/exp001_query_sets.json` — exact query rid lists and construction metadata
- `results/exp001_modified_cosine_activation_diagnostic.json` — the separate cross-adduct diagnostic
- `results/exp001_partial_run1_KILLED_preliminary.md` — the killed run, labeled preliminary
- `results/exp001_results.json` — the very first leakage-uncontrolled preliminary run

---

## 1. Hypothesis

Restated from `research/decision_log.md` (DEC-002): because the test set is 100%
timsTOF and `enveda-180` is train's dominant (and near-exclusive) timsTOF library, we
hypothesized `enveda-180` would be at least as good a Class 1 reference library as the
full 2.5M-row train set, and that adding the other ~1.39M non-timsTOF spectra would
dilute or slow retrieval without adding recall. This was a hypothesis derived from
metadata correlation, explicitly flagged as untested until this experiment.

A second, more basic question this experiment answers: **if the correct molecule's
spectrum genuinely exists somewhere in the library, how often can precursor-filtered
+ Modified Cosine retrieval actually find it, and where does it fail when it doesn't?**

## 2. Validation construction

Two modes, both drawn from `enveda-180`/`enveda-np-examples` (timsTOF) molecules so the
query population resembles the real test set's instrument profile:

**Mode A — near-exact duplicate ceiling test.** Sample query spectra from metadata-groups
`(inchikey14, adduct, precursor_mz, num_peaks)` that were independently validated (see
§3) as true near-duplicates by actual peak-level cosine similarity. Exclude only the
query row itself from the reference library; a validated near-duplicate sibling
deliberately remains, so this measures the practical ceiling of spectral-library
matching when the "same" measurement is genuinely available.

**Mode B — same molecule, different spectrum.** Sample molecules with ≥2 distinct
metadata-groups. Pick one group as the query; exclude the query's **entire group**
(every row sharing its adduct/precursor_mz/num_peaks combination) from the reference
library, not just the query row. Whatever remains for that molecule comes from a
different adduct, different collision-energy grouping, or different acquisition —
never a copy of the query. This measures library matching under genuine (if modest)
spectral variation, one step short of the harder Class 2 problem.

**Honesty check on Mode B's scope (explicitly requested):** Mode B guarantees the
surviving reference spectrum is not a metadata-duplicate of the query. It does **not**
guarantee a different instrument or lab — most surviving spectra are still drawn from
`enveda-180` itself (just a different adduct/CE/acquisition), since that's what the
molecule population mostly consists of. This is "same-library, different-spectrum," a
real but easier problem than fully cross-instrument Class 2. This distinction is the
basis for the "recommended next experiment" in §17.

## 3. Leakage controls

Metadata equality alone is **not** used as the near-duplicate definition. A 30-group
manual check earlier in this project found metadata-matching groups had actual
peak-level cosine similarities ranging from 0.05 to 0.9998 (median 0.95) — metadata
equality was a poor proxy on its own. The final protocol:

1. Candidate near-duplicate groups (same `inchikey14`/`adduct`/`precursor_mz`/`num_peaks`,
   size ≥2) are checked in randomly-shuffled batches by running the actual matchms
   `CosineGreedy` on their real peak lists, until enough validated groups exist to fill
   `MODE_A_N`.
2. **Threshold: direct-cosine ≥ 0.95** marks a group as a true near-duplicate, used only
   for Mode A. Chosen as the point where two spectra are unambiguously "the same
   measurement, possibly re-normalized" rather than a coincidence of peak count. Final
   run: checked 600/8,777 candidate groups, validated 257 (hit rate 43%, median sim
   0.916 among checked).
3. Mode B excludes the query's **whole metadata group**, a stricter and safer rule than
   excluding only the query row — this is what prevents an accidental near-duplicate
   from leaking into what's supposed to be a "different spectrum" test.
4. Query self-exclusion is structural (baked into every SQL candidate query's `NOT IN`
   clause), not just a convention — verified with concrete rid-level examples in the
   smoke-test evidence report (query's own group_id never appears among its own
   candidates).

## 4. Reference library definitions

| Key | SQL filter | Content |
|---|---|---|
| A | `ingest_lib = 'enveda-180'` | ~1.15M rows, 183,191 molecules, timsTOF |
| B | `ingest_lib IN ('enveda-180','enveda-np-examples')` | A + 1,184 extra rows |
| C | `TRUE` | All 2.54M train rows, all instruments/libraries |

## 5. Retrieval pipeline

```
query spectrum
    -> precursor m/z filter (|candidate.precursor_mz - query.precursor_mz| <= 0.01 Da)
    -> Modified Cosine rescoring (matchms 0.33.1 ModifiedCosineGreedy) on all surviving candidates
    -> molecule-level aggregation (max score per inchikey14)
    -> rank molecules, truncate to top 25
```

No separate approximate/sparse-cosine prefilter stage was inserted between the
precursor filter and Modified Cosine: candidate pools after precursor filtering were
already small (median 232–690 candidates per query across conditions, well within what
Modified Cosine can score directly), so an intermediate approximate stage would add
complexity without a demonstrated need. This is a documented design choice, not an
omission.

## 6. Parameters

- Precursor tolerance: **0.01 Da**, absolute, adduct-naive (see limitation in §16)
- Peak m/z tolerance (matchms default): **0.1 Da**
- Intensity weighting: `mz_power=0.0`, `intensity_power=1.0` (matchms defaults)
- Modified Cosine implementation: `matchms.similarity.ModifiedCosineGreedy` v0.33.1,
  verified via a hand-derived toy check (§9 of `research/16_modified_cosine_notes.md`;
  re-verified inline in this run's own log: score=0.9903, matches=3, exact match to the
  hand-derived expected value)
- Aggregation rule: max score per `inchikey14` (deduplicates candidate spectra to
  molecule level before ranking)
- Sample sizes: Mode A n=200, Mode B n=400, multi-spectrum add-on n=150 molecules × 2
  held-out query spectra each
- Seed: 42 (see §16 for a caveat on exact reproducibility)

## 7. Results table (primary conditions, Modified Cosine)

| Condition | candGenRecall | Recall@1 | Recall@5 | Recall@10 | Recall@25 | MRR@25 | median candidates |
|---|---|---|---|---|---|---|---|
| A / enveda-180 | 1.0000 | 0.5650 | 0.7950 | 0.8550 | 0.9600 | 0.6654 | 232 |
| A / enveda-180+np | 1.0000 | 0.5650 | 0.7950 | 0.8550 | 0.9600 | 0.6654 | 232 |
| A / all-train | 1.0000 | 0.5600 | 0.7950 | 0.8550 | 0.9600 | 0.6629 | 300 |
| B / enveda-180 | 0.9650 | 0.5300 | 0.8025 | 0.8750 | 0.9150 | 0.6502 | 400 |
| B / enveda-180+np | 0.9650 | 0.5300 | 0.8025 | 0.8750 | 0.9150 | 0.6502 | 400 |
| B / all-train | 0.9700 | 0.5350 | 0.8050 | 0.8725 | 0.9150 | 0.6510 | 690 |

Direct (unshifted) Cosine comparison, Mode B / all-train: Recall@1=0.5350, Recall@5=0.8050,
Recall@10=0.8725, Recall@25=0.9150, MRR@25=0.6510, median candidates=690 — **byte-identical**
to Modified Cosine on this condition. See §13 for why.

## 8. Candidate-generation recall (before final ranking)

Mode A is trivially 1.0000 everywhere by construction (a validated sibling always
survives in every library variant, since it's already part of `enveda-180`). Mode B is
the informative number: **96.5% (library A/B) to 97.0% (library C)** — meaning 3.0–3.5%
of the time, no candidate within the ±0.01 Da precursor window belongs to the true
molecule at all, regardless of how much library is searched. Library C recovers 2 more
molecules into the candidate pool than A/B (14 missing vs 12 missing out of 400) — a
real but tiny effect, and one that doesn't translate into a final Recall@25 difference
(both land at 0.9150), because those few extra recovered candidates apparently don't
rank inside the top 25 anyway.

## 9. Final MRR@25

Mode A (ceiling test): **0.6629–0.6654** across library variants — barely distinguishable
from each other.
Mode B (realistic test): **0.6502–0.6510** across library variants — again barely
distinguishable. All-train's MRR@25 (0.6510) is marginally *higher* than enveda-180-only
(0.6502), the opposite direction from what DEC-002's hypothesis predicted, though the
difference (+0.0008) is well within noise for n=400.

## 10. Recall@1/5/10/25

See §7 table. Headline: Mode B (realistic "different spectrum" case) reaches
Recall@25 ≈ 0.915 and Recall@1 ≈ 0.53–0.535 regardless of which library is searched.

## 11. Multi-spectrum results

Using the 150-molecule × 2-held-out-query-spectra add-on (library C, Modified Cosine),
three strategies compared:

| Strategy | Recall@1 | Recall@25 | MRR@25 | Real/implementable? |
|---|---|---|---|---|
| Single spectrum (first query only, arbitrary) | 0.4067 | 0.9133 | 0.5562 | Yes |
| Max-aggregation across both query spectra | 0.4333 | 0.9267 | 0.5694 | Yes |
| Oracle best-of-2 (hindsight pick using ground truth) | 0.5867 | 0.9867 | 0.7243 | **No — upper bound only** |

The real, implementable max-aggregation strategy **does** beat the real single-spectrum
baseline on every metric (Recall@1 +2.66pp, Recall@25 +1.34pp, MRR@25 +0.0132) — a
modest but genuine, honestly-measured improvement from using multiple query spectra per
molecule. The much larger gap to the oracle (MRR@25 0.72 vs 0.57) shows there's real
room for a smarter aggregation strategy (e.g. weighting by spectrum quality) beyond
naive max — flagged as a candidate for a future experiment, not built here per
"don't over-engineer this stage."

## 12. Runtime / memory

Full run: **2,517.3 seconds (~42 minutes)**, final RSS 1,838 MB. Peak RSS during the
single most expensive condition (Mode B / all-train) was **3,857 MB**, always returning
to ~1,600–1,800 MB after that condition's data was released — i.e. memory was bounded
per-condition, not accumulating across the whole run. This was the fix for a prior
attempt that was killed by a system memory-pressure safeguard after 4/6 conditions
(preserved in `results/exp001_partial_run1_KILLED_preliminary.md`); see
`exp001_full_v2.py`'s docstring for the specific changes (per-condition fetch/release,
incremental checkpointing, no global rid→metadata dict covering all 2.5M rows).

Per-condition fetch time (dominated by a full-file rescan, ~30–97s, roughly independent
of candidate count) plus scoring time (17s for the smallest Mode A condition up to 397s
for the largest Mode B condition).

## 13. Failure analysis (Mode B / all-train / Modified Cosine, n=400)

| Category | Count | % |
|---|---|---|
| Correct molecule ranked 1–25 (success) | 366 | 91.5% |
| Correct candidate found, ranked below 25 | 22 | 5.5% |
| Molecule absent from candidate pool entirely | 11 | 2.8% |
| Candidate generation total failure (0 candidates) | 1 | 0.2% |

Of the 34 non-rank-1-success failure cases with further sub-categorization attempted:
**0** were attributable to an adduct mismatch (i.e. the true molecule's surviving
candidates never shared the query's exact adduct), **1** involved an unusually large
spectrum (>2,000 peaks), and **21** fell into "other" — meaning most ranking failures
are **not** explained by the two specific mechanisms checked here, and remain an open
question (§16).

**Important, load-bearing finding (not a bug):** Modified Cosine and direct Cosine
produced byte-identical results on Mode B/all-train (confirmed at both smoke-test and
full scale). Traced to matchms's own source
(`ModifiedCosineGreedy.pair()`: `if abs(mass_shift) <= self.tolerance: return
CosineGreedy(...).pair(...)`, `tolerance` default 0.1 Da). Our precursor pre-filter
(0.01 Da) guarantees every candidate's `mass_shift` is smaller than matchms's own
shift-fallback threshold, so the shift-matching branch is provably never taken inside
this pipeline. **A separate diagnostic outside the main pipeline** confirms the branch
does activate given a genuine mass difference: same real molecule (`inchikey14`
`OQQGVHNCEDTIBM`), `[M+H]+` (precursor 341.1871) vs `[M+Na]+` (precursor 363.1684, a
21.98 Da difference) — direct Cosine found 6 matched peak pairs (score 0.9245),
Modified Cosine found **9** matched peak pairs (score 0.9245, same to 4 decimals — the
3 extra shifted matches happened to be low-intensity in this particular pair, so they
didn't move the score, but they are real, additional matches the direct method cannot
see). This confirms the scorer works correctly; it simply never gets the chance to show
its advantage under this experiment's tight precursor window.

## 14. Interpretation

Given a real molecule whose spectrum exists somewhere in the timsTOF-like population:
- If a near-identical spectrum is available (Mode A), simple precursor-filtered cosine
  retrieval finds the correct molecule in the top 25 about **96%** of the time, and at
  rank 1 about **56–57%** of the time.
- If only a genuinely different spectrum of the same molecule is available (Mode B, the
  more realistic case), top-25 coverage drops to about **91.5%**, rank-1 to about
  **53%**.
- The gap between candidate-generation recall (~96.5–97%) and final Recall@25 (91.5%)
  shows retrieval loses a further ~5.5 points purely to **ranking**, not candidate
  absence — i.e. roughly as many queries fail because Modified Cosine ranked the right
  answer too low as fail because it was never found at all.

## 15. What this tells us about Class 1 coverage

The DEC-002 hypothesis ("search `enveda-180` only, it beats the full train set") is
**not confirmed as stated** — library choice made essentially no difference in either
mode (MRR@25 differences of 0.0008–0.0025, well within sampling noise for n=200–400).
The more precise finding: **the precursor window (0.01 Da) is a far stronger filter
than library identity** — most non-timsTOF spectra never enter the candidate pool for a
timsTOF query regardless of which library variant is nominally being searched, because
their reported precursor_mz values differ from the query's by more than 0.01 Da even
for the same molecule. This means it's safe (and simpler) to search the full train set
rather than manually restricting to `enveda-180`, since the practical filtering is
already happening via the precursor window, not via a library allowlist.

Class 1, defined as "the exact spectrum-or-a-near-duplicate is in the library," is
strongly solvable with this simple pipeline (~96% top-25 coverage). The harder,
more test-realistic case ("some different spectrum of the same molecule is in the
library") is still quite solvable (~91.5% top-25 coverage) but leaves a real,
measurable gap — about 8.5% of queries never reach the correct molecule in the top 25,
split roughly evenly between candidate-absence and ranking failure.

## 16. What remains unknown

1. **Why do 21/34 non-rank-1 failures fall into "other"?** Neither adduct mismatch nor
   an extreme peak count explains most ranking failures. This needs direct inspection
   of representative failure cases (query vs. best-scoring wrong candidate vs.
   true-molecule candidate) — not yet done.
2. **The precursor filter is adduct-naive.** A same-molecule candidate at a genuinely
   different adduct (mass difference of tens of Da, e.g. `[M+H]+` vs `[M+Na]+`) is
   structurally excluded from ever being considered, regardless of library. The
   cross-adduct diagnostic (§13) shows Modified Cosine *could* meaningfully compare such
   pairs if they were allowed into the candidate pool — but our candidate generation
   never gives it the chance. This may be suppressing real recall the pipeline is
   otherwise capable of.
3. **Exact reproducibility is only approximate, not bit-exact.** The killed v1 partial
   run and the completed v2 run used the identical seed (42) and identical sampling
   code, yet produced slightly different query samples (e.g. Mode A Recall@1 0.575 vs
   0.565). Traced to `duckdb`'s `PRAGMA threads=4` — multi-threaded execution does not
   guarantee a stable row order within a `list()` aggregate across separate runs, which
   cascades into different "first pair" choices during near-duplicate validation and
   thus slightly different sampled query sets. The methodology and conclusions are
   unaffected (numbers agree within ~1 percentage point), but a bit-exact rerun would
   require `PRAGMA threads=1` or an explicit `ORDER BY` before the `list()` aggregate.
4. Whether Recall@25/MRR@25 would improve meaningfully if candidate generation were
   made adduct-aware (neutral-mass-based) rather than raw-precursor-mz-based — directly
   related to point 2, untested.

## 17. Recommended next experiment

**EXP-002 candidate: adduct-aware candidate generation.** Convert each candidate's
`(precursor_mz, adduct)` to an estimated neutral mass (using a small, documented
adduct-mass-delta table for the handful of adducts observed in test/`enveda-180`) and
filter by neutral-mass proximity instead of raw precursor_mz proximity. This would let
genuinely cross-adduct same-molecule candidates into the pool, giving Modified Cosine's
shift-matching logic (confirmed working in §13) an actual chance to contribute inside
the real pipeline, and would directly test whether the ~8.5% Mode B gap narrows.
Secondary candidate: a targeted look at the 21 "other" failure cases from §16, since
their cause is currently unexplained and may point to something cheaper to fix first.

Per the stop condition on this experiment: **no further work (Class 2, neural models,
COCONUT, ranker tuning) proceeds until this report has been reviewed.**

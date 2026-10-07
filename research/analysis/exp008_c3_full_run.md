# EXP-008 — Full run (407 queries): preparation record

Date: 2026-09-24. Status: **PREPARED — NOT EXECUTED.** The 407-query run requires separate authorization.
The results sections are to be written after an authorized execution.

This is a performance-only re-implementation of the locked experiment. The scientific reference remains
`research/scripts/exp008_c3_smoke.py` (sha256 `a49fa4a0…263a1b`, unchanged). The EXP-007 4-decimal vs
±0.01 Da caveat remains binding.

## 1. Profiling (one smoke query, rid 179633, [M+H]+, 5,000-rep sample; reference code)

The ±150 Da window covers essentially the whole same-adduct representative library: 164–167k [M+H]+ and ~66k [M-H]-
per arm. With the query and decoy arms, that is ~334k Modified Cosine pairs per [M+H]+ query. Per evidence spectrum:

| component | cost | share |
|---|---|---|
| pandas `peaks.loc[rid]` lookup | 134 µs | 16% |
| `make_spectrum` → matchms `Spectrum` (metadata harmonization via PickyDict/regex) | 225 µs | 27% |
| `ModifiedCosineGreedy.pair` (Python wrapper, Fragments copies/vstack/_is_sorted around the numba helpers) | 463 µs | 56% |
| Morgan fingerprints + Tanimoto + S(c) aggregation (pool 37, cold fp cache) | 0.14 s per query | negligible |
| evidence fetch | one global scan (not per query) | one-off |

Repeated work across queries: every query rebuilt Spectrum objects for the same ~167k representatives, twice
(query and decoy arm). The scoring itself (query × evidence) is intrinsically per query and cannot be shared.

## 2. Optimization (v1, `research/scripts/exp008_c3_fast.py`)

1. **Flat peak store** (`results/exp008_c3_full/cache/rep_store.npz`, 1.9 GB): all 255,492 enveda-180
   monomer-adduct representatives. Peaks are pre-sorted with exactly `make_spectrum`'s `np.argsort`, and precursors are
   asserted equal to the representative table. It is built once (142 s).
2. **Batch numba kernel** `batch_modcos`: one call per query arm. It calls **matchms's own compiled helpers**
   (`collect_peak_pairs`, `score_best_matches`) with the same argument roles (reference = query, query = evidence),
   the same `vstack(...).T` array layout, `mass_shift = pm_query − pm_evidence`, the CosineGreedy fallback for
   |shift| ≤ 0.1, and the stable-mergesort-then-reversed pair ordering. It runs `prange` over evidence spectra with
   6 threads. Each score is independent, so results do not depend on scheduling.
3. **Unchanged:** representative selection, ±150 Da filter (inclusive bounds), target/decoy group exclusion,
   the top-20 ordering key (−s, inchikey14), S(c), tie-aware RR, chance, the direct ranker (reference `sim.pair`
   path, ~hundreds of pairs/query), failure categories, bootstrap. All of these are imported from the reference script.
4. **Resumable output:** one jsonl line per completed query, fsync'd. A restart skips completed rids. The summary is
   computed in manifest order.
5. **Gates:** `full` refuses without `--authorized`, without a passing regression report for the identical script
   sha256, or if the reference script has changed.

The process-level parallelism considered in the brief was not needed. Thread-level parallelism inside the kernel is deterministic.

## 3. Verification

- **Kernel unit test:** 12,000 real enveda-180 pairs (4 queries × 3,000), including 57 cosine-fallback pairs.
  **0 bit-mismatches** against `ModifiedCosineGreedy.pair`, max abs difference 0.
- **Population:** `prepare-full` recomputed eligibility, tiers and decoys for all 600 queries with the reference rules.
  It gave 407 eligible (T-low 141 / T-mid 128 / T-high 138), identical to the preflight. All 30 smoke manifest rows were
  reproduced field-for-field, and the seeded sample was redrawn identically.
- **30-query regression** (`results/exp008_c3_full/regression_report.json`): every record field except `sec` was
  compared recursively against `results/exp008_c3_smoke_query_results.jsonl`. That covers rid, tier, pool sizes, pool
  membership flags, A/B/C RRs and rank spans, the top-20 hits with s and Tc, top-25 candidates with scores, decoy RRs,
  failure category, and the leakage check fields. **0 differences, max float difference 0.0 → PASS.**

## 4. Runtime and memory

| | reference (smoke) | optimized |
|---|---|---|
| median s/query | 434 | **50.0** |
| mean s/query | 481 | 73.8 |
| p75 / p90 / max | – / – / 1,456 | 101 / 156 / 277 |
| queries > 120 s | 25/30 | 7/30 |
| total for 30 queries | 14,417 s | 2,215 s + 150 s setup + 17 s JIT |

- The median runtime gate (< 120 s/query) is met. A tail remains: 7/30 queries exceed 120 s.
- The remaining cost is the numba helpers themselves (typed-list construction in `find_matches`/`collect_peak_pairs`),
  which grows with the query's peak count. Reducing it further would mean re-implementing matchms's pair collection,
  which was not done in order to keep numerics identical by construction.
- Peak working set was **3.03 GB**: store 1.9 GB, train metadata, structure universe, direct-ranker peaks.
  8 logical CPUs, 6 numba threads.
- Projected full run: 407 × 73.8 s ≈ 8.3 h plus about 3 min setup. This is an extrapolation from the 30-query mean, not a measurement.

## 5. Full-run manifest

`results/exp008_c3_full/manifest.json` contains:
- experiment ID, script sha256 `b1d501a2…381bf8` and reference sha256, and optimization version `exp008-fast-v1`
- population count 407 with all query IDs and the per-query tier and decoy
- seeds (20260924 / 20260925 / 20260926), the EXP-002 monomer adduct table, the 5 ppm rule, the ±150 Da rule, and H=20
- Morgan r=2 / 2048 / Tanimoto, and the scorer parameters
- software versions (Python 3.14.0, matchms 0.33.1, RDKit 2026.03.6, duckdb 1.5.5, numba 0.67.0, numpy 2.4.4)
- the regression result and the runtime benchmark

## 6. Command (not executed)

```
python research/scripts/exp008_c3_fast.py full --authorized
```

Outputs:
- `results/exp008_c3_full/query_results.jsonl` (resumable)
- `results/exp008_c3_full/summary.json` (overall and per tier: A/B/C MRR, R@1/5/10/25, lifts C−A, decoy−A, C−decoy with bootstrap CIs, leakage checks)
- this report, to be completed after execution

## 7. Results

*Not executed.*

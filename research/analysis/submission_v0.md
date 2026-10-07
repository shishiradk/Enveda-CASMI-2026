# CASMI 2026 Submission — V0

## Status
**V0 is the first complete end-to-end submission produced by this pipeline.** It is a full optimized run of the validated V0 algorithm (Variant-A over all training data, molecule-level max aggregation, v1-equivalent candidate adducts). Sanity gate (20-molecule subset vs the validated V1 smoke20 reference) passed with bit-identical scores before the full run.

## Exact algorithm / configuration (locked)

- **Candidate generation**: Variant-A — raw `precursor_mz` ± 0.01 Da, all-train library (no adduct/neutral-mass filtering).
- **Scoring**: matchms `ModifiedCosineGreedy` with
  - `tolerance=0.1`
  - `mz_power=0.0`
  - `intensity_power=1.0`
- **Scorer version**: matchms `0.33.1` (numba kernels unchanged).
- **Aggregation**: molecule-level max of per-(spectrum, inchikey14, cand_adduct) scores.
- **Ranking/selection**: score-descending, then deterministic `inchikey14` ascending tie-break; top-25 per molecule; SMILES via RDKit canonicalization of the selected inchikey14.
- **Execution**: optimized resumable implementation (`research/scripts/v0_make_submission_fast.py`) — one-time sorted candidate `pool.parquet`, lazy candidate-Spectrum cache, chunk-level `ProcessPoolExecutor` parallelism with fixed-order deterministic assembly. Scientific behavior is identical to the validated sequential engine; byte-identical scores on the sanity gate.

## Test counts

- Test spectra scored: **1,213** (every test spectrum scored exactly once).
- Molecule IDs in submission: **400** exactly matching `sample_submission.csv` order, no missing, no extra, no duplicates.
- Score rows before aggregation: **355,810** (unique per spectrum/inchikey14/cand_adduct).

## Candidate statistics

- Candidate spectra scored (per-spectrum window pairs, summed): **1,481,241**
- Candidates per spectrum: avg **1,221.1**, median **1,097**, max **3,125**
- Spectra per molecule: avg 3.03, median 3, max 9
- Predictions per submission row: min **4**, max **25** (all rows within 1–25)

## Runtime

- Score stage (all 21 chunks, 8 workers, band fetch): **384 s** wall
- Aggregate + validation stage: **33.2 s**
- Total approximate: **~7 min** including pool build (pool itself is a one-time cache, reused here).
- Resumability verified: on rerun the completed 21/21 chunks were skipped (0 s score stage).

## Validation A–E (all PASS, `validation_report.json`)

- **A (IDs & coverage)**: 400 rows; exact same 400 sample IDs, same order, no missing/extra/duplicates; all 1,213 spectra present exactly once.
- **B (prediction count)**: every row 1–25 predictions; max 25, min 4, no rows >25, no empty rows.
- **C (chemistry)**: 0 invalid SMILES, 0 empty fields, 0 duplicate within-molecule predictions (RDKit parse + canonical check).
- **D (format)**: exact columns `molecule_id,smiles`; no index column; no NaN; semicolon-separated SMILES.
- **E (ranking)**: 0 misordered molecules — emitted order equals recomputed score-desc, inchikey14-asc top-25 ranking.

## Software versions

- Python 3.14.0
- matchms 0.33.1, RDKit 2026.03.6, duckdb 1.5.5, numpy 2.4.4
- All on Windows / PowerShell.

## Artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Submission CSV | `results/submission_v0/submission_v0.csv` | `4cbddddd6d01041c6a4be021ef4a215897f14ac9d9c998cb504ac312f48b791c` |
| Score rows | `results/submission_v0/scores.parquet` | — (see manifest) |
| Validation report | `results/submission_v0/validation_report.json` | — |
| Runtime summary | `results/submission_v0/runtime_summary.json` | — |
| Run manifest | `results/submission_v0/run_manifest.json` | embeds above + versions |
| Sanity gate (opt) | `results/submission_v0/sanity_gate_opt.json` | — |

## Limitations

- Submission was generated with the V0/V1-equivalent variant; other candidate-generation variants (v1a same-adduct, v1b sum-fusion) and the neutral-mass (EXP-002) / C2 (EXP-007) ideas were explicitly **excluded**.
- Band-fetch candidate superset is restricted to the test `precursor_mz` range ± 0.01 Da; per-query windows are provably identical to the all-train engine (verified 0.0 max score diff on the sanity gate).
- No claim about leaderboard performance is made; performance is unknown and not estimated here.
- Memory: `psutil` unavailable, so peak RSS not recorded; no memory failures observed during run.
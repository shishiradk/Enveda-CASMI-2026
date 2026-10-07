# Dataset Report — Enveda CASMI 2026

Status: PHASE 0 / PHASE 1 (dataset reconnaissance only). No modeling has been done yet.

Every number below was produced by an actual query against the real competition files
(`train.parquet`, `test.parquet`, `sample_submission.csv`) using `duckdb`, not estimated.
Queries are reproducible via `research/01_dataset_recon.ipynb` and `src/data/loader.py`.

---

## 0. Competition facts (external, not invented)

Source: Kaggle competition page (`kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra`)
search results, cross-checked against a third-party public repo whose stated train/test
row counts match our own measurements exactly (2,539,608 train spectra, 1,213 test spectra,
400 test molecules) — that match is our main evidence the repo's other claims are trustworthy,
though anything below marked PUBLIC COMPETITOR CLAIM has not been independently verified by us.

**FACT (from Kaggle rules/search results):**
- Task: predict a molecule's 2D structure (SMILES) from LC-MS/MS spectra.
- Metric: MRR@25. Matching is via RDKit tautomer canonicalization + InChIKey14
  (connectivity only; stereochemistry/tautomers ignored).
- Submission: up to 25 ranked SMILES per `molecule_id`, joined by semicolons, best first
  (matches `sample_submission.csv` format we observed).
- Max team size: 5. Data license CC BY-NC 4.0, no redistribution. Code developed on
  competition data may not be privately shared outside the team (rule 3.6).

**PUBLIC COMPETITOR CLAIM (from a third-party GitHub README, not yet verified against the
official Kaggle rules/data page directly — Kaggle's overview page is JS-rendered and our
fetch tool could not retrieve its text):**
- Timeline: start Sep 14 2026, entry/merge deadline Dec 7 2026, final submission deadline
  Dec 14 2026.
- Max 5 submissions/day, 2 final submissions selected for scoring.
- Kaggle kernel time limit ~9 hours; a baseline library-search pipeline reportedly runs the
  full test set in ~1-2 minutes.
- A public baseline approach: sparse-cosine pre-filter over a deduplicated library (~500k
  unique structure/adduct spectra) drawn from five sources (enveda-180, enveda-np-examples,
  GNPS, RIKEN, pluskal_ms2), then exact Modified Cosine rescoring. More advanced public
  submissions reportedly add the COCONUT natural-products database (~729k candidates) and a
  neural fingerprint-prediction model to get past the library-only ceiling.

**ACTION ITEM:** we should get the *official* Kaggle rules/data page text directly (e.g. by
having the user paste it, or via an authenticated fetch) rather than relying solely on a
third-party repo, before finalizing runtime/compute/external-data assumptions.

---

## 1. File inventory

| File | Size | Rows | Row groups |
|---|---|---|---|
| `train.parquet` | 2.82 GB | 2,539,608 | 21 |
| `test.parquet` | 4.6 MB | 1,213 | 1 |
| `sample_submission.csv` | 42.6 KB | 400 | — |

**Engineering gotcha (important, save this):** `pandas.read_parquet()` /
`pyarrow.parquet.read_table()` fail on both parquet files in this environment with
`OSError: Repetition level histogram size mismatch` (pyarrow 19.0.0). This looks like a
pyarrow reader bug tied to the repetition-level stats for the nested `list<double>` columns.
`duckdb` reads both files without any issue and is what we used for every query in this
report. Use `src/data/loader.py` (duckdb-backed) for all data access going forward, and
re-check whether this pyarrow bug is still present before assuming `pandas.read_parquet`
will work in the final Kaggle submission notebook (Kaggle's pyarrow version may differ).

---

## 2. Schema

### `train.parquet` (2,539,608 rows, 18 columns)

| Column | Type | Meaning (inferred) |
|---|---|---|
| `ingest_lib` | string | Source spectral library this record came from (see §5) |
| `normalized_smiles` | string | Canonicalized SMILES of the molecule |
| `inchikey` | string | Full InChIKey (structure + stereochemistry layer) |
| `inchikey14` | string | First InChIKey block only — connectivity identity, matches the competition's scoring key |
| `molecular_formula` | string | Molecular formula, e.g. `C20H15N3O2` |
| `ionization_mode` | string | `positive` / `negative` |
| `instrument_type` | string | Free-text instrument label (33,053 nulls; 20+ distinct spellings, see §5) |
| `adduct` | string | Normalized adduct, e.g. `[M+H]+` |
| `adduct_orig` | string | Adduct as originally recorded by the source library |
| `precursor_mz` | double | Precursor ion m/z |
| `precursor_error_ppm` | double | ppm error between theoretical and observed precursor m/z (train-only; absent from test) |
| `ms2_mzs` | list<double> | Fragment m/z values (the peak list x-axis) |
| `ms2_normalized_intensities` | list<double> | Fragment intensities, normalized into `[~6.6e-8, 1.0]` (base peak = 1.0) |
| `num_peaks` | int64 | Peak count (equals `len(ms2_mzs)`) |
| `base_peak_intensity` | double | Mostly null (1,163,138 / 2,539,608 = 45.8% null) — likely raw (pre-normalization) intensity of the base peak, only populated by some source libraries |
| `collision_energy_ev` | list<double> | Collision energy(ies) in eV, as a list (spectra can be a stitched/ramped combination of several CEs) |
| `collision_energy_orig` | string | Collision energy as originally recorded (e.g. `"20,40,60"`, `"[20.0,30.0,60.0,40.0]"`, `"40"`, `"-60"`, or `"unknown"`) |
| `collision_energy_orig_units` | string | `eV` / `NCE` / `V` / `unknown` — **units are not uniform**, `collision_energy_ev` should be preferred over parsing `collision_energy_orig` by hand |

### `test.parquet` (1,213 rows, 12 columns)

Same spectral columns as train, **minus all structure/identity columns**
(`normalized_smiles`, `inchikey`, `inchikey14`, `molecular_formula`, `ingest_lib`,
`precursor_error_ppm`) — these are exactly what we must predict — **plus**:

| Column | Type | Meaning |
|---|---|---|
| `molecule_id` | string | Opaque id, e.g. `m_e8359d`. This is the prediction key (matches `sample_submission.csv`). One molecule can have several spectra. |
| `spectrum_id` | string | Opaque id, e.g. `s_24a837e2`. One row = one spectrum. |

`sample_submission.csv`: 400 rows, columns `molecule_id, smiles`. `smiles` in the sample is a
placeholder (`"CCO;CCO;...;CCO"`, 25 copies) — confirms up to 25 semicolon-joined SMILES per
molecule.

---

## 3. Scale: spectra, molecules, spectra-per-molecule

- Train: 2,539,608 spectra / **275,810 unique molecules** (by `inchikey14`) / 277,566 unique
  `normalized_smiles` strings / 275,950 unique full `inchikey` (i.e. very few
  stereoisomer pairs share an `inchikey14` — consistent with the eventual connectivity-only
  scoring).
- Test: 1,213 spectra / 400 unique `molecule_id` (matches `sample_submission.csv` exactly,
  1-to-1, no orphans in either direction).
- Spectra per train molecule is heavy-tailed: 17,033 molecules have exactly 1 spectrum, but
  the mode is 3-8 spectra/molecule (a molecule commonly appears at several adducts /
  collision energies / from several libraries); some molecules have 30+.
- Spectra per test molecule: 1,213 / 400 ≈ 3.0 average; we saw one example (`m_e8359d`) with
  9 spectra spanning 3 adducts (`[M+H]+`, `[M-H]-`, `[M+Na]+`) and multiple collision
  energies each (20, 40, 60, and the combined "20,40,60" ramp). **108 of the 400 test
  molecules have more than one adduct present.** This directly motivates §26 (multi-spectrum
  aggregation) as a near-certain requirement, not just an optional idea.

---

## 4. Peak statistics

| | train (all) | train (timsTOF subset only) | test |
|---|---|---|---|
| rows | 2,539,608 | 1,154,969 | 1,213 |
| min peaks | 1 | 4 | 4 |
| median peaks | 42 | 138 | 230 |
| max peaks | 73,318 (outlier — needs inspection) | — | — |

Test spectra have noticeably **more peaks on average than train overall**, but are closer to
(still higher than) the timsTOF-only train subset. This is worth understanding before
picking peak-filtering/top-N parameters — using global train peak-count statistics to tune
preprocessing would likely mismatch the test distribution.

Intensity values in `ms2_normalized_intensities` are pre-normalized into roughly
`[6.6e-8, 1.0]` (base peak = 1.0) in a 100k-row train sample we checked — confirms these are
not raw intensities and are already comparable in scale within a spectrum.

---

## 5. Instrument / library structure — the most important finding so far

`instrument_type` in train has 20+ distinct spellings (`timsTOF`, `Orbitrap`, `QTOF`,
`LC-ESI-QTOF`, `qTof`, `qtof`, `ToF`, `orbitrap`, ... — clearly **not normalized**, will need
canonicalization if used as a feature) plus 33,053 nulls.

**`test.parquet`'s `instrument_type` is 100% `timsTOF` (1,213/1,213 rows).**

Cross-referencing:
- `ingest_lib` counts in train: `enveda-180` (1,153,785), `pluskal_ms2` (527,581), `riken`
  (347,171), `gnps` (220,849), `massbank` (101,727), `mona` (92,416), `spectraverse`
  (50,933), `msdial` (40,765), `drug_plus` (2,545), `enveda-np-examples` (1,184), `masaryk`
  (652).
- **`instrument_type == 'timsTOF'` in train is essentially the `enveda-180` library**:
  1,153,785 of `enveda-180`'s 1,153,785 rows are timsTOF (100%), plus a tiny 1,184-row
  `enveda-np-examples` library also on timsTOF. Every other `ingest_lib` (GNPS, MassBank,
  MoNA, RIKEN, pluskal_ms2, spectraverse, msdial, drug_plus, masaryk) is **entirely
  non-timsTOF**.
- The timsTOF/`enveda-180` library covers 183,191 unique molecules; the other libraries
  combined cover 95,157 unique molecules; **the overlap between the two sets is only 2,538
  molecules** — i.e. `enveda-180` is almost entirely disjoint in molecular content from the
  public spectral libraries (GNPS/MassBank/MoNA/RIKEN/etc.) bundled into `train.parquet`.
- Test's adduct set (`[M+H]+`, `[M-H]-`, `[M+CH2O2-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+Cl]-`,
  `[M+K]+`) is a strict subset of `enveda-180`'s adduct set, and `enveda-180`'s
  collision-energy formatting (bare numbers or comma/semicolon lists like `"20,40,60"`,
  including negative-mode entries written with a leading `-`) matches test's
  `collision_energy_orig` formatting closely; other libraries mostly use the
  `[20.0,30.0,60.0]`-style bracketed list format instead.

**Working hypothesis (NOT yet tested):** the intended "home" library for Class 1 retrieval
against the test set is `enveda-180` (proprietary, natural-product-flavored, timsTOF-only,
~183k molecules), not the full 2.5M-row train file. The other ~1.39M spectra (GNPS, MassBank,
MoNA, RIKEN, pluskal_ms2, spectraverse, msdial, drug_plus, masaryk) are a different
instrument population that is 97%+ disjoint in molecule identity from `enveda-180`, and their
main value is probably Class 2 (learned/instrument-invariant similarity) and Class 3
(broadening formula/structure evidence), not direct Class 1 spectral matching against
timsTOF test queries. **This must be tested, not assumed** — see §8 for the first proposed
experiment.

---

## 6. Missing values

Train:
| Column | Nulls | % |
|---|---|---|
| `normalized_smiles`, `inchikey`, `molecular_formula`, `ionization_mode`, `adduct`, `precursor_mz`, `num_peaks` | 0 | 0% |
| `instrument_type` | 33,053 | 1.3% |
| `base_peak_intensity` | 1,163,138 | 45.8% |
| `collision_energy_orig` | 320,464 | 12.6% |

Test: **zero nulls in any column** (all 12 columns fully populated for all 1,213 rows).

`collision_energy_orig_units` in train: `eV` (1,368,908), `NCE` (789,066, i.e. normalized
collision energy — a % of an instrument-dependent reference, NOT directly comparable to eV
without conversion), `unknown` (325,078), `V` (56,556). Test is 100% `eV`. **This means any
feature or filter built on `collision_energy_orig` interpreted at face value would silently
mix eV and NCE scales for ~31% of train** — always use `collision_energy_ev` (already
normalized to eV) instead, unless it too is null.

---

## 7. Duplicate / near-duplicate analysis

- 178,988 groups of exact duplicates in train by `(inchikey14, adduct, precursor_mz,
  num_peaks)`, covering 523,532 rows (~20.6% of train) — i.e. a large fraction of train rows
  are repeated measurements of the same molecule/adduct/precursor/peak-count combination
  (different libraries or acquisitions of the same public spectrum). This needs a real
  near-duplicate check (comparing actual peak lists, not just counts) before it's used for
  anything like a validation split — naive random-row splitting would leak near-identical
  spectra between "train" and "validation".
- No exact duplicate spectra found within test (`(precursor_mz, ms2_mzs)` unique across all
  1,213 rows).
- 519,426 distinct `(inchikey14, adduct)` pairs across all of train — close to the "~500k
  unique structure/adduct spectra" figure in the public baseline claim (§0), which is
  reassuring corroboration of both our own numbers and that public claim.

---

## 8. Precursor / mass ranges

- Train `precursor_mz`: min 2.0 (implausible for a real precursor — likely a data artifact,
  needs inspection), median 350.1.
- Test `precursor_mz`: min 245.1, max not yet tabulated precisely, median 329.2 — broadly
  similar central tendency to train, no obvious gross mismatch.
- `precursor_error_ppm` exists only in train; not present in test (expected — it requires
  knowing the true structure/formula to compute).

---

## 9. Potential leakage — status

We **cannot** directly check train/test structural overlap (i.e. "is the correct answer
already sitting in `train.parquet`?") because `test.parquet` deliberately withholds all
structure identity columns. This is the central open question for class-mix estimation
(§33 of the master prompt) and can only be approached indirectly, e.g. via retrieval
experiments (does a naive library search already find a very high-confidence match for a
given test spectrum?) rather than a direct SQL join. No leakage conclusion should be drawn
yet.

Other leakage-adjacent observations already made:
- Train contains within-library duplication (§7) — relevant to validation-split design, not
  test leakage.
- `enveda-180` (timsTOF) is 97%+ disjoint in molecule identity from the other libraries
  within train itself (§5) — worth remembering when building any structure-disjoint
  validation split (a split that's "structure-disjoint" against the whole 2.5M train set
  might still leak an `enveda-180` molecule's *other* `enveda-180` spectra into validation
  if the split isn't done carefully within-library).

---

## 10. Important observations (summary)

1. This is real public+proprietary spectral library data (GNPS, MassBank, MoNA, RIKEN,
   pluskal_ms2, spectraverse, msdial, drug_plus, masaryk, plus Enveda's own `enveda-180` and
   `enveda-np-examples`), not synthetic.
2. Test is 100% timsTOF and closely matches `enveda-180`'s adduct set and collision-energy
   formatting; `enveda-180` is by far the largest single library (45% of train rows) and is
   almost fully disjoint (structurally) from the rest of train.
3. Multi-spectrum-per-molecule is the norm, not the exception, on both train (avg ~9.2
   spectra/molecule) and test (avg ~3.0 spectra/molecule, 27% of test molecules have >1
   adduct) — multi-spectrum aggregation (§26) is very likely to matter early, not just as a
   late-stage refinement.
4. `collision_energy_orig` mixes eV/NCE/V/unknown units and multiple string formats; always
   use `collision_energy_ev`.
5. `pandas`/`pyarrow` cannot read these parquet files in this environment (pyarrow 19.0.0
   bug); `duckdb` works and is now the standard data-access path (`src/data/loader.py`).
6. `base_peak_intensity` is unreliable (46% null in train, but 0% null in test) — don't
   depend on it for a cross-cutting feature without a null-handling plan.

## 11. Unknowns requiring investigation

1. **Class mix**: what fraction of the 400 test molecules have their correct structure
   already present somewhere in train (any library), and specifically in `enveda-180`? Not
   directly queryable — requires a retrieval experiment (§33 of master prompt).
2. Is the "enveda-180 is the matching library for test" hypothesis (§5) actually true, or
   coincidental? Needs a controlled first experiment (proposed below).
3. What causes `precursor_mz` = 2.0 in train — data error, or a real tiny fragment/adduct
   case? Needs row-level inspection.
4. What causes the `num_peaks` = 73,318 outlier row — real high-resolution spectrum, or a
   parsing artifact? Needs row-level inspection.
5. Real near-duplicate rate in train (peak-list-level, not just metadata-level) — needed
   before designing a leak-safe validation split.
6. Whether `normalized_smiles` is already RDKit-canonical / whether `inchikey14` can be
   trusted as-is for the competition's own tautomer-canonicalization + InChIKey14 matching
   rule, or whether we need to re-canonicalize ourselves to match the grader exactly.
7. Official Kaggle rules/data page text directly (runtime limits, internet policy, external
   data policy) — currently relying on third-party corroboration only (§0).

## 12. Resolved: the two outliers from §11

- **`precursor_mz` = 2.0** (3 rows, all `[M+2H]2+`, `qTof`): a doubly-charged adduct with
  `precursor_mz` = 2.0 is not physically plausible (implies a molecular mass near zero) —
  this is a **data error / placeholder value**, not a real spectrum. Only 52/2,539,608 rows
  (0.002%) have `precursor_mz` < 50 at all, so this is a tiny, droppable population, not a
  systemic issue — but any precursor-based filtering code should not assume `precursor_mz`
  is always physically sane.
- **`num_peaks` = 73,318** (and 2,371 other rows with >5,000 peaks, ~0.09% of train, mostly
  `gnps`/`mona` `qTof`/`QQQ` entries): a molecule with `precursor_mz` ≈ 233 cannot plausibly
  produce tens of thousands of *real* fragments — these are almost certainly raw, unfiltered
  spectra dominated by noise peaks that the source library never cleaned up. This directly
  motivates §23 (spectral preprocessing / noise removal / top-N peak filtering) as something
  we will need for robustness, not just speed — Modified Cosine run on a 73k-peak noisy
  spectrum would both be slow and likely dominated by spurious matches.

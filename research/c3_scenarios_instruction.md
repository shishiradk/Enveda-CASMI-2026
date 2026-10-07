# opencode task: timsTOF Class-3 and PubChem-only query sets (Task S4)

Run with: opencode app, model `opencode/big-pickle`, folder `D:\Enveda-CASMI-2026`. Message:
`Read research/c3_scenarios_instruction.md and do Task S4 exactly as written.`

## Why

We have a calibrated test proxy (`research/analysis/c3_calibration_report.md`). Two of its buckets have no timsTOF data:
- **PubChem-only:** the truth is in PubChem but not in COCONUT.
- **Class 3:** the truth is in neither PubChem nor COCONUT.

The existing S3 scenario has GNPS spectra, but the hidden test is 100% Bruker timsTOF. We need timsTOF query sets for
both buckets, drawn from the `enveda-180` library of `train.parquet`.

## Inputs (all local, read-only)

- `train.parquet`: 2.54M spectra. Columns include `ingest_lib`, `inchikey14`, `normalized_smiles`,
  `molecular_formula`, `adduct`, `precursor_mz`, `instrument_type`, `ms2_mzs`, `ms2_normalized_intensities`,
  `collision_energy_ev`, `ionization_mode`.
- `results/c3/train_classes.parquet`: one row per train `inchikey14`, with columns `ik, smiles, formula, libs, n_spec,
  n_tims, in_coco, in_pc`. Here `in_pc` is a **raw** InChIKey14 match against PubChem; tautomers are not reconciled.
- `external/pubchem/pubchem_rows.parquet`: 100.9M rows, columns `ik, cid, smiles, mass`. Use DuckDB only; never load
  it into pandas.
- `results/kaggle_v1_assets/universe.parquet`: columns `ik, smiles, mass, src`, where `src = 'coconut'` marks COCONUT
  structures.
- `results/bench/queries_S12.parquet`: the format to copy. Read its schema and copy its columns exactly.

## What to build

1. **Candidates.** Take molecules with `'enveda-180'` in `libs` and `n_tims >= 1`, as two sets:
   - **C3 candidates:** `in_coco = false` AND `in_pc = false` (about 5,240);
   - **PC candidates:** `in_coco = false` AND `in_pc = true`.
2. **Tautomer verification of the C3 candidates.** This is the metric's key. For each C3 candidate:
   - compute the canonical-tautomer InChIKey14 with RDKit: `rdMolStandardize.TautomerEnumerator().Canonicalize(mol)`,
     then `Chem.MolToInchiKey(...)[:14]`;
   - also do this for every PubChem and COCONUT structure with the **same molecular formula** (PubChem rows within
     ±0.003 Da of the candidate's monoisotopic mass are enough to find them);
   - drop the candidate if any of those has the same canonical-tautomer key.

   Report how many were dropped. Use multiprocessing, wrapped in `if __name__ == "__main__":` (Windows spawn). Use at
   most 4 processes and keep memory under 6 GB.
3. **Sampling.** Use seed `20261003`. Sample **300 molecules** from the verified C3 set and **300** from the PC set,
   stratified into 4 precursor-mass quartiles. Keep only molecules with at least one `[M+H]+` or `[M-H]-` spectrum.
4. **Query files.** For each sampled molecule, take **all** its enveda-180 timsTOF spectra, up to 16 per molecule. Use
   the same columns as `results/bench/queries_S12.parquet`, with `molecule_id` = the inchikey14. Write:
   - `results/c3/queries_S4C3.parquet`
   - `results/c3/queries_S4PC.parquet`
5. **Truth and hold-out keys.** Write `results/c3/truth_S4.parquet` with columns `qset, mid, smiles, formula, correct`.
   `correct` is a `;`-joined set of: the inchikey14, plus its canonical-tautomer key if different. Use the same format
   as `results/bench/truth.parquet`. Also write `results/c3/held_S4.parquet` (column `ik`): every sampled key, plus
   any train inchikey14 that shares its canonical-tautomer key.
6. **Report.** Write `research/analysis/c3_s4_scenarios.md` (under 80 lines) covering:
   - all counts at every step;
   - how many C3 candidates the tautomer check removed;
   - the mass, adduct and spectra-per-molecule distributions of each set, compared with `queries_S12.parquet`;
   - 10 example SMILES per set;
   - a natural-product-likeness note: the share of each set with RDKit's NP-likeness score > 0, from
     `rdkit.Contrib.NP_Score` if available, otherwise say "not computed".

   Label every claim **FACT** (measured, with the script name) or **INFERENCE**.

## Rules

- **Do not modify** any existing file. New files only, and only under `results/c3/`, `research/scripts/c3_s4_*.py`
  and `research/analysis/c3_s4_scenarios.md`.
- **Do not run** `research/bench/bench.py` or any `.bat` file.
- Keep each script under 250 lines, with a docstring saying what it does and how to run it.
- If the tautomer step over the PubChem same-formula sets is too slow (over 1 hour), restrict it to the
  same-formula structures within ±0.003 Da and say so in the report.

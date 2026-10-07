"""Scenario SV ("re-measured library hit"): the 400 visible test molecules, queried with their own test spectra,
against a library from which only the exact duplicates of those spectra are removed.

    python research/scripts/sv_prepare.py

Why: the visible test spectra are exact enveda-180 train rows, so an unmasked run sees library similarity ~1.0 and
cannot test any gate on library similarity (E5 failed on the leaderboard for exactly this reason, DEC-008). With
the duplicates masked, each molecule keeps only its *other* enveda-180 spectra, as a hidden-test molecule re-measured
in an extract would. V2 never uses enveda-180 references, so its existing lists for these queries apply unchanged.

Writes results/bench/sv_dup_rows.npy (train file_row_numbers to mask) and results/bench/truth_SV.parquet
(qset 'SV', mid, smiles, formula, correct = inchikey14 [+ canonical-tautomer key if different]).
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "bench"
TE, TR = (ROOT / "test.parquet").as_posix(), (ROOT / "train.parquet").as_posix()


def main():
    d = duckdb.sql(f"""
        WITH t AS (SELECT molecule_id, precursor_mz, adduct, ms2_mzs FROM read_parquet('{TE}')),
             r AS (SELECT file_row_number AS row, inchikey14, normalized_smiles, molecular_formula,
                          precursor_mz, adduct, ms2_mzs
                   FROM read_parquet('{TR}', file_row_number = true)
                   WHERE ingest_lib = 'enveda-180' AND precursor_mz IN (SELECT precursor_mz FROM t))
        SELECT t.molecule_id, r.row, r.inchikey14, r.normalized_smiles, r.molecular_formula
        FROM t JOIN r USING (precursor_mz, adduct) WHERE r.ms2_mzs = t.ms2_mzs""").df()
    n_test = duckdb.sql(f"SELECT count(*) FROM read_parquet('{TE}')").fetchone()[0]
    assert d.molecule_id.nunique() == 400, d.molecule_id.nunique()
    assert d.groupby("molecule_id").inchikey14.nunique().max() == 1, "a test molecule maps to two structures"
    rows = np.unique(d.row.values.astype(np.int64))
    np.save(OUT / "sv_dup_rows.npy", rows)
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()
    truth = []
    for mid, g in d.groupby("molecule_id"):
        ik, smi, f = g.inchikey14.iloc[0], g.normalized_smiles.iloc[0], g.molecular_formula.iloc[0]
        keys = {ik}
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is not None:
            keys.add(Chem.MolToInchiKey(te.Canonicalize(m))[:14])
        truth.append(dict(qset="SV", mid=mid, smiles=smi, formula=f, correct=";".join(sorted(keys))))
    pd.DataFrame(truth).to_parquet(OUT / "truth_SV.parquet", index=False)
    other = duckdb.sql(f"""SELECT count(DISTINCT inchikey14) FROM read_parquet('{TR}', file_row_number = true)
                           WHERE inchikey14 IN (SELECT inchikey14 FROM d) AND file_row_number NOT IN
                           (SELECT row FROM d)""").fetchone()[0]
    print(f"test spectra {n_test}, duplicate train rows {len(rows)}, molecules {d.molecule_id.nunique()}, "
          f"molecules with other train spectra left {other}")


if __name__ == "__main__":
    main()

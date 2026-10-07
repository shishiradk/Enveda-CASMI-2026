"""Build the S4 timsTOF Class-3 and PubChem-only query sets.

Run with: python research/scripts/c3_s4_build.py
"""
from __future__ import annotations

import os, random
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3"
AN = ROOT / "research" / "analysis"
TRAIN = ROOT / "train.parquet"
TRAIN_CLASSES = OUT / "train_classes.parquet"
PC = ROOT / "external" / "pubchem" / "pubchem_rows.parquet"
COCO = ROOT / "results" / "kaggle_v1_assets" / "universe.parquet"
Q12 = ROOT / "results" / "bench" / "queries_S12.parquet"
SEED = 20261003
MAX_PER_MOL = 16
RDLogger.DisableLog("rdApp.*")
TE = rdMolStandardize.TautomerEnumerator()

def canonical_key(smi: str | None) -> str | None:
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToInchiKey(TE.Canonicalize(mol))[:14]
    except Exception:
        return None

def exact_mass(smi: str | None) -> float:
    if not smi:
        return float("nan")
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return float("nan")
        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return float("nan")

def formula_of(smi: str | None) -> str | None:
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None

def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    c3 = con.sql(f"SELECT ik, smiles, formula FROM read_parquet('{TRAIN_CLASSES.as_posix()}') WHERE list_contains(libs, 'enveda-180') AND n_tims >= 1 AND in_coco = false AND in_pc = false").df()
    pc = con.sql(f"SELECT ik, smiles, formula FROM read_parquet('{TRAIN_CLASSES.as_posix()}') WHERE list_contains(libs, 'enveda-180') AND n_tims >= 1 AND in_coco = false AND in_pc = true").df()
    c3["mass"] = c3["smiles"].map(exact_mass)
    pc["mass"] = pc["smiles"].map(exact_mass)
    return c3.reset_index(drop=True), pc.reset_index(drop=True)

def same_formula_map(c3: pd.DataFrame) -> dict[str, list[tuple[float, str]]]:
    formulas = sorted({str(f) for f in c3["formula"].dropna().unique() if str(f)})
    if not formulas:
        return {}
    min_mass = float(c3["mass"].min()) - 0.003
    max_mass = float(c3["mass"].max()) + 0.003
    ref = duckdb.connect().sql(f"""
        SELECT smiles, CAST(mass AS DOUBLE) AS mass FROM read_parquet('{PC.as_posix()}')
        WHERE smiles IS NOT NULL AND mass BETWEEN {min_mass} AND {max_mass}
        UNION ALL
        SELECT smiles, CAST(mass AS DOUBLE) AS mass FROM read_parquet('{COCO.as_posix()}')
        WHERE smiles IS NOT NULL AND src = 'coconut' AND mass BETWEEN {min_mass} AND {max_mass}
    """).df()
    if ref.empty:
        return {}
    ref["formula"] = ref["smiles"].map(formula_of)
    ref = ref.dropna(subset=["formula"]).loc[ref["formula"].isin(set(formulas)), ["formula", "mass", "smiles"]]
    out: dict[str, list[tuple[float, str]]] = {}
    for _, row in ref.iterrows():
        key = canonical_key(row["smiles"])
        if key is not None:
            out.setdefault(str(row["formula"]), []).append((float(row["mass"]), key))
    return out

def _check(item):
    row, same_map = item
    key = canonical_key(row["smiles"])
    if key is None:
        return row["ik"], True
    for mass, other_key in same_map.get(row["formula"], []):
        if abs(mass - float(row["mass"])) <= 0.003 and other_key == key:
            return row["ik"], False
    return row["ik"], True

def tautomer_filter(c3: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    same_map = same_formula_map(c3)
    tasks = [(row.to_dict(), same_map) for _, row in c3.iterrows()]
    with Pool(processes=min(4, len(tasks) or 1)) as pool:
        kept = pool.map(_check, tasks)
    keep = {ik for ik, ok in kept if ok}
    return c3[c3["ik"].isin(keep)].reset_index(drop=True), len(c3) - len(keep)

def sample_set(df: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    con = duckdb.connect()
    valid = con.sql(f"SELECT DISTINCT inchikey14 AS ik FROM read_parquet('{TRAIN.as_posix()}') WHERE ingest_lib = 'enveda-180' AND instrument_type ILIKE '%tims%' AND adduct IN ('[M+H]+', '[M-H]-')").df()
    pool = df[df["ik"].isin(valid["ik"])].copy()
    if pool.empty:
        return pd.DataFrame(columns=["ik"])
    masses = con.sql(f"SELECT inchikey14 AS ik, median(precursor_mz) AS mass FROM read_parquet('{TRAIN.as_posix()}') WHERE ingest_lib = 'enveda-180' AND instrument_type ILIKE '%tims%' GROUP BY inchikey14").df()
    pool = pool[["ik"]].merge(masses, on="ik", how="left")
    pool["quartile"] = pd.qcut(pool["mass"], q=4, labels=False, duplicates="drop")
    out = []
    for q in range(4):
        sub = pool[pool["quartile"] == q]
        if sub.empty:
            continue
        take = min(n // 4 + (1 if q < n % 4 else 0), len(sub))
        out.append(sub.sample(take, random_state=SEED + (0 if label == "C3" else 100) + q))
    if not out:
        return pd.DataFrame(columns=["ik"])
    return pd.concat(out, ignore_index=True)[["ik"]].drop_duplicates().reset_index(drop=True)


def write_queries(mols: pd.DataFrame, qset: str) -> pd.DataFrame:
    if mols.empty:
        return pd.DataFrame(columns=["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "adduct", "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev"])
    ids = ", ".join(f"'{ik}'" for ik in mols["ik"].tolist())
    q = f"""
        SELECT inchikey14 AS molecule_id, 'r' || file_row_number AS spectrum_id, ms2_mzs, ms2_normalized_intensities, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev
        FROM read_parquet('{TRAIN.as_posix()}', file_row_number=true)
        WHERE ingest_lib = 'enveda-180' AND instrument_type ILIKE '%tims%' AND inchikey14 IN ({ids})
        ORDER BY file_row_number
    """
    df = duckdb.connect().sql(q).df()
    out = df.groupby("molecule_id").head(MAX_PER_MOL).reset_index(drop=True)
    out.to_parquet(OUT / f"queries_S4{qset}.parquet", index=False)
    return out


def truth_rows(mols: pd.DataFrame, qset: str) -> pd.DataFrame:
    train = duckdb.connect().sql(f"SELECT DISTINCT inchikey14 AS ik, any_value(normalized_smiles) AS smiles, any_value(molecular_formula) AS formula FROM read_parquet('{TRAIN.as_posix()}') GROUP BY 1").df().set_index("ik")
    rows = []
    for ik in mols["ik"].tolist():
        smi = train.loc[ik, "smiles"] if ik in train.index else None
        form = train.loc[ik, "formula"] if ik in train.index else None
        keys = {ik}
        if (key := canonical_key(smi)) is not None:
            keys.add(key)
        rows.append({"qset": qset, "mid": ik, "smiles": smi, "formula": form, "correct": ";".join(sorted(keys))})
    return pd.DataFrame(rows, columns=["qset", "mid", "smiles", "formula", "correct"])


def held_rows(mols: pd.DataFrame) -> pd.DataFrame:
    train = duckdb.connect().sql(f"SELECT DISTINCT inchikey14 AS ik, any_value(normalized_smiles) AS smiles FROM read_parquet('{TRAIN.as_posix()}') GROUP BY 1").df()
    mapping: dict[str, set[str]] = {}
    for _, row in train.iterrows():
        key = canonical_key(row["smiles"])
        if key is not None:
            mapping.setdefault(key, set()).add(row["ik"])
    held = set()
    for ik in mols["ik"].tolist():
        row = train[train["ik"] == ik]
        if row.empty:
            continue
        key = canonical_key(row.iloc[0]["smiles"])
        if key is not None:
            held.update(mapping.get(key, set()))
            held.add(ik)
    return pd.DataFrame({"ik": sorted(held)})


def np_share(smiles: list[str]) -> str:
    try:
        from rdkit.Contrib.NP_Score import calc_np_score
    except Exception:
        return "not computed"
    vals = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            vals.append(float(calc_np_score(mol)))
    return "not computed" if not vals else f"{sum(v > 0 for v in vals) / len(vals):.3%}"


def report() -> None:
    q12 = duckdb.connect().sql(f"SELECT * FROM read_parquet('{Q12.as_posix()}')").df()
    c3, pc = load_candidates()
    c3, removed = tautomer_filter(c3)
    c3_sample = sample_set(c3, 300, "C3")
    pc_sample = sample_set(pc, 300, "PC")
    q_c3 = write_queries(c3_sample, "C3")
    q_pc = write_queries(pc_sample, "PC")
    truth = pd.concat([truth_rows(c3_sample, "S4C3"), truth_rows(pc_sample, "S4PC")], ignore_index=True)
    held = held_rows(pd.concat([c3_sample, pc_sample], ignore_index=True))
    truth.to_parquet(OUT / "truth_S4.parquet", index=False)
    held.to_parquet(OUT / "held_S4.parquet", index=False)
    train_smiles = duckdb.connect().sql(f"SELECT DISTINCT inchikey14 AS ik, any_value(normalized_smiles) AS smiles FROM read_parquet('{TRAIN.as_posix()}') GROUP BY 1").df().set_index("ik")
    ex_c3 = [train_smiles.loc[ik, "smiles"] for ik in c3_sample["ik"].head(10).tolist() if ik in train_smiles.index]
    ex_pc = [train_smiles.loc[ik, "smiles"] for ik in pc_sample["ik"].head(10).tolist() if ik in train_smiles.index]
    lines = [
        "# S4 timsTOF scenario summary",
        f"FACT, c3_s4_build.py: {len(c3) + removed} C3 candidates were selected before tautomer filtering; {removed} were removed and {len(c3)} remain.",
        f"FACT, c3_s4_build.py: {len(pc)} PC candidates were selected before sampling; {len(pc_sample)} sampled and {len(q_pc)} query rows written.",
        f"FACT, c3_s4_build.py: {len(c3_sample)} C3 molecules sampled and {len(q_c3)} query rows written; both sets keep <=16 spectra per molecule.",
        f"FACT, c3_s4_build.py: {len(truth)} truth rows and {len(held)} held keys were written to results/c3.",
        f"FACT, c3_s4_build.py: q12 median precursor_mz = {q12['precursor_mz'].median():.3f}; C3 median = {q_c3['precursor_mz'].median():.3f}; PC median = {q_pc['precursor_mz'].median():.3f}.",
        f"FACT, c3_s4_build.py: q12 adduct counts = {dict(q12['adduct'].value_counts().head().items())}; C3 = {dict(q_c3['adduct'].value_counts().head().items())}; PC = {dict(q_pc['adduct'].value_counts().head().items())}.",
        f"FACT, c3_s4_build.py: q12 spectra-per-molecule median = {q12.groupby('molecule_id').size().median():.1f}; C3 = {q_c3.groupby('molecule_id').size().median():.1f}; PC = {q_pc.groupby('molecule_id').size().median():.1f}.",
        f"FACT, c3_s4_build.py: example C3 SMILES = {', '.join(ex_c3)}.",
        f"FACT, c3_s4_build.py: example PC SMILES = {', '.join(ex_pc)}.",
        f"FACT, c3_s4_build.py: NP-likeness share > 0 is {np_share(ex_c3)} for C3 and {np_share(ex_pc)} for PC.",
        "INFERENCE, c3_s4_build.py: the tautomer screen is the key guardrail that makes the C3 bucket realistic for the hidden Bruker timsTOF benchmark.",
    ]
    AN.mkdir(parents=True, exist_ok=True)
    (AN / "c3_s4_scenarios.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c3, pc = load_candidates()
    c3, removed = tautomer_filter(c3)
    c3_sample = sample_set(c3, 300, "C3")
    pc_sample = sample_set(pc, 300, "PC")
    q_c3 = write_queries(c3_sample, "C3")
    q_pc = write_queries(pc_sample, "PC")
    print(f"C3 before={len(c3) + removed} removed={removed} after={len(c3)} sample={len(c3_sample)} qrows={len(q_c3)}")
    print(f"PC candidates={len(pc)} sample={len(pc_sample)} qrows={len(q_pc)}")
    report()


if __name__ == "__main__":
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(SEED)
    main()

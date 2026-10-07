#!/usr/bin/env python
"""Build a same-formula isomer benchmark from train.parquet for the forward-model check.

For N random enveda-180 (timsTOF) molecules with [M+H]+ spectra and neutral mass
200-700 Da: candidates = true structure + up to 20 isomers with the identical
molecular formula (10 most similar by Morgan Tanimoto + 10 random), drawn from
results/kaggle_v1_assets/universe.parquet and external/pubchem/pubchem_rows.parquet.
Candidates are de-duplicated on the InChIKey first block (ICEBERG ignores stereo).
The candidate list is shuffled with a fixed seed; the truth index is stored
separately (results/forward_model/val_truth.json), never in the runner input.

Reads parquet with DuckDB only.
"""
import argparse
import json
import random
from pathlib import Path

import duckdb
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OK_ELEMENTS = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Se", "As", "Na", "K"}


def info(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or "." in smi:
        return None
    if Chem.GetFormalCharge(mol) != 0 or any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return None
    if any(a.GetSymbol() not in OK_ELEMENTS or a.GetIsotope() != 0 for a in mol.GetAtoms()):
        return None
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not ik:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return {"formula": CalcMolFormula(mol), "ik14": ik[:14], "fp": fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--n-similar", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--max-spectra", type=int, default=4)
    ap.add_argument("--pool", choices=["both", "universe"], default="both",
                    help="'universe' = decoys only from universe.parquet (train + COCONUT structures): harder, closer analogues")
    ap.add_argument("--min-decoys", type=int, default=5)
    ap.add_argument("--pool-cap", type=int, default=800, help="max isomer rows inspected per molecule (random subset)")
    ap.add_argument("--prefix", default="val")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "forward_model"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("set threads=4; set memory_limit='4GB'")
    train = str(ROOT / "train.parquet")
    uni = str(ROOT / "results" / "kaggle_v1_assets" / "universe.parquet")
    pub = str(ROOT / "external" / "pubchem" / "pubchem_rows.parquet")
    msg = str(ROOT / "external" / "forward_model" / "msg15_meta.parquet")

    msg_fold = {}
    for ik, fold in con.execute(f"select inchikey, fold from '{msg}' group by all").fetchall():
        msg_fold.setdefault(ik, set()).add(fold)

    mols = con.execute(f"""
        select inchikey14, any_value(normalized_smiles) smi, avg(precursor_mz) - 1.007276 mass
        from '{train}' where ingest_lib = 'enveda-180' and adduct = '[M+H]+' and ionization_mode = 'positive'
        group by 1 having mass between 200 and 700 order by 1""").fetchall()
    print("eligible molecules", len(mols))
    rng.shuffle(mols)

    cands_json, truth = {}, {}
    for ik14, smi, mass in mols:
        if len(cands_json) >= args.n:
            break
        t = info(smi)
        if t is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        exact = Chem.Descriptors.ExactMolWt(mol) if hasattr(Chem, "Descriptors") else None
        from rdkit.Chem.Descriptors import ExactMolWt
        exact = ExactMolWt(mol)
        lo, hi = exact - 0.0006, exact + 0.0006
        rows = con.execute(f"select ik, smiles, 'universe:' || src from '{uni}' where mass between ? and ?", [lo, hi]).fetchall()
        if args.pool == "both":
            rows += con.execute(f"select ik, smiles, 'pubchem' from '{pub}' where mass between ? and ? limit 12000", [lo, hi]).fetchall()
        rng.shuffle(rows)
        seen, pool = {t["ik14"], ik14}, []
        for ik, s, src in rows[: args.pool_cap]:
            if not ik or ik[:14] in seen:
                continue
            d = info(s)
            if d is None or d["formula"] != t["formula"] or d["ik14"] in seen:
                continue
            seen.add(d["ik14"])
            seen.add(ik[:14])
            pool.append({"smiles": s, "src": src, "ik14": d["ik14"],
                         "tani": float(DataStructs.TanimotoSimilarity(t["fp"], d["fp"]))})
        if len(pool) < args.min_decoys:
            continue
        by_sim = sorted(pool, key=lambda r: -r["tani"])
        similar = by_sim[: args.n_similar]
        rest = by_sim[args.n_similar:]
        rng.shuffle(rest)
        rand = rest[: args.n_random]
        for r in similar:
            r["kind"] = "similar"
        for r in rand:
            r["kind"] = "random"
        items = [{"smiles": smi, "src": "truth", "ik14": t["ik14"], "tani": 1.0, "kind": "truth"}] + similar + rand
        rng.shuffle(items)
        spectra = con.execute(f"""
            select ms2_mzs, ms2_normalized_intensities, precursor_mz, collision_energy_ev
            from '{train}' where ingest_lib = 'enveda-180' and adduct = '[M+H]+' and inchikey14 = ?
            order by len(collision_energy_ev), collision_energy_ev[1]""", [ik14]).fetchall()
        # one spectrum per distinct collision-energy setting, at most max_spectra
        uniq, chosen = set(), []
        for mzs, its, pmz, ces in spectra:
            key = tuple(ces or [])
            if key in uniq:
                continue
            uniq.add(key)
            chosen.append({"mzs": list(mzs), "intensities": list(its), "adduct": "[M+H]+",
                           "precursor_mz": pmz, "collision_energies": list(ces) if ces else None})
        chosen = chosen[: args.max_spectra]
        mid = f"{args.prefix}_{ik14}"
        cands_json[mid] = {"smiles": [r["smiles"] for r in items], "adduct": "[M+H]+", "spectra": chosen}
        folds = sorted(msg_fold.get(t["ik14"], set()) | msg_fold.get(ik14, set()))
        truth[mid] = {"truth_index": [r["kind"] for r in items].index("truth"), "formula": t["formula"], "mass": exact,
                      "kinds": [r["kind"] for r in items], "src": [r["src"] for r in items],
                      "tani": [round(r["tani"], 4) for r in items], "ik14": [r["ik14"] for r in items],
                      "msg_folds": folds, "n_isomers_pool": len(pool), "n_spectra": len(chosen)}
        if len(cands_json) % 20 == 0:
            print(len(cands_json), "built", flush=True)

    with open(out_dir / f"{args.prefix}_input.json", "w") as fh:
        json.dump(cands_json, fh)
    with open(out_dir / f"{args.prefix}_truth.json", "w") as fh:
        json.dump(truth, fh)
    n_c = [len(v["smiles"]) for v in cands_json.values()]
    print("molecules", len(cands_json), "candidates", sum(n_c), "mean", np.mean(n_c),
          "in MassSpecGym", sum(1 for v in truth.values() if v["msg_folds"]),
          "in MSG train fold", sum(1 for v in truth.values() if "train" in v["msg_folds"]))


if __name__ == "__main__":
    main()

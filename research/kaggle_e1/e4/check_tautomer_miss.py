"""How many pool structures without a PubChem match by plain InChIKey14 are in PubChem as another tautomer?

    python research/kaggle_e1/e4/check_tautomer_miss.py [n_sample=300] [workers=3]

Needs results/kaggle_e4/pool_pop_rows.parquet (build_lookup.py).  For a random sample of unmatched pool structures
(and, as a control, of matched ones) every PubChem row of the same monoisotopic mass (+-1e-4 Da) is
tautomer-canonicalised with the local RDKit and compared with the tautomer-canonical key of the pool structure.
Also reports how often the pool key equals the tautomer-canonical key of the pool SMILES.
Writes results/kaggle_e4/tautomer_miss.json.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "kaggle_e4"
PUB = ROOT / "external" / "pubchem" / "pubchem_rows_pop.parquet"
_g = {}


def canon(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g:
        _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:  # noqa: BLE001
        return None


def mass(smi):
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    m = Chem.MolFromSmiles(smi)
    return float(ExactMolWt(m)) if m is not None else float("nan")


def main():
    import duckdb
    import pandas as pd
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t0 = time.time()
    df = pd.read_parquet(OUT / "pool_pop_rows.parquet").drop_duplicates("key")
    rng = np.random.default_rng(0)
    res = {}
    con = duckdb.connect()
    con.execute("SET threads=3; SET memory_limit='4GB'")
    with Pool(workers) as mp:
        gen = df.sample(3000, random_state=0)
        ck = mp.map(canon, list(gen.smiles), chunksize=50)
        res["pool_key_equals_canonical_key_of_pool_smiles"] = round(float(np.mean([a == b for a, b in zip(ck, gen.key)])), 4)
        print(res, f"{time.time() - t0:.0f}s", flush=True)
        for name, sub in (("unmatched", df[df.g_cid == 0]), ("matched_control", df[df.g_cid > 0])):
            s = sub.iloc[rng.choice(len(sub), min(n, len(sub)), replace=False)].copy()
            s["mass"] = mp.map(mass, list(s.smiles), chunksize=50)
            s = s[np.isfinite(s.mass) & (s.mass < 900)]          # canonicalisation of very large isomer sets is too slow
            s["ck"] = mp.map(canon, list(s.smiles), chunksize=20)
            con.register("q", s[["key", "mass"]])
            rows = con.execute(f"SELECT q.key AS key, p.ik AS ik, p.smiles AS smiles, p.n_sid AS n_sid, p.n_pmid AS n_pmid "
                               f"FROM q JOIN '{PUB.as_posix()}' p ON p.mass BETWEEN q.mass - 1e-4 AND q.mass + 1e-4").df()
            con.unregister("q")
            per = rows.groupby("key").size()
            keep = set(per[per <= 4000].index)                  # skip the few structures with huge isomer sets
            rows = rows[rows.key.isin(keep)]
            s = s[s.key.isin(keep) | ~s.key.isin(set(per.index))]
            us = rows.smiles.drop_duplicates().tolist()
            print(name, "structures", len(s), "pubchem isomer rows", len(rows), "distinct smiles", len(us), f"{time.time() - t0:.0f}s", flush=True)
            c = dict(zip(us, mp.map(canon, us, chunksize=100)))
            rows["ck"] = rows.smiles.map(c)
            want = dict(zip(s.key, s.ck))
            hit = rows[rows.ck == rows.key.map(want)]
            direct = hit[hit.ik == hit.key]
            extra = hit[hit.ik != hit.key]
            g = extra.groupby("key").agg(n_sid=("n_sid", "sum"), n_pmid=("n_pmid", "sum"), n=("ik", "size"))
            res[name] = {"structures": int(len(s)), "isomer_rows": int(len(rows)),
                         "structures_with_a_pubchem_row_under_another_plain_key": int(len(g)),
                         "share": round(len(g) / max(len(s), 1), 4),
                         "extra_cids": int(g.n.sum()) if len(g) else 0, "extra_n_sid_sum": int(g.n_sid.sum()) if len(g) else 0,
                         "extra_n_sid_median": float(g.n_sid.median()) if len(g) else 0.0,
                         "extra_n_pmid_sum": int(g.n_pmid.sum()) if len(g) else 0,
                         "direct_cids": int(len(direct)), "direct_n_sid_sum": int(direct.n_sid.sum()),
                         "examples": [(k, int(v.n_sid), int(v.n_pmid)) for k, v in g.head(8).iterrows()]}
            print(name, res[name], flush=True)
    res["sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "tautomer_miss.json", "w"), indent=1)


if __name__ == "__main__":
    main()

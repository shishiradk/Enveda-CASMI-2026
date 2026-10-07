"""PubChem popularity lookup for the E4 engine pool: key -> (n_sid, n_pmid).

    python research/kaggle_e1/e4/build_lookup.py [workers=3]

Pool = what casmi_engine.build_pool sees: COCONUT (prvsiyan/coconut-casmi26-candidates), our ChEBI / LIPID MAPS table
(casmi-bio-clean) and the structures of train.parquet (read with DuckDB only).

Join (documented in research/analysis/e4_popularity.md):
  * pubchem_rows_pop.parquet has one row per CID with ``ik`` = plain InChIKey first block of the PubChem structure.
  * every pool structure belongs to one group = its pool key (the engine's tautomer-canonical InChIKey14).
  * probes of a group = the pool key itself + the plain InChIKey14 of every pool SMILES of the group (computed here).
  * counts of a group = sum of n_sid / n_pmid over all CIDs whose ``ik`` is one of the group's probes.
  * the table holds one row per probe (pool keys and plain keys), with the group's counts; zero-count rows are kept so
    the run can tell "no PubChem record" from "structure unknown to the table".
Output: results/kaggle_e4/pop_lookup/pop_lookup.npz (keys S14 sorted, n_sid int32, n_pmid int32) + coverage.json.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "kaggle_e4" / "pop_lookup"
PUB = ROOT / "external" / "pubchem" / "pubchem_rows_pop.parquet"
COCO = ROOT / "external" / "bench_inputs" / "coco" / "coco_meta.pkl"
BIO = ROOT / "results" / "structdb" / "dropin" / "bio_meta.pkl"
TRAIN = ROOT / "train.parquet"


def plain_ik14(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi)
        k = Chem.MolToInchiKey(m) if m is not None else ""
        return k[:14] if k else ""
    except Exception:  # noqa: BLE001
        return ""


def quant(v):
    v = np.asarray(v)
    if not len(v):
        return {}
    q = np.percentile(v, [10, 25, 50, 75, 90, 99])
    return {"p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p90": float(q[4]),
            "p99": float(q[5]), "max": int(v.max()), "mean": round(float(v.mean()), 2)}


def main():
    import duckdb
    import pandas as pd
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cm = pickle.load(open(COCO, "rb"))
    bm = pickle.load(open(BIO, "rb"))
    con = duckdb.connect()
    con.execute("SET threads=3; SET memory_limit='5GB'")
    tr = con.execute(f"SELECT DISTINCT inchikey14 AS key, normalized_smiles AS smiles FROM '{TRAIN.as_posix()}' "
                     "WHERE inchikey14 IS NOT NULL AND normalized_smiles IS NOT NULL").df()
    df = pd.concat([pd.DataFrame({"key": np.asarray(cm["keys"], dtype=object), "smiles": np.asarray(cm["smiles"], dtype=object), "src": "coconut"}),
                    pd.DataFrame({"key": np.asarray(bm["keys"], dtype=object), "smiles": np.asarray(bm["smiles"], dtype=object), "src": "bio"}),
                    tr.assign(src="train")], ignore_index=True)
    df = df[df.key.map(lambda k: isinstance(k, str) and len(k) == 14)].reset_index(drop=True)
    print("pool rows", len(df), df.src.value_counts().to_dict(), "distinct keys", df.key.nunique(), flush=True)

    us = df.smiles.drop_duplicates().tolist()
    with Pool(workers) as mp:
        pk = mp.map(plain_ik14, us, chunksize=2000)
    s2p = dict(zip(us, pk))
    df["plain"] = df.smiles.map(s2p)
    print(f"plain keys done {time.time() - t0:.0f}s | failed {(df.plain == '').sum()} | plain == pool key {(df.plain == df.key).mean():.4f}", flush=True)

    probes = pd.DataFrame({"k": sorted(set(df.key) | (set(df.plain) - {""}))})
    con.register("probes", probes)
    pc = con.execute(f"SELECT ik, SUM(n_sid)::BIGINT AS n_sid, SUM(n_pmid)::BIGINT AS n_pmid, COUNT(*) AS n_cid "
                     f"FROM '{PUB.as_posix()}' WHERE ik IN (SELECT k FROM probes) GROUP BY ik").df()
    print(f"pubchem rows matched: {len(pc)} keys, {int(pc.n_cid.sum())} CIDs, {time.time() - t0:.0f}s", flush=True)
    sid, pmid, ncid = (dict(zip(pc.ik, pc[c])) for c in ("n_sid", "n_pmid", "n_cid"))

    # group = pool key; probes of the group = pool key + plain keys of its SMILES
    grp = {}
    for k, p in zip(df.key.values, df.plain.values):
        g = grp.setdefault(k, {k})
        if p:
            g.add(p)
    gs = {k: sum(sid.get(x, 0) for x in g) for k, g in grp.items()}
    gp = {k: sum(pmid.get(x, 0) for x in g) for k, g in grp.items()}
    gc = {k: sum(ncid.get(x, 0) for x in g) for k, g in grp.items()}
    table = {}
    for k, g in grp.items():
        for x in g:
            old = table.get(x)
            if old is None or gs[k] + gp[k] > old[0] + old[1]:      # a probe shared by two groups takes the larger one
                table[x] = (gs[k], gp[k])
    keys = np.array(sorted(table), dtype="S14")
    n_sid = np.array([min(table[k.decode()][0], 2**31 - 1) for k in keys], np.int32)
    n_pmid = np.array([min(table[k.decode()][1], 2**31 - 1) for k in keys], np.int32)
    assert (np.diff(np.argsort(keys, kind="mergesort")) == 1).all() and len(np.unique(keys)) == len(keys)
    np.savez_compressed(OUT / "pop_lookup.npz", keys=keys, n_sid=n_sid, n_pmid=n_pmid)

    # ---- coverage ------------------------------------------------------------------------------------
    df["g_sid"] = df.key.map(gs); df["g_pmid"] = df.key.map(gp); df["g_cid"] = df.key.map(gc)
    df["hit_plain"] = df.plain.map(lambda x: x in ncid)
    df["hit_key"] = df.key.map(lambda x: x in ncid)
    df.to_parquet(OUT.parent / "pool_pop_rows.parquet", index=False)
    cov = {"pubchem_rows": str(PUB.name), "table_rows": int(len(keys)), "table_rows_with_record": int(((n_sid > 0) | (n_pmid > 0)).sum()),
           "npz_bytes": (OUT / "pop_lookup.npz").stat().st_size, "plain_key_failed": int((df.plain == "").sum()),
           "plain_equals_pool_key": round(float((df.plain == df.key).mean()), 4), "by_source": {}}
    for name, d in list(df.groupby("src")) + [("all", df)]:
        u = d.drop_duplicates("key")
        has = u.g_cid > 0
        cov["by_source"][name] = {
            "structures": int(len(u)), "with_pubchem_record": round(float(has.mean()), 4),
            "matched_by_plain_key_of_smiles": round(float(d.groupby("key").hit_plain.any().mean()), 4),
            "matched_by_pool_key_only": round(float((d.groupby("key").hit_key.any() & ~d.groupby("key").hit_plain.any()).mean()), 4),
            "with_pubmed_link": round(float((u.g_pmid > 0).mean()), 4),
            "n_sid_le_3_among_matched": round(float((u.g_sid[has] <= 3).mean()), 4),
            "n_cid_per_matched_group": quant(u.g_cid[has]), "n_sid_matched": quant(u.g_sid[has]), "n_pmid_matched": quant(u.g_pmid[has]),
            "pop_all": quant(np.log1p(u.g_sid) + np.log1p(u.g_pmid))}
    cov["sec"] = round(time.time() - t0, 1)
    json.dump(cov, open(OUT.parent / "coverage.json", "w"), indent=1)
    print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()

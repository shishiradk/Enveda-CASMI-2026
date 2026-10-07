"""Build results/c3gen/np_pool.parquet: COCONUT structures + natural-product-like train structures.

Train structure kept when RDKit Contrib NP_Score > 0, or when it appears in libs gnps/riken/massbank/mona/
enveda-np-examples and NP_Score > -0.5. COCONUT structures are all kept. One row per plain InChIKey14, sorted by mass.
Columns: ik14, smiles, mass, formula, np_score, src (coconut | train | coconut+train; FACT: no overlap occurs, the bench
COCONUT export excludes train keys), libs (train libs, ';'-joined), heavy, in_coco_full (train row is in full COCONUT
per results/c3/train_classes.parquet; True for COCONUT rows). Checkpointed per chunk in results/c3gen/work/np/. 3 workers, < 1 GB each.

Usage: python research/c3gen/build_np_pool.py
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

R = "D:/Enveda-CASMI-2026/"
WORK = R + "results/c3gen/work/np/"
OUT = R + "results/c3gen/np_pool.parquet"
NP_LIBS = {"gnps", "riken", "massbank", "mona", "enveda-np-examples"}
CHUNK = 20000
_g = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import io, contextlib
    from rdkit.Contrib.NP_Score import npscorer
    with contextlib.redirect_stderr(io.StringIO()):
        _g["fs"] = npscorer.readNPModel()


def _chunk(args):
    i, smis = args
    p = WORK + f"np_{i:04d}.pkl"
    if os.path.exists(p):
        return i, len(smis), True
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Contrib.NP_Score import npscorer
    sc = np.full(len(smis), np.nan, np.float32); ms = np.full(len(smis), np.nan); fo = [None] * len(smis)
    hv = np.zeros(len(smis), np.int16)
    for j, s in enumerate(smis):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            ms[j] = ExactMolWt(m); fo[j] = CalcMolFormula(m); hv[j] = m.GetNumHeavyAtoms()
            sc[j] = npscorer.scoreMol(m, _g["fs"])
        except Exception:
            pass
    pickle.dump(dict(np=sc, mass=ms, formula=fo, heavy=hv), open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)
    return i, len(smis), False


def main():
    import duckdb
    import pandas as pd
    os.makedirs(WORK, exist_ok=True)
    T0 = time.time()
    cm = pickle.load(open(R + "external/bench_inputs/coco/coco_meta.pkl", "rb"))
    co_k = np.asarray(cm["keys"], object); co_s = np.asarray(cm["smiles"], object); del cm
    con = duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
    tr = con.sql(f"select ik, smiles, array_to_string(libs, ';') libs, in_coco from '{R}results/c3/train_classes.parquet'").fetchnumpy()
    tr_k = np.asarray(tr["ik"], object); tr_s = np.asarray(tr["smiles"], object); tr_l = np.asarray(tr["libs"], object)
    log(f"COCONUT {len(co_k):,}  train {len(tr_k):,}")
    smis = np.concatenate([co_s, tr_s])
    jobs = [(i, list(smis[s:s + CHUNK])) for i, s in enumerate(range(0, len(smis), CHUNK))]
    with Pool(3, initializer=_init) as mp:
        for n, (i, k, cached) in enumerate(mp.imap_unordered(_chunk, jobs)):
            if n % 5 == 0 or n == len(jobs) - 1:
                log(f"chunk {n + 1}/{len(jobs)} (#{i}{' cached' if cached else ''})  {time.time() - T0:.0f}s")
    parts = [pickle.load(open(WORK + f"np_{i:04d}.pkl", "rb")) for i, _ in jobs]
    sc = np.concatenate([p["np"] for p in parts]); ms = np.concatenate([p["mass"] for p in parts])
    hv = np.concatenate([p["heavy"] for p in parts]); fo = np.array(sum((p["formula"] for p in parts), []), object)
    nc = len(co_k)
    # train selection
    t_np = sc[nc:]
    in_np_lib = np.array([bool(NP_LIBS & set(l.split(";"))) for l in tr_l], bool)
    keep_t = np.isfinite(t_np) & ((t_np > 0) | (in_np_lib & (t_np > -0.5)))
    tr_libs = dict(zip(tr_k, tr_l))
    tr_cf = dict(zip(tr_k, np.asarray(tr["in_coco"], bool)))
    co_set = set(co_k)
    df = pd.DataFrame(dict(ik14=np.concatenate([co_k, tr_k]), smiles=smis, mass=ms, formula=fo, np_score=sc,
                           heavy=hv, _from=np.r_[np.zeros(nc, np.int8), np.ones(len(tr_k), np.int8)]))
    sel = np.r_[np.ones(nc, bool), keep_t & np.array([k not in co_set for k in tr_k], bool)]
    df = df[sel & np.isfinite(ms)]
    df = df.drop_duplicates("ik14", keep="first")
    in_tr = df.ik14.isin(set(tr_k)).to_numpy()
    df["src"] = np.where(df._from == 1, "train", np.where(in_tr, "coconut+train", "coconut"))
    df["libs"] = [tr_libs.get(k, "") for k in df.ik14]
    # FACT: the bench COCONUT export (coco_meta) shares no key with train; train_classes.in_coco (full COCONUT) does
    df["in_coco_full"] = [bool(tr_cf.get(k, True)) for k in df.ik14]
    df = df.drop(columns="_from").sort_values("mass", kind="mergesort").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    st = dict(rows=len(df), by_src=df.src.value_counts().to_dict(),
              train_total=int(len(tr_k)), train_np_like=int(keep_t.sum()),
              train_np_like_by_rule=dict(np_gt0=int((np.isfinite(t_np) & (t_np > 0)).sum()),
                                         lib_and_gt_m05=int((in_np_lib & np.isfinite(t_np) & (t_np > -0.5)).sum())),
              train_np_like_in_coconut=int((keep_t & np.array([k in co_set for k in tr_k])).sum()),
              coconut_np_score_median=float(np.nanmedian(sc[:nc])), rdkit_fail=int((~np.isfinite(ms)).sum()),
              train_rows_in_full_coconut=int((df.src == "train").to_numpy().__and__(df.in_coco_full.to_numpy()).sum()),
              mass_150_900=int(((df.mass >= 150) & (df.mass <= 900)).sum()), seconds=round(time.time() - T0))
    import json
    json.dump(st, open(R + "results/c3gen/np_pool_stats.json", "w"), indent=1)
    log(json.dumps(st))


if __name__ == "__main__":
    main()

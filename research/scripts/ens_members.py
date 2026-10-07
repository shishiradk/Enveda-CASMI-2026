"""Ensemble bench, stage 1: one FP net pair's fp @ zlog for every S2/S3 PubChem-window structure.

    python research/scripts/ens_members.py --tag ho1s10 --files models/fp_ens_ho1/fp_single_ho1s10.pt models/fp_ens_ho1/fp_merged_ho1s10.pt

Generalises c2gap_ho2.py (same logits, same ce=25 fallback, same window rows) so every ho1-family ensemble member is
scored identically. Only nets trained on the ho1 package (bench molecules held out) are leak-free here.
Writes results/ens/<tag>_logits.pkl and results/ens/<tag>_scores.npy (float32, row-aligned with
windows_5.0ppm.parquet filtered to S2/S3).
"""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "results" / "c3" / "e6_work"
OUTD = ROOT / "results" / "ens"
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench  # noqa: E402


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--files", nargs=2, required=True, help="the single and merged .pt of one member")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    lp = OUTD / f"{a.tag}_logits.pkl"
    if lp.exists():
        Z = pickle.load(open(lp, "rb"))
    else:
        bench.setup_env(ROOT / "research/bench/eng", a.workers)
        import torch
        torch.set_num_threads(a.workers)
        import casmi_engine as E, pv_fp
        _pce = E.parse_ce
        E.parse_ce = lambda v: (lambda x: x if np.isfinite(x) else 25.0)(_pce(v))
        s, m, _ = pv_fp.load_fp_models(sorted(str(ROOT / f) for f in a.files), "cpu")
        Z = {}
        for scen, q in (("S2", "queries_S12.parquet"), ("S3", "queries_S3.parquet")):
            te = pd.read_parquet(bench.OUT / q)
            Z[scen] = bench.alt_zlogs(E, pv_fp, (s, m, "cpu"), te)
            log(f"{scen}: {a.tag} logits for {len(Z[scen])} molecules")
        pickle.dump(Z, open(lp, "wb"), protocol=4)
        del s, m
    t = pq.read_table(W / "windows_5.0ppm.parquet", columns=["scen", "mid", "ik"])
    t = t.filter(pc.is_in(t["scen"], value_set=pa.array(["S2", "S3"])))
    w = t.to_pandas()
    del t
    keys = sorted(set(zip(w.scen, w.mid)))
    gid = pd.Series(range(len(keys)), index=pd.MultiIndex.from_tuples(keys)).reindex(list(zip(w.scen, w.mid))).values
    Zm = np.stack([Z[sc][m] if Z[sc].get(m) is not None else np.zeros(6930, np.float32) for sc, m in keys])
    nz = Zm.shape[1]
    chunks = sorted((W / "fp").glob("chunk_*.npz"))
    ik_all = np.concatenate([np.load(f, allow_pickle=True)["ik"].astype("U14") for f in chunks])
    pos = pd.Series(np.arange(len(ik_all)), index=ik_all).reindex(w.ik.values).values.astype(np.int64)
    del ik_all
    sc = np.full(len(w), -np.inf, np.float32)
    order = np.argsort(pos, kind="stable")
    bounds = np.searchsorted(pos[order], np.arange(len(chunks) + 1) * 50000)
    for c, f in enumerate(chunks):
        rows = order[bounds[c]:bounds[c + 1]]
        if len(rows) == 0:
            continue
        z = np.load(f, allow_pickle=True)
        fp, ok = z["fp"], z["ok"]
        for j in range(0, len(rows), 4000):
            r = rows[j:j + 4000]
            loc = pos[r] - c * 50000
            F = np.unpackbits(fp[loc], axis=1)[:, :nz].astype(np.float32)
            sc[r] = np.where(ok[loc], np.einsum("ij,ij->i", F, Zm[gid[r]]), -np.inf)
    np.save(OUTD / f"{a.tag}_scores.npy", sc)
    log(f"{a.tag} done: {np.isfinite(sc).sum():,}/{len(sc):,} scored")


if __name__ == "__main__":
    main()

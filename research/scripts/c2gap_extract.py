"""C2 gap diagnosis, stage 1: where is the truth in the PubChem channel (S3 PubChem-only bucket, and S2)?

    python research/scripts/c2gap_extract.py

Read-only on existing outputs: results/c3/e6_work/{windows_5.0ppm.parquet, scores_5.0ppm.npy, logits.pkl,
pc_lists_*.pkl}, results/bench/e1/{S2,S3}.pkl, results/bench/truth.parquet, external/pubchem/pubchem_rows_pop.parquet.
Scores are the held-out ho1 nets' fp @ zlog (leak-free on the bench). Writes results/c2gap/:
  mol.parquet   one row per (scen, mid): bucket, window size, truth rank in the full window (raw PubChem ik14),
                channel rank (top-40 metric-key list), engine rank, E6 merged rank, same-mass isomer count, ...
  cand.parquet  top-100 window candidates per molecule (+ the truth row if deeper): score, popularity, mass, smiles.
Peak memory about 1.2 GB (DuckDB memory_limit 1GB).

--work points the four cache inputs at another dir (default unchanged), so a re-extraction against full-net scores
reads results/c2gap_full/work/ while the ho1 cache stays put. --tag keeps the two output names apart in
results/c2gap/.
"""
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb as _ddb
import pyarrow.parquet as pq
import pyarrow.compute as pc


def _pq_read(path, columns=None):
    """DuckDB-backed parquet read returning a pandas DataFrame: avoids pyarrow 'Repetition level
    histogram size mismatch' on files written with older pyarrow versions."""
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    return _ddb.sql(f"SELECT {cols} FROM '{Path(path).as_posix()}'").df()

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "results" / "c3" / "e6_work"
OUTD = ROOT / "results" / "c2gap"
sys.path.insert(0, str(ROOT / "research" / "bench"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import bench, bench_eval as BE  # noqa: E402
from e6_merge_eval import merge  # noqa: E402

TOPK = 100


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    global W
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scens", default="S2,S3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work", default=str(W), help="cache dir holding logits.pkl/windows/scores/pc_lists")
    a = ap.parse_args()
    W = Path(a.work)
    SC = a.scens.split(",")
    BASE = dict(SV="blend", S1="clean_kf", S2="clean_kf", S3="blend")
    TQ = dict(SV="SV", S1="S12", S2="S12", S3="S3")
    OUTD.mkdir(parents=True, exist_ok=True)
    truth = bench.load_truth()
    cls = _pq_read(ROOT / "results/c3/train_classes.parquet", columns=["ik", "in_pc", "in_coco"]).set_index("ik")
    L = pickle.load(open(W / "logits.pkl", "rb"))
    target = {(s, m): t for s in SC for m, (t, _) in L[s].items()}
    del L

    t = _pq_read(W / "windows_5.0ppm.parquet", columns=["scen", "mid", "ik"])
    sc_all = np.load(W / "scores_5.0ppm.npy")
    keep = pc.is_in(__import__("pyarrow").array(t["scen"].tolist()),
                    value_set=__import__("pyarrow").array(SC))
    idx = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
    t = t.iloc[idx].reset_index(drop=True)
    w = t.copy()
    w["score"] = sc_all[idx]
    del t, sc_all, idx
    w = w[np.isfinite(w.score)]
    w = w.sort_values(["scen", "mid", "score"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True)
    w["wrank"] = w.groupby(["scen", "mid"]).cumcount() + 1
    log(f"windows {SC}: {len(w):,} rows")

    # masses + popularity for every window structure (DuckDB, streaming over 100M rows)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='1GB'; SET threads=2; SET temp_directory='{(OUTD / 'duck_tmp').as_posix()}'")
    uk = pd.DataFrame({"ik": w.ik.unique()})
    con.register("uk", uk)
    pop = con.sql(f"""SELECT p.ik, min(p.mass) AS mass, sum(p.n_sid) AS n_sid, sum(p.n_pmid) AS n_pmid,
                             count(*) AS n_cid, min(p.cid) AS cid0
                      FROM read_parquet('{(ROOT / 'external/pubchem/pubchem_rows_pop.parquet').as_posix()}') p
                      SEMI JOIN uk ON p.ik = uk.ik GROUP BY p.ik""").df()
    log(f"popularity rows: {len(pop):,} of {len(uk):,} structures")
    w = w.merge(pop, on="ik", how="left")
    del pop, uk

    # engine lists + E6 merged lists, as e6_merge_eval.py builds them
    eng = {}
    for scen in SC:
        base = BASE[scen]
        R = pickle.load(open(bench.OUT / "e1" / f"{scen}.pkl", "rb"))
        P = pickle.load(open(W / f"pc_lists_{scen}.pkl", "rb"))
        Tq = truth[TQ[scen]]
        for mid in sorted(R):
            r = R[mid]
            _, keys, ix = BE.engine_list(r, base)
            e = dict(keys=keys, raw=[r["keys"][i] for i in ix])
            p = P.get(mid, {})
            mk, mr = merge(e, p, 1, "alt")
            cor = Tq[mid]["correct"]
            eng[(scen, mid)] = dict(
                eng_rank=BE.rank_in(e["keys"], e["raw"], r, cor),
                ch_rank=BE.rank_in(p.get("keys", []), p.get("raw", []), r, cor),
                e6_rank=BE.rank_in(mk, mr, r, cor),
                eng_lib1=float(r["lib"][ix[0]]) if ix else 0.0,
                eng_top1=keys[0] if keys else None, ch_top1=(p.get("keys") or [None])[0],
                truth_canon=r["truth_canon"], truth_in_pool=bool(r["truth_in_pool"]))

    rows, cands = [], []
    for (scen, mid), g in w.groupby(["scen", "mid"], sort=False):
        Tq = truth[TQ[scen]][mid]
        cor = set(Tq["correct"]) | {eng[(scen, mid)]["truth_canon"]} if (scen, mid) in eng else set(Tq["correct"])
        hit = g.ik.isin(cor).values
        tr = int(g.wrank.values[hit.argmax()]) if hit.any() else 0
        if scen == "S3":
            b = "PC" if (mid in cls.index and cls.at[mid, "in_pc"] and not cls.at[mid, "in_coco"]) else "C3"
        else:
            b = scen
        tm = float(g.mass.values[hit.argmax()]) if hit.any() else np.nan
        m0 = g.mass.values
        tg = target.get((scen, mid), np.nan)
        row = dict(scen=scen, mid=mid, b=b, n_win=len(g), truth_wrank=tr, target=tg,
                   n_iso_truth=int(np.sum(np.abs(m0 - tm) < 2e-4)) if hit.any() else 0,
                   top1_same_mass=bool(hit.any() and abs(m0[0] - tm) < 2e-4),
                   truth_score=float(g.score.values[hit.argmax()]) if hit.any() else np.nan,
                   top1_score=float(g.score.values[0]), top2_score=float(g.score.values[1]) if len(g) > 1 else np.nan,
                   truth_ppm=(tm - tg) / tg * 1e6 if hit.any() else np.nan)
        row.update(eng.get((scen, mid), {}))
        rows.append(row)
        sel = g.wrank.values <= TOPK
        sel |= hit
        c = g[sel].copy()
        c["is_truth"] = hit[sel]
        c["b"] = b
        cands.append(c)
    mol = pd.DataFrame(rows)
    cand = pd.concat(cands, ignore_index=True)
    # smiles for the candidate table only
    con.register("ck", pd.DataFrame({"cid0": cand.cid0.dropna().astype("int64").unique()}))
    smi = con.sql(f"""SELECT p.cid AS cid0, p.smiles FROM read_parquet('{(ROOT / 'external/pubchem/pubchem_rows_pop.parquet').as_posix()}') p
                      SEMI JOIN ck ON p.cid = ck.cid0""").df()
    cand = cand.merge(smi, on="cid0", how="left")
    mol.to_parquet(OUTD / f"mol{a.tag}.parquet", index=False)
    cand.to_parquet(OUTD / f"cand{a.tag}.parquet", index=False)
    log(f"mol {len(mol)} rows, cand {len(cand):,} rows")


if __name__ == "__main__":
    main()

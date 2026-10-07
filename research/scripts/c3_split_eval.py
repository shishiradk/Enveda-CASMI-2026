"""MRR@25 of saved bench runs split by the truth's database class (COCONUT / PubChem-only / neither = real Class 3).

    python research/scripts/c3_split_eval.py [--tag e1] [--sets blend,pv,clean_kf,fp_only]

Uses the bench's own list builder and matcher (bench_eval.engine_list / rank_in, metric key + tautomer aware), so the
numbers per scenario equal bench_eval's when the classes are pooled. Our V2 engine ("ours") is scored from the
cached lists in results/bench/cache/ours_<scen>.pkl. Classes come from results/c3/train_classes.parquet (c3_census.py).
Writes results/c3/split_<tag>.json and per-molecule rows to results/c3/split_<tag>.csv.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402

OUT = ROOT / "results" / "c3"


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None, None]
    r = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[r].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--sets", default="blend,pv,clean_kf,fp_only,fp_only_mk,pv_mk")
    a = ap.parse_args()
    cls = pd.read_parquet(OUT / "train_classes.parquet").set_index("ik")
    label = lambda m: "coconut" if cls.at[m, "in_coco"] else ("pubchem_only" if cls.at[m, "in_pc"] else "neither")
    truth = bench.load_truth()
    rows = []
    for p in sorted((bench.OUT / a.tag).glob("S*.pkl")):
        scen = p.stem
        if scen.startswith("recs"):
            continue
        R = pickle.load(open(p, "rb"))
        T = truth["S12" if scen[:2] in ("S1", "S2") else "S3"]
        ours_p = bench.OUT / "cache" / f"ours_{'S3' if scen.startswith('S3') else scen}.pkl"
        ours = pickle.load(open(ours_p, "rb")) if ours_p.exists() and scen != "S3i" else {}
        for m in sorted(R):
            row = dict(scen=scen, mid=m, cls=label(m) if m in cls.index else "unknown")
            for s in a.sets.split(","):
                if "scores" in R[m] and s in R[m]["scores"]:
                    _, keys, idx = BE.engine_list(R[m], s)
                    row[s] = BE.rr(BE.rank_in(keys, [R[m]["keys"][i] for i in idx], R[m], T[m]["correct"]))
            if m in ours:
                o = ours[m]
                row["ours"] = BE.rr(BE.rank_in(o["keys"], o["raw"], R[m], T[m]["correct"]))
                # E2: RRF(engine list, ours), BETA 0.4 -- as submitted (blend) and with the honest ranker (clean_kf)
                for base, col in (("blend", "e2_fusion"), ("clean_kf", "e2_honest")):
                    if "scores" in R[m] and base in R[m]["scores"]:
                        _, keys, idx = BE.engine_list(R[m], base)
                        order, raws = BE.fuse(dict(keys=keys, raw=[R[m]["keys"][i] for i in idx]), o, 0.4)
                        row[col] = BE.rr(BE.rank_in(order, raws, R[m], T[m]["correct"]))
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"split_{a.tag}.csv", index=False)
    res = {}
    for (scen, c), g in D.groupby(["scen", "cls"]):
        res.setdefault(scen, {})[c] = {"n": len(g), **{s: boot(g[s].dropna()) for s in D.columns[3:] if g[s].notna().any()}}
    json.dump(res, open(OUT / f"split_{a.tag}.json", "w"), indent=1)
    for scen, d in res.items():
        for c, v in d.items():
            print(scen, f"{c:13}", "n", v["n"], " ".join(f"{k} {x[0]:.3f}" for k, x in v.items() if k != "n" and x[0] is not None))


if __name__ == "__main__":
    main()

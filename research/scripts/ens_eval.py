"""Ensemble bench, stage 2: truth rank in the FULL 5 ppm window for ho1 alone vs. averages of ho1-family members.

    python research/scripts/ens_eval.py [--members ho1,ho1s10,ho1s20,ho1s30,ho1big]  -> results/ens/ens_eval.json

Inputs: results/ens/<tag>_scores.npy from ens_members.py (all leak-free: ho1 package, bench molecules held out).
Same buckets, ranks, MRR@25 and E6 merge simulation as c2gap_ens_window.py. score = fp @ zlog is linear in zlog,
so the raw mean of member scores equals scoring with the averaged zlog (what a deployed ensemble would do).
"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench  # noqa
W = ROOT / "results/c3/e6_work"; D = ROOT / "results/c2gap"; E = ROOT / "results/ens"

ap = argparse.ArgumentParser()
ap.add_argument("--members", default="ho1,ho1s10,ho1s20,ho1s30,ho1big")
a = ap.parse_args()
M = a.members.split(",")

t = pq.read_table(W / "windows_5.0ppm.parquet", columns=["scen", "mid", "ik"])
t = t.filter(pc.is_in(t["scen"], value_set=pa.array(["S2", "S3"])))
w = t.to_pandas(); del t
S = {m: np.load(E / f"{m}_scores.npy") for m in M}
assert all(len(v) == len(w) for v in S.values())
mol = pd.read_parquet(D / "mol.parquet").set_index(["scen", "mid"])
truth = bench.load_truth()


def rk(v, hit):
    return int((v > v[hit].max()).sum() + 1) if hit.any() else 0


def z(v):
    f = np.isfinite(v); o = np.full(len(v), -np.inf, np.float64)
    o[f] = (v[f] - v[f].mean()) / (v[f].std() + 1e-6); return o


combos = {"ho1": ["ho1"], **{m: [m] for m in M[1:]}}
std = [m for m in M if m != "ho1big"]
combos[f"mean{len(std)}"] = std
combos[f"mean{len(M)}"] = M
if len(std) >= 3:
    combos["mean3"] = std[:3]
if "ho1" in M and len(M) > 1:
    combos["mean2"] = M[:2]

rows = []
for (s, m), ix in w.groupby(["scen", "mid"], sort=False).indices.items():
    b = mol.at[(s, m), "b"]
    if b not in ("PC", "S2"):
        continue
    cor = truth["S12" if s == "S2" else "S3"][m]["correct"] | {mol.at[(s, m), "truth_canon"]}
    hit = pd.Series(w.ik.values[ix]).isin(cor).values
    seg = {k: S[k][ix].astype(np.float64) for k in M}
    r = dict(b=b, eng=int(mol.at[(s, m), "eng_rank"]))
    for name, mem in combos.items():
        r[name] = rk(np.mean([seg[k] for k in mem], 0), hit)
        if len(mem) > 1:
            r["z" + name] = rk(np.mean([z(seg[k]) for k in mem], 0), hit)
    rows.append(r)
r = pd.DataFrame(rows)
keys = [c for c in r.columns if c not in ("b", "eng")]
rr = lambda x: np.where((x > 0) & (x <= 25), 1 / np.where(x > 0, x, 1), 0)
def mrr(x): return round(float(rr(np.asarray(x)).mean()), 4)
def e6(ch, eng): return [min([1 if e == 1 else 2 * (e - 1)] * (e > 0) + [2 * c + 1] * (c > 0)) if (e > 0 or c > 0) else 0 for c, e in zip(ch, eng)]
def ci(d):
    bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
out = {}
for b, g in list(r.groupby("b")) + [("ALL", r)]:
    out[b] = {"n": len(g)}
    for k in keys:
        out[b][k] = dict(window_mrr=mrr(g[k]), e6_sim=mrr(e6(g[k], g.eng)), top1=int((g[k] == 1).sum()),
                         le25=int(((g[k] > 0) & (g[k] <= 25)).sum()), le100=int(((g[k] > 0) & (g[k] <= 100)).sum()))
    for k in keys:
        if k != "ho1" and len(combos.get(k.lstrip("z"), [0, 0])) > 1:
            out[b][f"{k}_minus_ho1_window_ci"] = ci(rr(g[k].values) - rr(g.ho1.values))
            out[b][f"{k}_minus_ho1_e6_ci"] = ci(rr(np.array(e6(g[k], g.eng))) - rr(np.array(e6(g.ho1, g.eng))))
json.dump(out, open(E / "ens_eval.json", "w"), indent=1)
for b in out:
    print(f"== {b} (n={out[b]['n']})")
    for k in keys:
        x = out[b][k]; c = out[b].get(f"{k}_minus_ho1_window_ci", ""); c2 = out[b].get(f"{k}_minus_ho1_e6_ci", "")
        print(f"  {k:10s} win {x['window_mrr']:.4f}  e6 {x['e6_sim']:.4f}  top1 {x['top1']:3d}  <=25 {x['le25']:3d}  {c} {c2}")

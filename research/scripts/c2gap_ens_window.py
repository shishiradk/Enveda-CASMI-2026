"""C2 gap diagnosis: truth rank in the FULL 5 ppm window under ho1, ho2 and the ho1+ho2 mean (leak-free nets).

    python research/scripts/c2gap_ens_window.py    -> results/c2gap/ens_window.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench  # noqa
W = ROOT / "results/c3/e6_work"; D = ROOT / "results/c2gap"
t = pq.read_table(W / "windows_5.0ppm.parquet", columns=["scen", "mid", "ik"])
idx = np.flatnonzero(pc.is_in(t["scen"], value_set=pa.array(["S2", "S3"])).to_numpy(zero_copy_only=False))
w = t.take(idx).to_pandas(); w["s1"] = np.load(W / "scores_5.0ppm.npy")[idx]
h = pd.read_parquet(D / "ho2_scores.parquet"); assert (h.ik.values == w.ik.values).all()
w["s2"] = h.s_ho2.values; del h
mol = pd.read_parquet(D / "mol.parquet").set_index(["scen", "mid"])
truth = bench.load_truth()
def rk(v, hit):
    return int((v > v[hit].max()).sum() + 1) if hit.any() else 0
rows = []
for (s, m), g in w.groupby(["scen", "mid"], sort=False):
    b = mol.at[(s, m), "b"]
    if b not in ("PC", "S2"): continue
    cor = truth["S12" if s == "S2" else "S3"][m]["correct"] | {mol.at[(s, m), "truth_canon"]}
    hit = g.ik.isin(cor).values
    a, c = g.s1.values, g.s2.values
    za, zc = (a - a.mean()) / (a.std() + 1e-6), (c - c.mean()) / (c.std() + 1e-6)
    rows.append(dict(b=b, eng=int(mol.at[(s, m), "eng_rank"]), ho1=rk(a, hit), ho2=rk(c, hit), mean=rk(a + c, hit), zmean=rk(za + zc, hit),
                     max_=rk(np.maximum(za, zc), hit)))
r = pd.DataFrame(rows)
def mrr(x): x = np.asarray(x); return round(float(np.where((x > 0) & (x <= 25), 1 / np.where(x > 0, x, 1), 0).mean()), 4)
def e6(ch, eng): return [min([1 if e == 1 else 2 * (e - 1)] * (e > 0) + [2 * c + 1] * (c > 0)) if (e > 0 or c > 0) else 0 for c, e in zip(ch, eng)]
out = {}
for b, g in r.groupby("b"):
    out[b] = {k: dict(window_mrr=mrr(g[k]), e6_sim=mrr(e6(g[k], g.eng)), top1=int((g[k] == 1).sum()), le40=int(((g[k] > 0) & (g[k] <= 40)).sum()),
                      le100=int(((g[k] > 0) & (g[k] <= 100)).sum())) for k in ("ho1", "ho2", "mean", "zmean", "max_")}
    # paired bootstrap mean - ho1 on window MRR
    rr = lambda x: np.where((x > 0) & (x <= 25), 1 / np.where(x > 0, x, 1), 0)
    d = rr(g["mean"].values) - rr(g.ho1.values); bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
    out[b]["mean_minus_ho1_ci"] = [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
json.dump(out, open(D / "ens_window.json", "w"), indent=1); print(json.dumps(out, indent=1))

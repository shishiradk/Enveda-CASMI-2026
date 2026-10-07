"""E5 design: gated fusion of the engine list with our V2 (PubChem) list, scored with the calibrated proxy.

    python research/scripts/e5_gate_eval.py

Rule (per molecule): g = library similarity of the engine's top-1 candidate (rec["lib"]).
  g >= t  -> keep the engine list unchanged (library answer protected)
  g <  t  -> weighted RRF of engine list (weight 1) and V2 list (weight beta), K = 3, top 40 (bench_eval.fuse)
Engine list = clean_kf (honest ranker) on S1/S2, blend on S3 (no clean_kf there). The visible set is protected by
construction for t <= 0.6 (E3b's protection already covered every visible first place, e4_popularity.md), so V = 1.
Proxy = 0.146*V + 0.21*S1 + 0.11*PC (c3_calibration_report.md); S2 and Class 3 reported separately.
Selection is checked out of sample: settings chosen on a random half of molecules, scored on the other half.
Writes results/c3/e5_gate.json.
"""
import itertools, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402

W = dict(V=0.146, S1=0.21, PC=0.11)
TS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)  # 1.01 = always fuse
BETAS = (0.2, 0.4, 0.7, 1.0, 2.0)


def load():
    truth = bench.load_truth()
    cls = pd.read_parquet(ROOT / "results/c3/train_classes.parquet").set_index("ik")
    mol = []
    for scen, base, ours_name in (("S1", "clean_kf", "S1"), ("S2", "clean_kf", "S2"), ("S3", "blend", "S3")):
        R = pickle.load(open(bench.OUT / "e1" / f"{scen}.pkl", "rb"))
        O = pickle.load(open(bench.OUT / "cache" / f"ours_{ours_name}.pkl", "rb"))
        T = truth["S12" if scen != "S3" else "S3"]
        for m in sorted(R):
            r = R[m]
            _, keys, idx = BE.engine_list(r, base)
            g = float(r["lib"][idx[0]]) if len(idx) else 0.0
            c = "coconut" if cls.at[m, "in_coco"] else ("pc" if cls.at[m, "in_pc"] else "c3")
            bucket = scen if scen != "S3" else ("PC" if c == "pc" else "C3")
            mol.append(dict(bucket=bucket, mid=m, g=g, rec=r, correct=T[m]["correct"],
                            e=dict(keys=keys, raw=[r["keys"][i] for i in idx]), o=O.get(m, {"keys": [], "raw": []})))
    return mol


def score(mol, t, beta):
    out = []
    for x in mol:
        if x["g"] >= t:
            order, raws = x["e"]["keys"], x["e"]["raw"]
        else:
            order, raws = BE.fuse(x["e"], x["o"], beta)
        out.append(BE.rr(BE.rank_in(order, raws, x["rec"], x["correct"])))
    return np.array(out)


def proxy(rr, buckets, sel=None):
    sel = np.ones(len(rr), bool) if sel is None else sel
    m = {b: rr[(buckets == b) & sel].mean() for b in ("S1", "S2", "PC", "C3")}
    return W["V"] * 1.0 + W["S1"] * m["S1"] + W["PC"] * m["PC"], m


def main():
    mol = load()
    buckets = np.array([x["bucket"] for x in mol])
    base = score(mol, 9.0, 0.0)  # t > any g: engine list only = E1
    p0, m0 = proxy(base, buckets)
    grid = {}
    for t, b in itertools.product(TS, BETAS):
        rr = score(mol, t, b)
        p, m = proxy(rr, buckets)
        grid[(t, b)] = (rr, p, m)
    best = max(grid, key=lambda k: grid[k][1])
    rng = np.random.default_rng(20261003)
    oos = []
    for _ in range(20):  # choose on one half, report on the other
        half = rng.random(len(mol)) < 0.5
        k = max(grid, key=lambda k: proxy(grid[k][0], buckets, half)[0])
        oos.append(proxy(grid[k][0], buckets, ~half)[0] - proxy(base, buckets, ~half)[0])
    res = {"E1": {"proxy": round(p0, 4), **{k: round(v, 4) for k, v in m0.items()}},
           "best": {"t": best[0], "beta": best[1], "proxy": round(grid[best][1], 4),
                    **{k: round(v, 4) for k, v in grid[best][2].items()}},
           "best_minus_E1": round(grid[best][1] - p0, 4),
           "out_of_sample_gain": [round(float(np.mean(oos)), 4), round(float(np.percentile(oos, 10)), 4),
                                  round(float(np.percentile(oos, 90)), 4)],
           "grid": {f"t{t}_b{b}": {"proxy": round(v[1], 4), **{k: round(x, 4) for k, x in v[2].items()}}
                    for (t, b), v in grid.items()},
           "gate_share_fused": {bk: round(float(np.mean([x["g"] < best[0] for x in mol if x["bucket"] == bk])), 3)
                                for bk in ("S1", "S2", "PC", "C3")}}
    json.dump(res, open(ROOT / "results/c3/e5_gate.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1))
    for t in TS:
        print(f"t {t:4}", "  ".join(f"b{b}:{grid[(t, b)][1]:.4f}(S1 {grid[(t, b)][2]['S1']:.3f} PC {grid[(t, b)][2]['PC']:.3f})"
                                   for b in BETAS))


if __name__ == "__main__":
    main()

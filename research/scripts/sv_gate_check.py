"""What does a library-similarity gate do on re-measured library hits (scenario SV)?

    python research/scripts/sv_gate_check.py

E1 lists = bench SV records (blend). V2 lists = results/kaggle_v2_local/our_lists.json (V2 never uses enveda-180,
so its lists for these queries are exact). Writes results/c3/sv_gate.json.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402


def main():
    R = pickle.load(open(bench.OUT / "e1" / "SV.pkl", "rb"))
    T = bench.load_truth()["SV"]
    V2 = json.load(open(ROOT / "results/kaggle_v2_local/our_lists.json"))
    bench.setup_env(ROOT / "research/bench/eng", 4)
    import casmi_engine as E
    C = bench.canon_many(E, sorted({s for v in V2.values() for s in v["smiles"] if s}), 4)
    rows = []
    for m in sorted(R):
        r = R[m]
        _, keys, idx = BE.engine_list(r, "blend")
        e = dict(keys=keys, raw=[r["keys"][i] for i in idx])
        o = dict(keys=[C.get(s) for s in V2.get(m, {"smiles": []})["smiles"]])
        o["raw"] = [None] * len(o["keys"])
        g = float(r["lib"][idx[0]]) if len(idx) else 0.0
        rr_e1 = BE.rr(BE.rank_in(e["keys"], e["raw"], r, T[m]["correct"]))
        rr_v2 = BE.rr(BE.rank_in(o["keys"], o["raw"], r, T[m]["correct"]))
        fused = {}
        for t, b in ((0.7, 2.0), (0.7, 0.4), (0.5, 0.4), (0.3, 0.4), (0.0, 0.0), (9.0, 0.4)):  # 9.0 = always fuse (E2)
            if g >= t:
                fused[f"t{t}_b{b}"] = rr_e1
            else:
                order, raws = BE.fuse(e, o, b)
                fused[f"t{t}_b{b}"] = BE.rr(BE.rank_in(order, raws, r, T[m]["correct"]))
        rows.append(dict(mid=m, g=g, e1=rr_e1, v2=rr_v2, **fused))
    g = np.array([x["g"] for x in rows])
    res = {"n": len(rows), "E1": round(float(np.mean([x["e1"] for x in rows])), 4),
           "V2": round(float(np.mean([x["v2"] for x in rows])), 4),
           "top1_lib_quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                          np.round(np.percentile(g, [10, 25, 50, 75, 90]), 3).tolist())),
           "share_gate_below_0.7": round(float((g < 0.7).mean()), 3),
           "fused": {k: round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0] if k.startswith("t")}}
    json.dump(res, open(ROOT / "results/c3/sv_gate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

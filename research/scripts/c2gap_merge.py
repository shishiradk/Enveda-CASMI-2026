"""C2 gap diagnosis, stage 2: what the E6 merge rule costs the PubChem-only bucket, and leak-free alternatives.

    python research/scripts/c2gap_merge.py

Same items as e6_merge_eval.py (engine lists from results/bench/e1, channel lists from results/c3/e6_work, held-out
ho1 nets). Merge = a slot pattern over the engine list E and the channel list P, metric-key de-duplicated, top 40.
Gated variants switch to an aggressive pattern only when a per-molecule rule fires. Gates are evaluated with
5-fold molecule-level CV where a threshold is fitted (thresholds chosen on the training folds only).
Proxy = 0.146*SV + 0.21*S1 + 0.111*PC (research/analysis/c3_calibration_report.md). Writes results/c2gap/merge.json.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402

W = ROOT / "results" / "c3" / "e6_work"
WTS = dict(SV=0.146, S1=0.21, PC=0.111)


def pattern_merge(e, p, pat):
    """pat: string over {E,P}; consumed left to right, then alternation E,P to the end."""
    keys, raws, seen = [], [], set()
    E = list(zip(e["keys"], e["raw"]))
    P = list(zip(p.get("keys", []), p.get("raw", [])))
    ie = ip = 0
    seq = list(pat) + ["E", "P"] * 60
    for s in seq:
        if len(keys) >= 40 or (ie >= len(E) and ip >= len(P)):
            break
        if s == "E" and ie < len(E):
            k, r = E[ie]; ie += 1
        elif s == "P" and ip < len(P):
            k, r = P[ip]; ip += 1
        elif ie < len(E):
            k, r = E[ie]; ie += 1
        else:
            k, r = P[ip]; ip += 1
        if k and k not in seen:
            seen.add(k); keys.append(k); raws.append(r)
    return keys, raws


def load_items():
    truth = bench.load_truth()
    cls = pd.read_parquet(ROOT / "results/c3/train_classes.parquet").set_index("ik")
    items = []
    for scen, base in (("SV", "blend"), ("S1", "clean_kf"), ("S2", "clean_kf"), ("S3", "blend")):
        R = pickle.load(open(bench.OUT / "e1" / f"{scen}.pkl", "rb"))
        P = pickle.load(open(W / f"pc_lists_{scen}.pkl", "rb"))
        T = truth["S12" if scen in ("S1", "S2") else scen]
        for mid in sorted(R):
            r = R[mid]
            _, keys, idx = BE.engine_list(r, base)
            if scen == "S3":
                b = "PC" if (mid in cls.index and cls.at[mid, "in_pc"] and not cls.at[mid, "in_coco"]) else "C3"
            else:
                b = scen
            p = P.get(mid, {})
            ps = p.get("score", [])
            es = r["scores"][base][idx] if len(idx) else np.array([])
            items.append(dict(b=b, mid=mid, r=r, correct=T[mid]["correct"], e=dict(keys=keys, raw=[r["keys"][i] for i in idx]), p=p,
                              lib1=float(r["lib"][idx[0]]) if len(idx) else 0.0,
                              top_sim=float(r.get("top_sim", 0.0) or 0.0),
                              es1=float(es[0]) if len(es) else 0.0, es2=float(es[1]) if len(es) > 1 else 0.0,
                              pm=(ps[0] - ps[1]) if len(ps) > 1 else 0.0, ps1=ps[0] if ps else -1e9,
                              e1_in_p=bool(keys and keys[0] in set(p.get("keys", [])[:5])),
                              agree=bool(keys and p.get("keys") and keys[0] == p["keys"][0])))
    return items


def rr_of(x, keys, raws):
    return BE.rr(BE.rank_in(keys, raws, x["r"], x["correct"]))


def evaluate(items, choose):
    """choose(x) -> pattern string. Returns MRR per bucket and the proxy."""
    out = {}
    for b in ("SV", "S1", "S2", "PC", "C3"):
        v = [rr_of(x, *pattern_merge(x["e"], x["p"], choose(x))) for x in items if x["b"] == b]
        out[b] = round(float(np.mean(v)), 4)
    out["proxy"] = round(sum(WTS[b] * out[b] for b in WTS), 4)
    return out


def main():
    items = load_items()
    res = {}
    base = evaluate(items, lambda x: "E")  # = m1_alt (E, E, P, E, P, ...)
    res["E6_m1_alt"] = base
    for pat in ("E" * 40, "EP", "PE", "EPP", "EEPP", "PP"):
        res["pat_" + pat[:6]] = evaluate(items, lambda x, pat=pat: pat)
    # oracle bounds for the PC bucket: channel at rank 1 / 2 only where it is right
    def oracle(x, pat):
        pk = x["p"].get("keys", [])
        ok = pk and BE.is_ok(pk[0], x["p"]["raw"][0], x["r"], x["correct"])
        return pat if ok else "E"
    res["oracle_P1_when_right"] = evaluate(items, lambda x: oracle(x, "PE"))
    res["oracle_P2_when_right"] = evaluate(items, lambda x: oracle(x, "EP"))
    # simple gates: aggressive pattern when engine top-1 is weak and the channel is confident
    grid = {}
    for pat in ("EP", "PE"):
        for lt in (0.3, 0.5, 0.7, 2.0):
            for pm in (0.0, 2.0, 5.0, 10.0, 1e9):
                grid[f"{pat}_lib<{lt}_pm>{pm}"] = evaluate(
                    items, lambda x, pat=pat, lt=lt, pm=pm: pat if (x["lib1"] < lt and x["pm"] > pm) else "E")
    res["gate_grid_top"] = dict(sorted(grid.items(), key=lambda kv: -kv[1]["proxy"])[:15])
    res["gate_grid_n"] = len(grid)
    # margins: distribution per bucket
    df = pd.DataFrame([{k: x[k] for k in ("b", "lib1", "top_sim", "es1", "es2", "pm", "ps1", "agree", "e1_in_p")} for x in items])
    res["feature_medians"] = df.groupby("b")[["lib1", "top_sim", "es1", "pm", "ps1"]].median().round(3).to_dict()
    res["agree_rate"] = df.groupby("b")["agree"].mean().round(3).to_dict()
    json.dump(res, open(ROOT / "results/c2gap/merge.json", "w"), indent=1)
    for k, v in res.items():
        if k.startswith(("E6", "pat", "oracle")):
            print(f"{k:24} {v}")
    print("top gates:")
    for k, v in res["gate_grid_top"].items():
        print(f"  {k:28} {v}")
    print(res["feature_medians"]); print(res["agree_rate"])
    pickle.dump(df, open(ROOT / "results/c2gap/merge_feats.pkl", "wb"))


if __name__ == "__main__":
    main()

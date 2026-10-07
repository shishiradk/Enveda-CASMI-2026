"""E6 offline: merge rules for the engine list + our PubChem channel (append-only family), per bench bucket.

    python research/scripts/e6_merge_eval.py

Engine list: blend on SV and S3, the honest ranker clean_kf on S1/S2 (bench e1 records). PubChem list:
results/c3/e6_work/pc_lists_<scen>.pkl (e6_pc_channel.py, held-out ho1 nets).
Rule (m, mode, gate):
  engine top-m always first (library answers and confident engine answers keep their slots);
  if gate is set and the engine's top-1 library similarity >= gate: engine list unchanged;
  otherwise the remaining slots are filled by `mode`:
    alt   -- alternate engine[m:], pc, engine, pc, ...
    block -- the PubChem list first, then engine[m:]
  de-duplicated on the metric key, top 40.
Acceptance: SV and S1 may not fall more than 0.005 below the engine; among those, maximise the PubChem-only bucket.
Writes results/c3/e6_merge.json.
"""
import itertools, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402

W = ROOT / "results" / "c3" / "e6_work"


def merge(e, p, m, mode):
    keys, raws, seen = [], [], set()

    def add(k, r):
        if k and k not in seen:
            seen.add(k); keys.append(k); raws.append(r)
    for k, r in zip(e["keys"][:m], e["raw"][:m]):
        add(k, r)
    rest_e = list(zip(e["keys"][m:], e["raw"][m:]))
    rest_p = list(zip(p.get("keys", []), p.get("raw", [])))
    if mode == "block":
        seq = rest_p + rest_e
    else:
        seq = [x for pair in itertools.zip_longest(rest_e, rest_p) for x in pair if x is not None]
    for k, r in seq:
        add(k, r)
        if len(keys) >= 40:
            break
    return keys, raws


def main():
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
            items.append(dict(b=b, r=r, correct=T[mid]["correct"], g=float(r["lib"][idx[0]]) if len(idx) else 0.0,
                              e=dict(keys=keys, raw=[r["keys"][i] for i in idx]), p=P.get(mid, {})))
    B = np.array([x["b"] for x in items])

    def run(m, mode, gate):
        out = []
        for x in items:
            if gate is not None and x["g"] >= gate:
                ks, rs = x["e"]["keys"], x["e"]["raw"]
            else:
                ks, rs = merge(x["e"], x["p"], m, mode)
            out.append(BE.rr(BE.rank_in(ks, rs, x["r"], x["correct"])))
        out = np.array(out)
        return {b: round(float(out[B == b].mean()), 4) for b in ("SV", "S1", "S2", "PC", "C3")}

    base = run(10 ** 6, "alt", None)  # m beyond the list = engine only
    pc_only = {b: round(float(np.mean([BE.rr(BE.rank_in(x["p"].get("keys", []), x["p"].get("raw", []), x["r"],
                                                      x["correct"])) for x in items if x["b"] == b])), 4)
               for b in ("SV", "S1", "S2", "PC", "C3")}
    grid = {}
    for m, mode, gate in itertools.product((1, 2, 3, 5, 10), ("alt", "block"), (None, 0.5, 0.7)):
        grid[f"m{m}_{mode}_g{gate}"] = run(m, mode, gate)
    ok = {k: v for k, v in grid.items() if v["SV"] >= base["SV"] - 0.005 and v["S1"] >= base["S1"] - 0.005}
    best = max(ok, key=lambda k: ok[k]["PC"]) if ok else None
    res = dict(engine=base, pubchem_channel_alone=pc_only, best=best, best_scores=grid.get(best), grid=grid,
               n={b: int((B == b).sum()) for b in ("SV", "S1", "S2", "PC", "C3")})
    json.dump(res, open(ROOT / "results/c3/e6_merge.json", "w"), indent=1)
    print("engine           ", base)
    print("PubChem channel  ", pc_only)
    for k, v in sorted(grid.items(), key=lambda kv: -kv[1]["PC"])[:12]:
        flag = "ok" if k in ok else "--"
        print(f"{flag} {k:18} {v}")
    print("best (SV and S1 within 0.005):", best, grid.get(best))


if __name__ == "__main__":
    main()

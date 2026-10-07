"""How closely does the local CPU engine reproduce the lists the Kaggle E1 run wrote (results/bench/kaggle_e1_output)?

    python research/bench/bench_parity.py [N_MOLECULES=40] [WORKERS=3]      # ~10 min; loads the 3 GB library

Runs the unmasked engine on the first N test molecules (bench.py scenario "T"), keeps the local lists
(results/bench/parity/lists_T.json, recs_T.pkl) and compares them with eng_lists.json. To give the differences a scale
the same comparison is made between two disjoint halves of the local ranker ensemble (seed split): if Kaggle-vs-local
is not larger than half-vs-half, the difference is ranker seed/fit noise rather than a different engine.
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402


def compare(A, B):
    """A, B: {mid: [keys]}; B is the reference."""
    import numpy as np
    rows = []
    for m in A:
        a, b = A[m], B.get(m, [])
        if not a or not b:
            continue
        pos = {k: i + 1 for i, k in enumerate(a)}
        r = dict(top1_same=a[0] == b[0], top3_same_order=a[:3] == b[:3], top5_same_order=a[:5] == b[:5],
                 top5_same_set=set(a[:5]) == set(b[:5]),
                 top25_jaccard=len(set(a[:25]) & set(b[:25])) / len(set(a[:25]) | set(b[:25])),
                 top25_same_order=a[:25] == b[:25])
        for k in (1, 2, 3, 5, 10):  # where does the reference's rank-k candidate sit in the other list
            if len(b) >= k:
                r[f"rr_of_ref_rank{k}"] = 1.0 / pos[b[k - 1]] if b[k - 1] in pos and pos[b[k - 1]] <= 25 else 0.0
                r[f"absdisp_ref_rank{k}"] = abs(pos.get(b[k - 1], 41) - k)
        rows.append(r)
    keys = sorted({k for r in rows for k in r})
    return {k: round(float(np.mean([r[k] for r in rows if k in r])), 4) for k in keys} | {"n": len(rows)}


def parity2(E, P, recs, ours, blocks, pvr, out, workers):
    import numpy as np
    K = json.load(open(bench.OUT / "kaggle_e1_output" / "eng_lists.json"))
    pickle.dump(recs, open(out / "recs_T.pkl", "wb"), protocol=4)
    sets = {"local": (pvr, ours), "half_A": ([pvr[i] for i in (0, 1, 4, 5)], [ours[i] for i in (0, 2)]),
            "half_B": ([pvr[i] for i in (2, 3, 6, 7)], [ours[i] for i in (1, 3)])}
    order = {}
    for nm, (p, o) in sets.items():
        bl = E.blend_scores(E.score_molecules(recs, p, ["base"], use_fp=True),
                            E.score_molecules(recs, o, blocks, use_fp=False), bench.W_PV)
        order[nm] = {r["mid"]: r["cand"][np.argsort(-bl[r["mid"]], kind="mergesort")[:80]] for r in recs if r["mid"] in bl}
    C = bench.canon_many(E, [P["smiles"][c] for od in order.values() for o in od.values() for c in o], workers)
    L = {}
    for nm, od in order.items():
        L[nm] = {}
        for mid, o in od.items():
            keys, seen = [], set()
            for c in o:
                k = C.get(P["smiles"][c])
                if k is None or k in seen:
                    continue
                seen.add(k); keys.append(k)
                if len(keys) >= 40:
                    break
            L[nm][str(mid)] = keys
    kag = {m: K[m]["keys"] for m in L["local"] if m in K}
    res = {"local_vs_kaggle": compare(L["local"], kag), "halfA_vs_halfB": compare(L["half_A"], L["half_B"]),
           "halfA_vs_kaggle": compare(L["half_A"], kag), "local_vs_halfA": compare(L["local"], L["half_A"]),
           "same_length": float(np.mean([len(L["local"][m]) == len(kag[m]) for m in kag])),
           "same_top40_set": float(np.mean([set(L["local"][m]) == set(kag[m]) for m in kag]))}
    json.dump(L, open(out / "lists_T.json", "w"))
    json.dump(res, open(out / "parity2.json", "w"), indent=1)
    ks = [k for k in res["local_vs_kaggle"]]
    print("| metric | " + " | ".join(k for k in res if isinstance(res[k], dict)) + " |")
    for k in ks:
        print(f"| {k} | " + " | ".join(str(v.get(k)) for v in res.values() if isinstance(v, dict)) + " |")
    print("same list length:", res["same_length"], "| same top-40 set:", res["same_top40_set"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bench.parity = parity2
    bench.run(SimpleNamespace(eng_dir=str(HERE / "eng"), workers=w, tag="parity", scen="T", limit=n, train="",
                              premasked=False, no_alt_fp=True, inject=False))

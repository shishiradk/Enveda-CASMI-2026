"""Find which query groups of prvsiyan's rank_train.npz are our S1/S2 molecules (the 250 enveda-np-examples targets).

    python research/bench/bench_pvgroups.py [TAG=e1]

rank_train.npz ships no identities, only a group id G and the simulation flag M (0 = class-1 sim, 1 = class-2 sim).
Per group the base block carries constants of the query: column 13 = similarity of the best analog, column 2 = best
library similarity among the candidates. The bench records the same quantities for every S1 / S2 molecule, so groups
are matched by nearest neighbour in (top_sim S1, top_sim S2, lvmax S1). The match is reported, not assumed: the script
prints the distance distribution and how the matched ids cluster. Output: results/bench/cache/pv_np_groups.json, read
by bench.py (load_rankers) to fit the pv_cv ranker without those groups.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "bench"
tag = sys.argv[1] if len(sys.argv) > 1 else "e1"
z = np.load(ROOT / "external/bench_inputs/rank/rank_train.npz")
X, G, M = z["X"], z["G"], z["M"].astype(int)
ng = int(G.max()) + 1
feat = np.zeros((ng, 3), np.float32)
for m, cols in ((0, ((0, 13), (2, 2))), (1, ((1, 13),))):
    first = {}
    for i in np.where(M == m)[0]:
        first.setdefault(int(G[i]), i)
    for g, i in first.items():
        for j, c in cols:
            feat[g, j] = X[i, c]
R = {s: pickle.load(open(OUT / tag / f"recs_{s}.pkl", "rb"))["recs"] for s in ("S1", "S2")}
mine = {}
for s, j in (("S1", 0), ("S2", 1)):
    for r in R[s]:
        v = mine.setdefault(r["mid"], np.zeros(3, np.float32))
        v[j] = float(r["ana"][0][1]) if r.get("ana") else 0.0
        if s == "S1":
            v[2] = float(r["lib"].max()) if len(r.get("lib", [])) else 0.0
mids = sorted(mine)
A = np.stack([mine[m] for m in mids])
D = np.abs(A[:, None, :] - feat[None, :, :]).max(2)
best = D.argmin(1); dist = D.min(1)
second = np.sort(D, 1)[:, 1]
ok = dist < 0.02
print(f"molecules {len(mids)} | matched within 0.02: {int(ok.sum())} | median distance {np.median(dist):.4f} "
      f"| median second-best {np.median(second):.4f} | unique groups {len(set(best[ok].tolist()))}")
print("matched group ids: min", best[ok].min(), "max", best[ok].max(), "| share with id < 250:", round(float((best[ok] < 250).mean()), 3))
hist = np.bincount(best[ok] // 50, minlength=17)
print("matched ids per block of 50:", hist.tolist())
groups = sorted(set(best[ok].tolist()))
lo, hi = int(np.percentile(best[ok], 1)), int(np.percentile(best[ok], 99))
contiguous = (best[ok] < 250).mean() > 0.95
if contiguous:  # the matched ids form the first block -> drop the whole block (also covers the unmatched few)
    groups = list(range(250))
json.dump(dict(groups=groups, matched=int(ok.sum()), n=len(mids), contiguous_first_250=bool(contiguous),
               median_dist=float(np.median(dist))), open(OUT / "cache" / "pv_np_groups.json", "w"))
print("groups to drop:", len(groups), "| contiguous first 250:", bool(contiguous))

"""Does a bigger class-3 generation budget raise MRR@25 under the existing ranker_v0?  python research/scratch/gen_budget_c3.py --n 500"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/v4n_rebuild")); sys.path.insert(0, str(ROOT / "research/v4n_rebuild/sim"))
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
import simulate as S, lightgbm as lgb
S._init(str(ROOT / "results/v4n/tables"), [str(ROOT / "models/cft_hoR_a/out_cft/export/cft_hoR_a.pt")], 4, True)
E = S.W["E"]
mdl = pickle.load(open(ROOT / "results/v4n/sim/hoR_RA/ranker_v0.pkl", "rb"))
boost = [lgb.Booster(model_str=b) for b in mdl["boosters"]]
q = pd.read_parquet(ROOT / "results/v4n/sim/hoR_RA/queries.parquet")
q = q[q.regime == "c3"].sample(a.n, random_state=a.seed).reset_index(drop=True)
CFG = {"engine(6x60,150)": dict(gen_n_analog=6, gen_max_per_parent=60, gen_max_total=150),
       "12x60,400": dict(gen_n_analog=12, gen_max_per_parent=60, gen_max_total=400),
       "25x60,800": dict(gen_n_analog=25, gen_max_per_parent=60, gen_max_total=800)}
res = {k: dict(rr=[], has=[], ncand=[]) for k in CFG}
t0 = time.time()
for i, r in q.iterrows():
    for name, kw in CFG.items():
        for k, v in kw.items():
            setattr(E.cfg, k, v)
        rows, qs = S._run_one(r.to_dict())
        n = len(rows); rr = 0.0; has = int(rows.y.max() > 0) if n else 0
        if n and has:
            X = rows[["f_" + f for f in mdl["features"]]].values.astype(np.float32)
            p = np.mean([b.predict(X) for b in boost], 0)
            rank = 1 + int((p > p[rows.y.values.argmax()]).sum())
            rr = 1.0 / rank if rank <= 25 else 0.0
        res[name]["rr"].append(rr); res[name]["has"].append(has); res[name]["ncand"].append(n)
    if (i + 1) % 25 == 0:
        print(i + 1, f"{time.time()-t0:.0f}s", {k: (round(np.mean(v["has"]), 3), round(np.mean(v["rr"]), 4)) for k, v in res.items()}, flush=True)
rep = {k: dict(recall=float(np.mean(v["has"])), mrr25=float(np.mean(v["rr"])), median_cands=float(np.median(v["ncand"]))) for k, v in res.items()}
json.dump(rep, open(ROOT / "results/v4n/gen_budget_c3.json", "w"), indent=1); print(json.dumps(rep, indent=1))

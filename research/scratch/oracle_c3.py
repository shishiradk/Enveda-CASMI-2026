"""Class-3 oracle: can the truth be derived (engine grammar, <=2 steps) from a structurally close, mass-compatible
library structure?  Compares with the engine's actual candidate recall in the same simulated queries.

    python research/scratch/oracle_c3.py --n 600 --out results/v4n/oracle_c3.json
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "v4n_rebuild"))
from engine import derive  # noqa: E402

LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
TOL = 0.003


def tanimoto(a, B):
    inter = LUT[np.bitwise_and(B, a)].sum(1, dtype=np.int32)
    na = int(LUT[a].sum()); nb = LUT[B].sum(1, dtype=np.int32)
    return inter / np.maximum(na + nb - inter, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/v4n/oracle_c3.json"))
    ap.add_argument("--ext", default=None, help="JSON list of {name, smarts, delta} transforms appended to the grammar (test only)")
    a = ap.parse_args()
    if a.ext:
        for t in json.load(open(a.ext)):
            assert t["name"] not in derive.NAMES, t["name"]
            derive.GRAMMAR.append((t["name"], t["smarts"], t["delta"])); derive.NAMES.append(t["name"])
            derive.DELTA[t["name"]] = derive._delta(t["delta"]); derive._SMARTS[t["name"]] = t["smarts"]
        derive._COMBOS.clear()
        print("grammar extended:", len(derive.NAMES), "transforms,", len(derive.combos()), "combos", flush=True)
    T = ROOT / "results/v4n/tables"
    st = pd.read_parquet(T / "train_structs.parquet", columns=["sid", "inchikey14", "mass", "smiles"])
    fp = np.load(T / "train_fp_raw.npy", mmap_mode="r")
    mass = st.mass.values.astype(np.float64)
    order = np.argsort(mass, kind="mergesort"); ms = mass[order]
    ik = st.inchikey14.values
    q = pd.read_parquet(ROOT / "results/v4n/sim/hoR_RA/queries.parquet", columns=["qid", "regime", "sid0", "target"])
    q = q[q.regime == "c3"].sample(a.n, random_state=a.seed).reset_index(drop=True)
    # engine's actual recall for these queries (truth among candidates, any rank)
    import duckdb
    rows = glob.glob(str(ROOT / "results/v4n/sim/hoR_RA/rows/*.parquet"))
    ids = ",".join(map(str, q.qid.tolist()))
    hit = duckdb.sql(f"select qid, max(y) y from read_parquet({[r.replace(chr(92), '/') for r in rows]}) "
                     f"where qid in ({ids}) group by qid").df().set_index("qid").y
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDLogger.DisableLog("rdApp.*")
    combos = derive.combos()
    cm = np.array([m for _, m in combos]); cn = np.array([len(c) for c, _ in combos])
    out = []
    t0 = time.time()
    for i, r in q.iterrows():
        sid = int(r.sid0); M = float(st.mass.values[sid]); tik = ik[sid]
        # parents whose mass + some combo mass = M
        pars = set()
        for c in range(len(cm)):
            lo = np.searchsorted(ms, M - cm[c] - TOL, "left"); hi = np.searchsorted(ms, M - cm[c] + TOL, "right")
            if hi > lo:
                pars.update(order[lo:hi].tolist())
        pars = np.array([p for p in pars if ik[p] != tik], np.int64)
        rec = dict(qid=int(r.qid), n_parents=int(len(pars)), engine_has_truth=int(hit.get(int(r.qid), 0) > 0),
                   best_tan=None, reach_top1=0, reach_top5=0, reach_top30=0, reach_1step=0, rank_of_hit=None)
        if len(pars):
            tan = tanimoto(np.asarray(fp[sid]), np.asarray(fp[pars]))
            o = np.argsort(-tan, kind="stable")[:a.topn]
            rec["best_tan"] = float(tan[o[0]])
            for rank, j in enumerate(o):
                p = int(pars[j])
                got = False
                for smi, combo in derive.derive(st.smiles.values[p], M - mass[p], tol=TOL, max_steps=2, max_out=300):
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        continue
                    try:
                        k = inchi.MolToInchiKey(m)[:14]
                    except Exception:
                        continue
                    if k == tik:
                        got = True
                        rec["reach_1step"] = int(rec["reach_1step"] or len(combo) == 1)
                        break
                if got:
                    rec["rank_of_hit"] = rank
                    rec["reach_top30"] = 1; rec["reach_top5"] = int(rank < 5); rec["reach_top1"] = int(rank < 1)
                    break
        out.append(rec)
        if (i + 1) % 50 == 0:
            d = pd.DataFrame(out)
            print(f"{i+1}/{len(q)} {time.time()-t0:.0f}s | reach top30 {d.reach_top30.mean():.3f} top5 {d.reach_top5.mean():.3f} "
                  f"top1 {d.reach_top1.mean():.3f} | engine {d.engine_has_truth.mean():.3f}", flush=True)
    d = pd.DataFrame(out)
    both = (d.reach_top30 == 1) & (d.engine_has_truth == 1)
    rep = dict(n=len(d), engine_recall=float(d.engine_has_truth.mean()),
               has_mass_compatible_parent=float((d.n_parents > 0).mean()),
               oracle_reach_top30=float(d.reach_top30.mean()), oracle_reach_top5=float(d.reach_top5.mean()),
               oracle_reach_top1=float(d.reach_top1.mean()),
               reach_by_1_step_only=float(d.reach_1step.mean()),
               reached_and_engine_found=float(both.mean()),
               reached_but_engine_missed=float(((d.reach_top30 == 1) & (d.engine_has_truth == 0)).mean()),
               engine_found_but_not_reached=float(((d.reach_top30 == 0) & (d.engine_has_truth == 1)).mean()),
               best_parent_tanimoto_median=float(d.best_tan.dropna().median()) if d.best_tan.notna().any() else None,
               best_parent_tanimoto_ge_0p7=float((d.best_tan.fillna(0) >= 0.7).mean()),
               median_parents=float(d.n_parents.median()), seconds=round(time.time() - t0))
    json.dump(dict(summary=rep, rows=out), open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

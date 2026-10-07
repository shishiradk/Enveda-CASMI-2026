"""Query-held-out refit of BOTH rankers for the S1/S2 molecules (the clean answer to the ranker-row leak).

    python research/bench/bench_cvrank.py [TAG=e1] [WORKERS=3]

Both shipped ranker training sets contain rows simulated from the very 250 enveda-np-examples molecules that S1/S2
evaluate, in sorted-InChIKey order (verified here against the bench's own channel values, they agree to 1e-4):
  rank_train.npz (prvsiyan)        group g      = molecule g            (g < 250; groups 250..818 are other structures)
  sim_rank_rows_nofp.npz (megayak) groups 2g, 2g+1 = molecule g         (query set "np")
bench.py's ours_cv / pv_cv drop ALL of these rows, which also removes every timsTOF natural-product training query and
is therefore pessimistic (a domain shift on top of the leak). Here each molecule is scored by rankers fitted on
everything except the rows of its own fold (5 folds, fold = g mod 5), i.e. the author's "held out by query group" CV.
Adds score sets pv_kf, ours_kf and clean_kf = 0.88 * rank(pv_kf) + 0.12 * rank(ours_kf) to <TAG>/S1.pkl and S2.pkl.
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

NF = 5

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "e1"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bench.setup_env(HERE / "eng", workers)
    import numpy as np
    import rdkit
    import casmi_engine as E
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    out = bench.OUT / tag
    pc = bench.OUT / "cache" / f"pool_{bench.TRAIN.stem}_{bench.TRAIN.stat().st_size}_{rdkit.__version__}.pkl"
    E.POOL = P = pickle.load(open(pc, "rb"))
    recs = {s: pickle.load(open(out / f"recs_{s}.pkl", "rb"))["recs"] for s in ("S1", "S2")}
    mids = sorted(r["mid"] for r in recs["S1"])
    assert mids == sorted(r["mid"] for r in recs["S2"]) and len(mids) == 250
    gi = {m: i for i, m in enumerate(mids)}
    zp = np.load(E.find("rank_train.npz"))
    zs = np.load(E.find("sim_rank_rows_nofp.npz"), allow_pickle=True)
    blocks = [str(b) for b in zs["blocks"]]
    npq = list(zs["sets"]).index("np")
    # ---- identity check: top analog similarity (base column 13) of the class-1 / class-2 rows vs the bench records
    ok = {"pv": [], "mk": []}
    for s, flag in (("S1", 0), ("S2", 1)):
        for r in recs[s]:
            if not r.get("ana"):
                continue
            g, t = gi[r["mid"]], float(r["ana"][0][1])
            i = np.where((zp["G"] == g) & (zp["M"] == flag))[0]
            j = np.where((zs["Q"] == npq) & (zs["G"] == 2 * g + flag))[0]
            if len(i): ok["pv"].append(abs(float(zp["X"][i[0], 13]) - t) < 1e-3)
            if len(j): ok["mk"].append(abs(float(zs["X"][j[0], 13]) - t) < 1e-3)
    bench.log(f"group identity check (top analog sim equal within 1e-3): rank_train {np.mean(ok['pv']):.3f} of {len(ok['pv'])}, "
              f"megayak sim {np.mean(ok['mk']):.3f} of {len(ok['mk'])}")
    gbm = dict(W_A=E.RANK.W_A, SEEDS=E.RANK.SEEDS, GBM=E.RANK.GBM, PVGBM=E.CFG.GBM, sk=__import__("sklearn").__version__)
    S = {s: {"pv_kf": {}, "ours_kf": {}} for s in recs}
    for f in range(NF):
        def fit_pv(f=f):
            keep = (zp["G"] >= 250) | (zp["G"] % NF != f)
            X, Y, M = zp["X"][keep], zp["Y"][keep], zp["M"][keep]
            return [E._hgb(X, Y, np.where(M == 0, w1, 1.0 - w1), sd, E.CFG.GBM) for w1 in (0.30, 0.60) for sd in (0, 1, 2, 3)]

        def fit_mk(f=f):
            keep = (zs["Q"] != npq) | ((zs["G"] // 2) % NF != f)
            X, Y, Sx = zs["X"][keep], zs["Y"][keep], zs["S"][keep]
            return [E._hgb(X, Y, np.where(Sx == 0, wA, 1.0 - wA), sd, E.RANK.GBM) for wA in E.RANK.W_A for sd in E.RANK.SEEDS]
        pvr = bench.fit_or_load(f"pv_kf{f}", fit_pv, dict(gbm, nf=NF))
        mkr = bench.fit_or_load(f"ours_kf{f}", fit_mk, dict(gbm, nf=NF))
        for s in recs:
            sub = [r for r in recs[s] if gi[r["mid"]] % NF == f]
            S[s]["pv_kf"].update(E.score_molecules(sub, pvr, ["base"], use_fp=True))
            S[s]["ours_kf"].update(E.score_molecules(sub, mkr, blocks, use_fp=False))
        bench.log(f"fold {f} done")
    for s in recs:
        S[s]["clean_kf"] = E.blend_scores(S[s]["pv_kf"], S[s]["ours_kf"], bench.W_PV)
        R = pickle.load(open(out / f"{s}.pkl", "rb"))
        need = {}
        for r in recs[s]:
            if not len(r["cand"]):
                continue
            u = set()
            for sc in S[s].values():
                u |= set(np.argsort(-sc[r["mid"]], kind="mergesort")[:80].tolist())
            need[r["mid"]] = (r["cand"], sorted(u - set(R[r["mid"]]["top"])))
        C = bench.canon_many(E, [P["smiles"][c[i]] for c, v in need.values() for i in v], workers)
        for m, (c, v) in need.items():
            assert np.array_equal(P["keys"][c].astype(str), R[m]["keys"]), "S pkl and recs disagree on the candidates"
            for i in v:
                R[m]["top"][int(i)] = (P["smiles"][c[i]], C.get(P["smiles"][c[i]]))
            for nm, sc in S[s].items():
                R[m]["scores"][nm] = sc[m].astype(np.float32)
        pickle.dump(R, open(out / f"{s}.pkl", "wb"))
        bench.log(f"{s}: pv_kf / ours_kf / clean_kf added")

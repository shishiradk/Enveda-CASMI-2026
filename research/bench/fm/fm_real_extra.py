"""E3c study, supporting checks (research/analysis/e3c_real_score.md).

    python research/bench/fm/fm_real_extra.py [WORKERS=3]     (after fm_real.py)   -> results/bench/fm/real_extra.txt

1. Provenance of the raw ranker scores: clean_kf must be exactly blend_scores(pv_kf, ours_kf); the stored pv_kf / ours_kf of
   fold f must be reproduced by the cached fold-f rankers (fitted without the rows of the fold's molecules) and must
   differ from what the shipped (leaky) rankers and another fold's rankers give.
2. Selection under leaderboard-like class mixes: the criterion and the reported number are w1 * S1 + w2 * S2.
3. Representative rules on the as-submitted lists (blend; raw scores pv / ours) of S1, S2 and S3.
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

if __name__ == "__main__":
    import bench
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    bench.setup_env(HERE.parent / "eng", workers)
    import numpy as np
    import rdkit
    import casmi_engine as E
    import fm_lib as F
    import fm_real as R
    lines = []

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    # ---------------- 1. provenance ----------------
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    pr("## 1. provenance of pv_kf / ours_kf")
    for s in ("S1", "S2"):
        S = pickle.load(open(bench.OUT / "e1" / f"{s}.pkl", "rb"))
        sc = {k: {m: r["scores"][k] for m, r in S.items() if "scores" in r and "pv_kf" in r["scores"]} for k in ("pv_kf", "ours_kf", "clean_kf", "pv", "ours")}
        bl = E.blend_scores({m: v.astype(np.float64) for m, v in sc["pv_kf"].items()}, {m: v.astype(np.float64) for m, v in sc["ours_kf"].items()}, bench.W_PV)
        same_order = np.mean([np.array_equal(np.argsort(-bl[m], kind="mergesort")[:60], np.argsort(-sc["clean_kf"][m].astype(np.float64), kind="mergesort")[:60]) for m in bl])
        dmax = max(float(np.abs(bl[m] - sc["clean_kf"][m]).max()) for m in bl)
        pr(f"{s}: clean_kf vs blend_scores(pv_kf, ours_kf): max abs difference {dmax:.2e}, identical top-60 order for {same_order:.3f} of {len(bl)} molecules")
    pc = bench.OUT / "cache" / f"pool_{bench.TRAIN.stem}_{bench.TRAIN.stat().st_size}_{rdkit.__version__}.pkl"
    E.POOL = pickle.load(open(pc, "rb"))
    recs = pickle.load(open(bench.OUT / "e1" / "recs_S1.pkl", "rb"))["recs"]
    S1 = pickle.load(open(bench.OUT / "e1" / "S1.pkl", "rb"))
    mids = sorted(r["mid"] for r in recs)
    gi = {m: i for i, m in enumerate(mids)}
    gbm = dict(W_A=E.RANK.W_A, SEEDS=E.RANK.SEEDS, GBM=E.RANK.GBM, PVGBM=E.CFG.GBM, sk=__import__("sklearn").__version__)

    def nofit():
        raise RuntimeError("fold ranker cache missing: run bench_cvrank.py")
    sub = [r for r in recs if gi[r["mid"]] % 5 == 0 and len(r["cand"])]
    for f in (0, 1):
        pvr = bench.fit_or_load(f"pv_kf{f}", nofit, dict(gbm, nf=5))
        got = E.score_molecules(sub, pvr, ["base"], use_fp=True)
        d = [float(np.abs(got[r["mid"]] - S1[r["mid"]]["scores"]["pv_kf"]).max()) for r in sub]
        pr(f"S1 fold-0 molecules ({len(sub)}) scored with the fold-{f} pv rankers: max |difference to stored pv_kf| = {max(d):.2e}, "
           f"molecules identical within 1e-6: {int(np.sum(np.array(d) < 1e-6))}")
    d = [float(np.abs(S1[r["mid"]]["scores"]["pv"] - S1[r["mid"]]["scores"]["pv_kf"]).max()) for r in sub]
    pr(f"stored shipped-ranker pv vs stored pv_kf on the same molecules: max |difference| median {np.median(d):.3f}, identical {int(np.sum(np.array(d) < 1e-6))}")
    zp = np.load(E.find("rank_train.npz"))
    pr(f"rank_train.npz: rows of groups < 250 with G % 5 == 0 (excluded from the fold-0 fit): {int(((zp['G'] < 250) & (zp['G'] % 5 == 0)).sum())} of {len(zp['G'])}")
    del E.POOL

    # ---------------- 2. leaderboard-like mixes ----------------
    Rd = pickle.load(open(F.FM / "real_score_delta.pkl", "rb"))
    names = [k for k in Rd if not k.startswith("control")]
    real = [k for k in names if not k.startswith("rank")]
    n = len(Rd[R.E3B]["S1"])
    pr("\n## 2. selection and report under a class mix w1 * S1 + w2 * S2 (the rest of the test set cannot change); 20 splits x 2")
    pr("| mix (w1, w2) | family | held-out vs E1 | held-out vs E3b | E3b itself vs E1 | most picked |\n|---|---|---|---|---|---|")
    M1 = {k: Rd[k]["S1"] for k in names}; M2 = {k: Rd[k]["S2"] for k in names}
    for w1, w2 in ((0.4, 0.0), (0.3, 0.05), (0.2, 0.1), (0.0, 0.1), (0.5, 0.5)):
        for fn, ks in (("real-score terms (gated or not)", real), ("everything incl. E3b", names)):
            rng = np.random.default_rng(7)
            held = np.zeros(n); picks = {}
            A = np.array([w1 * M1[k] + w2 * M2[k] for k in ks])
            for _ in range(20):
                p = rng.permutation(n)
                for a, b in ((p[: n // 2], p[n // 2:]), (p[n // 2:], p[: n // 2])):
                    j = int(np.argmax(A[:, a].mean(1)))
                    picks[ks[j]] = picks.get(ks[j], 0) + 1
                    held[b] += A[j, b] / 20
            e3b = w1 * M1[R.E3B] + w2 * M2[R.E3B]
            top = max(picks, key=picks.get)
            pr(f"| {w1}, {w2} | {fn} | {F.fmtd(F.boot(held))} | {F.fmtd(F.boot(held - e3b))} | {F.fmtd(F.boot(e3b))} | {top} x{picks[top]} |")

    # ---------------- 3. as-submitted lists, S3 ----------------
    D, _ = F.load()
    pr("\n## 3. as-submitted rankers (blend lists, raw scores pv / ours), versus no re-ordering")
    pr("| rule | S1 | S2 | S3 (46 listed truths of 300) |\n|---|---|---|---|")
    rules = {"E3b (rank lam 0.5, protect 0.6)": dict(term="rank", lam=0.5, protect=0.6), "E3 (rank lam 1)": dict(term="rank", lam=1.0, protect=None),
             "zl_bl lam 0.5": dict(term="zl_bl", lam=0.5, protect=None), "zl_bl lam 1.0": dict(term="zl_bl", lam=1.0, protect=None),
             "abs a 1.5": dict(term="abs", lam=1.5, protect=None), "zl_bl lam 1.0, protect 0.6": dict(term="zl_bl", lam=1.0, protect=0.6)}
    for nm, kw in rules.items():
        c = [F.fmtd(F.boot(R.deltas(D, s, kw, ss="blend", pv="pv", ours="ours"))) for s in ("S1", "S2", "S3")]
        pr(f"| {nm} | " + " | ".join(c) + " |")
    (F.FM / "real_extra.txt").write_text("\n".join(lines), encoding="utf-8")

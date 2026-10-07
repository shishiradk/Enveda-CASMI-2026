"""Real-score rules on the labelled visible test molecules (the only real-test data with labels).

    python research/bench/fm/visible_real_check.py <downloaded E3 kernel output dir> [WORKERS=3]

E3's Kaggle run exported the rank blend only, so the raw ranker probabilities are recomputed locally for the 40 visible
molecules of the parity run (results/bench/parity/recs_T.pkl: unmasked engine channels) with the shipped rankers
(cached fits; no leak question here: the visible molecules are not in the rankers' training rows) and joined on SMILES
to the Kaggle lists and the Kaggle forward scores.  Base order = the Kaggle E1 order.
"""
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

if __name__ == "__main__":
    import bench
    d = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bench.setup_env(HERE.parent / "eng", workers)
    import numpy as np
    import rdkit
    import casmi_engine as E
    import fm_lib as F
    import fm_real as R
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    pc = bench.OUT / "cache" / f"pool_{bench.TRAIN.stem}_{bench.TRAIN.stat().st_size}_{rdkit.__version__}.pkl"
    E.POOL = P = pickle.load(open(pc, "rb"))
    RK = bench.load_rankers(E)
    recs = pickle.load(open(bench.OUT / "parity" / "recs_T.pkl", "rb"))
    s_pv = E.score_molecules(recs, *RK["pv"], use_fp=True)
    s_ours = E.score_molecules(recs, *RK["ours"], use_fp=False)
    L = json.load(open(d / "eng_lists.json"))
    ice = json.load(open(d / "fm_ice.json"))["scores"]
    gl = json.load(open(d / "fm_gl.json"))["scores"]
    inp = json.load(open(d / "fm_input.json"))
    lab = json.load(open(F.FM / "visible_test_labels.json"))
    from visible_test_check import canon_key
    lists, miss = {}, 0
    for r in recs:
        m = str(r["mid"])
        if m not in L or r["mid"] not in s_pv or m not in inp:
            continue
        e = L[m]
        by = {P["smiles"][c]: i for i, c in enumerate(r["cand"])}
        idx = [by.get(s) for s in e["smiles"]]
        miss += sum(i is None for i in idx)
        if any(i is None for i in idx):
            continue
        keys, smi = lab[m]
        tk = set(keys) | {canon_key(smi)}
        pos = {s: i for i, s in enumerate(inp[m]["smiles"])}
        lists[m] = dict(smiles=e["smiles"], scores=e["scores"], ok=[k in tk for k in e["keys"]], formula=[F.formula(s) for s in e["smiles"]],
                        pv_kf=[float(s_pv[r["mid"]][i]) for i in idx], ours_kf=[float(s_ours[r["mid"]][i]) for i in idx],
                        lib=[float(r["lib"][i]) for i in idx],
                        ice=[ice[m][pos[s]] if s in pos else None for s in e["smiles"]],
                        gl=[gl[m][pos[s]] if s in pos else None for s in e["smiles"]])
    print(f"molecules joined {len(lists)} of {len(recs)} | Kaggle candidates not found locally {miss}")
    rules = {"E3 (rank, lam 1, no protection)": dict(term="rank", lam=1.0, protect=None),
             "rank lam 0.5 no protection": dict(term="rank", lam=0.5, protect=None),
             "rank lam 0.1 no protection": dict(term="rank", lam=0.1, protect=None),
             "E3b (rank, lam 0.5, protect 0.6)": dict(term="rank", lam=0.5, protect=0.6)}
    for term, lam in (("zl_bl", 0.5), ("zl_bl", 1.0), ("zl_pv", 0.5), ("zl_pv", 1.0), ("zl_list", 0.5), ("zl_list", 1.0),
                      ("abs", 0.5), ("abs", 1.0), ("abs", 1.5), ("abs", 2.0), ("abs", 4.0)):
        rules[f"{term} {lam} no protection"] = dict(term=term, lam=lam, protect=None)
    out = {"n": len(lists)}
    marg, ptruth, tlib = [], [], []
    for m, l in lists.items():
        dd = R.prep(l)
        if True in l["ok"]:
            t = l["ok"].index(True)
            others = np.delete(dd["L"], t) if t < dd["top"] else dd["L"]
            marg.append(float(dd["L"][t] - others.max())); ptruth.append(float(dd["ppv"][t])); tlib.append(float(dd["lib"][t]))
    print("truth: pv probability median / min", np.round(np.median(ptruth), 3), np.round(np.min(ptruth), 3),
          "| logit-blend margin over the best other candidate, median / min", np.round(np.median(marg), 2), np.round(np.min(marg), 2),
          "| library similarity median / min", np.round(np.median(tlib), 3), np.round(np.min(tlib), 3))
    out["truth"] = dict(pv_median=float(np.median(ptruth)), pv_min=float(np.min(ptruth)), margin_median=float(np.median(marg)),
                        margin_min=float(np.min(marg)), lib_median=float(np.median(tlib)), lib_min=float(np.min(tlib)))
    for name, kw in rules.items():
        a = np.array([F.rr_of(l["ok"]) for l in lists.values()])
        b = np.array([F.rr_of(l["ok"], R.perm_of(R.prep(l), **kw)) for l in lists.values()])
        out[name] = dict(delta=float((b - a).mean()), demoted=int(((a == 1) & (b < 1)).sum()), base_top1=int((a == 1).sum()))
        print(f"{name}: MRR change {(b - a).mean():+.4f} | correct top-1 demoted {out[name]['demoted']} of {out[name]['base_top1']}")
    json.dump(out, open(F.FM / "visible_real_check.json", "w"), indent=1)

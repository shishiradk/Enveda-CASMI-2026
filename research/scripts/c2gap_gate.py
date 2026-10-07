"""C2 gap diagnosis, stage 6: re-ranked channel + a confidence gate that moves the channel's #1 up, on all buckets.

    python research/scripts/c2gap_gate.py

Candidates: results/c2gap/cand{,_S1,_SV}.parquet (ho1 top-100 per molecule + truth row). Re-ranker as in
c2gap_rerank.py (LightGBM lambdarank), trained on PC+S2 molecules only, 5 folds by molecule id (S1 and S2 share
molecules, so they share folds). SV (enveda-180 visible, re-measured) is never trained on.
Merged rank is simulated from engine rank e and channel rank c with a slot pattern (dedup ignored; it was within
0.006 of the real E6 merge). Patterns: E = m1_alt (e0 e1 p0 e2 p1 ...), EP (e0 p0 e1 p1 ...), PE (p0 e0 p1 e1 ...).
Gate: aggressive pattern when engine top-1 library similarity < L and the re-ranker's top-1 softmax share > T.
Thresholds: in-sample grid (optimistic) and 2-fold cross-fit by molecule (honest). Proxy = .146 SV + .21 S1 + .111 PC.
Writes results/c2gap/gate.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
from c2gap_rerank import featurize, add_rel, SETS  # noqa: E402

D = ROOT / "results" / "c2gap"
WTS = dict(SV=0.146, S1=0.21, PC=0.111)


def seq_pos(pat, n=60):
    labels = list(pat) + ["E", "P"] * n
    pe, pp, ie, ip = {}, {}, 0, 0
    for i, l in enumerate(labels, 1):
        if l == "E":
            pe[ie] = i; ie += 1
        else:
            pp[ip] = i; ip += 1
    return pe, pp


POS = {p: seq_pos(p) for p in ("E", "EP", "PE")}


def merged(e, c, pat):
    pe, pp = POS[pat]
    cand = ([pe.get(e - 1, 999)] if e > 0 else []) + ([pp.get(c - 1, 999)] if c > 0 else [])
    r = min(cand) if cand else 0
    return r if r <= 25 else 0


def rr(r):
    return 1.0 / r if r > 0 else 0.0


def main():
    import lightgbm as lgb
    cs, ms = [], []
    for tag in ("", "_S1", "_SV"):
        cs.append(pd.read_parquet(D / f"cand{tag}.parquet")); ms.append(pd.read_parquet(D / f"mol{tag}.parquet"))
    c = pd.concat(cs, ignore_index=True); mol = pd.concat(ms, ignore_index=True)
    c = c[c.b != "C3"].reset_index(drop=True); mol = mol[mol.b != "C3"].reset_index(drop=True)
    c = add_rel(featurize(c), mol)
    mids = sorted(mol.mid.unique())
    fold = dict(zip(mids, np.random.default_rng(0).permutation(len(mids)) % 5))
    c["fold"] = c.mid.map(fold)
    res = {}
    for fs_name in ("ho1+struct", "ho1+struct+pop"):
        fs = SETS[fs_name]
        preds = np.zeros(len(c))
        for k in range(5):
            tr = c[(c.fold != k) & c.b.isin(["PC", "S2"])]
            tr = tr[tr.groupby(["scen", "mid"]).is_truth.transform("any")].sort_values(["scen", "mid"])
            ds = lgb.Dataset(tr[fs], tr.is_truth.astype(int), group=tr.groupby(["scen", "mid"], sort=False).size().values)
            mdl = lgb.train(dict(objective="lambdarank", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50,
                                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=25,
                                 verbose=-1, seed=k, num_threads=2), ds, num_boost_round=200)
            te = c.fold == k
            preds[te.values] = mdl.predict(c.loc[te, fs])
        if fs_name == "ho1+struct":
            imp = pd.Series(mdl.feature_importance("gain"), index=fs).sort_values(ascending=False)
            res["importance_ho1+struct_lastfold"] = (imp / imp.sum()).round(3).to_dict()
        c["pred"] = preds
        rows = []
        for (s, m), g in c.groupby(["scen", "mid"], sort=False):
            g = g.sort_values("pred", ascending=False)
            t = np.flatnonzero(g.is_truth.values)
            p = g.pred.values
            ex = np.exp(p - p.max())
            rows.append(dict(scen=s, mid=m, cr=int(t[0] + 1) if len(t) and t[0] < 100 else 0,
                             share=float(ex[0] / ex.sum()), margin=float(p[0] - p[1]) if len(p) > 1 else 0.0))
        r = pd.DataFrame(rows).merge(mol[["scen", "mid", "b", "eng_rank", "e6_rank", "ch_rank", "eng_lib1"]], on=["scen", "mid"])

        def score(df, choose):
            out = {}
            for b in ("SV", "S1", "S2", "PC"):
                g = df[df.b == b]
                out[b] = float(np.mean([rr(merged(e, cr, choose(x))) for e, cr, x in zip(g.eng_rank, g.cr, g.itertuples())]))
            out["proxy"] = sum(WTS[b] * out[b] for b in WTS)
            return out
        R = {}
        R["real_E6"] = {b: float(np.mean([rr(x) if x <= 25 else 0 for x in r[r.b == b].e6_rank])) for b in ("SV", "S1", "S2", "PC")}
        R["real_E6"]["proxy"] = sum(WTS[b] * R["real_E6"][b] for b in WTS)
        old = r.assign(cr=r.ch_rank)
        R["sim_E6_old_channel"] = score(old, lambda x: "E")
        R["sim_rerank_E"] = score(r, lambda x: "E")
        R["sim_rerank_EP_all"] = score(r, lambda x: "EP")
        R["sim_rerank_PE_all"] = score(r, lambda x: "PE")
        grid = {}
        for pat in ("EP", "PE"):
            for L in (0.3, 0.5, 0.7, 0.9, 2.0):
                for T in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                    grid[(pat, L, T)] = lambda x, pat=pat, L=L, T=T: pat if (x.eng_lib1 < L and x.share > T) else "E"
        ins = {f"{k[0]}_lib<{k[1]}_share>{k[2]}": score(r, f) for k, f in grid.items()}
        top = sorted(ins.items(), key=lambda kv: -kv[1]["proxy"])[:8]
        R["gate_in_sample_top"] = dict(top)
        # 2-fold cross-fit by molecule id
        half = r.mid.map(lambda m: fold[m] % 2)
        cf_vals = {b: [] for b in ("SV", "S1", "S2", "PC")}
        chosen = []
        for h in (0, 1):
            A, B = r[half == h], r[half != h]
            best = max(grid, key=lambda k: score(A, grid[k])["proxy"])
            chosen.append(f"{best[0]}_lib<{best[1]}_share>{best[2]}")
            for b in cf_vals:
                g = B[B.b == b]
                cf_vals[b] += [rr(merged(e, cr, grid[best](x))) for e, cr, x in zip(g.eng_rank, g.cr, g.itertuples())]
        cfm = {b: float(np.mean(v)) for b, v in cf_vals.items()}
        cfm["proxy"] = sum(WTS[b] * cfm[b] for b in WTS)
        R["gate_crossfit"] = cfm; R["gate_crossfit_chosen"] = chosen
        R["share_median_by_bucket"] = r.groupby("b").share.median().round(3).to_dict()
        res[fs_name] = {k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) and all(isinstance(vv, float) for vv in v.values())
                            else v) for k, v in R.items()}
        print(fs_name, json.dumps(res[fs_name], indent=1, default=float), flush=True)
    json.dump(res, open(D / "gate.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

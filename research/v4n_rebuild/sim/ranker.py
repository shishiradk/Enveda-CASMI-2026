"""Ranker v0 (REBUILD_SPEC 3.9 / 6): 4-seed LightGBM lambdarank on the simulation rows, grouped 5-fold CV by the
split folds F0..F4 (folds are per truth score key), regime weights, f.z-only baseline, feature importance, and the
V5c provenance probe.  Our code, MIT.

    python research/v4n_rebuild/sim/ranker.py --run pilot_ho2 [--rounds 500] [--seeds 0,1,2,3] [--mix c1=0.2,c2=0.55,c3=0.25]

Outputs in results/v4n/sim/<run>/: cv_report.json, cv_pred.parquet (qid, row, pred), ranker_v0.pkl (all-data fit;
dict(features, params, rounds, boosters=[model strings]) ), feature_importance.csv.
Weights: each query gets mix[regime] / n_queries[regime] (rescaled to mean 1), i.e. each regime contributes its
target share and, within a regime, every structure counts once (one query per structure and regime).
Training uses only queries whose candidate list contains the truth (no-positive groups carry no lambdarank gradient);
evaluation uses every query (no positive => reciprocal rank 0).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import time

import numpy as np
import pandas as pd

from common import MIX, OUT_ROOT, REGIMES, mrr_and_top1

PARAMS = dict(objective="lambdarank", metric="None", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_depth=-1,
              lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1, num_threads=int(__import__("os").environ.get("LGB_THREADS", "4")))
CHECKPOINTS = (100, 200, 300, 500)


def load(run):
    from engine.engine import FEATURES
    d = OUT_ROOT / run
    files = sorted(glob.glob(str(d / "rows" / "rows_*.parquet")))
    R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    R = R.sort_values(["qid"], kind="stable").reset_index(drop=True)
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "qstats" / "q_*.parquet")))], ignore_index=True)
    S = S.sort_values("qid").reset_index(drop=True)
    feats = ["f_" + f for f in FEATURES]
    return R, S, feats


def group_sizes(g):
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return np.diff(np.r_[starts, len(g)])


def fit(X, y, g, w, feats, seed, rounds):
    import lightgbm as lgb
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    ds = lgb.Dataset(X, label=y, group=group_sizes(g), weight=w, feature_name=feats, free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=rounds)


def summarise(score, R, Q, mix):
    """Per-regime MRR@25 / top-1 (per query, unweighted inside a regime) + the mix-weighted total."""
    rr, t1 = mrr_and_top1(score, R.y.values, R.qid.values)
    qids = R.qid.values[np.r_[True, R.qid.values[1:] != R.qid.values[:-1]]]
    df = pd.DataFrame(dict(qid=qids, rr=rr, t1=t1)).set_index("qid")
    q = Q.set_index("qid").join(df, how="left").fillna({"rr": 0.0, "t1": 0.0})   # queries with 0 candidates -> 0
    out = {}
    for r in REGIMES:
        s = q[q.regime == r]
        if len(s):
            out[r] = dict(n=int(len(s)), mrr25=round(float(s.rr.mean()), 4), top1=round(float(s.t1.mean()), 4),
                          truth_in_cands=round(float(s.has_pos.mean()), 4),
                          mrr25_if_present=round(float(s.rr[s.has_pos > 0].mean()), 4))
    out["mix"] = dict(mrr25=round(sum(mix[r] * out[r]["mrr25"] for r in mix if r in out) /
                                  sum(mix[r] for r in mix if r in out), 4),
                      top1=round(sum(mix[r] * out[r]["top1"] for r in mix if r in out) /
                                 sum(mix[r] for r in mix if r in out), 4))
    per_fold = {}
    for f, s in q.groupby("fold"):
        per_fold[f] = {r: round(float(s[s.regime == r].rr.mean()), 4) for r in REGIMES if (s.regime == r).any()}
    out["per_fold"] = per_fold
    return out, q


def cv(R, Q, feats, mix, seeds, rounds, extra=None):
    X = R[feats + (extra or [])].values.astype(np.float32)
    y = R.y.values.astype(np.int32)
    g = R.qid.values
    qmeta = Q.set_index("qid")
    rfold = qmeta.fold.reindex(g).values
    rpos = qmeta.has_pos.reindex(g).values > 0
    reg_n = Q.regime.value_counts()
    qw = Q.regime.map(lambda r: mix.get(r, 0.0) / reg_n[r]).values
    qw = qw / qw.mean()
    rw = pd.Series(qw, index=Q.qid).reindex(g).values
    preds = {c: np.zeros(len(y)) for c in sorted({c for c in CHECKPOINTS if c < rounds} | {rounds})}
    gain = np.zeros(len(feats) + len(extra or []))
    T0 = time.time()
    for f in sorted(Q.fold.unique()):
        tr = (rfold != f) & rpos
        te = rfold == f
        P = {c: np.zeros(te.sum()) for c in preds}
        for s in seeds:
            b = fit(X[tr], y[tr], g[tr], rw[tr], feats + (extra or []), s, rounds)
            for c in preds:
                P[c] += b.predict(X[te], num_iteration=c) / len(seeds)
            gain += b.feature_importance("gain")
        for c in preds:
            preds[c][te] = P[c]
        print(f"  fold {f}: train rows {tr.sum():,} test rows {te.sum():,} ({time.time() - T0:.0f}s)", flush=True)
    return preds, gain / gain.sum()


def provenance_probe(score, R, Q):
    """V5c: in c2, where do pool decoys of train origin (src 0) vs COCONUT origin (src 1) land?  Mean rank percentile
    (0 = top) per source, and the share of train-origin decoys among the top-5 decoys vs their share of all decoys."""
    q2 = set(Q.qid[(Q.regime == "c2")])
    m = R.qid.isin(q2).values
    sub = R.loc[m, ["qid", "y", "cand_src"]].copy()
    sub["s"] = score[m]
    sub["pct"] = sub.groupby("qid").s.rank(ascending=False, method="first", pct=True)
    sub["rk"] = sub.groupby("qid").s.rank(ascending=False, method="first")
    d = sub[(sub.y == 0) & (sub.cand_src >= 0)]
    dec = d.copy()
    dec["drk"] = dec.groupby("qid").s.rank(ascending=False, method="first")
    top5 = dec[dec.drk <= 5]
    return dict(pct_train_src=round(float(d.pct[d.cand_src == 0].mean()), 4),
                pct_coconut_src=round(float(d.pct[d.cand_src == 1].mean()), 4),
                share_train_all_decoys=round(float((d.cand_src == 0).mean()), 4),
                share_train_top5_decoys=round(float((top5.cand_src == 0).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--mix", default=",".join(f"{k}={v}" for k, v in MIX.items()))
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    mix = {k: float(v) for k, v in (x.split("=") for x in a.mix.split(","))}
    seeds = [int(s) for s in a.seeds.split(",")]
    d = OUT_ROOT / a.run
    R, Q, feats = load(a.run)
    Q = Q[Q.qid.isin(R.qid.unique()) | (Q.n_cand == 0)].copy()
    Q["has_pos"] = Q.has_pos.clip(lower=0)
    print(f"rows {len(R):,} queries {len(Q):,} positives {int(R.y.sum()):,} "
          f"by regime {Q.regime.value_counts().to_dict()}", flush=True)
    t0 = time.time()
    preds, imp = cv(R, Q, feats, mix, seeds, a.rounds)
    cv_sec = time.time() - t0
    fz = R["f_fz"].values.astype(np.float64)
    base, _ = summarise(fz, R, Q, mix)
    rep = dict(run=a.run, rows=int(len(R)), queries=int(len(Q)), positives=int(R.y.sum()), mix=mix, seeds=seeds,
               rounds=a.rounds, params=PARAMS, cv_seconds=round(cv_sec), fz_baseline=base, ranker={})
    for c, p in preds.items():
        rep["ranker"][f"it{c}"], qdf = summarise(p, R, Q, mix)
    final = preds[max(preds)]
    pd.DataFrame(dict(qid=R.qid.values, row=np.arange(len(R)), pred=final)).to_parquet(d / "cv_pred.parquet",
                                                                                       index=False)
    fi = pd.DataFrame(dict(feature=[f[2:] for f in feats], gain_share=imp)).sort_values("gain_share", ascending=False)
    fi.to_csv(d / "feature_importance.csv", index=False)
    rep["feature_importance_top20"] = {r.feature: round(float(r.gain_share), 4) for r in fi.head(20).itertuples()}
    rep["features_zero_gain"] = [r.feature for r in fi.itertuples() if r.gain_share == 0]
    if not a.no_probe:
        rep["v5c_provenance"] = dict(ranker=provenance_probe(final, R, Q), fz=provenance_probe(fz, R, Q))
        # src-shortcut probe: would an explicit candidate-origin feature help?  (1 seed, same rounds)
        R["src_feat"] = R.cand_src.astype(np.float32)
        p_src, _ = cv(R, Q, feats, mix, [0], a.rounds, extra=["src_feat"])
        p_ref, _ = cv(R, Q, feats, mix, [0], a.rounds)
        s_src, _ = summarise(p_src[max(p_src)], R, Q, mix)
        s_ref, _ = summarise(p_ref[max(p_ref)], R, Q, mix)
        rep["v5c_provenance"]["src_feature_probe"] = {
            r: dict(without=s_ref[r]["mrr25"], with_src=s_src[r]["mrr25"],
                    delta=round(s_src[r]["mrr25"] - s_ref[r]["mrr25"], 4)) for r in REGIMES if r in s_ref}
    # all-data fit (pilot ranker; only meaningful as a smoke artefact)
    X = R[feats].values.astype(np.float32)
    pos = Q.set_index("qid").has_pos.reindex(R.qid.values).values > 0
    reg_n = Q.regime.value_counts()
    qw = Q.regime.map(lambda r: mix.get(r, 0.0) / reg_n[r]).values
    rw = pd.Series(qw / qw.mean(), index=Q.qid).reindex(R.qid.values).values
    boosters = [fit(X[pos], R.y.values[pos], R.qid.values[pos], rw[pos], feats, s, a.rounds).model_to_string()
                for s in seeds]
    pickle.dump(dict(features=[f[2:] for f in feats], params=PARAMS, rounds=a.rounds, boosters=boosters, mix=mix,
                     run=a.run), open(d / "ranker_v0.pkl", "wb"))
    json.dump(rep, open(d / "cv_report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("fz_baseline", "ranker")}, indent=1))
    print(json.dumps(rep.get("v5c_provenance", {}), indent=1))
    print("top features", rep["feature_importance_top20"])


if __name__ == "__main__":
    main()

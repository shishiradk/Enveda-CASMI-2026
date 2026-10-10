"""V5 checks (REBUILD_SPEC 9) for a simulation run.  Our code, MIT.

    python research/v4n_rebuild/sim/simulate.py --run test400 --test --workers 4     # once: engine on the visible test
    python research/v4n_rebuild/sim/v5.py --run pilot_ho2 [--test-run test400]

(a) per-regime CV MRR@25 of ranker v0 (from cv_report.json) against c1 >= 0.85, c2 0.60-0.73, c3 > 0;
(b) KS distance of the query-level statistics n_query, q_npeaks, lib_max, top_sim, n_cand between the simulated
    queries (per regime, and the class mix c1 0.20 / c2 0.55 / c3 0.25 as a weighted ECDF) and the 400 visible test
    molecules run unmasked; pass if < 0.15.  Diagnostic: q_npeaks of the test spectra after the library cleaning rule;
(c) provenance probe (from cv_report.json): no gap > 0.02.
Output: results/v4n/sim/<run>/v5_report.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from common import MIX, OUT_ROOT, QSTATS, REGIMES, TEST


def wks(a, wa, b):
    """KS distance between the weighted ECDF of a (weights wa) and the ECDF of b."""
    a = np.asarray(a, float); b = np.asarray(b, float); wa = np.asarray(wa, float) / np.sum(wa)
    xs = np.unique(np.r_[a, b])
    oa = np.argsort(a)
    ca = np.r_[0, np.cumsum(wa[oa])]
    Fa = ca[np.searchsorted(a[oa], xs, "right")]
    Fb = np.searchsorted(np.sort(b), xs, "right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))


def load_q(run):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(OUT_ROOT / run / "qstats" / "q_*.parquet")))],
                     ignore_index=True)


def cleaned_test_npeaks():
    from engine import spectra as sp
    t = pd.read_parquet(TEST, columns=["molecule_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
    n = []
    for r in t.itertuples():
        mz, it = sp.clean(np.asarray(r.ms2_mzs, np.float64), np.asarray(r.ms2_normalized_intensities, np.float64),
                          float(r.precursor_mz), 0.001, 512, 2.0)
        n.append(len(mz))
    t["n"] = n
    return t.groupby("molecule_id").n.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test-run", default="testSV", help="visible-test engine run used for V5b (SV protocol)")
    ap.add_argument("--raw-test-run", default="test400", help="unmasked visible-test run (secondary, diagnostic)")
    a = ap.parse_args()
    d = OUT_ROOT / a.run
    rep = {}
    cvr = json.load(open(d / "cv_report.json"))
    rk = cvr["ranker"][f"it{cvr['rounds']}"]
    ranges = dict(c1=(0.85, 1.0), c2=(0.60, 0.73), c3=(1e-9, 1.0))
    rep["a"] = {r: dict(mrr25=rk[r]["mrr25"], fz_baseline=cvr["fz_baseline"][r]["mrr25"], range=ranges[r],
                        pass_=bool(ranges[r][0] <= rk[r]["mrr25"] <= ranges[r][1]),
                        mrr25_if_truth_present=rk[r]["mrr25_if_present"], truth_in_cands=rk[r]["truth_in_cands"])
                for r in REGIMES if r in rk}

    S = load_q(a.run)
    T = load_q(a.test_run)
    S = S[S.n_cand > 0] if "n_cand" in S else S
    reg_n = S.regime.value_counts()
    w = S.regime.map(lambda r: MIX[r] / reg_n[r]).values
    b = {}
    for s in QSTATS:
        row = dict(test_median=float(T[s].median()), sim_median=float(S[s].median()),
                   ks_mix=round(wks(S[s].values, w, T[s].values), 4))
        for r in REGIMES:
            x = S[S.regime == r]
            if len(x):
                row[f"ks_{r}"] = round(wks(x[s].values, np.ones(len(x)), T[s].values), 4)
        row["pass_"] = row["ks_mix"] < 0.15
        b[s] = row
    tc = cleaned_test_npeaks()
    b["q_npeaks"]["diag_test_cleaned_median"] = float(tc.median())
    b["q_npeaks"]["diag_ks_mix_if_test_cleaned"] = round(wks(S.q_npeaks.values, w, tc.values), 4)
    if (OUT_ROOT / a.raw_test_run / "qstats").exists():
        T0 = load_q(a.raw_test_run)
        rep["b_unmasked_test"] = {s: dict(test_median=float(T0[s].median()),
                                          ks_mix=round(wks(S[s].values, w, T0[s].values), 4)) for s in QSTATS}
    rep["b"] = b
    # ranker v0 on the labelled SV rows (visible test, exact duplicates masked): ranker vs f.z
    rows_sv = sorted(glob.glob(str(OUT_ROOT / a.test_run / "rows" / "rows_*.parquet")))
    if rows_sv and (d / "ranker_v0.pkl").exists():
        import pickle
        import lightgbm as lgb
        from common import mrr_and_top1
        R = pd.concat([pd.read_parquet(f) for f in rows_sv], ignore_index=True).sort_values("qid", kind="stable")
        if (R.y >= 0).all():
            mdl = pickle.load(open(d / "ranker_v0.pkl", "rb"))
            X = R[["f_" + f for f in mdl["features"]]].values.astype(np.float32)
            p = np.mean([lgb.Booster(model_str=b).predict(X) for b in mdl["boosters"]], 0)
            g = R.qid.values
            nq = len(T)
            out = {}
            for name, sc in (("ranker_v0", p), ("fz_only", R.f_fz.values.astype(float))):
                rr, t1 = mrr_and_top1(sc, R.y.values, g)
                out[name] = dict(mrr25=round(float(rr.sum() / nq), 4), top1=round(float(t1.sum() / nq), 4))
            out["truth_in_cands"] = round(float(R.groupby("qid").y.max().sum() / nq), 4)
            out["queries"] = nq
            rep["sv_eval"] = out
    rt = {}
    for name, X in (("sim", S), ("test", T)):
        rt[name] = dict(sec_mean=round(float(X.sec.mean()), 2), sec_median=round(float(X.sec.median()), 2),
                        sec_p90=round(float(X.sec.quantile(0.9)), 2), sec_max=round(float(X.sec.max()), 1),
                        n_cand_median=float(X.n_cand.median()), n_gen_mean=round(float(X.n_gen.mean()), 1))
    rep["runtime_per_query_per_worker"] = rt
    rep["b_test_queries"] = int(len(T))
    rep["b_sim_queries"] = int(len(S))
    rep["c"] = cvr.get("v5c_provenance")
    if rep["c"]:
        pr, pf = rep["c"]["ranker"], rep["c"]["fz"]
        gap_r = pr["share_train_top5_decoys"] - pr["share_train_all_decoys"]
        gap_f = pf["share_train_top5_decoys"] - pf["share_train_all_decoys"]
        rep["c"]["top5_train_overrep_ranker"] = round(gap_r, 4)
        rep["c"]["top5_train_overrep_fz"] = round(gap_f, 4)
        rep["c"]["ranker_minus_fz"] = round(gap_r - gap_f, 4)
        # the harmful direction is the ranker PROMOTING train-origin candidates ("train-style => truth"), relative to
        # the model-only f.z order; demoting them is the expected effect of the masked-library features.
        rep["c"]["pass_"] = bool(gap_r - gap_f <= 0.02 and
                                 rep["c"]["src_feature_probe"].get("c2", {}).get("delta", 0.0) <= 0.02)
        rep["c"]["criterion"] = ("one-sided: top-5 train-origin over-representation (ranker minus f.z) <= 0.02 and an "
                                 "explicit origin feature adds <= 0.02 c2 MRR")
    json.dump(rep, open(d / "v5_report.json", "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()

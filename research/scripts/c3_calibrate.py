"""Fit the hidden-test class mix from our leaderboard scores and bench profiles.

    python research/scripts/c3_calibrate.py

Model: LB_i = sum_k w_k * score_i(bucket k), w_k >= 0, sum w_k <= 1 (the rest scores 0). Buckets:
  V   visible-like exact library hit       (results/c3/visible_eval.json)
  S1  library hit from another library     (bench S1, honest ranker = clean_kf)
  S2  COCONUT structure, no spectra        (bench S2, clean_kf)
  PC  PubChem-only structure                (bench S3, truth in PubChem, not COCONUT)
  C3  in neither database (real Class 3)    (bench S3, truth in neither)
Profiles of E3 / E3b / E4 are E1 plus the paired bench deltas of their reports (S1, S2, visible); their PC and C3
are E1's (the re-orderings act on the engine's own list). Every number is a public-LB reading on ~130 molecules
(noise about +-0.016), so the fit is a rough estimate. Writes results/c3/calibration.json.
"""
import itertools, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
C3 = ROOT / "results" / "c3"
sp = json.load(open(C3 / "split_e1.json"))
vis = json.load(open(C3 / "visible_eval.json"))
SV_MODE = "--sv" in __import__("sys").argv   # v2: library-hit bucket = scenario SV (re-measured), E5 added (DEC-008)
if SV_MODE:
    sv = json.load(open(C3 / "sv_gate.json"))
    vis = {"E1": {"mrr": sv["E1"]}, "V2": {"mrr": sv["V2"]}, "E2": {"mrr": sv["fused"]["t9.0_b0.4"]}}
e5g = json.load(open(C3 / "e5_gate.json"))["best"]
g = lambda scen, cls, s: sp[scen][cls][s][0]

E1 = dict(V=vis["E1"]["mrr"], S1=g("S1", "coconut", "clean_kf"), S2=g("S2", "coconut", "clean_kf"),
          PC=g("S3", "pubchem_only", "blend"), C3=g("S3", "neither", "blend"))
SUBS = {
    "E1": (0.353, E1),
    "V2": (0.251, dict(V=vis["V2"]["mrr"], S1=g("S1", "coconut", "ours"), S2=g("S2", "coconut", "ours"),
                       PC=g("S3", "pubchem_only", "ours"), C3=g("S3", "neither", "ours"))),
    "E2": (0.350, dict(V=vis["E2"]["mrr"], S1=g("S1", "coconut", "e2_honest"), S2=g("S2", "coconut", "e2_honest"),
                       PC=g("S3", "pubchem_only", "e2_fusion"), C3=g("S3", "neither", "e2_fusion"))),
    # deltas: e3_diagnosis.md (E3 lam 1/1: S1 -0.057, S2 +0.075, visible -0.084; E3b: S1 +0.004, S2 +0.078, visible 0)
    "E3": (0.322, {**E1, "V": E1["V"] - 0.084, "S1": E1["S1"] - 0.057, "S2": E1["S2"] + 0.075}),
    "E3b": (0.355, {**E1, "S1": E1["S1"] + 0.004, "S2": E1["S2"] + 0.078}),
    # e4_popularity.md: mu 0.25 logit vs E3b: S1 +0.011, S2 +0.088 (bench truths' own popularity), visible 0
    "E4": (0.347, {**E1, "S1": E1["S1"] + 0.015, "S2": E1["S2"] + 0.166}),
}
if SV_MODE:  # E5 v2 = E3b + gate (lib < 0.7 -> RRF with V2, BETA 2.0): bench S1/S2/PC/C3 from e5_gate.json, SV measured
    SUBS["E5"] = (0.287, dict(V=sv["fused"]["t0.7_b2.0"], S1=e5g["S1"], S2=e5g["S2"], PC=e5g["PC"], C3=e5g["C3"]))
B = ["V", "S1", "S2", "PC", "C3"]
names = list(SUBS)
y = np.array([SUBS[n][0] for n in names])
X = np.array([[SUBS[n][1][b] for b in B] for n in names])


def fit(X, y):
    k = X.shape[1]
    cons = [{"type": "ineq", "fun": lambda w: 1 - w.sum()}]
    r = minimize(lambda w: ((X @ w - y) ** 2).sum(), np.full(k, 0.1), bounds=[(0, 1)] * k, constraints=cons,
                 method="SLSQP")
    return r.x


w = fit(X, y)
res = {"buckets": B, "profiles": {n: dict(zip(B, map(float, X[i]))) for i, n in enumerate(names)},
       "lb": dict(zip(names, map(float, y))), "weights_all": dict(zip(B, np.round(w, 3).tolist())),
       "remainder_scoring_0": round(float(1 - w.sum()), 3),
       "pred": {n: round(float(X[i] @ w), 4) for i, n in enumerate(names)}}
loo = {}
for i, n in enumerate(names):  # leave one submission out
    m = np.arange(len(names)) != i
    wi = fit(X[m], y[m])
    loo[n] = {"pred": round(float(X[i] @ wi), 4), "obs": float(y[i]), "w": dict(zip(B, np.round(wi, 3).tolist()))}
res["leave_one_out"] = loo
res["loo_rmse"] = round(float(np.sqrt(np.mean([(v["pred"] - v["obs"]) ** 2 for v in loo.values()]))), 4)
# without E4 (its S2 delta depends on the truths' popularity and may be -0.057 instead of +0.088)
m = np.array([n != "E4" for n in names])
res["weights_without_E4"] = dict(zip(B, np.round(fit(X[m], y[m]), 3).tolist()))
json.dump(res, open(C3 / ("calibration_sv.json" if SV_MODE else "calibration.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("weights_all", "remainder_scoring_0", "pred", "lb", "loo_rmse",
                                      "weights_without_E4")}, indent=1))
for n, v in loo.items():
    print(f"LOO {n:4} obs {v['obs']:.3f} pred {v['pred']:.3f}  w {v['w']}")

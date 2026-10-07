"""(c) Learned combiner for the same-formula re-ordering, strict molecule-level held-out evaluation.

    python research/bench/fm/fm_real_learned.py     -> results/bench/fm/real_learned.txt / real_learned.json

Rows: candidates of the formula groups (size >= 2, at least one forward model covers the group) of the honest E1 lists
(clean_kf order, raw ranker outputs pv_kf / ours_kf from the query-held-out refit) of S1 and S2.
Split: random halves of the 250 MOLECULES (S1 and S2 rows of a molecule always fall on the same side), 20 splits, both
directions; the model is trained on one half and re-orders the groups of the other half.  Every molecule is therefore
scored 20 times by models that never saw it in either scenario.
Models: logistic regression (standardised features) and a small LightGBM LambdaRank.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_lib as F  # noqa: E402
import fm_real as R  # noqa: E402

SC = ("S1", "S2")
FEATS = ["lpv", "lours", "L", "L_minus_max", "base_rank_in_group", "z_ice", "z_gl", "ice", "gl", "ice_minus_max", "gl_minus_max",
         "lib", "lib_max_group", "log_size", "margin_L", "L_z_list"]
SETS = {
    "ranker+forward (4)": ["L", "L_minus_max", "z_ice", "z_gl"],
    "ranker+forward+lib+size (8)": ["L", "L_minus_max", "z_ice", "z_gl", "lib", "lib_max_group", "log_size", "margin_L"],
    "all (16)": FEATS,
}


def rows(D):
    """-> list of groups: dict(scen, mi (molecule index), mem, X, y)."""
    G = []
    mids = sorted(D["S1"])
    for s in SC:
        for mi, m in enumerate(mids):
            l = D[s][m]["lists"].get("clean_kf")
            if l is None or not l["smiles"] or "ice" not in l:
                continue
            d = R.prep(l)
            ok = np.array(l["ok"][:d["top"]], bool)
            for mem, zi, ci, zg, cg in R.groups_of(d, None):
                if not (ci or cg):
                    continue
                ice = np.nan_to_num(d["ice"][mem], nan=float(np.nanmean(d["ice"][mem])) if np.isfinite(d["ice"][mem]).any() else 0.0)
                gl = np.nan_to_num(d["gl"][mem], nan=float(np.nanmean(d["gl"][mem])) if np.isfinite(d["gl"][mem]).any() else 0.0)
                L = d["L"][mem]
                sL = np.sort(L)[::-1]
                X = np.column_stack([d["lpv"][mem], d["lours"][mem], L, L - L.max(), np.arange(len(mem)) / len(mem), zi, zg, ice, gl,
                                     ice - ice.max(), gl - gl.max(), d["lib"][mem], np.full(len(mem), d["lib"][mem].max()),
                                     np.full(len(mem), np.log(len(mem))), np.full(len(mem), sL[0] - sL[1]), d["Lz_list"][mem]])
                G.append(dict(scen=s, mi=mi, mem=mem, X=X, y=ok[mem].astype(int), n=d["n"], ok=l["ok"]))
    return G, mids


def fit_predict(kind, cols, tr, te):
    Xtr = np.vstack([g["X"][:, cols] for g in tr]); ytr = np.concatenate([g["y"] for g in tr])
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        return [m.decision_function((g["X"][:, cols] - mu) / sd) for g in te], m.coef_[0] / sd
    import lightgbm as lgb
    keep = [g for g in tr if g["y"].any()]                  # groups without the truth carry no ranking information
    Xk = np.vstack([g["X"][:, cols] for g in keep]); yk = np.concatenate([g["y"] for g in keep])
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=120, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=3, random_state=0)
    m.fit(Xk, yk, group=[len(g["y"]) for g in keep])
    return [m.predict(g["X"][:, cols]) for g in te], m.feature_importances_


def main():
    warnings.filterwarnings("ignore")
    D, _ = F.load()
    G, mids = rows(D)
    n = len(mids)
    base = {s: np.array([F.rr_of(D[s][m]["lists"]["clean_kf"]["ok"]) if D[s][m]["lists"].get("clean_kf") else 0.0 for m in mids]) for s in SC}
    e3b = {s: R.deltas(D, s, dict(term="rank", lam=0.5, protect=0.6)) for s in SC}
    lines = []

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    pr(f"groups {len(G)} (with the truth: {sum(int(g['y'].any()) for g in G)}), rows {sum(len(g['y']) for g in G)}")
    pr("| model | features | S1 vs E1 | S2 vs E1 | mean vs E1 | S1 vs E3b | S2 vs E3b | mean vs E3b |\n|---|---|---|---|---|---|---|---|")
    out = {}
    reps = 20
    for kind in ("logreg", "lgbm"):
        for sn, names in SETS.items():
            cols = [FEATS.index(c) for c in names]
            held = {s: np.zeros(n) for s in SC}
            coefs = []
            rng = np.random.default_rng(7)
            for _ in range(reps):
                p = rng.permutation(n)
                side = np.zeros(n, int); side[p[n // 2:]] = 1
                for a in (0, 1):
                    tr = [g for g in G if side[g["mi"]] == a]
                    te = [g for g in G if side[g["mi"]] != a]
                    pred, cf = fit_predict(kind, cols, tr, te)
                    coefs.append(cf)
                    perms = {}
                    for g, sc in zip(te, pred):
                        perm = perms.setdefault((g["scen"], g["mi"]), [list(range(g["n"])), g["ok"]])[0]
                        order = sorted(range(len(g["mem"])), key=lambda j: (-round(float(sc[j]), 9), j))
                        for slot, j in zip(g["mem"], order):
                            perm[int(slot)] = int(g["mem"][j])
                    for (s, mi), (perm, ok) in perms.items():
                        held[s][mi] += (F.rr_of(ok, perm) - F.rr_of(ok)) / reps
            c = [F.boot(held["S1"]), F.boot(held["S2"]), F.boot((held["S1"] + held["S2"]) / 2),
                 F.boot(held["S1"] - e3b["S1"]), F.boot(held["S2"] - e3b["S2"]), F.boot((held["S1"] + held["S2"] - e3b["S1"] - e3b["S2"]) / 2)]
            pr(f"| {kind} | {sn} | " + " | ".join(F.fmtd(x) for x in c) + " |")
            cm = np.mean(coefs, 0)
            pr("   " + ("coef per raw unit: " if kind == "logreg" else "split counts: ") + ", ".join(f"{a}={b:.3g}" for a, b in zip(names, cm)))
            out[f"{kind}|{sn}"] = dict(zip(("S1", "S2", "mean", "S1_vs_e3b", "S2_vs_e3b", "mean_vs_e3b"), c), coef=dict(zip(names, map(float, cm))))
    json.dump(out, open(F.FM / "real_learned.json", "w"), indent=1)
    (F.FM / "real_learned.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

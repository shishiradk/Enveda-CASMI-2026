"""Seed-Tc-aware re-ranker, stage 2: grouped 5-fold CV (stratified by nb1) for the hand rule and lambdarank.

    python -u research/c3gen/seedtc_rank.py [--workers 2]

Loads results/c3gen/seedtc/feat_{default,strip_edit1}.parquet (rows are the shipped rr_bt union, <=200).

Methods:
  hand_rule : score = rrs + w(seed_tc) * z_z,  rrs = 1/(rank_rr+10),  w = w0 * max(0, (t1 - seed_tc)/t1).
              Grid over w0 in {0.5, 1, 2, 4} and t1 in {0.5, 0.7, 0.85, 1.05}; every cell is evaluated by the same
              grouped CV. The selected cell maximises nb1-no MRR on the default bench subject to strip-edit1 overall
              >= rr_bt(strip) - 0.01 (= 0.389), ties broken toward larger t1 then smaller w0.
  lmb       : LightGBM lambdarank on 12 candidate-side features (no truth), trained per fold on the DEFAULT-mode rows
              (the deploy-realistic feature space) and applied to both modes' held-out molecules.

Outputs (results/c3gen/seedtc/): cv_summary.json (grid + selected + per-fold metrics), rank_hand_best_{mode}.json and
rank_lmb_{mode}.json ({mid: [smiles]}, fold-pooled, <=200, rr_bt tie order).
"""
import argparse, glob, json, os, sys, time

import numpy as np
import pandas as pd

ROOT = "D:/Enveda-CASMI-2026/"
OUT = ROOT + "results/c3gen/seedtc/"
MODES = ["default", "strip_edit1"]
W0 = [0.5, 1.0, 2.0, 4.0]
T1 = [0.5, 0.7, 0.85, 1.05]
LMB_FEATS = ["rank_rr", "rrs", "rk_a", "rk_b", "rma", "rmb", "gen_a", "gen_b", "z", "z_z", "seed_tc", "ha_delta"]
SEED = 0


def mrr_from_labels(mids, ranked, lab_of):
    """List of reciprocal ranks @25. ranked[m] = candidate smiles best-first; lab_of[(mid, smiles)] = truth flag."""
    rr = []
    for m in mids:
        rk = 0
        for j, s in enumerate(ranked.get(m, []), 1):
            if lab_of.get((m, s), 0):
                rk = j
                break
        rr.append(1.0 / rk if 0 < rk <= 25 else 0.0)
    return np.asarray(rr)


def assign_folds(mids, nb1_of, nfolds=5, seed=SEED):
    """Grouped, nb1-stratified fold assignment (deterministic shuffle within each nb1 stratum)."""
    rng = np.random.default_rng(seed)
    folds = {}
    for v in (False, True):
        ids = sorted(m for m in mids if nb1_of[m] == v)
        per = rng.permutation(len(ids))
        for k, m in enumerate(ids):
            folds[m] = int(per[k] % nfolds)
    return folds


def rank_molecule(rows, col="score"):
    """rows: DataFrame (one molecule) -> top-200 smiles sorted by col desc, rank_rr asc (stable tie order)."""
    out = rows.sort_values([col, "rank_rr"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)["smiles"].tolist()[:200]


def metric_partition(F, test_mids, ranked, lab_of):
    """Overall + nb1 strata + subset + ins/oos label-based MRR on the test molecules (one fold)."""
    Fm = F[F.mid.isin(test_mids)].drop_duplicates("mid")
    rrs = mrr_from_labels(test_mids, ranked, lab_of)
    out = {"all": float(rrs.mean())}
    meta = Fm.set_index("mid")[["nb1", "subset", "ins"]]
    out["nb1_yes"] = float(rrs[[meta.loc[m, "nb1"] for m in test_mids]].mean()) if any(meta.loc[m, "nb1"] for m in test_mids) else np.nan
    out["nb1_no"] = float(rrs[~np.asarray([meta.loc[m, "nb1"] for m in test_mids])].mean())
    out["ins"] = float(rrs[np.asarray([meta.loc[m, "ins"] for m in test_mids])].mean())
    out["oos"] = float(rrs[~np.asarray([meta.loc[m, "ins"] for m in test_mids])].mean())
    for sub in ("npex", "s3pc", "s3none"):
        sel = [m for m in test_mids if meta.loc[m, "subset"] == sub]
        out[sub] = float(mrr_from_labels(sel, ranked, lab_of).mean()) if sel else np.nan
    df = pd.DataFrame(dict(mid=test_mids, rr=rrs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.workers = max(1, min(a.workers, 2))
    t0 = time.time()
    F = {m: pd.read_parquet(OUT + f"feat_{m}.parquet") for m in MODES}
    for m in MODES:
        F[m] = F[m].assign(rrs=1.0 / (F[m].rank_rr + 10),
                           rma=np.where(F[m].rk_a > 0, 1.0 / (F[m].rk_a + 10), 0.0),
                           rmb=np.where(F[m].rk_b > 0, 1.0 / (F[m].rk_b + 10), 0.0),
                           gen_a=F[m].gen_a.astype(np.int8), gen_b=F[m].gen_b.astype(np.int8))
    mids = sorted(F["default"].mid.unique())
    nb1_of = F["default"].drop_duplicates("mid").set_index("mid").nb1.to_dict()
    folds = assign_folds(mids, nb1_of)
    nb1_share = {f_: np.mean([nb1_of[m] for m in mids if folds[m] == f_]) for f_ in range(5)}
    print("folds nb1 shares", {k: round(v, 3) for k, v in nb1_share.items()}, flush=True)

    # --- hand-rule grid (label-based CV, both modes) ---
    grid = {}
    for mode in MODES:
        G = F[mode]
        lab_of = {(r.mid, r.smiles): r.label for r in G.itertuples()}
        grid[mode] = {}
        for w0 in W0:
            for t1 in T1:
                G1 = G.copy()
                G1["score"] = G1.rrs + (w0 * np.maximum(t1 - G1.seed_tc, 0.0) / t1) * G1.z_z
                rr_all, rr_yn, rr_nn = [], [], []
                for f_ in range(5):
                    test = [m for m in mids if folds[m] == f_]
                    ranked = {m: rank_molecule(G1[G1.mid == m]) for m in test}
                    mrr = metric_partition(G, test, ranked, lab_of)
                    rr_all.append(mrr["all"]); rr_yn.append(mrr["nb1_yes"]); rr_nn.append(mrr["nb1_no"])
                grid[mode][(w0, t1)] = dict(all=float(np.mean(rr_all)), nb1_yes=float(np.mean(rr_yn)),
                                            nb1_no=float(np.mean(rr_nn)))
        print(mode, "grid best nb1-no", max(grid[mode].items(), key=lambda kv: kv[1]["nb1_no"]), flush=True)
    # select: maximise default nb1-no, subject to strip all >= 0.389, tie-break t1 then w0
    cells = [(w0, t1) for w0 in W0 for t1 in T1]
    ok = [c for c in cells if grid["strip_edit1"][c]["all"] >= 0.389]
    if not ok:
        print("WARN no cell meets the strip-edit1 overall constraint", flush=True)
    best = max(ok, key=lambda c: (grid["default"][c]["nb1_no"], c[1], -c[0])) if ok else \
        max(cells, key=lambda c: grid["default"][c]["nb1_no"])
    print("selected hand cell", best, flush=True)
    # fold-pooled ranked lists for the selected cell (both modes)
    ranked_out = {mode: {} for mode in MODES}
    for mode in MODES:
        G = F[mode].copy()
        w0, t1 = best
        G["score"] = G.rrs + (w0 * np.maximum(t1 - G.seed_tc, 0.0) / t1) * G.z_z
        for m in mids:
            ranked_out[mode][m] = rank_molecule(G[G.mid == m])
        json.dump(ranked_out[mode], open(OUT + f"rank_hand_best_{mode}.json", "w"))

    # --- lambdarank (per-fold, trained on default-mode rows with a label, applied to both modes) ---
    try:
        import lightgbm as lgb
    except Exception as e:
        print("lightgbm unavailable:", e, flush=True); return
    ranked_lmb = {mode: {} for mode in MODES}
    lmb_metric = {mode: [] for mode in MODES}
    for f_ in range(5):
        tr = [m for m in mids if folds[m] != f_]
        te = [m for m in mids if folds[m] == f_]
        Tr = F["default"][F["default"].mid.isin(tr)]
        grp = Tr.groupby("mid", sort=True).size()
        keep = grp[grp >= 2].index
        Tr = Tr[Tr.mid.isin(keep)]
        Tr = Tr[Tr.groupby("mid")["label"].transform("sum") > 0]     # only queries with a positive
        g = Tr.groupby("mid", sort=True).size().values
        X = Tr[LMB_FEATS].values
        y = Tr.label.values.astype(np.int32)
        mdl = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", ndcg_eval_at=[25],
                             n_estimators=150, learning_rate=0.05, num_leaves=15,
                             subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
                             reg_lambda=1.0, random_state=SEED, n_jobs=a.workers, verbosity=-1)
        mdl.fit(X, y, group=g)
        for mode in MODES:
            G = F[mode][F[mode].mid.isin(te)].copy()
            G["score"] = mdl.predict(G[LMB_FEATS].values)
            for m in te:
                ranked_lmb[mode][m] = rank_molecule(G[G.mid == m])
            lab_of = {(r.mid, r.smiles): r.label for r in F[mode].itertuples()}
            lmb_metric[mode].append(metric_partition(F[mode], te, ranked_lmb[mode], lab_of))
        print(f"fold {f_} lmb done {time.time() - t0:.0f}s", flush=True)
    for mode in MODES:
        json.dump(ranked_lmb[mode], open(OUT + f"rank_lmb_{mode}.json", "w"))
        agg = {k: float(np.mean([d[k] for d in lmb_metric[mode]])) for k in
               ("all", "nb1_yes", "nb1_no", "ins", "oos", "npex", "s3pc", "s3none")}
        grid[mode]["lmb"] = agg
        print(mode, "lmb", agg, flush=True)
    grid_j = {"_meta": dict(selected_hand=list(best), folds=nb1_share,
                            rule="rrs + w0*max(0,(t1-seed_tc)/t1)*z_z", rrs="1/(rank_rr+10)",
                            constraint_strip_all_ge="0.389 (rr_bt strip 0.3993 - 0.01)",
                            lmb_feats=LMB_FEATS, boot_seed=SEED)}
    for mode in MODES:
        grid_j[mode] = {f"{w0}__{t1}": grid[mode][(w0, t1)] for w0 in W0 for t1 in T1}
        grid_j[mode]["lmb"] = grid[mode]["lmb"]
    for mode in MODES:
        print("---", mode, "hand grid (all / nb1-no) ---", flush=True)
        for w0 in W0:
            print("  " + " ".join(f"{t1}:{grid[mode][(w0, t1)]['all']:.4f}/{grid[mode][(w0, t1)]['nb1_no']:.4f}"
                                 for t1 in T1), flush=True)
    json.dump(grid_j, open(OUT + "cv_summary.json", "w"), indent=1, default=float)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
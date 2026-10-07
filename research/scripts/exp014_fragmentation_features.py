"""EXP-014: Fragmentation-aware candidate scoring (design: research/analysis/exp014_fragmentation_features_design.md).

Completely independent of EXP-013 (no error taxonomy) and of the V0/v1b pipeline. For each (query spectrum,
candidate molecule) pair we build a feature vector (mass/adduct + spectral evidence when the candidate has
reference MS/MS in the training library), train simple classifiers, and rank candidate pools head-to-head against
the EXP-011/012 spectrum-kNN (B3) baseline on identical pools.

Stages:  build feats train knn_val eval report
    python research/scripts/exp014_fragmentation_features.py build|feats|train|knn_val|eval|report
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402
import exp012_coconut_pools as C12  # noqa: E402

NEG = E.NEG
AD10 = E.AD10
PRIMARY = 5
TOL = 0.1
N_REF = 3
TOPK_Q = 100
TOPK_NL = 20
NL_MAX = 500.0
BIN = 1.0
MZ_MAX = 1500.0
MAX_POOL_NEG = 32
SEED = 20260925

MASS_FEATS = ["mass_diff_ppm", "mass_diff_mDa", "adduct_err_ppm", "best_adduct_err_ppm", "adduct_is_best"]
PEAK_FEATS = ["matched_peaks", "frag_overlap_frac", "top10_overlap_frac", "frag_cos_binned"]
INT_FEATS = ["int_weighted_frac", "int_total_ratio", "top5_frac_q", "top5_frac_ref", "entropy_q", "entropy_ref",
             "peak_ratio", "n_peaks_ref"]
NL_FEATS = ["nl_overlap_frac", "nl_int_frac"]
MODCOS_FEATS = ["mod_cos"]
SPECTRAL_FEATS = PEAK_FEATS + INT_FEATS + NL_FEATS + MODCOS_FEATS
GLOBAL_FEATS = ["has_spectra", "n_ref_spectra"]
ALL_FEATS = MASS_FEATS + SPECTRAL_FEATS + GLOBAL_FEATS
ABLATIONS = {
    "mass": MASS_FEATS,
    "mass+peak": MASS_FEATS + PEAK_FEATS,
    "mass+peak+int": MASS_FEATS + PEAK_FEATS + INT_FEATS,
    "mass+peak+int+nl": MASS_FEATS + PEAK_FEATS + INT_FEATS + NL_FEATS,
    "mass+peak+int+nl+mc": MASS_FEATS + PEAK_FEATS + INT_FEATS + NL_FEATS + MODCOS_FEATS,
    "all_features": MASS_FEATS + PEAK_FEATS + INT_FEATS + NL_FEATS + MODCOS_FEATS + GLOBAL_FEATS,
}
MAX_POOL_NEG_DISPLAY = MAX_POOL_NEG


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


SMOKE = False


def O(dir_=None):
    sub = "exp014_fragfeats_smoke" if SMOKE else "exp014_fragfeats"
    return (Path(dir_) if dir_ else ROOT / "results" / sub)


E11 = ROOT / "results" / "exp011_class2_proxy"
O12 = ROOT / "results" / "exp012_coconut"


# ----------------------------------------------------------------------------- build
def build_ref_index(c, split_of, subset_iks, splits):
    """per ik -> reference spectra (mz, intensity, precursor) capped at N_REF, from root train.parquet.
    split_of: {rid: 'train'|'val'|None}. splits: which split values are eligible as reference evidence."""
    tmp = pd.DataFrame({"ik": sorted(subset_iks)})
    c.register("wantik", tmp)
    meta = c.execute("""
        SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm,
               len(ms2_mzs) AS npeaks
        FROM read_parquet('%(t)s', file_row_number=true) WHERE inchikey14 IN (SELECT ik FROM wantik)""" % {"t": E.TRAIN}).df()
    c.unregister("wantik")
    meta["split"] = meta["rid"].map(split_of).fillna("train")
    meta = meta[meta["adduct"].isin(AD10) & meta["split"].isin(splits)]
    # pick the N_REF spectra with the most peaks per molecule
    meta = meta.sort_values(["ik", "npeaks", "pm"], ascending=[True, False, False]).groupby("ik").head(N_REF)
    rids = meta["rid"].tolist()
    c.register("wantrid", pd.DataFrame({"rid": rids}))
    spec = c.execute("""
        SELECT file_row_number AS rid, ms2_mzs, ms2_normalized_intensities, precursor_mz AS pm
        FROM read_parquet('%(t)s', file_row_number=true)
        WHERE file_row_number IN (SELECT rid FROM wantrid)""" % {"t": E.TRAIN}).df()
    c.unregister("wantrid")
    spec_by_rid = {r.rid: (np.sort(np.asarray(r.ms2_mzs, dtype=float)),
                           np.asarray(r.ms2_normalized_intensities, dtype=float), float(r.pm))
                   for r in spec.itertuples()}
    out = {}
    for ik, sub in meta.groupby("ik"):
        arr = [spec_by_rid[r.rid] for r in sub.itertuples() if r.rid in spec_by_rid]
        if arr:
            out[ik] = arr
    return out


def rid_spectra(c, rids):
    c.register("wantrid", pd.DataFrame({"rid": list(rids)}))
    spec = c.execute("""
        SELECT file_row_number AS rid, ms2_mzs, ms2_normalized_intensities, precursor_mz AS pm
        FROM read_parquet('%(t)s', file_row_number=true)
        WHERE file_row_number IN (SELECT rid FROM wantrid)""" % {"t": E.TRAIN}).df()
    c.unregister("wantrid")
    return {r.rid: (np.sort(np.asarray(r.ms2_mzs, dtype=float)),
                    np.asarray(r.ms2_normalized_intensities, dtype=float), float(r.pm))
            for r in spec.itertuples()}


def stage_build(a):
    t0 = time.time()
    d = O()
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    c = E.con_()
    U, C = C12.universes()
    UCT = U["COCONUT+train"].sort_values("mass").reset_index(drop=True)
    UC = U["COCONUT"].sort_values("mass").reset_index(drop=True)
    UCT.to_parquet(d / "uct.parquet", index=False)
    UC.to_parquet(d / "uc.parquet", index=False)
    tr = pd.read_parquet(E11 / "training.parquet")
    split_of = dict(zip(tr["rid"], tr["split"]))
    cm = tr[tr["adduct"].isin(AD10)].copy()
    cm["M"] = cm["pm"] - cm["adduct"].map(AD10)
    unass = UCT["mass"].values
    unass_keys = UCT["ik"].values
    aliases = json.load(open(O12 / "target_aliases.json"))

    def pool_window(Uf, M):
        m = Uf["mass"].values
        lo, hi = C12.window(m, M, PRIMARY)
        return Uf["ik"].values[lo:hi].tolist(), m[lo:hi]

    def sample_queries(split_name, n, pool_of):
        elig = cm[cm["split"] == split_name]
        counts = elig.groupby("ik").size()
        elig_iks = counts[counts >= 2].index.values
        chosen = rng.choice(elig_iks, n, replace=False)
        out = []
        for ik in chosen:
            sub = elig[elig["ik"] == ik].sort_values("M")
            q = sub.iloc[0]
            refs = sub["rid"].values[1:N_REF + 1]
            pool, _ = pool_window(UCT, q["M"])
            if ik not in pool:
                pool.append(ik)
            out.append({"rid": int(q["rid"]), "ik": ik, "adduct": q["adduct"], "pm": float(q["pm"]),
                        "M": float(q["M"]), "ref_rids": refs.tolist(), "pop": split_name, "pool": pool,
                        "pool_name": pool_of})
        return out

    n_tr = 250 if a.smoke else 2000
    n_va = 60 if a.smoke else 400
    q_train = sample_queries("train", n_tr, "COCONUT+train")
    q_val = sample_queries("val", n_va, "COCONUT+train")
    pd.DataFrame(q_train).to_parquet(d / "queries_train.parquet", index=False)
    pd.DataFrame(q_val).to_parquet(d / "queries_val.parquet", index=False)

    qq = pd.read_parquet(E11 / "queries.parquet")
    tt = pd.read_parquet(E11 / "targets.parquet")
    tpop = dict(zip(tt["ik"], tt["pop"]))
    qq = qq[qq["adduct"].isin(AD10)].copy()
    qq["M"] = qq["pm"] - qq["adduct"].map(AD10)
    rows = []
    for r in qq.itertuples():
        a = aliases.get(r.ik, {"ik": []})
        ok = set(a["ik"]) | set(a.get("parent", [])) | set(a.get("tautomer", [])) | {r.ik}
        for uname, Uf in (("COCONUT", UC), ("COCONUT+train", UCT)):
            pool, _ = pool_window(Uf, r.M)
            truth = [x for x in pool if x in ok]
            rows.append({"rid": int(r.rid), "ik": r.ik, "pop": tpop[r.ik], "adduct": r.adduct, "pm": float(r.pm),
                         "M": float(r.M), "pool": pool, "truth": truth, "pool_name": uname})
    qt = pd.DataFrame(rows)
    qt.to_parquet(d / "queries_target.parquet", index=False)

    # reference evidence indexes (only for iks that appear in some pool)
    pool_iks = set()
    for dfn in ("queries_train.parquet", "queries_val.parquet", "queries_target.parquet"):
        for iks in pd.read_parquet(d / dfn)["pool"]:
            pool_iks |= set(iks)
    log(f"reference fetch for {len(pool_iks)} pool iks ...")
    ref_train = build_ref_index(c, split_of, pool_iks, {"train"})
    ref_lib = build_ref_index(c, split_of, pool_iks, {"train", "val"})
    with open(d / "ref_train.pkl", "wb") as f:
        pickle.dump(ref_train, f, protocol=4)
    with open(d / "ref_lib.pkl", "wb") as f:
        pickle.dump(ref_lib, f, protocol=4)
    # query spectrum cache: every rid in the three query tables (+ their ref rids)
    need_rids = set()
    for dfn in ("queries_train.parquet", "queries_val.parquet", "queries_target.parquet"):
        dfq = pd.read_parquet(d / dfn)
        need_rids |= set(dfq["rid"])
        if "ref_rids" in dfq:
            for rr in dfq["ref_rids"]:
                need_rids |= set(rr)
    rid_spec = rid_spectra(c, need_rids)
    with open(d / "rid_spec.pkl", "wb") as f:
        pickle.dump(rid_spec, f, protocol=4)
    c.close()
    json.dump({"n_train": n_tr, "n_val": n_va, "n_target_spectra": int(len(qt)),
               "target_spectra_covered": int(qt["truth"].apply(len).sum() > 0),
               "pool_iks_with_refs_train": len(ref_train), "pool_iks_with_refs_lib": len(ref_lib),
               "rid_spec": len(rid_spec), "sec": round(time.time() - t0, 1)},
              open(d / "build.json", "w"), indent=1)
    log(f"build done in {time.time() - t0:.0f}s: refs train {len(ref_train)}, lib {len(ref_lib)}, rids {len(rid_spec)}")


# ----------------------------------------------------------------------------- features
_W = {}


def _init_worker(d):
    from matchms.similarity import spectrum_similarity_functions as ssf
    _W["ssf"] = ssf
    with open(d / "ref_train.pkl", "rb") as f:
        _W["ref_train"] = pickle.load(f)
    with open(d / "ref_lib.pkl", "rb") as f:
        _W["ref_lib"] = pickle.load(f)
    with open(d / "rid_spec.pkl", "rb") as f:
        _W["rid_spec"] = pickle.load(f)


def _binned(mz, it):
    h = np.zeros(int(MZ_MAX / BIN), dtype=np.float64)
    i = (mz / BIN).astype(np.int64)
    m = (i >= 0) & (i < len(h))
    np.add.at(h, i[m], np.sqrt(it[m]))
    return h


def _refs_for(ik, qname, q):
    if ik == q["ik"]:
        rid2 = _W["rid_spec"]
        return [rid2[r] for r in q.get("ref_rids", []) if r in rid2]
    return _W["ref_train" if qname == "train" else "ref_lib"].get(ik, [])


def _pair_features(qname, q, ik, Mc):
    """Feature vector for (query spectrum q, candidate ik). Order = ALL_FEATS (mass, spectral, global).
    Spectral block is all-zero when the candidate has no reference MS/MS in the library."""
    qm, qi, qpm = _W["rid_spec"][q["rid"]]
    qi = np.maximum(qi, 0.0)
    s = qi.sum()
    if s <= 0:
        qi = np.ones_like(qi)
    Mq = q["M"]
    ad = AD10[q["adduct"]]
    ad_err = abs(q["pm"] - (Mc + ad)) / (Mc + ad) * 1e6
    best_err = min(abs(q["pm"] - (Mc + x)) / (Mc + x) * 1e6 for x in AD10.values())
    best_ad = min(AD10, key=lambda k: abs(q["pm"] - (Mc + AD10[k])))
    mass_feats = [1e6 * (Mq - Mc) / Mc, 1000.0 * (Mq - Mc), ad_err, best_err, int(best_ad == q["adduct"])]
    refs = _refs_for(ik, qname, q)
    if refs:
        # peak overlap of query top-100 across all ref peaks (within TOL)
        allmz = np.sort(np.concatenate([r[0] for r in refs]))
        qtop_idx = np.argsort(-qi)[:TOPK_Q]
        qtop = qm[qtop_idx]
        ov = np.zeros(len(qtop), dtype=bool)
        for j, t in enumerate(qtop):
            lo = np.searchsorted(allmz, t - TOL)
            hi = np.searchsorted(allmz, t + TOL, side="right")
            ov[j] = hi - lo > 0
        match_qint = (qi[qtop_idx][ov]).sum() / qi.sum()
        top5q = qi[np.argsort(-qi)[:5]].sum() / qi.sum()
        top5r = max(r[1][np.argsort(-r[1])[:5]].sum() / max(r[1].sum(), 1e-12) for r in refs)
        p = qi / qi.sum()
        ent_q = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        ent_r = max(float(-(p * np.log(p + 1e-12)).sum()) for r in refs for p in [(r[1] / r[1].sum())])
        npr = max(len(r[1]) for r in refs)
        # neutral losses (query top-20 |pm - mz|)
        qnl = (qpm - qm[np.argsort(-qi)[:TOPK_NL]])
        qnl = qnl[(qnl > 0.5) & (qnl < NL_MAX)]
        nl_union = np.concatenate([(r[2] - r[0])[r[0] < r[2]] for r in refs]) if refs else np.array([])
        nlu = np.sort(nl_union[(nl_union > 0.5) & (nl_union < NL_MAX)])
        n_m = 0.0
        for l in qnl:
            lo = np.searchsorted(nlu, l - TOL)
            hi = np.searchsorted(nlu, l + TOL, side="right")
            if hi - lo > 0:
                n_m += 1.0
        nl_ov = n_m / max(len(qnl), 1)
        nl_int = float(min(1.0, nl_ov))
        int_total_ratio = np.log2(max(1.0, sum(r[1].sum() for r in refs) / max(qi.sum(), 0.1)))
        # modified cosine vs each ref (matchms exact kernels); best value across refs
        best_sc, best_mt, best_bin = 0.0, 0, 0.0
        ssf = _W["ssf"]
        hq = _binned(qm, qi)
        for (rm, ri, rpm) in refs:
            spec1 = np.column_stack([rm, ri]).astype(np.float64)
            spec2 = np.column_stack([qm, qi]).astype(np.float64)
            shift = rpm - qpm
            if abs(shift) <= TOL:
                pairs = ssf.collect_peak_pairs(spec1, spec2, TOL, 0.0, 0.0, 1.0)
                if pairs is None:
                    continue
                pairs = pairs[np.argsort(pairs[:, 2], kind="mergesort")[::-1], :]
            else:
                z = ssf.collect_peak_pairs(spec1, spec2, TOL, 0.0, 0.0, 1.0)
                s_ = ssf.collect_peak_pairs(spec1, spec2, TOL, shift, 0.0, 1.0)
                pairs = np.concatenate([np.zeros((0, 3)) if z is None else z,
                                        np.zeros((0, 3)) if s_ is None else s_], axis=0)
                if pairs.shape[0] == 0:
                    continue
                pairs = pairs[np.argsort(pairs[:, 2], kind="mergesort")[::-1], :]
            sc, mt = ssf.score_best_matches(pairs, spec1, spec2, 0.0, 1.0)
            if sc > best_sc:
                best_sc, best_mt = float(sc), int(mt)
            hr = _binned(rm, ri)
            c = float(hq @ hr / (np.linalg.norm(hq) * np.linalg.norm(hr) + 1e-12))
            if c > best_bin:
                best_bin = c
        spectral = [float(ov.sum()), float(ov.mean()), float(ov[:10].mean()), best_bin,
                    float(match_qint), float(int_total_ratio), float(top5q), float(top5r),
                    float(ent_q), float(ent_r), float(npr / max(len(qm), 1)), float(npr),
                    float(nl_ov), float(nl_int), float(best_sc)]
    else:
        spectral = [0.0] * len(SPECTRAL_FEATS)
    global_feats = [1.0 if refs else 0.0, float(len(refs))]
    return mass_feats + spectral + global_feats


def _worker_f(job):
    qname, q, ik, Mc = job
    feats = _pair_features(qname, q, ik, Mc)
    label = 0.0
    if qname in ("train", "val"):
        label = float(ik == q["ik"])
    elif qname == "target":
        label = float(ik in q["truth"])
    return [q["rid"], q["ik"], qname, q.get("pool_name", ""), q.get("pop", ""), ik] + feats + [label]


def stage_feats(a):
    t0 = time.time()
    d = O()
    rows = []
    qc = {"train": pd.read_parquet(d / "queries_train.parquet"),
          "val": pd.read_parquet(d / "queries_val.parquet"),
          "target": pd.read_parquet(d / "queries_target.parquet")}
    uct = pd.read_parquet(d / "uct.parquet")
    uc = pd.read_parquet(d / "uc.parquet")
    mass_uct = dict(zip(uct["ik"], uct["mass"]))
    mass_uc = dict(zip(uc["ik"], uc["mass"]))

    jobs = []  # (qname, q, ik, Mc)
    for qname, df in qc.items():
        for q in df.to_dict("records"):
            for ik in q["pool"]:
                Mc = (mass_uct if q.get("pool_name") != "COCONUT" else mass_uc).get(ik)
                if Mc is None:
                    continue
                jobs.append((qname, q, ik, float(Mc)))
    log(f"{len(jobs)} feature rows")

    with Pool(6, initializer=_init_worker, initargs=(d,)) as pool:
        res = pool.map(_worker_f, jobs, chunksize=512)
    cols = ["rid", "qik", "qname", "pool_name", "pop", "cik"] + ALL_FEATS + ["label"]
    df = pd.DataFrame(res, columns=cols)
    df.to_parquet(d / "pairs.parquet", index=False)
    log(f"pairs {len(df)} rows written")
    json.dump({"rows": int(len(df)), "pos_train": int(((df.qname == 'train') & (df.label == 1)).sum()),
               "sec": round(time.time() - t0, 1)}, open(d / "feats.json", "w"), indent=1)


# ----------------------------------------------------------------------------- train
def stage_train(a):
    t0 = time.time()
    d = O()
    df = pd.read_parquet(d / "pairs.parquet")
    trn = df[df["qname"] == "train"]
    val = df[df["qname"] == "val"]
    from sklearn.metrics import roc_auc_score as ras
    import lightgbm as lgb
    res = {}
    ytr = trn["label"].values.astype(int)
    yva = val["label"].values.astype(int)
    preds = {}
    for nm, cols in ABLATIONS.items():
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                               class_weight="balanced", subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, verbosity=-1)
        t0m = time.time()
        m.fit(trn[cols].values, ytr)
        pv = m.predict_proba(val[cols].values)[:, 1]
        pt = m.predict_proba(trn[cols].values)[:, 1]
        preds[nm] = pv
        res[nm] = {"fit_s": round(time.time() - t0m, 1), "tr_auc": round(float(ras(ytr, pt)), 4),
                   "val_auc": round(float(ras(yva, pv)), 4)}
        log(f"  {nm}: {res[nm]}")
    np.save(d / "val_ablation_preds.npy", np.column_stack([preds[n] for n in ABLATIONS]))
    json.dump(res, open(d / "train_ablation.json", "w"), indent=1)
    # model families on ALL features
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    Xtr, Xva = trn[ALL_FEATS].values, val[ALL_FEATS].values
    sc = StandardScaler().fit(Xtr)
    families = []
    families.append(("LR", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED), sc))
    families.append(("ET", ExtraTreesClassifier(n_estimators=200, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=4), None))
    families.append(("RF", RandomForestClassifier(n_estimators=200, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=4), None))
    families.append(("LGB", lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, class_weight="balanced", subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1), None))
    families.append(("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1), sc))
    try:
        import xgboost as xgb
        pos_w = float((ytr == 0).sum() / max(ytr.sum(), 1))
        families.append(("XGB", xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, scale_pos_weight=pos_w, random_state=SEED, n_jobs=4, eval_metric="auc"), None))
    except Exception as ex:
        log(f"xgboost skip: {ex}")
    res2 = {}
    for nm, m, xsc in families:
        t0m = time.time()
        m.fit(xsc.transform(Xtr) if xsc else Xtr, ytr)
        pv = m.predict_proba(xsc.transform(Xva) if xsc else Xva)[:, 1]
        pt = m.predict_proba(xsc.transform(Xtr) if xsc else Xtr)[:, 1]
        res2[nm] = {"fit_s": round(time.time() - t0m, 1), "tr_auc": round(float(ras(ytr, pt)), 4),
                    "val_auc": round(float(ras(yva, pv)), 4)}
        with open(d / f"model_{nm}.pkl", "wb") as f:
            pickle.dump({"model": m, "scaler": xsc}, f)
        log(f"  {nm}: {res2[nm]}")
    json.dump(res2, open(d / "train_models.json", "w"), indent=1)
    log(f"train done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- knn_val
def stage_knn_val(a):
    """B3-style spectrum-kNN predicted fingerprint for the VAL query set (reuses EXP-011 matrices)."""
    t0 = time.time()
    d = O()
    from scipy import sparse
    val = pd.read_parquet(d / "queries_val.parquet")
    tr = pd.read_parquet(E11 / "training.parquet").set_index("rid")
    Xall = sparse.load_npz(E11 / "X_train.npz")
    rall = np.load(E11 / "rid_train.npy")
    keep = np.isin(rall, tr.loc[tr["split"] == "train"].index.values)
    Xt, rt = Xall[keep], rall[keep]
    t_neg = tr.loc[rt, "adduct"].isin(NEG).values
    F = E.FP(E11)
    t_rows = np.array([F.row[k] for k in tr.loc[rt, "ik"].values])
    XtT = Xt.T.tocsr()
    rloc = {r: i for i, r in enumerate(rall)}
    val_r = [rloc[r] for r in val["rid"].values]
    Xq = Xall[val_r]
    k = 20
    preds = {}
    for i in range(0, Xq.shape[0], 256):
        S = (Xq[i:i + 256] @ XtT).toarray()
        for j in range(S.shape[0]):
            neg = val.iloc[i + j]["adduct"] in NEG
            s = np.where(t_neg == neg, S[j], -1.0)
            nn = np.argpartition(-s, k)[:k]
            w = np.clip(s[nn], 0, None) + 1e-6
            bits = np.unpackbits(F.packed[t_rows[nn]], axis=1).astype(np.float32)
            preds[val.iloc[i + j]["rid"]] = (w[:, None] * bits).sum(0) / w.sum()
        log(f"  kNN-val {min(i + 256, Xq.shape[0])}/{Xq.shape[0]}")
    by_mol = {}
    for r, p in preds.items():
        by_mol.setdefault(val.loc[val["rid"] == r, "ik"].iloc[0], []).append(p)
    by_mol = {k: np.mean(v, 0) for k, v in by_mol.items()}
    with open(d / "knn_val.pkl", "wb") as f:
        pickle.dump(by_mol, f)
    log(f"kNN-val preds for {len(by_mol)} molecules in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- eval
def _cand_bits(iks, F, crow, cfp):
    """Vectorized: (n, FP_BITS) float bits. Missing fingerprint -> zero row."""
    n = len(iks)
    out = np.zeros((n, E.FP_BITS), np.float32)
    crow_r = crow.get
    for i, k in enumerate(iks):
        row = crow_r(k)
        if row is None:
            row = F.row.get(k)
            if row is None:
                continue
            out[i] = np.unpackbits(F.packed[row])
        else:
            out[i] = np.unpackbits(cfp[row])
    return out


def rank_evals(score_dfs, meta):
    """rank pools, tie-aware per EXP-011 statistic."""
    out = {}
    for key, df in score_dfs.items():
        ranks = []
        for q, g in df.groupby("rid"):
            sc = dict(zip(g["cik"], g["score"]))
            truth = meta[q]
            st = E.expected_rank_stats(sc, truth)
            if st:
                ranks.append(st)
        out[key] = pd.DataFrame(ranks) if ranks else None
    return out


def summarize(rd, entries):
    rep = {}
    for key, df in rd.items():
        if df is None or len(df) == 0:
            rep[key] = {"n": 0}
            continue
        rep[key] = {"n": len(df), "mrr25": float(df["rr25"].mean()), "r@1": float(df["r@1"].mean()),
                    "r@10": float(df["r@10"].mean()), "median_rank": float(df["rank"].median())}
    return rep


def stage_eval(a):
    t0 = time.time()
    d = O()
    df = pd.read_parquet(d / "pairs.parquet")
    qt = pd.read_parquet(d / "queries_target.parquet").to_dict("records")
    for q in qt:
        q["truth"] = list(q["truth"]) if q["truth"] is not None else []
    import lightgbm as lgb
    model = pickle.load(open(d / "model_LGB.pkl", "rb"))["model"]
    tcol = df["qname"] == "target"
    Xt_ = df.loc[tcol, ALL_FEATS].values
    pt_ = model.predict_proba(Xt_)[:, 1]
    tgt = df[tcol].copy()
    tgt["score"] = pt_
    uct = pd.read_parquet(d / "uct.parquet")
    uc = pd.read_parquet(d / "uc.parquet")
    aliases = json.load(open(O12 / "target_aliases.json"))
    cdb = pd.read_parquet(O12 / "coconut_db.parquet")
    cdb_ok = cdb[cdb["parse_ok"]].reset_index(drop=True)
    cfp = np.load(O12 / "coconut_fp_packed.npy")[cdb["parse_ok"].values]
    crow = dict(zip(cdb_ok["ik"], np.arange(len(cdb_ok))))
    F = E.FP(E11)
    muct = dict(zip(uct["ik"], uct["mass"]))
    muc = dict(zip(uc["ik"], uc["mass"]))
    pb3z = np.load(E11 / "pred_B3.npz", allow_pickle=True)
    pred_b3 = dict(zip(pb3z["iks"], pb3z["pred"]))
    # ---- molecule-level aggregates: pool = window(median M, 5 ppm), score = mean over that molecule's spectra
    tscore = {}
    for (uname, ik), g in tgt.groupby(["pool_name", "qik"]):
        tscore.setdefault((uname, ik), {}).update(g.groupby("cik")["score"].mean().to_dict())
    groups = {}
    for q in qt:
        groups.setdefault((q["pool_name"], q["ik"]), []).append(q)
    r_rows, b_rows = [], []
    n_g = len(groups)
    log(f"eval targets: {n_g} (universe, molecule) groups")
    for gi, ((uname, ik), qs) in enumerate(groups.items()):
        if (gi % 500) == 0 and gi:
            log(f"  target groups {gi}/{n_g} in {time.time() - t0:.0f}s")
        Uf = uct if uname == "COCONUT+train" else uc
        m = Uf["mass"].values
        M = float(np.median([q["M"] for q in qs]))
        lo, hi = C12.window(m, M, PRIMARY)
        pool = Uf["ik"].values[lo:hi].tolist()
        a = aliases.get(ik, {"ik": []})
        truth_set = set(a["ik"]) | set(a.get("parent", [])) | set(a.get("tautomer", [])) | {ik}
        truth = [c for c in pool if c in truth_set]
        if not truth:
            continue
        pb3 = pred_b3
        st_b3 = None
        if ik in pb3:
            cb = _cand_bits(pool, F, crow, cfp)
            b3 = E.cosine_scores(pb3[ik], cb)
            st_b3 = E.expected_rank_stats(dict(zip(pool, b3)), truth[0])
        mm = muct if uname == "COCONUT+train" else muc
        merr = [-abs(mm[c] - M) for c in pool]
        st_b1 = E.expected_rank_stats(dict(zip(pool, merr)), truth[0])
        b0_ranks = np.arange(1, len(pool) + 1)
        st_b0 = {"rank": float(b0_ranks.mean()), "rr25": float(np.mean(np.where(b0_ranks <= 25, 1 / b0_ranks, 0))),
                 "r@1": float(np.mean(b0_ranks <= 1)), "r@10": float(np.mean(b0_ranks <= 10))}
        # E14: mean pairwise score over the molecule's spectra for candidates present in the pool
        sc = tscore.get((uname, ik), {})
        gr = {c: sc.get(c, 0.0) for c in pool}
        st_e = E.expected_rank_stats(gr, truth[0])
        if st_e:
            r_rows.append({"universe": uname, "pop": qs[0]["pop"], "ik": ik, "n_cand": len(pool),
                           **{f"E14_{k}": v for k, v in st_e.items()}})
        rec = {"universe": uname, "pop": qs[0]["pop"], "ik": ik, "n_cand": len(pool),
               "B0_rank": st_b0["rank"], "B0_rr25": st_b0["rr25"], "B0_r@1": st_b0["r@1"], "B0_r@10": st_b0["r@10"],
               "B1_rank": st_b1["rank"], "B1_rr25": st_b1["rr25"], "B1_r@1": st_b1["r@1"], "B1_r@10": st_b1["r@10"]}
        if st_b3:
            rec.update({"B3_rank": st_b3["rank"], "B3_rr25": st_b3["rr25"], "B3_r@1": st_b3["r@1"], "B3_r@10": st_b3["r@10"]})
        b_rows.append(rec)
    r14 = pd.DataFrame(r_rows)
    r14.to_parquet(d / "eval_target_E14.parquet", index=False)
    bdf = pd.DataFrame(b_rows)
    bdf.to_parquet(d / "eval_target_baselines.parquet", index=False)

    rep = {}
    for (uname, pop), g in r14.groupby(["universe", "pop"]):
        b = bdf[(bdf["universe"] == uname) & (bdf["pop"] == pop)]
        m = g.merge(b, on="ik", suffixes=("_e", "_b"))
        if len(m) == 0:
            continue
        rec = {"n": len(m), "pool_median": float(m["n_cand_b"].median())}
        for tag in ("E14", "B3", "B1"):
            if f"{tag}_rr25" not in m:
                continue
            rec[f"{tag}_rr25"] = round(float(m[f"{tag}_rr25"].mean()), 4)
            rec[f"{tag}_r@1"] = round(float(m[f"{tag}_r@1"].mean()), 4)
            rec[f"{tag}_r@10"] = round(float(m[f"{tag}_r@10"].mean()), 4)
            rec[f"{tag}_median_rank"] = round(float(m[f"{tag}_rank"].median()), 2)
        if "B3_rr25" in rec and "E14_rr25" in rec:
            rec["E14_wins_rr25_vs_B3"] = int((m["E14_rr25"] > m["B3_rr25"]).sum())
            rec["E14_wins_r@1_vs_B3"] = int((m["E14_r@1"] > m["B3_r@1"]).sum())
        Uf = uct if uname == "COCONUT+train" else uc
        mm = dict(zip(Uf["ik"], Uf["mass"]))
        for lo, hi, lab in C12.BINS:
            msub = m[(m["n_cand_b"] >= lo) & (m["n_cand_b"] <= hi)]
            if len(msub):
                rec[f"b{lab}_n"] = len(msub)
                rec[f"b{lab}_E14_rr25"] = round(float(msub["E14_rr25"].mean()), 4)
                rec[f"b{lab}_B3_rr25"] = round(float(msub["B3_rr25"].mean()), 4)
        # hard negatives: >=2 pool candidates within 10 mDa of the molecule's mass (vectorized)
        Uf_all = uct if uname == "COCONUT+train" else uc
        mm2 = dict(zip(Uf_all["ik"], Uf_all["mass"]))
        ms_all = np.sort(Uf_all["mass"].values)
        mv = np.asarray([mm2.get(ik, np.nan) for ik in m["ik"].values], dtype=float)
        near_vect = []
        for M in mv:
            if np.isnan(M):
                near_vect.append(False)
                continue
            lo_, hi_ = np.searchsorted(ms_all, M - 0.01), np.searchsorted(ms_all, M + 0.01, side="right")
            near_vect.append(hi_ - lo_ >= 2)
        hard_ik = m["ik"].values[np.array(near_vect)].tolist()
        if hard_ik:
            h = m[m["ik"].isin(hard_ik)]
            rec["hard_n"] = len(h)
            rec["hard_E14_rr25"] = round(float(h["E14_rr25"].mean()), 4)
            rec["hard_B3_rr25"] = round(float(h["B3_rr25"].mean()), 4)
        rep[f"{uname}|{pop}"] = rec

    # ---- val: E14 ablation ladder + kNN-val + B1 on identical val pools
    vq = pd.read_parquet(d / "queries_val.parquet").to_dict("records")
    valdf = df[df["qname"] == "val"].copy()
    knn = pickle.load(open(d / "knn_val.pkl", "rb"))
    vmass = uct.set_index("ik")["mass"].to_dict()
    ab_preds = np.load(d / "val_ablation_preds.npy")  # rows = val pairs (df order), cols = ab_names
    ab_names = list(ABLATIONS)
    val_rows = []
    for q in vq:
        sub_idx = valdf["rid"].values == q["rid"]
        sub = valdf[sub_idx]
        if sub.empty:
            continue
        ploc = np.where(sub_idx)[0]
        row = {"rid": q["rid"], "ik": q["ik"]}
        for j, nm in enumerate(ab_names):
            scv = dict(zip(sub["cik"].values, ab_preds[ploc, j]))
            st = E.expected_rank_stats(scv, q["ik"])
            if st:
                row[nm + "_rr25"] = st["rr25"]
                row[nm + "_r@1"] = st["r@1"]
        # kNN-val + B1 baselines on the SAME pool
        if q["ik"] in knn:
            cb = _cand_bits(q["pool"], F, crow, cfp)
            b3 = E.cosine_scores(knn[q["ik"]], cb)
            merr = [-abs(vmass.get(c, q["M"]) - q["M"]) for c in q["pool"]]
            st3 = E.expected_rank_stats(dict(zip(q["pool"], b3)), q["ik"])
            st1 = E.expected_rank_stats(dict(zip(q["pool"], merr)), q["ik"])
            if st3:
                row["knn_B3_rr25"] = st3["rr25"]
                row["knn_B3_r@1"] = st3["r@1"]
            if st1:
                row["mass_B1_rr25"] = st1["rr25"]
                row["mass_B1_r@1"] = st1["r@1"]
        val_rows.append(row)
    vr = pd.DataFrame(val_rows)
    vr.to_parquet(d / "eval_val_ablation.parquet", index=False)
    if len(vr):
        vrep = {"n": len(vr)}
        for nm in ab_names:
            if nm + "_rr25" in vr:
                vrep[nm + "_mrr25"] = round(float(vr[nm + "_rr25"].mean()), 4)
                vrep[nm + "_r@1"] = round(float(vr[nm + "_r@1"].mean()), 4)
        for k in ("knn_B3_rr25", "knn_B3_r@1", "mass_B1_rr25", "mass_B1_r@1"):
            if k in vr:
                vrep[k] = round(float(vr[k].mean()), 4)
        rep["VAL"] = vrep
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(d / "eval.json", "w"), indent=1)
    log(json.dumps(rep, indent=1)[:6000])


def stage_report(a):
    """Machine-readable EXP-014 artifacts under research/analysis/exp014/, plus a short findings JSON."""
    t0 = time.time()
    d = O()
    out = ROOT / "research" / "analysis" / "exp014"
    out.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(d / "eval.json"))
    ta = json.load(open(d / "train_ablation.json"))
    tm = json.load(open(d / "train_models.json"))
    build = json.load(open(d / "build.json"))
    # feature importance from the LGB all-features model
    import lightgbm as lgb
    m = pickle.load(open(d / "model_LGB.pkl", "rb"))["model"]
    imp = dict(zip(ALL_FEATS, m.feature_importances_))
    pd.DataFrame({"feature": ALL_FEATS, "importance": [imp[f] for f in ALL_FEATS],
                  "group": [("mass" if f in MASS_FEATS else "spectral" if f in SPECTRAL_FEATS else "global") for f in ALL_FEATS]}
                 ).to_csv(out / "feature_importances.csv", index=False)
    # summary table (same layout as eval.json) as CSV
    rows = []
    for key, v in rep.items():
        if key == "sec" or not isinstance(v, dict):
            continue
        r = {"group": key}
        r.update({k: v[k] for k in sorted(v)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    # per-row eval tables (molecule-level target; val ablation ladder)
    for fn in ("eval_target_E14.parquet", "eval_target_baselines.parquet", "eval_val_ablation.parquet", "eval_val_baselines.parquet"):
        src = d / fn
        if src.exists():
            pd.read_parquet(src).to_csv(out / fn.replace(".parquet", ".csv"), index=False)
    # pairwise per-query metric: merge E14 vs B3 vs B1 for the report
    e14 = pd.read_parquet(d / "eval_target_E14.parquet")
    bb = pd.read_parquet(d / "eval_target_baselines.parquet")
    m = e14.merge(bb, on=["universe", "pop", "ik"], suffixes=("_e14", "_b"))
    m.to_csv(out / "target_pairwise.csv", index=False)
    findings = {"design_q": "does explicit fragmentation-aware feature engineering provide information the current "
                           "spectrum-kNN representation is missing?",
                "smoke": SMOKE, "build": build, "ablation_lgb": ta, "models": tm,
                "eval_summary": rep, "feature_importance": {k: int(v) for k, v in imp.items()},
                "note": "smoke outputs live in results/exp014_fragfeats_smoke/; full-run outputs expected after full pipeline"}
    json.dump(findings, open(out / "findings.json", "w"), indent=1)
    log(f"report artifacts written to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("EXP014_SMOKE", "1" if a.smoke else "0")
    globals()["SMOKE"] = a.smoke
    d = O()
    d.mkdir(parents=True, exist_ok=True)
    {"build": stage_build, "feats": stage_feats, "train": stage_train, "knn_val": stage_knn_val,
     "eval": stage_eval, "report": stage_report}[a.stage](a)
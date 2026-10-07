"""EXP-016: pre-registered confirmation of EXP-014 v2 fragmentation features on a fresh natural-product population.
Design (locked): research/analysis/exp016_fragmentation_confirmation_design.md

    python research/scripts/exp016_fragmentation_confirmation.py build   # population, exclusions, pools, NP-train set, gates
    python research/scripts/exp016_fragmentation_confirmation.py knn     # frozen EXP-011 B3 kNN predictions for the targets
    python research/scripts/exp016_fragmentation_confirmation.py frag    # fragments for new structures (EXP-014 v2 cache reused read-only)
    python research/scripts/exp016_fragmentation_confirmation.py feats   # pair features (targets + FRAG-NP training queries)
    python research/scripts/exp016_fragmentation_confirmation.py trainnp # FRAG-NP (same hyperparameters as EXP-014 v2)
    python research/scripts/exp016_fragmentation_confirmation.py eval
Imports helper FUNCTIONS from exp011/exp014v2/exp015 scripts; never writes into their output folders.
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402
import exp014_fragmentation_features_v2 as F14  # noqa: E402
import exp015_spectrum_representation as S15  # noqa: E402  (helpers only: rd_tools, aliases_in)

E11, E12 = ROOT / "results" / "exp011_class2_proxy", ROOT / "results" / "exp012_coconut"
V2 = ROOT / "results" / "exp014_fragfeats_v2"
R = ROOT / "results" / "exp016"
A = ROOT / "research" / "analysis" / "exp016"
SEED, N_NP_TRAIN, Q_PER_MOL = 20260930, 2000, 4


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def v2_exclusions(C, U8):
    """Exactly the EXP-014 v2 exclusion set X (targets, EXP-011 aliases, D, EXP-012 aliases, target parent groups)."""
    targets = set(pd.read_parquet(E11 / "targets.parquet")["ik"])
    man = json.load(open(E11 / "manifest.json"))
    A12 = json.load(open(E12 / "target_aliases.json"))
    D = set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
    tpar = set(U8.loc[U8["ik"].isin(targets), "parent"]) - {None}
    return (targets | {x for v in man["aliases"].values() for x in v} | D
            | {x for a in A12.values() for k in ("ik", "parent", "tautomer") for x in a[k]}
            | set(U8.loc[U8["parent"].isin(tpar), "ik"]) | set(C.loc[C["parent"].isin(tpar), "ik"]))


# ============================================================================ build
def stage_build():
    t0 = time.time()
    R.mkdir(parents=True, exist_ok=True); A.mkdir(parents=True, exist_ok=True)
    C, U8 = F14.structures()
    Cs = C.sort_values("mass").reset_index(drop=True)
    UCT = pd.concat([C, U8[~U8["ik"].isin(set(C["ik"]))]]).sort_values("mass").reset_index(drop=True)
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='3GB'")
    meta = con.execute(f"""SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm, ingest_lib AS lib
                           FROM read_parquet('{E.TRAIN}', file_row_number=true)""").df()
    m10 = meta[meta["adduct"].isin(E.AD10)]
    tr = pd.read_parquet(E11 / "training.parquet")
    ref = tr[tr["split"] == "train"]
    ref_mols = set(ref["ik"])
    D = set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
    used15 = set(pd.read_parquet(ROOT / "results" / "exp015" / "primary_targets.parquet")["ik"])  # read-only
    prev = set(pd.read_parquet(E11 / "targets.parquet")["ik"]) | used15
    elig = sorted((D & set(C["ik"]) & set(m10["ik"])) - prev)
    X = v2_exclusions(C, U8)

    # FRAG-NP training queries (EXP-011 split=train molecules in COCONUT, >=2 spectra; query = lowest-M spectrum)
    cm = tr[(tr["split"] == "train") & tr["ik"].isin(set(C["ik"]))].copy()
    cm["M"] = cm["pm"] - cm["adduct"].map(E.AD10)
    cnt = cm.groupby("ik").size()
    rng = np.random.default_rng(SEED)
    np_iks = sorted(rng.choice(cnt[cnt >= 2].index.values, N_NP_TRAIN, replace=False).tolist())
    frag_train_mols = (set(pd.read_parquet(ROOT / "results" / "exp014_fragfeats" / "queries_train.parquet")["ik"])
                       | set(pd.read_parquet(ROOT / "results" / "exp014_fragfeats" / "queries_val.parquet")["ik"]) | set(np_iks))

    cache = {"tools": S15.rd_tools()}
    u8 = U8.drop_duplicates("ik").set_index("ik")
    par8 = dict(zip(U8["ik"], U8["parent"]))
    ref_par = {par8.get(k) for k in ref_mols} - {None}
    ftr_par = {par8.get(k) for k in frag_train_mols} - {None}
    q_all = m10[m10["ik"].isin(elig)].sample(frac=1.0, random_state=SEED).groupby("ik").head(Q_PER_MOL)
    con.register("qr", pd.DataFrame({"rid": q_all["rid"].values})); con.register("rr", pd.DataFrame({"rid": ref["rid"].values}))
    hq = con.execute(f"""SELECT file_row_number AS rid, hash(ms2_mzs, ms2_normalized_intensities) AS h FROM read_parquet('{E.TRAIN}', file_row_number=true)
                         WHERE file_row_number IN (SELECT rid FROM qr)""").df()
    hr = set(con.execute(f"""SELECT hash(ms2_mzs, ms2_normalized_intensities) AS h FROM read_parquet('{E.TRAIN}', file_row_number=true)
                             WHERE file_row_number IN (SELECT rid FROM rr)""").df()["h"])
    dup_q = set(hq.loc[hq["h"].isin(hr), "rid"])
    tstat, qrows, target_keys = [], [], set()
    for ik in elig:
        r = u8.loc[ik]
        al8 = S15.aliases_in(U8.sort_values("mass").reset_index(drop=True), ik, r["smiles"], r["mass"], r["formula"], r["parent"], cache) \
            if False else S15.aliases_in(UCT[UCT["ik"].isin(set(U8["ik"]))].reset_index(drop=True), ik, r["smiles"], r["mass"], r["formula"], r["parent"], cache)
        alC = S15.aliases_in(Cs, ik, r["smiles"], r["mass"], r["formula"], r["parent"], cache)
        okC = ({ik} & set(C["ik"])) | alC
        leak = {"alias_in_knn_ref": bool(({ik} | al8) & ref_mols), "parent_in_knn_ref": r["parent"] in ref_par,
                "alias_in_frag_training": bool(({ik} | al8) & frag_train_mols), "parent_in_frag_training": r["parent"] in ftr_par,
                "previous_target": ik in prev}
        qs = q_all[(q_all["ik"] == ik) & ~q_all["rid"].isin(dup_q)]
        status = "ok" if (not any(leak.values()) and len(qs)) else ("no_query" if not len(qs) else "leak:" + ",".join(k for k, v in leak.items() if v))
        M = float(np.median(qs["pm"] - qs["adduct"].map(E.AD10))) if len(qs) else np.nan
        lo, hi = (np.searchsorted(Cs["mass"].values, M * (1 - 5e-6)), np.searchsorted(Cs["mass"].values, M * (1 + 5e-6), side="right")) if len(qs) else (0, 0)
        pool = Cs["ik"].values[lo:hi].tolist()
        truth = [c for c in pool if c in okC]
        tstat.append({"ik": ik, "status": status, "M": M, "pool": pool, "truth": truth, "n_pool": len(pool),
                      "covered": bool(truth), **leak})
        if status == "ok":
            target_keys |= {ik} | al8 | okC
            for q in qs.itertuples():
                qrows.append({"rid": int(q.rid), "ik": ik, "adduct": q.adduct, "pm": float(q.pm), "lib": q.lib})
    T = pd.DataFrame(tstat)
    T.to_pickle(R / "targets.pkl")
    Q = pd.DataFrame(qrows)
    Q.to_parquet(R / "target_queries.parquet", index=False)

    # FRAG-NP training rows (pools exclude v2 X and every EXP-016 target key)
    Xnp = X | target_keys
    nrows = []
    for ik in np_iks:
        sub = cm[cm["ik"] == ik].sort_values("M")
        q = sub.iloc[0]
        m = UCT["mass"].values
        lo, hi = np.searchsorted(m, q["M"] * (1 - 5e-6)), np.searchsorted(m, q["M"] * (1 + 5e-6), side="right")
        pool = [c for c in UCT["ik"].values[lo:hi] if c == ik or c not in Xnp]
        if ik not in pool:
            pool.append(ik)
        nrows.append({"set": "trainNP", "rid": int(q["rid"]), "ik": ik, "adduct": q["adduct"], "pm": float(q["pm"]),
                      "M": float(q["M"]), "pool": pool, "truth": [ik]})
    NPQ = pd.DataFrame(nrows)
    NPQ.to_pickle(R / "np_train_queries.pkl")
    ok = T[T["status"] == "ok"]
    gates = {"accepted_targets_with_leak_flag": int(ok[["alias_in_knn_ref", "parent_in_knn_ref", "alias_in_frag_training",
                                                        "parent_in_frag_training", "previous_target"]].any(axis=1).sum()),
             "np_train_molecules_in_targets": len(set(np_iks) & target_keys),
             "np_train_pool_members_in_exclusions": sum(1 for p, i in zip(NPQ["pool"], NPQ["ik"]) for c in p if c != i and c in Xnp),
             "targets_overlap_exp011_015": len(set(ok["ik"]) & prev)}
    info = {"eligible": len(elig), "status_counts": T["status"].value_counts().to_dict(), "accepted": len(ok),
            "covered": int(ok["covered"].sum()), "query_spectra": len(Q), "query_dup_removed": len(dup_q),
            "query_library_mix": Q["lib"].value_counts().to_dict(), "np_train_queries": len(NPQ),
            "pool_median_covered": float(ok.loc[ok["covered"], "n_pool"].median()), "gates": gates,
            "gates_ok": all(v == 0 for v in gates.values()), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "build.json", "w"), indent=1, default=int)
    log(json.dumps(info, indent=1, default=int))
    assert info["gates_ok"], "leakage gate failed"


# ============================================================================ kNN (frozen EXP-011 B3)
def stage_knn():
    from scipy import sparse
    t0 = time.time()
    Q = pd.read_parquet(R / "target_queries.parquet")
    c = E.con_()
    Xq, rq, _ = E.fetch_features(c, Q["rid"].values) if E.fetch_features.__code__.co_argcount == 2 and False else (None, None, None)
    Xq, rq = _fetch_features_compat(c, Q["rid"].values)
    tr = pd.read_parquet(E11 / "training.parquet").set_index("rid")
    Xall, rall = sparse.load_npz(E11 / "X_train.npz"), np.load(E11 / "rid_train.npy")
    keep = np.isin(rall, tr.index[tr["split"] == "train"].values)
    Xt, rt = Xall[keep], rall[keep]
    del Xall
    t_neg = tr.loc[rt, "adduct"].isin(E.NEG).values
    F = E.FP(E11)
    t_rows = np.array([F.row[k] for k in tr.loc[rt, "ik"].values])
    XtT = Xt.T.tocsr()
    qd = Q.set_index("rid")
    q_neg = qd.loc[rq, "adduct"].isin(E.NEG).values
    preds = []
    for i in range(0, Xq.shape[0], 128):
        S = (Xq[i:i + 128] @ XtT).toarray()
        for j in range(S.shape[0]):
            s = np.where(t_neg == q_neg[i + j], S[j], -1.0)
            nn = np.argpartition(-s, 20)[:20]
            w = np.clip(s[nn], 0, None) + 1e-6
            preds.append((w[:, None] * np.unpackbits(F.packed[t_rows[nn]], axis=1).astype(np.float32)).sum(0) / w.sum())
        log(f"  kNN {min(i + 128, Xq.shape[0])}/{Xq.shape[0]}")
    by = {}
    for ik, p in zip(qd.loc[rq, "ik"].values, preds):
        by.setdefault(ik, []).append(p)
    by = {k: np.mean(v, axis=0) for k, v in by.items()}
    np.savez_compressed(R / "pred_knn.npz", iks=np.array(list(by)), pred=np.stack(list(by.values())))
    log(f"kNN done for {len(by)} molecules in {time.time() - t0:.0f}s")


def _fetch_features_compat(c, rids):
    out = E.fetch_features(c, rids)
    return out[0], out[1]


# ============================================================================ fragments
def stage_frag():
    t0 = time.time()
    C, U8 = F14.structures()
    smi = dict(zip(U8["ik"], U8["smiles"])); smi.update(dict(zip(C["ik"], C["smiles"])))
    T = pd.read_pickle(R / "targets.pkl")
    NPQ = pd.read_pickle(R / "np_train_queries.pkl")
    need = {c for p in T.loc[T["status"] == "ok", "pool"] for c in p} | {c for p in NPQ["pool"] for c in p}
    old = pickle.load(open(V2 / "fragments.pkl", "rb"))  # read-only reuse
    todo = sorted(need - set(old))
    with Pool(6) as pool:
        new = dict(pool.map(F14._frag_job, [(k, smi[k]) for k in todo], chunksize=200))
    fr = {k: old[k] for k in need & set(old)}
    fr.update(new)
    pickle.dump(fr, open(R / "fragments.pkl", "wb"))
    info = {"structures_needed": len(need), "reused_from_exp014v2": len(need & set(old)), "new": len(todo),
            "failed": sum(v is None for v in fr.values()), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "frag.json", "w"), indent=1)
    log(info)


# ============================================================================ features
def stage_feats():
    t0 = time.time()
    C, U8 = F14.structures()
    mass = dict(zip(U8["ik"], U8["mass"])); mass.update(dict(zip(C["ik"], C["mass"])))
    fr = pickle.load(open(R / "fragments.pkl", "rb"))
    T = pd.read_pickle(R / "targets.pkl")
    T = T[(T["status"] == "ok") & T["covered"]].set_index("ik")
    Q = pd.read_parquet(R / "target_queries.parquet")
    Q = Q[Q["ik"].isin(T.index)]
    NPQ = pd.read_pickle(R / "np_train_queries.pkl")
    con = duckdb.connect(); con.execute("SET threads=4")
    con.register("w", pd.DataFrame({"rid": np.r_[Q["rid"].values, NPQ["rid"].values]}))
    sp = con.execute(f"""SELECT file_row_number AS rid, precursor_mz AS pm, ms2_mzs AS mz, ms2_normalized_intensities AS it
                         FROM read_parquet('{E.TRAIN}', file_row_number=true) WHERE file_row_number IN (SELECT rid FROM w)""").df()
    spec = {}
    for r in sp.itertuples():
        mz, it = np.asarray(r.mz, float), np.asarray(r.it, float)
        o = np.argsort(mz)
        spec[int(r.rid)] = (mz[o], it[o], float(r.pm))
    rows = []
    for q in Q.itertuples():
        t = T.loc[q.ik]
        tru = set(t["truth"])
        for c in t["pool"]:
            rows.append(["target", q.rid, q.ik, c] + F14.pair_features(spec[q.rid], q.adduct, q.pm, t["M"], mass[c], fr.get(c)) + [float(c in tru)])
    for q in NPQ.itertuples():
        for c in q.pool:
            rows.append(["trainNP", q.rid, q.ik, c] + F14.pair_features(spec[q.rid], q.adduct, q.pm, q.M, mass[c], fr.get(c)) + [float(c == q.ik)])
    df = pd.DataFrame(rows, columns=["set", "rid", "qik", "cik"] + F14.ALL_FEATS + ["label"])
    df.to_parquet(R / "pairs.parquet", index=False)
    info = {"rows": df["set"].value_counts().to_dict(), "positives": df.groupby("set")["label"].sum().to_dict(),
            "nan_cells": int(df[F14.ALL_FEATS].isna().sum().sum()), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "feats.json", "w"), indent=1, default=float)
    log(info)


def stage_trainnp():
    import lightgbm as lgb
    df = pd.read_parquet(R / "pairs.parquet")
    tr = df[df["set"] == "trainNP"]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, class_weight="balanced",
                           subsample=0.8, colsample_bytree=0.8, random_state=F14.SEED, verbosity=-1)
    m.fit(tr[F14.ALL_FEATS].values, tr["label"].values)
    pickle.dump(m, open(R / "model_fragNP.pkl", "wb"))
    log(f"FRAG-NP trained on {tr['rid'].nunique()} queries / {len(tr)} pairs")


# ============================================================================ eval
def tie_iso(sc, ok, same_f):
    if not same_f.any():
        return np.nan
    b = sc[ok].max()
    return float(np.mean((sc[same_f] > b) + 0.5 * (sc[same_f] == b)))


def stage_eval():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from scipy.stats import rankdata
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    df = pd.read_parquet(R / "pairs.parquet")
    tg = df[df["set"] == "target"].copy()
    v2 = pickle.load(open(V2 / "models.pkl", "rb"))["LGB_all"]
    tg["FRAG"] = v2[0].predict_proba(tg[v2[1]].values)[:, 1]
    tg["FRAG_NP"] = pickle.load(open(R / "model_fragNP.pkl", "rb")).predict_proba(tg[F14.ALL_FEATS].values)[:, 1]
    T = pd.read_pickle(R / "targets.pkl")
    n_accepted = int((T["status"] == "ok").sum())
    T = T[(T["status"] == "ok") & T["covered"]].set_index("ik")
    C, U8 = F14.structures()
    mass = dict(zip(C["ik"], C["mass"]))
    smi = dict(zip(C["ik"], C["smiles"]))
    Cp = pd.read_parquet(E12 / "coconut_db.parquet")
    cfp = np.load(E12 / "coconut_fp_packed.npy")[Cp["parse_ok"].values]
    crow = {k: i for i, k in enumerate(Cp.loc[Cp["parse_ok"], "ik"])}
    z = np.load(R / "pred_knn.npz")
    pred = dict(zip(z["iks"], z["pred"]))
    form = {}

    def formula(k):
        if k not in form:
            form[k] = CalcMolFormula(Chem.MolFromSmiles(smi[k]))
        return form[k]
    rows = []
    for ik, g in tg.groupby("qik"):
        t = T.loc[ik]
        pool = list(t["pool"])
        ok = np.array([c in set(t["truth"]) for c in pool])
        agg = g.groupby("cik")[["FRAG", "FRAG_NP"]].mean().reindex(pool)
        bits = np.unpackbits(cfp[[crow[c] for c in pool]], axis=1).astype(np.float32)
        s = {"kNN": E.cosine_scores(pred[ik], bits), "B1": -np.abs(np.array([mass[c] for c in pool]) - t["M"]),
             "FRAG": agg["FRAG"].values, "FRAG_NP": agg["FRAG_NP"].values}
        s["COMBO"] = -(0.5 * rankdata(-s["kNN"]) + 0.5 * rankdata(-s["FRAG"]))
        tf = formula(pool[int(np.flatnonzero(ok)[0])])
        same_f = np.array([(not ok[i]) and formula(c) == tf for i, c in enumerate(pool)])
        n = len(pool)
        rec = {"ik": ik, "pool": n, "has_same_formula": bool(same_f.any()),
               "chance_rr": float(np.mean(1 / np.arange(1, n - ok.sum() + 2)))}
        for nm, sc in s.items():
            st = F14.rstats(sc, ok)
            rec.update({f"{nm}_{k}": v for k, v in st.items()})
            rec[f"{nm}_iso"] = tie_iso(sc, ok, same_f)
        rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(A / "per_target_results.csv", index=False)
    names = ["kNN", "B1", "FRAG", "FRAG_NP", "COMBO"]
    boot = F14.boot
    sf = D[D["has_same_formula"]]
    res = {"n_accepted": n_accepted, "n_ranked": len(D), "coverage": len(D) / n_accepted,
           "pool_median": float(D["pool"].median()), "chance_MRR": float(D["chance_rr"].mean()),
           "models": {nm: {"MRR": float(D[f"{nm}_rr"].mean()), "MRR@25": float(D[f"{nm}_rr25"].mean()),
                           "R@1": float(D[f"{nm}_r@1"].mean()), "R@10": float(D[f"{nm}_r@10"].mean()),
                           "sameformula_n": len(sf), "sameformula_MRR": float(sf[f"{nm}_rr"].mean()),
                           "isomer_error_tieaware": float(sf[f"{nm}_iso"].mean())} for nm in names}}
    bs = lambda x: boot(x.values, 20260930)
    res["H1"] = {"FRAG_minus_B1_MRR": bs(D["FRAG_rr"] - D["B1_rr"]), "FRAG_minus_B1_isoerr": bs(sf["FRAG_iso"] - sf["B1_iso"])}
    res["H2"] = {"COMBO_minus_kNN_MRR": bs(D["COMBO_rr"] - D["kNN_rr"]), "COMBO_minus_kNN_R@1": bs(D["COMBO_r@1"] - D["kNN_r@1"]),
                 "COMBO_minus_kNN_isoerr": bs(sf["COMBO_iso"] - sf["kNN_iso"])}
    res["H3"] = {"FRAGNP_minus_FRAG_MRR": bs(D["FRAG_NP_rr"] - D["FRAG_rr"]), "FRAGNP_minus_FRAG_isoerr": bs(sf["FRAG_NP_iso"] - sf["FRAG_iso"])}
    res["other"] = {"FRAG_minus_kNN_MRR": bs(D["FRAG_rr"] - D["kNN_rr"]), "FRAGNP_minus_kNN_MRR": bs(D["FRAG_NP_rr"] - D["kNN_rr"])}
    ck, cf, cc = D["kNN_r@1"] >= 0.5, D["FRAG_r@1"] >= 0.5, D["COMBO_r@1"] >= 0.5
    res["transitions"] = {"kNN_only": int((ck & ~cf).sum()), "FRAG_only": int((~ck & cf).sum()), "both": int((ck & cf).sum()),
                          "neither": int((~ck & ~cf).sum()), "COMBO_fixes_vs_kNN": int((~ck & cc).sum()),
                          "COMBO_breaks_vs_kNN": int((ck & ~cc).sum())}

    def verdict(lo_hi_pairs):
        return lo_hi_pairs
    h1a, h1b = res["H1"]["FRAG_minus_B1_MRR"][1] > 0, res["H1"]["FRAG_minus_B1_isoerr"][2] < 0
    res["gates"] = {"H1": "PASS" if (h1a and h1b) else ("FAIL" if not (h1a or h1b) else "INCONCLUSIVE"),
                    "H2": ("confirmed" if res["H2"]["COMBO_minus_kNN_MRR"][1] > 0 else
                           "harmful" if res["H2"]["COMBO_minus_kNN_MRR"][2] < 0 else "not confirmed"),
                    "H3": "confirmed" if res["H3"]["FRAGNP_minus_FRAG_MRR"][1] > 0 else "not confirmed"}
    res["sec"] = round(time.time() - t0, 1)
    json.dump(res, open(A / "summary.json", "w"), indent=1, default=float)
    log(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True); A.mkdir(parents=True, exist_ok=True)
    {"build": stage_build, "knn": stage_knn, "frag": stage_frag, "feats": stage_feats, "trainnp": stage_trainnp,
     "eval": stage_eval}[sys.argv[1]]()

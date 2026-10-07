"""EXP-014 v2: structure-derived (in-silico) fragmentation features for Class-2 ranking.
Design (locked): research/analysis/exp014_fragmentation_features_v2_design.md. v1 script/outputs untouched (INVALID).

    python research/scripts/exp014_fragmentation_features_v2.py prep     # exclusions, pools, query spectra, gates
    python research/scripts/exp014_fragmentation_features_v2.py frag     # in-silico fragments per candidate structure
    python research/scripts/exp014_fragmentation_features_v2.py feats    # pair features
    python research/scripts/exp014_fragmentation_features_v2.py train    # LightGBM (primary) + secondaries
    python research/scripts/exp014_fragmentation_features_v2.py eval     # target ranking, CIs, taxonomy, transitions
"""
import json
import pickle
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402

E11, E12 = ROOT / "results" / "exp011_class2_proxy", ROOT / "results" / "exp012_coconut"
V1 = ROOT / "results" / "exp014_fragfeats"               # read-only: query tables only
R = ROOT / "results" / "exp014_fragfeats_v2"
A = ROOT / "research" / "analysis" / "exp014" / "v2"
SEED, BOOT_SEED = 20260925, 20260929
H_MASS, PROTON = 1.007825, 1.007276
TOPK_PEAKS, MAX_BONDS_2CUT = 30, 60
MASS_FEATS = ["mass_diff_ppm", "mass_diff_mDa", "adduct_err_ppm", "best_adduct_err_ppm", "adduct_is_best"]
FRAG_FEATS = ["frac_explained", "int_explained", "n_explained", "frac_explained_1cut", "n_fragments_log"]
ALL_FEATS = MASS_FEATS + FRAG_FEATS
TC_NEAR, TC_UNREL = 0.50, 0.30


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ============================================================================ prep
def structures():
    """ik -> smiles/mass/parent for COCONUT (parse_ok) and train universe (EXP-008)."""
    C = pd.read_parquet(E12 / "coconut_db.parquet")
    C = C[C["parse_ok"]][["ik", "smiles", "mass", "parent", "formula"]]
    U8 = pd.read_parquet(E.UNIVERSE)
    U8 = U8[U8["parse_ok"]][["ik", "smiles", "mass", "parent", "formula"]]
    return C, U8


def stage_prep():
    t0 = time.time()
    R.mkdir(parents=True, exist_ok=True); A.mkdir(parents=True, exist_ok=True)
    C, U8 = structures()
    targets = pd.read_parquet(E11 / "targets.parquet")
    man = json.load(open(E11 / "manifest.json"))
    A12 = json.load(open(E12 / "target_aliases.json"))
    D = set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
    tset = set(targets["ik"])
    tpar = set(U8.loc[U8["ik"].isin(tset), "parent"]) - {None}
    X = (tset | {x for v in man["aliases"].values() for x in v} | D
         | {x for a in A12.values() for k in ("ik", "parent", "tautomer") for x in a[k]}
         | set(U8.loc[U8["parent"].isin(tpar), "ik"]) | set(C.loc[C["parent"].isin(tpar), "ik"]))
    par8 = dict(zip(U8["ik"], U8["parent"]))

    qtr, qva = pd.read_parquet(V1 / "queries_train.parquet"), pd.read_parquet(V1 / "queries_val.parquet")
    gates = {}
    for name, q in (("train", qtr), ("val", qva)):
        gates[f"{name}_query_molecules_in_exclusions"] = int(q["ik"].isin(X).sum())
        gates[f"{name}_query_parent_is_target_parent"] = int(q["ik"].map(par8).isin(tpar).sum())
    gates["val_train_molecule_overlap"] = len(set(qtr["ik"]) & set(qva["ik"]))
    tr_rows = pd.read_parquet(E11 / "training.parquet")
    split_of = dict(zip(tr_rows["rid"], tr_rows["split"]))
    gates["train_query_rid_not_train_split"] = int((qtr["rid"].map(split_of) != "train").sum())
    gates["val_query_rid_not_val_split"] = int((qva["rid"].map(split_of) != "val").sum())
    rows, removed = [], 0
    for name, q in (("train", qtr), ("val", qva)):
        for r in q.itertuples():
            pool = [c for c in r.pool if c == r.ik or c not in X]
            removed += len(r.pool) - len(pool)
            rows.append({"set": name, "rid": int(r.rid), "ik": r.ik, "adduct": r.adduct, "pm": float(r.pm), "M": float(r.M),
                         "pool": pool, "truth": [r.ik]})
    gates["heldout_candidates_removed_from_train_val_pools"] = removed
    gates["heldout_in_train_val_pools_after"] = sum(1 for r in rows for c in r["pool"] if c in X and c != r["ik"])

    # target groups (molecule level, exactly as v1 eval / EXP-012): T1 -> COCONUT, T2 -> COCONUT+train
    qt = pd.read_parquet(V1 / "queries_target.parquet").drop_duplicates("rid")[["rid", "ik", "pop", "adduct", "pm", "M"]]
    UC = C.sort_values("mass").reset_index(drop=True)
    UCT = pd.concat([C, U8[~U8["ik"].isin(set(C["ik"]))]]).sort_values("mass").reset_index(drop=True)
    tgt = []
    for ik, g in qt.groupby("ik"):
        pop = g["pop"].iloc[0]
        Uf = UC if pop == "T1_np" else UCT
        M = float(np.median(g["M"]))
        m = Uf["mass"].values
        lo, hi = np.searchsorted(m, M * (1 - 5e-6)), np.searchsorted(m, M * (1 + 5e-6), side="right")
        pool = Uf["ik"].values[lo:hi].tolist()
        a = A12.get(ik, {"ik": [], "parent": [], "tautomer": []})
        ok = set(a["ik"]) | set(a["parent"]) | set(a["tautomer"]) | ({ik} if pop != "T1_np" else set())
        truth = [c for c in pool if c in ok]
        for r in g.itertuples():
            rows.append({"set": "target", "rid": int(r.rid), "ik": ik, "pop": pop, "adduct": r.adduct, "pm": float(r.pm),
                         "M": M, "pool": pool, "truth": truth})
        tgt.append({"ik": ik, "pop": pop, "M": M, "pool": len(pool), "covered": bool(truth)})
    Q = pd.DataFrame(rows)
    Q.to_pickle(R / "queries.pkl")
    pd.DataFrame(tgt).to_parquet(R / "target_groups.parquet", index=False)

    # query spectra, fetched fresh with m/z and intensity sorted TOGETHER (v1 rid_spec.pkl sorted m/z only)
    con = duckdb.connect(); con.execute("SET threads=4")
    con.register("w", pd.DataFrame({"rid": Q["rid"].unique()}))
    sp = con.execute(f"""SELECT file_row_number AS rid, precursor_mz AS pm, ms2_mzs AS mz, ms2_normalized_intensities AS it
                         FROM read_parquet('{E.TRAIN}', file_row_number=true) WHERE file_row_number IN (SELECT rid FROM w)""").df()
    spec = {}
    for r in sp.itertuples():
        mz, it = np.asarray(r.mz, float), np.asarray(r.it, float)
        o = np.argsort(mz)
        spec[int(r.rid)] = (mz[o], it[o], float(r.pm))
    pickle.dump(spec, open(R / "query_spectra.pkl", "wb"))
    iks = sorted({c for p in Q["pool"] for c in p})
    smi = dict(zip(U8["ik"], U8["smiles"])); smi.update(dict(zip(C["ik"], C["smiles"])))  # COCONUT SMILES preferred
    pd.DataFrame({"ik": iks, "smiles": [smi[k] for k in iks]}).to_parquet(R / "candidates.parquet", index=False)
    gates["all_gates_zero"] = all(v == 0 for k, v in gates.items() if k != "heldout_candidates_removed_from_train_val_pools")
    info = {"gates": gates, "n_queries": Q["set"].value_counts().to_dict(), "n_candidate_structures": len(iks),
            "target_groups": pd.DataFrame(tgt).groupby("pop")["covered"].agg(["count", "sum"]).to_dict("index"),
            "exclusion_keys": len(X), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "prep.json", "w"), indent=1, default=int)
    log(json.dumps(info, indent=1, default=int))
    assert gates["all_gates_zero"], "leakage gate failed"


# ============================================================================ in-silico fragmentation
def _fragments(smi):
    """Neutral fragment masses (1- and 2-bond cleavages, implicit H kept on their atoms). Returns (all, onecut)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    pt = Chem.GetPeriodicTable()
    am = np.array([pt.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * H_MASS for a in m.GetAtoms()])
    n = m.GetNumAtoms()
    adj = [0] * n
    bonds = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i] |= 1 << j; adj[j] |= 1 << i
        bonds.append((i, j))
    full = (1 << n) - 1

    def comps(cut):
        a2 = list(adj)
        for i, j in cut:
            a2[i] &= ~(1 << j); a2[j] &= ~(1 << i)
        seen, out = 0, []
        for s in range(n):
            if seen >> s & 1:
                continue
            comp, frontier = 1 << s, 1 << s
            while frontier:
                nxt = 0
                f = frontier
                while f:
                    low = f & -f
                    nxt |= a2[low.bit_length() - 1]
                    f ^= low
                frontier = nxt & ~comp
                comp |= nxt
            seen |= comp
            out.append(comp)
        return out

    def mass_of(mask):
        idx = [k for k in range(n) if mask >> k & 1]
        return float(am[idx].sum())
    one, allm, cache = set(), set(), {}
    for b in bonds:
        for c in comps([b]):
            if c != full:
                v = cache.setdefault(c, mass_of(c)); one.add(round(v, 5)); allm.add(round(v, 5))
    if len(bonds) <= MAX_BONDS_2CUT:
        for b1, b2 in combinations(bonds, 2):
            for c in comps([b1, b2]):
                if c != full:
                    allm.add(round(cache.setdefault(c, mass_of(c)), 5))
    allm.add(round(float(am.sum()), 5))
    return np.array(sorted(allm)), np.array(sorted(one))


def _frag_job(args):
    ik, smi = args
    try:
        return ik, _fragments(smi)
    except Exception:
        return ik, None


def stage_frag():
    t0 = time.time()
    cand = pd.read_parquet(R / "candidates.parquet")
    with Pool(6) as pool:
        res = pool.map(_frag_job, list(zip(cand["ik"], cand["smiles"])), chunksize=200)
    fr = {ik: f for ik, f in res}
    pickle.dump(fr, open(R / "fragments.pkl", "wb"))
    ok = [f for f in fr.values() if f is not None]
    info = {"structures": len(fr), "failed": sum(f is None for f in fr.values()),
            "median_fragment_masses": float(np.median([len(f[0]) for f in ok])), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "frag.json", "w"), indent=1)
    log(info)


# ============================================================================ pair features
def pair_features(spec, adduct, pm, Mq, Mc, frag):
    mz, it, qpm = spec
    ad = E.AD10[adduct]
    ad_err = abs(pm - (Mc + ad)) / (Mc + ad) * 1e6
    errs = {k: abs(pm - (Mc + v)) / (Mc + v) * 1e6 for k, v in E.AD10.items()}
    best = min(errs, key=errs.get)
    mass = [1e6 * (Mq - Mc) / Mc, 1000.0 * (Mq - Mc), ad_err, errs[best], float(best == adduct)]
    keep = (mz < qpm - 2.0) & (it > 0)
    mzk, itk = mz[keep], it[keep]
    if len(mzk) > TOPK_PEAKS:
        o = np.argsort(-itk, kind="stable")[:TOPK_PEAKS]
        mzk, itk = mzk[o], itk[o]
    if frag is None or len(mzk) == 0:
        return mass + [0.0, 0.0, 0.0, 0.0, 0.0]
    neg = adduct in E.NEG
    hs = np.array([-1, 0, 1, 2]) if neg else np.array([-2, -1, 0, 1])
    base = -PROTON if neg else PROTON

    def explained(fm):
        ions = np.sort((fm[:, None] + base + hs[None, :] * H_MASS).ravel())
        tol = np.maximum(0.01, mzk * 20e-6)
        lo = np.searchsorted(ions, mzk - tol)
        hi = np.searchsorted(ions, mzk + tol, side="right")
        return hi > lo
    e_all, e_one = explained(frag[0]), explained(frag[1]) if len(frag[1]) else np.zeros(len(mzk), bool)
    return mass + [float(e_all.mean()), float(itk[e_all].sum() / itk.sum()), float(e_all.sum()), float(e_one.mean()),
                   float(np.log10(max(len(frag[0]), 1)))]


def stage_feats():
    t0 = time.time()
    Q = pd.read_pickle(R / "queries.pkl")
    spec = pickle.load(open(R / "query_spectra.pkl", "rb"))
    fr = pickle.load(open(R / "fragments.pkl", "rb"))
    C, U8 = structures()
    mass = dict(zip(U8["ik"], U8["mass"])); mass.update(dict(zip(C["ik"], C["mass"])))
    rows = []
    for q in Q.itertuples():
        s = spec[q.rid]
        tru = set(q.truth)
        for c in q.pool:
            rows.append([q.set, q.rid, q.ik, getattr(q, "pop", None), c]
                        + pair_features(s, q.adduct, q.pm, q.M, mass[c], fr.get(c)) + [float(c in tru)])
    df = pd.DataFrame(rows, columns=["set", "rid", "qik", "pop", "cik"] + ALL_FEATS + ["label"])
    df.to_parquet(R / "pairs.parquet", index=False)
    info = {"rows": len(df), "by_set": df["set"].value_counts().to_dict(),
            "positives": df.groupby("set")["label"].sum().to_dict(), "nan_cells": int(df[ALL_FEATS].isna().sum().sum()),
            "sec": round(time.time() - t0, 1)}
    json.dump(info, open(R / "feats.json", "w"), indent=1, default=float)
    log(info)


# ============================================================================ train
def stage_train():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    df = pd.read_parquet(R / "pairs.parquet")
    tr, va = df[df["set"] == "train"], df[df["set"] == "val"]
    mk = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, class_weight="balanced",
                                    subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1)
    models, info = {}, {}
    for name, feats in (("LGB_all", ALL_FEATS), ("LGB_fragonly", FRAG_FEATS), ("LGB_massonly", MASS_FEATS)):
        m = mk().fit(tr[feats].values, tr["label"].values)
        models[name] = (m, feats, None)
    sc = StandardScaler().fit(tr[ALL_FEATS].values)
    models["LR_all"] = (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
                        .fit(sc.transform(tr[ALL_FEATS].values), tr["label"].values), ALL_FEATS, sc)
    for name, (m, feats, s) in models.items():
        X = va[feats].values if s is None else s.transform(va[feats].values)
        p = m.predict_proba(X)[:, 1]
        v = va.assign(p=p)
        rr = []
        for _, g in v.groupby("rid"):
            st = E.expected_rank_stats(dict(zip(g["cik"], g["p"])), g["qik"].iloc[0])
            rr.append(st["rr25"])
        info[name] = {"val_auc": float(roc_auc_score(va["label"], p)), "val_mrr25_spectrum_level": float(np.mean(rr))}
    pickle.dump(models, open(R / "models.pkl", "wb"))
    info["sec"] = round(time.time() - t0, 1)
    json.dump(info, open(R / "train.json", "w"), indent=1)
    log(info)


# ============================================================================ eval
def boot(d, seed=BOOT_SEED):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def rstats(scores, ok):
    best = scores[ok].max()
    o = scores[~ok]
    better, tied = int((o > best).sum()), int((o == best).sum())
    ranks = np.arange(better + 1, better + tied + 2)
    return {"rank": float(ranks.mean()), "rr": float(np.mean(1 / ranks)), "rr25": float(np.mean(np.where(ranks <= 25, 1 / ranks, 0))),
            "r@1": float(np.mean(ranks <= 1)), "r@10": float(np.mean(ranks <= 10))}


def stage_eval():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    df = pd.read_parquet(R / "pairs.parquet")
    tg = df[df["set"] == "target"].copy()
    models = pickle.load(open(R / "models.pkl", "rb"))
    for name, (m, feats, s) in models.items():
        X = tg[feats].values if s is None else s.transform(tg[feats].values)
        tg[name] = m.predict_proba(X)[:, 1]
    Q = pd.read_pickle(R / "queries.pkl")
    Qt = Q[Q["set"] == "target"].drop_duplicates("ik").set_index("ik")
    C, U8 = structures()
    mass = dict(zip(U8["ik"], U8["mass"])); mass.update(dict(zip(C["ik"], C["mass"])))
    smi = dict(zip(U8["ik"], U8["smiles"])); smi.update(dict(zip(C["ik"], C["smiles"])))
    Cp = pd.read_parquet(E12 / "coconut_db.parquet")
    cfp = np.load(E12 / "coconut_fp_packed.npy")[Cp["parse_ok"].values]
    crow = {k: i for i, k in enumerate(Cp.loc[Cp["parse_ok"], "ik"])}
    F = E.FP(E11)
    z = np.load(E11 / "pred_B3.npz")
    pred = dict(zip(z["iks"], z["pred"]))
    form = {}

    def fp_packed(k):
        return cfp[crow[k]] if k in crow else F.packed[F.row[k]]

    def formula(k):
        if k not in form:
            form[k] = CalcMolFormula(Chem.MolFromSmiles(smi[k]))
        return form[k]
    rows = []
    for ik, g in tg.groupby("qik"):
        q = Qt.loc[ik]
        pool = list(q["pool"])
        ok = np.array([c in set(q["truth"]) for c in pool])
        if not ok.any() or ik not in pred:
            continue
        agg = g.groupby("cik")[list(models)].mean().reindex(pool)
        packed = np.stack([fp_packed(c) for c in pool])
        bits = np.unpackbits(packed, axis=1).astype(np.float32)
        s = {"kNN": E.cosine_scores(pred[ik], bits), "B1_mass": -np.abs(np.array([mass[c] for c in pool]) - q["M"])}
        for name in models:
            s[name] = agg[name].values
        ti = int(np.flatnonzero(ok)[0])
        inter = np.bitwise_count(np.bitwise_and(packed, packed[ti])).sum(1)
        union = np.bitwise_count(np.bitwise_or(packed, packed[ti])).sum(1)
        tc = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        tf = formula(pool[ti])
        same_f = np.array([(not ok[i]) and formula(c) == tf for i, c in enumerate(pool)])
        n = len(pool)
        rec = {"ik": ik, "pop": q["pop"], "pool": n, "has_same_formula": bool(same_f.any()),
               "chance_rr": float(np.mean(1 / np.arange(1, n - ok.sum() + 2)))}
        for name, sc in s.items():
            st = rstats(sc, ok)
            rec.update({f"{name}_{k}": v for k, v in st.items()})
            rec[f"{name}_iso_err"] = float(np.mean(sc[same_f] > sc[ok].max())) if same_f.any() else np.nan
            # POST-HOC (added after first eval, labelled in the report): tie-aware version, ties count 0.5.
            # The registered strict ">" version is degenerate for mass-only scorers (same-formula isomers tie the truth).
            rec[f"{name}_iso_err_tieaware"] = (float(np.mean((sc[same_f] > sc[ok].max()) + 0.5 * (sc[same_f] == sc[ok].max())))
                                               if same_f.any() else np.nan)
        wrong = np.flatnonzero(~ok)
        if len(wrong) == 0:  # pool contains only the truth: rank 1 for every scorer
            rec["kNN_top1_category"] = "no_competitor"
            rows.append(rec)
            continue
        top1 = wrong[np.argmax(s["kNN"][wrong])]
        rec["kNN_top1_category"] = ("correct" if rec["kNN_rank"] <= 1 else
                                    ("near_isomer" if tc[top1] >= TC_NEAR else "distant_isomer") if same_f[top1] else
                                    ("unrelated" if tc[top1] < TC_UNREL else "other"))
        rows.append(rec)
    T = pd.DataFrame(rows)
    A.mkdir(parents=True, exist_ok=True)
    T.to_csv(A / "per_target_results.csv", index=False)
    names = ["kNN", "B1_mass"] + list(models)
    rep = {}
    for pop, g in T.groupby("pop"):
        tgroups = pd.read_parquet(R / "target_groups.parquet")
        n_all = int((tgroups["pop"] == pop).sum())
        P = {"n_targets": n_all, "n_ranked": len(g), "coverage": len(g) / n_all, "pool_median": float(g["pool"].median()),
             "chance_MRR": float(g["chance_rr"].mean())}
        for nm in names:
            sf = g[g["has_same_formula"]]
            P[nm] = {"MRR": float(g[f"{nm}_rr"].mean()), "MRR@25": float(g[f"{nm}_rr25"].mean()),
                     "R@1": float(g[f"{nm}_r@1"].mean()), "R@10": float(g[f"{nm}_r@10"].mean()),
                     "sameformula_n": len(sf), "sameformula_MRR": float(sf[f"{nm}_rr"].mean()),
                     "isomer_error_rate": float(g[f"{nm}_iso_err"].mean())}
        p = "LGB_all"
        sf = g.dropna(subset=[f"{p}_iso_err", "B1_mass_iso_err", "kNN_iso_err"])
        P["deltas"] = {"LGB_all_minus_B1_MRR": boot(g[f"{p}_rr"] - g["B1_mass_rr"]),
                       "LGB_all_minus_kNN_MRR": boot(g[f"{p}_rr"] - g["kNN_rr"]),
                       "LGB_all_minus_B1_R@1": boot(g[f"{p}_r@1"] - g["B1_mass_r@1"]),
                       "LGB_all_minus_kNN_R@1": boot(g[f"{p}_r@1"] - g["kNN_r@1"]),
                       "LGB_fragonly_minus_chance_MRR": boot(g["LGB_fragonly_rr"] - g["chance_rr"]),
                       "LGB_all_minus_LGB_massonly_MRR": boot(g[f"{p}_rr"] - g["LGB_massonly_rr"]),
                       "isoerr_LGB_all_minus_B1": boot(sf[f"{p}_iso_err"] - sf["B1_mass_iso_err"]),
                       "isoerr_LGB_all_minus_kNN": boot(sf[f"{p}_iso_err"] - sf["kNN_iso_err"]),
                       "POSTHOC_isoerr_tieaware_LGB_all_minus_B1": boot(sf[f"{p}_iso_err_tieaware"] - sf["B1_mass_iso_err_tieaware"]),
                       "POSTHOC_isoerr_tieaware_LGB_all_minus_kNN": boot(sf[f"{p}_iso_err_tieaware"] - sf["kNN_iso_err_tieaware"])}
        for nm in names:
            P[nm]["POSTHOC_isomer_error_rate_tieaware"] = float(g[f"{nm}_iso_err_tieaware"].mean())
        ck, ce = g["kNN_r@1"] >= 0.5, g[f"{p}_r@1"] >= 0.5
        P["transitions_vs_kNN"] = {"kNN_correct_E14_wrong": int((ck & ~ce).sum()), "kNN_wrong_E14_correct": int((~ck & ce).sum()),
                                   "both_correct": int((ck & ce).sum()), "both_wrong": int((~ck & ~ce).sum())}
        P["by_kNN_top1_category"] = {c: {"n": len(h), "kNN_MRR": float(h["kNN_rr"].mean()), "E14_MRR": float(h[f"{p}_rr"].mean()),
                                         "B1_MRR": float(h["B1_mass_rr"].mean()), "E14_better": int((h[f"{p}_rank"] < h["kNN_rank"]).sum()),
                                         "E14_worse": int((h[f"{p}_rank"] > h["kNN_rank"]).sum())}
                                     for c, h in g.groupby("kNN_top1_category")}
        rep[pop] = P
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(A / "eval_summary.json", "w"), indent=1, default=float)
    log(json.dumps(rep, indent=1, default=float)[:8000])


if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True)
    {"prep": stage_prep, "frag": stage_frag, "feats": stage_feats, "train": stage_train, "eval": stage_eval}[sys.argv[1]]()

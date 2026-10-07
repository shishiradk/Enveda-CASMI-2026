"""EXP-018 -- learned candidate ranker (LightGBM lambdarank) over multi-engine features, leak-free by per-query exclusion.

One shared reference set R (the V1 reference construction: train spectra in the 10 test adducts, enveda-180 excluded,
<=3 per (InChIKey14, library)). Every query molecule carries its own exclusion:
  C2-type ("known structure, no public spectra"): every R row of the target (and of its parent group) is masked, and a
    train-only universe row of the target leaves its candidate pool (the truth stays only if COCONUT has it);
  C1-type ("has public spectra"): only the R rows of the target from the query's own library are masked; the target's
    spectra from other libraries remain retrievable.
R rows whose peak arrays are byte-identical to a query spectrum are masked in both types.

Populations (disjoint by InChIKey14 / parent group):
  TRAIN 3,000 and VAL 1,000 molecules sampled from natural-product libraries (gnps, riken, mona, massbank, msdial,
  pluskal_ms2), mass 157-1,159 Da, excluding every proxy target (V1 np-examples 250 + EXP-017 S3 300);
  30% C1-type where the molecule has spectra in >= 2 libraries, else C2-type; <= 4 query spectra from one library.
  Final tests, untouched by training/selection: S1, S2 (np-examples; V1 constructions) and S3 (PubChem-only NPs).

Features per (query molecule, candidate): kNN fingerprint cosine (mean prediction, max over spectra, z / gap / rank
within the pool), direct neighbour vote for the candidate, max binned cosine to the candidate's own unmasked R spectra
(any adduct, same adduct), number of unmasked R spectra, source flags, |mass error| ppm, pool size, same-formula count.

    python research/exp018/exp018_ranker.py build
    python research/exp018/exp018_ranker.py feats [--sets train,val,s1,s2,s3]
    python research/exp018/exp018_ranker.py train
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "kaggle_v1"))
sys.path.insert(0, str(ROOT / "research" / "kaggle_v2"))
import casmi_v1_kaggle as v1  # noqa: E402

TRAIN = (ROOT / "train.parquet").as_posix()
UNIV = (ROOT / "results" / "kaggle_v1_assets" / "universe.parquet").as_posix()
X8 = (ROOT / "results" / "exp008_structure_universe.parquet").as_posix()
O = ROOT / "results" / "exp018"
SEED = 20260930
NP_LIBS = ("gnps", "riken", "mona", "massbank", "msdial", "pluskal_ms2")
CFG = dict(v1.CFG)
FEATS = ["knn_mean", "knn_max", "knn_z", "knn_gap", "knn_rankpct", "nb_vote", "self_max", "self_same", "n_ref",
         "has_ref", "src_coconut", "src_train", "mass_err_ppm", "pool_size", "n_same_formula", "log_pool"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ----------------------------------------------------------------------------- build
def stage_build(a):
    O.mkdir(parents=True, exist_ok=True)
    ads = ",".join(f"'{x}'" for x in v1.AD10)
    libs = ",".join(f"'{x}'" for x in NP_LIBS)
    held = set(pd.read_parquet(ROOT / "results" / "kaggle_v1_proxy" / "held_keys.parquet")["ik"])
    held |= set(pd.read_parquet(ROOT / "results" / "kaggle_v2_proxy" / "s3_held.parquet")["ik"])
    m = duckdb.sql(f"""
        SELECT s.inchikey14 AS ik, list(DISTINCT s.ingest_lib) AS libs, any_value(x.parent) AS parent, any_value(x.mass) AS mass
        FROM '{TRAIN}' s JOIN '{X8}' x ON x.ik = s.inchikey14
        WHERE s.adduct IN ({ads}) AND s.ingest_lib IN ({libs}) AND x.parse_ok AND x.mass BETWEEN 157 AND 1159
        GROUP BY 1""").df()
    parents_held = set(pd.read_parquet(X8, columns=["ik", "parent"]).query("ik in @held")["parent"])
    m = m[~m["ik"].isin(held) & ~m["parent"].isin(parents_held)].sort_values("ik").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    # one molecule per parent group, then sample
    m = m.sample(frac=1.0, random_state=SEED).drop_duplicates("parent").sort_values("ik").reset_index(drop=True)
    pick = rng.choice(len(m), a.n_train + a.n_val, replace=False)
    T = m.iloc[pick].reset_index(drop=True)
    T["set"] = ["train"] * a.n_train + ["val"] * a.n_val
    T["nlib"] = T["libs"].map(len)
    T["qtype"] = np.where((T["nlib"] >= 2) & (rng.random(len(T)) < 0.30), "C1", "C2")
    T["qlib"] = [rng.choice(sorted(l)) for l in T["libs"]]
    T["libs"] = T["libs"].map(lambda l: ",".join(sorted(l)))
    # C1x: timsTOF queries (enveda-180) of molecules that also have spectra in other libraries -- the hidden test's
    # Class-1 geometry (timsTOF query vs references from other instruments; enveda-180 is not in R).
    x = duckdb.sql(f"""
        SELECT s.inchikey14 AS ik, list(DISTINCT s.ingest_lib) AS libs, any_value(x.parent) AS parent, any_value(x.mass) AS mass
        FROM '{TRAIN}' s JOIN '{X8}' x ON x.ik = s.inchikey14
        WHERE s.adduct IN ({ads}) AND x.parse_ok AND x.mass BETWEEN 157 AND 1159
        GROUP BY 1 HAVING bool_or(s.ingest_lib = 'enveda-180') AND bool_or(s.ingest_lib <> 'enveda-180')""").df()
    x = x[~x["ik"].isin(held) & ~x["parent"].isin(parents_held | set(T["parent"]))]
    x = x.sample(frac=1.0, random_state=SEED).drop_duplicates("parent")
    x = x.sample(min(len(x), a.n_c1x_train + a.n_c1x_val), random_state=SEED).reset_index(drop=True)
    x["set"] = ["train"] * min(a.n_c1x_train, len(x)) + ["val"] * (len(x) - min(a.n_c1x_train, len(x)))
    x["nlib"], x["qtype"], x["qlib"] = x["libs"].map(len), "C1", "enveda-180"
    x["libs"] = x["libs"].map(lambda l: ",".join(sorted(l)))
    x["pop"], T["pop"] = "c1x_tims", "np"
    T = pd.concat([T, x[T.columns.drop("pop").tolist() + ["pop"]]], ignore_index=True)
    T.to_parquet(O / "targets.parquet", index=False)
    duckdb.register("T", T[["ik", "qlib"]])
    q = duckdb.sql(f"""
        SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities, h FROM (
            SELECT s.inchikey14 AS molecule_id, 'r' || s.file_row_number AS spectrum_id, s.adduct, s.precursor_mz,
                   s.ms2_mzs, s.ms2_normalized_intensities, hash(s.ms2_mzs, s.ms2_normalized_intensities) AS h,
                   row_number() OVER (PARTITION BY s.inchikey14 ORDER BY hash(s.file_row_number + {SEED})) AS rn
            FROM read_parquet('{TRAIN}', file_row_number=true) s JOIN T ON T.ik = s.inchikey14 AND T.qlib = s.ingest_lib
            WHERE s.adduct IN ({ads}))
        WHERE rn <= 4""").df()
    q.to_parquet(O / "queries_trainval.parquet", index=False)
    info = {"eligible_parent_groups": int(len(m)), "targets": {f"{s}|{q}|{p}": int(n) for (s, q, p), n in T.groupby(["set", "qtype", "pop"]).size().items()},
            "query_spectra": int(len(q)), "held_proxy_keys": len(held)}
    json.dump({str(k): v for k, v in info.items()}, open(O / "build.json", "w"), indent=1, default=str)
    log(info)


# ----------------------------------------------------------------------------- reference with metadata
def load_reference():
    cache = O / "ref.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    t0 = time.time()
    ads = ",".join(f"'{x}'" for x in v1.AD10)
    con = v1.v0.connect(mem="6GB", threads=4, tmp=(O / "duck_tmp").as_posix())
    tbl = con.execute(f"""
        SELECT ik, lib, adduct, pm, h, ms2_mzs, ms2_normalized_intensities FROM (
            SELECT inchikey14 AS ik, ingest_lib AS lib, adduct, precursor_mz AS pm, ms2_mzs, ms2_normalized_intensities,
                   hash(ms2_mzs, ms2_normalized_intensities) AS h,
                   row_number() OVER (PARTITION BY inchikey14, ingest_lib ORDER BY hash(file_row_number + {SEED}), file_row_number) AS rn
            FROM read_parquet('{TRAIN}', file_row_number=true)
            WHERE adduct IN ({ads}) AND ingest_lib <> 'enveda-180' AND inchikey14 IS NOT NULL)
        WHERE rn <= {CFG['ref_cap']} ORDER BY ik, lib, adduct, pm""").arrow()
    con.close()
    if not hasattr(tbl, "column_names"):
        tbl = tbl.read_all()
    X = v1.featurize_arrow(tbl, CFG)
    meta = pd.DataFrame({c: tbl[c].to_numpy(zero_copy_only=False) for c in ("ik", "lib", "adduct", "h")})
    ref = {"X": X, "meta": meta}
    pickle.dump(ref, open(cache, "wb"), protocol=4)
    log(f"reference {X.shape} nnz {X.nnz} ({time.time() - t0:.0f}s)")
    return ref


# ----------------------------------------------------------------------------- per-query feature pass (worker)
_W = {}


def _init(ref_path, neg_path):
    from scipy import sparse
    _W["RT"] = sparse.load_npz(ref_path).T.tocsr()
    _W["neg"] = np.load(neg_path)


def _pass(args):
    """For each query spectrum: top-k neighbours after masking, and max similarity to each listed candidate's rows."""
    Xq, qneg, qadd, masks, cand_rows, k, radd = args
    S = (Xq @ _W["RT"]).toarray()
    S[qneg[:, None] != _W["neg"][None, :]] = -1.0
    for j, m in enumerate(masks):
        if len(m):
            S[j, m] = -1.0
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, axis=1)
    o = np.argsort(-sims, axis=1, kind="stable")
    idx, sims = np.take_along_axis(idx, o, axis=1), np.take_along_axis(sims, o, axis=1)
    selfsim = []
    for j, cr in enumerate(cand_rows):  # cr: {cand_ik: row array}
        d = {}
        for ik, rows in cr.items():
            v = S[j, rows]
            ok = v > -1.0
            if ok.any():
                same = v[ok & (radd[rows] == qadd[j])]
                d[ik] = (float(v[ok].max()), float(same.max()) if len(same) else 0.0, int(ok.sum()))
        selfsim.append(d)
    return idx, sims, selfsim


# ----------------------------------------------------------------------------- query sets
def query_sets(names):
    """name -> (queries DataFrame, targets DataFrame[ik, qtype, qlib, drop_train, correct(set)])."""
    out = {}
    if {"train", "val"} & set(names):
        T = pd.read_parquet(O / "targets.parquet")
        q = pd.read_parquet(O / "queries_trainval.parquet")
        for s in ("train", "val"):
            if s in names:
                t = T[T["set"] == s].copy()
                t["correct"] = [{ik} for ik in t["ik"]]
                t["mask_all_keys"] = [{ik} for ik in t["ik"]]
                out[s] = (q[q["molecule_id"].isin(set(t["ik"]))].reset_index(drop=True), t)
    if {"s1", "s2"} & set(names):
        import proxy_eval as pe1
        qp, targets, correct, held = pe1.build_queries()
        q = v1.v0.load_test(qp)
        q["h"] = duckdb.sql(f"SELECT spectrum_id, hash(ms2_mzs, ms2_normalized_intensities) AS h FROM '{qp.as_posix()}'").df() \
            .set_index("spectrum_id").loc[q["spectrum_id"], "h"].values
        A = json.load(open(ROOT / "results" / "exp012_coconut" / "target_aliases.json"))
        for s in ("s1", "s2"):
            if s in names:
                t = pd.DataFrame({"ik": targets})
                t["qtype"] = "C1" if s == "s1" else "C2"
                t["qlib"] = "enveda-np-examples"
                t["correct"] = [correct[x] for x in t["ik"]]
                t["mask_all_keys"] = [correct[x] | set(A.get(x, {}).get("parent", [])) for x in t["ik"]]
                out[s] = (q, t)
    if "s3" in names:
        import proxy_eval2 as pe2
        qp, T3, correct3, held3, _ = pe2.build_s3(300)
        q = v1.v0.load_test(qp)
        q["h"] = duckdb.sql(f"SELECT spectrum_id, hash(ms2_mzs, ms2_normalized_intensities) AS h FROM '{qp.as_posix()}'").df() \
            .set_index("spectrum_id").loc[q["spectrum_id"], "h"].values
        t = pd.DataFrame({"ik": T3["ik"]})
        t["qtype"], t["qlib"] = "C2", ""
        t["correct"] = [correct3[x] for x in t["ik"]]
        t["mask_all_keys"] = [correct3[x] | set(held3 & {x}) for x in t["ik"]]
        out["s3"] = (q, t)
    return out


# ----------------------------------------------------------------------------- features
def stage_feats(a):
    from scipy import sparse
    ref = load_reference()
    Xr, meta = ref["X"], ref["meta"]
    U = v1.Universe(Path(UNIV).parent)
    usrc = pd.read_parquet(UNIV, columns=["src"])["src"].values
    rows_of = meta.groupby("ik").indices                      # ik -> R rows
    rows_of_lib = meta.groupby(["ik", "lib"]).indices          # (ik, lib) -> R rows
    h2rows = meta.groupby("h").indices
    rrow = np.array([U.row.get(k, -1) for k in meta["ik"]])
    rneg = np.isin(meta["adduct"].to_numpy(dtype=object), list(v1.NEG))
    radd = meta["adduct"].to_numpy(dtype=object)
    IK = meta["ik"].to_numpy(dtype=object)
    rp, npth = O / "_ref.npz", O / "_ref_neg.npy"
    sparse.save_npz(rp, Xr, compressed=False); np.save(npth, rneg)
    for name, (q, t) in query_sets(a.sets.split(",")).items():
        t0 = time.time()
        q = q[q["adduct"].isin(v1.AD10)].reset_index(drop=True)
        tinfo = t.set_index("ik")
        # candidate pools per molecule (universe; C2 targets lose their train-only row)
        q["M"] = q["precursor_mz"] - q["adduct"].map(v1.AD10)
        Mmol = q.groupby("molecule_id")["M"].median()
        pools = {}
        for mid, M in Mmol.items():
            lo, hi = U.window(float(M), CFG["ppm"])
            r = np.arange(lo, hi)
            if tinfo.loc[mid, "qtype"] == "C2":
                keep = ~((usrc[r] == "train") & np.isin(U.ik[r], list(tinfo.loc[mid, "mask_all_keys"])))
                r = r[keep]
            pools[mid] = r
        # masks per query spectrum
        masks, cand_rows = [], []
        for r in q.itertuples():
            ti = tinfo.loc[r.molecule_id]
            if ti["qtype"] == "C2":
                m = [rows_of[k] for k in ti["mask_all_keys"] if k in rows_of]
            else:
                m = [rows_of_lib[(k, ti["qlib"])] for k in ti["mask_all_keys"] if (k, ti["qlib"]) in rows_of_lib]
            m += [h2rows[r.h]] if r.h in h2rows else []
            masks.append(np.unique(np.concatenate(m)) if m else np.zeros(0, int))
            cand_rows.append({ik: rows_of[ik] for ik in U.ik[pools[r.molecule_id]] if ik in rows_of})
        Xq = v1.featurize_df(q, CFG)
        qneg = q["adduct"].isin(v1.NEG).values
        qadd = q["adduct"].values
        B = 48
        jobs = [(Xq[i:i + B], qneg[i:i + B], qadd[i:i + B], masks[i:i + B], cand_rows[i:i + B], CFG["k"], radd)
                for i in range(0, len(q), B)]
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(str(rp), str(npth))) as pool:
            res = pool.map(_pass, jobs)
        idx = np.vstack([r[0] for r in res]); sims = np.vstack([r[1] for r in res])
        selfsim = [d for r in res for d in r[2]]
        # predicted fingerprints per spectrum
        preds = np.empty((len(q), v1.FP_BITS), np.float32)
        for s0 in range(0, len(q), 256):
            bits = np.unpackbits(U.fp[np.maximum(rrow[idx[s0:s0 + 256]], 0)], axis=2).astype(np.float32)
            w = np.clip(sims[s0:s0 + 256], 0, None) + 1e-6
            w = w * (rrow[idx[s0:s0 + 256]] >= 0)
            preds[s0:s0 + 256] = (w[:, :, None] * bits).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        nb_ik = IK[idx]
        rows_out = []
        for mid, g in q.groupby("molecule_id"):
            pr = pools[mid]
            if not len(pr):
                continue
            ii = g.index.values
            cb = U.bits(pr)
            cn = np.linalg.norm(cb, axis=1) + 1e-12
            P = preds[ii]
            per = (cb @ (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)).T) / cn[:, None]  # cand x spectra
            pm = P.mean(0)
            mean = (cb @ (pm / (np.linalg.norm(pm) + 1e-12))) / cn
            M = float(Mmol[mid])
            iks = U.ik[pr]
            vote = {}
            for j in ii:
                for n, s in zip(nb_ik[j], sims[j]):
                    vote[n] = vote.get(n, 0.0) + max(s, 0.0)
            ss = {}
            for j in ii:
                for ik, (mx, same, n) in selfsim[j].items():
                    o = ss.get(ik, (0.0, 0.0, 0))
                    ss[ik] = (max(o[0], mx), max(o[1], same), max(o[2], n))
            mass = U.mass[pr]
            fkey = np.round(mass, 4)
            _, inv, cnt = np.unique(fkey, return_inverse=True, return_counts=True)
            order = np.argsort(-mean, kind="stable")
            rank = np.empty(len(pr)); rank[order] = np.arange(len(pr))
            sd = mean.std() + 1e-6
            corr = tinfo.loc[mid, "correct"]
            for c in range(len(pr)):
                ik = iks[c]
                s3 = ss.get(ik, (0.0, 0.0, 0))
                rows_out.append((mid, ik, int(ik in corr), float(mean[c]), float(per[c].max()), float((mean[c] - mean.mean()) / sd),
                                 float(mean[c] - mean.max()), float(rank[c] / max(1, len(pr) - 1)), vote.get(ik, 0.0),
                                 s3[0], s3[1], s3[2], int(s3[2] > 0), int(usrc[pr[c]] == "coconut"), int(usrc[pr[c]] == "train"),
                                 float(abs(mass[c] - M) / M * 1e6), len(pr), int(cnt[inv[c]]), float(np.log(len(pr)))))
        df = pd.DataFrame(rows_out, columns=["mid", "ik", "label"] + FEATS)
        df.to_parquet(O / f"feats_{name}.parquet", index=False)
        cov = df.groupby("mid")["label"].max()
        log(f"{name}: {df['mid'].nunique()} molecules / {len(t)} targets, {len(df)} rows, truth-in-pool "
            f"{cov.mean():.3f}, {time.time() - t0:.0f}s")
    rp.unlink(); npth.unlink()


# ----------------------------------------------------------------------------- train / eval
def mrr(df, score):
    d = df.assign(s=score).sort_values(["mid", "s", "ik"], ascending=[True, False, True])
    d["r"] = d.groupby("mid").cumcount() + 1
    hit = d[d["label"] == 1].groupby("mid")["r"].min()
    rr = (1.0 / hit).where(hit <= 25, 0.0)
    return rr.reindex(d["mid"].unique()).fillna(0.0)


def stage_train(a):
    import lightgbm as lgb
    tr, va = pd.read_parquet(O / "feats_train.parquet"), pd.read_parquet(O / "feats_val.parquet")
    tr = tr[tr.groupby("mid")["label"].transform("max") == 1].sort_values("mid")  # learnable groups only
    va_l = va[va.groupby("mid")["label"].transform("max") == 1].sort_values("mid")
    feats = [f for f in FEATS if f not in a.drop.split(",")]
    dtr = lgb.Dataset(tr[feats], tr["label"], group=tr.groupby("mid", sort=True).size().values)
    dva = lgb.Dataset(va_l[feats], va_l["label"], group=va_l.groupby("mid", sort=True).size().values)
    params = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30,
                  seed=SEED, verbose=-1, num_threads=a.workers)
    bst = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    bst.save_model(str(O / f"ranker_{a.tag}.txt"))
    imp = dict(sorted(zip(feats, bst.feature_importance("gain").round(0)), key=lambda kv: -kv[1]))
    res = {"best_iter": bst.best_iteration, "features": feats, "gain": imp}
    for name in ("val", "s1", "s2", "s3"):
        p = O / f"feats_{name}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        n_all = {"val": a.n_val}.get(name)
        r_knn, r_lgb = mrr(d, d["knn_mean"].values), mrr(d, bst.predict(d[feats], num_iteration=bst.best_iteration))
        diff = (r_lgb - r_knn).values
        rng = np.random.default_rng(SEED)
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        res[name] = {"n_mol": int(len(r_knn)), "MRR_knn_V1": round(r_knn.mean(), 4), "MRR_ranker": round(r_lgb.mean(), 4),
                     "delta": [round(diff.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)]}
        if name == "val":
            T = pd.read_parquet(O / "targets.parquet").set_index("ik")
            for qt in ("C1", "C2", "c1x_tims"):
                ids = [m for m in r_knn.index if (T.loc[m, "pop"] if qt == "c1x_tims" else T.loc[m, "qtype"]) == qt]
                res[f"val_{qt}"] = {"n": len(ids), "MRR_knn_V1": round(r_knn[ids].mean(), 4), "MRR_ranker": round(r_lgb[ids].mean(), 4)}
    json.dump(res, open(O / f"train_{a.tag}.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "feats", "train"])
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-c1x-train", type=int, default=900)
    ap.add_argument("--n-c1x-val", type=int, default=300)
    ap.add_argument("--sets", default="train,val,s1,s2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--drop", default="")
    a = ap.parse_args()
    {"build": stage_build, "feats": stage_feats, "train": stage_train}[a.stage](a)

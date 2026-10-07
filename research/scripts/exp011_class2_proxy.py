"""EXP-011 Class-2 proxy (design: research/analysis/exp011_class2_proxy_design.md).

Stages (each writes to results/exp011_class2_proxy[_smoke]/ and can be rerun independently):
  build   targets, held-out set H (targets + parent/tautomer aliases), DB-only decoys D, training and query
          spectrum lists, manifest + leakage checks K1/K3/K5/K6. No scoring.
  feats   sparse binned spectrum features (training + queries), Morgan fingerprints, check K2.
  pools   candidate pools (3/5/10/20 ppm), baselines B0 chance, B1 mass error, B2 fingerprint prior.
  knn     B3 spectrum kNN fingerprint prediction.
  mlp     B4 learned spectrum -> fingerprint MLP.
  eval    metrics, paired bootstrap CIs, memorisation check, report json.

    python research/scripts/exp011_class2_proxy.py build [--smoke]   (then feats, pools, knn, mlp, eval)
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = (ROOT / "train.parquet").as_posix()
UNIVERSE = ROOT / "results" / "exp008_structure_universe.parquet"
SEED = 20260925
AD10 = {"[M+H]+": 1.007276, "[M+NH4]+": 18.033823, "[M-H2O+H]+": -17.003289, "[M-2H2O+H]+": -35.013854,
        "[M+Na]+": 22.989221, "[M+K]+": 38.963158, "[M-H]-": -1.007276, "[M-H2O-H]-": -19.017841,
        "[M+CH2O2-H]-": 44.998203, "[M+Cl]-": 34.969402}
ADDUCTS = sorted(AD10)
NEG = {"[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"}
BIN, MZ_MAX, NL_MAX, TOPK_PEAKS = 0.1, 1500.0, 500.0, 150
N_FRAG, N_NL = int(MZ_MAX / BIN), int(NL_MAX / BIN)
NFEAT = N_FRAG + N_NL
FP_BITS = 2048
PPMS = (3, 5, 10, 20)
PRIMARY_PPM = 5


def outdir(smoke):
    d = ROOT / "results" / ("exp011_class2_proxy_smoke" if smoke else "exp011_class2_proxy")
    d.mkdir(parents=True, exist_ok=True)
    return d


def con_():
    c = duckdb.connect()
    c.execute("SET threads=4; SET memory_limit='6GB'")
    return c


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- build
def stage_build(a, O):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    c = con_()
    meta = c.execute(f"""SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm, ingest_lib AS lib
                         FROM read_parquet('{TRAIN}', file_row_number=true)""").fetchdf()
    U = pd.read_parquet(UNIVERSE)
    U = U[U["parse_ok"]].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    m10 = meta[meta["adduct"].isin(AD10)]

    t1 = sorted(m10.loc[m10["lib"] == "enveda-np-examples", "ik"].unique())
    q1 = m10[(m10["lib"] == "enveda-np-examples")]
    e180 = m10[(m10["lib"] == "enveda-180") & ~m10["ik"].isin(t1)]
    pool_t2 = np.array(sorted(e180["ik"].unique()))
    t2 = sorted(rng.choice(pool_t2, 100 if a.smoke else 1000, replace=False).tolist())
    q2 = (e180[e180["ik"].isin(t2)].sample(frac=1.0, random_state=SEED)
          .groupby("ik").head(4))
    queries = pd.concat([q1.assign(pop="T1_np"), q2.assign(pop="T2_tims")], ignore_index=True)
    targets = pd.DataFrame({"ik": t1 + t2, "pop": ["T1_np"] * len(t1) + ["T2_tims"] * len(t2)})
    targets = targets[targets["ik"].isin(set(U["ik"]))].reset_index(drop=True)
    log(f"targets T1={len(t1)} T2={len(t2)}; query spectra {len(queries)}")

    # K5 alias search: parent group + tautomer-canonical InChIKey14 within +/-2 mDa of each target
    te = rdMolStandardize.TautomerEnumerator()
    lf, un = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
    Us = U.sort_values("mass").reset_index(drop=True)
    mass, smi_of, mass_of = Us["mass"].values, dict(zip(U["ik"], U["smiles"])), dict(zip(U["ik"], U["mass"]))
    parent_members = U.groupby("parent")["ik"].apply(set).to_dict()
    parent_of = dict(zip(U["ik"], U["parent"]))
    formula_of = dict(zip(U["ik"], U["formula"]))
    kcache = {}

    def tkeys(ik):
        if ik not in kcache:
            try:
                m = Chem.MolFromSmiles(smi_of[ik])
                k1 = Chem.MolToInchiKey(te.Canonicalize(m))[:14]
                k2 = Chem.MolToInchiKey(te.Canonicalize(un.uncharge(lf.choose(m))))[:14]
            except Exception:
                k1 = k2 = None
            kcache[ik] = (k1, k2)
        return kcache[ik]

    H, aliases = set(), {}
    for ik in targets["ik"]:
        grp = {ik} | parent_members.get(parent_of.get(ik), set())
        m0 = mass_of[ik]
        near = Us["ik"].values[np.searchsorted(mass, m0 - 0.002):np.searchsorted(mass, m0 + 0.002, side="right")]
        kt = tkeys(ik)
        # Tautomers share the molecular formula; salt/charge forms are covered by the parent group above
        # (and differ by >2 mDa anyway), so only same-formula neighbours need the costly canonicalisation.
        f0 = formula_of.get(ik)
        near = [n for n in near if n != ik and formula_of.get(n) == f0]
        al = {n for n in near if kt[0] and (tkeys(n)[0] == kt[0] or tkeys(n)[1] == kt[1])}
        if (grp | al) - {ik}:
            aliases[ik] = sorted((grp | al) - {ik})
        H |= grp | al
    others = np.array(sorted(set(U["ik"]) - H))
    D = set(rng.choice(others, int(0.10 * len(others)), replace=False).tolist())
    log(f"H={len(H)} (aliases for {len(aliases)} targets), D={len(D)}")

    tr = m10[~m10["ik"].isin(H | D)].copy()
    tr["_r"] = rng.random(len(tr))
    tr = tr.sort_values(["ik", "lib", "_r"]).groupby(["ik", "lib"]).head(3).drop(columns="_r")
    tr_mols = np.array(sorted(tr["ik"].unique()))
    val_mols = set(rng.choice(tr_mols, int(0.05 * len(tr_mols)), replace=False).tolist())
    tr["split"] = np.where(tr["ik"].isin(val_mols), "val", "train")
    if a.smoke:  # smoke: small training set
        tr = pd.concat([tr[tr.split == "train"].sample(50_000, random_state=SEED),
                        tr[tr.split == "val"].sample(5_000, random_state=SEED)])

    checks = {
        "K1_training_rows_in_H_or_D": int(tr["ik"].isin(H | D).sum()),
        "K3_targets_missing_from_U": int((~targets["ik"].isin(set(U["ik"]))).sum()),
        "K5_alias_search_done": True, "K5_targets_with_aliases": len(aliases),
        "K6_val_molecules_in_H": int(len(val_mols & H)),
        "K2a_query_rids_in_training": int(queries["rid"].isin(set(tr["rid"])).sum()),
        "query_molecules_not_targets": int((~queries["ik"].isin(set(targets["ik"]))).sum()),
    }
    queries.to_parquet(O / "queries.parquet", index=False)
    targets.to_parquet(O / "targets.parquet", index=False)
    tr.to_parquet(O / "training.parquet", index=False)
    pd.DataFrame({"ik": sorted(D)}).to_parquet(O / "db_only_decoys.parquet", index=False)
    man = {"seed": SEED, "smoke": a.smoke, "n_targets": targets["pop"].value_counts().to_dict(),
           "n_query_spectra": queries["pop"].value_counts().to_dict(), "H": len(H), "D": len(D),
           "aliases": aliases, "training_spectra": tr["split"].value_counts().to_dict(),
           "training_molecules": int(tr["ik"].nunique()), "checks": checks, "sec": round(time.time() - t0, 1)}
    json.dump(man, open(O / "manifest.json", "w"), indent=1)
    log(json.dumps({k: man[k] for k in ("n_targets", "n_query_spectra", "H", "D", "training_spectra", "checks")}))
    assert all(v == 0 for k, v in checks.items() if k.startswith(("K1", "K3", "K6", "K2a", "query")))


# ----------------------------------------------------------------------------- feats
def featurize(tbl):
    """Arrow table (rid, pm, adduct, ms2_mzs, ms2_normalized_intensities) -> CSR rows, L2-normalised."""
    import pyarrow.compute as pc
    from scipy import sparse
    n = tbl.num_rows
    lens = pc.list_value_length(tbl["ms2_mzs"]).fill_null(0).to_numpy()
    mz = pc.list_flatten(tbl["ms2_mzs"]).to_numpy(zero_copy_only=False)
    it = pc.list_flatten(tbl["ms2_normalized_intensities"]).to_numpy(zero_copy_only=False)
    row = np.repeat(np.arange(n), lens)
    pm = tbl["pm"].to_numpy()[row]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    row, mz, it, pm = row[ok], mz[ok], it[ok], pm[ok]
    order = np.lexsort((-it, row))  # per row by intensity desc
    row, mz, it, pm = row[order], mz[order], it[order], pm[order]
    starts = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    rank = np.arange(len(row)) - np.repeat(starts, np.diff(np.r_[starts, len(row)]))
    keep = rank < TOPK_PEAKS
    row, mz, it, pm = row[keep], mz[keep], it[keep], pm[keep]
    w = np.sqrt(it).astype(np.float32)
    fr = (mz / BIN).astype(np.int64)
    m1 = (fr >= 0) & (fr < N_FRAG)
    nl = pm - mz
    nb = (nl / BIN).astype(np.int64)
    m2 = (nl > 0.5) & (nb < N_NL)
    rows = np.r_[row[m1], row[m2]]
    cols = np.r_[fr[m1], N_FRAG + nb[m2]]
    vals = np.r_[w[m1], w[m2]]
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, NFEAT), dtype=np.float32)
    X.sum_duplicates()
    nrm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    nrm[nrm == 0] = 1.0
    return sparse.diags(1.0 / nrm).dot(X).tocsr().astype(np.float32)


def fetch_features(c, rids):
    import pyarrow as pa
    from scipy import sparse
    rids = np.sort(np.asarray(rids, dtype=np.int64))
    blocks, order, hashes = [], [], []
    for i in range(0, len(rids), 100_000):
        part = rids[i:i + 100_000]
        c.register("want", pd.DataFrame({"rid": part}))
        tbl = c.execute(f"""SELECT file_row_number AS rid, precursor_mz AS pm, ms2_mzs, ms2_normalized_intensities,
                                   hash(ms2_mzs, ms2_normalized_intensities) AS h
                            FROM read_parquet('{TRAIN}', file_row_number=true)
                            WHERE file_row_number IN (SELECT rid FROM want) ORDER BY rid""").arrow()
        if isinstance(tbl, pa.RecordBatchReader):
            tbl = tbl.read_all()
        c.unregister("want")
        blocks.append(featurize(tbl))
        order.append(tbl["rid"].to_numpy())
        hashes.append(tbl["h"].to_numpy())
        log(f"  features {min(i + 100_000, len(rids))}/{len(rids)}")
    got = np.concatenate(order)
    assert np.array_equal(got, rids)
    return sparse.vstack(blocks).tocsr(), rids, np.concatenate(hashes)


def morgan_packed(smiles_list):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = np.zeros((len(smiles_list), FP_BITS // 8), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        arr = np.zeros(FP_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), arr)
        out[i] = np.packbits(arr)
    return out


def stage_feats(a, O):
    from scipy import sparse
    t0 = time.time()
    c = con_()
    q = pd.read_parquet(O / "queries.parquet")
    tr = pd.read_parquet(O / "training.parquet")
    Xq, rq, hq = fetch_features(c, q["rid"].values)
    Xt, rt, ht = fetch_features(c, tr["rid"].values)
    sparse.save_npz(O / "X_query.npz", Xq); np.save(O / "rid_query.npy", rq)
    sparse.save_npz(O / "X_train.npz", Xt); np.save(O / "rid_train.npy", rt)
    k2b = int(np.isin(hq, ht).sum())  # K2b: byte-identical spectra between queries and training
    # fingerprints: training molecules + every universe molecule within 20 ppm of any target
    U = pd.read_parquet(UNIVERSE)
    U = U[U["parse_ok"]].sort_values("mass").reset_index(drop=True)
    t = pd.read_parquet(O / "targets.parquet")
    mass_of = dict(zip(U["ik"], U["mass"]))
    need = set(tr["ik"]) | set(t["ik"])
    qm = neutral_masses(q).groupby("ik")["M"].median()  # same molecule-level M as pools_for
    for M in qm.values:
        tol = M * 25e-6
        need |= set(U["ik"].values[np.searchsorted(U["mass"].values, M - tol):np.searchsorted(U["mass"].values, M + tol, side="right")])
    need = sorted(need)
    smi = dict(zip(U["ik"], U["smiles"]))
    fp = morgan_packed([smi[ik] for ik in need])
    np.save(O / "fp_packed.npy", fp)
    pd.DataFrame({"ik": need, "mass": [mass_of[k] for k in need]}).to_parquet(O / "fp_index.parquet", index=False)
    info = {"X_query": list(Xq.shape), "X_train": list(Xt.shape), "nnz_train": int(Xt.nnz),
            "K2b_query_spectra_byte_identical_in_training": k2b, "fingerprints": len(need), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(O / "feats.json", "w"), indent=1)
    log(info)
    assert k2b == 0, "query spectrum duplicated in training"


def neutral_masses(q):
    q = q.copy()
    q["M"] = q["pm"] - q["adduct"].map(AD10)
    return q


# ----------------------------------------------------------------------------- scoring helpers
class FP:
    def __init__(self, O):
        idx = pd.read_parquet(O / "fp_index.parquet")
        self.row = {k: i for i, k in enumerate(idx["ik"])}
        self.mass = idx["mass"].values
        self.ik = idx["ik"].values
        self.packed = np.load(O / "fp_packed.npy")

    def bits(self, iks):
        return np.unpackbits(self.packed[[self.row[k] for k in iks]], axis=1).astype(np.float32)


def pools_for(O):
    """Per target molecule: M (median over its spectra) and pools at each ppm (from fp_index = universe slice)."""
    q = neutral_masses(pd.read_parquet(O / "queries.parquet"))
    t = pd.read_parquet(O / "targets.parquet")
    idx = pd.read_parquet(O / "fp_index.parquet").sort_values("mass").reset_index(drop=True)
    Mmol = q.groupby("ik")["M"].median()
    out = {}
    for ik, pop in zip(t["ik"], t["pop"]):
        M = float(Mmol[ik])
        d = {"pop": pop, "M": M}
        for ppm in PPMS:
            tol = M * ppm * 1e-6
            lo, hi = np.searchsorted(idx["mass"].values, M - tol), np.searchsorted(idx["mass"].values, M + tol, side="right")
            d[ppm] = idx["ik"].values[lo:hi].tolist()
        out[ik] = d
    return out


def expected_rank_stats(scores: dict, target):
    """Tie-aware: returns (expected rank, expected RR, expected RR@25, P(rank<=k) for k in 1,10,50,100)."""
    if target not in scores:
        return None
    v = np.array(list(scores.values()), dtype=float)
    s = scores[target]
    better, tied = int(np.sum(v > s)), int(np.sum(v == s))
    ranks = np.arange(better + 1, better + tied + 1)
    return {"rank": float(ranks.mean()), "rr": float(np.mean(1.0 / ranks)),
            "rr25": float(np.mean(np.where(ranks <= 25, 1.0 / ranks, 0.0))),
            **{f"r@{k}": float(np.mean(ranks <= k)) for k in (1, 10, 50, 100)}}


def cosine_scores(pred, cand_bits):
    pn = pred / (np.linalg.norm(pred) + 1e-12)
    cn = cand_bits / (np.linalg.norm(cand_bits, axis=1, keepdims=True) + 1e-12)
    return cn @ pn


def loglik_scores(pred, cand_bits):
    p = np.clip(pred, 0.01, 0.99)
    return cand_bits @ np.log(p) + (1 - cand_bits) @ np.log(1 - p)


# ----------------------------------------------------------------------------- pools + B0/B1/B2
def stage_pools(a, O):
    t0 = time.time()
    P = pools_for(O)
    F = FP(O)
    tr = pd.read_parquet(O / "training.parquet")
    trm = pd.DataFrame({"ik": sorted(set(tr["ik"]))})
    trm["mass"] = [F.mass[F.row[k]] for k in trm["ik"]]
    trm = trm.sort_values("mass").reset_index(drop=True)
    res = {}
    for ik, d in P.items():
        pool = d[PRIMARY_PPM]
        r = {"pop": d["pop"], "M": d["M"], **{f"pool_{p}ppm": len(d[p]) for p in PPMS},
             **{f"in_pool_{p}ppm": ik in d[p] for p in PPMS}}
        if ik in pool:
            n = len(pool)
            ranks = np.arange(1, n + 1)
            r["B0"] = {"rank": (n + 1) / 2, "rr": float(np.mean(1 / ranks)), "rr25": float(np.mean(np.where(ranks <= 25, 1 / ranks, 0))),
                       **{f"r@{k}": min(k, n) / n for k in (1, 10, 50, 100)}}
            r["B1"] = expected_rank_stats({c: -abs(F.mass[F.row[c]] - d["M"]) for c in pool}, ik)
            lo, hi = np.searchsorted(trm["mass"].values, d["M"] - 5), np.searchsorted(trm["mass"].values, d["M"] + 5, side="right")
            prior = F.bits(trm["ik"].values[lo:hi]).mean(axis=0) if hi > lo else np.full(FP_BITS, 0.5, np.float32)
            cb = F.bits(pool)
            r["B2"] = expected_rank_stats(dict(zip(pool, cosine_scores(prior, cb))), ik)
            r["B2_ll"] = expected_rank_stats(dict(zip(pool, loglik_scores(prior, cb))), ik)
        res[ik] = r
    json.dump(res, open(O / "rank_B012.json", "w"))
    log(f"pools+B0-B2 done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- B3 kNN
def rank_with_predictions(O, pred_by_mol, tag):
    np.savez_compressed(O / f"pred_{tag}.npz", iks=np.array(list(pred_by_mol)), pred=np.stack(list(pred_by_mol.values())))
    P = pools_for(O)
    F = FP(O)
    D = set(pd.read_parquet(O / "db_only_decoys.parquet")["ik"])
    trset = set(pd.read_parquet(O / "training.parquet")["ik"])
    res, mem = {}, []
    for ik, d in P.items():
        pool = d[PRIMARY_PPM]
        if ik not in pool or ik not in pred_by_mol:
            continue
        cb = F.bits(pool)
        cs = cosine_scores(pred_by_mol[ik], cb)
        sc = dict(zip(pool, cs))
        top1 = max(sc, key=lambda k: (sc[k], k))
        tb, t1b = F.bits([ik])[0], F.bits([top1])[0]
        res[ik] = {tag: expected_rank_stats(sc, ik),
                   f"{tag}_ll": expected_rank_stats(dict(zip(pool, loglik_scores(pred_by_mol[ik], cb))), ik),
                   f"{tag}_tc_top1": float((tb * t1b).sum() / max(1.0, (np.maximum(tb, t1b)).sum()))}
        for c, s in sc.items():  # memorisation check: non-target candidates, in-training vs DB-only
            if c != ik and (c in D or c in trset):
                mem.append((c in trset, s - sc[ik]))
    m = pd.DataFrame(mem, columns=["in_training", "score_minus_target"])
    memo = m.groupby("in_training")["score_minus_target"].agg(["mean", "count"]).to_dict() if len(m) else {}
    return res, memo


def stage_knn(a, O, k=20):
    from scipy import sparse
    t0 = time.time()
    q = pd.read_parquet(O / "queries.parquet").set_index("rid")
    tr = pd.read_parquet(O / "training.parquet")
    tr = tr[tr["split"] == "train"].set_index("rid")
    Xq, rq = sparse.load_npz(O / "X_query.npz"), np.load(O / "rid_query.npy")
    Xt_all, rt_all = sparse.load_npz(O / "X_train.npz"), np.load(O / "rid_train.npy")
    keep = np.isin(rt_all, tr.index.values)
    Xt, rt = Xt_all[keep], rt_all[keep]
    t_neg = tr.loc[rt, "adduct"].isin(NEG).values
    F = FP(O)
    t_rows = np.array([F.row[k] for k in tr.loc[rt, "ik"].values])
    XtT = Xt.T.tocsr()
    preds = {}
    for i in range(0, Xq.shape[0], 256):
        S = (Xq[i:i + 256] @ XtT).toarray()
        for j in range(S.shape[0]):
            rid = rq[i + j]
            neg = q.loc[rid, "adduct"] in NEG
            s = np.where(t_neg == neg, S[j], -1.0)
            nn = np.argpartition(-s, k)[:k]
            w = np.clip(s[nn], 0, None) + 1e-6
            bits = np.unpackbits(F.packed[t_rows[nn]], axis=1).astype(np.float32)
            preds[rid] = (w[:, None] * bits).sum(0) / w.sum()
        log(f"  kNN {min(i + 256, Xq.shape[0])}/{Xq.shape[0]}")
    by_mol = {}
    for rid, p in preds.items():
        by_mol.setdefault(q.loc[rid, "ik"], []).append(p)
    by_mol = {k: np.mean(v, axis=0) for k, v in by_mol.items()}
    res, memo = rank_with_predictions(O, by_mol, "B3")
    json.dump({"ranks": res, "memorisation": memo, "sec": round(time.time() - t0, 1)}, open(O / "rank_B3.json", "w"))
    log(f"kNN done in {time.time() - t0:.0f}s; memorisation {memo}")


# ----------------------------------------------------------------------------- B4 MLP
def stage_mlp(a, O):
    import torch
    from scipy import sparse
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    t0 = time.time()
    tr = pd.read_parquet(O / "training.parquet").set_index("rid")
    Xt, rt = sparse.load_npz(O / "X_train.npz"), np.load(O / "rid_train.npy")
    F = FP(O)
    split = tr.loc[rt, "split"].values
    fp_row = np.array([F.row[k] for k in tr.loc[rt, "ik"].values])
    add_idx = np.array([ADDUCTS.index(x) for x in tr.loc[rt, "adduct"].values])

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.EmbeddingBag(NFEAT, 1024, mode="sum")
            self.add = torch.nn.Embedding(len(ADDUCTS), 1024)
            self.body = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 1024),
                                            torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(1024, FP_BITS))

        def forward(self, X, a):
            idx = torch.from_numpy(X.indices.astype(np.int64))
            off = torch.from_numpy(X.indptr[:-1].astype(np.int64))
            w = torch.from_numpy(X.data.astype(np.float32))
            return self.body(self.emb(idx, off, per_sample_weights=w) + self.add(a))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    tri, vai = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    rng = np.random.default_rng(SEED)
    epochs = 1 if a.smoke else 4
    best, hist = 1e9, []

    def batches(ix, bs=512):
        for s in range(0, len(ix), bs):
            b = ix[s:s + bs]
            yield Xt[b], torch.from_numpy(add_idx[b]), torch.from_numpy(np.unpackbits(F.packed[fp_row[b]], axis=1).astype(np.float32))

    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tri)
        tl, nb = 0.0, 0
        for Xb, ab, yb in batches(perm):
            opt.zero_grad()
            loss = lossf(net(Xb, ab), yb)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
            if nb % 200 == 0:
                log(f"  ep{ep} batch {nb} loss {tl / nb:.4f}")
        net.eval()
        with torch.no_grad():
            vl = np.mean([lossf(net(Xb, ab), yb).item() for Xb, ab, yb in batches(vai, 2048)])
        hist.append({"epoch": ep, "train_bce": tl / nb, "val_bce": float(vl), "sec": round(time.time() - t0, 1)})
        log(hist[-1])
        if vl < best:
            best = vl
            torch.save(net.state_dict(), O / "mlp.pt")
        else:
            break
    net.load_state_dict(torch.load(O / "mlp.pt"))
    net.eval()
    q = pd.read_parquet(O / "queries.parquet").set_index("rid")
    Xq, rq = sparse.load_npz(O / "X_query.npz"), np.load(O / "rid_query.npy")
    aq = torch.from_numpy(np.array([ADDUCTS.index(x) for x in q.loc[rq, "adduct"].values]))
    with torch.no_grad():
        pq = torch.sigmoid(net(Xq, aq)).numpy()
    by_mol = {}
    for rid, p in zip(rq, pq):
        by_mol.setdefault(q.loc[rid, "ik"], []).append(p)
    by_mol = {k: np.mean(v, axis=0) for k, v in by_mol.items()}
    res, memo = rank_with_predictions(O, by_mol, "B4")
    json.dump({"ranks": res, "memorisation": memo, "history": hist, "sec": round(time.time() - t0, 1)},
              open(O / "rank_B4.json", "w"))
    log(f"MLP done in {time.time() - t0:.0f}s; memorisation {memo}")


# ----------------------------------------------------------------------------- controls
def stage_controls(a, O):
    """Oracle (prediction = true fingerprint; plumbing positive control) and permutation (each molecule gets
    another molecule's prediction from the same population; must fall to ~chance) for every saved predictor."""
    P = pools_for(O)
    F = FP(O)
    rng = np.random.default_rng(SEED + 7)
    out = {}
    tgt = [ik for ik, d in P.items() if ik in d[PRIMARY_PPM]]
    oracle = [expected_rank_stats(dict(zip(P[ik][PRIMARY_PPM], cosine_scores(F.bits([ik])[0], F.bits(P[ik][PRIMARY_PPM])))), ik)
              for ik in tgt]
    out["oracle_true_fp"] = {"n": len(oracle), "mrr_cond": float(np.mean([r["rr"] for r in oracle])),
                             "r@1_cond": float(np.mean([r["r@1"] for r in oracle]))}
    for tag in ("B3", "B4"):
        f = O / f"pred_{tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        pred = dict(zip(z["iks"], z["pred"]))
        for pop in ("T1_np", "T2_tims"):
            mols = [ik for ik in tgt if P[ik]["pop"] == pop and ik in pred]
            if len(mols) < 3:
                continue
            perm = rng.permutation(len(mols))
            while np.any(perm == np.arange(len(mols))):
                perm = rng.permutation(len(mols))
            rr_real, rr_perm = [], []
            for i, ik in enumerate(mols):
                cb = F.bits(P[ik][PRIMARY_PPM])
                rr_real.append(expected_rank_stats(dict(zip(P[ik][PRIMARY_PPM], cosine_scores(pred[ik], cb))), ik)["rr"])
                rr_perm.append(expected_rank_stats(dict(zip(P[ik][PRIMARY_PPM], cosine_scores(pred[mols[perm[i]]], cb))), ik)["rr"])
            out[f"{tag}_{pop}"] = {"n": len(mols), "mrr_cond_real": float(np.mean(rr_real)),
                                   "mrr_cond_permuted": float(np.mean(rr_perm))}
    json.dump(out, open(O / "controls.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- eval
def stage_eval(a, O):
    base = json.load(open(O / "rank_B012.json"))
    extra = {}
    for f in ("rank_B3.json", "rank_B4.json"):
        if (O / f).exists():
            j = json.load(open(O / f))
            extra[f] = j
            for ik, r in j["ranks"].items():
                base[ik].update(r)
    rng = np.random.default_rng(20260926)
    rep = {}
    for pop in ("T1_np", "T2_tims"):
        mols = [ik for ik, r in base.items() if r["pop"] == pop]
        if not mols:
            continue
        R = {"n": len(mols)}
        for p in PPMS:
            R[f"pool_recall_{p}ppm"] = float(np.mean([base[m][f"in_pool_{p}ppm"] for m in mols]))
            R[f"pool_size_{p}ppm_p10_p50_p90"] = np.percentile([base[m][f"pool_{p}ppm"] for m in mols], [10, 50, 90]).tolist()
        inpool = [m for m in mols if base[m][f"in_pool_{PRIMARY_PPM}ppm"]]
        for b in ("B0", "B1", "B2", "B2_ll", "B3", "B3_ll", "B4", "B4_ll"):
            have = [m for m in inpool if b in base[m] and base[m][b]]
            if not have:
                continue
            def agg(key, cond):
                vals = [base[m][b][key] for m in have]
                return float(np.sum(vals) / (len(have) if cond else len(mols)))
            R[b] = {"n_ranked": len(have), **{f"{k}_uncond": agg(k, False) for k in ("rr", "rr25", "r@1", "r@10", "r@50", "r@100")},
                    **{f"{k}_cond": agg(k, True) for k in ("rr", "r@1", "r@10", "r@100")},
                    "median_rank_cond": float(np.median([base[m][b]["rank"] for m in have]))}
        for b in ("B3", "B4"):
            for ref in ("B0", "B2"):
                both = [m for m in inpool if base[m].get(b) and base[m].get(ref)]
                if both:
                    d = np.array([base[m][b]["rr"] - base[m][ref]["rr"] for m in both])
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                    R[f"delta_rr_{b}_minus_{ref}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            tc = [base[m][f"{b}_tc_top1"] for m in inpool if f"{b}_tc_top1" in base[m]]
            if tc:
                R[f"{b}_tanimoto_top1_to_truth_median"] = float(np.median(tc))
        rep[pop] = R
    rep["memorisation"] = {f: extra[f]["memorisation"] for f in extra}
    rep["runtime_sec"] = {f: extra[f]["sec"] for f in extra}
    rep["mlp_history"] = extra.get("rank_B4.json", {}).get("history")
    json.dump(rep, open(O / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "feats", "pools", "knn", "mlp", "controls", "eval"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    O = outdir(a.smoke)
    {"build": stage_build, "feats": stage_feats, "pools": stage_pools, "knn": stage_knn,
     "mlp": stage_mlp, "controls": stage_controls, "eval": stage_eval}[a.stage](a, O)

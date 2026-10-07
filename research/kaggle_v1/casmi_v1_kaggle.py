"""CASMI 2026 -- V1 hybrid pipeline: V0b library retrieval + COCONUT/train kNN fingerprint retrieval.

Branch L (Class 1, unchanged V0b science; functions imported from casmi_v0_kaggle):
  raw precursor_mz +/- 0.01 Da over all train spectra -> matchms ModifiedCosineGreedy(0.1, 0, 1)
  -> same-adduct candidates only -> per (spectrum, inchikey14) max -> SUM over the molecule's spectra.

Branch K (Class 2, EXP-011 B3 kNN with the EXP-015 representation):
  - references: train spectra in the 10 test adducts, enveda-180 excluded (EXP-013 R_noE180 finding),
    at most 3 spectra per (inchikey14, ingest_lib) (EXP-011 construction), fixed hash order.
  - features: top-150 peaks, fragment + neutral-loss bins, bin width / intensity transform from CFG,
    L2-normalised; cosine top-20 references of the same polarity.
  - per spectrum: similarity-weighted mean of the neighbours' Morgan r2/2048 fingerprints; per molecule: mean.
  - candidates: universe (COCONUT 2026-09 + train structures) within +/- 5 ppm of the molecule's neutral mass
    (median over its spectra of precursor_mz - adduct shift); score = cosine(predicted fp, candidate fp).

Fusion (CFG["tau"]): library candidates whose best same-adduct per-spectrum ModifiedCosine is >= tau go first,
in V0b order; the kNN list follows (duplicates skipped); top 25. tau = None skips branch L entirely (kNN only).
The local mixed proxy (research/kaggle_v1/proxy_eval.py) chose tau = None: any library gate or library score boost
lowered the Class-2 scenario far more than it raised the Class-1 scenario (research/kaggle_v1/README.md).

Usage (local):  python research/kaggle_v1/casmi_v1_kaggle.py --mode local
On Kaggle the notebook calls main(mode="kaggle").
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kaggle_v0"))  # local; on Kaggle both files sit in cwd
import casmi_v0_kaggle as v0  # noqa: E402

AD10 = {"[M+H]+": 1.007276, "[M+NH4]+": 18.033823, "[M-H2O+H]+": -17.003289, "[M-2H2O+H]+": -35.013854,
        "[M+Na]+": 22.989221, "[M+K]+": 38.963158, "[M-H]-": -1.007276, "[M-H2O-H]-": -19.017841,
        "[M+CH2O2-H]-": 44.998203, "[M+Cl]-": 34.969402}
NEG = {"[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"}
FP_BITS = 2048
TOP_N = 25
PLACEHOLDER_SMILES = "C"

CFG = {"bin_w": 0.01, "nl": True, "inten": "sqrt", "topk": 150, "mz_max": 1500.0, "nl_max": 500.0,
       "k": 20, "ppm": 5.0, "ref_cap": 3, "ref_exclude_libs": ("enveda-180",), "tau": None}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ----------------------------------------------------------------------------- inputs
def find_assets(mode, root=None):
    if mode == "local":
        root = root or Path(__file__).resolve().parents[2]
        return Path(root) / "results" / "kaggle_v1_assets"
    hits = sorted(glob.glob("/kaggle/input/**/universe_fp.npy", recursive=True))
    if len(hits) != 1:
        raise RuntimeError(f"casmi-v1-assets dataset not attached / ambiguous: {hits}")
    return Path(hits[0]).parent


class Universe:
    def __init__(self, d: Path, drop_train_iks=()):
        U = pd.read_parquet(d / "universe.parquet")
        fp = np.load(d / "universe_fp.npy")
        if len(drop_train_iks):  # LOCAL PROXY ONLY: remove held-out structures unless COCONUT has them
            keep = ~((U["src"] == "train") & U["ik"].isin(set(drop_train_iks))).values
            U, fp = U[keep].reset_index(drop=True), fp[keep]
        self.ik, self.smiles, self.mass, self.fp = U["ik"].values, U["smiles"].values, U["mass"].values, fp
        self.row = {k: i for i, k in enumerate(self.ik)}
        assert np.all(np.diff(self.mass) >= 0)

    def window(self, M, ppm):
        tol = M * ppm * 1e-6
        return np.searchsorted(self.mass, M - tol), np.searchsorted(self.mass, M + tol, side="right")

    def bits(self, rows):
        return np.unpackbits(self.fp[rows], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- features
def featurize(lens, mz, it, pm, cfg=CFG):
    """EXP-015 featurize (generalised EXP-011) on flat arrays. lens[i] peaks of row i; pm per row."""
    from scipy import sparse
    n = len(lens)
    n_frag, n_nl = int(cfg["mz_max"] / cfg["bin_w"]), int(cfg["nl_max"] / cfg["bin_w"])
    row = np.repeat(np.arange(n), lens)
    pmr = np.asarray(pm, dtype=np.float64)[row]
    mz, it = np.asarray(mz, dtype=np.float64), np.asarray(it, dtype=np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    row, mz, it, pmr = row[ok], mz[ok], it[ok], pmr[ok]
    order = np.lexsort((-it, row))
    row, mz, it, pmr = row[order], mz[order], it[order], pmr[order]
    st = np.r_[0, np.flatnonzero(np.diff(row)) + 1] if len(row) else np.array([], int)
    rank = np.arange(len(row)) - np.repeat(st, np.diff(np.r_[st, len(row)])) if len(row) else np.array([], int)
    keep = rank < cfg["topk"]
    row, mz, it, pmr = row[keep], mz[keep], it[keep], pmr[keep]
    w = {"sqrt": np.sqrt(it), "linear": it, "log": np.log1p(1000.0 * it)}[cfg["inten"]].astype(np.float32)
    fr = (mz / cfg["bin_w"]).astype(np.int64)
    m1 = (fr >= 0) & (fr < n_frag)
    rows, cols, vals = [row[m1]], [fr[m1]], [w[m1]]
    if cfg["nl"]:
        nlv = pmr - mz
        nb = (nlv / cfg["bin_w"]).astype(np.int64)
        m2 = (nlv > 0.5) & (nb < n_nl)
        rows.append(row[m2]); cols.append(n_frag + nb[m2]); vals.append(w[m2])
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n_frag + (n_nl if cfg["nl"] else 0)), dtype=np.float32)
    X.sum_duplicates()
    nrm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    nrm[nrm == 0] = 1.0
    return sparse.diags(1.0 / nrm).dot(X).tocsr().astype(np.float32)


def featurize_arrow(tbl, cfg=CFG):
    import pyarrow.compute as pc
    lens = pc.list_value_length(tbl["ms2_mzs"]).fill_null(0).to_numpy()
    mz = pc.list_flatten(tbl["ms2_mzs"]).to_numpy(zero_copy_only=False)
    it = pc.list_flatten(tbl["ms2_normalized_intensities"]).to_numpy(zero_copy_only=False)
    return featurize(lens, mz, it, tbl["pm"].to_numpy(), cfg)


def featurize_df(df, cfg=CFG):
    mzs = [np.asarray(x if x is not None else [], dtype=float) for x in df["ms2_mzs"]]
    its = [np.asarray(x if x is not None else [], dtype=float) for x in df["ms2_normalized_intensities"]]
    lens = np.array([min(len(a), len(b)) for a, b in zip(mzs, its)])
    mz = np.concatenate([a[:n] for a, n in zip(mzs, lens)]) if len(lens) else np.array([])
    it = np.concatenate([b[:n] for b, n in zip(its, lens)]) if len(lens) else np.array([])
    return featurize(lens, mz, it, df["precursor_mz"].values, cfg)


def load_reference(train, work, cfg=CFG, exclude_sql="", seed=20260930):
    """Reference spectra (ik, adduct, features). exclude_sql: LOCAL PROXY ONLY."""
    t0 = time.time()
    libs = ",".join(f"'{x}'" for x in cfg["ref_exclude_libs"]) or "''"
    ads = ",".join(f"'{a}'" for a in AD10)
    con = v0.connect(mem=os.environ.get("CASMI_DUCKDB_MEM", f"{max(2, int(v0.available_ram_gb() * 0.4))}GB"),
                     threads=4, tmp=(Path(work) / "duck_tmp").as_posix())
    tr = Path(train).as_posix()
    tbl = con.execute(f"""
        SELECT ik, adduct, pm, ms2_mzs, ms2_normalized_intensities FROM (
            SELECT inchikey14 AS ik, adduct, precursor_mz AS pm, ms2_mzs, ms2_normalized_intensities,
                   row_number() OVER (PARTITION BY inchikey14, ingest_lib
                                      ORDER BY hash(file_row_number + {seed}), file_row_number) AS rn
            FROM read_parquet('{tr}', file_row_number=true)
            WHERE adduct IN ({ads}) AND ingest_lib NOT IN ({libs}) AND inchikey14 IS NOT NULL {exclude_sql})
        WHERE rn <= {int(cfg['ref_cap'])} ORDER BY ik, adduct, pm""").arrow()
    con.close()
    if not hasattr(tbl, "column_names"):
        tbl = tbl.read_all()
    X = featurize_arrow(tbl, cfg)
    ik = tbl["ik"].to_numpy(zero_copy_only=False)
    neg = np.isin(tbl["adduct"].to_numpy(zero_copy_only=False), list(NEG))
    log(f"reference: {X.shape[0]} spectra, {len(set(ik))} molecules, nnz {X.nnz} ({time.time() - t0:.0f}s)")
    return X, ik, neg


# ----------------------------------------------------------------------------- neighbours
_W = {}


def _init(ref_path, neg_path):
    from scipy import sparse
    _W["RT"] = sparse.load_npz(ref_path).T.tocsr()
    _W["neg"] = np.load(neg_path)


def _topk(args):
    Xq, qneg, k = args
    S = (Xq @ _W["RT"]).toarray()
    S[qneg[:, None] != _W["neg"][None, :]] = -1.0
    k = min(k, S.shape[1] - 1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, axis=1)
    o = np.argsort(-sims, axis=1, kind="stable")
    return np.take_along_axis(idx, o, axis=1), np.take_along_axis(sims, o, axis=1)


def neighbours(Xq, qneg, Xr, rneg, k, work, workers):
    from scipy import sparse
    rp, npth = Path(work) / "_ref.npz", Path(work) / "_ref_neg.npy"
    sparse.save_npz(rp, Xr, compressed=False)
    np.save(npth, rneg)
    jobs = [(Xq[i:i + 64], qneg[i:i + 64], k) for i in range(0, Xq.shape[0], 64)]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init, initargs=(str(rp), str(npth))) as pool:
            res = pool.map(_topk, jobs)
    else:
        _init(str(rp), str(npth))
        res = [_topk(j) for j in jobs]
    rp.unlink(); npth.unlink()
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def knn_scores(test, U: Universe, ref, work, workers, cfg=CFG):
    """Per molecule_id: dict inchikey14 -> cosine(predicted fp, candidate fp) over the +/-ppm universe window."""
    t0 = time.time()
    Xr, rik, rneg = ref
    rrow = np.array([U.row.get(k, -1) for k in rik])
    ok_ref = rrow >= 0  # reference molecules without a universe fingerprint cannot vote
    Xr, rrow, rneg = Xr[ok_ref], rrow[ok_ref], rneg[ok_ref]
    q = test[test["adduct"].isin(AD10)].reset_index(drop=True)
    Xq = featurize_df(q, cfg)
    has_peaks = np.diff(Xq.indptr) > 0
    qneg = q["adduct"].isin(NEG).values
    idx, sims = neighbours(Xq, qneg, Xr, rneg, cfg["k"], work, workers)
    preds = np.empty((len(q), FP_BITS), np.float32)
    for s0 in range(0, len(q), 256):
        bits = np.unpackbits(U.fp[rrow[idx[s0:s0 + 256]]], axis=2).astype(np.float32)
        w = np.clip(sims[s0:s0 + 256], 0, None) + 1e-6
        preds[s0:s0 + 256] = (w[:, :, None] * bits).sum(1) / w.sum(1, keepdims=True)
    q["M"] = q["precursor_mz"] - q["adduct"].map(AD10)
    out, info = {}, {"spectra_used": int(has_peaks.sum()), "spectra_non_ad10": int(len(test) - len(q)),
                     "reference_without_fp": int((~ok_ref).sum()), "empty_pool_molecules": []}
    for mid, g in q.groupby("molecule_id"):
        rows = g.index.values[has_peaks[g.index.values]]
        if not len(rows):
            continue
        M = float(np.median(g["M"].values))
        lo, hi = U.window(M, cfg["ppm"])
        if hi <= lo:
            info["empty_pool_molecules"].append(mid)
            continue
        p = preds[rows].mean(axis=0)
        cb = U.bits(np.arange(lo, hi))
        cs = (cb @ (p / (np.linalg.norm(p) + 1e-12))) / (np.linalg.norm(cb, axis=1) + 1e-12)
        out[mid] = dict(zip(U.ik[lo:hi], cs.astype(float)))
    info.update(molecules_scored=len(out), sec=round(time.time() - t0, 1),
                median_pool=float(np.median([len(v) for v in out.values()])) if out else 0.0)
    log(f"kNN branch: {info['molecules_scored']} molecules, median pool {info['median_pool']} ({info['sec']}s)")
    return out, info


# ----------------------------------------------------------------------------- library branch (V0b)
def library_scores(scores: pd.DataFrame):
    """V0b: same-adduct rows; per (molecule, spectrum, ik) max; molecule score = SUM over spectra (order), and
    MAX over spectra (the confidence used by the fusion gate)."""
    s = scores[scores["query_adduct"] == scores["cand_adduct"]]
    per = s.groupby(["molecule_id", "spectrum_id", "inchikey14"], as_index=False)["score"].max()
    mol = per.groupby(["molecule_id", "inchikey14"])["score"].agg(["sum", "max"]).reset_index()
    out = {}
    for mid, g in mol.groupby("molecule_id"):
        g = g.sort_values(["sum", "inchikey14"], ascending=[False, True])
        out[mid] = list(zip(g["inchikey14"], g["sum"], g["max"]))
    return out


def fuse(lib_list, knn_dict, tau):
    """lib_list: [(ik, sum, max)] in V0b order; knn_dict: ik -> cosine. Returns top-25 inchikey14."""
    head = [ik for ik, _, mx in (lib_list or []) if mx >= tau]
    tail = sorted((knn_dict or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    out, seen = [], set()
    for ik in head + [k for k, _ in tail]:
        if ik not in seen:
            out.append(ik); seen.add(ik)
        if len(out) == TOP_N:
            break
    return out


# ----------------------------------------------------------------------------- checks
def validate(sub, expected_ids):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    parts = sub["smiles"].fillna("").map(lambda s: s.split(";") if s else [])
    invalid = sum(Chem.MolFromSmiles(p) is None for ps in parts for p in ps)
    rep = {"rows": len(sub), "ids_equal": sub["molecule_id"].tolist() == list(expected_ids),
           "no_dup_ids": sub["molecule_id"].is_unique, "min_preds": int(parts.map(len).min()),
           "max_preds": int(parts.map(len).max()), "empty_fields": int(sum(p == "" for ps in parts for p in ps)),
           "invalid_smiles": int(invalid), "schema": list(sub.columns) == ["molecule_id", "smiles"],
           "any_nan": bool(sub.isna().any().any())}
    rep["fatal_ok"] = (rep["ids_equal"] and rep["no_dup_ids"] and rep["schema"] and not rep["any_nan"]
                       and 1 <= rep["min_preds"] and rep["max_preds"] <= TOP_N and rep["empty_fields"] == 0)
    return rep


# ----------------------------------------------------------------------------- main
def run_branches(P, test, work, workers, cfg=CFG, lib_exclude_sql="", ref_exclude_sql="", drop_train_iks=(), assets=None,
                 use_library=True):
    """Both branches on one test frame. The exclusion arguments are LOCAL PROXY ONLY (empty on Kaggle)."""
    rep, lib = {}, {}
    if use_library:
        pool_path = Path(work) / "pool.parquet"
        rep["pool"] = v0.build_pool(P["train"], P["test"], pool_path, Path(work), lib_exclude_sql)
        log(f"library pool: {rep['pool']}")
        scores, timing, sec = v0.score_all(test, pool_path, workers)
        rep["library"] = {"score_sec": sec, "score_rows": len(scores), "candidate_pairs": int(timing["n_candidates"].sum())}
        lib = library_scores(scores)
        pool_path.unlink(missing_ok=True)
    U = Universe(assets, drop_train_iks)
    ref = load_reference(P["train"], work, cfg, ref_exclude_sql)
    rep["reference"] = {"spectra": int(ref[0].shape[0]), "molecules": int(len(set(ref[1])))}
    knn, rep["knn"] = knn_scores(test, U, ref, work, workers, cfg)
    return lib, knn, U, rep


def main(mode="kaggle", workers=None, root=None, cfg=CFG):
    T0 = time.time()
    P = v0.find_inputs(mode, Path(root) if root else None)
    work = Path(P["work"]).parent / "casmi_v1" if mode == "kaggle" else Path(P["work"]).parent / "kaggle_v1_local"
    work.mkdir(parents=True, exist_ok=True)
    out_csv = Path(P.get("out_csv") or work / "submission.csv")
    workers = workers or v0.n_workers()
    report = {"mode": mode, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "workers": workers}

    test = v0.load_test(P["test"])
    need = {"molecule_id", "spectrum_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"}
    assert need <= set(test.columns), f"test schema missing {need - set(test.columns)}"
    test = test[test["molecule_id"].notna()].reset_index(drop=True)
    test["molecule_id"] = test["molecule_id"].astype(str)
    if test["spectrum_id"].duplicated().any():
        test["spectrum_id"] = test["spectrum_id"].astype(str) + "#" + test.groupby("spectrum_id").cumcount().astype(str)
    test_ids = sorted(test["molecule_id"].unique())
    sample_ids = pd.read_csv(P["sample"])["molecule_id"].astype(str).tolist() if Path(P["sample"]).exists() else []
    expected_ids = sample_ids if set(sample_ids) == set(test_ids) and len(sample_ids) == len(test_ids) else test_ids
    report["test"] = {"spectra": len(test), "molecules": len(test_ids), "adducts": test["adduct"].value_counts().to_dict()}
    log(json.dumps(report["test"]))

    lib, knn, U, rep = run_branches(P, test, work, workers, cfg, assets=find_assets(mode, root),
                                   use_library=cfg["tau"] is not None)
    report.update(rep)

    tau = cfg["tau"] if cfg["tau"] is not None else float("inf")
    ranked = {m: fuse(lib.get(m), knn.get(m), tau) for m in expected_ids}
    all_iks = {ik for lst in ranked.values() for ik in lst}
    ik2smi = {ik: U.smiles[U.row[ik]] for ik in all_iks if ik in U.row}
    missing = all_iks - set(ik2smi)
    if missing:  # library hits whose structure failed RDKit parsing in the universe build: train SMILES
        ik2smi.update(v0.smiles_for(P["train"], missing))

    def row(m):
        preds = [ik2smi[ik] for ik in ranked[m] if ik in ik2smi]
        return ";".join(preds) if preds else PLACEHOLDER_SMILES
    sub = pd.DataFrame({"molecule_id": expected_ids, "smiles": [row(m) for m in expected_ids]})
    n_lib_head = [sum(1 for ik, _, mx in (lib.get(m) or []) if mx >= tau) for m in expected_ids]
    report["fusion"] = {"tau": cfg["tau"], "molecules_with_library_head": int(sum(n > 0 for n in n_lib_head)),
                        "molecules_with_knn": int(sum(m in knn for m in expected_ids)),
                        "placeholder_rows": int((sub["smiles"] == PLACEHOLDER_SMILES).sum()),
                        "iks_without_smiles": sorted(all_iks - set(ik2smi))}
    report["validation"] = validate(sub, expected_ids)
    log(f"fusion {report['fusion']} validation {report['validation']}")
    if not report["validation"]["fatal_ok"]:
        json.dump(report, open(work / "v1_run_report.json", "w"), indent=1, default=str)
        raise RuntimeError(f"submission would be rejected by the metric: {report['validation']}")
    sub.to_csv(out_csv, index=False)
    back = pd.read_csv(out_csv, keep_default_na=False)
    assert list(back.columns) == ["molecule_id", "smiles"] and (back["smiles"] != "").all()
    report["output"] = {"path": str(out_csv), "sha256_lf": v0.sha256_lf(out_csv)}
    report["runtime_sec_total"] = round(time.time() - T0, 1)
    json.dump(report, open(work / "v1_run_report.json", "w"), indent=1, default=str)
    log(json.dumps({k: report[k] for k in ("fusion", "output", "runtime_sec_total")}, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "kaggle"], default="local")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.workers)

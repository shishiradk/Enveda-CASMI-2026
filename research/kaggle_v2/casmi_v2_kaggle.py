"""CASMI 2026 -- V2: V1 kNN fingerprint retrieval + PubChem candidates (EXP-017).

V1 (research/kaggle_v1/casmi_v1_kaggle.py) is reused unchanged for features, references and neighbours.
New in V2:
  - the kNN prediction step is split out (predict_molecules) so candidates can come from several sources;
  - PubChem candidates: rows of pubchem_rows.parquet (build: research/kaggle_v1/build_pubchem.py join2; one row per CID,
    sorted by mass) inside the molecule's +/-ppm neutral-mass window, merged per InChIKey14 (representative = lowest CID,
    n_cid = merged CIDs), InChIKey14 not already in the V1 universe; Morgan r2/2048 computed on the fly for window rows;
  - score = cosine(predicted fp, candidate fp) + CFG["tier_bonus"] for COCONUT / train structures (0 for PubChem-only).
    tier_bonus is chosen on the local mixed proxy (research/kaggle_v2/proxy_eval2.py), never on the leaderboard.

Usage (local):  python research/kaggle_v2/casmi_v2_kaggle.py --mode local
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kaggle_v1"))  # local; on Kaggle all sources sit in cwd
import casmi_v1_kaggle as v1  # noqa: E402

v0 = v1.v0
CFG = {**v1.CFG, "pubchem": True, "tier_bonus": 0.05, "pc_max_window": 20000}


def log(m):
    v1.log(m)


def find_pubchem(mode, root=None):
    if mode == "local":
        return Path(root or HERE.parents[1]) / "external" / "pubchem" / "pubchem_rows.parquet"
    hits = sorted(glob.glob("/kaggle/input/**/pubchem_rows.parquet", recursive=True))
    if len(hits) != 1:
        raise RuntimeError(f"PubChem candidate dataset not attached / ambiguous: {hits}")
    return Path(hits[0])


# ----------------------------------------------------------------------------- kNN prediction (split from v1.knn_scores)
def predict_molecules(test, U, ref, work, workers, cfg=CFG):
    """molecule_id -> (neutral mass M, predicted fingerprint). Identical arithmetic to v1.knn_scores."""
    t0 = time.time()
    Xr, rik, rneg = ref
    rrow = np.array([U.row.get(k, -1) for k in rik])
    ok_ref = rrow >= 0
    Xr, rrow, rneg = Xr[ok_ref], rrow[ok_ref], rneg[ok_ref]
    q = test[test["adduct"].isin(v1.AD10)].reset_index(drop=True)
    n_nonfinite = int((~np.isfinite(q["precursor_mz"].astype(float))).sum())
    q = q[np.isfinite(q["precursor_mz"].astype(float))].reset_index(drop=True)  # review fix: NaN precursor poisons M
    info = {"spectra_non_ad10": int(len(test) - len(q) - n_nonfinite), "spectra_nonfinite_precursor": n_nonfinite,
            "reference_without_fp": int((~ok_ref).sum())}
    if not len(q):  # review fix: no usable spectrum at all -> no predictions instead of a crash
        log("predicted 0 molecules (no spectrum in the 10 adducts with a finite precursor)")
        return {}, info
    Xq = v1.featurize_df(q, cfg)
    has_peaks = np.diff(Xq.indptr) > 0
    idx, sims = v1.neighbours(Xq, q["adduct"].isin(v1.NEG).values, Xr, rneg, cfg["k"], work, workers)
    preds = np.empty((len(q), v1.FP_BITS), np.float32)
    for s0 in range(0, len(q), 256):
        bits = np.unpackbits(U.fp[rrow[idx[s0:s0 + 256]]], axis=2).astype(np.float32)
        w = np.clip(sims[s0:s0 + 256], 0, None) + 1e-6
        preds[s0:s0 + 256] = (w[:, :, None] * bits).sum(1) / w.sum(1, keepdims=True)
    q["M"] = q["precursor_mz"] - q["adduct"].map(v1.AD10)
    out = {}
    for mid, g in q.groupby("molecule_id"):
        rows = g.index.values[has_peaks[g.index.values]]
        if len(rows):
            out[mid] = (float(np.median(g["M"].values)), preds[rows].mean(axis=0))
    log(f"predicted {len(out)} molecules ({time.time() - t0:.0f}s)")
    return out, info


# ----------------------------------------------------------------------------- PubChem windows
def _fp_batch(smiles):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=v1.FP_BITS)
    out = np.zeros((len(smiles), v1.FP_BITS // 8), np.uint8)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        arr = np.zeros(v1.FP_BITS, np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        out[i], ok[i] = np.packbits(arr), True
    return out, ok


def pubchem_windows(pc_path, masses: dict, U, cfg=CFG, workers=4, drop_iks=()):
    """molecule_id -> DataFrame(ik, smiles, mass, n_cid, fp_row) of PubChem-only candidates; plus packed fps.
    drop_iks: LOCAL PROXY ONLY (unused on Kaggle)."""
    import duckdb
    t0 = time.time()
    win = pd.DataFrame([(m, M - M * cfg["ppm"] * 1e-6, M + M * cfg["ppm"] * 1e-6) for m, M in masses.items()],
                       columns=["molecule_id", "lo", "hi"])
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='6GB'")
    con.register("win", win)
    rows = con.execute(f"""
        SELECT w.molecule_id, p.ik, arg_min(p.smiles, p.cid) AS smiles, arg_min(p.mass, p.cid) AS mass, count(*) AS n_cid
        FROM win w JOIN read_parquet('{Path(pc_path).as_posix()}') p ON p.mass BETWEEN w.lo AND w.hi
        GROUP BY w.molecule_id, p.ik""").df()
    con.close()
    # Drop a PubChem row only when its universe twin is itself inside this window (it is then already a candidate).
    # Review fix: ~270 universe rows carry a protonated or isotope-labelled mass, so their twin can sit outside the window.
    if len(rows):
        Mw = rows["molecule_id"].map(masses).values
        urow = rows["ik"].map(U.row)
        has_u = urow.notna().values
        umass = np.where(has_u, U.mass[urow.fillna(0).astype(int).values], np.nan)
        in_win = has_u & (np.abs(umass - Mw) <= Mw * cfg["ppm"] * 1e-6)
        rows = rows[~in_win]
    if len(drop_iks):
        rows = rows[~rows["ik"].isin(set(drop_iks))]
    capped = []
    parts = []
    for mid, g in rows.groupby("molecule_id"):  # guard against pathological windows (keep the most-registered)
        if len(g) > cfg["pc_max_window"]:
            capped.append(mid)
            g = g.sort_values(["n_cid", "ik"], ascending=[False, True]).head(cfg["pc_max_window"])
        parts.append(g)
    rows = pd.concat(parts, ignore_index=True) if parts else rows
    uniq = rows.drop_duplicates("ik")[["ik", "smiles"]].reset_index(drop=True)
    chunks = [uniq["smiles"].values[i:i + 5000].tolist() for i in range(0, len(uniq), 5000)]
    if workers > 1 and len(chunks) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            res = pool.map(_fp_batch, chunks)
    else:
        res = [_fp_batch(c) for c in chunks]
    fp = np.concatenate([r[0] for r in res]) if res else np.zeros((0, v1.FP_BITS // 8), np.uint8)
    ok = np.concatenate([r[1] for r in res]) if res else np.zeros(0, bool)
    fprow = dict(zip(uniq["ik"], range(len(uniq))))
    rows["fp_row"] = rows["ik"].map(fprow)
    rows = rows[ok[rows["fp_row"].values]] if len(rows) else rows
    info = {"pubchem_rows": int(len(rows)), "unique_iks": int(len(uniq)), "fp_fail": int((~ok).sum()),
            "median_window": float(rows.groupby("molecule_id").size().median()) if len(rows) else 0.0,
            "capped_molecules": capped, "sec": round(time.time() - t0, 1)}
    log(f"PubChem windows: {info}")
    return {mid: g for mid, g in rows.groupby("molecule_id")}, fp, info


def score_molecules(pred, U, pc_win, pc_fp, cfg=CFG):
    """molecule_id -> list of (ik, score, src) sorted best first (all candidates, not truncated)."""
    out = {}
    for mid, (M, p) in pred.items():
        pn = p / (np.linalg.norm(p) + 1e-12)
        lo, hi = U.window(M, cfg["ppm"])
        cand = []
        if hi > lo:
            cb = U.bits(np.arange(lo, hi))
            cs = (cb @ pn) / (np.linalg.norm(cb, axis=1) + 1e-12)
            cand += [(ik, float(s) + cfg["tier_bonus"], "u") for ik, s in zip(U.ik[lo:hi], cs)]
        g = pc_win.get(mid)
        if g is not None and len(g):
            cb = np.unpackbits(pc_fp[g["fp_row"].values], axis=1).astype(np.float32)
            cs = (cb @ pn) / (np.linalg.norm(cb, axis=1) + 1e-12)
            cand += [(ik, float(s), "p") for ik, s in zip(g["ik"].values, cs)]
        out[mid] = sorted(cand, key=lambda t: (-t[1], t[0]))
    return out


def run_v2(P, test, work, workers, cfg=CFG, ref_exclude_sql="", drop_train_iks=(), drop_pc_iks=(), assets=None,
           pc_path=None):
    U = v1.Universe(assets, drop_train_iks)
    ref = v1.load_reference(P["train"], work, cfg, ref_exclude_sql)
    pred, info = predict_molecules(test, U, ref, work, workers, cfg)
    pc_win, pc_fp, info["pubchem"] = ({}, None, {}) if not cfg["pubchem"] else \
        pubchem_windows(pc_path, {m: M for m, (M, _) in pred.items()}, U, cfg, workers, drop_pc_iks)
    return pred, U, pc_win, pc_fp, info


# ----------------------------------------------------------------------------- main
def main(mode="kaggle", workers=None, root=None, cfg=CFG):
    T0 = time.time()
    P = v0.find_inputs(mode, Path(root) if root else None)
    work = Path(P["work"]).parent / ("casmi_v2" if mode == "kaggle" else "kaggle_v2_local")
    work.mkdir(parents=True, exist_ok=True)
    out_csv = Path(P.get("out_csv") or work / "submission.csv")
    workers = workers or v0.n_workers()
    report = {"mode": mode, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}
    test = v0.load_test(P["test"])
    test = test[test["molecule_id"].notna()].reset_index(drop=True)
    test["molecule_id"] = test["molecule_id"].astype(str)
    if test["spectrum_id"].duplicated().any():
        test["spectrum_id"] = test["spectrum_id"].astype(str) + "#" + test.groupby("spectrum_id").cumcount().astype(str)
    test_ids = sorted(test["molecule_id"].unique())
    sample_ids = pd.read_csv(P["sample"])["molecule_id"].astype(str).tolist() if Path(P["sample"]).exists() else []
    expected_ids = sample_ids if set(sample_ids) == set(test_ids) and len(sample_ids) == len(test_ids) else test_ids
    report["test"] = {"spectra": len(test), "molecules": len(test_ids)}

    pred, U, pc_win, pc_fp, report["predict"] = run_v2(P, test, work, workers, cfg, assets=v1.find_assets(mode, root),
                                                       pc_path=find_pubchem(mode, root) if cfg["pubchem"] else None)
    scored = score_molecules(pred, U, pc_win, pc_fp, cfg)
    smi = {}
    for mid, g in pc_win.items():
        smi.update(zip(g["ik"].values, g["smiles"].values))
    rows, src_top1 = [], {"u": 0, "p": 0}
    for m in expected_ids:
        lst = scored.get(m, [])[:v1.TOP_N]
        preds = [U.smiles[U.row[ik]] if src == "u" else smi[ik] for ik, _, src in lst]
        if lst:
            src_top1[lst[0][2]] += 1
        rows.append(";".join(preds) if preds else v1.PLACEHOLDER_SMILES)
    sub = pd.DataFrame({"molecule_id": expected_ids, "smiles": rows})
    report["fusion"] = {"top1_source": src_top1, "placeholder_rows": int((sub["smiles"] == v1.PLACEHOLDER_SMILES).sum())}
    report["validation"] = v1.validate(sub, expected_ids)
    log(f"{report['fusion']} validation {report['validation']}")
    if not report["validation"]["fatal_ok"]:
        json.dump(report, open(work / "v2_run_report.json", "w"), indent=1, default=str)
        raise RuntimeError(f"submission would be rejected by the metric: {report['validation']}")
    sub.to_csv(out_csv, index=False)
    back = pd.read_csv(out_csv, keep_default_na=False)
    assert list(back.columns) == ["molecule_id", "smiles"] and (back["smiles"] != "").all()
    report["output"] = {"path": str(out_csv), "sha256_lf": v0.sha256_lf(out_csv)}
    report["runtime_sec_total"] = round(time.time() - T0, 1)
    json.dump(report, open(work / "v2_run_report.json", "w"), indent=1, default=str)
    log(json.dumps({k: report[k] for k in ("fusion", "output", "runtime_sec_total")}, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "kaggle"], default="local")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.workers)

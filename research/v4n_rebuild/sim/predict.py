"""Inference: test.parquet -> v4r engine rows (same path as `simulate.py --test`) -> ranker_v0 (mean of its boosters)
-> top-25 unique by score key -> submission.csv (molecule_id, smiles joined by ';').  Our code, MIT.

    python research/v4n_rebuild/sim/predict.py --test test.parquet --ckpt models/cft_hoR_a/out_cft/export/cft_hoR_a.pt \
        --ranker results/v4n/sim/hoR_RA/ranker_v0.pkl --out submission.csv [--workers 4] [--limit N] [--key-check 200]
        [--truth results/bench/truth_SV.parquet] [--compare research/kaggle_e1/e7/v9_output/submission.csv]

Query spectra are cleaned like the simulation's library queries (V5b, sim/README.md): spectra.clean(.., 0.001, 512, 2.0).
Lists with < 25 unique keys are padded with the pool rows nearest in mass to the target (counted in the log).
--key-check N: recompute the score key of N random pool rows with the installed RDKit and compare with the table keys.
--truth: diagnostic MRR@25 / top-1 (on the visible test this is LEAKY: the truths and their spectra are in the library).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from common import TABLES, TEST
from simulate import test_queries

K = 25
CLEAN = (0.001, 512, 2.0)            # floor, topk, over: = library cleaning (build_tables) and v5.cleaned_test_npeaks
W: dict = {}


def _init(tables, ckpts, ranker, threads, gen):
    os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    import torch
    torch.set_num_threads(threads)
    import lightgbm as lgb
    from engine.engine import Engine, EngineCfg, FEATURES
    from engine.model import CFTBank
    from engine.tables import Library, Pool
    L, P = Library(tables, verbose=False), Pool(tables, verbose=False)
    E = Engine(L, P, CFTBank(ckpts, threads=threads), EngineCfg(generate=gen), verbose=False)
    mdl = pickle.load(open(ranker, "rb"))
    cols = [FEATURES.index(f) for f in mdl["features"]]
    boosters = [lgb.Booster(model_str=b) for b in mdl["boosters"]]
    for b in boosters:
        b.params["num_threads"] = threads
    W.update(E=E, P=P, cols=cols, boosters=boosters)


def clean_spectra(spectra):
    from engine import spectra as sp
    out = []
    for s in spectra:
        mz, it = sp.clean(np.asarray(s["mz"], np.float64), np.asarray(s["it"], np.float64), float(s["prec"]), *CLEAN)
        if len(mz):
            out.append(dict(s, mz=mz, it=it))
    return out or spectra             # never drop a molecule because every spectrum cleaned to nothing


def _pad(P, target, have, n):
    """Pool rows nearest in mass to target whose key is not yet listed."""
    if not np.isfinite(target):
        target = float(np.median(P.mass))
    c = int(np.searchsorted(P.mass, target))
    lo, hi, out = c - 1, c, []
    while len(out) < n and (lo >= 0 or hi < len(P.mass)):
        if hi >= len(P.mass) or (lo >= 0 and target - P.mass[lo] <= P.mass[hi] - target):
            i, lo = lo, lo - 1
        else:
            i, hi = hi, hi + 1
        k, s = P.key[i], P.smiles[i]
        if s and k not in have:
            have.add(k)
            out.append((s, k))
    return out


def _predict(job):
    q, spectra = job
    E, P, cols, boosters = W["E"], W["P"], W["cols"], W["boosters"]
    t0 = time.time()
    err = ""
    try:
        C, X, info = E.run(spectra, q["target"])
        n = len(C.pid)
        if n:
            Xr = np.ascontiguousarray(X[:, cols], dtype=np.float32)
            score = np.mean([b.predict(Xr) for b in boosters], 0)
            order = np.argsort(-score, kind="stable")
        else:
            order = np.zeros(0, np.int64)
        lib_max, n_gen = float(info["lib_max"]), int(info["n_gen"])
    except Exception as e:                          # recorded and padded, never a crash
        C, order, n, lib_max, n_gen, err = None, np.zeros(0, np.int64), 0, np.nan, 0, repr(e)[:300]
    picks, have = [], set()
    for i in order:
        k, s = C.key[i], C.smiles[i]
        if s and k not in have:
            have.add(k)
            picks.append((s, k))
            if len(picks) == K:
                break
    n_eng = len(picks)
    if len(picks) < K:
        picks += _pad(P, q["target"], have, K - len(picks))
    return dict(molecule_id=q["key"], smiles=";".join(s for s, _ in picks), keys=[k for _, k in picks],
                n_cand=int(n), n_engine=n_eng, lib_max=lib_max, n_gen=n_gen, sec=round(time.time() - t0, 3),
                n_spec=len(spectra), error=err)


def key_check(tables, n, seed=0):
    from engine import chem
    from engine.tables import Pool
    P = Pool(tables, verbose=False)
    idx = np.random.default_rng(seed).choice(len(P.mass), n, replace=False)
    bad = [(int(i), P.key[i], chem.score_key(P.smiles[i])) for i in idx if chem.score_key(P.smiles[i]) != P.key[i]]
    return dict(checked=int(n), mismatches=len(bad), examples=bad[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(TEST))
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--ranker", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--key-check", type=int, default=0)
    ap.add_argument("--truth", default=None, help="diagnostic: parquet (mid, smiles)")
    ap.add_argument("--compare", default=None, help="diagnostic: another submission.csv, top-1 agreement by key")
    a = ap.parse_args()
    T0 = time.time()
    rep = dict(test=a.test, ckpt=a.ckpt, ranker=a.ranker, clean=not a.no_clean)
    import rdkit
    rep["rdkit"] = rdkit.__version__
    if a.key_check:
        rep["key_check"] = key_check(a.tables, a.key_check)
        print("key check", rep["key_check"], flush=True)
    import simulate
    simulate.TEST = Path(a.test)                    # test_queries reads the module-level TEST
    Q, M = test_queries()
    if a.limit:
        Q = Q[:a.limit]
    jobs = [(q, M[q["key"]] if a.no_clean else clean_spectra(M[q["key"]])) for q in Q]
    print(f"{len(Q)} molecules, {sum(len(s) for _, s in jobs)} spectra", flush=True)
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    initargs = (a.tables, a.ckpt, a.ranker, a.threads, not a.no_generate)
    t1 = time.time()
    if a.workers <= 1:
        _init(*initargs)
        it = map(_predict, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=initargs)
        it = pool.imap(_predict, jobs, chunksize=1)
    res = []
    for j, r in enumerate(it):
        res.append(r)
        if (j + 1) % 25 == 0 or j + 1 == len(jobs):
            el = time.time() - t1
            print(f"{time.strftime('%H:%M:%S')} {j + 1}/{len(jobs)} {el / (j + 1):.2f} s/mol wall", flush=True)
    if a.workers > 1:
        pool.close(); pool.join()
    R = pd.DataFrame(res)
    R[["molecule_id", "smiles"]].to_csv(a.out, index=False)
    R.drop(columns=["smiles"]).to_parquet(Path(a.out).with_suffix(".diag.parquet"), index=False)
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    nsm = R.smiles.str.split(";").map(len)
    nuniq = R.smiles.str.split(";").map(lambda x: len(set(x)))
    nvalid = R.smiles.str.split(";").map(lambda x: sum(Chem.MolFromSmiles(s) is not None for s in x))
    rep.update(molecules=int(len(R)), rows_25=int((nsm == K).sum()), rows_25_unique=int((nuniq == K).sum()),
               rows_25_valid=int((nvalid == K).sum()), rows_padded=int((R.n_engine < K).sum()),
               errors=int((R.error != "").sum()), sec_load_and_run=round(time.time() - t1, 1),
               sec_total=round(time.time() - T0, 1), wall_s_per_mol=round((time.time() - t1) / max(len(R), 1), 3),
               worker_s_per_mol=dict(mean=round(float(R.sec.mean()), 2), median=round(float(R.sec.median()), 2),
                                     p90=round(float(R.sec.quantile(0.9)), 2), max=round(float(R.sec.max()), 1)),
               n_cand_median=float(R.n_cand.median()))
    from engine import chem
    if a.truth:
        tt = pd.read_parquet(a.truth)
        tk = {m: chem.score_key(s) for m, s in zip(tt.mid, tt.smiles)}
        rr = []
        for m, ks in zip(R.molecule_id, R["keys"]):
            p = [i for i, k in enumerate(ks) if k == tk.get(m)]
            rr.append(1.0 / (p[0] + 1) if p else 0.0)
        rr = np.array(rr)
        rep["visible_LEAKY"] = dict(mrr25=round(float(rr.mean()), 4), top1=round(float((rr == 1).mean()), 4),
                                    in_top25=round(float((rr > 0).mean()), 4),
                                    note="leaky: visible-test truths and their exact spectra are in the library")
    if a.compare:
        o = pd.read_csv(a.compare).set_index("molecule_id").smiles
        agree = [chem.score_key(o[m].split(";")[0]) == ks[0] for m, ks in zip(R.molecule_id, R["keys"]) if m in o.index]
        rep["top1_agree_with_compare"] = dict(n=len(agree), agree=int(sum(agree)),
                                              share=round(float(np.mean(agree)), 4) if agree else None)
    json.dump(rep, open(Path(a.out).with_suffix(".report.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()

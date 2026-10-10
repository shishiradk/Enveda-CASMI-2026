"""Simulation driver (REBUILD_SPEC 4): run the v4r engine on the simulated queries with the regime masks and write
labelled candidate rows as parquet shards.  Resumable, process-parallel, memory-bounded.  Our code, MIT.

    python research/v4n_rebuild/sim/simulate.py --run pilot_ho2 --workers 4                       # cft_ho2 bank
    python research/v4n_rebuild/sim/simulate.py --run hoR_RA --workers 4 --ckpt models/<R-A>.pt   # full run
    python research/v4n_rebuild/sim/simulate.py --run test400 --test --workers 4                  # visible test, unmasked
    python research/v4n_rebuild/sim/simulate.py --run testSV --test --workers 4            --exclude-rows results/bench/sv_dup_rows.npy --truth results/bench/truth_SV.parquet       # V5b: SV protocol
    options: --chunk 20 (queries per shard) --limit N (first N queries) --shard-of i/n (split the chunk list over
             machines / Kaggle sessions: this process does chunks c with c % n == i) --threads 1 (torch/numba per worker)

Inputs: results/v4n/sim/<run>/queries.parquet (build_queries.py), the engine tables (results/v4n/tables), the bank
checkpoint(s).  Output, per chunk c (written atomically, skipped if present => resume):
  rows/rows_<c>.parquet    one row per candidate: qid, regime, key (truth), fold, cand_key, cand_pid (-1 = generated),
                           cand_src (0 train / 1 COCONUT / -1 generated), y (cand key == truth key), f_<86 features>
  qstats/q_<c>.parquet     one row per query: qid, regime, key, fold, n_query, q_npeaks, lib_max, top_sim, n_cand, n_gen,
                           has_pos, fz_rank (rank of the truth by f.z, 0-based, -1 absent), sec
In --test mode key = molecule_id; y = -1 unless --truth is given.
Workers: spawn processes; each loads Library/Pool (memory-mapped) + the bank once.  Set --threads 1 when the PC is
shared; numba and torch threads per worker are pinned before import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from common import CKPT_HO2, OUT_ROOT, TABLES, TEST, peak_rss_mb, private_mb

W: dict = {}


def _init(tables, ckpts, threads, gen):
    os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    import torch
    torch.set_num_threads(threads)
    from engine.engine import Engine, EngineCfg
    from engine.model import CFTBank
    from engine.tables import Library, Pool
    L, P = Library(tables, verbose=False), Pool(tables, verbose=False)
    _slim(L, P)
    E = Engine(L, P, CFTBank(ckpts, threads=threads), EngineCfg(generate=gen), verbose=False)
    W.update(E=E, L=L, P=P, ex=np.zeros(len(L.sid), bool))


class _Coded:
    """Read-only string column stored as int32 codes + uniques (saves ~100 B per row of Python str objects)."""

    def __init__(self, arr):
        self.codes, self.uniq = pd.factorize(pd.Series(arr), use_na_sentinel=False)
        self.uniq = np.asarray(self.uniq, dtype=object)

    def __getitem__(self, i):
        return self.uniq[self.codes[i]]

    def __len__(self):
        return len(self.codes)


class _Blank:
    def __getitem__(self, i):
        return ""


def _slim(L, P):
    """Drop / compress the object columns the simulation never reads (per-worker private RAM)."""
    L.adduct = _Coded(L.adduct)
    L.instrument_type = _Coded(L.instrument_type)
    L.struct_ik = None
    P.smiles_tc = None
    P.smiles = _Blank()               # candidate SMILES are not needed for rows (keys are)
    import gc
    gc.collect()


def _query_spectra(L, idx):
    return [L.query_spectrum(int(i)) for i in idx]


def _run_one(q, test_spectra=None):
    """One query -> (rows DataFrame, qstat dict)."""
    from engine.engine import FEATURES
    E, L, P, ex = W["E"], W["L"], W["P"], W["ex"]
    t0 = time.time()
    if test_spectra is not None:
        spectra, target, truth = test_spectra, q["target"], q.get("truth")
        ex_rows = q.get("exclude_rows")
        if ex_rows is not None and len(ex_rows):
            ex[ex_rows] = True
        try:
            C, X, info = E.run(spectra, target, exclude=ex if ex_rows is not None and len(ex_rows) else None)
        finally:
            if ex_rows is not None and len(ex_rows):
                ex[ex_rows] = False
    else:
        spectra = _query_spectra(L, q["spec_idx"])
        target, truth = float(q["target"]), q["key"]
        mi = np.asarray(q["mask_idx"], np.int64)
        ex[mi] = True
        try:
            drop = P.pid_of_key(truth) if q["drop"] else -1        # by key: robust to a pool re-assemble
            C, X, info = E.run(spectra, target, exclude=ex, exclude_sid=int(q["sid0"]),
                               exclude_lib=int(q["exclude_lib"]), drop_pid=int(drop))
        finally:
            ex[mi] = False
    sec = time.time() - t0
    n = len(C.pid)
    keys = np.asarray(C.key, dtype=object)
    y = (keys == truth).astype(np.int8) if truth is not None else np.full(n, -1, np.int8)
    pid = C.pid.astype(np.int64)
    src = np.where(pid >= 0, P.src[np.maximum(pid, 0)], -1).astype(np.int8)
    df = pd.DataFrame(X, columns=["f_" + f for f in FEATURES])
    head = pd.DataFrame(dict(qid=np.full(n, q["qid"], np.int64), regime=q["regime"], key=q["key"], fold=q["fold"],
                             cand_key=keys.astype(str), cand_pid=pid, cand_src=src, y=y))
    rows = pd.concat([head, df], axis=1)
    fz = X[:, FEATURES.index("fz")] if n else np.zeros(0)
    fz_rank = -1
    if truth is not None and y.any():
        o = np.argsort(-fz, kind="stable")
        fz_rank = int(np.flatnonzero(y[o] > 0)[0])
    qs = dict(qid=int(q["qid"]), regime=q["regime"], key=q["key"], fold=q["fold"], n_query=len(spectra),
              q_npeaks=float(np.mean([len(s["mz"]) for s in spectra])) if spectra else 0.0,
              lib_max=float(info["lib_max"]), top_sim=float(info["top_sim"]), n_cand=int(n), n_gen=int(info["n_gen"]),
              has_pos=int(bool(y.any()) if truth is not None else -1), fz_rank=fz_rank, sec=round(sec, 3),
              n_pos=sum(1 for s in spectra if s["mode"] > 0))
    return rows, qs


def _run_chunk(job):
    c, Q, out, test_map = job
    t0 = time.time()
    R, S = [], []
    for q in Q:
        try:
            r, s = _run_one(q, test_map.get(q["key"]) if test_map else None)
        except Exception as e:                 # a failing query is recorded, never silently dropped
            print(f"[chunk {c}] qid {q['qid']} FAILED: {e!r}", flush=True)
            s = dict(qid=int(q["qid"]), regime=q["regime"], key=q["key"], fold=q["fold"], n_cand=0, has_pos=0,
                     fz_rank=-1, sec=0.0, error=repr(e)[:300])
            r = None
        if r is not None and len(r):
            R.append(r)
        S.append(s)
    rows = pd.concat(R, ignore_index=True) if R else pd.DataFrame()
    for sub, name, df in (("rows", f"rows_{c:05d}.parquet", rows), ("qstats", f"q_{c:05d}.parquet", pd.DataFrame(S))):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, d / name)               # atomic: a chunk is either complete or absent
    return c, len(Q), len(rows), time.time() - t0, peak_rss_mb(), private_mb()


def test_queries(exclude_rows=None, truth=None):
    """Visible test molecules.  exclude_rows: .npy of library rows to hide for every query (the SV protocol masks the
    exact enveda-180 duplicates of the test spectra, results/bench/sv_dup_rows.npy; spec_meta row = train.parquet row).
    truth: parquet (mid, smiles) -> rows labelled by the score key of the truth SMILES."""
    from engine import chem
    from engine.engine import spectra_from_rows, target_of
    t = pd.read_parquet(TEST)
    ex = np.load(exclude_rows).astype(np.int64) if exclude_rows else None
    tk = {}
    if truth:
        tt = pd.read_parquet(truth)
        tk = {m: chem.score_key(s_) for m, s_ in zip(tt.mid, tt.smiles)}
    Q, M = [], {}
    for i, (mid, rows) in enumerate(t.groupby("molecule_id", sort=True)):
        M[mid] = spectra_from_rows(rows)
        Q.append(dict(qid=i, regime="test", key=mid, fold="test", target=target_of(rows), exclude_rows=ex,
                      truth=tk.get(mid)))
    return Q, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", action="store_true", help="run the visible test molecules unmasked (V5b)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-of", default="0/1")
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--ckpt", nargs="+", default=[str(CKPT_HO2)])
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--exclude-rows", default=None, help="--test: .npy of library rows hidden for every query")
    ap.add_argument("--truth", default=None, help="--test: truth parquet (mid, smiles) to label rows")
    a = ap.parse_args()
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    test_map = None
    if a.test:
        Q, test_map = test_queries(a.exclude_rows, a.truth)
    else:
        Q = pd.read_parquet(out / "queries.parquet").to_dict("records")
    if a.limit:
        Q = Q[:a.limit]
    si, sn = map(int, a.shard_of.split("/"))
    chunks = [(c, Q[c * a.chunk:(c + 1) * a.chunk]) for c in range((len(Q) + a.chunk - 1) // a.chunk)]
    todo = [(c, q) for c, q in chunks if c % sn == si and not (out / "rows" / f"rows_{c:05d}.parquet").exists()
            and not (out / "qstats" / f"q_{c:05d}.parquet").exists()]
    cfg = dict(run=a.run, ckpt=a.ckpt, tables=a.tables, workers=a.workers, threads=a.threads, chunk=a.chunk,
               queries=len(Q), chunks=len(chunks), todo=len(todo), shard_of=a.shard_of, generate=not a.no_generate,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(cfg, open(out / f"simulate_cfg_{si}of{sn}.json", "w"), indent=1)
    print(json.dumps(cfg), flush=True)
    if not todo:
        return
    os.environ["NUMBA_NUM_THREADS"] = str(a.threads)      # inherited by spawned workers before numba import
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    jobs = [(c, q, out, {q_["key"]: test_map[q_["key"]] for q_ in q} if test_map else None) for c, q in todo]
    T0 = time.time()
    done_q = 0
    log = open(out / "simulate.log", "a")
    initargs = (a.tables, a.ckpt, a.threads, not a.no_generate)
    if a.workers <= 1:
        _init(*initargs)
        it = map(_run_chunk, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=initargs)
        it = pool.imap_unordered(_run_chunk, jobs)
    for c, nq, nr, dt, rss, priv in it:
        done_q += nq
        el = time.time() - T0
        msg = (f"{time.strftime('%H:%M:%S')} chunk {c} q {nq} rows {nr} {dt:.0f}s | done {done_q}/"
               f"{sum(len(q) for _, q in todo)} q, {el / done_q:.2f} s/q wall, peakRSS {rss:.0f} MB private {priv:.0f} MB")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    if a.workers > 1:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()

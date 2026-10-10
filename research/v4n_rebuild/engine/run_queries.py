"""Run the v4r engine on a parquet of query spectra (test.parquet column layout; one molecule = one molecule_id).

    python research/v4n_rebuild/engine/run_queries.py --queries test.parquet --out results/v4n/runs/test_x.pkl
           [--ckpt models/cft_ho2_akriti/cft_ho2.pt] [--limit 20] [--no-generate] [--threads 4]

Writes a pickle: {molecule_id: dict(pid, key, smiles, formula, X (float32 [n, 86]), features, info, sec)} plus a
small JSON summary next to it.  Candidates are in engine order (pool window order, then generated); a ranker
re-orders them later.  For bench parquets the `fz`-only top-25 hit rate against a truth table can be printed with
--truth results/bench/truth.parquet --qset S12.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=str(ROOT / "results" / "v4n" / "tables"))
    ap.add_argument("--ckpt", nargs="+", default=[str(ROOT / "models" / "cft_ho2_akriti" / "cft_ho2.pt")])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--qset", default=None)
    a = ap.parse_args()
    from engine import chem
    from engine.engine import FEATURES, Engine, EngineCfg, spectra_from_rows, target_of
    from engine.model import CFTBank
    from engine.tables import Library, Pool
    L, P = Library(a.tables), Pool(a.tables)
    E = Engine(L, P, CFTBank(a.ckpt, threads=a.threads), EngineCfg(generate=not a.no_generate))
    q = pd.read_parquet(a.queries)
    mids = list(dict.fromkeys(q.molecule_id))
    if a.limit:
        mids = mids[:a.limit]
    truth = None
    if a.truth:
        t = pd.read_parquet(a.truth)
        t = t[t.qset == a.qset] if a.qset else t
        truth = {m: chem.score_key(s) for m, s in zip(t.mid, t.smiles)}
    res, rr = {}, []
    T0 = time.time()
    for i, mid in enumerate(mids):
        rows = q[q.molecule_id == mid]
        t0 = time.time()
        C, X, info = E.run(spectra_from_rows(rows), target_of(rows))
        dt = time.time() - t0
        res[mid] = dict(pid=C.pid, key=C.key, smiles=C.smiles, formula=C.formula, X=X, info=info, sec=dt)
        if truth is not None and mid in truth:
            o = np.argsort(-X[:, FEATURES.index("fz")], kind="stable") if len(C.pid) else []
            ks = list(dict.fromkeys(C.key[j] for j in o))[:25]
            rr.append(1.0 / (ks.index(truth[mid]) + 1) if truth[mid] in ks else 0.0)
        if i % 10 == 0:
            print(f"{i + 1}/{len(mids)} {mid} n_cand {len(C.pid)} gen {info['n_gen']} {dt:.1f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(features=FEATURES, results=res), open(a.out, "wb"))
    secs = [r["sec"] for r in res.values()]
    summ = dict(molecules=len(res), sec_mean=float(np.mean(secs)), sec_median=float(np.median(secs)),
                sec_total=round(time.time() - T0), n_cand_median=float(np.median([len(r["pid"]) for r in res.values()])))
    if rr:
        summ["fz_only_mrr25"] = float(np.mean(rr))
    json.dump(summ, open(str(a.out) + ".json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

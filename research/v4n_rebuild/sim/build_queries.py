"""Build the simulated query table (REBUILD_SPEC 4) for the v4r ranker simulation.  Our code, MIT.

    # pilot: ho2 truths only (the only keys cft_ho2 never saw), targets <= 470 Da (pool COCONUT coverage)
    python research/v4n_rebuild/sim/build_queries.py --run pilot_ho2 --truths ho2 --max-mass 470
    # full run (after R-A exists and the pool tail is assembled): every HO_R key
    python research/v4n_rebuild/sim/build_queries.py --run hoR_RA --truths hoR

One truth = one score key (tautomer sids together).  Per truth and regime one query:
  query library l = enveda-180 if the key has (non-empty) spectra there, else the library holding most of them;
  query spectra   = a random subset of the key's spectra in l with |precursor error| <= 10 ppm (test adducts preferred), size drawn from the visible
                    test distribution of spectra per molecule (1-9, median 3); target = median neutral mass;
  c1  mask every spectrum of every key sid in l, hide the analog representatives of (sid, l); eligible only if the key
      has spectra in another library;
  c2  mask every spectrum of every key sid (all libraries);
  c3  as c2 and drop the truth's pool row (generation may still produce it).
Closure rule (leakage): a key is used only if EVERY sid sharing it is in the model's hold-out set (ho2 list or HO_R).
Output: results/v4n/sim/<run>/queries.parquet (+ build_queries.json).  Query order is shuffled (seeded) so any prefix
of chunks is a random sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from common import HO2_FILES, LIBS, OUT_ROOT, SPLIT, TABLES, TEST, REGIMES


def test_count_dist():
    t = pd.read_parquet(TEST, columns=["molecule_id", "adduct"])
    n = t.groupby("molecule_id").size().value_counts().sort_index()
    return n.index.values.astype(int), (n.values / n.values.sum()), sorted(t.adduct.unique().tolist())


def seed_of(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--truths", choices=["ho2", "hoR"], required=True)
    ap.add_argument("--max-mass", type=float, default=0.0, help="truth structure mass cap (0 = none)")
    ap.add_argument("--regimes", default="c1,c2,c3")
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prec-ppm", type=float, default=10.0,
                    help="query spectra must have |precursor_error_ppm| <= this (enveda-180, = the test instrument, "
                         "is 100%% within 10 ppm); falls back to all spectra of the key if none qualify")
    ap.add_argument("--limit", type=int, default=0, help="max truth keys (random), 0 = all")
    a = ap.parse_args()
    t0 = time.time()
    from engine import chem
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    regimes = [r for r in a.regimes.split(",") if r in REGIMES]

    st = pd.read_parquet(f"{a.tables}/train_structs.parquet", columns=["sid", "inchikey14", "key", "mass", "pid"])
    sp = pd.read_parquet(SPLIT, columns=["ik", "in_hoR", "fold", "stratum", "primary_lib"])
    assert (sp.ik.values == st.inchikey14.values).all(), "split not in train_structs sid order"
    pool = pd.read_parquet(f"{a.tables}/pool_meta.parquet", columns=["key", "train_sid", "src"])
    in_hoR = sp.in_hoR.values.astype(bool)
    if a.truths == "ho2":
        held = set(pd.concat([pd.read_parquet(f, columns=["ik"]) for f in HO2_FILES]).ik)
        hold = st.inchikey14.isin(held).values            # what the model never saw (by inchikey14)
        cand = hold & in_hoR
    else:
        hold = in_hoR
        cand = in_hoR
    rep = dict(truths=a.truths, sids_candidate=int(cand.sum()))

    # ---- truth keys: in the pool, mass cap, closure over every sid sharing the key
    key = st.key.values.astype(object)
    df = pd.DataFrame(dict(sid=st.sid.values, key=key, hold=hold, cand=cand, mass=st.mass.values, pid=st.pid.values))
    g = df.groupby("key").agg(all_hold=("hold", "all"), any_cand=("cand", "any"), n_sid=("sid", "size"),
                              pid=("pid", "max"))
    keys = g[g.any_cand & (g.pid >= 0)]
    rep["keys_candidate_in_pool"] = int(len(keys))
    rep["keys_dropped_closure"] = int((~keys.all_hold).sum())
    keys = keys[keys.all_hold]
    pid = keys.pid.values.astype(np.int64)
    sid0 = pool.train_sid.values[pid]                       # the sid that owns the pool row
    mass0 = st.mass.values[sid0]
    if a.max_mass > 0:
        keep = mass0 <= a.max_mass
        rep["keys_dropped_mass"] = int((~keep).sum())
        keys, pid, sid0, mass0 = keys[keep], pid[keep], sid0[keep], mass0[keep]
    if a.limit:
        r = np.random.default_rng(a.seed).choice(len(keys), size=min(a.limit, len(keys)), replace=False)
        r.sort()
        keys, pid, sid0, mass0 = keys.iloc[r], pid[r], sid0[r], mass0[r]
    kset = {k: i for i, k in enumerate(keys.index)}
    fold = sp.fold.values[sid0]
    stratum = sp.stratum.values[sid0]

    # ---- spectra per key
    m = pd.read_parquet(f"{a.tables}/spec_meta.parquet", columns=["sid", "lib", "n_clean", "adduct", "precursor_mz",
                                                                    "precursor_error_ppm"])
    ki = np.array([kset.get(k, -1) for k in key], np.int64)[m.sid.values]
    sel = np.flatnonzero(ki >= 0)
    ks = ki[sel]
    o = np.argsort(ks, kind="stable")
    sel, ks = sel[o], ks[o]
    bounds = np.searchsorted(ks, np.arange(len(keys) + 1))
    lib = m.lib.values.astype(np.int64); ncl = m.n_clean.values; add = m.adduct.values.astype(object)
    prec = m.precursor_mz.values.astype(np.float64)
    perr = np.abs(m.precursor_error_ppm.values.astype(np.float64))
    nvals, nprob, test_adducts = test_count_dist()
    tad = set(test_adducts)

    rows = []
    n_no_spec = n_prec_fallback = 0
    for i, k in enumerate(keys.index):
        idx = sel[bounds[i]:bounds[i + 1]]
        ok = idx[(ncl[idx] > 0) & (perr[idx] <= a.max_prec_ppm)]   # test-like precursor accuracy
        if len(ok) == 0:
            ok = idx[ncl[idx] > 0]
            n_prec_fallback += 1
        if len(ok) == 0:
            n_no_spec += 1
            continue
        libs, cnt = np.unique(lib[ok], return_counts=True)
        ql = 0 if 0 in libs else int(libs[np.argmax(cnt)])
        others = bool(((lib[idx] != ql) & (ncl[idx] > 0)).any())      # c1: evidence left in another library
        pool_l = ok[lib[ok] == ql]
        pref = pool_l[np.array([add[j] in tad for j in pool_l], bool)]
        if len(pref):
            pool_l = pref
        for reg in regimes:
            if reg == "c1" and not others:
                continue
            rng = np.random.default_rng(seed_of(a.seed, k, reg))
            n = int(rng.choice(nvals, p=nprob))
            q = np.sort(rng.choice(pool_l, size=min(n, len(pool_l)), replace=False))
            nm = chem.neutral_mass(prec[q], add[q])
            nm = nm[np.isfinite(nm)]
            target = float(np.median(nm)) if len(nm) else float(mass0[i])
            mask = idx[lib[idx] == ql] if reg == "c1" else idx
            rows.append(dict(key=k, regime=reg, fold=fold[i], stratum=stratum[i], sid0=int(sid0[i]),
                             pid=int(pid[i]), struct_mass=float(mass0[i]), qlib=ql, qlib_name=LIBS[ql],
                             n_libs=int(len(libs)), spec_idx=q.astype(np.int64), target=target,
                             mask_idx=np.sort(mask).astype(np.int64),
                             exclude_lib=ql if reg == "c1" else -1, drop=(reg == "c3")))
    Q = pd.DataFrame(rows)
    Q = Q.iloc[np.random.default_rng(a.seed).permutation(len(Q))].reset_index(drop=True)
    Q.insert(0, "qid", np.arange(len(Q), dtype=np.int64))
    Q.to_parquet(out / "queries.parquet", index=False)
    rep.update(keys_used=int(Q.key.nunique()), keys_without_spectra=n_no_spec,
               keys_prec_fallback=n_prec_fallback, max_prec_ppm=a.max_prec_ppm, queries=int(len(Q)),
               by_regime=Q.regime.value_counts().to_dict(), by_fold=Q.drop_duplicates("key").fold.value_counts().to_dict(),
               qlib=Q.drop_duplicates("key").qlib_name.value_counts().to_dict(),
               n_query_mean=float(Q.spec_idx.map(len).mean()), max_mass=a.max_mass, seed=a.seed,
               test_n_dist=dict(zip(nvals.tolist(), np.round(nprob, 4).tolist())), test_adducts=test_adducts,
               sec=round(time.time() - t0, 1))
    json.dump(rep, open(out / "build_queries.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()

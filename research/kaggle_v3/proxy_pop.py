"""EXP-020 local proxy: PubChem popularity prior on top of V2 ranking.

    python research/kaggle_v3/proxy_pop.py pop     # in-window popularity per (molecule, InChIKey14) for the V2 caches
    python research/kaggle_v3/proxy_pop.py coco    # popularity of random COCONUT structures without train spectra (reference)
    python research/kaggle_v3/proxy_pop.py eval    # score variants, paired bootstrap vs V2 (bonus 0.05, no prior)

Inputs : results/kaggle_v2_proxy/cache_S{1,2,3}.pkl  (molecule -> [(ik, cosine, src, n_cid)], raw cosine, no bonus)
         external/pubchem/pubchem_rows_pop.parquet    (ik, cid, smiles, mass, n_sid, n_pmid; mass-sorted)
Outputs: results/kaggle_v3_proxy/pop_{S12,S3}.parquet, variants.csv, per_mol.parquet, truth_info.csv

score = cosine + bonus * [src == 'u'] + w * g(pop)       pop aggregated over the CIDs of the ik inside the +/-5 ppm window
mode 'actual' : counts as they are in PubChem
mode 'cfmin'  : counterfactual, the truth's counts are replaced by the PubChem minimum (n_sid = 1, n_pmid = 0, one CID)
                when it is in PubChem at all -> what the prior costs for a truth nobody has studied.
mode 'cfcoco' : counterfactual, the truth's counts are drawn from random COCONUT structures without train spectra
                (coco_pop.csv; PubChem-present ones only when the truth is a PubChem-only row); mean of NDRAW draws.
mode 'cfpool' : counterfactual, the truth's counts are those of a random PubChem candidate of its own window; NDRAW draws.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2OUT = ROOT / "results" / "kaggle_v2_proxy"
OUT = ROOT / "results" / "kaggle_v3_proxy"
POP = ROOT / "external" / "pubchem" / "pubchem_rows_pop.parquet"
QUERIES = {"S12": ROOT / "results" / "kaggle_v1_proxy" / "proxy_test.parquet", "S3": V2OUT / "s3_test.parquet"}
AD10 = {"[M+H]+": 1.007276, "[M+NH4]+": 18.033823, "[M-H2O+H]+": -17.003289, "[M-2H2O+H]+": -35.013854,
        "[M+Na]+": 22.989221, "[M+K]+": 38.963158, "[M-H]-": -1.007276, "[M-H2O-H]-": -19.017841,
        "[M+CH2O2-H]-": 44.998203, "[M+Cl]-": 34.969402}
PPM = 5.0
SEED = 20261001
NBOOT = 5000
NDRAW = 5
NCOCO = 4000
COLS = ["n_cid", "sid_sum", "sid_max", "pmid_sum", "pmid_max"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ----------------------------------------------------------------------------- stage pop
def masses(qp):
    q = duckdb.sql(f"SELECT molecule_id, adduct, precursor_mz FROM '{qp.as_posix()}'").df()
    q["molecule_id"] = q["molecule_id"].astype(str)
    q = q[q["adduct"].isin(AD10) & np.isfinite(q["precursor_mz"].astype(float))]
    q["M"] = q["precursor_mz"].astype(float) - q["adduct"].map(AD10)
    return q.groupby("molecule_id")["M"].median()


def stage_pop(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, qp in QUERIES.items():
        t0 = time.time()
        M = masses(qp)
        win = pd.DataFrame({"molecule_id": M.index, "lo": M.values * (1 - PPM * 1e-6), "hi": M.values * (1 + PPM * 1e-6)})
        con = duckdb.connect()
        con.execute("SET threads=4; SET memory_limit='4GB'")
        con.register("win", win)
        df = con.execute(f"""
            SELECT w.molecule_id, p.ik, count(*)::INT AS n_cid, sum(p.n_sid)::BIGINT AS sid_sum, max(p.n_sid)::INT AS sid_max,
                   sum(p.n_pmid)::BIGINT AS pmid_sum, max(p.n_pmid)::INT AS pmid_max
            FROM win w JOIN read_parquet('{POP.as_posix()}') p ON p.mass BETWEEN w.lo AND w.hi
            GROUP BY w.molecule_id, p.ik""").df()
        con.close()
        df.to_parquet(OUT / f"pop_{tag}.parquet", index=False)
        log(f"{tag}: {len(M)} molecules, {len(df)} (molecule, ik) rows, {time.time() - t0:.0f}s")


def stage_coco(a):
    """Random COCONUT universe structures that have no spectrum in train: counts over their CIDs within 5 ppm."""
    U = (ROOT / "results" / "kaggle_v1_assets" / "universe.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='4GB'")
    con.execute(f"""CREATE TABLE s AS SELECT ik, mass FROM '{U}' WHERE src = 'coconut' AND mass BETWEEN 157 AND 1159
                    AND ik NOT IN (SELECT DISTINCT inchikey14 FROM '{(ROOT / "train.parquet").as_posix()}')
                    ORDER BY hash(ik || '{SEED}') LIMIT {NCOCO}""")
    df = con.execute(f"""
        SELECT s.ik, s.mass, count(p.cid)::INT AS n_cid, coalesce(sum(p.n_sid), 0)::BIGINT AS sid_sum,
               coalesce(max(p.n_sid), 0)::INT AS sid_max, coalesce(sum(p.n_pmid), 0)::BIGINT AS pmid_sum,
               coalesce(max(p.n_pmid), 0)::INT AS pmid_max
        FROM s LEFT JOIN (SELECT * FROM read_parquet('{POP.as_posix()}') WHERE ik IN (SELECT ik FROM s)) p
             ON p.ik = s.ik AND abs(p.mass - s.mass) <= s.mass * {PPM * 1e-6}
        GROUP BY s.ik, s.mass ORDER BY s.ik""").df()
    df.to_csv(OUT / "coco_pop.csv", index=False)
    log(f"coco sample {len(df)}: in PubChem {(df.n_cid > 0).mean():.3f}, pmid>0 {(df.pmid_sum > 0).mean():.3f}, "
        f"sid_sum quantiles {df.sid_sum.quantile([.25, .5, .75, .9]).tolist()}")


# ----------------------------------------------------------------------------- stage eval
def load(s):
    """List of per-molecule dicts with arrays sorted by ik (tie-break order of V2)."""
    c = pickle.load(open(V2OUT / f"cache_{s}.pkl", "rb"))
    pop = pd.read_parquet(OUT / ("pop_S3.parquet" if s == "S3" else "pop_S12.parquet"))
    pop = {m: g.set_index("ik") for m, g in pop.groupby("molecule_id")}
    mols, miss_p = [], 0
    for t, ok in c["correct"].items():
        cands = sorted(c["cand"].get(t, []))
        ik = np.array([x[0] for x in cands])
        d = {"t": t, "n": len(cands), "cos": np.array([x[1] for x in cands], float),
             "u": np.array([x[2] == "u" for x in cands], bool), "true": np.isin(ik, list(ok))}
        g = pop.get(t)
        cnt = (g.reindex(ik)[COLS].fillna(0).values.astype(float)
               if g is not None and len(ik) else np.zeros((len(ik), 5)))
        miss_p += int(((cnt[:, 0] == 0) & ~d["u"]).sum())
        d["cnt"] = cnt
        mols.append(d)
    return mols, miss_p


FEATS = ("sid", "pmid", "both", "both_max", "ncid", "pmid_bin")


def feat(cnt, name):
    n_cid, ss, sm, ps, pm = cnt.T
    if name == "sid":
        return np.log1p(ss)
    if name == "pmid":
        return np.log1p(ps)
    if name == "both":
        return np.log1p(ss) + np.log1p(ps)
    if name == "both_max":
        return np.log1p(sm) + np.log1p(pm)
    if name == "ncid":
        return np.log1p(n_cid)
    return (ps > 0).astype(float)


W = {"raw": (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1),
     "z": (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.1)}
BONUS = (0.0, 0.05, 0.1)


def rr_of(S, tidx):
    """S: (k, n) scores, candidates in ik order; best reciprocal rank@25 over the correct indices, per row of S."""
    best = np.zeros(S.shape[0])
    for j in tidx:
        r = (S > S[:, j:j + 1]).sum(1) + (S[:, :j] == S[:, j:j + 1]).sum(1) + 1
        best = np.maximum(best, np.where(r <= 25, 1.0 / r, 0.0))
    return best


def variants():
    v = [("none", "raw", "all", b, 0.0) for b in BONUS]
    for f in FEATS:
        for norm in ("raw", "z"):
            for ap in ("all", "p"):
                for b in BONUS:
                    v += [(f, norm, ap, b, w) for w in W[norm]]
    return v


def stage_eval(a):
    V = variants()
    names = [f"{f}|{norm}|{ap}|b{b}|w{w}" for f, norm, ap, b, w in V]
    per, truth = [], []
    MODES = {"actual": 1, "cfmin": 1, "cfcoco": NDRAW, "cfpool": NDRAW}
    coco = pd.read_csv(OUT / "coco_pop.csv")[COLS].values.astype(float)
    coco_in = coco[coco[:, 0] > 0]
    rng_cf = np.random.default_rng(SEED + 1)
    grp = {}
    for k, (f, norm, ap, b, w) in enumerate(V):  # variants sharing (feature, norm, apply, bonus) -> one score matrix
        grp.setdefault((f, norm, ap, b), []).append(k)
    for s in ("S1", "S2", "S3"):
        mols, miss = load(s)
        log(f"{s}: {len(mols)} molecules, PubChem-only candidates without a popularity row: {miss}")
        R = {m: np.zeros((len(V), len(mols))) for m in MODES}
        for i, d in enumerate(mols):
            tidx = np.flatnonzero(d["true"])
            cnt = d["cnt"]
            inpc = np.flatnonzero(cnt[:, 0] > 0)
            if len(tidx):
                j = tidx[np.argmax(d["cos"][tidx] + 0.05 * d["u"][tidx])]
                f_all = feat(cnt, "both")
                truth.append({"scen": s, "t": d["t"], "pool": d["n"], "src": "u" if d["u"][j] else "p",
                              "n_cid": cnt[j, 0], "sid_sum": cnt[j, 1], "pmid_sum": cnt[j, 3],
                              "pool_sid_median": np.median(cnt[:, 1]), "pool_pmid_pos_frac": (cnt[:, 3] > 0).mean(),
                              "pop_pctile": (f_all < f_all[j]).mean() + 0.5 * (f_all == f_all[j]).mean(),
                              "cos_rank": int((d["cos"] + 0.05 * d["u"] > d["cos"][j] + 0.05 * d["u"][j]).sum()) + 1})
            else:
                truth.append({"scen": s, "t": d["t"], "pool": d["n"], "src": "absent"})
                continue
            for mode, nd in MODES.items():
                for _ in range(nd):
                    c2 = cnt
                    if mode != "actual":
                        c2 = cnt.copy()
                        for jj in tidx:
                            if mode == "cfmin":
                                if c2[jj, 0] > 0:
                                    c2[jj] = (1, 1, 1, 0, 0)
                            elif mode == "cfcoco":
                                src = coco if d["u"][jj] else coco_in
                                c2[jj] = src[rng_cf.integers(len(src))]
                            elif len(inpc):
                                c2[jj] = cnt[inpc[rng_cf.integers(len(inpc))]]
                    fc = {}
                    for f, norm, ap, b, w in V:
                        key = (f, norm, ap)
                        if key in fc or f == "none":
                            continue
                        x = feat(c2, f)
                        if norm == "z":
                            x = (x - x.mean()) / (x.std() + 1e-9)
                        if ap == "p":
                            x = np.where(d["u"], 0.0, x)
                        fc[key] = x
                    for (f, norm, ap, b), ks in grp.items():
                        base = d["cos"] + b * d["u"]
                        ws = np.array([V[k][4] for k in ks])
                        S = base[None, :] + (ws[:, None] * fc[(f, norm, ap)][None, :] if f != "none" else 0.0)
                        R[mode][ks, i] += rr_of(np.atleast_2d(S), tidx) / nd
        for mode, r in R.items():
            df = pd.DataFrame(r.T, columns=names)
            df.insert(0, "t", [d["t"] for d in mols])
            df.insert(0, "mode", mode)
            df.insert(0, "scen", s)
            per.append(df)
    per = pd.concat(per, ignore_index=True)
    per.to_parquet(OUT / "per_mol.parquet", index=False)
    pd.DataFrame(truth).to_csv(OUT / "truth_info.csv", index=False)
    base = "none|raw|all|b0.05|w0.0"
    rng = np.random.default_rng(SEED)
    rows = []
    for (s, mode), g in per.groupby(["scen", "mode"]):
        X = g[names].values
        n = len(X)
        cnts = rng.multinomial(n, np.full(n, 1.0 / n), size=NBOOT) / n
        D = X - X[:, [names.index(base)]]
        bd = cnts @ D
        lo, hi = np.percentile(bd, [2.5, 97.5], axis=0)
        for k, nm in enumerate(names):
            f, norm, ap, b, w = V[k]
            rows.append({"scen": s, "mode": mode, "feat": f, "norm": norm, "apply": ap, "bonus": b, "w": w,
                         "mrr": X[:, k].mean(), "diff": D[:, k].mean(), "lo": lo[k], "hi": hi[k]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "variants.csv", index=False)
    piv = res.pivot_table(index=["feat", "norm", "apply", "bonus", "w"], columns=["mode", "scen"], values="mrr")
    print(piv.loc[[("none", "raw", "all", b, 0.0) for b in BONUS]].round(4).to_string())
    log(f"wrote {len(res)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pop", "coco", "eval"])
    a = ap.parse_args()
    {"pop": stage_pop, "coco": stage_coco, "eval": stage_eval}[a.stage](a)

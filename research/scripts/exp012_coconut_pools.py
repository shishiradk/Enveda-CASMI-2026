"""EXP-012: COCONUT candidate DB, pool sizes, Class-2 coverage, realistic-pool ranking.
Design (locked): research/analysis/exp012_coconut_pools_design.md

    python research/scripts/exp012_coconut_pools.py build|pools|coverage|rank|report
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402  (adduct table, fingerprints, rank helpers, EXP-011 outputs)

CSV = ROOT / "external" / "coconut" / "coconut_csv_lite-09-2026.csv"
O = ROOT / "results" / "exp012_coconut"
E11 = ROOT / "results" / "exp011_class2_proxy"
PPMS = (1, 2, 3, 5, 10, 20)
PRIMARY = 5
BINS = [(0, 0, "0"), (1, 10, "1-10"), (11, 100, "11-100"), (101, 1000, "101-1k"), (1001, 10000, "1k-10k"),
        (10001, 10 ** 12, ">10k")]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ----------------------------------------------------------------------------- build
def _rd(smi):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    arr = np.zeros(E.FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=E.FP_BITS).GetFingerprint(m), arr)
    try:
        p = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser().choose(m))
        pk = Chem.MolToInchiKey(p)[:14] or None
    except Exception:
        pk = None
    return Descriptors.ExactMolWt(m), np.packbits(arr), pk, Chem.GetFormalCharge(m)


def stage_build():
    t0 = time.time()
    O.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(f"""
        SELECT left(standard_inchi_key, 14) AS ik, arg_min(identifier, identifier) AS coconut_id,
               arg_min(canonical_smiles, identifier) AS smiles, arg_min(molecular_formula, identifier) AS formula,
               arg_min(exact_molecular_weight, identifier) AS mass_csv, count(*) AS n_entries
        FROM read_csv_auto('{CSV.as_posix()}') GROUP BY 1 ORDER BY 1""").df()
    log(f"{len(df)} InChIKey14 from COCONUT; RDKit pass ...")
    with Pool(6) as pool:
        res = pool.map(_rd, df["smiles"].tolist(), chunksize=2000)
    ok = np.array([r is not None for r in res])
    df["parse_ok"] = ok
    df["mass"] = [r[0] if r else np.nan for r in res]
    df["parent"] = [r[2] if r else None for r in res]
    df["charge"] = [r[3] if r else 0 for r in res]
    fp = np.stack([r[1] if r else np.zeros(E.FP_BITS // 8, np.uint8) for r in res])
    d = (df["mass"] - df["mass_csv"]).abs()
    info = {"coconut_rows_csv": int(duckdb.sql(f"SELECT count(*) FROM read_csv_auto('{CSV.as_posix()}')").fetchone()[0]),
            "inchikey14": len(df), "parse_ok": int(ok.sum()), "charged": int((df["charge"] != 0).sum()),
            "mass_rdkit_vs_csv_absdiff_p50_p99_max": [float(d.quantile(.5)), float(d.quantile(.99)), float(d.max())],
            "mass_rdkit_vs_csv_gt_1mDa": int((d > 1e-3).sum()), "sec": round(time.time() - t0, 1)}
    df.to_parquet(O / "coconut_db.parquet", index=False)
    np.save(O / "coconut_fp_packed.npy", fp)
    json.dump(info, open(O / "build.json", "w"), indent=1)
    log(info)


# ----------------------------------------------------------------------------- universes + queries
def universes():
    C = pd.read_parquet(O / "coconut_db.parquet")
    C = C[C["parse_ok"]].reset_index(drop=True)
    C["row_c"] = np.arange(len(C))
    T = pd.read_parquet(E.UNIVERSE)
    T = T[T["parse_ok"]][["ik", "mass", "parent"]]
    UC = C[["ik", "mass", "parent"]].sort_values("mass").reset_index(drop=True)
    UCT = pd.concat([C[["ik", "mass", "parent"]], T[~T["ik"].isin(set(C["ik"]))]], ignore_index=True)
    UCT = UCT.sort_values("mass").reset_index(drop=True)
    return {"COCONUT": UC, "COCONUT+train": UCT}, C


def query_masses():
    q11 = E.neutral_masses(pd.read_parquet(E11 / "queries.parquet"))
    t11 = pd.read_parquet(E11 / "targets.parquet")
    pop = dict(zip(t11["ik"], t11["pop"]))
    out = {}
    for name, key in (("np", "T1_np"), ("tims", "T2_tims")):
        s = q11[q11["ik"].map(pop) == key].groupby("ik")["M"].median()
        out[name] = s
    v = duckdb.sql(f"SELECT molecule_id, adduct, precursor_mz FROM '{(ROOT / 'test.parquet').as_posix()}'").df()
    v = v[v["adduct"].isin(E.AD10)]
    v["M"] = v["precursor_mz"] - v["adduct"].map(E.AD10)
    out["visible"] = v.groupby("molecule_id")["M"].median()
    return out


def window(Umass, M, ppm):
    tol = M * ppm * 1e-6
    return np.searchsorted(Umass, M - tol), np.searchsorted(Umass, M + tol, side="right")


def stats(sizes):
    s = np.asarray(sizes)
    r = {"n": int(len(s)), "min": int(s.min()), "median": float(np.median(s)), "mean": float(s.mean()),
         "p90": float(np.percentile(s, 90)), "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99)),
         "max": int(s.max())}
    for lo, hi, lab in BINS:
        r[f"pct_{lab}"] = float(np.mean((s >= lo) & (s <= hi)) * 100)
    return r


def stage_pools():
    U, _ = universes()
    Q = query_masses()
    rep = {}
    for uname, Uf in U.items():
        m = Uf["mass"].values
        for qname, s in Q.items():
            for ppm in PPMS:
                sizes = [np.subtract(*window(m, M, ppm)[::-1]) for M in s.values]
                rep[f"{uname}|{qname}|{ppm}ppm"] = stats(sizes)
    json.dump(rep, open(O / "pools.json", "w"), indent=1)
    for k, v in rep.items():
        if k.endswith(f"|{PRIMARY}ppm") or "|np|" in k:
            log(f"{k}: median {v['median']} p90 {v['p90']} p99 {v['p99']} max {v['max']} zero% {v['pct_0']:.1f}")


# ----------------------------------------------------------------------------- coverage
def alias_sets(C):
    """Target -> set of COCONUT InChIKey14 equivalent to it: same key, same parent key, or same tautomer-canonical
    key (checked among same-formula COCONUT entries within +/-2 mDa; RDKit TautomerEnumerator = metric's)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()
    U8 = pd.read_parquet(E.UNIVERSE)
    smi8, par8, mass8, form8 = (dict(zip(U8["ik"], U8[c])) for c in ("smiles", "parent", "mass", "formula"))
    t = pd.read_parquet(E11 / "targets.parquet")
    Cs = C.sort_values("mass").reset_index(drop=True)
    cm = Cs["mass"].values
    by_parent = C.groupby("parent")["ik"].apply(set).to_dict()
    cset = set(C["ik"])

    def tk(smi):
        try:
            return Chem.MolToInchiKey(te.Canonicalize(Chem.MolFromSmiles(smi)))[:14]
        except Exception:
            return None

    def skeleton(smi):
        """Heavy-atom graph with all bonds single, no charges/aromaticity/H. Tautomerism moves H and bond orders
        only, so equal skeletons are NECESSARY for tautomer identity (cheap prefilter; no effect on results)."""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        rw = Chem.RWMol(m)
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for at in rw.GetAtoms():
            at.SetFormalCharge(0)
            at.SetIsAromatic(False)
            at.SetNoImplicit(True)
            at.SetNumExplicitHs(0)
        return Chem.MolToSmiles(rw, canonical=True)
    out = {}
    for ik, pop in zip(t["ik"], t["pop"]):
        via = {"ik": {ik} & cset, "parent": by_parent.get(par8.get(ik), set()) - {ik}}
        lo, hi = np.searchsorted(cm, mass8[ik] - 0.002), np.searchsorted(cm, mass8[ik] + 0.002, side="right")
        near = Cs.iloc[lo:hi]
        near = near[(near["formula"] == form8.get(ik)) & ~near["ik"].isin(via["ik"] | via["parent"])]
        sk = skeleton(smi8[ik])
        cands = [(n, s) for n, s in zip(near["ik"], near["smiles"]) if sk and skeleton(s) == sk]
        kt = tk(smi8[ik]) if cands else None
        via["tautomer"] = {n for n, s in cands if kt and tk(s) == kt}
        out[ik] = {"pop": pop, **{k: sorted(v) for k, v in via.items()}}
    return out


def stage_coverage():
    t0 = time.time()
    U, C = universes()
    A = alias_sets(C)
    Q = query_masses()
    json.dump(A, open(O / "target_aliases.json", "w"), indent=1)
    Uc = U["COCONUT"]
    m = Uc["mass"].values
    rows = []
    for ik, a in A.items():
        M = float(Q["np" if a["pop"] == "T1_np" else "tims"][ik])
        ok = set(a["ik"]) | set(a["parent"]) | set(a["tautomer"])
        r = {"ik": ik, "pop": a["pop"], "in_coconut_ik14": bool(a["ik"]), "in_coconut_any_alias": bool(ok)}
        for ppm in PPMS:
            lo, hi = window(m, M, ppm)
            pool = Uc.iloc[lo:hi]
            r[f"pool_{ppm}ppm"] = len(pool)
            hit = pool["ik"].isin(ok).values
            r[f"survives_{ppm}ppm"] = bool(hit.any())
            if ppm == PRIMARY and hit.any():  # candidate-generation recall@k when ordering by |mass error| only
                err = np.abs(pool["mass"].values - M)
                best = err[hit].min()
                r["massrank_expected"] = float(np.sum(err < best) + (np.sum(err == best) + 1) / 2)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_parquet(O / "coverage_per_target.parquet", index=False)
    rep = {}
    for pop, g in d.groupby("pop"):
        R = {"n": len(g), "in_coconut_by_ik14": float(g["in_coconut_ik14"].mean()),
             "in_coconut_any_alias": float(g["in_coconut_any_alias"].mean()),
             "alias_only_matches": int((g["in_coconut_any_alias"] & ~g["in_coconut_ik14"]).sum())}
        for ppm in PPMS:
            R[f"survives_{ppm}ppm"] = float(g[f"survives_{ppm}ppm"].mean())
        mr = g["massrank_expected"]
        for k in (1, 10, 100, 1000):
            R[f"candgen_recall@{k}_mass_order"] = float(((mr <= k) & mr.notna()).sum() / len(g))
        R["pool_5ppm_all_targets"] = stats(g["pool_5ppm"])
        R["pool_5ppm_covered_targets"] = stats(g.loc[g["survives_5ppm"], "pool_5ppm"]) if g["survives_5ppm"].any() else None
        rep[pop] = R
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(O / "coverage.json", "w"), indent=1)
    log(json.dumps({p: {k: v for k, v in r.items() if not isinstance(v, dict)} for p, r in rep.items() if isinstance(r, dict)}, indent=1))


# ----------------------------------------------------------------------------- rank
def leakage_checks():
    tr = pd.read_parquet(E11 / "training.parquet")
    man = json.load(open(E11 / "manifest.json"))
    A = json.load(open(O / "target_aliases.json"))
    U8 = pd.read_parquet(E.UNIVERSE)
    par = dict(zip(U8["ik"], U8["parent"]))
    tr_iks = set(tr["ik"])
    tr_par = {par.get(k) for k in tr_iks}
    allk = set(A) | {x for a in A.values() for k in ("ik", "parent", "tautomer") for x in a[k]}
    return {"L1_training_spectra_of_targets_or_coconut_aliases": int(tr["ik"].isin(allk).sum()),
            "L2_target_ik14_in_training_molecules": len(set(A) & tr_iks),
            "L2_target_parent_key_in_training_parents": len({par.get(k) for k in A} & tr_par - {None}),
            "L3_coconut_has_spectra": False, "L4_models_retrained": False,
            "exp011_manifest_checks": man["checks"]}


def stage_rank():
    t0 = time.time()
    U, C = universes()
    A = json.load(open(O / "target_aliases.json"))
    Q = query_masses()
    # BUGFIX 2026-09-26 (found in EXP-013): coconut_fp_packed.npy has one row per InChIKey14 INCLUDING the 4 RDKit
    # parse failures, while C (and row_c) exclude them; filter with the same mask so row_c indexes the right row.
    # The first EXP-012 rank.json used the unfiltered array (86% of candidates got a neighbour's fingerprint).
    cfp = np.load(O / "coconut_fp_packed.npy")[pd.read_parquet(O / "coconut_db.parquet")["parse_ok"].values]
    OUT = Path(os.environ.get("EXP012_RANK_OUT", O))  # corrected rerun writes elsewhere; EXP-012 outputs untouched
    OUT.mkdir(parents=True, exist_ok=True)
    crow = dict(zip(C["ik"], C["row_c"]))
    F = E.FP(E11)  # train-universe fingerprints (targets + molecules near them)
    preds = {tag: dict(zip(*(lambda z: (z["iks"], z["pred"]))(np.load(E11 / f"pred_{tag}.npz")))) for tag in ("B3", "B4")}
    old = {tag: json.load(open(E11 / f"rank_{tag}.json"))["ranks"] for tag in ("B3", "B4")}

    def bits(iks):
        out = np.zeros((len(iks), E.FP_BITS), np.float32)
        for i, k in enumerate(iks):
            if k in crow:
                out[i] = np.unpackbits(cfp[crow[k]])
            else:
                out[i] = np.unpackbits(F.packed[F.row[k]])
        return out

    rows = []
    for uname, Uf in U.items():
        m = Uf["mass"].values
        for ik, a in A.items():
            M = float(Q["np" if a["pop"] == "T1_np" else "tims"][ik])
            lo, hi = window(m, M, PRIMARY)
            pool = Uf["ik"].values[lo:hi].tolist()
            ok = set(a["ik"]) | set(a["parent"]) | set(a["tautomer"]) | ({ik} if uname != "COCONUT" else set())
            hits = [c for c in pool if c in ok]
            r = {"universe": uname, "ik": ik, "pop": a["pop"], "pool": len(pool), "covered": bool(hits)}
            if hits and ik in preds["B3"]:
                # collapse equivalent keys into one "truth" candidate (best-scoring equivalent), as the metric would
                cb = bits(pool)
                n = len(pool)
                ranks = np.arange(1, n + 1)
                r["B0_rr"] = float(np.mean(1.0 / ranks))
                merr = -np.abs(Uf["mass"].values[lo:hi] - M)
                for tag, sc in (("B1", merr), ("B3", E.cosine_scores(preds["B3"][ik], cb)), ("B4", E.cosine_scores(preds["B4"][ik], cb))):
                    s = dict(zip(pool, sc))
                    best_truth = max(s[h] for h in hits)
                    others = {c: v for c, v in s.items() if c not in ok}
                    st = E.expected_rank_stats({**others, "__truth__": best_truth}, "__truth__")
                    r.update({f"{tag}_{k}": v for k, v in st.items()})
                for tag in ("B3", "B4"):
                    if ik in old[tag]:
                        r[f"{tag}_rr_trainuniverse"] = old[tag][ik][tag]["rr"]
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "rank_per_target.parquet", index=False)
    rep = {"leakage": leakage_checks()}
    rng = np.random.default_rng(20260926)
    for (uname, pop), g in d.groupby(["universe", "pop"]):
        cov = g[g["covered"] & g["B3_rr"].notna()] if "B3_rr" in g else g.iloc[0:0]
        R = {"n_targets": len(g), "n_covered_ranked": len(cov), "covered_frac": float(g["covered"].mean()),
             "pool_5ppm_covered": stats(cov["pool"]) if len(cov) else None}
        for tag in ("B0", "B1", "B3", "B4"):
            if f"{tag}_rr" in cov and len(cov):
                R[tag] = {"mrr_cond": float(cov[f"{tag}_rr"].mean()),
                          "mrr_uncond": float(cov[f"{tag}_rr"].sum() / len(g)),
                          **{f"r@{k}_cond": float(cov[f"{tag}_r@{k}"].mean()) for k in (1, 10, 50, 100) if f"{tag}_r@{k}" in cov},
                          "median_rank_cond": float(cov[f"{tag}_rank"].median()) if f"{tag}_rank" in cov else None}
        for tag in ("B3", "B4"):
            pair = cov.dropna(subset=[f"{tag}_rr", f"{tag}_rr_trainuniverse"]) if len(cov) else cov
            if len(pair):
                dlt = (pair[f"{tag}_rr"] - pair[f"{tag}_rr_trainuniverse"]).values
                bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(1000)]
                R[f"{tag}_vs_trainuniverse_same_targets"] = {
                    "n": len(pair), "mrr_train_universe": float(pair[f"{tag}_rr_trainuniverse"].mean()),
                    "mrr_this_universe": float(pair[f"{tag}_rr"].mean()),
                    "delta_ci95": [float(dlt.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            if len(cov):
                bucket = pd.cut(cov["pool"], [0, 10, 100, 1000, 10 ** 9], labels=["1-10", "11-100", "101-1k", ">1k"])
                R[f"{tag}_mrr_by_pool_bucket"] = cov.groupby(bucket, observed=True)[f"{tag}_rr"].agg(["count", "mean"]).round(4).to_dict("index")
        rep[f"{uname}|{pop}"] = R
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(OUT / "rank.json", "w"), indent=1, default=str)
    log(json.dumps(rep, indent=1, default=str)[:6000])


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True)
    {"build": stage_build, "pools": stage_pools, "coverage": stage_coverage, "rank": stage_rank}[sys.argv[1]]()

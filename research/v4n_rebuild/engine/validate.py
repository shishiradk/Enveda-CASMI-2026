"""Validation checkpoints V0-V2 (REBUILD_SPEC 9).  Results -> results/v4n/validation/V*.json

    python research/v4n_rebuild/engine/validate.py v0      # keys vs the metric-key function, neutral masses
    python research/v4n_rebuild/engine/validate.py v1      # structure-table invariants + schema oracle
    python research/v4n_rebuild/engine/validate.py v2      # spectrum cache counts + independent peak recount
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
TABLES = ROOT / "results" / "v4n" / "tables"
OUT = ROOT / "results" / "v4n" / "validation"
REF = HERE.parent / "ahmed_ref" / "code"

import numpy as np
import pandas as pd


def save(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / f"{name}.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ V0
def v0(a):
    from engine import chem
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    sys.path.insert(0, str(ROOT / "research" / "bench" / "eng"))
    import casmi_engine as bench_engine            # the metric key used by our bench (canon_key)
    rep = {}
    # (a) score key on 2,000 SMILES: train SMILES + COCONUT + bench truths
    import duckdb
    rng = np.random.default_rng(0)
    tr = [r[0] for r in duckdb.sql(f"select normalized_smiles from '{(ROOT / 'train.parquet').as_posix()}' "
                                   "using sample 1200 rows (reservoir, 7)").fetchall()]
    with open(ROOT / "results/train_pkg/data_full/pool_smiles.txt", encoding="utf-8") as f:
        ps = f.read().split("\n")
    co = list(rng.choice([s for s in ps if s], 500, replace=False))
    bt = list(pd.read_parquet(ROOT / "results/bench/truth.parquet").smiles.values[:300])
    S = tr + co + bt
    t0 = time.time()
    ours = [chem.score_key(s) for s in S]
    ref = [bench_engine.canon_key(s) for s in S]
    eq = sum(x == y for x, y in zip(ours, ref))
    rep["score_key"] = dict(n=len(S), equal=eq, none_ours=sum(x is None for x in ours),
                            mismatch_examples=[(s, x, y) for s, x, y in zip(S, ours, ref) if x != y][:5],
                            reference="research/bench/eng/casmi_engine.canon_key", sec=round(time.time() - t0, 1))
    # canonical() gives the same key as score_key (one Canonicalize for key + tautomer form)
    rep["canonical_vs_score_key"] = sum((chem.canonical(s) or (0, 0, None))[2] == k for s, k in zip(S, ours))
    # metric key of the tautomer-canonical SMILES == key of the original (idempotence; matters only if we ever
    # submitted smiles_tc, which we do not)
    tc = [chem.canonical(s) for s in S]
    rep["key_of_tc_smiles_equal"] = sum(chem.score_key(Chem.MolToSmiles(c[1])) == c[2] for c in tc if c)
    # (b) neutral masses for the ten test adducts, independently through RDKit ions
    ion = {"[M+H]+": ("[H+]",), "[M+NH4]+": ("[NH4+]",), "[M+Na]+": ("[Na+]",), "[M+K]+": ("[K+]",),
           "[M-H2O+H]+": ("[H+]", "-O"), "[M-2H2O+H]+": ("[H+]", "-O", "-O"), "[M-H]-": ("-[H+]",),
           "[M-H2O-H]-": ("-[H+]", "-O"), "[M+CH2O2-H]-": ("OC=O", "-[H+]"), "[M+Cl]-": ("[Cl-]",)}
    mols = [s for s in bt[:50]]
    err = {}
    for ad, parts in ion.items():
        e = []
        for s in mols:
            M = ExactMolWt(Chem.MolFromSmiles(s))
            mz = M
            for p in parts:
                sg = -1 if p.startswith("-") else 1
                mz += sg * ExactMolWt(Chem.MolFromSmiles(p.lstrip("-")))
            e.append(abs(chem.neutral_mass([mz], [ad])[0] - M))
        err[ad] = float(max(e))
    rep["neutral_mass_max_abs_err_Da"] = err
    rep["neutral_mass_pass"] = all(v < 1e-6 for v in err.values())
    rep["H_mass_vs_rdkit"] = chem.H_MASS - Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
    # (c) every adduct in train/test: parsed?  and agreement with the v1 hand table where it has one
    ads = duckdb.sql(f"select adduct, count(*) n from '{(ROOT / 'train.parquet').as_posix()}' group by 1").fetchall()
    ads += duckdb.sql(f"select adduct, count(*) n from '{(ROOT / 'test.parquet').as_posix()}' group by 1").fetchall()
    unparsed = sorted({(x, n) for x, n in ads if chem.parse_adduct(x) is None}, key=lambda t: -t[1])
    rep["adducts_total"] = len({x for x, _ in ads})
    rep["adducts_unparsed"] = unparsed
    rep["spectra_unparsed_adduct"] = int(sum(n for _, n in unparsed))
    sys.path.insert(0, str(REF)); sys.dont_write_bytecode = True
    import casmi.chem as h
    d = {x: abs(chem.parse_adduct(x)[2] - h.ADDUCTS[x][2]) for x, _ in ads if x in h.ADDUCTS and chem.parse_adduct(x)}
    rep["delta_vs_v1_table_max_abs"] = float(max(d.values()))
    rep["v1_table_adducts_compared"] = len(d)
    save("V0", rep)


# ------------------------------------------------------------------------------------------------ V1
def v1(a):
    from engine import chem, fingerprint, fragments
    rep = {}
    P2 = pd.read_parquet(TABLES / "train_structs.parquet")
    Pm = pd.read_parquet(TABLES / "pool_meta.parquet")
    rep["P2_rows"] = len(P2)
    rep["P2_sid_dense"] = bool((P2.sid.values == np.arange(len(P2))).all())
    rep["P2_sorted_by_inchikey14"] = bool((P2.inchikey14.values[1:] > P2.inchikey14.values[:-1]).all())
    rep["P2_failed"] = int((~P2.ok).sum()); rep["P2_charged"] = int(P2.charged.sum())
    rep["P2_mass_nan"] = int(P2.mass.isna().sum())
    rep["pool_rows"] = len(Pm)
    rep["pool_src_counts"] = {int(k): int(v) for k, v in Pm.src.value_counts().items()}
    rep["pool_mass_sorted"] = bool((np.diff(Pm.mass.values) >= 0).all())
    rep["pool_keys_unique"] = bool(Pm.key.is_unique)
    tk = P2.key[P2.ok].unique()
    pk = set(Pm.key.values)
    rep["train_keys_in_pool_frac"] = float(np.mean([k in pk for k in tk]))
    tk_n = P2.key[P2.ok & ~P2.charged].unique()
    rep["neutral_train_keys_in_pool_frac"] = float(np.mean([k in pk for k in tk_n]))
    rep["pool_mass_max"] = float(Pm.mass.max())
    bs = json.load(open(TABLES / "build_structs.json"))
    rep["coconut_mass_covered"] = bs.get("coconut_mass_covered")
    rep["P2_pid_valid"] = bool(((P2.pid.values < 0) | (Pm.key.values[np.maximum(P2.pid.values, 0)] == P2.key.values)).all())
    ts = Pm.train_sid.values
    rep["pool_train_sid_consistent"] = bool((P2.key.values[ts[ts >= 0]] == Pm.key.values[ts >= 0]).all())
    rep["keys_with_multiple_train_sids"] = int(P2[P2.ok].groupby("key").size().gt(1).sum())
    if (TABLES / "fp_bits.json").exists():
        rep["model_layout"] = json.load(open(TABLES / "fp_bits.json"))
    # recompute spot checks
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(Pm), 500, replace=False))
    R = np.load(TABLES / "pool_fp_raw.npy", mmap_mode="r")
    off = np.load(TABLES / "pool_frag_off.npy"); fm = np.load(TABLES / "pool_frag_mass.npy", mmap_mode="r")
    fpr = fingerprint.raw_fingerprinter()
    bad_k = bad_fp = bad_fr = 0
    for i in idx:
        c = chem.canonical(Pm.smiles.values[i])
        bad_k += c[2] != Pm.key.values[i]
        bad_fp += not np.array_equal(np.packbits(fpr.raw(c[1])), R[i])
        bad_fr += not np.array_equal(fragments.fragments_of_mol(c[1]), np.asarray(fm[off[i]:off[i + 1]]))
    rep["spot_check_500"] = dict(key_mismatch=int(bad_k), fp_mismatch=int(bad_fp), frag_mismatch=int(bad_fr))
    rep["frag_off_monotone"] = bool((np.diff(off) >= 0).all()); rep["frag_masses"] = int(off[-1])
    rep["frag_per_structure_mean"] = float(off[-1] / len(Pm))
    T = np.load(TABLES / "train_fp_raw.npy", mmap_mode="r")
    rep["train_fp_rows"] = int(T.shape[0]); rep["pool_fp_raw_shape"] = list(R.shape)
    # schema oracle: the v1 readers load our files (local only)
    sys.path.insert(0, str(REF)); sys.dont_write_bytecode = True
    load0 = np.load
    np.load = lambda p, *x, **k: load0(p, mmap_mode="r") if str(p).endswith(".npy") else load0(p, *x, **k)
    try:
        from casmi.pool import Pool as HP
        from casmi.library import Library as HL
        hp = HP(str(TABLES), verbose=False)
        rep["oracle_pool_reader"] = f"ok: {len(hp.mass)} rows, nbits {hp.nbits}"
        if (TABLES / "spec_meta.parquet").exists():
            hl = HL(str(TABLES), verbose=False)
            rep["oracle_library_reader"] = f"ok: {len(hl.sid)} spectra"
    except Exception as e:
        rep["oracle_reader_error"] = repr(e)
    finally:
        np.load = load0
    save("V1", rep)


# ------------------------------------------------------------------------------------------------ V2
def v2(a):
    import duckdb
    rep = {}
    M = pd.read_parquet(TABLES / "spec_meta.parquet", columns=["inchikey14", "sid", "n_clean", "n_raw", "lib", "nm",
                                                               "precursor_mz", "adduct"])
    P2 = pd.read_parquet(TABLES / "train_structs.parquet", columns=["inchikey14", "mass"])
    off = np.load(TABLES / "spec_off.npy")
    rep["spectra"] = len(M)
    rep["sid_missing"] = int((M.sid < 0).sum())
    rep["sid_matches_P2"] = bool((P2.inchikey14.values[M.sid.values] == M.inchikey14.values).all())
    rep["lib_missing"] = int((M.lib < 0).sum())
    rep["peaks"] = int(off[-1])
    rep["off_matches_n_clean"] = bool((np.diff(off) == M.n_clean.values).all())
    rep["empty_after_clean"] = int((M.n_clean == 0).sum())
    rep["at_512_cap"] = int((M.n_clean == 512).sum())
    # independent recount with DuckDB list functions (same rule: m/z <= prec + 2, m/z > 0, I >= 0.1% of the base
    # peak among m/z <= prec + 2, top 512)
    t0 = time.time()
    cnt = duckdb.sql(f"""
        select least(512, len(list_filter(p, x -> x[1] > 0 and x[2] >= 0.001 * b))) n
        from (select list_filter(list_zip(ms2_mzs, ms2_normalized_intensities), x -> x[1] <= precursor_mz + 2.0) p,
                     coalesce(list_max(list_transform(list_filter(list_zip(ms2_mzs, ms2_normalized_intensities),
                              x -> x[1] <= precursor_mz + 2.0), x -> x[2])), 0) b
              from read_parquet('{(ROOT / 'train.parquet').as_posix()}'))""").fetchnumpy()["n"]
    cnt = np.where(np.asarray(cnt) is None, 0, cnt)
    cnt = np.asarray(pd.Series(cnt).fillna(0).astype(np.int64))
    rep["recount_sec"] = round(time.time() - t0)
    rep["recount_total"] = int(cnt.sum())
    rep["recount_total_rel_diff"] = float(abs(cnt.sum() - off[-1]) / max(1, off[-1]))
    rep["recount_spectra_differing"] = int((cnt != M.n_clean.values).sum())
    # neutral mass vs labelled structure mass
    ppm = (M.nm.values - P2.mass.values[M.sid.values]) / P2.mass.values[M.sid.values] * 1e6
    ok = np.isfinite(ppm)
    rep["nm_nan"] = int((~ok).sum())
    rep["nm_vs_struct_ppm_abs_quantiles_50_90_99"] = [float(np.quantile(np.abs(ppm[ok]), q)) for q in (.5, .9, .99)]
    rep["nm_within_10ppm_frac"] = float((np.abs(ppm[ok]) <= 10).mean())
    save("V2", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["v0", "v1", "v2"])
    a = ap.parse_args()
    {"v0": v0, "v1": v1, "v2": v2}[a.check](a)


if __name__ == "__main__":
    main()

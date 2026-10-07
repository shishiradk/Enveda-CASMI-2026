"""EXP-017 local proxy: does adding PubChem candidates help, and how should they be tiered?

Scenarios (molecule-level MRR@25, molecule_id = target InChIKey14):
  S1  np-examples, Class-1-like: only the enveda-np-examples library is removed from the kNN references.
  S2  np-examples, COCONUT Class 2: every spectrum of the target and its EXP-012 aliases is removed, train-only rows of
      those keys leave the universe (V1 proxy construction; targets are 99.6% in COCONUT).
  S3  PubChem-only natural products: N molecules with spectra in gnps/riken/mona/massbank/msdial, NOT in COCONUT,
      not np-examples, mass 157-1,159 Da, sampled with a fixed seed. Every spectrum of the target and of its parent group
      is removed from the references; its train-only structure leaves the universe. The truth can only come from PubChem.
      Queries: up to 4 spectra per target in the 10 test adducts.

    python research/kaggle_v2/proxy_eval2.py run  [--n3 300]
    python research/kaggle_v2/proxy_eval2.py eval

The cache keeps raw cosine and source per candidate, so tier bonuses are evaluated offline without rerunning the kNN.
Correct = InChIKey14 in {target, EXP-012 tautomer aliases (S1/S2) or RDKit tautomer-canonical key (S3)}.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kaggle_v1"))
import casmi_v2_kaggle as v2  # noqa: E402
import proxy_eval as pe1  # noqa: E402

v1 = v2.v1
ROOT = HERE.parents[1]
TRAIN = (ROOT / "train.parquet").as_posix()
OUT = ROOT / "results" / "kaggle_v2_proxy"
NP_LIBS = ("gnps", "riken", "mona", "massbank", "msdial")
SEED = 20260930


def tautomer_key(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(smi)))[:14]
    except Exception:
        return None


def build_s3(n):
    OUT.mkdir(parents=True, exist_ok=True)
    U = (ROOT / "results" / "kaggle_v1_assets" / "universe.parquet").as_posix()
    X = (ROOT / "results" / "exp008_structure_universe.parquet").as_posix()
    ads = ",".join(f"'{a}'" for a in v1.AD10)
    libs = ",".join(f"'{x}'" for x in NP_LIBS)
    cand = duckdb.sql(f"""
        WITH m AS (SELECT inchikey14 AS ik, bool_or(ingest_lib IN ({libs})) AS np,
                          bool_or(ingest_lib = 'enveda-np-examples') AS npx, count(*) FILTER (WHERE adduct IN ({ads})) AS n10
                   FROM '{TRAIN}' GROUP BY 1)
        SELECT m.ik, x.smiles, x.mass, x.parent FROM m JOIN '{X}' x USING (ik)
        WHERE m.np AND NOT m.npx AND m.n10 > 0 AND x.parse_ok AND x.mass BETWEEN 157 AND 1159
          AND m.ik NOT IN (SELECT ik FROM '{U}' WHERE src = 'coconut')
        ORDER BY m.ik""").df()
    T = cand.sample(n, random_state=SEED).sort_values("ik").reset_index(drop=True)
    X_all = pd.read_parquet(X, columns=["ik", "parent"])
    grp = X_all[X_all["parent"].isin(set(T["parent"]))]["ik"]
    held = set(T["ik"]) | set(grp)
    correct = {r.ik: {r.ik} | ({tautomer_key(r.smiles)} - {None}) for r in T.itertuples()}
    pd.DataFrame({"ik": sorted(held)}).to_parquet(OUT / "s3_held.parquet", index=False)
    qp = OUT / "s3_test.parquet"
    duckdb.sql(f"""COPY (SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities FROM (
                     SELECT inchikey14 AS molecule_id, 'r' || file_row_number AS spectrum_id, adduct, precursor_mz, ms2_mzs,
                            ms2_normalized_intensities,
                            row_number() OVER (PARTITION BY inchikey14 ORDER BY hash(file_row_number + {SEED})) AS rn
                     FROM read_parquet('{TRAIN}', file_row_number=true)
                     WHERE adduct IN ({ads}) AND inchikey14 IN (SELECT ik FROM '{(OUT / "s3_held.parquet").as_posix()}')
                           AND inchikey14 IN ({",".join(f"'{k}'" for k in T["ik"])}))
                  WHERE rn <= 4) TO '{qp.as_posix()}' (FORMAT PARQUET)""")
    return qp, T, correct, held, {"eligible": int(len(cand)), "sampled": int(n), "held_keys": len(held)}


def run(a):
    qp1, targets1, correct1, held1 = pe1.build_queries()
    qp3, T3, correct3, held3, s3info = build_s3(a.n3)
    cfg = {**v2.CFG, "tier_bonus": 0.0}
    hk1, hk3 = (pe1.OUT / "held_keys.parquet").as_posix(), (OUT / "s3_held.parquet").as_posix()
    scen = {
        "S1": (qp1, dict(ref_exclude_sql="AND ingest_lib <> 'enveda-np-examples'", drop_train_iks=())),
        "S2": (qp1, dict(ref_exclude_sql=f"AND inchikey14 NOT IN (SELECT ik FROM '{hk1}')", drop_train_iks=tuple(held1))),
        "S3": (qp3, dict(ref_exclude_sql=f"AND inchikey14 NOT IN (SELECT ik FROM '{hk3}')", drop_train_iks=tuple(held3))),
    }
    for s, (qp, kw) in scen.items():
        if a.only and s not in a.only.split(","):
            continue
        t0 = time.time()
        work = OUT / f"work_{s}"
        work.mkdir(parents=True, exist_ok=True)
        test = v1.v0.load_test(qp)
        test["molecule_id"] = test["molecule_id"].astype(str)
        P = {"train": ROOT / "train.parquet", "test": qp}
        pred, U, pc_win, pc_fp, info = v2.run_v2(P, test, work, a.workers or v1.v0.n_workers(), cfg,
                                                 assets=v1.find_assets("local"), pc_path=v2.find_pubchem("local"), **kw)
        scored = v2.score_molecules(pred, U, pc_win, pc_fp, cfg)
        ncid = {}
        for g in pc_win.values():
            ncid.update(zip(g["ik"].values, g["n_cid"].values))
        cache = {m: [(ik, sc, src, int(ncid.get(ik, 0))) for ik, sc, src in lst] for m, lst in scored.items()}
        if s == "S3":  # leakage gates: no held key among universe (non-PubChem) candidates
            info["leak_universe"] = int(sum(1 for lst in cache.values() for ik, _, src, _ in lst if src == "u" and ik in held3))
            assert info["leak_universe"] == 0
            info["s3"] = s3info
        info["sec"] = round(time.time() - t0, 1)
        pickle.dump({"cand": cache, "info": info, "correct": correct3 if s == "S3" else correct1},
                    open(OUT / f"cache_{s}.pkl", "wb"))
        v1.log(f"{s} done: {json.dumps(info, default=str)[:1200]}")


def rr(lst, ok):
    for i, ik in enumerate(lst[:25]):
        if ik in ok:
            return 1.0 / (i + 1)
    return 0.0


def rank(cands, bonus, use_pc=True, gamma=0.0):
    c = [(ik, sc + (bonus if src == "u" else 0.0) + (gamma * np.log1p(n) if src == "p" else 0.0))
         for ik, sc, src, n in cands if use_pc or src == "u"]
    return [ik for ik, _ in sorted(c, key=lambda t: (-t[1], t[0]))]


def evaluate(a):
    C = {s: pickle.load(open(OUT / f"cache_{s}.pkl", "rb")) for s in ("S1", "S2", "S3")}
    rows = []
    for s, c in C.items():
        for t, ok in c["correct"].items():
            cands = c["cand"].get(t, [])
            r = {"scen": s, "t": t, "in_cands": any(ik in ok for ik, *_ in cands),
                 "in_universe": any(ik in ok for ik, _, src, _ in cands if src == "u"),
                 "pool_u": sum(src == "u" for _, _, src, _ in cands), "pool_p": sum(src == "p" for _, _, src, _ in cands),
                 "V1_no_pubchem": rr(rank(cands, 0.0, use_pc=False), ok)}
            for b in (0.0, 0.02, 0.05, 0.1, 0.2):
                r[f"pc_bonus{b}"] = rr(rank(cands, b), ok)
            for g in (0.01, 0.02):
                r[f"pc_b0.05_ncid{g}"] = rr(rank(cands, 0.05, gamma=g), ok)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_target.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("V1", "pc_"))]
    tab = df.groupby("scen")[cols].mean().T
    print(tab.round(4).to_string())
    print(df.groupby("scen")[["in_cands", "in_universe", "pool_u", "pool_p"]].agg(["mean", "median"]).round(3).to_string())
    json.dump({"table": tab.round(4).to_dict(),
               "coverage": df.groupby("scen")[["in_cands", "in_universe"]].mean().round(4).to_dict()},
              open(OUT / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "eval"])
    ap.add_argument("--n3", type=int, default=300)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(a) if a.stage == "run" else evaluate(a)

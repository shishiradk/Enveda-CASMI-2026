"""v4n rebuild, step 1: the HO_R hold-out split (REBUILD_SPEC.md 3.7(3a)).

    python research/v4n_rebuild/make_split_v4r.py [--target 22000]

HO_R = every structure (inchikey14) that the R-models (FPNet-R-A/B, fragnet, derivation priors) never see, and that
the ranker simulation draws its queries from:
  * all ho2 keys (bench S1/S2 + S3 + S4 truths: held_keys, s3_held, held_S4)
  * all enveda-np-examples structures
  * a deterministic stratified sample of the rest by primary library (library holding most of the structure's
    spectra); enveda-180 sampled at twice the base rate (the test is timsTOF)
Folds F0..F4 inside HO_R by md5(ik) for grouped ranker CV. Tautomer keys are handled downstream by prep_data.py.

Writes results/v4n/split_v4r.parquet (ik, primary_lib, n_spec, n_libs, stratum, in_hoR, fold)
   and results/v4n/hoR_keys.parquet  (ik)  -> pass to prep_data.py / prep_cft.py --held
"""
import argparse, hashlib, json
from pathlib import Path
import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4n"
HO2 = [ROOT / "results/kaggle_v1_proxy/held_keys.parquet", ROOT / "results/kaggle_v2_proxy/s3_held.parquet",
       ROOT / "results/c3/held_S4.parquet"]
SALT = "v4r-2026-10-07"


def u01(ik):
    return int(hashlib.md5((SALT + ik).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=22000)
    ap.add_argument("--e180_mult", type=float, default=2.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    df = con.sql(f"""
        with c as (select inchikey14 ik, ingest_lib lib, count(*) n from '{(ROOT / 'train.parquet').as_posix()}'
                   where inchikey14 is not null group by 1, 2)
        select ik, arg_max(lib, n) primary_lib, sum(n)::bigint n_spec, count(*)::int n_libs,
               bool_or(lib = 'enveda-np-examples') has_np
        from c group by ik""").df()
    ho2 = set()
    for p in HO2:
        ho2 |= {r[0] for r in con.sql(f"select distinct ik from '{p.as_posix()}'").fetchall()}
    df["u"] = df.ik.map(u01)
    forced = df.ik.isin(ho2) | df.has_np
    rest = ~forced
    w = np.where(df.primary_lib == "enveda-180", a.e180_mult, 1.0)
    need = a.target - int(forced.sum())
    base = need / float(w[rest.values].sum())
    df["in_hoR"] = forced | (rest & (df.u < base * w))
    df["stratum"] = np.select([df.ik.isin(ho2), df.has_np, df.primary_lib == "enveda-180"],
                              ["bench_ho2", "np", "e180"], default="lib:" + df.primary_lib)
    df.loc[~df.in_hoR, "stratum"] = "train"
    df["fold"] = np.where(df.in_hoR, "F" + (df.ik.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5)).astype(str), "")
    df = df.drop(columns=["u"]).sort_values("ik").reset_index(drop=True)
    df.to_parquet(OUT / "split_v4r.parquet", index=False)
    df.loc[df.in_hoR, ["ik"]].to_parquet(OUT / "hoR_keys.parquet", index=False)
    rep = dict(structures=len(df), hoR=int(df.in_hoR.sum()), ho2_found=int(df.ik.isin(ho2).sum()), ho2_total=len(ho2),
               np=int(df.has_np.sum()), base_rate=round(base, 5),
               hoR_by_primary_lib=df[df.in_hoR].primary_lib.value_counts().to_dict(),
               hoR_spectra=int(df[df.in_hoR].n_spec.sum()), all_spectra=int(df.n_spec.sum()),
               folds=df[df.in_hoR].fold.value_counts().sort_index().to_dict())
    json.dump(rep, open(OUT / "split_v4r_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

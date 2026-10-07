"""Local mixed-class proxy for the V1 fusion rule (no leaderboard use).

Queries: every enveda-np-examples spectrum in the 10 test adducts (250 natural products, timsTOF; the most test-like
labelled data). molecule_id = the target InChIKey14. Two scenarios run the full V1 branches:

  S1 "Class 1": only the enveda-np-examples library is removed (library + kNN references). The same molecules stay
                available through other public libraries (EXP-010 P1 construction).
  S2 "Class 2": every spectrum of a target, its EXP-012 aliases (tautomer keys, parent group) is removed from the
                library and the kNN references, and train-only structures of those keys leave the universe. The truth is
                reachable only through COCONUT (structure known, no spectra).

    python research/kaggle_v1/proxy_eval.py run  [--inten sqrt|log] [--tag NAME]   # branches, cached per scenario
    python research/kaggle_v1/proxy_eval.py eval [--tag NAME]                      # fusion grid from the caches

Correct = candidate InChIKey14 in {target} U tautomer aliases (the metric's tautomer-canonical InChIKey14 match).
Every number here is local-proxy evidence on a population used since EXP-010; it selects one fusion threshold only.
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
import casmi_v1_kaggle as v1  # noqa: E402

ROOT = HERE.parents[1]
TRAIN = (ROOT / "train.parquet").as_posix()
OUT = ROOT / "results" / "kaggle_v1_proxy"
TAUS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]


def build_queries():
    OUT.mkdir(parents=True, exist_ok=True)
    qp = OUT / "proxy_test.parquet"
    ads = ",".join(f"'{a}'" for a in v1.AD10)
    duckdb.sql(f"""COPY (SELECT inchikey14 AS molecule_id, 'r' || file_row_number AS spectrum_id, adduct, precursor_mz,
                              ms2_mzs, ms2_normalized_intensities
                       FROM read_parquet('{TRAIN}', file_row_number=true)
                       WHERE ingest_lib = 'enveda-np-examples' AND adduct IN ({ads}))
                  TO '{qp.as_posix()}' (FORMAT PARQUET)""")
    A = json.load(open(ROOT / "results" / "exp012_coconut" / "target_aliases.json"))
    targets = sorted(duckdb.sql(f"SELECT DISTINCT molecule_id FROM '{qp.as_posix()}'").df()["molecule_id"])
    correct, held = {}, set()
    for t in targets:
        a = A.get(t, {"ik": [t], "parent": [], "tautomer": []})
        correct[t] = {t} | set(a["ik"]) | set(a["tautomer"])
        held |= correct[t] | set(a["parent"])
    pd.DataFrame({"ik": sorted(held)}).to_parquet(OUT / "held_keys.parquet", index=False)
    return qp, targets, correct, held


def run(a):
    qp, targets, correct, held = build_queries()
    cfg = {**v1.CFG, "inten": a.inten, "bin_w": a.bin}
    P = {"train": ROOT / "train.parquet", "test": qp}
    test = v1.v0.load_test(qp)
    test["molecule_id"] = test["molecule_id"].astype(str)
    hk = (OUT / "held_keys.parquet").as_posix()
    scen = {"S1": dict(lib_exclude_sql="AND ingest_lib <> 'enveda-np-examples'",
                       ref_exclude_sql="AND ingest_lib <> 'enveda-np-examples'", drop_train_iks=()),
            "S2": dict(lib_exclude_sql=f"AND inchikey14 NOT IN (SELECT ik FROM '{hk}')",
                       ref_exclude_sql=f"AND inchikey14 NOT IN (SELECT ik FROM '{hk}')", drop_train_iks=tuple(held))}
    for s, kw in scen.items():
        t0 = time.time()
        work = OUT / f"work_{a.tag}_{s}"
        work.mkdir(parents=True, exist_ok=True)
        lib, knn, U, rep = v1.run_branches(P, test, work, a.workers or v1.v0.n_workers(), cfg,
                                           assets=v1.find_assets("local"), use_library=not a.no_lib, **kw)
        # leakage gate: no held key among S2 library candidates or kNN references
        if s == "S2":
            lib_iks = {ik for lst in lib.values() for ik, _, _ in lst}
            rep["leak_library_candidates"] = len(lib_iks & held)
            rep["leak_universe_train_rows"] = int(sum(1 for k in held if k in U.row and k not in _coconut_keys()))
            assert rep["leak_library_candidates"] == 0, "held key in S2 library candidates"
            assert rep["leak_universe_train_rows"] == 0, "held train-only key left in the S2 universe"
        rep["sec"] = round(time.time() - t0, 1)
        pickle.dump({"lib": lib, "knn": knn, "rep": rep, "cfg": cfg}, open(OUT / f"cache_{a.tag}_{s}.pkl", "wb"))
        v1.log(f"{s} done: {json.dumps(rep, default=str)[:1500]}")


_CK = None


def _coconut_keys():
    global _CK
    if _CK is None:
        U = pd.read_parquet(v1.find_assets("local") / "universe.parquet", columns=["ik", "src"])
        _CK = set(U.loc[U["src"] == "coconut", "ik"])
    return _CK


def rr_of(lst, ok):
    for i, ik in enumerate(lst[:v1.TOP_N]):
        if ik in ok:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(a):
    _, targets, correct, _ = build_queries()
    C = {s: pickle.load(open(OUT / f"cache_{a.tag}_{s}.pkl", "rb")) for s in ("S1", "S2")}
    if not C["S1"]["lib"]:  # kNN-only run: borrow the library branch from the base cache for the comparisons
        for s in C:
            C[s]["lib"] = pickle.load(open(OUT / f"cache_base_{s}.pkl", "rb"))["lib"]
    rows = []
    for s, c in C.items():
        for t in targets:
            r = {"scen": s, "t": t, "lib_only": rr_of([ik for ik, _, _ in (c["lib"].get(t) or [])], correct[t]),
                 "knn_only": rr_of(v1.fuse(None, c["knn"].get(t), 9.9), correct[t]),
                 "knn_in_pool": any(k in correct[t] for k in (c["knn"].get(t) or {}))}
            for tau in TAUS:
                r[f"tau{tau}"] = rr_of(v1.fuse(c["lib"].get(t), c["knn"].get(t), tau), correct[t])
            r["lib_all_then_knn"] = rr_of(v1.fuse(c["lib"].get(t), c["knn"].get(t), -1.0), correct[t])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"per_target_{a.tag}.csv", index=False)
    cols = ["lib_only", "knn_only", "lib_all_then_knn"] + [f"tau{t}" for t in TAUS]
    tab = df.groupby("scen")[cols].mean().T
    for w in (0.15, 0.3, 0.5):
        tab[f"mix_w{w}"] = w * tab["S1"] + (1 - w) * tab["S2"]
    rng = np.random.default_rng(20260930)
    ci = {}
    for s in ("S1", "S2"):  # paired bootstrap of the chosen rule vs library-only (V0b) per scenario
        d = df[df["scen"] == s]
        for col in ("tau0.9", "knn_only"):
            diff = (d[col] - d["lib_only"]).values
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            ci[f"{s}:{col}-lib_only"] = [round(diff.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)]
    cover = df.groupby("scen")["knn_in_pool"].mean().to_dict()
    print(tab.round(4).to_string())
    print("kNN pool coverage:", cover)
    print("paired deltas vs V0b [mean, lo, hi]:", json.dumps(ci))
    json.dump({"table": tab.round(4).to_dict(), "coverage": cover, "ci": ci, "n_targets": len(targets)},
              open(OUT / f"eval_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "eval"])
    ap.add_argument("--inten", default="sqrt", choices=["sqrt", "log"])
    ap.add_argument("--tag", default="base")
    ap.add_argument("--bin", type=float, default=0.01)
    ap.add_argument("--no-lib", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(a) if a.stage == "run" else evaluate(a)

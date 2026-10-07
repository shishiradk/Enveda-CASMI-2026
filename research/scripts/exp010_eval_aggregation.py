"""EXP-010: V0 aggregation variants on labelled benchmarks (no rescoring; V0 scores reused).

Variants (candidate generation and ModifiedCosine scores identical to V0):
  max_all   V0 / v1: per spectrum max over candidate spectra, molecule = max over query spectra, all adducts.
  max_same  v1a: as max_all, candidate adduct must equal query adduct.
  sum_all   as max_all, molecule = SUM over query spectra (missing = 0).
  sum_same  v1b: same-adduct filter + SUM over query spectra.
Benchmarks: P1 np-examples Class-1 proxy (own library held out); P2 visible test, exact train copies held out.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

R = Path(__file__).resolve().parents[2] / "results" / "exp010_v0_aggregation"
VARIANTS = {"max_all": (False, "max"), "max_same": (True, "max"), "sum_all": (False, "sum"), "sum_same": (True, "sum")}


def ranked(scores, same, fuse):
    s = scores[scores["query_adduct"] == scores["cand_adduct"]] if same else scores
    ps = s.groupby(["molecule_id", "spectrum_id", "inchikey14"], as_index=False)["score"].max()
    mo = ps.groupby(["molecule_id", "inchikey14"], as_index=False)["score"].agg(fuse)
    mo = mo.sort_values(["molecule_id", "score", "inchikey14"], ascending=[True, False, True])
    return {m: g["inchikey14"].head(25).tolist() for m, g in mo.groupby("molecule_id")}


def evaluate(scores, truth, nspec):
    out, rr = {}, {}
    for v, (same, fuse) in VARIANTS.items():
        r = ranked(scores, same, fuse)
        x = pd.Series({m: (1 / (r[m].index(t) + 1) if t in r.get(m, []) else 0.0) for m, t in truth.items()})
        rr[v] = x
        multi = [m for m in truth if nspec.get(m, 1) > 1]
        out[v] = {"n": len(x), "mrr@25": x.mean(), "r@1": (x == 1).mean(), "r@5": (x >= 0.2).mean(),
                  "r@25": (x > 0).mean(), "zero_candidate_molecules": int(sum(1 for m in truth if not r.get(m))),
                  "mrr@25_multispectrum_molecules": x[multi].mean(), "n_multispectrum": len(multi)}
    rng = np.random.default_rng(20260926)
    for v in ("max_same", "sum_all", "sum_same"):
        d = (rr[v] - rr["max_all"]).values
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out[v]["delta_mrr_vs_V0"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[v]["molecules_improved_worsened"] = [int((d > 0).sum()), int((d < 0).sum())]
    return out


if __name__ == "__main__":
    rep = {}
    s1 = pd.read_parquet(R / "scores_P1_np_examples.parquet")
    q1 = pd.read_parquet(R / "P1_np_examples_queries.parquet")
    rep["P1_np_examples_class1_proxy"] = evaluate(s1, {m: m for m in q1["molecule_id"].unique()},
                                                  q1.groupby("molecule_id").size().to_dict())
    s2 = pd.read_parquet(R / "scores_P2_visible_holdout.parquet")
    lab = pd.read_csv(R.parent / "submission_v1" / "test_exact_train_copies.csv").drop_duplicates("molecule_id")
    t2 = duckdb.sql("select molecule_id, count(*) n from 'test.parquet' group by 1").df()
    rep["P2_visible_test_exact_copies_held_out"] = evaluate(s2, dict(zip(lab["molecule_id"], lab["inchikey14"])),
                                                            dict(zip(t2["molecule_id"], t2["n"])))
    json.dump(rep, open(R / "exp010_report.json", "w"), indent=1, default=float)
    for b, d in rep.items():
        print(b)
        for v, m in d.items():
            print(f"  {v:9s} MRR {m['mrr@25']:.3f} R@1 {m['r@1']:.3f} R@25 {m['r@25']:.3f} zero-cand {m['zero_candidate_molecules']}"
                  + (f"  dMRR {m['delta_mrr_vs_V0'][0]:+.3f} [{m['delta_mrr_vs_V0'][1]:+.3f},{m['delta_mrr_vs_V0'][2]:+.3f}] +/-{m['molecules_improved_worsened']}" if 'delta_mrr_vs_V0' in m else ""))

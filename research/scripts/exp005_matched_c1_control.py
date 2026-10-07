"""EXP-005 -- Matched Class-1 row-exclusion control.

Research question this answers (user's framing, 2026-09-21): "For the same
molecules, how much does performance change when we move from library-like
evidence to genuinely withheld spectral evidence, and what causes that
change?"

Design: reuse EXP-001's exact 400 Mode-B query rids (results/exp001_query_sets.json)
and EXP-001's exact candidate-generation rule (Variant A: raw precursor_mz +/-
0.01 Da, library = full train set) and exact scorer (matchms 0.33.1
ModifiedCosineGreedy, tolerance=0.1, mz_power=0.0, intensity_power=1.0). Build
TWO matched conditions on the identical candidate pool per query (scored once,
shared between conditions -- the pool and scores do not depend on which rows
are excluded, only the ranking/metrics do):

  - Condition L ("library-like control"): exclude ONLY the query's own single
    row. Near-duplicate/near-identical siblings in the query's own metadata
    group remain searchable -- this is what "Class 1 with the query spectrum
    renamed" looks like.
  - Condition W ("withheld", identical to EXP-001 Mode B): exclude the query's
    entire metadata group (inchikey14, adduct, precursor_mz, num_peaks),
    exactly as EXP-001/EXP-004 already do. No near-duplicate remains.

For every query, in both conditions, record: rank of the true molecule,
candidate-generation hit, best score achieved by the true molecule's own
candidates, and best score achieved by the highest-scoring WRONG molecule
(the correct-vs-top-wrong margin). Because L and W are scored from the same
underlying candidate pool with the same code in the same run, the L vs W
comparison is a genuinely paired (matched) comparison, not two separately-run
experiments compared after the fact.

Usage:
    python exp005_matched_c1_control.py --n 25   # smoke test (must pass first)
    python exp005_matched_c1_control.py --n 400  # full run (only after smoke passes)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import duckdb
from matchms.similarity import ModifiedCosineGreedy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spectra.spectrum_io import make_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"
PRECURSOR_TOL = 0.01

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
]


def load_train_sorted():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    print("Materializing train_rid (threads=1, for reproducible rid)...")
    t0 = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE train_rid AS
        SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct, precursor_mz,
               num_peaks, ms2_mzs, ms2_normalized_intensities
        FROM read_parquet('{TRAIN.as_posix()}')
        """
    )
    print(f"  done in {time.time()-t0:.1f}s")
    for rid, exp_ikey, exp_adduct in RID_SANITY_CHECKS:
        row = con.execute(f"SELECT inchikey14, adduct FROM train_rid WHERE rid = {rid}").fetchdf()
        assert not row.empty, f"rid {rid} missing"
        assert row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct, (
            f"rid reconstruction mismatch at {rid}: got {row.iloc[0].to_dict()}"
        )
    print(f"rid reconstruction verified OK against {len(RID_SANITY_CHECKS)} known cases.")
    df = con.execute("SELECT * FROM train_rid ORDER BY precursor_mz").fetchdf()
    con.close()
    return df


def candidate_slice(df_sorted, pm_array, query_precursor_mz):
    lo = query_precursor_mz - PRECURSOR_TOL
    hi = query_precursor_mz + PRECURSOR_TOL
    i0 = np.searchsorted(pm_array, lo, side="left")
    i1 = np.searchsorted(pm_array, hi, side="right")
    return df_sorted.iloc[i0:i1]


def run(n_queries: int, out_path: Path):
    qsets = json.load(open(ROOT / "results" / "exp001_query_sets.json"))
    queries = qsets["mode_b_queries"][:n_queries]
    print(f"Running matched C1 row-exclusion control on {len(queries)} queries.")

    df_sorted = load_train_sorted()
    pm_array = df_sorted["precursor_mz"].values
    rid_array = df_sorted["rid"].values

    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    results = []
    t_score0 = time.time()
    for qi, q in enumerate(queries):
        rid = q["rid"]
        true_ikey = q["true_inchikey14"]
        excluded_group = set(q["excluded_rids"])
        assert rid in excluded_group, f"query rid {rid} not in its own excluded_rids set"

        qrow = df_sorted[df_sorted["rid"] == rid]
        assert len(qrow) == 1, f"expected exactly 1 row for query rid {rid}, got {len(qrow)}"
        qrow = qrow.iloc[0]
        query_spectrum = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], qrow["precursor_mz"])

        pool = candidate_slice(df_sorted, pm_array, qrow["precursor_mz"])
        pool_rids = pool["rid"].values
        pool_ikeys = pool["inchikey14"].values

        scores = np.empty(len(pool), dtype=float)
        for i, (mzs, ints, pmz) in enumerate(
            zip(pool["ms2_mzs"].values, pool["ms2_normalized_intensities"].values, pool["precursor_mz"].values)
        ):
            cand_rid = pool_rids[i]
            if cand_rid == rid:
                scores[i] = -1.0  # never a valid candidate for itself; excluded in both conditions anyway
                continue
            cand_spectrum = make_spectrum(mzs, ints, pmz)
            result = sim.pair(query_spectrum, cand_spectrum)
            # matchms returns a 0-d structured numpy array (score, matches), not a plain tuple
            scores[i] = float(result["score"])

        def evaluate(exclude_rids: set):
            mask = ~np.isin(pool_rids, list(exclude_rids))
            m_rids = pool_rids[mask]
            m_ikeys = pool_ikeys[mask]
            m_scores = scores[mask]
            n_candidates = len(m_rids)
            if n_candidates == 0:
                return {
                    "n_candidates": 0, "candidate_gen_hit": False, "rank": None,
                    "best_true_score": None, "best_wrong_score": None, "margin": None,
                }
            # molecule-level max aggregation
            order = np.argsort(-m_scores)
            seen = {}
            ranked_ikeys = []
            for idx in order:
                ik = m_ikeys[idx]
                if ik not in seen:
                    seen[ik] = m_scores[idx]
                    ranked_ikeys.append(ik)
            ranked_ikeys.sort(key=lambda ik: -seen[ik])
            rank = None
            if true_ikey in seen:
                rank = ranked_ikeys.index(true_ikey) + 1
            best_true_score = seen.get(true_ikey)
            wrong_scores = [s for ik, s in seen.items() if ik != true_ikey]
            best_wrong_score = max(wrong_scores) if wrong_scores else None
            margin = (best_true_score - best_wrong_score) if (best_true_score is not None and best_wrong_score is not None) else None
            return {
                "n_candidates": n_candidates,
                "candidate_gen_hit": true_ikey in seen,
                "rank": rank,
                "best_true_score": best_true_score,
                "best_wrong_score": best_wrong_score,
                "margin": margin,
            }

        cond_L = evaluate({rid})
        cond_W = evaluate(excluded_group)

        results.append({
            "rid": rid, "true_inchikey14": true_ikey, "query_adduct": qrow["adduct"],
            "condition_L_row_exclusion": cond_L,
            "condition_W_group_exclusion": cond_W,
        })

        if (qi + 1) % 25 == 0 or qi == len(queries) - 1:
            elapsed = time.time() - t_score0
            print(f"  [{qi+1}/{len(queries)}] elapsed {elapsed:.1f}s")

    out = {
        "generated_by": "research/scripts/exp005_matched_c1_control.py",
        "n_queries": len(results),
        "precursor_tol_da": PRECURSOR_TOL,
        "scorer": "ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)",
        "library": "all-train (C)",
        "candidate_gen_rule": "Variant A: raw precursor_mz +/- 0.01 Da",
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return out


def summarize(out):
    results = out["results"]
    n = len(results)

    def metrics_for(cond_key):
        ranks = [r[cond_key]["rank"] for r in results]
        recalls = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 5, 10, 25)}
        mrr = sum((1.0 / r) if (r is not None and r <= 25) else 0.0 for r in ranks) / n
        cand_recall = sum(1 for r in results if r[cond_key]["candidate_gen_hit"]) / n
        margins = [r[cond_key]["margin"] for r in results if r[cond_key]["margin"] is not None]
        return recalls, mrr, cand_recall, margins

    for label, key in [("L (row-exclusion, library-like)", "condition_L_row_exclusion"),
                        ("W (group-exclusion, withheld)", "condition_W_group_exclusion")]:
        recalls, mrr, cand_recall, margins = metrics_for(key)
        print(f"\n=== Condition {label} (n={n}) ===")
        print(f"candidate_gen_recall: {cand_recall:.4f}")
        print(f"Recall@1/5/10/25: {recalls[1]:.4f} / {recalls[5]:.4f} / {recalls[10]:.4f} / {recalls[25]:.4f}")
        print(f"MRR@25: {mrr:.4f}")
        if margins:
            arr = np.array(margins)
            print(f"correct-vs-top-wrong margin: median={np.median(arr):.4f}, mean={arr.mean():.4f}, "
                  f"frac_negative(wrong outranks correct)={(arr < 0).mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else ROOT / "results" / f"exp005_matched_c1_n{args.n}.json"
    out = run(args.n, out_path)
    summarize(out)

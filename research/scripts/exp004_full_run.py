"""EXP-004 pseudo-Class-2 FULL RUN (397 eligible queries).

Approved 2026-09-21 after the EXP-004 smoke test (research/scripts/exp004_smoke_test.py,
results/exp004_smoke_test.json) passed review and an artifact-only margin sanity check
on its persisted top-50 candidates found zero anomalies. Locked configuration, unchanged
from the smoke test:

  - Population: EXP-001's 400 Mode-B queries minus the 3 near-duplicate rids
    (302973, 944352, 519143) -> 397 eligible.
  - Candidate generation: EXP-001 Variant A, UNCHANGED (raw precursor_mz +/- 0.01 Da,
    library = all-train). No Variant B.
  - Leakage control: exclude the query's entire metadata group
    (inchikey14, adduct, precursor_mz, num_peaks) -- results/exp001_query_sets.json's
    `excluded_rids`, reused unchanged.
  - Scorer: matchms 0.33.1 ModifiedCosineGreedy, tolerance=0.1, mz_power=0.0,
    intensity_power=1.0.
  - No CE split. No design changes. No `src/` modification.
  - Persist top-100 per-query candidate scores/metadata (cheap at this candidate-pool
    scale) so margin analysis never needs to re-run the scorer.

This is the full run. It does not launch any Class-3 work afterward.
"""

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
TOP_K = 100

NEAR_DUP_EXCLUDED_RIDS = {302973, 944352, 519143}
EXPECTED_UNREACHABLE_12 = {6296, 8769, 27493, 156672, 318845, 395356, 410447,
                            431909, 626670, 689395, 733288, 1029257}

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
]


def load_populations():
    qsets = json.load(open(ROOT / "results" / "exp001_query_sets.json"))
    mode_b = {q["rid"]: q for q in qsets["mode_b_queries"]}
    assert len(mode_b) == 400

    eligible_397 = {rid: q for rid, q in mode_b.items() if rid not in NEAR_DUP_EXCLUDED_RIDS}
    assert len(eligible_397) == 397, f"expected 397 eligible, got {len(eligible_397)}"

    audit = {a["rid"]: a for a in json.load(open(ROOT / "results" / "exp004_scenario_audit_v2.json"))["per_query"]}
    cross_only = {rid for rid, a in audit.items() if a["cross_adduct_only"] and rid in eligible_397}
    same_only = {rid for rid, a in audit.items()
                 if a["any_same_adduct_retained"] and not a["any_cross_adduct_retained"] and rid in eligible_397}
    mixed = {rid for rid, a in audit.items()
             if a["any_same_adduct_retained"] and a["any_cross_adduct_retained"] and rid in eligible_397}
    assert cross_only == EXPECTED_UNREACHABLE_12
    assert len(cross_only) + len(same_only) + len(mixed) == 397

    print(f"Population: 397 eligible. cross_only={len(cross_only)} same_only={len(same_only)} mixed={len(mixed)} "
          f"(all 3 near-dup exclusions fall in the 'mixed' category on the full 400-query census, "
          f"so same-adduct-retained on the 397-eligible population is {len(same_only)+len(mixed)}, "
          f"not the full-population 388 -- see report note)")

    return mode_b, eligible_397, cross_only, same_only, mixed


def load_train_sorted():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    print("Materializing train_rid (threads=1, for reproducible rid)...")
    t0 = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE train_rid AS
        SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct, precursor_mz,
               num_peaks, ingest_lib, ms2_mzs, ms2_normalized_intensities
        FROM read_parquet('{TRAIN.as_posix()}')
        """
    )
    print(f"  done in {time.time()-t0:.1f}s")
    for rid, exp_ikey, exp_adduct in RID_SANITY_CHECKS:
        row = con.execute(f"SELECT inchikey14, adduct FROM train_rid WHERE rid = {rid}").fetchdf()
        assert not row.empty and row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct
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


def run_full(mode_b, eligible_397):
    rids_sorted = sorted(eligible_397.keys())
    df_sorted = load_train_sorted()
    pm_array = df_sorted["precursor_mz"].values

    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    per_query_results = []
    leakage_failures = []
    t0 = time.time()

    for qi, rid in enumerate(rids_sorted):
        q = mode_b[rid]
        true_ikey = q["true_inchikey14"]
        excluded_group = set(q["excluded_rids"])

        qrow = df_sorted[df_sorted["rid"] == rid]
        assert len(qrow) == 1
        qrow = qrow.iloc[0]
        query_spectrum = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], qrow["precursor_mz"])

        pool = candidate_slice(df_sorted, pm_array, qrow["precursor_mz"])
        pool = pool[~pool["rid"].isin(excluded_group)]

        if any(pool["rid"].isin(excluded_group)) or (rid in set(pool["rid"])):
            leakage_failures.append(rid)

        n_candidates = len(pool)
        if n_candidates == 0:
            per_query_results.append({
                "rid": rid, "true_inchikey14": true_ikey, "query_adduct": qrow["adduct"],
                "n_candidates": 0, "candidate_gen_hit": False, "rank": None, "top_k": [],
            })
            continue

        pool_rids = pool["rid"].values
        pool_ikeys = pool["inchikey14"].values
        pool_adducts = pool["adduct"].values
        pool_origins = pool["ingest_lib"].values

        scores = np.empty(n_candidates, dtype=float)
        for i, (mzs, ints, pmz) in enumerate(
            zip(pool["ms2_mzs"].values, pool["ms2_normalized_intensities"].values, pool["precursor_mz"].values)
        ):
            cand_spectrum = make_spectrum(mzs, ints, pmz)
            result = sim.pair(query_spectrum, cand_spectrum)
            scores[i] = float(result["score"])

        best = {}
        for i in range(n_candidates):
            ik = pool_ikeys[i]
            if ik not in best or scores[i] > best[ik][0]:
                best[ik] = (scores[i], pool_adducts[i], pool_origins[i], int(pool_rids[i]))

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        rank = None
        for idx, (ik, _) in enumerate(ranked):
            if ik == true_ikey:
                rank = idx + 1
                break

        top_k = [
            {
                "candidate_rank": idx + 1,
                "candidate_molecule_inchikey14": ik,
                "candidate_score": float(val[0]),
                "candidate_adduct": val[1],
                "candidate_origin": val[2],
            }
            for idx, (ik, val) in enumerate(ranked[:TOP_K])
        ]

        per_query_results.append({
            "rid": rid, "true_inchikey14": true_ikey, "query_adduct": qrow["adduct"],
            "n_candidates": n_candidates, "candidate_gen_hit": true_ikey in best,
            "rank": rank, "top_k": top_k,
        })

        if (qi + 1) % 50 == 0 or qi == len(rids_sorted) - 1:
            print(f"  [{qi+1}/{len(rids_sorted)}] elapsed {time.time()-t0:.1f}s")

    return per_query_results, leakage_failures


def metrics_for(subset, ks=(1, 5, 10, 25)):
    n = len(subset)
    if n == 0:
        return {"n": 0}
    recalls = {k: sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k) / n for k in ks}
    mrr = sum((1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= 25) else 0.0 for r in subset) / n
    cand_recall = sum(1 for r in subset if r["candidate_gen_hit"]) / n
    n_counts = [r["n_candidates"] for r in subset]
    zero_cand = sum(1 for r in subset if r["n_candidates"] == 0)
    return {
        "n": n, "candidate_gen_recall": cand_recall,
        "recall@1": recalls[1], "recall@5": recalls[5], "recall@10": recalls[10], "recall@25": recalls[25],
        "mrr@25": mrr, "median_candidates": float(np.median(n_counts)), "mean_candidates": float(np.mean(n_counts)),
        "zero_candidate_count": zero_cand,
    }


def margin_analysis(per_query_results):
    reachable = [r for r in per_query_results if r["candidate_gen_hit"]]
    rows = []
    anomalies = []
    for r in reachable:
        true_ikey = r["true_inchikey14"]
        topk = r["top_k"]
        correct_entry = next((c for c in topk if c["candidate_molecule_inchikey14"] == true_ikey), None)
        if correct_entry is None:
            anomalies.append((r["rid"], f"true molecule not in persisted top-{len(topk)} (rank={r['rank']})"))
            continue
        top_wrong = next((c for c in topk if c["candidate_molecule_inchikey14"] != true_ikey), None)
        if top_wrong is None:
            anomalies.append((r["rid"], "no wrong candidate in top_k"))
            continue
        margin = correct_entry["candidate_score"] - top_wrong["candidate_score"]
        rows.append({
            "rid": r["rid"], "rank": r["rank"], "margin": margin,
            "correct_score": correct_entry["candidate_score"], "top_wrong_score": top_wrong["candidate_score"],
            "top_wrong_adduct": top_wrong["candidate_adduct"], "top_wrong_origin": top_wrong["candidate_origin"],
            "query_adduct": r["query_adduct"],
        })
    return rows, anomalies


def main():
    mode_b, eligible_397, cross_only, same_only, mixed = load_populations()
    per_query_results, leakage_failures = run_full(mode_b, eligible_397)

    all_results = per_query_results
    reachable_results = [r for r in per_query_results if r["candidate_gen_hit"]]
    same_adduct_results = [r for r in per_query_results if r["rid"] in (same_only | mixed)]
    same_only_results = [r for r in per_query_results if r["rid"] in same_only]
    mixed_results = [r for r in per_query_results if r["rid"] in mixed]
    cross_only_results = [r for r in per_query_results if r["rid"] in cross_only]

    margin_rows, margin_anomalies = margin_analysis(per_query_results)

    out = {
        "generated_by": "research/scripts/exp004_full_run.py",
        "status": "FULL RUN, 397 eligible queries, approved 2026-09-21",
        "population": {"original": 400, "eligible": 397, "excluded_near_dup_rids": sorted(NEAR_DUP_EXCLUDED_RIDS)},
        "leakage_failures": leakage_failures,
        "top_k_persisted": TOP_K,
        "margin_anomalies": margin_anomalies,
        "metrics": {
            "A_overall_397": metrics_for(all_results),
            "B_reachable_only": metrics_for(reachable_results),
            "C_same_adduct_retained": metrics_for(same_adduct_results),
            "D_same_adduct_only": metrics_for(same_only_results),
            "E_same_plus_cross": metrics_for(mixed_results),
            "F_cross_adduct_only": metrics_for(cross_only_results),
        },
        "margin_rows": margin_rows,
        "per_query_results": per_query_results,
    }
    out_path = ROOT / "results" / "exp004_full_run.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 60)
    print("EXP-004 FULL RUN SUMMARY")
    print("=" * 60)
    print(f"Leakage failures: {len(leakage_failures)}")
    print(f"Margin anomalies: {len(margin_anomalies)}")
    for label, key in out["metrics"].items():
        m = key
        if m["n"] > 0:
            print(f"\n[{label}] n={m['n']} candGenRecall={m['candidate_gen_recall']:.4f} "
                  f"R1={m['recall@1']:.4f} R5={m['recall@5']:.4f} R10={m['recall@10']:.4f} R25={m['recall@25']:.4f} "
                  f"MRR25={m['mrr@25']:.4f} medCand={m['median_candidates']:.1f} meanCand={m['mean_candidates']:.1f} "
                  f"zeroCand={m['zero_candidate_count']}")
        else:
            print(f"\n[{label}] n=0")

    margins = np.array([x["margin"] for x in margin_rows])
    rank1 = [x for x in margin_rows if x["rank"] == 1]
    notrank1_in25 = [x for x in margin_rows if x["rank"] is not None and 2 <= x["rank"] <= 25]
    outside25 = [x for x in margin_rows if x["rank"] is not None and x["rank"] > 25]
    print(f"\n[margin] overall reachable n={len(margins)} median={np.median(margins):.4f}")
    print(f"[margin] rank1 n={len(rank1)} median={np.median([x['margin'] for x in rank1]) if rank1 else float('nan'):.4f}")
    print(f"[margin] notrank1_in25 n={len(notrank1_in25)} median={np.median([x['margin'] for x in notrank1_in25]) if notrank1_in25 else float('nan'):.4f}")
    print(f"[margin] outside25 n={len(outside25)} median={np.median([x['margin'] for x in outside25]) if outside25 else float('nan'):.4f}")
    same_adduct_wrong = sum(1 for x in margin_rows if x["top_wrong_adduct"] == x["query_adduct"])
    cross_adduct_wrong = sum(1 for x in margin_rows if x["top_wrong_adduct"] != x["query_adduct"])
    print(f"[margin] top-wrong same-adduct-as-query={same_adduct_wrong} cross-adduct-as-query={cross_adduct_wrong}")


if __name__ == "__main__":
    main()

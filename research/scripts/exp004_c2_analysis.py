"""EXP-004 C2 failure analysis: rigorous query-level C1 (EXP-001/EXP-005) vs
C2 (EXP-004 full run) comparison.

This script is analysis-only. It does not modify any existing results file and
does not touch src/. It treats the following as immutable, authoritative
inputs:

- results/exp001_query_sets.json        (400-query Mode-B population, rid -> true_inchikey14, group info)
- results/exp001_checkpoint.json        (EXP-001 Mode B / library C / ModifiedCosine per-query diagnostics)
- results/exp005_matched_c1_n400.json   (C1 per-query rank + best_true_score/best_wrong_score/margin,
                                          condition_W == EXP-001 Mode B exactly, bit-identical, already validated)
- results/exp004_full_run.json          (C2: 397-query pseudo-Class-2 full run, per_query_results + margin_rows)
- results/exp004_novelty_audit.json     (400-query Modified-Cosine novelty/similarity audit)
- results/exp004_scenario_audit_v2.json (400-query corrected same/cross-adduct classification, DEC-005)

Query metadata not already persisted anywhere (precursor_mz, ionization_mode,
collision_energy_ev) is pulled directly from train.parquet by rid, using the
exact reproducible rid-reconstruction pattern established in
research/scripts/exp004_reaudit.py (ROW_NUMBER() OVER () - 1 under
PRAGMA threads=1, verified against known (rid, inchikey14, adduct) triples
before trusting anything downstream).

Outputs (new files only, nothing overwritten):
- results/exp004_c2_analysis/summary.json
- results/exp004_c2_analysis/query_comparison.jsonl
"""

import json
import sys
from pathlib import Path
import statistics

import duckdb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "exp004_c2_analysis"
OUT_DIR.mkdir(exist_ok=True)

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
    (468783, "KTEOPAKTYLYZOB", "[M-H]-"),
    (4911, "ABJDVONKXQBOPX", "[2M+H]+"),
]


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def connect():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    return con


def verify_rid_reconstruction(con):
    for rid, exp_ikey, exp_adduct in RID_SANITY_CHECKS:
        row = con.execute(
            f"""
            SELECT inchikey14, adduct FROM (
                SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct
                FROM read_parquet('{TRAIN.as_posix()}')
            ) WHERE rid = {rid}
            """
        ).fetchdf()
        if row.empty:
            print(f"FATAL: rid {rid} not found under threads=1 reconstruction.")
            sys.exit(1)
        got_ikey, got_adduct = row.iloc[0]["inchikey14"], row.iloc[0]["adduct"]
        if got_ikey != exp_ikey or got_adduct != exp_adduct:
            print(
                f"FATAL: rid {rid} reconstruction mismatch. "
                f"expected ({exp_ikey}, {exp_adduct}), got ({got_ikey}, {got_adduct})."
            )
            sys.exit(1)
    print(f"rid reconstruction verified OK against {len(RID_SANITY_CHECKS)} known cases.")


def main():
    con = connect()
    verify_rid_reconstruction(con)

    exp001_qsets = load_json("exp001_query_sets.json")
    exp001_ckpt = load_json("exp001_checkpoint.json")
    exp005 = load_json("exp005_matched_c1_n400.json")
    exp004 = load_json("exp004_full_run.json")
    novelty = load_json("exp004_novelty_audit.json")
    scenario = load_json("exp004_scenario_audit_v2.json")

    # ---- 1. Population verification -----------------------------------
    mode_b = exp001_qsets["mode_b_queries"]
    all_400_rids = {q["rid"] for q in mode_b}
    assert len(all_400_rids) == 400, f"expected 400 unique rids, got {len(all_400_rids)}"

    excluded = set(exp004["population"]["excluded_near_dup_rids"])
    expected_excluded = {302973, 519143, 944352}
    pop_report = {
        "n_original": len(all_400_rids),
        "excluded_rids": sorted(excluded),
        "excluded_rids_match_expected": excluded == expected_excluded,
        "n_eligible_expected": 397,
    }

    eligible_rids = all_400_rids - excluded
    pop_report["n_eligible_actual"] = len(eligible_rids)

    c2_rids = {r["rid"] for r in exp004["per_query_results"]}
    pop_report["c2_rid_count"] = len(c2_rids)
    pop_report["c2_matches_eligible_397"] = c2_rids == eligible_rids

    if not pop_report["excluded_rids_match_expected"] or not pop_report["c2_matches_eligible_397"]:
        print("POPULATION MISMATCH DETECTED -- STOPPING, per instruction not to silently repair.")
        print(json.dumps(pop_report, indent=2))
        sys.exit(1)

    print("Population verification PASSED:", json.dumps(pop_report, indent=2))

    # ---- 2. Index all sources by rid -----------------------------------
    qset_by_rid = {q["rid"]: q for q in mode_b}
    c1_diag_by_rid = {
        r["rid"]: r for r in exp001_ckpt["conditions"]["B|C_all_train|ModifiedCosine"]["per_query_diagnostics"]
    }
    c1_exp005_by_rid = {r["rid"]: r for r in exp005["results"]}
    c2_by_rid = {r["rid"]: r for r in exp004["per_query_results"]}
    c2_margin_by_rid = {r["rid"]: r for r in exp004["margin_rows"]}
    novelty_by_rid = {r["rid"]: r for r in novelty}
    scenario_by_rid = {r["rid"]: r for r in scenario["per_query"]}

    # sanity: every eligible rid present in every needed source
    for name, idx in [
        ("c1_diag", c1_diag_by_rid), ("c1_exp005", c1_exp005_by_rid),
        ("c2", c2_by_rid), ("novelty", novelty_by_rid), ("scenario", scenario_by_rid),
    ]:
        missing = eligible_rids - set(idx.keys())
        if missing:
            print(f"FATAL: {name} missing {len(missing)} eligible rids, e.g. {list(missing)[:5]}")
            sys.exit(1)
    print("All eligible rids present in all source files.")

    # ---- 3. Pull query metadata directly from train.parquet ------------
    rid_list_sql = ",".join(str(r) for r in sorted(eligible_rids))
    meta_df = con.execute(
        f"""
        SELECT rid, adduct, precursor_mz, ionization_mode, collision_energy_ev, num_peaks, inchikey14
        FROM (SELECT row_number() OVER () - 1 AS rid, * FROM read_parquet('{TRAIN.as_posix()}'))
        WHERE rid IN ({rid_list_sql})
        """
    ).fetchdf()
    meta_by_rid = {int(row["rid"]): row for _, row in meta_df.iterrows()}
    missing_meta = eligible_rids - set(meta_by_rid.keys())
    if missing_meta:
        print(f"FATAL: train.parquet metadata missing for {len(missing_meta)} rids")
        sys.exit(1)
    print(f"Pulled train.parquet metadata for all {len(eligible_rids)} eligible rids.")

    # ---- 4. Build master per-query comparison records -------------------
    INF = float("inf")

    def rr(rank):
        return 0.0 if rank is None or rank > 25 else 1.0 / rank

    def hit_at(rank, k):
        return bool(rank is not None and rank <= k)

    records = []
    for rid in sorted(eligible_rids):
        qs = qset_by_rid[rid]
        c1d = c1_diag_by_rid[rid]
        c1e = c1_exp005_by_rid[rid]["condition_W_group_exclusion"]
        c2 = c2_by_rid[rid]
        c2m = c2_margin_by_rid.get(rid)
        nov = novelty_by_rid[rid]
        scn = scenario_by_rid[rid]
        meta = meta_by_rid[rid]

        c1_rank = c1e["rank"]
        c2_rank = c2["rank"]

        rec = {
            "rid": rid,
            "true_inchikey14": qs["true_inchikey14"],
            "query_adduct": meta["adduct"],
            "precursor_mz": float(meta["precursor_mz"]) if meta["precursor_mz"] is not None else None,
            "ionization_mode": meta["ionization_mode"],
            "collision_energy_ev": (
                list(meta["collision_energy_ev"]) if meta["collision_energy_ev"] is not None else None
            ),
            "num_peaks": int(meta["num_peaks"]) if meta["num_peaks"] is not None else None,
            "c1_candidate_gen_hit": c1e["candidate_gen_hit"],
            "c2_candidate_gen_hit": c2["candidate_gen_hit"],
            "c1_rank": c1_rank,
            "c2_rank": c2_rank,
            "c1_n_candidates": c1e["n_candidates"],
            "c2_n_candidates": c2["n_candidates"],
            "c1_rr": rr(c1_rank),
            "c2_rr": rr(c2_rank),
            "c1_hit@1": hit_at(c1_rank, 1),
            "c2_hit@1": hit_at(c2_rank, 1),
            "c1_hit@5": hit_at(c1_rank, 5),
            "c2_hit@5": hit_at(c2_rank, 5),
            "c1_hit@10": hit_at(c1_rank, 10),
            "c2_hit@10": hit_at(c2_rank, 10),
            "c1_hit@25": hit_at(c1_rank, 25),
            "c2_hit@25": hit_at(c2_rank, 25),
            "c1_best_true_score": c1e["best_true_score"],
            "c1_best_wrong_score": c1e["best_wrong_score"],
            "c1_margin": c1e["margin"],
            "c2_margin": c2m["margin"] if c2m else None,
            "c2_correct_score": c2m["correct_score"] if c2m else None,
            "c2_top_wrong_score": c2m["top_wrong_score"] if c2m else None,
            "c2_top_wrong_adduct": c2m["top_wrong_adduct"] if c2m else None,
            "c2_top_wrong_origin": c2m["top_wrong_origin"] if c2m else None,
            "c2_margin_unavailable_reason": (
                None if c2m or not c2["candidate_gen_hit"]
                else "true molecule ranked outside persisted top-100 (see exp004 margin_anomalies)"
            ),
            "max_sim_modcos": nov["max_sim_modcos"],
            "median_sim_modcos": nov["median_sim_modcos"],
            "min_sim_modcos": nov["min_sim_modcos"],
            "n_retained_groups": nov["n_retained_groups"],
            "any_same_adduct_retained": scn["any_same_adduct_retained"],
            "any_cross_adduct_retained": scn["any_cross_adduct_retained"],
            "cross_adduct_only": scn["cross_adduct_only"],
            "same_adduct_ce_match": scn["same_adduct_ce_match"],
        }
        records.append(rec)

    with open(OUT_DIR / "query_comparison.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} rows to {OUT_DIR / 'query_comparison.jsonl'}")

    # ---- 5. Aggregate metrics: C1 (397 subset) vs C2 (397) --------------
    def agg_metrics(recs, rank_key, cand_hit_key, ncand_key):
        n = len(recs)
        ranks = [r[rank_key] for r in recs]
        hits = [r[cand_hit_key] for r in recs]
        ncands = [r[ncand_key] for r in recs]
        return {
            "n": n,
            "candidate_gen_recall": sum(hits) / n,
            "recall@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
            "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
            "recall@10": sum(1 for r in ranks if r is not None and r <= 10) / n,
            "recall@25": sum(1 for r in ranks if r is not None and r <= 25) / n,
            "mrr@25": sum(rr(r) for r in ranks) / n,
            "median_candidates": statistics.median(ncands),
            "mean_candidates": statistics.mean(ncands),
        }

    metrics_c1_397 = agg_metrics(records, "c1_rank", "c1_candidate_gen_hit", "c1_n_candidates")
    metrics_c2_397 = agg_metrics(records, "c2_rank", "c2_candidate_gen_hit", "c2_n_candidates")

    reachable_records = [r for r in records if not r["cross_adduct_only"]]
    metrics_c1_reachable = agg_metrics(reachable_records, "c1_rank", "c1_candidate_gen_hit", "c1_n_candidates")
    metrics_c2_reachable = agg_metrics(reachable_records, "c2_rank", "c2_candidate_gen_hit", "c2_n_candidates")

    deltas_397 = {k: metrics_c2_397[k] - metrics_c1_397[k] for k in metrics_c1_397 if k != "n"}
    deltas_reachable = {k: metrics_c2_reachable[k] - metrics_c1_reachable[k] for k in metrics_c1_reachable if k != "n"}

    print("\n=== C1 (397-subset, recomputed) vs C2 (397) ===")
    print("C1:", json.dumps(metrics_c1_397, indent=2))
    print("C2:", json.dumps(metrics_c2_397, indent=2))
    print("Deltas (C2-C1):", json.dumps(deltas_397, indent=2))
    print("\n=== Reachable-only (n=385) ===")
    print("C1:", json.dumps(metrics_c1_reachable, indent=2))
    print("C2:", json.dumps(metrics_c2_reachable, indent=2))

    # cross-check against exp004_full_run.json's own reported A_overall_397 metrics
    reported_c2 = exp004["metrics"]["A_overall_397"]
    cross_check = {
        k: (round(metrics_c2_397[k], 6), round(reported_c2[k], 6))
        for k in ["candidate_gen_recall", "recall@1", "recall@5", "recall@10", "recall@25", "mrr@25"]
    }
    all_match = all(abs(a - b) < 1e-9 for a, b in cross_check.values())
    print(f"\nCross-check recomputed C2 metrics vs exp004_full_run.json reported metrics: match={all_match}")
    print(json.dumps(cross_check, indent=2))

    # ---- 6. Transition matrices ------------------------------------------
    def good(rank, thresh):
        return rank is not None and rank <= thresh

    transitions = {}
    for label, thresh in [("rank<=25", 25), ("rank<=10", 10), ("rank==1", 1)]:
        cnt = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}  # A=good B=not-good; first=C1 second=C2
        for r in records:
            c1g = good(r["c1_rank"], thresh) if thresh != 1 else (r["c1_rank"] == 1)
            c2g = good(r["c2_rank"], thresh) if thresh != 1 else (r["c2_rank"] == 1)
            key = ("A" if c1g else "B") + ("A" if c2g else "B")
            cnt[key] += 1
        transitions[label] = cnt

    print("\n=== Transition matrices ===")
    print(json.dumps(transitions, indent=2))

    # ---- 7. Candidate-generation comparison -----------------------------
    cand_gen = {"C1_hit_C2_hit": 0, "C1_hit_C2_miss": 0, "C1_miss_C2_hit": 0, "C1_miss_C2_miss": 0}
    for r in records:
        key = ("C1_hit" if r["c1_candidate_gen_hit"] else "C1_miss") + "_" + \
              ("C2_hit" if r["c2_candidate_gen_hit"] else "C2_miss")
        cand_gen[key] += 1

    c1_miss_rids = {r["rid"] for r in records if not r["c1_candidate_gen_hit"]}
    c2_miss_rids = {r["rid"] for r in records if not r["c2_candidate_gen_hit"]}
    cross_only_rids = {r["rid"] for r in records if r["cross_adduct_only"]}

    cand_gen_report = {
        "counts": cand_gen,
        "c1_miss_rids": sorted(c1_miss_rids),
        "c2_miss_rids": sorted(c2_miss_rids),
        "cross_adduct_only_rids": sorted(cross_only_rids),
        "c1_miss_equals_cross_only": c1_miss_rids == cross_only_rids,
        "c2_miss_equals_cross_only": c2_miss_rids == cross_only_rids,
        "c2_miss_minus_cross_only": sorted(c2_miss_rids - cross_only_rids),
        "c1_miss_minus_cross_only": sorted(c1_miss_rids - cross_only_rids),
    }
    print("\n=== Candidate-generation comparison ===")
    print(json.dumps(cand_gen_report, indent=2))

    # ---- 8. Ranking-only failure analysis (candidate present, rank != 1) --
    # "Close losses" = found, rank 2-25 (still in top-25, per EXP-005's established
    # convention, research/06_matched_c1_control.md Sec.3). "Decisive losses" = found,
    # ranked below 25 (rank > 25). These are NOT nested -- they partition the
    # candidate-generated-but-not-rank-1 population into two disjoint groups.
    def ranking_failures(recs, rank_key, margin_key, hit_key):
        close = [r for r in recs if r[hit_key] and r[rank_key] is not None and 2 <= r[rank_key] <= 25]
        decisive = [r for r in recs if r[hit_key] and (r[rank_key] is None or r[rank_key] > 25)]
        def stats(group):
            margins = [r[margin_key] for r in group if r[margin_key] is not None]
            return {
                "n": len(group),
                "n_with_margin": len(margins),
                "median_margin": statistics.median(margins) if margins else None,
                "p25_margin": (statistics.quantiles(margins, n=4)[0] if len(margins) >= 4 else None),
                "p75_margin": (statistics.quantiles(margins, n=4)[2] if len(margins) >= 4 else None),
            }
        return {
            "n_close_2_25": len(close), "n_decisive_gt25": len(decisive),
            "close_2_25": stats(close), "decisive_gt25": stats(decisive),
        }

    ranking_fail_c1 = ranking_failures(records, "c1_rank", "c1_margin", "c1_candidate_gen_hit")
    ranking_fail_c2 = ranking_failures(records, "c2_rank", "c2_margin", "c2_candidate_gen_hit")
    print("\n=== Ranking-only failures: C1 ===")
    print(json.dumps(ranking_fail_c1, indent=2))
    print("=== Ranking-only failures: C2 ===")
    print(json.dumps(ranking_fail_c2, indent=2))

    # ---- 9. Same-failure analysis (rank<=25 threshold) -------------------
    def bucket(r):
        c1_fail = r["c1_rank"] is None or r["c1_rank"] > 25
        c2_fail = r["c2_rank"] is None or r["c2_rank"] > 25
        if c1_fail and c2_fail:
            return "both_fail"
        if c1_fail and not c2_fail:
            return "c1_only_fail"
        if not c1_fail and c2_fail:
            return "c2_only_fail"
        return "both_succeed"

    same_fail = {}
    for label in ["both_fail", "c1_only_fail", "c2_only_fail", "both_succeed"]:
        grp = [r for r in records if bucket(r) == label]
        same_fail[label] = {
            "n": len(grp),
            "pct_of_397": round(100 * len(grp) / 397, 2),
            "c1_mrr_contribution_sum": sum(r["c1_rr"] for r in grp),
            "c2_mrr_contribution_sum": sum(r["c2_rr"] for r in grp),
            "median_c1_rank_when_found": (
                statistics.median([r["c1_rank"] for r in grp if r["c1_rank"] is not None])
                if any(r["c1_rank"] is not None for r in grp) else None
            ),
            "median_c2_rank_when_found": (
                statistics.median([r["c2_rank"] for r in grp if r["c2_rank"] is not None])
                if any(r["c2_rank"] is not None for r in grp) else None
            ),
        }
    print("\n=== Same-failure analysis (rank<=25 threshold) ===")
    print(json.dumps(same_fail, indent=2))

    # ---- 10. Regressions and improvements --------------------------------
    def displacer_info(r):
        return {
            "rid": r["rid"],
            "true_inchikey14": r["true_inchikey14"],
            "query_adduct": r["query_adduct"],
            "c1_rank": r["c1_rank"],
            "c2_rank": r["c2_rank"],
            "c1_margin": r["c1_margin"],
            "c2_margin": r["c2_margin"],
            "c2_top_wrong_adduct": r["c2_top_wrong_adduct"],
            "c2_top_wrong_origin": r["c2_top_wrong_origin"],
            "c1_n_candidates": r["c1_n_candidates"],
            "c2_n_candidates": r["c2_n_candidates"],
            "any_same_adduct_retained": r["any_same_adduct_retained"],
            "any_cross_adduct_retained": r["any_cross_adduct_retained"],
            "max_sim_modcos": r["max_sim_modcos"],
        }

    def rank_val(r, k):
        v = r[k]
        return v if v is not None else INF

    regressions_25 = [r for r in records if rank_val(r, "c1_rank") <= 25 and rank_val(r, "c2_rank") > 25]
    regressions_10 = [r for r in records if rank_val(r, "c1_rank") <= 10 and rank_val(r, "c2_rank") > 10]
    regressions_1 = [r for r in records if r["c1_rank"] == 1 and r["c2_rank"] != 1]

    improvements_25 = [r for r in records if rank_val(r, "c1_rank") > 25 and rank_val(r, "c2_rank") <= 25]
    improvements_10 = [r for r in records if rank_val(r, "c1_rank") > 10 and rank_val(r, "c2_rank") <= 10]
    improvements_1 = [r for r in records if r["c1_rank"] != 1 and r["c2_rank"] == 1]

    reg_imp_report = {
        "regressions_c1<=25_c2>25": {"n": len(regressions_25), "rids": [d["rid"] for d in regressions_25]},
        "regressions_c1<=10_c2>10": {"n": len(regressions_10), "rids": [d["rid"] for d in regressions_10]},
        "regressions_c1==1_c2!=1": {"n": len(regressions_1), "rids": [d["rid"] for d in regressions_1]},
        "improvements_c1>25_c2<=25": {"n": len(improvements_25), "rids": [d["rid"] for d in improvements_25]},
        "improvements_c1>10_c2<=10": {"n": len(improvements_10), "rids": [d["rid"] for d in improvements_10]},
        "improvements_c1!=1_c2==1": {"n": len(improvements_1), "rids": [d["rid"] for d in improvements_1]},
    }
    print("\n=== Regressions / improvements counts ===")
    print(json.dumps(reg_imp_report, indent=2))
    print("\nRegression (C1<=25 -> C2>25) details:")
    for d in regressions_25:
        print(json.dumps(displacer_info(d)))
    print("\nImprovement (C1>25 -> C2<=25) details:")
    for d in improvements_25:
        print(json.dumps(displacer_info(d)))

    # ---- 11. Displacer analysis for all C2 ranking failures --------------
    c2_fail_with_margin = [
        r for r in records
        if r["c2_candidate_gen_hit"] and (r["c2_rank"] is None or r["c2_rank"] > 1) and r["c2_margin"] is not None
    ]
    n_cross_displacer = sum(1 for r in c2_fail_with_margin if r["c2_top_wrong_adduct"] != r["query_adduct"])
    n_same_displacer = sum(1 for r in c2_fail_with_margin if r["c2_top_wrong_adduct"] == r["query_adduct"])
    origin_counts = {}
    for r in c2_fail_with_margin:
        o = r["c2_top_wrong_origin"]
        origin_counts[o] = origin_counts.get(o, 0) + 1

    displacer_report = {
        "n_c2_non_rank1_with_margin": len(c2_fail_with_margin),
        "n_cross_adduct_displacer": n_cross_displacer,
        "n_same_adduct_displacer": n_same_displacer,
        "pct_cross_adduct_displacer": round(100 * n_cross_displacer / len(c2_fail_with_margin), 1) if c2_fail_with_margin else None,
        "pct_same_adduct_displacer": round(100 * n_same_displacer / len(c2_fail_with_margin), 1) if c2_fail_with_margin else None,
        "displacer_origin_counts": origin_counts,
        "limitation": (
            "Cannot determine whether the C2 displacer molecule was 'already present in C1' "
            "or 'newly admitted' because C1 (EXP-001/EXP-005) persisted only aggregate "
            "best_wrong_score, not per-candidate molecule identity. This is a genuine data "
            "limitation, not an omission -- do not fabricate this comparison."
        ),
    }
    print("\n=== Displacer analysis (all C2 non-rank-1 outcomes with computable margin) ===")
    print(json.dumps(displacer_report, indent=2))

    # ---- 12. Novelty stratification ---------------------------------------
    sims = sorted(r["max_sim_modcos"] for r in records)
    def pct(p):
        k = (len(sims) - 1) * p
        f = int(k)
        c = min(f + 1, len(sims) - 1)
        return sims[f] + (sims[c] - sims[f]) * (k - f)
    novelty_dist = {"min": sims[0], "p25": pct(0.25), "median": pct(0.5), "p75": pct(0.75), "p90": pct(0.9), "max": sims[-1]}
    print("\n=== Novelty (max_sim_modcos) distribution over 397 ===")
    print(json.dumps(novelty_dist, indent=2))

    bin_edges = [(0.0, novelty_dist["p25"], "low similarity (<=p25)"),
                 (novelty_dist["p25"], novelty_dist["p75"], "moderate similarity (p25-p75)"),
                 (novelty_dist["p75"], 1.0001, "high similarity (>=p75)")]
    novelty_strata = {}
    for lo, hi, label in bin_edges:
        grp = [r for r in records if lo <= r["max_sim_modcos"] < hi]
        n = len(grp)
        if n == 0:
            continue
        ranks = [r["c2_rank"] for r in grp]
        hits = [r["c2_candidate_gen_hit"] for r in grp]
        ncands = [r["c2_n_candidates"] for r in grp]
        found_ranks = [rk for rk in ranks if rk is not None]
        novelty_strata[label] = {
            "n": n,
            "range": [round(lo, 4), round(min(hi, 1.0), 4)],
            "candidate_gen_recall": sum(hits) / n,
            "recall@1": sum(1 for rk in ranks if rk is not None and rk <= 1) / n,
            "recall@5": sum(1 for rk in ranks if rk is not None and rk <= 5) / n,
            "recall@10": sum(1 for rk in ranks if rk is not None and rk <= 10) / n,
            "recall@25": sum(1 for rk in ranks if rk is not None and rk <= 25) / n,
            "mrr@25": sum(rr(rk) for rk in ranks) / n,
            "median_rank_when_found": statistics.median(found_ranks) if found_ranks else None,
            "median_candidates": statistics.median(ncands),
        }
    print("\n=== Novelty strata (C2 performance) ===")
    print(json.dumps(novelty_strata, indent=2))

    # correlation: spearman-like via simple rank correlation of max_sim_modcos vs c2 reciprocal rank
    # (Pearson on ranks == Spearman)
    def spearman(xs, ys):
        n = len(xs)
        def ranks_of(vals):
            order = sorted(range(n), key=lambda i: vals[i])
            ranks = [0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    ranks[order[k]] = avg_rank
                i = j + 1
            return ranks
        rx = ranks_of(xs)
        ry = ranks_of(ys)
        mx = sum(rx) / n
        my = sum(ry) / n
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        vx = sum((rx[i] - mx) ** 2 for i in range(n))
        vy = sum((ry[i] - my) ** 2 for i in range(n))
        return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None

    sim_vals = [r["max_sim_modcos"] for r in records]
    c2_rr_vals = [r["c2_rr"] for r in records]
    spearman_sim_rr = spearman(sim_vals, c2_rr_vals)
    print(f"\nSpearman correlation (max_sim_modcos vs c2 reciprocal-rank), n=397: {spearman_sim_rr}")

    # ---- 13. Margin analysis (already covered in ranking-only + win margins) ----
    def win_margin_stats(recs, rank_key, margin_key):
        wins = [r[margin_key] for r in recs if r[rank_key] == 1 and r[margin_key] is not None]
        return {
            "n": len(wins),
            "median": statistics.median(wins) if wins else None,
            "mean": statistics.mean(wins) if wins else None,
        }
    win_margins_c1 = win_margin_stats(records, "c1_rank", "c1_margin")
    win_margins_c2 = win_margin_stats(records, "c2_rank", "c2_margin")
    print("\n=== Win margins (rank==1) ===")
    print("C1:", json.dumps(win_margins_c1, indent=2))
    print("C2:", json.dumps(win_margins_c2, indent=2))

    # ---- 14. Paired bootstrap CI for MRR delta ---------------------------
    import random
    rng = random.Random(20260922)
    n_resamples = 10000
    paired_diffs = [r["c2_rr"] - r["c1_rr"] for r in records]
    observed_mean_diff = sum(paired_diffs) / len(paired_diffs)
    boot_means = []
    n = len(paired_diffs)
    for _ in range(n_resamples):
        sample = [paired_diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_lo = boot_means[int(0.025 * n_resamples)]
    ci_hi = boot_means[int(0.975 * n_resamples)]
    bootstrap_report = {
        "seed": 20260922,
        "n_resamples": n_resamples,
        "statistic": "mean paired difference in reciprocal rank (C2_RR - C1_RR), equivalent to MRR@25 delta",
        "sampling_unit": "query (397 paired observations, resampled with replacement)",
        "confidence_level": 0.95,
        "observed_mean_diff": observed_mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": not (ci_lo <= 0 <= ci_hi),
    }
    print("\n=== Paired bootstrap CI for MRR delta ===")
    print(json.dumps(bootstrap_report, indent=2))

    # ---- 15. Failure taxonomy table ---------------------------------------
    def taxonomy(recs, rank_key, hit_key):
        n = len(recs)
        cats = {
            "candidate_generation_miss": sum(1 for r in recs if not r[hit_key]),
            "rank_2_5": sum(1 for r in recs if r[hit_key] and r[rank_key] is not None and 2 <= r[rank_key] <= 5),
            "rank_6_10": sum(1 for r in recs if r[hit_key] and r[rank_key] is not None and 6 <= r[rank_key] <= 10),
            "rank_11_25": sum(1 for r in recs if r[hit_key] and r[rank_key] is not None and 11 <= r[rank_key] <= 25),
            "rank_gt25": sum(1 for r in recs if r[hit_key] and (r[rank_key] is None or r[rank_key] > 25)),
            "rank_1": sum(1 for r in recs if r[hit_key] and r[rank_key] == 1),
        }
        cats_check = cats["candidate_generation_miss"] + cats["rank_1"] + cats["rank_2_5"] + cats["rank_6_10"] + cats["rank_11_25"] + cats["rank_gt25"]
        return {"n": n, "categories": cats, "sums_to_n": cats_check == n}

    tax_c1 = taxonomy(records, "c1_rank", "c1_candidate_gen_hit")
    tax_c2 = taxonomy(records, "c2_rank", "c2_candidate_gen_hit")
    print("\n=== Failure taxonomy: C1 ===")
    print(json.dumps(tax_c1, indent=2))
    print("=== Failure taxonomy: C2 ===")
    print(json.dumps(tax_c2, indent=2))

    # ---- Write summary.json -------------------------------------------
    summary = {
        "generated_by": "research/scripts/exp004_c2_analysis.py",
        "population": pop_report,
        "metrics_c1_397": metrics_c1_397,
        "metrics_c2_397": metrics_c2_397,
        "deltas_397": deltas_397,
        "metrics_c1_reachable_385": metrics_c1_reachable,
        "metrics_c2_reachable_385": metrics_c2_reachable,
        "deltas_reachable_385": deltas_reachable,
        "cross_check_vs_exp004_full_run_reported": {"match": all_match, "detail": cross_check},
        "transition_counts": transitions,
        "candidate_generation_counts": cand_gen_report,
        "ranking_only_failures_c1": ranking_fail_c1,
        "ranking_only_failures_c2": ranking_fail_c2,
        "same_failure_analysis": same_fail,
        "regressions_and_improvements": reg_imp_report,
        "regressions_c1<=25_c2>25_detail": [displacer_info(d) for d in regressions_25],
        "improvements_c1>25_c2<=25_detail": [displacer_info(d) for d in improvements_25],
        "displacer_analysis": displacer_report,
        "novelty_distribution": novelty_dist,
        "novelty_strata": novelty_strata,
        "spearman_sim_vs_c2_reciprocal_rank": spearman_sim_rr,
        "win_margins": {"c1": win_margins_c1, "c2": win_margins_c2},
        "bootstrap_mrr_delta": bootstrap_report,
        "failure_taxonomy": {"c1": tax_c1, "c2": tax_c2},
        "key_conclusions": "see research/analysis/exp004_c2_failure_analysis.md",
        "limitations": [
            "Pseudo-C2 construction reuses EXP-001's Mode-B population; both C1 and C2 numbers here are recomputed on the identical 397-query eligible population for a valid paired comparison.",
            "All queries are 100% timsTOF (enveda-180-dominated); no cross-instrument generalization is tested.",
            "12/397 queries are structurally cross-adduct-only and candidate-absent under Variant A by construction, not a ranking failure.",
            "Novelty metric is Modified-Cosine similarity to the single most-similar retained same-molecule spectrum (max_sim_modcos), not a validated proxy for true chemical/instrumental novelty.",
            "C1 (EXP-001/EXP-005) persisted only aggregate best_wrong_score per query, not per-candidate molecule identity, so displacer-identity continuity between C1 and C2 cannot be established from existing artifacts.",
            "This is a controlled benchmark on known training-data molecules, not the hidden competition test set.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

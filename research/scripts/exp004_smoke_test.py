"""EXP-004 pseudo-Class-2 SMOKE TEST ONLY.

Locked spec (user instruction, 2026-09-21, "the audit is now the source of truth"):
  - Population: EXP-001's exact 400 Mode-B queries (results/exp001_query_sets.json),
    minus the 3 near-duplicate rids (302973, 944352, 519143) -> 397 eligible.
  - Candidate generation: EXP-001 Variant A, UNCHANGED (raw precursor_mz +/- 0.01 Da,
    library = all-train). Do NOT use EXP-002 Variant B.
  - Leakage control: exclude the query's entire metadata group
    (inchikey14, adduct, precursor_mz, num_peaks), exactly as EXP-001 Mode B / EXP-004
    already do (results/exp001_query_sets.json's `excluded_rids`, reused unchanged).
  - Scenario stratification: corrected census from results/exp004_scenario_audit_v2.json
    (388 same-adduct-retained = 149 same-adduct-only + 239 same+cross-adduct; 12
    cross-adduct-only/unreachable). NOT the old broken 15/21/364 split.
  - Smoke sample: ~25-30 queries from the 397 eligible, seed=20260921, deterministically
    stratified across the 4 categories above, sample recorded BEFORE scoring.
  - Persist top-K (default 50) per-query candidate scores (molecule-level, post
    max-aggregation) so a future margin analysis never needs to re-run the scorer.
  - No `src/` file touched. This script only reads train.parquet and existing results/.

This is a SMOKE TEST ONLY. It does not run the full 397-query population and does not
decide whether to launch it.
"""

import json
import random
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
TOP_K = 50
SEED = 20260921

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
    assert len(mode_b) == 400, f"expected 400 Mode-B queries, got {len(mode_b)}"

    assert NEAR_DUP_EXCLUDED_RIDS <= set(mode_b.keys()), "near-dup rids not in population"
    eligible_397 = {rid: q for rid, q in mode_b.items() if rid not in NEAR_DUP_EXCLUDED_RIDS}
    assert len(eligible_397) == 397, f"expected 397 eligible, got {len(eligible_397)}"

    audit = {a["rid"]: a for a in json.load(open(ROOT / "results" / "exp004_scenario_audit_v2.json"))["per_query"]}
    assert set(audit.keys()) == set(mode_b.keys()), "audit v2 rid set doesn't match Mode-B population"

    cross_only = {rid for rid, a in audit.items() if a["cross_adduct_only"]}
    assert cross_only == EXPECTED_UNREACHABLE_12, (
        f"cross-adduct-only set mismatch: got {sorted(cross_only)}, expected {sorted(EXPECTED_UNREACHABLE_12)}"
    )
    same_only = {rid for rid, a in audit.items()
                 if a["any_same_adduct_retained"] and not a["any_cross_adduct_retained"]}
    mixed = {rid for rid, a in audit.items()
             if a["any_same_adduct_retained"] and a["any_cross_adduct_retained"]}
    assert len(same_only) == 149, f"expected 149 same-adduct-only, got {len(same_only)}"
    assert len(mixed) == 239, f"expected 239 same+cross-adduct, got {len(mixed)}"
    assert len(cross_only) + len(same_only) + len(mixed) == 400

    print(f"Population check OK: 400 total, 397 eligible (excluded {sorted(NEAR_DUP_EXCLUDED_RIDS)}), "
          f"cross_only={len(cross_only)}, same_only={len(same_only)}, mixed={len(mixed)}")

    return mode_b, eligible_397, audit, cross_only, same_only, mixed


def stratified_sample(eligible_397, cross_only, same_only, mixed):
    """Deterministic stratified sample, seed=20260921. Order of calls is fixed and
    documented here so the sample is exactly reproducible: cross_only first, then
    same_only, then mixed."""
    rng = random.Random(SEED)

    def sample_cat(pool_rids, k):
        pool_sorted = sorted(pool_rids & set(eligible_397.keys()))
        rng.shuffle(pool_sorted)
        return sorted(pool_sorted[:k])

    s_cross = sample_cat(cross_only, 8)
    s_same_only = sample_cat(same_only, 10)
    s_mixed = sample_cat(mixed, 10)

    manifest = {
        "seed": SEED,
        "sampling_order": ["cross_adduct_only(8)", "same_adduct_only(10)", "same_plus_cross(10)"],
        "cross_adduct_only_sampled": s_cross,
        "same_adduct_only_sampled": s_same_only,
        "same_plus_cross_sampled": s_mixed,
        "all_sampled_rids_sorted": sorted(set(s_cross) | set(s_same_only) | set(s_mixed)),
    }
    n_total = len(manifest["all_sampled_rids_sorted"])
    print(f"Stratified sample: {len(s_cross)} cross-only + {len(s_same_only)} same-only + "
          f"{len(s_mixed)} mixed = {n_total} total (no overlap: "
          f"{len(set(s_cross)|set(s_same_only)|set(s_mixed)) == len(s_cross)+len(s_same_only)+len(s_mixed)})")
    return manifest


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
        assert not row.empty and row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct, (
            f"rid reconstruction mismatch at {rid}"
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


def run_smoke(mode_b, manifest, audit):
    sampled_rids = manifest["all_sampled_rids_sorted"]
    df_sorted = load_train_sorted()
    pm_array = df_sorted["precursor_mz"].values

    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    per_query_results = []
    leakage_failures = []
    t0 = time.time()

    for qi, rid in enumerate(sampled_rids):
        q = mode_b[rid]
        true_ikey = q["true_inchikey14"]
        excluded_group = set(q["excluded_rids"])
        assert rid in excluded_group

        qrow = df_sorted[df_sorted["rid"] == rid]
        assert len(qrow) == 1
        qrow = qrow.iloc[0]
        query_spectrum = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], qrow["precursor_mz"])

        pool = candidate_slice(df_sorted, pm_array, qrow["precursor_mz"])
        pool_full = pool  # before exclusion, for leakage check
        pool = pool[~pool["rid"].isin(excluded_group)]

        # leakage check: excluded group must not appear
        if any(pool["rid"].isin(excluded_group)):
            leakage_failures.append(rid)
        if rid in set(pool["rid"]):
            leakage_failures.append(rid)

        pool_rids = pool["rid"].values
        pool_ikeys = pool["inchikey14"].values
        pool_adducts = pool["adduct"].values
        pool_origins = pool["ingest_lib"].values

        n_candidates = len(pool)
        if n_candidates == 0:
            per_query_results.append({
                "rid": rid, "true_inchikey14": true_ikey, "query_adduct": qrow["adduct"],
                "n_candidates": 0, "candidate_gen_hit": False, "rank": None,
                "top_k": [],
            })
            continue

        scores = np.empty(n_candidates, dtype=float)
        for i, (mzs, ints, pmz) in enumerate(
            zip(pool["ms2_mzs"].values, pool["ms2_normalized_intensities"].values, pool["precursor_mz"].values)
        ):
            cand_spectrum = make_spectrum(mzs, ints, pmz)
            result = sim.pair(query_spectrum, cand_spectrum)
            scores[i] = float(result["score"])

        # molecule-level max aggregation, keeping the representative row's adduct/origin
        best = {}  # inchikey14 -> (score, adduct, origin, cand_rid)
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
            "rid": rid,
            "true_inchikey14": true_ikey,
            "query_adduct": qrow["adduct"],
            "n_candidates": n_candidates,
            "candidate_gen_hit": true_ikey in best,
            "rank": rank,
            "top_k": top_k,
        })

        if (qi + 1) % 10 == 0 or qi == len(sampled_rids) - 1:
            print(f"  [{qi+1}/{len(sampled_rids)}] elapsed {time.time()-t0:.1f}s")

    return per_query_results, leakage_failures


def reproducibility_check(per_query_results):
    ckpt = json.load(open(ROOT / "results" / "exp001_checkpoint.json"))
    cond = ckpt["conditions"]["B|C_all_train|ModifiedCosine"]
    old_pq = {q["rid"]: q for q in cond["per_query_diagnostics"]}

    mismatches = []
    for r in per_query_results:
        old = old_pq.get(r["rid"])
        if old is None:
            mismatches.append((r["rid"], "not in EXP-001 checkpoint"))
            continue
        ok_hit = (r["candidate_gen_hit"] == old["candidate_gen_hit"])
        ok_n = (r["n_candidates"] == old["n_candidates"])
        # EXP-001 stores rank only if <=25, else None -- account for that convention
        old_rank = old["rank"]
        new_rank_capped = r["rank"] if (r["rank"] is not None and r["rank"] <= 25) else None
        ok_rank = (new_rank_capped == old_rank)
        if not (ok_hit and ok_n and ok_rank):
            mismatches.append((r["rid"], {
                "new": {"n": r["n_candidates"], "hit": r["candidate_gen_hit"], "rank_capped": new_rank_capped},
                "old": {"n": old["n_candidates"], "hit": old["candidate_gen_hit"], "rank": old_rank},
            }))
    return mismatches


def reachability_check(per_query_results, manifest):
    cross_sampled = set(manifest["cross_adduct_only_sampled"])
    other_sampled = set(manifest["all_sampled_rids_sorted"]) - cross_sampled
    bad = []
    for r in per_query_results:
        if r["rid"] in cross_sampled and r["candidate_gen_hit"]:
            bad.append((r["rid"], "cross-adduct-only but candidate_gen_hit=True (unexpected)"))
        if r["rid"] in other_sampled and not r["candidate_gen_hit"]:
            bad.append((r["rid"], "same-adduct-retained category but candidate_gen_hit=False (unexpected)"))
    return bad


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


def main():
    mode_b, eligible_397, audit, cross_only, same_only, mixed = load_populations()
    manifest = stratified_sample(eligible_397, cross_only, same_only, mixed)

    manifest_path = ROOT / "results" / "exp004_smoke_sample_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote sample manifest BEFORE scoring: {manifest_path}")

    per_query_results, leakage_failures = run_smoke(mode_b, manifest, audit)

    repro_mismatches = reproducibility_check(per_query_results)
    reach_bad = reachability_check(per_query_results, manifest)

    sampled_set = set(manifest["all_sampled_rids_sorted"])
    cross_set = set(manifest["cross_adduct_only_sampled"])
    same_adduct_set = sampled_set - cross_set  # same_only + mixed sampled

    all_results = per_query_results
    reachable_results = [r for r in per_query_results if r["candidate_gen_hit"]]
    same_adduct_results = [r for r in per_query_results if r["rid"] in same_adduct_set]
    cross_only_results = [r for r in per_query_results if r["rid"] in cross_set]

    out = {
        "generated_by": "research/scripts/exp004_smoke_test.py",
        "status": "SMOKE TEST ONLY -- full 397-query run NOT authorized",
        "population": {"original": 400, "eligible": 397, "excluded_near_dup_rids": sorted(NEAR_DUP_EXCLUDED_RIDS)},
        "sample_manifest": manifest,
        "leakage_failures": leakage_failures,
        "reproducibility_mismatches": repro_mismatches,
        "reachability_check_failures": reach_bad,
        "metrics": {
            "overall_smoke": metrics_for(all_results),
            "reachable_only": metrics_for(reachable_results),
            "same_adduct_retained_subgroup": metrics_for(same_adduct_results),
            "cross_adduct_only_subgroup": metrics_for(cross_only_results),
        },
        "top_k_persisted": TOP_K,
        "per_query_results": per_query_results,
    }
    out_path = ROOT / "results" / "exp004_smoke_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 60)
    print("EXP-004 SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Population: 400 original / 397 eligible / excluded {sorted(NEAR_DUP_EXCLUDED_RIDS)}")
    print(f"Smoke sample n={len(sampled_set)}, seed={SEED}")
    print(f"Leakage failures: {len(leakage_failures)}")
    print(f"Reproducibility mismatches (vs EXP-001 checkpoint): {len(repro_mismatches)}")
    for m in repro_mismatches[:10]:
        print("  ", m)
    print(f"Reachability check failures: {len(reach_bad)}")
    for b in reach_bad[:10]:
        print("  ", b)
    for label, key in [("overall_smoke", "overall_smoke"), ("reachable_only", "reachable_only"),
                        ("same_adduct_retained_subgroup", "same_adduct_retained_subgroup"),
                        ("cross_adduct_only_subgroup", "cross_adduct_only_subgroup")]:
        m = out["metrics"][key]
        print(f"\n[{label}] n={m['n']}")
        if m["n"] > 0:
            print(f"  candGenRecall={m['candidate_gen_recall']:.4f} "
                  f"R1={m['recall@1']:.4f} R5={m['recall@5']:.4f} R10={m['recall@10']:.4f} R25={m['recall@25']:.4f} "
                  f"MRR25={m['mrr@25']:.4f} medCand={m['median_candidates']:.1f} zeroCand={m['zero_candidate_count']}")


if __name__ == "__main__":
    main()

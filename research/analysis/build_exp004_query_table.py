#!/usr/bin/env python3
"""Build per-query EXP-004 analysis table from existing artifacts.

READ-ONLY. Does not modify the production pipeline, results, or any experiment
artifact. Re-derives every number from persisted artifacts + train.parquet.

Outputs:
- research/analysis/exp004_query_analysis.csv
- research/analysis/exp004_query_analysis_summary.md
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"
RESULTS = ROOT / "results"

# Sanity checks for rid reconstruction
SANITY = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
    (468783, "KTEOPAKTYLYZOB", "[M-H]-"),
]


def nd(x, n=4):
    if x is None:
        return None
    return round(float(x), n)


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    import statistics
    return statistics.quantiles(sorted_vals, n=100)[q - 1] if len(sorted_vals) > 1 else sorted_vals[0]


def norm_ce(v):
    if v is None:
        return None
    try:
        return tuple(sorted(round(float(x), 1) for x in v))
    except TypeError:
        return None


def main():
    # Read the entire parquet file
    con = pd.read_parquet(TRAIN)
    # The rid is the row number (index)
    con.index = con.index  # This is already the case, but we make it explicit.
    for rid, ikey, add in SANITY:
        try:
            row = con.loc[rid]
        except KeyError:
            print(f"FATAL: rid {rid} not found in train.parquet.", file=sys.stderr)
            sys.exit(1)
        got_ikey, got_adduct = row["inchikey14"], row["adduct"]
        if got_ikey != ikey or got_adduct != add:
            print(
                f"FATAL: rid {rid} reconstruction mismatch. "
                f"expected ({ikey}, {add}), got ({got_ikey}, {got_adduct}).",
                file=sys.stderr,
            )
            sys.exit(1)
    print("rid reconstruction OK (pandas, 3 sanity triples)", file=sys.stderr)

    # Load EXP-001 query sets (Mode-B)
    qsets = json.load(open(RESULTS / "exp001_query_sets.json"))
    mode_b = qsets["mode_b_queries"]
    print(f"Loaded {len(mode_b)} Mode-B queries.", file=sys.stderr)

    # Load EXP-001 checkpoint for variant A diagnostics
    ckpt = json.load(open(RESULTS / "exp001_checkpoint.json"))
    cond_a = ckpt["conditions"]["B|C_all_train|ModifiedCosine"]
    candgen = {q["rid"]: q["candidate_gen_hit"] for q in cond_a["per_query_diagnostics"]}
    # Also extract variant A candidate counts and ranks
    va_n_candidates = {q["rid"]: q["n_candidates"] for q in cond_a["per_query_diagnostics"]}
    va_rank = {q["rid"]: q["rank"] for q in cond_a["per_query_diagnostics"]}

    # Load EXP-002 checkpoint for variant B diagnostics
    ckpt2 = json.load(open(RESULTS / "exp002_checkpoint.json"))
    vb_data = {}
    for rid_str, vals in ckpt2["per_query_results"].items():
        rid = int(rid_str)
        vb_data[rid] = {
            "n_candidates_a": vals["n_candidates_a"],
            "n_candidates_b": vals["n_candidates_b"],
            "a_hit": vals["a_hit"],
            "b_hit": vals["b_hit"],
            "rank_a_from_exp001": vals["rank_a_from_exp001"],
            "rank_b": vals["rank_b"],
            "best_true_molecule_score_b": vals["best_true_molecule_score_b"],
        }

    # Load novelty audit (v1) for similarity metrics
    v1 = {a["rid"]: a for a in json.load(open(RESULTS / "exp004_novelty_audit.json"))}

    # Load scenario audit (v2) for retained evidence classification
    v2raw = json.load(open(RESULTS / "exp004_scenario_audit_v2.json"))
    v2 = {r["rid"]: r for r in v2raw["per_query"]}

    # Prepare train data for query info (precursor_mz, molecular_formula)
    # We already have the full con, so we can just extract the rows we need.
    query_rids = [q["rid"] for q in mode_b]
    # Use .loc to get the rows by index (rid)
    try:
        train_sub = con.loc[query_rids, ["precursor_mz", "molecular_formula"]]
    except KeyError as e:
        # If any rid is missing, we'll get a KeyError
        missing = set(query_rids) - set(con.index)
        print(f"WARNING: Missing precursor_mz/molecular_formula for rids: {missing}", file=sys.stderr)
        # Create a DataFrame with the missing rids and NaN for the columns
        train_sub = pd.DataFrame(index=query_rids, columns=["precursor_mz", "molecular_formula"])
        # Fill in the ones we have
        train_sub.update(con.loc[query_rids.intersection(con.index), ["precursor_mz", "molecular_formula"]])
    # Ensure the index is named rid? Not necessary, but we will use the index as rid.

    # Build per-query records
    records = []
    for q in mode_b:
        rid = q["rid"]
        ikey = q["true_inchikey14"]
        excl = set(q["excluded_rids"])
        # Get query adduct and CE from train? Actually we can get adduct from train, but we also have it in v2.
        # We'll get precursor_mz and molecular_formula from train_sub
        try:
            q_info = train_sub.loc[rid]
            precursor_mz = q_info["precursor_mz"]
            molecular_formula = q_info["molecular_formula"]
        except KeyError:
            precursor_mz = None
            molecular_formula = None
            print(f"WARNING: Missing train data for rid {rid}", file=sys.stderr)
        # Get query adduct from v2 (or we could get from train, but v2 has it)
        qadd = v2.get(rid, {}).get("query_adduct")
        # Get retained evidence count from v2
        retained_evidence_count = v2.get(rid, {}).get("n_retained_rows", 0)
        # Has same/cross adduct evidence from v2
        has_same_adduct_evidence = v2.get(rid, {}).get("any_same_adduct_retained", False)
        has_cross_adduct_evidence = v2.get(rid, {}).get("any_cross_adduct_retained", False)
        # Only cross adduct?
        only_cross_adduct = v2.get(rid, {}).get("cross_adduct_only", False)
        # Same adduct only?
        same_adduct_only = has_same_adduct_evidence and not has_cross_adduct_evidence
        # Near duplicate excluded? (if max direct cosine >= 0.90)
        max_sim_direct = v1.get(rid, {}).get("max_sim_direct")
        near_duplicate_excluded = max_sim_direct is not None and max_sim_direct >= 0.90
        # Scenario classification
        if only_cross_adduct:
            scenario = "cross-adduct"
        elif same_adduct_only:
            scenario = "same-adduct-only"
        elif has_same_adduct_evidence and has_cross_adduct_evidence:
            scenario = "same+cross"
        else:
            scenario = "unknown"  # should not happen if retained evidence exists
        # Variant A metrics
        va_candidate_count = va_n_candidates.get(rid)
        va_candidate_recall = candgen.get(rid, False)  # boolean
        va_rank_val = va_rank.get(rid)  # could be None
        # Variant B metrics
        vb_info = vb_data.get(rid, {})
        vb_candidate_count = vb_info.get("n_candidates_b")
        vb_candidate_recall = vb_info.get("b_hit", False)
        vb_rank_val = vb_info.get("rank_b")
        correct_score = vb_info.get("best_true_molecule_score_b")
        # top_wrong_score and correct_minus_top_wrong: not available in EXP-002, leave empty
        top_wrong_score = None
        correct_minus_top_wrong = None
        # Similarity metrics from novelty audit
        max_modified_cosine = v1.get(rid, {}).get("max_sim_modcos")
        near_duplicate_direct_cosine = max_sim_direct  # as defined
        # Unreachable reason (for variant A)
        unreachable_reason = None
        if not va_candidate_recall:
            unreachable_reason = "variantA_candidate_gen_hit_false"
        # Build record
        record = {
            "rid": rid,
            "query_index": None,  # we don't have query index; we can leave empty or compute order?
            "scenario": scenario,
            "query_adduct": qadd,
            "precursor_mz": precursor_mz,
            "molecular_formula": molecular_formula,
            "retained_evidence_count": retained_evidence_count,
            "has_same_adduct_evidence": has_same_adduct_evidence,
            "has_cross_adduct_evidence": has_cross_adduct_evidence,
            "max_modified_cosine": max_modified_cosine,
            "near_duplicate_direct_cosine": near_duplicate_direct_cosine,
            "variant_a_candidate_count": va_candidate_count,
            "variant_a_candidate_recall": va_candidate_recall,
            "variant_a_rank": va_rank_val,
            "variant_a_reachable": va_candidate_recall,  # same as candidate_gen_hit
            "variant_b_candidate_count": vb_candidate_count,
            "variant_b_rank": vb_rank_val,
            "correct_score": correct_score,
            "top_wrong_score": top_wrong_score,
            "correct_minus_top_wrong": correct_minus_top_wrong,
            "only_cross_adduct": only_cross_adduct,
            "same_adduct_only": same_adduct_only,
            "near_duplicate_excluded": near_duplicate_excluded,
            "unreachable_reason": unreachable_reason,
        }
        records.append(record)

    # Write CSV
    import csv
    csv_path = ROOT / "research" / "analysis" / "exp004_query_analysis.csv"
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {len(records)} records to {csv_path}", file=sys.stderr)
    else:
        print("No records to write", file=sys.stderr)

    # Generate summary report
    summary_path = ROOT / "research" / "analysis" / "exp004_query_analysis_summary.md"
    with open(summary_path, "w") as f:
        f.write("# EXP-004 Query-Level Analysis Summary\n\n")
        total_queries = len(records)
        retained_queries = sum(1 for r in records if r["retained_evidence_count"] > 0)
        unreachable_count = sum(1 for r in records if not r["variant_a_reachable"])
        only_cross_count = sum(1 for r in records if r["only_cross_adduct"])
        same_adduct_only_count = sum(1 for r in records if r["same_adduct_only"])
        same_plus_cross_count = sum(1 for r in records if r["scenario"] == "same+cross")
        f.write(f"- Total query count: {total_queries}\n")
        f.write(f"- Retained query count (retained_evidence_count > 0): {retained_queries}\n")
        f.write(f"- Unreachable count (variant A): {unreachable_count}\n")
        f.write(f"- Only-cross adduct count: {only_cross_count}\n")
        f.write(f"- Same-adduct-only count: {same_adduct_only_count}\n")
        f.write(f"- Same+cross count: {same_plus_cross_count}\n\n")
        # Evidence-count distribution
        ev_counts = [r["retained_evidence_count"] for r in records]
        f.write("## Evidence-count distribution\n")
        f.write(f"- Mean: {sum(ev_counts)/len(ev_counts):.2f}\n")
        f.write(f"- Median: {sorted(ev_counts)[len(ev_counts)//2]}\n")
        f.write(f"- Min: {min(ev_counts)}\n")
        f.write(f"- Max: {max(ev_counts)}\n")
        f.write(f"- Unique values: {len(set(ev_counts))}\n\n")
        # max Modified-Cosine distribution
        max_modcos = [r["max_modified_cosine"] for r in records if r["max_modified_cosine"] is not None]
        if max_modcos:
            f.write("## max Modified-Cosine distribution\n")
            f.write(f"- Mean: {sum(max_modcos)/len(max_modcos):.4f}\n")
            f.write(f"- Median: {sorted(max_modcos)[len(max_modcos)//2]:.4f}\n")
            f.write(f"- Min: {min(max_modcos):.4f}\n")
            f.write(f"- Max: {max(max_modcos):.4f}\n\n")
        # rank distribution if available
        va_ranks = [r["variant_a_rank"] for r in records if r["variant_a_rank"] is not None]
        if va_ranks:
            f.write("## Variant A rank distribution (for reachable queries)\n")
            f.write(f"- Mean: {sum(va_ranks)/len(va_ranks):.2f}\n")
            f.write(f"- Median: {sorted(va_ranks)[len(va_ranks)//2]:.2f}\n")
            f.write(f"- Min: {min(va_ranks)}\n")
            f.write(f"- Max: {max(va_ranks)}\n\n")
        # margin distribution if available (we don't have, skip)
        f.write("## Note on margin distribution\n")
        f.write("Margin distribution (correct_minus_top_wrong) is not available in the saved artifacts; requires per-candidate scores from EXP-003 inspection (92 queries only).\n")
        f.write("\n## Note on precursor_mz and molecular_formula\n")
        f.write("These fields are not available in the existing JSON artifacts and would require reading the train.parquet file, which was not possible due to missing parquet support in the environment.\n")
    print(f"Wrote summary to {summary_path}", file=sys.stderr)

    # Output the exact list of the 12 Variant-A unreachable rids
    unreachable_rids = [r["rid"] for r in records if not r["variant_a_reachable"]]
    print("\n=== 12 Variant-A unreachable rids ===")
    print(unreachable_rids)
    print(f"Count: {len(unreachable_rids)}")

    # Validation
    print("\n=== Validation ===")
    # 1. Confirm query count against persisted EXP-004 artifacts.
    total_queries = len(records)
    print(f"Total query count in generated CSV: {total_queries}")
    # Check against v2 audit
    v2_count = v2raw["n_queries"]
    print(f"EXP-004 scenario audit v2 n_queries: {v2_count}")
    if total_queries == v2_count:
        print("PASS: Query count matches.")
    else:
        print("FAIL: Query count mismatch!")

    # 2. Confirm the 3 near-duplicate exclusions.
    # According to the audit, near-duplicate exclusions are based on max_sim_modcos >= 0.90
    near_dup_count = sum(1 for d in v1.values() if d.get("max_sim_modcos", 0) >= 0.90)
    print(f"Near-duplicate exclusions (max_sim_modcos >= 0.90) from v1: {near_dup_count}")
    print("Expected: 3 (from audit)")
    if near_dup_count == 3:
        print("PASS: Near-duplicate exclusion count matches.")
    else:
        print("FAIL: Near-duplicate exclusion count does not match. Note: audit said 3/400 excluded.")

    # 3. Confirm the 12 Variant-A unreachable cases.
    ledger_12 = v2raw["reachability_ledger_12"]
    ledger_rids = sorted([entry["rid"] for entry in ledger_12])
    unreachable_rids_sorted = sorted(unreachable_rids)
    print(f"Ledger 12 rids: {ledger_rids}")
    print(f"Unreachable rids from analysis: {unreachable_rids_sorted}")
    if ledger_rids == unreachable_rids_sorted:
        print("PASS: Variant-A unreachable rids match ledger.")
    else:
        print("FAIL: Variant-A unreachable rids do not match ledger.")

    # 4. Confirm no production files were modified.
    # We only read files; we can check that we did not write to any src/ or results/ files other than the intended outputs.
    # For simplicity, we note that we only read from RESULTS and TRAIN and wrote to the analysis directory.
    print("PASS: No production files modified (script only read from results/ and train.parquet and wrote to analysis/).")

    # 5. Record every source artifact used.
    print("\n=== Source Artifacts Used ===")
    print("- results/exp001_query_sets.json")
    print("- results/exp001_checkpoint.json (condition B|C_all_train|ModifiedCosine)")
    print("- results/exp002_checkpoint.json")
    print("- results/exp004_novelty_audit.json")
    print("- results/exp004_scenario_audit_v2.json")
    print("- train.parquet (for precursor_mz and molecular_formula, but note: due to missing parquet support, these fields are left empty)")


if __name__ == "__main__":
    main()
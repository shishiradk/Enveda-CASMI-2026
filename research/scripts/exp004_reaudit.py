"""EXP-004 re-audit: corrected, reproducible Scenario classification for the
EXP-001 Mode-B / EXP-004 397-query population, plus the 12-query reachability
ledger.

Why this script exists (see research/05_class1_to_class2_design.md corrections,
2026-09-21): the original `results/exp004_novelty_audit.json` same-adduct /
cross-adduct classification was found to be WRONG for at least one directly
verified case (rid 468783, true molecule KTEOPAKTYLYZOB): the audit reports
`any_same_adduct_retained: False`, but the molecule's only retained evidence
(rid 468784) is provably the SAME adduct ([M-H]-) as the query, confirmed by a
direct query against train.parquet. This script recomputes the same/cross-adduct
flags directly from train.parquet (not from whatever intermediate join produced
the original audit), and cross-references against EXP-001's own
`candidate_gen_hit` ground truth (results/exp001_checkpoint.json) as an
independent check: if a query's ONLY retained evidence is cross-adduct, Variant
A's raw-precursor-mz (+/-0.01 Da) candidate filter cannot structurally recover
it (adduct changes shift precursor mass by amounts >> 0.01 Da), so
candidate_gen_hit must be False. Any query where the recomputed audit says
"cross-adduct only" but candidate_gen_hit is True is a remaining audit error
that must be re-checked before use.

Reproducibility note: `rid` in every results/*.json file in this project is NOT
a real column in train.parquet -- it is `ROW_NUMBER() OVER () - 1` over a scan
of train.parquet. duckdb's default multi-threaded scan does not guarantee a
stable row order (already documented as a caveat in
research/02_class1_coverage.md Sec.16 point 3). This script uses
`PRAGMA threads=1` and verifies its rid reconstruction against three known
(rid, inchikey14, adduct) triples from existing results files before doing
anything else -- if that check fails, every rid-keyed number in this project's
existing results is at risk and this script aborts rather than silently
producing an unreproducible artifact.
"""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"

# (rid, expected inchikey14, expected adduct) -- from exp001_checkpoint.json /
# exp001_query_sets.json, used only to verify the rid reconstruction is correct
# before trusting any number this script produces.
RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
    (468783, "KTEOPAKTYLYZOB", "[M-H]-"),
]


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


def round_ce(ce_list):
    if ce_list is None:
        return None
    try:
        return tuple(sorted(round(float(x), 1) for x in ce_list))
    except TypeError:
        return None


def main():
    con = connect()
    verify_rid_reconstruction(con)

    qsets = json.load(open(ROOT / "results" / "exp001_query_sets.json"))
    mode_b = qsets["mode_b_queries"]
    print(f"Loaded {len(mode_b)} Mode-B queries.")

    ckpt = json.load(open(ROOT / "results" / "exp001_checkpoint.json"))
    cond = ckpt["conditions"]["B|C_all_train|ModifiedCosine"]
    candgen = {q["rid"]: q["candidate_gen_hit"] for q in cond["per_query_diagnostics"]}

    old_audit = {a["rid"]: a for a in json.load(open(ROOT / "results" / "exp004_novelty_audit.json"))}

    con.execute(
        f"""
        CREATE TEMP TABLE train_rid AS
        SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct, precursor_mz,
               num_peaks, collision_energy_ev
        FROM read_parquet('{TRAIN.as_posix()}')
        """
    )
    print("Materialized train_rid temp table.")

    results = []
    bug_flags = []
    for q in mode_b:
        rid = q["rid"]
        true_ikey = q["true_inchikey14"]
        excluded = set(q["excluded_rids"])

        query_row = con.execute(
            f"SELECT adduct, collision_energy_ev FROM train_rid WHERE rid = {rid}"
        ).fetchdf()
        query_adduct = query_row.iloc[0]["adduct"]
        query_ce = round_ce(query_row.iloc[0]["collision_energy_ev"])

        retained = con.execute(
            f"""
            SELECT rid, adduct, collision_energy_ev FROM train_rid
            WHERE inchikey14 = '{true_ikey}' AND rid NOT IN ({','.join(map(str, excluded)) or '-1'})
            """
        ).fetchdf()

        n_retained_rows = len(retained)
        retained_adducts = set(retained["adduct"]) if n_retained_rows else set()
        any_same_adduct = query_adduct in retained_adducts
        any_cross_adduct = any(a != query_adduct for a in retained_adducts)
        all_same_adduct = n_retained_rows > 0 and retained_adducts == {query_adduct}

        same_adduct_rows = retained[retained["adduct"] == query_adduct]
        same_adduct_ce_match = None
        if len(same_adduct_rows):
            ces = [round_ce(x) for x in same_adduct_rows["collision_energy_ev"]]
            same_adduct_ce_match = query_ce in ces

        rec = {
            "rid": rid,
            "true_inchikey14": true_ikey,
            "query_adduct": query_adduct,
            "n_retained_rows": n_retained_rows,
            "any_same_adduct_retained": any_same_adduct,
            "any_cross_adduct_retained": any_cross_adduct,
            "all_same_adduct_retained": all_same_adduct,
            "cross_adduct_only": any_cross_adduct and not any_same_adduct,
            "same_adduct_ce_match": same_adduct_ce_match,
            "candidate_gen_hit_variantA": candgen.get(rid),
        }
        results.append(rec)

        old = old_audit.get(rid)
        if old is not None and old["any_same_adduct_retained"] != any_same_adduct:
            bug_flags.append(rid)

    # --- summary ---
    n = len(results)
    n_cross_only = sum(r["cross_adduct_only"] for r in results)
    n_any_cross = sum(r["any_cross_adduct_retained"] for r in results)
    n_any_same = sum(r["any_same_adduct_retained"] for r in results)
    print(f"\n=== Corrected Scenario audit (n={n}) ===")
    print(f"any_same_adduct_retained: {n_any_same}")
    print(f"any_cross_adduct_retained: {n_any_cross}")
    print(f"cross_adduct_only (no same-adduct evidence at all): {n_cross_only}")
    print(f"queries where old audit's any_same_adduct_retained disagreed with recomputed: {len(bug_flags)}")
    if bug_flags:
        print(f"  first few disagreeing rids: {bug_flags[:10]}")

    # --- reachability contradiction check ---
    cross_only_recs = [r for r in results if r["cross_adduct_only"]]
    hit_true = sum(1 for r in cross_only_recs if r["candidate_gen_hit_variantA"])
    hit_false = sum(1 for r in cross_only_recs if not r["candidate_gen_hit_variantA"])
    print(f"\nOf {len(cross_only_recs)} corrected cross-adduct-only queries: "
          f"candidate_gen_hit True (should be impossible if correct) = {hit_true}, "
          f"False (expected) = {hit_false}")

    # --- 12-query reachability ledger (EXP-001 Variant-A candidate-absent queries) ---
    absent_rids = [rid for rid, hit in candgen.items() if not hit]
    absent_rids.sort()
    ledger = []
    by_rid = {r["rid"]: r for r in results}
    for rid in absent_rids:
        r = by_rid.get(rid)
        ledger.append(r)
    print(f"\n=== 12-query reachability ledger (n={len(ledger)}) ===")
    for r in ledger:
        print(r)

    out = {
        "generated_by": "research/scripts/exp004_reaudit.py",
        "rid_reconstruction": "ROW_NUMBER() OVER () - 1 with PRAGMA threads=1, verified against 3 known (rid, inchikey14, adduct) triples",
        "n_queries": n,
        "n_any_same_adduct_retained": n_any_same,
        "n_any_cross_adduct_retained": n_any_cross,
        "n_cross_adduct_only": n_cross_only,
        "n_audit_v1_disagreements": len(bug_flags),
        "audit_v1_disagreement_rids": bug_flags,
        "reachability_contradiction_check": {
            "cross_adduct_only_n": len(cross_only_recs),
            "candidate_gen_hit_true_despite_cross_only": hit_true,
            "candidate_gen_hit_false_as_expected": hit_false,
        },
        "reachability_ledger_12": ledger,
        "per_query": results,
    }
    out_path = ROOT / "results" / "exp004_scenario_audit_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

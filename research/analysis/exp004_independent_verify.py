"""Independent verification of the claims EXP-004's revised design relies on.

READ-ONLY. Does not modify the production pipeline, results, or any experiment
artifact. Re-derives every number from persisted artifacts + train.parquet and
adjudicates between results/exp004_novelty_audit.json (v1) and
results/exp004_scenario_audit_v2.json (v2), neither of which is trusted a priori.

Outputs a single JSON blob to stdout (result sections) and optionally to
research/analysis/exp004_independent_verify_out.json.
"""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"
RESULTS = ROOT / "results"

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


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    for rid, ikey, add in SANITY:
        r = con.execute(
            f"""SELECT inchikey14, adduct FROM (
                SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct
                FROM read_parquet('{TRAIN.as_posix()}')) WHERE rid={rid}"""
        ).fetchdf()
        assert not r.empty and r.iloc[0]["inchikey14"] == ikey and r.iloc[0]["adduct"] == add, (rid, r)
    print("rid reconstruction OK (threads=1, 3 sanity triples)", file=sys.stderr)

    con.execute(
        f"""CREATE TEMP TABLE train_rid AS
        SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct, precursor_mz,
               num_peaks, collision_energy_ev, instrument_type
        FROM read_parquet('{TRAIN.as_posix()}')"""
    )

    qsets = json.load(open(RESULTS / "exp001_query_sets.json"))
    mode_b = qsets["mode_b_queries"]
    ckpt = json.load(open(RESULTS / "exp001_checkpoint.json"))
    candgen = {q["rid"]: q["candidate_gen_hit"] for q in ckpt["conditions"]["B|C_all_train|ModifiedCosine"]["per_query_diagnostics"]}
    v1 = {a["rid"]: a for a in json.load(open(RESULTS / "exp004_novelty_audit.json"))}
    v2raw = json.load(open(RESULTS / "exp004_scenario_audit_v2.json"))
    v2 = {r["rid"]: r for r in v2raw["per_query"]}

    def norm_ce(v):
        if v is None:
            return None
        try:
            return tuple(sorted(round(float(x), 1) for x in v))
        except TypeError:
            return None

    mine = []
    for q in mode_b:
        rid = q["rid"]
        ikey = q["true_inchikey14"]
        excl = set(q["excluded_rids"])
        qrow = con.execute(f"SELECT adduct, collision_energy_ev FROM train_rid WHERE rid={rid}").fetchdf()
        qadd = qrow.iloc[0]["adduct"]
        qce = norm_ce(qrow.iloc[0]["collision_energy_ev"])
        ret = con.execute(
            f"""SELECT rid, adduct, collision_energy_ev FROM train_rid
                WHERE inchikey14='{ikey}' AND rid NOT IN ({','.join(map(str, excl)) or '-1'})"""
        ).fetchdf()
        n_ret = len(ret)
        adds = set(ret["adduct"]) if n_ret else set()
        any_same = qadd in adds
        any_cross = any(a != qadd for a in adds)
        all_same = n_ret > 0 and adds == {qadd}
        same_rows = ret[ret["adduct"] == qadd]
        same_ces = [norm_ce(x) for x in same_rows["collision_energy_ev"]] if len(same_rows) else []
        ce_match = (qce in same_ces) if len(same_rows) else None
        mine.append({
            "rid": rid, "true_inchikey14": ikey, "query_adduct": qadd,
            "n_retained_rows": n_ret, "retained_adducts": sorted(adds),
            "any_same_adduct_retained": bool(any_same),
            "any_cross_adduct_retained": bool(any_cross),
            "all_same_adduct_retained": bool(all_same),
            "cross_adduct_only": bool(any_cross and not any_same),
            "same_adduct_ce_match": ce_match,
            "query_ce": qce,
            "query_ce_null": qce is None,
            "same_adduct_ces": [list(x) if x is not None else None for x in same_ces],
            "same_adduct_all_ce_null": (len(same_ces) > 0 and all(x is None for x in same_ces)),
            "candidate_gen_hit_variantA": candgen.get(rid),
        })
    mine_by = {r["rid"]: r for r in mine}
    n = len(mine)

    # --- three-way adjudication ---
    d_v1_mine = [r["rid"] for r in mine if v1[r["rid"]]["any_same_adduct_retained"] != r["any_same_adduct_retained"]]
    d_v2_mine = [r["rid"] for r in mine if v2[r["rid"]]["any_same_adduct_retained"] != r["any_same_adduct_retained"]]

    def census(flags_source, getf):
        s = sum(getf(e) for e in flags_source)
        return s

    census_out = {
        "mine": {
            "any_same": sum(r["any_same_adduct_retained"] for r in mine),
            "any_cross": sum(r["any_cross_adduct_retained"] for r in mine),
            "all_same": sum(r["all_same_adduct_retained"] for r in mine),
            "cross_adduct_only": sum(r["cross_adduct_only"] for r in mine),
        },
        "v1_audit": {
            "any_same": sum(a["any_same_adduct_retained"] for a in v1.values()),
            "any_cross": sum(a["any_cross_adduct_retained"] for a in v1.values()),
            "all_same": sum(a["all_same_adduct_retained"] for a in v1.values()),
        },
        "v2_audit": {
            "any_same": sum(r["any_same_adduct_retained"] for r in v2.values()),
            "any_cross": sum(r["any_cross_adduct_retained"] for r in v2.values()),
            "all_same": sum(r["all_same_adduct_retained"] for r in v2.values()),
            "cross_adduct_only": sum(r["cross_adduct_only"] for r in v2.values()),
        },
    }

    # A/B split under row-level CE rule
    all_same_recs = [r for r in mine if r["all_same_adduct_retained"]]
    A = [r for r in all_same_recs if r["same_adduct_ce_match"] is True]
    B = [r for r in all_same_recs if r["same_adduct_ce_match"] is False]
    ce_na = [r for r in all_same_recs if r["same_adduct_ce_match"] is None]

    # CE availability / reproducibility detail
    ce_stats = {
        "train_total_rows": con.execute("SELECT count(*) FROM train_rid").fetchone()[0],
        "train_ce_null_rows": con.execute("SELECT count(*) FROM train_rid WHERE collision_energy_ev IS NULL").fetchone()[0],
        "v1_any_different_ce_retained_n": sum(1 for a in v1.values() if a["any_different_ce_retained"]),
        "all_same_query_ce_null_n": sum(1 for r in all_same_recs if r["query_ce_null"]),
        "all_same_all_retained_ce_null_n": sum(1 for r in all_same_recs if r["same_adduct_all_ce_null"]),
        "A_or_B_defined": len(A) + len(B),
        "A_rids": [r["rid"] for r in A],
    }
    ce_stats["ce_null_pct"] = round(100.0 * ce_stats["train_ce_null_rows"] / ce_stats["train_total_rows"], 2)

    # reachability cross-check
    nohit = sorted([rid for rid, h in candgen.items() if not h])
    cross_only = sorted([r["rid"] for r in mine if r["cross_adduct_only"]])
    reach_out = {
        "candgen_nohit_n": len(nohit), "candgen_nohit": nohit,
        "your_cross_only_n": len(cross_only),
        "your_cross_only_equals_nohit": cross_only == nohit,
        "nohit_not_cross_only": sorted(set(nohit) - set(cross_only)),
        "cross_only_not_nohit": sorted(set(cross_only) - set(nohit)),
    }

    # 12-query ledger
    ledger = []
    for rid in nohit:
        m = mine_by[rid]
        e = {"rid": rid, "true_inchikey14": m["true_inchikey14"], "query_adduct": m["query_adduct"],
             "n_retained_rows": m["n_retained_rows"], "retained_adducts": m["retained_adducts"],
             "cross_adduct_only": m["cross_adduct_only"],
             "v1_any_same": v1[rid]["any_same_adduct_retained"],
             "v2_any_same": v2[rid]["any_same_adduct_retained"]}
        ledger.append(e)

    # --- EXP-003 (Q4) ---
    e3 = list(json.load(open(RESULTS / "exp003_inspection.json")).values())
    q4 = {"n": len(e3)}
    same_disp = cross_disp = 0
    disp_new_b = disp_in_a = 0
    correct_same = 0
    correct_in_a = 0
    disp_same_as_correct = 0
    for e in e3:
        if e.get("correct_is_same_adduct"):
            correct_same += 1
        if e.get("correct_candidate_was_in_variant_a"):
            correct_in_a += 1
        tw = e.get("top_wrong_overall")
        if tw and tw.get("adduct") == e.get("correct_adduct"):
            disp_same_as_correct += 1
        if tw:
            if tw.get("adduct") == e.get("query_adduct"):
                same_disp += 1
            else:
                cross_disp += 1
            if tw.get("was_in_variant_a") is True:
                disp_in_a += 1
            elif tw.get("was_in_variant_a") is False:
                disp_new_b += 1
    q4.update({"correct_is_same_adduct": correct_same,
               "correct_candidate_was_in_variant_a": correct_in_a,
               "top_wrong_same_adduct": same_disp, "top_wrong_cross_adduct": cross_disp,
               "top_wrong_in_variant_a": disp_in_a, "top_wrong_new_in_b": disp_new_b,
               "top_wrong_adduct_eq_correct_adduct": disp_same_as_correct})

    # --- Q5 margin feasibility ---
    margins = []
    for e in e3:
        cs = e.get("correct_score")
        tw = e.get("top_wrong_overall")
        if cs is not None and tw and tw.get("score") is not None:
            margins.append(float(cs) - float(tw["score"]))
    margins.sort()
    q5 = {
        "exp003_subset_n": len(e3), "margins_computable_n": len(margins),
        "median_margin": nd(margins[len(margins) // 2]) if margins else None,
        "min": nd(margins[0]) if margins else None, "max": nd(margins[-1]) if margins else None,
        "n_negative_margin": sum(1 for m in margins if m < 0),
        "full_397_margin_persisted": False,
        "why": "exp001_checkpoint per_query_diagnostics has no scores; exp002 per_query_results has only best_true_molecule_score_b (no per-candidate/wrong-molecule scores). Only exp003_inspection (92 selected queries) has correct_score + top_wrong score.",
    }

    out = {
        "rid_reconstruction": "ROW_NUMBER() OVER () - 1, PRAGMA threads=1, verified vs 3 triples",
        "n_queries": n,
        "adjudication": {
            "v1_vs_mine_disagreements_on_any_same": len(d_v1_mine),
            "v1_disagreement_rids_first20": d_v1_mine[:20],
            "v2_vs_mine_disagreements_on_any_same": len(d_v2_mine),
            "v2_disagreement_rids": d_v2_mine,
        },
        "census": census_out,
        "scenario_ce": {
            "all_same_adduct_n": len(all_same_recs),
            "scenario_A_ce_match": len(A), "scenario_B_ce_no_match": len(B),
            "ce_match_undefined": len(ce_na),
            "ce_stats": ce_stats,
        },
        "reachability": reach_out,
        "ledger_12": ledger,
        "exp003_q4": q4,
        "q5_margin": q5,
    }
    json.dump(out, open(ROOT / "research" / "analysis" / "exp004_independent_verify_out.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

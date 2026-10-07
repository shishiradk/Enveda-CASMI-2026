"""EXP-004 Q5: margin analysis.

Margin := score(correct molecule) - score(top-ranking wrong molecule).

FEASIBILITY (verified against persisted artifacts):
- exp001_checkpoint.json per_query_diagnostics: NO scores (only rank/hit/adduct).
- exp002_checkpoint.json per_query_results: only best_true_molecule_score_b
  (correct molecule's best score under Variant B); no per-candidate or
  wrong-molecule scores.
- exp003_inspection.json (92 outcome-selected queries): correct_score +
  top_wrong_overall.score ARE persisted -> margins computable for this subset.
Therefore full-397 margins cannot be extracted from saved artifacts; computing
them requires a read-only re-scoring pass (Variant A candidate lists are not
persisted either). This script extracts the 92-query subset margins.
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def quantiles_sorted(vals):
    n = len(vals)
    if n == 0:
        return {}
    if n == 1:
        return {"min": vals[0], "p25": vals[0], "median": vals[0], "p75": vals[0], "max": vals[0]}
    qs = statistics.quantiles(vals, n=100)
    return {
        "min": vals[0],
        "p10": qs[9], "p25": qs[24], "median": qs[49],
        "p75": qs[74], "p90": qs[89], "max": vals[-1],
    }


def main():
    insp = json.load(open(RESULTS / "exp003_inspection.json"))
    records = []
    for rid_s, e in insp.items():
        cs = e.get("correct_score")
        tw = e.get("top_wrong_overall")
        if cs is None or not tw or tw.get("score") is None:
            continue
        margin = cs - tw["score"]
        records.append({
            "rid": int(rid_s),
            "query_adduct": e.get("query_adduct"),
            "correct_score": cs,
            "top_wrong_adduct": tw.get("adduct"),
            "top_wrong_was_in_variant_a": tw.get("was_in_variant_a"),
            "margin": margin,
        })

    marg = sorted(r["margin"] for r in records)
    neg = sum(1 for m in marg if m < 0)

    by_adduct = {"same": [], "cross": []}
    by_src = {"in_a": [], "new_in_b": []}
    for r in records:
        cls = "same" if r["top_wrong_adduct"] == r["query_adduct"] else "cross"
        by_adduct[cls].append(r["margin"])
        cls2 = "in_a" if r["top_wrong_was_in_variant_a"] else "new_in_b"
        by_src[cls2].append(r["margin"])

    out = {
        "feasibility": {
            "full_397_margin_persisted": False,
            "why": "No persisted artifact stores per-candidate scores for the full population. "
                   "exp001_checkpoint per_query_diagnostics has no scores; exp002_checkpoint "
                   "per_query_results has only best_true_molecule_score_b. Only "
                   "exp003_inspection.json (92 queries) stores correct_score + top_wrong_overall.score. "
                   "Variant A candidate lists are also not persisted.",
            "needed_for_full": "Read-only re-scoring pass or a future run that persists "
                               "top-25 (molecule, score) lists per query.",
        },
        "exp003_subset": {
            "n": len(records),
            "n_negative_margin": neg,
            "all_negative": neg == len(records),
            "margin_stats": {k: round(v, 6) for k, v in quantiles_sorted(marg).items()},
            "by_top_wrong_adduct_vs_query": {
                "same_adduct": {k: round(v, 6) for k, v in quantiles_sorted(sorted(by_adduct["same"])).items()},
                "cross_adduct": {k: round(v, 6) for k, v in quantiles_sorted(sorted(by_adduct["cross"])).items()},
            },
            "by_top_wrong_source": {
                "already_in_variant_a": {k: round(v, 6) for k, v in quantiles_sorted(sorted(by_src["in_a"])).items()},
                "newly_admitted_by_b": {k: round(v, 6) for k, v in quantiles_sorted(sorted(by_src["new_in_b"])).items()},
            },
        },
    }
    json.dump(out, open(ROOT / "research" / "analysis" / "exp004_margin_analysis_out.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
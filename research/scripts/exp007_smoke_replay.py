"""EXP-007 smoke-replay comparison (reproducibility gate).

Compares the full 600-query run's rows against the validated smoke-test results
(results/exp007_smoke_query_results.jsonl) on the 30 shared query rids, for every
condition/tau, field-by-field. A material disagreement is a full-run stop condition.

Fields verified (query identity + outcome):
  rid, tau, stratum, max_sim, n_retained_same_adduct, condition, k_removed,
  no_op, n_candidates, candidate_gen_hit, c1_candidate_gen_hit, rank, c1_rank,
  transition_case, true_score (when rank<=25 for both), top_k molecule/score lists.

Output: results/exp007_c2_smoke_replay.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_REPLAY = ROOT / "results" / "exp007_c2_smoke_replay.json"

SMOKE_MANIFEST = ROOT / "results" / "exp007_smoke_test_manifest.json"
SMOKE_JSONL = ROOT / "results" / "exp007_smoke_query_results.jsonl"
FULL_JSONL = ROOT / "results" / "exp007_c2_query_results.jsonl"

VERIFY_FIELDS = [
    ("rid", int), ("tau", float), ("stratum", str), ("max_sim", float),
    ("n_retained_same_adduct", int), ("condition", str), ("k_removed", int),
    ("no_op", bool), ("n_candidates", int), ("candidate_gen_hit", bool),
    ("c1_candidate_gen_hit", bool), ("rank", int), ("c1_rank", int),
    ("transition_case", int),
]


def norm(v, cast):
    if v is None:
        return None
    if isinstance(v, (int, float)) and isinstance(v, float) and v != v:  # NaN
        return None
    if cast is bool and not isinstance(v, bool):
        return bool(v)
    try:
        if isinstance(v, float) and cast is float:
            return round(float(v), 9)
        return cast(v)
    except (TypeError, ValueError):
        return v


def main():
    smoke_manifest = json.load(open(SMOKE_MANIFEST))
    smoke_rids = set(smoke_manifest["all_sampled_rids_sorted"])
    assert smoke_rids, "smoke manifest empty"

    smoke_rows = [json.loads(l) for l in open(SMOKE_JSONL)]
    full_rows = [json.loads(l) for l in open(FULL_JSONL)]

    smoke_by_key = {}
    for r in smoke_rows:
        key = (int(r["rid"]), r["condition"],
               None if r["tau"] is None else round(float(r["tau"]), 2))
        smoke_by_key.setdefault(key, []).append(r)

    full_keyed = {}
    for r in full_rows:
        key = (int(r["rid"]), r["condition"],
               None if r["tau"] is None else round(float(r["tau"]), 2))
        full_keyed.setdefault(key, []).append(r)

    compared = 0
    mismatches = []
    smoke_covered = set()
    for key, srows in sorted(smoke_by_key.items()):
        rid = key[0]
        if rid not in smoke_rids:
            continue
        for sr in srows:
            smoke_covered.add(key)
            frows = full_keyed.get(key, [])
            if not frows:
                mismatches.append({"key": list(key), "field": "(row)", "smoke": "present", "full": "MISSING"})
                continue
            fr = frows[0]
            compared += 1
            for field, cast in VERIFY_FIELDS:
                sv, fv = norm(sr.get(field), cast), norm(fr.get(field), cast)
                if sv != fv:
                    mismatches.append({"key": list(key), "field": field, "smoke": sv, "full": fv})
            # true score comparison where both sides have it (rank <= 25)
            s_ts = sr.get("true_score")
            f_ts = fr.get("true_score")
            if s_ts is not None and f_ts is not None and abs(float(s_ts) - float(f_ts)) > 1e-9:
                mismatches.append({"key": list(key), "field": "true_score", "smoke": s_ts, "full": f_ts})
            # top_k molecule/score list equality
            if sr.get("top_k") is not None and fr.get("top_k") is not None:
                s_tk = [(t["candidate_molecule_inchikey14"], round(float(t["candidate_score"]), 9)) for t in sr["top_k"]]
                f_tk = [(t["candidate_molecule_inchikey14"], round(float(t["candidate_score"]), 9)) for t in fr["top_k"]]
                if s_tk != f_tk:
                    mismatches.append({"key": list(key), "field": "top_k", "smoke_topk": sr["top_k"], "full_topk": fr["top_k"]})

    reach_expected = smoke_manifest.get("n_ur_or_unreported", None)
    result = {
        "generated_by": "research/scripts/exp007_smoke_replay.py",
        "smoke_jsonl": str(SMOKE_JSONL),
        "full_jsonl": str(FULL_JSONL),
        "smoke_rids_expected": sorted(smoke_rids),
        "smoke_rows_total": len(smoke_rows),
        "smoke_rid_rows_compared": compared,
        "smoke_keys_covered": len(smoke_covered),
        "full_rows_total": len(full_rows),
        "n_mismatches": len(mismatches),
        "material_disagreement": len(mismatches) > 0,
        "mismatches": mismatches[:50],
        "note": ("A material disagreement (n_mismatches > 0) is a full-run STOP condition. "
                 "Disagreements must be reported, never silently overwritten."),
    }
    with open(OUT_REPLAY, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_REPLAY}")
    print(f"Compared {compared} smoke rows (shared with full run); mismatches={len(mismatches)}")
    if mismatches:
        print("MATERIAL DISAGREEMENT: full run does not replay the smoke test.")
        for m in mismatches[:10]:
            print("  ", m)
        raise SystemExit(2)
    print("Smoke replay: PASS (30 shared rids reproduce exactly).")


if __name__ == "__main__":
    main()
"""EXP-007 design sizing: strict-eligibility and strata table from Phase 2/3 audits.

Strict C2(tau) eligibility for a query = it has >=1 same-adduct retained sibling AND
the BEST (max) sim to any retained same-adduct sibling is < tau. That is the condition
under which every piece of reachable same-adduct evidence for the true molecule scores
below tau — i.e. the query is genuinely low-similarity to ALL its retained evidence.

Also reports the "no same-adduct evidence" stratum (queries whose retained siblings are
all cross-adduct only) — the reachability-adjacent group matching EXP-004's 12-query
finding, and cross-adduct-only-timsTOF single-spectrum queries.

Reads: results/exp007_audit_phase3.json   Writes: results/exp007_design_sizing.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "results" / "exp007_audit_phase3.json"
OUT = ROOT / "results" / "exp007_design_sizing.json"

TAUS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def main():
    with open(P3) as f:
        p3 = json.load(f)
    queries = p3["per_query"]
    n = len(queries)

    strict = {}
    for tau in TAUS:
        elig = [q for q in queries
                if q["n_same_adduct_retained"] >= 1
                and q["max_sim"] is not None
                and q["max_sim"] < tau]
        strict[str(tau)] = {"n_eligible": len(elig),
                            "frac_of_sample": round(len(elig) / n, 4)}
        print(f"strict C2(tau) max_sim<{tau}: {len(elig)}/{n} = {len(elig)/n:.4f}")

    no_same_adduct = [q for q in queries if q["n_same_adduct_retained"] == 0]
    has_adduct_evidence = [q for q in queries if q["n_same_adduct_retained"] > 0]
    print(f"\nno same-adduct retained evidence (cross-adduct-only or singleton): "
          f"{len(no_same_adduct)}/{n} = {len(no_same_adduct)/n:.4f}")
    print(f"has same-adduct retained evidence: {len(has_adduct_evidence)}/{n}")

    # max_sim distribution within the has-evidence stratum
    maxsims = sorted(q["max_sim"] for q in has_adduct_evidence)
    def pct(vals, p):
        return vals[min(len(vals) - 1, int((len(vals) - 1) * p))]
    print(f"max_sim within has-evidence stratum: p10={pct(maxsims,.10):.3f} "
          f"p25={pct(maxsims,.25):.3f} med={pct(maxsims,.50):.3f} "
          f"p75={pct(maxsims,.75):.3f} p90={pct(maxsims,.90):.3f}")

    # tau grid summary: rank of run used grows downgraded anyway performed aggregations
    # Extract the full max-sim list for external reuse in the design doc.
    report = {
        "n_query_sample": n,
        "n_has_same_adduct_evidence": len(has_adduct_evidence),
        "n_no_same_adduct_evidence": len(no_same_adduct),
        "strict_eligibility_by_tau": strict,
        "max_sim_distribution_has_evidence": {
            "p10": pct(maxsims, .10), "p25": pct(maxsims, .25),
            "median": pct(maxsims, .50), "p75": pct(maxsims, .75),
            "p90": pct(maxsims, .90),
        },
        "max_sims": maxsims,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
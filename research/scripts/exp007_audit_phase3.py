"""EXP-007 dataset audit, Phase 3: query-centric C2 eligibility census.

For a deterministic sample of timsTOF spectra treated as candidate C2 QUERIES,
measure, for each query's true molecule:
  - number of other SAME-ADDUCT spectra (the only evidence Variant A can structurally
    reach, per DEC-003/EXP-002), excluding the query's whole metadata group
  - ModifiedCosine similarity from the query to each same-adduct sibling
  - for each threshold tau in a grid: does the query have >=1 same-adduct sibling with
    sim < tau (i.e. is a C2(tau) construction non-empty)?
  - resulting "eligible query" count per tau

Answers the section 18 population-size questions with real spectral scoring on a
sample, so the full-run size in the design doc is evidence-based, not guessed.

rid reconstruction is over the FULL table only (row_number() OVER () - 1 under
PRAGMA threads=1), then filtered by instrument_type in the same pass.

Scorer/config byte-identical to EXP-001/004: ModifiedCosineGreedy(tolerance=0.1,
mz_power=0.0, intensity_power=1.0).

Output: ../../results/exp007_audit_phase3.json
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN = str(ROOT / "train.parquet")
OUT = str(ROOT / "results" / "exp007_audit_phase3.json")

N_QUERY_SAMPLE = 600
SEED = 20260922
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
]


def make_spectrum(mzs, ints, precursor_mz):
    from matchms import Spectrum
    return Spectrum(mz=np.asarray(mzs, dtype=float),
                    intensities=np.asarray(ints, dtype=float),
                    metadata={"precursor_mz": float(precursor_mz)})


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")

    for rid, exp_ikey, exp_adduct in RID_SANITY_CHECKS:
        row = con.execute(
            f"SELECT inchikey14, adduct FROM (SELECT row_number() OVER () - 1 AS rid, "
            f"inchikey14, adduct FROM read_parquet('{TRAIN}')) WHERE rid = {rid}"
        ).fetchdf()
        assert not row.empty and row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct
    print("rid reconstruction verified OK.")

    from matchms.similarity import ModifiedCosineGreedy
    sim_func = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    # Load the full table once with reproducible rids.
    full_t = con.execute(
        f"""
        SELECT rid, inchikey14, adduct, precursor_mz, collision_energy_orig, num_peaks,
               ms2_mzs, ms2_normalized_intensities, instrument_type, ingest_lib
        FROM (SELECT row_number() OVER () - 1 AS rid, *
              FROM read_parquet('{TRAIN}'))
        ORDER BY rid
        """
    ).fetchdf()
    print(f"full rows: {len(full_t)}; timsTOF rows: {(full_t['instrument_type']=='timsTOF').sum()}")

    tims = full_t[full_t["instrument_type"] == "timsTOF"].copy()
    rng = random.Random(SEED)
    tims_rids = tims["rid"].tolist()
    rng.shuffle(tims_rids)
    sample_rids = set(sorted(tims_rids[:N_QUERY_SAMPLE]))

    query_rows = tims[tims["rid"].isin(sample_rids)]
    print(f"sampled {len(query_rows)} timsTOF query spectra")

    # Load all spectra (any instrument / library) for the query molecules.
    q_ikeys = sorted(query_rows["inchikey14"].unique().tolist())
    in_clause = ", ".join(f"'{k}'" for k in q_ikeys)
    siblings = full_t[full_t["inchikey14"].isin(q_ikeys)].copy()
    print(f"sibling rows for query molecules: {len(siblings)}")

    # Group by (inchikey14, adduct, round(precursor,4)) -> same-adduct family
    def family_key(row):
        return (row["inchikey14"], row["adduct"], round(row["precursor_mz"], 4))

    groups = {}
    for idx, row in siblings.iterrows():
        groups.setdefault(family_key(row), []).append(row)

    per_query = []
    t0 = time.time()
    for _, qrow in query_rows.iterrows():
        r = int(qrow["rid"])
        fk = family_key(qrow)
        fam = groups.get(fk, [])
        # The query's whole metadata group: same inchikey14, adduct, round(precursor,4),
        # num_peaks (EXP-001's identical group definition).
        group_rids = {int(s["rid"]) for s in fam if int(s["num_peaks"]) == int(qrow["num_peaks"])}
        retained = [s for s in fam if int(s["rid"]) not in group_rids]

        qspec = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], qrow["precursor_mz"])
        sims = []
        for s in retained:
            sspec = make_spectrum(s["ms2_mzs"], s["ms2_normalized_intensities"], s["precursor_mz"])
            res = sim_func.pair(qspec, sspec)
            sims.append(float(res["score"]))

        rec = {
            "rid": r,
            "inchikey14": qrow["inchikey14"],
            "adduct": qrow["adduct"],
            "n_same_adduct_retained": len(retained),
            "n_from_other_libraries": int(sum(1 for s in retained if s["ingest_lib"] != "enveda-180")),
            "n_diff_ce": int(sum(1 for s in retained
                                 if str(s["collision_energy_orig"]) != str(qrow["collision_energy_orig"]))),
            "sims_to_retained": [round(v, 6) for v in sims],
            "max_sim": max(sims) if sims else None,
            "min_sim": min(sims) if sims else None,
        }
        per_query.append(rec)

    print(f"scored {len(per_query)} queries in {time.time()-t0:.1f}s")

    n_total = len(per_query)
    elig = {}
    for tau in THRESHOLDS:
        below = [q for q in per_query
                 if q["n_same_adduct_retained"] >= 1 and any(s < tau for s in q["sims_to_retained"])]
        elig[str(tau)] = {"n_eligible": len(below), "frac": round(len(below) / n_total, 4)}

    print("\n=== C2(tau) eligibility (>=1 same-adduct retained sibling with sim < tau) ===")
    for k, v in elig.items():
        print(f"  tau={k}: {v['n_eligible']}/{n_total} = {v['frac']}")

    have = [q for q in per_query if q["n_same_adduct_retained"] >= 1]
    dist_n = Counter(q["n_same_adduct_retained"] for q in have)

    max_sims = sorted(q["max_sim"] for q in have if q["max_sim"] is not None)

    def pct(vals, p):
        k = (len(vals) - 1) * p
        f = int(k)
        return vals[f]

    maxsim_dist = {
        "n": len(max_sims), "p25": pct(max_sims, .25), "median": pct(max_sims, .5),
        "p75": pct(max_sims, .75), "p90": pct(max_sims, .9),
    }

    report = {
        "generated_by": "research/scripts/exp007_audit_phase3.py",
        "seed": SEED,
        "n_query_sample": n_total,
        "n_queries_with_same_adduct_retained": len(have),
        "eligibility_by_tau": elig,
        "n_same_adduct_retained_distribution": {str(k): v for k, v in sorted(dist_n.items())},
        "max_sim_distribution": maxsim_dist,
        "per_query": per_query,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
"""EXP-007 dataset audit, Phase 2: empirical within-molecule spectral similarity.

Answers the design-critical question empirically (not assumed):
  - Within a molecule, how similar are its spectra at the SAME adduct but different
    collision energy?  (i.e. is "different CE = different spectrum" actually true?)
  - How similar across adducts?
  - How many molecules have >=2 spectra that are genuinely distinct (well below a
    candidate threshold) at the same adduct -> the raw material for a same-adduct C2?
  - What does the peak-level similarity distribution look like (to justify thresholds
    from data instead of borrowing 0.90/0.95)?

Strategy (efficient sampling, per instruction):
  - Sample groups of molecules from timsTOF (test-relevant) that have >=2 same-adduct
    spectra at different CE, and a separate sample with >=2 distinct adducts.
  - For every molecule in the sample, compute ALL within-molecule pairwise
    ModifiedCosineGreedy similarities (the same scorer/config as EXP-001/004).
  - Tag each pair: same/cross adduct, same/diff CE, same/diff (CE per row).
  - Report distributions + threshold-survival counts.

Scorer config (byte-identical to EXP-001/002/003/004/005):
    matchms 0.33.1 ModifiedCosineGreedy, tolerance=0.1, mz_power=0.0, intensity_power=1.0

Output: ../../results/exp007_audit_phase2.json  + printed summary
Runtime: bounded by sampling sizes below; intended to be cheap.
"""

import json
import random
import time
from pathlib import Path

import duckdb
import numpy as np
from matchms.similarity import ModifiedCosineGreedy

ROOT = Path(__file__).resolve().parents[2]
TRAIN = str(ROOT / "train.parquet")
OUT = str(ROOT / "results" / "exp007_audit_phase2.json")

SAME_ADDUCT_SAMPLE = 300          # molecules with >=2 same-adduct diff-CE spectra
CROSS_ADDUCT_SAMPLE = 150         # molecules with >=2 distinct adducts
SEED = 20260922
MAX_SPECTRA_PER_MOLECULE = 12     # cap pairwise count per molecule for the audit

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
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
                FROM read_parquet('{TRAIN}')
            ) WHERE rid = {rid}
            """
        ).fetchdf()
        assert not row.empty and row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct
    print(f"rid reconstruction verified OK against {len(RID_SANITY_CHECKS)} known cases.")


def sample_molecules(con, where_sql, n):
    """Deterministic sample of n inchikey14 values satisfying where_sql, from timsTOF."""
    rng = random.Random(SEED)
    ikeys = [r[0] for r in con.execute(
        f"SELECT inchikey14 FROM (SELECT inchikey14 FROM read_parquet('{TRAIN}') "
        f"WHERE instrument_type='timsTOF' GROUP BY inchikey14 {where_sql}) ORDER BY inchikey14"
    ).fetchall()]
    rng.shuffle(ikeys)
    return sorted(set(ikeys[:n]))


def load_spectra_for_molecules(con, ikeys, cap=None):
    """Load spectra (rid, ikey, adduct, prec, ce, mzs, ints) for target molecules.
    Uses reproducible rid via row_number() OVER () - 1 under threads=1."""
    in_clause = ", ".join(f"'{k}'" for k in ikeys)
    df = con.execute(
        f"""
        SELECT rid, inchikey14, adduct, precursor_mz, collision_energy_orig, num_peaks,
               ms2_mzs, ms2_normalized_intensities, ingest_lib
        FROM (
            SELECT row_number() OVER () - 1 AS rid, *
            FROM read_parquet('{TRAIN}')
        )
        WHERE inchikey14 IN ({in_clause})
        ORDER BY inchikey14, adduct, precursor_mz, collision_energy_orig
        """
    ).fetchdf()
    grouped = {}
    for _, row in df.iterrows():
        grouped.setdefault(row["inchikey14"], []).append(row)
    return grouped


def make_spectrum(mzs, ints, precursor_mz):
    from matchms import Spectrum
    return Spectrum(mz=np.asarray(mzs, dtype=float),
                    intensities=np.asarray(ints, dtype=float),
                    metadata={"precursor_mz": float(precursor_mz)})


def score_pairs(rows, sim_func):
    """All within-molecule pairs among rows (capped), score with ModifiedCosineGreedy.
    Returns list of dicts: pair_a, pair_b, sim, same_adduct, same_ce, adduct_a,
    adduct_b, ce_a, ce_b, num_peaks_a/b, source_a/b."""
    out = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]
            sa = make_spectrum(a["ms2_mzs"], a["ms2_normalized_intensities"], a["precursor_mz"])
            sb = make_spectrum(b["ms2_mzs"], b["ms2_normalized_intensities"], b["precursor_mz"])
            res = sim_func.pair(sa, sb)
            out.append({
                "inchikey14": a["inchikey14"],
                "sim": float(res["score"]),
                "same_adduct": a["adduct"] == b["adduct"],
                "same_ce": a["collision_energy_orig"] == b["collision_energy_orig"],
                "adduct_a": a["adduct"], "adduct_b": b["adduct"],
                "ce_a": str(a["collision_energy_orig"]), "ce_b": str(b["collision_energy_orig"]),
                "num_peaks_a": int(a["num_peaks"]), "num_peaks_b": int(b["num_peaks"]),
                "source_a": a["ingest_lib"], "source_b": b["ingest_lib"],
                "rid_a": int(a["rid"]), "rid_b": int(b["rid"]),
            })
    return out


def main():
    con = connect()
    verify_rid_reconstruction(con)

    sim_func = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    all_pairs = []
    summaries = {}

    # --- Sample 1: same-adduct, diff-CE multispectra molecules ---
    print(f"\nSampling {SAME_ADDUCT_SAMPLE} molecules with >=2 same-adduct spectra at different CE...")
    ikeys_same = sample_molecules(con, "HAVING count(*) >= 2", SAME_ADDUCT_SAMPLE)
    # refine: ensure at least one same-adduct pair actually exists with diff CE
    groups_same = load_spectra_for_molecules(con, ikeys_same, MAX_SPECTRA_PER_MOLECULE)
    sampled = 0
    pairs_same = []
    for ikey in ikeys_same:
        rows = groups_same[ikey][:MAX_SPECTRA_PER_MOLECULE]
        ps = score_pairs(rows, sim_func)
        if any(p["same_adduct"] and not p["same_ce"] for p in ps):
            pairs_same.extend(ps)
            sampled += 1
    summaries["sample1_same_adduct_diff_ce"] = {
        "target": SAME_ADDUCT_SAMPLE, "sampled_molecules_with_same_adduct_diff_ce": sampled,
        "n_pairs": len(pairs_same), "n_same_adduct_pairs": sum(1 for p in pairs_same if p["same_adduct"]),
        "n_same_adduct_diff_ce_pairs": sum(1 for p in pairs_same if p["same_adduct"] and not p["same_ce"]),
    }
    print(json.dumps(summaries["sample1_same_adduct_diff_ce"], indent=2))
    all_pairs.extend(pairs_same)

    # --- Sample 2: multi-adduct molecules ---
    print(f"\nSampling {CROSS_ADDUCT_SAMPLE} molecules with >=2 distinct adducts...")
    ikeys_multi = sample_molecules(con, "HAVING count(DISTINCT adduct) >= 2", CROSS_ADDUCT_SAMPLE)
    groups_multi = load_spectra_for_molecules(con, ikeys_multi, MAX_SPECTRA_PER_MOLECULE)
    pairs_multi = []
    cross_sampled = 0
    for ikey in ikeys_multi:
        rows = groups_multi[ikey][:MAX_SPECTRA_PER_MOLECULE]
        ps = score_pairs(rows, sim_func)
        if any(not p["same_adduct"] for p in ps):
            pairs_multi.extend(ps)
            cross_sampled += 1
    summaries["sample2_cross_adduct"] = {
        "target": CROSS_ADDUCT_SAMPLE, "sampled_molecules_with_cross_adduct_pairs": cross_sampled,
        "n_pairs": len(pairs_multi),
        "n_cross_adduct_pairs": sum(1 for p in pairs_multi if not p["same_adduct"]),
    }
    print(json.dumps(summaries["sample2_cross_adduct"], indent=2))
    all_pairs.extend(pairs_multi)

    def dist(vals):
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        def pct(p):
            k = (n - 1) * p
            f = int(k)
            return s[f]
        return {"n": n, "min": s[0], "p10": pct(.10), "p25": pct(.25), "median": pct(.5),
                "p75": pct(.75), "p90": pct(.9), "p95": pct(.95), "max": s[-1]}

    dists = {
        "all_pairs": dist([p["sim"] for p in all_pairs]),
        "same_adduct": dist([p["sim"] for p in all_pairs if p["same_adduct"]]),
        "same_adduct_diff_ce": dist([p["sim"] for p in all_pairs if p["same_adduct"] and not p["same_ce"]]),
        "same_adduct_same_ce": dist([p["sim"] for p in all_pairs if p["same_adduct"] and p["same_ce"]]),
        "cross_adduct": dist([p["sim"] for p in all_pairs if not p["same_adduct"]]),
        "cross_adduct_only_sample": dist([p["sim"] for p in pairs_multi if not p["same_adduct"]]),
    }
    print("\n=== within-molecule pairwise ModifiedCosine distributions ===")
    for label, d in dists.items():
        if d:
            print(f"{label}: n={d['n']} min={d['min']:.4f} p25={d['p25']:.4f} med={d['median']:.4f} "
                  f"p75={d['p75']:.4f} p90={d['p90']:.4f} max={d['max']:.4f}")

    # Threshold survival: fraction of same-adduct-diff-CE pairs below each threshold
    print("\n=== same-adduct diff-CE pair survival below threshold ===")
    sa_diff_ce = [p["sim"] for p in all_pairs if p["same_adduct"] and not p["same_ce"]]
    threshold_report = {}
    if sa_diff_ce:
        for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            c = sum(1 for v in sa_diff_ce if v < thresh)
            threshold_report[str(thresh)] = {"n_below": c, "frac_below": round(c / len(sa_diff_ce), 4)}
            print(f"  <{thresh}: {c}/{len(sa_diff_ce)} = {c/len(sa_diff_ce):.4f}")

    # Molecule-level: how many molecules have a same-adduct pair below threshold?
    print("\n=== molecule-level: any same-adduct pair < threshold (sample 1) ===")
    mol_report = {}
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        counts = {}
        for p in pairs_same:
            if not p["same_adduct"]:
                continue
            counts.setdefault(p["inchikey14"], {"min_sim": 1.0, "below": False})
            if p["sim"] < counts[p["inchikey14"]]["min_sim"]:
                counts[p["inchikey14"]]["min_sim"] = p["sim"]
            if p["sim"] < thresh:
                counts[p["inchikey14"]]["below"] = True
        n_below = sum(1 for v in counts.values() if v["below"])
        mol_report[str(thresh)] = {"n_molecules_with_any_same_adduct_pair_below": n_below,
                                   "of_sampled": len(counts)}
        print(f"  <{thresh}: {n_below}/{len(counts)} molecules have ANY same-adduct pair below threshold")

    report = {
        "generated_by": "research/scripts/exp007_audit_phase2.py",
        "seed": SEED,
        "scorer": "matchms 0.33.1 ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)",
        "sample_summaries": summaries,
        "pairwise_distributions": dists,
        "same_adduct_diff_ce_threshold_survival": threshold_report,
        "molecule_level_any_same_adduct_pair_below": mol_report,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
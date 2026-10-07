"""V0 validated-baseline competition submission: CASMI 2026.

Pipeline (no new parameters; exact validated Class-1 / Variant-A config):
  candidate generation : raw precursor_mz +/- 0.01 Da against the all-train library
  scoring             : matchms 0.33.1 ModifiedCosineGreedy(tolerance=0.1,
                        mz_power=0.0, intensity_power=1.0) -- the EXP-001 C1 config
  aggregation         : molecule-level max -- max candidate-spectrum score per
                        (molecule_id, spectrum_id, inchikey14), then max over the
                        molecule_id's query spectra
  output              : top-25 ranked SMILES per molecule_id, rank 1 = highest score
                        (deterministic inchikey14 tie-break), one row per molecule

Scoring engine is reused verbatim from src/submission/make_submission.py (cmd_score)
so the identical validated code path is exercised; only I/O naming differs (results/
submission_v0). The same-adduct (v1a) and sum-fusion (v1b) variants are NOT part of V0.

Root cause of the "1,213 vs 400" shape: test.parquet holds 1,213 spectra but only 400
distinct molecule_id (each molecule has 1-7 spectra across adducts/CEs). The official
sample_submission.csv -- confirmed against the competition rules and public CASMI-2026
toolkits -- requires ONE ranked SMILES list per molecule_id => exactly 400 rows. All
1,213 test spectra are scored exactly once and aggregated per molecule.

Usage:
  python research/scripts/v0_make_submission.py sanity   # deterministic gate vs V1 smoke20
  python research/scripts/v0_make_submission.py full     # complete 400-molecule V0
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.submission.make_submission as ms  # noqa: E402

OUT = ROOT / "results" / "submission_v0"
SAMPLE_CSV = ROOT / "sample_submission.csv"
SEED = 20260924


def connect():
    return ms.connect()


def tt(fn):
    return str(fn)


def run_score(args):
    """Score using the exact validated engine (cmd_score), then keep artifacts."""
    ms.OUT_DIR = OUT
    ns = argparse.Namespace(limit_molecules=args.limit_molecules,
                            holdout_exact_copies=False, seed=SEED)
    t0 = time.time()
    ms.cmd_score(ns)
    score_sec = time.time() - t0
    tag = f"_smoke{args.limit_molecules}" if args.limit_molecules else ""
    shutil.move(tt(OUT / f"scores{tag}.parquet"), tt(OUT / f"scores_{args.label}.parquet"))
    shutil.move(tt(OUT / f"timing{tag}.csv"), tt(OUT / f"timing_{args.label}.csv"))
    return score_sec


def ranked_top25(scores: pd.DataFrame) -> dict:
    """Reuses make_submission.rank_variant(scores, 'v1'): all candidate adducts,
    molecule-level max aggregation, deterministic (score desc, inchikey14 asc) order."""
    return ms.rank_variant(scores, "v1")


def aggregate(args):
    scores_path = OUT / f"scores_{args.label}.parquet"
    scores = pd.read_parquet(tt(scores_path))
    if args.limit_molecules:
        expected = sorted(scores["molecule_id"].unique())
    else:
        expected = pd.read_csv(SAMPLE_CSV)["molecule_id"].tolist()
    ranked = ranked_top25(scores)
    all_iks = {ik for lst in ranked.values() for ik in lst}
    ik2smi = ms.smiles_for(all_iks)
    missing = all_iks - set(ik2smi)
    assert not missing, f"{len(missing)} inchikey14 without SMILES"

    sub = pd.DataFrame({
        "molecule_id": expected,
        "smiles": [";".join(ik2smi[ik] for ik in ranked.get(m, [])) for m in expected],
    })
    out_csv = OUT / f"submission_{args.label}.csv"
    sub.to_csv(tt(out_csv), index=False)
    return sub, scores, ranked, ik2smi, out_csv


def validate(sub: pd.DataFrame, scores: pd.DataFrame, ranked: dict,
             ik2smi: dict, out_csv: Path) -> dict:
    """Validation A-E against the official competition format."""
    report = {}

    # A. IDs: exactly the 400 sample_submission molecule_ids, no missing/extra/dup.
    sample_ids = pd.read_csv(SAMPLE_CSV)["molecule_id"].tolist()
    sub_ids = sub["molecule_id"].tolist()
    report["A_ids"] = {
        "rows": len(sub_ids),
        "expected_n": len(sample_ids),
        "no_extra": set(sub_ids) == set(sample_ids),
        "no_missing": all(m in set(sub_ids) for m in sample_ids),
        "no_duplicates": len(sub_ids) == len(set(sub_ids)),
        "order_matches_sample": sub_ids == sample_ids,
        "ok": (set(sub_ids) == set(sample_ids)
               and all(m in set(sub_ids) for m in sample_ids)
               and len(sub_ids) == len(set(sub_ids))
               and sub_ids == sample_ids),
    }
    # All 1,213 test spectra scored exactly once.
    spec_counts = scores.groupby(["molecule_id", "spectrum_id"]).ngroups
    report["A_spectrum_coverage"] = {
        "test_spectra": 1213,
        "spectra_in_scores": spec_counts,
        "all_scored_once": spec_counts == 1213,
        "ok": spec_counts == 1213 and scores["spectrum_id"].nunique() == 1213,
    }

    # B. Prediction count: 1..25 per row.
    lens = sub["smiles"].str.split(";").apply(len)
    report["B_prediction_count"] = {
        "max": int(lens.max()), "min": int(lens.min()),
        "rows_gt_25": int((lens > 25).sum()),
        "rows_zero": int((lens == 0).sum()),
        "ok": bool(lens.between(1, 25).all()),
    }

    # C. Chemistry: RDKit-parse, canonical-form validity, no dup within molecule.
    invalid, empty, dup = 0, 0, 0
    bad_missing_smiles = 0
    per_row = []
    for s in sub["smiles"]:
        parts = s.split(";") if s else []
        if not parts:
            empty += 1
        canon_seen = {}
        for p in parts:
            if not p:
                empty += 1
                continue
            m = Chem.MolFromSmiles(p)
            if m is None:
                invalid += 1
                continue
            c = Chem.MolToSmiles(m)
            canon_seen.setdefault(c, 0)
            canon_seen[c] += 1
        dup += sum(v - 1 for v in canon_seen.values() if v > 1)
        per_row.append(len(parts))
    report["C_chemistry"] = {
        "invalid_smiles": int(invalid),
        "empty_fields": int(empty),
        "duplicate_within_molecule": int(dup),
        "ok": invalid == 0 and empty == 0 and dup == 0,
    }

    # D. Submission format: exact columns, no index col, no NaN/null.
    cols = list(sub.columns)
    report["D_format"] = {
        "columns": cols,
        "exact_schema": cols == ["molecule_id", "smiles"],
        "no_index_column": True,
        "any_nan": bool(sub.isna().any().any()),
        "ok": cols == ["molecule_id", "smiles"] and not sub.isna().any().any(),
    }

    # E. Ranking: emitted order == (score desc, inchikey14 asc) from V0 scores.
    # Recompute per-molecule top-25 scores the same way rank_variant does, and verify
    # the emitted list equals it (no candidate with a higher score was skipped/out of order).
    bad_rank = 0
    for m, lst in ranked.items():
        per_spec = scores[scores["molecule_id"] == m].groupby(
            ["spectrum_id", "inchikey14"], as_index=False)["score"].max()
        mol = per_spec.groupby("inchikey14", as_index=False)["score"].max()
        mol = mol.sort_values(["score", "inchikey14"], ascending=[False, True])
        sorted_ik = mol["inchikey14"].head(25).tolist()
        if sorted_ik != lst:
            bad_rank += 1
    report["E_ranking"] = {
        "molecules_misordered": int(bad_rank),
        "ok": bad_rank == 0,
    }

    report["ok"] = all(report[k]["ok"] for k in ("A_ids", "A_spectrum_coverage",
                                                 "B_prediction_count", "C_chemistry",
                                                 "D_format", "E_ranking"))
    return report


def write_runtime(scores: pd.DataFrame, timing_csv: Path, score_sec: float,
                  agg_sec: float, report: dict) -> dict:
    timing = pd.read_csv(tt(timing_csv))
    n_spec = len(timing)
    per_mol = scores.groupby("molecule_id")["spectrum_id"].nunique()

    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:
        rss = None

    def stats(s):
        return {"avg": float(np.mean(s)), "median": float(np.median(s)),
                "max": float(np.max(s))}

    out = {
        "n_test_spectra": int(n_spec),
        "n_molecule_ids": int(len(per_mol)),
        "n_candidate_spectra_scored": int(timing["n_candidates"].sum()),
        "candidates_per_spectrum": stats(timing["n_candidates"]),
        "sec_per_spectrum": stats(timing["sec"]),
        "spectra_per_molecule": stats(per_mol),
        "score_stage_sec": round(score_sec, 3),
        "aggregate_stage_sec": round(agg_sec, 3),
        "total_rss_bytes": rss,
        "memory_note": "psutil unavailable" if rss is None else "rss of aggregate process",
    }
    return out


def manifest(out_csv: Path, runtime: dict) -> dict:
    import hashlib
    import matchms
    import rdkit
    import duckdb
    import numpy
    h = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    return {
        "submission": str(out_csv),
        "sha256": h,
        "test_data": str(ROOT / "test.parquet"),
        "train_data": str(ROOT / "train.parquet"),
        "n_molecule_ids": 400,
        "n_test_spectra": 1213,
        "max_predictions_per_molecule": 25,
        "candidate_gen": "Variant-A raw precursor_mz +/- 0.01 Da, all-train library",
        "scorer": "matchms ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)",
        "aggregation": "molecule-level max",
        "v1_equivalent_variant": "v1 (all candidate adducts)",
        "excludes": ["v1a same-adduct", "v1b sum-fusion", "EXP-002 neutral-mass",
                     "EXP-007 C2 intervention"],
        "python": sys.version.split()[0],
        "versions": {"matchms": matchms.__version__, "rdkit": rdkit.__version__,
                     "duckdb": duckdb.__version__, "numpy": numpy.__version__},
        "seed": SEED,
        "runtime_summary": runtime,
    }


def cmd_sanity(args):
    print(">> V0 sanity gate: 20 molecule_ids vs results/submission_v1/submission_v1_smoke20.csv")
    ms.OUT_DIR = OUT
    s = run_score(argparse.Namespace(limit_molecules=20, label="sanity20"))
    sub, scores, ranked, ik2smi, out_csv = aggregate(argparse.Namespace(
        limit_molecules=20, label="sanity20"))

    ref = pd.read_csv(tt(ROOT / "results" / "submission_v1" / "submission_v1_smoke20.csv"))
    ref_ids = set(ref["molecule_id"])
    sel_ids = set(sub["molecule_id"])
    same_ids = ref_ids == sel_ids
    if not same_ids:
        print(f"  molecule_id sets differ! ref has {len(ref_ids - sel_ids)} extra, "
              f"sel has {len(sel_ids - ref_ids)} extra")
        sys.exit(1)
    ref_map = dict(zip(ref["molecule_id"], ref["smiles"]))
    mism = {m: (sub[sub["molecule_id"] == m]["smiles"].iloc[0], ref_map[m])
            for m in sorted(sel_ids)
            if sub[sub["molecule_id"] == m]["smiles"].iloc[0] != ref_map[m]}
    report = {
        "same_molecule_ids": same_ids,
        "n_smoke_molecules": len(sel_ids),
        "mismatched_rows": len(mism),
        "mismatch_detail": {m: {"v0": a, "v1": b} for m, (a, b) in list(mism.items())[:5]},
        "ok": same_ids and len(mism) == 0,
        "score_stage_sec": round(s, 3),
    }
    json.dump(report, open(tt(OUT / "sanity_gate.json"), "w"), indent=2)
    print(" ", json.dumps(report, indent=2))
    if report["ok"]:
        print(">> SANITY GATE PASSED -- V0 reproduces the validated V1 smoke20 output exactly.")
    else:
        print(">> SANITY GATE FAILED -- STOP. Investigate before full run.")
        sys.exit(1)


def cmd_full(args):
    print(">> V0 full run: all 1,213 test spectra -> 400 molecule rows")
    ms.OUT_DIR = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    score_sec = run_score(argparse.Namespace(limit_molecules=0, label="full"))
    t0 = time.time()
    sub, scores, ranked, ik2smi, out_csv = aggregate(argparse.Namespace(
        limit_molecules=0, label="v0"))
    report = validate(sub, scores, ranked, ik2smi, out_csv)
    agg_sec = time.time() - t0
    runtime = write_runtime(scores, OUT / "timing_full.csv", score_sec, agg_sec, report)
    sheet = manifest(out_csv, runtime)
    json.dump(report, open(tt(OUT / "validation_report.json"), "w"), indent=2)
    json.dump(sheet, open(tt(OUT / "run_manifest.json"), "w"), indent=2)
    json.dump(runtime, open(tt(OUT / "runtime_summary.json"), "w"), indent=2)
    print("  validation_report.json", json.dumps(report, indent=2))
    print(f"  submission_v0.csv :: {out_csv.stat().st_size} bytes, "
          f"sha256={sheet['sha256'][:16]}...")
    print(f"  runtime: score {runtime['score_stage_sec']}s, "
          f"aggregate {runtime['aggregate_stage_sec']}s, "
          f"{runtime['n_test_spectra']} spectra / {runtime['n_molecule_ids']} molecules, "
          f"{runtime['n_candidate_spectra_scored']} candidate spectra scored")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sanity", "full"])
    args = p.parse_args()
    {"sanity": cmd_sanity, "full": cmd_full}[args.command](args)


if __name__ == "__main__":
    main()
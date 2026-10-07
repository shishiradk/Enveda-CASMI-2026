"""Real-test submission pipeline: test.parquet -> submission CSV.

Two stages, so aggregation variants never require re-scoring:

  score      (expensive) For every test spectrum: Variant-A candidates
             (raw precursor_mz +/- 0.01 Da, all-train library), matchms 0.33.1
             ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
             -- the exact EXP-001 C1 configuration. Persists, per
             (spectrum_id, candidate inchikey14, candidate adduct), the max score.

  aggregate  (seconds) Reads the score cache and writes one submission per variant:
             v1   BASELINE-V1: all candidate adducts; molecule score = max over its
                  candidate spectra, then max over the molecule_id's query spectra.
             v1a  UPGRADE-V1A: as v1, but only candidate spectra whose adduct equals
                  the query spectrum's adduct (a same-m/z, different-adduct candidate
                  has a different neutral mass, so it is never the true molecule).
             v1b  UPGRADE-V1B: as v1a per spectrum, but fuse across the molecule_id's
                  query spectra by SUM (missing = 0) instead of max.

Usage:
  python src/submission/make_submission.py score --limit-molecules 20   # smoke
  python src/submission/make_submission.py score                         # full
  python src/submission/make_submission.py aggregate
  # labeled benchmark: local test with its exact train copies held out
  python src/submission/make_submission.py score    --holdout-exact-copies
  python src/submission/make_submission.py evaluate --holdout-exact-copies
"""

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from matchms.similarity import ModifiedCosineGreedy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectra.spectrum_io import make_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Env overrides so the same file runs in a Kaggle notebook (/kaggle/input/...).
TRAIN = os.environ.get("CASMI_TRAIN", (ROOT / "train.parquet").as_posix())
TEST = os.environ.get("CASMI_TEST", (ROOT / "test.parquet").as_posix())
SAMPLE = Path(os.environ.get("CASMI_SAMPLE", ROOT / "sample_submission.csv"))
OUT_DIR = Path(os.environ.get("CASMI_OUT", ROOT / "results" / "submission_v1"))
PRECURSOR_TOL = 0.01
TOP_N = 25
CHUNK = 60  # test spectra per train fetch; bounds peak-list memory
VARIANTS = ("v1", "v1a", "v1b")


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=2")
    return con


def run_tag(args) -> str:
    return (f"_smoke{args.limit_molecules}" if args.limit_molecules else "") +            ("_holdout" if args.holdout_exact_copies else "")


def cmd_score(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    if args.holdout_exact_copies:
        con.execute(f"CREATE TEMP TABLE te_all AS SELECT precursor_mz, ms2_mzs, ms2_normalized_intensities "
                    f"FROM '{TEST}'")
    holdout_sql = ("AND NOT EXISTS (SELECT 1 FROM te_all x WHERE x.precursor_mz = t.precursor_mz "
                   "AND x.ms2_mzs = t.ms2_mzs AND x.ms2_normalized_intensities = t.ms2_normalized_intensities)"
                   if args.holdout_exact_copies else "")
    test = con.execute(
        f"SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities "
        f"FROM '{TEST}'"
    ).fetchdf()
    if args.limit_molecules:
        keep = sorted(test["molecule_id"].unique())
        keep = pd.Series(keep).sample(n=args.limit_molecules, random_state=args.seed).tolist()
        test = test[test["molecule_id"].isin(keep)]
    test = test.sort_values("precursor_mz").reset_index(drop=True)
    print(f"Scoring {len(test)} test spectra / {test['molecule_id'].nunique()} molecule_ids")

    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    rows, timing = [], []
    t0 = time.time()
    for c0 in range(0, len(test), CHUNK):
        chunk = test.iloc[c0:c0 + CHUNK]
        con.register("q", chunk[["spectrum_id", "precursor_mz"]])
        lib = con.execute(
            f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
                FROM '{TRAIN}' t SEMI JOIN q
                ON t.precursor_mz BETWEEN q.precursor_mz - {PRECURSOR_TOL} AND q.precursor_mz + {PRECURSOR_TOL}
                WHERE TRUE {holdout_sql}"""
        ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
        con.unregister("q")
        pm = lib["precursor_mz"].values
        l_ik, l_ad = lib["inchikey14"].values, lib["adduct"].values
        l_mz, l_int = lib["ms2_mzs"].values, lib["ms2_normalized_intensities"].values
        for _, qr in chunk.iterrows():
            ts = time.time()
            qspec = make_spectrum(qr["ms2_mzs"], qr["ms2_normalized_intensities"], qr["precursor_mz"])
            i0 = np.searchsorted(pm, qr["precursor_mz"] - PRECURSOR_TOL, side="left")
            i1 = np.searchsorted(pm, qr["precursor_mz"] + PRECURSOR_TOL, side="right")
            best = {}
            for j in range(i0, i1):
                s = float(sim.pair(qspec, make_spectrum(l_mz[j], l_int[j], pm[j]))["score"])
                key = (l_ik[j], l_ad[j])
                if s > best.get(key, -1.0):
                    best[key] = s
            for (ik, cad), s in best.items():
                rows.append((qr["molecule_id"], qr["spectrum_id"], qr["adduct"], ik, cad, s))
            timing.append({"spectrum_id": qr["spectrum_id"], "n_candidates": int(i1 - i0),
                           "n_peaks": len(qr["ms2_mzs"]), "sec": time.time() - ts})
        del lib
        print(f"  [{min(c0 + CHUNK, len(test))}/{len(test)}] elapsed {time.time() - t0:.0f}s")

    scores = pd.DataFrame(rows, columns=["molecule_id", "spectrum_id", "query_adduct",
                                         "inchikey14", "cand_adduct", "score"])
    tag = run_tag(args)
    scores.to_parquet(OUT_DIR / f"scores{tag}.parquet", index=False)
    pd.DataFrame(timing).to_csv(OUT_DIR / f"timing{tag}.csv", index=False)
    print(f"Wrote {len(scores)} score rows; total {time.time() - t0:.0f}s")


def rank_variant(scores: pd.DataFrame, variant: str) -> dict:
    s = scores
    if variant in ("v1a", "v1b"):
        s = s[s["query_adduct"] == s["cand_adduct"]]
    per_spec = s.groupby(["molecule_id", "spectrum_id", "inchikey14"], as_index=False)["score"].max()
    fuse = "sum" if variant == "v1b" else "max"
    mol = per_spec.groupby(["molecule_id", "inchikey14"], as_index=False)["score"].agg(fuse)
    # Deterministic tie-break on inchikey14.
    mol = mol.sort_values(["molecule_id", "score", "inchikey14"], ascending=[True, False, True])
    return {mid: g["inchikey14"].head(TOP_N).tolist() for mid, g in mol.groupby("molecule_id")}


def smiles_for(inchikeys: set) -> dict:
    con = connect()
    con.register("k", pd.DataFrame({"ik": sorted(inchikeys)}))
    df = con.execute(
        f"""SELECT inchikey14, normalized_smiles, count(*) n FROM '{TRAIN}'
            WHERE inchikey14 IN (SELECT ik FROM k) GROUP BY 1, 2"""
    ).fetchdf()
    df = df.sort_values(["inchikey14", "n", "normalized_smiles"], ascending=[True, False, True])
    return df.drop_duplicates("inchikey14").set_index("inchikey14")["normalized_smiles"].to_dict()


def validate(sub: pd.DataFrame, expected_ids: list, smiles_to_ik: dict) -> dict:
    from rdkit import Chem
    report = {"rows": len(sub), "ids_match": sorted(sub["molecule_id"]) == sorted(expected_ids)}
    lens, bad, dup = [], 0, 0
    for s in sub["smiles"]:
        parts = s.split(";") if s else []
        lens.append(len(parts))
        bad += sum(Chem.MolFromSmiles(p) is None for p in parts)
        iks = [smiles_to_ik[p] for p in parts]
        dup += len(iks) - len(set(iks))
    report.update(min_len=min(lens), max_len=max(lens), invalid_smiles=bad, duplicate_ik14=dup,
                  empty_rows=sum(n == 0 for n in lens))
    report["ok"] = (report["ids_match"] and report["max_len"] <= TOP_N and bad == 0 and dup == 0
                    and report["empty_rows"] == 0)
    return report


def cmd_aggregate(args):
    tag = run_tag(args)
    scores = pd.read_parquet(OUT_DIR / f"scores{tag}.parquet")
    expected = sorted(scores["molecule_id"].unique()) if args.limit_molecules else pd.read_csv(SAMPLE)["molecule_id"].tolist()
    ranked = {v: rank_variant(scores, v) for v in VARIANTS}
    all_iks = {ik for r in ranked.values() for lst in r.values() for ik in lst}
    ik2smi = smiles_for(all_iks)
    missing = all_iks - set(ik2smi)
    assert not missing, f"{len(missing)} inchikey14 without SMILES"
    smi2ik = {v: k for k, v in ik2smi.items()}
    summary = {}
    for v, r in ranked.items():
        sub = pd.DataFrame({"molecule_id": expected,
                            "smiles": [";".join(ik2smi[ik] for ik in r.get(m, [])) for m in expected]})
        path = OUT_DIR / f"submission_{v}{tag}.csv"
        sub.to_csv(path, index=False)
        summary[v] = validate(sub, expected, smi2ik)
        print(v, path.name, summary[v])
    # Pairwise overlap of top-1 choices, to see how much the variants actually differ.
    for a, b in (("v1", "v1a"), ("v1a", "v1b")):
        same = np.mean([ranked[a].get(m, [None])[:1] == ranked[b].get(m, [None])[:1] for m in expected])
        summary[f"top1_agreement_{a}_{b}"] = float(same)
        print(f"top-1 agreement {a} vs {b}: {same:.3f}")
    json.dump(summary, open(OUT_DIR / f"validation{tag}.json", "w"), indent=2)


def cmd_evaluate(args):
    """Labeled evaluation on the local test set with exact train copies held out.

    Labels: every local test spectrum is byte-identical to one enveda-180 train row
    (results/submission_v1/test_exact_train_copies.csv), which fixes one inchikey14 per
    molecule_id. Only meaningful on scores produced with --holdout-exact-copies.
    """
    assert args.holdout_exact_copies, "evaluate requires --holdout-exact-copies scores"
    tag = run_tag(args)
    scores = pd.read_parquet(OUT_DIR / f"scores{tag}.parquet")
    labels = (pd.read_csv(OUT_DIR / "test_exact_train_copies.csv")
              .drop_duplicates("molecule_id").set_index("molecule_id")["inchikey14"])
    mids = sorted(pd.read_csv(OUT_DIR / "test_exact_train_copies.csv")["molecule_id"].unique())
    if args.limit_molecules:
        mids = sorted(scores["molecule_id"].unique())
    reachable = set(scores.merge(labels.rename("true").reset_index(), on="molecule_id")
                    .query("inchikey14 == true")["molecule_id"])
    out = {}
    for v in VARIANTS:
        ranked = rank_variant(scores, v)
        rr = {}
        for m in mids:
            lst = ranked.get(m, [])
            rr[m] = 1.0 / (lst.index(labels[m]) + 1) if labels[m] in lst else 0.0
        r = pd.Series(rr)
        out[v] = {"n": len(r), "mrr@25": r.mean(), "r@1": (r == 1).mean(), "r@25": (r > 0).mean(),
                  "n_reachable": len(reachable), "mrr@25_reachable": r[r.index.isin(reachable)].mean()}
        r.rename("rr").to_csv(OUT_DIR / f"eval_rr_{v}{tag}.csv")
        print(v, {k: round(x, 4) if isinstance(x, float) else x for k, x in out[v].items()})
    json.dump(out, open(OUT_DIR / f"eval{tag}.json", "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["score", "aggregate", "evaluate"])
    p.add_argument("--limit-molecules", type=int, default=0)
    p.add_argument("--holdout-exact-copies", action="store_true",
                   help="drop train rows byte-identical to local test spectra (labeled benchmark)")
    p.add_argument("--seed", type=int, default=20260924)
    args = p.parse_args()
    {"score": cmd_score, "aggregate": cmd_aggregate, "evaluate": cmd_evaluate}[args.stage](args)


if __name__ == "__main__":
    main()

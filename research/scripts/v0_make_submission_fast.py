"""Optimized V0 scoring: identical science, much faster fetch, resumable chunks.

Scientific behavior is preserved EXACTLY (scores bit-identical to the validated
engine in src/submission/make_submission.py cmd_score):
  - Variant-A candidate generation: raw precursor_mz +/- 0.01 Da, all-train library.
  - matchms 0.33.1 ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0).
  - molecule-level max aggregation (v1), top-25, deterministic inchikey14 tie-break.
  - Per query the candidate set is the same sorted-by-precursor library rows within
    +/- 0.01 Da via np.searchsorted => same (inchikey14, adduct) pairs and the same
    maximum score per (spectrum_id, inchikey14, cand_adduct). Pair order and score
    arithmetic are unchanged.

Execution differences (performance only):
  1. Pool: train.parquet is scanned ONCE into a sorted pool.parquet restricted to
     the test precursor range (+/- 0.01). Per-chunk library fetch reads only the
     m/z slice (row-group-pruned) instead of re-scanning 2.8GB of train 21 times.
  2. Candidate Spectrum objects are constructed lazily, once per library row that
     is actually scored, then reused across the chunk's queries. matchms pair()
     does not mutate spectra, so results are identical.
  3. Chunks are independent and scored by workers; results are concatenated in
     fixed chunk order => the scores DataFrame is byte-identical to the sequential
     engine's. Determinism holds regardless of worker scheduling.
  4. Resumable: each completed chunk is written to results/submission_v0/opt/ and a
     manifest records finished chunk indices; a rerun skips finished chunks.

Usage:
  python research/scripts/v0_make_submission_fast.py build-pool        # one-time cache
  python research/scripts/v0_make_submission_fast.py sanity            # gate vs V1 smoke20
  python research/scripts/v0_make_submission_fast.py score  --label full [--workers 8]
  python research/scripts/v0_make_submission_fast.py assemble --label full
  python research/scripts/v0_make_submission_fast.py full    [--workers 8]  # score+assemble+validate
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.submission.make_submission as ms  # noqa: E402
from spectra.spectrum_io import make_spectrum  # noqa: E402

OUT = ROOT / "results" / "submission_v0"
OPT_DIR = OUT / "opt"
MANIFEST = OPT_DIR / "manifest.json"
SAMPLE_CSV = ROOT / "sample_submission.csv"
POOL = OUT / "pool.parquet"
SEED = 20260924


def connect():
    return ms.connect()


def test_universe(limit_molecules=0):
    con = connect()
    test = con.execute(
        "SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities "
        f"FROM '{ms.TEST}'"
    ).fetchdf()
    if limit_molecules:
        keep = sorted(test["molecule_id"].unique())
        keep = pd.Series(keep).sample(n=limit_molecules, random_state=SEED).tolist()
        test = test[test["molecule_id"].isin(keep)]
    return test.sort_values("precursor_mz").reset_index(drop=True)


def cmd_build_pool(args):
    """One-time scan of train.parquet -> sorted pool.parquet (zstd, small row groups)."""
    if POOL.exists() and not (args.force or args.rebuild):
        print(f"pool exists ({POOL.stat().st_size / 1e6:.0f}MB); use --rebuild to rebuild")
        return
    con = connect()
    t0 = time.time()
    df = con.execute(
        f"""SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
            FROM '{ms.TRAIN}'
            WHERE precursor_mz BETWEEN
                (SELECT min(precursor_mz) - 0.01 FROM 'test.parquet')
                AND (SELECT max(precursor_mz) + 0.01 FROM 'test.parquet')
            ORDER BY precursor_mz"""
    ).fetchdf()
    print(f"pool rows: {len(df)}  (scan+sort {time.time() - t0:.1f}s)")

    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    POOL.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), POOL,
                   row_group_size=4096, compression="zstd")
    print(f"wrote {POOL} ({POOL.stat().st_size / 1e6:.0f}MB) {time.time() - t0:.1f}s")


def fetch_library(con, chunk, mode):
    """Per-chunk candidate library. Both modes yield IDENTICAL per-query windows.

    mode='semi' : SEMI JOIN over pool (union of per-spectrum windows -- the same
                  rows the sequential engine fetches from train).
    mode='band' : single range [chunk min-0.01, chunk max+0.01] superset; the per
                  query searchsorted window inside it selects the same candidate set.
    """
    if mode == "semi":
        con.register("q", chunk[["spectrum_id", "precursor_mz"]])
        lib = con.execute(
            f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
                FROM '{POOL.as_posix()}' t SEMI JOIN q
                ON t.precursor_mz BETWEEN q.precursor_mz - {ms.PRECURSOR_TOL}
                                      AND q.precursor_mz + {ms.PRECURSOR_TOL}
                WHERE TRUE"""
        ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
        con.unregister("q")
        return lib
    loq = float(chunk["precursor_mz"].min()) - ms.PRECURSOR_TOL
    hiq = float(chunk["precursor_mz"].max()) + ms.PRECURSOR_TOL
    return con.execute(
        f"""SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
            FROM '{POOL.as_posix()}'
            WHERE precursor_mz BETWEEN {loq} AND {hiq}"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)


def score_chunk(chunk: pd.DataFrame, mode: str = "semi") -> tuple:
    """Score one chunk; returns (rows, timing) exactly mirroring the engine.

    The per-query candidate window and the maximum score per (spectrum_id,
    inchikey14, cand_adduct) are computed identically to cmd_score: same
    searchsorted bounds, same pair() calls in the same order over the same
    candidate rows, same strict-> max update. Candidate Spectrum objects are
    constructed lazily once per scored library row and reused (pair() read-only).
    """
    rows, timing = [], []
    sim = ms.ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    con = connect()
    lib = fetch_library(con, chunk, mode)
    con.close()
    pm = lib["precursor_mz"].values
    l_ik, l_ad = lib["inchikey14"].values, lib["adduct"].values
    l_mz, l_int = lib["ms2_mzs"].values, lib["ms2_normalized_intensities"].values

    # Pass 1: find every library row that any query will actually score.
    ranges = []
    used = np.zeros(len(lib), dtype=bool)
    for _, qr in chunk.iterrows():
        i0 = np.searchsorted(pm, qr["precursor_mz"] - ms.PRECURSOR_TOL, side="left")
        i1 = np.searchsorted(pm, qr["precursor_mz"] + ms.PRECURSOR_TOL, side="right")
        ranges.append((i0, i1))
        used[i0:i1] = True
    lib_specs = {}
    for j in np.nonzero(used)[0]:
        lib_specs[j] = make_spectrum(l_mz[j], l_int[j], pm[j])

    # Pass 2: exactly the engine's per-query loop.
    for (i0, i1), (_, qr) in zip(ranges, chunk.iterrows()):
        ts = time.time()
        qspec = make_spectrum(qr["ms2_mzs"], qr["ms2_normalized_intensities"], qr["precursor_mz"])
        best = {}
        for j in range(i0, i1):
            s = float(sim.pair(qspec, lib_specs[j])["score"])
            key = (l_ik[j], l_ad[j])
            if s > best.get(key, -1.0):
                best[key] = s
        for (ik, cad), s in best.items():
            rows.append((qr["molecule_id"], qr["spectrum_id"], qr["adduct"], ik, cad, s))
        timing.append({"spectrum_id": qr["spectrum_id"], "n_candidates": int(i1 - i0),
                       "n_peaks": len(qr["ms2_mzs"]), "sec": time.time() - ts})
    return rows, timing


SCORE_COLS = ["molecule_id", "spectrum_id", "query_adduct", "inchikey14", "cand_adduct", "score"]


def done_chunks() -> list:
    if not MANIFEST.exists():
        return []
    return json.load(open(MANIFEST))["done"]


def mark_done(i):
    done = done_chunks()
    if i not in done:
        done.append(i)
        json.dump({"done": sorted(done)}, open(MANIFEST, "w"))


def cmd_score(args):
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    test = test_universe(args.limit_molecules)
    print(f"Scoring {len(test)} test spectra / {test['molecule_id'].nunique()} molecule_ids "
          f"(fetch={args.fetch}, workers={args.workers}, chunk={ms.CHUNK})")
    chunk_offsets = list(range(0, len(test), ms.CHUNK))
    done = done_chunks()
    pending = [c0 for c0 in chunk_offsets if c0 not in done]
    print(f"  chunks total={len(chunk_offsets)} done={len(done)} remaining={len(pending)}")

    import concurrent.futures as cf
    t0 = time.time()
    for batch in range(0, len(pending), args.workers):
        todo = pending[batch:batch + args.workers]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut = {c0: ex.submit(score_chunk, test.iloc[c0:c0 + ms.CHUNK], args.fetch) for c0 in todo}
            for c0, f in fut.items():
                rows, timing = f.result()
                pd.DataFrame(rows, columns=SCORE_COLS).to_parquet(
                    OPT_DIR / f"chunk_{c0:04d}.parquet", index=False)
                pd.DataFrame(timing).to_csv(OPT_DIR / f"timing_{c0:04d}.csv", index=False)
                mark_done(c0)
                print(f"  chunk@{c0}: {len(rows)} score rows "
                      f"(elapsed {time.time() - t0:.0f}s)")
        time.sleep(0.5)

    print(f"score complete: {len(done_chunks())}/{len(chunk_offsets)} chunks "
          f"({time.time() - t0:.0f}s)")


def collect_scores() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(OPT_DIR.glob("chunk_*.parquet"))]
    scores = pd.concat(frames, ignore_index=True)
    return scores[sorted(SCORE_COLS)][SCORE_COLS]


def cmd_assemble(args):
    tag = f"_smoke{args.limit_molecules}" if args.limit_molecules else ""
    scores = collect_scores()
    expected = sorted(scores["molecule_id"].unique()) if args.limit_molecules \
        else pd.read_csv(SAMPLE_CSV)["molecule_id"].tolist()
    assert len(scores) == len(scores.drop_duplicates(["spectrum_id", "inchikey14", "cand_adduct"])), \
        "score rows are not unique per (spectrum, ik, adduct)"
    scores.to_parquet(OUT / f"scores{tag}.parquet", index=False)
    timing = pd.concat([pd.read_csv(p) for p in sorted(OPT_DIR.glob("timing_*.csv"))],
                       ignore_index=True)
    timing.to_csv(OUT / f"timing{tag}.csv", index=False)
    print(f"assembled {len(scores)} score rows / {scores['spectrum_id'].nunique()} spectra -> "
          f"scores{tag}.parquet")

    ranked = ms.rank_variant(scores, "v1")
    all_iks = {ik for lst in ranked.values() for ik in lst}
    ik2smi = ms.smiles_for(all_iks)
    missing = all_iks - set(ik2smi)
    assert not missing, f"{len(missing)} inchikey14 without SMILES"
    sub = pd.DataFrame({"molecule_id": expected,
                        "smiles": [";".join(ik2smi[ik] for ik in ranked.get(m, [])) for m in expected]})
    out_csv = OUT / f"submission_{args.label}.csv"
    sub.to_csv(out_csv, index=False)
    print(f"submission -> {out_csv} ({len(sub)} rows)")
    return sub, scores, ranked, ik2smi, out_csv


def chunks_eq(scores_a, scores_b):
    a = scores_a.sort_values(["spectrum_id", "inchikey14", "cand_adduct"]).reset_index(drop=True)
    b = scores_b.sort_values(["spectrum_id", "inchikey14", "cand_adduct"]).reset_index(drop=True)
    if a.shape != b.shape:
        return False, None
    keys = (a[["spectrum_id", "inchikey14", "cand_adduct"]].values ==
            b[["spectrum_id", "inchikey14", "cand_adduct"]].values).all()
    d = float(np.abs(a["score"].values - b["score"].values).max())
    return bool(keys), d


def cmd_sanity(args):
    """Gate the optimized path against the V1 smoke20 reference.

    1. semi mode: scores must be byte-identical to results/submission_v1/
       scores_smoke20.parquet and the submission CSV identical to
       submission_v1_smoke20.csv (order-independent).
    2. band-vs-semi: on the smoke20 subset the band fetch selects the same
       candidates as semi (self-consistency of the two fetch modes).
    """
    print(">> OPT sanity gate: 20 molecule_ids vs V1 smoke20 reference")
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    ref_scores = pd.read_parquet(OUT.parent / "submission_v1" / "scores_smoke20.parquet")
    ref_csv = pd.read_csv(OUT.parent / "submission_v1" / "submission_v1_smoke20.csv")
    ref_map = dict(zip(ref_csv["molecule_id"], ref_csv["smiles"]))

    import shutil
    for p in OPT_DIR.glob("*"):
        p.unlink()
    if MANIFEST.exists():
        MANIFEST.unlink()

    test = test_universe(20)
    import concurrent.futures as cf
    report = {}
    sub = None
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {m: ex.submit(score_chunk, test, m) for m in ("semi", "band")}
        for mode in ("semi", "band"):
            rows, timing = futs[mode].result()
            tsec = float(np.sum([r["sec"] for r in timing]))
            scores = pd.DataFrame(rows, columns=SCORE_COLS)
            keys_ok, max_diff = chunks_eq(scores, ref_scores)
            shape_ok = scores.shape == ref_scores.shape
            mode_pass = shape_ok and keys_ok and max_diff == 0.0
            report[mode] = {"score_rows": int(len(scores)), "shape_ok": bool(shape_ok),
                            "keys_equal": keys_ok, "max_abs_score_diff": max_diff,
                            "sum_score_sec": round(tsec, 3), "id": "ok" if mode_pass else "FAIL"}
            print(" ", mode, report[mode])
            # Persist the (semi, identical) chunk rows so aggregate + CSV check run.
            if mode == "semi":
                for p in OPT_DIR.glob("chunk_*.parquet"):
                    p.unlink()
                pd.DataFrame(rows, columns=SCORE_COLS).to_parquet(
                    OPT_DIR / "chunk_0000.parquet", index=False)
                pd.DataFrame(timing).to_csv(OPT_DIR / "timing_0000.csv", index=False)
                mark_done(0)
    # order-independent submission CSV comparison via the same aggregation path
    sub, _, _, _, out_csv = cmd_assemble(argparse.Namespace(limit_molecules=20, label="sanity20"))
    sel_map = dict(zip(sub["molecule_id"], sub["smiles"]))
    mism = {m: (sel_map[m], ref_map[m]) for m in sorted(ref_map)
            if sel_map.get(m) != ref_map[m]}
    csv_identical = (len(mism) == 0 and set(sel_map) == set(ref_map))

    ok = all(report[m]["id"] == "ok" for m in ("semi", "band"))
    report["submission"] = {"rows": int(len(sub)), "csv_identical": csv_identical,
                            "mismatched_rows": len(mism), "id": "ok" if csv_identical else "FAIL"}
    json.dump({"sanity_gate_optimized": True, "modes": report},
              open(OUT / "sanity_gate_opt.json", "w"), indent=2)
    print("  submission:", report["submission"])
    if ok and csv_identical:
        print(">> OPT SANITY GATE PASSED -- scores bit-identical to V1 reference "
              "(both fetch modes); submission identical.")
    else:
        print(">> OPT SANITY GATE FAILED -- STOP.")
        sys.exit(1)


def cmd_full(args):
    print(">> OPT V0 full run: all 1,213 test spectra -> 400 molecule rows")
    t_score = time.time()
    cmd_score(argparse.Namespace(limit_molecules=0, workers=args.workers, fetch=args.fetch))
    score_sec = time.time() - t_score
    sub, scores, ranked, ik2smi, out_csv = cmd_assemble(
        argparse.Namespace(limit_molecules=0, label="v0"))
    st = time.time()
    report = validate_all(sub, scores, ranked, ik2smi, out_csv)
    agg_sec = time.time() - st
    sheet = manifest_json(out_csv, len(scores),
                          agg_sec=agg_sec,
                          n_test_spectra=int(scores["spectrum_id"].nunique()),
                          n_mols=int(scores["molecule_id"].nunique()))
    import importlib.util
    spec = importlib.util.spec_from_file_location("v0x",
                                                  ROOT / "research" / "scripts" / "v0_make_submission.py")
    v0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v0)
    runtime = v0.write_runtime(scores, OUT / "timing.csv", score_sec, agg_sec, report)
    sheet["runtime_summary"] = runtime
    json.dump(report, open(OUT / "validation_report.json", "w"), indent=2)
    json.dump(runtime, open(OUT / "runtime_summary.json", "w"), indent=2)
    json.dump(sheet, open(OUT / "run_manifest.json", "w"), indent=2)
    print("  validation_report.json", json.dumps(report, indent=2))
    print(f"  submission_v0.csv :: {out_csv.stat().st_size} bytes, "
          f"sha256={sheet['sha256'][:16]}...")
    print(f"  score rows: {len(scores)}")


def validate_all(sub, scores, ranked, ik2smi, out_csv):
    import importlib.util
    spec = importlib.util.spec_from_file_location("v0m",
                                                  ROOT / "research" / "scripts" / "v0_make_submission.py")
    v0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v0)
    return v0.validate(sub, scores, ranked, ik2smi, out_csv)


def manifest_json(out_csv, n_score_rows, agg_sec, n_test_spectra, n_mols):
    import matchms, rdkit, numpy  # noqa: E401
    h = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    return {
        "submission": str(out_csv), "sha256": h,
        "test_data": str(ROOT / "test.parquet"), "train_data": str(ROOT / "train.parquet"),
        "pool": str(POOL), "n_molecule_ids": int(n_mols), "n_test_spectra": int(n_test_spectra),
        "n_score_rows": int(n_score_rows), "max_predictions_per_molecule": 25,
        "candidate_gen": "Variant-A raw precursor_mz +/- 0.01 Da, all-train library",
        "scorer": "matchms ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)",
        "aggregation": "molecule-level max", "v1_equivalent_variant": "v1 (all candidate adducts)",
        "excludes": ["v1a same-adduct", "v1b sum-fusion", "EXP-002 neutral-mass",
                     "EXP-007 C2 intervention"],
        "optimizations": ["one-time sorted pool.parquet", "lazy candidate Spectrum cache per scored row",
                          "chunk-level parallelism with fixed-order assembly"],
        "python": sys.version.split()[0],
        "versions": {"matchms": matchms.__version__, "rdkit": rdkit.__version__,
                     "duckdb": duckdb.__version__, "numpy": numpy.__version__},
        "seed": SEED, "assemble_stage_sec": round(agg_sec, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build-pool", "sanity", "score", "assemble", "full"])
    p.add_argument("--limit-molecules", type=int, default=0)
    p.add_argument("--label", default="v0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fetch", choices=["semi", "band"], default="band")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()
    {"build-pool": cmd_build_pool, "sanity": cmd_sanity,
     "score": cmd_score, "assemble": cmd_assemble, "full": cmd_full}[a.command](a)


if __name__ == "__main__":
    main()
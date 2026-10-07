"""CASMI 2026 -- V0 Kaggle notebook pipeline (self-contained; no project imports).

V0 SCIENCE (locked; copied verbatim from src/submission/make_submission.py + v0_make_submission_fast.py):
  - Variant-A candidates: raw precursor_mz +/- 0.01 Da over the ALL-train library.
  - matchms 0.33.1 ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0).
  - per (spectrum_id, candidate inchikey14, candidate adduct): max score (strict >).
  - molecule level: max over candidate spectra, then max over the molecule_id's query spectra
    (rank_variant 'v1', all candidate adducts).
  - order: score desc, inchikey14 asc; top 25; SMILES = most frequent train normalized_smiles
    per inchikey14 (ties -> lexicographically smallest).

EXECUTION ONLY (no effect on outputs):
  - DuckDB streams the sorted candidate pool straight to parquet (no pandas materialisation).
  - Per chunk of 60 precursor-sorted test spectra, one precursor band is read from the pool.
  - Candidate Spectrum objects built lazily once per scored library row, reused within a chunk.
  - Chunks scored in parallel processes; results assembled in fixed chunk order.

ZERO-CANDIDATE POLICY (v2, user-approved 2026-09-25): a molecule_id with no V0 candidate gets the
single placeholder PLACEHOLDER_SMILES. It is NOT a prediction (it scores 0, exactly like an omitted
row); it only keeps the file valid, because the metric rejects rows with no guesses. Every such
molecule, and every spectrum skipped as malformed, is recorded in v0_run_report.json. All other rows
are byte-identical V0.

Usage (local reproduction):  python research/kaggle_v0/casmi_v0_kaggle.py --mode local
On Kaggle the notebook calls main(mode="kaggle").
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PRECURSOR_TOL = 0.01
TOP_N = 25
CHUNK = 60
MATCHMS_VERSION = "0.33.1"
PLACEHOLDER_SMILES = "C"  # non-predictive filler for zero-candidate molecule_ids only (see docstring)

# Parity reference: results/submission_v0/submission_v0.csv (validated local V0 output).
REF_SHA256_RAW = "4cbddddd6d01041c6a4be021ef4a215897f14ac9d9c998cb504ac312f48b791c"  # as written on Windows (CRLF)
REF_SHA256_LF = "bd885d283a2407c38fd4d17aee13481f29f2a07a2c2d71319f6a3ffb9a0db684"   # same bytes, CRLF -> LF
# sha256 of sorted "molecule_id|spectrum_id" lines of the VISIBLE test.parquet (1,213 spectra).
# Parity is enforced only when the present test matches it (commit run); the hidden rerun skips it.
VISIBLE_TEST_FINGERPRINT = "f396b01668daffce6785b5fcd00f6ca1b476ebc569711931a492417938b8d8df"


# ----------------------------------------------------------------------------- paths / env
def find_inputs(mode: str, root: Path | None = None) -> dict:
    if mode == "local":
        root = root or Path(__file__).resolve().parents[2]
        return {"train": root / "train.parquet", "test": root / "test.parquet",
                "sample": root / "sample_submission.csv", "work": root / "results" / "kaggle_v0_local",
                "ref_csv": root / "results" / "submission_v0" / "submission_v0.csv"}
    hits = sorted(glob.glob("/kaggle/input/**/test.parquet", recursive=True))
    hits = [h for h in hits if os.path.exists(os.path.join(os.path.dirname(h), "train.parquet"))]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one competition dir with train+test parquet, found {hits}")
    d = Path(hits[0]).parent
    return {"train": d / "train.parquet", "test": d / "test.parquet", "sample": d / "sample_submission.csv",
            "work": Path("/kaggle/temp/casmi_v0") if os.path.isdir("/kaggle/temp") else Path("/tmp/casmi_v0"),
            "out_csv": Path("/kaggle/working/submission.csv"), "ref_csv": None}


def n_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def available_ram_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 16.0


def connect(mem="2GB", threads=1, tmp=None):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET threads={threads}; SET preserve_insertion_order=false")
    if tmp:
        con.execute(f"SET temp_directory='{tmp}'")
    return con


# ----------------------------------------------------------------------------- V0 primitives
def make_spectrum(mzs, intensities, precursor_mz):
    """Identical to src/spectra/spectrum_io.py:make_spectrum."""
    from matchms import Spectrum
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    order = np.argsort(mzs)
    return Spectrum(mz=mzs[order], intensities=intensities[order],
                    metadata={"precursor_mz": float(precursor_mz)})


def load_test(test_path) -> pd.DataFrame:
    con = connect(mem="4GB", threads=2)
    t = con.execute(
        "SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities "
        f"FROM '{Path(test_path).as_posix()}'").fetchdf()
    con.close()
    return t.sort_values("precursor_mz").reset_index(drop=True)


def build_pool(train, test, pool_path: Path, work: Path, exclude_sql: str = ""):
    """Same rows as v0_make_submission_fast.cmd_build_pool, streamed by DuckDB (no pandas).
    exclude_sql is used ONLY by the local stress test (held-out molecules); empty on Kaggle."""
    if pool_path.exists():
        pool_path.unlink()
    t0 = time.time()
    mem_gb = max(2, int(available_ram_gb() * 0.45))  # leave room for the scoring workers that follow
    con = connect(mem=os.environ.get("CASMI_DUCKDB_MEM", f"{mem_gb}GB"), threads=4,
                  tmp=(work / "duck_tmp").as_posix())
    tr, te = Path(train).as_posix(), Path(test).as_posix()
    con.execute(f"""COPY (
        SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
        FROM '{tr}'
        WHERE precursor_mz BETWEEN (SELECT min(precursor_mz) - 0.01 FROM '{te}')
                               AND (SELECT max(precursor_mz) + 0.01 FROM '{te}')
              {exclude_sql}
        ORDER BY precursor_mz)
        TO '{pool_path.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 4096, COMPRESSION ZSTD)""")
    n = con.execute(f"SELECT count(*) FROM '{pool_path.as_posix()}'").fetchone()[0]
    con.close()
    return {"pool_rows": int(n), "pool_sec": round(time.time() - t0, 1), "duckdb_mem_gb": mem_gb,
            "pool_bytes": pool_path.stat().st_size}


def spectrum_ok(qr) -> str | None:
    """Reason a query spectrum cannot be scored, else None. Valid spectra take the V0 path unchanged."""
    pm, mz, it = qr["precursor_mz"], qr["ms2_mzs"], qr["ms2_normalized_intensities"]
    if pm is None or not np.isfinite(pm):
        return "precursor_mz missing/non-finite"
    if mz is None or it is None or len(mz) == 0:
        return "no peaks"
    if len(mz) != len(it):
        return "mz/intensity length mismatch"
    if not (np.all(np.isfinite(np.asarray(mz, dtype=float))) and np.all(np.isfinite(np.asarray(it, dtype=float)))):
        return "non-finite peak values"
    return None


def score_chunk(args):
    """Verbatim V0 per-query loop (band fetch). Returns (chunk_offset, rows, timing)."""
    c0, chunk, pool_path = args
    from matchms.similarity import ModifiedCosineGreedy
    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    fin = chunk["precursor_mz"][np.isfinite(chunk["precursor_mz"].astype(float))]
    loq = float(fin.min()) - PRECURSOR_TOL if len(fin) else 0.0
    hiq = float(fin.max()) + PRECURSOR_TOL if len(fin) else -1.0  # empty band when no finite precursor
    con = connect(mem="2GB", threads=1)
    lib = con.execute(
        f"""SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
            FROM '{pool_path}' WHERE precursor_mz BETWEEN {loq!r} AND {hiq!r}"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    con.close()
    pm = lib["precursor_mz"].values
    l_ik, l_ad = lib["inchikey14"].values, lib["adduct"].values
    l_mz, l_int = lib["ms2_mzs"].values, lib["ms2_normalized_intensities"].values

    ranges, used, skip = [], np.zeros(len(lib), dtype=bool), []
    for _, qr in chunk.iterrows():
        why = spectrum_ok(qr)
        skip.append(why)
        if why:
            ranges.append((0, 0))
            continue
        i0 = np.searchsorted(pm, qr["precursor_mz"] - PRECURSOR_TOL, side="left")
        i1 = np.searchsorted(pm, qr["precursor_mz"] + PRECURSOR_TOL, side="right")
        ranges.append((i0, i1))
        used[i0:i1] = True
    lib_specs = {j: make_spectrum(l_mz[j], l_int[j], pm[j]) for j in np.nonzero(used)[0]}

    rows, timing = [], []
    for (i0, i1), why, (_, qr) in zip(ranges, skip, chunk.iterrows()):
        ts = time.time()
        best, err = {}, why
        if not why:
            try:
                qspec = make_spectrum(qr["ms2_mzs"], qr["ms2_normalized_intensities"], qr["precursor_mz"])
                for j in range(i0, i1):
                    s = float(sim.pair(qspec, lib_specs[j])["score"])
                    key = (l_ik[j], l_ad[j])
                    if s > best.get(key, -1.0):
                        best[key] = s
            except Exception as e:  # recorded; this spectrum contributes no candidates
                best, err = {}, f"scoring error: {type(e).__name__}: {e}"[:300]
        for (ik, cad), s in best.items():
            rows.append((qr["molecule_id"], qr["spectrum_id"], qr["adduct"], ik, cad, s))
        mz = qr["ms2_mzs"]
        timing.append({"spectrum_id": qr["spectrum_id"], "molecule_id": qr["molecule_id"],
                       "n_candidates": 0 if err else int(i1 - i0), "n_peaks": 0 if mz is None else len(mz),
                       "skip_reason": err, "sec": time.time() - ts})
    return c0, rows, timing


SCORE_COLS = ["molecule_id", "spectrum_id", "query_adduct", "inchikey14", "cand_adduct", "score"]


def score_all(test: pd.DataFrame, pool_path: Path, workers: int):
    offsets = list(range(0, len(test), CHUNK))
    jobs = [(c0, test.iloc[c0:c0 + CHUNK], pool_path.as_posix()) for c0 in offsets]
    results = {}
    t0 = time.time()
    if workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(score_chunk, job): job[0] for job in jobs}
                for f in cf.as_completed(futs):
                    try:
                        c0, rows, timing = f.result()
                    except Exception as e:  # e.g. a worker killed for memory: redo that chunk serially
                        print(f"  chunk@{futs[f]} worker failed ({type(e).__name__}); will rerun serially", flush=True)
                        continue
                    results[c0] = (rows, timing)
                    print(f"  chunk@{c0}: {len(rows)} rows (elapsed {time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  process pool failed ({type(e).__name__}: {e}); falling back to serial", flush=True)
    for job in jobs:  # serial path, and retry for any chunk the pool did not return
        if job[0] not in results:
            c0, rows, timing = score_chunk(job)
            results[c0] = (rows, timing)
            print(f"  chunk@{c0} (serial): {len(rows)} rows (elapsed {time.time() - t0:.0f}s)", flush=True)
    rows = [r for c0 in offsets for r in results[c0][0]]          # fixed chunk order
    timing = [t for c0 in offsets for t in results[c0][1]]
    scores = pd.DataFrame(rows, columns=SCORE_COLS)
    return scores, pd.DataFrame(timing), round(time.time() - t0, 1)


# Aggregation variant. "max_all" is V0 (make_submission 'v1') and the only setting the parity gate accepts.
# "sum_same" is EXP-010's v1b (same-adduct candidates, SUM over the molecule's query spectra); it is built
# into a separate notebook (research/kaggle_v0b/) and never replaces V0 silently.
AGGREGATION = "max_all"


def rank_v1(scores: pd.DataFrame) -> dict:
    """Verbatim make_submission.rank_variant(scores, 'v1') when AGGREGATION == 'max_all'."""
    same, fuse = {"max_all": (False, "max"), "sum_same": (True, "sum")}[AGGREGATION]
    if same:
        scores = scores[scores["query_adduct"] == scores["cand_adduct"]]
    per_spec = scores.groupby(["molecule_id", "spectrum_id", "inchikey14"], as_index=False)["score"].max()
    mol = per_spec.groupby(["molecule_id", "inchikey14"], as_index=False)["score"].agg(fuse)
    mol = mol.sort_values(["molecule_id", "score", "inchikey14"], ascending=[True, False, True])
    return {mid: g["inchikey14"].head(TOP_N).tolist() for mid, g in mol.groupby("molecule_id")}


def smiles_for(train, inchikeys: set) -> dict:
    """Verbatim make_submission.smiles_for."""
    con = connect(mem="4GB", threads=4)
    con.register("k", pd.DataFrame({"ik": sorted(inchikeys)}))
    df = con.execute(
        f"""SELECT inchikey14, normalized_smiles, count(*) n FROM '{Path(train).as_posix()}'
            WHERE inchikey14 IN (SELECT ik FROM k) GROUP BY 1, 2""").fetchdf()
    con.close()
    df = df.sort_values(["inchikey14", "n", "normalized_smiles"], ascending=[True, False, True])
    return df.drop_duplicates("inchikey14").set_index("inchikey14")["normalized_smiles"].to_dict()


# ----------------------------------------------------------------------------- checks
def validate(sub, scores, ranked, test, expected_ids, timing=None, placeholder_ids=()) -> dict:
    """V0 validation A-E (v0_make_submission.validate), with the expected ids/spectra taken from
    the test file actually present, plus explicit top-25 / NaN / empty / schema checks.
    rep['fatal_ok'] covers only what the official metric rejects on (ids, schema, NaN, empty, >25);
    the remaining V0 checks are reported in rep['ok'] but do not stop the run."""
    from rdkit import Chem
    rep = {}
    placeholder_ids = set(placeholder_ids)
    sub_ids = sub["molecule_id"].tolist()
    rep["A_ids"] = {"rows": len(sub_ids), "expected_n": len(expected_ids),
                    "set_equal": set(sub_ids) == set(expected_ids),
                    "no_duplicates": len(sub_ids) == len(set(sub_ids)),
                    "order_matches_expected": sub_ids == list(expected_ids)}
    rep["A_ids"]["ok"] = all(rep["A_ids"][k] for k in ("set_equal", "no_duplicates", "order_matches_expected"))
    n_spec = len(test)
    cov = scores.groupby(["molecule_id", "spectrum_id"]).ngroups
    processed = int(timing["spectrum_id"].nunique()) if timing is not None else int(cov)
    no_cand = int((timing["n_candidates"] == 0).sum()) if timing is not None else 0
    rep["A_spectrum_coverage"] = {"test_spectra": n_spec, "spectra_processed_once": processed,
                                  "spectra_with_candidates": int(cov), "spectra_without_candidates": no_cand,
                                  "ok": processed == n_spec and cov + no_cand == n_spec}
    lens = sub["smiles"].fillna("").map(lambda s: len([p for p in s.split(";")]) if s else 0)
    rep["B_prediction_count"] = {"min": int(lens.min()), "max": int(lens.max()),
                                 "rows_gt_25": int((lens > TOP_N).sum()), "rows_zero": int((lens == 0).sum()),
                                 "ok": bool(lens.between(1, TOP_N).all())}
    invalid = empty = dup = 0
    for s in sub["smiles"].fillna(""):
        parts = s.split(";") if s else []
        empty += (not parts)
        seen = {}
        for p in parts:
            if not p:
                empty += 1
                continue
            m = Chem.MolFromSmiles(p)
            if m is None:
                invalid += 1
                continue
            c = Chem.MolToSmiles(m)
            seen[c] = seen.get(c, 0) + 1
        dup += sum(v - 1 for v in seen.values() if v > 1)
    rep["C_chemistry"] = {"invalid_smiles": invalid, "empty_fields": empty, "duplicate_within_molecule": dup,
                          "ok": invalid == 0 and empty == 0 and dup == 0}
    cols = list(sub.columns)
    rep["D_format"] = {"columns": cols, "exact_schema": cols == ["molecule_id", "smiles"],
                       "any_nan": bool(sub.isna().any().any()),
                       "ok": cols == ["molecule_id", "smiles"] and not sub.isna().any().any()}
    bad = 0
    for m, lst in ranked.items():
        sm = scores[scores["molecule_id"] == m]
        if AGGREGATION == "sum_same":
            sm = sm[sm["query_adduct"] == sm["cand_adduct"]]
        ps = sm.groupby(["spectrum_id", "inchikey14"], as_index=False)["score"].max()
        mo = ps.groupby("inchikey14", as_index=False)["score"].agg("sum" if AGGREGATION == "sum_same" else "max")
        mo = mo.sort_values(["score", "inchikey14"], ascending=[False, True])
        bad += mo["inchikey14"].head(TOP_N).tolist() != lst
    rep["E_ranking"] = {"molecules_misordered": int(bad), "ok": bad == 0}
    rows_ph = sub[sub["molecule_id"].isin(placeholder_ids)]
    rep["F_placeholder"] = {"rows": int(len(rows_ph)),
                            "ok": bool((rows_ph["smiles"] == PLACEHOLDER_SMILES).all())
                            and not (set(ranked) & placeholder_ids)}
    rep["ok"] = all(rep[k]["ok"] for k in list(rep))
    rep["fatal_ok"] = (rep["A_ids"]["set_equal"] and rep["A_ids"]["no_duplicates"] and rep["B_prediction_count"]["ok"]
                       and rep["D_format"]["ok"] and rep["C_chemistry"]["empty_fields"] == 0)
    return rep


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_lf(p) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_fingerprint(test: pd.DataFrame) -> str:
    key = "\n".join(sorted(test["molecule_id"] + "|" + test["spectrum_id"]))
    return hashlib.sha256(key.encode()).hexdigest()


# ----------------------------------------------------------------------------- main
def main(mode="kaggle", workers=None, root=None, test_override=None, holdout_labels=None, work_override=None):
    """test_override / holdout_labels / work_override: LOCAL STRESS TEST ONLY (a train-derived test file
    whose molecules are removed from the library). All three are None on Kaggle."""
    T0 = time.time()
    import matchms
    assert matchms.__version__ == MATCHMS_VERSION, f"matchms {matchms.__version__} != {MATCHMS_VERSION}"
    P = find_inputs(mode, Path(root) if root else None)
    exclude_sql = ""
    if test_override:
        P.update(test=Path(test_override), ref_csv=None, sample=Path("__none__"))
    if holdout_labels:
        exclude_sql = (f"AND inchikey14 NOT IN (SELECT DISTINCT inchikey14 FROM '{Path(holdout_labels).as_posix()}')")
    if work_override:
        P["work"] = Path(work_override)
    work = Path(P["work"]); work.mkdir(parents=True, exist_ok=True)
    out_csv = Path(P.get("out_csv") or work / "submission.csv")
    workers = workers or n_workers()
    report = {"mode": mode, "aggregation": AGGREGATION, "paths": {k: str(v) for k, v in P.items()}, "workers": workers,
              "versions": {"python": sys.version.split()[0], "matchms": matchms.__version__,
                           "duckdb": duckdb.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    try:
        import numba, rdkit
        report["versions"].update(numba=numba.__version__, rdkit=rdkit.__version__)
    except Exception as e:  # rdkit is required for validation below; numba for matchms
        raise RuntimeError(f"dependency import failed: {e}")

    # 1. inputs + schema
    test = load_test(P["test"])
    need = {"molecule_id", "spectrum_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"}
    assert need <= set(test.columns), f"test schema missing {need - set(test.columns)}"
    anomalies = {"duplicate_spectrum_id": int(test["spectrum_id"].duplicated().sum()),
                 "null_molecule_id_rows": int(test["molecule_id"].isna().sum())}
    if anomalies["null_molecule_id_rows"]:  # cannot be submitted under any id; recorded, not scored
        test = test[test["molecule_id"].notna()].reset_index(drop=True)
    if anomalies["duplicate_spectrum_id"]:  # keep every row; make the internal key unique, ids unchanged
        test["spectrum_id"] = test["spectrum_id"].astype(str) + "#" + test.groupby("spectrum_id").cumcount().astype(str)
    test_ids = sorted(test["molecule_id"].astype(str).unique())
    test["molecule_id"] = test["molecule_id"].astype(str)
    sample_ids = pd.read_csv(P["sample"])["molecule_id"].tolist() if Path(P["sample"]).exists() else []
    expected_ids = sample_ids if set(sample_ids) == set(test_ids) and len(sample_ids) == len(test_ids) else test_ids
    fp = test_fingerprint(test)
    report["test"] = {"spectra": len(test), "molecules": len(test_ids), "fingerprint": fp, "anomalies": anomalies,
                      "id_order": "sample_submission" if expected_ids is sample_ids else "sorted_test_ids",
                      "adducts": test["adduct"].value_counts().to_dict(),
                      "precursor_mz_range": [float(test["precursor_mz"].min()), float(test["precursor_mz"].max())]}
    print(json.dumps(report["test"], indent=1), flush=True)

    # 2. candidate pool (streamed)
    pool_path = work / "pool.parquet"
    report["pool"] = build_pool(P["train"], P["test"], pool_path, work, exclude_sql)
    print("pool:", report["pool"], flush=True)

    # 3. score every spectrum exactly once
    scores, timing, score_sec = score_all(test, pool_path, workers)
    report["score"] = {"score_sec": score_sec, "score_rows": len(scores),
                       "candidate_pairs": int(timing["n_candidates"].sum()),
                       "spectra_scored": int(timing["spectrum_id"].nunique())}
    assert timing["spectrum_id"].is_unique and len(timing) == len(test), "each spectrum must be processed exactly once"
    assert len(scores) == len(scores.drop_duplicates(["spectrum_id", "inchikey14", "cand_adduct"]))

    # 4. zero-candidate accounting; placeholder for zero-candidate molecule_ids ONLY (see docstring)
    zero_spec = timing[timing["n_candidates"] == 0]
    ranked = rank_v1(scores)
    zero_mols = sorted(set(test_ids) - set(ranked))
    report["zero_candidate"] = {"spectra": int(len(zero_spec)), "molecules": len(zero_mols),
                                "molecule_ids": zero_mols, "placeholder_smiles": PLACEHOLDER_SMILES,
                                "spectra_detail": zero_spec[["molecule_id", "spectrum_id", "skip_reason"]]
                                .to_dict("records")}
    print(f"zero-candidate: {len(zero_spec)} spectra, {len(zero_mols)} molecules -> placeholder", flush=True)

    # 5. assemble (V0 rows unchanged; zero-candidate rows get the placeholder)
    all_iks = {ik for lst in ranked.values() for ik in lst}
    ik2smi = smiles_for(P["train"], all_iks)
    no_smi = sorted(all_iks - set(ik2smi))
    report["inchikey14_without_smiles"] = no_smi  # dropped from lists (never observed; train always has SMILES)

    def row(m):
        preds = [ik2smi[ik] for ik in ranked.get(m, []) if ik in ik2smi]
        return ";".join(preds) if preds else PLACEHOLDER_SMILES
    sub = pd.DataFrame({"molecule_id": expected_ids, "smiles": [row(m) for m in expected_ids]})
    placeholder_ids = [m for m, s in zip(sub["molecule_id"], sub["smiles"]) if s == PLACEHOLDER_SMILES and not ranked.get(m)]
    report["zero_candidate"]["placeholder_rows"] = len(placeholder_ids)

    # 6. checks: metric-rejecting conditions are fatal; the other V0 checks are reported
    report["validation"] = validate(sub, scores, ranked, test, expected_ids, timing, placeholder_ids)
    print("validation ok:", report["validation"]["ok"], "fatal_ok:", report["validation"]["fatal_ok"], flush=True)
    json.dump(report, open(work / "v0_run_report.json", "w"), indent=1, default=str)
    if not report["validation"]["fatal_ok"]:
        raise RuntimeError(f"submission would be rejected by the metric: {report['validation']}")

    # 7. write + parity
    sub.to_csv(out_csv, index=False)
    back = pd.read_csv(out_csv, keep_default_na=False)
    assert list(back.columns) == ["molecule_id", "smiles"] and (back["smiles"] != "").all()
    report["output"] = {"path": str(out_csv), "sha256_raw": sha256_file(out_csv), "sha256_lf": sha256_lf(out_csv)}
    # The V0 byte-parity reference applies to the V0 aggregation only.
    parity = {"visible_test": fp == VISIBLE_TEST_FINGERPRINT and AGGREGATION == "max_all", "aggregation": AGGREGATION}
    if P.get("ref_csv") and Path(P["ref_csv"]).exists():
        ref = pd.read_csv(P["ref_csv"], keep_default_na=False)
        parity.update(ref_sha256_raw=sha256_file(P["ref_csv"]), raw_identical=sha256_file(P["ref_csv"]) == sha256_file(out_csv),
                      rows_identical=ref.equals(back))
    if parity["visible_test"] and REF_SHA256_LF:
        parity["lf_identical_to_v0"] = report["output"]["sha256_lf"] == REF_SHA256_LF
    report["parity"] = parity
    report["runtime_sec_total"] = round(time.time() - T0, 1)
    timing.to_csv(work / "timing.csv", index=False)
    json.dump(report, open(work / "v0_run_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("score", "zero_candidate", "output", "parity", "runtime_sec_total")},
                     indent=1, default=str), flush=True)
    if parity.get("visible_test") and parity.get("lf_identical_to_v0") is False:
        raise RuntimeError("PARITY FAILURE on the visible test: output differs from validated V0")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "kaggle"], default="local")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--test-override", help="LOCAL STRESS TEST: alternative test parquet")
    ap.add_argument("--holdout-labels", help="LOCAL STRESS TEST: parquet with inchikey14 to remove from the library")
    ap.add_argument("--work", help="LOCAL STRESS TEST: output directory")
    ap.add_argument("--aggregation", choices=["max_all", "sum_same"], default="max_all")
    a = ap.parse_args()
    AGGREGATION = a.aggregation
    main(a.mode, a.workers, test_override=a.test_override, holdout_labels=a.holdout_labels, work_override=a.work)

"""MRR@25 of past submissions on the visible test.parquet (a train sample: every spectrum is an exact enveda-180
duplicate, so the truth is known and the library has the exact spectrum).

    python research/scripts/c3_visible_eval.py

Lists: E1 = results/bench/kaggle_e1_output/eng_lists.json, V2 = results/kaggle_v2_local/our_lists.json,
E2 = weighted RRF of the two exactly as bench_eval.fuse (BETA 0.4, K 3). Truth = train inchikey14 of the exact
duplicate spectrum. Candidate keys use the bench's canonicaliser (metric key, tautomer canonical; cached).
Writes results/c3/visible_eval.json.
"""
import json, sys
from pathlib import Path
import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench, bench_eval as BE  # noqa: E402



def rr_of(keys, t):
    for i, k in enumerate(keys[:25]):
        if k == t:
            return 1.0 / (i + 1)
    return 0.0


def visible_truth():
    """{test molecule_id: train inchikey14} via the exact enveda-180 duplicate spectrum."""
    TE, TR = (ROOT / "test.parquet").as_posix(), (ROOT / "train.parquet").as_posix()
    return dict(duckdb.sql(f"""  -- prefilter enveda-180 rows on scalar columns, then compare peak lists
        WITH t AS (SELECT molecule_id, precursor_mz, adduct, ms2_mzs FROM read_parquet('{TE}')),
             r AS (SELECT inchikey14, precursor_mz, adduct, ms2_mzs FROM read_parquet('{TR}')
                   WHERE ingest_lib = 'enveda-180'
                     AND precursor_mz IN (SELECT precursor_mz FROM t))
        SELECT t.molecule_id, any_value(r.inchikey14)
        FROM t JOIN r USING (precursor_mz, adduct)
        WHERE r.ms2_mzs = t.ms2_mzs
        GROUP BY 1""").fetchall())


def main():
    truth = visible_truth()
    print(f"truth for {len(truth)} visible molecules", flush=True)
    E1 = json.load(open(ROOT / "results/bench/kaggle_e1_output/eng_lists.json"))
    V2 = json.load(open(ROOT / "results/kaggle_v2_local/our_lists.json"))
    bench.setup_env(ROOT / "research/bench/eng", 4)
    import casmi_engine as E  # noqa: E402
    smis = sorted({s for L in (E1, V2) for v in L.values() for s in v["smiles"] if s})
    C = bench.canon_many(E, smis, 4)
    key = lambda s: C.get(s) if s else None


    res = {}
    rows = {"E1": [], "V2": [], "E2": []}
    for m, t in truth.items():
        e = [key(s) for s in E1.get(m, {"smiles": []})["smiles"]]
        o = [key(s) for s in V2.get(m, {"smiles": []})["smiles"]]
        order, _ = BE.fuse({"keys": e, "raw": [None] * len(e)}, {"keys": o, "raw": [None] * len(o)}, 0.4)
        for nm, L in (("E1", e), ("V2", o), ("E2", order)):
            rows[nm].append(rr_of(L, t))
    for nm, v in rows.items():
        v = np.array(v)
        res[nm] = {"n": len(v), "mrr": round(float(v.mean()), 4), "top1": int((v == 1).sum())}
    json.dump(res, open(ROOT / "results/c3/visible_eval.json", "w"), indent=1)
    print(f"{len(truth)} visible molecules with a train duplicate;", res)


if __name__ == "__main__":  # Windows spawn: workers of canon_many re-import this file
    main()

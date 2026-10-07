"""EXP-010 scoring (V0 scorer, unchanged): produce per-(spectrum, candidate) V0 scores for two
labelled benchmarks, so aggregation variants can be compared without rescoring.

  P1  Class-1 proxy: enveda-np-examples spectra (same pipeline as the hidden test) as queries;
      library = all train EXCEPT the enveda-np-examples library (same molecules stay in public libs).
      Built on 2026-09-25 by the c1proxy run (scores copied to results/exp010_v0_aggregation/).
  P2  visible test.parquet as queries; library = all train minus the 1,213 byte-identical copies of the
      test spectra (same holdout rule as make_submission.py --holdout-exact-copies).

    python research/scripts/exp010_score_benchmarks.py P2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_v0"))
import casmi_v0_kaggle as K  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "exp010_v0_aggregation"

if __name__ == "__main__":
    assert sys.argv[1] == "P2"
    OUT.mkdir(parents=True, exist_ok=True)
    test = K.load_test("test.parquet")
    excl = ("AND hash(precursor_mz, ms2_mzs, ms2_normalized_intensities) NOT IN "
            "(SELECT hash(precursor_mz, ms2_mzs, ms2_normalized_intensities) FROM 'test.parquet')")
    print(K.build_pool("train.parquet", "test.parquet", OUT / "pool_P2.parquet", OUT, excl), flush=True)
    scores, timing, sec = K.score_all(test, OUT / "pool_P2.parquet", 8)
    scores.to_parquet(OUT / "scores_P2_visible_holdout.parquet")
    timing.to_csv(OUT / "timing_P2.csv", index=False)
    (OUT / "pool_P2.parquet").unlink()
    print("scored", len(scores), "rows in", sec, "s", flush=True)

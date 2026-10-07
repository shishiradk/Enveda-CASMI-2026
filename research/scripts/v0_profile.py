"""Profile the V0 scoring hot path on the smoke set (20 molecules, 56 spectra).

Times three components separately, mirroring src/submission/make_submission.py cmd_score:
  fetch : per-chunk DuckDB SEMI JOIN over train.parquet -> candidate library rows
  build : make_spectrum() construction of query + candidate Spectrum objects
  pair  : ModifiedCosineGreedy.pair() scoring calls

Also reports how many times the same library row gets rebuilt across queries
(duplicate construction factor) to size the Spectrum-cache optimization.

Usage:
  python research/scripts/v0_profile.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.submission.make_submission as ms  # noqa: E402
from spectra.spectrum_io import make_spectrum  # noqa: E402

OUT = ROOT / "results" / "submission_v0"


def main():
    ms.OUT_DIR = OUT
    con = ms.connect()
    test = con.execute(
        "SELECT molecule_id, spectrum_id, adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities "
        f"FROM '{ms.TEST}'"
    ).fetchdf()
    keep = sorted(test["molecule_id"].unique())
    keep = pd.Series(keep).sample(n=20, random_state=20260924).tolist()
    test = test[test["molecule_id"].isin(keep)]
    test = test.sort_values("precursor_mz").reset_index(drop=True)
    print(f"smoke: {len(test)} spectra / {test['molecule_id'].nunique()} molecule_ids")

    sim = ms.ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)

    t_fetch = t_build_q = t_build_c = t_pair = 0.0
    n_lib_rows = 0
    n_pairs = 0
    n_dup_builds = 0
    raw_lens = []

    for c0 in range(0, len(test), ms.CHUNK):
        chunk = test.iloc[c0:c0 + ms.CHUNK]
        t0 = time.time()
        con.register("q", chunk[["spectrum_id", "precursor_mz"]])
        lib = con.execute(
            f"""SELECT t.precursor_mz, t.adduct, t.inchikey14,
                       t.ms2_mzs, t.ms2_normalized_intensities
                FROM '{ms.TRAIN}' t SEMI JOIN q
                ON t.precursor_mz BETWEEN q.precursor_mz - {ms.PRECURSOR_TOL}
                                      AND q.precursor_mz + {ms.PRECURSOR_TOL}
                WHERE TRUE"""
        ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
        con.unregister("q")
        t_fetch += time.time() - t0

        pm = lib["precursor_mz"].values
        l_ik, l_ad = lib["inchikey14"].values, lib["adduct"].values
        l_mz, l_int = lib["ms2_mzs"].values, lib["ms2_normalized_intensities"].values

        # Simulate the exact current loop, timing each phase.
        for _, qr in chunk.iterrows():
            t0 = time.time()
            qspec = make_spectrum(qr["ms2_mzs"], qr["ms2_normalized_intensities"], qr["precursor_mz"])
            t_build_q += time.time() - t0
            i0 = np.searchsorted(pm, qr["precursor_mz"] - ms.PRECURSOR_TOL, side="left")
            i1 = np.searchsorted(pm, qr["precursor_mz"] + ms.PRECURSOR_TOL, side="right")
            best = {}
            for j in range(i0, i1):
                t0 = time.time()
                cand = make_spectrum(l_mz[j], l_int[j], pm[j])
                t_build_c += time.time() - t0
                raw_lens.append(len(l_mz[j]))
                t0 = time.time()
                s = float(sim.pair(qspec, cand)["score"])
                t_pair += time.time() - t0
                key = (l_ik[j], l_ad[j])
                if s > best.get(key, -1.0):
                    best[key] = s
            for (ik, cad), s in best.items():
                pass
            n_pairs += i1 - i0
        n_lib_rows += len(lib)
        del lib
        print(f"  chunk@{c0}: fetch={t_fetch:.1f}s build_q={t_build_q:.1f}s "
              f"build_c={t_build_c:.1f}s pair={t_pair:.1f}s pairs={n_pairs}")

    print()
    print(f"TOTAL fetch       : {t_fetch:.1f}s  ({t_fetch / max(1, (len(test) - 1) // ms.CHUNK + 1):.2f}s/chunk)")
    print(f"TOTAL build query : {t_build_q:.1f}s")
    print(f"TOTAL build cand  : {t_build_c:.1f}s")
    print(f"TOTAL pair        : {t_pair:.1f}s")
    print(f"gamma: fetch+pair = {(t_fetch + t_pair):.1f}s, build = {(t_build_q + t_build_c):.1f}s")
    print(f"n_lib_rows(chunks summed) = {n_lib_rows}, n_pairs = {n_pairs}, "
          f"rebuilt_per_lib_row = {n_pairs / max(1, len(raw_lens)):.2f}")
    print(f"lib m/z list len: n={len(raw_lens)} mean={np.mean(raw_lens):.1f} "
          f"median={np.median(raw_lens):.1f} max={np.max(raw_lens)}")


if __name__ == "__main__":
    main()
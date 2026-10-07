"""Smoke test for the review fixes in casmi_v2_kaggle.py: NaN precursor, unknown adduct, empty input.

    python research/kaggle_v2/smoke_review_fixes.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casmi_v2_kaggle as v2  # noqa: E402

v1 = v2.v1


def main():
    P = v1.v0.find_inputs("local")
    work = Path("results/kaggle_v2_local")
    work.mkdir(parents=True, exist_ok=True)
    t = v1.v0.load_test(P["test"])
    t["molecule_id"] = t["molecule_id"].astype(str)
    keep = sorted(t["molecule_id"].unique())[:20]
    t = t[t["molecule_id"].isin(keep)].reset_index(drop=True)
    m0 = next(m for m in keep if (t["molecule_id"] == m).sum() >= 2)
    t.loc[t.index[t["molecule_id"] == m0][0], "precursor_mz"] = np.nan  # NaN precursor in a multi-spectrum molecule
    m1 = next(m for m in keep if m != m0)
    t.loc[t["molecule_id"] == m1, "adduct"] = "[M+HCOO]-"  # whole molecule with an unknown adduct
    pred, U, pc, fp, info = v2.run_v2(P, t, work, 4, v2.CFG, assets=v1.find_assets("local"), pc_path=v2.find_pubchem("local"))
    print(info)
    assert m0 in pred and np.isfinite(pred[m0][0]), "NaN-precursor molecule lost"
    assert m1 not in pred, "unknown-adduct molecule should have no prediction (placeholder downstream)"
    sc = v2.score_molecules(pred, U, pc, fp)
    print("molecules scored", len(sc), "median candidates", np.median([len(v) for v in sc.values()]))
    from scipy import sparse
    ref0 = (sparse.csr_matrix((0, 200000), dtype=np.float32), np.array([], dtype=object), np.array([], dtype=bool))
    empty, einfo = v2.predict_molecules(t.assign(adduct="[X]"), U, ref0, work, 1)
    assert empty == {}, empty
    print("all-unknown-adduct ->", empty, einfo)
    print("SMOKE OK")


if __name__ == "__main__":
    main()

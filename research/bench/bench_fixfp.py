"""Recompute the fingerprint logits of a saved scenario with a missing collision energy read as 25 eV.

    python research/bench/bench_fixfp.py [TAG=e1] [SCEN=S3]
    python research/bench/bench.py rescore --tag e1 --scen S3        # afterwards

Why: casmi_engine.parse_ce() returns NaN (not its 25.0 default) for a NULL collision_energy_ev, the per-spectrum FPNet
then outputs NaN and the whole molecule's logit vector becomes NaN. The competition test set has a collision energy on
every row, so Kaggle is unaffected, but the S3 queries are public-library spectra and 451 of 829 have none: 235 of 300
S3 molecules ran with a dead fingerprint channel. 25 eV is what the author's trainer feeds for a missing energy
(train_fp.py: ce < 0 -> 25.0). Only `zlog` (public nets) and `zmk` (megayak nets) in recs_<SCEN>.pkl are replaced; the
library / analog / fragmentation channels do not depend on them. The as-run file is kept as recs_<SCEN>_nanfp.pkl.
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "e1"
    scen = sys.argv[2] if len(sys.argv) > 2 else "S3"
    bench.setup_env(HERE / "eng", 3)
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    torch.set_num_threads(3)
    import casmi_engine as E
    import pv_fp
    _pce = E.parse_ce

    def parse_ce(v):
        x = _pce(v)
        return x if np.isfinite(x) else 25.0
    E.parse_ce = parse_ce
    out = bench.OUT / tag
    D = pickle.load(open(out / f"recs_{scen}.pkl", "rb"))
    te = pq.read_table(bench.OUT / f"queries_{'S12' if scen[:2] in ('S1', 'S2') else 'S3'}.parquet").to_pandas()
    te = te[te.molecule_id.isin({r["mid"] for r in D["recs"]})]
    pub = pv_fp.load_fp_models(sorted(str(p) for p in (bench.INPUTS / "fp").glob("fp_*.pt")), "cpu")
    z_pub = bench.alt_zlogs(E, pv_fp, (pub[0], pub[1], "cpu"), te)
    z_mk = {}
    if bench.ALT_FP.exists():
        mk = pv_fp.load_fp_models(sorted(str(p) for p in bench.ALT_FP.glob("*.pt")), "cpu")
        z_mk = bench.alt_zlogs(E, pv_fp, (mk[0], mk[1], "cpu"), te)
    was_nan = same = 0
    dmax = 0.0
    for r in D["recs"]:
        if not len(r["cand"]):
            continue
        old, new = r.get("zlog"), z_pub.get(r["mid"])
        if old is not None and np.isnan(old).any():
            was_nan += 1
        elif old is not None and new is not None:
            same += 1; dmax = max(dmax, float(np.abs(old - new).max()))
        r["zlog"] = new
    print(f"{scen}: {was_nan} molecules had NaN logits; {same} others recomputed, max |old - new| = {dmax:.2e}; "
          f"NaN left: {sum(1 for r in D['recs'] if r.get('zlog') is not None and np.isnan(r['zlog']).any())}")
    if z_mk:
        D["zmk"] = z_mk
    bak = out / f"recs_{scen}_nanfp.pkl"
    if not bak.exists():
        (out / f"recs_{scen}.pkl").replace(bak)
    pickle.dump(D, open(out / f"recs_{scen}.pkl", "wb"), protocol=4)

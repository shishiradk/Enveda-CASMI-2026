"""Local test (not shipped to the GPU machine): the inference API on RAW train.parquet rows must reproduce the
prepared training arrays, and score() must agree with the validation code.

    python research/train_cft/parity_test.py --data results/train_pkg/data results/train_cft/data --ckpt <cft_*.pt>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cft_model as M
import train_cft as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", default=str(HERE.parent.parent / "train.parquet"))
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import duckdb, torch
    D = T.Data(a.data, mmap=True, log=print)
    sc = M.CFTScorer(a.ckpt)
    rows = np.load(T.find_file(D.dirs, "spec_row.npy"))
    rng = np.random.default_rng(5)
    n_pk = np.diff(D.off)
    capped = np.where(n_pk == M.MAX_PEAKS)[0]
    ids = np.unique(np.concatenate([rng.choice(len(rows), a.n, replace=False), rng.choice(capped, a.n // 4, replace=False)]))
    con = duckdb.connect(); con.execute("set threads=2; set memory_limit='2GB'")
    con.execute("create table want(r BIGINT)"); con.register("w", {"r": rows[ids].astype(np.int64)})
    con.execute("insert into want select r from w")
    got = con.execute(f"""select file_row_number r, ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct, collision_energy_ev
                          from read_parquet('{Path(a.train).as_posix()}', file_row_number=true)
                          where file_row_number in (select r from want)""").fetchall()
    by = {g[0]: g for g in got}
    bad_len = bad_val = bad_meta = 0; mx = 0.0; capped_checked = 0
    for i in ids:
        _, mz, it, prec, ad, ce = by[int(rows[i])]
        p = sc.prepare(dict(mz=mz, intensity=it, precursor_mz=prec, adduct=ad, collision_energy=ce))
        ref_mz = np.asarray(D.mz[D.off[i]:D.off[i + 1]]); ref_it = np.asarray(D.it[D.off[i]:D.off[i + 1]])
        capped_checked += len(ref_mz) == M.MAX_PEAKS
        if len(p["mz"]) != len(ref_mz): bad_len += 1; continue
        d = max(float(np.abs(p["mz"] - ref_mz).max()), float(np.abs(p["it"] - ref_it).max())); mx = max(mx, d)
        bad_val += d > 1e-6
        ce_ok = (p["ce"] < 0 and D.ce[i] < 0) or abs(p["ce"] - D.ce[i]) < 1e-3
        bad_meta += not (p["ad"] == D.ad[i] and ce_ok and abs(p["prec"] - D.prec[i]) < 1e-6)
    res = dict(spectra_checked=int(len(ids)), at_peak_cap=int(capped_checked), different_peak_count=int(bad_len),
               different_values=int(bad_val), max_abs_diff_when_same_count=mx, different_adduct_ce_or_precursor=int(bad_meta))
    # score() against the validation code on a few molecules
    V = D.build_val(30, pcv_limit=5, log=lambda *x: None)
    Z = T.val_logits(sc.net, V, "cpu", False)
    worst = 0.0
    for j in np.where(V["in_pool_set"])[0][:10]:
        specs, sid = D.spectra_of(int(V["mols"][j]), 6, 12345)
        raw = []
        for i in sid:
            g = con.execute(f"""select ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct, collision_energy_ev,
                                instrument_type from read_parquet('{Path(a.train).as_posix()}', file_row_number=true)
                                where file_row_number = {int(rows[i])}""").fetchone()
            raw.append(dict(mz=g[0], intensity=g[1], precursor_mz=g[2], adduct=g[3], collision_energy=g[4],
                            instrument=D.instruments[int(D.ins[i])]))
        cr = V["c_rows"][V["c_owner"] == j][:50]
        if not len(cr): continue
        s_api = sc.score(raw, np.asarray(D.fp[cr]))
        s_val = (torch.as_tensor(np.unpackbits(np.asarray(D.fp[cr]), axis=1)[:, :D.nbits]).float() @ Z[j]).numpy()
        worst = max(worst, float(np.abs(s_api - s_val).max() / max(1.0, np.abs(s_val).max())))
    res["score_api_vs_validation_max_rel_diff"] = worst
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

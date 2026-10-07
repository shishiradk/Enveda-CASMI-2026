"""E6 with CFT as the PubChem-channel scorer, on the S3 molecules (PubChem-only and Class-3 buckets), compared with
the ho1-net channel on identical windows.

    python research/scripts/e6_cft_channel.py [--workers 4] [--ckpt models/cft_kaggle/out_cft/export/cft_default.pt]

Windows: results/c3/e6_work/windows_5.0ppm.parquet (scen S3, same as e6_pc_channel.py). Queries: the S3 bench
spectra (results/kaggle_v2_proxy/s3_test.parquet; no CE / instrument, CFT defaults apply). Fingerprints: CFT's own
11,170-bit Fingerprinter(sc.fp_bits), cached in results/c3/e6_work/cft_fp_S3.npz. Score = fp @ CFT logits, top 60,
metric-key de-duplication (bench canonicaliser), 40 kept -> results/c3/e6_work/pc_lists_cft_S3.pkl.
CFT was trained without the S1/S2/S3 held keys (prep_cft.py --held), so S3 is leak-free for it.
Prints the channel-alone MRR@25 of both scorers and the m1-alt merge per bucket; writes results/c3/e6_cft.json.
"""
import argparse, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "results" / "c3" / "e6_work"
sys.path.insert(0, str(ROOT / "research" / "bench"))
sys.path.insert(0, str(ROOT / "research" / "train_cft"))
_g = {}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def _init(bits):
    import cft_fp
    _g["F"] = cft_fp.Fingerprinter(bits)


def _fp(smi):
    try:
        return _g["F"].packed(smi)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=str(ROOT / "models/cft_kaggle/out_cft/export/cft_default.pt"))
    a = ap.parse_args()
    import cft_model
    sc = cft_model.CFTScorer(a.ckpt, device="cpu")
    w = pd.read_parquet(W / "windows_5.0ppm.parquet")
    w = w[w.scen == "S3"].reset_index(drop=True)
    u = sorted(set(w.ik))
    smi_of = dict(zip(w.ik, w.smiles))
    f = W / "cft_fp_S3.npz"
    if not f.exists():
        with Pool(a.workers, initializer=_init, initargs=(sc.fp_bits,)) as pool:
            fps = pool.map(_fp, [smi_of[k] for k in u], chunksize=200)
        ok = np.array([x is not None for x in fps])
        nb = len(next(x for x in fps if x is not None))
        np.savez(f, ik=np.asarray(u, dtype="U14"), fp=np.stack([x if x is not None else np.zeros(nb, np.uint8)
                                                                  for x in fps]), ok=ok)
        log(f"CFT fingerprints: {len(u):,} structures, {ok.mean():.4f} parsed")
    z = np.load(f)
    pos = pd.Series(np.arange(len(z["ik"])), index=z["ik"])
    fp_all, ok_all = z["fp"], z["ok"]
    q = pd.read_parquet(ROOT / "results/kaggle_v2_proxy/s3_test.parquet")
    top = {}
    for mid, g in w.groupby("mid"):
        sp = q[q.molecule_id == mid]
        if not len(sp):
            continue
        specs = [dict(mz=np.asarray(r.ms2_mzs), intensity=np.asarray(r.ms2_normalized_intensities),
                      precursor_mz=r.precursor_mz, adduct=r.adduct, collision_energy=None) for r in sp.itertuples()]
        p = pos.reindex(g.ik.values).values.astype(np.int64)
        p = p[ok_all[p]]
        s = sc.score(specs, fp_all[p])
        o = np.argsort(-s, kind="mergesort")[:60]
        top[mid] = [(z["ik"][p[j]], float(s[j])) for j in o]
    log(f"CFT scored {len(top)} molecules")
    import bench
    bench.setup_env(ROOT / "research/bench/eng", a.workers)
    import casmi_engine as E
    C = bench.canon_many(E, sorted({smi_of[k] for v in top.values() for k, _ in v}), a.workers)
    res = {}
    for mid, v in top.items():
        rec = {"keys": [], "raw": [], "smiles": [], "score": []}
        for k, s in v:
            key = C.get(smi_of[k]) or k
            if key in rec["keys"]:
                continue
            rec["keys"].append(key); rec["raw"].append(k); rec["smiles"].append(smi_of[k]); rec["score"].append(s)
            if len(rec["keys"]) >= 40:
                break
        res[mid] = rec
    pickle.dump(res, open(W / "pc_lists_cft_S3.pkl", "wb"), protocol=4)

    import bench_eval as BE, e6_merge_eval as M
    truth = bench.load_truth()["S3"]
    cls = pd.read_parquet(ROOT / "results/c3/train_classes.parquet").set_index("ik")
    R = pickle.load(open(bench.OUT / "e1" / "S3.pkl", "rb"))
    H = pickle.load(open(W / "pc_lists_S3.pkl", "rb"))
    out = {}
    for name, P in (("ho1", H), ("cft", res)):
        rows = []
        for mid in sorted(R):
            r = R[mid]
            b = "PC" if (mid in cls.index and cls.at[mid, "in_pc"] and not cls.at[mid, "in_coco"]) else "C3"
            _, keys, idx = BE.engine_list(r, "blend")
            e = dict(keys=keys, raw=[r["keys"][i] for i in idx])
            p = P.get(mid, {})
            ch = BE.rr(BE.rank_in(p.get("keys", []), p.get("raw", []), r, truth[mid]["correct"]))
            ks, rs = M.merge(e, p, 1, "alt")
            mg = BE.rr(BE.rank_in(ks, rs, r, truth[mid]["correct"]))
            rows.append((b, ch, mg))
        d = pd.DataFrame(rows, columns=["b", "channel", "m1_alt"])
        out[name] = {b: dict(n=int(len(g)), channel=round(g.channel.mean(), 4), m1_alt=round(g.m1_alt.mean(), 4))
                     for b, g in d.groupby("b")}
        log(f"{name}: {out[name]}")
    json.dump(out, open(ROOT / "results/c3/e6_cft.json", "w"), indent=1)


if __name__ == "__main__":
    main()

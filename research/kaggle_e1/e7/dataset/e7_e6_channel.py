"""E6 PubChem channel as a drop-in engine: writes ranked PubChem candidate lists, never submission.csv.

E7 copy of research/kaggle_e1/e6/e6_channel.py (shipped in dataset casmi-e7-c3assets). The E6 lists it writes are
unchanged: the only additions are --dump-ctx (per-molecule target, zlog and the top --dump-n PubChem window rows by
fp @ zlog, for the E7 Class-3 channel e7_c3_channel.py) and --duck-mem (DuckDB memory limit, default the E6 value).

    python e7_e6_channel.py TEST_PARQUET OUT_JSON --eng-dir DIR --nets DIR --pubchem PARQUET --fp-bits NPY
                         [--workers 4] [--budget 14400] [--ppm 5] [--limit N]
                         [--dump-ctx CTX.pkl] [--dump-n 280] [--duck-mem 10GB]

Per molecule: target = median neutral mass of its spectra (pv.neutral_mass, as casmi_engine.compute_channels);
zlog = our fingerprint nets' logits (pv_fp.molecule_logits, as bench.alt_zlogs); every PubChem structure within
+-ppm of the target (pubchem_rows.parquet); its 6,930-bit fingerprint (ECFP4|ECFP6|RDKitFP|MACCS -> fp_bits, as
train_pkg/prep_data.fp_and_mass); score = fp @ zlog; top 60; tautomer-canonical InChIKey14 de-duplication; 40 kept.
Same arithmetic as research/scripts/e6_pc_channel.py (offline bench run). Molecules are processed in batches until
--budget seconds have passed; molecules left over get no list (the notebook then keeps the engine list unchanged).
OUT_JSON = {molecule_id: {"smiles": [...], "keys": [...], "scores": [...]}} plus "_stats".
CTX.pkl = {"mols": {molecule_id: {"target": float, "zlog": float32[6930], "pc": [(ik, smiles, score), ...] best first
(None when the budget ran out before the molecule)}}, "meta": {...}}, written even when the scoring loop stops early.
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

_g = {}


def _init(bits_path):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    _g["bits"] = np.load(bits_path)
    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    _g["te"] = rdMolStandardize.TautomerEnumerator()


def _fp(smi):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]
        return np.packbits(fp)
    except Exception:
        return None


def _key(smi):
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def parse_ce(v):  # casmi_engine.parse_ce
    try:
        a = np.abs(np.atleast_1d(np.asarray(v, float)))
        return float(a.mean()) if len(a) else 25.0
    except Exception:
        return 25.0


def targets_and_logits(te, a):
    sys.path.insert(0, a.eng_dir)
    import pv, pv_fp, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = sorted(glob.glob(os.path.join(a.nets, "*.pt")))
    s, m, _ = pv_fp.load_fp_models(nets, dev)
    print("E6 nets:", [os.path.basename(p) for p in nets], "| single", len(s), "merged", len(m), "|", dev, flush=True)
    out = {}
    with torch.no_grad():
        for mid, sub in te.groupby("molecule_id", sort=True):
            nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)
            nms = nms[np.isfinite(nms)]
            if not len(nms):
                continue
            specs = []
            for r in sub.itertuples():
                a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)
                k_ = a_ <= float(r.precursor_mz) + 2.0
                a_, b_ = a_[k_], b_[k_]
                specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))
            modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)
            ces = np.array([parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)
            z = pv_fp.molecule_logits((s, m, dev), specs, sub.precursor_mz.values.astype(np.float32),
                                      list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),
                                      ces, modes)
            if z is not None:
                out[str(mid)] = (float(np.median(nms)), np.asarray(z, np.float32))
    return out


def windows(L, a):
    import duckdb
    iv = pd.DataFrame([(mid, t * (1 - a.ppm * 1e-6), t * (1 + a.ppm * 1e-6)) for mid, (t, _) in L.items()],
                      columns=["mid", "lo", "hi"])
    ivb = pd.DataFrame([(r.mid, r.lo, r.hi, b) for r in iv.itertuples()
                        for b in range(int(np.floor(r.lo * 100)), int(np.floor(r.hi * 100)) + 1)],
                       columns=["mid", "lo", "hi", "bin"])
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '{a.duck_mem}'")
    con.register("ivb", ivb)
    w = con.sql(f"""SELECT ivb.mid, p.ik, p.smiles FROM read_parquet('{a.pubchem}') p
                    JOIN ivb ON CAST(floor(p.mass * 100) AS BIGINT) = ivb.bin
                    WHERE p.mass BETWEEN ivb.lo AND ivb.hi""").df()
    return w.drop_duplicates(["mid", "ik"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test"); ap.add_argument("out")
    ap.add_argument("--eng-dir", required=True); ap.add_argument("--nets", required=True)
    ap.add_argument("--pubchem", required=True); ap.add_argument("--fp-bits", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--budget", type=float, default=14400)
    ap.add_argument("--ppm", type=float, default=5.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-ctx", default=""); ap.add_argument("--dump-n", type=int, default=280)
    ap.add_argument("--duck-mem", default="10GB")
    a = ap.parse_args()
    T0 = time.time()
    te = pd.read_parquet(a.test)
    te = te[te.molecule_id.notna()]
    te["molecule_id"] = te.molecule_id.astype(str)
    if a.limit:
        te = te[te.molecule_id.isin(sorted(te.molecule_id.unique())[:a.limit])]
    L = targets_and_logits(te, a)
    w = windows(L, a)
    print(f"E6 logits {len(L)} molecules | windows {len(w):,} pairs, {w.ik.nunique():,} structures | "
          f"{time.time() - T0:.0f}s", flush=True)
    FP, res, mids = {}, {}, sorted(w.mid.unique())
    smi_of = dict(zip(w.ik, w.smiles))
    G = dict(tuple(w.groupby("mid")))
    DUMP = {m: dict(target=float(t), zlog=np.asarray(z, np.float32), pc=None) for m, (t, z) in L.items()}
    for m in DUMP:                               # molecules without any PubChem structure in the window
        if m not in G:
            DUMP[m]["pc"] = []
    with Pool(a.workers, initializer=_init, initargs=(a.fp_bits,)) as pool:
        for i in range(0, len(mids), 20):
            if time.time() - T0 > a.budget:
                print(f"E6 budget reached after {i} molecules", flush=True)
                break
            batch = mids[i:i + 20]
            new = sorted({k for m in batch for k in G[m].ik.values} - set(FP))
            FP.update(zip(new, pool.map(_fp, [smi_of[k] for k in new], chunksize=64)))
            top = {}
            for m in batch:
                ks = [k for k in G[m].ik.values if FP[k] is not None]
                if not ks:
                    continue
                t, z = L[m]
                F = np.unpackbits(np.stack([FP[k] for k in ks]), axis=1)[:, :len(z)].astype(np.float32)
                sc = F @ z
                o = np.argsort(-sc, kind="mergesort")[:60]
                top[m] = [(ks[j], float(sc[j])) for j in o]
                if a.dump_ctx:
                    od = np.argsort(-sc, kind="mergesort")[:a.dump_n]
                    DUMP[m]["pc"] = [(ks[j], smi_of[ks[j]], float(sc[j])) for j in od]
            for m in batch:
                if a.dump_ctx and DUMP.get(m) is not None and DUMP[m]["pc"] is None:
                    DUMP[m]["pc"] = []                # window rows exist but none could be fingerprinted
            need = sorted({smi_of[k] for v in top.values() for k, _ in v})
            C = dict(zip(need, pool.map(_key, need, chunksize=16)))
            for m, v in top.items():
                rec = {"smiles": [], "keys": [], "scores": []}
                for k, s in v:
                    key = C.get(smi_of[k]) or k
                    if key in rec["keys"]:
                        continue
                    rec["smiles"].append(smi_of[k]); rec["keys"].append(key); rec["scores"].append(s)
                    if len(rec["keys"]) >= 40:
                        break
                res[m] = rec
            print(f"E6 {min(i + 20, len(mids))}/{len(mids)} molecules | {len(FP):,} structures | "
                  f"{time.time() - T0:.0f}s", flush=True)
    res["_stats"] = dict(n_test=int(te.molecule_id.nunique()), n_logits=len(L), n_lists=len(res),
                         n_pairs=int(len(w)), sec=round(time.time() - T0, 1))
    json.dump(res, open(a.out, "w"))
    if a.dump_ctx:
        import pickle
        meta = dict(ppm=a.ppm, dump_n=a.dump_n, nets=sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.nets, "*.pt"))),
                    n_mols=len(DUMP), n_with_window=sum(v["pc"] is not None for v in DUMP.values()))
        pickle.dump(dict(mols=DUMP, meta=meta), open(a.dump_ctx + ".tmp", "wb"), protocol=4)
        os.replace(a.dump_ctx + ".tmp", a.dump_ctx)
        print("E6 ctx dump", meta, flush=True)
    print("E6 channel done", res["_stats"], flush=True)


if __name__ == "__main__":
    main()

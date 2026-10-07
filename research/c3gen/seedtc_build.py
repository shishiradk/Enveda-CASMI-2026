"""Seed-Tc-aware re-ranker, stage 1: per-candidate feature tables for the C3NP bench.

    python -u research/c3gen/seedtc_build.py [--modes default strip_edit1] [--workers 2]

Reuses the fingerprint recipe and zlog scoring of biotransform._zfull_many (the scorer the tournament combos used)
and c3np_strip.strip_keys("edit1") to reproduce the seed sets the --strip edit1 generators saw.

Writes, per mode, results/c3gen/seedtc/feat_<mode>.parquet with one row per candidate of the shipped rr_bt union
(capped 200, exactly the combo_rr_bt{sfx}.json order -- asserted per molecule):

  mid, smiles, ik14, label, rank_rr (1-based), rk_a/rk_b (rank in mmp_edit / biotransform, 0 if absent),
  gen_a/gen_b (which generator(s) produced it), z (full-6930-bit fp @ zlog), z_z (within-molecule z-score),
  seed_tc (max Tanimoto r2/2048 to the closest analog/window seed), ha_delta (heavy atoms - closest seed),
  plus oracle-side nb1, nb2, tc_band, subset, ins (stratification / reporting only, never features).

No truth-derived candidate feature: label uses the truth (training target); nb1/nb2/tc_band are oracle strata used
only for stratified CV and reporting.
"""
import argparse, json, os, pickle, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = "D:/Enveda-CASMI-2026/"
sys.path.insert(0, ROOT + "research/c3gen")
sys.path.insert(0, ROOT + "research/scripts")
import biotransform as B          # noqa: E402   the shipped scorer (fp recipe + zlog)
import c3np_eval as EV            # noqa: E402   key_many reuses the evaluator's canonical-key caches
import c3np_strip as ST          # noqa: E402   strip_keys for --strip edit1 seed sets

OUT = ROOT + "results/c3gen/seedtc/"
_MODE_SFX = {"default": "", "strip_edit1": "_strip_edit1"}
_G = {}


def _rd():
    if not _G:
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
        _G["Chem"] = B._rd()
        _G["mg"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        _G["MD"] = rdMolDescriptors
    return _G


def _fp8(m):
    """Full 6930-bit uint8 fingerprint of the zlog recipe (byte-identical to biotransform._zfull_many's input)."""
    g = B._G
    bits = g["bits"]
    full = np.concatenate([g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                           g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                           g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                           np.array(g["MACCS"].GenMACCSKeys(m), dtype=np.uint8)])[bits]
    return np.ascontiguousarray(full, np.uint8)


def _morgan_packed(m):
    g = _rd()
    a = g["mg"].GetFingerprintAsNumPy(m)
    return np.packbits(a.astype(np.uint8)).copy()   # 256 bytes


def _job(a):
    """RDKit-derived blocks for one molecule; never touches the truth."""
    mid, union_smiles, seed_smiles = a
    g = _rd()
    Chem = g["Chem"]
    fp8, cm, ha_c, bad = [], [], [], []
    for s in union_smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            bad.append(s)
            continue
        fp8.append(_fp8(m))
        cm.append(_morgan_packed(m))
        ha_c.append(int(g["MD"].CalcNumHeavyAtoms(m)))
    sm, ha_s, seed_bad = [], [], []
    for s in seed_smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            seed_bad.append(s)
            continue
        sm.append(_morgan_packed(m))
        ha_s.append(int(g["MD"].CalcNumHeavyAtoms(m)))
    return dict(mid=mid, bad=bad, seed_bad=seed_bad,
                fp8=np.asarray(fp8, np.uint8) if fp8 else np.zeros((0, 6930 // 8), np.uint8),
                cm=np.asarray(cm, np.uint8) if cm else np.zeros((0, 256), np.uint8),
                ha_c=np.asarray(ha_c, np.int32) if ha_c else np.zeros(0, np.int32),
                sm=np.asarray(sm, np.uint8) if sm else np.zeros((0, 256), np.uint8),
                ha_s=np.asarray(ha_s, np.int32) if ha_s else np.zeros(0, np.int32))


def _interleave(LA, LB):
    """Biotransform-first round robin on (smiles, key) pairs, deduped on key (combine.py's interleave)."""
    out, seen = [], set()
    for i in range(max(len(LA), len(LB))):
        for L in (LB, LA):
            if i < len(L) and L[i][1] not in seen:
                seen.add(L[i][1])
                out.append(L[i])
    return out


def union_rr(A, Bt, kfun):
    """Shipped rr_bt union: ([smiles], ik14s, rk_a, rk_b), byte-identical order to combine.py."""
    smi_a, smi_b, rk_a, rk_b = {}, {}, {}, {}
    for i, s in enumerate(A):
        k = kfun(s)
        if k and k not in smi_a:
            smi_a[k] = s
            rk_a[k] = i + 1
    for i, s in enumerate(Bt):
        k = kfun(s)
        if k and k not in smi_b:
            smi_b[k] = s
            rk_b[k] = i + 1
    smi = {}
    for k, s in smi_a.items():
        smi[k] = s
    for k, s in smi_b.items():
        smi.setdefault(k, s)   # first-occurrence (mmp first) copy, like combine
    LA = [(smi[k], k) for k in sorted(rk_a, key=rk_a.__getitem__)]
    LB = [(smi[k], k) for k in sorted(rk_b, key=rk_b.__getitem__)]
    rr = _interleave(LA, LB)[:200]
    keys = [k for _, k in rr]
    return [smi[k] for k in keys], keys, rk_a, rk_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["default", "strip_edit1"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.workers = max(1, min(a.workers, 2))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    C = pickle.load(open(ROOT + "results/c3np/context.pkl", "rb"))["mols"]
    M = pd.read_parquet(ROOT + "results/c3np/molecules.parquet")
    T = pd.read_parquet(ROOT + "results/c3np/strata.parquet")[["mid", "nb1", "nb2", "tc_band"]]
    M = M.merge(T, on="mid", how="left")
    M["ins"] = False
    M.loc[0:59, "ins"] = True          # analyze.py's in-sample slices (rows 0-59, 300-359)
    M.loc[300:359, "ins"] = True
    strip = ST.strip_keys("edit1")
    for mode in a.modes:
        sfx = _MODE_SFX[mode]
        A = json.load(open(ROOT + f"results/c3gen/full/mmp_edit{sfx}_full.json"))
        Bt = json.load(open(ROOT + f"results/c3gen/full/biotransform{sfx}_full.json"))
        ref = json.load(open(ROOT + f"results/c3gen/full/combo_rr_bt{sfx}.json"))
        mids = sorted(set(A) | set(Bt))
        all_smi = [s for m_ in mids for s in (A.get(m_, []) + Bt.get(m_, []))]
        print(mode, "mols", len(mids), "cand smiles", len(all_smi), f"{time.time() - t0:.0f}s", flush=True)
        K = EV.key_many(all_smi, a.workers)                     # (ckey, raw); caches are warm from the evals
        kfun = lambda s: (K.get(s) or (None, None))[1]
        excl = strip if mode == "strip_edit1" else {}
        seeds_of = {}
        for m_ in mids:
            ctx = C[m_]
            ex = excl.get(m_, frozenset())
            seeds_of[m_] = {}
            for x in (ctx.get("analogs") or []):
                if not ({x.get("ik14"), x.get("ckey")} & ex):
                    seeds_of[m_].setdefault(x.get("ik14") or x.get("ckey"), x.get("smiles"))
            for x in (ctx.get("window") or []):
                if not ({x.get("ik"), x.get("ckey")} & ex):
                    seeds_of[m_].setdefault(x.get("ik") or x.get("ckey"), x.get("smiles"))
        jobs = [(m_, list(dict.fromkeys(A.get(m_, []) + Bt.get(m_, []))), list(seeds_of[m_].values()))
                for m_ in mids]
        rows = []
        meta_of = {x.mid: x for x in M.itertuples()}
        correct_of = {x.mid: set(str(x.correct).split(";")) for x in M.itertuples()}
        with Pool(a.workers, initializer=_rd) as mp:
            for i, r in enumerate(mp.imap(_job, jobs, chunksize=2)):
                mid = r["mid"]
                mr = meta_of[mid]
                if r["bad"]:
                    print("WARN unparsable candidates", len(r["bad"]), mid, flush=True)
                if r["seed_bad"]:
                    print("WARN unparsable seeds", len(r["seed_bad"]), mid, flush=True)
                sm_is, ks, rk_a, rk_b = union_rr(A.get(mid, []), Bt.get(mid, []), kfun)
                shipped = ref.get(mid)
                if shipped is not None:
                    refk = [kfun(s) for s in shipped]
                    if ks != refk[:len(ks)]:
                        raise SystemExit(f"PARITY FAIL {mid}")
                uniq = list(dict.fromkeys(A.get(mid, []) + Bt.get(mid, [])))
                pos = {s: i for i, s in enumerate(uniq)}
                idx = np.asarray([pos[s] for s in sm_is], np.int64)
                r["fp8"] = r["fp8"][idx]; r["cm"] = r["cm"][idx]; r["ha_c"] = r["ha_c"][idx]
                zlog = np.asarray(C[mid]["zlog"], np.float32)
                z = (r["fp8"].astype(np.float32) @ zlog).astype(np.float64)
                zz = (z - z.mean()) / (z.std() + 1e-9) if len(z) else z
                cf, sf = r["cm"].view(np.uint64), r["sm"].view(np.uint64)
                pc = np.bitwise_count(cf).sum(1).astype(np.int32)
                ps = np.bitwise_count(sf).sum(1).astype(np.int32)
                if len(sf):
                    inter = np.bitwise_count(cf[:, None, :] & sf[None, :, :]).sum(2).astype(np.int32)
                    tc = inter / np.maximum(pc[:, None] + ps[None, :] - inter, 1)
                    bj = tc.argmax(1)
                    seed_tc = tc[np.arange(len(tc)), bj]
                    ha_delta = (r["ha_c"] - r["ha_s"][bj]).astype(np.int32)
                else:
                    seed_tc = np.zeros(len(z)); ha_delta = np.zeros(len(z), np.int32)
                ok = correct_of[mid]
                for j, s in enumerate(sm_is):
                    ck, raw = K.get(s) or (None, None)
                    rows.append(dict(mid=mid, smiles=s, ik14=ks[j], label=bool(ck in ok or raw in ok),
                                     rank_rr=j + 1, rk_a=rk_a.get(ks[j], 0), rk_b=rk_b.get(ks[j], 0),
                                     gen_a=bool(rk_a.get(ks[j])), gen_b=bool(rk_b.get(ks[j])),
                                     z=float(z[j]), z_z=float(zz[j]), seed_tc=float(seed_tc[j]),
                                     ha_delta=int(ha_delta[j]), nb1=bool(mr.nb1), nb2=bool(mr.nb2),
                                     tc_band=mr.tc_band, subset=mr.subset, ins=bool(mr.ins)))
                if i % 50 == 0:
                    print(mode, i, mid, len(ks), f"{time.time() - t0:.0f}s", flush=True)
        D = pd.DataFrame(rows, columns=["mid", "smiles", "ik14", "label", "rank_rr", "rk_a", "rk_b",
                                        "gen_a", "gen_b", "z", "z_z", "seed_tc", "ha_delta",
                                        "nb1", "nb2", "tc_band", "subset", "ins"])
        D.to_parquet(OUT + f"feat_{mode}.parquet", index=False)
        print(mode, "rows", len(D), "pos-labels", int(D.label.sum()), f"{time.time() - t0:.0f}s", flush=True)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
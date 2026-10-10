"""Table builders for the v4r engine (REBUILD_SPEC 3.1-3.6).  Writes to results/v4n/tables/.

    python research/v4n_rebuild/engine/build_tables.py featurize [--workers 7] [--coco-max-mass M]  (resumable)
    python research/v4n_rebuild/engine/build_tables.py assemble
        P2 train_structs.parquet, P4 pool_meta.parquet, P3 train_fp_raw.npy, P5 pool_fp_raw.npy,
        P6 pool_frag_off.npy / pool_frag_mass.npy        (all fingerprints/fragments from the tautomer-canonical form)
    python research/v4n_rebuild/engine/build_tables.py project --bits models/cft_ho2_akriti/cft_ho2.pt
        model-layout copies fp_bits.npy, pool_fp.npy, train_fp_sel.npy (bits read from a CFT checkpoint or a .npy)
    python research/v4n_rebuild/engine/build_tables.py lib
        L1 spectrum cache spec_meta.parquet, spec_off.npy, spec_mz.npy, spec_it.npy  (needs train_structs.parquet)

Inputs: train.parquet (structures + spectra) and the COCONUT SMILES of our existing 712,199-row pool
(results/train_pkg/data_full/pool_smiles.txt rows with pool_src == 0, i.e. 436,389 cleaned COCONUT structures).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "v4n" / "tables"
TRAIN = ROOT / "train.parquet"
OLD_POOL = ROOT / "results" / "train_pkg" / "data_full"

import numpy as np

LIBS = ["enveda-180", "pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial", "drug_plus",
        "enveda-np-examples", "masaryk"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ structures
CHUNKS = "_chunks"
CH = 2000


def _work_chunk(args):
    """Worker: featurize one chunk of SMILES.  Writes fp_/fo_/fm_ (fingerprints, fragment CSR) and meta_ files.
    If the fp/fragment files already exist (an interrupted earlier run), only the metadata is recomputed."""
    ci, smiles, is_train, tmp = args
    import pickle
    from engine import chem, fingerprint, fragments
    fpr = fingerprint.raw_fingerprinter()
    n = len(smiles)
    have = all(os.path.exists(f"{tmp}/{p}_{ci:05d}.npy") for p in ("fp", "fo", "fm"))
    fo_old = np.load(f"{tmp}/fo_{ci:05d}.npy") if have else None
    fp = np.zeros((n, fingerprint.RAW_BYTES), np.uint8)
    frs, meta = [], []
    for i, s in enumerate(smiles):
        charged = False
        std = chem.standardize(s) if isinstance(s, str) else None
        if std is None and is_train[i] and isinstance(s, str):
            std = chem.standardize(s, keep_charged=True) or s
            charged = True
        c = chem.canonical(std) if std is not None else None
        rec = None
        if c is not None and c[2] is not None:
            try:
                m, tm, key = c
                formula, mass, nh = chem.mol_props(m)
                rec = (key, chem.rd()["Chem"].MolToSmiles(tm), formula, mass, nh)
                if not have:
                    fp[i] = np.packbits(fpr.raw(tm))
                    frs.append(fragments.fragments_of_mol(tm))
            except Exception:
                rec = None
        if rec is None:
            meta.append((False, std, charged, None, None, None, np.nan, -1, 0))
            if not have:
                fp[i] = 0
                if len(frs) == i + 1:
                    frs[-1] = np.zeros(0, np.float32)
                else:
                    frs.append(np.zeros(0, np.float32))
            continue
        nf = int(fo_old[i + 1] - fo_old[i]) if have else len(frs[-1])
        meta.append((True, std, charged) + rec + (nf,))
    if not have:
        np.save(f"{tmp}/fp_{ci:05d}.npy", fp)
        np.save(f"{tmp}/fm_{ci:05d}.npy", np.concatenate(frs) if frs else np.zeros(0, np.float32))
        off = np.zeros(n + 1, np.int64)
        off[1:] = np.cumsum([len(x) for x in frs])
        np.save(f"{tmp}/fo_{ci:05d}.npy", off)
    pickle.dump(meta, open(f"{tmp}/meta_{ci:05d}.pkl.part", "wb"))
    os.replace(f"{tmp}/meta_{ci:05d}.pkl.part", f"{tmp}/meta_{ci:05d}.pkl")
    return ci


def _inputs():
    """(train table, COCONUT SMILES ascending by mass, their masses); deterministic, so chunk ids are stable."""
    import pickle
    cache = OUT / CHUNKS / "inputs.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    import duckdb
    con = duckdb.connect()
    con.execute("set threads=4; set memory_limit='2GB'")
    T = con.sql(f"""
        with c as (select inchikey14 ik, normalized_smiles smi, molecular_formula f, count(*) n,
                          min(file_row_number) r
                   from read_parquet('{TRAIN.as_posix()}', file_row_number=true) group by 1, 2, 3),
             r as (select *, row_number() over (partition by ik order by n desc, r asc) rk,
                          sum(n) over (partition by ik) n_spec, count(*) over (partition by ik) n_smiles from c)
        select ik, smi, f, n_spec::bigint n_spec, n_smiles::int n_smiles from r where rk = 1 order by ik""").df()
    psrc = np.load(OLD_POOL / "pool_src.npy")
    pmass = np.load(OLD_POOL / "pool_mass.npy")
    with open(OLD_POOL / "pool_smiles.txt", encoding="utf-8") as f:
        psmi = f.read().split("\n")[:len(psrc)]
    sel = np.flatnonzero(psrc == 0)                 # the old pool is mass-sorted -> COCONUT ascending by mass
    out = (T, [psmi[i] for i in sel], pmass[sel])
    (OUT / CHUNKS).mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def featurize(a):
    """Resumable: every chunk without a meta_ file (train chunks first, then COCONUT by mass); optionally COCONUT only
    up to --coco-max-mass (chunk start mass)."""
    from multiprocessing import Pool
    T0 = time.time()
    tmp = OUT / CHUNKS
    T, coco, coco_mass = _inputs()
    U = list(T.smi) + coco
    is_tr = np.r_[np.ones(len(T), bool), np.zeros(len(coco), bool)]
    umass = np.r_[np.zeros(len(T)), coco_mass]
    tasks = []
    for ci, s in enumerate(range(0, len(U), CH)):
        if (tmp / f"meta_{ci:05d}.pkl").exists():
            continue
        if a.coco_max_mass and not is_tr[s] and umass[s] > a.coco_max_mass:
            continue
        tasks.append((ci, U[s:s + CH], is_tr[s:s + CH], str(tmp)))
    log(f"featurize: {len(tasks)} chunks to do ({len(U):,} structures, {-(-len(U) // CH)} chunks total)")
    with Pool(a.workers) as mp:
        for k, ci in enumerate(mp.imap_unordered(_work_chunk, tasks)):
            if (k + 1) % 10 == 0 or k + 1 == len(tasks):
                log(f"  chunk {k + 1}/{len(tasks)} (id {ci})  {time.time() - T0:.0f}s")


def assemble(a):
    """Tables from all finished chunks: every train chunk is required; COCONUT is used up to the first missing chunk
    (a contiguous mass prefix) and the covered mass is recorded in build_structs.json."""
    import pickle
    import pandas as pd
    from engine import chem, fingerprint
    T0 = time.time()
    tmp = OUT / CHUNKS
    T, coco, coco_mass = _inputs()
    nT, nU = len(T), len(T) + len(coco)
    nch = -(-nU // CH)
    done = [(tmp / f"meta_{ci:05d}.pkl").exists() for ci in range(nch)]
    n_tr_ch = -(-nT // CH)
    assert all(done[:n_tr_ch]), "train chunks missing: run featurize first"
    last = n_tr_ch - 1
    while last + 1 < nch and done[last + 1]:
        last += 1
    use = list(range(last + 1))
    metas = [pickle.load(open(tmp / f"meta_{ci:05d}.pkl", "rb")) for ci in use]
    sizes = [min(CH, nU - ci * CH) for ci in use]
    M = pd.DataFrame([m for mm in metas for m in mm],
                     columns=["ok", "smiles", "charged", "key", "smiles_tc", "formula", "mass", "n_heavy", "n_frag"])
    M["chunk"] = np.repeat(use, sizes)
    M["row"] = np.concatenate([np.arange(k) for k in sizes])
    n_coco_used = len(M) - nT
    coco_cover = float(coco_mass[n_coco_used - 1]) if n_coco_used < len(coco) else float("inf")
    log(f"assemble: {len(use)} chunks, COCONUT {n_coco_used:,}/{len(coco):,} (mass <= {coco_cover:.2f})")
    # ---------------- P2
    P2 = M.iloc[:nT].reset_index(drop=True).copy()
    P2.insert(0, "sid", np.arange(nT, dtype=np.int64))
    P2.insert(1, "inchikey14", T.ik.values)
    P2["n_spec"] = T.n_spec.values
    P2["n_smiles"] = T.n_smiles.values
    P2["src_smiles"] = T.smi.values
    bad = ~P2.ok.values.astype(bool)
    if bad.any():          # unparseable: keep the raw SMILES and the formula mass so the library window still works
        P2.loc[bad, "smiles"] = T.smi.values[bad]
        P2.loc[bad, "formula"] = T.f.values[bad]
        P2.loc[bad, "mass"] = [chem.formula_mass(chem.parse_formula(f) or {}) if isinstance(f, str) else np.nan
                               for f in T.f.values[bad]]
    # ---------------- pool: train (neutral) + COCONUT, dedup by score key (train first, most spectra first)
    C = M.iloc[nT:].reset_index(drop=True).copy()
    cand_t = P2[~bad & ~P2.charged.values.astype(bool)].copy()
    cand_t["src"] = 0; cand_t["train_sid"] = cand_t.sid; cand_t["prio"] = -cand_t.n_spec
    cand_c = C[C.ok.values.astype(bool)].copy()
    cand_c["src"] = 1; cand_c["train_sid"] = -1; cand_c["prio"] = np.arange(len(cand_c))
    cols = ["key", "smiles", "smiles_tc", "formula", "mass", "n_heavy", "n_frag", "chunk", "row", "src", "train_sid",
            "prio"]
    Pc = pd.concat([cand_t[cols], cand_c[cols]], ignore_index=True)
    Pc = Pc.sort_values(["key", "src", "prio"], kind="mergesort")
    n_sids_key = P2[~bad].groupby("key").size()
    Pc = Pc.drop_duplicates("key", keep="first")
    Pc = Pc.iloc[np.lexsort((Pc.key.values, Pc.mass.values.astype(np.float64)))].reset_index(drop=True)
    Pc["n_train_sids"] = Pc.key.map(n_sids_key).fillna(0).astype(np.int16)
    k2p = dict(zip(Pc.key.values, range(len(Pc))))
    P2["pid"] = [k2p.get(k, -1) if isinstance(k, str) else -1 for k in P2.key.values]
    log(f"pool: {len(Pc):,} rows ({int((Pc.src == 0).sum()):,} train, {int((Pc.src == 1).sum()):,} COCONUT); "
        f"P2 charged {int(P2.charged.sum())}, failed {int(bad.sum())}")
    # ---------------- fingerprints / fragments in final order
    NB = fingerprint.RAW_BYTES
    pool_fp = np.lib.format.open_memmap(OUT / "pool_fp_raw.npy", "w+", np.uint8, (len(Pc), NB))
    train_fp = np.lib.format.open_memmap(OUT / "train_fp_raw.npy", "w+", np.uint8, (nT, NB))
    foff = np.zeros(len(Pc) + 1, np.int64)
    foff[1:] = np.cumsum(Pc.n_frag.values.astype(np.int64))
    fmass = np.lib.format.open_memmap(OUT / "pool_frag_mass.npy", "w+", np.float32, (int(foff[-1]),))
    pchunk = Pc.chunk.values; prow = Pc.row.values
    tchunk = P2.chunk.values; trow = P2.row.values
    for ci in use:
        fp = np.load(tmp / f"fp_{ci:05d}.npy")
        fo = np.load(tmp / f"fo_{ci:05d}.npy"); fm = np.load(tmp / f"fm_{ci:05d}.npy")
        sel = np.flatnonzero(pchunk == ci)
        if len(sel):
            pool_fp[sel] = fp[prow[sel]]
            for j in sel:
                r = prow[j]
                assert fo[r + 1] - fo[r] == foff[j + 1] - foff[j]
                fmass[foff[j]:foff[j + 1]] = fm[fo[r]:fo[r + 1]]
        st = np.flatnonzero(tchunk == ci)
        if len(st):
            train_fp[st] = fp[trow[st]]
    pool_fp.flush(); train_fp.flush(); fmass.flush()
    del pool_fp, train_fp, fmass
    np.save(OUT / "pool_frag_off.npy", foff)
    P2 = P2.drop(columns=["chunk", "row", "ok"]).assign(ok=~bad)
    P2["charged"] = P2.charged.astype(bool)
    P2.to_parquet(OUT / "train_structs.parquet", index=False)
    Pm = Pc.drop(columns=["chunk", "row", "prio"])
    Pm["src"] = Pm.src.astype(np.int8); Pm["train_sid"] = Pm.train_sid.astype(np.int64)
    Pm["n_heavy"] = Pm.n_heavy.astype(np.int32); Pm["mass"] = Pm.mass.astype(np.float64)
    Pm.to_parquet(OUT / "pool_meta.parquet", index=False)
    rep = dict(train_structs=nT, train_failed=int(bad.sum()), train_charged=int(P2.charged.sum()),
               coconut_total=len(coco), coconut_processed=n_coco_used, coconut_mass_covered=coco_cover,
               coconut_failed=int((~C.ok.values.astype(bool)).sum()), pool=len(Pc),
               pool_train=int((Pm.src == 0).sum()), pool_coconut=int((Pm.src == 1).sum()),
               train_keys_multi_sid=int((n_sids_key > 1).sum()), frag_masses=int(foff[-1]),
               seconds=round(time.time() - T0))
    json.dump(rep, open(OUT / "build_structs.json", "w"), indent=1)
    log(json.dumps(rep))


# ============================================================================================ model-layout projection
def load_bits(path):
    if str(path).endswith(".pt"):
        import torch
        return np.asarray(torch.load(path, map_location="cpu", weights_only=False)["fp_bits"], np.int64)
    return np.load(path).astype(np.int64)


def project(a):
    from engine import fingerprint
    bits = load_bits(a.bits)
    np.save(OUT / "fp_bits.npy", bits.astype(np.int32))
    for src, dst in (("pool_fp_raw.npy", "pool_fp.npy"), ("train_fp_raw.npy", "train_fp_sel.npy")):
        R = np.load(OUT / src, mmap_mode="r")
        nb = (len(bits) + 7) // 8
        D = np.lib.format.open_memmap(OUT / dst, "w+", np.uint8, (len(R), nb))
        for s in range(0, len(R), 50000):
            D[s:s + 50000] = np.packbits(fingerprint.select(R[s:s + 50000], bits), axis=1)
        D.flush(); del D
        log(f"{dst}: {len(R):,} x {nb} bytes ({len(bits)} bits)")
    json.dump(dict(bits_from=str(a.bits), nbits=int(len(bits)), blocks=fingerprint.block_counts(bits)),
              open(OUT / "fp_bits.json", "w"), indent=1)


# ============================================================================================ L1 spectrum cache
def _ce_stats(off, vals):
    n = np.diff(off)
    mean = np.full(len(n), np.nan); mn = mean.copy(); mx = mean.copy()
    nz = n > 0
    if nz.any():
        s = np.add.reduceat(vals, off[:-1][nz]) if len(vals) else np.zeros(int(nz.sum()))
        mean[nz] = s / n[nz]
        mn[nz] = np.minimum.reduceat(vals, off[:-1][nz]); mx[nz] = np.maximum.reduceat(vals, off[:-1][nz])
    return mean, mn, mx, n


def build_lib(a):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from engine import chem, model, spectra
    T0 = time.time()
    import duckdb
    # sid = rank of inchikey14 in sorted order (the P2 order; checked against train_structs.parquet in validate V2)
    iks = [r[0] for r in duckdb.sql(f"select distinct inchikey14 from '{TRAIN.as_posix()}' order by 1").fetchall()]
    k2s = {k: i for i, k in enumerate(iks)}
    tmp = OUT / "_tmp_lib"
    tmp.mkdir(exist_ok=True)

    pf = pq.ParquetFile(TRAIN)
    cols = ["inchikey14", "ingest_lib", "instrument_type", "adduct", "ionization_mode", "precursor_mz",
            "precursor_error_ppm", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    metas = []
    npk = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        L = t.column("ms2_mzs").combine_chunks(); I = t.column("ms2_normalized_intensities").combine_chunks()
        off = L.offsets.to_numpy().astype(np.int64); off = off - off[0]
        mz = L.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        it = I.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        assert np.array_equal(I.offsets.to_numpy() - I.offsets.to_numpy()[0], off)
        prec = t.column("precursor_mz").to_numpy().astype(np.float64)
        oo, om, oi = spectra.clean_all(off, mz, it, prec, 0.001, 512, 2.0)
        np.save(tmp / f"o_{rg:03d}.npy", np.diff(oo)); np.save(tmp / f"m_{rg:03d}.npy", om)
        np.save(tmp / f"i_{rg:03d}.npy", oi)
        ce = t.column("collision_energy_ev").combine_chunks()
        ce_off = ce.offsets.to_numpy().astype(np.int64); ce_off = ce_off - ce_off[0]
        ce_val = ce.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        cmean, cmin, cmax, cn = _ce_stats(ce_off, ce_val)
        d = t.drop_columns(["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]).to_pandas()
        d["n_raw"] = np.diff(off).astype(np.int32)
        d["n_clean"] = np.diff(oo).astype(np.int32)
        d["ce_mean"], d["ce_min"], d["ce_max"], d["ce_n"] = cmean, cmin, cmax, cn.astype(np.int16)
        d["sid"] = d.inchikey14.map(k2s).fillna(-1).astype(np.int64)
        d["lib"] = d.ingest_lib.map({l: i for i, l in enumerate(LIBS)}).fillna(-1).astype(np.int8)
        d["instr"] = d.instrument_type.map(lambda s: model.INSTRUMENTS.index(model.instrument_family(s))).astype(np.int8)
        d["adduct_ix"] = d.adduct.map(chem.adduct_index).astype(np.int16)
        d["mode"] = np.where(d.ionization_mode == "positive", 1, -1).astype(np.int8)
        d["nm"] = chem.neutral_mass(d.precursor_mz.values, d.adduct.values)
        d = d.drop(columns=["ionization_mode", "ingest_lib"])
        d.to_parquet(tmp / f"meta_{rg:03d}.parquet", index=False)
        metas.append(len(d))
        npk += len(om)
        log(f"rg {rg}: {len(d):,} spectra, {len(om):,} peaks (total {npk:,})  {time.time() - T0:.0f}s")
        del t, L, I, mz, it
    meta = pd.concat([pd.read_parquet(tmp / f"meta_{rg:03d}.parquet") for rg in range(len(metas))],
                     ignore_index=True)
    off = np.zeros(len(meta) + 1, np.int64)
    off[1:] = np.cumsum(meta.n_clean.values)
    MZ = np.lib.format.open_memmap(OUT / "spec_mz.npy", "w+", np.float32, (int(off[-1]),))
    IT = np.lib.format.open_memmap(OUT / "spec_it.npy", "w+", np.float32, (int(off[-1]),))
    p = 0
    for rg in range(pf.metadata.num_row_groups):
        m = np.load(tmp / f"m_{rg:03d}.npy"); i = np.load(tmp / f"i_{rg:03d}.npy")
        MZ[p:p + len(m)] = m; IT[p:p + len(m)] = i
        p += len(m)
    MZ.flush(); IT.flush(); del MZ, IT
    np.save(OUT / "spec_off.npy", off)
    meta["spec_ix"] = np.arange(len(meta), dtype=np.int64)
    meta.to_parquet(OUT / "spec_meta.parquet", index=False)
    rep = dict(spectra=len(meta), peaks=int(off[-1]), sid_missing=int((meta.sid < 0).sum()),
               lib_missing=int((meta.lib < 0).sum()), nm_nan=int(meta.nm.isna().sum()),
               empty_after_clean=int((meta.n_clean == 0).sum()), seconds=round(time.time() - T0))
    json.dump(rep, open(OUT / "build_lib.json", "w"), indent=1)
    log(json.dumps(rep))
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["featurize", "assemble", "project", "lib"])
    ap.add_argument("--coco-max-mass", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bits", default=str(ROOT / "models" / "cft_ho2_akriti" / "cft_ho2.pt"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    {"featurize": featurize, "assemble": assemble, "project": project, "lib": build_lib}[a.step](a)


if __name__ == "__main__":
    main()

"""Extra data for the CFT model, built on top of the prepared warm-up data (results/train_pkg/data).

    python research/train_cft/prep_cft.py            (CPU, 3 workers, about 1 hour)

Adds (results/train_cft/data):
  cfp_bits.npy                         selected bits of the cfp1 fingerprint (cft_fp.py)
  cpool_*.npy, cpool_smiles.txt        the CFT decoy pool = existing pool (COCONUT + training structures)
                                       + PubChem decoys sampled from the +-10 ppm windows of the training molecules
                                       + the PubChem representation of training / validation structures ("alt")
                                       fingerprints (cfp1, bit-packed), mass (sorted), key id, formula id,
                                       element counts, source
  mol_cpool.npy, mol_alt.npy           molecule -> pool row of its training-style / PubChem-style structure
  pcv_*.npy, pcv_smiles.txt            "PubChem validation": for 500 validation molecules up to 512 random
                                       PubChem structures of the same mass (cfp1 + the warm-up 6,930-bit fingerprint)
  cft_meta.json                        counts and formats
Nothing existing is rebuilt or modified. PubChem / held-key parquet files are read with DuckDB only.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import cft_fp

T0 = time.time()
_g = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _init(bits, base_bits):
    _g["fp"] = cft_fp.Fingerprinter(bits)
    _g["base_bits"] = base_bits
    if base_bits is not None:
        from rdkit.Chem import rdFingerprintGenerator as G
        _g["b"] = [G.GetMorganGenerator(radius=2, fpSize=4096), G.GetMorganGenerator(radius=3, fpSize=4096),
                   G.GetRDKitFPGenerator(fpSize=2048, maxPath=6)]


def _raw_chunk(smiles):
    """Bit counts of the raw fingerprint over a chunk (for the bit selection)."""
    f = _g["fp"]; c = np.zeros(cft_fp.RAW_BITS, np.int64); n = 0
    for s in smiles:
        m = f.mol(s)
        if m is None: continue
        c += f.raw(m); n += 1
    return c, n


def _desc_chunk(arg):
    smiles, want_base = arg
    f = _g["fp"]; nb = (f.nbits + 7) // 8; n = len(smiles)
    fp = np.zeros((n, nb), np.uint8); mass = np.full(n, np.nan); el = np.zeros((n, len(cft_fp.ELEMENTS)), np.uint8)
    form = [""] * n; ok = np.zeros(n, bool)
    fb = np.zeros((n, (len(_g["base_bits"]) + 7) // 8), np.uint8) if want_base else None
    for i, s in enumerate(smiles):
        r = f.describe(s)
        if r is None: continue
        fp[i], mass[i], form[i], el[i] = r; ok[i] = True
        if want_base:                    # warm-up fingerprint: ECFP4 | ECFP6 | RDKit path | MACCS -> fp_bits
            from rdkit.Chem import MACCSkeys
            m = f.mol(s)
            v = np.concatenate([g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in _g["b"]]
                               + [np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["base_bits"]]
            fb[i] = np.packbits(v)
    return fp, mass, form, el, ok, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/train_pkg/data"))
    ap.add_argument("--out", default=str(ROOT / "results/train_cft/data"))
    ap.add_argument("--pubchem", default=str(ROOT / "external/pubchem/pubchem_rows_pop.parquet"))
    ap.add_argument("--held", nargs="*", default=[str(ROOT / "results/kaggle_v1_proxy/held_keys.parquet"),
                                                  str(ROOT / "results/kaggle_v2_proxy/s3_held.parquet")])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per_mol", type=int, default=2, help="PubChem decoys drawn per training molecule")
    ap.add_argument("--max_decoys", type=int, default=400000, help="cap on the PubChem decoys (disk budget)")
    ap.add_argument("--pcv_mols", type=int, default=500); ap.add_argument("--pcv_cands", type=int, default=384)
    ap.add_argument("--ppm", type=float, default=10.0); ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--limit", type=int, default=0, help="(test) use only this many molecules / pool rows")
    a = ap.parse_args()
    import duckdb
    d = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); S = {}
    con = duckdb.connect(); con.execute("set threads=3; set memory_limit='3GB'")
    pc = Path(a.pubchem).as_posix()

    meta0 = json.load(open(d / "meta.json"))
    pool_mass = np.load(d / "pool_mass.npy"); pool_key = np.load(d / "pool_key.npy"); pool_src = np.load(d / "pool_src.npy")
    pool_smi = [l.rstrip("\n") for l in open(d / "pool_smiles.txt", encoding="utf-8")]
    mol_key = np.load(d / "mol_key.npy"); mol_pool = np.load(d / "mol_pool.npy"); mol_split = np.load(d / "mol_split.npy")
    NP = len(pool_mass); assert len(pool_smi) == NP == len(pool_key)
    held = set()
    for p in a.held:
        held |= {r[0] for r in con.sql(f"select distinct ik from '{Path(p).as_posix()}'").fetchall()}
    held |= set(meta0["excluded"].get("tautomer_keys", []))
    S["held_keys_excluded"] = len(held)
    pool_keys_s = np.char.decode(pool_key, "ascii"); pool_set = set(pool_keys_s.tolist())
    assert not (pool_set & held)

    # ---------------------------------------------------------------- PubChem rows to fetch
    pm = con.execute(f"select mass from read_parquet('{pc}')").fetchnumpy()["mass"]
    assert (np.diff(pm) >= 0).all(), "PubChem table must be sorted by mass"
    log(f"PubChem masses {len(pm):,} ({pm[0]:.3f} .. {pm[-1]:.3f})")
    tr = np.where((mol_split == 0) & (mol_pool >= 0))[0]; va = np.where((mol_split == 1) & (mol_pool >= 0))[0]
    if a.limit: tr = tr[:a.limit]
    x = pool_mass[mol_pool[tr]]; tol = x * a.ppm / 1e6
    lo = np.searchsorted(pm, x - tol, "left"); hi = np.searchsorted(pm, x + tol, "right"); n = hi - lo
    r = lo[:, None] + (rng.random((len(tr), a.per_mol)) * n[:, None]).astype(np.int64)
    sup_rows = np.unique(r[n > 0].ravel())
    S["train_mols"] = int(len(tr)); S["train_mols_with_pubchem_window"] = int((n > 0).sum())
    S["pubchem_window_size_train_pct_10_50_90"] = [float(v) for v in np.percentile(n, [10, 50, 90])]
    # PubChem validation molecules: random validation molecules whose mass is inside the PubChem table range
    xv = pool_mass[mol_pool[va]]
    okv = va[(xv > pm[0] + 1) & (xv < pm[-1] - 1)]
    pcv_mol = np.sort(rng.choice(okv, size=min(a.pcv_mols, len(okv)), replace=False))
    xv = pool_mass[mol_pool[pcv_mol]]; tol = xv * a.ppm / 1e6
    vlo = np.searchsorted(pm, xv - tol, "left"); vhi = np.searchsorted(pm, xv + tol, "right")
    extra = 64                                                    # spare rows: duplicates / the truth are dropped later
    pcv_rows = [np.sort(rng.choice(np.arange(l, h), size=min(h - l, a.pcv_cands + extra), replace=False))
                for l, h in zip(vlo, vhi)]
    del pm
    allrows = np.unique(np.concatenate([sup_rows] + pcv_rows))
    log(f"fetching {len(allrows):,} PubChem rows ({len(sup_rows):,} decoy rows, {sum(map(len, pcv_rows)):,} validation rows)")
    con.execute("create table ids(r BIGINT)"); con.register("ids_np", {"r": allrows})
    con.execute("insert into ids select r from ids_np")
    got = con.execute(f"""select file_row_number r, ik, smiles from read_parquet('{pc}', file_row_number=true)
                          where file_row_number in (select r from ids) order by r""").fetchnumpy()
    assert len(got["r"]) == len(allrows)
    row_ik = dict(zip(got["r"].tolist(), zip(got["ik"].tolist(), got["smiles"].tolist()))); del got
    log(f"fetched ({time.time() - T0:.0f}s); looking up the PubChem form of the training structures")
    con.execute("create table mk(ik VARCHAR)"); con.register("mk_np", {"ik": np.char.decode(mol_key, "ascii")[mol_pool >= 0]})
    con.execute("insert into mk select ik from mk_np")
    alt = con.execute(f"""select ik, arg_max(smiles, n_sid) smiles from read_parquet('{pc}')
                          where ik in (select ik from mk) group by ik""").fetchall()
    alt = dict(alt); S["molecules_found_in_pubchem"] = len(alt)
    log(f"{len(alt):,} of {int((mol_pool >= 0).sum()):,} molecules have their InChIKey14 in PubChem ({time.time() - T0:.0f}s)")

    # ---------------------------------------------------------------- lists of structures
    seen = set(); sup = []                                         # PubChem decoys: new keys only
    for r_ in sup_rows.tolist():
        ik, smi = row_ik[r_]
        if ik in pool_set or ik in held or ik in seen or not smi: continue
        seen.add(ik); sup.append((ik, smi))
    S["pubchem_decoys_drawn"] = int(len(sup_rows)); S["pubchem_decoys_new_keys"] = len(sup)
    if len(sup) > a.max_decoys:
        sup = [sup[i] for i in np.sort(rng.choice(len(sup), size=a.max_decoys, replace=False))]
    S["pubchem_decoys_kept"] = len(sup)
    mk = np.char.decode(mol_key, "ascii")
    alt_mols = [i for i in np.where(mol_pool >= 0)[0].tolist() if mk[i] in alt]
    if a.limit: alt_mols = sorted(set(alt_mols[:a.limit]) | (set(pcv_mol.tolist()) & set(alt_mols)))
    pcv = []                                                       # (mol, [(ik, smi)...], n_full)
    for i, rows, l, h in zip(pcv_mol.tolist(), pcv_rows, vlo, vhi):
        s2 = set(); c = []
        for r_ in rows.tolist():
            ik, smi = row_ik[r_]
            if ik == mk[i] or ik in held or ik in s2 or not smi: continue
            s2.add(ik); c.append((ik, smi))
            if len(c) >= a.pcv_cands: break
        pcv.append((i, c, int(h - l)))
    del row_ik
    if a.limit: pool_ix = np.arange(min(NP, a.limit))
    else: pool_ix = np.arange(NP)

    # ---------------------------------------------------------------- bit selection
    base_bits = np.load(d / "fp_bits.npy")
    samp = [pool_smi[i] for i in rng.choice(NP, size=min(60000, NP), replace=False)] + \
           [sup[i][1] for i in rng.choice(len(sup), size=min(40000, len(sup)), replace=False)]
    ch = [samp[i:i + 1000] for i in range(0, len(samp), 1000)]
    with Pool(a.workers, initializer=_init, initargs=(None, None)) as mp:
        cnt = np.zeros(cft_fp.RAW_BITS, np.int64); nn = 0
        for c, k in mp.imap_unordered(_raw_chunk, ch): cnt += c; nn += k
    fr = cnt / nn
    bits = np.where((fr >= 0.005) & (fr <= 0.995))[0].astype(np.int64); nbits = len(bits)
    edges = np.cumsum([0] + [b for _, b in cft_fp.BLOCKS])
    S["bits"] = dict(raw=cft_fp.RAW_BITS, selected=int(nbits), sample=int(nn), rule="frequency in [0.005, 0.995]",
                     per_block={name: int(((bits >= edges[j]) & (bits < edges[j + 1])).sum())
                                for j, (name, _) in enumerate(cft_fp.BLOCKS)})
    log(f"bit selection: {nbits} of {cft_fp.RAW_BITS} ({S['bits']['per_block']}) ({time.time() - T0:.0f}s)")
    np.save(out / "cfp_bits.npy", bits)

    # ---------------------------------------------------------------- fingerprints
    def run(smiles, want_base, name):
        ch = [(smiles[i:i + 2000], want_base) for i in range(0, len(smiles), 2000)]
        res = []
        with Pool(a.workers, initializer=_init, initargs=(bits, base_bits)) as mp:
            for j, r_ in enumerate(mp.imap(_desc_chunk, ch)):
                res.append(r_)
                if j % 50 == 0: log(f"  {name}: chunk {j + 1}/{len(ch)} ({time.time() - T0:.0f}s)")
        fp = np.concatenate([r_[0] for r_ in res]); mass = np.concatenate([r_[1] for r_ in res])
        form = [f for r_ in res for f in r_[2]]; el = np.concatenate([r_[3] for r_ in res])
        ok = np.concatenate([r_[4] for r_ in res])
        fb = np.concatenate([r_[5] for r_ in res]) if want_base else None
        return fp, mass, form, el, ok, fb

    A = list(run([pool_smi[i] for i in pool_ix], False, "pool"))
    assert A[4].all(), f"{int((~A[4]).sum())} pool structures failed in RDKit"
    dm = np.abs(A[1] - pool_mass[pool_ix]); S["pool_mass_recomputed_max_abs_diff"] = float(dm.max())
    B = list(run([s for _, s in sup], False, "pubchem decoys"))
    C = run([alt[mk[i]] for i in alt_mols], False, "pubchem form of training structures")
    pcv_alt = [i for i in pcv_mol.tolist() if mk[i] in alt]
    Cb = run([alt[mk[i]] for i in pcv_alt], True, "pubchem form of the PubChem-validation truths")
    flat = [s for _, c, _ in pcv for _, s in c]
    Dv = run(flat, True, "pubchem validation candidates")

    # alt entries: kept in the pool only when the PubChem form gives a different fingerprint
    p2 = {int(p): j for j, p in enumerate(pool_ix)}
    alt_keep = []; same = 0; alt_fail = 0
    for j, i in enumerate(alt_mols):
        if not C[4][j]: alt_fail += 1; continue
        q = p2.get(int(mol_pool[i]))
        if q is None: continue
        if (C[0][j] == A[0][q]).all(): same += 1; continue
        alt_keep.append((j, i, q))
    S["alt"] = dict(found=len(alt_mols), rdkit_failed=alt_fail, identical_fingerprint=same, added=len(alt_keep))
    log(f"PubChem form: {S['alt']}")
    okB = B[4] & np.isfinite(B[1]); S["pubchem_decoys_rdkit_failed"] = int((~okB).sum())
    okD = Dv[4]

    forms = sorted(set(A[2]) | {f for f, o in zip(B[2], okB) if o} | {C[2][j] for j, _, _ in alt_keep}
                   | {f for f, o in zip(Dv[2], okD) if o} | {""})
    fid = {f: i for i, f in enumerate(forms)}
    aj = np.array([j for j, _, _ in alt_keep], np.int64); aq = np.array([q for _, _, q in alt_keep], np.int64)
    fp = np.concatenate([A[0], B[0][okB], C[0][aj]]); A[0] = B[0] = None
    mass = np.concatenate([pool_mass[pool_ix], B[1][okB], pool_mass[pool_ix][aq]])   # alt: mass of its pool twin
    key = np.concatenate([pool_key[pool_ix], np.array([k for (k, _), o in zip(sup, okB) if o], "S14"),
                          pool_key[pool_ix][aq]])
    src = np.concatenate([pool_src[pool_ix], np.full(int(okB.sum()), 2, np.uint8), np.full(len(aj), 3, np.uint8)])
    form = np.concatenate([np.array([fid[f] for f in A[2]], np.int32),
                           np.array([fid[f] for f, o in zip(B[2], okB) if o], np.int32),
                           np.array([fid[C[2][j]] for j in aj], np.int32)])
    el = np.concatenate([A[3], B[3][okB], C[3][aj]])
    orig = np.concatenate([pool_ix.astype(np.int32), np.full(int(okB.sum()) + len(aj), -1, np.int32)])
    smi = [pool_smi[i] for i in pool_ix] + [s for (_, s), o in zip(sup, okB) if o] + [alt[mk[i]] for _, i, _ in alt_keep]
    o = np.argsort(mass, kind="mergesort"); inv = np.empty(len(o), np.int64); inv[o] = np.arange(len(o))
    _, kid = np.unique(key[o], return_inverse=True)
    mmf = np.lib.format.open_memmap(out / "cpool_fp.npy", mode="w+", dtype=np.uint8, shape=fp.shape)
    for s_ in range(0, len(o), 100000): mmf[s_:s_ + 100000] = fp[o[s_:s_ + 100000]]
    mmf.flush(); del mmf, fp
    np.save(out / "cpool_mass.npy", mass[o]); np.save(out / "cpool_key.npy", key[o])
    np.save(out / "cpool_kid.npy", kid.astype(np.int32)); np.save(out / "cpool_src.npy", src[o])
    np.save(out / "cpool_form.npy", form[o]); np.save(out / "cpool_elem.npy", el[o]); np.save(out / "cpool_orig.npy", orig[o])
    with open(out / "cpool_smiles.txt", "w", encoding="utf-8", newline="\n") as f:
        for i in o: f.write(smi[i] + "\n")
    with open(out / "cft_formulas.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(forms) + "\n")
    mol_cpool = np.full(len(mol_key), -1, np.int32); mol_alt = np.full(len(mol_key), -1, np.int32)
    have = np.array([int(p) in p2 for p in mol_pool]) & (mol_pool >= 0)
    mol_cpool[have] = inv[[p2[int(p)] for p in mol_pool[have]]]
    nA, nB = len(pool_ix), int(okB.sum())
    for t, (_, i, _) in enumerate(alt_keep): mol_alt[i] = inv[nA + nB + t]
    np.save(out / "mol_cpool.npy", mol_cpool); np.save(out / "mol_alt.npy", mol_alt)
    S["cpool"] = dict(n=int(len(o)), from_existing_pool=nA, pubchem_decoys=nB, pubchem_form_of_training=len(aj),
                      nbits=int(nbits), bytes_per_row=int((nbits + 7) // 8), formulas=len(forms))
    log(f"cpool {S['cpool']}")

    # ---------------------------------------------------------------- PubChem validation set
    alt_ix = {i: j for j, i in enumerate(pcv_alt)}; C = Cb
    off = [0]; keep = []; pos = 0; n_full = []; in_pc = []; tfp = []; tfb = []
    for i, c, nf in pcv:
        k = [pos + t for t in range(len(c)) if okD[pos + t]]; pos += len(c)
        keep += k; off.append(len(keep)); n_full.append(nf)
        j = alt_ix.get(i)
        good = j is not None and C[4][j]
        in_pc.append(good)
        tfp.append(C[0][j] if good else np.zeros(C[0].shape[1], np.uint8))
        tfb.append(C[5][j] if good else np.zeros(C[5].shape[1], np.uint8))
    keep = np.array(keep, np.int64)
    np.save(out / "pcv_mol.npy", np.array([i for i, _, _ in pcv], np.int32)); np.save(out / "pcv_off.npy", np.array(off, np.int64))
    np.save(out / "pcv_fp.npy", Dv[0][keep]); np.save(out / "pcv_fpb.npy", Dv[5][keep])
    np.save(out / "pcv_form.npy", np.array([fid[Dv[2][t]] for t in keep], np.int32))
    np.save(out / "pcv_nfull.npy", np.array(n_full, np.int64)); np.save(out / "pcv_truth_in_pc.npy", np.array(in_pc, bool))
    np.save(out / "pcv_truth_fp.npy", np.stack(tfp)); np.save(out / "pcv_truth_fpb.npy", np.stack(tfb))
    with open(out / "pcv_smiles.txt", "w", encoding="utf-8", newline="\n") as f:
        for t in keep: f.write(flat[t] + "\n")
    S["pcv"] = dict(molecules=len(pcv), candidates=int(len(keep)), truth_in_pubchem=int(sum(in_pc)),
                    full_window_pct_10_50_90=[float(v) for v in np.percentile(n_full, [10, 50, 90])],
                    sampled_per_molecule_median=float(np.median(np.diff(off))))
    log(f"pcv {S['pcv']}")

    files = {p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.suffix in (".npy", ".txt")}
    M = dict(S, format=dict(
        version=1, nbits=int(nbits), ppm=a.ppm, seed=a.seed, elements=list(cft_fp.ELEMENTS),
        fingerprint="cfp1 (cft_fp.py): " + " | ".join(f"{n}:{b}" for n, b in cft_fp.BLOCKS) + " -> cfp_bits.npy",
        cpool_fp="np.packbits rows (MSB first), first nbits bits valid; pool sorted by exact mass",
        cpool_src="0 COCONUT, 1 training structure, 2 PubChem decoy, 3 PubChem form of a training/validation structure",
        cpool_kid="integer id of the InChIKey14; rows with the kid of the truth are never negatives",
        mol="mol_cpool / mol_alt: row of the molecule's structure in the pool (training SMILES / PubChem SMILES), -1 = none",
        pcv="pcv_mol: molecule index; candidates of molecule j = rows pcv_off[j]:pcv_off[j+1] of pcv_fp (cfp1) and "
            "pcv_fpb (warm-up 6,930-bit fingerprint); the truth is NOT among them; pcv_truth_fp/_fpb = fingerprint of "
            "the truth from its PubChem SMILES (valid when pcv_truth_in_pc); pcv_nfull = size of the full PubChem window"),
        files_bytes=files, seconds=round(time.time() - T0),
        versions=dict(rdkit=__import__("rdkit").__version__, duckdb=duckdb.__version__, numpy=np.__version__),
        base_data=dict(folder="casmi-train-pkg", pool_n=int(NP), meta_seconds=meta0.get("seconds")))
    json.dump(M, open(out / "cft_meta.json", "w"), indent=1)
    log(f"done: {sum(files.values()) / 1e9:.2f} GB in {out} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()

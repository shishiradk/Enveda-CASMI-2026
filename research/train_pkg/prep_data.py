"""Build the training arrays for the spectrum -> fingerprint networks (runs on CPU, here, once).

  python research/train_pkg/prep_data.py            (from the project root)

Reads train.parquet with DuckDB only. Removes every spectrum whose InChIKey14 is in the proxy
held-out key files (plus tautomers of those molecules), prepares peaks exactly as the engine does at
inference (fp_model.prep_peaks), builds the candidate pool used for hard negatives
(COCONUT + training structures, sorted by mass, held-out keys removed) and writes plain .npy files that
can be memory-mapped. See research/analysis/train_pkg_report.md for the layout.
"""
import argparse, hashlib, json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import fp_model as M

LIBS = ["enveda-180", "pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial",
        "drug_plus", "enveda-np-examples", "masaryk", "<other>"]
_g = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- structures
def _fp_init(bits_path):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _g["bits"] = np.load(bits_path)
    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)


def fp_and_mass(smi):
    """ECFP4(4096) | ECFP6(4096) | RDKitFP(2048, maxPath 6) | MACCS(167) -> fp_bits selection (public recipe)."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]
        return np.packbits(fp), float(ExactMolWt(m))
    except Exception:
        return None


def canon_key(smi):
    """Tautomer-canonical InChIKey14, the competition metric key (same as casmi_engine.canon_key)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------- spectra
def _spec_init(train, mol_keys, mol_ok):
    import duckdb
    _g["con"] = duckdb.connect()
    _g["con"].execute("set threads=1; set memory_limit='1500MB'")
    _g["train"] = train
    _g["k2m"] = {k: i for i, k in enumerate(mol_keys) if mol_ok[i]}


def spec_chunk(rng_):
    a, b = rng_
    con = _g["con"]; k2m = _g["k2m"]
    src = (f"from read_parquet('{_g['train']}', file_row_number=true) "
           f"where file_row_number>={a} and file_row_number<{b}")
    m = con.sql("select file_row_number r, precursor_mz p, adduct, instrument_type, ionization_mode, ingest_lib, "
                "inchikey14 ik, list_avg(list_transform(collision_energy_ev, x->abs(x))) ce, len(ms2_mzs) n "
                + src + " order by r").fetchnumpy()
    keep = np.array([k in k2m for k in m["ik"]], bool)
    out = dict(row=[], mol=[], prec=[], ad=[], ins=[], ce=[], mode=[], lib=[], n=[], mz=[], it=[])
    st = dict(rows=int(len(keep)), excluded_rows=int((~keep).sum()), empty=0, unk_adduct=0)
    if keep.any():
        p = con.sql("select file_row_number r, unnest(ms2_mzs) mz, unnest(ms2_normalized_intensities) it "
                    + src + " order by r").fetchnumpy()
        pr = np.asarray(p["r"]); pmz = np.ma.filled(p["mz"], np.nan); pit = np.ma.filled(p["it"], 0.0)
        lo = np.searchsorted(pr, m["r"], "left"); hi = np.searchsorted(pr, m["r"], "right")
        ce = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(m["ce"], dtype=np.float64)), -1.0)
        libix = {l: i for i, l in enumerate(LIBS)}
        for j in np.where(keep)[0]:
            prec = np.float32(m["p"][j])                      # the engine passes float32 precursors
            x, y = M.prep_peaks(pmz[lo[j]:hi[j]], pit[lo[j]:hi[j]], prec)
            if len(x) == 0:
                st["empty"] += 1; continue
            ad = M.ADDUCT_IX.get(m["adduct"][j], M.ADDUCT_IX["<unk>"])
            st["unk_adduct"] += int(ad == M.ADDUCT_IX["<unk>"])
            out["row"].append(m["r"][j]); out["mol"].append(k2m[m["ik"][j]]); out["prec"].append(prec)
            out["ad"].append(ad); out["ins"].append(M.instr_family(m["instrument_type"][j]))
            out["ce"].append(ce[j]); out["mode"].append(1 if m["ionization_mode"][j] == "positive" else -1)
            out["lib"].append(libix.get(m["ingest_lib"][j], len(LIBS) - 1)); out["n"].append(len(x))
            out["mz"].append(x); out["it"].append(y)
    res = dict(row=np.array(out["row"], np.int32), mol=np.array(out["mol"], np.int32),
               prec=np.array(out["prec"], np.float32), ad=np.array(out["ad"], np.uint8),
               ins=np.array(out["ins"], np.uint8), ce=np.array(out["ce"], np.float32),
               mode=np.array(out["mode"], np.int8), lib=np.array(out["lib"], np.uint8),
               n=np.array(out["n"], np.int64),
               mz=np.concatenate(out["mz"]) if out["mz"] else np.zeros(0, np.float32),
               it=np.concatenate(out["it"]) if out["it"] else np.zeros(0, np.float32))
    return res, st


def sha_head(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f: h.update(f.read(n))
    return f"{os.path.getsize(p)}:{h.hexdigest()[:16]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "train.parquet"))
    ap.add_argument("--held", nargs="*", default=[str(ROOT / "results/kaggle_v1_proxy/held_keys.parquet"),
                                                  str(ROOT / "results/kaggle_v2_proxy/s3_held.parquet")])
    ap.add_argument("--coco", default=str(ROOT / "external/bench_inputs/coco"))
    ap.add_argument("--out", default=str(ROOT / "results/train_pkg/data"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--val_mod", type=int, default=50, help="1 molecule in val_mod goes to validation")
    a = ap.parse_args()
    import duckdb
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    con = duckdb.connect(); con.execute("set threads=4; set memory_limit='3GB'")
    tr = a.train.replace("\\", "/")
    S = {"inputs": {"train": sha_head(a.train)}}

    # ---- held-out keys
    held = set()
    for p in a.held:
        k = [r[0] for r in con.sql(f"select distinct ik from '{Path(p).as_posix()}'").fetchall()]
        S["inputs"][Path(p).name] = len(k); held |= set(k)
    n_all, m_all = con.sql(f"select count(*), count(distinct inchikey14) from '{tr}'").fetchone()
    mols = con.sql(f"""select inchikey14 ik, arg_min(normalized_smiles, file_row_number) smi,
                        arg_min(molecular_formula, file_row_number) f, count(*) n
                        from read_parquet('{tr}', file_row_number=true) group by 1 order by 1""").fetchnumpy()
    keys = np.array(mols["ik"], object); smis = np.array(mols["smi"], object); forms = np.array(mols["f"], object)
    nspec = np.asarray(mols["n"])
    is_held = np.array([k in held for k in keys], bool)
    log(f"train: {n_all:,} spectra, {m_all:,} molecules; held keys {len(held)} "
        f"({int(is_held.sum())} present, {int(nspec[is_held].sum()):,} spectra)")

    # ---- tautomers of held molecules (same formula, same tautomer-canonical key) are held out too
    hf = set(forms[is_held])
    cand = np.where(~is_held & np.array([f in hf for f in forms], bool))[0]
    log(f"tautomer check: {len(cand):,} same-formula training molecules to canonicalise")
    with Pool(a.workers) as mp:
        ck_held = set(mp.map(canon_key, list(smis[is_held]), chunksize=8)) - {None}
        ck_cand = mp.map(canon_key, list(smis[cand]), chunksize=8)
    taut = np.zeros(len(keys), bool)
    taut[cand[[c in ck_held for c in ck_cand]]] = True
    excl = is_held | taut
    excl_keys = held | set(keys[taut])
    log(f"tautomer-excluded molecules: {int(taut.sum())} ({int(nspec[taut].sum()):,} spectra)")
    S["excluded"] = dict(held_keys=len(held), held_molecules_in_train=int(is_held.sum()),
                         held_spectra=int(nspec[is_held].sum()), tautomer_molecules=int(taut.sum()),
                         tautomer_spectra=int(nspec[taut].sum()), tautomer_keys=sorted(keys[taut]),
                         same_formula_checked=int(len(cand)))

    # ---- candidate pool = COCONUT U training structures (public recipe), minus excluded keys
    bits_path = str(Path(a.coco) / "fp_bits.npy")
    bits = np.load(bits_path); nbits = len(bits)
    cm = pickle.load(open(Path(a.coco) / "coco_meta.pkl", "rb"))
    assert cm["nbits"] == nbits
    co_fp = np.load(Path(a.coco) / "coco_fp.npy"); co_mass = np.load(Path(a.coco) / "coco_mass.npy")
    co_keys = np.asarray(cm["keys"], object); co_smi = np.asarray(cm["smiles"], object)
    co_set = set(co_keys)
    need = np.where(~excl & np.array([k not in co_set for k in keys], bool))[0]
    log(f"COCONUT {len(co_keys):,}; fingerprinting {len(need):,} training structures not in COCONUT")
    with Pool(a.workers, initializer=_fp_init, initargs=(bits_path,)) as mp:
        res = mp.map(fp_and_mass, list(smis[need]), chunksize=500)
    ok = np.array([r is not None and np.isfinite(r[1]) for r in res], bool)
    tr_fp = np.stack([res[i][0] for i in np.where(ok)[0]]); tr_mass = np.array([res[i][1] for i in np.where(ok)[0]])
    ck = np.array([k not in excl_keys for k in co_keys], bool) & np.isfinite(co_mass)
    fp = np.vstack([co_fp[ck], tr_fp]); del co_fp
    mass = np.concatenate([co_mass[ck], tr_mass])
    pkeys = np.concatenate([co_keys[ck], keys[need][ok]])
    psmi = np.concatenate([co_smi[ck], smis[need][ok]])
    psrc = np.concatenate([np.zeros(int(ck.sum()), np.uint8), np.ones(int(ok.sum()), np.uint8)])
    o = np.argsort(mass, kind="mergesort")
    fp, mass, pkeys, psmi, psrc = fp[o], mass[o], pkeys[o], psmi[o], psrc[o]
    assert not (set(pkeys) & excl_keys)
    k2p = {k: i for i, k in enumerate(pkeys)}
    mol_pool = np.array([k2p.get(k, -1) for k in keys], np.int32)
    mol_pool[excl] = -1
    mol_ok = mol_pool >= 0
    S["pool"] = dict(n=int(len(mass)), coconut=int(ck.sum()), coconut_removed=int((~ck).sum()),
                     train_only=int(ok.sum()), rdkit_failed=int((~ok).sum()), nbits=int(nbits),
                     train_mols_with_target=int(mol_ok.sum()),
                     train_mols_target_from_coconut=int((mol_ok & np.array([k in co_set for k in keys])).sum()))
    log(f"pool {len(mass):,} ({S['pool']})")

    # ---- molecule-level split (deterministic hash of the key)
    hv = np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) for k in keys], np.int64)
    mol_split = np.where(hv % a.val_mod == 0, 1, 0).astype(np.uint8)       # 1 = validation
    mol_fold = ((hv // a.val_mod) % 5).astype(np.uint8)                     # spare 5-fold id for later jobs
    mol_split[~mol_ok] = 255                                                # not used

    # ---- spectra
    chunks = [(s, min(s + a.chunk, n_all)) for s in range(0, n_all, a.chunk)]
    parts = []; st = dict(rows=0, excluded_rows=0, empty=0, unk_adduct=0)
    with Pool(a.workers, initializer=_spec_init, initargs=(tr, list(keys), mol_ok)) as mp:
        for i, (r, s) in enumerate(mp.imap(spec_chunk, chunks)):
            parts.append(r)
            for k in st: st[k] += s[k]
            if i % 8 == 0 or i == len(chunks) - 1:
                log(f"  spectra chunk {i + 1}/{len(chunks)}  kept {sum(len(p['row']) for p in parts):,}  "
                    f"{time.time() - T0:.0f}s")
    cat = lambda k: np.concatenate([p[k] for p in parts])
    n = cat("n"); off = np.zeros(len(n) + 1, np.int64); np.cumsum(n, out=off[1:])
    mol = cat("mol")
    A = dict(spec_off=off, spec_mz=cat("mz"), spec_it=cat("it"), spec_prec=cat("prec"), spec_ad=cat("ad"),
             spec_ins=cat("ins"), spec_ce=cat("ce"), spec_mode=cat("mode"), spec_lib=cat("lib"),
             spec_row=cat("row"), spec_mol=mol, spec_split=mol_split[mol],
             mol_key=np.array(list(keys), "S14"), mol_pool=mol_pool, mol_split=mol_split, mol_fold=mol_fold,
             pool_fp=fp, pool_mass=mass, pool_key=np.array(list(pkeys), "S14"), pool_src=psrc, fp_bits=bits)
    del parts
    assert len(A["spec_mz"]) == off[-1] and (A["spec_split"] < 2).all()
    for k, v in A.items(): np.save(out / f"{k}.npy", v)
    with open(out / "mol_smiles.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("inchikey14\tsmiles\tn_spectra_train_parquet\n")
        for k, s, c in zip(keys, smis, nspec): f.write(f"{k}\t{s}\t{c}\n")
    with open(out / "pool_smiles.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(psmi) + "\n")

    sp = A["spec_split"]; um = lambda m: int(len(np.unique(mol[m])))
    S["spectra"] = dict(total_in_train_parquet=int(n_all), molecules_in_train_parquet=int(m_all),
                        rows_removed_excluded_or_no_target=st["excluded_rows"], empty_after_prep=st["empty"],
                        kept=int(len(mol)), kept_molecules=um(slice(None)),
                        train=int((sp == 0).sum()), train_molecules=um(sp == 0),
                        val=int((sp == 1).sum()), val_molecules=um(sp == 1),
                        peaks=int(off[-1]), mean_peaks=float(n.mean()), share_at_cap=float((n == M.MAX_PEAKS).mean()),
                        unk_adduct=st["unk_adduct"], ce_missing=int((A["spec_ce"] < 0).sum()),
                        by_lib={LIBS[i]: int(c) for i, c in enumerate(np.bincount(A["spec_lib"], minlength=len(LIBS)))},
                        by_instr={M.INSTR_LIST[i]: int(c) for i, c in enumerate(np.bincount(A["spec_ins"], minlength=5))})
    S["lists"] = dict(adducts=M.ADDUCT_LIST, instruments=M.INSTR_LIST, libs=LIBS)
    S["format"] = dict(version=1, max_peaks=M.MAX_PEAKS, nbits=int(nbits), val_mod=a.val_mod,
                       split="0=train 1=validation (by molecule, md5(inchikey14) % val_mod == 0)",
                       intensity="sqrt(i / max i) after prep_peaks; m/z ascending; <=128 peaks per spectrum",
                       ce="mean |collision energy| in eV, -1 when missing (training substitutes 25.0)",
                       pool_fp="np.packbits rows (MSB first), first nbits bits valid, pool sorted by exact mass")
    S["files_bytes"] = {p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.is_file()}
    S["seconds"] = round(time.time() - T0)
    import rdkit
    S["versions"] = dict(rdkit=rdkit.__version__, duckdb=duckdb.__version__, numpy=np.__version__)
    json.dump(S, open(out / "meta.json", "w"), indent=1)
    log(json.dumps({k: S[k] for k in ("excluded", "pool", "spectra")}, indent=1)[:3000])
    log(f"done in {time.time() - T0:.0f}s; {sum(S['files_bytes'].values()) / 1e9:.2f} GB in {out}")


if __name__ == "__main__":
    main()

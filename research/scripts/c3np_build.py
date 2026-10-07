"""C3NP: a leak-free natural-product Class-3 PROXY bench.

    python research/scripts/c3np_build.py [--workers 4]

The 550 bench molecules of results/bench/truth.parquet (S12 = 250 enveda-np-examples, timsTOF; S3 = 300 GNPS
natural products outside COCONUT) are turned into Class-3 queries: the true structure, and every structure that the
competition metric would count as the same answer, is treated as absent from every structure database. A de novo /
analog-editing generator then gets, per molecule, only leak-free context, and baseline retrieval scores 0 by
construction (checked by the baselines at the end).

Stages (each writes a checkpoint under results/c3np/work/ and is skipped when that file exists):
  base      truth keys: raw InChIKey14, tautomer-canonical key (ckey), exact mass, formula, element-graph hash,
            engine target / adducts / modes / n_spec (ho1 bench records: S2 for S12 molecules, S3 for S3).
  pool      the bench structure pool E.POOL (results/bench/cache/pool_train_*.pkl, 772,653 rows = COCONUT +
            ChEBI/LIPID MAPS + training structures) without its fingerprints -> work/pool_lite.parquet.
  isomers   every PubChem / COCONUT (universe + bench_inputs/coco) / E.POOL structure within +-1e-5 Da of a
            truth's exact mass (same-formula isomers; PubChem `mass` is exact), plus every row whose raw key is a
            bench `correct` key or bench parent-group key (any mass).
  verify    tautomer-invariant prefilter (element/degree multiset, then canonical bond-order-free skeleton SMILES),
            then RDKit tautomer canonicalisation of the skeleton matches; aliases = rows whose ckey equals the truth's ckey (or the ckey of a bench alias).
  molecules -> results/c3np/molecules.parquet, results/c3np/forbidden.parquet.
  analogs   top library analogs (ho1 bench `ana`: S2 records for S12 = library masked by held_keys; S3 records for
            S3), forbidden keys removed by raw key AND by canonical key, top 50.
  window    +-5 ppm (around the engine target) PubChem window (results/c3/e6_work, scored fp @ zlog with the ho1
            nets) + COCONUT window (bench_inputs/coco, scored here); forbidden keys removed; top 200 after
            canonical-key de-duplication.
  context   -> results/c3np/context.pkl.
  reach     -> results/c3np/reach.parquet, reach_summary.json (Morgan r2 2048-bit Tanimoto of the truth to (i) the
            sanitized analogs, (ii) the sanitized window, (iii) E.POOL shifted by every single edit, (iv) the whole
            sanitized E.POOL), plus a 1-edit substructure reachability proxy.
  baselines window / analog candidate files -> research/scripts/c3np_eval.py -> results/c3np/eval_{window,analog}.json.

Resource rules: DuckDB with memory_limit 3GB for PubChem; at most --workers (<=4) processes; no train.parquet read.
"""
import argparse, json, pickle, re, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3np"
WK = OUT / "work"
PC = (ROOT / "external/pubchem/pubchem_rows.parquet").as_posix()
UNI = (ROOT / "results/kaggle_v1_assets/universe.parquet").as_posix()
COCO = ROOT / "external/bench_inputs/coco"
EPOOL = ROOT / "results/bench/cache/pool_train_3033286496_2026.03.6.pkl"
BENCH_CANON = ROOT / "results/bench/cache/canon.pkl"
E6 = ROOT / "results/c3/e6_work"
TOL_ISO = 1e-5      # Da, same-formula isomer window (exact masses)
PPM = 5.0           # window / edit-shift tolerance
N_ANA, N_WIN = 50, 200
SCEN = {"S12": "S2", "S3": "S3"}           # which ho1 bench records serve each query set
QFILE = {"S12": "results/bench/queries_S12.parquet", "S3": "results/bench/queries_S3.parquet"}

AM = {"C": 12.0, "H": 1.00782503207, "O": 15.99491461956, "N": 14.0030740048, "S": 31.971071, "P": 30.97376163,
      "Cl": 34.968852682}
# single-step edits (prior-art table, research brief section 3): name -> element delta of (product - parent)
EDITS = [("hydrogenation", "H2"), ("methylenedioxy", "C"), ("CH2->C=O", "O,H-2"), ("methylation", "CH2"),
         ("amination", "NH"), ("hydroxylation", "O"), ("hydration", "H2O"), ("C2H2", "C2H2"), ("ethyl", "C2H4"),
         ("methoxylation", "CH2O"), ("dihydroxylation", "O2"), ("chlorination", "Cl,H-1"), ("acetyl", "C2H2O"),
         ("carboxylation", "CO2"), ("glycine", "C2H3NO"), ("prenyl", "C5H8"), ("sulfate", "SO3"),
         ("phosphate", "HPO3"), ("isovaleryl", "C5H8O"), ("malonyl", "C3H2O3"), ("succinyl", "C4H4O3"),
         ("benzoyl", "C7H4O"), ("pentose", "C5H8O4"), ("geranyl", "C10H16"), ("p-coumaroyl", "C9H6O2"),
         ("deoxyhexose", "C6H10O4"), ("galloyl", "C7H4O4"), ("caffeoyl", "C9H6O3"), ("hexose", "C6H10O5"),
         ("glucuronide", "C6H8O6"), ("feruloyl", "C10H8O3"), ("sinapoyl", "C11H10O4"),
         ("hexose+pentose", "C11H18O8"), ("rutinoside", "C12H20O9"), ("dihexoside", "C12H20O10")]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def parse_formula(f):
    """'C6H10O5' -> Counter; 'O,H-2' -> {'O': 1, 'H': -2} (comma form allows negative counts)."""
    c = Counter()
    if "," in f:
        for part in f.split(","):
            m = re.fullmatch(r"([A-Z][a-z]?)(-?\d*)", part)
            c[m.group(1)] += int(m.group(2)) if m.group(2) not in ("", "-") else (1 if m.group(2) == "" else -1)
        return c
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f.split("+")[0].split("-")[0]):
        c[el] += int(n) if n else 1
    return c


EDIT_DELTA = [(n, parse_formula(f), sum(AM[e] * k for e, k in parse_formula(f).items())) for n, f in EDITS]


# ---------------------------------------------------------------------------------------------- RDKit workers
_G = {}


def _rd():
    if not _G:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors, rdMolHash
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _G.update(Chem=Chem, DS=DataStructs, D=Descriptors, RMD=rdMolDescriptors, MH=rdMolHash,
                  te=rdMolStandardize.TautomerEnumerator(),
                  mg=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    return _G


def canon_one(smi):
    """Competition key: TautomerEnumerator().Canonicalize -> MolToInchiKey[:14] (= casmi_engine.canon_key)."""
    g = _rd()
    try:
        m = g["Chem"].MolFromSmiles(smi) if smi else None
        return g["Chem"].MolToInchiKey(g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def eg_one(smi):
    g = _rd()
    try:
        m = g["Chem"].MolFromSmiles(smi) if smi else None
        return g["MH"].MolHash(m, g["MH"].HashFunction.ElementGraph) if m is not None else None
    except Exception:
        return None


def _bare(smi):
    """Unsanitised parse with explicit H atoms removed (bond orders / aromaticity / charges are then ignored)."""
    g = _rd()
    m = g["Chem"].MolFromSmiles(smi, sanitize=False) if smi else None
    if m is None:
        return None
    if any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        rw = g["Chem"].RWMol(m)
        for i in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
            rw.RemoveAtom(i)
        m = rw.GetMol()
    return m


def degseq_one(smi):
    """Cheap tautomer-invariant prefilter: sorted (element, heavy degree) multiset."""
    try:
        m = _bare(smi)
        return None if m is None else "|".join(sorted(f"{a.GetSymbol()}{a.GetDegree()}" for a in m.GetAtoms()))
    except Exception:
        return None


def skel_one(smi):
    """Tautomer-invariant skeleton: canonical SMILES of the heavy-atom graph with every bond single, no aromatic
    flags, charges, H, stereo or isotopes. (RDKit's ElementGraph MolHash is NOT invariant: it keeps aromaticity,
    so e.g. a flavanone and its aromatic-enol tautomer hash differently - found by the window ckey check.)"""
    g = _rd()
    Chem = g["Chem"]
    try:
        m = _bare(smi)
        if m is None:
            return None
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False); b.SetStereo(Chem.BondStereo.STEREONONE)
        for a in m.GetAtoms():
            a.SetIsAromatic(False); a.SetFormalCharge(0); a.SetNumExplicitHs(0); a.SetNoImplicit(True)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED); a.SetIsotope(0); a.SetNumRadicalElectrons(0)
        m.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def truth_info(smi):
    g = _rd()
    m = g["Chem"].MolFromSmiles(smi)
    return dict(ik14=g["Chem"].MolToInchiKey(m)[:14], ckey=canon_one(smi), exact_mass=g["D"].ExactMolWt(m),
                formula_rdkit=g["RMD"].CalcMolFormula(m), eg=eg_one(smi))


def _fp(smi):
    g = _rd()
    try:
        m = g["Chem"].MolFromSmiles(smi) if smi else None
        return g["mg"].GetFingerprint(m) if m is not None else None
    except Exception:
        return None


def tan_task(args):
    """(truth_smiles, [smiles]) -> (max Tanimoto, index of max, n>=0.7, n>=0.85, n==1.0, n parsed)."""
    tsmi, smis = args
    g = _rd()
    t = _fp(tsmi)
    fps = [_fp(s) for s in smis]
    ok = [i for i, f in enumerate(fps) if f is not None]
    if t is None or not ok:
        return (np.nan, -1, 0, 0, 0, len(ok))
    s = np.asarray(g["DS"].BulkTanimotoSimilarity(t, [fps[i] for i in ok]))
    j = int(np.argmax(s))
    return (float(s[j]), ok[j], int((s >= 0.7).sum()), int((s >= 0.85).sum()), int((s >= 0.9999).sum()), len(ok))


def fpbin_task(smis):
    out = []
    for s in smis:
        f = _fp(s)
        out.append(f.ToBinary() if f is not None else None)
    return out


def edit_task(args):
    """(truth_smiles, truth_eg, [(pool_row, smiles, edit_idx, sign)]) -> [(pool_row, edit_idx, sign, tan, reach)].
    reach (1-edit substructure proxy) requires the exact formula delta of the edit and, for H2, an identical bond-order-free
    skeleton; for any other addition (truth = parent + edit) the parent must be a substructure of the truth, for a
    removal (truth = parent - edit) the truth a substructure of the parent."""
    tsmi, teg, items = args
    g = _rd()
    Chem = g["Chem"]
    tm = Chem.MolFromSmiles(tsmi)
    tf = parse_formula(g["RMD"].CalcMolFormula(tm))
    tfp = g["mg"].GetFingerprint(tm)
    res = []
    for row, smi, ei, sign in items:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        tan = float(g["DS"].TanimotoSimilarity(tfp, g["mg"].GetFingerprint(m)))
        reach = False
        if ei >= 0:
            pf = parse_formula(g["RMD"].CalcMolFormula(m))
            d = Counter(tf); d.subtract(pf)
            want = Counter({k: sign * v for k, v in EDIT_DELTA[ei][1].items()})
            if all(d.get(k, 0) == want.get(k, 0) for k in set(d) | set(want)):
                try:
                    if EDIT_DELTA[ei][0] == "hydrogenation":
                        reach = skel_one(smi) == teg
                    elif sign > 0:
                        reach = tm.HasSubstructMatch(m)
                    else:
                        reach = m.HasSubstructMatch(tm)
                except Exception:
                    reach = False
        res.append((row, ei, sign, tan, bool(reach)))
    return res


def canon_many(smiles, workers):
    """smiles -> ckey with a disk cache (work/canon_cache.pkl), prefilled read-only from the bench cache."""
    p = WK / "canon_cache.pkl"
    if p.exists():
        C = pickle.load(open(p, "rb"))
    else:
        C = pickle.load(open(BENCH_CANON, "rb")) if BENCH_CANON.exists() else {}
    need = sorted({s for s in smiles if isinstance(s, str)} - set(C))
    if need:
        t0 = time.time()
        with Pool(workers) as mp:
            C.update(zip(need, mp.map(canon_one, need, chunksize=16)))
        pickle.dump(C, open(p, "wb"), protocol=4)
        log(f"canonicalised {len(need):,} new SMILES in {time.time() - t0:.0f}s (cache {len(C):,})")
    return C


def duck():
    import duckdb
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '3GB'; SET threads = 4; SET enable_progress_bar = false; "
            f"SET temp_directory = '{(WK / 'duck_tmp').as_posix()}'")
    return con


# ---------------------------------------------------------------------------------------------------- stages
def stage_base(a):
    f = WK / "base.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    T = pd.read_parquet(ROOT / "results/bench/truth.parquet")
    with Pool(a.workers) as mp:
        info = pd.DataFrame(mp.map(truth_info, T.smiles.tolist()))
    T = pd.concat([T.reset_index(drop=True), info], axis=1)
    rec, zlog = {}, {}
    for q, scen in SCEN.items():
        D = pickle.load(open(ROOT / f"results/bench/ho1/recs_{scen}.pkl", "rb"))
        S = pickle.load(open(ROOT / f"results/bench/ho1/{scen}.pkl", "rb"))
        for r in D["recs"]:
            z = D["zmk"].get(r["mid"])
            zlog[r["mid"]] = None if z is None else np.asarray(z, np.float32)
            rec[r["mid"]] = dict(target=float(r["target"]), adducts=list(r["adducts"]),
                                 modes=[int(x) for x in np.asarray(r["modes"]).ravel()],
                                 n_spec=int(S[r["mid"]]["n_spec"]) if r["mid"] in S else len(r["adducts"]),
                                 ana=[(int(p), float(s)) for p, s in r["ana"]],
                                 truth_canon_bench=S.get(r["mid"], {}).get("truth_canon"))
        del D, S
    A = json.load(open(ROOT / "results/exp012_coconut/target_aliases.json"))
    T["parents"] = [A.get(m, {}).get("parent", []) if q == "S12" else [] for m, q in zip(T.mid, T.qset)]
    res = dict(T=T, rec=rec, zlog=zlog)
    pickle.dump(res, open(f, "wb"), protocol=4)
    log(f"base: {len(T)} truths; raw key == mid {int((T.ik14 == T.mid).sum())}; "
        f"ckey == bench truth_canon {sum(rec[m]['truth_canon_bench'] == c for m, c in zip(T.mid, T.ckey))}; "
        f"zlog for {sum(v is not None for v in zlog.values())}")
    return res


def stage_pool():
    f = WK / "pool_lite.parquet"
    if f.exists():
        return pd.read_parquet(f)
    P = pickle.load(open(EPOOL, "rb"))
    d = pd.DataFrame({"key": P["keys"].astype(str), "smiles": P["smiles"].astype(str), "mass": P["mass"]})
    del P
    d.to_parquet(f, index=True)
    log(f"pool_lite: {len(d):,} E.POOL rows")
    return pd.read_parquet(f)


def stage_isomers(B, PL):
    f = WK / "isomers.parquet"
    if f.exists():
        return pd.read_parquet(f)
    T = B["T"]
    qb = pd.DataFrame([(r.mid, r.exact_mass, b) for r in T.itertuples()
                       for b in range(int(np.floor((r.exact_mass - TOL_ISO) * 1e4)),
                                      int(np.floor((r.exact_mass + TOL_ISO) * 1e4)) + 1)],
                      columns=["mid", "tmass", "bin"])
    kmap = pd.DataFrame([(r.mid, k) for r in T.itertuples()
                         for k in set(r.correct.split(";")) | {r.mid, r.ckey} | set(r.parents)],
                        columns=["mid", "ik"])
    con = duck()
    con.register("qb", qb); con.register("kmap", kmap)
    parts = []
    for src, sql in (("pubchem", f"SELECT ik, smiles, mass FROM read_parquet('{PC}')"),
                     ("universe_" , f"SELECT ik, smiles, mass, src FROM read_parquet('{UNI}')")):
        if src == "pubchem":
            w = con.sql(f"""SELECT qb.mid, s.ik, s.smiles, s.mass, 'pubchem' AS src FROM ({sql}) s
                            JOIN qb ON CAST(floor(s.mass * 1e4) AS BIGINT) = qb.bin
                            WHERE abs(s.mass - qb.tmass) <= {TOL_ISO}""").df()
            k = con.sql(f"""SELECT kmap.mid, s.ik, s.smiles, s.mass, 'pubchem' AS src FROM ({sql}) s
                            JOIN kmap USING (ik)""").df()
        else:
            w = con.sql(f"""SELECT qb.mid, s.ik, s.smiles, s.mass, 'universe_' || s.src AS src FROM ({sql}) s
                            JOIN qb ON CAST(floor(s.mass * 1e4) AS BIGINT) = qb.bin
                            WHERE abs(s.mass - qb.tmass) <= {TOL_ISO}""").df()
            k = con.sql(f"""SELECT kmap.mid, s.ik, s.smiles, s.mass, 'universe_' || s.src AS src FROM ({sql}) s
                            JOIN kmap USING (ik)""").df()
        parts += [w, k]
        log(f"isomers[{src}]: {len(w):,} mass-window rows, {len(k):,} key-lookup rows")
    # E.POOL and the bench_inputs COCONUT set: in memory (sorted exact masses)
    cm = pickle.load(open(COCO / "coco_meta.pkl", "rb"))
    coco = pd.DataFrame({"ik": np.asarray(cm["keys"]).astype(str), "smiles": np.asarray(cm["smiles"]).astype(str),
                         "mass": np.load(COCO / "coco_mass.npy")})
    del cm
    for src, d in (("epool", PL.rename(columns={"key": "ik"})), ("coco", coco)):
        m = d.mass.values
        rows = []
        for r in T.itertuples():
            lo, hi = np.searchsorted(m, r.exact_mass - TOL_ISO), np.searchsorted(m, r.exact_mass + TOL_ISO, "right")
            rows.append(d.iloc[lo:hi][["ik", "smiles", "mass"]].assign(mid=r.mid))
        w = pd.concat(rows, ignore_index=True)
        k = d[d.ik.isin(set(kmap.ik))][["ik", "smiles", "mass"]].merge(kmap, on="ik")
        parts += [w.assign(src=src), k.assign(src=src)]
        log(f"isomers[{src}]: {len(w):,} mass-window rows, {len(k):,} key-lookup rows")
    iso = pd.concat(parts, ignore_index=True).drop_duplicates(["mid", "src", "ik"]).reset_index(drop=True)
    iso.to_parquet(f, index=False)
    log(f"isomers: {len(iso):,} (truth, source, structure) rows; {iso.smiles.nunique():,} unique SMILES; "
        f"median {int(iso.groupby('mid').size().median())} per truth")
    return iso


def stage_verify(a, B, iso):
    f = WK / "verified.parquet"
    if f.exists():
        return pd.read_parquet(f)
    T = B["T"].set_index("mid")
    u = iso.drop_duplicates("smiles").smiles.tolist()
    t0 = time.time()
    tds = {m: degseq_one(s) for m, s in zip(T.index, T.smiles)}
    tsk = {m: skel_one(s) for m, s in zip(T.index, T.smiles)}
    with Pool(a.workers) as mp:
        H = dict(zip(u, mp.map(degseq_one, u, chunksize=500)))
        iso = iso.assign(ds=iso.smiles.map(H))
        del H
        same_ds = np.array([d is not None and d == tds[m] for m, d in zip(iso.mid, iso.ds)])
        u2 = iso[same_ds].drop_duplicates("smiles").smiles.tolist()
        S2 = dict(zip(u2, mp.map(skel_one, u2, chunksize=50)))
    log(f"skeletons: degree-sequence prefilter on {len(u):,} SMILES, {len(u2):,} skeletons, {time.time() - t0:.0f}s")
    iso = iso.assign(sk=iso.smiles.map(S2))
    keyset = {m: set(r.correct.split(";")) | {m, r.ckey} | set(r.parents) for m, r in T.iterrows()}
    known = np.array([k in keyset[m] for m, k in zip(iso.mid, iso.ik)])
    same_eg = np.array([s is not None and s == tsk[m] for m, s in zip(iso.mid, iso.sk)])
    sel = iso[same_eg | known].copy()
    log(f"skeleton prefilter: {len(iso):,} rows -> {int(same_ds.sum()):,} same degree sequence -> "
        f"{int(same_eg.sum()):,} share the truth's bond-order-free skeleton; {int(known.sum()):,} have a known bench key")
    C = canon_many(sel.smiles.tolist(), a.workers)
    sel["ckey"] = sel.smiles.map(C)
    # truth ckey set: the truth's own ckey + the ckeys of its bench `correct` aliases (EXP-012 / S3 tautomer keys)
    cks = {m: {r.ckey} for m, r in T.iterrows()}
    corr = {m: set(r.correct.split(";")) | {m} for m, r in T.iterrows()}
    for r in sel.itertuples():
        if r.ik in corr[r.mid] and isinstance(r.ckey, str):
            cks[r.mid].add(r.ckey)
    sel["match"] = [("ckey" if (isinstance(c, str) and c in cks[m]) else "") +
                    ("|rawkey" if k in keyset[m] else "") for m, k, c in zip(sel.mid, sel.ik, sel.ckey)]
    sel.to_parquet(f, index=False)
    log(f"verify: {len(sel):,} rows canonicalised; {int((sel.match != '').sum()):,} alias rows")
    return sel


def stage_molecules(B, ver):
    fm, ff = OUT / "molecules.parquet", OUT / "forbidden.parquet"
    if fm.exists() and ff.exists():
        return pd.read_parquet(fm), pd.read_parquet(ff)
    T, rec = B["T"].copy(), B["rec"]
    al = ver[ver.match != ""]
    corr = {m: set(c.split(";")) for m, c in zip(T.mid, T.correct)}
    F = []
    for r in T.itertuples():
        F += [(r.mid, r.ik14, "truth_ik14"), (r.mid, r.ckey, "truth_ckey")]
        F += [(r.mid, k, "bench_correct") for k in sorted(corr[r.mid])]
        F += [(r.mid, k, "bench_parent_group") for k in r.parents]
    for x in al.itertuples():
        F.append((x.mid, x.ik, f"alias_raw:{x.src}"))
        if isinstance(x.ckey, str):
            F.append((x.mid, x.ckey, f"alias_ckey:{x.src}"))
    F = pd.DataFrame(F, columns=["mid", "key", "why"]).dropna()
    # one row per (mid, key); keep the first reason in priority order, list the others
    pri = {"truth_ik14": 0, "truth_ckey": 1, "bench_correct": 2, "bench_parent_group": 3}
    F["p"] = F.why.map(lambda w: pri.get(w, 4))
    F = F.sort_values(["mid", "key", "p", "why"])
    F = F.groupby(["mid", "key"], as_index=False).agg(why=("why", lambda s: ";".join(dict.fromkeys(s))))
    base = set(zip(T.mid, T.ik14)) | set(zip(T.mid, T.ckey)) | {(m, k) for m, ks in corr.items() for k in ks}
    F["extra"] = [(m, k) not in base for m, k in zip(F.mid, F.key)]
    F.to_parquet(ff, index=False)
    # membership (by ckey-equivalence or raw key)
    src = al.groupby("mid").src.agg(lambda s: set(s))
    T["in_pc"] = [("pubchem" in src.get(m, set())) for m in T.mid]
    T["in_coco"] = [bool({"universe_coconut", "coco"} & src.get(m, set())) for m in T.mid]
    T["in_epool"] = [("epool" in src.get(m, set())) for m in T.mid]
    T["subset"] = np.where(T.qset == "S12", "npex", np.where(T.in_pc, "s3pc", "s3none"))
    T["s3_in_coco_alias"] = (T.qset == "S3") & T.in_coco
    # correct = ik14 + ckey + bench aliases + raw keys of verified tautomer aliases (ckey-equal only)
    tw = al[al.match.str.startswith("ckey")].groupby("mid").ik.agg(set)
    T["correct_bench"] = T.correct
    T["correct"] = [";".join(sorted(corr[m] | {i, c} | tw.get(m, set()))) for m, i, c in zip(T.mid, T.ik14, T.ckey)]
    T["target"] = T.mid.map(lambda m: rec[m]["target"])
    T["adducts"] = T.mid.map(lambda m: ";".join(sorted(set(rec[m]["adducts"]))))
    T["n_spec"] = T.mid.map(lambda m: rec[m]["n_spec"])
    T["scen"] = T.qset.map(SCEN)
    T["spectra_file"] = T.qset.map(QFILE)
    M = T[["mid", "subset", "qset", "scen", "smiles", "ik14", "ckey", "correct", "correct_bench", "formula",
           "formula_rdkit", "exact_mass", "target", "adducts", "n_spec", "in_pc", "in_coco", "in_epool",
           "s3_in_coco_alias", "spectra_file"]]
    M.to_parquet(fm, index=False)
    log(f"molecules: {M.subset.value_counts().to_dict()}; forbidden {len(F):,} keys, {int(F.extra.sum())} beyond "
        f"ik14/ckey/bench-correct; S3 with a COCONUT tautomer alias {int(M.s3_in_coco_alias.sum())}")
    return M, F


def fset(F):
    return F.groupby("mid").key.agg(set).to_dict()


def stage_analogs(a, B, M, F, PL):
    f = WK / "analogs.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    FS = fset(F)
    keys, smis, mass = PL.key.values, PL.smiles.values, PL.mass.values
    pre, stats = {}, Counter()
    for r in M.itertuples():
        lst = []
        for p, s in B["rec"][r.mid]["ana"]:
            if keys[p] in FS[r.mid]:
                stats["drop_raw"] += 1
                continue
            lst.append((p, s))
            if len(lst) >= N_ANA + 20:
                break
        pre[r.mid] = lst
    C = canon_many([smis[p] for v in pre.values() for p, _ in v], a.workers)
    res = {}
    for r in M.itertuples():
        out, seen = [], set()
        for p, s in pre[r.mid]:
            ck = C.get(smis[p])
            if ck in FS[r.mid]:
                stats["drop_ckey"] += 1
                continue
            if ck is not None and ck in seen:
                stats["dup_ckey"] += 1
                continue
            seen.add(ck)
            out.append(dict(smiles=smis[p], ik14=keys[p], ckey=ck, sim=round(s, 6),
                            dmass=float(mass[p] - B["rec"][r.mid]["target"]), pool_row=int(p)))
            if len(out) >= N_ANA:
                break
        res[r.mid] = out
    pickle.dump(res, open(f, "wb"), protocol=4)
    log(f"analogs: {dict(stats)}; median kept {int(np.median([len(v) for v in res.values()]))}")
    return res


def stage_window(a, B, M, F):
    ff, ft = WK / "window_full.parquet", WK / "window_top.pkl"
    if ff.exists() and ft.exists():
        return pd.read_parquet(ff), pickle.load(open(ft, "rb"))
    con = duck()
    wf = (E6 / "windows_5.0ppm.parquet").as_posix()
    want = pd.DataFrame({"scen": M.scen, "mid": M.mid})
    con.register("want", want)
    w = con.sql(f"""SELECT w.file_row_number AS row, w.mid, w.ik, w.smiles
                    FROM read_parquet('{wf}', file_row_number = true) w JOIN want USING (scen, mid)""").df()
    sc = np.load(E6 / "scores_5.0ppm.npy", mmap_mode="r")
    w["score"] = np.asarray(sc[w.row.values], np.float32)
    w["src"] = "pubchem"
    log(f"window[pubchem]: {len(w):,} rows for {w.mid.nunique()} molecules")
    # spot-check the cached scores against the cached fingerprints (fp @ zlog)
    chunks = sorted((E6 / "fp").glob("chunk_*.npz"))
    firsts = [str(np.load(c, allow_pickle=True)["ik"][0]) for c in chunks]
    rng = np.random.default_rng(0)
    samp = w[np.isfinite(w.score)].iloc[rng.choice(int(np.isfinite(w.score).sum()), 300, replace=False)]
    err = []
    for r in samp.itertuples():
        c = int(np.searchsorted(firsts, r.ik, "right") - 1)
        z = np.load(chunks[c], allow_pickle=True)
        ik = z["ik"].astype(str); j = int(np.searchsorted(ik, r.ik))
        assert ik[j] == r.ik
        fp = np.unpackbits(z["fp"][j])[:6930].astype(np.float32)
        err.append(abs(float(fp @ B["zlog"][r.mid]) - r.score))
    log(f"window score spot-check: max |cached - recomputed| over 300 rows = {max(err):.2e}")
    assert max(err) < 1e-2
    # COCONUT window (bench_inputs/coco), scored here
    cm = pickle.load(open(COCO / "coco_meta.pkl", "rb"))
    ck_, cs_ = np.asarray(cm["keys"]).astype(str), np.asarray(cm["smiles"]).astype(str)
    del cm
    cmass = np.load(COCO / "coco_mass.npy"); cfp = np.load(COCO / "coco_fp.npy", mmap_mode="r")
    parts = [w.drop(columns="row")]
    for r in M.itertuples():
        lo = np.searchsorted(cmass, r.target * (1 - PPM * 1e-6))
        hi = np.searchsorted(cmass, r.target * (1 + PPM * 1e-6), "right")
        if hi <= lo or B["zlog"][r.mid] is None:
            continue
        fp = np.unpackbits(np.asarray(cfp[lo:hi]), axis=1)[:, :6930].astype(np.float32)
        parts.append(pd.DataFrame({"mid": r.mid, "ik": ck_[lo:hi], "smiles": cs_[lo:hi],
                                   "score": fp @ B["zlog"][r.mid], "src": "coconut"}))
    w = pd.concat(parts, ignore_index=True)
    w = w[np.isfinite(w.score)]
    n0 = len(w)
    w["_o"] = (w.src != "pubchem").astype(np.int8)
    w = w.sort_values(["mid", "ik", "_o"], kind="mergesort")
    w.loc[w.duplicated(["mid", "ik"], keep=False).values, "src"] = "coconut+pubchem"
    w = w.drop_duplicates(["mid", "ik"]).drop(columns="_o")   # pubchem row (and SMILES) kept first
    FK = F[["mid", "key"]].rename(columns={"key": "ik"}).assign(_f=1)
    w = w.merge(FK, on=["mid", "ik"], how="left")
    n_forb = int(w._f.notna().sum())
    w = w[w._f.isna()].drop(columns="_f")
    w = w.sort_values(["mid", "score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    w.to_parquet(ff, index=False)
    log(f"window: {n0:,} scored rows -> {len(w):,} after (mid, ik) de-dup and dropping {n_forb} forbidden rows; "
        f"src {w.src.value_counts().to_dict()}")
    # top N_WIN after canonical de-dup and canonical forbidden check
    head = w.groupby("mid").head(N_WIN + 80)
    C = canon_many(head.smiles.tolist(), a.workers)
    FS = fset(F)
    top, stats = {}, Counter()
    for mid, g in head.groupby("mid", sort=False):
        out, seen = [], set()
        for ik, smi, s, src in zip(g.ik, g.smiles, g.score, g.src):
            c = C.get(smi)
            if c in FS[mid]:
                stats["drop_ckey"] += 1
                continue
            if c is not None and c in seen:
                stats["dup_ckey"] += 1
                continue
            seen.add(c)
            out.append(dict(ik=ik, ckey=c, smiles=smi, score=float(s), src=src))
            if len(out) >= N_WIN:
                break
        top[mid] = out
    pickle.dump(top, open(ft, "wb"), protocol=4)
    log(f"window top: {dict(stats)}; molecules with a window {len(top)}; "
        f"median size {int(np.median([len(v) for v in top.values()]))}")
    return w, top


def stage_context(B, M, ana, top):
    f = OUT / "context.pkl"
    if f.exists():
        return
    ctx = {}
    for r in M.itertuples():
        rc = B["rec"][r.mid]
        ctx[r.mid] = dict(subset=r.subset, qset=r.qset, target=rc["target"], adducts=rc["adducts"],
                          modes=rc["modes"], n_spec=rc["n_spec"], zlog=B["zlog"][r.mid],
                          spectra=dict(file=r.spectra_file, molecule_id=r.mid),
                          analogs=ana[r.mid], window=top.get(r.mid, []))
    meta = dict(built=time.strftime("%Y-%m-%d %H:%M"), zlog="ho1 nets (models/fp_ho1_akriti), held keys excluded",
                analogs=f"top {N_ANA} of the ho1 bench `ana` list (pool rows of E.POOL, entropy similarity with "
                        "precursor shift), forbidden keys removed by raw and canonical key; S12 from the S2 "
                        "records (library masked by held_keys), S3 from the S3 records",
                window=f"+-{PPM} ppm around target, PubChem + COCONUT, score = fp6930 @ zlog, forbidden removed, "
                       f"canonical de-dup, top {N_WIN}",
                not_included="truth formula / exact mass / smiles (see molecules.parquet; generators must not read it)")
    pickle.dump(dict(meta=meta, mols=ctx), open(f, "wb"), protocol=4)
    log(f"context: {len(ctx)} molecules -> {f}")


def stage_reach(a, B, M, F, PL, ana, wfull, top):
    fr, fs = OUT / "reach.parquet", OUT / "reach_summary.json"
    if fr.exists() and fs.exists():
        return
    FS = fset(F)
    T = M.set_index("mid")
    SK = {m: skel_one(s) for m, s in zip(M.mid, M.smiles)}
    R = pd.DataFrame(index=M.mid)
    t0 = time.time()
    with Pool(a.workers) as mp:
        # (i) analogs
        res = mp.map(tan_task, [(T.smiles[m], [x["smiles"] for x in ana[m]]) for m in M.mid], chunksize=4)
        R["ana_max_tan"] = [x[0] for x in res]
        R["ana_best"] = [ana[m][x[1]]["smiles"] if x[1] >= 0 else None for m, x in zip(M.mid, res)]
        R["ana_top1_tan"] = [x[0] for x in mp.map(tan_task, [(T.smiles[m], [x["smiles"] for x in ana[m][:1]])
                                                               for m in M.mid])]
        log(f"reach (i) analogs done {time.time() - t0:.0f}s")
        # (ii) full sanitized window; drop rows whose canonical key is known forbidden (top part was canonicalised)
        Ccache = pickle.load(open(WK / "canon_cache.pkl", "rb"))
        g = {m: d for m, d in wfull.groupby("mid", sort=False)}
        jobs, kept = [], {}
        for m in M.mid:
            d = g.get(m)
            if d is None:
                jobs.append((T.smiles[m], [])); kept[m] = []; continue
            sm = [s for s in d.smiles if Ccache.get(s) not in FS[m]]
            kept[m] = sm
            jobs.append((T.smiles[m], sm))
        del Ccache
        res = mp.map(tan_task, jobs, chunksize=2)
        R["win_n"] = [len(kept[m]) for m in M.mid]
        R["win_max_tan"] = [x[0] for x in res]
        R["win_n_ge07"] = [x[2] for x in res]
        R["win_n_ge085"] = [x[3] for x in res]
        R["win_n_tan1"] = [x[4] for x in res]
        R["win_best"] = [kept[m][x[1]] if x[1] >= 0 else None for m, x in zip(M.mid, res)]
        res = mp.map(tan_task, [(T.smiles[m], [x["smiles"] for x in top.get(m, [])]) for m in M.mid], chunksize=4)
        R["wintop_max_tan"] = [x[0] for x in res]
        log(f"reach (ii) window done {time.time() - t0:.0f}s")
        # (iii) E.POOL shifted by each single edit (and the isomer route, delta 0)
        pm, pk, ps = PL.mass.values, PL.key.values, PL.smiles.values
        jobs = []
        for m in M.mid:
            em, items = T.exact_mass[m], []
            for ei, (_, _, dm) in [(-1, (None, None, 0.0))] + list(enumerate(EDIT_DELTA)):
                for sign in ((1,) if ei < 0 else (1, -1)):
                    c = em - sign * dm          # parent mass: truth = parent + sign * edit
                    lo, hi = np.searchsorted(pm, c * (1 - PPM * 1e-6)), np.searchsorted(pm, c * (1 + PPM * 1e-6), "right")
                    items += [(int(p), ps[p], ei, sign) for p in range(lo, hi) if pk[p] not in FS[m]]
            jobs.append((T.smiles[m], SK[m], items))
        res = mp.map(edit_task, jobs, chunksize=2)
        names = ["isomer"] + [n for n, _, _ in EDIT_DELTA]
        rows = []
        for m, rr_ in zip(M.mid, res):
            iso = [x for x in rr_ if x[1] < 0]
            ed = [x for x in rr_ if x[1] >= 0]
            best = max(ed, key=lambda x: x[3]) if ed else None
            reach = [x for x in ed if x[4]]
            rows.append(dict(edit_n=len(ed), edit_max_tan=best[3] if best else np.nan,
                             edit_best=(("+" if best[2] > 0 else "-") + names[best[1] + 1]) if best else None,
                             edit_best_smiles=ps[best[0]] if best else None,
                             iso_n=len(iso), iso_max_tan=max((x[3] for x in iso), default=np.nan),
                             edit1_reach=bool(reach),
                             edit1_reach_edits=";".join(sorted({("+" if x[2] > 0 else "-") + names[x[1] + 1]
                                                                 for x in reach})),
                             edit1_reach_max_tan=max((x[3] for x in reach), default=np.nan)))
        R = R.join(pd.DataFrame(rows, index=M.mid))
        log(f"reach (iii) edits done {time.time() - t0:.0f}s")
        # (iv) nearest neighbour in the whole sanitized E.POOL
        CH = 20000
        bins = []
        for out in mp.imap(fpbin_task, [ps[i:i + CH].tolist() for i in range(0, len(ps), CH)]):
            bins.extend(out)
        log(f"reach (iv) E.POOL Morgan fingerprints {len(bins):,} in {time.time() - t0:.0f}s")
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect
    ok = np.array([b is not None for b in bins])
    fps = [ExplicitBitVect(b) for b in bins if b is not None]
    del bins
    okidx = np.where(ok)[0]
    k2i = pd.Series(np.arange(len(pk)), index=pk)
    nn, nn_s = [], []
    g = _rd()
    for m in M.mid:
        t = g["mg"].GetFingerprint(g["Chem"].MolFromSmiles(T.smiles[m]))
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(t, fps))
        bad = k2i.reindex(list(FS[m])).dropna().astype(int).values
        pos = np.searchsorted(okidx, bad)
        pos = pos[(pos < len(okidx)) & (okidx[np.minimum(pos, len(okidx) - 1)] == bad)]
        s[pos] = -1
        j = int(np.argmax(s))
        nn.append(float(s[j])); nn_s.append(ps[okidx[j]])
    R["pool_nn_tan"], R["pool_nn_smiles"] = nn, nn_s
    log(f"reach (iv) done {time.time() - t0:.0f}s")
    R["any_max_tan"] = R[["ana_max_tan", "win_max_tan", "edit_max_tan"]].max(axis=1)
    R = R.reset_index().merge(M[["mid", "subset"]], on="mid")
    R.to_parquet(fr, index=False)

    def summ(d):
        o = dict(n=len(d))
        for col in ("ana_max_tan", "ana_top1_tan", "win_max_tan", "wintop_max_tan", "edit_max_tan", "iso_max_tan",
                    "pool_nn_tan", "any_max_tan"):
            v = d[col].fillna(0).values
            o[col] = dict(median=round(float(np.median(v)), 4), ge07=round(float((v >= 0.7).mean()), 4),
                          ge085=round(float((v >= 0.85).mean()), 4), eq1=int((v >= 0.9999).sum()))
        o["edit1_reach_rate"] = round(float(d.edit1_reach.mean()), 4)
        o["window_n_tan1_molecules"] = int((d.win_n_tan1 > 0).sum())
        o["nn_strata"] = {k: int(v) for k, v in pd.cut(d.pool_nn_tan.fillna(0), [-1, 0.5, 0.7, 0.85, 1.01],
                                                        right=False, labels=["<0.5", "0.5-0.7", "0.7-0.85", ">=0.85"]
                                                        ).value_counts().sort_index().items()}
        o["edit1_reach_edits_top"] = Counter(e for s in d.edit1_reach_edits for e in s.split(";") if e).most_common(12)
        return o
    S = {"overall": summ(R)}
    for k, d in R.groupby("subset"):
        S[k] = summ(d)
    S["notes"] = dict(fingerprint="RDKit Morgan radius 2, 2048 bits, Tanimoto",
                      routes=dict(ana_max_tan="(i) sanitized top-50 library analogs",
                                  win_max_tan="(ii) full sanitized +-5 ppm PubChem+COCONUT window (around target)",
                                  wintop_max_tan="(ii') the top-200 window handed to generators",
                                  edit_max_tan="(iii) E.POOL structures at truth exact mass -+ each single-edit delta "
                                               "(+-5 ppm), 35 edits x 2 directions",
                                  iso_max_tan="(iii') E.POOL same-mass (+-5 ppm) isomers",
                                  pool_nn_tan="(iv) nearest neighbour in the whole sanitized E.POOL",
                                  any_max_tan="max of (i), (ii), (iii)"),
                      edit1_reach="INFERENCE-level proxy: an E.POOL structure differs from the truth by exactly one "
                                  "edit's formula AND (H2: same bond-order-free skeleton; addition: parent is a substructure of "
                                  "the truth; removal: truth is a substructure of the parent)")
    json.dump(S, open(fs, "w"), indent=1, default=str)
    log("reach summary: " + json.dumps({k: {c: v[c]["ge07"] for c in ("ana_max_tan", "win_max_tan", "edit_max_tan",
                                                                        "pool_nn_tan")} | {"edit1": v["edit1_reach_rate"]}
                                         for k, v in S.items() if k != "notes"}))


def stage_baselines(a, M, ana, top):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import c3np_eval
    for name, cands in (("window", {m: [x["smiles"] for x in top.get(m, [])] for m in M.mid}),
                        ("analog", {m: [x["smiles"] for x in ana[m]] for m in M.mid})):
        p = WK / f"cand_{name}.json"
        json.dump(cands, open(p, "w"))
        r = c3np_eval.evaluate(cands, name, workers=a.workers)
        log(f"baseline {name}: " + json.dumps({k: (v["mrr25"], v["hit25"]) for k, v in r["by_subset"].items()}
                                              | {"overall": (r["overall"]["mrr25"], r["overall"]["hit25"])}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.workers = min(a.workers, 4)
    WK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    B = stage_base(a)
    PL = stage_pool()
    iso = stage_isomers(B, PL)
    ver = stage_verify(a, B, iso)
    del iso
    M, F = stage_molecules(B, ver)
    ana = stage_analogs(a, B, M, F, PL)
    wfull, top = stage_window(a, B, M, F)
    stage_context(B, M, ana, top)
    stage_reach(a, B, M, F, PL, ana, wfull, top)
    del wfull
    stage_baselines(a, M, ana, top)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""E7 Class-3 channel: per test molecule, rr_bt = round-robin union of the biotransform and mmp_edit generators
(biotransform first), top --top entries with metric keys. Writes candidate lists only, never submission.csv.

    python e7_c3_channel.py TEST_PARQUET OUT_JSON --ctx E6_CTX.pkl --ana ENG_ANA.json --assets DIR [--coco DIR]
                            [--workers N] [--budget SEC] [--top 25] [--limit N]

Inputs (all produced earlier in the same Kaggle run; nothing is read from repo paths):
  --ctx    e7_e6_channel.py --dump-ctx output: per molecule target (median neutral mass, = engine target), zlog (our
           E6 nets, casmi-e6-pcnets) and the top PubChem +-5 ppm window rows by fp @ zlog.
  --ana    eng_ana.json from the E7 engine runner: casmi_engine.top_analogs output (library analogs, best first) as
           [smiles, ik14, sim, pool mass] plus target / adducts.
  --coco   directory of prvsiyan's coco_fp.npy / coco_mass.npy / coco_meta.pkl (COCONUT window, scored fp @ zlog here).
  --assets directory holding this script, biotransform.py, mmp_edit.py, assets.py and the c3_* asset files.
Context per molecule (same recipe as research/scripts/c3np_build.py, the C3NP bench builder):
  analogs: first 70 analogs -> tautomer-canonical key -> canonical de-dup -> top 50 {smiles, ik14, ckey, sim, dmass};
  window : PubChem + COCONUT rows, de-dup on ik (PubChem row kept), sort by score desc (ties: ik), first 280 rows
           canonicalised, canonical de-dup, top 200 {ik, ckey, smiles, score, fpscore, src};
  mid    : an opaque id ("c3_0000"); forbidden = frozenset() (deploy mode: full mined rules at min_freq 3, full pool).
Generators: generate(ctx, k=200) of each; rr_bt exactly as research/scratch_wf/c3_tournament/combine.py (plain-InChIKey14
de-dup, SMILES of a key seen in both lists taken from mmp_edit, interleave biotransform/mmp_edit), capped at 200; the
top --top entries get the competition metric key (TautomerEnumerator canonical InChIKey14).
Failure handling: a molecule whose context is missing (no zlog / no E6 window record / no engine analog record), or
whose context build or either generator raises, gets no list; when --coco is given and the COCONUT window cannot be
built, no molecule gets a list. Results are written atomically every 25 molecules, so a killed run leaves complete
lists only. --budget stops collecting (and terminates the workers) once the wall clock is exceeded.
OUT_JSON = {molecule_id: {"smiles": [...], "keys": [...], "src": ["bt"|"mmp", ...], "prov": [...]},
            "_stats": {...}, "_info": {molecule_id: {...}}}.
"""
import argparse, json, os, pickle, sys, time, traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_ANA, N_ANA_PRE = 50, 70          # c3np_build: N_ANA, N_ANA + 20
N_WIN, WIN_HEAD = 200, 280         # c3np_build: N_WIN, N_WIN + 80
PPM_WIN = 5.0
K_GEN, RR_CAP = 200, 200
FAILSAFE = dict(bt=(90.0, 120.0), mmp=(26.0, 36.0))   # wall-clock failsafes only (deterministic budgets bind first)
_W = {}


# ------------------------------------------------------------------------------------------- chemistry helpers
def _rd():
    if "Chem" not in _W:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _W.update(Chem=Chem, te=rdMolStandardize.TautomerEnumerator())
    return _W["Chem"]


def canon_key(smi):
    """Competition key: TautomerEnumerator().Canonicalize -> MolToInchiKey[:14] (= casmi_engine.canon_key)."""
    Chem = _rd()
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return Chem.MolToInchiKey(_W["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------- context builders
def build_analogs(ana, target, canon=canon_key, exclude=frozenset(), n_ana=N_ANA, n_pre=N_ANA_PRE):
    """ana = [[smiles, ik14, sim, mass], ...] in engine order. c3np_build.stage_analogs recipe: raw-key exclusion,
    first n_pre, canonical-key exclusion and de-dup (None keys are not de-duplicated), top n_ana."""
    pre = []
    for smi, ik, s, mass in ana:
        if ik in exclude:
            continue
        pre.append((smi, ik, float(s), float(mass)))
        if len(pre) >= n_pre:
            break
    out, seen = [], set()
    for smi, ik, s, mass in pre:
        ck = canon(smi)
        if ck in exclude:
            continue
        if ck is not None and ck in seen:
            continue
        seen.add(ck)
        out.append(dict(smiles=smi, ik14=ik, ckey=ck, sim=round(s, 6), dmass=float(mass - target)))
        if len(out) >= n_ana:
            break
    return out


def build_window(pc, coco, canon=canon_key, exclude=frozenset(), n_win=N_WIN, head=WIN_HEAD):
    """pc / coco = [(ik, smiles, score), ...]. c3np_build.stage_window recipe: non-finite scores dropped, de-dup on ik
    with the PubChem row kept (src 'coconut+pubchem' when in both), raw-key exclusion, sort by score desc (ties: ik
    ascending), first `head` rows canonicalised, canonical exclusion and de-dup, top n_win."""
    rows = {}
    for src, L in (("pubchem", pc or []), ("coconut", coco or [])):
        for ik, smi, sc in L:
            sc = float(sc)
            if not np.isfinite(sc):
                continue
            r = rows.get(ik)
            if r is not None:
                if r["src"] != src:
                    r["src"] = "coconut+pubchem"
                continue
            rows[ik] = dict(ik=ik, smiles=smi, score=sc, src=src)
    L = sorted((r for r in rows.values() if r["ik"] not in exclude), key=lambda r: (-r["score"], r["ik"]))[:head]
    out, seen = [], set()
    for r in L:
        c = canon(r["smiles"])
        if c in exclude:
            continue
        if c is not None and c in seen:
            continue
        seen.add(c)
        out.append(dict(ik=r["ik"], ckey=c, smiles=r["smiles"], score=r["score"], fpscore=r["score"], src=r["src"]))
        if len(out) >= n_win:
            break
    return out


def coco_windows(tz, coco_dir, ppm=PPM_WIN, head=WIN_HEAD):
    """{mid: (target, zlog)} -> {mid: [(ik, smiles, score), ...]} top `head` COCONUT rows within +-ppm by fp @ zlog
    (ties: ik). coco_* = prvsiyan/coconut-casmi26-candidates (mass-sorted, 6,930-bit packed fingerprints)."""
    cm = pickle.load(open(os.path.join(coco_dir, "coco_meta.pkl"), "rb"))
    ck, cs = np.asarray(cm["keys"]).astype(str), np.asarray(cm["smiles"]).astype(str)
    del cm
    cmass = np.load(os.path.join(coco_dir, "coco_mass.npy"))
    cfp = np.load(os.path.join(coco_dir, "coco_fp.npy"), mmap_mode="r")
    out = {}
    for mid, (t, z) in tz.items():
        lo = int(np.searchsorted(cmass, t * (1 - ppm * 1e-6)))
        hi = int(np.searchsorted(cmass, t * (1 + ppm * 1e-6), "right"))
        if hi <= lo:
            out[mid] = []
            continue
        z = np.asarray(z, np.float32)
        sc = np.unpackbits(np.asarray(cfp[lo:hi]), axis=1)[:, :len(z)].astype(np.float32) @ z
        o = np.lexsort((ck[lo:hi], -sc))[:head]
        out[mid] = [(ck[lo + j], cs[lo + j], float(sc[j])) for j in o]
    return out


# ------------------------------------------------------------------------------------------- rr_bt
def rr_bt(bt, mm, cap=RR_CAP):
    """combine.py rr_bt: bt / mm = generator outputs [(smiles, score, prov), ...] best first. Returns
    [(smiles, plain ik14, src, prov), ...]: interleave(biotransform, mmp_edit) on plain InChIKey14, cap entries."""
    Chem = _rd()
    keys, smis, src, prov = {}, [], {}, {}
    for which, L in (("a", mm), ("b", bt)):          # a = mmp_edit first: a shared key keeps mmp_edit's SMILES
        for r, x in enumerate(L):
            m = Chem.MolFromSmiles(x[0])
            if m is None:
                continue
            k = Chem.MolToInchiKey(m)[:14]
            if k not in keys:
                keys[k] = len(smis)
                smis.append(x[0])
                src[k] = {}
            if which not in src[k]:
                src[k][which] = r
                prov[(k, which)] = x[2]
    LA = sorted([(k, d["a"]) for k, d in src.items() if "a" in d], key=lambda x: x[1])
    LB = sorted([(k, d["b"]) for k, d in src.items() if "b" in d], key=lambda x: x[1])
    out, seen = [], set()
    for i in range(max(len(LA), len(LB))):
        for which, L in (("b", LB), ("a", LA)):
            if i < len(L) and L[i][0] not in seen:
                k = L[i][0]
                seen.add(k)
                out.append((smis[keys[k]], k, "bt" if which == "b" else "mmp", prov[(k, which)]))
    return out[:cap]


# ------------------------------------------------------------------------------------------- workers
def _winit(code_dir, asset_dir):
    sys.path.insert(0, code_dir)
    import biotransform as B
    import mmp_edit as M
    B.set_asset_dir(asset_dir)
    M.set_asset_dir(asset_dir)
    B.T_GEN, B.T_TOTAL = FAILSAFE["bt"]
    M.T_GEN, M.T_ALL = FAILSAFE["mmp"]
    B._rd(); B._pool()
    M._ensure_assets()
    _rd()
    _W.update(B=B, M=M)


def run_one(raw, ntop=25, exclude=frozenset(), B=None, M=None, forbidden=frozenset(), k=K_GEN):
    """raw = dict(qid, target, zlog, adducts, ana, pc, coco, spectra) -> (record, info). Raises on any failure."""
    B = B or _W["B"]
    M = M or _W["M"]
    t0 = time.time()
    target = float(raw["target"])
    ctx = dict(mid=raw["qid"], target=target, adducts=list(raw.get("adducts") or []),
               zlog=np.asarray(raw["zlog"], np.float32),
               analogs=build_analogs(raw["ana"], target, exclude=exclude),
               window=build_window(raw["pc"], raw.get("coco"), exclude=exclude),
               spectra=raw.get("spectra") or [])
    t1 = time.time()
    bt, bi = B.generate(ctx, k=k, forbidden=forbidden, return_table=True)
    t2 = time.time()
    mm, mi = M.generate(ctx, k=k, forbidden=forbidden, return_feats=True)
    t3 = time.time()
    rr = rr_bt(bt, mm)
    top = rr[:ntop]
    rec = dict(smiles=[x[0] for x in top], keys=[canon_key(x[0]) for x in top], src=[x[2] for x in top],
               prov=[x[3] for x in top])
    info = dict(n_ana=len(ctx["analogs"]), n_win=len(ctx["window"]), n_bt=len(bt), n_mmp=len(mm), n_rr=len(rr),
                t_ctx=round(t1 - t0, 2), t_bt=round(t2 - t1, 2), t_mmp=round(t3 - t2, 2), sec=round(time.time() - t0, 2),
                bt_guard=bi.get("guard", ""), bt_trunc=bi.get("truncated", ""), mmp_guard=mi.get("guard", ""))
    return rec, info, dict(bt=bt, mmp=mm, rr=rr, ctx=ctx)


def _job(args):
    mid, raw, ntop = args
    try:
        rec, info, _ = run_one(raw, ntop)
        return mid, rec, info, None
    except Exception:
        return mid, None, {}, traceback.format_exc(limit=4)


# ------------------------------------------------------------------------------------------- main
def _write(path, res, info, stats):
    out = dict(sorted(res.items()))
    out["_stats"] = stats
    out["_info"] = info
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)


def load_spectra(test, mids):
    import pandas as pd
    te = pd.read_parquet(test, columns=["molecule_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct"])
    te = te[te.molecule_id.notna()]
    te["molecule_id"] = te.molecule_id.astype(str)
    te = te[te.molecule_id.isin(set(mids))]
    sp = {}
    for r in te.itertuples():
        sp.setdefault(r.molecule_id, []).append(dict(mz=[float(x) for x in r.ms2_mzs], intensity=[float(x) for x in r.ms2_normalized_intensities],
                                                     precursor_mz=float(r.precursor_mz), adduct=str(r.adduct)))
    return sp, sorted(te.molecule_id.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test"); ap.add_argument("out")
    ap.add_argument("--ctx", required=True); ap.add_argument("--ana", required=True)
    ap.add_argument("--assets", default=HERE); ap.add_argument("--coco", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--budget", type=float, default=7200); ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    T0 = time.time()
    stats = dict(n_test=0, n_jobs=0, n_lists=0, n_fail=0, n_missing=0, budget_hit=False, coco_ok=None)
    D = pickle.load(open(a.ctx, "rb"))["mols"]
    ANA = json.load(open(a.ana))
    sp, test_mids = load_spectra(a.test, set(D) | set(ANA))
    import pandas as pd
    all_test = sorted(pd.read_parquet(a.test, columns=["molecule_id"]).molecule_id.dropna().astype(str).unique())
    if a.limit:
        all_test = all_test[:a.limit]
    stats["n_test"] = len(all_test)
    info, res, ready = {}, {}, []
    for mid in all_test:
        d, an = D.get(mid), ANA.get(mid)
        if d is None or d.get("zlog") is None or d.get("pc") is None or an is None or an.get("ana") is None:
            info[mid] = dict(status="missing", ctx=d is not None, pc=d is not None and d.get("pc") is not None,
                             ana=an is not None)
            continue
        ready.append(mid)
    stats["n_missing"] = len(all_test) - len(ready)
    coco = {}
    if a.coco:
        try:
            coco = coco_windows({m: (D[m]["target"], D[m]["zlog"]) for m in ready}, a.coco)
            stats["coco_ok"] = True
        except Exception:
            stats["coco_ok"] = False
            print("E7 C3: COCONUT window FAILED -> no lists\n" + traceback.format_exc(limit=3), flush=True)
            _write(a.out, {}, info, stats)
            return
    print(f"E7 C3: {len(all_test)} test molecules, {len(ready)} with full context, COCONUT {stats['coco_ok']} | "
          f"{time.time() - T0:.0f}s", flush=True)
    # pre-flight: a missing asset file raises inside the Pool initializer, and multiprocessing.Pool then respawns the
    # worker forever, so imap_unordered never returns and the run hangs instead of failing. Check first.
    NEED = ("c3_pool_mass.npy", "c3_pool_fp.npy", "c3_pool_key.npy", "c3_pool_smiles.txt", "c3_fp_bits.npy",
            "c3_mmp_rules.parquet", "c3_pool_np.npy")
    miss = [f for f in NEED if not os.path.exists(os.path.join(a.assets, f))]
    if miss:
        stats["assets_missing"] = miss
        print("E7 C3: assets MISSING in " + a.assets + " -> no lists: " + ", ".join(miss), flush=True)
        _write(a.out, {}, info, stats)
        return
    jobs = []
    for i, mid in enumerate(ready):
        d = D[mid]
        raw = dict(qid="c3_%04d" % i, target=d["target"], zlog=d["zlog"], adducts=ANA[mid].get("adducts") or [],
                   ana=ANA[mid]["ana"], pc=d["pc"], coco=coco.get(mid, []), spectra=sp.get(mid, []))
        jobs.append((mid, raw, a.top))
    stats["n_jobs"] = len(jobs)
    from multiprocessing import Pool, TimeoutError as MPTimeout
    pool = Pool(max(1, a.workers), initializer=_winit, initargs=(HERE, a.assets))
    try:
        it = pool.imap_unordered(_job, jobs, chunksize=1)
        for n in range(len(jobs)):
            left = a.budget - (time.time() - T0)
            if left <= 0:
                stats["budget_hit"] = True
                break
            try:
                mid, rec, inf, err = it.next(timeout=left)
            except MPTimeout:
                stats["budget_hit"] = True
                break
            if err:
                stats["n_fail"] += 1
                info[mid] = dict(status="error", err=err[-1500:])
                print("E7 C3 ERROR", mid, err[-600:], flush=True)
            else:
                res[mid] = rec
                info[mid] = dict(status="ok", **inf)
            if (n + 1) % 25 == 0 or n + 1 == len(jobs):
                stats["n_lists"] = len(res); stats["sec"] = round(time.time() - T0, 1)
                _write(a.out, res, info, stats)
                print(f"E7 C3 {n + 1}/{len(jobs)} molecules | lists {len(res)} | fail {stats['n_fail']} | "
                      f"{time.time() - T0:.0f}s", flush=True)
    finally:
        pool.terminate()
        pool.join()
    secs = [v["sec"] for v in info.values() if v.get("status") == "ok"]
    stats.update(n_lists=len(res), sec=round(time.time() - T0, 1),
                 mol_sec_mean=round(float(np.mean(secs)), 2) if secs else None,
                 mol_sec_max=round(float(np.max(secs)), 2) if secs else None,
                 bt_guard=sum(1 for v in info.values() if v.get("bt_guard")),
                 mmp_guard=sum(1 for v in info.values() if v.get("mmp_guard")))
    _write(a.out, res, info, stats)
    print("E7 C3 channel done", stats, flush=True)


if __name__ == "__main__":
    main()

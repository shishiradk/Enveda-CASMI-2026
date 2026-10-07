"""C3NP congener-strip tables (r2 realism fix): which database structures are close congeners of each bench truth.

    python -u research/scripts/c3np_strip.py [--workers 2]

The C3NP truths are known natural products with close relatives in the seed sources, so bench MRR mostly measures
"edit a near-identical database neighbour" (review r2: ~75% of mmp-edit RR from 1-heavy-atom edits; biotransform
loses 36% of npex MRR when seeds with Tanimoto >= 0.7 to the truth are removed). This script writes ORACLE tables
(it reads the truth) used ONLY by the bench, never by generate():

  results/c3np/strip.parquet   one row per (mid, key, src) congener: src in {pool, analog, window}; key = plain
                               InChIKey14 (pool_key for pool rows; the context entry's ik14/ik for analog/window rows,
                               plus its ckey when different); tc = Morgan r2/2048 Tanimoto to the truth; edit =
                               '' or the 1-edit label ('+CH2', '-O', '+CH2O', ...) when the formula differs from the
                               truth by exactly CH2 / O / CH2O and the smaller structure is a substructure of the larger.
                               Rows are kept when tc >= 0.70 or edit != ''.
  results/c3np/strata.parquet  per molecule: best_tc (max tc over pool / analogs / window), tc_band, nb1 (a 1-heavy-
                               atom neighbour, +-CH2 / +-O, exists in a seed source), nb2 (+-CH2O), n_tc070, n_tc085,
                               n_edit1.
  results/c3np/work/pool_m2_2048.npy   packed Morgan r2/2048 fingerprints of results/train_pkg/data pool (cache).

Strip modes (generator CLIs: --strip MODE; keys go to ctx['exclude'], which removes them as SEEDS only, never from
the outputs, and the CLI drops matching analog / window entries from ctx):
  tc085 : every congener with tc >= 0.85         tc070 : tc >= 0.70
  edit1 : every 1-edit neighbour (+-CH2, +-O, +-CH2O; any tc)   ("harder C3NP stratum" of the mmp-edit review)
np_pool rows that are not in the train_pkg pool are never seeds, so they need no strip.
"""
import argparse, json, pickle, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3np"
TD = ROOT / "results" / "train_pkg" / "data"
PFP = OUT / "work" / "pool_m2_2048.npy"
PPM = 10.0
AM = {"C": 12.0, "H": 1.00782503207, "O": 15.99491461956}
EDITS = {"CH2": {"C": 1, "H": 2}, "O": {"O": 1}, "CH2O": {"C": 1, "H": 2, "O": 1}}
MODES = {"tc085": "tc >= 0.85", "tc070": "tc >= 0.70", "edit1": "edit != ''"}
_G = {}


def _rd():
    if not _G:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        RDLogger.DisableLog("rdApp.*")
        _G.update(Chem=Chem, DS=DataStructs, CMF=CalcMolFormula,
                  mg=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    return _G


def _formula(f):
    import re
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f.split("+")[0].split("-")[0]):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def edit_label(tm, tform, m):
    """'+CH2' etc. if m = truth +- one of EDITS (exact formula difference) and the smaller is a substructure of the
    larger; else ''."""
    g = _rd()
    f = _formula(g["CMF"](m))
    keys = set(f) | set(tform)
    diff = {e: f.get(e, 0) - tform.get(e, 0) for e in keys}
    diff = {e: v for e, v in diff.items() if v}
    for name, d in EDITS.items():
        for sgn in (1, -1):
            if diff == {e: sgn * v for e, v in d.items()}:
                big, small = (m, tm) if sgn > 0 else (tm, m)
                try:
                    if big.HasSubstructMatch(small):
                        return ("+" if sgn > 0 else "-") + name
                except Exception:
                    return ""
    return ""


def fp_chunk(args):
    lo, hi = args
    g = _rd()
    with open(TD / "pool_smiles.txt", "rb") as fh:
        lines = fh.read().split(b"\n")[lo:hi]
    out = np.zeros((hi - lo, 256), np.uint8)
    for i, s in enumerate(lines):
        m = g["Chem"].MolFromSmiles(s.decode().rstrip("\r"))
        if m is not None:
            out[i] = np.packbits(g["mg"].GetFingerprintAsNumPy(m).astype(np.uint8))
    return lo, out


def pool_fps(workers):
    n = len(np.load(TD / "pool_key.npy"))
    if PFP.exists() and PFP.stat().st_mtime > (TD / "pool_smiles.txt").stat().st_mtime:
        F = np.load(PFP)
        if len(F) == n:
            return F
    t0 = time.time()
    F = np.zeros((n, 256), np.uint8)
    step = 20000
    with Pool(workers) as mp:
        for lo, blk in mp.imap_unordered(fp_chunk, [(i, min(i + step, n)) for i in range(0, n, step)]):
            F[lo:lo + len(blk)] = blk
    tmp = str(PFP) + ".tmp.npy"
    np.save(tmp, F)
    Path(tmp).replace(PFP)
    print(f"pool fingerprints {n:,} in {time.time() - t0:.0f}s", flush=True)
    return F


def mol_job(args):
    """Analog / window entries + pool edit neighbours for one truth."""
    mid, tsmi, emass, ents, pool_rows = args
    g = _rd()
    tm = g["Chem"].MolFromSmiles(tsmi)
    tf = g["mg"].GetFingerprint(tm)
    tform = _formula(g["CMF"](tm))
    rows = []
    for src, keys, smi in ents:
        m = g["Chem"].MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        tc = float(g["DS"].TanimotoSimilarity(tf, g["mg"].GetFingerprint(m)))
        ed = edit_label(tm, tform, m)
        if tc >= 0.70 or ed:
            raw = g["Chem"].MolToInchiKey(m)[:14]
            for k in sorted({k for k in keys if k} | {raw}):
                rows.append(dict(mid=mid, key=k, src=src, tc=round(tc, 4), edit=ed, smiles=smi))
    for row, key, smi in pool_rows:              # pool rows in the +-CH2/O/CH2O mass windows
        m = g["Chem"].MolFromSmiles(smi)
        if m is None:
            continue
        ed = edit_label(tm, tform, m)
        if ed:
            tc = float(g["DS"].TanimotoSimilarity(tf, g["mg"].GetFingerprint(m)))
            rows.append(dict(mid=mid, key=key, src="pool", tc=round(tc, 4), edit=ed, smiles=smi, row=int(row)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.workers = max(1, min(a.workers, 2))
    t0 = time.time()
    M = pd.read_parquet(OUT / "molecules.parquet")
    C = pickle.load(open(OUT / "context.pkl", "rb"))["mols"]
    pkey = np.load(TD / "pool_key.npy")
    pmass = np.load(TD / "pool_mass.npy")
    with open(TD / "pool_smiles.txt", "rb") as fh:
        psmi = fh.read().split(b"\n")
    F = pool_fps(a.workers)
    F64 = F.view(np.uint64)                       # (n, 32)
    pc = np.bitwise_count(F64).sum(1).astype(np.int32)
    g = _rd()
    rows, jobs = [], []
    for r in M.itertuples():
        tm = g["Chem"].MolFromSmiles(r.smiles)
        t = np.packbits(g["mg"].GetFingerprintAsNumPy(tm).astype(np.uint8)).view(np.uint64)
        tpc = int(np.bitwise_count(t).sum())
        hit = []
        for lo in range(0, len(F64), 100000):     # chunked Tanimoto vs the whole pool (~25 MB temp)
            inter = np.bitwise_count(F64[lo:lo + 100000] & t).sum(1)
            tc = inter / np.maximum(pc[lo:lo + 100000] + tpc - inter, 1)
            j = np.flatnonzero(tc >= 0.70)
            hit += [(lo + int(x), float(tc[x])) for x in j]
        for i, tc in hit:
            rows.append(dict(mid=r.mid, key=pkey[i].decode(), src="pool", tc=round(tc, 4), edit="",
                             smiles=psmi[i].decode().rstrip("\r"), row=i))
        prow = []
        for d in EDITS.values():
            dm = sum(AM[e] * v for e, v in d.items())
            for sgn in (1, -1):
                mt = r.exact_mass + sgn * dm
                lo, hi = np.searchsorted(pmass, [mt - mt * PPM * 1e-6, mt + mt * PPM * 1e-6])
                prow += [(i, pkey[i].decode(), psmi[i].decode().rstrip("\r")) for i in range(lo, hi)]
        c = C[r.mid]
        ents = [("analog", (x.get("ik14"), x.get("ckey")), x.get("smiles")) for x in c["analogs"]] + \
               [("window", (x.get("ik"), x.get("ckey")), x.get("smiles")) for x in c["window"]]
        jobs.append((r.mid, r.smiles, float(r.exact_mass), ents, prow))
    del psmi, C
    print(f"pool Tanimoto done {time.time() - t0:.0f}s; {len(rows):,} pool rows tc>=0.7", flush=True)
    with Pool(a.workers) as mp:
        for rr in mp.imap(mol_job, jobs, chunksize=4):
            rows += rr
    S = pd.DataFrame(rows)
    S["row"] = S["row"].fillna(-1).astype(np.int64)
    # merge pool duplicates (tc row + edit row for the same structure)
    S = (S.sort_values(["mid", "src", "key", "edit"], ascending=[True, True, True, False])
          .groupby(["mid", "src", "key"], as_index=False)
          .agg(tc=("tc", "max"), edit=("edit", "first"), smiles=("smiles", "first"), row=("row", "max")))
    S.to_parquet(OUT / "strip.parquet", index=False)
    # strata
    st = []
    for r in M.itertuples():
        d = S[S.mid == r.mid]
        best = float(d.tc.max()) if len(d) else 0.0
        st.append(dict(mid=r.mid, subset=r.subset, best_tc_ge070=best,
                       nb1=bool(d.edit.isin(["+CH2", "-CH2", "+O", "-O"]).any()),
                       nb2=bool(d.edit.isin(["+CH2O", "-CH2O"]).any()),
                       n_tc070=int(d.key[d.tc >= 0.70].nunique()), n_tc085=int(d.key[d.tc >= 0.85].nunique()),
                       n_edit1=int(d.key[d.edit != ""].nunique())))
    T = pd.DataFrame(st)
    R = pd.read_parquet(OUT / "reach.parquet")[["mid", "pool_nn_tan", "ana_max_tan", "win_max_tan"]]
    T = T.merge(R, on="mid", how="left")
    T["best_tc"] = T[["best_tc_ge070", "pool_nn_tan", "ana_max_tan", "win_max_tan"]].max(1)
    T["tc_band"] = pd.cut(T.best_tc, [-1, 0.5, 0.7, 0.85, 1.01], right=False,
                          labels=["<0.5", "0.5-0.7", "0.7-0.85", ">=0.85"]).astype(str)
    T.to_parquet(OUT / "strata.parquet", index=False)
    summ = dict(n_rows=len(S), seconds=round(time.time() - t0),
                rows_by_src=S.src.value_counts().to_dict(),
                edit_rows=S[S.edit != ""].edit.value_counts().to_dict(),
                by_subset={k: dict(n=len(d), nb1=round(float(d.nb1.mean()), 3), nb2=round(float(d.nb2.mean()), 3),
                                   nb1_or_nb2=round(float((d.nb1 | d.nb2).mean()), 3),
                                   tc_band=d.tc_band.value_counts().to_dict(),
                                   mean_n_tc070=round(float(d.n_tc070.mean()), 1),
                                   mean_n_tc085=round(float(d.n_tc085.mean()), 1),
                                   mean_n_edit1=round(float(d.n_edit1.mean()), 1))
                           for k, d in T.groupby("subset")})
    json.dump(summ, open(OUT / "strip_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def strip_keys(mode):
    """{mid: frozenset(keys)} for a strip mode (bench-only; reads the oracle table)."""
    if mode in (None, "", "none"):
        return {}
    S = pd.read_parquet(OUT / "strip.parquet")
    S = S.query(MODES[mode])
    return S.groupby("mid").key.apply(frozenset).to_dict()


def apply_strip(ctx, keys):
    """Drop analog / window entries whose key is stripped and pass the keys as ctx['exclude'] (seed-only)."""
    if not keys:
        return ctx
    ctx = dict(ctx)
    ctx["analogs"] = [x for x in ctx.get("analogs") or [] if not ({x.get("ik14"), x.get("ckey")} & keys)]
    ctx["window"] = [x for x in ctx.get("window") or [] if not ({x.get("ik"), x.get("ckey")} & keys)]
    ctx["exclude"] = frozenset(keys)
    return ctx


if __name__ == "__main__":
    main()

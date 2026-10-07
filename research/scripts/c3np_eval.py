"""Score a ranked candidate file on the C3NP bench (results/c3np/molecules.parquet).

    python research/scripts/c3np_eval.py CANDIDATES.json [--name X] [--workers 4]
    python research/scripts/c3np_eval.py --selftest

CANDIDATES.json = {mid: [smiles, ...]} ranked best first. Each candidate is canonicalised to the competition key
(RDKit TautomerEnumerator().Canonicalize -> MolToInchiKey[:14]); duplicates (same key) are removed keeping the first;
an unparsable SMILES keeps its rank position but can never be correct. A molecule's answer is correct at rank r if
the candidate's canonical key (or its raw InChIKey14) is in molecules.parquet `correct` (truth ik14 + truth ckey +
bench aliases + verified tautomer aliases). Reciprocal rank counts only r <= 25. Molecules absent from the file
score 0. Reports MRR@25 (with a 99.9% bootstrap CI, 20k resamples), top-1, hit@10, hit@25, mean top-1 Tanimoto and
mean max-Tanimoto@25 to the truth (RDKit Morgan r2, 2048 bits; molecules without candidates count 0), close-match
(top-1 Tanimoto >= 0.675) and meaningful-match (>= 0.4) rates; per subset (npex / s3pc / s3none) and overall.
Writes results/c3np/eval_<name>.json and eval_<name>_per_mol.csv.

r2 additions:
  --only_present  score only the mids present in the file (smoke slices; otherwise absent mids score 0).
  Congener strata (when results/c3np/strata.parquet exists, built by c3np_strip.py; oracle, bench-only):
  `by_stratum` reports MRR for nb1 yes/no (a +-CH2 / +-O neighbour of the truth with a substructure relation exists in
  the seed sources), nb12 (also +-CH2O) and best seed-source Tanimoto band, plus `projection_nb1` =
  p * MRR(nb1=yes) + (1 - p) * MRR(nb1=no) for assumed shares p of hidden Class-3 molecules that have such a
  neighbour (INFERENCE: p is not measured for the hidden test). Bench MRR is NOT a Class-3 forecast: read it through
  these strata, and through the generator CLIs' --strip {tc085,tc070,edit1} modes, which remove the truth's
  congeners from the seed sources.
"""
import argparse, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3np"
CACHES = [OUT / "work" / "canon_cache.pkl", ROOT / "results/bench/cache/canon.pkl"]
EVAL_CACHE = OUT / "work" / "eval_canon_cache.pkl"
_G = {}


def _rd():
    if not _G:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _G.update(Chem=Chem, DS=DataStructs, te=rdMolStandardize.TautomerEnumerator(),
                  mg=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    return _G


def keys_one(smi):
    """(canonical key, raw ik14) or (None, None)."""
    g = _rd()
    try:
        m = g["Chem"].MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if m is None:
            return None, None
        raw = g["Chem"].MolToInchiKey(m)[:14]
        return g["Chem"].MolToInchiKey(g["te"].Canonicalize(m))[:14], raw
    except Exception:
        return None, None


def raw_one(smi):
    g = _rd()
    try:
        m = g["Chem"].MolFromSmiles(smi)
        return g["Chem"].MolToInchiKey(m)[:14] if m is not None else None
    except Exception:
        return None


def tan_one(args):
    """(truth_smiles, [smiles]) -> list of Tanimoto (nan for unparsable)."""
    t, smis = args
    g = _rd()
    tm = g["Chem"].MolFromSmiles(t)
    tf = g["mg"].GetFingerprint(tm)
    out = []
    for s in smis:
        m = g["Chem"].MolFromSmiles(s) if isinstance(s, str) and s else None
        out.append(float(g["DS"].TanimotoSimilarity(tf, g["mg"].GetFingerprint(m))) if m is not None else np.nan)
    return out


def key_many(smiles, workers, use_disk=True):
    """smiles -> (ckey, raw). Canonical keys come from the build caches when present; raw keys are always computed
    (cheap) for the not-cached ones only via keys_one."""
    C = {}
    if use_disk and EVAL_CACHE.exists():
        C = pickle.load(open(EVAL_CACHE, "rb"))
    need = sorted({s for s in smiles if isinstance(s, str)} - set(C))
    if need:
        t0 = time.time()
        pre = {}
        if use_disk:   # canonical keys already computed by the build (same recipe); raw keys are recomputed
            for f in CACHES:
                if f.exists():
                    d = pickle.load(open(f, "rb"))
                    pre.update({s: d[s] for s in need if s in d and s not in pre and d[s] is not None})
                    del d
        rest = [s for s in need if s not in pre]
        hit = list(pre)
        with Pool(workers) as mp:
            C.update(zip(rest, mp.map(keys_one, rest, chunksize=16)))
            C.update((s, (pre[s], r)) for s, r in zip(hit, mp.map(raw_one, hit, chunksize=64)))
        if use_disk:
            EVAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(C, open(EVAL_CACHE, "wb"), protocol=4)
        print(f"keyed {len(need):,} SMILES ({len(hit):,} canonical keys from build caches) in {time.time() - t0:.0f}s", flush=True)
    return C


def rank_of(cands, K, ok):
    """1-based rank (after canonical de-dup) of the first correct candidate within the whole list, 0 if none;
    also returns the de-duplicated list of SMILES."""
    seen, dd = set(), []
    for s in cands:
        ck, raw = K.get(s, (None, None)) if isinstance(s, str) else (None, None)
        if ck is not None:
            if ck in seen:
                continue
            seen.add(ck)
        dd.append((s, ck, raw))
    for i, (_, ck, raw) in enumerate(dd):
        if (ck is not None and ck in ok) or (raw is not None and raw in ok):
            return i + 1, [x[0] for x in dd]
    return 0, [x[0] for x in dd]


def boot_ci(v, n=20000, alpha=0.001, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(0, n, 1000):
        idx = rng.integers(0, len(v), (min(1000, n - i), len(v)))
        means[i:i + len(idx)] = v[idx].mean(1)
    return [round(float(np.quantile(means, alpha / 2)), 4), round(float(np.quantile(means, 1 - alpha / 2)), 4)]


def summarize(d):
    rr = d.rr.values
    return dict(n=len(d), mrr25=round(float(rr.mean()), 4), mrr25_ci999=boot_ci(rr),
                top1=round(float((d["rank"] == 1).mean()), 4),
                hit10=round(float(((d["rank"] > 0) & (d["rank"] <= 10)).mean()), 4),
                hit25=round(float(((d["rank"] > 0) & (d["rank"] <= 25)).mean()), 4),
                found_any_rank=round(float((d["rank"] > 0).mean()), 4),
                top1_tan_mean=round(float(d.top1_tan.fillna(0).mean()), 4),
                maxtan25_mean=round(float(d.maxtan25.fillna(0).mean()), 4),
                close_match_rate=round(float((d.top1_tan.fillna(0) >= 0.675).mean()), 4),
                meaningful_match_rate=round(float((d.top1_tan.fillna(0) >= 0.4).mean()), 4),
                n_with_candidates=int((d.n_cand > 0).sum()), median_n_cand=float(d.n_cand.median()))


def evaluate(cands, name, workers=4, M=None, write=True, use_disk=True, only_present=False):
    if M is None:
        M = pd.read_parquet(OUT / "molecules.parquet")
    if only_present:
        M = M[M.mid.isin(set(cands))].reset_index(drop=True)
    extra = sorted(set(cands) - set(M.mid))
    if extra:
        print(f"warning: {len(extra)} mids not in the bench are ignored (e.g. {extra[:3]})")
    allsmi = [s for m in M.mid for s in cands.get(m, [])]
    K = key_many(allsmi, workers, use_disk)
    rows, tjobs = [], []
    for r in M.itertuples():
        ok = set(r.correct.split(";"))
        rank, dd = rank_of(cands.get(r.mid, []), K, ok)
        rows.append(dict(mid=r.mid, subset=r.subset, rank=rank, rr=1.0 / rank if 0 < rank <= 25 else 0.0,
                         n_cand=len(dd)))
        tjobs.append((r.smiles, dd[:25]))
    with Pool(workers) as mp:
        tans = mp.map(tan_one, tjobs, chunksize=8)
    D = pd.DataFrame(rows)
    D["top1_tan"] = [t[0] if t else np.nan for t in tans]
    D["maxtan25"] = [np.nanmax(t) if t and not np.all(np.isnan(t)) else np.nan for t in tans]
    res = dict(name=name, n_molecules=len(M), n_missing=int((D.n_cand == 0).sum()), overall=summarize(D),
               by_subset={k: summarize(d) for k, d in D.groupby("subset")})
    st = OUT / "strata.parquet"
    if st.exists() and "selftest" not in set(D.subset):
        T = pd.read_parquet(st)[["mid", "nb1", "nb2", "tc_band"]]
        D = D.merge(T, on="mid", how="left")
        D["nb12"] = D.nb1.fillna(False) | D.nb2.fillna(False)
        short = lambda d: dict(n=len(d), mrr25=round(float(d.rr.mean()), 4) if len(d) else None,
                               hit25=round(float(((d["rank"] > 0) & (d["rank"] <= 25)).mean()), 4) if len(d) else None)
        bs = {}
        for col in ("nb1", "nb12"):
            bs[col] = {("yes" if v else "no"): short(d) for v, d in D.groupby(D[col].fillna(False).astype(bool))}
            bs[col + "_by_subset"] = {k: {("yes" if v else "no"): short(e) for v, e in d.groupby(d[col].fillna(False).astype(bool))}
                                      for k, d in D.groupby("subset")}
        bs["tc_band"] = {k: short(d) for k, d in D.groupby("tc_band")}
        res["by_stratum"] = bs
        y, n = bs["nb1"].get("yes", {}).get("mrr25"), bs["nb1"].get("no", {}).get("mrr25")
        if y is not None and n is not None:
            res["projection_nb1"] = {f"p={p}": round(p * y + (1 - p) * n, 4) for p in (0.1, 0.25, 0.5, 0.75)}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT / f"eval_{name}.json", "w"), indent=1)
        D.to_csv(OUT / f"eval_{name}_per_mol.csv", index=False)
    return res


def selftest(workers):
    truth = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"          # caffeine
    hp, pyr = "O=C1CCCCC1", "OC1=CCCCC1"            # cyclohexanone / its enol (keto-enol tautomers)
    fill = ["C" * n for n in range(1, 41)]            # alkanes: distinct keys, never correct
    K = {s: keys_one(s) for s in [truth, hp, pyr] + fill}
    assert K[hp][0] == K[pyr][0] and K[hp][1] != K[pyr][1], "tautomer pair must share the canonical key only"
    ck, raw = K[truth]
    hk = K[hp]
    mols = [("r1", truth, [truth] + fill[:5], 1.0), ("r3", truth, fill[:2] + [truth], 1 / 3),
            ("r25", truth, fill[:24] + [truth], 1 / 25), ("r26", truth, fill[:25] + [truth], 0.0),
            ("absent", truth, fill[:30], 0.0), ("taut", hp, [fill[0], pyr, hp], 0.5),
            ("dups", truth, ["CCO", "OCC", "C(C)O", "CCN", truth], 1 / 3),
            ("bad_smiles", truth, ["not_a_smiles", truth], 0.5), ("missing", truth, None, 0.0)]
    M = pd.DataFrame([dict(mid=m, subset="selftest", smiles=t, correct=";".join(sorted({K[t][0], K[t][1]})))
                      for m, t, _, _ in mols])
    cands = {m: c for m, _, c, _ in mols if c is not None}
    res = evaluate(cands, "selftest", workers=workers, M=M, write=False, use_disk=False)
    D = pd.DataFrame([dict(mid=m, want=w) for m, _, _, w in mols])
    # recompute per-molecule rr through the same path
    Kc = key_many([s for v in cands.values() for s in v], workers, use_disk=False)
    got = {r.mid: rank_of(cands.get(r.mid, []), Kc, set(r.correct.split(";")))[0] for r in M.itertuples()}
    for m, _, _, w in mols:
        rk = got[m]
        rr = 1.0 / rk if 0 < rk <= 25 else 0.0
        assert abs(rr - w) < 1e-12, (m, rk, rr, w)
        print(f"  {m:10s} rank {rk:2d} rr {rr:.4f} (expected {w:.4f}) ok")
    want = float(D.want.mean())
    assert abs(res["overall"]["mrr25"] - round(want, 4)) < 1e-9, (res["overall"]["mrr25"], want)
    assert res["overall"]["top1"] == round(1 / len(mols), 4)
    assert res["overall"]["hit25"] == round(6 / len(mols), 4)
    print(f"selftest passed: MRR@25 {res['overall']['mrr25']} (expected {want:.4f}), top1 {res['overall']['top1']}, "
          f"hit25 {res['overall']['hit25']}, mean top-1 Tanimoto {res['overall']['top1_tan_mean']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only_present", action="store_true", help="score only the mids in the file (smoke slices)")
    a = ap.parse_args()
    a.workers = min(a.workers, 4)
    if a.selftest:
        selftest(a.workers)
        return
    if not a.candidates:
        ap.error("CANDIDATES.json is required (or --selftest)")
    name = a.name or Path(a.candidates).stem
    res = evaluate(json.load(open(a.candidates)), name, a.workers, only_present=a.only_present)
    print(json.dumps({k: res[k] for k in ("overall", "by_subset", "by_stratum", "projection_nb1") if k in res}, indent=1))
    print(f"-> {OUT / f'eval_{name}.json'}")


if __name__ == "__main__":
    main()

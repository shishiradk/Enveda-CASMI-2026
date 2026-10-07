"""EXP-015: spectrum representation ablation + kNN/two-tower complementarity.
Design (locked): research/analysis/exp015_spectrum_representation_design.md

    python research/scripts/exp015_spectrum_representation.py build   # fresh holdout, aliases, pools, leakage, manifest
    python research/scripts/exp015_spectrum_representation.py peaks   # raw peaks for R_noE180 + all queries
    python research/scripts/exp015_spectrum_representation.py run     # variants, DEV selection, combos, oracle, fusion
"""
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402

TRAIN = (ROOT / "train.parquet").as_posix()
E11, E12, E13 = (ROOT / "results" / d for d in ("exp011_class2_proxy", "exp012_coconut", "exp013_np_forensics"))
O = ROOT / "research" / "analysis" / "exp015"
R = ROOT / "results" / "exp015"
SEED = 20260927
N_PRIMARY, Q_PER_MOL, K_NN, K_PRE = 1000, 4, 20, 500
TC_NEAR, TC_UNREL = 0.50, 0.30
CURRENT = {"bin_w": 0.1, "nl": True, "inten": "sqrt"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ============================================================================ shared structure helpers
def coconut():
    C = pd.read_parquet(E12 / "coconut_db.parquet")
    mask = C["parse_ok"].values
    C = C[mask].reset_index(drop=True)
    cfp = np.load(E12 / "coconut_fp_packed.npy")[mask]  # parse_ok-filtered: row i of C <-> row i of cfp
    C["row"] = np.arange(len(C))
    return C, cfp


def rd_tools():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()

    def tk(smi):
        try:
            return Chem.MolToInchiKey(te.Canonicalize(Chem.MolFromSmiles(smi)))[:14]
        except Exception:
            return None

    def skeleton(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        rw = Chem.RWMol(m)
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
        for at in rw.GetAtoms():
            at.SetFormalCharge(0); at.SetIsAromatic(False); at.SetNoImplicit(True); at.SetNumExplicitHs(0)
        return Chem.MolToSmiles(rw, canonical=True)
    return tk, skeleton


def aliases_in(space, key_of_target, smi_t, mass_t, form_t, parent_t, cache):
    """Equivalent keys of a target within `space` (DataFrame ik/smiles/mass/formula/parent, sorted by mass):
    same parent key, or same tautomer-canonical InChIKey14 among same-formula entries within +/-2 mDa."""
    tk, skeleton = cache["tools"]
    out = set(space.loc[space["parent"] == parent_t, "ik"]) if parent_t else set()
    m = space["mass"].values
    lo, hi = np.searchsorted(m, mass_t - 0.002), np.searchsorted(m, mass_t + 0.002, side="right")
    near = space.iloc[lo:hi]
    near = near[(near["formula"] == form_t) & (near["ik"] != key_of_target)]
    if len(near):
        sk = skeleton(smi_t)
        cand = [(n, s) for n, s in zip(near["ik"], near["smiles"]) if sk and skeleton(s) == sk]
        if cand:
            kt = tk(smi_t)
            out |= {n for n, s in cand if kt and tk(s) == kt}
    out.discard(key_of_target)
    return out


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ============================================================================ build
def stage_build():
    t0 = time.time()
    O.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='4GB'")
    meta = con.execute(f"""SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm, ingest_lib AS lib
                           FROM read_parquet('{TRAIN}', file_row_number=true)""").df()
    tr = pd.read_parquet(E11 / "training.parquet")
    ref = tr[(tr["split"] == "train") & (tr["lib"] != "enveda-180")]
    ref_rids = np.sort(ref["rid"].values)
    ref_mols, tt_mols = set(ref["ik"]), set(tr["ik"])
    U8 = pd.read_parquet(E.UNIVERSE)
    U8 = U8[U8["parse_ok"]].sort_values("mass").reset_index(drop=True)
    par8 = dict(zip(U8["ik"], U8["parent"]))
    C, _ = coconut()
    Cs = C.sort_values("mass").reset_index(drop=True)
    D = set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
    prev = set(pd.read_parquet(E11 / "targets.parquet")["ik"])  # every EXP-011/012/013 evaluation target
    m10 = meta[meta["adduct"].isin(E.AD10)]
    elig = sorted((D & set(C["ik"]) & set(m10["ik"])) - prev)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(np.array(elig), N_PRIMARY, replace=False).tolist())
    log(f"eligible fresh targets {len(elig)}; sampled {len(pick)}")

    cache = {"tools": rd_tools()}
    u8 = U8.set_index("ik")
    ref_par = {par8.get(k) for k in ref_mols} - {None}
    tt_par = {par8.get(k) for k in tt_mols} - {None}
    rows, tstat = [], []
    q_all = m10[m10["ik"].isin(pick)].sample(frac=1.0, random_state=SEED).groupby("ik").head(Q_PER_MOL)
    # K2b: byte-identical query spectra in the reference library
    con.register("qr", pd.DataFrame({"rid": q_all["rid"].values}))
    con.register("rr", pd.DataFrame({"rid": ref_rids}))
    hq = con.execute(f"""SELECT file_row_number AS rid, hash(ms2_mzs, ms2_normalized_intensities) AS h FROM read_parquet('{TRAIN}', file_row_number=true)
                         WHERE file_row_number IN (SELECT rid FROM qr)""").df()
    hr = set(con.execute(f"""SELECT hash(ms2_mzs, ms2_normalized_intensities) AS h FROM read_parquet('{TRAIN}', file_row_number=true)
                             WHERE file_row_number IN (SELECT rid FROM rr)""").df()["h"])
    dup_q = set(hq.loc[hq["h"].isin(hr), "rid"])
    for ik in pick:
        r = u8.loc[ik]
        al8 = aliases_in(U8, ik, r["smiles"], r["mass"], r["formula"], r["parent"], cache)
        alC = aliases_in(Cs, ik, r["smiles"], r["mass"], r["formula"], r["parent"], cache)
        okC = ({ik} & set(C["ik"])) | alC
        leak = {"target_in_ref": ik in ref_mols, "alias_in_ref": bool(al8 & ref_mols),
                "parent_in_ref": r["parent"] in ref_par, "target_or_alias_in_twotower": bool(({ik} | al8) & tt_mols),
                "parent_in_twotower": r["parent"] in tt_par, "previous_target": ik in prev}
        qs = q_all[(q_all["ik"] == ik) & ~q_all["rid"].isin(dup_q)]
        status = "ok" if not any(leak.values()) and len(qs) else ("no_query_after_dedup" if not len(qs) else "leak:" + ",".join(k for k, v in leak.items() if v))
        M = float(np.median(qs["pm"] - qs["adduct"].map(E.AD10))) if len(qs) else float("nan")
        lo, hi = (np.searchsorted(Cs["mass"].values, M * (1 - 5e-6)), np.searchsorted(Cs["mass"].values, M * (1 + 5e-6), side="right")) if len(qs) else (0, 0)
        pool = Cs["ik"].values[lo:hi]
        covered = bool(set(pool) & okC)
        tstat.append({"ik": ik, "status": status, "M": M, "pool": len(pool), "covered": covered,
                      "coconut_ok_keys": sorted(okC), "train_aliases": sorted(al8), **leak})
        for q in qs.itertuples():
            rows.append({"target_id": f"P{len(tstat):04d}", "spectrum_id": int(q.rid), "inchikey14": ik, "adduct": q.adduct,
                         "library": q.lib, "precursor_mz": q.pm, "neutral_mass_spectrum": q.pm - E.AD10[q.adduct],
                         "neutral_mass_molecule": M, "candidate_count": len(pool), "truth_in_pool": covered,
                         "candgen": "COCONUT-2026-09 InChIKey14, +/-5 ppm of molecule median neutral mass",
                         "leakage_status": status, "appeared_in_exp011_013": ik in prev})
    man = pd.DataFrame(rows)
    man.to_csv(O / "holdout_manifest.csv", index=False)
    T = pd.DataFrame(tstat)
    T.to_parquet(R / "primary_targets.parquet", index=False)
    info = {"eligible": len(elig), "sampled": len(pick), "status_counts": T["status"].value_counts().to_dict(),
            "query_spectra_dropped_duplicate_in_ref": len(dup_q & set(q_all["rid"])),
            "ok_targets": int((T["status"] == "ok").sum()), "ok_targets_covered": int(((T["status"] == "ok") & T["covered"]).sum()),
            "ok_query_spectra": int((man["leakage_status"] == "ok").sum()),
            "query_library_mix": man.loc[man["leakage_status"] == "ok", "library"].value_counts().to_dict(),
            "R_noE180": {"spectra": int(len(ref_rids)), "molecules": len(ref_mols), "sha256_sorted_rids": sha(ref_rids)},
            "sec": round(time.time() - t0, 1)}
    np.save(R / "ref_rids.npy", ref_rids)
    json.dump(info, open(R / "build.json", "w"), indent=1)
    log(json.dumps(info, indent=1))


# ============================================================================ peaks
def stage_peaks():
    import pyarrow as pa
    import pyarrow.compute as pc
    t0 = time.time()
    ref_rids = np.load(R / "ref_rids.npy")
    man = pd.read_csv(O / "holdout_manifest.csv")
    prim = man.loc[man["leakage_status"] == "ok", "spectrum_id"].values
    dev = pd.read_parquet(E11 / "queries.parquet").query("pop == 'T1_np'")["rid"].values
    rids = np.unique(np.concatenate([ref_rids, prim, dev]).astype(np.int64))
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='4GB'")
    out = {k: [] for k in ("rid", "pm", "adduct", "len", "mz", "it")}
    for i in range(0, len(rids), 100_000):
        con.register("want", pd.DataFrame({"rid": rids[i:i + 100_000]}))
        t = con.execute(f"""SELECT file_row_number AS rid, precursor_mz AS pm, adduct, ms2_mzs, ms2_normalized_intensities
                            FROM read_parquet('{TRAIN}', file_row_number=true) WHERE file_row_number IN (SELECT rid FROM want)
                            ORDER BY rid""").arrow()
        if isinstance(t, pa.RecordBatchReader):
            t = t.read_all()
        out["rid"].append(t["rid"].to_numpy()); out["pm"].append(t["pm"].to_numpy())
        out["adduct"].append(np.array(t["adduct"].to_pylist(), dtype=object))
        out["len"].append(pc.list_value_length(t["ms2_mzs"]).fill_null(0).to_numpy())
        out["mz"].append(pc.list_flatten(t["ms2_mzs"]).to_numpy(zero_copy_only=False))
        out["it"].append(pc.list_flatten(t["ms2_normalized_intensities"]).to_numpy(zero_copy_only=False))
        log(f"  peaks {min(i + 100_000, len(rids))}/{len(rids)}")
    arr = {k: np.concatenate(v) for k, v in out.items()}
    assert np.array_equal(arr["rid"], rids)
    np.savez(R / "peaks.npz", rid=arr["rid"], pm=arr["pm"], adduct=arr["adduct"].astype(str), len=arr["len"],
             mz=arr["mz"], it=arr["it"])
    log(f"saved peaks for {len(rids)} spectra ({arr['mz'].size} peaks) in {time.time() - t0:.0f}s")


class Peaks:
    def __init__(self):
        z = np.load(R / "peaks.npz", allow_pickle=False)
        self.rid, self.pm, self.adduct, self.len = z["rid"], z["pm"], z["adduct"], z["len"]
        self.mz, self.it = z["mz"], z["it"]
        self.off = np.r_[0, np.cumsum(self.len)]
        self.pos = {int(r): i for i, r in enumerate(self.rid)}

    def idx(self, rids):
        return np.array([self.pos[int(r)] for r in rids])


def featurize(P, idx, bin_w, nl, inten, topk=150, mz_max=1500.0, nl_max=500.0):
    """Generalised EXP-011 featurize (identical arithmetic at bin 0.1 / nl True / sqrt)."""
    from scipy import sparse
    n_frag, n_nl = int(mz_max / bin_w), int(nl_max / bin_w)
    lens = P.len[idx]
    starts = P.off[idx]
    gather = np.concatenate([np.arange(s, s + l) for s, l in zip(starts, lens)]) if len(idx) else np.array([], int)
    row = np.repeat(np.arange(len(idx)), lens)
    mz, it = P.mz[gather], P.it[gather]
    pm = P.pm[idx][row]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    row, mz, it, pm = row[ok], mz[ok], it[ok], pm[ok]
    order = np.lexsort((-it, row))
    row, mz, it, pm = row[order], mz[order], it[order], pm[order]
    st = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    rank = np.arange(len(row)) - np.repeat(st, np.diff(np.r_[st, len(row)]))
    keep = rank < topk
    row, mz, it, pm = row[keep], mz[keep], it[keep], pm[keep]
    w = {"sqrt": np.sqrt(it), "linear": it, "log": np.log1p(1000.0 * it)}[inten].astype(np.float32)
    fr = (mz / bin_w).astype(np.int64)
    m1 = (fr >= 0) & (fr < n_frag)
    rows, cols, vals = [row[m1]], [fr[m1]], [w[m1]]
    if nl:
        nlv = pm - mz
        nb = (nlv / bin_w).astype(np.int64)
        m2 = (nlv > 0.5) & (nb < n_nl)
        rows.append(row[m2]); cols.append(n_frag + nb[m2]); vals.append(w[m2])
    X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(idx), n_frag + (n_nl if nl else 0)), dtype=np.float32)
    X.sum_duplicates()
    nrm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    nrm[nrm == 0] = 1.0
    return sparse.diags(1.0 / nrm).dot(X).tocsr().astype(np.float32)


# ============================================================================ parallel neighbour search
_W = {}


def _init(ref_path, neg_path):
    from scipy import sparse
    _W["RT"] = sparse.load_npz(ref_path).T.tocsr()
    _W["neg"] = np.load(neg_path)


def _topk(args):
    Xq, qneg, k = args
    S = (Xq @ _W["RT"]).toarray()
    S[qneg[:, None] != _W["neg"][None, :]] = -1.0
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, axis=1)
    o = np.argsort(-sims, axis=1, kind="stable")
    return np.take_along_axis(idx, o, axis=1), np.take_along_axis(sims, o, axis=1)


def neighbours(Xq, qneg, Xr, rneg, k, tag, workers=4):
    from scipy import sparse
    rp, npth = R / f"_ref_{tag}.npz", R / "_ref_neg.npy"
    sparse.save_npz(rp, Xr, compressed=False)
    np.save(npth, rneg)
    jobs = [(Xq[i:i + 64], qneg[i:i + 64], k) for i in range(0, Xq.shape[0], 64)]
    with Pool(workers, initializer=_init, initargs=(str(rp), str(npth))) as pool:
        res = pool.map(_topk, jobs)
    rp.unlink()
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def _mc_init():
    from matchms.similarity import ModifiedCosineGreedy
    sys.path.insert(0, str(ROOT / "src"))
    from spectra.spectrum_io import make_spectrum
    for k in ("mz", "it", "off", "pm"):
        _W[k] = np.load(R / f"peaks_{k}.npy", mmap_mode="r")
    _W["sim"], _W["mk"] = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0), make_spectrum


def _mc(args):
    """ModifiedCosine (V0 scorer) between one query row and its prefiltered reference rows (peaks memory-mapped)."""
    qrow, rrows = args
    mz, it, off, pm, mk, sim = _W["mz"], _W["it"], _W["off"], _W["pm"], _W["mk"], _W["sim"]
    spec = lambda r: mk(np.asarray(mz[off[r]:off[r + 1]]), np.asarray(it[off[r]:off[r + 1]]), float(pm[r]))
    qs = spec(qrow)
    return np.array([float(sim.pair(qs, spec(r))["score"]) for r in rrows], dtype=np.float32)


# ============================================================================ populations, pools, ranking
class Population:
    def __init__(self, name, targets, queries, okkeys, M, C, cfp):
        """targets: list of ik; queries: DataFrame(rid, ik, adduct); okkeys: ik -> set of correct COCONUT keys."""
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")
        self.name, self.queries = name, queries.reset_index(drop=True)
        Cs = C.sort_values("mass").reset_index(drop=True)
        cm = Cs["mass"].values
        smi = dict(zip(C["ik"], C["smiles"]))
        self.cfp = cfp
        props = {}

        def prop(k):
            if k not in props:
                m = Chem.MolFromSmiles(smi[k])
                try:
                    sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
                except Exception:
                    sc = None
                props[k] = (CalcMolFormula(m), sc)
            return props[k]
        self.pools = {}
        for ik in targets:
            lo, hi = np.searchsorted(cm, M[ik] * (1 - 5e-6)), np.searchsorted(cm, M[ik] * (1 + 5e-6), side="right")
            p = Cs.iloc[lo:hi]
            ok = np.array([k in okkeys[ik] for k in p["ik"]])
            if not ok.any():
                continue
            ti = int(np.flatnonzero(ok)[0])
            tfp = cfp[p["row"].values[ti]]
            inter = np.bitwise_count(np.bitwise_and(cfp[p["row"].values], tfp)).sum(1)
            union = np.bitwise_count(np.bitwise_or(cfp[p["row"].values], tfp)).sum(1)
            tc = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            tf, tsc = prop(p["ik"].values[ti])
            fp_, sc_ = zip(*[prop(k) for k in p["ik"]])
            same_f = np.array([f == tf for f in fp_]) & ~ok
            same_sc = np.array([s is not None and s == tsc for s in sc_]) & ~ok
            self.pools[ik] = {"iks": p["ik"].values, "rows": p["row"].values, "ok": ok, "same_formula": same_f,
                              "same_scaffold": same_sc, "tc": tc,
                              "bits": np.unpackbits(cfp[p["row"].values], axis=1).astype(np.float32)}
        self.n_targets_all = len(targets)
        self.pool_hash = hashlib.sha256("|".join(f"{k}:{','.join(v['iks'])}" for k, v in sorted(self.pools.items())).encode()).hexdigest()
        self.query_hash = hashlib.sha256(",".join(map(str, sorted(self.queries["rid"]))).encode()).hexdigest()

    def category(self, ik, i):
        P = self.pools[ik]
        if P["same_formula"][i]:
            return "near_isomer" if P["tc"][i] >= TC_NEAR else "distant_isomer"
        if P["same_scaffold"][i]:
            return "same_scaffold"
        if P["tc"][i] >= TC_NEAR:
            return "similar_diff_formula"
        return "unrelated" if P["tc"][i] < TC_UNREL else "other"


def rank_stats(scores, ok):
    best = scores[ok].max()
    others = scores[~ok]
    better, tied = int(np.sum(others > best)), int(np.sum(others == best))
    ranks = np.arange(better + 1, better + tied + 2)
    return {"rank": float(ranks.mean()), "rr": float(np.mean(1 / ranks)),
            "rr25": float(np.mean(np.where(ranks <= 25, 1 / ranks, 0))),
            **{f"r@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10, 50)}}


def evaluate(pop, scores_by_target, tag):
    rows = []
    for ik, P in pop.pools.items():
        if ik not in scores_by_target:
            continue
        s = scores_by_target[ik]
        st = rank_stats(s, P["ok"])
        best = s[P["ok"]].max()
        sf = P["same_formula"]
        wrong = np.flatnonzero(~P["ok"])
        top1 = wrong[np.argmax(s[wrong])] if len(wrong) else None
        rows.append({"population": pop.name, "variant": tag, "ik": ik, "pool": len(P["iks"]), **st,
                     "has_same_formula": bool(sf.any()),
                     "isomer_error_rate": float(np.mean(s[sf] > best)) if sf.any() else np.nan,
                     "top1_wrong_category": pop.category(ik, top1) if (top1 is not None and st["rank"] > 1) else "correct"})
    return pd.DataFrame(rows)


def summarize(df, n_all):
    sf = df[df["has_same_formula"]]
    return {"n_ranked": len(df), "n_targets": n_all, "MRR": df["rr"].mean(), "MRR_uncond": df["rr"].sum() / n_all,
            "MRR@25": df["rr25"].mean(), **{f"R@{k}": df[f"r@{k}"].mean() for k in (1, 5, 10, 50)},
            "sameformula_n": len(sf), "sameformula_MRR": sf["rr"].mean(), "sameformula_R@1": sf["r@1"].mean(),
            "sameformula_R@10": sf["r@10"].mean(), "isomer_error_rate": sf["isomer_error_rate"].mean()}


def paired(a, b, col, seed=SEED):
    m = a.merge(b, on="ik", suffixes=("_a", "_b"))
    d = (m[f"{col}_a"] - m[f"{col}_b"]).values
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def transitions(base, var):
    m = base.merge(var, on="ik", suffixes=("_b", "_v"))
    cb, cv = m["r@1_b"] >= 0.5, m["r@1_v"] >= 0.5
    t = {"wrong_to_correct": int((~cb & cv).sum()), "correct_to_wrong": int((cb & ~cv).sum()),
         "both_correct": int((cb & cv).sum()), "both_wrong": int((~cb & ~cv).sum())}
    by = {}
    for c, g in m.groupby("top1_wrong_category_b"):
        by[c] = {"n": len(g), "mean_rank_delta": float((g["rank_v"] - g["rank_b"]).mean()),
                 "improved": int((g["rank_v"] < g["rank_b"]).sum()), "worsened": int((g["rank_v"] > g["rank_b"]).sum()),
                 "MRR_base": float(g["rr_b"].mean()), "MRR_var": float(g["rr_v"].mean())}
    return t, by


# ============================================================================ run
def stage_run():
    import torch
    t0 = time.time()
    C, cfp = coconut()
    F = E.FP(E11)
    P = Peaks()
    man = pd.read_csv(O / "holdout_manifest.csv")
    T = pd.read_parquet(R / "primary_targets.parquet")
    T = T[T["status"] == "ok"]
    pq = man[man["leakage_status"] == "ok"].rename(columns={"spectrum_id": "rid", "inchikey14": "ik"})[["rid", "ik", "adduct"]]
    prim = Population("PRIMARY", T["ik"].tolist(), pq, {r.ik: set(r.coconut_ok_keys) for r in T.itertuples()},
                      dict(zip(T["ik"], T["M"])), C, cfp)
    A12 = {k: v for k, v in json.load(open(E12 / "target_aliases.json")).items() if v["pop"] == "T1_np"}
    dq = pd.read_parquet(E11 / "queries.parquet").query("pop == 'T1_np'")[["rid", "ik", "adduct"]]
    Md = (dq.assign(M=pd.read_parquet(E11 / "queries.parquet").set_index("rid").loc[dq["rid"], "pm"].values
                    - dq["adduct"].map(E.AD10).values).groupby("ik")["M"].median().to_dict())
    dev = Population("DEV", list(A12), dq, {k: set(a["ik"]) | set(a["parent"]) | set(a["tautomer"]) for k, a in A12.items()},
                     Md, C, cfp)
    log(f"PRIMARY ranked targets {len(prim.pools)}/{prim.n_targets_all}; DEV {len(dev.pools)}/{dev.n_targets_all}")

    ref_rids = np.load(R / "ref_rids.npy")
    tr = pd.read_parquet(E11 / "training.parquet").set_index("rid")
    ref_ik = tr.loc[ref_rids, "ik"].values
    ref_neg = tr.loc[ref_rids, "adduct"].isin(E.NEG).values
    ref_fprow = np.array([F.row[k] for k in ref_ik])
    Q = pd.concat([prim.queries.assign(pop="PRIMARY"), dev.queries.assign(pop="DEV")], ignore_index=True)
    q_neg = Q["adduct"].isin(E.NEG).values
    ri, qi = P.idx(ref_rids), P.idx(Q["rid"].values)

    def predict(nb_idx, nb_w):
        out = np.empty((len(nb_idx), E.FP_BITS), np.float32)
        for s0 in range(0, len(nb_idx), 256):  # chunked: (256, k, 2048) floats at a time
            bits = np.unpackbits(F.packed[ref_fprow[nb_idx[s0:s0 + 256]]], axis=2).astype(np.float32)
            w = np.clip(nb_w[s0:s0 + 256], 0, None) + 1e-6
            out[s0:s0 + 256] = (w[:, :, None] * bits).sum(1) / w.sum(1, keepdims=True)
        return out

    def scores_from_pred(pop, pred_q):
        by = {}
        for ik, p in zip(Q["ik"].values, pred_q):
            by.setdefault(ik, []).append(p)
        out = {}
        for ik, Pl in pop.pools.items():
            if ik in by:
                out[ik] = E.cosine_scores(np.mean(by[ik], axis=0), Pl["bits"])
        return out

    results, preds_cache = [], {}

    def run_rep(tag, params, keep_pre=False):
        tq = time.time()
        Xr, Xq = featurize(P, ri, **params), featurize(P, qi, **params)
        idx, sims = neighbours(Xq, q_neg, Xr, ref_neg, K_NN, tag)
        pre = neighbours(Xq, q_neg, Xr, ref_neg, K_PRE, tag + "_pre")[0] if keep_pre else None
        pred = predict(idx, sims)
        for pop in (prim, dev):
            results.append(evaluate(pop, scores_from_pred(pop, pred), tag))
        preds_cache[tag] = pred
        log(f"{tag} {params} done in {time.time() - tq:.0f}s")
        return idx, sims, pre, Xr, Xq

    # --- gate: current featurizer reproduces EXP-011 features exactly
    from scipy import sparse
    Xt, rt = sparse.load_npz(E11 / "X_train.npz"), np.load(E11 / "rid_train.npy")
    samp = np.sort(np.random.default_rng(SEED).choice(ref_rids, 2000, replace=False))
    a = featurize(P, P.idx(samp), **CURRENT)
    b = Xt[np.searchsorted(rt, samp)]
    feat_repro = float(abs(a - b).max()) if (a - b).nnz else 0.0
    log(f"feature reproduction max |diff| = {feat_repro}")

    variants = {"B0_current": CURRENT, "V2_fragment_only": {**CURRENT, "nl": False},
                "V3a_bin0.01": {**CURRENT, "bin_w": 0.01}, "V3b_bin0.5": {**CURRENT, "bin_w": 0.5},
                "V4a_linear": {**CURRENT, "inten": "linear"}, "V4b_log": {**CURRENT, "inten": "log"}}
    pre_idx, b0_idx = None, None
    for tag, params in variants.items():
        idx, sims, pre_, _, _ = run_rep(tag, params, keep_pre=(tag == "B0_current"))
        if tag == "B0_current":
            pre_idx, b0_idx = pre_, idx
    # determinism gate: B0 recomputed twice on the identical top-20 path
    idx2, sims2 = neighbours(featurize(P, qi, **CURRENT), q_neg, featurize(P, ri, **CURRENT), ref_neg, K_NN, "det")
    det_ok = bool(np.array_equal(idx2, b0_idx))

    def run_modcos(tag, pre):
        tq = time.time()
        for k, v in (("mz", P.mz), ("it", P.it), ("off", P.off), ("pm", P.pm)):
            if not (R / f"peaks_{k}.npy").exists():
                np.save(R / f"peaks_{k}.npy", v)
        jobs = [(int(qrow), ri[pre[j]].astype(np.int64)) for j, qrow in enumerate(qi)]
        with Pool(4, initializer=_mc_init) as pool:
            mc = np.stack(pool.map(_mc, jobs, chunksize=8))
        o = np.argsort(-mc, axis=1, kind="stable")[:, :K_NN]
        pred = predict(np.take_along_axis(pre, o, axis=1), np.take_along_axis(mc, o, axis=1))
        for pop in (prim, dev):
            results.append(evaluate(pop, scores_from_pred(pop, pred), tag))
        preds_cache[tag] = pred
        log(f"{tag} done in {time.time() - tq:.0f}s")

    run_modcos("V5_modcos_rerank", pre_idx)

    # --- DEV-only selection of levels, then combinations
    res = pd.concat(results, ignore_index=True)
    devm = res[res["population"] == "DEV"].groupby("variant")["rr"].mean()
    choose = lambda opts: max(opts, key=lambda kv: (round(devm[kv[0]], 6), kv[0] == "B0_current"))[1]
    best_nl = choose([("B0_current", True), ("V2_fragment_only", False)])
    best_bin = choose([("B0_current", 0.1), ("V3a_bin0.01", 0.01), ("V3b_bin0.5", 0.5)])
    best_int = choose([("B0_current", "sqrt"), ("V4a_linear", "linear"), ("V4b_log", "log")])
    selection = {"best_nl": best_nl, "best_bin": best_bin, "best_intensity": best_int,
                 "dev_MRR": devm.to_dict(), "rule": "highest DEV MRR per factor; tie -> current"}
    log(f"DEV selection {selection}")
    combos = {"C1_bestNL_bestInt": {"bin_w": 0.1, "nl": best_nl, "inten": best_int},
              "C2_bestNL_bestBin_bestInt": {"bin_w": best_bin, "nl": best_nl, "inten": best_int}}
    for tag, params in combos.items():
        if params == CURRENT:
            continue
        idx, _, pre_, _, _ = run_rep(tag, params, keep_pre=(tag.startswith("C2")))
        if tag.startswith("C2"):
            run_modcos("C3_C2_plus_modcos", pre_)
    if combos["C2_bestNL_bestBin_bestInt"] == CURRENT:
        log("C2 equals the current representation; C3 = V5")
    res = pd.concat(results, ignore_index=True)

    # --- oracle (diagnostic) on PRIMARY
    orc = evaluate(prim, {ik: E.cosine_scores(Pl["bits"][np.flatnonzero(Pl["ok"])[0]], Pl["bits"]) for ik, Pl in prim.pools.items()}, "oracle")
    oracle = {"coverage": len(prim.pools) / prim.n_targets_all, "MRR": orc["rr"].mean(), "R@1": orc["r@1"].mean(),
              "same_formula_prevalence": float(np.mean([P_["same_formula"].any() for P_ in prim.pools.values()])),
              "median_pool": float(np.median([len(P_["iks"]) for P_ in prim.pools.values()]))}

    # --- fusion (PRIMARY): B0 kNN + EXP-013 hard-negative two-tower, fixed 50/50 average rank
    from scipy.stats import rankdata
    net = Towers(); net.load_state_dict(torch.load(E13 / "D_hardneg.pt")); net.eval()
    Xq_cur = featurize(P, qi, **CURRENT)
    with torch.no_grad():
        emb = net.spec(Xq_cur, torch.from_numpy(np.array([E.ADDUCTS.index(x) for x in Q["adduct"]]))).numpy()
    knn_sc = scores_from_pred(prim, preds_cache["B0_current"])
    tt_sc, fu_sc = {}, {}
    by = {}
    for ik, e, pp in zip(Q["ik"].values, emb, Q["pop"].values):
        if pp == "PRIMARY":
            by.setdefault(ik, []).append(e)
    for ik, Pl in prim.pools.items():
        with torch.no_grad():
            ce = net.cand(cfp[Pl["rows"]]).numpy()
        tt_sc[ik] = ce @ np.mean(by[ik], axis=0)
        fu_sc[ik] = -(0.5 * rankdata(-knn_sc[ik]) + 0.5 * rankdata(-tt_sc[ik]))  # higher = better
    fk, ft, ff = evaluate(prim, knn_sc, "kNN_B0"), evaluate(prim, tt_sc, "two_tower"), evaluate(prim, fu_sc, "fusion_50_50")
    ck, ct, cf = (d.set_index("ik")["r@1"] >= 0.5 for d in (fk, ft, ff))
    fusion = pd.DataFrame([{**summarize(d, prim.n_targets_all), "model": n} for n, d in (("kNN_B0", fk), ("two_tower", ft), ("fusion_50_50", ff))])
    fus_counts = {"kNN_only_correct": int((ck & ~ct).sum()), "two_tower_only_correct": int((~ck & ct).sum()),
                  "both_correct": int((ck & ct).sum()), "neither": int((~ck & ~ct).sum()),
                  "fusion_fixes_vs_kNN": int((~ck & cf).sum()), "fusion_regressions_vs_kNN": int((ck & ~cf).sum()),
                  "delta_MRR_fusion_vs_kNN": paired(ff, fk, "rr"), "delta_MRR_two_tower_vs_kNN": paired(ft, fk, "rr"),
                  "delta_R@1_fusion_vs_kNN": paired(ff, fk, "r@1")}

    # --- assemble outputs
    rows, trans = [], {}
    for pop in ("PRIMARY", "DEV"):
        base = res[(res["population"] == pop) & (res["variant"] == "B0_current")]
        n_all = prim.n_targets_all if pop == "PRIMARY" else dev.n_targets_all
        for v, g in res[res["population"] == pop].groupby("variant", sort=False):
            r = {"population": pop, "variant": v, **summarize(g, n_all)}
            if v != "B0_current":
                r.update({"dMRR_vs_B0": paired(g, base, "rr"), "dR@1_vs_B0": paired(g, base, "r@1"),
                          "d_isomer_error_rate_vs_B0": paired(g.dropna(subset=["isomer_error_rate"]),
                                                              base.dropna(subset=["isomer_error_rate"]), "isomer_error_rate")})
                t, bycat = transitions(base, g)
                trans[f"{pop}|{v}"] = {"transitions": t, "by_B0_top1_category": bycat}
            rows.append(r)
    vr = pd.DataFrame(rows)
    vr[vr["variant"] == "B0_current"].to_csv(O / "baseline_results.csv", index=False)
    vr.to_csv(O / "variant_results.csv", index=False)
    b0 = res[res["variant"] == "B0_current"][["population", "ik", "rank", "rr"]].rename(columns={"rank": "B0_rank", "rr": "B0_rr"})
    pqr = res.merge(b0, on=["population", "ik"])
    pqr["rank_delta_vs_B0"] = pqr["rank"] - pqr["B0_rank"]
    pqr.to_csv(O / "per_query_results.csv", index=False)
    fusion.to_csv(O / "fusion_results.csv", index=False)
    json.dump(oracle, open(O / "oracle_results.json", "w"), indent=1, default=float)
    build = json.load(open(R / "build.json"))
    Tall = pd.read_parquet(R / "primary_targets.parquet")
    leak = {"feature_reproduction_max_abs_diff": feat_repro, "B0_deterministic": det_ok,
            "pools_identical_across_variants": True, "pool_hash_PRIMARY": prim.pool_hash, "pool_hash_DEV": dev.pool_hash,
            "query_hash_PRIMARY": prim.query_hash, "query_hash_DEV": dev.query_hash,
            "accepted_targets_with_any_leak_flag": int(Tall.loc[Tall["status"] == "ok",
                ["target_in_ref", "alias_in_ref", "parent_in_ref", "target_or_alias_in_twotower", "parent_in_twotower", "previous_target"]].any(axis=1).sum()),
            "excluded_targets_by_reason": Tall["status"].value_counts().to_dict(),
            "query_spectra_removed_duplicate_in_ref": build["query_spectra_dropped_duplicate_in_ref"],
            "R_noE180": build["R_noE180"], "coconut_fp_indexing": "parse_ok-filtered (EXP-012 bug not reproduced)",
            "selection_used_PRIMARY": False, "seeds": {"holdout": SEED, "bootstrap": SEED}}
    leak["VALID"] = bool(feat_repro < 1e-6 and det_ok and leak["accepted_targets_with_any_leak_flag"] == 0)
    json.dump(leak, open(O / "leakage_audit.json", "w"), indent=1, default=str)
    summary = {"build": build, "selection": selection, "oracle": oracle, "fusion_counts": fus_counts,
               "transitions": trans, "runtime_sec": round(time.time() - t0, 1), "valid": leak["VALID"]}
    json.dump(summary, open(O / "summary.json", "w"), indent=1, default=float)
    log("done")
    print(vr.to_string())
    print(json.dumps({"oracle": oracle, "fusion_counts": fus_counts, "valid": leak["VALID"]}, indent=1, default=float))


class Towers:  # placeholder, replaced below (needs torch import at module level only when used)
    pass


def _make_towers():
    import torch

    def bits_bag(packed):
        b = np.unpackbits(packed, axis=1)
        r, c = np.nonzero(b)
        off = np.r_[0, np.cumsum(np.bincount(r, minlength=len(b)))[:-1]]
        return torch.from_numpy(c.astype(np.int64)), torch.from_numpy(off.astype(np.int64))

    class _Towers(torch.nn.Module):  # identical to EXP-013 stage_d
        def __init__(self):
            super().__init__()
            self.se = torch.nn.EmbeddingBag(E.NFEAT, 512, mode="sum")
            self.sa = torch.nn.Embedding(len(E.ADDUCTS), 512)
            self.sh = torch.nn.Linear(512, 512)
            self.ce = torch.nn.EmbeddingBag(E.FP_BITS, 512, mode="sum")
            self.ch = torch.nn.Linear(512, 512)

        def spec(self, X, a):
            e = self.se(torch.from_numpy(X.indices.astype(np.int64)), torch.from_numpy(X.indptr[:-1].astype(np.int64)),
                        per_sample_weights=torch.from_numpy(X.data.astype(np.float32))) + self.sa(a)
            return torch.nn.functional.normalize(self.sh(torch.relu(e)), dim=-1)

        def cand(self, packed):
            i, o = bits_bag(packed)
            return torch.nn.functional.normalize(self.ch(torch.relu(self.ce(i, o))), dim=-1)
    return _Towers


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        Towers = _make_towers()  # noqa: F811
    {"build": stage_build, "peaks": stage_peaks, "run": stage_run}[sys.argv[1]]()

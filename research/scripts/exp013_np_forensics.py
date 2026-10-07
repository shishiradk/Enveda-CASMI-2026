"""EXP-013 forensic diagnosis of the natural-product Class-2 ranking failure.
Design (locked): research/analysis/exp013_np_ranking_forensics_design.md

    python research/scripts/exp013_np_forensics.py ab   # A error taxonomy + B nearest-structure / fingerprint oracle
    python research/scripts/exp013_np_forensics.py c    # C reference-library test (kNN unchanged)
    python research/scripts/exp013_np_forensics.py d    # D hard-negative two-tower ranker + random-negative ablation
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402
import exp012_coconut_pools as C12  # noqa: E402

O = ROOT / "results" / "exp013_np_forensics"
E11, E12 = ROOT / "results" / "exp011_class2_proxy", ROOT / "results" / "exp012_coconut"
SEED = 20260926
NP_LIBS = {"riken", "gnps", "massbank", "mona", "spectraverse", "msdial", "masaryk"}
TC_NEAR, TC_UNREL = 0.50, 0.30
EPOCH_SPECTRA = 250_000


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def tanimoto_packed(a, B):
    """a: (256,) uint8 packed; B: (n,256) packed -> Tanimoto."""
    inter = np.bitwise_count(np.bitwise_and(B, a)).sum(1, dtype=np.int64)  # popcount; identical to unpackbits().sum()
    union = np.bitwise_count(np.bitwise_or(B, a)).sum(1, dtype=np.int64)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


# ----------------------------------------------------------------------------- common setup
class Setup:
    def __init__(self):
        self.A = {k: v for k, v in json.load(open(E12 / "target_aliases.json")).items() if v["pop"] == "T1_np"}
        C = pd.read_parquet(E12 / "coconut_db.parquet")
        self.C = C[C["parse_ok"]].reset_index(drop=True)
        self.cfp = np.load(E12 / "coconut_fp_packed.npy")[C["parse_ok"].values]
        self.Cs = self.C.assign(row=np.arange(len(self.C))).sort_values("mass").reset_index(drop=True)
        self.cm = self.Cs["mass"].values
        self.M = C12.query_masses()["np"]
        self.pools = {}
        for ik, a in self.A.items():
            lo, hi = C12.window(self.cm, float(self.M[ik]), C12.PRIMARY)
            p = self.Cs.iloc[lo:hi]
            ok = set(a["ik"]) | set(a["parent"]) | set(a["tautomer"])
            self.pools[ik] = {"iks": p["ik"].tolist(), "rows": p["row"].values, "mass": p["mass"].values,
                              "ok": np.array([k in ok for k in p["ik"]])}

    def rank(self, ik, scores):
        """Truth collapsed to its best-scoring equivalent key; returns stats + ordered pool indices."""
        P = self.pools[ik]
        ok = P["ok"]
        best = scores[ok].max()
        others = scores[~ok]
        st = E.expected_rank_stats({**{i: s for i, s in enumerate(others)}, "__t__": best}, "__t__")
        order = np.lexsort((np.array(P["iks"]), -scores))
        return st, order

    def pool_bits(self, ik):
        return np.unpackbits(self.cfp[self.pools[ik]["rows"]], axis=1).astype(np.float32)


def summarize(rrs, n_all, extra=None):
    rr = np.array([r["rr"] for r in rrs])
    return {"n_ranked": len(rr), "mrr_cond": float(rr.mean()), "mrr_uncond": float(rr.sum() / n_all),
            "mrr25_cond": float(np.mean([r["rr25"] for r in rrs])),
            **{f"r@{k}_cond": float(np.mean([r[f"r@{k}"] for r in rrs])) for k in (1, 10, 50, 100)}, **(extra or {})}


def paired_ci(d, seed=SEED):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def chance_rr(n):
    return float(sum(1 / r for r in range(1, n + 1)) / n)


def leakage_report():
    return C12.leakage_checks()


# ----------------------------------------------------------------------------- A + B
def stage_ab():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    S = Setup()
    z = np.load(E11 / "pred_B3.npz")
    pred = dict(zip(z["iks"], z["pred"]))
    smi = dict(zip(S.C["ik"], S.C["smiles"]))
    props = {}

    def prop(ik):
        if ik not in props:
            m = Chem.MolFromSmiles(smi[ik])
            try:
                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
            except Exception:
                sc = None
            props[ik] = (CalcMolFormula(m), sc)
        return props[ik]

    def category(same_f, same_sc, tc):
        if same_f:
            return "1_same_formula_near_isomer" if tc >= TC_NEAR else "2_same_formula_distant_isomer"
        if same_sc:
            return "3_same_scaffold"
        if tc >= TC_NEAR:
            return "4_similar_different_formula"
        if tc < TC_UNREL:
            return "5_unrelated"
        return "6_other_moderate"

    tgt_rows, cand_rows = [], []
    for ik, a in S.A.items():
        P = S.pools[ik]
        if not P["ok"].any():
            tgt_rows.append({"ik": ik, "covered": False, "category_target": "8_candidate_generation",
                             "pool": len(P["iks"])})
            continue
        truth_i = int(np.flatnonzero(P["ok"])[0])
        tfp = S.cfp[P["rows"][truth_i]]
        tf, tsc = prop(P["iks"][truth_i])
        tc_all = tanimoto_packed(tfp, S.cfp[P["rows"]])
        cb = S.pool_bits(ik)
        sc = E.cosine_scores(pred[ik], cb)
        st, order = S.rank(ik, sc)
        ost, _ = S.rank(ik, E.cosine_scores(np.unpackbits(tfp).astype(np.float32), cb))  # fingerprint oracle
        wrong_order = [i for i in order if not P["ok"][i]]
        top10_pos = [i for i in order[:10] if not P["ok"][i]]
        cats = {}
        for i in range(len(P["iks"])):
            if P["ok"][i]:
                continue
            f, s = prop(P["iks"][i])
            c = category(f == tf, s is not None and s == tsc, tc_all[i])
            cats[i] = c
            role = "top1_wrong" if wrong_order and i == wrong_order[0] else ("top10_wrong" if i in top10_pos else "pool")
            cand_rows.append({"target": ik, "cand": P["iks"][i], "role": role, "above_truth": sc[i] > sc[P["ok"]].max(),
                              "score": float(sc[i]), "tc_to_truth": float(tc_all[i]), "same_formula": f == tf,
                              "same_scaffold": s is not None and s == tsc,
                              "mass_diff_ppm": float((P["mass"][i] - P["mass"][truth_i]) / P["mass"][truth_i] * 1e6),
                              "category": c})
        n_wrong = len(P["iks"]) - int(P["ok"].sum())
        tc_w = tc_all[~P["ok"]]
        top1 = wrong_order[0] if wrong_order else None
        tgt_rows.append({
            "ik": ik, "covered": True, "formula": tf, "neutral_mass": float(S.M[ik]), "pool": len(P["iks"]),
            "rank": st["rank"], "rr": st["rr"], "rr25": st["rr25"], **{f"r@{k}": st[f"r@{k}"] for k in (1, 10, 50, 100)},
            "oracle_rr": ost["rr"], "oracle_r@1": ost["r@1"], "chance_rr": chance_rr(len(P["iks"]) - int(P["ok"].sum()) + 1),
            "top1_cand": P["iks"][top1] if top1 is not None else None,
            "top1_category": cats.get(top1) if st["rank"] > 1 else "correct",
            "top10": [P["iks"][i] for i in order[:10]],
            "tc_top1_wrong": float(tc_all[top1]) if top1 is not None else None,
            "tc_top10_wrong_mean": float(np.mean(tc_all[top10_pos])) if top10_pos else None,
            "tc_pool_wrong_max": float(tc_w.max()) if n_wrong else None,
            "tc_pool_wrong_mean": float(tc_w.mean()) if n_wrong else None,
            "n_wrong_tc_ge_0.70": int((tc_w >= 0.70).sum()), "n_wrong_tc_ge_0.85": int((tc_w >= 0.85).sum()),
            "n_same_formula_wrong": int(sum(1 for i, c in cats.items() if c.startswith(("1_", "2_")))),
            "top1_tc_percentile_in_pool": float((tc_w <= tc_all[top1]).mean()) if top1 is not None else None,
            "pred_cos_truth": float(E.cosine_scores(pred[ik], cb[[truth_i]])[0]),
            "pred_cos_top1wrong": float(sc[top1]) if top1 is not None else None,
        })
    T = pd.DataFrame(tgt_rows)
    Cd = pd.DataFrame(cand_rows)
    O.mkdir(parents=True, exist_ok=True)
    T.to_parquet(O / "A_targets.parquet", index=False)
    Cd.to_parquet(O / "A_candidates.parquet", index=False)

    cov = T[T["covered"]]
    wrong_top1 = cov[cov["rank"] > 1]
    def dist(s):
        v = s.value_counts()
        return {k: {"n": int(n), "pct": round(100 * n / v.sum(), 1)} for k, n in v.sort_index().items()}
    rep = {
        "n_targets": len(T), "n_covered": len(cov), "n_candidate_generation_failures": int((~T["covered"]).sum()),
        "kNN": summarize(cov.to_dict("records"), len(T)),
        "fingerprint_oracle": {"mrr_cond": float(cov["oracle_rr"].mean()), "r@1_cond": float(cov["oracle_r@1"].mean())},
        "chance_mrr_cond": float(cov["chance_rr"].mean()),
        "A_top1_error_categories": dist(wrong_top1["top1_category"]),
        "A_top10_wrong_categories": dist(Cd.loc[Cd["role"].isin(["top1_wrong", "top10_wrong"]), "category"]),
        "A_wrong_above_truth_categories": dist(Cd.loc[Cd["above_truth"], "category"]),
        "A_pool_base_rate_categories": dist(Cd["category"]),
        "A_stereochemical": "0 by construction (InChIKey14 collapses stereoisomers)",
        "B": {
            "tc_top1_wrong_median": float(wrong_top1["tc_top1_wrong"].median()),
            "tc_top10_wrong_mean_median": float(cov["tc_top10_wrong_mean"].median()),
            "tc_pool_wrong_max_median": float(cov["tc_pool_wrong_max"].median()),
            "tc_pool_wrong_mean_median": float(cov["tc_pool_wrong_mean"].median()),
            "top1_tc_percentile_in_pool_median": float(wrong_top1["top1_tc_percentile_in_pool"].median()),
            "frac_targets_with_wrong_tc_ge_0.70": float((cov["n_wrong_tc_ge_0.70"] > 0).mean()),
            "frac_targets_with_wrong_tc_ge_0.85": float((cov["n_wrong_tc_ge_0.85"] > 0).mean()),
            "frac_targets_with_same_formula_wrong": float((cov["n_same_formula_wrong"] > 0).mean()),
            "pred_cos_truth_median": float(cov["pred_cos_truth"].median()),
            "pred_cos_top1wrong_median": float(wrong_top1["pred_cos_top1wrong"].median()),
            "kNN_mrr_by_nearest_wrong_tc": cov.groupby(pd.cut(cov["tc_pool_wrong_max"], [0, .5, .7, .85, 1.01]),
                                                       observed=True)["rr"].agg(["count", "mean"]).round(3).rename(index=str).astype(object).to_dict("index"),
            "oracle_mrr_by_nearest_wrong_tc": cov.groupby(pd.cut(cov["tc_pool_wrong_max"], [0, .5, .7, .85, 1.01]),
                                                          observed=True)["oracle_rr"].agg(["count", "mean"]).round(3).rename(index=str).astype(object).to_dict("index"),
        },
        "leakage": leakage_report(), "sec": round(time.time() - t0, 1)}
    json.dump(rep, open(O / "AB_report.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


# ----------------------------------------------------------------------------- C
def knn_predict(S_chunked, ref_mask, t_rows, t_neg, q_neg, packed, k=20):
    """Unchanged EXP-011 B3 rule on a column subset: top-k by cosine among same-polarity references in ref_mask."""
    preds = []
    for S, qn in S_chunked:
        for j in range(S.shape[0]):
            s = np.where((t_neg == qn[j]) & ref_mask, S[j], -1.0)
            nn = np.argpartition(-s, k)[:k]
            w = np.clip(s[nn], 0, None) + 1e-6
            bits = np.unpackbits(packed[t_rows[nn]], axis=1).astype(np.float32)
            preds.append((w[:, None] * bits).sum(0) / w.sum())
    return np.stack(preds)


def stage_c():
    from scipy import sparse
    t0 = time.time()
    S = Setup()
    q = pd.read_parquet(E11 / "queries.parquet").set_index("rid")
    tr = pd.read_parquet(E11 / "training.parquet")
    tr = tr[tr["split"] == "train"].set_index("rid")
    Xq, rq = sparse.load_npz(E11 / "X_query.npz"), np.load(E11 / "rid_query.npy")
    Xt_all, rt_all = sparse.load_npz(E11 / "X_train.npz"), np.load(E11 / "rid_train.npy")
    keep = np.isin(rt_all, tr.index.values)
    Xt, rt = Xt_all[keep], rt_all[keep]
    lib, ikt = tr.loc[rt, "lib"].values, tr.loc[rt, "ik"].values
    t_neg = tr.loc[rt, "adduct"].isin(E.NEG).values
    F = E.FP(E11)
    t_rows = np.array([F.row[k] for k in ikt])
    sel = np.array([q.loc[r, "ik"] in S.A for r in rq])
    Xq1, rq1 = Xq[sel], rq[sel]
    q_neg = q.loc[rq1, "adduct"].isin(E.NEG).values

    rng = np.random.default_rng(SEED)
    cset = set(S.C["ik"])
    in_coco = np.array([k in cset for k in ikt])
    masks = {"R_full": np.ones(len(rt), bool), "R_noE180": lib != "enveda-180",
             "R_NPlib": np.isin(lib, list(NP_LIBS)), "R_COCONUTmol": in_coco}
    cm_mols = np.array(sorted(set(ikt[in_coco])))
    for frac in (0.25, 0.50):
        pick = set(rng.choice(cm_mols, int(frac * len(cm_mols)), replace=False))
        masks[f"R_COCONUTmol_{int(frac * 100)}pct"] = in_coco & np.isin(ikt, list(pick))
    all_mols = rng.permutation(np.array(sorted(set(ikt))))  # size control: random molecules until spectrum count matches
    cnt = pd.Series(ikt).value_counts()
    target_n, acc, chosen = int(in_coco.sum()), 0, []
    for m in all_mols:
        chosen.append(m); acc += cnt[m]
        if acc >= target_n:
            break
    masks["R_sizecontrol_random"] = np.isin(ikt, chosen)

    XtT = Xt.T.tocsr()
    chunks = [((Xq1[i:i + 128] @ XtT).toarray(), q_neg[i:i + 128]) for i in range(0, Xq1.shape[0], 128)]
    log(f"similarities computed ({Xq1.shape[0]} T1 queries x {Xt.shape[0]} refs) in {time.time() - t0:.0f}s")
    q_ik = q.loc[rq1, "ik"].values
    ref_b3 = dict(zip(*(lambda z: (z["iks"], z["pred"]))(np.load(E11 / "pred_B3.npz"))))
    rep = {"leakage": leakage_report(), "references": {}}
    per = {}
    for name, mk in masks.items():
        P = knn_predict(chunks, mk, t_rows, t_neg, q_neg, F.packed)
        by = {}
        for ik, p in zip(q_ik, P):
            by.setdefault(ik, []).append(p)
        by = {k: np.mean(v, axis=0) for k, v in by.items()}
        if name == "R_full":  # must reproduce EXP-011 B3 exactly (kNN unchanged)
            rep["R_full_reproduces_B3_max_abs_diff"] = float(max(np.abs(by[k] - ref_b3[k]).max() for k in by))
        rrs, rows = [], {}
        for ik in S.A:
            if not S.pools[ik]["ok"].any():
                continue
            st, _ = S.rank(ik, E.cosine_scores(by[ik], S.pool_bits(ik)))
            rrs.append(st); rows[ik] = st["rr"]
        per[name] = rows
        refs = {"spectra": int(mk.sum()), "molecules": int(len(set(ikt[mk]))),
                "pct_enveda180": float(np.mean(lib[mk] == "enveda-180")), "pct_in_coconut": float(np.mean(in_coco[mk]))}
        rep["references"][name] = {"reference": refs, **summarize(rrs, len(S.A))}
        log(f"{name}: {refs} MRR {rep['references'][name]['mrr_cond']:.3f}")
    iks = sorted(per["R_full"])
    ch = {ik: chance_rr(len(S.pools[ik]["iks"]) - int(S.pools[ik]["ok"].sum()) + 1) for ik in iks}
    for name in per:
        rep["references"][name]["lift_vs_chance"] = paired_ci([per[name][k] - ch[k] for k in iks])
        rep["references"][name]["delta_vs_R_full"] = paired_ci([per[name][k] - per["R_full"][k] for k in iks])
    rep["pool"] = {"coverage": float(np.mean([S.pools[k]["ok"].any() for k in S.A])),
                   "size_median": float(np.median([len(S.pools[k]["iks"]) for k in S.A]))}
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(O / "C_report.json", "w"), indent=1)
    pd.DataFrame(per).to_parquet(O / "C_rr_per_target.parquet")
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------- D
def stage_d():
    import torch
    from scipy import sparse
    t0 = time.time()
    torch.manual_seed(SEED); torch.set_num_threads(4)
    S = Setup()
    F = E.FP(E11)
    man = json.load(open(E11 / "manifest.json"))
    targets = set(pd.read_parquet(E11 / "targets.parquet")["ik"])
    excl = targets | {x for v in man["aliases"].values() for x in v} | set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
    A12 = json.load(open(E12 / "target_aliases.json"))
    excl |= {x for a in A12.values() for k in ("ik", "parent", "tautomer") for x in a[k]}
    U8 = pd.read_parquet(E.UNIVERSE)
    tpar = set(U8.loc[U8["ik"].isin(targets), "parent"])
    excl |= set(U8.loc[U8["parent"].isin(tpar), "ik"]) | set(S.C.loc[S.C["parent"].isin(tpar), "ik"])
    # negative universe: COCONUT + train-structure fingerprints available (EXP-011 store), minus exclusions
    nu_ik = list(S.C["ik"]); nu_fp = [S.cfp]; nu_m = list(S.C["mass"])
    idx = pd.read_parquet(E11 / "fp_index.parquet")
    extra = idx[~idx["ik"].isin(set(S.C["ik"]))]
    nu_ik += extra["ik"].tolist(); nu_fp.append(F.packed[[F.row[k] for k in extra["ik"]]]); nu_m += extra["mass"].tolist()
    nu_fp = np.concatenate(nu_fp); nu_m = np.array(nu_m); nu_ik = np.array(nu_ik)
    keepn = ~np.isin(nu_ik, list(excl))
    o = np.argsort(nu_m[keepn]); nu_ik, nu_fp, nu_m = nu_ik[keepn][o], nu_fp[keepn][o], nu_m[keepn][o]
    log(f"negative universe {len(nu_ik)} structures; excluded {len(excl)} target/alias/decoy keys")

    tr = pd.read_parquet(E11 / "training.parquet").set_index("rid")
    Xt, rt = sparse.load_npz(E11 / "X_train.npz"), np.load(E11 / "rid_train.npy")
    split, tik = tr.loc[rt, "split"].values, tr.loc[rt, "ik"].values
    add = np.array([E.ADDUCTS.index(a) for a in tr.loc[rt, "adduct"].values])
    pos_fp = F.packed[[F.row[k] for k in tik]]
    pos_m = F.mass[[F.row[k] for k in tik]]
    leak = {"train_molecules_in_exclusions": int(np.isin(tik, list(excl)).sum()),
            "negatives_in_exclusions": int(np.isin(nu_ik, list(excl)).sum()),
            "val_molecules_overlap_train": len(set(tik[split == "val"]) & set(tik[split == "train"]))}
    log(leak)
    assert all(v == 0 for v in leak.values())

    def bits_bag(packed):
        b = np.unpackbits(packed, axis=1)
        r, c = np.nonzero(b)
        off = np.r_[0, np.cumsum(np.bincount(r, minlength=len(b)))[:-1]]
        return torch.from_numpy(c.astype(np.int64)), torch.from_numpy(off.astype(np.int64))

    class Towers(torch.nn.Module):
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

    def negatives(b, rng, hard):
        out = np.empty((len(b), 31), dtype=np.int64)
        for j, s in enumerate(b):
            M = pos_m[s]
            if not hard:
                out[j] = rng.integers(0, len(nu_ik), 31); continue
            lo5, hi5 = np.searchsorted(nu_m, M * (1 - 5e-6)), np.searchsorted(nu_m, M * (1 + 5e-6), side="right")
            lo50, hi50 = np.searchsorted(nu_m, M * (1 - 50e-6)), np.searchsorted(nu_m, M * (1 + 50e-6), side="right")
            same = np.arange(lo5, hi5)
            same = same[nu_ik[same] != tik[s]]
            if len(same) > 64:
                same = rng.choice(same, 64, replace=False)
            picks = []
            if len(same):
                tc = tanimoto_packed(pos_fp[s], nu_fp[same])
                top = same[np.argsort(-tc)[:8]]
                picks += top.tolist()
                rest = np.setdiff1d(same, top)
                picks += rng.choice(rest, min(8, len(rest)), replace=False).tolist() if len(rest) else []
            band = np.r_[np.arange(lo50, lo5), np.arange(hi5, hi50)]
            band = band[nu_ik[band] != tik[s]] if len(band) else band
            picks += rng.choice(band, min(8, len(band)), replace=False).tolist() if len(band) else []
            fill = 31 - len(picks)
            picks += rng.integers(0, len(nu_ik), fill).tolist()
            out[j] = picks[:31]
        return out

    def run(hard):
        tag = "hardneg" if hard else "randneg"
        net = Towers()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rng = np.random.default_rng(SEED)
        tri, vai = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
        vrng = np.random.default_rng(SEED + 1)
        vsub = vrng.choice(vai, min(20000, len(vai)), replace=False)
        vneg = negatives(vsub, np.random.default_rng(SEED + 2), hard)
        best, hist = 1e9, []

        def loss_on(b, neg):
            se = net.spec(Xt[b], torch.from_numpy(add[b]))
            ce = net.cand(np.concatenate([pos_fp[b][:, None, :], nu_fp[neg]], axis=1).reshape(-1, E.FP_BITS // 8))
            ce = ce.view(len(b), 32, -1)
            logits = torch.einsum("bd,bkd->bk", se, ce) / 0.1
            return torch.nn.functional.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long))

        for ep in range(3):
            # epoch = 250k random training spectra (set before any D result was seen; runtime cap, same for both models)
            net.train(); perm = rng.choice(tri, min(EPOCH_SPECTRA, len(tri)), replace=False); tl = 0.0; nb = 0
            for s0 in range(0, len(perm), 256):
                b = perm[s0:s0 + 256]
                opt.zero_grad()
                l = loss_on(b, negatives(b, rng, hard))
                l.backward(); opt.step(); tl += l.item(); nb += 1
                if nb % 500 == 0:
                    log(f"  {tag} ep{ep} batch {nb} loss {tl / nb:.4f}")
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([loss_on(vsub[i:i + 1024], vneg[i:i + 1024]).item() for i in range(0, len(vsub), 1024)]))
            hist.append({"epoch": ep, "train_loss": tl / nb, "val_loss": vl, "sec": round(time.time() - t0, 1)})
            log(hist[-1])
            if vl < best:
                best = vl; torch.save(net.state_dict(), O / f"D_{tag}.pt")
            else:
                break
        net.load_state_dict(torch.load(O / f"D_{tag}.pt")); net.eval()
        q = pd.read_parquet(E11 / "queries.parquet").set_index("rid")
        Xq, rq = sparse.load_npz(E11 / "X_query.npz"), np.load(E11 / "rid_query.npy")
        sel = np.array([q.loc[r, "ik"] in S.A for r in rq])
        with torch.no_grad():
            emb = net.spec(Xq[sel], torch.from_numpy(np.array([E.ADDUCTS.index(a) for a in q.loc[rq[sel], "adduct"]]))).numpy()
        by = {}
        for ik, e in zip(q.loc[rq[sel], "ik"].values, emb):
            by.setdefault(ik, []).append(e)
        rows = {}
        for ik in S.A:
            if not S.pools[ik]["ok"].any():
                continue
            me = np.mean(by[ik], axis=0)
            with torch.no_grad():
                ce = net.cand(S.cfp[S.pools[ik]["rows"]]).numpy()
            st, _ = S.rank(ik, ce @ me)
            rows[ik] = st
        return rows, hist

    res = {}
    for hard in (True, False):
        rows, hist = run(hard)
        res["hardneg" if hard else "randneg"] = (rows, hist)
    T = pd.read_parquet(O / "A_targets.parquet").set_index("ik")
    cov = T[T["covered"]]
    knn = {ik: {"rr": cov.loc[ik, "rr"], "rr25": cov.loc[ik, "rr25"], **{f"r@{k}": cov.loc[ik, f"r@{k}"] for k in (1, 10, 50, 100)}}
           for ik in cov.index}
    rep = {"leakage": {**leak, **leakage_report()}, "negative_universe": int(len(nu_ik)), "models": {}}
    iks = sorted(knn)
    same_formula = [ik for ik in iks if cov.loc[ik, "n_same_formula_wrong"] > 0]
    for tag, (rows, hist) in res.items():
        R = {"history": hist, **summarize([rows[k] for k in iks], len(S.A))}
        R["delta_vs_kNN"] = paired_ci([rows[k]["rr"] - knn[k]["rr"] for k in iks])
        R["lift_vs_chance"] = paired_ci([rows[k]["rr"] - cov.loc[k, "chance_rr"] for k in iks])
        R["same_formula_targets"] = {"n": len(same_formula), "mrr": float(np.mean([rows[k]["rr"] for k in same_formula])),
                                     "kNN_mrr": float(np.mean([knn[k]["rr"] for k in same_formula]))}
        R["by_kNN_top1_category"] = {c: {"n": len(g), "mrr": float(np.mean([rows[k]["rr"] for k in g.index])),
                                         "kNN_mrr": float(g["rr"].mean())} for c, g in cov.groupby("top1_category")}
        rep["models"][tag] = R
    rep["kNN"] = summarize([knn[k] for k in iks], len(S.A))
    rep["hardneg_minus_randneg"] = paired_ci([res["hardneg"][0][k]["rr"] - res["randneg"][0][k]["rr"] for k in iks])
    rep["sec"] = round(time.time() - t0, 1)
    json.dump(rep, open(O / "D_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "models"}, indent=1, default=str))
    for tag, R in rep["models"].items():
        print(tag, {k: R[k] for k in R if k not in ("history", "by_kNN_top1_category")})


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True)
    {"ab": stage_ab, "c": stage_c, "d": stage_d}[sys.argv[1]]()

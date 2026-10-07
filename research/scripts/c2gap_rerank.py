"""C2 gap diagnosis, stage 5: a learned re-ranker over the PubChem channel's top-100 window (S3-PC and S2 buckets).

    python research/scripts/c2gap_rerank.py

Input: results/c2gap/cand.parquet (top-100 per molecule by ho1 fp@zlog, plus the truth row when deeper) and, if
present, results/c2gap/ho2_scores.parquet (ho2 nets; ensemble = mean score). All nets are held out from the bench.
Model: LightGBM lambdarank, 5-fold CV grouped by molecule, trained on PC+S2 together (S1 shares S2's windows and
queries, so it is not used). Only molecules whose truth is in the top-100 can be trained on; MRR is over all molecules
of the bucket (truth beyond 100 or absent = 0). The E6 merged rank is simulated from the channel rank with the m1_alt
slot rule (engine e_k at 2k, channel p_j at 2j+3), which reproduces e6_rank (checked in the output).
Popularity bias check: 'cf' = at test time the truth's popularity features are replaced by those of a random
non-truth candidate of the same molecule (truth treated as an ordinary compound).
Writes results/c2gap/rerank.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "c2gap"
RNG = np.random.default_rng(0)


def featurize(c):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors as RD, Descriptors
    from rdkit.Contrib.NP_Score import npscorer
    RDLogger.DisableLog("rdApp.*")
    npm = npscorer.readNPModel()
    rows = []
    for s in c.smiles.fillna("").values:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            rows.append([np.nan] * 10); continue
        rows.append([npscorer.scoreMol(m, npm), RD.CalcNumRings(m), RD.CalcNumAromaticRings(m), RD.CalcFractionCSP3(m),
                     len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
                     int("." in s), sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()),
                     int(any(a.GetIsotope() for a in m.GetAtoms())), RD.CalcNumHBD(m), Descriptors.MolLogP(m)])
    f = pd.DataFrame(rows, columns=["np", "rings", "arom", "fsp3", "nstereo", "multi", "charge", "isotope", "hbd", "logp"])
    return pd.concat([c.reset_index(drop=True), f], axis=1)


def add_rel(c, mol):
    # the extract keeps the truth row even when it is deeper than rank 100; such a row would be a label leak
    # (rank 101 = truth), so only the top-100 window is ever ranked: deeper truths count as absent.
    c = c[c.wrank <= 100].reset_index(drop=True)
    g = c.groupby(["scen", "mid"])
    c["ens"] = (c.score + c.s_ho2) / 2 if "s_ho2" in c else c.score
    for s in ("score", "s_ho2", "ens"):
        if s not in c:
            continue
        c[f"{s}_gap"] = c[s] - g[s].transform("max")
        c[f"{s}_z"] = (c[s] - g[s].transform("mean")) / (g[s].transform("std") + 1e-6)
        c[f"{s}_rk"] = g[s].rank(ascending=False, method="first")
    c["lsid"], c["lpmid"], c["lcid"] = np.log1p(c.n_sid.fillna(0)), np.log1p(c.n_pmid.fillna(0)), np.log1p(c.n_cid.fillna(0))
    c["lsid_rel"] = c.lsid - g["lsid"].transform("max")
    c["lpmid_rel"] = c.lpmid - g["lpmid"].transform("max")
    t = mol.set_index(["scen", "mid"]).target
    tg = t.reindex(pd.MultiIndex.from_arrays([c.scen, c.mid])).values
    c["ppm"] = (c.mass - tg) / tg * 1e6
    c["abs_ppm"] = c.ppm.abs()
    c["np_rel"] = c.np - g["np"].transform("mean")
    return c


SETS = {
    "ho1_only": ["score"],
    "ho1+struct": ["score", "score_gap", "score_z", "score_rk", "ppm", "abs_ppm", "np", "np_rel", "rings", "arom", "fsp3",
                   "nstereo", "multi", "charge", "isotope", "hbd", "logp"],
    "ho1+struct+pop": None,  # filled below
    "pop_only": ["lsid", "lpmid", "lcid", "lsid_rel", "lpmid_rel"],
}
SETS["ho1+struct+pop"] = SETS["ho1+struct"] + SETS["pop_only"]
SETS["ho1+pop"] = ["score", "score_gap", "score_z", "score_rk"] + SETS["pop_only"]


def mrr_from_ranks(r):
    r = np.asarray(r, float)
    return float(np.where((r > 0) & (r <= 25), 1 / np.where(r > 0, r, 1), 0).mean())


def e6_sim(ch, eng):
    """merged rank under m1_alt: engine e_k at 1 (k=0) or 2k, channel p_j (0-based) at 2j+3; dedup ignored."""
    out = []
    for c, e in zip(ch, eng):
        cand = []
        if e > 0:
            cand.append(1 if e == 1 else 2 * (e - 1))
        if c > 0:
            cand.append(2 * (c - 1) + 3)
        out.append(min(cand) if cand else 0)
    return out


def run_cv(c, mol, feats, cf=False, label=""):
    import lightgbm as lgb
    mids = mol[["scen", "mid"]].drop_duplicates().reset_index(drop=True)
    fold = pd.Series(RNG.permutation(len(mids)) % 5, index=pd.MultiIndex.from_frame(mids))
    c = c.assign(fold=fold.reindex(pd.MultiIndex.from_arrays([c.scen, c.mid])).values)
    ranks = {}
    for k in range(5):
        tr = c[(c.fold != k)]
        trg = tr.groupby(["scen", "mid"]).is_truth.transform("any")
        tr = tr[trg].sort_values(["scen", "mid"])
        te = c[c.fold == k].copy()
        if cf:  # truth gets the popularity of a random non-truth candidate of its molecule
            for (s, m), g in te.groupby(["scen", "mid"]):
                if g.is_truth.any() and (~g.is_truth).sum() > 0:
                    donor = g[~g.is_truth].sample(1, random_state=int(RNG.integers(1e9))).iloc[0]
                    ti = g.index[g.is_truth.values]
                    for f in ("lsid", "lpmid", "lcid", "n_sid", "n_pmid", "n_cid"):
                        te.loc[ti, f] = donor[f]
            gg = te.groupby(["scen", "mid"])
            te["lsid_rel"] = te.lsid - gg["lsid"].transform("max")
            te["lpmid_rel"] = te.lpmid - gg["lpmid"].transform("max")
        if feats == ["score"] or feats == ["ens"] or feats == ["s_ho2"]:
            te["pred"] = te[feats[0]]
        else:
            ds = lgb.Dataset(tr[feats], tr.is_truth.astype(int), group=tr.groupby(["scen", "mid"], sort=False).size().values)
            mdl = lgb.train(dict(objective="lambdarank", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50,
                                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=25,
                                 verbose=-1, seed=k, num_threads=2), ds, num_boost_round=200)
            te["pred"] = mdl.predict(te[feats])
        te = te.sort_values(["scen", "mid", "pred"], ascending=[True, True, False])
        te["r"] = te.groupby(["scen", "mid"]).cumcount() + 1
        for (s, m), g in te.groupby(["scen", "mid"]):
            h = g[g.is_truth]
            ranks[(s, m)] = int(h.r.iloc[0]) if len(h) and h.r.iloc[0] <= 100 else 0
    mm = mol.copy()
    mm["ch"] = [ranks.get((s, m), 0) for s, m in zip(mm.scen, mm.mid)]
    mm["e6"] = e6_sim(mm.ch, mm.eng_rank)
    res = {}
    for b in ("PC", "S2"):
        g = mm[mm.b == b]
        res[b] = dict(channel=round(mrr_from_ranks(g.ch), 4), e6_sim=round(mrr_from_ranks(g.e6), 4),
                      top1=int((g.ch == 1).sum()), top10=int(((g.ch > 0) & (g.ch <= 10)).sum()))
    return res


def main():
    c = pd.read_parquet(D / "cand.parquet")
    mol = pd.read_parquet(D / "mol.parquet")
    mol = mol[mol.b.isin(["PC", "S2"])].reset_index(drop=True)
    c = c[c.b.isin(["PC", "S2"])].reset_index(drop=True)
    if (D / "ho2_scores.parquet").exists():
        h = pd.read_parquet(D / "ho2_scores.parquet")
        c = c.merge(h, on=["scen", "mid", "ik"], how="left")
    c = featurize(c)
    c = add_rel(c, mol)
    out = {}
    # check: simulated E6 from the actual channel rank vs the real E6 rank
    m2 = mol.copy(); m2["sim"] = e6_sim(m2.ch_rank, m2.eng_rank)
    out["check_e6_sim"] = {b: dict(real=round(mrr_from_ranks(g.e6_rank), 4), sim=round(mrr_from_ranks(g.sim), 4),
                                   channel_real=round(mrr_from_ranks(g.ch_rank), 4)) for b, g in m2.groupby("b")}
    print(out["check_e6_sim"])
    sets = dict(SETS)
    if "s_ho2" in c:
        sets["ho2_only"] = ["s_ho2"]
        sets["ens_only"] = ["ens"]
        sets["ens+struct+pop"] = ["ens", "ens_gap", "ens_z", "ens_rk", "score", "s_ho2"] + SETS["ho1+struct+pop"][4:]
        sets["ens+struct"] = ["ens", "ens_gap", "ens_z", "ens_rk", "score", "s_ho2"] + SETS["ho1+struct"][4:]
    for name, fs in sets.items():
        out[name] = run_cv(c, mol, fs)
        print(name, out[name], flush=True)
        if any(f in fs for f in ("lsid", "lpmid")):
            out[name + "__cf"] = run_cv(c, mol, fs, cf=True)
            print(name + "__cf", out[name + "__cf"], flush=True)
    # truth vs top-1 impostor contrasts (ho1 order), molecules with truth in window but not at rank 1
    imp = []
    for (s, m), g in c.groupby(["scen", "mid"]):
        if not g.is_truth.any() or g.is_truth.values[np.argmax(g.score.values)]:
            continue
        t = g[g.is_truth].iloc[0]; i = g.loc[g.score.idxmax()]
        imp.append(dict(b=g.b.iloc[0], dscore=t.score - i.score, dlsid=t.lsid - i.lsid, dlpmid=t.lpmid - i.lpmid,
                        dnp=t.np - i.np, dabs_ppm=t.abs_ppm - i.abs_ppm, same_mass=abs(t.mass - i.mass) < 2e-4,
                        dmulti=t.multi - i.multi, dcharge=t.charge - i.charge, dstereo=t.nstereo - i.nstereo,
                        dho2=(t.s_ho2 - i.s_ho2) if "s_ho2" in c else np.nan))
    imp = pd.DataFrame(imp)
    out["impostor_contrast"] = {b: {k: (round(float(g[k].median()), 3) if k != "same_mass" else round(float(g[k].mean()), 3))
                                    for k in g.columns if k != "b"} | {"n": len(g),
                                    "truth_more_popular_sid": round(float((g.dlsid > 0).mean()), 3),
                                    "truth_ho2_better": round(float((g.dho2 > 0).mean()), 3) if "s_ho2" in c else None,
                                    "truth_more_np": round(float((g.dnp > 0).mean()), 3)}
                                for b, g in imp.groupby("b")}
    print(json.dumps(out["impostor_contrast"], indent=1))
    json.dump(out, open(D / "rerank.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

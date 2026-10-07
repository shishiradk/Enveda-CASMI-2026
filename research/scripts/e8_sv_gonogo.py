"""TASK D: E8 go/no-go -- re-score SV with the deploy nets, then apply the PubChem re-ranker.

    python research/scripts/e8_sv_gonogo.py [--feats full] [--seed 0] [--out results/c2gap_full/sv_gonogo.json]

The re-ranker in research/analysis/c2_gap_diagnosis.md was fitted on scores from the held-out **ho1** nets, but E6/E8
deploy the **full-data** nets, so score / score_gap / score_z / score_rk shift.  SV is the only bucket the full nets
never trained on, so it is the honest check.

  train  : results/c2gap/{cand,mol}.parquet  -- bench PC+S2 rows, ho1 scores (as shipped)
  apply  : results/c2gap/{cand,mol}_fullSV.parquet -- SV rows, full-net scores
  merge  : e6_merge_eval.merge(e, p, 1, "alt") -- the shipped E6 rule, unchanged

Feature sets (--feats):
  full    SETS["ho1+struct+pop"]                       (the shipped set)
  rankov  full minus raw `score`                       (instruction's rank-only variant)
  rankz   full minus raw `score` and `score_gap`       (stricter: only rank/z/net-invariant terms)

Writes the JSON with per-molecule merged ranks so the bootstrap CI is reproducible.
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import bench, bench_eval as BE  # noqa: E402
from e6_merge_eval import merge  # noqa: E402
from c2gap_rerank import SETS, add_rel, featurize, mrr_from_ranks  # noqa: E402

CG = ROOT / "results" / "c2gap"
FULL = ROOT / "results" / "c2gap_full"
LGB = dict(objective="lambdarank", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=25, verbose=-1, num_threads=2)
ROUND = 200


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def feats_of(name):
    f = list(SETS["ho1+struct+pop"])
    if name == "full":
        return f
    f.remove("score")
    if name == "rankz":
        f.remove("score_gap")
    return f


def _pq_read(path):
    import duckdb
    return duckdb.read_parquet(str(path)).df()


def train_model(feats, seed, train_tag=""):
    import lightgbm as lgb
    cand_file = CG / f"cand{train_tag}.parquet"
    mol_file = CG / f"mol{train_tag}.parquet"
    c = _pq_read(cand_file)
    mol = _pq_read(mol_file)
    c = c[c.b.isin(["PC", "S2"])].reset_index(drop=True)
    mol = mol[mol.b.isin(["PC", "S2"])].reset_index(drop=True)
    c = featurize(c)
    c = add_rel(c, mol)
    keep = c.groupby(["scen", "mid"]).is_truth.transform("any")
    c = c[keep].sort_values(["scen", "mid"])
    log(f"train ({cand_file.name}): {c.scen.nunique()} scens, {c.groupby(['scen','mid']).ngroups} molecules, "
        f"{len(c):,} rows, {len(feats)} feats")
    ds = lgb.Dataset(c[feats], c.is_truth.astype(int), group=c.groupby(["scen", "mid"], sort=False).size().values)
    return lgb.train(dict(LGB, seed=seed), ds, num_boost_round=ROUND)


def sv_rows():
    c = _pq_read(CG / "cand_fullSV.parquet")
    mol = _pq_read(CG / "mol_fullSV.parquet")
    return featurize(c), mol


def channel_scores(model, feats, c, P):
    """Booster score for every entry of every channel list, in list order."""
    idx = c.set_index(["mid", "ik"])
    flat = [(mid, p["raw"][j]) for mid, p in P.items() for j in range(len(p["raw"]))]
    want = pd.MultiIndex.from_tuples(flat, names=["mid", "ik"])
    sub = idx.reindex(want)
    miss = int(sub.score.isna().sum())
    pred = model.predict(sub[feats].astype(float))
    out, i = {}, 0
    for mid, p in P.items():
        n = len(p["raw"])
        o = np.argsort(-pred[i:i + n], kind="stable")
        out[mid] = {k: [v[j] for j in o] for k, v in p.items()}
        i += n
    if miss:
        log(f"  WARNING {miss} channel entries missing features (defaulted to NaN path)")
    return out


def engine_lists(scen, base):
    R = pickle.load(open(ROOT / "results" / "bench" / "e1" / f"{scen}.pkl", "rb"))
    out = {}
    for mid in sorted(R):
        r = R[mid]
        _, keys, ix = BE.engine_list(r, base)
        out[mid] = (r, dict(keys=keys, raw=[r["keys"][i] for i in ix]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="full", choices=["full", "rankov", "rankz"])
    ap.add_argument("--train-tag", default="", help="Suffix on cand/mol parquets for training, e.g. _fullS23")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=str(FULL / "sv_gonogo.json"))
    a = ap.parse_args()

    t0 = time.time()
    feats = feats_of(a.feats)
    log(f"feats ({a.feats}): {feats} (train_tag: '{a.train_tag}')")

    model = train_model(feats, a.seed, a.train_tag)
    c, mol = sv_rows()
    c = add_rel(c, mol)
    log(f"SV: {mol.scen.nunique()} scen, {len(mol)} molecules, {len(c):,} candidate rows")

    truth = bench.load_truth()
    tq = truth["SV"]
    E = engine_lists("SV", "blend")
    P0 = pickle.load(open(FULL / "work" / "pc_lists_SV.pkl", "rb"))
    P1 = channel_scores(model, feats, c, P0)

    # ho1-scored SV for the "E6 as shipped" reference column
    P_ho1 = pickle.load(open(ROOT / "results" / "c3" / "e6_work" / "pc_lists_SV.pkl", "rb"))

    mids = sorted(E)
    r_ship, r_base, r_rr, ch_base, ch_rr = [], [], [], [], []
    for mid in mids:
        r, e = E[mid]
        mk, mr = merge(e, P0[mid], 1, "alt")
        r_base.append(BE.rank_in(mk, mr, r, tq[mid]["correct"]))
        mk, mr = merge(e, P1[mid], 1, "alt")
        r_rr.append(BE.rank_in(mk, mr, r, tq[mid]["correct"]))
        mk, mr = merge(e, P_ho1[mid], 1, "alt")
        r_ship.append(BE.rank_in(mk, mr, r, tq[mid]["correct"]))
        p0, p1 = P0[mid], P1[mid]
        ch_base.append(BE.rank_in(p0["keys"], p0["raw"], r, tq[mid]["correct"]))
        ch_rr.append(BE.rank_in(p1["keys"], p1["raw"], r, tq[mid]["correct"]))

    r_ship, r_base, r_rr = map(np.asarray, (r_ship, r_base, r_rr))
    ch_base, ch_rr = map(np.asarray, (ch_base, ch_rr))
    d = np.where(r_rr > 0, 1 / np.where(r_rr > 0, r_rr, 1), 0) - np.where(r_base > 0, 1 / np.where(r_base > 0, r_base, 1), 0)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (a.boot, len(d)))
    means = d[idx].mean(axis=1)
    ci = np.percentile(means, [2.5, 97.5])

    res = dict(feats=a.feats, train_tag=a.train_tag, feature_list=feats, seed=a.seed, n_mol=len(mids),
               mrr=dict(e6_as_shipped=round(mrr_from_ranks(r_ship), 4),
                        fullnet_no_rerank=round(mrr_from_ranks(r_base), 4),
                        fullnet_with_rerank=round(mrr_from_ranks(r_rr), 4),
                        channel_no_rerank=round(mrr_from_ranks(ch_base), 4),
                        channel_with_rerank=round(mrr_from_ranks(ch_rr), 4)),
               delta=round(float(d.mean()), 4), ci95=[round(float(ci[0]), 4), round(float(ci[1]), 4)],
               changed=int((np.abs(d) > 1e-9).sum()), improved=int((d > 1e-9).sum()),
               regressed=int((d < -1e-9).sum()), runtime_s=round(time.time() - t0, 1),
               ranks=dict(mid=mids, e6_as_shipped=[int(x) for x in r_ship],
                          fullnet_no_rerank=[int(x) for x in r_base],
                          fullnet_with_rerank=[int(x) for x in r_rr]))

    go = (res["mrr"]["fullnet_with_rerank"] >= res["mrr"]["e6_as_shipped"]) and (res["ci95"][0] >= -0.005)
    res["decision"] = "GO" if go else "NO-GO"
    json.dump(res, open(a.out, "w"), indent=1)
    log(json.dumps(res, indent=1))
    print(f"\n{res['decision']}  ({a.feats})")


if __name__ == "__main__":
    main()

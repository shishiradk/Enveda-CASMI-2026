"""Offline evaluation of a bench run (results/bench/<tag>/<scen>.pkl): MRR@25 with bootstrap CIs for every score set,
E2-style fusion with our V2 lists over a BETA grid (paired CIs versus BETA 0), and the class-wise error analysis.

    python research/bench/bench.py eval --tag e1 [--score blend]

Score sets (per-candidate scores kept by bench.py run):
  blend        E1 as submitted: 0.88 * rank(pv ranker with public FPNets) + 0.12 * rank(megayak no-FP ranker)
  pv / ours    the two rankers alone;  pv_nofp: pv ranker with the FP features zeroed (out of its training distribution)
  ours_cv      megayak ranker refitted WITHOUT the enveda-np-examples rows (the S1/S2 molecules); blend_cv uses it
  pv_mk        pv ranker fed megayak's leak-free FPNets;  blend_mk / blend_mk_cv accordingly
  pv_cv        pv ranker refitted WITHOUT the rank_train.npz query groups that are the S1/S2 molecules (bench_pvgroups.py)
  clean        0.88 * rank(pv_cv) + 0.12 * rank(ours_cv): E1 with no ranker row of an evaluated S1/S2 molecule
  clean_mk     the same with megayak's FPNets (pv_mk_cv)
  fp_only(_mk) FP channel alone (f.z) with the public / megayak nets: used to size the model leak
A candidate is correct if its tautomer-canonical InChIKey14 (local RDKit) equals the truth's, or its raw InChIKey14 is in
the proxy's alias set (EXP-012 aliases for S1/S2, RDKit tautomer key for S3) - the same rule as proxy_eval2.py.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "bench"
V2P = ROOT / "results" / "kaggle_v2_proxy"
BETAS = (0.0, 0.2, 0.4, 0.6, 1.0)
KRR = 3.0
TIER_BONUS = 0.05  # research/kaggle_v2/casmi_v2_kaggle.py CFG["tier_bonus"], as used by v2_lists.py in E2
NB = 10000
RNG = np.random.default_rng(20261002)


def boot(x, paired_to=None):
    x = np.asarray(x, float)
    d = x - np.asarray(paired_to, float) if paired_to is not None else x
    idx = RNG.integers(0, len(d), (NB, len(d)))
    bs = d[idx].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def fmt(c):
    return f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"


def fmtd(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def engine_list(rec, score, topn=40):
    """eng_runner.py: top-80 by score, drop unparsable / duplicate metric keys, keep 40."""
    if "scores" not in rec or score not in rec["scores"]:
        return [], [], []
    order = np.argsort(-rec["scores"][score], kind="mergesort")[:80]
    smis, keys, idx, seen = [], [], [], set()
    for i in order:
        s, k = rec["top"][int(i)]
        if k is None or k in seen:
            continue
        seen.add(k); smis.append(s); keys.append(k); idx.append(int(i))
        if len(smis) >= topn:
            break
    return smis, keys, idx


def is_ok(key, raw, rec, correct):
    return (key is not None and key == rec["truth_canon"]) or key in correct or (raw is not None and raw in correct)


def rank_in(keys, raws, rec, correct):
    for i, k in enumerate(keys):
        if is_ok(k, raws[i] if raws is not None else None, rec, correct):
            return i + 1
    return 0


def rr(rank, n=25):
    return 1.0 / rank if 0 < rank <= n else 0.0


def our_lists(scen, mids, workers=3):
    """Our V2 engine's lists for the proxy scenario, rebuilt from the proxy cache exactly as v2_lists.py emits them:
    score = cosine + tier bonus for COCONUT/train structures, metric-key dedup, top 40."""
    import duckdb
    sys.path.insert(0, str(HERE))
    cp = OUT / "cache" / f"ours_{scen}.pkl"
    done = pickle.load(open(cp, "rb")) if cp.exists() else {}
    mids = [m for m in mids if m not in done]
    if not mids:
        return done
    c = pickle.load(open(V2P / f"cache_{scen}.pkl", "rb"))["cand"]
    top = {}
    for m in mids:
        cs = [(ik, sc + (TIER_BONUS if src == "u" else 0.0), src) for ik, sc, src, _ in c.get(m, [])]
        top[m] = sorted(cs, key=lambda t: (-t[1], t[0]))[:70]
    need = pd.DataFrame({"ik": sorted({ik for v in top.values() for ik, _, _ in v})})
    U = (ROOT / "results/kaggle_v1_assets/universe.parquet").as_posix()
    PC = (ROOT / "results/kaggle_v2_pubchem/pubchem_rows.parquet").as_posix()
    su = dict(duckdb.sql(f"SELECT ik, any_value(smiles) FROM '{U}' WHERE ik IN (SELECT ik FROM need) GROUP BY 1").fetchall())
    sp = dict(duckdb.sql(f"SELECT ik, arg_min(smiles, cid) FROM '{PC}' WHERE ik IN (SELECT ik FROM need) GROUP BY 1").fetchall())
    import bench
    bench.setup_env(HERE / "eng", workers)
    import casmi_engine as E
    smi_of = {m: [(su.get(ik) if src == "u" else sp.get(ik)) or su.get(ik) or sp.get(ik) for ik, _, src in v]
              for m, v in top.items()}
    C = bench.canon_many(E, [s for v in smi_of.values() for s in v if s], workers)
    out = {}
    for m, v in top.items():
        rec = {"smiles": [], "keys": [], "raw": []}
        for (ik, _, _), s in zip(v, smi_of[m]):
            if not s:
                continue
            k = C.get(s) or ik
            if k in rec["keys"]:
                continue
            rec["smiles"].append(s); rec["keys"].append(k); rec["raw"].append(ik)
            if len(rec["keys"]) >= 40:
                break
        out[m] = rec
    done.update(out)
    pickle.dump(done, open(cp, "wb"))
    return done


def fuse(e, o, beta):
    """The E2 notebook cell: weighted reciprocal-rank fusion, engine weight 1, ours BETA, K = 3, top 40."""
    sc, raw = {}, {}
    for w, L in ((1.0, e), (beta, o)):
        for r, k in enumerate(L.get("keys", []), 1):
            if not k:
                continue
            sc[k] = sc.get(k, 0.0) + w / (KRR + r)
            raw.setdefault(k, (L.get("raw") or [None] * len(L["keys"]))[r - 1])
    order = sorted(sc, key=lambda k: -sc[k])[:40]
    return order, [raw[k] for k in order]


def formula(smi, _c={}):
    if smi not in _c:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        m = Chem.MolFromSmiles(smi) if smi else None
        _c[smi] = CalcMolFormula(m) if m is not None else None
    return _c[smi]


def main(tag="e1", score="blend"):
    sys.path.insert(0, str(HERE))
    import bench
    truth = bench.load_truth()
    d = OUT / tag
    res = {"tag": tag, "score": score, "mrr": {}, "paired": {}, "fusion": {}, "errors": {}}
    rows = []
    for p in sorted(d.glob("S*.pkl")):
        scen = p.stem
        R = pickle.load(open(p, "rb"))
        T = truth["S12" if scen[:2] in ("S1", "S2") else ("SV" if scen == "SV" else "S3")]
        mids = sorted(R)
        per = {}
        for sset in bench.SCORE_SETS:
            if not any("scores" in R[m] and sset in R[m]["scores"] for m in mids):
                continue
            ranks = []
            for m in mids:
                _, keys, idx = engine_list(R[m], sset)
                ranks.append(rank_in(keys, [R[m]["keys"][i] for i in idx], R[m], T[m]["correct"]))
            per[sset] = np.array([rr(r) for r in ranks])
        res["mrr"][scen] = {k: boot(v) for k, v in per.items()}
        res["mrr"][scen]["n"] = len(mids)
        res["paired"][scen] = {f"{a}-{b}": boot(per[a], per[b]) for a, b in
                               (("clean_kf", "blend"), ("pv_kf", "pv"), ("ours_kf", "ours"), ("clean", "blend"), ("clean_mk", "blend"), ("clean_mk", "clean"), ("pv_cv", "pv"),
                                ("blend_cv", "blend"), ("blend_mk", "blend"), ("blend_mk_cv", "blend"), ("ours_cv", "ours"),
                                ("pv_mk", "pv"), ("pv_nofp", "pv"), ("fp_only_mk", "fp_only"), ("pv", "blend"),
                                ("ours", "blend")) if a in per and b in per}
        # ---------------- fusion with our lists ----------------
        cs = scen[:2]
        if (V2P / f"cache_{cs}.pkl").exists():
            O = our_lists(cs, mids, int(os.environ.get('BENCH_WORKERS', 3)))
            fr = {}
            for beta in BETAS:
                v = []
                for m in mids:
                    smis, keys, idx = engine_list(R[m], score)
                    e = dict(keys=keys, raw=[R[m]["keys"][i] for i in idx])
                    order, raws = fuse(e, O.get(m, {"keys": []}), beta)
                    v.append(rr(rank_in(order, raws, R[m], T[m]["correct"])))
                fr[beta] = np.array(v)
            ours_only = np.array([rr(rank_in(O[m]["keys"], O[m]["raw"], R[m], T[m]["correct"])) if m in O else 0.0 for m in mids])
            res["fusion"][scen] = {"ours_only": boot(ours_only),
                                   **{f"beta{b}": boot(fr[b]) for b in BETAS},
                                   **{f"beta{b}-beta0.0": boot(fr[b], fr[0.0]) for b in BETAS if b}}
        # ---------------- error analysis for the chosen score ----------------
        cat = {k: 0 for k in ("not_in_pool", "pool_not_window", "window_gt40", "r26_40", "r2_25", "r1")}
        same_f = {"top40_not1": [0, 0], "r2_25": [0, 0], "r26_40": [0, 0], "window_gt40": [0, 0]}
        raw_ranks, ncand, lib_truth = [], [], []
        for m in mids:
            r = R[m]
            ok = T[m]["correct"]
            smis, keys, idx = engine_list(r, score)
            rk = rank_in(keys, [r["keys"][i] for i in idx], r, ok) if keys else 0
            inwin = "keys" in r and (bool(set(r["keys"]) & ok) or any(k == r["truth_canon"] for _, k in r["top"].values()))
            raw_rank = 0
            if inwin and score in r["scores"]:
                o = np.argsort(-r["scores"][score], kind="mergesort")
                hit = [j for j, i in enumerate(o) if r["keys"][i] in ok]
                raw_rank = hit[0] + 1 if hit else rk
                ti = [i for i in range(len(r["keys"])) if r["keys"][i] in ok]
                lib_truth.append(float(max(r["lib"][i] for i in ti)) if ti else 0.0)
            if rk == 1: c = "r1"
            elif 2 <= rk <= 25: c = "r2_25"
            elif rk > 25: c = "r26_40"
            elif inwin: c = "window_gt40"
            elif r["truth_in_pool"]: c = "pool_not_window"
            else: c = "not_in_pool"
            cat[c] += 1
            ncand.append(len(r.get("keys", [])))
            if c in ("r2_25", "r26_40", "window_gt40") and smis:
                same = int(formula(smis[0]) == formula(T[m]["smiles"]))
                same_f[c][0] += same; same_f[c][1] += 1
                if c != "window_gt40":
                    same_f["top40_not1"][0] += same; same_f["top40_not1"][1] += 1
            rows.append(dict(scen=scen, mid=m, cat=c, rank=rk, raw_rank=raw_rank, n_cand=ncand[-1], n_spec=r["n_spec"],
                             truth_lib_sim=lib_truth[-1] if inwin and score in r["scores"] else np.nan,
                             **{f"rr_{k}": float(v[mids.index(m)]) for k, v in per.items()}))
        n = len(mids)
        res["errors"][scen] = dict(n=n, frac={k: round(v / n, 4) for k, v in cat.items()}, count=cat,
                                   winner_same_formula={k: [v[0], v[1], round(v[0] / v[1], 3) if v[1] else None]
                                                        for k, v in same_f.items()},
                                   median_candidates=float(np.median(ncand)),
                                   mrr_lost={  # MRR points lost per category (1 - rr summed over the category) / n
                                       k: round(float(sum(1 - rr(x["rank"]) for x in rows if x["scen"] == scen and x["cat"] == k) / n), 4)
                                       for k in cat},
                                   truth_has_library_spectrum=round(float(np.mean(np.array(lib_truth) > 0)), 4) if lib_truth else None)
    pd.DataFrame(rows).to_csv(d / f"per_molecule_{score}.csv", index=False)
    json.dump(res, open(d / f"eval_{score}.json", "w"), indent=1)
    # ---------------- report ----------------
    scens = list(res["mrr"])
    print(f"\n## MRR@25 [95% bootstrap CI], tag={tag}\n")
    print("| score set | " + " | ".join(f"{s} (n={res['mrr'][s]['n']})" for s in scens) + " |\n|---|" + "---|" * len(scens))
    for k in bench.SCORE_SETS:
        if any(k in res["mrr"][s] for s in scens):
            print(f"| {k} | " + " | ".join(fmt(res["mrr"][s][k]) if k in res["mrr"][s] else "-" for s in scens) + " |")
    print("\n## paired differences [95% CI]\n")
    ks = sorted({k for s in scens for k in res["paired"][s]})
    print("| a - b | " + " | ".join(scens) + " |\n|---|" + "---|" * len(scens))
    for k in ks:
        print(f"| {k} | " + " | ".join(fmtd(res["paired"][s][k]) if k in res["paired"][s] else "-" for s in scens) + " |")
    print(f"\n## fusion of `{score}` with our V2 lists (weighted RRF, K=3)\n")
    fs = [s for s in scens if s in res["fusion"]]
    print("| | " + " | ".join(fs) + " |\n|---|" + "---|" * len(fs))
    for k in ["ours_only"] + [f"beta{b}" for b in BETAS]:
        print(f"| {k} | " + " | ".join(fmt(res["fusion"][s][k]) for s in fs) + " |")
    for b in BETAS[1:]:
        k = f"beta{b}-beta0.0"
        print(f"| {k} | " + " | ".join(fmtd(res["fusion"][s][k]) for s in fs) + " |")
    print(f"\n## error analysis of `{score}`\n")
    print("| | " + " | ".join(scens) + " |\n|---|" + "---|" * len(scens))
    for k in ("not_in_pool", "pool_not_window", "window_gt40", "r26_40", "r2_25", "r1"):
        print(f"| {k} | " + " | ".join(f"{res['errors'][s]['frac'][k]:.3f} (lost {res['errors'][s]['mrr_lost'][k]:.3f})" for s in scens) + " |")
    for k in ("top40_not1", "r2_25", "r26_40", "window_gt40"):
        print(f"| winner same formula, {k} | " + " | ".join("{}/{}".format(*res["errors"][s]["winner_same_formula"][k][:2]) for s in scens) + " |")
    print("| median candidates | " + " | ".join(str(res["errors"][s]["median_candidates"]) for s in scens) + " |")
    print("| truth has library spectrum (of in-window) | " + " | ".join(str(res["errors"][s]["truth_has_library_spectrum"]) for s in scens) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Bench check of the E4 stage (run: python research/kaggle_e1/e4/bench_e4.py).

E4's own stage functions on the bench lists of the E3 diagnosis (results/bench/fm; honest ranker scores clean_kf /
pv_kf / ours_kf; Kaggle forward scores) with the shipped lookup.  Paired MRR@25 differences against E3b.

The bench truths are textbook natural products (research/analysis/exp020_popularity_prior.md section 4), so the
'actual' rows are an upper bound.  Counterfactual rows replace the TRUTH's counts only:
  cfcoco: counts of a random COCONUT pool structure (mean over 5 draws),  cfmin: 1 substance record, no PubMed link.
Writes results/kaggle_e4/bench_e4.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "e3"))
sys.path.insert(0, str(HERE.parent / "e3b"))
sys.path.insert(0, str(HERE))
sys.path.insert(3, str(ROOT / "research" / "bench" / "fm"))
import e4_stage as E  # noqa: E402
import pop_stage as P  # noqa: E402
import fm_lib as F  # noqa: E402

LAMS = {"ice": 0.5, "gl": 0.5}


def K(i):
    return "K" * 12 + chr(65 + i // 26) + chr(65 + i % 26)


def rr(ok_by_smiles, smiles):
    for i, s in enumerate(smiles[:25]):
        if ok_by_smiles[s]:
            return 1.0 / (i + 1)
    return 0.0


def main():
    import pandas as pd
    lk = P.load_lookup(ROOT / "results" / "kaggle_e4" / "pop_lookup" / "pop_lookup.npz")
    D, _ = F.load()
    pool = pd.read_parquet(ROOT / "results" / "kaggle_e4" / "pool_pop_rows.parquet", columns=["key", "src", "g_sid", "g_pmid"])
    coco = pool[pool.src == "coconut"].drop_duplicates("key")[["g_sid", "g_pmid"]].values
    rng = np.random.default_rng(0)
    cache, out, truth_pop, cand_pop = {}, {}, [], []
    for scen in ("S1", "S2"):
        mols = []
        for m in sorted(D[scen]):
            l = D[scen][m]["lists"].get("clean_kf")
            if l is None or not l["smiles"]:
                mols.append(None)
                continue
            n = len(l["smiles"])
            cnt = []
            for s in l["smiles"]:
                if s not in cache:
                    cache[s] = P.candidate_counts(s, None, lk)
                cnt.append(cache[s][:2])
            e = dict(smiles=list(l["smiles"]), keys=[K(i) for i in range(n)], scores=list(l["scores"]), lib=list(l["lib"]),
                     pv=list(l["pv_kf"]), ours=list(l["ours_kf"]))
            by = {t: {"m": dict(zip(l["smiles"], l[t]))} for t in ("ice", "gl")} if "ice" in l else {}
            mols.append((e, cnt, dict(zip(l["smiles"], l["ok"])), by))
            if scen == "S1":
                truth_pop += [P.pop_value(*c) for c, ok in zip(cnt, l["ok"]) if ok]
                cand_pop += [P.pop_value(*c) for c, ok in zip(cnt[:60], l["ok"][:60]) if not ok]

        def run(mu, mode, cf=None, forward=True):
            vals, dem, pro = [], 0, 0
            for item in mols:
                if item is None:
                    vals.append((0.0, 0.0))
                    continue
                e, cnt, ok, by = item
                cnt = list(cnt)
                if cf is not None:
                    for i, s in enumerate(e["smiles"]):
                        if ok[s]:
                            cnt[i] = (1, 0) if cf == "min" else tuple(int(x) for x in coco[rng.integers(len(coco))])
                look = P.PopLookup(np.array(e["keys"], dtype="S14"), [c[0] for c in cnt], [c[1] for c in cnt])
                base = E.apply_forward({"m": e}, {"m"}, by, LAMS)[0]["m"] if (by and forward) else e
                pp, _ = P.run_pop_stage({"m": e}, look, mu=mu, score_mode=mode, key_fn=None)
                new = E.apply_forward(pp, {"m"}, by, LAMS)[0]["m"] if (by and forward) else pp["m"]
                a, b = rr(ok, base["smiles"]), rr(ok, new["smiles"])
                vals.append((a, b)); dem += int(a == 1 and b < 1); pro += int(a < 1 and b == 1)
            v = np.array(vals)
            return dict(e3b=round(float(v[:, 0].mean()), 4), delta=F.boot(v[:, 1] - v[:, 0]), demoted=dem, promoted=pro)

        for mode in ("logit", "rank"):
            for mu in (0.0, 0.1, 0.15, 0.25, 0.4):
                r = run(mu, mode)
                assert mu > 0 or (r["delta"][0] == 0 and r["demoted"] == 0 and r["promoted"] == 0)
                row = {"actual": r}
                if mu > 0:
                    row["cfmin"] = run(mu, mode, "min")
                    reps = [run(mu, mode, "coco") for _ in range(5)]
                    row["cfcoco"] = dict(delta_mean=round(float(np.mean([x["delta"][0] for x in reps])), 4),
                                         delta_range=[round(min(x["delta"][0] for x in reps), 4), round(max(x["delta"][0] for x in reps), 4)])
                out[f"{scen}|{mode}|mu={mu}"] = row
                print(scen, mode, "mu", mu, "| E3b", r["e3b"], "| E4 - E3b", F.fmtd(r["delta"]), "| top-1 truths demoted / promoted", r["demoted"], r["promoted"],
                      ("| cfmin " + F.fmtd(row["cfmin"]["delta"]) + f" dem {row['cfmin']['demoted']}" + " | cfcoco " + f"{row['cfcoco']['delta_mean']:+.3f}") if mu > 0 else "", flush=True)
    q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 25, 50, 75, 90])]  # noqa: E731
    out["pop_quantiles_10_25_50_75_90"] = {"bench_truths": q(truth_pop), "other_listed_candidates": q(cand_pop),
                                           "coconut_pool": q(np.log1p(coco[:, 0]) + np.log1p(coco[:, 1]))}
    print(out["pop_quantiles_10_25_50_75_90"])
    json.dump(out, open(ROOT / "results" / "kaggle_e4" / "bench_e4.json", "w"), indent=1)


if __name__ == "__main__":
    main()

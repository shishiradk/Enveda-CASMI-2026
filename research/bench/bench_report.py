"""Checks and the five-way error table on top of bench_eval.py (reads results/bench/<tag>/<scen>.pkl only).

    python research/bench/bench_report.py [TAG=e1] [SCORE=blend]

Adds what bench_eval.py does not print:
  1. key audit: MRR with the strict metric rule (tautomer-canonical InChIKey14 of the candidate == that of the truth,
     both from the local RDKit) versus the proxy rule bench_eval uses (strict OR raw InChIKey14 in the alias set);
  2. pool-provenance audit: targets whose truth is in the bench pool only as a train-origin structure, i.e. kept
     because OUR COCONUT copy has it although the candidate files E1 ships (prvsiyan COCONUT + ChEBI/LIPID MAPS) do
     not. On Kaggle such a Class-2 molecule would be unreachable, so "MRR, shipped pool only" zeroes them;
  3. the five-way error table asked for in the report (absent from pool / in pool but not in the top 40 / 26-40 /
     2-25 / rank 1) and the same-formula share of the rank-1 candidate when the truth is listed but not first;
  4. the fingerprint-channel leak diagnostics: FP-only MRR with the public and the megayak nets, FP gain of the pv
     ranker (pv - pv_nofp) per scenario.
Writes results/bench/<tag>/report_<score>.json and prints markdown.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
import bench_eval as be  # noqa: E402


def main(tag="e1", score="blend"):
    truth = bench.load_truth()
    d = bench.OUT / tag
    out = {}
    for p in sorted(d.glob("S*.pkl")):
        scen = p.stem
        R = pickle.load(open(p, "rb"))
        T = truth["S12" if scen[:2] in ("S1", "S2") else "S3"]
        mids = sorted(R)
        lax, strict, shipped, cats, samef, n_alias_only, tr_only = [], [], [], [], [], 0, 0
        for m in mids:
            r, ok = R[m], T[m]["correct"]
            smis, keys, idx = be.engine_list(r, score)
            raws = [r["keys"][i] for i in idx]
            rk = be.rank_in(keys, raws, r, ok)
            rk_s = next((i + 1 for i, k in enumerate(keys) if k == r["truth_canon"]), 0)
            lax.append(be.rr(rk)); strict.append(be.rr(rk_s))
            n_alias_only += int(be.rr(rk) != be.rr(rk_s))
            # provenance of the truth inside the candidate window
            ti = [i for i in range(len(r.get("keys", []))) if r["keys"][i] in ok]
            # the shipped files may hold the same structure under another raw key: match by metric key as well
            tj = sorted(set(ti) | {i for i, (_, k) in r.get("top", {}).items() if k == r["truth_canon"]})
            only_train = bool(tj) and all(bool(r["from_train"][i]) for i in tj)
            tr_only += int(only_train)
            shipped.append(0.0 if only_train else be.rr(rk))
            inwin = "keys" in r and (bool(ti) or any(k == r["truth_canon"] for _, k in r["top"].values()))
            c = ("rank1" if rk == 1 else "rank2_25" if 2 <= rk <= 25 else "rank26_40" if rk > 25 else
                 "in_pool_not_top40" if (inwin or r["truth_in_pool"]) else "absent_from_pool")
            cats.append(c)
            if c in ("rank2_25", "rank26_40") and smis:
                samef.append(int(be.formula(smis[0]) == be.formula(T[m]["smiles"])))
        n = len(mids)
        frac = {c: round(cats.count(c) / n, 4) for c in ("absent_from_pool", "in_pool_not_top40", "rank26_40", "rank2_25", "rank1")}
        lost = {c: round(float(sum(1 - x for x, k in zip(lax, cats) if k == c)) / n, 4) for c in frac}
        o = dict(n=n, mrr=be.boot(lax), mrr_strict_key=be.boot(strict), molecules_where_rules_differ=n_alias_only,
                 truth_train_origin_only=tr_only, mrr_shipped_pool_only=be.boot(shipped), frac=frac, mrr_lost=lost,
                 listed_not_first=len(samef), rank1_same_formula=int(sum(samef)),
                 rank1_same_formula_frac=round(float(np.mean(samef)), 3) if samef else None)
        per = {}
        for s in ("fp_only", "fp_only_mk", "pv", "pv_nofp", "pv_mk", "pv_cv", "clean", "clean_mk", "blend"):
            if any("scores" in R[m] and s in R[m]["scores"] for m in mids):
                v = []
                for m in mids:
                    _, keys, idx = be.engine_list(R[m], s)
                    v.append(be.rr(be.rank_in(keys, [R[m]["keys"][i] for i in idx], R[m], T[m]["correct"])))
                per[s] = np.array(v)
        o["leak"] = {k: be.boot(v) for k, v in per.items()}
        for a, b in (("pv", "pv_nofp"), ("pv_mk", "pv_nofp"), ("fp_only", "fp_only_mk"), ("blend", "clean"), ("blend", "clean_mk")):
            if a in per and b in per:
                o["leak"][f"{a}-{b}"] = be.boot(per[a], per[b])
        out[scen] = o
    json.dump(out, open(d / f"report_{score}.json", "w"), indent=1)
    sc = list(out)
    print(f"\n## five-way error table of `{score}` (share of molecules; MRR points lost)\n")
    print("| | " + " | ".join(f"{s} (n={out[s]['n']})" for s in sc) + " |\n|---|" + "---|" * len(sc))
    for c in ("absent_from_pool", "in_pool_not_top40", "rank26_40", "rank2_25", "rank1"):
        print(f"| {c} | " + " | ".join(f"{out[s]['frac'][c]:.3f} ({out[s]['mrr_lost'][c]:.3f})" for s in sc) + " |")
    print("| rank-1 has the truth's formula (truth listed, not first) | " +
          " | ".join(f"{out[s]['rank1_same_formula']}/{out[s]['listed_not_first']}" for s in sc) + " |")
    print("\n## audits\n")
    print("| | " + " | ".join(sc) + " |\n|---|" + "---|" * len(sc))
    for k, f in (("mrr", be.fmt), ("mrr_strict_key", be.fmt), ("mrr_shipped_pool_only", be.fmt)):
        print(f"| {k} | " + " | ".join(f(out[s][k]) for s in sc) + " |")
    for k in ("molecules_where_rules_differ", "truth_train_origin_only"):
        print(f"| {k} | " + " | ".join(str(out[s][k]) for s in sc) + " |")
    print("\n## fingerprint-channel diagnostics\n")
    ks = []
    for s in sc:
        ks += [k for k in out[s]["leak"] if k not in ks]
    print("| | " + " | ".join(sc) + " |\n|---|" + "---|" * len(sc))
    for k in ks:
        print(f"| {k} | " + " | ".join((be.fmtd if "-" in k else be.fmt)(out[s]["leak"][k]) if k in out[s]["leak"] else "-" for s in sc) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:3])

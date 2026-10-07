#!/usr/bin/env python
"""Rank of the true structure among same-formula isomers by forward-model score alone.

Usage: python eval_validation.py --scores results/forward_model/val_scores.json [--key scores|cosine]
Ties get the mid-rank.  The random baseline is the exact expectation for a
uniformly random order of the same candidate set (MRR = H_n / n, top-1 = 1/n).
95% CIs: percentile bootstrap over molecules (5000 resamples, fixed seed).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def rank_of_truth(scores, truth_idx, keep):
    idx = [i for i in range(len(scores)) if keep[i] and scores[i] is not None]
    if truth_idx not in idx or len(idx) < 2:
        return None
    s = np.array([scores[i] for i in idx], dtype=float)
    t = scores[truth_idx]
    rank = 1 + float((s > t).sum()) + 0.5 * float((s == t).sum() - 1)
    return rank, len(idx)


def summarise(rows, rng, label):
    if not rows:
        return {"subset": label, "n": 0}
    rr = np.array([1.0 / r for r, _ in rows])
    top1 = np.array([1.0 if r < 1.5 else 0.0 for r, _ in rows])
    top3 = np.array([1.0 if r <= 3 else 0.0 for r, _ in rows])
    rnd_rr = np.array([sum(1.0 / k for k in range(1, n + 1)) / n for _, n in rows])
    rnd_t1 = np.array([1.0 / n for _, n in rows])
    n = len(rows)
    boot = rng.integers(0, n, size=(5000, n))

    def ci(x):
        m = x[boot].mean(1)
        return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]

    return {"subset": label, "n": n, "mean_candidates": round(float(np.mean([c for _, c in rows])), 2),
            "mrr": round(float(rr.mean()), 4), "mrr_ci95": ci(rr),
            "top1": round(float(top1.mean()), 4), "top1_ci95": ci(top1),
            "top3": round(float(top3.mean()), 4),
            "random_mrr": round(float(rnd_rr.mean()), 4), "random_top1": round(float(rnd_t1.mean()), 4),
            "mrr_minus_random": round(float((rr - rnd_rr).mean()), 4), "mrr_minus_random_ci95": ci(rr - rnd_rr),
            "top1_minus_random": round(float((top1 - rnd_t1).mean()), 4), "top1_minus_random_ci95": ci(top1 - rnd_t1),
            "median_rank": float(np.median([r for r, _ in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth", default=str(ROOT / "results" / "forward_model" / "val_truth.json"))
    ap.add_argument("--key", default="scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = json.load(open(args.scores))
    truth = json.load(open(args.truth))
    sc = res[args.key]
    rng = np.random.default_rng(0)
    subsets = {"all": [], "truth_not_in_MassSpecGym": [], "truth_in_MassSpecGym": [],
               "vs_similar_decoys_only": [], "vs_random_decoys_only": [],
               "vs_similar_decoys_only__not_in_MSG": []}
    n_total = n_scored = 0
    for mid, t in truth.items():
        if mid not in sc:
            continue
        n_total += 1
        kinds = t["kinds"]
        full = rank_of_truth(sc[mid], t["truth_index"], [True] * len(kinds))
        if full is None:
            continue
        n_scored += 1
        subsets["all"].append(full)
        subsets["truth_in_MassSpecGym" if t["msg_folds"] else "truth_not_in_MassSpecGym"].append(full)
        for name, kind in (("vs_similar_decoys_only", "similar"), ("vs_random_decoys_only", "random")):
            r = rank_of_truth(sc[mid], t["truth_index"], [k in ("truth", kind) for k in kinds])
            if r is not None:
                subsets[name].append(r)
                if name == "vs_similar_decoys_only" and not t["msg_folds"]:
                    subsets["vs_similar_decoys_only__not_in_MSG"].append(r)
    out = {"key": args.key, "molecules_in_truth": n_total, "molecules_with_truth_scored": n_scored,
           "runner_meta": {k: v for k, v in res.get("meta", {}).items() if k != "argv"},
           "results": [summarise(v, rng, k) for k, v in subsets.items()]}
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

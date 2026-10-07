"""EXP-011 post-hoc: how would ranking quality change with larger candidate pools (COCONUT/PubChem-sized)?

Model (PROJECTION, not measurement): each extra candidate outranks the target independently with the
probability q the target showed inside its own pool. rank = 1 + Binomial(N-1, q), so
E[RR] = (1 - (1-q)^N) / (N q). q is estimated as (expected_rank - 0.5) / n  (smoothed, so rank 1 in a
tiny pool is not treated as q = 0).
Self-check: on T2 (larger pools), estimate q from random subsamples of ~9 candidates (the T1 median pool),
project back to each molecule's true pool size, and compare with the measured MRR.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp011_class2_proxy as E  # noqa: E402

O = E.outdir(False)
NS = (10, 25, 50, 100, 250, 500, 1000, 5000)


def err(q, N):
    q = np.clip(q, 1e-9, 1)
    return (1 - (1 - q) ** N) / (N * q)


def main():
    base = json.load(open(O / "rank_B012.json"))
    P = E.pools_for(O)
    F = E.FP(O)
    out = {}
    for tag in ("B3", "B4"):
        f = O / f"pred_{tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        pred = dict(zip(z["iks"], z["pred"]))
        rng = np.random.default_rng(E.SEED + 11)
        for pop in ("T1_np", "T2_tims"):
            qs, qs_sub, n_true, rr_true = [], [], [], []
            for ik, d in P.items():
                pool = d[E.PRIMARY_PPM]
                if d["pop"] != pop or ik not in pool or ik not in pred:
                    continue
                sc = dict(zip(pool, E.cosine_scores(pred[ik], F.bits(pool))))
                st = E.expected_rank_stats(sc, ik)
                n = len(pool)
                qs.append((st["rank"] - 0.5) / n)
                n_true.append(n)
                rr_true.append(st["rr"])
                others = [c for c in pool if c != ik]
                if len(others) >= 8:  # self-check: subsample to 9 candidates incl. target
                    sub = [ik] + list(rng.choice(others, 8, replace=False))
                    s2 = E.expected_rank_stats({c: sc[c] for c in sub}, ik)
                    qs_sub.append(((s2["rank"] - 0.5) / 9, n, st["rr"]))
            qs = np.array(qs)
            R = {"n": len(qs), "measured_mrr_cond": float(np.mean(rr_true)),
                 "projected_mrr_cond_by_pool_size": {N: float(np.mean(err(qs, N))) for N in NS},
                 "chance_mrr_by_pool_size": {N: float(sum(1 / r for r in range(1, N + 1)) / N) for N in NS}}
            if qs_sub:
                qsub = np.array([x[0] for x in qs_sub]); nsz = np.array([x[1] for x in qs_sub])
                R["selfcheck_from_9_candidate_subsamples"] = {
                    "n": len(qs_sub), "measured_mrr_cond": float(np.mean([x[2] for x in qs_sub])),
                    "projected_mrr_cond": float(np.mean(err(qsub, nsz)))}
            out[f"{tag}_{pop}"] = R
    json.dump(out, open(O / "pool_projection.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

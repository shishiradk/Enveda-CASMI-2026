"""Second stage: a small family of "safe" rules around the two fixes suggested by the diagnosis
(ranker term from the ranker's probability instead of its rank; do not touch molecules / candidates with strong library
evidence), selected on one half of the molecules and reported on the other.

    python research/bench/fm/fm_reco.py [score variant, default runner score]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_lib as F  # noqa: E402
from fm_analysis import SC, attach_rescore, crossfit, show  # noqa: E402


def rule(l, e, lam=0.5, term="rank", gate=None, protect=None, cols=("ice", "gl"), lam_gl=None):
    """term 'rank': z of the exported rank blend (E3) | 'logit': z of logit(pv probability).
    gate: skip the molecule when its best library similarity >= gate.
    protect: candidates whose own library similarity >= protect keep their slot (they leave the group)."""
    n = len(l["smiles"])
    if gate is not None and e["lib_max"] >= gate:
        return list(range(n))
    if protect is None:
        return F.general_perm(l, lam, lam if lam_gl is None else lam_gl, fm_cols=cols,
                              ranker="pv_kf" if term == "logit" else "scores", rank_term="logit" if term == "logit" else "z")
    l2 = dict(l)
    l2["formula"] = [None if l["lib"][i] >= protect else f for i, f in enumerate(l["formula"])]
    return F.general_perm(l2, lam, lam if lam_gl is None else lam_gl, fm_cols=cols,
                          ranker="pv_kf" if term == "logit" else "scores", rank_term="logit" if term == "logit" else "z")


def ev(D, scen, kw, ss="clean_kf"):
    mids = sorted(D[scen])
    out = []
    for m in mids:
        e = D[scen][m]
        l = e["lists"].get(ss)
        if l is None or not l["smiles"] or "ice" not in l:
            out.append(0.0)
            continue
        out.append(F.rr_of(l["ok"], rule(l, e, **kw)) - F.rr_of(l["ok"]))
    return np.array(out)


def main(variant=None):
    D, _ = F.load()
    attach_rescore(D)
    cols = ("ice", "gl") if not variant else (f"ice:{variant}", f"gl:{variant}")
    mids = sorted(D["S1"])
    fam = {}
    for term, lam, gate, protect in itertools.product(("rank", "logit"), (0.1, 0.25, 0.5, 1.0), (None, 0.6, 0.7, 0.8), (None, 0.6, 0.7, 0.8)):
        if gate is not None and protect is not None:
            continue
        fam[f"{term} lam={lam} gate={gate} protect={protect}"] = dict(lam=lam, term=term, gate=gate, protect=protect, cols=cols)
    for term, (li, lg) in itertools.product(("rank", "logit"), ((0.5, 0.0), (0.0, 0.5), (1.0, 0.0), (0.0, 1.0), (0.25, 0.0), (0.0, 0.25))):
        fam[f"{term} lam_ice={li} lam_gl={lg}"] = dict(lam=li, lam_gl=lg, term=term, cols=cols)
    R = {n: {s: ev(D, s, kw) for s in SC} for n, kw in fam.items()}
    print("| setting | S1 | S2 | mean of S1, S2 |\n|---|---|---|---|")
    store = {}
    for n in R:
        c = show(n, R[n])
        store[n] = {**{s: c[s] for s in SC}, "mean": F.boot(np.mean([R[n][s] for s in SC], 0))}
    # ---- one fixed split (the pre-declared protocol) ----
    rng = np.random.default_rng(20261002)
    perm = rng.permutation(len(mids))
    A, Bh = perm[:125], perm[125:]
    avg = {n: np.mean([R[n][s] for s in SC], 0) for n in R}
    out = {"family": store, "split": {}}
    for nameA, a, b in (("A->B", A, Bh), ("B->A", Bh, A)):
        best = max(R, key=lambda n: avg[n][a].mean())
        safe = max(R, key=lambda n: (min(R[n]["S1"][a].mean(), 0.0), R[n]["S2"][a].mean()))   # no S1 loss first, then S2 gain
        print(f"\nsplit {nameA}: chosen on the selection half = `{best}` (selection-half mean {avg[best][a].mean():+.3f})")
        print("| held-out half | S1 | S2 | mean |\n|---|---|---|---|")
        for tag, n in (("best mean", best), ("no-S1-loss then S2", safe), ("E3 rule (rank lam=1.0)", "rank lam=1.0 gate=None protect=None")):
            cells = [F.boot(R[n]["S1"][b]), F.boot(R[n]["S2"][b]), F.boot(avg[n][b])]
            print(f"| {tag}: {n} | " + " | ".join(F.fmtd(c) for c in cells) + " |")
            out["split"].setdefault(nameA, {})[tag] = dict(setting=n, S1=cells[0], S2=cells[1], mean=cells[2])
    held, picks = crossfit(D, R, mids)
    print("\nrepeated cross-fitting (20 random splits, both directions), all 250 molecules held out:")
    print("| | S1 | S2 | mean |\n|---|---|---|---|")
    c = show("held-out, safe family", held)
    out["crossfit"] = {**{s: c[s] for s in SC}, "mean": F.boot(np.mean([held[s] for s in SC], 0)), "picks": picks}
    print("picked:", picks)
    # as-submitted (leaky) ranker scores for the short list of interest
    print("\nsame settings on the as-submitted ranker scores (blend lists; pv instead of pv_kf is not exported there -> rank term only)")
    json.dump(out, open(F.FM / f"reco{'_' + variant if variant else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

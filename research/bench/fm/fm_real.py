"""Real-score ranker term for the forward-model re-ordering (E3c study, research/analysis/e3c_real_score.md).

    python research/bench/fm/fm_real.py            -> results/bench/fm/real_score.txt / real_score.json / real_score_delta.pkl

The engine's exported score is a within-molecule rank blend.  Here the ranker term is built from the two rankers' RAW
outputs (mean predict_proba of the HistGradientBoosting ensembles; the query-held-out refit pv_kf / ours_kf of
bench_cvrank.py) and combined with the forward-model scores inside formula groups.  The base list is always the honest
E1 list (clean_kf order), so "no re-ordering" is E1 and every rule is slot-preserving like E3 / E3b.

Ranker terms (group = same formula inside the top 60, protected candidates removed):
  rank     z over the group of the exported rank blend                      (E3, E3b)
  zl_pv    z over the group of logit(pv)                                    (diagnosis variant)
  zl_bl    z over the group of L = 0.88 logit(pv) + 0.12 logit(ours)
  zl_list  (L - mean) / sd over the WHOLE top-60 list (one scale per molecule, margins between groups kept)
  abs      L / T: absolute logit units, no standardisation at all (T = temperature)
fused = ranker term + lam * (z(ICEBERG) + z(GLACIER)), z over the group, ddof 1, as in fm_rerank.
Gate (b): the forward term is applied to a group only if the ranker margin between its two best members is below a
threshold (margin in probability of pv, or in L).
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_lib as F  # noqa: E402

SC = ("S1", "S2")
W_PV = 0.88
TOPN = 60
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def zs(x):
    """fm_rerank.zscores: z over the finite values (ddof 1), 0 for missing; covered flag."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = np.zeros(len(x))
    if ok.sum() < 2:
        return out, False
    sd = x[ok].std(ddof=1)
    if not sd > 1e-12:
        return out, False
    out[ok] = (x[ok] - x[ok].mean()) / sd
    return out, True


def prep(l, pv="pv_kf", ours="ours_kf"):
    """Arrays of one list (cached in the list dict)."""
    key = "_prep_" + pv
    if key in l:
        return l[key]
    n = len(l["smiles"])
    top = min(n, TOPN)
    d = dict(n=n, top=top, ok=l["ok"], base=np.asarray(l["scores"], float)[:top],
             ppv=np.asarray(l[pv], float)[:top], pours=np.asarray(l[ours], float)[:top],
             lib=np.asarray(l["lib"], float)[:top], formula=l["formula"][:top],
             ice=F.col(l, "ice")[:top], gl=F.col(l, "gl")[:top])
    d["lpv"], d["lours"] = logit(d["ppv"]), logit(d["pours"])
    d["L"] = W_PV * d["lpv"] + (1 - W_PV) * d["lours"]
    d["pbl"] = W_PV * d["ppv"] + (1 - W_PV) * d["pours"]
    sd = d["L"].std(ddof=1) if top > 1 else 0.0
    d["Lz_list"] = (d["L"] - d["L"].mean()) / sd if sd > 1e-12 else np.zeros(top)
    d["groups"] = {}
    l[key] = d
    return d


def groups_of(d, protect):
    if protect not in d["groups"]:
        g = {}
        for i in range(d["top"]):
            f = d["formula"][i]
            if f is None or (protect is not None and d["lib"][i] >= protect):
                continue
            g.setdefault(f, []).append(i)
        out = []
        for mem in g.values():
            if len(mem) < 2:
                continue
            mem = np.array(mem)
            zi, ci = zs(d["ice"][mem])
            zg, cg = zs(d["gl"][mem])
            out.append((mem, zi, ci, zg, cg))
        d["groups"][protect] = out
    return d["groups"][protect]


def perm_of(d, term="rank", lam=0.5, lam_gl=None, protect=None, T=1.0, gate=None, gate_col="ppv", lam0_keep=True):
    """Slot-preserving permutation.  gate: (threshold) on the margin of gate_col between the group's two best members."""
    perm = list(range(d["n"]))
    lg = lam if lam_gl is None else lam_gl
    for mem, zi, ci, zg, cg in groups_of(d, protect):
        if not ((ci and lam) or (cg and lg)):
            continue
        if gate is not None:
            s = np.sort(d[gate_col][mem])[::-1]
            if s[0] - s[1] >= gate:
                continue
        if term == "rank":
            r = zs(d["base"][mem])[0]
        elif term == "zl_pv":
            r = zs(d["lpv"][mem])[0]
        elif term == "zl_bl":
            r = zs(d["L"][mem])[0]
        elif term == "zl_list":
            r = d["Lz_list"][mem]
        elif term == "abs":
            r = d["L"][mem] / T
        elif term == "abs_pv":
            r = d["lpv"][mem] / T
        else:
            raise ValueError(term)
        fused = r + (lam * zi if ci else 0.0) + (lg * zg if cg else 0.0)
        fused = np.round(fused / 1e-9) * 1e-9
        order = sorted(range(len(mem)), key=lambda j: (-fused[j], j))
        for slot, j in zip(mem, order):
            perm[int(slot)] = int(mem[j])
    return perm


def deltas(D, scen, kw, ss="clean_kf", pv="pv_kf", ours="ours_kf"):
    """Per-molecule RR@25 difference versus no re-ordering (0 for molecules without a list / forward scores)."""
    out = []
    for m in sorted(D[scen]):
        l = D[scen][m]["lists"].get(ss)
        if l is None or not l["smiles"] or "ice" not in l:
            out.append(0.0)
            continue
        d = prep(l, pv, ours)
        out.append(F.rr_of(l["ok"], perm_of(d, **kw)) - F.rr_of(l["ok"]))
    return np.array(out)


def family():
    fam = {}
    lams = (0.1, 0.25, 0.5, 1.0, 2.0)
    for term, lam, prot in itertools.product(("rank", "zl_pv", "zl_bl", "zl_list"), lams, (None, 0.6)):
        fam[f"{term} lam={lam} protect={prot}"] = dict(term=term, lam=lam, protect=prot)
    # absolute logits: fused = L / T + lam z  ==  L + (lam T) z ; one free parameter a = lam * T (logit units per forward z)
    for a, prot in itertools.product((0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), (None, 0.6)):
        fam[f"abs a={a} protect={prot}"] = dict(term="abs", lam=a, T=1.0, protect=prot)
    # (b) margin gates
    for term, lam, prot, (gc, gt) in itertools.product(("rank", "zl_bl"), (0.25, 0.5, 1.0), (None, 0.6),
                                                       (("ppv", 0.1), ("ppv", 0.2), ("ppv", 0.3), ("L", 1.0), ("L", 2.0), ("L", 3.0))):
        fam[f"{term} lam={lam} protect={prot} gate {gc}<{gt}"] = dict(term=term, lam=lam, protect=prot, gate=gt, gate_col=gc)
    for a, prot, (gc, gt) in itertools.product((1.0, 2.0, 4.0), (None, 0.6), (("ppv", 0.2), ("L", 2.0))):
        fam[f"abs a={a} protect={prot} gate {gc}<{gt}"] = dict(term="abs", lam=a, protect=prot, gate=gt, gate_col=gc)
    return fam


E3B = "rank lam=0.5 protect=0.6"
E3 = "rank lam=1.0 protect=None"


def crossfit(R, names, n, reps=20, seed=7, crit="mean"):
    """Repeated 2-fold: choose among `names` on one half, apply to the other.  -> held-out per-molecule deltas, picks."""
    rng = np.random.default_rng(seed)
    held = {s: np.zeros(n) for s in SC}
    picks = {}
    M1 = np.array([R[k]["S1"] for k in names]); M2 = np.array([R[k]["S2"] for k in names])
    for _ in range(reps):
        p = rng.permutation(n)
        halves = (p[: n // 2], p[n // 2:])
        for a, b in (halves, halves[::-1]):
            s1, s2 = M1[:, a].mean(1), M2[:, a].mean(1)
            if crit == "mean":
                sc = (s1 + s2) / 2
            elif crit == "safe":           # no S1 loss first, then the S2 gain
                sc = np.minimum(s1, 0.0) * 1000 + s2
            else:                          # "min": the worse of the two scenarios (consistency)
                sc = np.minimum(s1, s2)
            best = names[int(np.argmax(sc))]
            picks[best] = picks.get(best, 0) + 1
            held["S1"][b] += R[best]["S1"][b] / reps
            held["S2"][b] += R[best]["S2"][b] / reps
    return held, dict(sorted(picks.items(), key=lambda kv: -kv[1]))


def main():
    D, _ = F.load()
    n = len(D["S1"])
    fam = family()
    fam["abs a=0 (raw-score blend order inside groups, no forward term)"] = None
    R = {}
    for name, kw in fam.items():
        if kw is None:
            continue
        R[name] = {s: deltas(D, s, kw) for s in SC}
    # lam = 0 control: what the raw-score order alone does inside formula groups (forward term multiplied by 1e-9)
    R["control: raw-score blend order, forward term off"] = {s: deltas(D, s, dict(term="abs", lam=1e-9, protect=None)) for s in SC}
    R["control: raw-score blend order, forward term off, protect 0.6"] = {s: deltas(D, s, dict(term="abs", lam=1e-9, protect=0.6)) for s in SC}
    lines = []

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    base = {s: np.array([F.rr_of(D[s][m]["lists"]["clean_kf"]["ok"]) if D[s][m]["lists"].get("clean_kf") else 0.0 for m in sorted(D[s])]) for s in SC}
    pr("baseline MRR@25 (clean_kf):", {s: round(float(base[s].mean()), 4) for s in SC})
    pr("\n## full-sample table: versus no re-ordering (E1) and versus E3b; 10,000-resample paired bootstrap\n")
    pr("| setting | S1 vs E1 | S2 vs E1 | mean vs E1 | S1 vs E3b | S2 vs E3b | mean vs E3b |\n|---|---|---|---|---|---|---|")
    store = {}
    for name, r in R.items():
        c = [F.boot(r["S1"]), F.boot(r["S2"]), F.boot((r["S1"] + r["S2"]) / 2),
             F.boot(r["S1"] - R[E3B]["S1"]), F.boot(r["S2"] - R[E3B]["S2"]), F.boot((r["S1"] + r["S2"] - R[E3B]["S1"] - R[E3B]["S2"]) / 2)]
        store[name] = dict(zip(("S1", "S2", "mean", "S1_vs_e3b", "S2_vs_e3b", "mean_vs_e3b"), c))
        pr(f"| {name} | " + " | ".join(F.fmtd(x) for x in c) + " |")
    out = {"baseline": {s: float(base[s].mean()) for s in SC}, "full": store, "crossfit": {}}

    pr("\n## selection on one half, report on the other (20 random splits x both directions; every molecule held out 20 times)\n")
    pr("| family | criterion | S1 vs E1 | S2 vs E1 | mean vs E1 | S1 vs E3b | S2 vs E3b | mean vs E3b | picks |\n|---|---|---|---|---|---|---|---|---|")
    names = [k for k in R if not k.startswith("control")]
    fams = {
        "rank term (E3 / E3b family)": [k for k in names if k.startswith("rank") and "gate" not in k],
        "(a) real-score terms, no gate": [k for k in names if not k.startswith("rank") and "gate" not in k],
        "(a) real-score terms, no gate, protect 0.6 only": [k for k in names if not k.startswith("rank") and "gate" not in k and "protect=0.6" in k],
        "(a) absolute logit only": [k for k in names if k.startswith("abs") and "gate" not in k],
        "(b) gated, all terms": [k for k in names if "gate" in k],
        "(a)+(b) real-score, gated or not": [k for k in names if not k.startswith("rank")],
        "everything": names,
    }
    for fn, ks in fams.items():
        for crit in ("mean", "safe", "min"):
            held, picks = crossfit(R, ks, n, crit=crit)
            c = [F.boot(held["S1"]), F.boot(held["S2"]), F.boot((held["S1"] + held["S2"]) / 2),
                 F.boot(held["S1"] - R[E3B]["S1"]), F.boot(held["S2"] - R[E3B]["S2"]),
                 F.boot((held["S1"] + held["S2"] - R[E3B]["S1"] - R[E3B]["S2"]) / 2)]
            top = "; ".join(f"{k} x{v}" for k, v in list(picks.items())[:3])
            pr(f"| {fn} ({len(ks)}) | {crit} | " + " | ".join(F.fmtd(x) for x in c) + f" | {top} |")
            out["crossfit"][f"{fn}|{crit}"] = dict(zip(("S1", "S2", "mean", "S1_vs_e3b", "S2_vs_e3b", "mean_vs_e3b"), c), picks=picks)
    json.dump(out, open(F.FM / "real_score.json", "w"), indent=1)
    pickle.dump(R, open(F.FM / "real_score_delta.pkl", "wb"))
    (F.FM / "real_score.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Offline analysis of the forward-model re-ordering on the bench (E3 diagnosis).

    python research/bench/fm/fm_analysis.py            -> results/bench/fm/analysis.json + printed tables

All differences are paired per molecule against "no re-ordering" (the engine list), MRR@25, 10,000-resample bootstrap.
Honest ranker scores (clean_kf lists) unless stated.  Molecules without a covered spectrum count with difference 0.
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_lib as F  # noqa: E402

LAMS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
SC = ("S1", "S2")
OUT = {}


def attach_rescore(D):
    p = F.FM / "rescore.pkl"
    if not p.exists():
        return False
    RS = pickle.load(open(p, "rb"))
    for scen in D:
        for m, e in D[scen].items():
            r = RS.get(e["qm"])
            if r is None:
                continue
            pos = {s: i for i, s in enumerate(r["smiles"])}
            for l in e["lists"].values():
                for t in ("gl", "ice"):
                    for v, arr in r[t].items():
                        l[f"{t}:{v}"] = [float(arr[pos[s]]) if np.isfinite(arr[pos[s]]) else None for s in l["smiles"]]
    return True


def delta(D, scen, ss, fn, gate=None):
    _, a, b = F.evaluate(D, scen, ss, fn, gate)
    return b - a


def row(D, fn, gate=None, ss="clean_kf", scens=SC):
    return {s: delta(D, s, ss, fn, gate) for s in scens}


def show(name, r, store=None):
    cells = {s: F.boot(v) for s, v in r.items()}
    both = F.boot(np.mean([r[s] for s in r], 0)) if len(r) > 1 else None
    print(f"| {name} | " + " | ".join(F.fmtd(cells[s]) for s in r) + (f" | {F.fmtd(both)} |" if both else " |"))
    if store is not None:
        OUT.setdefault(store, {})[name] = {**{s: cells[s] for s in r}, **({"mean": both} if both else {})}
    return cells


def crossfit(D, configs, mids, seed=7, reps=20):
    """2-fold cross-fitting, repeated: pick the config with the best mean (S1+S2)/2 difference on one half, apply it to the
    other half.  Returns the held-out per-molecule difference (averaged over repeats) and how often each config is picked."""
    names = list(configs)
    M = {n: {s: configs[n][s] for s in SC} for n in names}          # per-molecule differences, aligned with mids
    avg = np.array([np.mean([M[n][s] for s in SC], 0) for n in names])
    rng = np.random.default_rng(seed)
    held = {s: np.zeros(len(mids)) for s in SC}
    picks = {}
    for _ in range(reps):
        perm = rng.permutation(len(mids))
        halves = (perm[: len(mids) // 2], perm[len(mids) // 2:])
        for a, b in (halves, halves[::-1]):
            best = int(np.argmax(avg[:, a].mean(1)))
            picks[names[best]] = picks.get(names[best], 0) + 1
            for s in SC:
                held[s][b] += M[names[best]][s][b] / reps
    return held, dict(sorted(picks.items(), key=lambda kv: -kv[1]))


def main():
    D, res = F.load()
    have_rs = attach_rescore(D)
    mids = sorted(D["S1"])
    hdr = "| setting | S1 | S2 | mean of S1, S2 |\n|---|---|---|---|"

    # ---------------- 0. baseline and coverage ----------------
    print("\n## 0. baseline (no re-ordering)\n")
    for scen, ss in (("S1", "clean_kf"), ("S2", "clean_kf"), ("S1", "blend"), ("S2", "blend"), ("S3", "blend")):
        ms, a, _ = F.evaluate(D, scen, ss, lambda l: list(range(len(l["smiles"]))))
        nfm = sum(1 for m in ms if "ice" in D[scen][m]["lists"].get(ss, {}))
        sc = sum(sum(v is not None for v in D[scen][m]["lists"][ss]["ice"]) for m in ms if "ice" in D[scen][m]["lists"].get(ss, {}))
        tot = sum(len(D[scen][m]["lists"][ss]["smiles"]) for m in ms if "ice" in D[scen][m]["lists"].get(ss, {}))
        print(f"{scen} {ss}: n={len(ms)} MRR@25 {a.mean():.4f} | molecules with forward scores {nfm} | candidates scored {sc}/{tot}")
        OUT.setdefault("baseline", {})[f"{scen}_{ss}"] = dict(n=len(ms), mrr=float(a.mean()), n_fm=nfm, scored=sc, cands=tot)

    # ---------------- (a) E3's exact rule ----------------
    print("\n## (a) E3's exact rule (fm_rerank.rerank_molecule, top 60, lam 1/1, z of the exported rank-blend score)\n")
    print(hdr)
    e3 = row(D, F.e3_perm)
    show("E3 rule, honest ranker (clean_kf)", e3, "a")
    show("E3 rule, as-submitted ranker (blend)", row(D, F.e3_perm, ss="blend"), "a")
    show("same rule through general_perm (check)", row(D, F.general_perm), "a")
    d3 = delta(D, "S3", "blend", F.e3_perm)
    print("S3 (46 listed truths of 300; blend):", F.fmtd(F.boot(d3)))
    OUT["a"]["S3_blend"] = F.boot(d3)
    for s in SC:
        _, a, b = F.evaluate(D, s, "clean_kf", F.e3_perm)
        OUT["a"][f"{s}_counts"] = dict(top1_lost=int(((a == 1) & (b < 1)).sum()), top1_gained=int(((a < 1) & (b == 1)).sum()),
                                       worse=int((b < a).sum()), better=int((b > a).sum()), top1_before=int((a == 1).sum()), top1_after=int((b == 1).sum()))
        print(s, OUT["a"][f"{s}_counts"])

    # ---------------- (b) lambda sweep ----------------
    print("\n## (b) lam_ice x lam_gl, E3 rule otherwise (mean of S1 and S2 differences; rows lam_ice, columns lam_gl)\n")
    grid = {}
    for li, lg in itertools.product(LAMS, LAMS):
        grid[(li, lg)] = row(D, lambda l, li=li, lg=lg: F.e3_perm(l, li, lg))
    for s in SC + ("mean",):
        print(f"\n{s}\n| lam_ice \\ lam_gl | " + " | ".join(str(x) for x in LAMS) + " |\n|---|" + "---|" * len(LAMS))
        for li in LAMS:
            vals = [(np.mean([grid[(li, lg)][t] for t in SC], 0) if s == "mean" else grid[(li, lg)][s]).mean() for lg in LAMS]
            print(f"| {li} | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")
    OUT["b"] = {f"{li}_{lg}": {**{s: F.boot(grid[(li, lg)][s]) for s in SC}, "mean": F.boot(np.mean([grid[(li, lg)][s] for s in SC], 0))}
                for li, lg in grid}
    print("\nselected cells with CI\n" + hdr)
    for li, lg in ((0.1, 0), (0.25, 0), (0.5, 0), (1, 0), (2, 0), (0, 0.1), (0, 0.25), (0, 0.5), (0, 1), (0, 2), (0.1, 0.1), (0.25, 0.25), (0.5, 0.5), (1, 1), (2, 2)):
        show(f"lam_ice {li}, lam_gl {lg}", grid[(float(li), float(lg))])
    held, picks = crossfit(D, {f"{a}_{b}": grid[(a, b)] for a, b in grid}, mids)
    print("\ncross-fitted (lambda pair chosen on one half, applied to the other; 20 random splits):")
    show("held-out, lambda grid on E3's rule", held, "b_crossfit")
    print("picked:", picks)
    OUT["b_crossfit_picks"] = picks

    # ---------------- (c) variants ----------------
    print("\n## (c) variants (each row: one change to E3's rule; lam as stated)\n")
    V = {}

    def add(name, fn, gate=None):
        V[name] = row(D, fn, gate)

    gp = F.general_perm
    for lam in (0.25, 0.5, 1.0):
        add(f"E3 rule, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam))
        # ranker term from the real ranker probability instead of the exported rank blend
        add(f"ranker = z(logit pv_kf), lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, ranker="pv_kf", rank_term="logit"))
        add(f"ranker = z(pv_kf prob), lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, ranker="pv_kf"))
        for k in (2, 3, 5, 10):
            add(f"top-{k} of the group only, lam {lam}/{lam}", lambda l, lam=lam, k=k: gp(l, lam, lam, topk=k))
        for mg in (0.05, 0.1, 0.2, 0.4):
            add(f"only if pv_kf margin < {mg}, lam {lam}/{lam}", lambda l, lam=lam, mg=mg: gp(l, lam, lam, margin=mg))
        add(f"z over the whole top-N, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, zscope="list"))
        add(f"whole top-N as one group, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, scope="list"))
        add(f"rank sum, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, combine="rank"))
        add(f"RRF k=3, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, combine="rrf"))
        add(f"topN 25, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, topn=25))
        add(f"topN 40, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, topn=40))
        add(f"topN 10, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, topn=10))
        for thr in (0.5, 0.7, 0.8):
            add(f"only molecules with best library similarity < {thr}, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam),
                gate=lambda e, l, thr=thr: e["lib_max"] < thr)
            add(f"only molecules with best library similarity >= {thr}, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam),
                gate=lambda e, l, thr=thr: e["lib_max"] >= thr)
        add(f"cosine instead of entropy, lam {lam}/{lam}", lambda l, lam=lam: gp(l, lam, lam, fm_cols=("ice_cos", "gl_cos")))
        if have_rs:
            for v in ("e3", "ce_best", "ce_mean", "ce_all", "cutprec", "rel1", "top30", "noprec"):
                add(f"score variant {v}, lam {lam}/{lam}", lambda l, lam=lam, v=v: gp(l, lam, lam, fm_cols=(f"ice:{v}", f"gl:{v}")))
    print(hdr)
    for n, r in V.items():
        show(n, r, "c")
    held, picks = crossfit(D, V, mids)
    print("\ncross-fitted over ALL variants above plus the lambda grid:")
    allc = dict(V); allc.update({f"lam {a}/{b}": grid[(a, b)] for a, b in grid})
    held, picks = crossfit(D, allc, mids)
    show("held-out, all variants", held, "c_crossfit")
    print("picked:", picks)
    OUT["c_crossfit_picks"] = picks

    # ---------------- (d) diagnostics ----------------
    print("\n## (d) diagnostics\n")
    for s in SC:
        rows = []
        for m in mids:
            e = D[s][m]
            l = e["lists"].get("clean_kf")
            if l is None or "ice" not in l or True not in l["ok"]:
                continue
            t = l["ok"].index(True)
            grp = [i for i in range(len(l["smiles"])) if l["formula"][i] == l["formula"][t]]
            if len(grp) < 2:
                continue
            perm = F.e3_perm(l)
            ice, gl = F.col(l, "ice")[grp], F.col(l, "gl")[grp]
            if not (np.isfinite(ice).all() and np.isfinite(gl).all()):
                continue
            ti = grp.index(t)
            zs = F._z(ice)[0] + F._z(gl)[0]
            ads = sorted({c[2] for c in e["meta"]["covered"]})
            rows.append(dict(mid=m, size=len(grp), r_rank=ti + 1, ice_rank=1 + int((ice > ice[ti]).sum()), gl_rank=1 + int((gl > gl[ti]).sum()),
                             fm_rank=1 + int((zs > zs[ti]).sum()), list_rank=t + 1, new_rank=perm.index(t) + 1, lib_max=e["lib_max"],
                             truth_lib=l["lib"][t], n_cov=len(e["meta"]["covered"]), n_spec=e["n_spec"], adducts="/".join(ads),
                             pv_top=max(l["pv_kf"]), pv_truth=l["pv_kf"][t], ice_truth=float(ice[ti]), ice_max=float(ice.max()),
                             margin=float(np.sort(np.array(l["pv_kf"])[grp])[::-1][0] - np.sort(np.array(l["pv_kf"])[grp])[::-1][1]),
                             mass=float(e["meta"]["covered"][0][0] >= 0) and 0.0))
        n = len(rows)
        g = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
        d = dict(n_groups=n, median_size=float(np.median(g("size"))),
                 truth_first_ranker=float((g("r_rank") == 1).mean()), truth_first_ice=float((g("ice_rank") == 1).mean()),
                 truth_first_gl=float((g("gl_rank") == 1).mean()), truth_first_fm_zsum=float((g("fm_rank") == 1).mean()),
                 group_mrr_ranker=float((1 / g("r_rank")).mean()), group_mrr_ice=float((1 / g("ice_rank")).mean()),
                 group_mrr_gl=float((1 / g("gl_rank")).mean()), group_mrr_fm_zsum=float((1 / g("fm_rank")).mean()),
                 both_first=float(((g("r_rank") == 1) & (g("fm_rank") == 1)).mean()),
                 ranker_first_fm_not=float(((g("r_rank") == 1) & (g("fm_rank") > 1)).mean()),
                 fm_first_ranker_not=float(((g("r_rank") > 1) & (g("fm_rank") == 1)).mean()),
                 neither=float(((g("r_rank") > 1) & (g("fm_rank") > 1)).mean()))
        dem = [r for r in rows if r["list_rank"] == 1 and r["new_rank"] > 1]
        pro = [r for r in rows if r["list_rank"] > 1 and r["new_rank"] == 1]
        kept = [r for r in rows if r["list_rank"] == 1 and r["new_rank"] == 1]

        def prof(R):
            if not R:
                return {}
            h = lambda k: np.array([r[k] for r in R], float)  # noqa: E731
            return dict(n=len(R), median_group_size=float(np.median(h("size"))), median_lib_max=float(np.median(h("lib_max"))),
                        median_truth_lib=float(np.median(h("truth_lib"))), share_truth_has_lib_spectrum=float((h("truth_lib") > 0).mean()),
                        median_n_covered_spectra=float(np.median(h("n_cov"))), median_pv_margin=float(np.median(h("margin"))),
                        median_pv_truth=float(np.median(h("pv_truth"))), median_new_rank=float(np.median(h("new_rank"))),
                        median_fm_rank=float(np.median(h("fm_rank"))), median_ice_truth=float(np.median(h("ice_truth"))),
                        share_na_adduct=float(np.mean(["Na" in r["adducts"] for r in R])))
        d.update(demoted=prof(dem), kept_first=prof(kept), promoted=prof(pro))
        OUT.setdefault("d", {})[s] = d
        print(s, json.dumps(d, indent=1))
        pickle.dump(rows, open(F.FM / f"diag_rows_{s}.pkl", "wb"))
    json.dump(OUT, open(F.FM / "analysis.json", "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))


if __name__ == "__main__":
    main()

"""Re-score the bench candidates offline from the raw predictions dumped by the Kaggle run (fm_<model>.json.pred.pkl),
with the runner's own functions, under alternative merges / observed-spectrum treatments.

    python research/bench/fm/fm_rescore.py [workers]

Writes results/bench/fm/rescore.pkl: {qm: {"smiles": [...], model: {variant: array(n_cand) (mean over covered spectra)}}}
Variants
  e3          the spectrum's own energies merged (must equal the runner's score)        [entropy]
  e3_cos      same, square-root cosine
  ce_best     max over single-energy predictions at 20 / 40 / 60 eV
  ce_mean     mean of the three single-energy scores
  ce_all      20 + 40 + 60 merged, whatever the spectrum's own energy
  cutprec     own energies; observed peaks above precursor + 2 removed (as E1's engine does)
  rel1        own energies; observed peaks below 1 % of the base peak removed, and cutprec
  top30       own energies; 30 strongest observed peaks, and cutprec
  noprec      rel1 + precursor peak (+-0.02) removed from observed and predicted
"""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FM = ROOT / "results" / "bench" / "fm"
sys.path.insert(0, str(ROOT / "research" / "forward_model"))
import ice_runner as R  # noqa: E402

VARIANTS = ("e3", "e3_cos", "ce_best", "ce_mean", "ce_all", "cutprec", "rel1", "top30", "noprec")
G = {}


class A:  # runner defaults used by E3
    default_ces = "20,40,60"; ce_round = 5.0; ce_min = 5.0; max_ces = 3


def init():
    for t in ("gl", "ice"):
        G[t] = pickle.load(open(FM / "output" / f"fm_{t}.json.pred.pkl", "rb"))
    G["inp"] = json.load(open(FM / "input" / "fm_input.json"))


def sim(obs, preds, cos=False, drop=None):
    mz_p, it_p = R.merge_predictions(preds, np, 100)
    if drop is not None and mz_p.size:
        k = np.abs(mz_p - drop) > 0.02
        mz_p, it_p = mz_p[k], it_p[k]
        if it_p.size:
            it_p = it_p / it_p.max()
    if mz_p.size == 0 or obs[0].size == 0:
        return np.nan
    a, b = R.align(obs[0], obs[1], mz_p, it_p, np, 0.01, 20.0)
    return R.cosine_similarity(a, b, np) if cos else R.entropy_similarity(a, b, np)


def work(qm):
    inp = G["inp"]
    ents = sorted(e for e in inp if e.startswith(qm + "#s"))
    smiles = inp[ents[0]]["smiles"]
    out = {"smiles": smiles}
    for t in ("gl", "ice"):
        P, C = G[t]["pred"], G[t]["canon"]
        acc = {v: np.full((len(ents), len(smiles)), np.nan) for v in VARIANTS}
        for si, e in enumerate(ents):
            sp = inp[e]["spectra"][0]
            ad, pmz = sp["adduct"], float(sp["precursor_mz"])
            ces = R.ce_list(sp["collision_energies"], A)
            mz, it = np.asarray(sp["mzs"], float), np.asarray(sp["intensities"], float)
            o_raw = R.clean_spectrum(mz, it, np)
            k = mz <= pmz + 2.0
            o_cut = R.clean_spectrum(mz[k], it[k], np)
            o_rel = R.clean_spectrum(mz[k], it[k], np, 0, 0.01)
            o_top = R.clean_spectrum(mz[k], it[k], np, 30, 0.0)
            k2 = k & (np.abs(mz - pmz) > 0.02)
            o_np = R.clean_spectrum(mz[k2], it[k2], np, 0, 0.01)
            for ci, s in enumerate(smiles):
                c = C.get(s, (None, None))[0]
                if c is None:
                    continue
                own = [P.get((c[0], ad, ce)) for ce in ces]
                if any(p is None for p in own):
                    continue
                own = [p.astype(np.float64) for p in own]
                acc["e3"][si, ci] = sim(o_raw, own)
                acc["e3_cos"][si, ci] = sim(o_raw, own, cos=True)
                acc["cutprec"][si, ci] = sim(o_cut, own)
                acc["rel1"][si, ci] = sim(o_rel, own)
                acc["top30"][si, ci] = sim(o_top, own)
                acc["noprec"][si, ci] = sim(o_np, own, drop=pmz)
                grid = [P.get((c[0], ad, ce)) for ce in (20.0, 40.0, 60.0)]
                if all(p is not None for p in grid):
                    grid = [p.astype(np.float64) for p in grid]
                    one = [sim(o_raw, [p]) for p in grid]
                    acc["ce_best"][si, ci] = np.nanmax(one)
                    acc["ce_mean"][si, ci] = np.nanmean(one)
                    acc["ce_all"][si, ci] = sim(o_raw, grid)
        out[t] = {v: a.mean(0) for v, a in acc.items()}          # NaN when any covered spectrum is missing
        out[t + "_per_spec"] = acc["e3"]
    return qm, out


def main(workers=3):
    init()
    qms = sorted({e.split("#")[0] for e in G["inp"]})
    with Pool(workers, initializer=init) as mp:
        res = dict(mp.imap_unordered(work, qms, chunksize=4))
    pickle.dump(res, open(FM / "rescore.pkl", "wb"))
    # parity with the runner's own scores
    for t in ("gl", "ice"):
        run = json.load(open(FM / "output" / f"fm_{t}.json"))["scores"]
        d = []
        for qm, o in res.items():
            ents = sorted(e for e in G["inp"] if e.startswith(qm + "#s"))
            for si, e in enumerate(ents):
                for a, b in zip(run[e], o[t + "_per_spec"][si]):
                    if a is not None and np.isfinite(b):
                        d.append(abs(a - b))
        print(t, "offline vs runner: n", len(d), "max abs diff", float(np.max(d)) if d else None)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)

"""Bench candidate lists + query spectra -> input of the forward-model runner (Kaggle dataset casmi-bench-fm-input).

    python research/bench/fm/build_fm_input.py

lists.pkl   {scen: {score_set: {mid: dict(smiles, keys, raw, scores, idx, truncated)}}}  E3-style lists (up to 60, metric-key dedup)
fm_input.json  runner input.  Entries "<mid>#s<i>" = one covered query spectrum with its own collision energies (the mean
               over these entries is exactly E3's score); "<mid>#g<adduct index>_<ce>" = the first spectrum of that adduct
               with a single forced energy (only there to make the runner predict the 20/40/60 grid).
"""
import json
import math
import pickle
from collections import OrderedDict
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "results" / "bench"
OUT = B / "fm"
ADDUCTS = ("[M+H]+", "[M+Na]+")
KEEP = 60
SETS = {"S1": ("clean_kf", "blend"), "S2": ("clean_kf", "blend"), "S3": ("blend",)}


def e3_list(rec, score, keep=KEEP):
    """E3 runner: blend order, metric-key dedup; E1 stops after 80 candidates when fewer than 40 were kept."""
    sc = rec["scores"][score]
    order = np.argsort(-sc, kind="mergesort")[:max(80, 3 * keep)]
    out = dict(smiles=[], keys=[], raw=[], scores=[], idx=[], truncated=False)
    seen = set()
    for j, i in enumerate(order):
        i = int(i)
        if j >= 80 and len(out["smiles"]) < 40:
            break
        if i not in rec["top"]:          # the bench kept SMILES for the top 80 of every score set only
            out["truncated"] = True
            break
        s, k = rec["top"][i]
        if k is None or k in seen:
            continue
        seen.add(k)
        out["smiles"].append(s); out["keys"].append(k); out["raw"].append(rec["keys"][i]); out["scores"].append(float(sc[i])); out["idx"].append(i)
        if len(out["smiles"]) >= max(40, keep):
            break
    return out


def main():
    con = duckdb.connect()
    T = {}
    for q, mid, smi, formula, correct in con.execute(f"select * from read_parquet('{(B / 'truth.parquet').as_posix()}')").fetchall():
        T[(q, mid)] = dict(smiles=smi, formula=formula, correct=set(correct.split(";")))
    lists, cands = {}, {}
    for scen, sets in SETS.items():
        R = pickle.load(open(B / "e1" / f"{scen}.pkl", "rb"))
        lists[scen] = {}
        for ss in sets:
            lists[scen][ss] = {}
            for mid, rec in R.items():
                if "scores" not in rec or ss not in rec["scores"]:
                    continue
                L = e3_list(rec, ss)
                if scen == "S3":  # only molecules whose truth is listed can change
                    ok = T[("S3", mid)]["correct"]
                    if not any(k == rec["truth_canon"] or k in ok or r in ok for k, r in zip(L["keys"], L["raw"])):
                        continue
                lists[scen][ss][mid] = L
                u = cands.setdefault((scen[:2] if scen == "S3" else "S12", mid), [])
                for s in L["smiles"]:
                    if s not in u:
                        u.append(s)
    pickle.dump(lists, open(OUT / "lists.pkl", "wb"))
    spectra = {}
    for q, f in (("S12", "queries_S12.parquet"), ("S3", "queries_S3.parquet")):
        rows = con.execute(f"select molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, adduct, precursor_mz, collision_energy_ev, "
                           f"instrument_type from read_parquet('{(B / f).as_posix()}') order by molecule_id, spectrum_id").fetchall()
        for mid, sid, mz, it, ad, pmz, ce, inst in rows:
            spectra.setdefault((q, mid), []).append(dict(sid=sid, mzs=mz, intensities=it, adduct=ad, precursor_mz=pmz,
                                                         collision_energies=ce, instrument=inst))
    inp, meta, n_pred, n_cand = OrderedDict(), {}, 0, 0
    for (q, mid), smi in sorted(cands.items()):
        cov = [(i, sp) for i, sp in enumerate(spectra.get((q, mid), [])) if sp["adduct"] in ADDUCTS]
        if not cov or len(smi) < 2:
            continue
        keys = set()
        for i, sp in cov:
            inp[f"{q}|{mid}#s{i}"] = dict(smiles=smi, spectra=[{k: sp[k] for k in ("mzs", "intensities", "adduct", "precursor_mz", "collision_energies")}])
            ces = [abs(float(c)) for c in (sp["collision_energies"] or []) if c is not None and math.isfinite(c) and c > 0] or [20.0, 40.0, 60.0]
            ces = sorted({max(5.0, round(c / 5.0) * 5.0) for c in ces})
            if len(ces) > 3:
                ces = [ces[j] for j in sorted({round(t * (len(ces) - 1) / 2) for t in range(3)})]
            keys |= {(sp["adduct"], c) for c in ces}
        for a, ad in enumerate(ADDUCTS):
            first = [sp for _, sp in cov if sp["adduct"] == ad]
            if not first:
                continue
            for ce in (20.0, 40.0, 60.0):
                if (ad, ce) not in keys:
                    keys.add((ad, ce))
                    sp = first[0]
                    inp[f"{q}|{mid}#g{a}_{int(ce)}"] = dict(smiles=smi, spectra=[dict(mzs=sp["mzs"], intensities=sp["intensities"], adduct=ad,
                                                                                 precursor_mz=sp["precursor_mz"], collision_energies=[ce])])
        meta[f"{q}|{mid}"] = dict(n_cand=len(smi), covered=[(i, sp["sid"], sp["adduct"], sp["collision_energies"], sp["instrument"]) for i, sp in cov],
                                  n_spec=len(spectra[(q, mid)]), keys=sorted(keys))
        n_pred += len(smi) * len(keys); n_cand += len(smi)
    (OUT / "input").mkdir(exist_ok=True)
    json.dump(inp, open(OUT / "input" / "fm_input.json", "w"))
    json.dump(meta, open(OUT / "input" / "fm_input_meta.json", "w"))
    print("molecules", len(meta), "entries", len(inp), "candidates", n_cand, "predictions per model", n_pred)
    for scen in lists:
        for ss, L in lists[scen].items():
            n = [len(v["smiles"]) for v in L.values()]
            print(scen, ss, "lists", len(L), "len pct", np.percentile(n, [0, 10, 50, 90, 100]), "truncated", sum(v["truncated"] for v in L.values()),
                  "sent", sum(f"{'S3' if scen == 'S3' else 'S12'}|{m}" in meta for m in L))


if __name__ == "__main__":
    main()

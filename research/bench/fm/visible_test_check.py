"""Labelled check of the actual E3 Kaggle run: the 400 visible test molecules are placeholder rows of train (enveda-180),
so their structures are known.  E1 has their own spectra in its library (exact match), i.e. this is an "easy Class 1" set.

    python research/bench/fm/visible_test_check.py <downloaded E3 kernel output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_lib import FM, boot, fmtd, formula, rerank_molecule, rr_of  # noqa: E402


def canon_key(smi, _g={}):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g:
        _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:  # noqa: BLE001
        return None


def main(d):
    d = Path(d)
    L = json.load(open(d / "eng_lists.json"))
    ice = json.load(open(d / "fm_ice.json"))["scores"]
    gl = json.load(open(d / "fm_gl.json"))["scores"]
    inp = json.load(open(d / "fm_input.json"))
    lab = json.load(open(FM / "visible_test_labels.json"))
    base, new, rows = [], [], []
    for m, e in L.items():
        keys, smi = lab[m]
        tk = set(keys) | {canon_key(smi)}
        ok = [k in tk for k in e["keys"]]
        f = [formula(s) for s in e["smiles"]]
        b = rr_of(ok)
        base.append(b)
        if m not in inp:
            new.append(b)
            continue
        pos = {s: i for i, s in enumerate(inp[m]["smiles"])}
        per = {"ice": [ice[m][pos[s]] if s in pos else None for s in e["smiles"]],
               "gl": [gl[m][pos[s]] if s in pos else None for s in e["smiles"]]}
        perm, _ = rerank_molecule(f, e["scores"], per, {"ice": 1.0, "gl": 1.0}, 60)
        new.append(rr_of(ok, perm))
        if True in ok:
            t = ok.index(True)
            grp = [i for i in range(len(f)) if f[i] == f[t]]
            for tag, v in per.items():
                x = [v[i] for i in grp]
                if all(a is not None for a in x):
                    rows.append((tag, len(grp), 1 + sum(a > v[t] for a in x), grp.index(t) + 1))
    base, new = np.array(base), np.array(new)
    out = {"n": len(base), "E1_mrr": float(base.mean()), "E3_mrr": float(new.mean()), "delta": boot(new - base),
           "E1_top1": float((base == 1).mean()), "E3_top1": float((new == 1).mean()),
           "truth_first_in_E1_and_demoted": int(((base == 1) & (new < 1)).sum()), "promoted_to_first": int(((base < 1) & (new == 1)).sum())}
    for tag in ("ice", "gl"):
        r = np.array([(a, b, c) for t, a, b, c in rows if t == tag], float)
        out[f"{tag}_alone_truth_first_in_group"] = float((r[:, 1] == 1).mean())
        out[f"{tag}_alone_group_mrr"] = float((1 / r[:, 1]).mean())
        out["ranker_truth_first_in_group"] = float((r[:, 2] == 1).mean())
        out["n_groups_with_truth"] = len(r)
        out["median_group_size"] = float(np.median(r[:, 0]))
    print(json.dumps(out, indent=1))
    print("E3 - E1 on the visible (placeholder) test:", fmtd(out["delta"]))
    json.dump(out, open(FM / "visible_test_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

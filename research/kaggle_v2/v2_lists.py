"""V2 as a drop-in engine for another notebook: writes ranked candidate lists, never submission.csv.

    python v2_lists.py OUT_JSON [TOPN=40] [WORKERS=4]

OUT_JSON = {molecule_id: {"smiles": [...], "keys": [...], "scores": [...], "src": [...]}}, best first, TOPN per molecule.
keys = tautomer-canonical InChIKey14 computed with the installed RDKit (the competition metric's matching key), so lists
fuse correctly with engines that key candidates the same way. Duplicate keys are collapsed (first, i.e. best, kept).
Needs on Kaggle: competition data, dataset casmi-v1-assets, dataset casmi-v2-pubchem, and this folder (casmi_v0/v1/v2
sources). Uses RDKit, DuckDB, scipy, numpy, pandas only (no matchms).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import casmi_v2_kaggle as v2  # noqa: E402

v1, v0 = v2.v1, v2.v0


def tautomer_key(smi, _te=[]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if not _te:
        _te.append(rdMolStandardize.TautomerEnumerator())
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_te[0].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def main(out_json, topn=40, workers=None, mode="kaggle"):
    T0 = time.time()
    P = v0.find_inputs(mode)
    work = Path(P["work"]).parent / "casmi_v2_lists"
    work.mkdir(parents=True, exist_ok=True)
    workers = workers or v0.n_workers()
    test = v0.load_test(P["test"])
    test = test[test["molecule_id"].notna()].reset_index(drop=True)
    test["molecule_id"] = test["molecule_id"].astype(str)
    if test["spectrum_id"].duplicated().any():
        test["spectrum_id"] = test["spectrum_id"].astype(str) + "#" + test.groupby("spectrum_id").cumcount().astype(str)
    pred, U, pc_win, pc_fp, info = v2.run_v2(P, test, work, workers, v2.CFG, assets=v1.find_assets(mode),
                                             pc_path=v2.find_pubchem(mode))
    scored = v2.score_molecules(pred, U, pc_win, pc_fp, v2.CFG)
    smi = {}
    for g in pc_win.values():
        smi.update(zip(g["ik"].values, g["smiles"].values))
    out = {}
    for mid, lst in scored.items():
        rec = {"smiles": [], "keys": [], "scores": [], "src": []}
        for ik, sc, src in lst:
            s = U.smiles[U.row[ik]] if src == "u" else smi[ik]
            k = tautomer_key(s) or ik
            if k in rec["keys"]:
                continue
            rec["smiles"].append(s); rec["keys"].append(k); rec["scores"].append(round(float(sc), 6)); rec["src"].append(src)
            if len(rec["smiles"]) >= topn:
                break
        out[mid] = rec
    json.dump(out, open(out_json, "w"))
    print(json.dumps({"molecules": len(out), "test_molecules": int(test["molecule_id"].nunique()), "info": info,
                      "sec": round(time.time() - T0, 1)}, default=str), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 40, int(a[2]) if len(a) > 2 else None,
         a[3] if len(a) > 3 else "kaggle")

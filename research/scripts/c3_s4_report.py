"""Statistics for the S4 report (research/analysis/c3_s4_scenarios.md): counts, mass / adduct / spectra-per-molecule
distributions of queries_S4C3 and queries_S4PC against queries_S12, NP-likeness share, example SMILES.

    python research/scripts/c3_s4_report.py      (prints JSON; also writes results/c3/s4_work/report.json)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
C3 = ROOT / "results" / "c3"


def np_share(smiles):
    try:
        import os, sys
        from rdkit import Chem, RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
        import npscorer
        mdl = npscorer.readNPModel()
        v = [npscorer.scoreMol(m, mdl) for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]
        return dict(share_np_gt0=round(float(np.mean(np.array(v) > 0)), 3), median=round(float(np.median(v)), 2))
    except Exception as e:
        return f"not computed ({e!r})"


def main():
    T = pd.read_parquet(C3 / "truth_S4.parquet")
    ver = json.load(open(C3 / "s4_work" / "verified.json"))
    cands = pd.read_parquet(C3 / "s4_work" / "cands.parquet")
    out = dict(candidates=cands.qset.value_counts().to_dict(),
               tautomer_check={k: v for k, v in ver.items() if k not in ("verified", "dropped_examples")},
               n_verified_c3=len(ver["verified"]), held_keys=len(pd.read_parquet(C3 / "held_S4.parquet")), sets={})
    sets = {"S4C3": C3 / "queries_S4C3.parquet", "S4PC": C3 / "queries_S4PC.parquet",
            "S12 (bench)": ROOT / "results" / "bench" / "queries_S12.parquet"}
    for name, f in sets.items():
        q = pd.read_parquet(f)
        per = q.groupby("molecule_id").size()
        d = dict(molecules=int(q.molecule_id.nunique()), spectra=len(q),
                 precursor_mz_quartiles=[round(float(x), 1) for x in q.precursor_mz.quantile([.25, .5, .75])],
                 spectra_per_molecule=dict(median=float(per.median()), mean=round(float(per.mean()), 2),
                                           max=int(per.max())),
                 adducts={k: round(v, 3) for k, v in q.adduct.value_counts(normalize=True).head(6).items()},
                 instruments=q.instrument_type.value_counts().to_dict() if "instrument_type" in q else None)
        if name.startswith("S4"):
            t = T[T.qset == name]
            d["np_likeness"] = np_share(t.smiles.tolist())
            d["examples"] = t.smiles.head(10).tolist()
            d["truth_with_tautomer_alias"] = int((t.correct.str.count(";") > 0).sum())
        out["sets"][name] = d
    json.dump(out, open(C3 / "s4_work" / "report.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

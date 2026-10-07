"""Check a finished E5 Kaggle run before it is submitted.

    python research/kaggle_e1/e5/check_e5.py [--out results/kaggle_e5]

Pass conditions:
  1. submission.csv is valid: 400 rows, at most 25 unique SMILES per row;
  2. on the visible set, MRR@25 = 1.000 (every library answer is protected by the gate);
  3. the log reports that the V2 engine ran and the gate fused at least one molecule.
Visible truth = the exact enveda-180 duplicate (research/scripts/c3_visible_eval.visible_truth); candidate keys use
the bench canonicaliser (metric key). Prints PASS / FAIL and writes <out>/check_e5.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "bench"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "kaggle_e5"))
    a = ap.parse_args()
    out = Path(a.out)
    import bench
    from c3_visible_eval import visible_truth
    sub = pd.read_csv(out / "submission.csv")
    if (out / "e5_lists.json").exists():
        L = json.load(open(out / "e5_lists.json"))
    else:  # the output download sometimes omits it: rebuild the lists from submission.csv (gate values unknown)
        L = {str(m): dict(smiles=s.split(";"), gate_lib=None) for m, s in zip(sub.molecule_id, sub.smiles)}
    n = sub.smiles.str.split(";").map(len)
    ok_sub = len(sub) == 400 and sub.molecule_id.is_unique and bool((n <= 25).all()) and \
        bool((sub.smiles.str.split(";").map(lambda v: len(set(v))) == n).all())
    truth = visible_truth()
    bench.setup_env(ROOT / "research/bench/eng", 4)
    import casmi_engine as E
    smis = sorted({s for v in L.values() for s in v["smiles"] if s})
    C = bench.canon_many(E, smis, 4)
    rr = []
    for m, t in truth.items():
        keys = [C.get(s) for s in L.get(m, {}).get("smiles", [])][:25]
        rr.append(1.0 / (keys.index(t) + 1) if t in keys else 0.0)
    rr = np.array(rr)
    g = np.array([v["gate_lib"] for v in L.values() if v.get("gate_lib") is not None])
    log = "\n".join(p.read_text(errors="ignore") for p in out.glob("*.log"))
    ran = "our lists" in log and "E5 gate lib" in log
    res = dict(submission_valid=ok_sub, visible_mrr=round(float(rr.mean()), 4), visible_top1=int((rr == 1).sum()),
               visible_lost=[m for (m, _), x in zip(truth.items(), rr) if x < 1],
               gate_min=float(g.min()) if len(g) else None, gate_share_below=float((g < 0.7).mean()) if len(g) else None,
               v2_and_gate_in_log=ran)
    res["PASS"] = bool(ok_sub and res["visible_mrr"] == 1.0 and ran)
    json.dump(res, open(out / "check_e5.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("PASS" if res["PASS"] else "FAIL")


if __name__ == "__main__":
    main()

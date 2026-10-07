"""Check a downloaded E4 kernel output (run: python research/kaggle_e1/e4/check_run.py <E4 output dir> [<E3b output dir>]).

Validity of submission.csv, stage status, popularity statistics, and the comparison of the submitted rows with
(a) the E3b reference computed inside the same run and (b) the submission of E3b's own Kaggle run.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def rows(p):
    return {r["molecule_id"]: r["smiles"].split(";") for r in csv.DictReader(open(p, encoding="utf-8"))}


def compare(a, b):
    t1 = [m for m in a if a[m][:1] != b[m][:1]]
    return dict(top1=len(t1), order=sum(a[m] != b[m] for m in a), members=sum(set(a[m]) != set(b[m]) for m in a), top1_mids=t1)


def main():
    d = Path(sys.argv[1])
    sub = rows(d / "submission.csv")
    sample = [r["molecule_id"] for r in csv.DictReader(open(ROOT / "sample_submission.csv", encoding="utf-8"))]
    n = [len(v) for v in sub.values()]
    ok = (list(sub) == sample and len(sub) == 400 and max(n) <= 25 and min(n) >= 1 and all(len(set(v)) == len(v) for v in sub.values())
          and all(all(x for x in v) for v in sub.values()))
    print("submission valid:", ok, "| rows", len(sub), "| SMILES per row min / median / max", min(n), int(np.median(n)), max(n),
          "| fallback rows (CCO only)", sum(v == ["CCO"] for v in sub.values()))
    fm, pop = json.load(open(d / "fm_stats.json")), json.load(open(d / "pop_stats.json"))
    print("forward stage:", fm.get("status"), {t: (m.get("status"), m.get("n_candidates_scored"), m.get("n_candidates_unscored"), m.get("wall_sec"))
                                               for t, m in (fm.get("models") or {}).items()}, "| sent", fm.get("n_candidates_sent"),
          "| protected", fm.get("n_candidates_protected"))
    print("popularity stage:", pop.get("status"), {k: v for k, v in pop.items() if k not in ("status", "e4_vs_e3b")})
    eng, e4 = json.load(open(d / "eng_lists.json")), json.load(open(d / "e4_lists.json"))
    c = pop.get("n_candidates") or 1
    print(f"candidates in the top-60 windows: {c} | with a PubChem record {pop.get('n_with_record')} ({pop.get('n_with_record', 0) / c:.1%}) | "
          f"with a PubMed link {pop.get('n_with_pubmed')} ({pop.get('n_with_pubmed', 0) / c:.1%}) | unknown to the table {pop.get('n_unknown_to_table')} | "
          f"molecules with at least one record {pop.get('n_molecules_with_record')} of {pop.get('n_molecules')}")
    ref = rows(d / "e3b_order_reference.csv")
    r1 = compare(sub, ref)
    print("E4 vs E3b (same run): top-1 changed", r1["top1"], "| top-25 order changed", r1["order"], "| top-25 membership changed", r1["members"])
    e1 = compare(sub, rows(d / "e1_order_reference.csv"))
    print("E4 vs E1  (same run): top-1 changed", e1["top1"], "| top-25 order changed", e1["order"], "| top-25 membership changed", e1["members"])
    new_in, moved, same_pos = [], [], []
    for m in sub:
        a, b = ref[m], sub[m]
        new_in.append(len(set(b) - set(a)))
        same_pos.append(sum(x == y for x, y in zip(a, b)) / max(len(b), 1))
        pos = {s: i for i, s in enumerate(eng[m]["smiles"])} if m in eng else {}
        moved += [pos[s] for s in b if s in pos and pos[s] >= 25]
    print(f"per row: candidates in E4's top 25 that E3b did not list: mean {np.mean(new_in):.2f}, median {np.median(new_in):.0f}, max {max(new_in)} | "
          f"same candidate in the same slot: mean {np.mean(same_pos):.1%} | E4 top-25 entries from engine ranks 26-60: {len(moved)} ({len(moved) / 400:.2f} per row)")
    prot = sum(1 for m, e in eng.items() for i, x in enumerate(e["lib"][:60]) if x >= 0.6)
    kept = sum(1 for m, e in eng.items() for i, x in enumerate(e["lib"][:60]) if x >= 0.6 and e4[m]["keys"][i] == e["keys"][i])
    top1_prot = sum(1 for e in eng.values() if e["lib"] and e["lib"][0] >= 0.6)
    print(f"protected candidates {prot}, in their slot after E4 {kept} | molecules whose engine top-1 is protected: {top1_prot} of {len(eng)}")
    for m in r1["top1_mids"]:
        e, f = eng[m], e4[m]
        i_old, i_new = e["smiles"].index(ref[m][0]), e["smiles"].index(sub[m][0])
        print(f"  top-1 changed {m}: E3b top-1 = engine rank {i_old + 1} (lib {e['lib'][i_old]:.2f}, pv {e['pv'][i_old]:.4f}, pop {f['pop'][f['smiles'].index(ref[m][0])]}) -> "
              f"E4 top-1 = engine rank {i_new + 1} (lib {e['lib'][i_new]:.2f}, pv {e['pv'][i_new]:.4f}, pop {f['pop'][0]}) | lib_max {e['lib_max']:.2f}")
    if len(sys.argv) > 2:
        d3 = Path(sys.argv[2])
        s3, g3 = rows(d3 / "submission.csv"), json.load(open(d3 / "eng_lists.json"))
        r2 = compare(sub, s3)
        r3 = compare(ref, s3)
        same_eng = sum(eng[m]["keys"] == g3[m]["keys"] for m in eng)
        print("E4 vs E3b's own Kaggle run (version 1): top-1 changed", r2["top1"], "| top-25 order changed", r2["order"], "| top-25 membership changed", r2["members"])
        print("in-run E3b reference vs E3b's own Kaggle run: identical rows", 400 - r3["order"], "of 400 | top-1 differs", r3["top1"],
              "| engine lists identical (keys, order)", same_eng, "of", len(eng))
        if r2["top1_mids"]:
            print("  top-1 differs from E3b's run for:", r2["top1_mids"])
        lab = ROOT / "results" / "bench" / "fm" / "visible_test_labels.json"
        if lab.exists():        # the visible molecules are train placeholders: labelled MRR@25 on the engine's metric keys
            L = json.load(open(lab))

            def mrr(lists):
                v = []
                for m, e in lists.items():
                    tk = set(L[m][0])
                    v.append(next((1.0 / (i + 1) for i, k in enumerate(e["keys"][:25]) if k in tk), 0.0))
                return float(np.mean(v)), int(sum(x == 1.0 for x in v)), len(v)
            print("labelled visible molecules (truth key from the train row; MRR@25, top-1 count, n): engine", mrr(eng), "| E4", mrr(e4))


if __name__ == "__main__":
    main()

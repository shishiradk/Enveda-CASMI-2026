"""Rule-reach sanity check on the C3NP bench (not a generator): for each of the 550 C3NP molecules, apply the mined MMP
rules ONCE to the top-N sanitized library analogs (results/c3np/context.pkl) at the engine target mass (+-ppm) and
count how often the truth (metric key) is among the products, per subset (npex / s3pc / s3none).

Leak hygiene (FACT by construction):
  * rules = assets.rules_excluding(assets.bench_truth_keys(), min_support=1): every supporting molecule pair that
    involves any of the 550 truth keys (correct / ckey / plain ik14 of the truth SMILES) or any
    results/c3np/forbidden.parquet key is removed and the rules are recounted. Reported at clean support >=1 and >=3
    (a product's support = max clean freq over the rules that produce it, so >=3 is derived from the same run).
  * analogs come from context.pkl (already sanitized by raw and canonical key); any analog whose ik14/ckey is a
    bench truth/forbidden key is skipped again here.
  * the target mass is the engine target from context.pkl (from the spectrum), not the truth exact mass.
Truth matching: product plain InChIKey14 in {correct, ckey, ik14}; products with the truth formula and Morgan-r2
Tanimoto >= 0.6 to the truth also get the tautomer-canonical key (assets.taut_key) and match on it.

Usage: python research/c3gen/rules_reach_check.py [--n_analogs 10] [--ppm 10] [--max_mols 0] [--out ...]
Single process, peak ~1-1.5 GB (rules + support arrays).
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter

import numpy as np

R = "D:/Enveda-CASMI-2026/"
sys.path.insert(0, R + "research/c3gen")
import assets as A  # noqa: E402


def _rk(fr, f):
    """0-based rank of a product with support f among products sorted by support desc, ties split evenly."""
    return int((fr > f).sum() + ((fr == f).sum() - 1) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_analogs", type=int, default=10)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--max_mols", type=int, default=0)
    ap.add_argument("--out", default=R + "results/c3gen/rules_reach_c3np.json")
    a = ap.parse_args()
    import pandas as pd
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    T0 = time.time()
    M = pd.read_parquet(R + "results/c3np/molecules.parquet")
    if a.max_mols:
        M = M.head(a.max_mols)
    ctx = pickle.load(open(R + "results/c3np/context.pkl", "rb"))["mols"]
    bad = A.bench_truth_keys()
    miss = [m for m in M.itertuples() if not {*m.correct.split(";"), m.ckey, m.ik14} <= bad]
    assert not miss, f"truth keys missing from bench_truth_keys: {len(miss)}"
    rules_all = A.load_rules(min_freq=1)
    rules = A.rules_excluding(bad, min_support=1, rules=rules_all)
    n_rules = dict(directed_all=int(len(rules_all)), directed_clean_ge1=int(len(rules)),
                   directed_clean_ge3=int((rules.freq >= 3).sum()),
                   directed_dropped=int(len(rules_all) - len(rules)))
    print("rules", n_rules, f"{time.time() - T0:.0f}s", flush=True)
    t1 = time.time()
    per = []
    for i, m in enumerate(M.itertuples()):
        c = ctx[m.mid]
        TK = {*m.correct.split(";"), m.ckey, m.ik14}
        tm = Chem.MolFromSmiles(m.smiles)
        tfp = gen.GetFingerprint(tm)
        tform = CalcMolFormula(tm)
        ana = [x for x in c["analogs"] if x["ik14"] not in bad and x["ckey"] not in bad][:a.n_analogs]
        prods = {}            # plain ik14 -> [max freq, best analog rank, smirks, smiles]
        for rank, x in enumerate(ana):
            for q in A.apply_rules(x["smiles"], rules, target_mass=c["target"], ppm=a.ppm, taut_dedupe=False):
                p = prods.get(q["key"])
                if p is None:
                    prods[q["key"]] = [q["freq"], rank, q["smirks"], q["smiles"]]
                else:
                    if q["freq"] > p[0]:
                        p[0], p[2] = q["freq"], q["smirks"]
                    p[1] = min(p[1], rank)
        hit = None
        for k, p in prods.items():
            match = k in TK
            if not match:
                pm = Chem.MolFromSmiles(p[3])
                if pm is not None and CalcMolFormula(pm) == tform and \
                        DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(pm)) >= 0.6:
                    match = A.taut_key(pm) in TK
            if match and (hit is None or p[0] > hit[0]):
                hit = p
        fr = np.array([p[0] for p in prods.values()], int)
        rec = dict(mid=m.mid, subset=m.subset, n_analogs=len(ana), n_prod=len(prods), n_prod3=int((fr >= 3).sum()))
        if hit is not None:
            rec.update(hit=1, hit_freq=int(hit[0]), hit_ana_rank=int(hit[1]), hit_smirks=hit[2],
                       hit_rank_ge1=_rk(fr, hit[0]), hit_rank_ge3=_rk(fr[fr >= 3], hit[0]))
        else:
            rec.update(hit=0)
        per.append(rec)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(M)} hits {sum(r['hit'] for r in per)}  {time.time() - t1:.0f}s", flush=True)
    P = pd.DataFrame(per)
    P.to_parquet(a.out.replace(".json", ".parquet"), index=False)
    summ = {}
    for sub, g in [("all", P)] + list(P.groupby("subset")):
        h = g[g.hit == 1]
        h3 = h[h.hit_freq >= 3]
        summ[sub] = dict(
            n=int(len(g)), with_analogs=int((g.n_analogs > 0).sum()),
            hit_ge1=int(len(h)), hit_ge1_frac=round(len(h) / len(g), 3),
            hit_ge3=int(len(h3)), hit_ge3_frac=round(len(h3) / len(g), 3),
            hit_ge1_by_top_analogs={n: int((h.hit_ana_rank < n).sum()) for n in (1, 3, 5, 10) if n <= a.n_analogs},
            hit_ge3_rank_lt25=int((h3.hit_rank_ge3 < 25).sum()),
            hit_ge3_rank_lt200=int((h3.hit_rank_ge3 < 200).sum()),
            prod_median=float(g.n_prod.median()), prod_p90=float(g.n_prod.quantile(.9)),
            prod3_median=float(g.n_prod3.median()), prod3_p90=float(g.n_prod3.quantile(.9)))
    top_sm = Counter(P.loc[P.hit == 1, "hit_smirks"]).most_common(15)
    out = dict(bench="C3NP", n_analogs=a.n_analogs, ppm=a.ppm, rules=n_rules, summary=summ,
               hit_smirks_top15=top_sm, seconds=round(time.time() - T0), apply_seconds=round(time.time() - t1))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

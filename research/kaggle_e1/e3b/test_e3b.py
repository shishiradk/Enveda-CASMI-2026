"""Checks of the E3b stage (run: python research/kaggle_e1/e3b/test_e3b.py).

1. unit: a protected candidate (lib >= threshold) never moves and is never sent; protect_lib=None reproduces E3's selection.
2. bench: E3b's own select() + E3's rerank_molecule() on the bench lists and the Kaggle forward scores
   (results/bench/fm) -> paired MRR@25 difference versus no re-ordering, S1 / S2, honest ranker (clean_kf).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e3"))          # fm_rerank.py (E3's rule, unchanged)
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / "bench" / "fm"))
import fm_stage as S  # noqa: E402  (the E3b stage: HERE is first on sys.path)
from fm_rerank import rerank_molecule  # noqa: E402

assert Path(S.__file__).resolve().parent == HERE


def reorder(e, fwd_by_smiles, lams, topn=60, protect=0.6):
    """What run_fm_stage does for one molecule once the runner scores are in."""
    sel = S.select({"m": e}, {"m"}, topn, protect)
    n = len(e["smiles"])
    if "m" not in sel:
        return list(range(n)), 0
    f = sel["m"]["formulas"] + [None] * (n - len(sel["m"]["formulas"]))
    per = {}
    for tag, sc in fwd_by_smiles.items():
        col = [None] * n
        for i in sel["m"]["send"]:
            col[i] = sc.get(e["smiles"][i])
        per[tag] = col
    perm, _ = rerank_molecule(f, e.get("scores") or [], per, lams, topn)
    return perm, len(sel["m"]["send"])


def test_unit():
    e = dict(smiles=["CCO", "COC", "CCCO", "CC(C)O", "COCC"], scores=[1.0, 0.9, 0.8, 0.7, 0.6], lib=[0.9, 0.0, 0.0, 0.7, 0.1])
    fwd = {"ice": {"CCO": 0.0, "COC": 1.0, "CCCO": 0.1, "CC(C)O": 0.9, "COCC": 0.8}}
    perm, sent = reorder(e, fwd, {"ice": 5.0, "gl": 0.0})
    assert perm[0] == 0 and perm[3] == 3, perm              # protected slots
    assert perm[1] == 1                                     # COC is now alone in its formula group
    assert [perm[2], perm[4]] == [4, 2] and sent == 2       # C3H8O: only the two unprotected members were sent and swapped
    perm0, sent0 = reorder(e, fwd, {"ice": 5.0, "gl": 0.0}, protect=None)
    assert perm0[:2] == [1, 0] and sent0 == 5               # E3 behaviour: the library-matched candidate is demoted
    try:
        S.select({"m": dict(smiles=["CCO", "COC"])}, {"m"}, 60, 0.6)
        raise AssertionError("missing lib must raise (the notebook then keeps the engine lists)")
    except ValueError:
        pass
    print("unit ok")


def test_bench():
    import fm_lib as F
    D, _ = F.load()
    for prot, lam in ((0.6, 0.5), (None, 1.0)):
        for scen in ("S1", "S2"):
            d, dem, pro, sent = [], 0, 0, 0
            for m in sorted(D[scen]):
                l = D[scen][m]["lists"].get("clean_kf")
                if l is None or not l["smiles"]:
                    d.append(0.0)
                    continue
                b = F.rr_of(l["ok"])
                if "ice" not in l:
                    d.append(0.0)
                    continue
                e = dict(smiles=l["smiles"], scores=l["scores"], lib=l["lib"])
                fwd = {t: dict(zip(l["smiles"], l[t])) for t in ("ice", "gl")}
                perm, ns = reorder(e, fwd, {"ice": lam, "gl": lam}, 60, prot)
                a = F.rr_of(l["ok"], perm)
                d.append(a - b); dem += int(b == 1 and a < 1); pro += int(b < 1 and a == 1); sent += ns
            print(f"protect {prot} lam {lam} {scen}: {F.fmtd(F.boot(d))} | truths demoted from rank 1: {dem}, promoted to rank 1: {pro} | candidates sent {sent}")


if __name__ == "__main__":
    test_unit()
    test_bench()

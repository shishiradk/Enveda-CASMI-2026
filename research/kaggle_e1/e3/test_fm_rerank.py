"""Unit tests for fm_rerank (synthetic input).  Run: python research/kaggle_e1/e3/test_fm_rerank.py"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_rerank import list_changes, rerank_molecule, zscores  # noqa: E402

L1 = {"ice": 1.0, "gl": 1.0}


def test_zscores():
    z, cov = zscores([1.0, 2.0, 3.0])
    assert cov and [round(x, 9) for x in z] == [-1.0, 0.0, 1.0]
    z, cov = zscores([1.0, None, 3.0])
    assert cov and z[1] == 0.0 and abs(z[0] + 1 / math.sqrt(2)) < 1e-12
    assert zscores([1.0, None]) == ([0.0, 0.0], False)
    assert zscores([2.0, 2.0, 2.0]) == ([0.0] * 3, False)
    assert zscores([float("nan"), 1.0])[1] is False


def test_slots_preserved():
    # A B A C B A : formula A at 0,2,5 ; B at 1,4 ; C at 3
    f = ["A", "B", "A", "C", "B", "A"]
    r = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    fwd = {"ice": [0.1, 0.2, 0.2, 0.9, 0.9, 0.9], "gl": [None] * 6}
    perm, info = rerank_molecule(f, r, fwd, L1)
    assert sorted(perm) == list(range(6))
    assert perm[3] == 3                                   # other formula never moves
    assert {perm[0], perm[2], perm[5]} == {0, 2, 5} and {perm[1], perm[4]} == {1, 4}
    # group A: zr = [0.927, 0.132, -1.059]; z_ice of (.1,.2,.9) = [-0.688, -0.459, 1.147]
    # fused = [0.239, -0.326, 0.088] -> order 0, 5, 2
    assert [perm[0], perm[2], perm[5]] == [0, 5, 2]
    # group B: two members, zr = [.707, -.707], z_ice = [-.707, .707] -> exact tie -> order kept
    assert [perm[1], perm[4]] == [1, 4]
    assert info["groups"] == 2 and info["groups_covered"] == 2 and info["groups_changed"] == 1


def test_uncovered_groups_untouched():
    f = ["A", "A", "A", "B", "B"]
    r = [5, 4, 3, 2, 1]
    # A: only one member scored; B: equal scores -> no group is covered
    fwd = {"ice": [None, 0.9, None, 0.3, 0.3], "gl": [None, None, None, None, None]}
    perm, info = rerank_molecule(f, r, fwd, L1)
    assert perm == list(range(5)) and info["groups_covered"] == 0
    assert rerank_molecule(f, r, {}, L1)[0] == list(range(5))
    assert rerank_molecule(f, r, {"ice": []}, L1)[0] == list(range(5))


def test_missing_scores_use_ranker_only():
    # one formula; members 0 and 3 have no forward score -> forward z = 0 for them
    f = ["A"] * 4
    r = [4.0, 3.0, 2.0, 1.0]
    fwd = {"ice": [None, 0.1, 0.9, None]}
    perm, _ = rerank_molecule(f, r, fwd, {"ice": 1.0})
    # zr = [1.162, .387, -.387, -1.162]; z_ice = [0, -.707, .707, 0] -> fused [1.162, -.320, .320, -1.162]
    assert perm == [0, 2, 1, 3]
    # unscored members keep their ranker order relative to each other whatever the forward scores are
    for seed in range(200):
        rnd = random.Random(seed)
        n = rnd.randint(3, 12)
        rr = sorted((rnd.random() for _ in range(n)), reverse=True)
        ff = [rnd.random() if rnd.random() < 0.6 else None for _ in range(n)]
        p, _ = rerank_molecule(["A"] * n, rr, {"ice": ff}, {"ice": 1.0})
        miss = [i for i in p if ff[i] is None]
        assert miss == sorted(miss)


def test_two_models_and_lambda():
    f = ["A", "A", "A"]
    r = [3.0, 2.0, 1.0]                                   # zr = [1, 0, -1]
    fwd = {"ice": [0.1, 0.2, 0.3], "gl": [0.3, 0.2, 0.1]}  # cancel each other exactly
    assert rerank_molecule(f, r, fwd, L1)[0] == [0, 1, 2]
    assert rerank_molecule(f, r, fwd, {"ice": 1.0, "gl": 0.0})[0] == [0, 1, 2]      # zr + z_ice all equal -> tie -> kept
    assert rerank_molecule(f, r, fwd, {"ice": 2.0, "gl": 0.0})[0] == [2, 1, 0]
    # a model that scored a single member of the group contributes nothing
    fwd = {"ice": [0.1, 0.2, 0.9], "gl": [None, None, 0.0]}
    a = rerank_molecule(f, r, fwd, L1)[0]
    b = rerank_molecule(f, r, {"ice": fwd["ice"]}, L1)[0]
    assert a == b


def test_ties_keep_order():
    f = ["A"] * 4
    r = [1.0, 1.0, 1.0, 1.0]                              # zero ranker spread
    fwd = {"ice": [0.5, 0.9, 0.5, 0.9]}
    assert rerank_molecule(f, r, fwd, {"ice": 1.0})[0] == [1, 3, 0, 2]


def test_topn_and_unknown_formula():
    f = ["A", None, "A", None, "A", "A"]
    r = [6, 5, 4, 3, 2, 1]
    fwd = {"ice": [0.0, 0.5, 0.1, 0.5, 0.2, 1.0]}
    perm, _ = rerank_molecule(f, r, fwd, {"ice": 5.0}, topn=5)
    assert perm[5] == 5 and perm[1] == 1 and perm[3] == 3  # beyond topn / unknown formula stay
    assert [perm[0], perm[2], perm[4]] == [4, 2, 0]
    # missing ranker scores -> list position is used instead
    perm2, _ = rerank_molecule(["A"] * 3, [None, 1.0, float("nan")], {"ice": [0.1, 0.2, 0.3]}, {"ice": 0.5})
    assert perm2 == [0, 1, 2]
    assert rerank_molecule([], [], {"ice": []}, L1)[0] == []


def test_random_invariants():
    for seed in range(500):
        rnd = random.Random(1000 + seed)
        n = rnd.randint(1, 70)
        f = [rnd.choice(["A", "B", "C", "D", None]) for _ in range(n)]
        r = sorted((rnd.random() for _ in range(n)), reverse=True)
        fwd = {m: [rnd.choice([None, round(rnd.random(), 2)]) for _ in range(n)] for m in ("ice", "gl")}
        perm, _ = rerank_molecule(f, r, fwd, L1, topn=60)
        assert sorted(perm) == list(range(n))
        for new, old in enumerate(perm):
            assert f[new] == f[old]                        # every slot keeps its formula
            if new >= 60 or f[new] is None:
                assert new == old
        # no forward information at all -> identity
        assert rerank_molecule(f, r, {"ice": [None] * n, "gl": [None] * n}, L1)[0] == list(range(n))


def test_list_changes():
    assert list_changes(list("abcd"), list("abcd")) == {"top1": False, "order": False, "set": False}
    assert list_changes(list("abcd"), list("bacd"), k=2) == {"top1": True, "order": True, "set": False}
    assert list_changes(list("abcd"), list("acbd"), k=2) == {"top1": False, "order": True, "set": True}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "tests passed")

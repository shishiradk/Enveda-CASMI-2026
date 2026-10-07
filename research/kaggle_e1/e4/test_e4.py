"""Checks of the E4 stage (run: python research/kaggle_e1/e4/test_e4.py [E3b kernel output dir]).

Synthetic: lookup, isotonic score, missing counts, protected candidates, ties, window end, mu = 0 identity, missing raw
scores, forward replay after the prior.  With an E3b kernel output directory (eng_lists.json, fm_*.json,
submission.csv): the replay reproduces E3b's submitted rows exactly at mu = 0, through the notebook's own write rule.
"""
import copy
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e3"))           # fm_rerank.py (E3's rule, unchanged)
sys.path.insert(0, str(HERE.parent / "e3b"))          # fm_stage.py (E3b, unchanged)
sys.path.insert(0, str(HERE))
import e4_stage as E  # noqa: E402
import pop_stage as P  # noqa: E402

ROOT = HERE.parents[2]


def lookup_of(d):
    ks = sorted(d)
    return P.PopLookup(np.array(ks, dtype="S14"), [d[k][0] for k in ks], [d[k][1] for k in ks])


def K(i):
    return "K" * 12 + chr(65 + i // 26) + chr(65 + i % 26)


def entry(n, lib=None, pv=None, ours=None):
    pv = pv or [0.5 / (i + 1) for i in range(n)]
    return dict(smiles=[f"S{i}" for i in range(n)], keys=[K(i) for i in range(n)], scores=[1.0 - 0.01 * i for i in range(n)],
                lib=lib or [0.0] * n, pv=pv, ours=ours or list(pv), lib_max=max(lib or [0.0]))


def nokey(_s):
    return None                                       # synthetic SMILES: look up by the engine key only


def test_lookup():
    lk = lookup_of({K(0): (3, 1), K(2): (0, 0)})
    assert lk.get(K(0)) == (3, 1) and lk.get(K(2)) == (0, 0) and lk.get(K(1)) is None and lk.get(None) is None and lk.get("short") is None
    assert P.candidate_counts("x", K(0), lk, nokey) == (3, 1, "canon") and P.candidate_counts("x", K(1), lk, nokey) == (0, 0, "unknown")
    assert P.candidate_counts("x", K(1), lk, lambda s: K(0)) == (3, 1, "plain")       # the plain key of the SMILES wins
    try:
        P.PopLookup(np.array([K(1), K(0)], dtype="S14"), [1, 1], [0, 0])
        raise AssertionError("unsorted keys must raise")
    except ValueError:
        pass
    assert abs(P.pop_value(9, 0) - np.log(10)) < 1e-12 and P.pop_value(0, 0) == 0.0
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(Path(d) / "pop_lookup.npz", keys=lk.keys, n_sid=lk.n_sid.astype(np.int32), n_pmid=lk.n_pmid.astype(np.int32))
        assert P.load_lookup(P.find_lookup(d)).get(K(0)) == (3, 1)


def test_isotonic():
    assert P.isotonic_nonincreasing([3, 2, 1]) == [3, 2, 1]
    assert P.isotonic_nonincreasing([1, 3, 2]) == [2, 2, 2]
    assert P.isotonic_nonincreasing([5, 1, 3, 0]) == [5, 2, 2, 0]
    v = P.isotonic_nonincreasing(list(np.random.default_rng(0).normal(size=200)))
    assert all(a >= b for a, b in zip(v, v[1:]))


def test_prior():
    n = 6
    e = entry(n)
    lk = lookup_of({K(4): (2000, 500), K(1): (1, 0), K(2): (0, 0)})                    # K(0), K(3), K(5): unknown to the table
    new, st = P.run_pop_stage({"m": e}, lk, mu=0.25, key_fn=nokey)
    o = [int(s[1:]) for s in new["m"]["smiles"]]
    assert o[0] == 4, o                                                               # the famous candidate moves up
    assert [x for x in o if x != 4] == [0, 1, 2, 3, 5] or o.index(1) < o.index(2), o  # the rest keeps the ranker order (small priors)
    assert new["m"]["scores"] == e["scores"]                                          # scores stay with the slots
    assert new["m"]["keys"] == [K(i) for i in o] and new["m"]["pv"] == [e["pv"][i] for i in o]
    assert st["n_unknown_to_table"] == 3 and st["n_with_record"] == 2 and st["n_found_canon"] == 3 and st["n_with_pubmed"] == 1
    assert e["smiles"] == [f"S{i}" for i in range(n)]                                 # input untouched
    # candidates without counts compete with pop = 0: with equal counts everywhere the order cannot change
    new, _ = P.run_pop_stage({"m": e}, lookup_of({K(i): (7, 7) for i in range(n)}), mu=5.0, key_fn=nokey)
    assert new["m"]["smiles"] == e["smiles"]
    new, _ = P.run_pop_stage({"m": e}, lookup_of({}), mu=5.0, key_fn=nokey)
    assert new["m"]["smiles"] == e["smiles"]


def test_protected():
    e = entry(6, lib=[0.95, 0.0, 0.7, 0.0, 0.0, 0.59])
    lk = lookup_of({K(5): (5000, 5000), K(4): (500, 50), K(2): (9, 9), K(0): (0, 0)})
    new, st = P.run_pop_stage({"m": e}, lk, mu=1.0, key_fn=nokey)
    o = [int(s[1:]) for s in new["m"]["smiles"]]
    assert o[0] == 0 and o[2] == 2, o                 # protected slots, although K(0) has no record and others are famous
    assert o == [0, 5, 2, 4, 1, 3], o                 # the unprotected ones are sorted into slots 1, 3, 4, 5
    assert st["n_protected"] == 2
    new, _ = P.run_pop_stage({"m": e}, lk, mu=1.0, protect_lib=None, key_fn=nokey)
    assert new["m"]["smiles"][0] == "S5"              # without protection the library hit is overtaken
    try:
        P.run_pop_stage({"m": {k: v for k, v in e.items() if k != "lib"}}, lk, mu=1.0, key_fn=nokey)
        raise AssertionError("missing lib must raise")
    except ValueError:
        pass


def test_ties_and_window():
    e = entry(5, pv=[0.2] * 5)                        # identical ranker scores
    lk = lookup_of({K(i): (10, 0) for i in range(5)})
    new, _ = P.run_pop_stage({"m": e}, lk, mu=0.25, key_fn=nokey)
    assert new["m"]["smiles"] == e["smiles"]          # exact ties keep their current order
    lk = lookup_of({K(0): (10, 0), K(1): (10, 0), K(2): (50, 0), K(3): (10, 0), K(4): (50, 0)})
    new, _ = P.run_pop_stage({"m": e}, lk, mu=0.25, key_fn=nokey)
    assert new["m"]["smiles"] == ["S2", "S4", "S0", "S1", "S3"]
    e = entry(8)
    lk = lookup_of({K(7): (10 ** 6, 10 ** 6), K(3): (100, 0)})
    new, _ = P.run_pop_stage({"m": e}, lk, mu=1.0, topn=5, key_fn=nokey)
    assert new["m"]["smiles"][5:] == e["smiles"][5:] and new["m"]["smiles"][0] == "S3"   # nothing enters or leaves the window
    # the isotonic score: where raw logits disagree with the list order the pair is tied and popularity decides
    e = entry(3, pv=[0.30, 0.31, 0.01])
    new, _ = P.run_pop_stage({"m": e}, lookup_of({K(1): (1, 0)}), mu=0.01, key_fn=nokey)
    assert new["m"]["smiles"] == ["S1", "S0", "S2"]
    lk = lookup_of({K(2): (10 ** 6, 10 ** 6)})
    for bad in ({k: v for k, v in entry(3).items() if k != "pv"}, dict(entry(3), ours=[0.1, float("nan"), 0.1])):
        try:
            P.run_pop_stage({"m": bad}, lk, mu=0.25, key_fn=nokey)
            raise AssertionError("raw scores missing for every molecule must raise in logit mode")
        except ValueError:
            pass
        new, st = P.run_pop_stage({"m": bad, "g": entry(3)}, lk, mu=0.25, key_fn=nokey)      # one bad molecule: it keeps its order
        assert new["m"]["smiles"] == bad["smiles"] and new["g"]["smiles"][0] == "S2" and st["n_molecules_skipped_no_raw_scores"] == 1
    new, _ = P.run_pop_stage({"m": {k: v for k, v in entry(3).items() if k != "pv"}}, lk, mu=0.25, score_mode="rank", key_fn=nokey)
    assert len(new["m"]["smiles"]) == 3


def test_mu0_identity():
    rng = np.random.default_rng(1)
    lists = {}
    for m in range(50):
        n = int(rng.integers(1, 61))
        pv = sorted(rng.random(n).tolist(), reverse=True)
        lists[str(m)] = entry(n, lib=(rng.random(n) * 0.9).tolist(), pv=pv, ours=rng.random(n).tolist())
    lk = lookup_of({K(i): (int(rng.integers(0, 500)), int(rng.integers(0, 50))) for i in range(0, 60, 2)})
    ref = copy.deepcopy(lists)
    for mode in ("logit", "rank"):
        new, st = P.run_pop_stage(lists, lk, mu=0.0, score_mode=mode, key_fn=nokey)
        assert all(new[m]["smiles"] == ref[m]["smiles"] and new[m]["keys"] == ref[m]["keys"] and new[m]["scores"] == ref[m]["scores"] for m in ref)
        assert st["n_molecules_reordered"] == 0
        flat = lookup_of({K(i): (12, 3) for i in range(60)})                           # the general path with a constant prior:
        new, st = P.run_pop_stage(lists, flat, mu=0.25, score_mode=mode, key_fn=nokey)  # the score alone keeps the order
        assert all(new[m]["smiles"] == ref[m]["smiles"] for m in ref), mode
    new, st = P.run_pop_stage(lists, lk, mu=0.25, key_fn=nokey)
    assert st["n_molecules_reordered"] > 0
    for m in ref:                                      # always a permutation; protected slots fixed
        assert sorted(new[m]["keys"]) == sorted(ref[m]["keys"])
        assert all(new[m]["keys"][i] == ref[m]["keys"][i] for i, x in enumerate(ref[m]["lib"]) if x >= 0.6)
    assert lists == ref                                # input untouched


def _fake_run(tmp, lists, fwd):
    """Write runner files as E3b's stage leaves them and return (E3b lists, stats) via E3b's own rule."""
    import fm_stage
    from fm_rerank import rerank_molecule
    covered = set(lists)
    sel = fm_stage.select(lists, covered, 60, 0.6)
    json.dump({m: {"smiles": [lists[m]["smiles"][i] for i in sel[m]["send"]]} for m in sel}, open(Path(tmp) / "fm_input.json", "w"))
    for tag in ("gl", "ice"):
        json.dump({"scores": {m: [fwd[tag].get(lists[m]["smiles"][i]) for i in sel[m]["send"]] for m in sel}, "meta": {}},
                  open(Path(tmp) / f"fm_{tag}.json", "w"))
    e3b = {}
    for m, e in lists.items():
        e3b[m] = e
        if m not in sel:
            continue
        n = len(e["smiles"])
        per = {}
        for tag in ("gl", "ice"):
            col = [None] * n
            for i in sel[m]["send"]:
                col[i] = fwd[tag].get(e["smiles"][i])
            per[tag] = col
        perm, _ = rerank_molecule(sel[m]["formulas"], e["scores"], per, {"ice": 0.5, "gl": 0.5}, 60)
        e3b[m] = dict(e, **{k: [e[k][i] for i in perm] for k in ("smiles", "keys", "scores", "lib")})
    return e3b, {"models": {t: {"status": "ok", "n_candidates_scored": 1} for t in ("gl", "ice")}}


def test_forward_replay():
    smi = ["CCO", "COC", "CCCO", "CC(C)O", "COCC", "CCN", "CNC"]
    e = dict(smiles=smi, keys=[K(i) for i in range(7)], scores=[1.0 - 0.1 * i for i in range(7)], lib=[0.9, 0, 0, 0, 0, 0, 0],
             pv=[0.9, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02], ours=[0.9, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02], lib_max=0.9)
    lists = {"m": e}
    fwd = {"gl": {"CCCO": 0.1, "CC(C)O": 0.9, "COCC": 0.5, "CCN": 0.2, "CNC": 0.21, "COC": 0.3},
           "ice": {"CCCO": 0.2, "CC(C)O": 0.8, "COCC": 0.5, "CCN": 0.3, "CNC": 0.31, "COC": None}}
    lk = lookup_of({K(6): (3000, 900), K(2): (40, 3), K(0): (1, 0)})
    with tempfile.TemporaryDirectory() as tmp:
        e3b, st = _fake_run(tmp, lists, fwd)
        orig = E.covered_molecules
        E.covered_molecules = lambda _p: {"m"}
        assert e3b["m"]["smiles"] != smi and e3b["m"]["smiles"][0] == "CCO"
        new0, s0 = E.run_e4(lists, e3b, st, lk, None, tmp, mu=0.0, key_fn=nokey)
        assert new0["m"]["keys"] == e3b["m"]["keys"] and new0["m"]["smiles"] == e3b["m"]["smiles"]      # mu = 0: E3b exactly
        assert s0["e4_vs_e3b"] == dict(n_top1_changed=0, n_top25_order_changed=0, n_top25_set_changed=0, top1_changed_molecules=[])
        new, s1 = E.run_e4(lists, e3b, st, lk, None, tmp, mu=0.25, key_fn=nokey)
        assert new["m"]["smiles"][0] == "CCO"                                                          # protected
        assert new["m"]["smiles"].index("CNC") < e3b["m"]["smiles"].index("CNC")                        # the famous amine moved up
        assert sorted(new["m"]["keys"]) == sorted(e["keys"]) and s1["forward_replayed"]
        # forward stage failed / off: the prior alone
        new2, s2 = E.run_e4(lists, lists, {}, lk, None, tmp, mu=0.25, key_fn=nokey)
        assert not s2["forward_replayed"] and new2["m"]["smiles"][0] == "CCO" and new2["m"]["smiles"][1] == "CNC"
        # a tampered E3b list is detected
        bad = {"m": dict(e3b["m"], keys=list(reversed(e3b["m"]["keys"])))}
        try:
            E.run_e4(lists, bad, st, lk, None, tmp, mu=0.25, key_fn=nokey)
            raise AssertionError("replay mismatch must raise")
        except AssertionError as exc:
            assert "replay differs" in str(exc)
        E.covered_molecules = orig


def test_real_e3b(out_dir):
    """mu = 0 on the real E3b Kaggle run: replay == the submitted rows; then the effect of the prior with our lookup."""
    out_dir = Path(out_dir)
    eng = json.load(open(out_dir / "eng_lists.json"))
    st = json.load(open(out_dir / "fm_stats.json"))
    sub = {r["molecule_id"]: r["smiles"].split(";") for r in csv.DictReader(open(out_dir / "submission.csv", encoding="utf-8"))}
    cov = E.covered_molecules(str(ROOT / "test.parquet"))
    by = E.scores_by_smiles(str(out_dir), st)
    rep, ch = E.apply_forward(eng, cov, by, {"ice": 0.5, "gl": 0.5}, 60, 0.6)
    same = sum(list(dict.fromkeys(rep[m]["smiles"]))[:25] == sub[m] for m in eng)
    print(f"real E3b run: replay rows identical to E3b's submission: {same} of {len(eng)} | changes vs engine {ch}")
    assert same == len(eng) and ch["top1"] == st["n_top1_changed"] and ch["order"] == st["n_top25_order_changed"] and ch["set"] == st["n_top25_set_changed"]
    lkp = ROOT / "results" / "kaggle_e4" / "pop_lookup" / "pop_lookup.npz"
    if not lkp.exists():
        print("no lookup yet: skipped the prior on the real lists")
        return
    lk = P.load_lookup(lkp)
    orig = E.covered_molecules
    E.covered_molecules = lambda _p: cov
    try:
        new0, s0 = E.run_e4(eng, rep, st, lk, None, str(out_dir), mu=0.0, score_mode="rank")
        assert all(new0[m]["keys"] == rep[m]["keys"] for m in eng) and s0["e4_vs_e3b"]["n_top25_order_changed"] == 0
        print("real E3b run, mu = 0: identical to E3b | candidates", s0["n_candidates"], "found by plain key", s0["n_found_plain"],
              "by engine key", s0["n_found_canon"], "unknown", s0["n_unknown_to_table"], "with a record", s0["n_with_record"],
              "with PubMed", s0["n_with_pubmed"], "| pop quantiles", s0.get("pop_quantiles_25_50_75_90_99"))
        for mu in (0.15, 0.25):
            new, s1 = E.run_e4(eng, rep, st, lk, None, str(out_dir), mu=mu, score_mode="rank")
            print(f"real E3b run, mu = {mu}, RANK score (the Kaggle lists of E3b have no raw probabilities): vs E3b", s1["e4_vs_e3b"]["n_top1_changed"],
                  s1["e4_vs_e3b"]["n_top25_order_changed"], s1["e4_vs_e3b"]["n_top25_set_changed"], "(top-1 / top-25 order / top-25 set)")
    finally:
        E.covered_molecules = orig


if __name__ == "__main__":
    test_lookup(); test_isotonic(); test_prior(); test_protected(); test_ties_and_window(); test_mu0_identity(); test_forward_replay()
    print("unit ok")
    if len(sys.argv) > 1:
        test_real_e3b(sys.argv[1])

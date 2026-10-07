"""Self-tests for research/c3gen/assets.py (loaders, rules_excluding, apply_rules).

    python research/c3gen/test_assets.py

Prints one PASS/FAIL line per test and exits non-zero on any failure. Peak ~1 GB (rules + support arrays).
"""
import sys
import time

import numpy as np

R = "D:/Enveda-CASMI-2026/"
sys.path.insert(0, R + "research/c3gen")
import assets as A  # noqa: E402

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not ok:
        FAIL.append(name)


def main():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    pool = A.load_np_pool(columns=["ik14", "smiles", "mass"])
    check("np_pool loads, ik14 unique", len(pool) > 0 and pool.ik14.is_unique, f"{len(pool):,} rows")
    rules = A.load_rules(min_freq=1)
    check("rules load, sorted by freq desc", len(rules) > 0 and bool((np.diff(rules.freq.to_numpy()) <= 0).all()),
          f"{len(rules):,} directed rules")
    check("each pid has exactly 2 directed rules", bool((rules.groupby("pid").size() == 2).all()))
    S = A.load_support()
    keys = pool.ik14.to_numpy()
    cnt = np.bincount(S["pid"], minlength=int(rules.pid.max()) + 1)
    check("support counts == rules.freq", bool((cnt[rules.pid.to_numpy()] == rules.freq.to_numpy()).all()))

    # 1. empty key set changes nothing
    R0 = A.rules_excluding(set(), min_support=1, rules=rules, support=S, pool_keys=keys)
    check("rules_excluding(empty) keeps all rules with freq unchanged",
          len(R0) == len(rules) and bool((R0.freq == R0.freq_all).all()))

    # 2. a rule supported only by pairs involving the excluded keys is dropped
    f3 = rules[rules.freq == 3].pid.unique()
    pid = int(f3[0])
    sel = S["pid"] == pid
    mol = np.unique(np.r_[S["a"][sel], S["b"][sel]])
    ex = set(keys[mol])
    R1 = A.rules_excluding(ex, min_support=1, rules=rules, support=S, pool_keys=keys)
    check("rule supported ONLY by excluded keys is dropped", pid not in set(R1.pid), f"pid {pid}, {len(ex)} keys")
    # independent recount for every rule
    bad = np.isin(keys, list(ex))
    ok = ~(bad[S["a"]] | bad[S["b"]])
    clean = np.bincount(S["pid"][ok], minlength=len(cnt))
    exp_kept = set(rules.pid[clean[rules.pid.to_numpy()] >= 1])
    check("kept set == independent recount (min_support=1)", set(R1.pid) == exp_kept,
          f"dropped {rules.pid.nunique() - len(exp_kept)} pids")
    check("freq == clean count", bool((R1.freq.to_numpy() == clean[R1.pid.to_numpy()]).all()))

    # 3. partial exclusion: excluding one molecule of a freq>=5 rule lowers its count but keeps it
    pid5 = int(rules[rules.freq >= 5].pid.iloc[len(rules[rules.freq >= 5]) // 2])
    sel = S["pid"] == pid5
    one = int(S["a"][sel][0])
    n_inv = int(((S["a"][sel] == one) | (S["b"][sel] == one)).sum())
    f_all = int(cnt[pid5])
    R2 = A.rules_excluding({keys[one]}, min_support=1, rules=rules, support=S, pool_keys=keys)
    row = R2[R2.pid == pid5]
    exp = f_all - n_inv
    check("partial exclusion recounts", (len(row) == 0 and exp == 0) or (len(row) == 2 and int(row.freq.iloc[0]) == exp),
          f"pid {pid5}: {f_all} -> {exp}")
    # min_support=3 threshold
    R3 = A.rules_excluding(ex, min_support=3, rules=rules, support=S, pool_keys=keys)
    check("min_support=3 keeps exactly clean>=3", set(R3.pid) == set(rules.pid[clean[rules.pid.to_numpy()] >= 3]))

    # 4. apply_rules reproduces the example pair of random rules (ex_a + rule -> ex_b)
    rng = np.random.default_rng(1)
    smi = pool.smiles.to_numpy()
    sample = rules.iloc[rng.choice(len(rules), 40, replace=False)]
    rep = 0; n = 0; prod_ok = True; mass_ok = True; dedupe_ok = True
    t1 = time.time()
    for r in sample.itertuples():
        a_s, b_s = smi[r.ex_a], smi[r.ex_b]
        mb = ExactMolWt(Chem.MolFromSmiles(b_s))
        sub = rules[rules.rule_id == r.rule_id]
        out = A.apply_rules(a_s, sub, target_mass=mb, ppm=10)
        n += 1
        kb = A.taut_key(b_s)
        rep += any(o["key"] == kb for o in out)
        for o in out:
            pm = Chem.MolFromSmiles(o["smiles"])
            prod_ok &= pm is not None
            mass_ok &= abs(o["mass"] - mb) <= mb * 10e-6 + 1e-9
        dedupe_ok &= len({o["key"] for o in out}) == len(out)
    check("apply_rules reproduces ex_b from ex_a (single rule)", rep >= 0.8 * n, f"{rep}/{n}, {time.time() - t1:.1f}s")
    check("apply_rules products sanitize", prod_ok)
    check("apply_rules products within ppm", mass_ok)
    check("apply_rules products unique by tautomer key", dedupe_ok)

    # 5. full rule set on a small natural product, mass-filtered and unfiltered
    q = "COc1cc(C=CC(=O)O)ccc1O"            # ferulic acid; +C6H10O5 (hexose) target
    mq = ExactMolWt(Chem.MolFromSmiles(q))
    t1 = time.time()
    out = A.apply_rules(q, rules[rules.freq >= 3], target_mass=mq + 162.05282, ppm=10)
    check("apply_rules ferulic acid + hexose finds glycosides", len(out) > 0,
          f"{len(out)} products, {time.time() - t1:.1f}s; top: {[o['smiles'] for o in out[:2]]}")
    print(f"total {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

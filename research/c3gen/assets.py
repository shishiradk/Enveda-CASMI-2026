"""Shared Class-3 generator assets: natural-product pool + single-cut MMP transformation rules.

Files (built by build_np_pool.py and build_mmp_rules.py):
  results/c3gen/np_pool.parquet          one row per plain InChIKey14 (row order = pool index used everywhere)
  results/c3gen/mmp_rules.parquet        directed rules  from_frag>>to_frag  ([*:1] = attachment point)
  results/c3gen/mmp_rule_support.parquet (pid, a, b): every distinct molecule pair supporting an unordered rule pair
                                          pid; a/b are np_pool row indices. Used for leak exclusion.

Key conventions (FACT, checked on 200 random rows each): np_pool.ik14 is the *plain* RDKit InChIKey14 of the source
(COCONUT coco_meta keys and train inchikey14 both match plain InChIKey on 200/200, tautomer-canonical on 180/200 and
195/200). The metric key is tautomer-canonical, so leak exclusion uses bench_truth_keys(), which unions the truth
'correct' keys, the plain keys of the truth SMILES, and results/c3np/forbidden.parquet (which already lists the raw and
tautomer-canonical aliases found in COCONUT / PubChem / the engine pool).

Typical use:
    import sys; sys.path.insert(0, 'D:/Enveda-CASMI-2026/research/c3gen'); import assets as A
    R = A.rules_excluding(A.bench_truth_keys(), min_support=3)       # leak-free rules for bench evaluation
    prods = A.apply_rules('CC(=O)Oc1ccccc1C(=O)O', R, target_mass=194.0579, ppm=10)
"""
import hashlib
import os

import numpy as np

ROOT = os.environ.get("CASMI_ROOT", "D:/Enveda-CASMI-2026").replace("\\", "/").rstrip("/") + "/"
POOL_PATH = ROOT + "results/c3gen/np_pool.parquet"
RULES_PATH = ROOT + "results/c3gen/mmp_rules.parquet"
SUPPORT_PATH = ROOT + "results/c3gen/mmp_rule_support.parquet"

MAX_VAR_HEAVY = 13      # variable part (the part that changes) at most 13 heavy atoms (a hexose-O is 12)
MAX_CUT_BONDS = 200     # rdMMPA default 20 silently skips large natural products

_G = {}


def _rd():
    if "Chem" not in _G:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        _G["Chem"] = Chem
    return _G["Chem"]


# ------------------------------------------------------------------ loaders
def load_np_pool(columns=None, path=POOL_PATH):
    """np_pool as a pandas DataFrame (columns: ik14, smiles, mass, formula, np_score, src, libs, mined).
    Small enough (~0.5M rows) to load; pass columns=[...] to project."""
    import pandas as pd
    return pd.read_parquet(path, columns=columns)


def load_rules(min_freq=3, max_var_heavy=None, path=RULES_PATH):
    """Directed MMP rules, sorted by freq desc. Columns: rule_id, pid (unordered pair id, shared by A>>B and B>>A),
    env (attachment atom of the constant: 'C', 'c', 'O', 'N', 'n', ...), from_frag, to_frag, smirks, delta_mass (to - from, Da), freq (distinct supporting molecule pairs),
    n_mols, from_heavy, to_heavy, ex_a, ex_b (one example pair, np_pool row indices; ex_a carries from_frag)."""
    import pandas as pd
    R = pd.read_parquet(path)
    R = R[R.freq >= min_freq]
    if max_var_heavy is not None:
        R = R[(R.from_heavy <= max_var_heavy) & (R.to_heavy <= max_var_heavy)]
    return R.reset_index(drop=True)


def load_support(path=SUPPORT_PATH):
    """(pid, a, b) int32 arrays: one row per distinct molecule pair supporting unordered rule pid."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return {c: t.column(c).to_numpy() for c in ("pid", "a", "b")}


def bench_truth_keys(include_forbidden=True):
    """Every key that must be treated as a bench truth molecule: truth.parquet 'correct' (tautomer-canonical),
    the plain InChIKey14 of the truth SMILES, and results/c3np/forbidden.parquet keys (aliases) when present."""
    import duckdb
    Chem = _rd()
    keys = set()
    for c, s in duckdb.sql(f"select correct, smiles from '{ROOT}results/bench/truth.parquet'").fetchall():
        keys |= set(c.split(";")) if c else set()
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            keys.add(Chem.MolToInchiKey(m)[:14])
    fp = ROOT + "results/c3np/forbidden.parquet"
    if include_forbidden and os.path.exists(fp):
        keys |= {k for (k,) in duckdb.sql(f"select distinct key from '{fp}'").fetchall()}
    return keys


def rules_excluding(keys, min_support=1, rules=None, support=None, pool_keys=None):
    """Leak hygiene: recount every rule using only supporting pairs where NEITHER molecule's ik14 is in `keys`.
    Rules whose clean support < min_support are dropped (min_support=1 = drop rules supported ONLY by pairs that
    involve the given keys; use 3 to keep the >=3 mining threshold after exclusion). Returns the rules frame with
    freq replaced by the clean count (original in freq_all)."""
    if rules is None:
        rules = load_rules(min_freq=1)
    if support is None:
        support = load_support()
    if pool_keys is None:
        pool_keys = load_np_pool(columns=["ik14"]).ik14.to_numpy()
    keys = set(keys)
    bad = np.fromiter((k in keys for k in pool_keys), bool, len(pool_keys))
    ok = ~(bad[support["a"]] | bad[support["b"]])
    npid = int(max(support["pid"].max(), rules.pid.max())) + 1
    clean = np.bincount(support["pid"][ok], minlength=npid)
    out = rules.copy()
    out["freq_all"] = out["freq"]
    out["freq"] = clean[out.pid.to_numpy()]
    out = out[out.freq >= min_support]
    return out.sort_values(["freq", "rule_id"], ascending=[False, True]).reset_index(drop=True)


# ------------------------------------------------------------------ chemistry helpers
def canon_frag(smi):
    """Canonical, non-isomeric SMILES of a fragment (keeps [*:1]); returns (smiles, heavy atoms excluding dummies)."""
    Chem = _rd()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, -1
    return Chem.MolToSmiles(m, isomericSmiles=False), sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)


def frag_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


_PATT = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]"   # rdMMPA's default single-cut bond pattern


def _fsmi(f):
    """Canonical non-isomeric SMILES of a one-dummy fragment Mol, the dummy written as [*:1]."""
    return _rd().MolToSmiles(f, isomericSmiles=False).replace("*", "[*:1]")


def _sym(at):
    s = at.GetSymbol()
    return s.lower() if at.GetIsAromatic() else s


def _env(frag):
    for at in frag.GetAtoms():
        if at.GetAtomicNum() == 0:
            nb = at.GetNeighbors()
            return _sym(nb[0]) if nb else "?"
    return "?"


def single_cuts(m, max_var_heavy=MAX_VAR_HEAVY, hcut=True, require_const_ge_var=True):
    """Set of (constant, variable, env): canonical fragment SMILES, both carrying [*:1], and env = the constant's
    attachment atom symbol (lower case when aromatic, e.g. 'C', 'c', 'O', 'N', 'n') so rules keep their context.
    - single acyclic cuts on rdMMPA's default bond pattern (via FragmentOnBonds), both orientations;
    - H-cuts: an implicit/explicit H replaced by the attachment (variable '[H][*:1]'), one per symmetry class.
    Filter: variable heavy atoms <= max_var_heavy and (optionally) constant heavy >= variable heavy.
    Strings are self-consistent between mining and apply_rules (same function): the dummy is an unlabeled '*'
    during canonicalisation and is written as [*:1] textually."""
    Chem = _rd()
    if "patt" not in _G:
        _G["patt"] = Chem.MolFromSmarts(_PATT)
    out = set()
    done = set()
    heavy = np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()] + [False, False], bool)
    for a1, a2 in m.GetSubstructMatches(_G["patt"]):
        bd = m.GetBondBetweenAtoms(a1, a2).GetIdx()
        if bd in done:
            continue
        done.add(bd)
        try:
            f = Chem.FragmentOnBonds(m, [bd], addDummies=True, dummyLabels=[(1, 1)])
            idx = Chem.GetMolFrags(f, asMols=False, sanitizeFrags=False)
        except Exception:
            continue
        if len(idx) != 2:
            continue
        h = [int(heavy[list(t)].sum()) for t in idx]      # dummies sit at the two appended indices (not heavy)
        ok = [(i, 1 - i) for i in (0, 1)
              if h[1 - i] <= max_var_heavy and (not require_const_ge_var or h[i] >= h[1 - i])]
        if not ok:
            continue
        fr = Chem.GetMolFrags(f, asMols=True, sanitizeFrags=False, frags=None, fragsMolAtomMapping=None)
        smi = [_fsmi(fr[0]), _fsmi(fr[1])]
        for ci, vi in ok:
            out.add((smi[ci], smi[vi], _env(fr[ci])))
    if hcut:
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        seen = set()
        for at in m.GetAtoms():
            if at.GetAtomicNum() <= 1 or at.GetTotalNumHs() == 0 or ranks[at.GetIdx()] in seen:
                continue
            seen.add(ranks[at.GetIdx()])
            rw = Chem.RWMol(m)
            d = Chem.Atom(0)
            d.SetIsotope(1)
            j = rw.AddAtom(d)
            x = rw.GetAtomWithIdx(at.GetIdx())
            if x.GetNumExplicitHs() > 0:
                x.SetNumExplicitHs(x.GetNumExplicitHs() - 1)
            x.SetNoImplicit(False)
            rw.AddBond(at.GetIdx(), j, Chem.BondType.SINGLE)
            try:
                mm = rw.GetMol()
                mm.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mm)
                out.add((_fsmi(mm), "[H][*:1]", _sym(at)))
            except Exception:
                pass
    return out


def frag_key(frag):
    """Hand-written fragment ('O[*:1]', 'CO[*]') -> the canonical form single_cuts / the mined rules use ('[*:1]O').
    apply_rules matches from_frag by exact string, so hand-made rule frames must pass their fragments through this."""
    if frag in ("[H][*:1]", "[H][*]", "[*][H]", "[*:1][H]"):
        return "[H][*:1]"
    m = _rd().MolFromSmiles(frag.replace("[*:1]", "*"))
    return _fsmi(m) if m is not None else None


def frag_mass(frag):
    """Exact mass of a [*:1]-fragment with the attachment capped by H (so delta = mass(to) - mass(from))."""
    Chem = _rd()
    from rdkit.Chem.Descriptors import ExactMolWt
    if frag == "[H][*:1]":
        return 2 * 1.00782503223
    m = Chem.MolFromSmiles(frag.replace("[*:1]", "[H]"))
    return float(ExactMolWt(m)) if m is not None else float("nan")


def _zip(const, var):
    """Join constant and variable fragments at [*:1]; returns a sanitized Mol or None."""
    Chem = _rd()
    try:
        if var == "[H][*:1]":
            m = Chem.MolFromSmiles(const.replace("[*:1]", "[H]"))
        else:
            m = Chem.molzip(Chem.MolFromSmiles(const), Chem.MolFromSmiles(var))
        if m is None:
            return None
        m = Chem.RemoveHs(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def taut_key(m_or_smi):
    """Tautomer-canonical InChIKey14 (the competition metric key; same recipe as casmi_engine.canon_key)."""
    Chem = _rd()
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if "te" not in _G:
        _G["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(m_or_smi) if isinstance(m_or_smi, str) else m_or_smi
        return Chem.MolToInchiKey(_G["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def _rule_index(rules):
    """{(from_frag, env): (sorted deltas array, [(to_frag, delta, freq, rule_id), ...] in delta order)}, cached."""
    key = ("idx", id(rules), len(rules), int(rules.rule_id.sum()) if len(rules) else 0)
    if _G.get("idx_key") != key:
        tmp = {}
        for f, e, t, d, q, r in zip(rules.from_frag.to_numpy(), rules.env.to_numpy(), rules.to_frag.to_numpy(),
                                    rules.delta_mass.to_numpy(), rules.freq.to_numpy(), rules.rule_id.to_numpy()):
            tmp.setdefault((f, e), []).append((t, float(d), int(q), int(r)))
        idx = {}
        for k, v in tmp.items():
            v.sort(key=lambda x: x[1])
            idx[k] = (np.array([x[1] for x in v]), v)
        _G["idx"], _G["idx_key"] = idx, key
    return _G["idx"]


def apply_rules(smiles, rules_subset, target_mass=None, ppm=10.0, abs_tol=0.0, max_products=None,
                taut_dedupe=True, exclude_keys=None, max_var_heavy=MAX_VAR_HEAVY):
    """Apply single-cut MMP rules once to `smiles`.

    rules_subset: frame with from_frag, env, to_frag, delta_mass, freq, rule_id (load_rules() / rules_excluding()).
                  from_frag must be in canonical single_cuts form ('[*:1]O', not 'O[*:1]'; see frag_key()).
                  A rule fires only where the cut's attachment atom (env) matches the atom it was mined on.
    target_mass: neutral monoisotopic mass; when given, only rules with mass(smiles) + delta within ppm of the
                 target (or abs_tol Da, whichever is larger) are tried (bisect on delta), and products are
                 re-checked on exact mass. None = every applicable rule (can be thousands of products).
    Returns a list of dicts {smiles, key, ik14, mass, rule_id, smirks, env, freq, const}: key is the
    tautomer-canonical InChIKey14 (metric key) when taut_dedupe=True (~0.05-0.3 s per product), else the plain
    InChIKey14; ik14 is always the plain key. Deduplicated by key (highest-freq rule kept), sorted by freq desc.
    exclude_keys: product keys to drop (checked against both key and ik14)."""
    Chem = _rd()
    from rdkit.Chem.Descriptors import ExactMolWt
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    idx = _rule_index(rules_subset)
    m0 = float(ExactMolWt(m))
    tol = None
    if target_mass is not None:
        tol = max(target_mass * ppm * 1e-6, abs_tol)
        lo_d, hi_d = target_mass - m0 - tol - 1e-4, target_mass - m0 + tol + 1e-4
    cuts = single_cuts(m, max_var_heavy=max_var_heavy, require_const_ge_var=False)
    cand = {}
    for const, var, env in cuts:
        ent = idx.get((var, env))
        if ent is None:
            continue
        ds, lst = ent
        if tol is not None:
            i0, i1 = np.searchsorted(ds, lo_d, "left"), np.searchsorted(ds, hi_d, "right")
            lst = lst[i0:i1]
        for to, d, q, rid in lst:
            p = _zip(const, to)
            if p is None:
                continue
            mass = float(ExactMolWt(p))
            if tol is not None and abs(mass - target_mass) > tol:
                continue
            s = Chem.MolToSmiles(p, isomericSmiles=False)
            if s not in cand or cand[s]["freq"] < q:
                cand[s] = dict(smiles=s, mass=mass, rule_id=rid, smirks=f"{var}>>{to}", env=env, freq=q,
                               const=const)
    out = {}
    excl = set(exclude_keys or ())
    for s, r in sorted(cand.items(), key=lambda kv: -kv[1]["freq"]):
        pm = Chem.MolFromSmiles(s)
        if pm is None:
            continue
        ik = Chem.MolToInchiKey(pm)[:14]
        k = taut_key(pm) if taut_dedupe else ik
        if not k or k in excl or ik in excl or k in out:
            continue
        r["key"] = k; r["ik14"] = ik
        out[k] = r
        if max_products and len(out) >= max_products:
            break
    return sorted(out.values(), key=lambda r: -r["freq"])

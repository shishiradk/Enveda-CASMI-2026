"""mmp-edit: Class-3 candidate generator that edits library analogs with mined matched-molecular-pair (MMP) rules.

Design: research/c3gen/designs/mmp-edit.md (+ judge grafts in designs/judges.json). Common generator interface:

    generate(ctx, k=200, forbidden=frozenset()) -> [(smiles, score, provenance), ...]   best first, <= k

ctx keys: mid, target (neutral monoisotopic mass), adducts, zlog (float32[6930]), analogs [{smiles, ik14, sim, dmass}],
window [{ik, smiles, fpscore}], spectra (unused here).

Algorithm (per molecule, deterministic, single core):
  1. Seeds (database inputs filtered by `forbidden` on plain ik14 / ckey):
       A  ctx['analogs'][:20]                          (spectral analogs; weight = analog sim)
       W  ctx['window'][:10] by fpscore                (same-mass parents: judges' formula-first graft)
       B  for the 30 most-supported rule deltas d: the 2 zlog-best structures of the train_pkg pool
          (COCONUT + train, results/train_pkg/data) at target - d (+-10 ppm)   (biotransform "Seeds B" graft)
  2. One-step single-cut MMP edits (assets.single_cuts, same strings as the mined rules): rules keyed by
     (variable, attachment env) whose delta matches target - mass(seed) within 10 ppm; env back-off (variable only)
     when a seed has no env-matched rule for its delta.
  3. Small non-MMP SMIRKS family (+-H2, +-H2O, +O epoxide, methylenedioxy closure) when the delta matches.
  4. Two-step edits (pairs of frequent small rules, d1 + d2 = delta; covers same-mass "move" edits) for the top
     analog/window seeds when one-step yields < 50 products and time allows.
  5. Validate (sanitize, |mass - target| <= 10 ppm), dedupe on plain InChIKey14 (no tautomer keys in here), drop
     products that are known database structures (window keys, np_pool / train_pkg pool keys) unless the key is in
     `forbidden` (outputs are never filtered by `forbidden`: the truth is the target).
  6. Two-stage score: stage 1 = z(Morgan-slice of fp @ zlog) + priors over all products; stage 2 = z(full 6930-bit
     prep_data fp @ zlog) + priors on the stage-1 top 400.  priors = L_FREQ*log1p(rule support)
     + L_SEED*seed weight + L_PROP*sum_s sim_s*Tc(product, analog s) - L_TWO*[two-step] - L_ENV*[env back-off]
     - L_RXN*[SMIRKS family].
Provenance: mmp:{A|W|B}{seed_rank}:{from}>>{to}:env{0|1}:n{support}:d{1|2}  (rxn:<name> for the SMIRKS family).

Rules: module default = assets.load_rules(min_freq=3) (all data: submission). For bench evaluation call
set_rules(assets.rules_excluding(<truth + forbidden keys>)) first (the CLI does this).

CLI (C3NP bench):
    python research/c3gen/mmp_edit.py --c3np --n 60 [--offset 0] [--workers 2] --out results/c3gen/mmp_edit_c3np.json
writes {mid: [smiles, ...]} (for research/scripts/c3np_eval.py), <out>.full.json ({mid: [[smiles, score, prov]]})
and <out>.summary.json (hit@1/@25/@200 by tautomer-canonical key, candidates, s/molecule, mass-valid share).
    --strip {tc085,tc070,edit1}: bench realism mode (research/scripts/c3np_strip.py): the truth's congeners are removed
    from the seed sources (ctx['exclude'] + filtered analogs / window) and treated as unknown structures. Bench MRR is
    not a Class-3 forecast; report it beside the strip numbers and c3np_eval's nb1 strata.
"""
import argparse
import bisect
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assets as A  # noqa: E402

ROOT = A.ROOT
TP = ROOT + "results/train_pkg/data/"
BITS_PATH = ROOT + "external/bench_inputs/coco/fp_bits.npy"
CLEAN_RULES = ROOT + "results/c3gen/work/mmp_rules_clean_c3np.parquet"
DEPLOY_DIR = None      # set_asset_dir(): deploy mode (Kaggle), every asset from one directory, no repo paths

PPM = 10.0
N_A, N_W, N_BD, N_BPER = 20, 10, 30, 2      # analog seeds, window seeds, Seeds-B deltas, structures per delta
MAX_PROD = 2500                             # product cap before scoring (by prior)
N_FULL = 400                                # stage-2 full-fingerprint rescoring
TWO_STEP_MIN = 50                           # do two-step only when one-step yields fewer products
# Deterministic work budgets are the real limits (r2: tightened so the worst C3NP molecule stays <= ~11 s on one
# free core; r1 budgets gave up to 14.7 s and, under CPU load, let the wall-clock guards fire). The wall-clock guards
# are only a hard failsafe for an overloaded machine. When one fires, feats['guard'] records it and every output
# provenance gets a '|guard=<gen|all>' suffix. T_ALL no longer mixes rescored and unrescored products: if it fires
# during stage 2, the WHOLE list falls back to the stage-1 order (deterministic unless T_GEN also fired).
W_GEN = 1.6e5                               # sum heavy(seed)*n_cuts + sum heavy(product) over zips
W_FULL = 3.5e6                              # sum heavy(product)**2.5 over stage-2 rescored products
W_VAL = 6.0e4                               # sum heavy(product) over validated products (InChIKey + mass)
T_GEN = 13.0                                # wall-clock failsafe for generation (s)
T_ALL = 18.0                                # wall-clock failsafe for stage-2 rescoring (s)
DROP_KNOWN = os.environ.get("MMP_EDIT_DROP_KNOWN", "1") != "0"   # drop products that are known DB structures
L_FREQ, L_SEED, L_PROP, L_TWO, L_ENV, L_RXN = 0.25, 1.0, 1.0, 0.75, 0.5, 0.5
SEED_W = {"W": 0.3, "B": 0.2}

# non-MMP edits (own SMARTS): name, reaction SMARTS, delta (product - parent), Da
RXNS = [
    ("red_CC", "[C:1]=[C:2]>>[C:1]-[C:2]", 2.01565),
    ("red_CO", "[CX3;!$(C-[O,N,S]):1]=[OX1:2]>>[C:1]-[O:2]", 2.01565),
    ("ox_CHOH", "[CX4;!H0:1]-[OX2H1:2]>>[C:1]=[O:2]", -2.01565),
    ("desat_CC", "[CX4;!H0:1]-[CX4;!H0:2]>>[C:1]=[C:2]", -2.01565),
    ("dehydrate", "[CX4;!H0:1]-[CX4:2]-[OX2H1]>>[C:1]=[C:2]", -18.01056),
    ("hydrate", "[C:1]=[C:2]>>[C:1]-[C:2]-[OH]", 18.01056),
    ("epoxide", "[C:1]=[C:2]>>[C:1]1-[C:2]-O-1", 15.99491),
    ("mdo", "[c:1](-[OX2H1:3]):[c:2]-[OX2:4]-[CH3:5]>>[c:1]1:[c:2]-[O:4]-[CH2:5]-[O:3]-1", -2.01565),
]

_S = {}


def set_asset_dir(d):
    """Deploy mode: read every asset from directory d ('c3_'-prefixed names, so no other Kaggle input's file-name
    lookups can collide): c3_pool_{mass,fp,key}.npy + c3_pool_smiles.txt (the full train_pkg pool, no held-key
    exclusions), c3_fp_bits.npy, c3_mmp_rules.parquet (all mined rules; installed with min_freq=3, the submission
    setting). Known-structure keys = the pool keys (the pool contains every np_pool structure: FACT, E7 build check).
    Nothing is written (pool_smiles offsets are kept in memory). Call before the first generate()."""
    global TP, BITS_PATH, DEPLOY_DIR
    DEPLOY_DIR = d.rstrip("/\\") + "/"
    TP = DEPLOY_DIR + "c3_"
    BITS_PATH = TP + "fp_bits.npy"
    for k in ("idx", "idx_var", "small_idx", "small_d", "seedb_d", "pmass", "pkey", "pfp", "known", "poff"):
        _S.pop(k, None)
    if "Chem" in _S:
        _S["bits"] = np.load(BITS_PATH)
        _S["mbit"] = _S["bits"] < 8192


def _rd():
    if "Chem" not in _S:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
        from rdkit.Chem.Descriptors import ExactMolWt
        RDLogger.DisableLog("rdApp.*")
        bits = np.load(BITS_PATH)
        _S.update(Chem=Chem, MACCS=MACCSkeys, EMW=ExactMolWt, bits=bits,
                  m2=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
                  m3=rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
                  rk=rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
                  mbit=bits < 8192,            # the Morgan part of the 6930-bit selection
                  rxns=[(n, AllChem.ReactionFromSmarts(s), d) for n, s, d in RXNS])
    return _S["Chem"]


# ------------------------------------------------------------------ assets
def set_rules(rules):
    """Install a rules frame (assets.load_rules / rules_excluding output) and build the lookup indexes."""
    idx, var_only = {}, {}
    f, e, t, d, q, r = (rules[c].to_numpy() for c in ("from_frag", "env", "to_frag", "delta_mass", "freq", "rule_id"))
    for i in np.lexsort((r, d)):                      # delta order (rule_id breaks ties) -> deterministic
        idx.setdefault((f[i], e[i]), []).append((float(d[i]), t[i], int(q[i]), int(r[i])))
        var_only.setdefault(f[i], {})
        k = (t[i], round(float(d[i]), 6))
        o = var_only[f[i]].get(k)
        var_only[f[i]][k] = (float(d[i]), t[i], int(q[i]) + (o[2] if o else 0), min(int(r[i]), o[3]) if o else int(r[i]))

    def pack(lst):
        lst.sort(key=lambda x: (x[0], x[3]))
        return np.array([x[0] for x in lst]), lst
    _S["idx"] = {k: pack(v) for k, v in idx.items()}
    _S["idx_var"] = {k: pack(list(v.values())) for k, v in var_only.items()}
    # small rules for two-step edits: frequent rules with <= 4 heavy atoms each side, plus frequent larger ones
    hv = np.maximum(rules.from_heavy.to_numpy(), rules.to_heavy.to_numpy())
    order = np.lexsort((r, -q))
    small = [i for i in order if hv[i] <= 4][:300] + [i for i in order if 4 < hv[i] <= 12][:100]
    sm = {}
    for i in small:
        sm.setdefault((f[i], e[i]), []).append((float(d[i]), t[i], int(q[i]), int(r[i])))
    _S["small_idx"] = {k: pack(v) for k, v in sm.items()}
    _S["small_d"] = np.sort(np.unique(np.round(d[small], 5)))
    # Seeds-B deltas: most supported distinct deltas (|d| > 0.5 Da)
    import pandas as pd
    g = pd.DataFrame(dict(d=np.round(d, 3), q=q))
    g = g[g.d.abs() > 0.5].groupby("d").q.sum().sort_values(ascending=False, kind="mergesort")
    _S["seedb_d"] = [float(x) for x in g.index[:N_BD]]
    _S["rules_n"] = len(rules)


def _ensure_assets(pool=True):
    _rd()
    if "idx" not in _S:
        set_rules(A.load_rules(min_freq=3, path=TP + "mmp_rules.parquet") if DEPLOY_DIR else A.load_rules(min_freq=3))
    if pool and DEPLOY_DIR and not os.path.exists(TP + "pool_fp.npy"):
        raise FileNotFoundError(TP + "pool_fp.npy")
    if pool and "pmass" not in _S and os.path.exists(TP + "pool_fp.npy"):
        _S["pmass"] = np.load(TP + "pool_mass.npy")
        _S["pkey"] = np.load(TP + "pool_key.npy")
        _S["pfp"] = np.load(TP + "pool_fp.npy", mmap_mode="r")
    if "known" not in _S:                         # sorted |S14 array (13 MB) instead of a Python set (~100 MB)
        kn = (np.zeros(0, "S14") if DEPLOY_DIR else
              A.load_np_pool(columns=["ik14"]).ik14.to_numpy().astype("S14"))
        if "pkey" in _S:
            kn = np.concatenate([kn, _S["pkey"]])
        _S["known"] = np.unique(kn)


def _is_known(ik):
    kn = _S["known"]
    b = ik.encode()
    j = int(np.searchsorted(kn, b))
    return j < len(kn) and kn[j] == b


def _pool_smiles(rows):
    """SMILES of train_pkg pool rows, read by byte offset (offsets cached in results/c3gen/work/)."""
    if _S.get("poff") is None:
        op = ROOT + "results/c3gen/work/pool_smiles_offsets.npy"
        if not DEPLOY_DIR and os.path.exists(op) and os.path.getmtime(op) > os.path.getmtime(TP + "pool_smiles.txt"):
            _S["poff"] = np.load(op)
        else:
            with open(TP + "pool_smiles.txt", "rb") as fh:
                data = fh.read()
            nl = np.frombuffer(data, np.uint8) == 10
            _S["poff"] = np.r_[0, np.flatnonzero(nl) + 1].astype(np.int64)
            del data, nl
            if DEPLOY_DIR:
                return _pool_smiles(rows)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            tmp = op + f".{os.getpid()}.tmp.npy"
            np.save(tmp, _S["poff"])
            os.replace(tmp, op)
    out = []
    with open(TP + "pool_smiles.txt", "rb") as fh:
        for i in rows:
            fh.seek(int(_S["poff"][i]))
            out.append(fh.readline().decode().rstrip("\r\n"))
    return out


def _byte_lut(zlog):
    """(867, 256) table: score of packed byte value v at byte position j = sum of zlog over its set bits, so
    unpackbits(fp)[:6930] @ zlog == lut[j, fp[:, j]].sum(1) without unpacking."""
    z = np.zeros(867 * 8, np.float32)
    z[:6930] = zlog
    bits = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32)   # (256, 8) MSB first
    return (z.reshape(867, 8) @ bits.T).astype(np.float32)


# ------------------------------------------------------------------ chemistry
def _ik(m):
    try:
        return _S["Chem"].MolToInchiKey(m)[:14]
    except Exception:
        return None


def _morgan_bits(m):
    return np.concatenate([_S["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                           _S["m3"].GetFingerprintAsNumPy(m).astype(np.uint8)])


def _full_fp(m, mg):
    """prep_data.fp_and_mass recipe (ECFP4 4096 | ECFP6 4096 | RDKitFP 2048 maxPath 6 | MACCS 167)[bits]."""
    v = np.concatenate([mg, _S["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                        np.array(_S["MACCS"].GenMACCSKeys(m), dtype=np.uint8)])
    return v[_S["bits"]]


def _seed_cuts(m):
    return sorted(A.single_cuts(m, require_const_ge_var=False))


def _rxn_products(m, delta, tol):
    Chem = _S["Chem"]
    out = []
    for name, rx, d in _S["rxns"]:
        if abs(d - delta) > tol + 1e-3:
            continue
        try:
            ps = rx.RunReactants((m,), maxProducts=200)
        except Exception:
            continue
        seen = set()
        for tup in ps:
            p = tup[0]
            try:
                Chem.SanitizeMol(p)
                s = Chem.MolToSmiles(p, isomericSmiles=False)
            except Exception:
                continue
            if s not in seen:
                seen.add(s)
                out.append((s, name))
    return sorted(out)


# ------------------------------------------------------------------ main entry
def generate(ctx, k=200, forbidden=frozenset(), return_feats=False):
    """Up to k (smiles, score, provenance), best first. See module docstring."""
    T0 = time.time()
    _ensure_assets()
    Chem = _S["Chem"]
    EMW = _S["EMW"]
    forbidden = set(forbidden or ())
    strip = set(ctx.get("exclude") or ())        # bench strip keys (c3np_strip.py): treated as not in any database
    excl = forbidden | strip                      # seed exclusion; outputs are never filtered by either
    target = float(ctx["target"])
    tol = target * PPM * 1e-6
    zlog = np.asarray(ctx["zlog"], np.float32) if ctx.get("zlog") is not None else None

    # ---- 1. seeds
    seeds = []                                   # (kind, rank, smiles, mol, weight)
    seen_seed = set()

    def add_seed(kind, rank, smi, keys, w):
        if not smi or any(x in excl for x in keys if x):
            return
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return
        ik = _ik(m)
        if ik is None or ik in excl or ik in seen_seed:
            return
        seen_seed.add(ik)
        seeds.append((kind, rank, Chem.MolToSmiles(m, isomericSmiles=False), m, float(w)))

    for i, a in enumerate((ctx.get("analogs") or [])[:N_A]):
        add_seed("A", i, a.get("smiles"), (a.get("ik14"), a.get("ckey")), max(float(a.get("sim") or 0), 0.0))
    win = sorted(ctx.get("window") or [], key=lambda w: -float(w.get("fpscore", w.get("score", 0)) or 0))
    window_keys = {w.get("ik") for w in ctx.get("window") or []} | {w.get("ckey") for w in ctx.get("window") or []}
    for i, w in enumerate(win[:N_W]):
        add_seed("W", i, w.get("smiles"), (w.get("ik"), w.get("ckey")), SEED_W["W"])
    if zlog is not None and "pmass" in _S:
        lut = _byte_lut(zlog)
        rows = []
        for d in _S["seedb_d"]:
            mt = target - d
            lo, hi = np.searchsorted(_S["pmass"], [mt - mt * PPM * 1e-6, mt + mt * PPM * 1e-6])
            if hi <= lo:
                continue
            ks = _S["pkey"][lo:hi]
            ok = np.array([x.decode() not in excl for x in ks], bool)
            if not ok.any():
                continue
            cand = np.arange(lo, hi)[ok]
            if len(cand) > 4000:                       # bounded memory / time
                cand = cand[:4000]
            sc = lut[np.arange(lut.shape[0])[None, :], np.asarray(_S["pfp"][cand])].sum(1)
            top = cand[np.lexsort((cand, -sc))[:N_BPER]]
            rows += [int(x) for x in top]
        for i, (r, s) in enumerate(zip(rows, _pool_smiles(rows))):
            add_seed("B", i, s, (_S["pkey"][r].decode(),), SEED_W["B"])

    t_seed = time.time() - T0
    # ---- 2-4. products
    P = {}                                       # smiles -> dict(prior fields)

    def put(s, kind, rank, w, frm, to, env_ok, freq, steps, rxn=None):
        r = P.get(s)
        prov = (f"rxn:{rxn}:{kind}{rank}" if rxn else
                f"mmp:{kind}{rank}:{frm}>>{to}:env{int(env_ok)}:n{freq}:d{steps}")
        cand = dict(w=w, freq=freq, two=int(steps == 2), env=int(not env_ok), rxn=int(rxn is not None), prov=prov)
        pr = _prior(cand)
        if r is None or pr > r["pr"]:
            cand["pr"] = pr
            P[s] = cand

    cut_cache = {}
    n_seed_done = 0
    work = 0.0
    guard = set()

    def over():
        if work > W_GEN:
            return True
        if time.time() - T0 > T_GEN:
            guard.add("gen")
            return True
        return False
    for kind, rank, ssmi, m, w in seeds:
        if over():
            break
        n_seed_done += 1
        delta = target - float(EMW(m))
        try:
            cuts = cut_cache.setdefault(ssmi, _seed_cuts(m))
        except Exception:
            continue
        work += m.GetNumHeavyAtoms() * len(cuts)
        got = 0
        for const, var, env in cuts:
            ent = _S["idx"].get((var, env))
            if ent is None:
                continue
            ds, lst = ent
            i0, i1 = bisect.bisect_left(ds, delta - tol - 1e-4), bisect.bisect_right(ds, delta + tol + 1e-4)
            for dd, to, q, rid in lst[i0:i1]:
                p = A._zip(const, to)
                if p is None:
                    continue
                work += p.GetNumHeavyAtoms()
                s = Chem.MolToSmiles(p, isomericSmiles=False)
                put(s, kind, rank, w, var, to, True, q, 1)
                got += 1
        if not got:                              # env back-off
            for const, var, env in cuts:
                ent = _S["idx_var"].get(var)
                if ent is None:
                    continue
                ds, lst = ent
                i0, i1 = bisect.bisect_left(ds, delta - tol - 1e-4), bisect.bisect_right(ds, delta + tol + 1e-4)
                for dd, to, q, rid in lst[i0:i1]:
                    p = A._zip(const, to)
                    if p is None:
                        continue
                    work += p.GetNumHeavyAtoms()
                    put(Chem.MolToSmiles(p, isomericSmiles=False), kind, rank, w, var, to, False, q, 1)
        for s, name in _rxn_products(m, delta, tol):
            put(s, kind, rank, w, "", "", True, 0, 1, rxn=name)

    n_one = len(P)
    if n_one < TWO_STEP_MIN:
        sd = _S["small_d"]
        for kind, rank, ssmi, m, w in [x for x in seeds if x[0] in "AW"][:6]:
            if over():
                break
            delta = target - float(EMW(m))
            inter = {}
            for const, var, env in cut_cache.get(ssmi) or _seed_cuts(m):
                ent = _S["small_idx"].get((var, env))
                if ent is None:
                    continue
                for d1, to, q, rid in ent[1]:
                    need = delta - d1                     # does some small rule supply the rest?
                    j = np.searchsorted(sd, need - tol - 2e-3)
                    if j >= len(sd) or sd[j] > need + tol + 2e-3:
                        continue
                    p = A._zip(const, to)
                    if p is None:
                        continue
                    work += p.GetNumHeavyAtoms()
                    s = Chem.MolToSmiles(p, isomericSmiles=False)
                    if s not in inter or inter[s][1] < q:
                        inter[s] = (p, q, var, to)
            for s1 in sorted(inter, key=lambda x: (-inter[x][1], x))[:40]:
                if over():
                    break
                p1, q1, v1, t1 = inter[s1]
                d2t = target - float(EMW(p1))
                try:
                    cuts2 = _seed_cuts(p1)
                except Exception:
                    continue
                work += p1.GetNumHeavyAtoms() * len(cuts2)
                for const, var, env in cuts2:
                    ent = _S["small_idx"].get((var, env))
                    if ent is None:
                        continue
                    ds, lst = ent
                    i0, i1 = bisect.bisect_left(ds, d2t - tol - 1e-4), bisect.bisect_right(ds, d2t + tol + 1e-4)
                    for dd, to, q, rid in lst[i0:i1]:
                        p = A._zip(const, to)
                        if p is None:
                            continue
                        work += p.GetNumHeavyAtoms()
                        s = Chem.MolToSmiles(p, isomericSmiles=False)
                        if s == ssmi:
                            continue
                        put(s, kind, rank, w, f"{v1}>>{t1};{var}", to, True, min(q, q1), 2)
    t_gen = time.time() - T0

    # ---- 5. validate, dedupe on plain ik14, drop known structures
    items = sorted(P.items(), key=lambda kv: (-kv[1]["pr"], kv[0]))
    prods, seen = [], set()
    vwork = 0
    for s, r in items:
        if len(prods) >= MAX_PROD or vwork > W_VAL:
            break
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        vwork += m.GetNumHeavyAtoms()
        mass = float(EMW(m))
        if abs(mass - target) > tol:
            continue
        ik = _ik(m)
        if ik is None or ik in seen:
            continue
        seen.add(ik)
        if ik in window_keys or ik in seen_seed or (DROP_KNOWN and ik not in forbidden and ik not in strip and _is_known(ik)):
            continue
        r.update(smiles=s, mol=m, ik=ik)
        prods.append(r)
    if not prods:
        return ([], dict(guard="+".join(sorted(guard)), n_seeds=len(seeds))) if return_feats else []

    t_val = time.time() - T0
    # ---- 6. scoring
    MG = np.stack([_morgan_bits(r["mol"]) for r in prods])                 # n x 8192 uint8
    pri = np.array([r["pr"] for r in prods], np.float32)
    # analog propagation: sum_s sim_s * Tc(product, seed) on Morgan r2 4096 bits
    an = [(x[3], x[4]) for x in seeds if x[0] == "A"]
    if an:
        SM = np.stack([_S["m2"].GetFingerprintAsNumPy(mm).astype(np.uint8) for mm, _ in an]).astype(np.float32)
        X = MG[:, :4096].astype(np.float32)
        inter = X @ SM.T
        un = X.sum(1)[:, None] + SM.sum(1)[None, :] - inter
        tc = inter / np.maximum(un, 1)
        prop = tc @ np.array([w for _, w in an], np.float32)
    else:
        prop = np.zeros(len(prods), np.float32)
    if zlog is not None:
        zm = zlog[_S["mbit"]]
        s1 = MG[:, _S["bits"][_S["mbit"]]].astype(np.float32) @ zm
        s1z = (s1 - s1.mean()) / (s1.std() + 1e-6)
    else:
        s1 = s1z = np.zeros(len(prods), np.float32)
    S1 = s1z + pri + L_PROP * prop
    o1 = np.lexsort((np.arange(len(prods)), -S1))
    full = np.full(len(prods), np.nan, np.float32)
    fwork = 0.0
    if zlog is not None:
        for i in o1[:N_FULL]:
            fwork += prods[i]["mol"].GetNumHeavyAtoms() ** 2.5
            if fwork > W_FULL:
                break
            if time.time() - T0 > T_ALL:
                guard.add("all")
                break
            try:
                full[i] = _full_fp(prods[i]["mol"], MG[i]).astype(np.float32) @ zlog
            except Exception:
                pass
    if "all" in guard:                            # failsafe: whole list in stage-1 order (no partial mixing)
        full[:] = np.nan
    fin = np.isfinite(full)
    S = S1 - 10.0                                 # not rescored -> below every rescored product
    if fin.any():
        f = full[fin]
        S[fin] = (f - f.mean()) / (f.std() + 1e-6) + pri[fin] + L_PROP * prop[fin]
    order = np.lexsort((np.arange(len(prods)), -S))[:k]
    gsuf = ("|guard=" + "+".join(sorted(guard))) if guard else ""
    out = [(prods[i]["smiles"], round(float(S[i]), 5), prods[i]["prov"] + gsuf) for i in order]
    if return_feats:
        feats = dict(work=work, vwork=vwork, fwork=fwork, guard="+".join(sorted(guard)),
                     t_seed=t_seed, t_gen=t_gen, t_val=t_val, n_full=int(fin.sum()), t_total=time.time() - T0, n_seeds=len(seeds), n_seed_done=n_seed_done,
                     n_one=n_one, n_raw=len(P), n_prod=len(prods),
                     rows=[dict(smiles=prods[i]["smiles"], ik=prods[i]["ik"], s1=float(s1[i]),
                                full=None if not np.isfinite(full[i]) else float(full[i]), pr=float(pri[i]),
                                prop=float(prop[i]), freq=prods[i]["freq"], w=prods[i]["w"], two=prods[i]["two"],
                                env=prods[i]["env"], rxn=prods[i]["rxn"], prov=prods[i]["prov"])
                           for i in o1[:1500]])
        return out, feats
    return out


def _prior(c):
    return (L_FREQ * math.log1p(c["freq"]) + L_SEED * c["w"] - L_TWO * c["two"] - L_ENV * c["env"]
            - L_RXN * c["rxn"])


# ------------------------------------------------------------------ CLI (C3NP bench)
def _clean_rules():
    """Leak-free rules for the C3NP bench: every supporting pair touching a bench truth / alias / forbidden key is
    removed (assets.rules_excluding), cached to CLEAN_RULES."""
    import pandas as pd
    if os.path.exists(CLEAN_RULES) and os.path.getmtime(CLEAN_RULES) > os.path.getmtime(A.RULES_PATH):
        return CLEAN_RULES
    keys = A.bench_truth_keys()
    F = pd.read_parquet(ROOT + "results/c3np/forbidden.parquet")
    M = pd.read_parquet(ROOT + "results/c3np/molecules.parquet")
    keys |= set(F.key) | set(M.ik14) | set(M.ckey) | {x for c in M.correct for x in c.split(";")}
    R = A.rules_excluding(keys, min_support=1, rules=A.load_rules(min_freq=1))
    os.makedirs(os.path.dirname(CLEAN_RULES), exist_ok=True)
    R.to_parquet(CLEAN_RULES, index=False)
    return CLEAN_RULES


def _winit(rules_path):
    import pandas as pd
    _rd()
    set_rules(pd.read_parquet(rules_path))
    _ensure_assets()


def _job(args):
    mid, ctx, forb, k = args
    t = time.time()
    try:
        out, feats = generate(ctx, k=k, forbidden=forb, return_feats=True)
        err = None
    except Exception as e:                                  # report, don't kill the batch
        import traceback
        out, feats, err = [], {}, traceback.format_exc()
    feats["sec"] = time.time() - t
    return mid, out, feats, err


def _score(M, res, k):
    """hit@1/25/200 by the metric key: plain ik14 match, or the tautomer-canonical key for same-formula products in
    the first 30 (deduplicated) positions or, below that, with Morgan Tc >= 0.5 to the truth (tautomer keys cost
    0.05-0.3 s each; a tautomer pair can have Tc as low as ~0.36, so hit@200 may miss a rare deep tautomer match)."""
    Chem = _rd()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    per = []
    for r in M.itertuples():
        ok = set(r.correct.split(";")) | {r.ik14, r.ckey}
        tm = Chem.MolFromSmiles(r.smiles)
        tf, tform = g.GetFingerprint(tm), CalcMolFormula(tm)
        out = res.get(r.mid, ([], {}, None))[0]
        rank, nvalid, seen = 0, 0, set()
        pos = 0
        for s, sc, pv in out:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            if abs(float(_S["EMW"](m)) - r.target) <= r.target * PPM * 1e-6:
                nvalid += 1
            ik = Chem.MolToInchiKey(m)[:14]
            key = ik
            if ik not in ok and CalcMolFormula(m) == tform and \
                    (pos < 30 or DataStructs.TanimotoSimilarity(tf, g.GetFingerprint(m)) >= 0.5):
                key = A.taut_key(m) or ik
            if key in seen:
                continue
            seen.add(key)
            pos += 1
            if not rank and (key in ok or ik in ok):
                rank = pos
        f = res.get(r.mid, ([], {}, None))[1]
        per.append(dict(mid=r.mid, subset=r.subset, rank=rank, n=len(out), valid=nvalid,
                        sec=f.get("sec", 0.0), n_seeds=f.get("n_seeds", 0), n_prod=f.get("n_prod", 0),
                        t_gen=f.get("t_gen", 0.0), guard=f.get("guard", "")))
    return per


def _summ(rows):
    n = len(rows)
    if not n:
        return {}
    rk = np.array([r["rank"] for r in rows])
    nc = np.array([r["n"] for r in rows])
    return dict(n=n, mrr25=round(float(np.where((rk > 0) & (rk <= 25), 1.0 / np.maximum(rk, 1), 0).mean()), 4),
                hit1=round(float((rk == 1).mean()), 4), hit25=round(float(((rk > 0) & (rk <= 25)).mean()), 4),
                hit200=round(float(((rk > 0) & (rk <= 200)).mean()), 4), mean_cands=round(float(nc.mean()), 1),
                zero_cands=int((nc == 0).sum()),
                mass_valid_share=round(float(sum(r["valid"] for r in rows) / max(nc.sum(), 1)), 4),
                sec_mean=round(float(np.mean([r["sec"] for r in rows])), 2),
                sec_max=round(float(np.max([r["sec"] for r in rows])), 2),
                mean_products=round(float(np.mean([r["n_prod"] for r in rows])), 1),
                guard_fired={g: sum(1 for r in rows if g in (r.get("guard") or "").split("+")) for g in ("gen", "all")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3np", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--out", default=ROOT + "results/c3gen/mmp_edit_c3np.json")
    ap.add_argument("--feats", action="store_true", help="also pickle per-product features (<out>.feats.pkl)")
    ap.add_argument("--strip", default="none", choices=["none", "tc085", "tc070", "edit1"],
                    help="bench realism mode: strip the truth's congeners from the seed sources (c3np_strip.py)")
    a = ap.parse_args()
    if not a.c3np:
        ap.error("only --c3np is implemented")
    import pandas as pd
    from multiprocessing import Pool
    T0 = time.time()
    a.workers = max(1, min(a.workers, 3))
    M = pd.read_parquet(ROOT + "results/c3np/molecules.parquet").iloc[a.offset:a.offset + a.n]
    F = pd.read_parquet(ROOT + "results/c3np/forbidden.parquet")
    forb = F.groupby("mid").key.apply(lambda s: frozenset(s)).to_dict()
    C = pickle.load(open(ROOT + "results/c3np/context.pkl", "rb"))["mols"]
    rp = _clean_rules()
    sys.path.insert(0, ROOT + "research/scripts")
    import c3np_strip
    skeys = c3np_strip.strip_keys(a.strip)
    print(time.strftime("%H:%M:%S"), f"{len(M)} molecules; clean rules {rp}", flush=True)
    jobs = []
    for r in M.itertuples():
        c = C[r.mid]
        ctx = dict(mid=r.mid, target=float(c["target"]), adducts=list(c["adducts"]), zlog=c["zlog"],
                   analogs=[dict(smiles=x["smiles"], ik14=x["ik14"], ckey=x.get("ckey"), sim=x["sim"],
                                 dmass=x["dmass"]) for x in c["analogs"]],
                   window=[dict(ik=x["ik"], ckey=x.get("ckey"), smiles=x["smiles"], fpscore=x["score"])
                           for x in c["window"]],
                   spectra=[])
        ctx = c3np_strip.apply_strip(ctx, skeys.get(r.mid, frozenset()))
        jobs.append((r.mid, ctx, forb.get(r.mid, frozenset()), a.k))
    del C
    res = {}
    with Pool(a.workers, initializer=_winit, initargs=(rp,)) as mp:
        for i, (mid, out, feats, err) in enumerate(mp.imap(_job, jobs, chunksize=1)):
            res[mid] = (out, feats, err)
            if err:
                print("ERROR", mid, err, flush=True)
            if i % 10 == 0 or i == len(jobs) - 1:
                print(time.strftime("%H:%M:%S"), f"{i + 1}/{len(jobs)} {mid} n={len(out)} "
                      f"{feats.get('sec', 0):.1f}s", flush=True)
    _rd()
    per = _score(M, res, a.k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({m: [s for s, _, _ in v[0]] for m, v in res.items()}, open(a.out, "w"))
    base = a.out[:-5] if a.out.endswith(".json") else a.out
    json.dump({m: v[0] for m, v in res.items()}, open(base + ".full.json", "w"))
    if a.feats:
        pickle.dump({m: v[1] for m, v in res.items()}, open(base + ".feats.pkl", "wb"), protocol=4)
    summ = dict(args=vars(a), n_errors=sum(v[2] is not None for v in res.values()),
                overall=_summ(per), by_subset={s: _summ([r for r in per if r["subset"] == s])
                                               for s in sorted({r["subset"] for r in per})},
                wall_seconds=round(time.time() - T0), per_mol=per)
    json.dump(summ, open(base + ".summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_mol"}, indent=1))


if __name__ == "__main__":
    main()

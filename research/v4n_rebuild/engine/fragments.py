"""Bond-cleavage fragments (MetFrag-lite) and subformula enumeration, plus the peak-explanation kernels (our code).

Fragment definition (v1 semantics, REBUILD_SPEC 3.6): connected components of the heavy-atom graph after removing one
or two bonds (ring bonds included), plus the intact molecule; component mass = atoms + implicit H; masses rounded to
1e-3 Da, unique, sorted.  The pool table stores them as float32 and run-time candidates get the same cast.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from . import chem

H = chem.H_MASS


def mol_graph(mol):
    """(atom weights incl. implicit H, CSR adjacency (start, nbr, edge id), n_edges) or None."""
    n = mol.GetNumAtoms()
    w = np.zeros(n, np.float64)
    for a in mol.GetAtoms():
        try:
            w[a.GetIdx()] = chem.mass_of(a.GetSymbol()) + a.GetTotalNumHs() * H
        except Exception:
            return None
    E = mol.GetNumBonds()
    deg = np.zeros(n + 1, np.int64)
    ends = np.zeros((E, 2), np.int64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ends[b.GetIdx()] = (i, j)
        deg[i + 1] += 1; deg[j + 1] += 1
    start = np.cumsum(deg)
    nbr = np.zeros(2 * E, np.int64); eid = np.zeros(2 * E, np.int64)
    fill = start[:-1].copy()
    for e in range(E):
        i, j = ends[e]
        nbr[fill[i]] = j; eid[fill[i]] = e; fill[i] += 1
        nbr[fill[j]] = i; eid[fill[j]] = e; fill[j] += 1
    return w, start, nbr, eid, E


@njit(cache=True)
def _label(n, start, nbr, eid, cut1, cut2, lab, stack):
    for i in range(n):
        lab[i] = -1
    nc = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = nc
        top = 1
        stack[0] = s
        while top > 0:
            top -= 1
            u = stack[top]
            for p in range(start[u], start[u + 1]):
                e = eid[p]
                if e == cut1 or e == cut2:
                    continue
                v = nbr[p]
                if lab[v] < 0:
                    lab[v] = nc
                    stack[top] = v
                    top += 1
        nc += 1
    return nc


@njit(cache=True)
def _frag_masses(w, start, nbr, eid, E, max_depth, max_out):
    n = len(w)
    buf = np.empty(max_out, np.float64)
    k = 0
    buf[k] = w.sum(); k += 1
    lab = np.empty(n, np.int64); stack = np.empty(n, np.int64); cm = np.zeros(n, np.float64)
    for e1 in range(E):
        for e2 in range(e1, E if max_depth >= 2 else e1 + 1):
            c2 = -1 if e2 == e1 else e2
            nc = _label(n, start, nbr, eid, e1, c2, lab, stack)
            if nc < 2:
                continue
            for c in range(nc):
                cm[c] = 0.0
            for i in range(n):
                cm[lab[i]] += w[i]
            for c in range(nc):
                if k < max_out:
                    buf[k] = cm[c]; k += 1
    r = np.empty(k, np.float64)
    for i in range(k):
        r[i] = np.floor(buf[i] * 1000.0 + 0.5) / 1000.0
    r.sort()
    out = np.empty(k, np.float64)
    m = 0
    for i in range(k):
        if m == 0 or r[i] != out[m - 1]:
            out[m] = r[i]; m += 1
    return out[:m]


def fragments_of_mol(mol, max_depth: int = 2, max_out: int = 200000) -> np.ndarray:
    """Sorted unique fragment masses, float32 (the table dtype)."""
    g = mol_graph(mol)
    if g is None:
        return np.zeros(0, np.float32)
    w, start, nbr, eid, E = g
    if E == 0:
        return np.array([np.floor(w.sum() * 1000.0 + 0.5) / 1000.0], np.float32)
    return _frag_masses(w, start, nbr, eid, E, max_depth, max_out).astype(np.float32)


def fragments_of_smiles(smiles, max_depth: int = 2) -> np.ndarray:
    c = chem.canonical(smiles)
    return np.zeros(0, np.float32) if c is None else fragments_of_mol(c[1], max_depth)


# ------------------------------------------------------------------------------------------ peak explanation
@njit(cache=True, fastmath=True)
def explained(masses, peak_mz, ion_delta, hmax, tol_da, tol_ppm):
    """bool per peak: some mass m with |peak - (m + ion_delta + s*H)| <= tol for integer s in [-hmax, hmax]."""
    n = len(peak_mz); M = len(masses)
    ok = np.zeros(n, np.bool_)
    if M == 0:
        return ok
    for i in range(n):
        p = peak_mz[i]
        t = max(tol_da, p * tol_ppm * 1e-6)
        for s in range(-hmax, hmax + 1):
            x = p - ion_delta - s * 1.00782503223
            j = np.searchsorted(masses, x)
            if (j < M and abs(masses[j] - x) <= t) or (j > 0 and abs(masses[j - 1] - x) <= t):
                ok[i] = True
                break
    return ok


def ion_offsets(adduct: str) -> List[float]:
    """Neutral fragment -> fragment ion mass offsets for a query adduct (protonated/deprotonated, plus the
    cationising / anion-attaching species named in the adduct)."""
    pol = 1
    info = chem.adduct_info(adduct)
    if info is not None:
        pol = info[3]
    elif isinstance(adduct, str) and adduct.endswith("-"):
        pol = -1
    a = adduct if isinstance(adduct, str) else ""
    e = chem.ELECTRON
    if pol > 0:
        out = [chem.PROTON]
        if "Na" in a:
            out.append(chem.mass_of("Na") - e)
        if "K" in a:
            out.append(chem.mass_of("K") - e)
        if "NH4" in a:
            out.append(chem.mass_of("N") + 4 * H - e)
        return out
    out = [-chem.PROTON]
    if "Cl" in a:
        out.append(chem.mass_of("Cl") + e)
    if "CH2O2" in a or "HCOO" in a:
        out.append(chem.mass_of("C") + 2 * H + 2 * chem.mass_of("O") - chem.PROTON)
    return out


def frag_features(frags: np.ndarray, peaks, tol_da=0.01, tol_ppm=15.0, hshift=2) -> np.ndarray:
    """peaks: [(mz, rel. intensity, adduct)].  -> [max int frac, mean int frac, max count frac, max top10 frac,
    max strict (H-shift <= 1) int frac]; intensities weighted by sqrt."""
    if len(frags) == 0 or not peaks:
        return np.zeros(5, np.float32)
    fr = np.asarray(frags, np.float64)
    I, Cn, T, S = [], [], [], []
    for mz, it, ad in peaks:
        if len(mz) == 0:
            continue
        w = np.sqrt(np.asarray(it, np.float64)); tot = w.sum()
        if tot <= 0:
            continue
        mz = np.asarray(mz, np.float64)
        ok = np.zeros(len(mz), bool); ok1 = np.zeros(len(mz), bool)
        for d in ion_offsets(ad):
            ok |= explained(fr, mz, d, hshift, tol_da, tol_ppm)
            ok1 |= explained(fr, mz, d, 1, tol_da, tol_ppm)
        I.append(w[ok].sum() / tot); Cn.append(ok.mean())
        T.append(ok[np.argsort(-w)[:10]].mean()); S.append(w[ok1].sum() / tot)
    if not I:
        return np.zeros(5, np.float32)
    return np.array([max(I), float(np.mean(I)), max(Cn), max(T), max(S)], np.float32)


# ------------------------------------------------------------------------------------------ subformulas
_EL_MASS = None


def _el_mass():
    global _EL_MASS
    if _EL_MASS is None:
        _EL_MASS = np.array([chem.mass_of(e) for e in chem.ELEMS10])
    return _EL_MASS


@njit(cache=True)
def _subformulas(cnt, em, max_out):
    """All sub-formulas of cnt (C H N O P S F Cl Br I) with H <= 2C+N+2P+2 and RDBE >= -1, as sorted masses."""
    out = np.empty(max_out, np.float64)
    k = 0
    nhal = cnt[6] + cnt[7] + cnt[8] + cnt[9]
    for c in range(cnt[0] + 1):
        for n in range(cnt[2] + 1):
            for o in range(cnt[3] + 1):
                for p in range(cnt[4] + 1):
                    for s in range(cnt[5] + 1):
                        base = c * em[0] + n * em[2] + o * em[3] + p * em[4] + s * em[5]
                        hcap = min(cnt[1], 2 * c + n + 2 + 2 * p)
                        for hal in range(nhal + 1):
                            for f in range(min(cnt[6], hal) + 1):
                                for cl in range(min(cnt[7], hal - f) + 1):
                                    for br in range(min(cnt[8], hal - f - cl) + 1):
                                        io = hal - f - cl - br
                                        if io > cnt[9]:
                                            continue
                                        hm = base + f * em[6] + cl * em[7] + br * em[8] + io * em[9]
                                        for h in range(hcap + 1):
                                            if c + n + o + p + s + hal + h == 0:
                                                continue
                                            if 2 * c - (h + hal) + (n + p) + 2 < -2:
                                                continue
                                            if k >= max_out:
                                                r = out[:k].copy()
                                                r.sort()
                                                return r
                                            out[k] = hm + h * em[1]
                                            k += 1
    r = out[:k].copy()
    r.sort()
    return r


_SF_CACHE: dict = {}


def subformula_masses(formula: str, max_out=2000000) -> Optional[np.ndarray]:
    if formula in _SF_CACHE:
        return _SF_CACHE[formula]
    v = chem.formula_vec(formula)
    r = None if v is None else _subformulas(v, _el_mass(), max_out)
    if len(_SF_CACHE) > 20000:
        _SF_CACHE.clear()
    _SF_CACHE[formula] = r
    return r


def subformula_features(formula: str, peaks, tol_da=0.005, tol_ppm=10.0) -> np.ndarray:
    """[max int frac, mean int frac, count frac, top10 frac] of peaks explained as subformula ions (no H shift)."""
    sub = subformula_masses(formula)
    if sub is None or not peaks:
        return np.zeros(4, np.float32)
    I, Cn, T = [], [], []
    for mz, it, ad in peaks:
        if len(mz) == 0:
            continue
        w = np.sqrt(np.asarray(it, np.float64)); tot = w.sum()
        if tot <= 0:
            continue
        mz = np.asarray(mz, np.float64)
        ok = np.zeros(len(mz), bool)
        for d in ion_offsets(ad):
            ok |= explained(sub, mz, d, 0, tol_da, tol_ppm)
        I.append(w[ok].sum() / tot); Cn.append(ok.mean()); T.append(ok[np.argsort(-w)[:10]].mean())
    if not I:
        return np.zeros(4, np.float32)
    return np.array([max(I), float(np.mean(I)), max(Cn), max(T)], np.float32)

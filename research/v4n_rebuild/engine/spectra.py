"""Spectrum kernels (numba, our code): cleaning, entropy-weighted spectra, entropy / cosine / mass-shifted hybrid
similarity, and batched library scoring.

Similarity is the spectral entropy similarity of Li et al. (Nat. Methods 2021): weights p (sum 1), low-entropy spectra
re-weighted by p^(0.25 + 0.25 S) when S < 3, similarity = 1 - (2 S(AB) - S(A) - S(B)) / ln 4 on the 50/50 mixture.
Peak matching: greedy two-pointer walk over m/z-sorted lists, tolerance max(tol Da, ppm * query m/z).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

LN4 = float(np.log(4.0))
ZERO = 1e-6          # direct-search similarities at or below this are float noise (deliberate v1 difference)


@njit(cache=True)
def clean(mz, it, prec, floor, topk, over):
    """Library cleaning: drop m/z > prec + over (and m/z <= 0), keep intensities >= floor * base (base taken over the
    peaks <= prec + over), the topk most intense, sorted by m/z, intensity / base.  float32 outputs."""
    n = len(mz)
    base = 0.0
    for i in range(n):
        if mz[i] <= prec + over and it[i] > base:
            base = it[i]
    if base <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * base
    sel = np.empty(n, np.int64)
    c = 0
    for i in range(n):
        if mz[i] <= prec + over and it[i] >= thr and mz[i] > 0:
            sel[c] = i
            c += 1
    sel = sel[:c]
    if c > topk:
        vals = np.empty(c, np.float64)
        for i in range(c):
            vals[i] = it[sel[i]]
        sel = sel[np.argsort(vals)[c - topk:]]
        c = topk
    m = np.empty(c, np.float64)
    v = np.empty(c, np.float32)
    for i in range(c):
        m[i] = mz[sel[i]]
        v[i] = it[sel[i]] / base
    o = np.argsort(m)
    return m[o].astype(np.float32), v[o]


@njit(cache=True, fastmath=True)
def entropy_weights(it, power, ent_weight):
    n = len(it)
    p = np.empty(n, np.float64)
    tot = 0.0
    for i in range(n):
        p[i] = it[i] ** power
        tot += p[i]
    if tot <= 0:
        return p
    for i in range(n):
        p[i] /= tot
    if ent_weight:
        S = 0.0
        for i in range(n):
            if p[i] > 0:
                S -= p[i] * np.log(p[i])
        if S < 3.0:
            e = 0.25 + 0.25 * S
            tot = 0.0
            for i in range(n):
                p[i] = p[i] ** e
                tot += p[i]
            if tot > 0:
                for i in range(n):
                    p[i] /= tot
    return p


@njit(cache=True, fastmath=True)
def weigh(mz, it, floor, topk, power, ent_weight):
    """Search-time preparation (no precursor cut): relative floor, topk by intensity (kept in m/z order), weights."""
    n = len(mz)
    base = 0.0
    for i in range(n):
        if it[i] > base:
            base = it[i]
    if base <= 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    thr = floor * base
    sel = np.empty(n, np.int64)
    c = 0
    for i in range(n):
        if it[i] >= thr:
            sel[c] = i
            c += 1
    sel = sel[:c]
    if c > topk:
        vals = np.empty(c, np.float64)
        for i in range(c):
            vals[i] = it[sel[i]]
        keep = sel[np.argsort(vals)[c - topk:]]
        keep.sort()
        sel = keep
        c = topk
    m = np.empty(c, np.float64)
    v = np.empty(c, np.float64)
    for i in range(c):
        m[i] = mz[sel[i]]
        v[i] = it[sel[i]]
    return m, entropy_weights(v, power, ent_weight)


@njit(cache=True, fastmath=True)
def _H(p):
    s = 0.0
    for x in range(len(p)):
        if p[x] > 0:
            s -= p[x] * np.log(p[x])
    return s


@njit(cache=True, fastmath=True)
def _hv(v):
    return -v * np.log(v) if v > 0 else 0.0


@njit(cache=True, fastmath=True)
def entropy_similarity(am, ap, bm, bp, tol, ppm):
    n = len(am); m = len(bm)
    if n == 0 or m == 0:
        return 0.0
    sab = 0.0
    i = 0; j = 0
    while i < n and j < m:
        t = max(tol, am[i] * ppm * 1e-6)
        d = am[i] - bm[j]
        if d < -t:
            sab += _hv(0.5 * ap[i]); i += 1
        elif d > t:
            sab += _hv(0.5 * bp[j]); j += 1
        else:
            sab += _hv(0.5 * (ap[i] + bp[j])); i += 1; j += 1
    while i < n:
        sab += _hv(0.5 * ap[i]); i += 1
    while j < m:
        sab += _hv(0.5 * bp[j]); j += 1
    s = 1.0 - (2.0 * sab - _H(ap) - _H(bp)) / LN4
    return min(1.0, max(0.0, s))


@njit(cache=True, fastmath=True)
def cosine_similarity(am, ap, bm, bp, tol, ppm):
    n = len(am); m = len(bm)
    if n == 0 or m == 0:
        return 0.0
    na = 0.0; nb = 0.0; dot = 0.0
    for x in range(n):
        na += ap[x] * ap[x]
    for x in range(m):
        nb += bp[x] * bp[x]
    i = 0; j = 0
    while i < n and j < m:
        t = max(tol, am[i] * ppm * 1e-6)
        d = am[i] - bm[j]
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            dot += ap[i] * bp[j]; i += 1; j += 1
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / np.sqrt(na * nb)


@njit(cache=True, fastmath=True)
def shifted_similarity(am, ap, bm, bp, tol, ppm, shift):
    """Hybrid search: each reference peak matches a query peak directly, or else at reference m/z + shift (query peaks
    used once; direct matches first); entropy similarity of the resulting alignment."""
    n = len(am); m = len(bm)
    if n == 0 or m == 0:
        return 0.0
    if abs(shift) < 1e-4:
        return entropy_similarity(am, ap, bm, bp, tol, ppm)
    used = np.zeros(n, np.bool_)
    got = np.zeros(m, np.float64)
    i = 0; j = 0
    while i < n and j < m:
        t = max(tol, am[i] * ppm * 1e-6)
        d = am[i] - bm[j]
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            got[j] = ap[i]; used[i] = True; i += 1; j += 1
    i = 0; j = 0
    while i < n and j < m:
        t = max(tol, am[i] * ppm * 1e-6)
        d = am[i] - (bm[j] + shift)
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            if not used[i] and got[j] == 0.0:
                got[j] = ap[i]; used[i] = True
            i += 1; j += 1
    sab = 0.0
    for j in range(m):
        sab += _hv(0.5 * (bp[j] + got[j]))
    for i in range(n):
        if not used[i]:
            sab += _hv(0.5 * ap[i])
    s = 1.0 - (2.0 * sab - _H(ap) - _H(bp)) / LN4
    return min(1.0, max(0.0, s))


@njit(cache=True, fastmath=True, parallel=True)
def score_library(qm, qp, idx, off, lmz, lit, tol, ppm, floor, topk, power, ent_weight, kind):
    """Similarity of a weighted query to library spectra idx (CSR off/lmz/lit). kind 0 entropy, 1 cosine."""
    out = np.zeros(len(idx), np.float32)
    for k in prange(len(idx)):
        a = off[idx[k]]; b = off[idx[k] + 1]
        if b <= a:
            continue
        cm, cp = weigh(lmz[a:b], lit[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        v = entropy_similarity(qm, qp, cm, cp, tol, ppm) if kind == 0 else cosine_similarity(qm, qp, cm, cp, tol, ppm)
        out[k] = v if v > ZERO else 0.0       # no shared peak gives 0 exactly; drop +-1e-15 rounding noise
    return out


@njit(cache=True, fastmath=True, parallel=True)
def score_library_shift(qm, qp, idx, shift, off, lmz, lit, tol, ppm, floor, topk, power, ent_weight):
    out = np.zeros(len(idx), np.float32)
    for k in prange(len(idx)):
        a = off[idx[k]]; b = off[idx[k] + 1]
        if b <= a:
            continue
        cm, cp = weigh(lmz[a:b], lit[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        out[k] = shifted_similarity(qm, qp, cm, cp, tol, ppm, shift[k])
    return out


def merge(pairs, tol=0.005):
    """Merge peak lists of one molecule: each normalised to base 1, pooled, sorted; of two neighbours closer than
    `tol` (measured from the last kept peak) the stronger survives (ties: the later one)."""
    ms, vs = [], []
    for mz, it in pairs:
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        if len(mz) and it.max() > 0:
            ms.append(mz); vs.append(it / it.max())
    if not ms:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mz = np.concatenate(ms); it = np.concatenate(vs)
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    last = 0
    for j in range(1, len(mz)):
        if mz[j] - mz[last] < tol:
            if it[j] >= it[last]:
                keep[last] = False
                last = j
            else:
                keep[j] = False
        else:
            last = j
    return mz[keep].astype(np.float32), it[keep].astype(np.float32)


@njit(cache=True)
def clean_all(off, mz, it, prec, floor, topk, over):
    """clean() over a CSR block of spectra -> (offsets, mz float32, it float32)."""
    n = len(off) - 1
    oo = np.zeros(n + 1, np.int64)
    om = np.empty(len(mz), np.float32)
    oi = np.empty(len(mz), np.float32)
    p = 0
    for i in range(n):
        m, v = clean(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]], prec[i], floor, topk, over)
        om[p:p + len(m)] = m
        oi[p:p + len(m)] = v
        p += len(m)
        oo[i + 1] = p
    return oo, om[:p], oi[:p]

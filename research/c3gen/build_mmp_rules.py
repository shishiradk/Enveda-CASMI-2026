"""Mine single-cut matched-molecular-pair transformation rules inside results/c3gen/np_pool.parquet.

Molecules: np_pool rows with 150 <= mass <= 900; all train-derived rows first, then COCONUT rows in a fixed random
order (seed 0); at most --n molecules; fragmentation stops at --max_minutes (completed chunks are kept).
Fragmentation: assets.single_cuts (single acyclic cuts on rdMMPA's default bond pattern + H-cuts; variable <= 13
heavy atoms and constant >= variable). Index by constant (64-bit blake2b hash); every two molecules sharing a constant
with different variables give one supporting pair for the unordered rule {var_a, var_b}.
Groups larger than --gcap members are randomly subsampled (logged).
Outputs:
  results/c3gen/mmp_rules.parquet          directed rules (both directions), freq >= --min_freq
  results/c3gen/mmp_rule_support.parquet   (pid, a, b) np_pool row indices of every distinct supporting pair
  results/c3gen/mmp_rules_stats.json       counts, runtime, top rules
Checkpoints: results/c3gen/work/mmp/frag_XXXX.pkl (re-runs skip finished chunks).

Usage: python research/c3gen/build_mmp_rules.py [--n 150000] [--max_minutes 40] [--workers 3]
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

R = "D:/Enveda-CASMI-2026/"
sys.path.insert(0, R + "research/c3gen")
import assets as A  # noqa: E402

CHUNK = 1500


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _frag_chunk(args):
    i, idx, smis, work = args
    p = work + f"frag_{i:04d}.pkl"
    if os.path.exists(p):
        try:                                   # a shutdown can leave a zero-filled checkpoint (seen: frag_0048)
            with open(p, "rb") as fh:
                pickle.load(fh)
            return i, True
        except Exception:
            os.replace(p, p + f".corrupt{int(time.time())}")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    C, V, M, E = [], [], [], []
    vocab = {}; evocab = {}
    for pi, s in zip(idx, smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            cuts = A.single_cuts(m)
        except Exception:
            continue
        for c, v, e in cuts:
            C.append(A.frag_hash(c)); V.append(vocab.setdefault(v, len(vocab))); M.append(pi)
            E.append(evocab.setdefault(e, len(evocab)))
    d = dict(C=np.array(C, np.int64), V=np.array(V, np.int32), M=np.array(M, np.int32), E=np.array(E, np.int16),
             vocab=list(vocab), evocab=list(evocab), n=len(idx))
    with open(p + ".tmp", "wb") as fh:
        pickle.dump(d, fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(p + ".tmp", p)
    return i, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--max_minutes", type=float, default=40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min_freq", type=int, default=3)
    ap.add_argument("--gcap", type=int, default=300)
    ap.add_argument("--out", default=R + "results/c3gen/")
    ap.add_argument("--pool", default=A.POOL_PATH)
    a = ap.parse_args()
    import pandas as pd
    work = a.out + "work/mmp/"
    os.makedirs(work, exist_ok=True)
    T0 = time.time()
    P = pd.read_parquet(a.pool, columns=["ik14", "smiles", "mass", "src"])
    ok = ((P.mass >= 150) & (P.mass <= 900)).to_numpy()
    tr = np.where(ok & (P.src != "coconut").to_numpy())[0]
    co = np.where(ok & (P.src == "coconut").to_numpy())[0]
    rng = np.random.default_rng(0)
    order = np.r_[rng.permutation(tr), rng.permutation(co)][:a.n]
    log(f"pool {len(P):,}; 150-900 Da {int(ok.sum()):,} (train-derived {len(tr):,}, coconut-only {len(co):,}); "
        f"mining {len(order):,}")
    smi = P.smiles.to_numpy()
    jobs = [(i, order[s:s + CHUNK], list(smi[order[s:s + CHUNK]]), work)
            for i, s in enumerate(range(0, len(order), CHUNK))]
    del smi
    t1 = time.time()
    done = set()
    mp = Pool(a.workers)
    try:
        for n, (i, cached) in enumerate(mp.imap_unordered(_frag_chunk, jobs)):
            done.add(i)
            if n % 5 == 0 or n == len(jobs) - 1:
                el = time.time() - t1
                log(f"frag chunk {n + 1}/{len(jobs)}{' (cached)' if cached else ''}  {el:.0f}s")
            if time.time() - t1 > a.max_minutes * 60:
                log("time cap reached; stopping fragmentation")
                break
    finally:
        mp.terminate(); mp.join()
    frag_sec = time.time() - t1
    done = sorted(i for i, *_ in jobs if os.path.exists(work + f"frag_{i:04d}.pkl"))
    # ---- merge chunks into global arrays
    vocab = {}; evocab = {}
    Cs, Vs, Ms, Es = [], [], [], []
    mined = []
    for i in done:
        d = pickle.load(open(work + f"frag_{i:04d}.pkl", "rb"))
        remap = np.array([vocab.setdefault(v, len(vocab)) for v in d["vocab"]], np.int32)
        eremap = np.array([evocab.setdefault(v, len(evocab)) for v in d["evocab"]] or [0], np.int16)
        Cs.append(d["C"]); Vs.append(remap[d["V"]] if len(d["V"]) else d["V"]); Ms.append(d["M"])
        Es.append(eremap[d["E"]] if len(d["E"]) else d["E"])
        mined.append(jobs[i][1])
    C = np.concatenate(Cs); V = np.concatenate(Vs); M = np.concatenate(Ms); E = np.concatenate(Es)
    del Cs, Vs, Ms, Es
    estr = np.array(list(evocab), object)
    assert len(estr) < 127 and len(vocab) < (1 << 28)
    mined = np.concatenate(mined)
    vstr = np.array(list(vocab), object)
    log(f"mined molecules {len(mined):,}; cut rows {len(C):,}; distinct variables {len(vstr):,}; "
        f"frag {frag_sec:.0f}s")
    # ---- group by constant
    o = np.lexsort((M, V, C)); C, V, M, E = C[o], V[o], M[o], E[o]
    keep = np.r_[True, (C[1:] != C[:-1]) | (V[1:] != V[:-1]) | (M[1:] != M[:-1])]
    C, V, M, E = C[keep], V[keep], M[keep], E[keep]
    ncut = int(len(C))
    b = np.r_[0, np.where(C[1:] != C[:-1])[0] + 1, len(C)]
    sizes = np.diff(b)
    multi = np.where(sizes >= 2)[0]
    log(f"constants {len(sizes):,}; with >=2 molecules {len(multi):,}; largest {sizes.max():,}")
    PK, PA, PB = [], [], []
    capped = 0
    buf_k, buf_a, buf_b = [], [], []
    nbuf = 0
    for g in multi:
        s, e = b[g], b[g + 1]
        v = V[s:e]; m = M[s:e]
        if v.min() == v.max():
            continue
        if e - s > a.gcap:
            sel = np.sort(rng.choice(e - s, a.gcap, replace=False)); v = v[sel]; m = m[sel]; capped += 1
        i, j = np.triu_indices(len(v), 1)
        dv = v[i] != v[j]
        dv &= m[i] != m[j]
        i, j = i[dv], j[dv]
        if not len(i):
            continue
        sw = v[i] > v[j]
        lo_v = np.where(sw, v[j], v[i]).astype(np.int64); hi_v = np.where(sw, v[i], v[j]).astype(np.int64)
        lo_m = np.where(sw, m[j], m[i]); hi_m = np.where(sw, m[i], m[j])
        buf_k.append((np.int64(E[s]) << 56) | (lo_v << 28) | hi_v); buf_a.append(lo_m); buf_b.append(hi_m); nbuf += len(i)
        if nbuf > 2_000_000:
            PK.append(np.concatenate(buf_k)); PA.append(np.concatenate(buf_a)); PB.append(np.concatenate(buf_b))
            buf_k, buf_a, buf_b, nbuf = [], [], [], 0
    if nbuf:
        PK.append(np.concatenate(buf_k)); PA.append(np.concatenate(buf_a)); PB.append(np.concatenate(buf_b))
    del C, V, M, E
    K = np.concatenate(PK); PA = np.concatenate(PA).astype(np.int32); PB = np.concatenate(PB).astype(np.int32)
    del PK
    log(f"raw supporting pairs {len(K):,}; capped groups {capped}")
    # distinct (rule, a, b)
    o = np.lexsort((PB, PA, K)); K, PA, PB = K[o], PA[o], PB[o]
    keep = np.r_[True, (K[1:] != K[:-1]) | (PA[1:] != PA[:-1]) | (PB[1:] != PB[:-1])]
    K, PA, PB = K[keep], PA[keep], PB[keep]
    uk, start, cnt = np.unique(K, return_index=True, return_counts=True)
    log(f"distinct pairs {len(K):,}; unordered rules {len(uk):,}; with freq>={a.min_freq}: "
        f"{int((cnt >= a.min_freq).sum()):,}")
    kr = cnt >= a.min_freq
    uk, start, cnt = uk[kr], start[kr], cnt[kr]                  # still sorted by key
    ordr = np.lexsort((uk, -cnt))                                # pid = rank by frequency
    pid_sorted = np.empty(len(uk), np.int32); pid_sorted[ordr] = np.arange(len(uk), dtype=np.int32)
    pos = np.minimum(np.searchsorted(uk, K), max(len(uk) - 1, 0))
    sk = (uk[pos] == K) if len(uk) else np.zeros(len(K), bool)
    S_pid = pid_sorted[pos[sk]]
    S_a, S_b = PA[sk], PB[sk]
    uk, start, cnt = uk[ordr], start[ordr], cnt[ordr]
    # molecules per rule
    pm = np.unique(np.r_[S_pid.astype(np.int64) << 32 | S_a, S_pid.astype(np.int64) << 32 | S_b])
    nmol = np.bincount((pm >> 32).astype(np.int64), minlength=len(uk))
    # ---- rule table
    ev = (uk >> 56).astype(np.int64); va = ((uk >> 28) & 0xFFFFFFF).astype(np.int64)
    vb = (uk & 0xFFFFFFF).astype(np.int64)
    used = np.unique(np.r_[va, vb])
    fm = {}; fh = {}
    for u in used:
        s = vstr[u]; fm[u] = A.frag_mass(s)
        fh[u] = 0 if s == "[H][*:1]" else A.canon_frag(s)[1]
    ex_a = PA[start]; ex_b = PB[start]
    rows = []
    for p in range(len(uk)):
        x, y = int(va[p]), int(vb[p])
        for f, t, ea, eb in ((x, y, ex_a[p], ex_b[p]), (y, x, ex_b[p], ex_a[p])):
            rows.append((p, estr[ev[p]], vstr[f], vstr[t], fm[t] - fm[f], int(cnt[p]), int(nmol[p]), fh[f], fh[t], int(ea), int(eb)))
    Rt = pd.DataFrame(rows, columns=["pid", "env", "from_frag", "to_frag", "delta_mass", "freq", "n_mols", "from_heavy",
                                     "to_heavy", "ex_a", "ex_b"])
    Rt["smirks"] = Rt.from_frag + ">>" + Rt.to_frag
    Rt = Rt.sort_values(["freq", "pid", "delta_mass"], ascending=[False, True, True], kind="mergesort")
    Rt.insert(0, "rule_id", np.arange(len(Rt), dtype=np.int32))
    Rt = Rt[["rule_id", "pid", "env", "from_frag", "to_frag", "smirks", "delta_mass", "freq", "n_mols", "from_heavy",
             "to_heavy", "ex_a", "ex_b"]]
    Rt.to_parquet(a.out + "mmp_rules.parquet", index=False)
    pd.DataFrame(dict(pid=S_pid, a=S_a, b=S_b)).to_parquet(a.out + "mmp_rule_support.parquet", index=False)
    np.save(a.out + "mmp_mined_idx.npy", np.sort(mined).astype(np.int32))
    top = Rt.head(100)
    st = dict(pool_rows=int(len(P)), eligible_150_900=int(ok.sum()), mined_molecules=int(len(mined)),
              mined_train_derived=int(np.isin(mined, tr).sum()), cut_rows=ncut,
              distinct_variables=int(len(vstr)), constants_multi=int(len(multi)), capped_groups=int(capped),
              distinct_supporting_pairs=int(len(K)), unordered_rules_freq_ge_min=int(len(uk)),
              directed_rules=int(len(Rt)), support_rows=int(len(S_pid)), min_freq=a.min_freq, gcap=a.gcap,
              frag_seconds=round(frag_sec), total_seconds=round(time.time() - T0),
              top100_directed=[dict(env=r.env, smirks=r.smirks, delta=round(r.delta_mass, 5), freq=int(r.freq),
                                    n_mols=int(r.n_mols)) for r in top.itertuples()])
    json.dump(st, open(a.out + "mmp_rules_stats.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in st.items() if k != "top100_directed"}))


if __name__ == "__main__":
    main()

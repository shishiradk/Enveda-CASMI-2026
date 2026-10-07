"""Train the CFT model (contrastive spectrum -> fingerprint transformer) and export it.

    python train_cft.py --data <folder(s) with the .npy files> --out <output folder>

`--data` takes one or more folders: the warm-up data (casmi-train-pkg) and the CFT additions (casmi-train-cft);
they may also be the same folder. Run the SAME command again after any interruption: it resumes from the last
checkpoint. Exit codes: 0 finished, 2 not finished (run again), 3 GPU out of memory (use a smaller --bs).

Method (details and reasoning: research/analysis/train_cft_report.md):
  input    peak tokens (Fourier features of m/z and of precursor - m/z, sqrt intensity) + one global token
  output   logits z over the cfp1 fingerprint bits (+ element counts); candidate score = f . z
  loss     softmax cross-entropy of f.z over the truth and K same-mass (+-10 ppm) decoys from the pool
           + w_bce * BCE(z, truth bits) + w_elem * smooth-L1(element counts)
  metric   rank of the truth among ALL same-mass pool candidates for held-out molecules (MRR@25, top-1),
           and among sampled same-mass PubChem structures ("pcv")
"""
import argparse, contextlib, hashlib, json, math, os, platform, shutil, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cft_model as M

_jit = M._jit
_merge = M.merge_peaks
NEAR_ROWS = 3000        # filler negatives are drawn from this many pool rows on each side of the mass window
EXPORT_NAME = "cft_{name}.pt"


# ------------------------------------------------------------------------------------ numba kernels
@_jit
def _build_rows(members, row_off, off, mz, it, MZ, IT, PAD, L):
    """Row r = the spectrum members[row_off[r]] alone, or the merge of members[row_off[r]:row_off[r+1]]."""
    for r in range(len(row_off) - 1):
        a = row_off[r]; b = row_off[r + 1]
        if b - a == 1:
            s = off[members[a]]; n = min(off[members[a] + 1] - s, L)
            for j in range(n):
                MZ[r, j] = mz[s + j]; IT[r, j] = it[s + j]; PAD[r, j] = False
        else:
            tot = 0
            for q in range(a, b): tot += min(off[members[q] + 1] - off[members[q]], L)
            m = np.empty(tot, np.float32); v = np.empty(tot, np.float32); p = 0
            for q in range(a, b):
                s = off[members[q]]; n = min(off[members[q] + 1] - s, L)
                for j in range(n):
                    m[p] = mz[s + j]; v[p] = it[s + j]; p += 1
            m, v = _merge(m, v, L)
            for j in range(len(m)):
                MZ[r, j] = m[j]; IT[r, j] = v[j]; PAD[r, j] = False


@_jit
def _sample_negs(pos, lo, hi, kid, form, allow, U, K, n_hard, near, out):
    """out[b, 0] = pos[b]; out[b, 1:] = K decoys for row b, without repetition while the window lasts:
    first up to n_hard same-formula rows of the mass window [lo, hi), then other window rows, then filler
    rows of nearby mass. Rows with the key of the truth, or with allow == 0, are never used."""
    N = len(kid)
    for b in range(len(pos)):
        p = pos[b]; out[b, 0] = p
        w = hi[b] - lo[b]
        S = np.empty(w, np.int64); O = np.empty(w, np.int64); ns = 0; no = 0
        for i in range(lo[b], hi[b]):
            if kid[i] == kid[p] or allow[i] == 0: continue
            if form[i] == form[p]:
                S[ns] = i; ns += 1
            else:
                O[no] = i; no += 1
        k = 0; u = 0
        take = min(ns, n_hard)
        for t in range(take):                                   # partial shuffle of S
            j = t + int(U[b, u] * (ns - t)); u += 1
            if j >= ns: j = ns - 1
            S[t], S[j] = S[j], S[t]
            out[b, 1 + k] = S[t]; k += 1
        R = np.empty(ns - take + no, np.int64); nr = 0
        for t in range(take, ns):
            R[nr] = S[t]; nr += 1
        for t in range(no):
            R[nr] = O[t]; nr += 1
        take2 = min(nr, K - k)
        for t in range(take2):
            j = t + int(U[b, u] * (nr - t)); u += 1
            if j >= nr: j = nr - 1
            R[t], R[j] = R[j], R[t]
            out[b, 1 + k] = R[t]; k += 1
        a0 = max(lo[b] - near, 0); a1 = min(hi[b] + near, N)
        while k < K:                                            # filler: nearby mass, any formula
            j = a0 + int(U[b, u % U.shape[1]] * (a1 - a0)); u += 1
            if j >= a1: j = a1 - 1
            tries = 0
            while (kid[j] == kid[p] or allow[j] == 0) and tries < 50:
                j = j + 1 if j + 1 < a1 else a0
                tries += 1
            out[b, 1 + k] = j; k += 1


# ------------------------------------------------------------------------------------ data
def find_file(dirs, name):
    for d in dirs:
        p = Path(d) / name
        if p.exists(): return p
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in dirs]}")


class Data:
    def __init__(self, dirs, mmap=False, log=print, ppm=10.0, pubchem_negs=True):
        self.dirs = dirs = [Path(d) for d in dirs]
        big = lambda n: (np.load(find_file(dirs, n + ".npy"), mmap_mode="r").view(np.ndarray) if mmap
                         else np.load(find_file(dirs, n + ".npy")))
        ld = lambda n: np.load(find_file(dirs, n + ".npy"))
        self.meta = json.load(open(find_file(dirs, "meta.json")))
        self.cmeta = json.load(open(find_file(dirs, "cft_meta.json")))
        self.adducts = self.meta["lists"]["adducts"]; self.instruments = self.meta["lists"]["instruments"]
        self.ad_mode = M.mode_of_adducts(self.adducts)
        self.off = big("spec_off"); self.mz = big("spec_mz"); self.it = big("spec_it")
        self.prec = ld("spec_prec"); self.ad = ld("spec_ad").astype(np.int64); self.ins = ld("spec_ins").astype(np.int64)
        self.ce = ld("spec_ce"); self.lib = ld("spec_lib"); self.six = ld("spec_mol").astype(np.int64)
        split = ld("spec_split"); self.mol_split = ld("mol_split")
        self.fp = big("cpool_fp"); self.cmass = ld("cpool_mass"); self.kid = ld("cpool_kid").astype(np.int64)
        self.form = ld("cpool_form").astype(np.int64); self.elem = ld("cpool_elem"); self.src = ld("cpool_src")
        self.mol_cpool = ld("mol_cpool").astype(np.int64); self.mol_alt = ld("mol_alt").astype(np.int64)
        self.fp_bits = ld("cfp_bits"); self.nbits = int(self.cmeta["format"]["nbits"]); self.ppm = ppm
        assert len(self.fp_bits) == self.nbits and self.fp.shape[1] * 8 >= self.nbits
        assert len(self.cmass) == len(self.fp) and (np.diff(self.cmass) >= 0).all(), "pool must be sorted by mass"
        assert (self.mol_cpool[self.six] >= 0).all()
        self.allow = np.ones(len(self.cmass), np.uint8)
        if not pubchem_negs: self.allow[self.src == 2] = 0
        self.tr = np.where(split == 0)[0]; self.va = np.where(split == 1)[0]
        o = np.argsort(self.six, kind="mergesort")                      # spectra of a molecule
        self._mol_order = o; self._mol_bounds = np.searchsorted(self.six[o], np.arange(self.six.max() + 2))
        mk = self.six * 64 + self.ad                                     # merge groups: molecule x adduct
        o = np.argsort(mk, kind="mergesort"); self._mg_order = o
        _, first, inv, cnt = np.unique(mk[o], return_index=True, return_inverse=True, return_counts=True)
        self._mg_start = np.empty(len(mk), np.int64); self._mg_cnt = np.empty(len(mk), np.int64)
        self._mg_start[o] = first[inv]; self._mg_cnt[o] = cnt[inv]
        self._make_groups()
        log(f"data: nbits={self.nbits} pool={len(self.cmass):,} (PubChem decoys {int((self.src == 2).sum()):,}, "
            f"used as negatives: {pubchem_negs}) train={len(self.tr):,} val={len(self.va):,} spectra, "
            f"{len(np.unique(self.six[self.tr])):,} / {len(np.unique(self.six[self.va])):,} molecules")

    def _make_groups(self):
        """Balanced sampling groups over the training spectra: molecule x polarity."""
        g = self.six[self.tr] * 2 + (np.asarray(self.ad_mode)[self.ad[self.tr]] > 0)
        o = np.argsort(g, kind="mergesort"); self._g_order = self.tr[o]
        _, self._g_start, self._g_cnt = np.unique(g[o], return_index=True, return_counts=True)

    def subset(self, n_mol, seed=0):
        """Smoke test: keep a random subset of the training molecules."""
        r = np.random.default_rng(seed); mols = np.unique(self.six[self.tr])
        pick = r.choice(mols, size=min(len(mols), n_mol), replace=False)
        self.tr = self.tr[np.isin(self.six[self.tr], pick)]; self._make_groups()

    def window(self, pos):
        x = self.cmass[pos]; tol = x * self.ppm / 1e6
        return np.searchsorted(self.cmass, x - tol, "left"), np.searchsorted(self.cmass, x + tol, "right")

    # ---------------------------------------------------------------- training batches
    def batch(self, B, K, rng, group_frac=0.5, merge_p=0.3, hard_frac=0.5, alt_p=0.5, aug=True):
        """One training batch. Returns (dict of network inputs, cand [B, 1+K] pool rows (column 0 = truth),
        packed candidate fingerprints [B, 1+K, nbytes], element targets [B, n_elem])."""
        B = min(B, len(self.tr)); L = M.MAX_PEAKS
        ng = int(round(B * group_frac))
        ids = self.tr[rng.integers(0, len(self.tr), B - ng)]
        g = rng.integers(0, len(self._g_start), ng)
        ids = np.concatenate([ids, self._g_order[self._g_start[g] + (rng.random(ng) * self._g_cnt[g]).astype(np.int64)]])
        members = []; row_off = [0]
        do_merge = (rng.random(B) < merge_p) & (self._mg_cnt[ids] > 1)
        for r in range(B):
            i = int(ids[r])
            if do_merge[r]:
                c = int(self._mg_cnt[i]); k = int(rng.integers(2, min(4, c) + 1))
                pk = self._mg_order[self._mg_start[i] + rng.choice(c, size=k, replace=False)]
                if i not in pk: pk[0] = i
                members += pk.tolist()
            else:
                members.append(i)
            row_off.append(len(members))
        members = np.asarray(members, np.int64); row_off = np.asarray(row_off, np.int64)
        N = int(min(L, max(1, np.add.reduceat(np.minimum(self.off[members + 1] - self.off[members], L), row_off[:-1]).max())))
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), np.bool_)
        _build_rows(members, row_off, self.off, self.mz, self.it, mz, it, pad, N)
        prec = self.prec[ids].copy(); ce = self.ce[ids].copy(); nm = np.ones(B, np.float32)
        for r in np.where(do_merge)[0]:
            pk = members[row_off[r]:row_off[r + 1]]
            prec[r] = np.median(self.prec[pk]); c = self.ce[pk]; c = c[c >= 0]
            ce[r] = c.mean() if len(c) else -1.0; nm[r] = len(pk)
        if aug:
            n0 = (~pad).sum(1)
            keep = rng.random((B, N)) > rng.uniform(0.0, 0.30, size=(B, 1))               # peak dropout
            few = ((~pad) & keep).sum(1) < np.minimum(n0, 3)                              # never below 3 peaks
            keep[few] = True
            pad = pad | (~keep)
            it = np.minimum(it * np.exp(rng.normal(0.0, 0.30, size=(B, N))).astype(np.float32), 1.5)
            mz = mz * (1.0 + rng.normal(0.0, 3e-6, size=(B, N))).astype(np.float32)       # 3 ppm m/z jitter
            prec = prec * (1.0 + rng.normal(0.0, 3e-6, size=B)).astype(np.float32)
            ce = np.where(rng.random(B) < 0.10, -1.0, ce).astype(np.float32)              # hide the collision energy
        inp = dict(mz=mz, it=it, pad=pad, prec=prec.astype(np.float32), ad=self.ad[ids], ins=self.ins[ids],
                   ce=np.where(ce < 0, 0.0, ce).astype(np.float32), cek=(ce >= 0).astype(np.float32), nm=nm,
                   mode=np.asarray(self.ad_mode, np.float32)[self.ad[ids]])
        mol = self.six[ids]; pos = self.mol_cpool[mol]; alt = self.mol_alt[mol]
        pos = np.where((alt >= 0) & (rng.random(B) < alt_p), alt, pos)
        lo, hi = self.window(pos)
        cand = np.empty((B, K + 1), np.int64)
        _sample_negs(pos, lo, hi, self.kid, self.form, self.allow, rng.random((B, 2 * K + 2)), K,
                     int(round(K * hard_frac)), NEAR_ROWS, cand)
        return inp, cand, self.fp[cand], self.elem[pos]

    # ---------------------------------------------------------------- validation set
    def spectra_of(self, mol, q, seed):
        """Query of a validation molecule: up to q spectra from ONE library (timsTOF 'enveda-180' if the molecule
        has it, else its largest library), like a test molecule. Returns prepared spectrum dicts and spectrum ids."""
        ids = self._mol_order[self._mol_bounds[mol]:self._mol_bounds[mol + 1]]
        libs = self.lib[ids]
        L = 0 if (libs == 0).any() else int(np.bincount(libs).argmax())
        ids = np.sort(ids[libs == L])
        if len(ids) > q: ids = np.sort(np.random.default_rng([seed, int(mol)]).choice(ids, size=q, replace=False))
        return [dict(mz=np.asarray(self.mz[self.off[i]:self.off[i + 1]]), it=np.asarray(self.it[self.off[i]:self.off[i + 1]]),
                     prec=float(self.prec[i]), ad=int(self.ad[i]), ins=int(self.ins[i]), ce=float(self.ce[i]))
                for i in ids], ids

    def build_val(self, n_mols, q=6, seed=12345, pcv_limit=None, log=print):
        """Fixed validation set: `n_mols` held-out molecules for the pool-window metric (0 = all) plus the
        PubChem-validation molecules. Views are built with the same functions the inference API uses."""
        V = {}
        vm = np.where((self.mol_split == 1) & (self.mol_cpool >= 0))[0]
        vm = vm[np.isin(vm, np.unique(self.six[self.va]))]
        pick = vm if (n_mols <= 0 or n_mols >= len(vm)) else \
            np.sort(np.random.default_rng(seed).choice(vm, size=n_mols, replace=False))
        pm = np.load(find_file(self.dirs, "pcv_mol.npy")).astype(np.int64)
        poff = np.load(find_file(self.dirs, "pcv_off.npy"))
        npc = len(pm) if pcv_limit is None else min(pcv_limit, len(pm))
        mols = np.unique(np.concatenate([pick, pm[:npc]])); ix = {int(m): j for j, m in enumerate(mols)}
        views = []; owner = []; kind = []; nq = []
        for j, m in enumerate(mols):
            specs, _ = self.spectra_of(int(m), q, seed)
            v, k = M.make_views(specs)
            views += v; kind += k; owner += [j] * len(v); nq.append(len(specs))
        V.update(mols=mols, arrays=M.collate(views, self.ad_mode), owner=np.array(owner), kind=np.array(kind),
                 n_query=np.array(nq), in_pool_set=np.isin(mols, pick))
        truth = self.mol_cpool[mols]; lo, hi = self.window(truth)
        rows = []; own = []
        for j in np.where(V["in_pool_set"])[0]:
            r = np.arange(lo[j], hi[j]); r = r[self.kid[r] != self.kid[truth[j]]]
            rows.append(r); own.append(np.full(len(r), j, np.int64))
        V["c_rows"] = np.concatenate(rows); V["c_owner"] = np.concatenate(own); V["truth"] = truth
        V["c_iso"] = self.form[V["c_rows"]] == self.form[truth][V["c_owner"]]
        V["c_orig"] = self.src[V["c_rows"]] < 2
        # PubChem validation candidates
        V["p_fp"] = np.load(find_file(self.dirs, "pcv_fp.npy"), mmap_mode="r")
        V["p_off"] = poff[:npc + 1]; n = int(poff[npc])
        V["p_owner"] = np.repeat(np.array([ix[int(m)] for m in pm[:npc]]), np.diff(V["p_off"]))
        V["p_mol"] = np.array([ix[int(m)] for m in pm[:npc]])
        V["p_nfull"] = np.load(find_file(self.dirs, "pcv_nfull.npy"))[:npc]
        V["p_in_pc"] = np.load(find_file(self.dirs, "pcv_truth_in_pc.npy"))[:npc]
        V["p_truth_fp"] = np.load(find_file(self.dirs, "pcv_truth_fp.npy"))[:npc]
        V["p_iso"] = np.load(find_file(self.dirs, "pcv_form.npy"))[:n] == self.form[truth][V["p_owner"]]
        assert len(V["p_owner"]) == n
        log(f"validation: {int(V['in_pool_set'].sum()):,} molecules for the pool window "
            f"({len(V['c_rows']):,} candidates, median {np.median(np.bincount(V['c_owner'], minlength=len(mols))[V['in_pool_set']]):.0f} "
            f"per molecule), {npc} PubChem-validation molecules ({n:,} candidates), {len(views):,} views, "
            f"median {np.median(nq):.0f} spectra per query")
        return V


# ------------------------------------------------------------------------------------ ranking metric
def compare(Z, truth_packed, cand_packed, owner, nbits, dev, chunk=2048):
    """Z [n_mol, nbits] torch; truth_packed [n_mol, nbytes]; cand_packed(s, e) -> packed rows s:e; owner [C].
    Returns (gt, eq): for every candidate whether its score f.z is above / equal to the truth's score."""
    import torch
    st = torch.empty(len(Z), device=dev)
    for s in range(0, len(Z), chunk):
        f = M.unpack_bits(torch.as_tensor(np.array(truth_packed[s:s + chunk]), device=dev), nbits).float()
        st[s:s + chunk] = (f * Z[s:s + chunk]).sum(1)
    C = len(owner); gt = np.zeros(C, bool); eq = np.zeros(C, bool)
    for s in range(0, C, chunk):
        e = min(C, s + chunk)
        f = M.unpack_bits(torch.as_tensor(np.array(cand_packed(s, e)), device=dev), nbits).float()
        o = torch.as_tensor(owner[s:e], device=dev)
        d = (f * Z[o]).sum(1) - st[o]
        tol = 1e-6 * st[o].abs().clamp(min=1.0)
        gt[s:e] = (d > tol).cpu().numpy(); eq[s:e] = (d.abs() <= tol).cpu().numpy()
    return gt, eq


def summarize(gt, eq, owner, n_mol, mask=None, mols=None, n_full=None):
    """Rank = 1 + (#candidates above the truth) + 0.5 * (#ties). Molecules without a candidate are left out.
    With n_full (true window size per molecule) also the rank extrapolated from the sample to the full window."""
    if mask is None: mask = np.ones(len(owner), bool)
    g = np.bincount(owner[mask], weights=gt[mask], minlength=n_mol)
    e = np.bincount(owner[mask], weights=eq[mask], minlength=n_mol)
    n = np.bincount(owner[mask], minlength=n_mol)
    use = n > 0
    if mols is not None:
        sel = np.zeros(n_mol, bool); sel[mols] = True; use &= sel
    if not use.any(): return dict(n_mol=0)
    rank = (1 + g + 0.5 * e)[use]
    out = dict(mrr=float(np.where(rank <= 25, 1 / rank, 0).mean()), top1=float((rank < 1.25).mean()),
               n_mol=int(use.sum()), mean_cands=float(n[use].mean()), median_rank=float(np.median(rank)),
               chance_mrr=float(np.mean([sum(1 / r for r in range(1, min(k + 1, 25) + 1)) / (k + 1) for k in n[use]])))
    if n_full is not None:
        nf = np.zeros(n_mol); nf[mols] = n_full
        rx = 1 + (rank - 1) * np.maximum(nf[use] / n[use], 1.0)
        out.update(mrr_fullwindow_est=float(np.where(rx <= 25, 1 / rx, 0).mean()), top1_fullwindow_est=float((rx < 1.25).mean()),
                   mean_full_window=float(nf[use].mean()))
    return out


def evaluate(D, V, Z, dev):
    """All validation numbers for per-molecule logits Z [len(V['mols']), nbits] (torch, on dev)."""
    n = len(V["mols"]); R = {}
    tfp = D.fp[V["truth"]]
    gt, eq = compare(Z, tfp, lambda s, e: D.fp[V["c_rows"][s:e]], V["c_owner"], D.nbits, dev)
    R["pool"] = summarize(gt, eq, V["c_owner"], n)                                   # headline
    R["pool_isomers"] = summarize(gt, eq, V["c_owner"], n, V["c_iso"])               # same-formula candidates only
    R["pool_orig"] = summarize(gt, eq, V["c_owner"], n, V["c_orig"])                 # without the PubChem decoys
    # PubChem validation: truth in its PubChem form when PubChem has it (the real inference situation)
    pt = tfp.copy(); pm = V["p_mol"]; pt[pm[V["p_in_pc"]]] = V["p_truth_fp"][V["p_in_pc"]]
    gt, eq = compare(Z, pt, lambda s, e: V["p_fp"][s:e], V["p_owner"], D.nbits, dev)
    R["pcv"] = summarize(gt, eq, V["p_owner"], n, None, pm, V["p_nfull"])
    R["pcv_isomers"] = summarize(gt, eq, V["p_owner"], n, V["p_iso"], pm)
    gt, eq = compare(Z, tfp, lambda s, e: V["p_fp"][s:e], V["p_owner"], D.nbits, dev)
    R["pcv_truth_train_form"] = summarize(gt, eq, V["p_owner"], n, None, pm, V["p_nfull"])
    return R


def val_logits(net, V, dev, amp):
    import torch
    with torch.no_grad():
        z, _ = M.forward_views(net, V["arrays"], dev, amp)
        return M.combine_views(z, V["owner"], V["kind"], len(V["mols"]))


# ------------------------------------------------------------------------------------ helpers
class Prefetch:
    """Builds batches for steps start, start+1, ... in background threads, returned in order.
    Each step has its own RNG stream, so the batch sequence does not depend on threads or on resuming."""
    def __init__(self, fn, start, stop, workers=2, depth=6):
        self.fn, self.next, self.stop, self.depth = fn, start, stop, depth
        self.ex = ThreadPoolExecutor(max(1, workers)); self.q = deque()
        self._top()

    def _top(self):
        while len(self.q) < self.depth and self.next < self.stop:
            self.q.append(self.ex.submit(self.fn, self.next)); self.next += 1

    def get(self):
        b = self.q.popleft().result(); self._top(); return b

    def close(self):
        for f in self.q: f.cancel()
        self.ex.shutdown(wait=True)


def atomic_save(obj, path):
    import torch
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def keep_awake(on=True):
    """Ask Windows not to go to sleep while training (the screen may still turn off)."""
    if platform.system() != "Windows": return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""): h.update(blk)
    return h.hexdigest()


def is_oom(e):
    return "out of memory" in str(e).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="model name in file names (default = preset)")
    ap.add_argument("--preset", default="default", choices=list(M.PRESETS), help="default: 8 layers x 512; small: 4 x 256")
    ap.add_argument("--bs", type=int, default=256, help="batch size (256 is meant to fit 8 GB VRAM; use 128 if out of memory)")
    ap.add_argument("--steps", type=int, default=40000, help="training steps = length of the learning-rate schedule")
    ap.add_argument("--patience", type=int, default=12000, help="stop after this many steps without a better validation MRR")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=63, help="decoys per example")
    ap.add_argument("--ppm", type=float, default=10.0, help="half-width of the mass window for decoys and validation")
    ap.add_argument("--hard_frac", type=float, default=0.5, help="share of the decoys taken from same-formula structures first")
    ap.add_argument("--no_pubchem_negs", action="store_true", help="ablation: do not use the PubChem decoys as negatives")
    ap.add_argument("--alt_p", type=float, default=0.5, help="probability of using the PubChem form of the truth as the positive")
    ap.add_argument("--group_frac", type=float, default=0.5, help="share of a batch sampled uniformly over molecule x polarity")
    ap.add_argument("--merge_p", type=float, default=0.3, help="share of inputs that are 2-4 merged spectra (same molecule and adduct)")
    ap.add_argument("--w_bce", type=float, default=1.0); ap.add_argument("--w_elem", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.999, help="decay of the weight average that is validated and exported (0 = off)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val_every", type=int, default=2000); ap.add_argument("--val_mols", type=int, default=2000)
    ap.add_argument("--val_q", type=int, default=6, help="spectra per validation query")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ckpt_minutes", type=float, default=10.0)
    ap.add_argument("--max_minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit); run again to resume")
    ap.add_argument("--resume_from", default="", help="folder with checkpoints from an earlier session (Kaggle)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--device", default="auto"); ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU/GPU test run of the whole pipeline")
    ap.add_argument("--smoke_mols", type=int, default=2000)
    ap.add_argument("--stop_after", type=int, default=0, help="(test) pretend to be interrupted after N steps")
    a = ap.parse_args()
    explicit = {x.split("=")[0] for x in sys.argv[1:] if x.startswith("--")}
    if a.smoke:
        for k, v in dict(steps=80, warm=10, val_every=20, val_mols=100, log_every=5, patience=10 ** 9).items():
            if "--" + k not in explicit: setattr(a, k, v)
        a.bs = min(a.bs, 32); a.mmap = True
        if "--ckpt_minutes" not in explicit: a.ckpt_minutes = 0.2
    name = a.name or a.preset

    import torch, torch.nn.functional as F
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.txt", "a", encoding="utf-8")
    metf = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def log(*x):
        s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(v) for v in x)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    def metric(**kw):
        metf.write(json.dumps(kw) + "\n"); metf.flush()

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev == "cuda" and not a.no_amp
    T_RUN = time.time()
    log("=" * 30, "run start", "=" * 30)
    log(f"device={dev}" + (f" ({torch.cuda.get_device_name(0)}, "
                           f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GB)" if dev == "cuda" else ""),
        f"amp={amp} torch={torch.__version__} python={platform.python_version()} numba={M.HAVE_NUMBA} os={platform.system()}")
    if not M.HAVE_NUMBA: log("WARNING: numba is not installed; batch building will be slow (pip install numba)")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    keep_awake(True)
    ckp = out / f"ckpt_{name}.pt"; bestp = out / f"best_{name}.pt"; spath = out / f"state_{name}.json"
    if a.resume_from and not a.fresh:
        for p in Path(a.resume_from).rglob("*"):
            if p.name in {ckp.name, bestp.name, spath.name} and not (out / p.name).exists():
                shutil.copy2(p, out / p.name); log("copied", p.name, "from", a.resume_from)

    D = Data(a.data, mmap=a.mmap, log=log, ppm=a.ppm, pubchem_negs=not a.no_pubchem_negs)
    if a.smoke: D.subset(a.smoke_mols); log(f"SMOKE subset: train={len(D.tr):,} spectra")
    V = D.build_val(a.val_mols, q=a.val_q, pcv_limit=40 if a.smoke else None, log=log)
    mcfg = dict(M.PRESETS[a.preset], nbits=D.nbits, drop=a.drop, n_adduct=len(D.adducts), n_instr=len(D.instruments),
                n_elem=D.elem.shape[1])
    cfg = {k: v for k, v in vars(a).items() if k not in ("data", "out", "resume_from", "fresh", "stop_after",
                                                         "max_minutes", "device", "workers", "mmap", "ckpt_minutes")}
    state = json.load(open(spath)) if spath.exists() and not a.fresh else {"status": "pending"}

    def losses(net, inp, fpk, el):
        order = ("mz", "it", "pad", "prec", "ad", "ins", "ce", "cek", "nm", "mode")
        x = [torch.as_tensor(inp[k], device=dev) for k in order]
        with (torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()):
            z, e = net(*x)
        z = z.float(); e = e.float()
        fc = M.unpack_bits(torch.as_tensor(fpk, device=dev), D.nbits)                    # bool [B, 1+K, nbits]
        raw = torch.cat([(fc[:, s:s + 16] * z[:, None, :]).sum(-1)                      # f.z in fp32, in slices
                         for s in range(0, fc.shape[1], 16)], 1)                        # (keeps peak memory low)
        sc = (raw - raw.mean(1, keepdim=True)) / math.sqrt(D.nbits) * net.log_scale.clamp(-3, 3).exp()
        lc = F.cross_entropy(sc, torch.zeros(len(sc), dtype=torch.long, device=dev))
        lb = F.binary_cross_entropy_with_logits(z, fc[:, 0].float())
        le = F.smooth_l1_loss(e, torch.log1p(torch.as_tensor(el, device=dev).float()))
        return lc, lb, le, (sc.argmax(1) == 0).float().mean()

    def lr_at(s):
        if s < a.warm: return a.lr * (s + 1) / max(1, a.warm)
        p = (s - a.warm) / max(1, a.steps - a.warm)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def make(s):
        rng = np.random.default_rng([a.seed, 7, s])
        return D.batch(a.bs, a.K, rng, a.group_frac, a.merge_p, a.hard_frac, a.alt_p, aug=True)

    interrupted = False; steps_this_run = 0; why = "steps"
    if state.get("status") == "done" and bestp.exists():
        log(f"[{name}] already finished (best step {state.get('best_step')}); nothing to train")
    else:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        net = M.build_model(mcfg).to(dev)
        ema = M.build_model(mcfg).to(dev) if a.ema > 0 else None
        if ema is not None:
            ema.load_state_dict(net.state_dict()); ema.eval()
            for p in ema.parameters(): p.requires_grad_(False)
        nodecay = [p for n_, p in net.named_parameters() if p.ndim < 2 or "adduct" in n_ or "instr" in n_]
        decay = [p for n_, p in net.named_parameters() if not (p.ndim < 2 or "adduct" in n_ or "instr" in n_)]
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=nodecay, weight_decay=0.0)],
                                lr=a.lr, betas=(0.9, 0.98))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        step = 0; best = -1.0; best_step = 0; best_val = {}; hist = []; secs = 0.0
        npar = sum(p.numel() for p in net.parameters())
        if ckp.exists() and not a.fresh:
            ck = torch.load(ckp, map_location=dev, weights_only=False)
            net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            if ema is not None and ck.get("ema") is not None: ema.load_state_dict(ck["ema"])
            step, best, best_step, best_val, hist, secs = (ck["step"], ck["best"], ck["best_step"], ck["best_val"],
                                                           ck["hist"], ck["secs"])
            torch.set_rng_state(ck["rng_cpu"])
            if dev == "cuda" and ck.get("rng_cuda") is not None: torch.cuda.set_rng_state_all(ck["rng_cuda"])
            diff = {k: (ck["cfg"].get(k), v) for k, v in cfg.items() if ck["cfg"].get(k) != v}
            log(f"[{name}] RESUMED from step {step} (best MRR {best:.4f} at {best_step})"
                + (f"; WARNING settings changed since the checkpoint: {diff}" if diff else ""))
        else:
            log(f"[{name}] new model, preset {a.preset}: {npar / 1e6:.1f} M parameters, d={mcfg['d']} "
                f"layers={mcfg['layers']} nbits={D.nbits} bs={a.bs} K={a.K} seed={a.seed}")
        state.update(status="running", name=name, params=npar); json.dump(state, open(spath, "w"), indent=1)

        def save_ckpt():
            atomic_save({"model": net.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                         "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "best": best,
                         "best_step": best_step, "best_val": best_val, "hist": hist, "secs": secs, "cfg": cfg,
                         "mcfg": mcfg, "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None}, ckp)

        def validate():
            res = {}
            net.eval()
            for tag, m in (("ema", ema), ("raw", net)):
                if m is None: continue
                res[tag] = evaluate(D, V, val_logits(m, V, dev, amp), dev)
            net.train()
            return res

        pf = Prefetch(make, step, a.steps, workers=a.workers)
        net.train(); t_ck = t_log = time.time(); acc = np.zeros(4); nr = 0; n_log = 0
        try:
            while step < a.steps:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                inp, cand, fpk, el = pf.get()
                opt.zero_grad(set_to_none=True)
                lc, lb, le, top1 = losses(net, inp, fpk, el)
                loss = lc + a.w_bce * lb + a.w_elem * le
                if not torch.isfinite(loss):
                    log(f"[{name}] non-finite loss at step {step}; batch skipped")
                    opt.zero_grad(set_to_none=True)
                else:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    if ema is not None:
                        dcy = min(a.ema, (1 + step) / (10 + step))
                        with torch.no_grad():
                            for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - dcy)
                    acc += [lc.item(), lb.item(), le.item(), top1.item()]; nr += 1
                step += 1; steps_this_run += 1; n_log += 1
                if step % a.log_every == 0:
                    now = time.time(); dt = now - t_log; t_log = now; secs += dt
                    sps = n_log / max(dt, 1e-9); n_log = 0; m = acc / max(nr, 1)
                    mem = f" vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G" if dev == "cuda" else ""
                    log(f"[{name}] step {step} ctr {m[0]:.4f} bce {m[1]:.4f} elem {m[2]:.4f} batch-top1 {m[3]:.3f} "
                        f"scale {net.log_scale.exp().item():.2f} lr {lr_at(step):.2e} {sps:.2f} step/s{mem}")
                    metric(model=name, kind="train", step=step, ctr=m[0], bce=m[1], elem=m[2], top1=m[3],
                           lr=lr_at(step), steps_per_s=sps)
                    acc[:] = 0; nr = 0
                if step % a.val_every == 0 or step == a.steps:
                    res = validate()
                    tag = max(res, key=lambda t: res[t]["pool"]["mrr"]); r = res[tag]
                    v = dict(step=step, weights=tag, mrr=r["pool"]["mrr"], top1=r["pool"]["top1"],
                             isomer_mrr=r["pool_isomers"].get("mrr"), pcv_mrr=r["pcv"].get("mrr"),
                             pcv_top1=r["pcv"].get("top1"), pcv_mrr_fullwindow_est=r["pcv"].get("mrr_fullwindow_est"),
                             detail=res)
                    hist.append({k: x for k, x in v.items() if k != "detail"}); better = v["mrr"] > best
                    f4 = lambda x: "-" if x is None else f"{x:.4f}"
                    log(f"[{name}]   VAL step {step} ({tag} weights) pool MRR {v['mrr']:.4f} top1 {v['top1']:.4f} | "
                        f"isomers-only MRR {f4(v['isomer_mrr'])} | PubChem sample MRR {f4(v['pcv_mrr'])} top1 "
                        f"{f4(v['pcv_top1'])}" + (f" | other weights pool MRR " + ", ".join(
                            f"{t} {res[t]['pool']['mrr']:.4f}" for t in res if t != tag) if len(res) > 1 else "")
                        + ("  <- best" if better else ""))
                    metric(model=name, kind="val", **v)
                    if better:
                        best, best_step, best_val = v["mrr"], step, v
                        src = ema if tag == "ema" else net
                        atomic_save({"model": {k: t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu()
                                               for k, t in src.state_dict().items()},
                                     "cfg": mcfg, "step": step, "weights": tag, "adducts": D.adducts,
                                     "instruments": D.instruments, "fp_bits": D.fp_bits, "val": v, "train_cfg": cfg}, bestp)
                    t_log = time.time(); n_log = 0
                    if step - best_step >= a.patience:
                        why = "patience"; break
                if time.time() - t_ck > a.ckpt_minutes * 60:
                    save_ckpt(); t_ck = time.time(); log(f"[{name}] checkpoint saved at step {step}")
                if (a.max_minutes and (time.time() - T_RUN) / 60 > a.max_minutes) or \
                        (a.stop_after and steps_this_run >= a.stop_after):
                    if step < a.steps: interrupted = True; break
        except KeyboardInterrupt:
            interrupted = True; log("Ctrl+C received - saving a checkpoint before exiting")
        except RuntimeError as e:
            if is_oom(e):
                log("ERROR: the GPU ran out of memory. Run again with a smaller batch, e.g.  --bs 128  "
                    "(the run resumes from the last checkpoint).")
                pf.close(); sys.exit(3)
            raise
        finally:
            pf.close()
        save_ckpt()
        state.update(steps_done=step, best_step=best_step, best_mrr=best, best_val=best_val, train_seconds=round(secs),
                     steps_per_s=round(step / secs, 3) if secs > 1 else None, val_history=hist,
                     vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None)
        if interrupted:
            log(f"[{name}] stopped at step {step}; checkpoint saved. Run the same command again to continue.")
        else:
            state.update(status="done", stop_reason=why)
            log(f"[{name}] finished at step {step} ({why}); best pool MRR {best:.4f} at step {best_step}")
        json.dump(state, open(spath, "w"), indent=1)

    # ---- export (also after an interruption, from the best checkpoint so far, marked as not final)
    done = state.get("status") == "done"
    exp = out / "export"; exp.mkdir(exist_ok=True); files = {}
    if bestp.exists():
        dst = exp / EXPORT_NAME.format(name=name); shutil.copy2(bestp, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
        files[name] = dict(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst), step=state.get("best_step"))
    res = dict(finished=done, name=name, config=vars(a), model_config=mcfg, files=files, state=state,
               data=dict(cft=D.cmeta.get("cpool"), bits=D.cmeta.get("bits"), pcv=D.cmeta.get("pcv"),
                         train_spectra_used=int(len(D.tr)), val_molecules=int(V["in_pool_set"].sum())),
               env=dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, amp=amp,
                        torch=torch.__version__, python=platform.python_version(), os=platform.platform(),
                        numba=M.HAVE_NUMBA, numpy=np.__version__),
               script_sha256={f: sha256(HERE / f) for f in ("train_cft.py", "cft_model.py")},
               wall_minutes_this_run=round((time.time() - T_RUN) / 60, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(res, open(exp / f"train_result_{name}.json", "w"), indent=1, default=str)
    for f in ("train_log.txt", "metrics.jsonl"):
        logf.flush(); metf.flush(); shutil.copy2(out / f, exp / f)
    keep_awake(False)
    if done:
        log(f"ALL DONE. Results are in {exp}  ->  " + ", ".join(v["file"] for v in files.values()) + f", train_result_{name}.json")
    else:
        log(f"NOT FINISHED. Run the same command again to continue. (Partial results so far are in {exp}.)")
    sys.exit(0 if done else 2)


if __name__ == "__main__":
    main()

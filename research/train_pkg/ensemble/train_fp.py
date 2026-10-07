"""Train the two spectrum -> fingerprint networks (per-spectrum "single" and "merged") and export them
in the format the engine loads.

    python train_fp.py --data <folder with the .npy files> --out <output folder>

Run the SAME command again after any interruption: it resumes from the last checkpoint.

Method = our re-implementation of the public recipe (prvsiyan, "Analog Propagation - CASMI 2026 baseline",
appendix training script): BCE on the 6,930 fingerprint bits + softmax cross-entropy of f.z against 63 decoys
from the same +-10 ppm mass window; AdamW 3e-4, wd 0.01, betas (0.9, 0.98), 2,000 warm-up steps then cosine to
2 % over 80,000 steps; batch 256; gradient clip 1.0; peak dropout / intensity jitter / 5 ppm m/z jitter;
merged model: 60 % of inputs are 2-4 spectra of the same molecule merged; keep the best-validation checkpoint.
Differences from the public script are listed in research/analysis/train_pkg_report.md.
"""
import argparse, contextlib, hashlib, json, math, os, platform, shutil, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_model as M

try:
    import numba
    _jit = numba.njit(cache=False, nogil=True)
    HAVE_NUMBA = True
except Exception:                                            # works without numba, only slower
    _jit = lambda f: f
    HAVE_NUMBA = False

MODELS = {  # name: (merge_p, seed, export file name). 'merged' must be in the merged file name (engine rule).
    "single": dict(merge_p=0.0, seed=2, file="fp_single_ho1.pt"),
    "merged": dict(merge_p=0.6, seed=1, file="fp_merged_ho1.pt"),
}


# ------------------------------------------------------------------------------------ batch building
@_jit
def _fill(ids, off, mz, it, lens, MZ, IT, PAD):
    for r in range(len(ids)):
        a = off[ids[r]]
        for j in range(lens[r]):
            MZ[r, j] = mz[a + j]; IT[r, j] = it[a + j]; PAD[r, j] = False


@_jit
def _merge_row(pk, off, mz, it, maxlen, MZ, IT, PAD, r):
    """Public Data._merged(): concatenate the peers' peaks, drop the weaker of two peaks closer than
    0.005 m/z, keep the `maxlen` most intense, m/z ascending. Written into row r."""
    tot = 0
    for j in pk:
        tot += min(off[j + 1] - off[j], maxlen)
    m = np.empty(tot, np.float32); v = np.empty(tot, np.float32)
    p = 0
    for j in pk:
        a = off[j]; b = min(off[j] + maxlen, off[j + 1])
        for q in range(a, b):
            m[p] = mz[q]; v[p] = it[q]; p += 1
    o = np.argsort(m, kind="mergesort")
    m = m[o]; v = v[o]
    keep = np.ones(tot, np.bool_)
    thr = np.float32(0.005)
    for j in range(1, tot):
        if np.float32(m[j] - m[j - 1]) < thr:
            if v[j] >= v[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    m = m[keep]; v = v[keep]
    if len(m) > maxlen:
        top = np.sort(np.argsort(-v, kind="mergesort")[:maxlen])
        m = m[top]; v = v[top]
    n2 = len(m)
    for j in range(MZ.shape[1]):
        MZ[r, j] = 0.0; IT[r, j] = 0.0; PAD[r, j] = True
    for j in range(n2):
        MZ[r, j] = m[j]; IT[r, j] = v[j]; PAD[r, j] = False


class Data:
    def __init__(self, d, mmap=False, log=print):
        d = Path(d)
        ld = lambda n: (np.load(d / f"{n}.npy", mmap_mode="r").view(np.ndarray) if mmap else np.load(d / f"{n}.npy"))
        self.meta = json.load(open(d / "meta.json"))
        self.off = ld("spec_off"); self.mz = ld("spec_mz"); self.it = ld("spec_it")
        self.prec = np.load(d / "spec_prec.npy"); self.ad = np.load(d / "spec_ad.npy").astype(np.int64)
        self.ins = np.load(d / "spec_ins.npy").astype(np.int64); self.ce = np.load(d / "spec_ce.npy")
        self.mode = np.load(d / "spec_mode.npy").astype(np.float32)
        self.six = np.load(d / "spec_mol.npy").astype(np.int64); split = np.load(d / "spec_split.npy")
        self.s2p = np.load(d / "mol_pool.npy").astype(np.int64)
        self.pmass = np.load(d / "pool_mass.npy"); self.pfp = ld("pool_fp")
        self.nbits = int(self.meta["format"]["nbits"])
        assert self.pfp.shape[1] * 8 >= self.nbits and len(self.pmass) == len(self.pfp)
        assert (np.diff(self.pmass) >= 0).all(), "pool must be sorted by mass"
        order = np.argsort(self.six, kind="mergesort")
        self._grp_order = order
        self._grp_bounds = np.searchsorted(self.six[order], np.arange(self.six.max() + 2))
        self.tr = np.where(split == 0)[0]; self.va = np.where(split == 1)[0]
        assert (self.s2p[self.six] >= 0).all()
        log(f"data: nbits={self.nbits} pool={len(self.pmass):,} train={len(self.tr):,} val={len(self.va):,} "
            f"spectra, {len(np.unique(self.six[self.tr])):,} / {len(np.unique(self.six[self.va])):,} molecules")

    def subset(self, n_mol, seed=0):
        """Smoke test: keep a random subset of molecules."""
        r = np.random.default_rng(seed)
        for name in ("tr", "va"):
            ids = getattr(self, name); mols = np.unique(self.six[ids])
            pick = r.choice(mols, size=min(len(mols), n_mol if name == "tr" else max(n_mol // 10, 20)), replace=False)
            setattr(self, name, ids[np.isin(self.six[ids], pick)])

    def sample_negs(self, pos, K, rng, ppm=10.0):
        m = self.pmass[pos]; tol = m * ppm / 1e6
        lo = np.searchsorted(self.pmass, m - tol, "left"); hi = np.searchsorted(self.pmass, m + tol, "right")
        n = hi - lo
        c = lo[:, None] + np.minimum((rng.random((len(pos), K)) * n[:, None]).astype(np.int64),
                                     np.maximum(n[:, None] - 1, 0))
        bad = c == pos[:, None]
        if bad.any():
            nxt = c + 1
            c = np.where(bad, np.where(nxt < hi[:, None], nxt, lo[:, None]), c)
        rnd = rng.integers(0, len(self.pmass), (len(pos), K))          # no isomers -> random decoys
        return np.where((n <= 1)[:, None], rnd, c)

    def peers(self, i, rng, kmax=4):
        s = self.six[i]
        a, b = self._grp_bounds[s], self._grp_bounds[s + 1]
        if b - a <= 1: return None
        k = int(rng.integers(2, min(kmax, b - a) + 1))
        pick = self._grp_order[a + rng.choice(b - a, size=k, replace=False)]
        if i not in pick: pick = np.concatenate([[i], pick[:-1]])
        return pick.astype(np.int64)

    def batch(self, ids, K, rng, aug=False, merge_p=0.0):
        """Returns numpy arrays: (mz, it, pad, prec, ad, ins, ce, mode), cand (B, K+1; column 0 = truth)."""
        ids = np.asarray(ids, np.int64); B = len(ids); L = M.MAX_PEAKS
        lens = np.minimum(self.off[ids + 1] - self.off[ids], L).astype(np.int64); N = max(int(lens.max()), 1)
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), np.bool_)
        _fill(ids, self.off, self.mz, self.it, lens, mz, it, pad)
        if merge_p > 0:
            for r, i in enumerate(ids):
                if rng.random() >= merge_p: continue
                pk = self.peers(i, rng)
                if pk is None: continue
                _merge_row(pk, self.off, self.mz, self.it, N, mz, it, pad, r)
        if aug:
            keep = rng.random((B, N)) > rng.uniform(0.0, 0.30, size=(B, 1))                 # peak dropout
            pad = pad | (~keep)
            it = it * np.exp(rng.normal(0.0, 0.25, size=(B, N))).astype(np.float32)        # intensity jitter
            mz = mz * (1.0 + rng.normal(0.0, 5e-6, size=(B, N))).astype(np.float32)        # ~5 ppm m/z jitter
            allpad = pad.all(1)
            if allpad.any(): pad[allpad, 0] = False
        ce = np.where(self.ce[ids] < 0, 25.0, self.ce[ids]).astype(np.float32)
        pos = self.s2p[self.six[ids]]
        cand = np.concatenate([pos[:, None], self.sample_negs(pos, K, rng)], 1)
        return (mz, it, pad, self.prec[ids], self.ad[ids], self.ins[ids], ce, self.mode[ids]), cand


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


# ------------------------------------------------------------------------------------ helpers
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=256, help="batch size (256 = public recipe; fits in 8 GB VRAM)")
    ap.add_argument("--models", default="single,merged")
    ap.add_argument("--steps", type=int, default=80000, help="length of the learning-rate schedule")
    ap.add_argument("--stop_step", type=int, default=40000, help="hard stop per model")
    ap.add_argument("--patience", type=int, default=10000, help="stop a model after this many steps without a better validation score")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--K", type=int, default=63); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--val_every", type=int, default=1000); ap.add_argument("--val_n", type=int, default=4096)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ckpt_minutes", type=float, default=10.0)
    ap.add_argument("--max_minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit); run again to resume")
    ap.add_argument("--resume_from", default="", help="folder with checkpoints from an earlier session (Kaggle)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--device", default="auto"); ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU/GPU test run of the whole pipeline")
    ap.add_argument("--stop_after", type=int, default=0, help="(test) pretend to be interrupted after N steps")
    ap.add_argument("--seed_offset", type=int, default=0, help="added to each model's seed (ensemble members)")
    ap.add_argument("--tag", default="ho1", help="export names fp_single_<tag>.pt / fp_merged_<tag>.pt")
    a = ap.parse_args()
    for _n, _s in MODELS.items():
        _s["seed"] += a.seed_offset; _s["file"] = f"fp_{_n}_{a.tag}.pt"
    if a.smoke:
        a.steps, a.stop_step, a.warm, a.val_every, a.val_n, a.log_every = 60, 60, 10, 20, 128, 5
        a.bs = min(a.bs, 32); a.ckpt_minutes = min(a.ckpt_minutes, 0.2); a.patience = 10 ** 9

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
        f"amp={amp} torch={torch.__version__} python={platform.python_version()} numba={HAVE_NUMBA} os={platform.system()}")
    if not HAVE_NUMBA: log("WARNING: numba is not installed; batch building will be slow (pip install numba)")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    keep_awake(True)
    if a.resume_from and not a.fresh:
        for p in Path(a.resume_from).rglob("*"):
            if p.name in {"state.json"} or (p.name.startswith(("ckpt_", "best_")) and p.suffix == ".pt"):
                if not (out / p.name).exists():
                    shutil.copy2(p, out / p.name); log("copied", p.name, "from", a.resume_from)

    D = Data(a.data, mmap=a.mmap, log=log)
    if a.smoke: D.subset(2000); log(f"SMOKE subset: train={len(D.tr):,} val={len(D.va):,} spectra")
    pool = torch.from_numpy(np.ascontiguousarray(D.pfp)).to(dev)             # bit-packed, ~0.6 GB
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def cand_fp(cand):
        """(B, C) pool indices -> (B, C, nbits) float32 0/1 (np.unpackbits order)."""
        x = pool[torch.as_tensor(cand, device=dev)]
        x = (x.unsqueeze(-1) >> shifts) & 1
        return x.reshape(x.shape[0], x.shape[1], -1)[..., :D.nbits].float()

    def to_dev(inp):
        return tuple(torch.as_tensor(x, device=dev) for x in inp)

    def losses(net, inp, cand, train):
        with (torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()):
            z = net(*to_dev(inp))
        z = z.float()
        fc = cand_fp(cand)
        lb = F.binary_cross_entropy_with_logits(z, fc[:, 0])
        raw = torch.bmm(fc, z.unsqueeze(-1)).squeeze(-1)                      # f.z in fp32
        sc = (raw - raw.mean(dim=1, keepdim=True)) / math.sqrt(D.nbits)       # per-example shift: ranking unchanged
        lc = F.cross_entropy(sc, torch.zeros(len(sc), dtype=torch.long, device=sc.device))
        return lb, lc, (sc.argmax(1) == 0).float().mean()

    val_ids = np.sort(np.random.default_rng(12345).choice(D.va, size=min(a.val_n, len(D.va)), replace=False))

    @torch.no_grad()
    def validate(net, merge_p):
        net.eval(); rng = np.random.default_rng(999); b = []; c = []; t = []
        for s in range(0, len(val_ids), 128):
            ii = val_ids[s:s + 128]
            inp, cand = D.batch(ii, a.K, rng, aug=False, merge_p=merge_p)
            lb, lc, acc = losses(net, inp, cand, False)
            b.append(lb.item() * len(ii)); c.append(lc.item() * len(ii)); t.append(acc.item() * len(ii))
        net.train(); n = len(val_ids)
        return sum(b) / n, sum(c) / n, sum(t) / n

    def lr_at(s):
        if s < a.warm: return a.lr * s / max(1, a.warm)
        p = (s - a.warm) / max(1, a.steps - a.warm)
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    cfg = {k: v for k, v in vars(a).items() if k not in ("data", "out", "resume_from", "fresh", "stop_after",
                                                         "max_minutes", "device", "workers", "mmap")}
    spath = out / "state.json"
    state = json.load(open(spath)) if spath.exists() and not a.fresh else {"models": {}}
    names = [n for n in a.models.split(",") if n]
    interrupted = False; steps_this_run = 0

    for mi, name in enumerate(names):
        spec = MODELS[name]; ms = state["models"].setdefault(name, {"status": "pending"})
        ckp = out / f"ckpt_{name}.pt"; bestp = out / f"best_{name}.pt"
        if ms["status"] == "done" and bestp.exists():
            log(f"[{name}] already finished (best step {ms.get('best_step')}); skipping"); continue
        torch.manual_seed(spec["seed"]); np.random.seed(spec["seed"])
        net = M.build_model(D.nbits, d=a.d, layers=a.layers).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        step = 0; best = -1.0; best_step = 0; best_val = {}; hist = []; secs = 0.0
        if ckp.exists() and not a.fresh:
            ck = torch.load(ckp, map_location=dev, weights_only=False)
            net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            step, best, best_step, best_val, hist, secs = (ck["step"], ck["best"], ck["best_step"], ck["best_val"],
                                                           ck["hist"], ck["secs"])
            torch.set_rng_state(ck["rng_cpu"])
            if dev == "cuda" and ck.get("rng_cuda") is not None: torch.cuda.set_rng_state_all(ck["rng_cuda"])
            diff = {k: (ck["cfg"].get(k), v) for k, v in cfg.items() if ck["cfg"].get(k) != v}
            log(f"[{name}] RESUMED from step {step} (best {best:.4f} at {best_step})"
                + (f"; WARNING settings changed since the checkpoint: {diff}" if diff else ""))
        else:
            log(f"[{name}] new model, {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, "
                f"merge_p={spec['merge_p']} seed={spec['seed']} bs={a.bs}")
        ms["status"] = "running"; json.dump(state, open(spath, "w"), indent=1)

        def save_ckpt():
            atomic_save({"model": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                         "step": step, "best": best, "best_step": best_step, "best_val": best_val, "hist": hist,
                         "secs": secs, "cfg": cfg, "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None,
                         "nbits": D.nbits, "d": a.d, "layers": a.layers}, ckp)

        def make(s, _seed=spec["seed"], _mp=spec["merge_p"]):
            rng = np.random.default_rng([_seed, 1000 + mi, s])
            ids = rng.choice(D.tr, size=min(a.bs, len(D.tr)), replace=False)
            return D.batch(ids, a.K, rng, aug=True, merge_p=_mp)

        pf = Prefetch(make, step, a.stop_step, workers=a.workers)
        net.train(); t_ck = t_log = time.time(); rb = rc = ra = 0.0; nr = 0; why = "stop_step"; n_log = 0
        try:
            while step < a.stop_step:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                inp, cand = pf.get()
                opt.zero_grad(set_to_none=True)
                lb, lc, acc = losses(net, inp, cand, True)
                loss = lb + a.lam * lc
                if not torch.isfinite(loss):
                    log(f"[{name}] non-finite loss at step {step}; batch skipped")
                    opt.zero_grad(set_to_none=True)
                else:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    rb += lb.item(); rc += lc.item(); ra += acc.item(); nr += 1
                step += 1; steps_this_run += 1; n_log += 1
                if step % a.log_every == 0:
                    now = time.time(); dt = now - t_log; t_log = now; secs += dt
                    sps = n_log / max(dt, 1e-9); n_log = 0
                    mem = f" vram {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G" if dev == "cuda" else ""
                    log(f"[{name}] step {step} bce {rb / max(nr, 1):.4f} ctr {rc / max(nr, 1):.4f} "
                        f"top1 {ra / max(nr, 1):.3f} lr {lr_at(step):.2e} {sps:.2f} step/s{mem}")
                    metric(model=name, kind="train", step=step, bce=rb / max(nr, 1), ctr=rc / max(nr, 1),
                           top1=ra / max(nr, 1), lr=lr_at(step), steps_per_s=sps)
                    rb = rc = ra = 0.0; nr = 0
                if step % a.val_every == 0 or step == a.stop_step:
                    own = validate(net, spec["merge_p"]); oth = validate(net, 0.6 if spec["merge_p"] == 0 else 0.0)
                    v = dict(step=step, bce=own[0], ctr=own[1], top1=own[2], other_view_top1=oth[2],
                             other_view_bce=oth[0])
                    hist.append(v); better = own[2] > best
                    log(f"[{name}]   VAL step {step} bce {own[0]:.4f} ctr {own[1]:.4f} hardneg-top1 {own[2]:.4f} "
                        f"(other input view {oth[2]:.4f})" + ("  <- best" if better else ""))
                    metric(model=name, kind="val", **v)
                    if better:
                        best, best_step, best_val = own[2], step, v
                        atomic_save({"model": {k: t.detach().cpu() for k, t in net.state_dict().items()},
                                     "step": step, "nbits": D.nbits, "d": a.d, "layers": a.layers}, bestp)
                    t_log = time.time(); n_log = 0
                    if step - best_step >= a.patience:
                        why = "patience"; break
                if time.time() - t_ck > a.ckpt_minutes * 60:
                    save_ckpt(); t_ck = time.time(); log(f"[{name}] checkpoint saved at step {step}")
                if (a.max_minutes and (time.time() - T_RUN) / 60 > a.max_minutes) or \
                        (a.stop_after and steps_this_run >= a.stop_after):
                    interrupted = True; break
        except KeyboardInterrupt:
            interrupted = True; log("Ctrl+C received - saving a checkpoint before exiting")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log("ERROR: the GPU ran out of memory. Run again with a smaller batch, e.g.  --bs 128  "
                    "(the run resumes from the last checkpoint).")
                pf.close(); sys.exit(3)
            raise
        finally:
            pf.close()
        save_ckpt()
        ms.update(steps_done=step, best_step=best_step, best_top1=best, best_val=best_val, train_seconds=round(secs),
                  steps_per_s=round(step / secs, 3) if secs > 1 else None, val_history=hist)
        if interrupted:
            log(f"[{name}] stopped at step {step}; checkpoint saved. Run the same command again to continue.")
            json.dump(state, open(spath, "w"), indent=1); break
        ms.update(status="done", stop_reason=why)
        json.dump(state, open(spath, "w"), indent=1)
        log(f"[{name}] finished at step {step} ({why}); best hardneg-top1 {best:.4f} at step {best_step}")
        del net, opt
        if dev == "cuda": torch.cuda.empty_cache()

    # ---- export (also after an interruption, from the best checkpoints so far, marked as not final)
    done = all(state["models"].get(n, {}).get("status") == "done" for n in names)
    exp = out / "export"; exp.mkdir(exist_ok=True); files = {}
    for name in names:
        bp = out / f"best_{name}.pt"
        if not bp.exists(): continue
        ck = torch.load(bp, map_location="cpu", weights_only=False)
        obj = {"model": {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()},
               "step": int(ck["step"]), "nbits": int(ck["nbits"]), "d": int(ck["d"]), "layers": int(ck["layers"])}
        dst = exp / MODELS[name]["file"]; atomic_save(obj, dst)
        files[name] = dict(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst), step=obj["step"])
    res = dict(finished=done, config=vars(a), model_defs={n: MODELS[n] for n in names}, files=files,
               models={n: {k: v for k, v in state["models"].get(n, {}).items()} for n in names},
               data=dict(excluded=D.meta.get("excluded"), spectra={k: v for k, v in D.meta.get("spectra", {}).items()
                                                                  if not isinstance(v, dict)}, pool=D.meta.get("pool"),
                         train_spectra_used=int(len(D.tr)), val_spectra_scored=int(len(val_ids))),
               env=dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, amp=amp,
                        torch=torch.__version__, python=platform.python_version(), os=platform.platform(),
                        numba=HAVE_NUMBA, numpy=np.__version__),
               script_sha256={f: sha256(HERE / f) for f in ("train_fp.py", "fp_model.py")},
               wall_minutes_this_run=round((time.time() - T_RUN) / 60, 1),
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(res, open(exp / "train_result.json", "w"), indent=1, default=str)
    for f in ("train_log.txt", "metrics.jsonl"):
        logf.flush(); metf.flush(); shutil.copy2(out / f, exp / f)
    keep_awake(False)
    if done:
        log(f"ALL DONE. Results are in {exp}  ->  " + ", ".join(v["file"] for v in files.values()) + ", train_result.json")
    else:
        log(f"NOT FINISHED. Run the same command again to continue. (Partial results so far are in {exp}.)")
    sys.exit(0 if done else 2)


if __name__ == "__main__":
    main()

"""CFT: contrastive spectrum -> fingerprint transformer. Model, peak preparation, views and the inference API.

Written from scratch for this project (MIT-releasable); it does not import the warm-up package.

Inference:
    sc = CFTScorer("cft_default.pt", device="cpu")
    z = sc.logits(spectra)                 # spectra of ONE molecule: list of dicts with keys
                                           #   mz, intensity (arrays), precursor_mz, adduct (str),
                                           #   collision_energy (float or None), instrument (default "timsTOF")
    scores = sc.score(spectra, cand_fp)    # cand_fp: packed uint8 [n, ceil(nbits/8)] or 0/1 [n, nbits]
                                           #   (from cft_fp.Fingerprinter(sc.fp_bits)); higher = better
Several spectra are combined as: every spectrum is one "single" view; spectra with the same adduct are also
merged into one "merged" view (if there are >= 2 of them); molecule logits = mean of the single-view logits,
averaged 50/50 with the mean of the merged-view logits when merged views exist.
"""
import math
import numpy as np

try:
    import numba
    _jit = numba.njit(cache=False, nogil=True)
    HAVE_NUMBA = True
except Exception:                                            # works without numba, only slower
    _jit = lambda f: f
    HAVE_NUMBA = False

MAX_PEAKS = 128
MERGE_TOL = 0.005          # Da: peaks closer than this are collapsed when spectra are merged
MAX_QUERY = 16             # at most this many spectra of a molecule are used
PRESETS = {                # d, layers, heads, head width
    "default": dict(d=512, layers=8, heads=8, head=2048),
    "small": dict(d=256, layers=4, heads=4, head=1024),
    "tiny": dict(d=64, layers=2, heads=2, head=128),         # CPU smoke test only
}


# ------------------------------------------------------------------------------------ peaks and views
def prepare_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """Raw peaks -> (m/z ascending float32, sqrt(relative intensity) float32), the format of spec_mz / spec_it.
    Rule (same rule the prepared training arrays were built with): drop peaks above precursor + 1.5 and below
    0.1 % of the base peak; if more than `max_peaks` remain take them in order of intensity, at most `per_win`
    per 50-Da window, then fill up with the strongest remaining ones."""
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    e = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    if len(mz) == 0: return e
    k = mz <= prec_mz + 1.5
    mz, it = mz[k], it[k]
    if len(mz) == 0 or it.max() <= 0: return e
    k = it >= floor * it.max()
    mz, it = mz[k], it[k]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        win_id = np.floor_divide(mz, win).astype(np.int64)
        used = {}; first = []; later = []
        for i in order:
            c = used.get(win_id[i], 0)
            if c < per_win: used[win_id[i]] = c + 1; first.append(i)
            else: later.append(i)
        sel = (first + later)[:max_peaks] if len(first) < max_peaks else first[:max_peaks]
        sel = np.asarray(sel); mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


@_jit
def merge_peaks(m, v, maxlen):
    """Pooled prepared peaks of several spectra -> merged peak list: sort by m/z, of two neighbours closer than
    MERGE_TOL keep the stronger, keep the `maxlen` strongest, m/z ascending."""
    o = np.argsort(m, kind="mergesort")
    m = m[o]; v = v[o]; n = len(m)
    keep = np.ones(n, np.bool_); last = 0
    for j in range(1, n):
        if m[j] - m[last] < 0.005:
            if v[j] > v[last]:
                keep[last] = False; last = j
            else:
                keep[j] = False
        else:
            last = j
    m = m[keep]; v = v[keep]
    if len(m) > maxlen:
        top = np.sort(np.argsort(-v, kind="mergesort")[:maxlen])
        m = m[top]; v = v[top]
    return m, v


def make_views(specs):
    """specs: prepared spectra of ONE molecule, each a dict(mz, it, prec, ad, ins, ce) with ce < 0 = unknown.
    Returns (views, kinds): views in the same dict format plus 'nm' (number of spectra merged); kinds 0 = single,
    1 = merged (one per adduct that has >= 2 spectra)."""
    specs = [s for s in specs if len(s["mz"])][:MAX_QUERY]
    views = [dict(s, nm=1) for s in specs]; kinds = [0] * len(specs)
    by = {}
    for s in specs: by.setdefault(int(s["ad"]), []).append(s)
    for ad, g in by.items():
        if len(g) < 2: continue
        m, v = merge_peaks(np.concatenate([s["mz"] for s in g]).astype(np.float32),
                           np.concatenate([s["it"] for s in g]).astype(np.float32), MAX_PEAKS)
        ces = [s["ce"] for s in g if s["ce"] >= 0]
        views.append(dict(mz=m, it=v, prec=float(np.median([s["prec"] for s in g])), ad=ad, ins=g[0]["ins"],
                          ce=float(np.mean(ces)) if ces else -1.0, nm=len(g)))
        kinds.append(1)
    return views, kinds


def collate(views, mode_of_adduct):
    """List of views -> dict of numpy arrays for the network."""
    B = len(views); N = max(1, max(len(v["mz"]) for v in views))
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), np.bool_)
    for r, v in enumerate(views):
        n = len(v["mz"]); mz[r, :n] = v["mz"]; it[r, :n] = v["it"]; pad[r, :n] = False
    ce = np.array([v["ce"] for v in views], np.float32)
    ad = np.array([v["ad"] for v in views], np.int64)
    return dict(mz=mz, it=it, pad=pad, prec=np.array([v["prec"] for v in views], np.float32), ad=ad,
                ins=np.array([v["ins"] for v in views], np.int64), ce=np.where(ce < 0, 0.0, ce).astype(np.float32),
                cek=(ce >= 0).astype(np.float32), nm=np.array([v["nm"] for v in views], np.float32),
                mode=np.asarray(mode_of_adduct, np.float32)[ad])


def combine_views(z, owner, kind, n_mol):
    """Per-view outputs z [V, D] (torch) -> per-molecule [n_mol, D]: mean of single views, averaged 50/50 with the
    mean of merged views where a molecule has merged views."""
    import torch
    owner = torch.as_tensor(owner, device=z.device, dtype=torch.long)
    kind = torch.as_tensor(kind, device=z.device, dtype=torch.long)
    out = []
    for k in (0, 1):
        w = (kind == k).to(z.dtype)
        s = torch.zeros(n_mol, z.shape[1], dtype=z.dtype, device=z.device).index_add_(0, owner, z * w[:, None])
        c = torch.zeros(n_mol, dtype=z.dtype, device=z.device).index_add_(0, owner, w)
        out.append((s / c.clamp(min=1)[:, None], c))
    (zs, _), (zm, cm) = out
    has = (cm > 0)[:, None]
    return torch.where(has, 0.5 * (zs + zm), zs)


def mode_of_adducts(adducts):
    """+1 / -1 from the charge sign at the end of the adduct string (0 for '<unk>')."""
    return [1.0 if a.endswith("+") else (-1.0 if a.endswith("-") else 0.0) for a in adducts]


def unpack_bits(packed, nbits):
    """torch uint8 [..., nbytes] (np.packbits order) -> bool [..., nbits]."""
    import torch
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    x = (packed.unsqueeze(-1) >> sh) & 1
    return x.reshape(*packed.shape[:-1], -1)[..., :nbits].bool()


# ------------------------------------------------------------------------------------ network
def build_model(cfg):
    """cfg: dict(nbits, d, layers, heads, head, drop, n_adduct, n_instr, n_elem). torch is imported lazily."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class Fourier(nn.Module):
        """sin / cos features with `dim/2` wavelengths log-spaced in [lo, hi]; the angle is formed in float64
        so that 0.01-Da wavelengths stay exact at m/z 1000."""
        def __init__(self, dim, lo, hi):
            super().__init__()
            wl = torch.logspace(math.log10(lo), math.log10(hi), dim // 2, dtype=torch.float64)
            self.register_buffer("w", 2 * math.pi / wl, persistent=False)

        def forward(self, x):
            a = x.double().unsqueeze(-1) * self.w
            return torch.cat([torch.sin(a), torch.cos(a)], -1).float()

    class Layer(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__(); self.h = h
            self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
            self.drop = nn.Dropout(drop)

        def forward(self, x, key_ok):
            B, N, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok)
            x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, N, D)))
            return x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.ln2(x))))))

    class CFT(nn.Module):
        def __init__(self, c):
            super().__init__()
            d = c["d"]; self.cfg = dict(c)
            self.f_mz = Fourier(d, 1e-2, 2e3); self.f_nl = Fourier(d, 1e-2, 2e3); self.f_prec = Fourier(d, 1e-2, 2e3)
            self.f_ce = Fourier(d // 4, 1.0, 400.0)
            self.peak = nn.Linear(2 * d + 2, d)
            self.glob = nn.Linear(d + d // 4 + 4, d)
            self.adduct = nn.Embedding(c["n_adduct"], d); self.instr = nn.Embedding(c["n_instr"], d)
            self.layers = nn.ModuleList([Layer(d, c["heads"], c["drop"]) for _ in range(c["layers"])])
            self.ln = nn.LayerNorm(d)
            self.fp_head = nn.Sequential(nn.Linear(2 * d, c["head"]), nn.GELU(), nn.Dropout(c["drop"]),
                                         nn.Linear(c["head"], c["nbits"]))
            self.el_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, c["n_elem"]))
            self.log_scale = nn.Parameter(torch.zeros(()))        # sharpness of the contrastive softmax

        def forward(self, mz, it, pad, prec, ad, ins, ce, cek, nm, mode):
            """mz, it, pad: [B, N] (it = sqrt relative intensity, pad True = no peak); the rest [B].
            Returns (fingerprint logits [B, nbits], log1p element counts [B, n_elem])."""
            B = mz.shape[0]
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.peak(torch.cat([self.f_mz(mz), self.f_nl(nl), it.unsqueeze(-1), (it * it).unsqueeze(-1)], -1))
            g = self.glob(torch.cat([self.f_prec(prec), self.f_ce(ce) * cek[:, None], cek[:, None],
                                     (nm / 4.0)[:, None], mode[:, None], (torch.log1p(prec) / 10.0)[:, None]], -1))
            g = g + self.adduct(ad) + self.instr(ins)
            x = torch.cat([g.unsqueeze(1), p], 1)
            ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=pad.device), ~pad], 1)
            for l in self.layers: x = l(x, ok[:, None, None, :])
            x = self.ln(x)
            w = ok[:, 1:].unsqueeze(-1).to(x.dtype)
            h = torch.cat([x[:, 0], (x[:, 1:] * w).sum(1) / w.sum(1).clamp(min=1)], -1)
            return self.fp_head(h), self.el_head(h)

    return CFT(cfg)


def forward_views(net, arrays, device, amp=False, bs=256):
    """Run the network on collated views (dict from `collate`) in batches. Returns (z, elem) float32 on device."""
    import torch
    n = len(arrays["prec"]); zs = []; es = []
    order = ("mz", "it", "pad", "prec", "ad", "ins", "ce", "cek", "nm", "mode")
    for s in range(0, n, bs):
        inp = [torch.as_tensor(arrays[k][s:s + bs], device=device) for k in order]
        if amp:
            with torch.autocast("cuda", dtype=torch.float16): z, e = net(*inp)
        else:
            z, e = net(*inp)
        zs.append(z.float()); es.append(e.float())
    return torch.cat(zs), torch.cat(es)


# ------------------------------------------------------------------------------------ inference API
class CFTScorer:
    def __init__(self, ckpt, device="cpu", amp=None):
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False) if not isinstance(ckpt, dict) else ckpt
        self.cfg = ck["cfg"]; self.adducts = list(ck["adducts"]); self.instruments = list(ck["instruments"])
        self.fp_bits = np.asarray(ck["fp_bits"]); self.nbits = int(self.cfg["nbits"]); self.step = ck.get("step")
        self.net = build_model(self.cfg).to(device).eval(); self.net.load_state_dict(ck["model"])
        self.device = device; self.amp = (device != "cpu") if amp is None else amp
        self._ad = {a: i for i, a in enumerate(self.adducts)}; self._mode = mode_of_adducts(self.adducts)
        self._ins = {a: i for i, a in enumerate(self.instruments)}

    def prepare(self, s):
        """Raw spectrum dict -> prepared spectrum dict (see module docstring)."""
        prec = float(np.float32(s["precursor_mz"]))
        mz, it = prepare_peaks(s["mz"], s["intensity"], prec)
        ce = s.get("collision_energy")
        if ce is not None and not np.isscalar(ce):
            ce = float(np.mean(np.abs(np.asarray(ce, float)))) if len(ce) else None
        ce = -1.0 if ce is None or not np.isfinite(ce) else abs(float(ce))
        return dict(mz=mz, it=it, prec=prec, ad=self._ad.get(s.get("adduct"), self._ad["<unk>"]),
                    ins=self._ins.get(s.get("instrument", "timsTOF"), self._ins["other"]), ce=ce)

    def logits_prepared(self, mols):
        """mols: list (one entry per molecule) of lists of prepared spectra -> torch [n_mol, nbits] on device."""
        import torch
        views = []; owner = []; kind = []
        for j, specs in enumerate(mols):
            v, k = make_views(specs)
            views += v; kind += k; owner += [j] * len(v)
        if not views: return torch.zeros(len(mols), self.nbits, device=self.device)
        with torch.no_grad():
            z, _ = forward_views(self.net, collate(views, self._mode), self.device, self.amp)
            return combine_views(z, owner, kind, len(mols))

    def logits(self, spectra):
        return self.logits_prepared([[self.prepare(s) for s in spectra]])[0].cpu().numpy()

    def score(self, spectra, cand_fp):
        z = self.logits(spectra)
        f = np.asarray(cand_fp)
        if f.shape[1] != self.nbits: f = np.unpackbits(f, axis=1)[:, :self.nbits]
        return f.astype(np.float32) @ z

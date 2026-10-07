"""Spectrum -> fingerprint network (FPNet) and peak preparation.

ADAPTED (kept identical on purpose, so the weights stay load-compatible with the engine's pv_fp.py)
from the public Kaggle notebook prvsiyan/analog-propagation-casmi-2026-baseline, cell "fpmodel":
ADDUCT_LIST, INSTR_LIST, instr_family, prep_peaks, SinEmb, Block, FPNet.
Everything else in this package (data preparation, training loop, checkpointing, export) is our own
re-implementation of the method described there.

DO NOT change the architecture, the adduct / instrument lists or prep_peaks: the engine rebuilds the
network from (nbits, d, layers) alone and loads the state dict strictly.
"""
import math
import numpy as np

MAX_PEAKS = 128
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]


def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4


def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """Filter -> window-diversified top-N -> sort by m/z. Returns (mz, sqrt-intensity)."""
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            ss = set(sel.tolist())                       # same result as the original, built once
            rest = np.array([i for i in order if i not in ss])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)


def build_model(nbits, d=512, layers=6, heads=8, drop=0.1):
    """Import torch lazily so the data-preparation workers never load it."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class SinEmb(nn.Module):
        def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
            super().__init__()
            n = dim // 2
            wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
            self.register_buffer('inv', (2 * math.pi) / wav)

        def forward(self, x):
            a = x.unsqueeze(-1) * self.inv
            return torch.cat([torch.sin(a), torch.cos(a)], -1)

    class Block(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__(); self.h = h
            self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, pad):
            B, N, D = x.shape; y = self.n1(x)
            q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            m = (~pad)[:, None, None, :]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
            return x + self.drop(self.ff(self.n2(x)))

    class FPNet(nn.Module):
        def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
            super().__init__()
            self.d = d
            self.mz_emb = SinEmb(d)
            self.nl_emb = SinEmb(d)
            self.pk = nn.Linear(2 * d + 1, d)
            self.prec_emb = SinEmb(d)
            self.ad = nn.Embedding(len(ADDUCT_LIST), d)
            self.ins = nn.Embedding(len(INSTR_LIST), d)
            self.gl = nn.Linear(d + 3, d)
            self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

        def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
            B, N = mz.shape
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
            g = self.gl(torch.cat([self.prec_emb(prec),
                                   (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                                   torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
            x = torch.cat([g.unsqueeze(1), p], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
            for b in self.blocks: x = b(x, pad)
            x = self.norm(x)
            cls = x[:, 0]
            msk = (~pad[:, 1:]).float().unsqueeze(-1)
            mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
            return self.head(torch.cat([cls, mean], -1))

    return FPNet(nbits, d=d, layers=layers, heads=heads, drop=drop)

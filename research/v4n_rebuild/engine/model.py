"""Pluggable spectrum -> fingerprint model interface of the engine, and the CFT adapter.

Interface (duck-typed; any object with these members can be the engine's bank):
    nbits            number of predicted fingerprint bits
    fp_bits          int array: indices of the predicted bits in the raw cfp1 layout (fingerprint.RAW_BITS)
    prepare(mz, it, prec) -> (mz float32 ascending, sqrt relative intensity float32)    model-specific peak prep
    logits_el(items) -> (z float32 [n, nbits], el float32 [n, 10])                       log-odds + log1p element counts
items: dict(mz, it (already prepared), prec (float), adduct (str), ce (float eV, NaN = unknown),
            nm (number of spectra merged into this view, 1 for a single spectrum), instrument (str))

CFTBank wraps one or more research/train_cft checkpoints (all must share fp_bits) and averages their outputs.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Sequence

import numpy as np

_CFT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "train_cft")
INSTRUMENTS = ("timsTOF", "Orbitrap", "QTOF", "IT", "other")


def instrument_family(s) -> str:
    """Instrument string -> one of INSTRUMENTS (the rule the CFT training arrays were built with)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "other"
    t = str(s).lower()
    if "timstof" in t:
        return "timsTOF"
    if any(k in t for k in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive")):
        return "Orbitrap"
    if "tof" in t:
        return "QTOF"
    if "trap" in t or "qq" in t:
        return "IT"
    return "other"


class CFTBank:
    def __init__(self, ckpts: Sequence[str], device: str = "cpu", threads: int = 0):
        import torch
        if _CFT_DIR not in sys.path:
            sys.path.insert(0, os.path.abspath(_CFT_DIR))
        import cft_model as CM
        self.CM = CM
        if threads:
            torch.set_num_threads(threads)
        self.nets, bits = [], None
        for p in ckpts:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            b = np.asarray(ck["fp_bits"], np.int64)
            if bits is None:
                bits, self.adducts, self.instruments = b, list(ck["adducts"]), list(ck["instruments"])
                self.cfg = ck["cfg"]
            elif not np.array_equal(bits, b) or list(ck["adducts"]) != self.adducts:
                raise ValueError("all checkpoints of a bank must share fp_bits and the adduct list")
            net = CM.build_model(ck["cfg"]).to(device).eval()
            net.load_state_dict(ck["model"])
            self.nets.append(net)
        self.fp_bits = bits
        self.nbits = len(bits)
        self.device = device
        self._ad = {a: i for i, a in enumerate(self.adducts)}
        self._mode = np.asarray(CM.mode_of_adducts(self.adducts), np.float32)
        self._ins = {a: i for i, a in enumerate(self.instruments)}

    def prepare(self, mz, it, prec):
        return self.CM.prepare_peaks(mz, it, float(np.float32(prec)))

    def _views(self, items):
        V = []
        for x in items:
            ce = x.get("ce", float("nan"))
            ce = -1.0 if ce is None or not np.isfinite(ce) else abs(float(ce))
            ins = x.get("instrument", "timsTOF")
            ins = ins if ins in self._ins else instrument_family(ins)
            V.append(dict(mz=np.asarray(x["mz"], np.float32), it=np.asarray(x["it"], np.float32),
                          prec=float(np.float32(x["prec"])), ad=self._ad.get(x["adduct"], self._ad["<unk>"]),
                          ins=self._ins.get(ins, self._ins["other"]), ce=ce, nm=int(x.get("nm", 1))))
        return V

    def logits_el(self, items):
        import torch
        if not items:
            return np.zeros((0, self.nbits), np.float32), np.zeros((0, 10), np.float32)
        arr = self.CM.collate(self._views(items), self._mode)
        zs, es = [], []
        with torch.no_grad():
            for net in self.nets:
                z, e = self.CM.forward_views(net, arr, self.device, amp=False)
                zs.append(z.cpu().numpy()); es.append(e.cpu().numpy())
        return np.mean(zs, 0).astype(np.float32), np.mean(es, 0).astype(np.float32)

"""Readers for the v4r tables (results/v4n/tables).  Everything large is memory-mapped.

Library  (L1 + P2): cleaned CSR spectra of train.parquet, per-spectrum metadata, train structures (sid-indexed).
Pool     (P4/P5/P6): mass-sorted candidate structures, fingerprints, fragment masses.
Fingerprints are served in the bank's bit layout: the model-layout file (pool_fp.npy / train_fp_sel.npy written by
`build_tables.py project`) when its fp_bits.npy equals the bank's bits, otherwise projected on the fly from the raw
16,384-bit cfp1 tables.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from . import fingerprint

DEFAULT_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "v4n", "tables")


class _FP:
    """Packed fingerprints of a table, served as float32 0/1 rows in a chosen bit layout."""

    def __init__(self, d, raw_name, sel_name, bits):
        self.bits = np.asarray(bits, np.int64)
        self.nbits = len(self.bits)
        self.raw = self.sel = None
        fb = os.path.join(d, "fp_bits.npy")
        if os.path.exists(fb) and os.path.exists(os.path.join(d, sel_name)):
            if np.array_equal(np.load(fb).astype(np.int64), self.bits):
                self.sel = np.load(os.path.join(d, sel_name), mmap_mode="r")
        if self.sel is None:  # raw layout only needed when the model's bits differ from the projected table
            self.raw = np.load(os.path.join(d, raw_name), mmap_mode="r")

    def get(self, rows) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        if self.sel is not None:
            return np.unpackbits(self.sel[rows], axis=1)[:, :self.nbits]
        return fingerprint.select(self.raw[rows], self.bits)


class Library:
    def __init__(self, d: str = DEFAULT_TABLES, verbose=True):
        t0 = time.time()
        self.dir = d
        self.off = np.load(os.path.join(d, "spec_off.npy"))
        self.mz = np.load(os.path.join(d, "spec_mz.npy"), mmap_mode="r")
        self.it = np.load(os.path.join(d, "spec_it.npy"), mmap_mode="r")
        m = pd.read_parquet(os.path.join(d, "spec_meta.parquet"),
                            columns=["sid", "lib", "mode", "adduct_ix", "adduct", "precursor_mz", "n_clean", "instr",
                                     "instrument_type", "ce_mean", "ce_n"])
        self.sid = m.sid.values.astype(np.int64)
        self.lib = m.lib.values.astype(np.int8)
        self.mode = m["mode"].values.astype(np.int8)
        from . import chem    # index recomputed from the string so the table never goes stale vs chem.ADDUCT_LIST
        self.adduct_ix = m.adduct.map(chem.ADDUCT_IX).fillna(chem.UNK_ADDUCT).values.astype(np.int16)
        self.adduct = m.adduct.values
        self.prec = m.precursor_mz.values.astype(np.float64)
        self.n_clean = m.n_clean.values.astype(np.int32)
        self.instr = m.instr.values.astype(np.int8)
        self.instrument_type = m.instrument_type.values
        self.ce_mean = m.ce_mean.values.astype(np.float64)
        self.ce_n = m.ce_n.values.astype(np.int32)
        st = pd.read_parquet(os.path.join(d, "train_structs.parquet"),
                             columns=["sid", "inchikey14", "mass", "key", "smiles", "smiles_tc", "pid"])
        assert (st.sid.values == np.arange(len(st))).all()
        self.struct_mass = st.mass.values.astype(np.float64)
        self.struct_key = st.key.values.astype(object)
        self.struct_smiles = st.smiles.values.astype(object)
        self.struct_ik = st.inchikey14.values.astype(object)
        self.struct_pid = st.pid.values.astype(np.int64)
        assert (self.sid >= 0).all(), "spectra without a train structure"
        self.smass = self.struct_mass[self.sid]
        self.order = np.argsort(self.smass, kind="mergesort")
        self.smass_sorted = self.smass[self.order]
        self.train_fp: Optional[_FP] = None
        if verbose:
            print(f"[Library] {len(self.sid):,} spectra, {len(st):,} structures ({time.time() - t0:.1f}s)", flush=True)

    def set_bits(self, bits):
        self.train_fp = _FP(self.dir, "train_fp_raw.npy", "train_fp_sel.npy", bits)

    def window(self, target, ppm, exclude=None, mode=None) -> np.ndarray:
        tol = target * ppm * 1e-6
        lo = np.searchsorted(self.smass_sorted, target - tol, "left")
        hi = np.searchsorted(self.smass_sorted, target + tol, "right")
        c = self.order[lo:hi]
        if mode is not None:
            c = c[self.mode[c] == mode]
        if exclude is not None and len(c):
            c = c[~exclude[c]]
        return c

    def peaks(self, i):
        a, b = self.off[i], self.off[i + 1]
        return np.asarray(self.mz[a:b], np.float64), np.asarray(self.it[a:b], np.float64)

    def query_spectrum(self, i) -> dict:
        """Library spectrum i as an engine query spectrum (simulation)."""
        mz, it = self.peaks(i)
        ce = self.ce_mean[i]
        return dict(mz=mz, it=it, mode=int(self.mode[i]), adduct=self.adduct[i], prec=float(self.prec[i]),
                    ce=float(ce) if np.isfinite(ce) else np.nan, ce_n=int(self.ce_n[i]) if self.ce_n[i] > 0 else 1,
                    instrument=self.instrument_type[i])


class Pool:
    def __init__(self, d: str = DEFAULT_TABLES, verbose=True):
        t0 = time.time()
        self.dir = d
        m = pd.read_parquet(os.path.join(d, "pool_meta.parquet"),
                            columns=["mass", "key", "smiles", "smiles_tc", "formula", "n_heavy", "src", "train_sid"])
        self.mass = m.mass.values.astype(np.float64)
        assert (np.diff(self.mass) >= 0).all()
        self.key = m.key.values.astype(object)
        self.smiles = m.smiles.values.astype(object)
        self.smiles_tc = m.smiles_tc.values.astype(object)
        self.formula = m.formula.values.astype(object)
        self.n_heavy = m.n_heavy.values.astype(np.int32)
        self.src = m.src.values.astype(np.int8)
        self.train_sid = m.train_sid.values.astype(np.int64)
        self.frag_off = np.load(os.path.join(d, "pool_frag_off.npy"))
        self.frag_mass = np.load(os.path.join(d, "pool_frag_mass.npy"), mmap_mode="r")
        self.fp: Optional[_FP] = None
        self.key2pid = None
        if verbose:
            print(f"[Pool] {len(self.mass):,} structures ({time.time() - t0:.1f}s)", flush=True)

    def set_bits(self, bits):
        self.fp = _FP(self.dir, "pool_fp_raw.npy", "pool_fp.npy", bits)

    @property
    def nbits(self):
        return self.fp.nbits

    def window(self, target, ppm) -> np.ndarray:
        tol = target * ppm * 1e-6
        return np.arange(np.searchsorted(self.mass, target - tol, "left"),
                         np.searchsorted(self.mass, target + tol, "right"))

    def fps(self, pids) -> np.ndarray:
        return self.fp.get(pids)

    def frags(self, pid) -> np.ndarray:
        return np.asarray(self.frag_mass[self.frag_off[pid]:self.frag_off[pid + 1]], np.float64)

    def pid_of_key(self, k) -> int:
        if self.key2pid is None:
            self.key2pid = {kk: i for i, kk in enumerate(self.key)}
        return self.key2pid.get(k, -1)

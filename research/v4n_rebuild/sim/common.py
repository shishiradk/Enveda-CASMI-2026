"""Shared paths, constants and small helpers of the v4r simulation (phase B).  Our code, MIT."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))          # makes `engine` importable

TABLES = Path(os.environ.get("V4R_TABLES", ROOT / "results" / "v4n" / "tables"))
SPLIT = ROOT / "results" / "v4n" / "split_v4r.parquet"
OUT_ROOT = Path(os.environ.get("V4R_SIM_OUT", ROOT / "results" / "v4n" / "sim"))   # Kaggle: /kaggle/working
CKPT_HO2 = ROOT / "models" / "cft_ho2_akriti" / "cft_ho2.pt"
TEST = ROOT / "test.parquet"
HO2_FILES = [ROOT / "results" / "kaggle_v1_proxy" / "held_keys.parquet",
             ROOT / "results" / "kaggle_v2_proxy" / "s3_held.parquet",
             ROOT / "results" / "c3" / "held_S4.parquet"]

REGIMES = ("c1", "c2", "c3")
MIX = {"c1": 0.20, "c2": 0.55, "c3": 0.25}         # REBUILD_SPEC 4, starting class mix (HYPOTHESIS)
LIBS = ["enveda-180", "pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial", "drug_plus",
        "enveda-np-examples", "masaryk"]          # = engine/build_tables.py LIBS (spec_meta.lib codes)
QSTATS = ["n_query", "q_npeaks", "lib_max", "top_sim", "n_cand"]   # V5b query-level statistics


def peak_rss_mb() -> float:
    """Peak working set of this process in MB (Windows; 0 elsewhere)."""
    if os.name != "nt":
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    try:
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
        f = ctypes.windll.psapi.GetProcessMemoryInfo
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        f(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 20
    except Exception:
        return 0.0


def private_mb() -> float:
    """Private (commit) bytes of this process in MB on Windows — what really competes for RAM (mmaps excluded)."""
    if os.name != "nt":
        return 0.0

    class PMCX(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    c = PMCX(); c.cb = ctypes.sizeof(PMCX)
    try:
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
        f = ctypes.windll.psapi.GetProcessMemoryInfo
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        f(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PrivateUsage / 2 ** 20
    except Exception:
        return 0.0


def mrr_and_top1(score, y, g, k=25):
    """Per-group reciprocal rank within the top k (0 if absent) and top-1 hit. Groups must be contiguous in `g`.
    Ties are broken by row order (stable), like a deterministic submission writer."""
    import numpy as np
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    rr = np.zeros(len(starts)); t1 = np.zeros(len(starts))
    for i, (s, e) in enumerate(zip(starts, ends)):
        o = np.argsort(-score[s:e], kind="stable")
        pos = np.flatnonzero(y[s:e][o] > 0)
        if len(pos) and pos[0] < k:
            rr[i] = 1.0 / (pos[0] + 1)
            t1[i] = float(pos[0] == 0)
    return rr, t1

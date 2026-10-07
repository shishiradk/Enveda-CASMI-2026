"""Import bootstrap for running upstream ms-pred (MIT, coleygroup) inference
without DGL / torch_scatter / pytorch_lightning / pygmtools binaries.

``setup(ms_pred_src, shims=True, extra_paths=())`` puts our pure-torch shims in
front of ``sys.path``, adds the unmodified upstream ``src`` directory, and
registers empty placeholder modules for purely cosmetic imports (plotting,
multiprocessing helpers) when those packages are not installed.

With ``shims=False`` the real DGL / torch_scatter / lightning of the current
environment are used (reference runs for equivalence checks).
"""
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MS_PRED_SRC = HERE.parents[1] / "external" / "forward_model" / "ms-pred" / "src"
DEFAULT_PYDEPS = HERE.parents[1] / "external" / "forward_model" / "pydeps"

# modules that upstream imports at module top level but inference never calls
_SOFT = {
    "seaborn": {},
    "cairosvg": {},
    "h5py": {},
    "multiprocess": {},
    "multiprocess.context": {"_force_start_method": lambda *a, **k: None},
    "multiprocess.queues": {},
    "pathos": {},
    "pathos.multiprocessing": {},
}


def _soft_stub(name, attrs):
    if name in sys.modules:
        return
    root = name.split(".")[0]
    if root not in sys.modules or not getattr(sys.modules[root], "__fm_stub__", False):
        try:
            if importlib.util.find_spec(root) is not None:
                return  # the real package exists, use it
        except (ImportError, ValueError):
            pass
    mod = types.ModuleType(name)
    mod.__fm_stub__ = True
    mod.__path__ = []  # behave like a package so submodule stubs can attach
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, mod)


def setup(ms_pred_src=None, shims=True, extra_paths=()):
    ms_pred_src = Path(ms_pred_src or os.environ.get("MS_PRED_SRC") or DEFAULT_MS_PRED_SRC)
    if not (ms_pred_src / "ms_pred").is_dir():
        raise FileNotFoundError(f"ms_pred package not found under {ms_pred_src}")
    paths = [str(ms_pred_src)]
    if shims:
        paths.insert(0, str(HERE / "shims"))
    for p in paths:
        if p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = paths
    extra = [str(p) for p in extra_paths]
    if DEFAULT_PYDEPS.is_dir():
        extra.append(str(DEFAULT_PYDEPS))
    for p in extra:  # appended: never shadow packages of the running environment
        if p not in sys.path:
            sys.path.append(p)
    for name, attrs in _SOFT.items():
        _soft_stub(name, attrs)
    # matplotlib must not try to open a display in a batch subprocess
    os.environ.setdefault("MPLBACKEND", "Agg")
    return ms_pred_src

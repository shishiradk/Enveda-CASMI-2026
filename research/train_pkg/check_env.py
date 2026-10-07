"""Prints whether this PC is ready for training. Called by setup_windows.bat."""
import platform, shutil, sys
from pathlib import Path

ok = True
print("Python  :", platform.python_version(), "(need 3.10 - 3.14; 3.12 recommended)")
if not (3, 10) <= sys.version_info[:2] <= (3, 14): ok = False; print("  PROBLEM: unsupported Python version")
try:
    import numpy, torch
    print("PyTorch :", torch.__version__, "| numpy", numpy.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU     :", p.name, f"| {p.total_memory / 2 ** 30:.1f} GB VRAM | CUDA build", torch.version.cuda)
        x = torch.randn(2048, 2048, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            y = (x @ x).float().mean().item()            # fails here if the build does not support this GPU
        print("GPU test: OK (mixed-precision matrix product ran)")
    else:
        ok = False
        print("  PROBLEM: PyTorch cannot see an NVIDIA GPU. See 'CUDA not available' in RUNSHEET.md.")
except Exception as e:
    ok = False; print("  PROBLEM:", type(e).__name__, e)
try:
    import numba; print("numba   :", numba.__version__)
except Exception:
    print("numba   : not installed (training still works, but the data loading is slower)")
here = Path(__file__).resolve().parent
need = ["spec_mz.npy", "spec_it.npy", "spec_off.npy", "pool_fp.npy", "pool_mass.npy", "mol_pool.npy", "meta.json"]
miss = [f for f in need if not (here / f).exists()]
print("Data    :", "all files found" if not miss else f"MISSING {miss} (they must be in the same folder as this file)")
ok &= not miss
free = shutil.disk_usage(here).free / 2 ** 30
print(f"Disk    : {free:.1f} GB free (need about 3 GB)")
ok &= free > 3
print()
print("SETUP OK - you can now run run_train.bat" if ok else "SETUP PROBLEM - fix the PROBLEM lines above first")
sys.exit(0 if ok else 1)

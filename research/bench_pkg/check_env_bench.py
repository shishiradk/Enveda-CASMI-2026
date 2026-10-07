"""Checks the bench environment: packages, versions that the caches depend on, GPU, input files."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ok = True


def bad(m):
    global ok
    ok = False
    print("PROBLEM:", m)


print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        (torch.ones(8, device="cuda") * 2).sum().item()
    else:
        print("note: no GPU - the bench still works on the CPU (the network part is only a few minutes)")
except Exception as e:
    bad(f"torch: {e}")
need = {"rdkit": "2026.03.6", "sklearn": "1.9.0", "pandas": "3.0.5"}  # cache keys / pickles depend on these
for mod in ("numpy", "pandas", "pyarrow", "duckdb", "numba", "rdkit", "sklearn"):
    try:
        v = __import__(mod).__version__
        print(f"{mod} {v}")
        if mod in need and v != need[mod]:
            bad(f"{mod} is {v}, the caches need {need[mod]} (run setup_bench.bat again)")
    except Exception as e:
        bad(f"{mod}: {e}")
for f in ("research/bench/bench.py", "external/bench_inputs/fp", "results/bench/truth.parquet",
          "results/bench/cache/pool_train_3033286496_2026.03.6.pkl"):
    if not (HERE / f).exists():
        bad(f"missing {f} - the package was not unpacked completely")
t = HERE / "train.parquet"
if not t.exists():
    bad("train.parquet is missing (see RUNSHEET_BENCH.md step 3)")
elif t.stat().st_size != 3033286496:
    bad(f"train.parquet has {t.stat().st_size} bytes, expected 3033286496 (wrong or incomplete file)")
print("SETUP OK" if ok else "SETUP NOT OK")
sys.exit(0 if ok else 1)

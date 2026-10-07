"""Assemble the evaluation-bench package (code + inputs + caches) for a teammate's Windows PC and upload it as a
private Kaggle dataset.

    python research/bench_pkg/make_bench_pkg.py --stage D:/casmi_bench_pkg            # assemble only
    python research/bench_pkg/make_bench_pkg.py --stage D:/casmi_bench_pkg --upload   # + kaggle datasets create/version

The staged folder mirrors the project root (research/bench, external/bench_inputs, results/...) so bench.py runs
unchanged (ROOT = two levels above bench.py). train.parquet is NOT included: setup_bench.bat downloads it from the
competition (the file size is part of the pool-cache key, so it must be the original file).
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = "shishiradhikari11/casmi-bench-pkg"

FILES = [  # (source relative to ROOT, also the destination)
    "results/bench/queries_S12.parquet", "results/bench/queries_S3.parquet", "results/bench/truth.parquet",
    "results/kaggle_v1_proxy/held_keys.parquet",
    "results/kaggle_v2_proxy/cache_S1.pkl", "results/kaggle_v2_proxy/cache_S2.pkl",
    "results/kaggle_v2_proxy/cache_S3.pkl", "results/kaggle_v2_proxy/s3_held.parquet",
    "results/kaggle_v1_assets/universe.parquet",
    "results/bench/e1/per_molecule_blend.csv",          # reference for the smoke parity check
    "results/bench/e1/eval_blend.json",                 # reference numbers (public + megayak nets)
]
DIRS = ["external/bench_inputs", "external/bench_inputs_alt/megayak_fp"]
CODE = ["bench*.py", "eng/*.py"]  # globs under research/bench; eng_runner.py is read by bench.run
PKG = ["RUNSHEET_BENCH.md", "setup_bench.bat", "smoke_bench.bat", "run_bench.bat", "send_bench_results.bat",
       "check_env_bench.py", "bench_check.py", "requirements_bench.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    st = Path(a.stage)
    st.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        (st / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, st / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, st / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    cache = st / "results/bench/cache"
    cache.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "results/bench/cache").iterdir():  # pool, rankers, canonical SMILES, V2 lists; not numba
        if p.is_file():
            shutil.copy2(p, cache / p.name)
    for pat in CODE:
        for src in (ROOT / "research/bench").glob(pat):
            dst = st / "research/bench" / src.relative_to(ROOT / "research/bench")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for f in PKG:
        shutil.copy2(HERE / f, st / f)
    size = sum(p.stat().st_size for p in st.rglob("*") if p.is_file())
    print(f"staged {st}: {size / 1e9:.2f} GB")
    if a.upload:
        if (st / "train.parquet").exists():
            sys.exit("refusing to upload: remove train.parquet from the staged folder (competition data, 3 GB)")
        meta = st / "dataset-metadata.json"
        json.dump({"title": "casmi-bench-pkg", "id": DATASET, "licenses": [{"name": "other"}]}, open(meta, "w"))
        r = subprocess.call(["kaggle", "datasets", "create", "-p", str(st), "-r", "zip"])
        if r:
            r = subprocess.call(["kaggle", "datasets", "version", "-p", str(st), "-r", "zip", "-m", "update"])
        sys.exit(r)


if __name__ == "__main__":
    main()

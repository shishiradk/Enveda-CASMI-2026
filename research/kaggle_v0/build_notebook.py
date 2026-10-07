"""Build the Kaggle notebook from casmi_v0_kaggle.py (single source of truth, embedded verbatim).

    python research/kaggle_v0/build_notebook.py                     # V0 -> research/kaggle_v0/
    python research/kaggle_v0/build_notebook.py --variant sum_same  # EXP-010 v1b -> research/kaggle_v0b/
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "casmi_v0_kaggle.py"
ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["max_all", "sum_same"], default="max_all")
VARIANT = ap.parse_args().variant
OUTDIR = HERE if VARIANT == "max_all" else HERE.parent / "kaggle_v0b"
OUTDIR.mkdir(exist_ok=True)
NB = OUTDIR / "casmi_v0_kaggle.ipynb"

src = SRC.read_text(encoding="utf-8")
body = src.split('\nif __name__ == "__main__":')[0].rstrip() + "\n"  # notebook runs main() explicitly
if VARIANT != "max_all":
    assert body.count('AGGREGATION = "max_all"') == 1
    body = body.replace('AGGREGATION = "max_all"', f'AGGREGATION = "{VARIANT}"')
src_sha = hashlib.sha256(src.encode()).hexdigest()

INSTALL = r'''# Offline dependency install (internet is disabled for this competition).
# Wheels come from the private dataset "casmi-v0-wheels" (cp312 manylinux). --no-deps keeps the
# image's numpy/numba/scipy/pandas/networkx untouched; only modules that are missing get installed.
import glob, importlib, os, subprocess, sys
hits = glob.glob("/kaggle/input/**/matchms-0.33.1-py3-none-any.whl", recursive=True)
assert hits, "casmi-v0-wheels dataset not attached (matchms-0.33.1 wheel not found under /kaggle/input)"
WHEELS = os.path.dirname(hits[0])
need = {"pickydict": "pickydict==0.5.0", "pubchempy": "pubchempy==1.0.5", "pyteomics": "pyteomics==5.0.1",
        "psims": "psims==1.4.0", "sparsestack": "sparsestack==0.7.1", "deprecated": "deprecated==1.3.1",
        "wrapt": "wrapt", "six": "six", "sqlalchemy": "sqlalchemy", "greenlet": "greenlet",
        "rdkit": "rdkit==2026.3.3", "matchms": "matchms==0.33.1"}
todo = []
for mod, req in need.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        todo.append(req)
print("installing:", todo)
if todo:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        f"--find-links={WHEELS}", *todo], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    assert r.returncode == 0, "offline install failed"
importlib.invalidate_caches()
import matchms, rdkit, numba, numpy, duckdb
from matchms.similarity import ModifiedCosineGreedy  # fail fast if the stack is broken
assert matchms.__version__ == "0.33.1", matchms.__version__
print({"python": sys.version.split()[0], "matchms": matchms.__version__, "rdkit": rdkit.__version__,
       "numba": numba.__version__, "numpy": numpy.__version__, "duckdb": duckdb.__version__,
       "cpus": len(os.sched_getaffinity(0))})
'''

RUN = '''report = main(mode="kaggle")
print("submission.csv written:", report["output"])
'''

TITLE = ("V0 library retrieval (Variant A + Modified Cosine)" if VARIANT == "max_all" else
         "V0b: V0 candidates/scores, same-adduct + SUM over spectra (EXP-010 v1b)")
HEADER = f"""# CASMI 2026 — {TITLE}

Aggregation: `{VARIANT}`.

Locked V0 science: raw precursor_mz ±0.01 Da over all train spectra → matchms 0.33.1
`ModifiedCosineGreedy(tolerance=0.1, mz_power=0, intensity_power=1)` → molecule-level max → top 25.

- Zero-candidate molecule_ids (no train spectrum within ±0.01 Da) get the placeholder `C`. It is not a
  prediction (scores 0, same as an omitted row) and only keeps the file valid. Every such case and every
  malformed spectrum is recorded in `v0_run_report.json`. All other rows are unchanged V0.
- Parity gate: on the visible `test.parquet` (commit run) the output must equal the validated V0 CSV
  (LF sha256 `bd885d28…`), otherwise the notebook raises.

Source: `research/kaggle_v0/casmi_v0_kaggle.py` sha256 `{src_sha}`.
"""


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell("markdown", HEADER), cell("code", INSTALL), cell("code", body), cell("code", RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
if VARIANT != "max_all":
    meta = json.loads((HERE / "kernel-metadata.json").read_text())
    meta.update(id="shishiradhikari11/casmi-v0b-sum-same-adduct", title="CASMI V0b sum same adduct")
    (OUTDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {NB} [{VARIANT}] (embedded source sha256 {src_sha})")

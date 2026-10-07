"""Build the V1 Kaggle notebook. casmi_v0_kaggle.py and casmi_v1_kaggle.py are embedded verbatim (%%writefile).

    python research/kaggle_v1/build_notebook.py

Kaggle inputs: the competition data, dataset "casmi-v0-wheels" (offline matchms/rdkit wheels) and dataset
"casmi-v1-assets" (universe.parquet + universe_fp.npy from research/kaggle_v1/build_assets.py).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V0 = HERE.parent / "kaggle_v0" / "casmi_v0_kaggle.py"
V1 = HERE / "casmi_v1_kaggle.py"
INSTALL = (HERE.parent / "kaggle_v0" / "build_notebook.py").read_text(encoding="utf-8").split("INSTALL = r'''")[1].split("'''")[0]
src0, src1 = V0.read_text(encoding="utf-8"), V1.read_text(encoding="utf-8")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
cfg = src1.split("CFG = ")[1].split("\n\n")[0]

HEADER = f"""# CASMI 2026 — V1 hybrid: V0b library retrieval + COCONUT kNN fingerprint retrieval

- **Class 1 branch (unchanged V0b):** raw precursor ±0.01 Da over all train spectra → matchms ModifiedCosine →
  same-adduct candidates → SUM over the molecule's spectra.
- **Class 2 branch:** kNN over train reference spectra (10 test adducts, enveda-180 excluded, ≤3 per structure and library),
  0.01 Da fragment + neutral-loss bins → similarity-weighted neighbour Morgan fingerprint → cosine against
  COCONUT 2026-09 (CC0) + train structures within ±5 ppm of the molecule's neutral mass.
- **Fusion:** library candidates with best ModifiedCosine ≥ tau first (V0b order), then the kNN list. tau was fitted on a
  local mixed Class-1/Class-2 proxy (research/kaggle_v1/proxy_eval.py), not on the leaderboard.

Config: `{cfg}`

Sources: casmi_v0_kaggle.py sha256 `{sha(src0)}`, casmi_v1_kaggle.py sha256 `{sha(src1)}`.
"""

RUN = '''import os, sys
os.chdir("/kaggle/working"); sys.path.insert(0, "/kaggle/working")
import casmi_v1_kaggle as v1
report = v1.main(mode="kaggle")
print("submission.csv written:", report["output"])
'''


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell("markdown", HEADER), cell("code", INSTALL),
                cell("code", "%%writefile /kaggle/working/casmi_v0_kaggle.py\n" + src0),
                cell("code", "%%writefile /kaggle/working/casmi_v1_kaggle.py\n" + src1),
                cell("code", RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "casmi_v1_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "shishiradhikari11/casmi-v1-hybrid-knn", "title": "CASMI V1 hybrid kNN", "code_file": "casmi_v1_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": ["shishiradhikari11/casmi-v0-wheels", "shishiradhikari11/casmi-v1-assets"],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", HERE / "casmi_v1_kaggle.ipynb", "v0", sha(src0)[:12], "v1", sha(src1)[:12])

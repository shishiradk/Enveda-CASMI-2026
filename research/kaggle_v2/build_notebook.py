"""Build the V2 Kaggle notebook. casmi_v0/v1/v2_kaggle.py are embedded verbatim (%%writefile) and imported.

    python research/kaggle_v2/build_notebook.py

Kaggle inputs: competition data, "casmi-v0-wheels" (offline matchms/rdkit wheels), "casmi-v1-assets" (COCONUT + train
universe), "casmi-v2-pubchem" (pubchem_rows.parquet from research/kaggle_v1/build_pubchem.py join2).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
SRC = {"casmi_v0_kaggle.py": R / "kaggle_v0" / "casmi_v0_kaggle.py", "casmi_v1_kaggle.py": R / "kaggle_v1" / "casmi_v1_kaggle.py",
       "casmi_v2_kaggle.py": HERE / "casmi_v2_kaggle.py"}
INSTALL = (R / "kaggle_v0" / "build_notebook.py").read_text(encoding="utf-8").split("INSTALL = r'''")[1].split("'''")[0]
text = {k: p.read_text(encoding="utf-8") for k, p in SRC.items()}
sha = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in text.items()}
cfg = text["casmi_v2_kaggle.py"].split("CFG = ")[1].split("\n")[0]

HEADER = f"""# CASMI 2026 — V2: kNN fingerprint retrieval over COCONUT + train structures + PubChem

- **Spectrum → fingerprint:** kNN over train reference spectra (10 test adducts, enveda-180 excluded, ≤3 per structure and
  library), 0.01 Da fragment + neutral-loss bins, top-20 same-polarity neighbours, similarity-weighted Morgan r2/2048.
- **Candidates:** COCONUT 2026-09 (CC0) + train structures + PubChem (public domain; ~101M filtered CIDs) within ±5 ppm of
  the molecule's neutral mass; PubChem rows merged per InChIKey14, fingerprints computed on the fly.
- **Score:** cosine(predicted, candidate) + tier bonus for COCONUT/train structures. The bonus was chosen on a local
  three-scenario proxy (Class 1, COCONUT Class 2, PubChem-only Class 2), not on the leaderboard.

Config: `{cfg}`

Sources (sha256): {", ".join(f"`{k}` {v[:12]}" for k, v in sha.items())}.
"""

RUN = '''import os, sys
os.chdir("/kaggle/working"); sys.path.insert(0, "/kaggle/working")
import casmi_v2_kaggle as v2
report = v2.main(mode="kaggle")
print("submission.csv written:", report["output"])
'''


def cell(kind, t):
    c = {"cell_type": kind, "metadata": {}, "source": t.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


cells = [cell("markdown", HEADER), cell("code", INSTALL)]
cells += [cell("code", f"%%writefile /kaggle/working/{k}\n" + v) for k, v in text.items()]
cells += [cell("code", RUN)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "casmi_v2_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "shishiradhikari11/casmi-v2-pubchem-knn", "title": "CASMI V2 PubChem kNN", "code_file": "casmi_v2_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["shishiradhikari11/casmi-v0-wheels", "shishiradhikari11/casmi-v1-assets", "shishiradhikari11/casmi-v2-pubchem"],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", HERE / "casmi_v2_kaggle.ipynb", {k: v[:12] for k, v in sha.items()})

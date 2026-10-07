"""Copy the package code next to the prepared data so that results/train_pkg/data is ONE flat folder that can be
uploaded as the private Kaggle dataset shishiradhikari11/casmi-train-pkg. Uploads nothing itself.

    python research/train_pkg/make_upload.py
    (then, after review:  kaggle datasets create -p results/train_pkg/data      -> private by default)
"""
import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE.parent.parent / "results/train_pkg/data"
CODE = ["fp_model.py", "train_fp.py", "check_result.py", "check_env.py", "setup_windows.bat", "run_train.bat",
        "check_result.bat", "smoke_test.bat", "send_results.bat", "RUNSHEET.md", "PROMPT_FOR_ASSISTANT.md",
        "kaggle/casmi_fp_train.ipynb", "kaggle/kernel-metadata.json"]
for f in CODE:
    shutil.copy2(HERE / f, DST / Path(f).name)
json.dump({"title": "casmi-train-pkg", "id": "shishiradhikari11/casmi-train-pkg",
           "licenses": [{"name": "other"}]}, open(DST / "dataset-metadata.json", "w"), indent=1)
tot = sum(p.stat().st_size for p in DST.iterdir() if p.is_file())
print(f"{DST}: {len(list(DST.iterdir()))} files, {tot / 1e9:.2f} GB")

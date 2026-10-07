"""Copy the CFT code next to the new data so that results/train_cft/data is ONE flat folder that can be uploaded
as the private Kaggle dataset shishiradhikari11/casmi-train-cft (used TOGETHER with casmi-train-pkg).
Uploads nothing itself.

    python research/train_cft/make_upload_cft.py
    (then, after review:  kaggle datasets create -p results/train_cft/data      -> private by default)
"""
import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE.parent.parent / "results/train_cft/data"
CODE = ["cft_model.py", "cft_fp.py", "train_cft.py", "eval_cft.py", "_cft_env.bat", "run_train_cft.bat",
        "check_result_cft.bat", "smoke_test_cft.bat", "send_results_cft.bat", "RUNSHEET_CFT.md",
        "kaggle/casmi_cft_train.ipynb"]
for f in CODE:
    shutil.copy2(HERE / f, DST / Path(f).name)
shutil.copy2(HERE / "kaggle/kernel-metadata.json", DST / "kernel-metadata-cft.json")   # rename back to push the notebook
json.dump({"title": "casmi-train-cft", "id": "shishiradhikari11/casmi-train-cft",
           "licenses": [{"name": "other"}]}, open(DST / "dataset-metadata.json", "w"), indent=1)
files = sorted(p for p in DST.iterdir() if p.is_file())
tot = sum(p.stat().st_size for p in files)
for p in files: print(f"  {p.stat().st_size / 1e6:10.2f} MB  {p.name}")
print(f"{DST}: {len(files)} files, {tot / 1e9:.2f} GB")

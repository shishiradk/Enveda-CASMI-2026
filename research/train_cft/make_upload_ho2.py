"""Make the two ho2 upload folders (held out: S1/S2 + S3 + S4 keys) for Akriti's laptop run (CFT) and our Kaggle run
(FPNet). Uploads nothing itself.

    python research/train_cft/make_upload_ho2.py

results/train_pkg/data_ho2 -> dataset akritirijal04/casmi-train-pkg-ho2 (prep_data.py --held ... held_S4)
results/train_cft/data_ho2 -> dataset akritirijal04/casmi-train-cft-ho2 (prep_cft.py  --held ... held_S4)
The CFT .bat files get the ho2 options baked in (--name ho2, results dataset casmi-cft-ho2-results), so the runsheet
research/train_pkg/AKRITI_HO2.md stays click-only.
"""
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG, CFT = ROOT / "results/train_pkg/data_ho2", ROOT / "results/train_cft/data_ho2"
TP, TC = ROOT / "research/train_pkg", ROOT / "research/train_cft"

for f in ["fp_model.py", "train_fp.py", "check_result.py", "check_env.py", "setup_windows.bat", "run_train.bat",
          "check_result.bat", "smoke_test.bat", "send_results.bat", "RUNSHEET.md", "PROMPT_FOR_ASSISTANT.md",
          "kaggle/casmi_fp_train.ipynb", "kaggle/kernel-metadata.json"]:
    shutil.copy2(TP / f, PKG / Path(f).name)
json.dump({"title": "casmi-train-pkg-ho2", "id": "akritirijal04/casmi-train-pkg-ho2",
           "licenses": [{"name": "other"}]}, open(PKG / "dataset-metadata.json", "w"), indent=1)

if (CFT / "cpool_fp.npy").exists():  # the CFT package is built after the FP package (prep_cft.py reads it)
    for f in ["cft_model.py", "cft_fp.py", "train_cft.py", "eval_cft.py", "_cft_env.bat", "check_result_cft.bat",
              "smoke_test_cft.bat", "RUNSHEET_CFT.md", "kaggle/casmi_cft_train.ipynb"]:
        shutil.copy2(TC / f, CFT / Path(f).name)
    shutil.copy2(TC / "kaggle/kernel-metadata.json", CFT / "kernel-metadata-cft.json")
    shutil.copy2(TP / "AKRITI_HO2.md", CFT / "AKRITI_HO2.md")

    run = (TC / "run_train_cft.bat").read_text()
    old = "%PYEXE% train_cft.py --data %DATA% --out out_cft %*"
    assert run.count(old) == 1
    (CFT / "run_train_cft.bat").write_text(run.replace(old, "%PYEXE% train_cft.py --data %DATA% --out out_cft --name ho2 %*"))

    send = (TC / "send_results_cft.bat").read_text()
    assert send.count("casmi-cft-results") >= 3, send.count("casmi-cft-results")
    (CFT / "send_results_cft.bat").write_text(send.replace("casmi-cft-results", "casmi-cft-ho2-results"))

    json.dump({"title": "casmi-train-cft-ho2", "id": "akritirijal04/casmi-train-cft-ho2",
               "licenses": [{"name": "other"}]}, open(CFT / "dataset-metadata.json", "w"), indent=1)
for d in (PKG, CFT) if (CFT / "cpool_fp.npy").exists() else (PKG,):
    files = [p for p in d.iterdir() if p.is_file()]
    print(f"{d}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")

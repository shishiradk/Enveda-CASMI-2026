"""Click-only laptop package for Akriti: FPNet F-B (CFT recipe, FULL data, seed 2, merge_p 0.6, bs 256) on her RTX 5050.
Data comes from her own private datasets akritirijal04/casmi-train-{pkg,cft}-full.
    python research/v4n_rebuild/make_akriti_full_b.py  ->  research/v4n_rebuild/akriti_full_b.zip
"""
import re, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TP, TC = ROOT / "research/train_pkg", ROOT / "research/train_cft"
PKG, CFT = ROOT / "results/train_pkg/data_full", ROOT / "results/train_cft/data_full"
OUT = ROOT / "research/v4n_rebuild/akriti_full_b"
OUT.mkdir(exist_ok=True)
files = {}
# downloader, generated from the uploaded folders (same files, same sizes)
dl = (TP / "download_ho2.bat").read_text(encoding="utf-8")
head, tail = dl.split("call :get", 1)[0], dl[dl.index("echo.\nif !BAD!"):]
lines = []
for ds, d in (("akritirijal04/casmi-train-pkg-full", PKG), ("akritirijal04/casmi-train-cft-full", CFT)):
    for f in sorted(d.iterdir()):
        if f.is_file() and f.name != "dataset-metadata.json":
            lines.append(f"call :get {ds} {f.name} {f.stat().st_size}\n")
files["download_full.bat"] = (head + "".join(lines) + tail).replace("download_ho2.bat", "download_full.bat")
for f in ["setup_windows.bat", "check_env.py"]:
    files[f] = (TP / f).read_text(encoding="utf-8")
for f in ["_cft_env.bat", "check_result_cft.bat", "smoke_test_cft.bat"]:
    files[f] = (TC / f).read_text(encoding="utf-8")
run = (TC / "run_train_cft.bat").read_text(encoding="utf-8")
old = "%PYEXE% train_cft.py --data %DATA% --out out_cft %*"
assert run.count(old) == 1
files["run_train_cft.bat"] = run.replace(old, "%PYEXE% train_cft.py --data %DATA% --out out_cft --name full_b --seed 2 --merge_p 0.6 %*")
send = (TC / "send_results_cft.bat").read_text(encoding="utf-8")
assert send.count("casmi-cft-results") >= 3
files["send_results_cft.bat"] = send.replace("casmi-cft-results", "casmi-cft-full-b-results")
files["AKRITI_FULL_B.md"] = """# Akriti: CFT "full_b" training on your laptop (RTX 5050)

Hi Akriti! Thanks for R-B - it worked perfectly. This is the next one: the same network on the full data. About
**4-6 hours** of laptop time; you can stop and restart any time. **Nothing gets submitted.**

**Before you start:** keep the laptop plugged in, sleep set to Never, close games. Need about **15 GB free** on C:.

1. Make an empty folder **C:\casmi-full** and unzip **akriti_full_b.zip** into it.
2. Double-click **download_full.bat**. It downloads about 3.6 GB from your own Kaggle account (the Kaggle login is the
   kaggle.json Shishir set up; if it says kaggle.json not found, tell Shishir). It must end with **ALL FILES DOWNLOADED**.
   If some files failed, double-click it again.
3. Double-click **setup_windows.bat**. It must end with **SETUP OK** and `torch.cuda.is_available(): True`.
4. Optional 5-minute test: **smoke_test_cft.bat** -> must end with **RESULT: ALL OK**.
5. Double-click **run_train_cft.bat**. After 15 minutes send Shishir one line with `step/s` and `vram`.
   If the window closes or the laptop restarts: double-click run_train_cft.bat again - it continues.
   If it says **OUT OF GPU MEMORY**: open cmd in the folder and run `run_train_cft.bat --bs 128`, and tell Shishir.
6. When it says **FINISHED**: double-click **check_result_cft.bat** -> must say **RESULT: ALL OK**.
7. Double-click **send_results_cft.bat**, type `akritirijal04`. Then send Shishir the text of
   `out_cft\export\check_summary_cft.txt`.

Thank you! 🙏
"""
for n, t in files.items():
    (OUT / n).write_text(t, encoding="utf-8", newline="\r\n" if n.endswith(".bat") else None)
with zipfile.ZipFile(ROOT / "research/v4n_rebuild/akriti_full_b.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in files: z.write(OUT / n, n)
print(sorted(files), len(lines), "files in downloader")

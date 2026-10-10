"""Click-only laptop package for Akriti: FPNet R-B (CFT recipe, HO_R held out, seed 2, merge_p 0.6) on her RTX 5050.
Data comes from her own private datasets akritirijal04/casmi-train-{pkg,cft}-hor (uploaded 2026-10-07).
    python research/v4n_rebuild/make_akriti_hor_b.py  ->  research/v4n_rebuild/akriti_hor_b.zip
"""
import re, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TP, TC = ROOT / "research/train_pkg", ROOT / "research/train_cft"
PKG, CFT = ROOT / "results/train_pkg/data_hoR", ROOT / "results/train_cft/data_hoR"
OUT = ROOT / "research/v4n_rebuild/akriti_hor_b"
OUT.mkdir(exist_ok=True)
files = {}
# downloader, generated from the uploaded folders (same files, same sizes)
dl = (TP / "download_ho2.bat").read_text(encoding="utf-8")
head, tail = dl.split("call :get", 1)[0], dl[dl.index("echo.\nif !BAD!"):]
lines = []
for ds, d in (("akritirijal04/casmi-train-pkg-hor", PKG), ("akritirijal04/casmi-train-cft-hor", CFT)):
    for f in sorted(d.iterdir()):
        if f.is_file() and f.name != "dataset-metadata.json":
            lines.append(f"call :get {ds} {f.name} {f.stat().st_size}\n")
files["download_hor.bat"] = (head + "".join(lines) + tail).replace("download_ho2.bat", "download_hor.bat")
for f in ["setup_windows.bat", "check_env.py"]:
    files[f] = (TP / f).read_text(encoding="utf-8")
for f in ["_cft_env.bat", "check_result_cft.bat", "smoke_test_cft.bat"]:
    files[f] = (TC / f).read_text(encoding="utf-8")
run = (TC / "run_train_cft.bat").read_text(encoding="utf-8")
old = "%PYEXE% train_cft.py --data %DATA% --out out_cft %*"
assert run.count(old) == 1
files["run_train_cft.bat"] = run.replace(old, "%PYEXE% train_cft.py --data %DATA% --out out_cft --name hoR_b --seed 2 --merge_p 0.6 %*")
send = (TC / "send_results_cft.bat").read_text(encoding="utf-8")
assert send.count("casmi-cft-results") >= 3
files["send_results_cft.bat"] = send.replace("casmi-cft-results", "casmi-cft-hor-b-results")
md = (TP / "AKRITI_HO2.md").read_text(encoding="utf-8")
md = md.replace('CFT "ho2"', 'CFT "hoR_b"').replace("casmi-ho2", "casmi-hor").replace("casmi-train-pkg-ho2", "casmi-train-pkg-hor") \
       .replace("casmi-train-cft-ho2", "casmi-train-cft-hor").replace("casmi-cft-ho2-results", "casmi-cft-hor-b-results") \
       .replace("download_ho2.bat", "download_hor.bat").replace("[ho2]", "[hoR_b]").replace("ho2 datasets are ready", "hoR datasets are ready")
md = md.replace("600 test molecules are held out", "22,000 molecules are held out")
md = md.replace("put the file **`download_hor.bat`** (Shishir sends it to you) into", "put **`download_hor.bat`** (it is in the zip Shishir sent) into")
files["AKRITI_HOR_B.md"] = md + "\n\n**Extra step for this job:** copy all files from the zip `akriti_hor_b.zip` into `C:\casmi-hor` (after step 2), replacing any with the same name.\n"
for n, t in files.items():
    (OUT / n).write_text(t, encoding="utf-8", newline="\r\n" if n.endswith(".bat") else None)
with zipfile.ZipFile(ROOT / "research/v4n_rebuild/akriti_hor_b.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in files: z.write(OUT / n, n)
print(sorted(files), len(lines), "files in downloader")

"""Check a finished Kaggle notebook before submitting it.

    python research/tools/verify_kernel.py shishiradhikari11/casmi-pub438-gen25 --version 1
Downloads only submission.csv + the log, prints PASS/FAIL and the exact submit command.
"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("slug"); ap.add_argument("--version", type=int, required=True)
ap.add_argument("--test", default=str(Path(__file__).resolve().parents[2] / "test.parquet"))
a = ap.parse_args()
d = Path(tempfile.mkdtemp())
st = subprocess.run(["kaggle", "kernels", "status", a.slug], capture_output=True, text=True).stdout.strip()
print(st)
if "COMPLETE" not in st:
    sys.exit("FAIL: kernel is not COMPLETE yet")
subprocess.run(["kaggle", "kernels", "output", a.slug, "-p", str(d), "--file-pattern", r"submission\.csv|.*\.json$"],
               capture_output=True, text=True, timeout=900)
sub = d / "submission.csv"
ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)
log = next(iter(d.glob("*.log")), None)
text = ""
if log:
    try:
        text = "".join(x.get("data", "") for x in json.load(open(log)))
    except Exception:
        text = open(log, encoding="utf-8", errors="ignore").read()
check("submission.csv present", sub.exists())
if sub.exists():
    s = pd.read_csv(sub, dtype=str, keep_default_na=False)
    n_test = pd.read_parquet(a.test, columns=["molecule_id"]).molecule_id.nunique()
    check("columns molecule_id,smiles", list(s.columns) == ["molecule_id", "smiles"], list(s.columns))
    check("one row per molecule", len(s) == n_test and s.molecule_id.is_unique, f"{len(s)} rows vs {n_test}")
    k = s.smiles.str.split(";").map(lambda x: len([y for y in x if y]))
    check("no empty rows, <=25 guesses", int(k.min()) >= 1 and int(k.max()) <= 25, f"min {k.min()} max {k.max()}")
check("RDKit 2026.03.3 in log", "rdkit 2026.03.3" in text)
check("no Traceback in log", "Traceback" not in text, f"{text.count('Traceback')} found" if "Traceback" in text else "")
print("\nRESULT:", "PASS" if ok else "FAIL")
if ok:
    print(f'kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k {a.slug} -v {a.version} '
          f'-f submission.csv -m "{a.slug.split("/")[1]} v{a.version}"')

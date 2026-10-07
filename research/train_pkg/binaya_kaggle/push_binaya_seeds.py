"""Push seed training kernels to binayaadhikari13's Kaggle account.

Usage:
    python research/train_pkg/binaya_kaggle/push_binaya_seeds.py [--seed 30|40|all]
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINAYA_CONFIG = ROOT / "research" / "scratch_wf" / "binaya_kaggle"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="30", choices=["30", "40", "all"])
    a = ap.parse_args()

    assert BINAYA_CONFIG.exists() and (BINAYA_CONFIG / "kaggle.json").exists(), \
        f"Missing {BINAYA_CONFIG / 'kaggle.json'}"

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(BINAYA_CONFIG)

    seeds = ["30", "40"] if a.seed == "all" else [a.seed]
    for s in seeds:
        folder = ROOT / "research" / "train_pkg" / "binaya_kaggle" / f"s{s}"
        print(f"Pushing kernel for Seed {s} from {folder}...")
        res = subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)], env=env, text=True, capture_output=True)
        print("stdout:", res.stdout)
        print("stderr:", res.stderr)
        if res.returncode == 0:
            print(f"SUCCESS: Kernel binayaadhikari13/casmi-fp-train-full-s{s} pushed and running!")
        else:
            print(f"FAIL: returncode {res.returncode}")

if __name__ == "__main__":
    main()

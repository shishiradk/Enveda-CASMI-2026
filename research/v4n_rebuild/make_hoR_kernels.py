"""v4n rebuild, step 2: stage the HO_R FPNet (CFT recipe) training runs R-A and R-B (REBUILD_SPEC.md 3.7(3b,c)).

    python research/v4n_rebuild/make_hoR_kernels.py

Data (built by prep_data.py / prep_cft.py with --held results/v4n/hoR_keys.parquet + ho2 files):
  results/train_pkg/data_hoR  -> dataset <owner>/casmi-train-pkg-hor   (base spectra + pool)
  results/train_cft/data_hoR  -> dataset <owner>/casmi-train-cft-hor   (CFT pool, PubChem decoys, code)
Kernels (one per teammate account, private):
  R-A  akritirijal04/casmi-cft-hor-a      --name hoR_a --seed 1 --merge_p 0.3
  R-B  binayaadhikari13/casmi-cft-hor-b   --name hoR_b --seed 2 --merge_p 0.6
Writes results/v4n/kernels/<run>/ and the code files into the data folders. Uploads nothing.
"""
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG, CFT = ROOT / "results/train_pkg/data_hoR", ROOT / "results/train_cft/data_hoR"
TP, TC = ROOT / "research/train_pkg", ROOT / "research/train_cft"
KD = ROOT / "results/v4n/kernels"
RUNS = {"hoR_a": dict(owner="akritirijal04", slug="casmi-cft-hor-a", seed=1, merge_p=0.3),
        "hoR_b": dict(owner="binayaadhikari13", slug="casmi-cft-hor-b", seed=2, merge_p=0.6)}

assert (PKG / "meta.json").exists() and (CFT / "cpool_fp.npy").exists(), "run prep_data.py and prep_cft.py first"
meta = json.load(open(PKG / "meta.json"))
assert meta["excluded"]["held_keys"] > 20000, meta["excluded"]  # HO_R really held out
for f in ["fp_model.py", "check_result.py"]:
    shutil.copy2(TP / f, PKG / f)
for f in ["cft_model.py", "cft_fp.py", "train_cft.py", "eval_cft.py"]:
    shutil.copy2(TC / f, CFT / f)

nb0 = json.load(open(TC / "kaggle/casmi_cft_train.ipynb", encoding="utf-8"))
for run, r in RUNS.items():
    nb = json.loads(json.dumps(nb0))
    c0 = "".join(nb["cells"][0]["source"])
    nb["cells"][0]["source"] = (f"# CASMI26 v4n rebuild - FPNet {run} (CFT recipe, HO_R held out: {meta['excluded']['held_keys']:,} keys)\n\n"
                                f"seed {r['seed']}, merge_p {r['merge_p']}. Data: casmi-train-pkg-hor + casmi-train-cft-hor.\n\n" + c0)
    c1 = "".join(nb["cells"][1]["source"])
    old = "prev = sorted(set(os.path.dirname(p) for p in glob.glob(f'/kaggle/input/**/ckpt_{PRESET}.pt', recursive=True)))"
    assert c1.count(old) == 1
    c1 = c1.replace(old, f"NAME = '{run}'\n" + old.replace("ckpt_{PRESET}", "ckpt_{NAME}"))
    nb["cells"][1]["source"] = c1.splitlines(keepends=True)
    c2 = "".join(nb["cells"][2]["source"])
    old = "'--preset', PRESET, '--max_minutes', '600']"
    assert c2.count(old) == 1
    c2 = c2.replace(old, f"'--preset', PRESET, '--name', NAME, '--seed', '{r['seed']}', '--merge_p', '{r['merge_p']}', '--max_minutes', '600']")
    nb["cells"][2]["source"] = c2.splitlines(keepends=True)
    d = KD / run
    d.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(d / "casmi_cft_train.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"id": f"{r['owner']}/{r['slug']}", "title": r["slug"], "code_file": "casmi_cft_train.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
               "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{r['owner']}/casmi-train-pkg-hor", f"{r['owner']}/casmi-train-cft-hor"],
               "competition_sources": [], "kernel_sources": [], "model_sources": []},
              open(d / "kernel-metadata.json", "w"), indent=1)
    print("staged", run, "->", d)
print("data:", PKG, CFT, "held keys", meta["excluded"]["held_keys"])

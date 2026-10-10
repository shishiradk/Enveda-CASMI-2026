"""Build a Kaggle kernel folder = the public 0.438 notebook with the class-3 generation budget changed.

    python research/tools/make_variant.py --name gen12 --gen-n 12 --gen-total 400
    then:  kaggle kernels push -p research/tools/variants/<name>
Only cell 11's EngineCfg line changes, so any score difference vs 0.438 is that budget and nothing else.
Defaults of the public notebook: gen_n_analog=6, gen_max_total=150.
"""
import argparse, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/scratch_opencode/pubnb/samanyu1808"
NB = "casmi26-iceberg-where-glacier-is-blind-lb-0-438.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--gen-n", type=int, default=6)
ap.add_argument("--gen-total", type=int, default=150)
ap.add_argument("--owner", default="shishiradhikari11")
a = ap.parse_args()
nb = json.load(open(SRC / NB, encoding="utf-8"))
s = "".join(nb["cells"][11]["source"])
old = "EngineCfg(generate=True), bank)"
assert s.count(old) == 1, "cell 11 changed; check the notebook"
s = s.replace(old, f"EngineCfg(generate=True, gen_n_analog={a.gen_n}, gen_max_total={a.gen_total}), bank)")
nb["cells"][11]["source"] = s.splitlines(keepends=True)
d = ROOT / "research/tools/variants" / a.name
d.mkdir(parents=True, exist_ok=True)
code = f"casmi_{a.name}.ipynb"
json.dump(nb, open(d / code, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
meta = json.load(open(SRC / "kernel-metadata.json"))
json.dump(dict(id=f"{a.owner}/casmi-pub438-{a.name}", title=f"casmi pub438 {a.name}", code_file=code, language="python",
               kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
               machine_shape="NvidiaTeslaT4", dataset_sources=meta["dataset_sources"],
               competition_sources=meta["competition_sources"], kernel_sources=[], model_sources=[]),
          open(d / "kernel-metadata.json", "w"), indent=1)
print(f"built {d}\npush:   kaggle kernels push -p {d}\nslug:   {a.owner}/casmi-pub438-{a.name}")

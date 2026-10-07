"""Stage the offline Kaggle dataset shishiradhikari11/casmi-fm-runner (private).

    python research/kaggle_e1/e3/build_fm_dataset.py <stage dir> <wheel dir>

<wheel dir> holds the Linux cp312 wheels fetched with
    pip download --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
        --python-version 312 --no-deps rdkit==2025.3.6 LinSATNet==0.1.3 einops==0.8.2

Layout: code/ (runner, fm_env, shims, ms_pred_src/ms_pred subset), weights/, wheels/, LICENSES/, ATTRIBUTION.md.
Nothing under research/forward_model or external/ is modified; the runner is copied and patched here.
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FM = ROOT / "research" / "forward_model"
UP = ROOT / "external" / "forward_model" / "ms-pred"
CK = ROOT / "external" / "forward_model" / "ckpt" / "stripped"
UPSTREAM_COMMIT = "708148c2a8eb0c32120644436fefd2fe90cd2214"
SUBSET = ["common", "iceberg", "glacier", "graphormer", "magma", "nn_utils"]

stage, wheels = Path(sys.argv[1]), Path(sys.argv[2])
if stage.exists():
    shutil.rmtree(stage)
code = stage / "code"
code.mkdir(parents=True)
ignore = shutil.ignore_patterns("__pycache__", "*.pyc")

# ---- our runner (copied, with two small additions for unattended GPU runs) ----------------------
src = (FM / "ice_runner.py").read_text(encoding="utf-8")


def patch(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


# 1. keep the first prediction traceback in the output's meta (CUDA problems must be visible in the notebook log)
src = patch(src, '            oom = "out of memory" in str(exc).lower()\n',
            '            oom = "out of memory" in str(exc).lower()\n'
            '            if "first_predict_error" not in meta:\n'
            '                meta["first_predict_error"] = traceback.format_exc()[-3000:]\n'
            '                log(args, "first prediction error\\n" + meta["first_predict_error"])\n')
# 2. stop early when nothing at all can be predicted (systematic failure) instead of burning the budget
src = patch(src, '                bump(f"predict_error:{type(exc).__name__}")\n',
            '                bump(f"predict_error:{type(exc).__name__}")\n'
            '                state["fail"] = state.get("fail", 0) + 1\n'
            '                if not state.get("ok") and state["fail"] >= 24:\n'
            '                    raise RuntimeError("24 single predictions failed and none succeeded: giving up")\n')
src = patch(src, '            pred_cache[(t[0], t[2], t[3])] = sp\n',
            '            pred_cache[(t[0], t[2], t[3])] = sp\n'
            '        state["ok"] = state.get("ok", 0) + len(tasks)\n')
(code / "ice_runner.py").write_text(src, encoding="utf-8", newline="\n")
shutil.copy(FM / "fm_env.py", code / "fm_env.py")
shutil.copytree(FM / "shims", code / "shims", ignore=ignore)
# 3. CUDA fix in the DGL stand-in: upstream ICEBERG calls ``cpu_graph.subgraph(cuda_mask)``; real DGL accepts
#    that, so the stand-in moves the node selection to the graph's device (found in the first T4 run).
shim = (code / "shims" / "dgl" / "__init__.py").read_text(encoding="utf-8")
shim = patch(shim, '        idx = nodes.long()\n    new_id = torch.full(',
             '        idx = nodes.long()\n    idx = idx.to(g.device)\n    new_id = torch.full(')
(code / "shims" / "dgl" / "__init__.py").write_text(shim, encoding="utf-8", newline="\n")

# ---- upstream ms_pred subset, unmodified ---------------------------------------------------------
dst = code / "ms_pred_src" / "ms_pred"
dst.mkdir(parents=True)
shutil.copy(UP / "src" / "ms_pred" / "__init__.py", dst / "__init__.py")
for name in SUBSET:
    shutil.copytree(UP / "src" / "ms_pred" / name, dst / name, ignore=ignore)
shutil.copy(UP / "LICENSE", code / "ms_pred_src" / "LICENSE")

# ---- weights and wheels ---------------------------------------------------------------------------
(stage / "weights").mkdir()
for f in ("iceberg_gen.pt", "iceberg_inten.pt", "glacier.pt"):
    shutil.copy(CK / f, stage / "weights" / f)
(stage / "wheels").mkdir()
lic = stage / "LICENSES"
lic.mkdir()
shutil.copy(UP / "LICENSE", lic / "ms-pred_LICENSE_MIT.txt")
for w in sorted(wheels.glob("*.whl")):
    shutil.copy(w, stage / "wheels" / w.name)
    with zipfile.ZipFile(w) as z:
        for n in z.namelist():
            low = n.lower()
            if ("dist-info" in low and "licen" in low) or low == "rdkit/license.txt":
                (lic / (w.name.split("-")[0] + "_" + n.replace("/", "_"))).write_bytes(z.read(n))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


rows = "\n".join(f"| `{p.relative_to(stage).as_posix()}` | {p.stat().st_size:,} | `{sha(p)}` |"
                 for p in sorted(list((stage / "weights").iterdir()) + list((stage / "wheels").iterdir())))
(stage / "ATTRIBUTION.md").write_text(f"""# casmi-fm-runner: forward-model re-scorer for CASMI 2026 (offline bundle)

| Part | Origin | Licence |
|---|---|---|
| `code/ms_pred_src/ms_pred/` ({", ".join(SUBSET)}; unmodified) | https://github.com/coleygroup/ms-pred, commit `{UPSTREAM_COMMIT}` | MIT, Copyright (c) 2023 Samuel Goldman (`code/ms_pred_src/LICENSE`) |
| `weights/iceberg_gen.pt`, `weights/iceberg_inten.pt` | ICEBERG 2.1 `msg_all` checkpoints linked from the ms-pred README ("open-source MassSpecGym-trained weights"); Lightning checkpoints reduced to state_dict + hyper-parameters + provenance, weights unchanged | distributed by the ms-pred authors with the MIT repository; no separate licence file in the download |
| `weights/glacier.pt` | GLACIER MassSpecGym checkpoint linked from the same README line; reduced the same way | as above |
| training data of the weights | MassSpecGym (https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) | MIT |
| `code/ice_runner.py`, `code/fm_env.py`, `code/shims/` (pure-torch stand-ins for DGL, torch_scatter, pytorch_lightning, pygmtools) | written for this project | MIT |
| `wheels/rdkit-2025.3.6-*.whl` | https://pypi.org/project/rdkit/2025.3.6/ | BSD-3-Clause (the wheel bundles third-party shared libraries under their own permissive licences, see the wheel) |
| `wheels/LinSATNet-0.1.3-*.whl` | https://pypi.org/project/LinSATNet/0.1.3/ | MIT |
| `wheels/einops-0.8.2-*.whl` | https://pypi.org/project/einops/0.8.2/ | MIT |

Licence texts: `LICENSES/`. The original checkpoint sha256 values are stored inside each weight file (`provenance`).

| File | Bytes | sha256 |
|---|---|---|
{rows}
""", encoding="utf-8", newline="\n")
(stage / "dataset-metadata.json").write_text(json.dumps({
    "title": "casmi-fm-runner", "id": "shishiradhikari11/casmi-fm-runner", "licenses": [{"name": "other"}]}, indent=2))
tot = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
print("staged", stage, f"{tot / 1e6:.1f} MB", sum(1 for p in stage.rglob('*') if p.is_file()), "files")

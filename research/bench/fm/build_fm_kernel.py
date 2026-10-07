"""Write the private Kaggle notebook shishiradhikari11/casmi-bench-fm: the E3 runner, E3 settings, on the bench lists.

    python research/bench/fm/build_fm_kernel.py

Uses e3/fm_stage.py (locate, install_site, run_model: same command line as E3).  The runner is copied to the working
directory with one addition: every raw prediction (canonical SMILES, adduct, energy) -> float32 peaks is pickled, so
other merges / similarities can be computed offline.  Scoring code is untouched.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3 = HERE.parents[1] / "kaggle_e1" / "e3"
OUT = HERE.parents[2] / "results" / "bench" / "fm" / "kernel"
FILES = {n: (E3 / n).read_text(encoding="utf-8") for n in ("fm_rerank.py", "fm_stage.py")}
CODE = r'''import os, sys, json, glob, shutil, time, subprocess
T0 = time.time()
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
D = '/kaggle/working/e3code'; os.makedirs(D, exist_ok=True)
for n, src in ''' + repr(FILES) + r'''.items():
    open(os.path.join(D, n), 'w').write(src)
sys.path.insert(0, D)
import fm_stage
WORK = '/tmp/fm_work'; os.makedirs(WORK, exist_ok=True)
paths = fm_stage.locate(None, WORK)
print(paths)
site, extra, ver = fm_stage.install_site(paths['root'], WORK, sys.executable)
print('pinned rdkit', ver)
code2 = '/kaggle/working/runner_code'
shutil.rmtree(code2, ignore_errors=True)
shutil.copytree(paths['code'], code2, ignore=shutil.ignore_patterns('ms_pred_src', '__pycache__'))
src = open(os.path.join(code2, 'ice_runner.py')).read()
a = "            pred_cache[(t[0], t[2], t[3])] = sp\n"
assert src.count(a) == 1
src = src.replace(a, a + "            PRED_STORE[(t[0], t[2], t[3])] = sp.astype(np.float32)\n")
b = "    if dump is not None:\n        write_json_atomic(dump, args.dump_pred)\n"
assert src.count(b) == 1
src = src.replace(b, "    import pickle\n    pickle.dump({'pred': PRED_STORE, 'canon': canon_cache}, open(args.output + '.pred.pkl', 'wb'), protocol=4)\n" + b)
c = "T0 = time.time()\n"
assert src.count(c) == 1
src = src.replace(c, c + "PRED_STORE = {}\n")
open(os.path.join(code2, 'ice_runner.py'), 'w').write(src)
paths['runner'] = os.path.join(code2, 'ice_runner.py')
inp = sorted(glob.glob('/kaggle/input/**/fm_input.json', recursive=True))[0]
print('input', inp, os.path.getsize(inp))
dummy = os.path.join(WORK, 'none.parquet')          # spectra are inside the input JSON
import pandas as pd
pd.DataFrame({'molecule_id': pd.Series([], dtype=str), 'ms2_mzs': pd.Series([], dtype=object), 'ms2_normalized_intensities': pd.Series([], dtype=object),
              'adduct': pd.Series([], dtype=str), 'precursor_mz': pd.Series([], dtype=float), 'collision_energy_ev': pd.Series([], dtype=object)}).to_parquet(dummy)
STATS = {}
for tag, budget in (('gl', 5400), ('ice', 7200)):
    out = os.path.join(WORK, f'fm_{tag}.json')
    cmdres = fm_stage.run_model(tag, paths, inp, out, dummy, budget, site, extra, sys.executable, 'auto', WORK)
    STATS[tag] = None if cmdres is None else cmdres.get('meta')
    for f in (out, out + '.pred.pkl', os.path.join(WORK, f'fm_{tag}.log')):
        if os.path.exists(f):
            shutil.copy(f, '/kaggle/working/' + os.path.basename(f))
    print(tag, 'done', f'{time.time() - T0:.0f}s', flush=True)
json.dump(STATS, open('/kaggle/working/fm_bench_stats.json', 'w'), indent=1)
shutil.rmtree(code2, ignore_errors=True); shutil.rmtree(D, ignore_errors=True)
print('total', f'{time.time() - T0:.0f}s')
'''
compile(CODE, "nb", "exec")
OUT.mkdir(parents=True, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "casmi_bench_fm.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "shishiradhikari11/casmi-bench-fm", "title": "casmi bench fm", "code_file": "casmi_bench_fm.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["shishiradhikari11/casmi-bench-fm-input", "shishiradhikari11/casmi-fm-runner"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", OUT)

"""Build E1: the prize-eligible base. Second engine of the public v4n-fusion notebook (prvsiyan analog-propagation
two-ranker, CC0 / CC BY inputs) run standalone with our CC BY ChEBI + LIPID MAPS table (casmi-bio-clean).

    python research/kaggle_e1/build_notebook.py <path to fork .ipynb>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
fork = json.load(open(sys.argv[1], encoding="utf-8"))
eng = next("".join(c["source"]) for c in fork["cells"] if "".join(c["source"]).startswith("# Our engine (two-ranker"))
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1"):
    assert bad not in eng, bad

SETUP = '''# E1: prize-eligible base. Inputs: competition data, prvsiyan/casmi26-fp-models-v2 (CC0),
# prvsiyan/casmi26-ranker-features (CC0), megayak/casmi26-simulated-ranker-rows (CC0),
# prvsiyan/coconut-casmi26-candidates (CC BY 4.0), shishiradhikari11/casmi-bio-clean (CC BY 4.0), RDKit wheel (BSD).
import os, sys, glob, re, subprocess, time, json, hashlib, pickle
import numpy as np, pandas as pd
T0 = time.time()
os.makedirs('/kaggle/working/numba_cache', exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = '/kaggle/working/numba_cache'
def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
print('rdkit wheel:', whl)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
import rdkit; print('rdkit', rdkit.__version__)
COMP = os.path.dirname(find('train.parquet'))
print('COMP', COMP, '| BIO', find('bio_fp.npy'), '| fp models', sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True)))
'''
WRITE = '''sub = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
assert len(ENG) > 0, 'engine produced no lists'
out, n_empty = [], 0
for mid in sub.molecule_id:
    e = ENG.get(str(mid)) or {}
    sm = list(dict.fromkeys(e.get('smiles', [])))[:25]
    n_empty += int(not sm)
    out.append((mid, ';'.join(sm) if sm else 'CCO'))
fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
assert fs.molecule_id.is_unique and fs.smiles.notna().all()
fs.to_csv('submission.csv', index=False)
print('E1 submission written:', len(fs), 'molecules | without a list:', n_empty, f'| {time.time()-T0:.0f}s')
'''


def cell(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


OURS = "# Our V2 engine (kNN fingerprint + COCONUT/train + PubChem 5 ppm windows) -> OURS lists, then weighted RRF with ENG.\nOURS, BETA, KRR = {}, 0.4, 3.0\ntry:\n    _t0 = time.time()\n    _code = os.path.dirname(find('v2_lists.py'))\n    rr = subprocess.run([sys.executable, os.path.join(_code, 'v2_lists.py'), '/kaggle/working/our_lists.json', '40', '4'],\n                        capture_output=True, text=True, timeout=3 * 3600)\n    print(rr.stdout[-2000:]); print(rr.stderr[-2000:])\n    OURS = json.load(open('/kaggle/working/our_lists.json'))\n    print('our lists', len(OURS), f'{time.time() - _t0:.0f}s')\nexcept Exception as e:\n    print('OUR ENGINE FAILED -> engine lists only:', repr(e))\nn_top1 = n_new = 0\nfor mid in set(ENG) | set(OURS):\n    e, o = ENG.get(mid) or {}, OURS.get(mid) or {}\n    sc, smi_of = {}, {}\n    for w, L in ((1.0, e), (BETA, o)):\n        for r, (s, k) in enumerate(zip(L.get('smiles', []), L.get('keys', [])), 1):\n            if not k:\n                continue\n            sc[k] = sc.get(k, 0.0) + w / (KRR + r); smi_of.setdefault(k, s)\n    order = sorted(sc, key=lambda k: -sc[k])[:40]\n    ek = e.get('keys', [])\n    n_top1 += int(bool(ek) and bool(order) and order[0] != ek[0])\n    n_new += len(set(order[:25]) - set(ek[:25]))\n    ENG[mid] = dict(smiles=[smi_of[k] for k in order], keys=order)\nprint('fused ENG + OURS: top-1 changed in', n_top1, '| new keys in top-25:', n_new, '| BETA', BETA)\n"
FUSE = "--fuse" in sys.argv
nb = {"cells": [cell(SETUP), cell(eng)] + ([cell(OURS)] if FUSE else []) + [cell(WRITE)], "metadata": fork["metadata"], "nbformat": 4, "nbformat_minor": 5}
(HERE / ("e2" if FUSE else ".") / ("casmi_e2.ipynb" if FUSE else "casmi_e1.ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "shishiradhikari11/casmi-e2-eligible-fused" if FUSE else "shishiradhikari11/casmi-e1-eligible-base",
        "title": "CASMI E2 eligible fused" if FUSE else "CASMI E1 eligible base", "code_file": "casmi_e2.ipynb" if FUSE else "casmi_e1.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["prvsiyan/casmi26-fp-models-v2", "prvsiyan/casmi26-ranker-features",
                            "megayak/casmi26-simulated-ranker-rows", "prvsiyan/coconut-casmi26-candidates",
                            "shishiradhikari11/casmi-bio-clean", "metric/rdkit-2026-3-3-wheel"] + (["shishiradhikari11/casmi-v2-code",
                            "shishiradhikari11/casmi-v1-assets", "shishiradhikari11/casmi-v2-pubchem"] if FUSE else []),
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
(HERE / ("e2" if FUSE else ".") / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", "E2" if FUSE else "E1", "engine cell chars", len(eng))

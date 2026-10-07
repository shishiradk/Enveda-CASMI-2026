"""Build E3 = E1 (prize-eligible base) + our forward-model re-scoring (GLACIER + ICEBERG, ms-pred MIT).

    python research/kaggle_e1/build_e3.py [--smoke N] [--budget SEC]

Reads research/kaggle_e1/casmi_e1.ipynb (not modified), writes research/kaggle_e1/e3/casmi_e3.ipynb and
kernel-metadata.json (kernel shishiradhikari11/casmi-e3-eligible-fm, private, T4, internet off).

Cells: 1 setup (E1's, plus the E3 switches) | 2 engine (E1's, with a runner that also exports the blend
scores, the best library similarity and up to 60 candidates) | 3 forward-model stage (e3/fm_stage.py +
e3/fm_rerank.py embedded; any failure keeps the engine lists) | 4 submission.csv with validity checks.
--smoke N runs the engine on the first N molecules only (debug runs; the other rows get the fallback).
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "e3"
SMOKE = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
BUDGET = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 5400

e1 = json.load(open(HERE / "casmi_e1.ipynb", encoding="utf-8"))
setup, eng, _write = ("".join(c["source"]) for c in e1["cells"])
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1", "casmi26-iceberg", "casmi26-glacier"):
    assert bad not in eng, bad

# ---- cell 1: E1 setup + E3 switches ---------------------------------------------------------------
_w = "if tag in w]"
assert setup.count(_w) == 1                      # the E3 inputs contain a second RDKit wheel (2025.3.6): never install that one here
setup = setup.replace(_w, "if tag in w and not os.path.basename(w).startswith('rdkit-2025')]")
setup = setup.replace("import rdkit; print('rdkit', rdkit.__version__)",
                      "import rdkit; print('rdkit', rdkit.__version__, '' if rdkit.__version__.startswith('2026.03') else '!!! UNEXPECTED RDKIT VERSION !!!')")
SETUP = setup.replace("# E1: prize-eligible base.", "# E3 = E1 (prize-eligible base) + forward-model re-scoring.", 1) + f'''
# E3 additions: shishiradhikari11/casmi-fm-runner (ms-pred MIT code + MassSpecGym weights, our runner and shims, RDKit 2025.3.6 wheel BSD).
SMOKE_N = {SMOKE}            # > 0: engine on the first N molecules only (debug)
FM_ENABLE = True
FM_BUDGET_SEC = {BUDGET}     # wall clock for GLACIER + ICEBERG together (installation and model loading included)
FM_TOPN = 60                 # same-formula groups are formed inside the first 60 candidates
LAM_ICE, LAM_GL = 1.0, 1.0
print('E3 switches: SMOKE_N', SMOKE_N, '| FM_BUDGET_SEC', FM_BUDGET_SEC, '| FM_TOPN', FM_TOPN, '| lam', LAM_ICE, LAM_GL)
try:
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
except Exception as e:
    print('nvidia-smi:', repr(e))
'''

# ---- cell 2: engine, with our extended runner -----------------------------------------------------
RUNNER = '''"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles, keys, scores, lib_max}} (blend order).

E3 version of the E1 runner.  The first 40 entries are exactly E1's list (first 80 candidates in blend order,
de-duplicated on the metric key, at most 40).  When those 40 are full the scan goes on up to 60 entries, so the
forward-model stage can form formula groups over a top 60.  scores = blend score of each kept candidate,
lib_max = best library similarity of any candidate of the molecule (used only to order the forward-model work).
"""
import os, sys, json, time
import numpy as np
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    keep = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    import casmi_engine as E
    _orig_find = E.find
    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    import glob
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    print('engine fp models', fp_models, flush=True)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,
                           workers=os.cpu_count(), limit=limit, w_pv=0.88)
    bl = E.blend_scores(S['pv'], S['ours'], 0.88)
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:max(80, 3 * keep)]
        smis, keys, scs, seen = [], [], [], set()
        for j, c in enumerate(r['cand'][order]):
            if j >= 80 and len(smis) < 40:         # E1 stops after 80 candidates
                break
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k); scs.append(float(bl[mid][order[j]]))
            if len(smis) >= max(40, keep):
                break
        lib = r.get('lib')
        res[str(mid)] = dict(smiles=smis, keys=keys, scores=scs,
                             lib_max=float(np.max(lib)) if lib is not None and len(lib) else 0.0)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)
'''
compile(RUNNER, "eng_runner.py", "exec")
tree = ast.parse(eng)
stmt = [ast.get_source_segment(eng, n) for n in tree.body]
old = [s for s in stmt if s.startswith("open(os.path.join(ENG_DIR, 'eng_runner.py')")]
assert len(old) == 1 and eng.count(old[0]) == 1
ENG = eng.replace(old[0], "open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write(" + repr(RUNNER) + ")")
arg = "'/kaggle/working/eng_lists.json']"
assert ENG.count(arg) == 1
ENG = ENG.replace(arg, "'/kaggle/working/eng_lists.json', str(SMOKE_N), str(FM_TOPN)]")
ENG = ENG.replace("# Our engine (two-ranker + BIO + AFIX) in a subprocess, then weighted reciprocal-rank fusion with the v4g lists.",
                  "# Our engine (two-ranker + BIO + AFIX) in a subprocess -> ENG (E1 lists; E3 also exports scores, lib_max, top 60).", 1)
ast.parse(ENG)
if "--stub" in sys.argv:  # debug only: no engine, ENG = first N molecules of a saved E1 eng_lists.json (no scores -> list position)
    _p, _n = sys.argv[sys.argv.index("--stub") + 1], int(sys.argv[sys.argv.index("--stub") + 2])
    _e = json.load(open(_p))
    ENG = ("# DEBUG STUB: engine skipped\nENG = json.loads(" + repr(json.dumps({m: _e[m] for m in sorted(_e)[:_n]}))
           + ")\nprint('stub lists', len(ENG))\n")

# ---- cell 3: forward-model stage ---------------------------------------------------------------------
FILES = {n: (OUT / n).read_text(encoding="utf-8") for n in ("fm_rerank.py", "fm_stage.py")}
FM = '''# Forward-model re-scoring (GLACIER + ICEBERG, ms-pred MIT, MassSpecGym weights) of same-formula candidates.
# Runs in subprocesses with their own RDKit 2025.3.6 site directory; this process keeps the metric's RDKit.
# Any failure leaves ENG exactly as the engine wrote it.
import traceback, copy
E3_DIR = '/kaggle/working/e3code'; os.makedirs(E3_DIR, exist_ok=True)
E3_FILES = ''' + repr(FILES) + '''
for _n, _src in E3_FILES.items():
    open(os.path.join(E3_DIR, _n), 'w').write(_src)
ENG_E1 = {m: list(e.get('keys', [])) for m, e in ENG.items()}
E1_TOP25 = {m: list(dict.fromkeys(e.get('smiles', [])))[:25] for m, e in ENG.items()}
FM_STATS = {'status': 'not run'}
_rd_before = rdkit.__version__
if FM_ENABLE and len(ENG):
    try:
        _t0 = time.time()
        if E3_DIR not in sys.path:
            sys.path.insert(0, E3_DIR)
        import fm_stage
        _new, _st = fm_stage.run_fm_stage(copy.deepcopy(ENG), os.path.join(COMP, 'test.parquet'), '/tmp/fm_work',
                                          budget_sec=FM_BUDGET_SEC, topn=FM_TOPN, lam_ice=LAM_ICE, lam_gl=LAM_GL)
        assert set(_new) == set(ENG), 'molecule set changed'
        for _m, _e in ENG.items():
            assert sorted(_new[_m]['keys']) == sorted(_e['keys']) and len(_new[_m]['smiles']) == len(_e['smiles']), _m
            assert len(set(_new[_m]['keys'])) == len(_new[_m]['keys']), _m
        ENG = _new
        FM_STATS = dict(_st, status='ok', total_sec=round(time.time() - _t0, 1))
    except Exception as e:
        FM_STATS = {'status': 'FAILED', 'error': repr(e)}
        print('!' * 100)
        print('FORWARD-MODEL STAGE FAILED -> E1 LISTS KEPT UNCHANGED:', repr(e))
        traceback.print_exc()
        print('!' * 100)
import rdkit as _rd
assert _rd.__version__ == _rd_before, 'the notebook RDKit changed'
for _f in sorted(glob.glob('/tmp/fm_work/fm_*')):
    try:
        import shutil; shutil.copy(_f, '/kaggle/working/' + os.path.basename(_f))
    except Exception as e:
        print('copy', _f, repr(e))
json.dump(FM_STATS, open('/kaggle/working/fm_stats.json', 'w'), indent=1)
print('notebook rdkit', _rd.__version__, f'| elapsed {time.time()-T0:.0f}s')
print('FM_STATS ' + json.dumps(FM_STATS))
'''

# ---- cell 4: submission ---------------------------------------------------------------------------------
WRITE = '''sub = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
assert len(ENG) > 0, 'engine produced no lists'
out, n_empty, n_top1, n_order, n_set = [], 0, 0, 0, 0
for mid in sub.molecule_id:
    e = ENG.get(str(mid)) or {}
    sm = list(dict.fromkeys(e.get('smiles', [])))[:25]
    old = E1_TOP25.get(str(mid), [])
    n_top1 += int(sm[:1] != old[:1]); n_order += int(sm != old); n_set += int(set(sm) != set(old))
    n_empty += int(not sm)
    out.append((mid, ';'.join(sm) if sm else 'CCO'))
fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
assert fs.molecule_id.is_unique and fs.smiles.notna().all() and len(fs) == len(sub) and list(fs.molecule_id) == list(sub.molecule_id)
_n = fs.smiles.str.split(';').map(len); _u = fs.smiles.str.split(';').map(lambda v: len(set(v)))
assert (_n <= 25).all() and (_n == _u).all() and (fs.smiles.str.len() > 0).all() and not fs.smiles.str.contains(';;').any()
fs.to_csv('submission.csv', index=False)
pd.DataFrame([(m, ';'.join(E1_TOP25.get(str(m), [])) or 'CCO') for m in sub.molecule_id],
             columns=['molecule_id', 'smiles']).to_csv('e1_order_reference.csv', index=False)
print('E3 submission written:', len(fs), 'rows | without a list:', n_empty, '| SMILES per row min/median/max:',
      int(_n.min()), int(_n.median()), int(_n.max()), '| all unique within row:', bool((_n == _u).all()))
print('E3 vs E1 submission rows: top-1 changed', n_top1, '| top-25 order changed', n_order, '| top-25 membership changed', n_set,
      '| forward stage', FM_STATS.get('status'), f'| {time.time()-T0:.0f}s')
'''
for name, src in (("SETUP", SETUP), ("FM", FM), ("WRITE", WRITE)):
    ast.parse(src)


def cell(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


nb = {"cells": [cell(SETUP), cell(ENG), cell(FM), cell(WRITE)], "metadata": e1["metadata"], "nbformat": 4, "nbformat_minor": 5}
OUT.mkdir(exist_ok=True)
(OUT / "casmi_e3.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": "shishiradhikari11/casmi-e3-eligible-fm", "title": "CASMI E3 eligible fm", "code_file": "casmi_e3.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["prvsiyan/casmi26-fp-models-v2", "prvsiyan/casmi26-ranker-features",
                            "megayak/casmi26-simulated-ranker-rows", "prvsiyan/coconut-casmi26-candidates",
                            "shishiradhikari11/casmi-bio-clean", "metric/rdkit-2026-3-3-wheel",
                            "shishiradhikari11/casmi-fm-runner"],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote E3 | smoke", SMOKE, "| budget", BUDGET, "| engine cell chars", len(ENG), "| fm cell chars", len(FM))

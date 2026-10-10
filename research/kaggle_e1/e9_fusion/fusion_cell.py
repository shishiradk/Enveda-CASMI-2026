# ===== E9: E7 (above) + our rebuilt v4n engine, fused by RRF on the metric key =====
import os, sys, glob, shutil, subprocess, time, json
import pandas as pd
T9 = time.time()
shutil.copy('submission.csv', 'e7_submission.csv')          # E7's final answer, kept as the fallback

def _find9(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

CODE9 = os.path.dirname(_find9('predict.py'))
ROOT9 = '/kaggle/working/v4n_code'
D9 = {'engine': f'{ROOT9}/research/v4n_rebuild/engine', 'sim': f'{ROOT9}/research/v4n_rebuild/sim', 'cft': f'{ROOT9}/research/train_cft'}
for d in D9.values():
    os.makedirs(d, exist_ok=True)
for f in glob.glob(f'{CODE9}/*.py'):
    b = os.path.basename(f)
    shutil.copy(f, D9['sim'] if b in ('common.py', 'simulate.py', 'predict.py') else D9['cft'] if b.startswith('cft_') else D9['engine'])
TAB9 = '/kaggle/working/v4n_tables'
shutil.copytree(os.path.dirname(_find9('pool_fp.npy')), TAB9, dirs_exist_ok=True)
TEST9 = _find9('test.parquet')
env9 = dict(os.environ, V4R_TABLES=TAB9, V4R_SIM_OUT='/kaggle/working/v4n_sim', NUMBA_CACHE_DIR='/kaggle/working/numba_cache')
cmd9 = [sys.executable, '-u', f"{D9['sim']}/predict.py", '--test', TEST9, '--tables', TAB9, '--ckpt', _find9('cft_hoR_a.pt'),
        '--ranker', _find9('ranker_v0.pkl'), '--workers', str(max(1, min(4, os.cpu_count() or 1)))]
W_E7, W_V4N, K0 = 1.0, 0.6, 3.0      # stronger list (E7, LB 0.363) primary; v4n (LB 0.341) secondary; fork constants
try:
    p = subprocess.run(cmd9 + ['--out', '/kaggle/working/v4n_smoke.csv', '--workers', '1', '--limit', '3'], env=env9, timeout=1200)
    assert p.returncode == 0, f'v4n smoke rc {p.returncode}'
    p = subprocess.run(cmd9 + ['--out', '/kaggle/working/v4n_submission.csv'], env=env9, timeout=5 * 3600)
    assert p.returncode == 0, f'v4n rc {p.returncode}'
    sys.path.insert(0, f'{ROOT9}/research/v4n_rebuild')
    from engine.chem import score_key
    e7 = pd.read_csv('e7_submission.csv'); v4 = pd.read_csv('/kaggle/working/v4n_submission.csv')
    V4 = dict(zip(v4.molecule_id.astype(str), v4.smiles))
    _kc = {}
    def _k(s):
        if s not in _kc:
            _kc[s] = score_key(s) or s
        return _kc[s]
    out, n_top1_kept, n_new = [], 0, 0
    for mid, s7 in zip(e7.molecule_id.astype(str), e7.smiles):
        L7 = [x for x in str(s7).split(';') if x]
        L4 = [x for x in str(V4.get(mid, '')).split(';') if x]
        sc, first = {}, {}
        for w, L in ((W_E7, L7), (W_V4N, L4)):
            for r, smi in enumerate(L):
                k = _k(smi)
                sc[k] = sc.get(k, 0.0) + w / (K0 + r)
                first.setdefault(k, smi)
        keys = sorted(sc, key=lambda k: -sc[k])[:25]
        fused = [first[k] for k in keys]
        if len(fused) < 25:
            fused += [x for x in L7 if x not in fused][:25 - len(fused)]
        n_top1_kept += bool(L7) and fused[0] == L7[0]
        n_new += len(set(keys) - {_k(x) for x in L7})
        out.append((mid, ';'.join(fused)))
    sub9 = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    n = sub9.smiles.str.split(';').map(lambda x: len(set(x)))
    assert len(sub9) == len(e7) and (n == 25).all(), (len(sub9), n.min())
    sub9.to_csv('submission.csv', index=False)
    print(f'E9 fused submission written: {len(sub9)} rows | E7 top-1 kept {n_top1_kept}/{len(sub9)} | '
          f'v4n-only candidates added {n_new} ({n_new / len(sub9):.1f}/row) | {time.time() - T9:.0f}s', flush=True)
except Exception as e:
    shutil.copy('e7_submission.csv', 'submission.csv')
    print('E9 fusion FAILED, submission = E7 unchanged:', repr(e), flush=True)
finally:
    shutil.rmtree(TAB9, ignore_errors=True); shutil.rmtree(ROOT9, ignore_errors=True)

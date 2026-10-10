# ===== E11: E7 (above) + our rebuilt v4n engine, RRF-fused to 40, forward-model re-rank of mixed formula groups =====
# Replaces the final E9 cell (research/kaggle_e1/e9_fusion/fusion_cell.py).  Ours, MIT.
#
# Per molecule:
#   1. E7's final list (e7_submission.csv) and the rebuilt engine's list (v4n_submission.csv), the latter cut to its
#      first n_engine entries (v4n_submission.diag.parquet); the rest is mass-nearest padding with no evidence.
#   2. RRF on the metric score key: W_E7 / (K0 + r_E7) + W_V4N / (K0 + r_v4n), r = 0-based rank in the key-deduplicated
#      list; ties keep E7-first insertion order; top KEEP_TOP (40) kept.
#   3. E7's forward-model code (fm_rerank.rerank_molecule, fm_stage.run_model) re-orders same-formula groups of those
#      40, with E7's library protection (lib >= FM_PROTECT_LIB keeps its slot; rebuilt-only candidates have lib 0).
#      Only groups holding at least one rebuilt-only candidate are re-ranked (FM_ONLY_MIXED_GROUPS): E7's own groups
#      were already re-ranked by E7's forward-model stage and are left exactly as E7 ordered them.
#      Forward scores E7 already computed (/kaggle/working/fm_*.json) are reused; only new SMILES are sent.
#   4. Cut to 25 unique keys; a molecule that cannot reach 25 unique keys / 25 unique SMILES keeps E7's list.
#   5. Any exception for one molecule -> E7's list for that molecule.  Any exception outside the per-molecule loops
#      -> submission.csv = E7's, unchanged.  The cell never raises.
#   6. deadline: past it, every remaining molecule keeps E7's list.
#   7. Score keys are computed at most once per SMILES; keys known from E7 (ENG, eng_lists.json) and from the diag
#      parquet are reused.
import json
import math
import os
import shutil
import sys
import time

W_E7, W_V4N, K0, KEEP_TOP, K_OUT = 1.0, 0.6, 3.0, 40, 25
FM_PROTECT_LIB_E11 = 0.6          # = E7's FM_PROTECT_LIB
FM_ONLY_MIXED_GROUPS = True


# ---------------------------------------------------------------- pure helpers (unit-tested, no RDKit)
def split_list(s):
    """'a;b;;c' -> ['a', 'b', 'c']; NaN / None / non-string -> []."""
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(';') if x.strip()]


class KeyCache:
    """SMILES -> metric key, computed at most once per SMILES.  compute(smiles) -> key | None."""

    def __init__(self, compute, prefill=None):
        self.compute = compute
        self.d = {}
        self.n_computed = 0
        for s, k in (prefill or {}).items():
            if isinstance(s, str) and isinstance(k, str) and k:
                self.d[s] = k

    def add(self, smiles, keys):
        for s, k in zip(smiles or [], keys or []):
            if isinstance(s, str) and isinstance(k, str) and k and s not in self.d:
                self.d[s] = k

    def __call__(self, s):
        if s not in self.d:
            self.n_computed += 1
            try:
                k = self.compute(s)
            except Exception:  # noqa: BLE001
                k = None
            self.d[s] = k if isinstance(k, str) and k else None
        return self.d[s]


def dedup(smiles, key_of):
    """Rank-ordered [(key, smiles)], first occurrence of each key, SMILES without a key dropped."""
    out, seen = [], set()
    for s in smiles:
        k = key_of(s)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append((k, s))
    return out


def rrf_fuse(l7, l4, key_of, w7=W_E7, w4=W_V4N, k0=K0, keep=KEEP_TOP):
    """l7, l4: rank-ordered SMILES.  Returns [dict(key, smiles, score, r7, r4)] best first, at most `keep`.
    The SMILES written for a key is E7's when E7 lists it, else the rebuilt engine's."""
    rows = {}
    for w, name, lst in ((w7, 'r7', dedup(l7, key_of)), (w4, 'r4', dedup(l4, key_of))):
        for r, (k, s) in enumerate(lst):
            e = rows.get(k)
            if e is None:
                e = rows[k] = dict(key=k, smiles=s, score=0.0, r7=None, r4=None, order=len(rows))
            e[name] = r
            e['score'] += w / (k0 + r)
    out = sorted(rows.values(), key=lambda e: (-round(e['score'], 12), e['order']))[:keep]
    for e in out:
        e.pop('order')
    return out


def fm_plan(fused, formula_of, lib_of, protect_lib=FM_PROTECT_LIB_E11, only_mixed=FM_ONLY_MIXED_GROUPS):
    """Formula per fused entry for fm_rerank (None = not in any group): protected (lib >= protect_lib) or unknown
    formula -> None; groups of size < 2 -> None; with only_mixed, groups without a rebuilt-only member -> None.
    Returns (formulas, send) where send = positions that belong to a group."""
    f = []
    for e in fused:
        lib = lib_of(e['key'])
        if lib is not None and float(lib) >= protect_lib:
            f.append(None)
            continue
        try:
            f.append(formula_of(e['smiles']))
        except Exception:  # noqa: BLE001
            f.append(None)
    groups = {}
    for i, x in enumerate(f):
        if x is not None:
            groups.setdefault(x, []).append(i)
    for x, mem in groups.items():
        new_member = any(fused[i]['r7'] is None for i in mem)
        if len(mem) < 2 or (only_mixed and not new_member):
            for i in mem:
                f[i] = None
    return f, [i for i, x in enumerate(f) if x is not None]


def fm_apply(fused, formulas, fwd, lams, rerank_fn, topn=KEEP_TOP):
    """fwd: {model: [score | None per fused entry]}.  Re-orders `fused` with E7's rerank_molecule, the RRF score as
    the ranker term.  Returns (new fused list, info)."""
    if not any(x is not None for x in formulas) or not fwd:
        return list(fused), {}
    perm, info = rerank_fn(list(formulas), [e['score'] for e in fused], fwd, lams, topn)
    if sorted(perm) != list(range(len(fused))):
        raise AssertionError('rerank_molecule did not return a permutation')
    return [fused[i] for i in perm], info


def cut(fused, k_out=K_OUT):
    """First k_out entries; asserts k_out unique keys and k_out unique SMILES without ';'."""
    top = fused[:k_out]
    smi = [e['smiles'] for e in top]
    assert len(top) == k_out, f'only {len(top)} unique keys'
    assert len({e['key'] for e in top}) == k_out, 'duplicate key'
    assert len(set(smi)) == k_out, 'duplicate SMILES'
    assert all(isinstance(s, str) and s and ';' not in s for s in smi), 'invalid SMILES string'
    return smi


def validate_submission(rows, e7_rows, k_out=K_OUT):
    """rows, e7_rows: [(molecule_id, smiles_str)].  Same molecules in the same order, no empty / NaN, <= k_out guesses,
    unique guesses per row."""
    assert [m for m, _ in rows] == [m for m, _ in e7_rows], 'molecule set or order differs from E7'
    assert len({m for m, _ in rows}) == len(rows), 'duplicate molecule_id'
    for m, s in rows:
        assert isinstance(s, str) and s and ';;' not in s, f'{m}: empty or malformed row'
        g = s.split(';')
        assert 0 < len(g) <= k_out and len(set(g)) == len(g), f'{m}: {len(g)} guesses / duplicates'


def fuse_all(e7_rows, v4, key_of, deadline_t, clock=time.time, formula_of=None, lib_of=None, fm_score_fn=None,
             rerank_fn=None, lams=None):
    """Core of E11, no I/O.
    e7_rows: [(molecule_id, smiles_str)] = e7_submission.csv in order.
    v4: {molecule_id: (smiles_list, keys_list_or_None, n_engine)}.
    deadline_t: absolute clock time; after it every remaining molecule keeps E7's row.
    formula_of(smiles) / lib_of(mid) -> (key -> lib or None) / fm_score_fn(requests, deadline_t) -> {model: {mid:
    {smiles: score}}} / rerank_fn = fm_rerank.rerank_molecule.  Without formula_of, fm_score_fn or rerank_fn the
    forward-model step is skipped (fused order kept).
    Per-molecule exceptions -> E7's row for that molecule.  Exceptions outside the loops propagate to the caller,
    which then keeps E7's submission unchanged.  Returns (rows, stats)."""
    st = dict(n=len(e7_rows), fused=0, e7_kept_error=0, e7_kept_deadline=0, e7_kept_no_v4n=0, e7_kept_short=0,
              v4n_dropped_padding=0, fm_groups_mixed=0, fm_molecules_reranked=0, new_keys_top25=0, top1_changed=0,
              errors=[])
    fused, plans = {}, {}
    use_fm = formula_of is not None and fm_score_fn is not None and rerank_fn is not None

    # ---- phase A: fuse every molecule to KEEP_TOP
    for mid, s7 in e7_rows:
        if clock() >= deadline_t:
            st['e7_kept_deadline'] += 1
            continue
        try:
            l7 = split_list(s7)
            v = v4.get(mid)
            if v is None:
                st['e7_kept_no_v4n'] += 1
                continue
            smi4, keys4, n_eng = v
            n_eng = max(0, min(int(n_eng), len(smi4)))
            st['v4n_dropped_padding'] += len(smi4) - n_eng
            if keys4 is not None and len(keys4) == len(smi4) and hasattr(key_of, 'add'):
                key_of.add(smi4[:n_eng], keys4[:n_eng])
            f = rrf_fuse(l7, smi4[:n_eng], key_of)
            fused[mid] = f
            if use_fm:
                lib = lib_of(mid) if lib_of else {}
                plans[mid] = fm_plan(f, formula_of, lambda k: (lib or {}).get(k, 0.0))
                st['fm_groups_mixed'] += len({x for x in plans[mid][0] if x is not None})
        except Exception as e:  # noqa: BLE001
            fused.pop(mid, None); plans.pop(mid, None)
            st['e7_kept_error'] += 1
            st['errors'].append(f'A {mid}: {e!r}'[:300])

    # ---- phase B: forward scores for the planned groups (outside the per-molecule loop)
    fwd_all = {}
    if use_fm and any(p[1] for p in plans.values()):
        req = {m: [fused[m][i]['smiles'] for i in p[1]] for m, p in plans.items() if p[1]}
        fwd_all = fm_score_fn(req, deadline_t) or {}

    # ---- phase C: re-rank, cut, write
    rows = []
    for mid, s7 in e7_rows:
        if mid not in fused:
            rows.append((mid, s7)); continue
        if clock() >= deadline_t:
            st['e7_kept_deadline'] += 1
            rows.append((mid, s7)); continue
        try:
            f = fused[mid]
            if use_fm and mid in plans and plans[mid][1]:
                fwd, send = {}, set(plans[mid][1])
                for tag, by_mid in fwd_all.items():
                    sc = (by_mid or {}).get(mid) or {}
                    col = [sc.get(e['smiles']) if i in send else None for i, e in enumerate(f)]
                    if any(x is not None for x in col):
                        fwd[tag] = col
                f2, info = fm_apply(f, plans[mid][0], fwd, lams or {}, rerank_fn)
                st['fm_molecules_reranked'] += int([e['key'] for e in f2] != [e['key'] for e in f])
                f = f2
            if len(f) < K_OUT:
                st['e7_kept_short'] += 1
                rows.append((mid, s7)); continue
            out = cut(f)
            l7 = split_list(s7)
            k7 = {e['key'] for e in f if e['r7'] is not None and e['r7'] < K_OUT}
            st['new_keys_top25'] += sum(1 for e in f[:K_OUT] if e['key'] not in k7)
            st['top1_changed'] += int(bool(l7) and out[0] != l7[0])
            rows.append((mid, ';'.join(out)))
            st['fused'] += 1
        except Exception as e:  # noqa: BLE001
            st['e7_kept_error'] += 1
            st['errors'].append(f'C {mid}: {e!r}'[:300])
            rows.append((mid, s7))
    validate_submission(rows, e7_rows)
    st['errors'] = st['errors'][:20]
    return rows, st


# ---------------------------------------------------------------- forward-model scoring with E7's runner
def load_fm_cache(work_dirs=('/kaggle/working', '/tmp/fm_work'), need_rdkit='2025.03'):
    """E7's forward scores: {model: {mid: {smiles: score}}} from fm_input.json + fm_{gl,ice}.json."""
    cache = {}
    for d in work_dirs:
        try:
            inp = json.load(open(os.path.join(d, 'fm_input.json')))
        except Exception:  # noqa: BLE001
            continue
        for tag in ('gl', 'ice'):
            try:
                res = json.load(open(os.path.join(d, f'fm_{tag}.json')))
            except Exception:  # noqa: BLE001
                continue
            if need_rdkit and not str(res.get('meta', {}).get('rdkit', '')).startswith(need_rdkit):
                continue
            for m, v in (res.get('scores') or {}).items():
                smi = (inp.get(m) or {}).get('smiles') or []
                if isinstance(v, list) and len(v) == len(smi):
                    dst = cache.setdefault(tag, {}).setdefault(m, {})
                    for s, x in zip(smi, v):
                        if isinstance(x, (int, float)) and math.isfinite(x):
                            dst.setdefault(s, float(x))
        if cache:
            break
    return cache


def make_fm_score_fn(cache, test_parquet, work, lams, budget_cap, reserve_sec, gl_share=0.4, need_rdkit='2025.03'):
    """fm_score_fn for fuse_all: cached scores first; the SMILES still missing are scored with E7's fm_stage runner
    (same adduct / instrument settings), inside min(budget_cap, deadline - now - reserve_sec)."""
    def fn(req, deadline_t):
        import pandas as pd
        import fm_stage                               # written to E3_DIR by E7's forward-model cell
        te = pd.read_parquet(test_parquet, columns=['molecule_id', 'adduct'])
        covered = set(te.loc[te.adduct.astype(str).isin(fm_stage.ADDUCTS), 'molecule_id'].astype(str))
        out = {tag: {m: dict(v) for m, v in by.items()} for tag, by in cache.items()}
        miss = {}
        for m, smis in req.items():
            if m not in covered:
                continue
            s_need = [s for s in dict.fromkeys(smis) if any(lams.get(t) and s not in out.get(t, {}).get(m, {})
                                                             for t in fm_stage.MODELS)]
            if s_need:
                miss[m] = s_need
        n_req = sum(len(v) for v in req.values()); n_miss = sum(len(v) for v in miss.values())
        print(f'[e11 fm] requested {n_req} candidates in {len(req)} molecules; {n_req - n_miss} cached or uncovered, '
              f'{n_miss} to score in {len(miss)} molecules', flush=True)
        budget = min(budget_cap, deadline_t - time.time() - reserve_sec)
        if not miss or budget < 300:
            print(f'[e11 fm] runner skipped (budget {budget:.0f}s)', flush=True)
            return out
        t_start = time.time()
        os.makedirs(work, exist_ok=True)
        paths = fm_stage.locate(None, work)
        site, extra, ver = fm_stage.install_site(paths['root'], work, sys.executable)
        inp = os.path.join(work, 'fm_input.json')
        json.dump({m: {'smiles': v} for m, v in miss.items()}, open(inp, 'w'))
        for k, tag in enumerate(fm_stage.MODELS):
            if not lams.get(tag):
                continue
            left = budget - (time.time() - t_start) - 30.0
            b = min(left, gl_share * budget) if k == 0 else left
            if b < 120:
                print(f'[e11 fm] {tag} skipped, {b:.0f}s left', flush=True)
                continue
            res = fm_stage.run_model(tag, paths, inp, os.path.join(work, f'fm_{tag}.json'), test_parquet, b, site,
                                     extra, sys.executable, 'auto', work)
            if res is None or (need_rdkit and not str(res.get('meta', {}).get('rdkit', '')).startswith(need_rdkit)):
                continue
            for m, v in (res.get('scores') or {}).items():
                if m in miss and isinstance(v, list) and len(v) == len(miss[m]):
                    dst = out.setdefault(tag, {}).setdefault(m, {})
                    for s, x in zip(miss[m], v):
                        if isinstance(x, (int, float)) and math.isfinite(x):
                            dst[s] = float(x)
        return out
    return fn


# ---------------------------------------------------------------- notebook glue
def _e11_main():
    import subprocess
    import pandas as pd
    g = globals()
    t_cell = time.time()
    shutil.copy('submission.csv', 'e7_submission.csv')       # E7's final answer = the fallback
    KERNEL_LIMIT_S = 9 * 3600                                 # Code Requirements: notebook <= 9 h
    SAFETY_S = 900
    deadline_t = g['T0'] + KERNEL_LIMIT_S - SAFETY_S if 'T0' in g else t_cell + 3 * 3600
    deadline_s = deadline_t - time.time()
    print(f'E11: {deadline_s:.0f}s until the E11 deadline', flush=True)
    stats = dict(status='not run')
    try:
        comp = g.get('COMP') or os.path.dirname(g['find']('train.parquet'))
        test_parquet = os.path.join(comp, 'test.parquet')     # never a glob: a dataset copy could shadow it

        def find9(name):
            import glob
            hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True), key=len)
            if not hits:
                raise FileNotFoundError(name)
            return hits[0]

        # ---- rebuilt engine (same invocation as E9), bounded by the deadline
        code9 = os.path.dirname(find9('predict.py'))
        root9 = '/kaggle/working/v4n_code'
        d9 = {'engine': f'{root9}/research/v4n_rebuild/engine', 'sim': f'{root9}/research/v4n_rebuild/sim',
              'cft': f'{root9}/research/train_cft'}
        for d in d9.values():
            os.makedirs(d, exist_ok=True)
        import glob
        for f in glob.glob(f'{code9}/*.py'):
            b = os.path.basename(f)
            shutil.copy(f, d9['sim'] if b in ('common.py', 'simulate.py', 'predict.py') else
                        d9['cft'] if b.startswith('cft_') else d9['engine'])
        tab9 = '/kaggle/working/v4n_tables'
        shutil.copytree(os.path.dirname(find9('pool_fp.npy')), tab9, dirs_exist_ok=True)
        env9 = dict(os.environ, V4R_TABLES=tab9, V4R_SIM_OUT='/kaggle/working/v4n_sim',
                    NUMBA_CACHE_DIR='/kaggle/working/numba_cache')
        cmd9 = [sys.executable, '-u', f"{d9['sim']}/predict.py", '--test', test_parquet, '--tables', tab9,
                '--ckpt', find9('cft_hoR_a.pt'), '--ranker', find9('ranker_v0.pkl'),
                '--workers', str(max(1, min(4, os.cpu_count() or 1)))]
        reserve = 1800                                        # forward models + fusion after the engine
        try:
            p = subprocess.run(cmd9 + ['--out', '/kaggle/working/v4n_smoke.csv', '--workers', '1', '--limit', '3'],
                               env=env9, timeout=max(60, min(1200, deadline_t - time.time() - reserve)))
            assert p.returncode == 0, f'v4n smoke rc {p.returncode}'
            p = subprocess.run(cmd9 + ['--out', '/kaggle/working/v4n_submission.csv'], env=env9,
                               timeout=max(60, min(5 * 3600, deadline_t - time.time() - reserve)))
            assert p.returncode == 0, f'v4n rc {p.returncode}'
        finally:
            shutil.rmtree(tab9, ignore_errors=True)

        # ---- inputs
        e7 = pd.read_csv('e7_submission.csv', dtype=str, keep_default_na=False)
        e7_rows = list(zip(e7.molecule_id.astype(str), e7.smiles))
        v4s = pd.read_csv('/kaggle/working/v4n_submission.csv', dtype=str, keep_default_na=False)
        diag = pd.read_parquet('/kaggle/working/v4n_submission.diag.parquet')
        dg = {str(m): (r.get('keys'), r.get('n_engine')) for m, r in
              zip(diag.molecule_id.astype(str), diag.to_dict('records'))}
        v4 = {}
        for m, s in zip(v4s.molecule_id.astype(str), v4s.smiles):
            keys, n_eng = dg.get(m, (None, None))
            if n_eng is None or (isinstance(n_eng, float) and not math.isfinite(n_eng)):
                n_eng = 0                                     # unknown -> no evidence-backed entries
            smi = split_list(s)
            keys = list(keys) if keys is not None and len(keys) == len(smi) else None
            v4[m] = (smi, keys, int(n_eng))
        ids7 = {m for m, _ in e7_rows}
        print(f'E11 inputs: E7 {len(e7_rows)} rows, rebuilt {len(v4)} rows, shared {len(ids7 & set(v4))}', flush=True)

        # ---- keys: E7's own lists first, then the engine dump; RDKit only for what is left
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        _taut = rdMolStandardize.TautomerEnumerator()

        def score_key(s):                                     # = engine/chem.score_key
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            ik = Chem.MolToInchiKey(_taut.Canonicalize(m))
            return ik[:14] if ik else None

        key_of = KeyCache(score_key)
        eng_full = {}
        try:
            eng_full = json.load(open('/kaggle/working/eng_lists.json'))
        except Exception as e:  # noqa: BLE001
            print('E11: eng_lists.json unreadable (lib = 0 everywhere):', repr(e), flush=True)
        for src in (g.get('ENG') or {}, eng_full):
            for e in src.values():
                if isinstance(e, dict):
                    key_of.add(e.get('smiles'), e.get('keys'))
        lib_by_mid = {}
        for m, e in eng_full.items():
            ks, lb = e.get('keys') or [], e.get('lib') or []
            if len(ks) == len(lb):
                lib_by_mid[str(m)] = {k: float(x) for k, x in zip(ks, lb) if k}

        # ---- forward models: E7's code, written by E7's forward-model cell
        e3 = g.get('E3_DIR', '/kaggle/working/e3code')
        if e3 not in sys.path:
            sys.path.insert(0, e3)
        rerank_fn = formula_of = fm_fn = None
        lams = {'ice': float(g.get('LAM_ICE', 0.5)), 'gl': float(g.get('LAM_GL', 0.5))}
        if g.get('FM_ENABLE', True):
            from fm_rerank import rerank_molecule as rerank_fn
            from rdkit.Chem.rdMolDescriptors import CalcMolFormula

            def formula_of(s):
                m = Chem.MolFromSmiles(s)
                return CalcMolFormula(m) if m is not None else None
            fm_fn = make_fm_score_fn(load_fm_cache(), test_parquet, '/tmp/fm_work_e11', lams,
                                     budget_cap=float(g.get('FM_BUDGET_SEC', 5400)), reserve_sec=600)

        rows, stats = fuse_all(e7_rows, v4, key_of, deadline_t, formula_of=formula_of,
                               lib_of=lambda m: lib_by_mid.get(m, {}), fm_score_fn=fm_fn, rerank_fn=rerank_fn,
                               lams=lams)
        sub = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
        sub.to_csv('submission.csv', index=False)
        stats.update(status='ok', keys_computed=key_of.n_computed, sec=round(time.time() - t_cell, 1))
    except Exception as e:  # noqa: BLE001
        shutil.copy('e7_submission.csv', 'submission.csv')
        stats = dict(status='FAILED', error=repr(e)[:500], sec=round(time.time() - t_cell, 1))
        print('E11 FAILED, submission = E7 unchanged:', repr(e), flush=True)
    finally:
        shutil.rmtree('/kaggle/working/v4n_code', ignore_errors=True)
    try:
        json.dump(stats, open('/kaggle/working/e11_stats.json', 'w'), indent=1, default=str)
    except Exception:  # noqa: BLE001
        pass
    print('E11_STATS ' + json.dumps(stats, default=str), flush=True)


if __name__ == '__main__':          # true when run as a notebook cell; false when imported by the tests
    try:
        _e11_main()
    except BaseException as _e:      # noqa: BLE001 - the cell never raises
        try:
            shutil.copy('e7_submission.csv', 'submission.csv')
        except Exception:  # noqa: BLE001
            pass
        print('E11 outer failure, submission = E7 unchanged:', repr(_e), flush=True)

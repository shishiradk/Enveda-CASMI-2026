"""Unit tests for the E11 fusion cell: pure python, toy inputs, no RDKit, no GPU, no files beyond the repo.

    python -m unittest research/kaggle_e1/e11_fusion/test_fusion_cell.py -v

E7's own fm_rerank.py is taken from the E7 notebook (casmi_e7.ipynb, the E3_FILES literal of the forward-model
cell), so the re-rank under test is exactly the code E7 ships.
"""
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
E7_NB = HERE.parent / 'e7' / 'casmi_e7.ipynb'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fc = _load('e11_fusion_cell', HERE / 'fusion_cell.py')


def _e7_fm_rerank():
    nb = json.load(open(E7_NB))
    for c in nb['cells']:
        src = ''.join(c['source'])
        if 'E3_FILES = {' not in src:
            continue
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'E3_FILES':
                files = ast.literal_eval(node.value)
                d = Path(tempfile.mkdtemp())
                (d / 'fm_rerank.py').write_text(files['fm_rerank.py'])
                return _load('fm_rerank', d / 'fm_rerank.py')
    raise RuntimeError('fm_rerank.py not found in the E7 notebook')


fm_rerank = _e7_fm_rerank()


def key_of_toy(s):
    """Toy metric key: lower-case, '~' suffix marks a tautomer spelling of the same key; 'BAD' has no key."""
    if s.startswith('BAD'):
        return None
    return s.split('~')[0].lower()


def e7_list(prefix, n):
    return [f'{prefix}{i}' for i in range(n)]


def row(lst):
    return ';'.join(lst)


def never(*_):
    return 0.0


FAR = 1e18


class TestHelpers(unittest.TestCase):
    def test_split_list(self):
        self.assertEqual(fc.split_list('a;b;;c; '), ['a', 'b', 'c'])
        self.assertEqual(fc.split_list(float('nan')), [])
        self.assertEqual(fc.split_list(None), [])

    def test_key_cache_computes_once(self):
        calls = []

        def comp(s):
            calls.append(s)
            return s.lower()
        kc = fc.KeyCache(comp, prefill={'P': 'pre'})
        kc.add(['Q', 'R'], ['q-key', None])
        for _ in range(3):
            self.assertEqual(kc('X'), 'x')
            self.assertEqual(kc('P'), 'pre')
            self.assertEqual(kc('Q'), 'q-key')
        self.assertEqual(kc('R'), 'r')                 # a None key in add() is not cached
        self.assertEqual(calls, ['X', 'R'])
        self.assertEqual(kc.n_computed, 2)

    def test_key_cache_exception_is_none(self):
        kc = fc.KeyCache(lambda s: 1 / 0)
        self.assertIsNone(kc('A'))

    def test_rrf_math_and_tie_order(self):
        f = fc.rrf_fuse(['a', 'b', 'c'], ['b', 'x'], key_of_toy)
        sc = {e['key']: e['score'] for e in f}
        self.assertAlmostEqual(sc['a'], 1 / 3)
        self.assertAlmostEqual(sc['b'], 1 / 4 + 0.6 / 3)
        self.assertAlmostEqual(sc['x'], 0.6 / 4)
        self.assertEqual([e['key'] for e in f], ['b', 'a', 'c', 'x'])
        # exact tie: E7-only rank r vs rebuilt-only rank r' with 1/(3+r) == 0.6/(3+r'): r=2 (0.2), r'=0 (0.2)
        f = fc.rrf_fuse(['a', 'b', 'c'], ['y'], key_of_toy)
        self.assertEqual([e['key'] for e in f], ['a', 'b', 'c', 'y'])   # E7 inserted first wins the tie

    def test_rrf_keeps_40(self):
        f = fc.rrf_fuse(e7_list('e', 30), e7_list('v', 30), key_of_toy)
        self.assertEqual(len(f), 40)


class TestFuseAll(unittest.TestCase):
    def run_fuse(self, e7_rows, v4, **kw):
        kw.setdefault('deadline_t', FAR)
        return fc.fuse_all(e7_rows, v4, fc.KeyCache(key_of_toy), **kw)

    def test_padding_dropped(self):
        l7 = e7_list('e', 25)
        smi4 = ['n0', 'n1', 'pad0', 'pad1', 'pad2']
        rows, st = self.run_fuse([('m1', row(l7))], {'m1': (smi4, None, 2)})
        out = rows[0][1].split(';')
        self.assertIn('n0', out)
        self.assertIn('n1', out)
        self.assertFalse(any(s.startswith('pad') for s in out))
        self.assertEqual(st['v4n_dropped_padding'], 3)
        self.assertEqual(len(out), 25)

    def test_padding_only_gives_e7_order(self):
        l7 = e7_list('e', 25)
        rows, _ = self.run_fuse([('m1', row(l7))], {'m1': (['pad0', 'pad1'], None, 0)})
        self.assertEqual(rows[0][1], row(l7))

    def test_molecule_missing_from_rebuilt(self):
        l7 = e7_list('e', 25)
        rows, st = self.run_fuse([('m1', row(l7)), ('m2', row(e7_list('f', 25)))], {'m1': (['n0'], None, 1)})
        self.assertEqual(rows[1], ('m2', row(e7_list('f', 25))))
        self.assertEqual(st['e7_kept_no_v4n'], 1)

    def test_short_union_keeps_e7(self):
        l7 = e7_list('e', 10)
        rows, st = self.run_fuse([('m1', row(l7))], {'m1': (e7_list('n', 5), None, 5)})
        self.assertEqual(rows[0][1], row(l7))
        self.assertEqual(st['e7_kept_short'], 1)

    def test_short_lists_completed_by_rebuilt(self):
        l7 = e7_list('e', 20)
        rows, st = self.run_fuse([('m1', row(l7))], {'m1': (e7_list('n', 10), None, 10)})
        out = rows[0][1].split(';')
        self.assertEqual(len(out), 25)
        self.assertEqual(len(set(out)), 25)
        self.assertEqual(st['fused'], 1)

    def test_duplicate_keys(self):
        # E7 lists 'e1' twice under two spellings; the rebuilt engine lists e1, e2 under other spellings and a BAD SMILES
        l7 = ['e0', 'e1', 'e1~t', 'e2'] + e7_list('x', 22)
        smi4 = ['e1~u', 'e2~v', 'BAD', 'n0']
        rows, _ = self.run_fuse([('m1', row(l7))], {'m1': (smi4, None, 4)})
        out = rows[0][1].split(';')
        keys = [key_of_toy(s) for s in out]
        self.assertEqual(len(keys), 25)
        self.assertEqual(len(set(keys)), 25)
        self.assertIn('e1', out)                     # E7's spelling is written for a shared key
        self.assertNotIn('e1~t', out)
        self.assertNotIn('e1~u', out)
        self.assertNotIn('BAD', out)
        self.assertEqual(out[0], 'e1')               # 1/4 + 0.6/3 > 1/3

    def test_diag_keys_reused(self):
        calls = []

        def comp(s):
            calls.append(s)
            return key_of_toy(s)
        l7 = e7_list('e', 25)
        kc = fc.KeyCache(comp, prefill={s: s for s in l7})
        fc.fuse_all([('m1', row(l7))], {'m1': (['n0', 'n1'], ['kn0', 'kn1'], 2)}, kc, FAR)
        self.assertEqual(calls, [])

    def test_exception_fallback_per_molecule(self):
        l7a, l7b = e7_list('a', 25), e7_list('b', 25)
        v4 = {'m1': (['n0'], None, 'not-a-number'), 'm2': (['n1'], None, 1)}
        rows, st = self.run_fuse([('m1', row(l7a)), ('m2', row(l7b))], v4)
        self.assertEqual(rows[0], ('m1', row(l7a)))
        self.assertIn('n1', rows[1][1].split(';'))
        self.assertEqual(st['e7_kept_error'], 1)

    def test_exception_in_rerank_falls_back_for_that_molecule(self):
        l7 = e7_list('e', 25)
        v4 = {'m1': (['n0'], None, 1), 'm2': (['n1'], None, 1)}

        def rerank_boom(formulas, *a):
            raise RuntimeError('boom')
        rows, st = self.run_fuse([('m1', row(l7)), ('m2', row(l7))], v4,
                                 formula_of=lambda s: 'F', lib_of=lambda m: {},
                                 fm_score_fn=lambda req, d: {'gl': {m: {s: float(i) for i, s in enumerate(v)}
                                                                    for m, v in req.items()}},
                                 rerank_fn=rerank_boom, lams={'gl': 0.5})
        self.assertEqual(rows, [('m1', row(l7)), ('m2', row(l7))])
        self.assertEqual(st['e7_kept_error'], 2)

    def test_exception_outside_loop_propagates(self):
        l7 = e7_list('e', 25)

        def fm_boom(req, d):
            raise RuntimeError('runner')
        with self.assertRaises(RuntimeError):
            self.run_fuse([('m1', row(l7))], {'m1': (['n0'], None, 1)}, formula_of=lambda s: 'F',
                          lib_of=lambda m: {}, fm_score_fn=fm_boom, rerank_fn=fm_rerank.rerank_molecule,
                          lams={'gl': 0.5})

    def test_deadline_fallback(self):
        mols = [(f'm{i}', row(e7_list(f'e{i}_', 25))) for i in range(6)]
        v4 = {m: ([f'n_{m}'], None, 1) for m, _ in mols}
        t = {'now': 0.0}

        def clock():
            t['now'] += 1.0
            return t['now']
        # phase A reads the clock once per molecule (1..6), phase C once per fused molecule (7..)
        rows, st = fc.fuse_all(mols, v4, fc.KeyCache(key_of_toy), deadline_t=4.0, clock=clock)
        fused = [m for (m, s), (_, s7) in zip(rows, mols) if s != s7]
        self.assertEqual(fused, [])                    # the 3 fused in A hit the deadline in C
        self.assertEqual(rows, mols)
        self.assertEqual(st['e7_kept_deadline'], 6)
        t['now'] = 0.0
        rows, st = fc.fuse_all(mols, v4, fc.KeyCache(key_of_toy), deadline_t=8.5, clock=clock)
        changed = [m for (m, s), (_, s7) in zip(rows, mols) if s != s7]
        self.assertEqual(changed, ['m0', 'm1'])         # A: m0..m5 at t=1..6; C: m0 t=7, m1 t=8, then past 8.5
        self.assertEqual(st['e7_kept_deadline'], 4)


class TestForwardModelStep(unittest.TestCase):
    """E7's rerank_molecule on a mixed formula group, a pure-E7 group and a library-protected candidate."""

    def setUp(self):
        # E7: e0 (protected lib 0.95) e1 e2 ... ; rebuilt: n0 only.  Formulas: e0, e2, n0 share 'F1'; e3, e4 share 'F2'.
        self.l7 = e7_list('e', 25)
        self.form = {'e0': 'F1', 'e2': 'F1', 'n0': 'F1', 'e3': 'F2', 'e4': 'F2'}
        self.lib = {'e0': 0.95}
        self.v4 = {'m1': (['n0'], None, 1)}

    def fm_scores(self, req, deadline):
        scores = {'e2': 0.0, 'n0': 1.0, 'e0': 9.0, 'e3': 0.0, 'e4': 9.0}
        self.req = req
        return {'gl': {m: {s: scores[s] for s in v} for m, v in req.items()}}

    def run_fm(self):
        return fc.fuse_all([('m1', row(self.l7))], self.v4, fc.KeyCache(key_of_toy), FAR,
                           formula_of=lambda s: self.form.get(s, 'U_' + s), lib_of=lambda m: self.lib,
                           fm_score_fn=self.fm_scores, rerank_fn=fm_rerank.rerank_molecule, lams={'gl': 5.0})

    def test_mixed_group_reranked_protected_and_e7_groups_untouched(self):
        base, _ = fc.fuse_all([('m1', row(self.l7))], self.v4, fc.KeyCache(key_of_toy), FAR)
        base = base[0][1].split(';')
        rows, st = self.run_fm()
        out = rows[0][1].split(';')
        self.assertEqual(self.req, {'m1': ['e2', 'n0']})   # only the mixed group, protected e0 never sent
        self.assertEqual(out[0], 'e0')                      # protected keeps its slot
        self.assertEqual(out.index('n0'), base.index('e2'))  # n0 (fm 1.0, lam 5) takes e2's slot
        self.assertEqual(out.index('e2'), base.index('n0'))
        self.assertEqual(out.index('e3'), base.index('e3'))  # pure-E7 group F2 left as E7 ordered it
        self.assertEqual(out.index('e4'), base.index('e4'))
        self.assertEqual(st['fm_molecules_reranked'], 1)
        self.assertEqual(len(set(out)), 25)

    def test_no_scores_keeps_fused_order(self):
        base, _ = fc.fuse_all([('m1', row(self.l7))], self.v4, fc.KeyCache(key_of_toy), FAR)
        rows, _ = fc.fuse_all([('m1', row(self.l7))], self.v4, fc.KeyCache(key_of_toy), FAR,
                              formula_of=lambda s: self.form.get(s, 'U_' + s), lib_of=lambda m: self.lib,
                              fm_score_fn=lambda req, d: {}, rerank_fn=fm_rerank.rerank_molecule, lams={'gl': 5.0})
        self.assertEqual(rows, base)


class TestValidate(unittest.TestCase):
    def test_rejects_bad_rows(self):
        e7 = [('m1', 'a;b')]
        with self.assertRaises(AssertionError):
            fc.validate_submission([('m1', 'a;a')], e7)
        with self.assertRaises(AssertionError):
            fc.validate_submission([('m1', '')], e7)
        with self.assertRaises(AssertionError):
            fc.validate_submission([('m2', 'a')], e7)
        with self.assertRaises(AssertionError):
            fc.validate_submission([('m1', ';'.join(e7_list('x', 26)))], e7)
        fc.validate_submission([('m1', 'a;b')], e7)

    def test_cut_asserts(self):
        f = fc.rrf_fuse(e7_list('e', 24), [], key_of_toy)
        with self.assertRaises(AssertionError):
            fc.cut(f)


if __name__ == '__main__':
    sys.exit(unittest.main())

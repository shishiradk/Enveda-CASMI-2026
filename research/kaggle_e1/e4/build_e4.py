"""Build E4 = E3b + PubChem popularity prior (research/analysis/e4_popularity.md).

    python research/kaggle_e1/e4/build_e4.py [--smoke N] [--budget SEC] [--mu MU] [--score logit|rank]

Reads research/kaggle_e1/e3b/casmi_e3b.ipynb (the pushed E3b notebook; not modified) and patches it by asserted text
replacements; embeds e4/pop_stage.py and e4/e4_stage.py.  Writes e4/casmi_e4.ipynb and e4/kernel-metadata.json
(kernel shishiradhikari11/casmi-e4-eligible-pop, private, T4, internet off, E3b's datasets + casmi-pop-lookup).

Cells: 1 setup (E3b's + the E4 switches) | 2 engine (E3b's; the runner also exports the two raw ranker probabilities;
a copy of the engine lists is kept) | 3 forward-model stage (E3b's cell, unchanged: ENG becomes the E3b lists) |
4 popularity prior + replay of E3b's rule on the adjusted lists (any failure keeps the E3b lists) | 5 submission.csv
with validity checks and the comparison against E1 and E3b.
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3B = HERE.parent / "e3b"


def arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SMOKE, BUDGET, MU, SCORE = arg("--smoke", 0, int), arg("--budget", 5400, int), arg("--mu", 0.25, float), arg("--score", "logit", str)
assert SCORE in ("logit", "rank")


def sub1(text, old, new):
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)


e3b = json.load(open(E3B / "casmi_e3b.ipynb", encoding="utf-8"))
setup, eng, fm, _write = ("".join(c["source"]) for c in e3b["cells"])
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1", "casmi26-iceberg", "casmi26-glacier", "dmitriigluzdov",
            "popularity-prior", "pool_lsid"):
    assert bad not in setup + eng + fm, bad

# ---- cell 1: E3b setup + E4 switches --------------------------------------------------------------------
setup = sub1(setup, "# E3 = E1 (prize-eligible base) + forward-model re-scoring.",
             "# E4 = E3b (E1 prize-eligible base + forward-model re-ordering with library protection) + PubChem popularity prior.")
setup = sub1(setup, "SMOKE_N = 0 ", f"SMOKE_N = {SMOKE} ")
setup = sub1(setup, "FM_BUDGET_SEC = 5400 ", f"FM_BUDGET_SEC = {BUDGET} ")
SETUP = setup + f'''
# E4 addition: shishiradhikari11/casmi-pop-lookup (our table of PubChem substance / PubMed counts per pool structure, built from
# PubChem's public CID-SID and CID-PMID files; PubChem data are in the public domain).
POP_ENABLE = True
POP_MU = {MU!r}               # f = z(ranker) + POP_MU * (log1p(n_sid) + log1p(n_pmid)); 0 = E3b exactly
POP_SCORE = {SCORE!r}         # 'logit': z of the isotonic logit blend of the two raw ranker probabilities; 'rank': z of the rank blend
print('E4 switches: POP_ENABLE', POP_ENABLE, '| POP_MU', POP_MU, '| POP_SCORE', POP_SCORE)
'''

# ---- cell 2: engine; the runner also exports pv / ours -----------------------------------------------------
tree = ast.parse(eng)
node = [n for n in tree.body if (ast.get_source_segment(eng, n) or "").startswith("open(os.path.join(ENG_DIR, 'eng_runner.py')")]
assert len(node) == 1
seg = ast.get_source_segment(eng, node[0])
runner = node[0].value.args[0].value
assert isinstance(runner, str) and eng.count(seg) == 1
runner = sub1(runner, "smis, keys, scs, libs, seen = [], [], [], [], set()", "smis, keys, scs, libs, seen = [], [], [], [], set()\n        pvs, ous = [], []")
runner = sub1(runner, "scs.append(float(bl[mid][order[j]]))",
              "scs.append(float(bl[mid][order[j]]))\n            pvs.append(float(S['pv'][mid][order[j]])); ous.append(float(S['ours'][mid][order[j]]))")
runner = sub1(runner, "res[str(mid)] = dict(smiles=smis, keys=keys, scores=scs, lib=libs,",
              "res[str(mid)] = dict(smiles=smis, keys=keys, scores=scs, lib=libs, pv=pvs, ours=ous,")
runner = sub1(runner, "E3 version of the E1 runner.", "E4 version of the E1 runner (E3b's, plus pv / ours = the two raw ranker probabilities of each kept candidate).")
compile(runner, "eng_runner.py", "exec")
ENG = eng.replace(seg, "open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write(" + repr(runner) + ")")
ENG += "\nimport copy as _copy\nENG_RAW = _copy.deepcopy(ENG)      # E4: the engine lists as written, before any re-ordering\n"
ast.parse(ENG)

# ---- cell 3: E3b's forward-model cell, unchanged -----------------------------------------------------------
FM = fm

# ---- cell 4: popularity prior + E3b's rule on the adjusted lists ---------------------------------------------
FILES = {n: (HERE / n).read_text(encoding="utf-8") for n in ("pop_stage.py", "e4_stage.py")}
for src in FILES.values():
    for bad in ("popularity_reorder", "pool_lsid", "pool_pop"):
        assert bad not in src, bad
POP = '''# E4: PubChem popularity prior on the engine lists, then E3b's rule again on the adjusted lists (forward scores re-used).
# Library-protected candidates (own library similarity >= FM_PROTECT_LIB) keep their slots.  Any failure keeps the E3b lists.
E4_FILES = ''' + repr(FILES) + '''
os.makedirs(E3_DIR, exist_ok=True)
for _n, _src in E4_FILES.items():
    open(os.path.join(E3_DIR, _n), 'w').write(_src)
ENG_E3B = ENG
E3B_TOP25 = {m: list(dict.fromkeys(e.get('smiles', [])))[:25] for m, e in ENG_E3B.items()}
POP_STATS = {'status': 'not run'}
if POP_ENABLE and POP_MU > 0 and len(ENG_RAW):
    try:
        _t0 = time.time()
        if E3_DIR not in sys.path:
            sys.path.insert(0, E3_DIR)
        import pop_stage, e4_stage
        _lkp = pop_stage.find_lookup('/kaggle/input')
        _lk = pop_stage.load_lookup(_lkp)
        print('popularity lookup', _lkp, len(_lk), 'keys')
        _fm_ok = FM_STATS.get('status') == 'ok'
        _new, _st = e4_stage.run_e4(ENG_RAW, ENG_E3B if _fm_ok else ENG_RAW, FM_STATS if _fm_ok else {}, _lk,
                                    os.path.join(COMP, 'test.parquet'), '/tmp/fm_work', mu=POP_MU, topn=FM_TOPN,
                                    lam_ice=LAM_ICE, lam_gl=LAM_GL, protect_lib=FM_PROTECT_LIB, score_mode=POP_SCORE)
        assert set(_new) == set(ENG_RAW), 'molecule set changed'
        for _m, _e in ENG_RAW.items():
            assert sorted(_new[_m]['keys']) == sorted(_e['keys']) and len(_new[_m]['smiles']) == len(_e['smiles']), _m
            assert len(set(_new[_m]['keys'])) == len(_new[_m]['keys']), _m
        ENG = _new
        POP_STATS = dict(_st, status='ok', total_sec=round(time.time() - _t0, 1))
    except Exception as e:
        POP_STATS = {'status': 'FAILED', 'error': repr(e)}
        print('!' * 100)
        print('POPULARITY STAGE FAILED -> E3B LISTS KEPT UNCHANGED:', repr(e))
        traceback.print_exc()
        print('!' * 100)
assert rdkit.__version__ == _rd_before, 'the notebook RDKit changed'
json.dump(POP_STATS, open('/kaggle/working/pop_stats.json', 'w'), indent=1)
json.dump(ENG, open('/kaggle/working/e4_lists.json', 'w'))
print(f'elapsed {time.time()-T0:.0f}s')
print('POP_STATS ' + json.dumps(POP_STATS))
'''

# ---- cell 5: submission ----------------------------------------------------------------------------------
WRITE = '''sub = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
assert len(ENG) > 0, 'engine produced no lists'
out, n_empty = [], 0
cmp = {'E1': [E1_TOP25, 0, 0, 0, []], 'E3b': [E3B_TOP25, 0, 0, 0, []]}
for mid in sub.molecule_id:
    e = ENG.get(str(mid)) or {}
    sm = list(dict.fromkeys(e.get('smiles', [])))[:25]
    for _c in cmp.values():
        old = _c[0].get(str(mid), [])
        _c[1] += int(sm[:1] != old[:1]); _c[2] += int(sm != old); _c[3] += int(set(sm) != set(old))
        if sm[:1] != old[:1]:
            _c[4].append(str(mid))
    n_empty += int(not sm)
    out.append((mid, ';'.join(sm) if sm else 'CCO'))
fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
assert fs.molecule_id.is_unique and fs.smiles.notna().all() and len(fs) == len(sub) and list(fs.molecule_id) == list(sub.molecule_id)
_n = fs.smiles.str.split(';').map(len); _u = fs.smiles.str.split(';').map(lambda v: len(set(v)))
assert (_n <= 25).all() and (_n == _u).all() and (fs.smiles.str.len() > 0).all() and not fs.smiles.str.contains(';;').any()
fs.to_csv('submission.csv', index=False)
for _name, _ref in (('e1_order_reference.csv', E1_TOP25), ('e3b_order_reference.csv', E3B_TOP25)):
    pd.DataFrame([(m, ';'.join(_ref.get(str(m), [])) or 'CCO') for m in sub.molecule_id],
                 columns=['molecule_id', 'smiles']).to_csv(_name, index=False)
print('E4 submission written:', len(fs), 'rows | without a list:', n_empty, '| SMILES per row min/median/max:',
      int(_n.min()), int(_n.median()), int(_n.max()), '| all unique within row:', bool((_n == _u).all()))
for _k, _c in cmp.items():
    print('E4 vs', _k, 'submission rows: top-1 changed', _c[1], '| top-25 order changed', _c[2], '| top-25 membership changed', _c[3])
    if _c[4]:
        print('   top-1 changed vs', _k, ':', _c[4][:60])
print('forward stage', FM_STATS.get('status'), '| popularity stage', POP_STATS.get('status'), f'| {time.time()-T0:.0f}s')
'''
for name, src in (("SETUP", SETUP), ("FM", FM), ("POP", POP), ("WRITE", WRITE)):
    ast.parse(src)


def cell(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


nb = {"cells": [cell(SETUP), cell(ENG), cell(FM), cell(POP), cell(WRITE)], "metadata": e3b["metadata"], "nbformat": 4, "nbformat_minor": 5}
(HERE / "casmi_e4.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = json.load(open(E3B / "kernel-metadata.json"))
meta.update(id="shishiradhikari11/casmi-e4-eligible-pop", title="CASMI E4 eligible pop", code_file="casmi_e4.ipynb", is_private=True,
            enable_internet=False, dataset_sources=list(meta["dataset_sources"]) + ["shishiradhikari11/casmi-pop-lookup"])
assert meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4" and len(meta["dataset_sources"]) == 8
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote E4 | smoke", SMOKE, "| budget", BUDGET, "| mu", MU, "| score", SCORE, "| cells", [len(x) for x in (SETUP, ENG, FM, POP, WRITE)])

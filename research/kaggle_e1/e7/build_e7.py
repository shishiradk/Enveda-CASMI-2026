"""Build E7 = E6 + Class-3 channel (rr_bt at ranks 4-8), NOT pushed.

    python research/kaggle_e1/e7/build_e7.py [--no-assets]

Rule (DEC-010, research/analysis/c3gen_tournament.md): final list = E6 list ranks 1-3 unchanged; ranks 4-8 = the first
5 candidates of rr_bt (round-robin union of biotransform and mmp_edit, biotransform first) whose metric key is not
already in the E6 top-25; then E6 ranks 4+; de-dup on the metric key; 25 per row. Ungated. Any failure of the C3
channel (exception, timeout, missing context) keeps the E6 list for that molecule / all molecules.

Cells: 0 = E6 setup (header updated) | 1 = E6 engine cell, its eng_runner.py additionally exports the engine's library
analogs (casmi_engine.top_analogs output) to /kaggle/working/eng_ana.json (eng_lists.json unchanged) | 2 = E6 forward
model stage (unchanged) | 3 = E6 PubChem channel + merge, run through e7_e6_channel.py (same E6 lists + --dump-ctx
/kaggle/working/e6_ctx.pkl; falls back to e6_channel.py when the E7 dataset is missing) | 4 = NEW: E7 C3 channel
(e7_c3_channel.py in a subprocess, time-budgeted) + E7 merge (e7_merge.py, inlined) | 5 = E6 submission cell, with
E7 names, e6_lists.json / e7_lists.json and the visible check (top-3 unchanged vs E6, rows with C3 insertions).
Reads e6/casmi_e6.ipynb, e6/kernel-metadata.json (not modified). Writes e7/casmi_e7.ipynb, e7/kernel-metadata.json
(kernel shishiradhikari11/casmi-e7-c3-channel, private, T4, E6 datasets + shishiradhikari11/casmi-e7-c3assets) and
e7/dataset/ (upload as casmi-e7-c3assets): code + c3_* assets + README + dataset-metadata.json.
"""
import ast, copy, json, shutil, sys
from pathlib import Path

import numpy as np

E7 = Path(__file__).resolve().parent
KE = E7.parent
ROOT = KE.parents[1]
D = E7 / "dataset"
NO_ASSETS = "--no-assets" in sys.argv

e6 = json.load(open(KE / "e6" / "casmi_e6.ipynb", encoding="utf-8"))
assert len(e6["cells"]) == 5
setup, eng, fm, e6cell, write = ("".join(c["source"]) for c in e6["cells"])
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1", "casmi26-iceberg", "casmi26-glacier"):
    assert bad not in setup + eng + fm + e6cell + write, bad

# ---------------------------------------------------------------- cell 0: header
H0 = "# E6 = E3b (prize-eligible base + forward-model re-scoring) + our PubChem channel, append-only."
assert setup.startswith(H0)
setup = setup.replace(H0, "# E7 = E6 (E3b + our PubChem channel, append-only) + our Class-3 channel (rr_bt) at ranks 4-8. "
                          "E7 input: shishiradhikari11/casmi-e7-c3assets\n# (our MIT code; assets derived from the "
                          "competition train structures and COCONUT, CC BY 4.0).", 1)

# ---------------------------------------------------------------- cell 1: engine runner exports its analogs
ANA_EXPORT = '''    json.dump(res, open(out, 'w'))
    try:   # E7: the engine's library analogs (top_analogs output) for the Class-3 channel; eng_lists.json is unchanged
        ana = {}
        for r in recs:
            if r.get('ana') is None or 'target' not in r:
                continue
            ana[str(r['mid'])] = dict(target=float(r['target']), adducts=[str(x) for x in r.get('adducts', [])],
                                      ana=[[str(E.POOL['smiles'][p]), str(E.POOL['keys'][p]), float(s), float(E.POOL['mass'][p])]
                                           for p, s in list(r['ana'])[:100]])
        json.dump(ana, open(os.path.join(os.path.dirname(os.path.abspath(out)), 'eng_ana.json'), 'w'))
        print('engine analogs exported', len(ana), flush=True)
    except Exception as e:
        print('ENGINE ANALOG EXPORT FAILED (E7 C3 channel will be skipped):', repr(e), flush=True)
'''
tree = ast.parse(eng)
lit = [n.args[0] for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "write"
       and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
       and n.args[0].value.startswith('"""Our engine')]
assert len(lit) == 1
runner = lit[0].value
assert runner == (ROOT / "results/kaggle_e6/eng/eng_runner.py").read_text(encoding="utf-8")   # as shipped in E6
anchor = "    json.dump(res, open(out, 'w'))\n"
assert runner.count(anchor) == 1
runner7 = runner.replace(anchor, ANA_EXPORT, 1)
seg = ast.get_source_segment(eng, lit[0])
assert eng.count(seg) == 1
eng = eng.replace(seg, repr(runner7), 1)
assert eng.startswith("# Our engine")

# ---------------------------------------------------------------- cell 3: E6 channel via the E7 copy (ctx dump)
old = "    _py = find('e6_channel.py')\n"
new = ("    _py = find('e6_channel.py')\n"
       "    try:   # E7: same E6 lists + per-molecule target / zlog / PubChem window dump for the Class-3 channel\n"
       "        _py7 = find('e7_e6_channel.py')\n"
       "    except Exception as _e:\n"
       "        _py7 = None\n"
       "        print('E7 channel copy not found -> e6_channel.py without the ctx dump:', repr(_e))\n")
assert e6cell.count(old) == 1
e6cell = e6cell.replace(old, new, 1)
old = "    rr = subprocess.run([sys.executable, _py, os.path.join(COMP, 'test.parquet'), '/kaggle/working/e6_pc.json',"
assert e6cell.count(old) == 1
e6cell = e6cell.replace(old, "    rr = subprocess.run([sys.executable, _py7 or _py, os.path.join(COMP, 'test.parquet'), '/kaggle/working/e6_pc.json',", 1)
old = "'--budget', str(int(_budget))],\n"
assert e6cell.count(old) == 1
e6cell = e6cell.replace(old, "'--budget', str(int(_budget))]\n"
                             "                        + (['--dump-ctx', '/kaggle/working/e6_ctx.pkl'] if _py7 else []),\n", 1)

# ---------------------------------------------------------------- cell 4 (new): E7 C3 channel + merge
MERGE_SRC = (E7 / "e7_merge.py").read_text(encoding="utf-8")
E7CELL = '''# E7: our Class-3 channel (DEC-010) -> C3L: rr_bt = round-robin union of the biotransform and mmp_edit generators
# (shishiradhikari11/casmi-e7-c3assets), inserted at ranks 4-8 of the E6 lists; E6 ranks 1-3 untouched. Ungated.
# Any failure (exception, timeout, missing context) keeps the E6 list of that molecule / of all molecules.
E6_SNAP = {m: dict(smiles=list((v or {}).get('smiles', [])), keys=list((v or {}).get('keys', []))) for m, v in ENG.items()}
C3L, E7_STATS, E7_MERGE = {}, {}, {}
_c3out = '/kaggle/working/e7_c3.json'
try:
    _t0 = time.time()
    _c3py = find('e7_c3_channel.py')
    _left = 8.4 * 3600 - (time.time() - T0)
    _budget = int(min(2.5 * 3600, _left))
    if _budget < 600:
        raise RuntimeError(f'only {_left:.0f}s left for the C3 channel')
    for _f in ('/kaggle/working/e6_ctx.pkl', '/kaggle/working/eng_ana.json'):
        if not os.path.exists(_f):
            raise FileNotFoundError(_f)
    _cmd = [sys.executable, _c3py, os.path.join(COMP, 'test.parquet'), _c3out, '--ctx', '/kaggle/working/e6_ctx.pkl',
            '--ana', '/kaggle/working/eng_ana.json', '--assets', os.path.dirname(_c3py),
            '--coco', os.path.dirname(find('coco_fp.npy')), '--workers', str(os.cpu_count() or 4), '--budget', str(_budget)]
    try:
        rr = subprocess.run(_cmd, capture_output=True, text=True, timeout=_budget + 900)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    except subprocess.TimeoutExpired as _e:
        print('E7 C3 channel timed out -> only the complete lists written so far are used:', repr(_e))
    C3L = json.load(open(_c3out))
    E7_STATS = C3L.pop('_stats', {})
    C3L.pop('_info', None)
    print('E7 C3 lists', len(C3L), '|', E7_STATS, f'| {time.time() - _t0:.0f}s')
except Exception as e:
    C3L = {}
    print('E7 C3 CHANNEL FAILED -> E6 lists kept:', repr(e))
''' + MERGE_SRC + '''
try:
    ENG, E7_MERGE = e7_merge(ENG, C3L)
except Exception as e:
    print('E7 MERGE FAILED -> E6 lists kept:', repr(e))
print('E7 merge:', E7_MERGE, f'| {time.time() - T0:.0f}s')
'''

# ---------------------------------------------------------------- cell 5: submission + visible check
write = write.replace("print('E6 submission written:'", "print('E7 submission written:'", 1)
write = write.replace("print('E6 vs E1 submission rows:", "print('E7 vs E1 submission rows:", 1)
assert "E7 submission written" in write and "E7 vs E1" in write
tail = write[write.index("json.dump({m: dict(smiles="):]
write = write.replace(tail, '''try:   # diagnostics only: submission.csv is already written
    _L6 = {m: list(dict.fromkeys((E6_SNAP.get(m) or {}).get('smiles', [])))[:25] for m in map(str, sub.molecule_id)}
    _L7 = {m: list(dict.fromkeys((ENG.get(m) or {}).get('smiles', [])))[:25] for m in map(str, sub.molecule_id)}
    json.dump({m: dict(smiles=v) for m, v in _L6.items()}, open('e6_lists.json', 'w'))
    json.dump({m: dict(smiles=v) for m, v in _L7.items()}, open('e7_lists.json', 'w'))
    json.dump(dict(e6=E6_STATS, e7_channel=E7_STATS, e7_merge=E7_MERGE), open('e7_stats.json', 'w'))
    print('E7 visible check: top-3 unchanged vs E6', sum(_L7[m][:3] == _L6[m][:3] for m in _L7), '/', len(_L7),
          '| 25 unique per row', sum(len(v) == 25 == len(set(v)) for v in _L7.values()), '/', len(_L7),
          '| rows with C3 insertions', sum(set(_L7[m]) != set(_L6[m]) for m in _L7))
    print('e6_lists.json / e7_lists.json written | E6 stats', E6_STATS, '| E7 channel', E7_STATS, '| E7 merge', E7_MERGE)
except Exception as e:
    print('E7 diagnostics failed (submission unaffected):', repr(e))
''')

nb = copy.deepcopy(e6)
srcs = [setup, eng, fm, e6cell, E7CELL, write]
nb["cells"] = []
for s in srcs:
    c = copy.deepcopy(e6["cells"][0])
    c["source"] = s.splitlines(keepends=True)
    c["outputs"], c["execution_count"] = [], None
    nb["cells"].append(c)
for s in srcs:
    compile(s, "cell", "exec")
compile(runner7, "eng_runner", "exec")
json.dump(nb, open(E7 / "casmi_e7.ipynb", "w", encoding="utf-8"), indent=1)

meta = json.load(open(KE / "e6" / "kernel-metadata.json"))
meta.update(id="shishiradhikari11/casmi-e7-c3-channel", title="CASMI E7 C3 channel", code_file="casmi_e7.ipynb",
            is_private=True)
assert "shishiradhikari11/casmi-e6-pcnets" in meta["dataset_sources"]
meta["dataset_sources"] = meta["dataset_sources"] + ["shishiradhikari11/casmi-e7-c3assets"]
json.dump(meta, open(E7 / "kernel-metadata.json", "w"), indent=1)

# ---------------------------------------------------------------- dataset casmi-e7-c3assets
D.mkdir(exist_ok=True)
for f in ("e7_c3_channel.py", "e7_e6_channel.py", "e7_merge.py"):
    shutil.copy(E7 / f, D / f)
for f in ("assets.py", "biotransform.py", "mmp_edit.py"):
    shutil.copy(ROOT / "research" / "c3gen" / f, D / f)
FULL = ROOT / "results" / "train_pkg" / "data_full"     # full pool: COCONUT + all train structures, no held keys
ASSETS = {"c3_pool_fp.npy": FULL / "pool_fp.npy", "c3_pool_mass.npy": FULL / "pool_mass.npy",
          "c3_pool_key.npy": FULL / "pool_key.npy", "c3_pool_smiles.txt": FULL / "pool_smiles.txt",
          "c3_fp_bits.npy": FULL / "fp_bits.npy", "c3_mmp_rules.parquet": ROOT / "results" / "c3gen" / "mmp_rules.parquet"}
if not NO_ASSETS:
    for dst, src in ASSETS.items():
        if not (D / dst).exists() or (D / dst).stat().st_size != src.stat().st_size:
            shutil.copy(src, D / dst)
    if not (D / "c3_pool_np.npy").exists():
        import pyarrow.parquet as pq
        pk = np.load(FULL / "pool_key.npy")
        npk = np.unique(np.asarray(pq.read_table(ROOT / "results/c3gen/np_pool.parquet", columns=["ik14"])
                                   .column("ik14").to_pylist()).astype("S14"))
        assert np.isin(npk, pk).all()          # every np_pool structure is in the full pool (mmp_edit 'known' = pool keys)
        np.save(D / "c3_pool_np.npy", np.isin(pk, npk))
json.dump({"title": "casmi-e7-c3assets", "id": "shishiradhikari11/casmi-e7-c3assets",
           "licenses": [{"name": "CC-BY-4.0"}],
           "description": "E7 Class-3 channel for CASMI 2026: our generator code (MIT) and structure assets derived "
                          "from the competition train structures and COCONUT (CC BY 4.0). See README.md."},
          open(D / "dataset-metadata.json", "w"), indent=1)
(D / "README.md").write_text("""# casmi-e7-c3assets (CASMI 2026, team shishiradhikari11)

Class-3 candidate generation for the E7 notebook (`shishiradhikari11/casmi-e7-c3-channel`): per test molecule, two
generators edit known structures toward the measured mass and the merged list (`rr_bt`) is inserted at ranks 4-8 of
our E6 lists.

## Code (MIT licence, written by our team)
- `e7_c3_channel.py` - the channel (context builder, both generators, round-robin union).
- `biotransform.py` - hand-written biotransformation rules (glycosylation, acylation, methylation, oxidation, ...).
- `mmp_edit.py` - single-cut matched-molecular-pair edits with rules mined by us.
- `assets.py` - shared fragment/rule helpers.
- `e7_e6_channel.py` - copy of our E6 PubChem channel (`casmi-e6-pcnets/e6_channel.py`) plus a context dump.
- `e7_merge.py` - the E7 list merge.
Dependencies: RDKit (BSD-3-Clause), NumPy, pandas, pyarrow.

## Data assets (CC BY 4.0, because they contain COCONUT-derived structures)
- `c3_pool_{fp,mass,key}.npy`, `c3_pool_smiles.txt` - 712,199 structures: 436,389 COCONUT natural products
  (COCONUT, https://coconut.naturalproducts.net, CC BY 4.0; obtained via prvsiyan/coconut-casmi26-candidates) and
  275,810 structures of the competition `train.parquet`. Mass-sorted; 6,930-bit packed fingerprints (ECFP4 | ECFP6 |
  RDKit FP | MACCS, prvsiyan's bit selection `c3_fp_bits.npy`), computed by us with RDKit.
- `c3_pool_np.npy` - boolean mask: natural-product-like pool rows (COCONUT, plus train structures with RDKit Contrib
  NP-likeness > 0, or > -0.5 when found in a natural-product library).
- `c3_mmp_rules.parquet` - matched-molecular-pair transformation rules mined by us from train and COCONUT structures
  (used with support >= 3).
No NIST data, no non-commercial datasets and no third-party model weights are included. Competition train structures
are used as the host allows for derived artifacts; please attribute COCONUT (Sorokina et al., J. Cheminform. 2021) when
reusing the structure files.
""", encoding="utf-8")
print("wrote", E7 / "casmi_e7.ipynb", "| datasets:", meta["dataset_sources"], "| dataset dir", D,
      "| files", sorted(p.name for p in D.iterdir()))

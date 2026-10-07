"""Build E5 = E3b + our V2 engine, fused only where the engine has no strong library answer (NOT pushed).

    python research/kaggle_e1/e5/build_e5.py

Rule (chosen on the calibrated proxy, research/scripts/e5_gate_eval.py, results/c3/e5_gate.json):
  g = library similarity of the current top-1 candidate (ENG[mid]['lib'][0], kept aligned by the E3b stage)
  g >= GATE_LIB (0.7)  -> E3b list unchanged
  g <  GATE_LIB        -> weighted RRF of the E3b list (weight 1) and the V2 list (weight BETA = 2.0), K = 3, top 40
Bench (honest rankers): proxy 0.350 -> 0.370, out-of-sample +0.017 [+0.012, +0.021]; S2 (≈0 % of the test) -0.06.

Cells: 0-2 = E3b's cells 0-2 unchanged (setup, engine, forward-model stage) | 3 = V2 engine + gated fusion (E2's
cell 2 with the gate) | 4 = E3b's submission cell, plus e5_lists.json (lists + gate decisions) for the visible check.
Reads e3b/casmi_e3b.ipynb, e3b/kernel-metadata.json, e2/kernel-metadata.json (not modified).
Writes e5/casmi_e5.ipynb and e5/kernel-metadata.json (kernel shishiradhikari11/casmi-e5-eligible-gated, private, T4).
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "e5"
GATE_LIB, BETA, KRR = 0.7, 2.0, 3.0

e3b = json.load(open(HERE / "e3b" / "casmi_e3b.ipynb", encoding="utf-8"))
assert len(e3b["cells"]) == 4
setup, eng, fm, write = ("".join(c["source"]) for c in e3b["cells"])
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1", "casmi26-iceberg", "casmi26-glacier"):
    assert bad not in setup + eng + fm, bad

# Kaggle's image moved to Python 3.13 and casmi-fm-runner has no cp313 RDKit 2025.3 wheel (E5 v1: forward stage
# FAILED). The wheel now comes from shishiradhikari11/casmi-rdkit2025-cp313; search every attached input for it.
_old = 'glob.glob(os.path.join(root, "**", "rdkit-2025.3.*.whl"), recursive=True)'
_new = ('(glob.glob(os.path.join(root, "**", "rdkit-2025.3.*.whl"), recursive=True) + '
        'glob.glob("/kaggle/input/**/rdkit-2025.3.*-" + tag + "-*.whl", recursive=True))')
assert fm.count(_old) == 1
fm = fm.replace(_old, _new)

FUSE = f'''# E5: our V2 engine (kNN fingerprint + COCONUT/train + PubChem 5 ppm windows) -> OURS, fused into ENG only where the
# current top-1 candidate has no strong library match (lib < GATE_LIB). Any failure keeps the E3b lists.
OURS, GATE_LIB, BETA, KRR = {{}}, {GATE_LIB}, {BETA}, {KRR}
GATE = {{}}
try:
    _t0 = time.time()
    _code = os.path.dirname(find('v2_lists.py'))
    rr = subprocess.run([sys.executable, os.path.join(_code, 'v2_lists.py'), '/kaggle/working/our_lists.json', '40', '4'],
                        capture_output=True, text=True, timeout=3 * 3600)
    print(rr.stdout[-2000:]); print(rr.stderr[-2000:])
    OURS = json.load(open('/kaggle/working/our_lists.json'))
    print('our lists', len(OURS), f'{{time.time() - _t0:.0f}}s')
except Exception as e:
    print('OUR ENGINE FAILED -> E3b lists only:', repr(e))
n_kept = n_fused = n_top1 = n_new = 0
for mid in set(ENG) | set(OURS):
    e, o = ENG.get(mid) or {{}}, OURS.get(mid) or {{}}
    libs = list(e.get('lib') or [])
    g = float(libs[0]) if libs else 0.0
    GATE[mid] = g
    if e.get('keys') and g >= GATE_LIB:
        n_kept += 1
        continue
    sc, smi_of = {{}}, {{}}
    for w, L in ((1.0, e), (BETA, o)):
        for r, (s, k) in enumerate(zip(L.get('smiles', []), L.get('keys', [])), 1):
            if not k:
                continue
            sc[k] = sc.get(k, 0.0) + w / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:40]
    if not order:
        continue
    ek = e.get('keys', [])
    n_fused += 1
    n_top1 += int(bool(ek) and order[0] != ek[0])
    n_new += len(set(order[:25]) - set(ek[:25]))
    ENG[mid] = dict(smiles=[smi_of[k] for k in order], keys=order)
print(f'E5 gate lib >= {{GATE_LIB}}: kept {{n_kept}} | fused {{n_fused}} (BETA {{BETA}}) | top-1 changed {{n_top1}} '
      f'| new keys in top-25 {{n_new}}')
'''

write = write.replace("print('E3 submission written:'", "print('E5 submission written:'", 1)
write = write.replace("print('E3 vs E1 submission rows:", "print('E5 vs E1 submission rows:", 1)
assert "E5 submission written" in write and "E5 vs E1" in write
write += '''
json.dump({m: dict(smiles=list(dict.fromkeys((ENG.get(m) or {}).get('smiles', [])))[:25], gate_lib=GATE.get(m))
           for m in map(str, sub.molecule_id)}, open('e5_lists.json', 'w'))
print('e5_lists.json written (lists + gate values, for the visible-set check)')
'''

nb = copy.deepcopy(e3b)
srcs = [setup.replace("# E3 = E1 (prize-eligible base) + forward-model re-scoring.",
                      "# E5 = E3b (prize-eligible base + forward-model re-scoring) + gated fusion with our V2 engine.", 1),
        eng, fm, FUSE, write]
cell = e3b["cells"][0]
nb["cells"] = []
for s in srcs:
    c = copy.deepcopy(cell)
    c["source"] = s.splitlines(keepends=True)
    c["outputs"], c["execution_count"] = [], None
    nb["cells"].append(c)
OUT.mkdir(exist_ok=True)
json.dump(nb, open(OUT / "casmi_e5.ipynb", "w", encoding="utf-8"), indent=1)

meta = json.load(open(HERE / "e3b" / "kernel-metadata.json"))
v2 = json.load(open(HERE / "e2" / "kernel-metadata.json"))["dataset_sources"]
meta.update(id="shishiradhikari11/casmi-e5-eligible-gated", title="CASMI E5 eligible gated", code_file="casmi_e5.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + [d for d in v2 if d not in meta["dataset_sources"]] +     ["shishiradhikari11/casmi-rdkit2025-cp313"]
json.dump(meta, open(OUT / "kernel-metadata.json", "w"), indent=1)
for s in srcs:  # every cell must at least parse
    compile(s, "cell", "exec")
print("wrote", OUT / "casmi_e5.ipynb", "| datasets:", meta["dataset_sources"])

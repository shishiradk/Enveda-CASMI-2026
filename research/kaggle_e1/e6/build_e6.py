"""Build E6 = E3b + our PubChem channel (own full-data fingerprint nets), merged append-only (NOT pushed).

    python research/kaggle_e1/e6/build_e6.py

Rule (chosen offline, research/scripts/e6_merge_eval.py, results/c3/e6_merge.json; leak-free ho1 nets on the bench):
  the engine's top-1 keeps its slot; below it, engine[1:] and the PubChem channel alternate (engine first),
  de-duplicated on the metric key, top 40. No gate (gating on library similarity lost most of the PubChem gain).
Bench: PubChem-only 0.215 -> 0.309, SV -0.0003, S1 -0.003, S2 -0.013; calibrated proxy about +0.010.

Cells: 0-2 = E5's cells 0-2 (E3b setup, engine, forward-model stage with the cp313 RDKit-wheel fix) | 3 = E6 channel
(e6_channel.py from dataset casmi-e6-pcnets, in a subprocess, time-budgeted) + merge | 4 = E5's submission cell with
e6_lists.json for the visible-set check. Any E6 failure keeps the E3b lists.
Reads e5/casmi_e5.ipynb, e3b/kernel-metadata.json (not modified). Writes e6/casmi_e6.ipynb, e6/kernel-metadata.json
(kernel shishiradhikari11/casmi-e6-pubchem-channel, private, T4) and e6/dataset/ (upload as casmi-e6-pcnets).
"""
import copy, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "e6"
ROOT = HERE.parents[1]

e5 = json.load(open(HERE / "e5" / "casmi_e5.ipynb", encoding="utf-8"))
assert len(e5["cells"]) == 5
setup, eng, fm, _fuse, write = ("".join(c["source"]) for c in e5["cells"])
for bad in ("ahmedberatozer", "casmi26-v4b", "casmi26-v3", "fpnet_full1", "casmi26-iceberg", "casmi26-glacier"):
    assert bad not in setup + eng + fm, bad

E6 = '''# E6: our PubChem channel -> PCL, merged append-only into ENG (engine top-1 kept, then engine/PubChem alternate).
# Nets: shishiradhikari11/casmi-e6-pcnets (our own FPNets trained on the competition train data only, MIT).
PCL, E6_STATS = {}, {}
try:
    _t0 = time.time()
    _budget = max(600, min(5 * 3600, 8.25 * 3600 - (time.time() - T0)))
    _py = find('e6_channel.py')
    _bits = os.path.join(os.path.dirname(find('coco_fp.npy')), 'fp_bits.npy')  # prvsiyan's 6,930-bit selection
    rr = subprocess.run([sys.executable, _py, os.path.join(COMP, 'test.parquet'), '/kaggle/working/e6_pc.json',
                         '--eng-dir', ENG_DIR, '--nets', os.path.dirname(_py), '--pubchem', find('pubchem_rows.parquet'),
                         '--fp-bits', _bits, '--workers', str(os.cpu_count() or 4), '--budget', str(int(_budget))],
                        capture_output=True, text=True, timeout=int(_budget) + 3600)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    PCL = json.load(open('/kaggle/working/e6_pc.json'))
    E6_STATS = PCL.pop('_stats', {})
    print('E6 PubChem lists', len(PCL), f'{time.time() - _t0:.0f}s')
except Exception as e:
    print('E6 CHANNEL FAILED -> E3b lists kept:', repr(e))
n_merged = n_new = 0
for mid, e in list(ENG.items()):
    p = PCL.get(mid) or {}
    if not e.get('keys') or not p.get('keys'):
        continue
    ek, es = e['keys'], e['smiles']
    seq = [(ek[0], es[0])]
    rest_e, rest_p = list(zip(ek[1:], es[1:])), list(zip(p['keys'], p['smiles']))
    for i in range(max(len(rest_e), len(rest_p))):
        seq += rest_e[i:i + 1] + rest_p[i:i + 1]
    keys, smis = [], []
    for k, s in seq:
        if k and k not in keys:
            keys.append(k); smis.append(s)
        if len(keys) >= 40:
            break
    n_merged += 1
    n_new += len(set(keys[:25]) - set(ek[:25]))
    ENG[mid] = dict(smiles=smis, keys=keys)
print(f'E6 merge: {n_merged}/{len(ENG)} molecules | new keys in top-25 {n_new} | {time.time()-T0:.0f}s')
'''

write = write.replace("print('E5 submission written:'", "print('E6 submission written:'", 1)
write = write.replace("print('E5 vs E1 submission rows:", "print('E6 vs E1 submission rows:", 1)
assert "E6 submission written" in write and "E6 vs E1" in write
tail = write[write.index("json.dump({m: dict(smiles="):]
write = write.replace(tail, '''json.dump({m: dict(smiles=list(dict.fromkeys((ENG.get(m) or {}).get('smiles', [])))[:25])
           for m in map(str, sub.molecule_id)}, open('e6_lists.json', 'w'))
json.dump(E6_STATS, open('e6_stats.json', 'w'))
print('e6_lists.json written (lists for the visible-set check) | E6 stats', E6_STATS)
''')

nb = copy.deepcopy(e5)
srcs = [setup.replace("# E5 = E3b (prize-eligible base + forward-model re-scoring) + gated fusion with our V2 engine.",
                      "# E6 = E3b (prize-eligible base + forward-model re-scoring) + our PubChem channel, append-only.", 1),
        eng, fm, E6, write]
assert srcs[0].startswith("# E6 =")
nb["cells"] = []
for s in srcs:
    c = copy.deepcopy(e5["cells"][0])
    c["source"] = s.splitlines(keepends=True)
    c["outputs"], c["execution_count"] = [], None
    nb["cells"].append(c)
OUT.mkdir(exist_ok=True)
json.dump(nb, open(OUT / "casmi_e6.ipynb", "w", encoding="utf-8"), indent=1)

meta = json.load(open(HERE / "e3b" / "kernel-metadata.json"))
meta.update(id="shishiradhikari11/casmi-e6-pubchem-channel", title="CASMI E6 PubChem channel", code_file="casmi_e6.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + [
    "shishiradhikari11/casmi-v2-pubchem", "shishiradhikari11/casmi-rdkit2025-cp313", "shishiradhikari11/casmi-e6-pcnets"]
json.dump(meta, open(OUT / "kernel-metadata.json", "w"), indent=1)

D = OUT / "dataset"
D.mkdir(exist_ok=True)
shutil.copy(OUT / "e6_channel.py", D / "e6_channel.py")
for v in ("merged", "single"):  # not fp_*.pt: the engine and setup cell glob that name for prvsiyan's nets
    src, dst = ROOT / "models" / "fp_full" / f"fp_{v}_full.pt", D / f"e6net_{v}_full.pt"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy(src, dst)
json.dump({"title": "casmi-e6-pcnets", "id": "shishiradhikari11/casmi-e6-pcnets",
           "licenses": [{"name": "MIT"}],
           "description": "E6 PubChem channel for CASMI 2026: e6_channel.py and two fingerprint nets trained by our team on the competition train data only."}, open(D / "dataset-metadata.json", "w"), indent=1)
(D / "README.md").write_text(
    "E6 PubChem channel for CASMI 2026: e6_channel.py (MIT) and two fingerprint networks (e6net_*_full.pt, "
    "prvsiyan FPNet architecture) trained by our team on the competition train.parquet only. Released under MIT.\n")
for s in srcs:
    compile(s, "cell", "exec")
print("wrote", OUT / "casmi_e6.ipynb", "| datasets:", meta["dataset_sources"], "| dataset dir", D)

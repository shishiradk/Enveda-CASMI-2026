# Fork Integration — Simple Steps

**Goal:** Add our V2 engine to your forked 0.399 notebook. It runs in parallel, our engine scores candidates, all three engines fuse together.

---

## Step 1: Add Three Datasets (in Kaggle notebook, right panel)

Click **"Add Input"** three times and search for:

1. `shishiradhikari11/casmi-v2-code` — our V0/V1/V2 code + drop-in runner
2. `shishiradhikari11/casmi-v1-assets` — COCONUT + train universe fingerprints
3. `shishiradhikari11/casmi-v2-pubchem` — PubChem table (2.7 GB)

All three will show in your **Input** section with little folder icons.

---

## Step 2: Make Code Changes

Copy-paste these **exactly as written** into the notebook cells shown.

### Change 2a: Insert New Cell (after cell 5)

Cell 5 is the one that says "Our engine (two-ranker + BIO + AFIX)". Right after it, insert a **new cell** with this code:

```python
# OUR V2 engine (kNN fingerprint + COCONUT/train + PubChem) in a subprocess -> OURS = {molecule_id: {smiles, keys}}
import glob, os, sys, json, subprocess, time
OURS, BETA = {}, 0.4          # BETA = weight of our list in the reciprocal-rank fusion (ENG uses ALPHA = 0.6)
try:
    _t0 = time.time()
    _code = os.path.dirname(glob.glob('/kaggle/input/**/v2_lists.py', recursive=True)[0])
    rr = subprocess.run([sys.executable, os.path.join(_code, 'v2_lists.py'), '/kaggle/working/our_lists.json', '40', '4'],
                        capture_output=True, text=True, timeout=3 * 3600)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    OURS = json.load(open('/kaggle/working/our_lists.json'))
    print('our lists', len(OURS), f'{time.time() - _t0:.0f}s')
except Exception as e:
    print('OUR ENGINE FAILED -> fork unchanged:', repr(e))
```

Run this cell. It will print "our lists 400" if successful.

### Change 2b: Edit Cell 12 (ICEBERG cell)

Find the cell with the comment `# ADDED (ICE_UNION)`. At the very top, add this **one line**:

```python
ENG_FOR_ICE = {m: {'smiles': (ENG.get(m, {}) or {}).get('smiles', [])[:40] + (OURS.get(m, {}) or {}).get('smiles', [])[:40],
                   'keys': (ENG.get(m, {}) or {}).get('keys', [])[:40] + (OURS.get(m, {}) or {}).get('keys', [])[:40]}
               for m in set(ENG) | set(OURS)}
```

Then find the line `if ENG:` and change it to:

```python
if ENG_FOR_ICE:
```

Then find where it says `e = ENG.get(str(m))` and change it to:

```python
e = ENG_FOR_ICE.get(str(m))
```

Then find the line with `es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]` and change it to:

```python
es = [s for s, k in zip(e['smiles'], e['keys']) if k and k not in bk]
```

### Change 2c: Replace Last Cell (Cell 15)

Delete the entire last cell and replace it with this:

```python
ALPHA, KRR = 0.6, 3.0
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2
from rdkit import Chem as _Ch2
def _form2(s):
    try:
        return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception:
        return s
def fuse3(v_smis, lists, n=40):
    """Weighted reciprocal-rank fusion: v4n list weight 1, then (smiles, keys, weight) lists."""
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for e_smis, e_keys, w in lists:
        for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
            if not k:
                continue
            sc[k] = sc.get(k, 0.0) + w / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]

if ENG or OURS:
    v4 = pd.read_csv('submission.csv')
    changed = n_ice = 0; out = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(';') if x and x != 'CCO']
        lists = []
        e = ENG.get(str(mid))
        if e and e.get('smiles'):
            lists.append((e['smiles'], e['keys'], ALPHA))
        o_ = OURS.get(str(mid))
        if o_ and o_.get('smiles'):
            lists.append((o_['smiles'], o_['keys'], BETA))
        if lists:
            fsm, fk, fsc = fuse3(vs, lists)
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if ice:
                try:
                    o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], ice, lam=ICE_LAM, top_n=len(fsm))
                    _g = gl_of(mid)
                    if _g:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [ice, _g], [ICE_LAM, GL_LAM], top_n=len(fsm))
                    n_ice += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused ICE rerank failed', mid, repr(ex))
            changed += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    assert len(fs) == len(v4) and fs.molecule_id.is_unique and fs.smiles.notna().all()
    assert fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    fs.to_csv('submission.csv', index=False)
    print('fused (v4n + ENG + OURS, +ICE/GL) written; top-1 changed in', changed, '| ICE reordered', n_ice,
          '| BETA', BETA, f'{time.time()-T0:.0f}s')
```

---

## Step 3: Save and Submit

1. **Save Version** (top right)
2. **Submit to Competition**
3. Wait for hidden test to score (usually 10–20 min)

---

## What You'll See

- Your fork runs unchanged (baseline, expect ~0.399)
- Our V2 engine runs in a subprocess (~8 min), outputs 40 candidates per molecule
- All three engines fuse together (their two + our one)
- ICEBERG + GLACIER re-score everything
- One submission.csv written

---

## After Scoring

Compare:
- **Fork baseline** (unchanged 0.399): ~0.399
- **Fork + V2 (BETA=0.4)**: TBD

If fork+V2 ≥ fork baseline → you can try BETA=0.6 (equal weight).

If fork+V2 < fork baseline → something is wrong; we debug.

**We also have V3 with popularity prior being built**, which scored 0.251 standalone and should go higher with the prior.

---

## Questions?

Everything is ready. Just follow the steps above in your Kaggle notebook fork.

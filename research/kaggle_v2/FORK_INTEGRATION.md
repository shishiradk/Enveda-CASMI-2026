# Adding our V2 engine to the forked 0.399 notebook

Fork: `shishiradhikari11/casmi26-v4n-engine-fusion-union-lb-0-399`. Three changes, then *Save Version* → *Submit*.

Our engine (kNN fingerprint retrieval over COCONUT + train structures + PubChem 5 ppm windows) is very different from
both engines already in the fork, which is what makes rank fusion pay off. It runs on CPU in its own subprocess
(~8–10 min on the hidden test, < 10 GB RAM) before the GPU stages, and never writes `submission.csv`.

## 0. Add inputs (right panel → *Add Input* → *Your Datasets*)

- `shishiradhikari11/casmi-v2-code` (our pipeline code: `casmi_v0_kaggle.py`, `casmi_v1_kaggle.py`, `casmi_v2_kaggle.py`, `v2_lists.py`)
- `shishiradhikari11/casmi-v1-assets` (COCONUT + train structures + fingerprints)
- `shishiradhikari11/casmi-v2-pubchem` (PubChem table, 2.7 GB)

## 1. NEW cell — insert directly after cell 5 (the "Our engine (two-ranker + BIO + AFIX)" code cell)

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

## 2. EDIT cell 12 (the "ICEBERG post-ranker isomer re-scoring" cell): three replacements

So our candidates also get ICEBERG + GLACIER forward scores (the "union" step that made 0.386 → 0.399).

**(a)** Add this line at the very top of the cell:

```python
ENG_FOR_ICE = {m: {'smiles': (ENG.get(m, {}) or {}).get('smiles', [])[:40] + (OURS.get(m, {}) or {}).get('smiles', [])[:40],
                   'keys': (ENG.get(m, {}) or {}).get('keys', [])[:40] + (OURS.get(m, {}) or {}).get('keys', [])[:40]}
               for m in set(ENG) | set(OURS)}
```

**(b)** Replace `    if ENG:                              # ADDED (ICE_UNION)` with
`    if ENG_FOR_ICE:                      # ADDED (ICE_UNION)` (keep the rest of that comment line).

**(c)** Inside that block, replace

```python
            e = ENG.get(str(m))
```
with
```python
            e = ENG_FOR_ICE.get(str(m))
```
and replace
```python
            es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]
```
with
```python
            es = [s for s, k in zip(e['smiles'], e['keys']) if k and k not in bk]
```

## 3. REPLACE the whole last cell (cell 15, "Fusion + forward re-rank") with

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

## Submissions to make (one variable at a time; public LB noise is about ±0.016)

1. The fork unchanged (baseline on our team; expect ~0.399).
2. The fork + our engine with `BETA = 0.4`.
3. If 2 ≥ baseline: `BETA = 0.6` (same weight as ENG).

Keep the order of decisions on the record in `research/analysis/submission_v1.md` (LB log).

## Licence note for the final (prize-eligible) selection

`prvsiyan/chebi-lipidmaps-casmi26` is CC BY-NC-SA 4.0. The winner licence is MIT, so a final selection must not depend
on it. Replace it with our own ChEBI/LIPID MAPS build (opencode Task C) before choosing the two final submissions.

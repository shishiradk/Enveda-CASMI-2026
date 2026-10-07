# Ceiling measurement for scaffold+substituent recombination on S1/S3 (truth-key analogs removed)
import pickle, os, sys, collections, numpy as np, duckdb
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit.Chem import AllChem, rdMolDescriptors as rd
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
here = os.path.dirname(__file__)
A = pickle.load(open('D:/Enveda-CASMI-2026/results/c3gen/ssr_probe_ana.pkl','rb'))
T = {m:(s,c) for q,m,s,c in duckdb.sql("select qset,mid,smiles,correct from 'D:/Enveda-CASMI-2026/results/bench/truth.parquet'").fetchall()}
TE = rdMolStandardize.TautomerEnumerator()
def ck(m):
    try: return Chem.MolToInchiKey(TE.Canonicalize(m))[:14]
    except Exception: return None
def nostereo(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.RemoveStereochemistry(m); return m
def scaf(m):
    try: return Chem.MolToSmiles(MS.GetScaffoldForMol(m))
    except Exception: return None
def gscaf(m):
    try: return Chem.MolToSmiles(MS.MakeScaffoldGeneric(MS.GetScaffoldForMol(m)))
    except Exception: return None
def subs(m, sc):
    core = Chem.MolFromSmiles(sc)
    if core is None or not m.HasSubstructMatch(core): return None
    sc_ = Chem.ReplaceCore(m, core, labelByIndex=False)
    if sc_ is None: return collections.Counter()
    fr = Chem.GetMolFrags(sc_, asMols=True, sanitizeFrags=False)
    return collections.Counter(Chem.MolToSmiles(f) for f in fr)
fpg = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
res = collections.defaultdict(list)
for b, rows in A.items():
    for r in rows:
        ts, tk = T[r['mid']]
        tm = nostereo(ts)
        if tm is None: continue
        tsc, tg = scaf(tm), gscaf(tm)
        tsub = subs(tm, tsc) if tsc else None
        tfp = fpg.GetFingerprint(tm)
        tf = rd.CalcMolFormula(tm)
        ana = []
        for (i, s, k, smi, mass) in r['ana']:
            am = nostereo(smi)
            if am is None: continue
            if k == tk or ck(am) == tk: continue  # truth removed
            ana.append((s, am))
            if len(ana) == 20: break
        x = dict(b=b, n=len(ana), nring=rd.CalcNumRings(tm), mw=r['target'])
        for K in (1, 5, 20):
            A_ = ana[:K]
            x[f'sc{K}'] = any(scaf(a)==tsc for _,a in A_)
            x[f'gs{K}'] = any(gscaf(a)==tg for _,a in A_)
            x[f'tan{K}'] = max([DataStructs.TanimotoSimilarity(tfp, fpg.GetFingerprint(a)) for _,a in A_] or [0])
        # substituent analysis among analogs sharing the exact scaffold
        same = [a for _,a in ana if scaf(a)==tsc]
        x['posiso'] = any(rd.CalcMolFormula(a)==tf for a in same)
        best = None; vocab = collections.Counter()
        for a in same:
            sa = subs(a, tsc)
            if sa is None: continue
            vocab |= sa
            d = sum(((tsub or collections.Counter()) - sa).values()) + sum((sa - (tsub or collections.Counter())).values())
            best = d if best is None else min(best, d)
        x['subdist'] = best
        x['vocab_cover'] = (tsub is not None and len(same)>0 and all(k in vocab for k in tsub))
        x['nsub'] = sum(tsub.values()) if tsub else 0
        # H-bearing scaffold positions
        core = Chem.MolFromSmiles(tsc) if tsc else None
        x['npos'] = sum(1 for at in core.GetAtoms() if at.GetTotalNumHs()>0) if core else 0
        x['acyclic'] = (tsc == '' )
        res[b].append(x)
for b, xs in res.items():
    n = len(xs); f = lambda key: np.mean([bool(x[key]) for x in xs])
    print(f'== {b} n={n} acyclic-truth={f("acyclic"):.2f}')
    for K in (1,5,20):
        print(f'  top{K}: exact-scaffold {f(f"sc{K}"):.3f}  generic-scaffold {f(f"gs{K}"):.3f}  median maxTan {np.median([x[f"tan{K}"] for x in xs]):.2f}  share maxTan>=0.7 {np.mean([x[f"tan{K}"]>=0.7 for x in xs]):.2f}')
    sd = [x['subdist'] for x in xs if x['subdist'] is not None]
    print(f'  pos-isomer analog (same scaffold+formula) {f("posiso"):.3f}; truth substituents all in shared-scaffold analog vocab {f("vocab_cover"):.3f}')
    print(f'  substituent multiset edit distance to closest same-scaffold analog: ' + str({d:sd.count(d) for d in sorted(set(sd))}))
    print(f'  median truth #substituents {np.median([x["nsub"] for x in xs])}, median H-bearing scaffold atoms {np.median([x["npos"] for x in xs])}')
    # by ring count
    for lo,hi in ((0,1),(1,3),(3,99)):
        g = [x for x in xs if lo<=x['nring']<hi]
        if g: print(f'   rings[{lo},{hi}) n={len(g)} sc20 {np.mean([x["sc20"] for x in g]):.2f} gs20 {np.mean([x["gs20"] for x in g]):.2f}')

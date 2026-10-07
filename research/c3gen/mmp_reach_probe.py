# Probe (not a generator): how often is the bench truth one matched-molecular-pair edit away from a top-N spectral library analog (truth removed)?
# Usage: python research/c3gen/mmp_reach_probe.py S1|S3 20   (loads the bench pool cache, ~1 GB RAM, ~8 min)
# Measure: how often is the truth one MMP edit away from a top-N library analog (truth removed)?
import pickle, sys, time, json, numpy as np, duckdb
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdMMPA, AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')
R='D:/Enveda-CASMI-2026/'
bench=sys.argv[1]; NA=int(sys.argv[2])
P=pickle.load(open(R+'results/bench/cache/pool_train_3033286496_2026.03.6.pkl','rb'))
smi=P['smiles']; keys=P['keys']; mass=P['mass']; del P
d=pickle.load(open(R+f'results/bench/ho1/recs_{bench}.pkl','rb'))
qs='S12' if bench=='S1' else 'S3'
T=duckdb.sql(f"select mid,smiles,correct from '{R}results/bench/truth.parquet' where qset='{qs}'").df().set_index('mid')
def canon(m): return Chem.MolToSmiles(m, isomericSmiles=False)
def nheavy(s): return sum(1 for a in Chem.MolFromSmiles(s,sanitize=False).GetAtoms() if a.GetAtomicNum()>1)
def cuts1(m):
    """single-cut: set of (constant, variable) canonical pairs, both orientations; plus H-cut (variable=[H])"""
    out=set()
    for core,ch in rdMMPA.FragmentMol(m,maxCuts=1,resultsAsMols=False):
        a,b=ch.split('.')
        a=canon(Chem.MolFromSmiles(a)); b=canon(Chem.MolFromSmiles(b))
        out.add((a,b)); out.add((b,a))
    mh=Chem.Mol(m)
    for at in m.GetAtoms():
        if at.GetTotalNumHs()>0:
            rw=Chem.RWMol(m); i=rw.AddAtom(Chem.Atom(0)); rw.GetAtomWithIdx(i).SetAtomMapNum(1)
            a=rw.GetAtomWithIdx(at.GetIdx()); a.SetNumExplicitHs(max(0,a.GetNumExplicitHs()-1)); a.SetNoImplicit(False)
            rw.AddBond(at.GetIdx(),i,Chem.BondType.SINGLE)
            try:
                mm=rw.GetMol(); Chem.SanitizeMol(mm); out.add((canon(mm),'[H][*:1]'))
            except Exception: pass
    return out
def cuts2(m):
    out=set()
    for core,ch in rdMMPA.FragmentMol(m,maxCuts=2,maxCutBonds=40,resultsAsMols=False):
        if core: out.add((ch, core))   # constant = two ends (as given string), variable = linker
    return out
TR={}
st=dict(cf=[],n=0,anyana=0,iso=0,mmp1=0,mmp1_small=0,mmp2=0,either=0,tan_best=[],dm_top1=[],rank_hit=[])
t0=time.time()
for r in d['recs']:
    mid=r['mid']
    if mid not in T.index: continue
    tm=Chem.MolFromSmiles(T.loc[mid,'smiles']); 
    if tm is None: continue
    tk=T.loc[mid,'correct']; tik=Chem.MolToInchiKey(tm)[:14]
    ana=[(p,s) for p,s in r['ana'] if keys[p]!=tk][:NA]
    am=[]
    for p,s in ana:
        m=Chem.MolFromSmiles(smi[p])
        if m is None: continue
        if Chem.MolToInchiKey(m)[:14] in (tk,tik): continue
        am.append(m)
    st['n']+=1
    if not am: continue
    st['anyana']+=1
    tfp=AllChem.GetMorganFingerprintAsBitVect(tm,2,2048)
    afp=[AllChem.GetMorganFingerprintAsBitVect(m,2,2048) for m in am]
    st['tan_best'].append(max(DataStructs.BulkTanimotoSimilarity(tfp,afp)))
    tmass=Descriptors.ExactMolWt(tm)
    st['dm_top1'].append(tmass-Descriptors.ExactMolWt(am[0]))
    st['iso']+=any(abs(tmass-Descriptors.ExactMolWt(m))<0.003 for m in am)
    th=tm.GetNumHeavyAtoms()
    tc1=cuts1(tm); tconst1={}
    for c,v in tc1: tconst1.setdefault(c,set()).add(v)
    hit1=hit1s=hit2=False; first=None; best_cf=0.0
    for j,m in enumerate(am):
        for c,v in cuts1(m):
            if c in tconst1 and tconst1[c]-{v}:
                nc=nheavy(c)-1
                for vt in tconst1[c]-{v}:
                    nvt=max(nheavy(vt)-1,0); nva=max(nheavy(v)-1,0)
                    cf=nc/max(th,1); best_cf=max(best_cf,cf)
                    if cf>=0.6 and nvt<=10 and nva<=10:
                        hit1=True
                        if first is None: first=j
                        if j<5: TR[(v,vt)]=TR.get((v,vt),0)+1
                        if nvt<=4 and nva<=4: hit1s=True
    st['cf'].append(best_cf)
    if th<=60:
        tc2={}
        for c,v in cuts2(tm): tc2.setdefault(c,set()).add(v)
        for m in am[:10]:
            if m.GetNumHeavyAtoms()>60: continue
            for c,v in cuts2(m):
                if c in tc2 and tc2[c]-{v} and nheavy(c)-2>=0.6*th and all(nheavy(x)-2<=10 for x in tc2[c]|{v}): hit2=True; break
            if hit2: break
    st['mmp1']+=hit1; st['mmp1_small']+=hit1s; st['mmp2']+=hit2; st['either']+=(hit1 or hit2)
    st['rank_hit'].append(first if first is not None else -1)
tb=np.array(st['tan_best']); dm=np.abs(np.array(st['dm_top1'])); rh=np.array(st['rank_hit'])
print(json.dumps(dict(bench=bench,NA=NA,n=st['n'],with_analogs=st['anyana'],isomer_in_analogs=st['iso'],
  mmp1=st['mmp1'],mmp1_var_le4=st['mmp1_small'],mmp2_top10=st['mmp2'],mmp_any=st['either'],
  tan_best_median=float(np.median(tb)),tan_ge07=int((tb>=0.7).sum()),tan_ge05=int((tb>=0.5).sum()),
  dm_top1_median=float(np.median(dm)),dm_top1_lt0p01=int((dm<0.01).sum()),
  first_hit_rank_hist={k:int(((rh>=lo)&(rh<hi)).sum()) for k,(lo,hi) in {'0':(0,1),'1-4':(1,5),'5-19':(5,20),'20+':(20,999)}.items()},
  sec=round(time.time()-t0,1),cf_ge06=int((np.array(st['cf'])>=0.6).sum()),cf_ge08=int((np.array(st['cf'])>=0.8).sum()),top_transforms=sorted(((v,'%s>>%s'%k) for k,v in TR.items()),reverse=True)[:30])))

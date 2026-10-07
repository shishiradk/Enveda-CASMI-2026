# extract analog smiles/keys for S1/S3 bench records (top 30) from bench pool cache; small output
import pickle, numpy as np, os, gc
os.chdir('D:/Enveda-CASMI-2026')
P = pickle.load(open('results/bench/cache/pool_train_3033286496_2026.03.6.pkl','rb'))
print(P.keys(), len(P['mass']))
out = {}
for b in ('S1','S3'):
    d = pickle.load(open(f'results/bench/ho1/recs_{b}.pkl','rb'))
    rows = []
    for r in d['recs']:
        ana = [(int(i), float(s), P['keys'][i], P['smiles'][i], float(P['mass'][i])) for i, s in r['ana'][:30]]
        win = [(P['keys'][i], P['smiles'][i]) for i in r['cand']]
        rows.append(dict(mid=r['mid'], target=r['target'], ana=ana, adducts=r['adducts'], nwin=len(win)))
    out[b] = rows
    del d; gc.collect()
pickle.dump(out, open('results/c3gen/ssr_probe_ana.pkl', 'wb'))


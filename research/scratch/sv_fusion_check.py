import sys, glob, pickle, json, itertools
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT/'research'/'bench')); sys.path.insert(0, str(ROOT/'research'/'scripts'))
import bench, bench_eval as BE
from e6_merge_eval import merge
import lightgbm as lgb
truth = bench.load_truth()['SV']
R = pickle.load(open(bench.OUT/'e1'/'SV.pkl','rb'))
P = pickle.load(open(ROOT/'results/c3/e6_work/pc_lists_SV.pkl','rb'))
# E6 merged (E7 family primary) lists
E6 = {}
for mid, r in R.items():
    _, keys, idx = BE.engine_list(r, 'blend')
    e = dict(keys=keys, raw=[r['keys'][i] for i in idx])
    ks, rs = merge(e, P.get(mid, {}), 1, 'alt')
    g = float(r['lib'][idx[0]]) if len(idx) else 0.0
    E6[mid] = dict(keys=ks[:25], raw=rs[:25], g=g)
# rebuilt engine lists on the same SV queries, ranker_v0
rows = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('results/v4n/sim/testSV_RA/rows/*.parquet'))], ignore_index=True)
mdl = pickle.load(open('results/v4n/sim/hoR_RA/ranker_v0.pkl','rb'))
X = rows[['f_'+f for f in mdl['features']]].values.astype(np.float32)
rows['p'] = np.mean([lgb.Booster(model_str=b).predict(X) for b in mdl['boosters']], 0)
V4 = {}
for mid, g in rows.groupby('key'):
    g = g.sort_values('p', ascending=False, kind='stable')
    V4[str(mid)] = list(dict.fromkeys(g.cand_key.tolist()))[:25]
mids = [m for m in R if m in V4]
print('molecules', len(R), 'with v4n lists', len(mids), '| key-space overlap sample:',
      np.mean([len(set(E6[m]['keys']) & set(V4[m]))>0 for m in mids]).round(3))
def rank_of(keys, mid):
    return BE.rank_in(keys, None, R[mid], truth[mid]['correct'])
def fuse(l1, l2, w2=0.6, K=3.0):
    sc = {}
    for w, L in ((1.0, l1), (w2, l2)):
        for r, k in enumerate(L):
            sc[k] = sc.get(k, 0.0) + w/(K+r)
    return sorted(sc, key=lambda k: -sc[k])[:25]
def rank_raw(keys, mid):  # E6 raw-aware rank for E6-only list
    return BE.rank_in(E6[mid]['keys'], E6[mid]['raw'], R[mid], truth[mid]['correct'])
def ev(name, fn):
    rr = np.array([BE.rr(fn(m)) for m in mids]); return name, rr
res = {}
res['E6 alone'] = np.array([BE.rr(rank_raw(None, m)) for m in mids])
res['v4n alone'] = np.array([BE.rr(rank_of(V4[m], m)) for m in mids])
for w in (0.3, 0.6, 1.0):
    res[f'fuse w={w}'] = np.array([BE.rr(rank_of(fuse(E6[m]['keys'], V4[m], w), m)) for m in mids])
def protect(m, w=0.6, thr=0.9):
    return E6[m]['keys'] if E6[m]['g'] >= thr else fuse(E6[m]['keys'], V4[m], w)
for thr in (0.9, 0.7):
    res[f'fuse w=0.6 + keep E6 when lib>={thr}'] = np.array([BE.rr(rank_of(protect(m, 0.6, thr), m)) for m in mids])
def append_only(m, k=3):
    base = E6[m]['keys'][:k]; out = list(base)
    for x in fuse(E6[m]['keys'], V4[m], 0.6):
        if x not in out: out.append(x)
    return out[:25]
res['fuse but E6 top-3 pinned'] = np.array([BE.rr(rank_of(append_only(m), m)) for m in mids])
base = res['E6 alone']
print(f"{'variant':42} MRR@25  top1   delta  lost-top1 gained-top1")
for k, v in res.items():
    r1 = v == 1.0
    print(f"{k:42} {v.mean():.4f} {r1.mean():.3f} {v.mean()-base.mean():+.4f} {int((base==1)[~r1].sum()):4d} {int(((base<1)&r1).sum()):6d}")
g = np.array([E6[m]['g'] for m in mids]); f = res['fuse w=0.6']
lost = (base == 1) & (f < 1)
print('lost top-1 when fused: n =', int(lost.sum()), '| their E6 lib sim: median', np.round(np.median(g[lost]),2) if lost.any() else None)
print('E6 top-1 correct count', int((base==1).sum()), '| lib>=0.9 among them', int(((base==1)&(g>=0.9)).sum()))

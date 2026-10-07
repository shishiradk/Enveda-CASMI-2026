"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles, keys, scores, lib_max}} (blend order).

E3 version of the E1 runner.  The first 40 entries are exactly E1's list (first 80 candidates in blend order,
de-duplicated on the metric key, at most 40).  When those 40 are full the scan goes on up to 60 entries, so the
forward-model stage can form formula groups over a top 60.  scores = blend score of each kept candidate,
lib_max = best library similarity of any candidate of the molecule (used only to order the forward-model work).
"""
import os, sys, json, time
import numpy as np
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    keep = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    import casmi_engine as E
    _orig_find = E.find
    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    import glob
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    print('engine fp models', fp_models, flush=True)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,
                           workers=os.cpu_count(), limit=limit, w_pv=0.88)
    bl = E.blend_scores(S['pv'], S['ours'], 0.88)
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:max(80, 3 * keep)]
        smis, keys, scs, libs, seen = [], [], [], [], set()
        lib = r.get('lib')
        for j, c in enumerate(r['cand'][order]):
            if j >= 80 and len(smis) < 40:         # E1 stops after 80 candidates
                break
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k); scs.append(float(bl[mid][order[j]]))
            libs.append(float(lib[order[j]]) if lib is not None and len(lib) else 0.0)
            if len(smis) >= max(40, keep):
                break
        res[str(mid)] = dict(smiles=smis, keys=keys, scores=scs, lib=libs,
                             lib_max=float(np.max(lib)) if lib is not None and len(lib) else 0.0)
    json.dump(res, open(out, 'w'))
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
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)

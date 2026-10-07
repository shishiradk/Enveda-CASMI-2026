def e7_merge(ENG, C3L, keep_top=3, n_ins=5, ref_n=25, cap=40):
    """E7SPEC (DEC-010): final list = E6 ranks 1-3 unchanged; then the first n_ins C3 (rr_bt) candidates whose metric
    key is not already in the E6 top ref_n (and not repeated); then E6 ranks 4+; de-dup on the metric key; cap entries
    (the submission cell writes the first 25). ENG = {mid: {smiles, keys}} after the E6 merge; C3L = {mid: {smiles,
    keys, ...}} from e7_c3_channel.py. Any molecule without a C3 list, or any error, keeps its E6 entry unchanged;
    a result whose first keep_top keys differ from E6's is discarded (post-condition). Returns (new ENG, stats)."""
    out = dict(ENG)
    st = dict(n_mol=len(ENG), n_c3=0, n_rows_ins=0, n_ins=0, n_err=0, n_post=0)
    for mid, e in ENG.items():
        c = C3L.get(mid) if isinstance(C3L, dict) else None
        if not c or not isinstance(e, dict) or not e.get('keys'):
            continue
        st['n_c3'] += 1
        try:
            ek, es = list(e['keys']), list(e['smiles'])
            ref = set(k for k in ek[:ref_n] if k)
            ins, seen = [], set()
            for k, s in zip(c.get('keys') or [], c.get('smiles') or []):
                if not k or not isinstance(s, str) or not s or ';' in s or k in ref or k in seen:
                    continue
                ins.append((k, s)); seen.add(k)
                if len(ins) >= n_ins:
                    break
            seq = list(zip(ek[:keep_top], es[:keep_top])) + ins + list(zip(ek[keep_top:], es[keep_top:]))
            keys, smis = [], []
            for k, s in seq:
                if k and k not in keys:
                    keys.append(k); smis.append(s)
                if len(keys) >= cap:
                    break
            if keys[:keep_top] != [k for k in ek if k][:keep_top]:
                st['n_post'] += 1
                continue
            out[mid] = dict(smiles=smis, keys=keys)
            n_new = len(set(keys[:ref_n]) - ref)
            st['n_ins'] += n_new
            st['n_rows_ins'] += int(n_new > 0)
        except Exception:
            st['n_err'] += 1
    return out, st

"""E4 = popularity prior (pop_stage) + E3b's forward-model re-ordering (ours, MIT).

E3b's ``fm_stage.run_fm_stage`` is used unchanged: it runs GLACIER / ICEBERG once on the engine lists and returns the
E3b lists.  The popularity stage only permutes unprotected candidates inside the same top-N window, so the set of
candidates E3b sends to the forward models is the same for the popularity-adjusted lists; their scores are read back
from the runner files and E3b's rule (``fm_stage.select`` + ``fm_rerank.rerank_molecule``) is applied a second time,
now to the popularity-adjusted lists.

``apply_forward`` on the ORIGINAL lists must reproduce the E3b lists key for key; ``run_e4`` checks this and raises
otherwise (the notebook then keeps the E3b lists).
"""
import json
import os

import fm_stage
from fm_rerank import list_changes, rerank_molecule

import pop_stage

MOVE_KEYS = ("smiles", "keys", "scores", "lib", "pv", "ours", "pop")


def scores_by_smiles(work, fm_stats, models=fm_stage.MODELS):
    """{model: {mid: {smiles: score | None}}} from fm_input.json / fm_<model>.json, with E3b's acceptance rules."""
    inp_path = os.path.join(work, "fm_input.json")
    if not os.path.exists(inp_path):
        return {}
    inp = json.load(open(inp_path))
    out = {}
    for tag in models:
        ms = (fm_stats.get("models") or {}).get(tag) or {}
        if "n_candidates_scored" not in ms or str(ms.get("status", "")).startswith("discarded"):
            continue                                     # skipped, no output, or discarded by E3b
        try:
            sc = json.load(open(os.path.join(work, f"fm_{tag}.json"))).get("scores", {})
        except Exception:  # noqa: BLE001
            continue
        got = {}
        for m, v in inp.items():
            x = sc.get(m)
            if isinstance(x, list) and len(x) == len(v["smiles"]):
                got[m] = {s: (y if isinstance(y, (int, float)) else None) for s, y in zip(v["smiles"], x)}
        out[tag] = got
    return out


def apply_forward(lists, covered, by_smiles, lams, topn=60, protect_lib=0.6):
    """E3b's re-ordering of ``lists`` with forward scores looked up by SMILES.  Returns (new lists, changes)."""
    sel = fm_stage.select(lists, covered, topn, protect_lib)
    new, ch = {}, {"top1": 0, "order": 0, "set": 0}
    for mid, e in lists.items():
        new[mid] = e
        if mid not in sel:
            continue
        n = len(e["smiles"])
        f = sel[mid]["formulas"] + [None] * (n - len(sel[mid]["formulas"]))
        per = {}
        for tag, got in by_smiles.items():
            if mid in got:
                col = [None] * n
                for i in sel[mid]["send"]:
                    col[i] = got[mid].get(e["smiles"][i])
                per[tag] = col
        if not per:
            continue
        perm, _ = rerank_molecule(f, e.get("scores") or [], per, lams, topn)
        if sorted(perm) != list(range(n)):
            raise AssertionError(f"not a permutation for {mid}")
        if perm == list(range(n)):
            continue
        e2 = dict(e)
        for key in MOVE_KEYS:
            if isinstance(e.get(key), list) and len(e[key]) == n:
                e2[key] = [e[key][i] for i in perm]
        new[mid] = e2
        c = list_changes(e["keys"], e2["keys"], 25)
        for k in ch:
            ch[k] += int(c[k])
    return new, ch


def covered_molecules(test_parquet):
    import pandas as pd
    te = pd.read_parquet(test_parquet, columns=["molecule_id", "adduct"])
    return set(te.loc[te.adduct.astype(str).isin(fm_stage.ADDUCTS), "molecule_id"].astype(str))


def run_e4(eng_lists, e3b_lists, fm_stats, lookup, test_parquet, work, mu=0.25, topn=60, lam_ice=0.5, lam_gl=0.5, protect_lib=0.6,
           score_mode="logit", key_fn=pop_stage.plain_key):
    """eng_lists: the engine lists; e3b_lists: what fm_stage.run_fm_stage returned for them (or eng_lists itself when
    the forward stage failed / is off, with fm_stats = {}).  Returns (e4 lists, stats)."""
    lams = {"ice": float(lam_ice), "gl": float(lam_gl)}
    forward_ok = e3b_lists is not eng_lists and bool(fm_stats.get("models"))
    by, covered = {}, set()
    if forward_ok:
        covered = covered_molecules(test_parquet)
        by = scores_by_smiles(work, fm_stats)
        chk, _ = apply_forward(eng_lists, covered, by, lams, topn, protect_lib)
        bad = [m for m in eng_lists if list(chk[m]["keys"]) != list(e3b_lists[m]["keys"])]
        if bad:
            raise AssertionError(f"E3b replay differs for {len(bad)} molecules, e.g. {bad[:3]}")
    popped, st = pop_stage.run_pop_stage(eng_lists, lookup, mu=mu, topn=topn, protect_lib=protect_lib, score_mode=score_mode, key_fn=key_fn)
    if forward_ok:
        new, ch = apply_forward(popped, covered, by, lams, topn, protect_lib)
    else:
        new, ch = popped, {"top1": 0, "order": 0, "set": 0}
    for m, e in eng_lists.items():
        if sorted(new[m]["keys"]) != sorted(e["keys"]) or len(new[m]["smiles"]) != len(e["smiles"]):
            raise AssertionError(f"candidate set changed for {m}")
        if protect_lib is not None:
            for i, x in enumerate(list(e.get("lib") or [])[:topn]):
                if float(x) >= protect_lib and new[m]["keys"][i] != e["keys"][i]:
                    raise AssertionError(f"protected candidate moved in {m} slot {i}")
    vs = {"top1": 0, "order": 0, "set": 0, "top1_mids": []}
    for m in eng_lists:
        c = list_changes(e3b_lists[m]["keys"], new[m]["keys"], 25)
        for k in ("top1", "order", "set"):
            vs[k] += int(c[k])
        if c["top1"]:
            vs["top1_mids"].append(m)
    st.update(forward_replayed=forward_ok, forward_models=sorted(by), forward_after_pop=ch,
              e4_vs_e3b=dict(n_top1_changed=vs["top1"], n_top25_order_changed=vs["order"], n_top25_set_changed=vs["set"],
                             top1_changed_molecules=vs["top1_mids"][:200]))
    return new, st

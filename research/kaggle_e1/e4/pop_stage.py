"""E4 popularity stage (ours, MIT): a PubChem popularity prior on the engine's candidate list, applied BEFORE the
E3b forward-model re-ordering.  Method and design choices: research/analysis/e4_popularity.md.

Per molecule, over the first ``topn`` entries of the engine list (best first):

1. ranker score ``s``
     score_mode 'logit' (default): L = w_pv * logit(pv) + (1 - w_pv) * logit(ours) from the two raw ranker
         probabilities the E4 engine runner exports, made non-increasing along the list by isotonic regression
         (pool-adjacent-violators).  The engine's order is therefore kept exactly; only the margins come from L.
     score_mode 'rank': the exported within-molecule rank blend (``scores``) as it is.
2. z = (s - mean) / (population sd + 1e-9) over all ``topn`` entries.
3. pop = log1p(n_sid) + log1p(n_pmid) from the lookup (0 for a candidate without a PubChem record or unknown to the
   table); f = z + mu * pop.
   A molecule whose list lacks usable raw probabilities keeps its order (counted); if that is true for every
   molecule the stage raises and the notebook keeps the E3b lists.
4. Candidates whose own library similarity is >= ``protect_lib`` keep their slot (E3b's protection).  All other
   candidates are sorted by f (descending, rounded to 1e-9, ties keep their current order) and written into the
   slots the unprotected candidates occupied.  Entries after ``topn`` never move.
5. ``scores`` stays attached to the SLOTS (it is not permuted): the forward-model stage then treats the
   popularity-adjusted order exactly as E3b treats the engine's order.

mu = 0 (or an empty lookup) is the identity.  ``run_pop_stage`` never changes its input.
"""
import glob
import math
import os

import numpy as np

GRID = 1e-9
MOVE_KEYS = ("smiles", "keys", "lib", "pv", "ours")      # per-candidate fields that travel with the candidate


class PopLookup:
    """Sorted S14 key array + two int32 arrays (pop_lookup.npz)."""

    def __init__(self, keys, n_sid, n_pmid):
        self.keys = np.asarray(keys, dtype="S14")
        self.n_sid = np.asarray(n_sid, np.int64)
        self.n_pmid = np.asarray(n_pmid, np.int64)
        if not (len(self.keys) == len(self.n_sid) == len(self.n_pmid)):
            raise ValueError("lookup arrays differ in length")
        if len(self.keys) > 1 and not (self.keys[1:] > self.keys[:-1]).all():
            raise ValueError("lookup keys are not sorted / unique")

    def __len__(self):
        return len(self.keys)

    def get(self, key):
        """(n_sid, n_pmid) or None when the key is not in the table."""
        if not isinstance(key, str) or len(key) != 14 or not len(self.keys):
            return None
        k = np.bytes_(key.encode("ascii", "replace"))
        i = int(np.searchsorted(self.keys, k))
        if i < len(self.keys) and self.keys[i] == k:
            return int(self.n_sid[i]), int(self.n_pmid[i])
        return None


def load_lookup(path):
    z = np.load(path, allow_pickle=False)
    return PopLookup(z["keys"], z["n_sid"], z["n_pmid"])


def find_lookup(root="/kaggle/input"):
    hits = sorted(glob.glob(os.path.join(root, "**", "pop_lookup.npz"), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError("pop_lookup.npz (dataset casmi-pop-lookup) not found")
    return hits[0]


def plain_key(smiles):
    """Plain InChIKey first block of the SMILES as listed (no tautomer canonicalisation) or None."""
    from rdkit import Chem, rdBase
    _quiet = rdBase.BlockLogs()  # noqa: F841
    try:
        m = Chem.MolFromSmiles(smiles)
        k = Chem.MolToInchiKey(m) if m is not None else ""
        return k[:14] if k else None
    except Exception:  # noqa: BLE001
        return None


def candidate_counts(smiles, key, lookup, key_fn=plain_key):
    """Counts of one candidate: by the plain key of its SMILES, else by the engine's (tautomer-canonical) key.
    Returns (n_sid, n_pmid, how) with how in 'plain' / 'canon' / 'unknown'."""
    k = key_fn(smiles) if key_fn is not None else None
    v = lookup.get(k) if k else None
    if v is not None:
        return v[0], v[1], "plain"
    v = lookup.get(key)
    if v is not None:
        return v[0], v[1], "canon"
    return 0, 0, "unknown"


def pop_value(n_sid, n_pmid):
    return math.log1p(max(int(n_sid), 0)) + math.log1p(max(int(n_pmid), 0))


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def isotonic_nonincreasing(v):
    """Least-squares non-increasing fit of v in list order (pool-adjacent-violators)."""
    blocks = []                                    # [sum, count]
    for x in v:
        blocks.append([float(x), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out.extend([s / c] * c)
    return out


def logit_blend(pv, ours, w_pv=0.88, eps=1e-6):
    def lg(p):
        p = min(max(float(p), eps), 1.0 - eps)
        return math.log(p / (1.0 - p))
    return [w_pv * lg(a) + (1.0 - w_pv) * lg(b) for a, b in zip(pv, ours)]


def ranker_term(entry, top, score_mode="logit", w_pv=0.88):
    """Non-increasing ranker score of the first ``top`` entries (see the module text)."""
    if score_mode == "logit":
        pv, ours = list(entry.get("pv") or [])[:top], list(entry.get("ours") or [])[:top]
        if len(pv) < top or len(ours) < top or not all(_finite(x) for x in pv + ours):
            raise ValueError("no raw ranker probabilities (pv / ours) in the engine list")
        return isotonic_nonincreasing(logit_blend(pv, ours, w_pv))
    if score_mode == "rank":
        sc = list(entry.get("scores") or [])[:top]
        if len(sc) < top or not all(_finite(x) for x in sc):
            sc = [-float(i) for i in range(top)]
        return isotonic_nonincreasing(sc)
    raise ValueError(f"unknown score_mode {score_mode!r}")


def pop_permutation(s, pop, protected, mu):
    """s, pop, protected: one entry per candidate of the window.  Returns perm with perm[new_slot] = old_slot."""
    n = len(s)
    perm = list(range(n))
    free = [i for i in range(n) if not protected[i]]
    if n < 2 or len(free) < 2 or not mu:
        return perm
    a = np.asarray(s, np.float64)
    z = (a - a.mean()) / (a.std() + 1e-9)
    f = {i: round((float(z[i]) + float(mu) * float(pop[i])) / GRID) * GRID for i in free}
    order = sorted(free, key=lambda i: (-f[i], i))
    for slot, i in zip(free, order):
        perm[slot] = i
    return perm


def run_pop_stage(lists, lookup, mu=0.25, topn=60, protect_lib=0.6, score_mode="logit", w_pv=0.88, key_fn=plain_key):
    """lists: {mid: {smiles, keys, scores, lib, pv, ours, ...}} -> (new lists, stats)."""
    new = {}
    st = {"mu": mu, "topn": topn, "protect_lib": protect_lib, "score_mode": score_mode, "lookup_rows": len(lookup) if lookup is not None else 0,
          "n_molecules": len(lists), "n_candidates": 0, "n_found_plain": 0, "n_found_canon": 0, "n_unknown_to_table": 0,
          "n_with_record": 0, "n_with_pubmed": 0, "n_protected": 0, "n_molecules_with_record": 0, "n_molecules_reordered": 0,
          "n_top1_changed": 0, "n_top25_order_changed": 0, "n_top25_set_changed": 0, "n_isotonic_ties": 0,
          "n_molecules_skipped_no_raw_scores": 0}
    pops, n_scored = [], 0
    for mid, e in lists.items():
        new[mid] = e
        smi = list(e.get("smiles", []))
        keys = list(e.get("keys", []))
        top = min(len(smi), int(topn))
        if top == 0:
            continue
        if protect_lib is not None:
            lib = list(e.get("lib") or [])
            if len(lib) < top:
                raise ValueError(f"{mid}: no per-candidate library similarity in the engine list")
            prot = [float(lib[i]) >= protect_lib for i in range(top)]
        else:
            prot = [False] * top
        cnt = [candidate_counts(smi[i], keys[i] if i < len(keys) else None, lookup, key_fn) for i in range(top)] if lookup is not None \
            else [(0, 0, "unknown")] * top
        pop = [pop_value(a, b) for a, b, _ in cnt]
        st["n_candidates"] += top
        st["n_found_plain"] += sum(h == "plain" for _, _, h in cnt)
        st["n_found_canon"] += sum(h == "canon" for _, _, h in cnt)
        st["n_unknown_to_table"] += sum(h == "unknown" for _, _, h in cnt)
        st["n_with_record"] += sum(p > 0 for p in pop)
        st["n_with_pubmed"] += sum(b > 0 for _, b, _ in cnt)
        st["n_protected"] += sum(prot)
        st["n_molecules_with_record"] += int(any(p > 0 for p in pop))
        pops.extend(pop)
        e2 = dict(e)
        e2["pop"] = [round(p, 4) for p in pop] + [0.0] * (len(smi) - top)
        new[mid] = e2
        if not mu or top < 2:
            continue
        try:
            s = ranker_term(e, top, score_mode, w_pv)
        except ValueError:                             # this molecule has no usable raw scores: its order is kept
            st["n_molecules_skipped_no_raw_scores"] += 1
            continue
        n_scored += 1
        st["n_isotonic_ties"] += sum(1 for i in range(1, top) if s[i] == s[i - 1])
        perm = pop_permutation(s, pop, prot, mu) + list(range(top, len(smi)))
        if perm == list(range(len(smi))):
            continue
        for key in MOVE_KEYS + ("pop",):
            if isinstance(e2.get(key), list) and len(e2[key]) == len(smi):
                e2[key] = [e2[key][i] for i in perm]
        a, b = keys[:25], e2["keys"][:25]
        st["n_molecules_reordered"] += 1
        st["n_top1_changed"] += int(a[:1] != b[:1])
        st["n_top25_order_changed"] += int(a != b)
        st["n_top25_set_changed"] += int(set(a) != set(b))
    if st["n_molecules_skipped_no_raw_scores"] and not n_scored:
        raise ValueError("no molecule has raw ranker probabilities (pv / ours) in the engine list")
    if pops:
        q = np.percentile(pops, [25, 50, 75, 90, 99])
        st["pop_quantiles_25_50_75_90_99"] = [round(float(x), 3) for x in q]
        st["pop_max"] = round(float(max(pops)), 3)
    return new, st

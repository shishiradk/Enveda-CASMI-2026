"""Shared loaders / re-ordering rules for the forward-model bench analysis (research/analysis/e3_diagnosis.md)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "results" / "bench"
FM = B / "fm"
sys.path.insert(0, str(ROOT / "research" / "kaggle_e1" / "e3"))
sys.path.insert(0, str(ROOT / "research" / "forward_model"))
from fm_rerank import rerank_molecule  # noqa: E402  (E3's own rule, unmodified)

NB = 10000


def formula(smi, _c={}):
    if smi not in _c:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(smi) if smi else None
        _c[smi] = CalcMolFormula(m) if m is not None else None
    return _c[smi]


def load(out_dir=None):
    """-> D[scen][mid] = dict(lists={score_set: {...}}, truth, meta...), raw runner outputs."""
    import duckdb
    out_dir = Path(out_dir or FM / "output")
    lists = pickle.load(open(FM / "lists.pkl", "rb"))
    T = {}
    for q, mid, smi, f, correct in duckdb.connect().execute(f"select * from read_parquet('{(B / 'truth.parquet').as_posix()}')").fetchall():
        T[(q, mid)] = dict(smiles=smi, formula=f, correct=set(correct.split(";")))
    inp = json.load(open(FM / "input" / "fm_input.json"))
    meta = json.load(open(FM / "input" / "fm_input_meta.json"))
    res = {t: json.load(open(out_dir / f"fm_{t}.json")) for t in ("gl", "ice")}
    fm = {}
    for ent, v in inp.items():
        qm, tag = ent.split("#")
        d = fm.setdefault(qm, {"smiles": v["smiles"], "ent": {}})
        d["ent"][tag] = {t: (res[t]["scores"].get(ent), res[t]["cosine"].get(ent)) for t in res}
    D = {}
    for scen, per in lists.items():
        q = "S3" if scen == "S3" else "S12"
        R = pickle.load(open(B / "e1" / f"{scen}.pkl", "rb"))
        D[scen] = {}
        for mid, rec in R.items():
            e = dict(n_spec=rec.get("n_spec"), top_sim=rec.get("top_sim"), lists={}, truth=T[(q, mid)], qm=f"{q}|{mid}",
                     meta=meta.get(f"{q}|{mid}"), lib_max=float(np.max(rec["lib"])) if "lib" in rec and len(rec["lib"]) else 0.0)
            ok = T[(q, mid)]["correct"]
            for ss, L in per.items():
                if mid not in L:
                    continue
                l = dict(L[mid])
                l["ok"] = [bool((k is not None and k == rec["truth_canon"]) or k in ok or r in ok) for k, r in zip(l["keys"], l["raw"])]
                l["formula"] = [formula(s) for s in l["smiles"]]
                for extra in ("pv_kf", "ours_kf", "pv", "ours"):
                    if extra in rec["scores"]:
                        l[extra] = [float(rec["scores"][extra][i]) for i in l["idx"]]
                l["lib"] = [float(rec["lib"][i]) for i in l["idx"]]
                f = fm.get(e["qm"])
                if f is not None:
                    pos = {s: i for i, s in enumerate(f["smiles"])}
                    stags = sorted(t for t in f["ent"] if t.startswith("s"))
                    for t in ("gl", "ice"):
                        for kind, w in (("", 0), ("_cos", 1)):
                            c = []
                            for s in l["smiles"]:
                                vals = [f["ent"][tg][t][w][pos[s]] if f["ent"][tg][t][w] is not None else None for tg in stags]
                                c.append(float(np.mean(vals)) if vals and all(v is not None for v in vals) else None)
                            l[t + kind] = c
                e["lists"][ss] = l
            D[scen][mid] = e
    return D, res


def rr_of(ok, perm=None, n=25):
    o = ok if perm is None else [ok[i] for i in perm]
    for i, v in enumerate(o[:n]):
        if v:
            return 1.0 / (i + 1)
    return 0.0


def boot(d, seed=20261002):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmtd(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def e3_perm(l, lam_ice=1.0, lam_gl=1.0, topn=60, ranker="scores"):
    """E3's exact rule (fm_rerank.rerank_molecule)."""
    fwd = {m: l[m] for m in ("ice", "gl") if m in l}
    if not fwd:
        return list(range(len(l["smiles"])))
    perm, _ = rerank_molecule(l["formula"], l[ranker], fwd, {"ice": lam_ice, "gl": lam_gl}, topn)
    return perm


def _z(x, ddof=1):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = np.zeros(len(x))
    if ok.sum() < 2:
        return out, False
    sd = x[ok].std(ddof=ddof)
    if not sd > 1e-12:
        return out, False
    out[ok] = (x[ok] - x[ok].mean()) / sd
    return out, True


def col(l, name):
    return np.array([np.nan if v is None else v for v in l.get(name, [None] * len(l["smiles"]))], float)


def general_perm(l, lam_ice=1.0, lam_gl=1.0, topn=60, ranker="scores", fm_cols=("ice", "gl"), combine="z", zscope="group",
                 topk=0, margin=None, margin_col="pv_kf", scope="group", rrf_k=3.0, rank_term="z", cap=None):
    """Slot-preserving re-ordering with options.
    combine: 'z' (z-score sum) | 'rank' (rank sum) | 'rrf';  zscope: 'group' | 'list' (statistics over the whole top-N)
    topk: re-order only the first k members of each group (0 = all);  margin: re-order a group only when the gap of
    margin_col between its two best members is below this;  scope 'list': one group = the whole top-N.
    rank_term: 'z' ranker enters as z-score | 'logit' z-score of logit(ranker);  cap: clip z_fm to +-cap."""
    n = len(l["smiles"])
    top = min(n, topn)
    perm = list(range(n))
    lams = dict(zip(fm_cols, (lam_ice, lam_gl)))
    cols = {m: col(l, m)[:top] for m in fm_cols if m in l and lams[m]}
    if not cols:
        return perm
    r = np.asarray(l[ranker][:top], float)
    if rank_term == "logit":
        p = np.clip(r, 1e-6, 1 - 1e-6)
        r = np.log(p / (1 - p))
    groups = {}
    if scope == "list":
        groups["*"] = list(range(top))
    else:
        for i in range(top):
            if l["formula"][i] is not None:
                groups.setdefault(l["formula"][i], []).append(i)
    zl = {m: _z(v)[0] for m, v in cols.items()} if zscope == "list" else None
    zrl = _z(r)[0] if zscope == "list" else None
    for mem in groups.values():
        if len(mem) < 2:
            continue
        if margin is not None:
            s = np.sort(np.asarray(l[margin_col], float)[mem])[::-1]
            if s[0] - s[1] >= margin:
                continue
        if topk and len(mem) > topk:
            mem = mem[:topk]
        mem = list(mem)
        covered = False
        f = np.zeros(len(mem))
        for m, v in cols.items():
            x = v[mem]
            okm = np.isfinite(x)
            cov = bool(okm.sum() >= 2 and np.std(x[okm]) > 1e-12)
            if combine == "z":
                z = zl[m][mem] if zscope == "list" else _z(x)[0]
                if cap is not None:
                    z = np.clip(z, -cap, cap)
            else:
                rk = np.full(len(mem), (okm.sum() + 1) / 2.0)
                rk[okm] = np.argsort(np.argsort(-x[okm], kind="mergesort"), kind="mergesort") + 1.0
                z = -rk if combine == "rank" else 1.0 / (rrf_k + rk)
            if cov:
                covered = True
                f += lams[m] * z
        if not covered:
            continue
        if combine == "z":
            zr = zrl[mem] if zscope == "list" else _z(r[mem])[0]
        else:
            rk = np.arange(len(mem)) + 1.0
            zr = -rk if combine == "rank" else 1.0 / (rrf_k + rk)
        fused = np.round((zr + f) / 1e-9) * 1e-9
        order = sorted(range(len(mem)), key=lambda j: (-fused[j], j))
        for slot, j in zip(mem, order):
            perm[slot] = mem[j]
    return perm


def evaluate(D, scen, ss, perm_fn, gate=None):
    """-> (mids, rr_base, rr_new).  Molecules without a list or without forward scores keep their order."""
    mids = sorted(D[scen])
    a, b = [], []
    for m in mids:
        l = D[scen][m]["lists"].get(ss)
        if l is None or not l["smiles"]:
            a.append(0.0); b.append(0.0)
            continue
        base = rr_of(l["ok"])
        a.append(base)
        if "ice" not in l or (gate is not None and not gate(D[scen][m], l)):
            b.append(base)
        else:
            b.append(rr_of(l["ok"], perm_fn(l)))
    return mids, np.array(a), np.array(b)

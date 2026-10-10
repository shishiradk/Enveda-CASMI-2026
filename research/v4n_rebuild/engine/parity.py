"""V3 engine parity: our engine vs the reference v1 engine (local test oracle only, never shipped) on OUR tables with
the same FP model (cft_ho2 through engine.model.CFTBank), on bench S12 molecules.

    python research/v4n_rebuild/engine/parity.py ours   --n 200      # our engine  -> results/v4n/parity/ours.pkl
    python research/v4n_rebuild/engine/parity.py oracle --n 200      # v1 engine   -> results/v4n/parity/oracle.pkl
    python research/v4n_rebuild/engine/parity.py compare             # report      -> results/v4n/parity/report.json

Regimes (REBUILD_SPEC 4): every query uses the S2 mask (all library spectra of every sid sharing the truth's score key
hidden, analog representatives of the truth sid hidden); every 3rd query is run as c3 (truth pool row dropped, so only
generation can recover it).  Generation is on.  Oracle harness adaptations (all "same inputs", none changes logic):
  * np.load memory-maps (RAM);  * adduct_index -> ours (our spec_meta adduct indices);
  * fpnet.prep_peaks -> the CFT peak prep;  * the bank -> CFTBank behind an item converter;
  * raw_fingerprint / fragments_for_smiles -> our tautomer-canonical cfp1 / fragment functions (generated candidates);
  * query spectra get ce_n = 1 so v1's n_merged equals our nm (single 1, merged = #spectra, cap 8).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
TABLES = ROOT / "results" / "v4n" / "tables"
OUT = ROOT / "results" / "v4n" / "parity"
CKPT = ROOT / "models" / "cft_ho2_akriti" / "cft_ho2.pt"
REF = HERE.parent / "ahmed_ref" / "code"
MAX_TARGET = 470.0

import numpy as np
import pandas as pd


def queries(n, seed=0):
    from engine import chem
    from engine.engine import spectra_from_rows, target_of
    q = pd.read_parquet(ROOT / "results/bench/queries_S12.parquet")
    tr = pd.read_parquet(ROOT / "results/bench/truth.parquet")
    tr = tr[tr.qset == "S12"].set_index("mid")
    st = pd.read_parquet(TABLES / "train_structs.parquet", columns=["sid", "inchikey14", "key", "pid"])
    ik2sid = dict(zip(st.inchikey14, st.sid))
    mids = sorted(q.molecule_id.unique())
    tmass = {m: target_of(r) for m, r in q.groupby("molecule_id")}
    mids = [m for m in mids if tmass[m] <= MAX_TARGET]      # pool COCONUT coverage (build_structs.json)
    rng = np.random.default_rng(seed)
    mids = sorted(rng.choice(mids, size=min(n, len(mids)), replace=False))
    out = []
    for j, mid in enumerate(mids):
        rows = q[q.molecule_id == mid]
        tkey = chem.score_key(tr.loc[mid, "smiles"])
        sids = st.sid.values[st.key.values == tkey]
        sid = ik2sid.get(mid, -1)
        sids = np.union1d(sids, [sid] if sid >= 0 else [])
        pid = int(st.pid.values[sid]) if sid >= 0 else -1
        out.append(dict(mid=mid, spectra=spectra_from_rows(rows), target=target_of(rows), truth_key=tkey,
                        mask_sids=sids.astype(np.int64), exclude_sid=int(sid), regime="c3" if j % 3 == 2 else "c2",
                        drop_pid=pid if j % 3 == 2 else -1))
    return out


def run_ours(a):
    from engine.engine import Engine, EngineCfg
    from engine.model import CFTBank
    from engine.tables import Library, Pool
    L, P = Library(str(TABLES)), Pool(str(TABLES))
    bank = CFTBank([str(CKPT)], threads=a.threads)
    E = Engine(L, P, bank, EngineCfg(merged_nm_cap=8, gen_true_steps=False))
    res = []
    Q = queries(a.n)
    for i, q in enumerate(Q):
        ex = np.isin(L.sid, q["mask_sids"])
        t0 = time.time()
        C, X, info = E.run(q["spectra"], q["target"], exclude=ex, exclude_sid=q["exclude_sid"], drop_pid=q["drop_pid"])
        dt = time.time() - t0
        res.append(dict(mid=q["mid"], regime=q["regime"], pid=C.pid, key=C.key, X=X, an=info["analogs"], sec=dt,
                        truth_key=q["truth_key"], gen_combo=C.gen_combo))
        if i % 10 == 0:
            print(f"ours {i}/{len(Q)} n_cand {len(C.pid)} gen {info['n_gen']} {dt:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / "ours.pkl", "wb"))


def run_oracle(a):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REF))
    from engine import chem as our_chem, fragments as our_frag, fingerprint as our_fp
    from engine.engine import FEATURES as OUR_F
    from engine.model import CFTBank
    import casmi.chem as hchem
    import casmi.fpnet as hfp
    import casmi.frag as hfrag
    from casmi.engine import Engine as HEngine, EngineCfg as HCfg, FEATURES as HF
    from casmi.library import Library as HLib
    from casmi.pool import Pool as HPool

    bank = CFTBank([str(CKPT)], threads=a.threads)
    hchem.adduct_index = our_chem.adduct_index
    hfp.prep_peaks = lambda mz, it, prec, **k: bank.prepare(mz, it, prec)
    fpr = our_fp.raw_fingerprinter()

    def raw_fp(smi):
        c = our_chem.canonical(smi)
        return None if c is None else fpr.raw(c[1])
    hchem.raw_fingerprint = raw_fp
    hfrag.fragments_for_smiles = lambda smi, *x, **k: our_frag.fragments_of_smiles(smi).astype(np.float64)

    class OracleBank:
        nbits = bank.nbits

        def logits_el(self, items):
            conv = [dict(mz=x["mz"], it=x["it"], prec=x["prec"], adduct=our_chem.ADDUCT_LIST[x["adduct_ix"]],
                         ce=x["ce"] if x["ce_known"] > 0 else np.nan, nm=x["n_merged"], instrument="timsTOF")
                    for x in items]
            return bank.logits_el(conv)

    if a.equalise:          # same near-zero clamp as our score_library -> isolates genuine logic differences
        import casmi.engine as heng
        s0 = heng.spmod.search

        def search_eq(*x):
            r = s0(*x)
            r[r <= 1e-6] = 0.0
            return r

        class _SP:
            def __getattr__(self, k):
                return search_eq if k == "search" else getattr(hsp, k)
        import casmi.spectra as hsp
        heng.spmod = _SP()
    load0 = np.load
    np.load = lambda p, *x, **k: load0(p, mmap_mode="r") if str(p).endswith(".npy") else load0(p, *x, **k)
    L = HLib(str(TABLES)); P = HPool(str(TABLES))
    tfp = np.load(str(TABLES / "train_fp_sel.npy"))
    np.load = load0
    assert np.array_equal(np.asarray(P.bits, np.int64), bank.fp_bits)
    E = HEngine(L, P, tfp, HCfg(generate=True), OracleBank())
    keep = [HF.index(f) for f in OUR_F]
    res = []
    Q = queries(a.n)
    for i, q in enumerate(Q):
        spectra = [dict(s, ce_n=1) for s in q["spectra"]]
        ex = np.isin(L.sid, q["mask_sids"])
        t0 = time.time()
        C, X, info = E.run(spectra, q["target"], exclude=ex, exclude_sid=q["exclude_sid"], drop_pid=q["drop_pid"])
        dt = time.time() - t0
        an = E.analogs(spectra, q["target"], ex, q["exclude_sid"], -1)
        res.append(dict(mid=q["mid"], regime=q["regime"], pid=C.pid, key=C.key, X=X[:, keep], an=an, sec=dt,
                        truth_key=q["truth_key"]))
        if i % 10 == 0:
            print(f"oracle {i}/{len(Q)} n_cand {len(C.pid)} gen {info.get('n_gen')} {dt:.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / ("oracle_eq.pkl" if a.equalise else "oracle.pkl"), "wb"))


RANK_FEATS = {"lib_max_rank", "ap_rank", "fz_rank", "el_rank", "fr_rank", "sf_rank", "mass_ppm_rank", "gen_rank_fz",
              "lib_x_fz", "ap_x_fz", "agree_lib", "agree_ap", "fr_x_fz"}


def compare(a):
    from engine.engine import FEATURES
    A = {r["mid"]: r for r in pickle.load(open(OUT / "ours.pkl", "rb"))}
    B = {r["mid"]: r for r in pickle.load(open(OUT / a.oracle_file, "rb"))}
    qd = pd.read_parquet(ROOT / "results/bench/queries_S12.parquet", columns=["molecule_id", "adduct", "ionization_mode"])
    tie = set()           # polarity groups whose most frequent adduct is tied (v1 picks by set order)
    for (m, md), g in qd.groupby(["molecule_id", "ionization_mode"]):
        c = g.adduct.value_counts()
        if len(c) > 1 and c.iloc[0] == c.iloc[1]:
            tie.add(m)
    mids = sorted(set(A) & set(B))
    if a.no_ties:
        mids = [m for m in mids if m not in tie]
    nF = len(FEATURES)
    worst = np.zeros(nF); nbad = np.zeros(nF, int); nbad_strict = np.zeros(nF, int)
    rep = dict(oracle_file=a.oracle_file, adduct_tie_queries_excluded=bool(a.no_ties),
               adduct_tie_queries=len(tie & (set(A) & set(B))), queries=len(mids), cand_pool_identical=0, cand_gen_identical=0, cand_identical=0, an_identical=0,
               rows_compared=0, rows_all_equal=0, gen_only_ours=0, gen_only_oracle=0, examples=[])
    t_ours, t_or = [], []
    for m in mids:
        x, y = A[m], B[m]
        t_ours.append(x["sec"]); t_or.append(y["sec"])
        px = [k for p, k in zip(x["pid"], x["key"]) if p >= 0]; py = [k for p, k in zip(y["pid"], y["key"]) if p >= 0]
        gx = [k for p, k in zip(x["pid"], x["key"]) if p < 0]; gy = [k for p, k in zip(y["pid"], y["key"]) if p < 0]
        rep["cand_pool_identical"] += px == py
        rep["cand_gen_identical"] += sorted(gx) == sorted(gy)
        rep["cand_identical"] += (px == py) and sorted(gx) == sorted(gy)
        rep["gen_only_ours"] += len(set(gx) - set(gy)); rep["gen_only_oracle"] += len(set(gy) - set(gx))
        ax = [(s, round(v, 5)) for s, v, *_ in x["an"]]; ay = [(s, round(v, 5)) for s, v, *_ in y["an"]]
        rep["an_identical"] += ax == ay
        iy = {k: i for i, k in enumerate(y["key"])}
        common = [(i, iy[k]) for i, k in enumerate(x["key"]) if k in iy]
        if not common:
            continue
        ix, jy = map(np.array, zip(*common))
        same_set = (px == py) and sorted(gx) == sorted(gy)
        D = np.abs(x["X"][ix].astype(np.float64) - y["X"][jy].astype(np.float64))
        rel = D / np.maximum(1.0, np.abs(y["X"][jy]))
        rep["rows_compared"] += len(ix)
        rep["rows_all_equal"] += int((rel <= 1e-5).all(1).sum())
        if same_set:      # group-level features are only comparable when the candidate sets agree
            worst = np.maximum(worst, rel.max(0))
            nbad += (rel > 1e-5).any(0)
            nbad_strict += (rel > 1e-3).any(0)
        if len(rep["examples"]) < 10 and (not same_set or (rel > 1e-3).any()):
            bad = [FEATURES[f] for f in np.flatnonzero((rel > 1e-3).any(0))]
            rep["examples"].append(dict(mid=m, regime=x["regime"], pool_same=px == py, n_pool=(len(px), len(py)),
                                        gen=(len(gx), len(gy)), gen_overlap=len(set(gx) & set(gy)), bad=bad[:12]))
    rep["features_queries_over_1e-5"] = {FEATURES[f]: int(nbad[f]) for f in np.argsort(-nbad) if nbad[f]}
    rep["features_queries_over_1e-3"] = {FEATURES[f]: int(nbad_strict[f]) for f in np.argsort(-nbad_strict)
                                         if nbad_strict[f]}
    rep["max_rel_diff"] = {FEATURES[f]: float(worst[f]) for f in np.argsort(-worst) if worst[f] > 1e-5}
    rep["sec_per_query"] = dict(ours_mean=float(np.mean(t_ours)), ours_median=float(np.median(t_ours)),
                                oracle_mean=float(np.mean(t_or)), oracle_median=float(np.median(t_or)))
    for name, R in (("ours", A), ("oracle", B)):
        hit = []
        for m in mids:
            r = R[m]
            hit.append(r["truth_key"] in set(r["key"]))
        rep[f"truth_in_cands_{name}"] = float(np.mean(hit))
    json.dump(rep, open(OUT / a.report, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=["ours", "oracle", "compare"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--equalise", action="store_true")
    ap.add_argument("--oracle-file", default="oracle.pkl")
    ap.add_argument("--no-ties", action="store_true")
    ap.add_argument("--report", default="report.json")
    a = ap.parse_args()
    {"ours": run_ours, "oracle": run_oracle, "compare": compare}[a.side](a)


if __name__ == "__main__":
    main()

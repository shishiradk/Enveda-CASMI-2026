"""Local dry run of the E7 kernel's C3 stage on the visible test.parquet (400 molecules). NOT a Kaggle run.

    python research/kaggle_e1/e7/dryrun_e7.py ctx|ana|c3|merge|all [--workers 2] [--limit N]

Stages (outputs in results/kaggle_e7_dry/):
  ctx    e6_ctx.pkl in the kernel's format (e7_e6_channel.py --dump-ctx): target + zlog from e7_e6_channel's own
         targets_and_logits() (CPU; the shipped E6 nets research/kaggle_e1/e6/dataset/e6net_*_full.pt = Kaggle's
         casmi-e6-pcnets; engine code results/kaggle_e6/eng = the E6 notebook's embedded code). PubChem window: the SV
         rows of results/c3/e6_work/windows_5.0ppm.parquet (3,061,067 pairs = Kaggle's n_pairs) with their cached
         fingerprints (results/c3/e6_work/fp), scored fp @ zlog, top 280 per molecule (= the kernel's dump). The
         PubChem table itself is never loaded. Check: local top-60 scores vs Kaggle's results/kaggle_e6/e6_pc.json.
  ana    eng_ana.json in the kernel's format, produced by the E7 eng_runner export code (exec'd from casmi_e7.ipynb)
         on SUBSTITUTE engine records: results/bench/e1/recs_SV.pkl (the same 400 test queries through our engine,
         with their 1,213 exact-duplicate train spectra masked; Kaggle's engine sees those duplicates, so its analog
         lists also contain the true structure at sim ~1). Pool = results/c3np/work/pool_lite.parquet (the bench E.POOL,
         772,653 rows = Kaggle's E.POOL).
  c3     e7_c3_channel.py (from research/kaggle_e1/e7/dataset, deploy assets) in a subprocess, COCONUT from
         external/bench_inputs/coco (= prvsiyan/coconut-casmi26-candidates files).
  merge  E6 lists = results/kaggle_e6/e6_lists.json (Kaggle E6 output, top 25), metric keys recomputed locally;
         the E7 merge function exec'd from casmi_e7.ipynb cell 4; visible checks; dry_report.json.
"""
import argparse, json, os, pickle, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E7 = ROOT / "research" / "kaggle_e1" / "e7"
OUT = ROOT / "results" / "kaggle_e7_dry"
WORK = ROOT / "results" / "c3" / "e6_work"
TEST = ROOT / "test.parquet"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def free_gb():
    o = subprocess.run(["powershell", "-c", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) / 1048576


def stage_ctx(a):
    import duckdb
    import pandas as pd
    sys.path.insert(0, str(E7))
    import e7_e6_channel as C6
    te = pd.read_parquet(TEST)
    te = te[te.molecule_id.notna()]
    te["molecule_id"] = te.molecule_id.astype(str)
    if a.limit:
        te = te[te.molecule_id.isin(sorted(te.molecule_id.unique())[:a.limit])]
    ns = argparse.Namespace(eng_dir=str(ROOT / "results/kaggle_e6/eng"), nets=str(E7.parent / "e6" / "dataset"))
    t0 = time.time()
    L = C6.targets_and_logits(te, ns)
    log(f"ctx: logits for {len(L)} molecules in {time.time() - t0:.0f}s")
    mids = sorted(L)
    mi = {m: i for i, m in enumerate(mids)}
    con = duckdb.connect()
    con.sql("SET memory_limit = '1GB'; SET threads = 2")
    wf = (WORK / "windows_5.0ppm.parquet").as_posix()
    chunks = sorted((WORK / "fp").glob("chunk_*.npz"))
    Z = np.stack([L[m][1] for m in mids]).astype(np.float32)          # (n_mol, 6930)
    parts_m, parts_k, parts_s = [], [], []
    n_rows = 0
    for ci, c in enumerate(chunks):
        z = np.load(c, allow_pickle=True)
        ik = z["ik"].astype(str)
        lo, hi = ik[0], ik[-1]
        w = con.sql(f"SELECT mid, ik FROM read_parquet('{wf}') WHERE scen = 'SV' AND ik >= '{lo}' AND ik <= '{hi}'").fetchnumpy()
        if not len(w["mid"]):
            continue
        sel = np.isin(w["mid"].astype(str), mids)
        wm, wk = w["mid"].astype(str)[sel], w["ik"].astype(str)[sel]
        j = np.searchsorted(ik, wk)
        assert (ik[j] == wk).all()
        ok = z["ok"][j].astype(bool) if "ok" in z.files else np.ones(len(j), bool)
        wm, wk, j = wm[ok], wk[ok], j[ok]
        mm = np.array([mi[m] for m in wm], np.int32)
        sc = np.empty(len(j), np.float32)
        for b in range(0, len(j), 2000):                 # bounded memory: 2000 x 6930 float32 per step
            F = np.unpackbits(z["fp"][j[b:b + 2000]], axis=1)[:, :6930].astype(np.float32)
            sc[b:b + 2000] = np.einsum("ij,ij->i", F, Z[mm[b:b + 2000]])
        parts_m.append(mm); parts_k.append(wk); parts_s.append(sc.astype(np.float32))
        n_rows += len(wk)
        if ci % 10 == 0:
            log(f"ctx: chunk {ci + 1}/{len(chunks)} rows {n_rows:,}")
    M_, K_, S_ = np.concatenate(parts_m), np.concatenate(parts_k), np.concatenate(parts_s)
    o = np.lexsort((K_, -S_, M_))       # per molecule: score desc (ties: ik; the kernel's mergesort tie order may differ)
    M_, K_, S_ = M_[o], K_[o], S_[o]
    starts = np.searchsorted(M_, np.arange(len(mids)))
    ends = np.searchsorted(M_, np.arange(len(mids)), "right")
    keep = np.concatenate([np.arange(s, min(e, s + 280)) for s, e in zip(starts, ends)])
    want = pd.DataFrame({"mid": np.array(mids)[M_[keep]], "ik": K_[keep]})
    con.register("want", want)
    smi = con.sql(f"SELECT DISTINCT w.ik, w.smiles FROM read_parquet('{wf}') w JOIN want USING (mid, ik) "
                  f"WHERE w.scen = 'SV'").df()
    smi_of = dict(zip(smi.ik, smi.smiles))
    DUMP = {}
    for i, m in enumerate(mids):
        s, e = starts[i], ends[i]
        e = min(e, s + 280)
        DUMP[m] = dict(target=float(L[m][0]), zlog=L[m][1], pc=[(K_[r], smi_of[K_[r]], float(S_[r])) for r in range(s, e)])
    meta = dict(source="dryrun_e7 local: SV window cache + local CPU zlog (e6net_*_full.pt)", n_rows=int(n_rows),
                n_mols=len(DUMP))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(mols=DUMP, meta=meta), open(OUT / "e6_ctx.pkl", "wb"), protocol=4)
    # check against Kaggle's E6 lists: scores of the same SMILES, and rank-1 agreement
    K = json.load(open(ROOT / "results/kaggle_e6/e6_pc.json"))
    K.pop("_stats", None)
    diffs, top1, n_in = [], 0, 0
    for m, rec in K.items():
        if m not in DUMP:
            continue
        loc = {s: sc for _, s, sc in DUMP[m]["pc"]}
        for s, sc in zip(rec["smiles"], rec["scores"]):
            if s in loc:
                diffs.append(abs(loc[s] - sc) / max(1.0, abs(sc)))
                n_in += 1
        top1 += int(bool(DUMP[m]["pc"]) and DUMP[m]["pc"][0][1] == rec["smiles"][0])
    chk = dict(n_kaggle_entries=sum(len(v["smiles"]) for v in K.values()), n_found_in_local_top280=n_in,
               rel_score_diff_max=float(np.max(diffs)) if diffs else None,
               rel_score_diff_median=float(np.median(diffs)) if diffs else None,
               top1_smiles_same=top1, n_mols=len(K), window_rows=int(n_rows))
    json.dump(chk, open(OUT / "ctx_check.json", "w"), indent=1)
    log(f"ctx: dump written; check vs Kaggle e6_pc.json {chk}")


def stage_ana(a):
    import pandas as pd
    import ast
    nb = json.load(open(E7 / "casmi_e7.ipynb", encoding="utf-8"))
    src = "".join(nb["cells"][1]["source"])
    lit = [n.args[0].value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "write" and n.args and isinstance(n.args[0], ast.Constant)
           and isinstance(n.args[0].value, str) and n.args[0].value.startswith('"""Our engine')][0]
    i0 = lit.index("    try:   # E7: the engine's library analogs")
    i1 = lit.index("    print('engine lists'")
    block = "\n".join(l[4:] for l in lit[i0:i1].splitlines())          # dedent the main-block snippet
    D = pickle.load(open(ROOT / "results/bench/e1/recs_SV.pkl", "rb"))
    P = pd.read_parquet(ROOT / "results/c3np/work/pool_lite.parquet", columns=["key", "smiles", "mass"])
    E = argparse.Namespace(POOL=dict(keys=P.key.to_numpy(), smiles=P.smiles.to_numpy(), mass=P.mass.to_numpy()))
    recs = D["recs"]
    if a.limit:
        keep = set(sorted(r["mid"] for r in recs)[:a.limit])
        recs = [r for r in recs if r["mid"] in keep]
    OUT.mkdir(parents=True, exist_ok=True)
    ns = dict(recs=recs, E=E, out=str(OUT / "eng_lists_unused.json"), json=json, os=os)
    exec(compile(block, "eng_runner_e7_export", "exec"), ns)
    A = json.load(open(OUT / "eng_ana.json"))
    log(f"ana: {len(A)} molecules exported; median analogs {int(np.median([len(v['ana']) for v in A.values()]))}")


def stage_c3(a):
    while free_gb() < 1.5:
        log(f"c3: waiting for RAM ({free_gb():.2f} GB free)")
        time.sleep(60)
    cmd = [sys.executable, "-u", str(E7 / "dataset" / "e7_c3_channel.py"), str(TEST), str(OUT / "e7_c3.json"),
           "--ctx", str(OUT / "e6_ctx.pkl"), "--ana", str(OUT / "eng_ana.json"), "--assets", str(E7 / "dataset"),
           "--coco", str(ROOT / "external/bench_inputs/coco"), "--workers", str(a.workers), "--budget", str(a.budget)]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    t0 = time.time()
    with open(OUT / "e7_c3.log", "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(OUT))
    log(f"c3: rc {r.returncode} in {time.time() - t0:.0f}s")


def _ck(s):
    sys.path.insert(0, str(E7))
    import e7_c3_channel as C
    return C.canon_key(s)


def stage_merge(a):
    import pandas as pd
    nb = json.load(open(E7 / "casmi_e7.ipynb", encoding="utf-8"))
    cell = "".join(nb["cells"][4]["source"])
    i0 = cell.index("def e7_merge(")
    i1 = cell.index("\ntry:\n    ENG, E7_MERGE = e7_merge(")
    ns = {}
    exec(compile(cell[i0:i1], "e7_merge_cell", "exec"), ns)
    e7_merge = ns["e7_merge"]
    L6 = json.load(open(ROOT / "results/kaggle_e6/e6_lists.json"))
    smis = sorted({s for v in L6.values() for s in v["smiles"]})
    cache = OUT / "e6_keys.pkl"
    KC = pickle.load(open(cache, "rb")) if cache.exists() else {}
    need = [s for s in smis if s not in KC]
    if need:
        with Pool(a.workers) as mp:
            KC.update(zip(need, mp.map(_ck, need, chunksize=32)))
        pickle.dump(KC, open(cache, "wb"))
    ENG = {m: dict(smiles=v["smiles"], keys=[KC[s] for s in v["smiles"]]) for m, v in L6.items()}
    C3 = json.load(open(OUT / "e7_c3.json"))
    stats, info = C3.pop("_stats", {}), C3.pop("_info", {})
    sub = pd.read_csv(ROOT / "sample_submission.csv")
    E7L, mst = e7_merge(ENG, C3)
    ids = [str(m) for m in sub.molecule_id]
    l6 = {m: list(dict.fromkeys(ENG[m]["smiles"]))[:25] for m in ids}
    l7 = {m: list(dict.fromkeys(E7L[m]["smiles"]))[:25] for m in ids}
    pos_ins = {}
    for m in ids:
        ins = [i + 1 for i, s in enumerate(l7[m]) if s not in set(l6[m])]
        if ins:
            pos_ins[m] = ins
    from rdkit import Chem
    bad = sum(1 for m in ids for s in l7[m] if Chem.MolFromSmiles(s) is None)
    rep = dict(c3_stats=stats, merge_stats=mst,
               n_rows=len(ids), top3_unchanged=sum(l7[m][:3] == l6[m][:3] for m in ids),
               rows_25_unique=sum(len(l7[m]) == 25 == len(set(l7[m])) for m in ids),
               rows_25_unique_by_key=sum(len(set(KC.get(s) or _ck(s) for s in l7[m])) == 25 for m in ids),
               rows_with_c3_insertions=len(pos_ins),
               n_inserted=sum(len(v) for v in pos_ins.values()),
               insert_positions_hist={str(p): sum(p in v for v in pos_ins.values()) for p in range(1, 26)},
               rows_with_5_insertions=sum(len(v) == 5 for v in pos_ins.values()),
               unparsable_smiles=bad,
               e6_rank_lost=sum(len(set(l6[m]) - set(l7[m])) for m in ids),
               status_counts=pd.Series([v.get("status") for v in info.values()]).value_counts().to_dict(),
               mol_sec_mean=stats.get("mol_sec_mean"), mol_sec_max=stats.get("mol_sec_max"))
    # the visible truth (enveda-180 copies): E6 / E7 MRR on the visible molecules (should be identical in top-3)
    T = pd.read_parquet(ROOT / "results/bench/truth_SV.parquet")
    corr = {r.mid: set(r.correct.split(";")) for r in T.itertuples()}

    def mrr(L):
        rr = []
        for m in ids:
            ks = [KC.get(s) or _ck(s) for s in L[m]]
            r = next((i + 1 for i, k in enumerate(ks) if k in corr.get(m, set())), 0)
            rr.append(1 / r if r else 0)
        return round(float(np.mean(rr)), 4)
    rep.update(visible_mrr_e6=mrr(l6), visible_mrr_e7=mrr(l7))
    sub7 = pd.DataFrame([(m, ";".join(l7[str(m)]) if l7[str(m)] else "CCO") for m in sub.molecule_id],
                        columns=["molecule_id", "smiles"])
    sub7.to_csv(OUT / "submission_e7_dry.csv", index=False)
    json.dump({m: dict(smiles=l7[m]) for m in ids}, open(OUT / "e7_lists_dry.json", "w"))
    json.dump(rep, open(OUT / "dry_report.json", "w"), indent=1)
    log(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ctx", "ana", "c3", "merge", "all"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=6 * 3600)
    a = ap.parse_args()
    for s in (["ctx", "ana", "c3", "merge"] if a.stage == "all" else [a.stage]):
        globals()["stage_" + s](a)


if __name__ == "__main__":
    main()

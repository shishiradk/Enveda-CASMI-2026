"""20-molecule parity check of the deployed E7 C3 channel (context builders + generators + rr_bt) on the C3NP bench.

    python research/kaggle_e1/e7/parity_e7.py bench|deploy [--n 20]

The deployed code is the dataset copy (research/kaggle_e1/e7/dataset/*.py). Inputs that can be matched to the bench
(results/c3np/context.pkl was built by research/scripts/c3np_build.py from these exact raw inputs):
  target, zlog : context.pkl (= the ho1 bench records; the deployed channel passes the E6 dump's values through)
  analogs      : raw engine `ana` lists of the ho1 bench records (results/bench/ho1/recs_{S2,S3}.pkl) -> converted by
                 the E7 eng_runner export code (exec'd from casmi_e7.ipynb) with the bench E.POOL (pool_lite) ->
                 e7_c3_channel.build_analogs(exclude = the molecule's forbidden keys)
  window       : PubChem rows of the bench window cache (results/c3/e6_work, cached ho1 scores, full window) +
                 e7_c3_channel.coco_windows (external/bench_inputs/coco, unlimited head) ->
                 e7_c3_channel.build_window(exclude = forbidden keys)
bench : bench-mode generators (train_pkg/data pool with held keys removed, leak-clean rules, forbidden keys) via
        e7_c3_channel.run_one -> compare ctx with context.pkl and the biotransform / mmp_edit / rr_bt lists with
        results/c3gen/full (biotransform also with the E7 work-budget full run results/c3gen/e7/bt_full_e7_c*.json).
deploy: the same raw inputs, deploy mode (dataset assets, full rules min_freq 3, full pool, forbidden = {}), to show
        what deploy mode changes (no truth key is ever excluded; the full pool contains every bench truth, so mmp_edit
        drops them as known structures by design). Writes results/kaggle_e7_dry/parity_<mode>.json.
"""
import argparse, glob, hashlib, json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E7 = ROOT / "research" / "kaggle_e1" / "e7"
DS = E7 / "dataset"
OUT = ROOT / "results" / "kaggle_e7_dry"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def pick(n):
    """n bench mids: half in-sample (rows 0-59, 300-359), half rest, spread over the bench; skipping rows where the
    old biotransform time guard fired (their old lists are machine dependent)."""
    import pandas as pd
    M = pd.read_parquet(ROOT / "results/c3np/molecules.parquet")
    fired = set()
    for f in glob.glob(str(ROOT / "results/c3gen/full/biotransform_c[0-9][0-9][0-9]_dump.pkl")):
        d = pickle.load(open(f, "rb"))
        fired |= {m for m, i in d["infos"].items() if i.get("truncated") in ("time", "stage2_time")}
    rows = [i for i in range(len(M)) if M.mid[i] not in fired]
    ins = [i for i in rows if i < 60 or 300 <= i < 360]
    rest = [i for i in rows if not (i < 60 or 300 <= i < 360)]
    sel = [ins[j] for j in np.linspace(0, len(ins) - 1, n // 2).astype(int)] + \
          [rest[j] for j in np.linspace(0, len(rest) - 1, n - n // 2).astype(int)]
    return M.iloc[sel].reset_index(drop=True)


def raw_inputs(M):
    import duckdb
    import pandas as pd
    import ast
    sys.path.insert(0, str(DS))
    import e7_c3_channel as C
    ctx = pickle.load(open(ROOT / "results/c3np/context.pkl", "rb"))["mols"]
    F = pd.read_parquet(ROOT / "results/c3np/forbidden.parquet")
    forb = F.groupby("mid").key.agg(set).to_dict()
    # engine analog export code, exactly as shipped in casmi_e7.ipynb (eng_runner.py)
    nb = json.load(open(E7 / "casmi_e7.ipynb", encoding="utf-8"))
    src = "".join(nb["cells"][1]["source"])
    lit = [n.args[0].value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "write" and n.args and isinstance(n.args[0], ast.Constant)
           and isinstance(n.args[0].value, str) and n.args[0].value.startswith('"""Our engine')][0]
    i0, i1 = lit.index("    try:   # E7: the engine's library analogs"), lit.index("    print('engine lists'")
    block = "\n".join(l[4:] for l in lit[i0:i1].splitlines())
    P = pd.read_parquet(ROOT / "results/c3np/work/pool_lite.parquet", columns=["key", "smiles", "mass"])
    E = argparse.Namespace(POOL=dict(keys=P.key.to_numpy(), smiles=P.smiles.to_numpy(), mass=P.mass.to_numpy()))
    recs = []
    for scen in sorted(set(M.scen)):
        D = pickle.load(open(ROOT / f"results/bench/ho1/recs_{scen}.pkl", "rb"))
        want = set(M.mid[M.scen == scen])
        recs += [r for r in D["recs"] if r["mid"] in want]
        del D
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "parity_tmp"
    tmp.mkdir(exist_ok=True)
    exec(compile(block, "eng_runner_e7_export", "exec"), dict(recs=recs, E=E, out=str(tmp / "x.json"), json=json, os=__import__("os")))
    ANA = json.load(open(tmp / "eng_ana.json"))
    # PubChem window rows (bench cache, ho1 scores) and COCONUT window
    con = duckdb.connect()
    con.sql("SET memory_limit = '1GB'; SET threads = 2")
    con.register("want", pd.DataFrame({"scen": M.scen, "mid": M.mid}))
    wf = (ROOT / "results/c3/e6_work/windows_5.0ppm.parquet").as_posix()
    w = con.sql(f"SELECT w.file_row_number AS row, w.mid, w.ik, w.smiles FROM read_parquet('{wf}', file_row_number = true) w "
                f"JOIN want USING (scen, mid)").df()
    sc = np.load(ROOT / "results/c3/e6_work/scores_5.0ppm.npy", mmap_mode="r")
    w["score"] = np.asarray(sc[w.row.values], np.float32)
    PC = {m: list(zip(g.ik, g.smiles, g.score.astype(float))) for m, g in w.groupby("mid")}
    tz = {m: (ctx[m]["target"], ctx[m]["zlog"]) for m in M.mid}
    CO = C.coco_windows(tz, str(ROOT / "external/bench_inputs/coco"), head=10 ** 9)
    raws = {}
    for i, m in enumerate(M.mid):
        raws[m] = dict(qid="c3_%04d" % i, target=ctx[m]["target"], zlog=ctx[m]["zlog"], adducts=ctx[m]["adducts"],
                       ana=ANA[m]["ana"], pc=PC.get(m, []), coco=CO.get(m, []), spectra=[])
    return raws, ctx, forb, ANA


def same_list(a, b, keys):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for k in keys:
            u, v = x.get(k), y.get(k)
            if isinstance(u, float) or isinstance(v, float):
                if u is None or v is None or abs(float(u) - float(v)) > 1e-4 * max(1.0, abs(float(v))):
                    return False
            elif u != v:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "deploy"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    import pandas as pd
    sys.path.insert(0, str(DS))
    for f in ("biotransform.py", "mmp_edit.py", "assets.py"):
        h = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in (DS / f, ROOT / "research/c3gen" / f)]
        assert h[0] == h[1], f"dataset copy of {f} differs from research/c3gen"
    import e7_c3_channel as C
    import biotransform as B
    import mmp_edit as Mm
    M = pick(a.n)
    t0 = time.time()
    raws, ctx, forb, ANA = raw_inputs(M)
    log(f"raw inputs for {len(raws)} molecules in {time.time() - t0:.0f}s")
    if a.mode == "bench":
        Mm._rd(); Mm.set_rules(pd.read_parquet(Mm.CLEAN_RULES)); Mm._ensure_assets()
    else:
        B.set_asset_dir(str(DS)); Mm.set_asset_dir(str(DS))
        B.T_GEN, B.T_TOTAL = C.FAILSAFE["bt"]; Mm.T_GEN, Mm.T_ALL = C.FAILSAFE["mmp"]
        Mm._ensure_assets()
    old_bt = json.load(open(ROOT / "results/c3gen/full/biotransform_full.json"))
    old_mm = json.load(open(ROOT / "results/c3gen/full/mmp_edit_full.json"))
    old_rr = json.load(open(ROOT / "results/c3gen/full/combo_rr_bt.json"))
    new_bt = {}
    for f in glob.glob(str(ROOT / "results/c3gen/e7/bt_full_e7_c[0-9][0-9][0-9].json")):
        new_bt.update(json.load(open(f)))
    rows = []
    for m in M.mid:
        f = forb.get(m, set()) if a.mode == "bench" else set()
        t = time.time()
        rec, info, x = C.run_one(raws[m], ntop=25, exclude=frozenset(f), B=B, M=Mm, forbidden=frozenset(f))
        bt, mm, rr, cx = [s for s, _, _ in x["bt"]], [s for s, _, _ in x["mmp"]], [s for s, *_ in x["rr"]], x["ctx"]
        r = dict(info, mid=m, subset=M.set_index("mid").subset[m], sec=round(time.time() - t, 1))
        if a.mode == "bench":
            r.update(
                analogs_same=same_list(cx["analogs"], ctx[m]["analogs"], ("smiles", "ik14", "ckey", "sim", "dmass")),
                window_same=same_list(cx["window"], ctx[m]["window"], ("ik", "ckey", "smiles", "score", "src")),
                n_analogs=(len(cx["analogs"]), len(ctx[m]["analogs"])), n_window=(len(cx["window"]), len(ctx[m]["window"])),
                mmp_same_as_full=mm == old_mm.get(m), bt_same_as_old_full=bt == old_bt.get(m),
                bt_same_as_e7_full=(bt == new_bt.get(m)) if m in new_bt else None,
                rr_same_as_full=rr == old_rr.get(m),
                rr_top25_same_as_full=rr[:25] == (old_rr.get(m) or [])[:25])
        else:
            r.update(top25_overlap_with_bench_rr=len(set(rr[:25]) & set((old_rr.get(m) or [])[:25])),
                     n_rr=len(rr))
        rows.append(r)
        log(json.dumps(r))
    D = pd.DataFrame(rows)
    summ = {"mode": a.mode, "n": len(D), "sec_mean": round(float(D.sec.mean()), 1), "sec_max": float(D.sec.max())}
    for c in D.columns:
        if D[c].dtype == bool or c.endswith("_same") or "same_as" in c:
            summ[c] = int(D[c].fillna(False).astype(bool).sum())
    json.dump(dict(summary=summ, rows=rows), open(OUT / f"parity_{a.mode}.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

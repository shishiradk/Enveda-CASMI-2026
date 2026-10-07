"""E3 forward-model stage (ours, MIT): score same-formula candidates of the engine's top list with
GLACIER and ICEBERG (ms-pred, MIT, MassSpecGym weights) in subprocesses that use their own RDKit
2025.03.x site directory, then re-order inside formula groups (fm_rerank).

``run_fm_stage`` never changes its input.  It returns (new_lists, stats); any exception is left to
the caller, which keeps the engine lists unchanged.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

from fm_rerank import list_changes, rerank_molecule

ADDUCTS = ("[M+H]+", "[M+Na]+")      # the only adducts in MassSpecGym, the models' training data
INSTRUMENT = "QTOF"                  # test spectra are all timsTOF (a Q-TOF); tokens seen in training: Orbitrap, QTOF
MODELS = ("gl", "ice")               # run order: GLACIER (cheap) first, ICEBERG with what is left
MODEL_NAME = {"gl": "glacier", "ice": "iceberg"}


def log(*a):
    print("[fm]", *a, flush=True)


def formulas_of(smiles):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import rdBase
    _quiet = rdBase.BlockLogs()  # noqa: F841 - silences parse errors until it goes out of scope
    out = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            out.append(CalcMolFormula(m) if m is not None else None)
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def locate(root_hint=None, work=None):
    """Find the casmi-fm-runner dataset (directory that holds code/ice_runner.py)."""
    base = root_hint or "/kaggle/input"
    hits = sorted(glob.glob(os.path.join(base, "**", "ice_runner.py"), recursive=True), key=len)
    if not hits and work:  # the dataset folders are still zip archives: unpack them
        import zipfile
        for cz in sorted(glob.glob(os.path.join(base, "**", "code.zip"), recursive=True), key=len):
            if any(n.endswith("ice_runner.py") for n in zipfile.ZipFile(cz).namelist()):
                dst = os.path.join(work, "ds")
                shutil.rmtree(dst, ignore_errors=True)
                for z in glob.glob(os.path.join(os.path.dirname(cz), "*.zip")):
                    zipfile.ZipFile(z).extractall(os.path.join(dst, os.path.splitext(os.path.basename(z))[0]))
                hits = sorted(glob.glob(os.path.join(dst, "**", "ice_runner.py"), recursive=True), key=len)
                break
    if not hits:
        raise FileNotFoundError("ice_runner.py (dataset casmi-fm-runner) not found")
    code = os.path.dirname(hits[0])
    root = os.path.dirname(code)

    def one(pattern):
        h = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True), key=len)
        if not h:
            raise FileNotFoundError(pattern)
        return h[0]
    src = os.path.dirname(os.path.dirname(one(os.path.join("ms_pred", "__init__.py"))))
    return dict(root=root, code=code, runner=hits[0], ms_pred_src=src, gen=one("iceberg_gen.pt"),
                inten=one("iceberg_inten.pt"), glacier=one("glacier.pt"))


def install_site(root, work, python_exe):
    """RDKit 2025.03.x into its own directory (the notebook process keeps the metric's RDKit)."""
    tag = "cp%d%d" % sys.version_info[:2] if python_exe == sys.executable else "cp"
    site, extra = os.path.join(work, "site_rdkit2025"), os.path.join(work, "site_extra")
    whl = [w for w in glob.glob(os.path.join(root, "**", "rdkit-2025.3.*.whl"), recursive=True) if tag in os.path.basename(w)]
    if not whl:
        raise FileNotFoundError(f"no rdkit-2025.3.x wheel for {tag}")
    for d, wheels in ((site, whl[:1]), (extra, sorted(glob.glob(os.path.join(root, "**", "*-py3-none-any.whl"), recursive=True)))):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        if not wheels:
            continue
        r = subprocess.run([python_exe, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", d] + wheels,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("pip install failed: " + r.stderr[-1500:])
    chk = subprocess.run([python_exe, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import rdkit; "
                          "from rdkit import Chem; assert Chem.MolFromSmiles('CCO') is not None; print(rdkit.__version__)", site],
                         capture_output=True, text=True)
    ver = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else ""
    if not ver.startswith("2025.03"):
        raise RuntimeError(f"pinned RDKit not importable: {ver!r} {chk.stderr[-1500:]}")
    return site, extra, ver


def select(lists, covered_mids, topn):
    """Per molecule: formulas of the top list and the positions that share a formula with another one."""
    sel = {}
    for mid, e in lists.items():
        smi = list(e.get("smiles", []))[:topn]
        if mid not in covered_mids or len(smi) < 2:
            continue
        f = formulas_of(smi)
        cnt = {}
        for x in f:
            if x is not None:
                cnt[x] = cnt.get(x, 0) + 1
        send = [i for i, x in enumerate(f) if x is not None and cnt[x] >= 2]
        if send:
            sel[mid] = dict(formulas=f, send=send)
    return sel


def run_model(tag, paths, inp, out, test_parquet, budget, site, extra, python_exe, device, work, extra_args=()):
    """One runner subprocess.  Returns the parsed output (possibly partial) or None."""
    cmd = [python_exe, paths["runner"], "--input", inp, "--output", out, "--model", MODEL_NAME[tag],
           "--spectra-parquet", test_parquet, "--ms-pred-src", paths["ms_pred_src"], "--extra-path", extra,
           "--device", device, "--budget-sec", str(int(budget)), "--adducts", ",".join(ADDUCTS),
           "--instrument", INSTRUMENT, "--default-ces", "20,40,60", "--save-every-sec", "60"]
    if site:
        cmd += ["--prepend-path", site]
    cmd += ["--glacier-ckpt", paths["glacier"]] if tag == "gl" else ["--gen-ckpt", paths["gen"], "--inten-ckpt", paths["inten"]]
    cmd += list(extra_args)
    logf = os.path.join(work, f"fm_{tag}.log")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    try:
        with open(logf, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=budget + 240, env=env, cwd=work).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout (killed)"
    try:
        tail = open(logf, errors="replace").read()[-2500:]
    except OSError:
        tail = ""
    log(f"{tag}: runner exit {rc} after {time.time() - t0:.0f}s (budget {budget:.0f}s); log tail:\n{tail}")
    try:
        res = json.load(open(out))
    except Exception as exc:  # noqa: BLE001
        log(f"{tag}: NO OUTPUT ({exc!r})")
        return None
    res["meta"]["wall_sec"] = round(time.time() - t0, 1)
    res["meta"]["exit"] = rc
    return res


def run_fm_stage(lists, test_parquet, work, budget_sec=5400.0, topn=60, lam_ice=1.0, lam_gl=1.0, gl_share=0.4,
                 root_hint=None, python_exe=None, site=None, extra=None, device="auto", extra_args=(), need_rdkit="2025.03"):
    import pandas as pd
    t_start = time.time()
    python_exe = python_exe or sys.executable
    test_parquet = os.path.abspath(test_parquet)
    os.makedirs(work, exist_ok=True)
    paths = locate(root_hint, work)
    if site is None:
        site, extra, ver = install_site(paths["root"], work, python_exe)
        log("pinned RDKit", ver, "in", site)
    lams = {"ice": float(lam_ice), "gl": float(lam_gl)}

    te = pd.read_parquet(test_parquet, columns=["molecule_id", "adduct"])
    covered = set(te.loc[te.adduct.astype(str).isin(ADDUCTS), "molecule_id"].astype(str))
    sel = select(lists, covered, topn)
    # weakest library match first: if the budget runs out, the well-matched molecules are the ones left unscored
    order = sorted(sel, key=lambda m: (float(lists[m].get("lib_max") or 0.0), m))
    inp = os.path.join(work, "fm_input.json")
    with open(inp, "w") as fh:
        json.dump({m: {"smiles": [lists[m]["smiles"][i] for i in sel[m]["send"]]} for m in order}, fh)
    n_sent = sum(len(sel[m]["send"]) for m in order)
    stats = {"n_molecules": len(lists), "n_without_covered_spectrum": sum(1 for m in lists if m not in covered),
             "n_covered_but_no_formula_group": sum(1 for m in lists if m in covered and m not in sel),
             "n_molecules_sent": len(order), "n_candidates_sent": n_sent, "topn": topn, "lams": lams,
             "budget_sec": budget_sec, "adducts": list(ADDUCTS), "instrument": INSTRUMENT, "models": {}}
    log(f"{len(lists)} molecules: {stats['n_without_covered_spectrum']} without a covered spectrum, "
        f"{stats['n_covered_but_no_formula_group']} covered but no formula group >= 2, {len(order)} sent with {n_sent} candidates")

    fwd = {}
    for k, tag in enumerate(MODELS):
        if not lams[tag] or not order:
            continue
        left = budget_sec - (time.time() - t_start) - 30.0
        budget = min(left, gl_share * budget_sec) if k == 0 else left
        if budget < 120:
            log(f"{tag}: SKIPPED, only {budget:.0f}s of budget left")
            stats["models"][tag] = {"status": "skipped_no_budget"}
            continue
        res = run_model(tag, paths, inp, os.path.join(work, f"fm_{tag}.json"), test_parquet, budget, site, extra,
                        python_exe, device, work, extra_args)
        if res is None:
            stats["models"][tag] = {"status": "no_output"}
            continue
        meta = res.get("meta", {})
        ms = {k2: meta.get(k2) for k2 in ("status", "device", "rdkit", "torch", "batch_size", "load_sec", "elapsed_sec", "wall_sec",
                                          "n_predictions", "predict_sec", "pred_per_sec", "molecules_done", "n_molecules_covered",
                                          "out_of_time", "reasons", "exit")}
        if meta.get("error") or meta.get("first_predict_error"):
            log(f"{tag}: runner error:\n{meta.get('error') or meta.get('first_predict_error')}")
            ms["error"] = (meta.get("error") or meta.get("first_predict_error"))[-600:]
        sc = res.get("scores", {})
        if need_rdkit and not str(meta.get("rdkit", "")).startswith(need_rdkit):
            log(f"{tag}: SCORES DISCARDED, runner used RDKit {meta.get('rdkit')!r} instead of {need_rdkit}.x")
            ms["status"] = f"discarded_wrong_rdkit({meta.get('status')})"
            sc = {}
        got = {}
        for m in order:
            v = sc.get(m)
            if isinstance(v, list) and len(v) == len(sel[m]["send"]):
                got[m] = [x if isinstance(x, (int, float)) else None for x in v]
        n_scored = sum(1 for v in got.values() for x in v if x is not None)
        mol_scored = sum(1 for v in got.values() if any(x is not None for x in v))
        ms.update(n_candidates_scored=n_scored, n_candidates_unscored=n_sent - n_scored, n_molecules_with_scores=mol_scored,
                  n_molecules_without_scores=len(order) - mol_scored)
        stats["models"][tag] = ms
        fwd[tag] = got
        log(f"{tag}: status {ms['status']}, {n_scored}/{n_sent} candidates scored in {mol_scored}/{len(order)} molecules, "
            f"{ms.get('n_predictions')} predictions, {ms.get('pred_per_sec')} pred/s, wall {ms.get('wall_sec')}s")

    new = {}
    ch = {"top1": 0, "order": 0, "set": 0}
    tot = {"groups": 0, "groups_covered": 0, "groups_changed": 0, "members_covered": 0}
    n_any = 0
    for mid, e in lists.items():
        new[mid] = e
        if mid not in sel:
            continue
        n = len(e["smiles"])
        f = sel[mid]["formulas"] + [None] * (n - len(sel[mid]["formulas"]))
        per = {}
        for tag, got in fwd.items():
            if mid in got:
                col = [None] * n
                for i, x in zip(sel[mid]["send"], got[mid]):
                    col[i] = x
                per[tag] = col
        if not per:
            continue
        perm, info = rerank_molecule(f, e.get("scores") or [], per, lams, topn)
        if sorted(perm) != list(range(n)):
            raise AssertionError(f"not a permutation for {mid}")
        for k2 in tot:
            tot[k2] += info[k2]
        n_any += int(info["groups_covered"] > 0)
        if perm == list(range(n)):
            continue
        e2 = dict(e)
        for key in ("smiles", "keys", "scores"):
            if isinstance(e.get(key), list) and len(e[key]) == n:
                e2[key] = [e[key][i] for i in perm]
        new[mid] = e2
        c = list_changes(e["keys"], e2["keys"], 25)
        for k2 in ch:
            ch[k2] += int(c[k2])
    stats.update(n_molecules_with_covered_group=n_any, formula_groups=tot, n_top1_changed=ch["top1"],
                 n_top25_order_changed=ch["order"], n_top25_set_changed=ch["set"], stage_sec=round(time.time() - t_start, 1))
    log(f"re-ordered: top-1 changed in {ch['top1']} molecules, top-25 order changed in {ch['order']}, "
        f"top-25 membership changed in {ch['set']} (of {len(lists)}); stage {stats['stage_sec']}s")
    return new, stats

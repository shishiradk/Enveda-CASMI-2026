#!/usr/bin/env python
"""ICEBERG / GLACIER forward-model scorer (our own runner; MIT).

For every molecule: predict an MS/MS spectrum for each candidate SMILES with
ICEBERG 2.1 (upstream ms-pred, MIT, MassSpecGym ``msg_all`` checkpoint) and
score it against the observed spectra of that molecule.

Standalone subprocess:

    python ice_runner.py --input cands.json --output scores.json \
        --gen-ckpt .../gen/best.ckpt --inten-ckpt .../inten_contr/best.ckpt \
        [--spectra-parquet test.parquet] [--ms-pred-src .../ms-pred/src] \
        [--device auto|cpu|cuda] [--budget-sec 2700]

Input JSON
    {molecule_id: {"smiles": [s1, s2, ...],          # candidates, ranker order
                   "adduct": "[M+H]+",               # optional default adduct
                   "spectra": [                      # optional if --spectra-parquet
                       {"mzs": [...], "intensities": [...],
                        "adduct": "[M+H]+", "precursor_mz": 301.07,
                        "collision_energies": [20, 40, 60]}, ...]}}
    Molecules are processed in the order of the file (put the ones that matter
    most first: when the time budget runs out the rest stay unscored).

Output JSON
    {"scores":  {molecule_id: [float | null, ...]},  # aligned with "smiles"
     "cosine":  {molecule_id: [float | null, ...]},
     "meta":    {...}}
    ``null`` = not scored (uncovered adduct, invalid structure, budget ran out,
    error).  The caller must leave such candidates where the ranker put them.

Score
    A spectrum is "covered" when its adduct is in --adducts (default [M+H]+ and
    [M+Na]+, the only adducts in MassSpecGym).  For a covered spectrum the
    candidate is predicted at each of the spectrum's collision energies
    (rounded to --ce-round eV, at least --ce-min; --default-ces if unknown) with
    the instrument token --instrument; the per-energy predictions are scaled to
    max 1, summed, merged at 4 decimals and cut to the --top-k strongest peaks.
    Predicted and observed peaks are matched one-to-one (greedy by intensity
    product) within max(--tol-da, --tol-ppm).  "scores" is the unweighted
    spectral-entropy similarity, "cosine" the square-root-intensity cosine;
    both are averaged over the molecule's covered spectra.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

T0 = time.time()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", choices=["iceberg", "glacier"], default="iceberg")
    p.add_argument("--gen-ckpt", default=None, help="ICEBERG fragment-generation checkpoint")
    p.add_argument("--inten-ckpt", default=None, help="ICEBERG intensity checkpoint")
    p.add_argument("--glacier-ckpt", default=None, help="GLACIER checkpoint (with --model glacier)")
    p.add_argument("--spectra-parquet", default=None,
                   help="parquet with molecule_id, ms2_mzs, ms2_normalized_intensities, adduct, precursor_mz, collision_energy_ev")
    p.add_argument("--ms-pred-src", default=None)
    p.add_argument("--prepend-path", action="append", default=[],
                   help="directories put in FRONT of sys.path (e.g. a pinned RDKit build)")
    p.add_argument("--extra-path", action="append", default=[],
                   help="directories appended to sys.path (pure-python deps)")
    p.add_argument("--real-dgl", action="store_true", help="use installed DGL/torch_scatter/lightning instead of our shims")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--budget-sec", type=float, default=1e9, help="wall-clock budget, counted from process start")
    p.add_argument("--safety-sec", type=float, default=20.0, help="stop this long before the budget ends")
    p.add_argument("--batch-size", type=int, default=0, help="0 = 64 on cuda, 16 on cpu")
    p.add_argument("--max-nodes", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--adducts", default="[M+H]+,[M+Na]+")
    p.add_argument("--instrument", default="QTOF")
    p.add_argument("--default-ces", default="20,40,60")
    p.add_argument("--ce-round", type=float, default=5.0)
    p.add_argument("--ce-min", type=float, default=5.0)
    p.add_argument("--max-ces", type=int, default=3, help="at most this many energies per spectrum (evenly thinned)")
    p.add_argument("--max-atoms", type=int, default=120, help="skip candidates with more heavy atoms")
    p.add_argument("--max-cands", type=int, default=0, help="score only the first N candidates per molecule (0 = all)")
    p.add_argument("--tol-da", type=float, default=0.01)
    p.add_argument("--tol-ppm", type=float, default=20.0)
    p.add_argument("--obs-top-k", type=int, default=0, help="keep only the N strongest observed peaks (0 = all)")
    p.add_argument("--obs-min-rel", type=float, default=0.0, help="drop observed peaks below this fraction of the base peak")
    p.add_argument("--save-every-sec", type=float, default=120.0)
    p.add_argument("--dump-pred", default=None, help="optional JSON with the predicted spectra")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def log(args, *msg):
    if not args.quiet:
        print(f"[ice_runner {time.time() - T0:7.1f}s]", *msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# spectra helpers (numpy only)
# --------------------------------------------------------------------------
def clean_spectrum(mzs, intens, np, top_k=0, min_rel=0.0):
    mz = np.asarray(mzs, dtype=np.float64).reshape(-1)
    it = np.asarray(intens, dtype=np.float64).reshape(-1)
    n = min(mz.size, it.size)
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0)
    mz, it = mz[ok], it[ok]
    if mz.size == 0:
        return mz, it
    if min_rel > 0:
        keep = it >= min_rel * it.max()
        mz, it = mz[keep], it[keep]
    if top_k and mz.size > top_k:
        keep = np.argsort(-it)[:top_k]
        mz, it = mz[keep], it[keep]
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    return mz, it / it.max()


def merge_predictions(specs, np, top_k=100, decimals=4):
    """specs: list of (n, 2) arrays [m/z, intensity]; each is scaled to max 1 and summed."""
    acc = {}
    for sp in specs:
        if sp is None or len(sp) == 0:
            continue
        mz, it = sp[:, 0], sp[:, 1]
        ok = it > 0
        mz, it = mz[ok], it[ok]
        if mz.size == 0:
            continue
        it = it / it.max()
        for m, i in zip(np.round(mz, decimals).tolist(), it.tolist()):
            acc[m] = acc.get(m, 0.0) + i
    if not acc:
        return np.zeros(0), np.zeros(0)
    mz = np.fromiter(acc.keys(), dtype=np.float64)
    it = np.fromiter(acc.values(), dtype=np.float64)
    if mz.size > top_k:
        keep = np.argsort(-it)[:top_k]
        mz, it = mz[keep], it[keep]
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    return mz, it / it.max()


def align(mz_a, it_a, mz_b, it_b, np, tol_da=0.01, tol_ppm=20.0):
    """One-to-one greedy matching (largest intensity product first).
    Returns two equally long intensity vectors (matched pairs, then leftovers)."""
    na, nb = mz_a.size, mz_b.size
    if na == 0 or nb == 0:
        return (np.concatenate([it_a, np.zeros(nb)]), np.concatenate([np.zeros(na), it_b]))
    tol = np.maximum(tol_da, mz_a * tol_ppm * 1e-6)
    lo = np.searchsorted(mz_b, mz_a - tol, side="left")
    hi = np.searchsorted(mz_b, mz_a + tol, side="right")
    pairs = []
    for i in range(na):
        for j in range(lo[i], hi[i]):
            pairs.append((it_a[i] * it_b[j], i, j))
    pairs.sort(reverse=True)
    used_a, used_b = np.zeros(na, bool), np.zeros(nb, bool)
    va, vb = [], []
    for _, i, j in pairs:
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = used_b[j] = True
        va.append(it_a[i])
        vb.append(it_b[j])
    ra, rb = it_a[~used_a], it_b[~used_b]
    a = np.concatenate([np.asarray(va, dtype=np.float64), ra, np.zeros(rb.size)])
    b = np.concatenate([np.asarray(vb, dtype=np.float64), np.zeros(ra.size), rb])
    return a, b


def _entropy(p, np):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entropy_similarity(a, b, np):
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    a, b = a / sa, b / sb
    val = 1.0 - (2.0 * _entropy((a + b) / 2.0, np) - _entropy(a, np) - _entropy(b, np)) / math.log(4.0)
    return float(min(1.0, max(0.0, val)))


def cosine_similarity(a, b, np, power=0.5):
    a, b = np.power(a, power), np.power(b, power)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else 0.0


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------
def ce_list(raw, args):
    vals = []
    if raw is not None:
        try:
            if isinstance(raw, (int, float)):
                raw = [raw]
            for v in raw:
                v = abs(float(v))
                if math.isfinite(v) and v > 0:
                    vals.append(v)
        except TypeError:
            vals = []
    if not vals:
        vals = [float(x) for x in args.default_ces.split(",") if x]
    out = []
    for v in vals:
        if args.ce_round > 0:
            v = round(v / args.ce_round) * args.ce_round
        v = max(args.ce_min, float(v))
        if v not in out:
            out.append(v)
    out.sort()
    if args.max_ces and len(out) > args.max_ces:
        idx = [round(i * (len(out) - 1) / (args.max_ces - 1)) for i in range(args.max_ces)] if args.max_ces > 1 else [len(out) // 2]
        out = [out[i] for i in sorted(set(idx))]
    return tuple(out)


def load_parquet_spectra(path, wanted):
    cols = "molecule_id, ms2_mzs, ms2_normalized_intensities, adduct, precursor_mz, collision_energy_ev"
    rows = None
    try:
        import duckdb
        rows = duckdb.connect().execute(f"select {cols} from read_parquet(?)", [str(path)]).fetchall()
    except ImportError:
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(path), columns=[c.strip() for c in cols.split(",")]).to_pydict()
        rows = list(zip(*[tbl[c.strip()] for c in cols.split(",")]))
    out = {}
    for mid, mzs, its, adduct, pmz, ces in rows:
        if mid in wanted:
            out.setdefault(mid, []).append({"mzs": mzs, "intensities": its, "adduct": adduct,
                                            "precursor_mz": pmz, "collision_energies": ces})
    return out


# --------------------------------------------------------------------------
def write_json_atomic(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    deadline = T0 + args.budget_sec - args.safety_sec

    with open(args.input) as fh:
        data = json.load(fh, object_pairs_hook=OrderedDict)
    scores = {mid: [None] * len(v.get("smiles", [])) for mid, v in data.items()}
    cosines = {mid: [None] * len(v.get("smiles", [])) for mid, v in data.items()}
    meta = {"status": "started", "n_molecules": len(data), "n_candidates": sum(len(v) for v in scores.values()),
            "argv": sys.argv[1:], "reasons": {}}

    def bump(reason, n=1):
        meta["reasons"][reason] = meta["reasons"].get(reason, 0) + n

    def save(status):
        meta["status"] = status
        meta["elapsed_sec"] = round(time.time() - T0, 2)
        write_json_atomic({"scores": scores, "cosine": cosines, "meta": meta}, args.output)

    save("started")  # a valid (all-null) output exists from the first second on
    try:
        run(args, data, scores, cosines, meta, bump, save, deadline)
    except Exception:
        meta["error"] = traceback.format_exc()[-4000:]
        log(args, "FATAL, keeping partial output\n" + meta["error"])
        save("failed")
        return 0  # graceful: the caller reads what is there
    return 0


def run(args, data, scores, cosines, meta, bump, save, deadline):
    for p in reversed(args.prepend_path):
        sys.path.insert(0, p)
    import numpy as np
    import fm_env
    fm_env.setup(args.ms_pred_src, shims=not args.real_dgl, extra_paths=args.extra_path)
    import torch
    if args.real_dgl:
        import pathlib
        if os.name == "nt":  # Lightning checkpoints written on Linux pickle PosixPath
            pathlib.PosixPath = pathlib.PurePosixPath
        import dgl
        _graph = dgl.graph

        def _graph64(data, *a, **k):  # Windows + numpy<2 would give int32 ids
            return _graph(tuple(torch.as_tensor(x).long() for x in data), *a, **k)
        dgl.graph = _graph64
    import warnings
    warnings.filterwarnings("ignore")
    from rdkit import Chem, RDLogger
    import rdkit
    RDLogger.DisableLog("rdApp.*")
    import ms_pred.common as common

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        device = "cuda:0"
    batch_size = args.batch_size or (64 if device.startswith("cuda") else 16)
    covered = {a.strip() for a in args.adducts.split(",") if a.strip()}
    covered = {a for a in covered if a in common.ion2onehot_pos}
    if args.instrument not in common.instrument2onehot_pos:
        raise ValueError(f"unknown instrument token {args.instrument}")

    if args.model == "iceberg":
        from ms_pred.iceberg import joint_model
        model = joint_model.JointModel.from_checkpoints(args.gen_ckpt, args.inten_ckpt)
    else:
        from ms_pred.glacier import joint_model
        model = joint_model.JointModel.load_from_checkpoint(args.glacier_ckpt, map_location="cpu")
    model.eval()
    model.to(device)
    meta.update({"model": args.model, "device": device, "batch_size": batch_size, "torch": torch.__version__, "rdkit": rdkit.__version__,
                 "shims": not args.real_dgl, "adducts": sorted(covered), "instrument": args.instrument,
                 "load_sec": round(time.time() - T0, 2)})
    log(args, f"model loaded on {device}; rdkit {rdkit.__version__}; torch {torch.__version__}")

    # ---- observed spectra ------------------------------------------------
    ext = {}
    if args.spectra_parquet:
        ext = load_parquet_spectra(args.spectra_parquet, set(data.keys()))

    canon_cache = {}

    def canonical(smi):
        """Same normalisation as upstream predict_mol (stereo removed, InChI round trip)."""
        if smi in canon_cache:
            return canon_cache[smi]
        res = None
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                why = "invalid_smiles"
            elif "." in smi:
                why = "multi_fragment"
            elif any(a.GetSymbol() not in common.VALID_ELEMENTS for a in mol.GetAtoms()):
                why = "unsupported_element"
            elif mol.GetNumAtoms() > args.max_atoms:
                why = "too_large"
            elif mol.GetNumAtoms() < 2:
                why = "too_small"
            else:
                if args.model == "iceberg":
                    s = common.rm_stereo(smi)
                    s = common.smiles_from_inchi(common.inchi_from_smiles(s))
                else:  # upstream GLACIER prediction script: plain RDKit canonical SMILES
                    s = Chem.MolToSmiles(Chem.RemoveHs(mol))
                m2 = Chem.MolFromSmiles(s) if s else None
                if m2 is None or "." in s:
                    why = "canonicalisation_failed"
                else:
                    res, why = (s, float(common.mass_from_smi(s))), None
        except Exception as exc:  # noqa: BLE001
            why = f"canon_error:{type(exc).__name__}"
        canon_cache[smi] = (res, why)
        return canon_cache[smi]

    # ---- plan ------------------------------------------------------------
    # plan[mid] = (list of (obs_mz, obs_int, adduct, ces), list of canonical entries per candidate)
    plan = OrderedDict()
    for mid, entry in data.items():
        specs = entry.get("spectra") or ext.get(mid) or []
        obs = []
        for sp in specs:
            adduct = sp.get("adduct") or entry.get("adduct")
            if adduct not in covered:
                continue
            mz, it = clean_spectrum(sp.get("mzs", []), sp.get("intensities", []), np, args.obs_top_k, args.obs_min_rel)
            if mz.size == 0:
                continue
            obs.append((mz, it, adduct, ce_list(sp.get("collision_energies"), args)))
        if not obs:
            bump("molecule_without_covered_spectrum")
            continue
        plan[mid] = obs
    meta["n_molecules_covered"] = len(plan)
    log(args, f"{len(plan)}/{len(data)} molecules have a covered spectrum")

    pred_cache = {}       # (canon_smiles, adduct, ce) -> (n,2) float array | None
    dump = {} if args.dump_pred else None
    n_pred, t_pred, last_save = 0, 0.0, time.time()
    state = {"bs": batch_size}

    def predict_batch(tasks):
        """tasks: list of (canon_smiles, mass, adduct, ce).  Fills pred_cache."""
        smis = [t[0] for t in tasks]
        adducts = [t[2] for t in tasks]
        ces = [float(t[3]) for t in tasks]
        pmz = [t[1] + float(common.ion2mass[t[2]]) for t in tasks]
        with torch.no_grad():
            if args.model == "iceberg":
                out = model.predict_mol(smis, collision_eng=ces, precursor_mz=pmz, adduct=adducts,
                                        threshold=args.threshold, device=device, max_nodes=args.max_nodes,
                                        instrument=[args.instrument] * len(tasks), binned_out=False,
                                        canonical_root_smi=True)
            else:
                out = model.predict_mol(smis, collision_eng=ces, adduct=adducts, device=device,
                                        instrument=[args.instrument] * len(tasks))
        for t, sp in zip(tasks, out["spec"]):
            sp = sp.detach().cpu().numpy().astype(np.float64)
            sp = sp[np.isfinite(sp).all(1) & (sp[:, 1] > 0)]
            if sp.shape[0] > args.top_k:
                sp = sp[np.argsort(-sp[:, 1])[: args.top_k]]
            pred_cache[(t[0], t[2], t[3])] = sp

    def predict_safe(tasks):
        nonlocal n_pred, t_pred
        t1 = time.time()
        try:
            predict_batch(tasks)
        except Exception as exc:  # noqa: BLE001
            oom = "out of memory" in str(exc).lower()
            if oom and device.startswith("cuda"):
                torch.cuda.empty_cache()
                state["bs"] = max(1, state["bs"] // 2)
                bump("cuda_oom_batch_halved")
            if len(tasks) == 1:
                pred_cache[(tasks[0][0], tasks[0][2], tasks[0][3])] = None
                bump(f"predict_error:{type(exc).__name__}")
            else:
                half = len(tasks) // 2
                t_pred += time.time() - t1
                predict_safe(tasks[:half])
                predict_safe(tasks[half:])
                return
        n_pred += len(tasks)
        t_pred += time.time() - t1

    out_of_time = False
    done_mols = 0
    for mid, obs in plan.items():
        if time.time() > deadline:
            out_of_time = True
            break
        smiles = data[mid]["smiles"]
        limit = len(smiles) if args.max_cands <= 0 else min(len(smiles), args.max_cands)
        cands = []
        for k in range(limit):
            res, why = canonical(smiles[k])
            if res is None:
                bump(why)
            cands.append(res)
        if time.time() > deadline:
            out_of_time = True
            break
        need = []
        for res in cands:
            if res is None:
                continue
            for _, _, adduct, ces in obs:
                for ce in ces:
                    key = (res[0], adduct, ce)
                    if key not in pred_cache and key not in need:
                        need.append(key)
        mass_of = {res[0]: res[1] for res in cands if res is not None}
        tasks = [(s, mass_of[s], a, c) for (s, a, c) in need]
        i = 0
        while i < len(tasks):
            if time.time() > deadline:
                out_of_time = True
                break
            chunk = tasks[i:i + state["bs"]]
            predict_safe(chunk)
            i += len(chunk)
        # score whatever is completely predicted (also on the molecule that ran out of time)
        for k, res in enumerate(cands):
            if res is None:
                continue
            ent, cos, complete = [], [], True
            for mz_o, it_o, adduct, ces in obs:
                preds = [pred_cache.get((res[0], adduct, ce), "missing") for ce in ces]
                if any(isinstance(p, str) or p is None for p in preds):
                    complete = False
                    break
                mz_p, it_p = merge_predictions(preds, np, args.top_k)
                if mz_p.size == 0:
                    complete = False
                    break
                a, b = align(mz_o, it_o, mz_p, it_p, np, args.tol_da, args.tol_ppm)
                ent.append(entropy_similarity(a, b, np))
                cos.append(cosine_similarity(a, b, np))
                if dump is not None:
                    dump.setdefault(mid, {}).setdefault(str(k), []).append(
                        {"adduct": adduct, "ces": list(ces), "mz": mz_p.round(4).tolist(), "intensity": it_p.round(5).tolist()})
            if complete and ent:
                scores[mid][k] = round(float(np.mean(ent)), 6)
                cosines[mid][k] = round(float(np.mean(cos)), 6)
            else:
                bump("candidate_not_scored")
        done_mols += 1
        if out_of_time:
            break
        if time.time() - last_save > args.save_every_sec:
            meta.update({"n_predictions": n_pred, "predict_sec": round(t_pred, 2), "molecules_done": done_mols})
            save("running")
            last_save = time.time()
            log(args, f"{done_mols}/{len(plan)} molecules, {n_pred} predictions, {n_pred / max(t_pred, 1e-9):.2f} pred/s")
        if len(pred_cache) > 200000:
            pred_cache.clear()

    meta.update({"n_predictions": n_pred, "predict_sec": round(t_pred, 2), "molecules_done": done_mols,
                 "pred_per_sec": round(n_pred / max(t_pred, 1e-9), 3), "out_of_time": out_of_time,
                 "n_scored": sum(1 for v in scores.values() for x in v if x is not None)})
    if dump is not None:
        write_json_atomic(dump, args.dump_pred)
    save("out_of_time" if out_of_time else "ok")
    log(args, f"done: {meta['n_scored']} candidates scored, {n_pred} predictions in {t_pred:.1f}s, status {meta['status']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())

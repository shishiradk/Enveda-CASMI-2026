"""Evaluate an exported CFT model and compare it with baselines on the SAME candidates.

    python eval_cft.py --data <data folder(s)> --out <output folder used for training>
    python eval_cft.py --data <data folder(s)> --ckpt path/to/cft_default.pt [--baseline folder_with_fp_*.pt]

Reports, for held-out validation molecules (query = up to 6 spectra of one library, combined by the inference API):
  pool        rank of the truth among all same-mass (+-10 ppm) structures of the CFT pool   <- headline
  pool_orig   the same, PubChem decoys left out (candidates the warm-up model can also score)
  isomers     same-formula candidates only
  pcv         rank among up to 384 random same-mass PubChem structures (+ estimate for the full PubChem window)
Baselines: 'prior' (no spectrum: log-odds of the mean training fingerprint) and, if the warm-up networks
fp_single*.pt / fp_merged*.pt are found, the warm-up model on pool_orig and pcv.
Writes <out>/export/eval_<name>.json and check_summary_cft.txt. Exit code 0 = model loaded and ran.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cft_model as M
import train_cft as T


def find_baseline(a, dirs):
    cands = [a.baseline] if a.baseline else []
    cands += [str(Path(a.out) / "export")] if a.out else []
    cands += [str(Path(d) / "out" / "export") for d in dirs] + [str(HERE.parent.parent / "results/train_pkg/smoke_out/export")]
    for c in cands:
        f = sorted(Path(c).glob("fp_single*.pt")) if c and Path(c).is_dir() else []
        if f: return Path(c)
    return None


def baseline_logits(bdir, code_dirs, D, V, dev):
    """Warm-up FPNet (6,930 bits): per-spectrum network on single views, merged network on merged views."""
    import torch
    for c in code_dirs:
        if (Path(c) / "fp_model.py").exists(): sys.path.insert(0, str(c)); break
    else:
        return None, "fp_model.py not found"
    import fp_model as F0
    nets = {}
    for kind, pat in (("single", "fp_single*.pt"), ("merged", "fp_merged*.pt")):
        f = sorted(bdir.glob(pat))
        if not f: continue
        ck = torch.load(f[0], map_location="cpu", weights_only=False)
        net = F0.build_model(ck["nbits"], d=ck["d"], layers=ck["layers"]).to(dev).eval(); net.load_state_dict(ck["model"])
        nets[kind] = (net, f[0].name, int(ck["step"]))
    A = V["arrays"]; n = len(A["prec"]); nb = next(iter(nets.values()))[0].head[-1].out_features
    z = torch.zeros(n, nb, device=dev)
    ce = np.where(A["cek"] > 0, A["ce"], 25.0).astype(np.float32)
    mode = np.where(A["mode"] >= 0, 1.0, -1.0).astype(np.float32)
    with torch.no_grad():
        for kind, k in (("single", 0), ("merged", 1)):
            net = nets.get(kind, nets.get("single"))[0]
            ix = np.where(V["kind"] == k)[0]
            for s in range(0, len(ix), 128):
                j = ix[s:s + 128]
                inp = [torch.as_tensor(x[j], device=dev) for x in (A["mz"], A["it"], A["pad"], A["prec"], A["ad"], A["ins"], ce, mode)]
                z[torch.as_tensor(j, device=dev)] = net(*inp).float()
        Z = M.combine_views(z, V["owner"], V["kind"], len(V["mols"]))
    return Z, ", ".join(f"{k}: {v[1]} (step {v[2]})" for k, v in nets.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+"); ap.add_argument("--out", default="")
    ap.add_argument("--ckpt", default="", help="exported cft_*.pt (default: every cft_*.pt in <out>/export)")
    ap.add_argument("--baseline", default="", help="folder with the warm-up fp_single*.pt / fp_merged*.pt")
    ap.add_argument("--val_mols", type=int, default=0, help="0 = all validation molecules")
    ap.add_argument("--val_q", type=int, default=6); ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="small validation set (pipeline test)")
    a = ap.parse_args()
    import torch
    assert a.out or a.ckpt, "give --out (training output folder) or --ckpt"
    exp = Path(a.out) / "export" if a.out else Path(a.ckpt).parent
    exp.mkdir(parents=True, exist_ok=True)
    files = [Path(a.ckpt)] if a.ckpt else sorted(exp.glob("cft_*.pt"))
    lines = []; ok = bool(files)

    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); lines.append(s)

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    say("CASMI CFT evaluation,", time.strftime("%Y-%m-%d %H:%M:%S"), "| device:", dev,
        torch.cuda.get_device_name(0) if dev == "cuda" else "", "| torch", torch.__version__)
    if not files: say(f"PROBLEM: no cft_*.pt found in {exp} - training has not written a model yet")
    D = T.Data(a.data, mmap=True, log=lambda *x: None)
    V = D.build_val(100 if a.smoke else a.val_mols, q=a.val_q, pcv_limit=40 if a.smoke else None, log=say)
    n = len(V["mols"]); R = {"validation": dict(molecules=int(V["in_pool_set"].sum()), pcv_molecules=int(len(V["p_mol"])),
                                                 spectra_per_query=a.val_q)}

    def row(tag, r):
        f = lambda k, w=".4f": format(r[k], w) if r.get(k) is not None else "   -  "
        return (f"  {tag:<24} MRR@25 {f('mrr')}  top1 {f('top1')}  median rank {f('median_rank', '6.1f')}  "
                f"molecules {r.get('n_mol', 0):>5}  mean candidates {f('mean_cands', '7.1f')}  chance MRR {f('chance_mrr')}"
                + (f"  | full PubChem window (est.): MRR {f('mrr_fullwindow_est')} top1 {f('top1_fullwindow_est')} "
                   f"mean window {f('mean_full_window', '.0f')}" if "mrr_fullwindow_est" in r else ""))

    # ---- the CFT model(s), through the inference API
    for f in files:
        try:
            sc = M.CFTScorer(str(f), device=dev)
            mols = [D.spectra_of(int(m), a.val_q, 12345)[0] for m in V["mols"]]
            Z = torch.cat([sc.logits_prepared(mols[s:s + 64]) for s in range(0, n, 64)])
            fin = bool(torch.isfinite(Z).all())
            Zv = T.val_logits(sc.net, V, dev, sc.amp)                 # the path used during training
            par = float((Z - Zv).abs().max())
            res = T.evaluate(D, V, Z, dev)
            R[f.name] = dict(metrics=res, step=sc.step, finite=fin, api_vs_training_path_max_abs_diff=par,
                             bytes=f.stat().st_size, sha256=T.sha256(f)[:16], cfg=sc.cfg)
            good = fin and sc.nbits == D.nbits
            ok &= good
            say(f"\n{f.name}: {'OK' if good else 'PROBLEM'} | {f.stat().st_size / 1e6:.1f} MB | step {sc.step} | "
                f"d {sc.cfg['d']} layers {sc.cfg['layers']} nbits {sc.nbits} | finite {fin} | "
                f"inference API vs training path: max abs logit difference {par:.2e}")
            for tag in ("pool", "pool_orig", "pool_isomers", "pcv", "pcv_isomers", "pcv_truth_train_form"):
                say(row(tag, res[tag]))
        except Exception as e:
            ok = False; say(f"{f.name}: PROBLEM - could not load or run: {type(e).__name__}: {e}")

    # ---- baseline 1: no spectrum at all (prior over fingerprint bits)
    tm = np.where((D.mol_split == 0) & (D.mol_cpool >= 0))[0]
    tm = np.sort(np.random.default_rng(0).choice(tm, size=min(20000, len(tm)), replace=False))
    freq = np.unpackbits(np.asarray(D.fp[np.sort(D.mol_cpool[tm])]), axis=1)[:, :D.nbits].mean(0).clip(1e-3, 1 - 1e-3)
    Zp = torch.as_tensor(np.log(freq / (1 - freq)), dtype=torch.float32, device=dev).repeat(n, 1)
    res = T.evaluate(D, V, Zp, dev); R["baseline_prior"] = res
    say("\nbaseline 'prior' (no spectrum; log-odds of the mean training fingerprint), same candidates:")
    for tag in ("pool", "pool_orig", "pcv"): say(row(tag, res[tag]))

    # ---- baseline 2: the warm-up networks (6,930-bit fingerprint) on the candidates they can score
    bdir = find_baseline(a, a.data)
    if bdir is None:
        say("\nbaseline 'warm-up networks': not found (give --baseline <folder with fp_single*.pt>)")
    else:
        try:
            Zb, desc = baseline_logits(bdir, list(a.data) + [HERE, HERE.parent / "train_pkg"], D, V, dev)
            if Zb is None: raise RuntimeError(desc)
            pfp = np.load(T.find_file(D.dirs, "pool_fp.npy"), mmap_mode="r"); mol_pool = np.load(T.find_file(D.dirs, "mol_pool.npy"))
            orig = np.load(T.find_file(D.dirs, "cpool_orig.npy")); nb = Zb.shape[1]
            keep = V["c_orig"]; rows = orig[V["c_rows"][keep]]; own = V["c_owner"][keep]
            tf = np.asarray(pfp[np.asarray(mol_pool[V["mols"]])])
            gt, eq = T.compare(Zb, tf, lambda s, e: pfp[rows[s:e]], own, nb, dev)
            rb = dict(pool_orig=T.summarize(gt, eq, own, n))
            fpb = np.load(T.find_file(D.dirs, "pcv_fpb.npy"), mmap_mode="r")
            tb = np.load(T.find_file(D.dirs, "pcv_truth_fpb.npy"))[:len(V["p_mol"])]
            pt = tf.copy(); pt[V["p_mol"][V["p_in_pc"]]] = tb[V["p_in_pc"]]
            gt, eq = T.compare(Zb, pt, lambda s, e: fpb[s:e], V["p_owner"], nb, dev)
            rb["pcv"] = T.summarize(gt, eq, V["p_owner"], n, None, V["p_mol"], V["p_nfull"])
            R["baseline_warmup"] = dict(metrics=rb, files=desc, folder=str(bdir))
            say(f"\nbaseline 'warm-up networks' from {bdir} ({desc}), same candidates:")
            for tag in ("pool_orig", "pcv"): say(row(tag, rb[tag]))
            if "smoke" in str(bdir): say("  NOTE: this baseline file is the 60-step SMOKE model, not a trained network.")
        except Exception as e:
            say(f"\nbaseline 'warm-up networks': could not be run ({type(e).__name__}: {e})")

    say("\nHow to read: 'pool' MRR is the selection metric. Compare CFT with the baselines on pool_orig and pcv "
        "(identical candidate sets). 'chance MRR' = random order.")
    for p in sorted(exp.glob("train_result_*.json")):
        res = json.load(open(p)); st = res.get("state", {})
        say(f"training [{res.get('name')}]: finished {res.get('finished')} | steps {st.get('steps_done')} | best pool MRR "
            f"{st.get('best_mrr')} at step {st.get('best_step')} | {st.get('steps_per_s')} step/s | "
            f"{round((st.get('train_seconds') or 0) / 60)} min | VRAM {st.get('vram_gb')} GB | stop {st.get('stop_reason')} | "
            f"gpu {res['env'].get('gpu')} amp {res['env'].get('amp')} bs {res['config'].get('bs')}")
        ok &= bool(res.get("finished")) or bool(a.ckpt)
    say("RESULT:", "ALL OK - send back the folder " + str(exp) if ok
        else "NOT READY (see PROBLEM lines above, or training has not finished)")
    json.dump(R, open(exp / "eval_cft.json", "w"), indent=1, default=str)
    open(exp / "check_summary_cft.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

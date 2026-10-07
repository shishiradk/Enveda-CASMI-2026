"""Verify the exported networks and print a short summary to send back.

    python check_result.py --data <data folder> --out <output folder used for training>

Loads each exported .pt the way the engine does (rebuild FPNet from nbits/d/layers, strict state-dict load),
runs it on validation spectra and writes <out>/export/check_summary.txt.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_model as M

KEYS = {"model", "step", "nbits", "d", "layers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    import torch
    import train_fp as T
    exp = Path(a.out) / "export"; lines = []; ok = True

    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); lines.append(s)

    say("CASMI fingerprint-network check,", time.strftime("%Y-%m-%d %H:%M:%S"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", "| torch", torch.__version__)
    rj = exp / "train_result.json"
    res = json.load(open(rj)) if rj.exists() else None
    if res is None: say("PROBLEM: train_result.json is missing - training has not written results yet"); ok = False
    else: say("training finished:", res["finished"])
    files = sorted(exp.glob("fp_*.pt"))
    if len(files) != 2: say(f"PROBLEM: expected 2 exported fp_*.pt files in {exp}, found {len(files)}"); ok = False
    D = T.Data(a.data, mmap=True, log=lambda *x: None)
    ids = np.sort(np.random.default_rng(12345).choice(D.va, size=min(a.n, len(D.va)), replace=False))
    for f in files:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            assert set(ck) == KEYS, f"keys {sorted(ck)}"
            net = M.build_model(ck["nbits"], d=ck["d"], layers=ck["layers"]).to(dev).eval()
            net.load_state_dict(ck["model"])                       # strict
            merged = "merged" in f.name
            rng = np.random.default_rng(999); hit = 0; fin = True
            with torch.no_grad():
                for s in range(0, len(ids), 64):
                    inp, cand = D.batch(ids[s:s + 64], 63, rng, aug=False, merge_p=0.6 if merged else 0.0)
                    z = net(*[torch.as_tensor(x, device=dev) for x in inp]).float().cpu().numpy()
                    fin &= bool(np.isfinite(z).all())
                    fc = np.unpackbits(np.asarray(D.pfp[cand.ravel()]), axis=1)[:, :D.nbits].reshape(*cand.shape, -1)
                    hit += int(((fc * z[:, None, :]).sum(-1).argmax(1) == 0).sum())
            top1 = hit / len(ids)
            good = fin and ck["nbits"] == D.nbits
            ok &= good
            say(f"{f.name}: {'OK' if good else 'PROBLEM'} | {f.stat().st_size / 1e6:.1f} MB | step {ck['step']} | "
                f"d {ck['d']} layers {ck['layers']} nbits {ck['nbits']} | view {'merged' if merged else 'single'} | "
                f"val hardneg-top1 on {len(ids)} spectra {top1:.3f} | finite {fin} | sha256 {T.sha256(f)[:16]}")
        except Exception as e:
            ok = False; say(f"{f.name}: PROBLEM - could not load or run: {type(e).__name__}: {e}")
    if res:
        for n, m in res["models"].items():
            say(f"training [{n}]: status {m.get('status')} | steps {m.get('steps_done')} | best top1 "
                f"{m.get('best_top1')} at step {m.get('best_step')} | {m.get('steps_per_s')} step/s | "
                f"{round((m.get('train_seconds') or 0) / 60)} min | stop {m.get('stop_reason')}")
        say("gpu:", res["env"].get("gpu"), "| amp", res["env"].get("amp"), "| batch size", res["config"].get("bs"))
    say("RESULT:", "ALL OK - send back the folder " + str(exp) if ok and res and res["finished"]
        else "NOT READY (see PROBLEM lines above, or training has not finished)")
    exp.mkdir(parents=True, exist_ok=True)
    open(exp / "check_summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

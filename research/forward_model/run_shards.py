#!/usr/bin/env python
"""Run a forward-model runner over an input JSON in N parallel subprocess shards
(local CPU validation only; on Kaggle the runner is called once on the GPU).

    python run_shards.py --python <interpreter> --runner ice_runner.py --input in.json \
        --output out.json --shards 4 -- <arguments passed through to the runner>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    rest = [a for a in args.rest if a != "--"]
    data = json.load(open(args.input))
    keys = list(data.keys())
    out = Path(args.output)
    procs, parts = [], []
    t0 = time.time()
    for s in range(args.shards):
        part_keys = keys[s::args.shards]
        if not part_keys:
            continue
        p_in = out.with_suffix(f".shard{s}.in.json")
        p_out = out.with_suffix(f".shard{s}.out.json")
        json.dump({k: data[k] for k in part_keys}, open(p_in, "w"))
        log = open(out.with_suffix(f".shard{s}.log"), "w")
        cmd = [args.python, args.runner, "--input", str(p_in), "--output", str(p_out)] + rest
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        parts.append((p_in, p_out))
    for p in procs:
        p.wait()
    merged = {"scores": {}, "cosine": {}, "meta": {"shards": [], "wall_sec": round(time.time() - t0, 1)}}
    for p_in, p_out in parts:
        if not p_out.exists():
            continue
        d = json.load(open(p_out))
        merged["scores"].update(d.get("scores", {}))
        merged["cosine"].update(d.get("cosine", {}))
        merged["meta"]["shards"].append({k: v for k, v in d.get("meta", {}).items() if k != "argv"})
        p_in.unlink()
        p_out.unlink()
    sh = merged["meta"]["shards"]
    merged["meta"]["n_predictions"] = sum(m.get("n_predictions", 0) for m in sh)
    merged["meta"]["predict_sec_sum"] = round(sum(m.get("predict_sec", 0) for m in sh), 1)
    merged["meta"]["n_scored"] = sum(m.get("n_scored", 0) for m in sh)
    merged["meta"]["status"] = [m.get("status") for m in sh]
    # keep the input order
    for key in ("scores", "cosine"):
        merged[key] = {k: merged[key][k] for k in keys if k in merged[key]}
    json.dump(merged, open(out, "w"))
    print(json.dumps({k: v for k, v in merged["meta"].items() if k != "shards"}))


if __name__ == "__main__":
    main()

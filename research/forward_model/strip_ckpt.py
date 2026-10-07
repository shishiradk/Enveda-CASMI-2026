#!/usr/bin/env python
"""Strip a Lightning checkpoint to {'state_dict', 'hyper_parameters', 'provenance'} (plain tensors,
loadable with torch.load(weights_only=True) on any OS).  Weights are unchanged.

    python strip_ckpt.py in.ckpt out.pt
"""
import hashlib
import pathlib
import sys

import torch


def main(src, dst):
    if not isinstance(pathlib.Path(), pathlib.PosixPath):
        pathlib.PosixPath = pathlib.PurePosixPath  # checkpoints pickled on Linux
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    best = ""
    for val in ckpt.get("callbacks", {}).values():
        if isinstance(val, dict) and "best_model_path" in val:
            best = str(val["best_model_path"])
    out = {"state_dict": {k: v.clone() for k, v in ckpt["state_dict"].items()},
           "hyper_parameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                for k, v in dict(ckpt["hyper_parameters"]).items()},
           "provenance": {"source_sha256": sha, "upstream_training_path": best,
                          "epoch": int(ckpt.get("epoch", -1)), "global_step": int(ckpt.get("global_step", -1))}}
    torch.save(out, dst)
    print(dst, out["provenance"], sum(v.numel() for v in out["state_dict"].values()), "parameters")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

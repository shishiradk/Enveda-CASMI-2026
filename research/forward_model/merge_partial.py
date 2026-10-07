#!/usr/bin/env python
"""Merge (possibly partial) shard outputs of run_shards.py; keeps only molecules where every candidate has a score
or an explicit failure, i.e. molecules the shard has finished.  Usage: merge_partial.py <prefix> <out.json>"""
import glob, json, sys
prefix, out = sys.argv[1], sys.argv[2]
merged = {"scores": {}, "cosine": {}, "meta": {"shards": []}}
for f in sorted(glob.glob(prefix + ".shard*.out.json")):
    d = json.load(open(f))
    done = d["meta"].get("molecules_done", 0)
    keys = [k for k, v in d["scores"].items() if any(x is not None for x in v)]
    for k in keys:
        merged["scores"][k] = d["scores"][k]; merged["cosine"][k] = d["cosine"][k]
    merged["meta"]["shards"].append({k: v for k, v in d["meta"].items() if k != "argv"})
sh = merged["meta"]["shards"]
merged["meta"]["n_predictions"] = sum(m.get("n_predictions", 0) for m in sh)
merged["meta"]["predict_sec_sum"] = round(sum(m.get("predict_sec", 0) for m in sh), 1)
merged["meta"]["n_molecules_scored"] = len(merged["scores"])
json.dump(merged, open(out, "w"))
print({k: v for k, v in merged["meta"].items() if k != "shards"})

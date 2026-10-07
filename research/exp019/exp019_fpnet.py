"""EXP-019 -- trained spectrum -> fingerprint network (FPNet-style MLP), an engine independent of the kNN.

Training spectra: the V1/EXP-018 reference construction (10 test adducts, enveda-180 excluded, <= 3 per (ik, library)),
minus every molecule that is an evaluation target anywhere (EXP-018 train+val targets, their parent groups, the V1
np-examples keys and the EXP-017 S3 keys). 5% of the remaining molecules are the early-stopping split.
Input: 0.1 Da fragment + neutral-loss bins (sqrt intensity, L2-normalised; EXP-011 features) + adduct one-hot.
Target: Morgan r2/2048 bits of the spectrum's structure. Loss: BCE with logits. CPU training (torch).

Evaluation on the EXP-018 feature frames (same candidates, same molecules): FPNet alone, V1 kNN alone, and a fixed
equal-weight fusion z(kNN) + z(FPNet) within each pool (no fitted weights).

    python research/exp019/exp019_fpnet.py data
    python research/exp019/exp019_fpnet.py train [--epochs 20]
    python research/exp019/exp019_fpnet.py eval  [--sets val,s1,s2]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "kaggle_v1"))
sys.path.insert(0, str(ROOT / "research" / "kaggle_v2"))
sys.path.insert(0, str(ROOT / "research" / "exp018"))
import casmi_v1_kaggle as v1  # noqa: E402

TRAIN = (ROOT / "train.parquet").as_posix()
UNIV = ROOT / "results" / "kaggle_v1_assets"
E18 = ROOT / "results" / "exp018"
O = ROOT / "results" / "exp019"
SEED = 20260930
ADDUCTS = sorted(v1.AD10)
FCFG = {**v1.CFG, "bin_w": 0.1}  # 15,000 fragment + 5,000 neutral-loss bins


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def excluded_keys():
    x8 = pd.read_parquet(ROOT / "results" / "exp008_structure_universe.parquet", columns=["ik", "parent"])
    keys = set(pd.read_parquet(E18 / "targets.parquet")["ik"])
    keys |= set(pd.read_parquet(ROOT / "results" / "kaggle_v1_proxy" / "held_keys.parquet")["ik"])
    keys |= set(pd.read_parquet(ROOT / "results" / "kaggle_v2_proxy" / "s3_held.parquet")["ik"])
    par = set(x8.loc[x8["ik"].isin(keys), "parent"])
    return keys | set(x8.loc[x8["parent"].isin(par), "ik"])


def stage_data(a):
    from scipy import sparse
    O.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ex = excluded_keys()
    pd.DataFrame({"ik": sorted(ex)}).to_parquet(O / "excluded_keys.parquet", index=False)
    ads = ",".join(f"'{x}'" for x in v1.AD10)
    con = v1.v0.connect(mem="6GB", threads=4, tmp=(O / "duck_tmp").as_posix())
    con.register("ex", pd.DataFrame({"ik": sorted(ex)}))
    tbl = con.execute(f"""
        SELECT ik, adduct, pm, ms2_mzs, ms2_normalized_intensities FROM (
            SELECT inchikey14 AS ik, adduct, precursor_mz AS pm, ms2_mzs, ms2_normalized_intensities,
                   row_number() OVER (PARTITION BY inchikey14, ingest_lib ORDER BY hash(file_row_number + {SEED}), file_row_number) AS rn
            FROM read_parquet('{TRAIN}', file_row_number=true)
            WHERE adduct IN ({ads}) AND ingest_lib <> 'enveda-180' AND inchikey14 IS NOT NULL
                  AND inchikey14 NOT IN (SELECT ik FROM ex))
        WHERE rn <= {FCFG['ref_cap']} ORDER BY ik""").arrow()
    con.close()
    if not hasattr(tbl, "column_names"):
        tbl = tbl.read_all()
    X = v1.featurize_arrow(tbl, FCFG)
    ik = np.asarray(tbl["ik"].to_numpy(zero_copy_only=False), dtype=object)
    add = np.array([ADDUCTS.index(x) for x in tbl["adduct"].to_numpy(zero_copy_only=False)], np.int64)
    U = v1.Universe(UNIV)
    row = np.array([U.row.get(k, -1) for k in ik])
    keep = (row >= 0) & (np.diff(X.indptr) > 0)
    X, ik, add, row = X[keep], ik[keep], add[keep], row[keep]
    mols = np.unique(ik)
    rng = np.random.default_rng(SEED)
    es = set(rng.choice(mols, int(0.05 * len(mols)), replace=False))
    split = np.array([k in es for k in ik])
    sparse.save_npz(O / "X.npz", X)
    np.savez(O / "meta.npz", add=add, row=row, es=split)
    info = {"spectra": int(X.shape[0]), "molecules": int(len(mols)), "early_stop_spectra": int(split.sum()),
            "excluded_keys": len(ex), "dim": int(X.shape[1]), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(O / "data.json", "w"), indent=1)
    log(info)


def make_model(dim):
    import torch
    import torch.nn as nn

    class FPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(dim, 1024)
            self.add = nn.Embedding(len(ADDUCTS), 1024)
            self.body = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, v1.FP_BITS))

        def forward(self, x, a):
            return self.body(self.inp(x) + self.add(a))
    torch.manual_seed(SEED)
    return FPNet()


def to_torch(Xb):
    import torch
    c = Xb.tocoo()
    return torch.sparse_coo_tensor(np.vstack([c.row, c.col]), c.data, c.shape, dtype=torch.float32).coalesce()


def stage_train(a):
    import torch
    from scipy import sparse
    torch.set_num_threads(a.threads)
    X = sparse.load_npz(O / "X.npz").tocsr()
    m = np.load(O / "meta.npz")
    add, row, es = m["add"], m["row"], m["es"]
    U = v1.Universe(UNIV)
    Y = U.fp  # packed; unpack per batch
    tr, va = np.flatnonzero(~es), np.flatnonzero(es)
    net = make_model(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    best, bad, hist = 1e9, 0, []

    def batches(idx, bs, shuffle):
        idx = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            yield (to_torch(X[b]), torch.from_numpy(add[b]),
                   torch.from_numpy(np.unpackbits(Y[row[b]], axis=1).astype(np.float32)))
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        tl = []
        for xb, ab, yb in batches(tr, 512, True):
            opt.zero_grad()
            loss = lossf(net(xb, ab), yb)
            loss.backward()
            opt.step()
            tl.append(loss.item())
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(net(xb, ab), yb).item() for xb, ab, yb in batches(va, 2048, False)]))
        hist.append({"epoch": ep, "train": float(np.mean(tl)), "val": vl, "sec": round(time.time() - t0, 1)})
        log(hist[-1])
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(net.state_dict(), O / "fpnet.pt")
        else:
            bad += 1
            if bad >= 3:
                break
    json.dump(hist, open(O / "train_hist.json", "w"), indent=1)


def predict_queries(q):
    """Per-molecule mean predicted bit probabilities for a query frame (molecule_id, adduct, peaks)."""
    import torch
    from scipy import sparse
    q = q[q["adduct"].isin(v1.AD10)].reset_index(drop=True)
    X = v1.featurize_df(q, FCFG)
    net = make_model(X.shape[1])
    net.load_state_dict(torch.load(O / "fpnet.pt"))
    net.eval()
    add = torch.from_numpy(np.array([ADDUCTS.index(x) for x in q["adduct"]], np.int64))
    with torch.no_grad():
        P = torch.sigmoid(net(to_torch(X), add)).numpy()
    ok = np.diff(X.indptr) > 0
    return {mid: P[g.index.values[ok[g.index.values]]].mean(0) for mid, g in q.groupby("molecule_id")
            if ok[g.index.values].any()}


def stage_eval(a):
    import exp018_ranker as e18
    U = v1.Universe(UNIV)
    res = {}
    for name, (q, t) in e18.query_sets(a.sets.split(",")).items():
        d = pd.read_parquet(E18 / f"feats_{name}.parquet")
        pred = predict_queries(q)
        fp = np.zeros(len(d))
        for mid, g in d.groupby("mid"):
            if mid not in pred:
                continue
            cb = U.bits(np.array([U.row[k] for k in g["ik"]]))
            p = pred[mid]
            fp[g.index.values] = (cb @ (p / np.linalg.norm(p))) / (np.linalg.norm(cb, axis=1) + 1e-12)
        d["fpnet"] = fp
        z = lambda c: (d[c] - d.groupby("mid")[c].transform("mean")) / (d.groupby("mid")[c].transform("std").fillna(0) + 1e-6)
        d["fuse"] = z("knn_mean") + z("fpnet")
        d.to_parquet(O / f"scored_{name}.parquet", index=False)
        r = {c: e18.mrr(d, d[c].values) for c in ("knn_mean", "fpnet", "fuse")}
        rng = np.random.default_rng(SEED)
        out = {c: round(v.mean(), 4) for c, v in r.items()}
        for c in ("fpnet", "fuse"):
            diff = (r[c] - r["knn_mean"]).values
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            out[f"{c}-knn"] = [round(diff.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)]
        if name == "val":
            T = pd.read_parquet(E18 / "targets.parquet").set_index("ik")
            for grp in ("c1x_tims", "C2"):
                ids = [m for m in r["knn_mean"].index if (T.loc[m, "pop"] if grp == "c1x_tims" else T.loc[m, "qtype"]) == grp]
                out[grp] = {c: round(r[c][ids].mean(), 4) for c in r}
        res[name] = out
        log(f"{name}: {out}")
    json.dump(res, open(O / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train", "eval"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sets", default="val,s1,s2")
    a = ap.parse_args()
    {"data": stage_data, "train": stage_train, "eval": stage_eval}[a.stage](a)

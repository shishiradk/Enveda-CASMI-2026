"""E6 offline: a PubChem candidate channel scored by our own fingerprint nets, for every bench molecule.

    python research/scripts/e6_pc_channel.py [--ppm 5] [--workers 6] [--nets models/fp_ho1_akriti]
                                           [--work results/c3/e6_work]

Per molecule (scenarios SV, S1, S2, S3): every PubChem structure within +-ppm of the target neutral mass
(external/pubchem/pubchem_rows.parquet, our own build), its 6,930-bit fingerprint (prep_data.fp_and_mass, the
engine's bit definition), score = fp @ zlog (the engine's FP-channel score = log-likelihood ratio), top 60 by score,
metric-key de-duplication (bench canonicaliser), 40 kept. zlog comes from the held-out ho1 nets, which never saw any
bench molecule, so the channel is leak-free on the bench (the full-data nets are not, and are for submissions only).

Resumable: every stage writes to results/c3/e6_work/ and is skipped when its output exists; fingerprints are cached
in chunks. Output: results/c3/e6_work/pc_lists_<scen>.pkl = {mid: {keys, raw, smiles, score}}.

--work only changes where those caches live (default unchanged), so a second scoring pass can run beside the ho1
cache instead of overwriting it. Note logits.pkl always takes S1/S2/S3 zlogs from bench/ho1/recs_*.pkl and only
derives SV from --nets, so --work with different nets changes only the SV rows.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results" / "c3" / "e6_work"
sys.path.insert(0, str(ROOT / "research" / "bench"))
sys.path.insert(0, str(ROOT / "research" / "train_pkg"))
SCENS = ("SV", "S1", "S2", "S3")


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def stage_logits(a):
    """{scen: {mid: (target_mass, zlog float32[6930])}} from the bench records; ho1 logits for SV computed here."""
    out = WORK / "logits.pkl"
    if out.exists():
        return pickle.load(open(out, "rb"))
    import bench
    res = {}
    for scen in ("S1", "S2", "S3"):
        D = pickle.load(open(bench.OUT / "ho1" / f"recs_{scen}.pkl", "rb"))
        res[scen] = {r["mid"]: (float(r["target"]), np.asarray(D["zmk"][r["mid"]], np.float32))
                     for r in D["recs"] if r["mid"] in D["zmk"] and D["zmk"][r["mid"]] is not None}
    D = pickle.load(open(bench.OUT / "e1" / "recs_SV.pkl", "rb"))
    bench.setup_env(ROOT / "research/bench/eng", a.workers)
    import casmi_engine as E, pv_fp, pyarrow.parquet as pq
    s, m, _ = pv_fp.load_fp_models(sorted(str(p) for p in Path(a.nets).glob("fp_*.pt")), "cpu")
    te = pq.read_table(ROOT / "test.parquet").to_pandas()
    z = bench.alt_zlogs(E, pv_fp, (s, m, "cpu"), te)
    res["SV"] = {r["mid"]: (float(r["target"]), np.asarray(z[r["mid"]], np.float32)) for r in D["recs"] if r["mid"] in z}
    pickle.dump(res, open(out, "wb"), protocol=4)
    log("logits: " + json.dumps({k: len(v) for k, v in res.items()}))
    return res


def stage_windows(a, L):
    out = WORK / f"windows_{a.ppm}ppm.parquet"
    if out.exists():
        return pd.read_parquet(out)
    import duckdb
    iv = pd.DataFrame([(scen, mid, t * (1 - a.ppm * 1e-6), t * (1 + a.ppm * 1e-6))
                       for scen, d in L.items() for mid, (t, _) in d.items()], columns=["scen", "mid", "lo", "hi"])
    # equi-join on 0.01 Da bins (a range join over 100M rows ran out of memory), exact window filter afterwards
    iv["b0"], iv["b1"] = np.floor(iv.lo * 100).astype(np.int64), np.floor(iv.hi * 100).astype(np.int64)
    ivb = pd.DataFrame([(r.scen, r.mid, r.lo, r.hi, b) for r in iv.itertuples() for b in range(r.b0, r.b1 + 1)],
                       columns=["scen", "mid", "lo", "hi", "bin"])
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '6GB'; SET temp_directory = '{(WORK / 'duck_tmp').as_posix()}'")
    con.register("ivb", ivb)
    pc = (ROOT / "external/pubchem/pubchem_rows.parquet").as_posix()
    w = con.sql(f"""SELECT ivb.scen, ivb.mid, p.ik, p.smiles
                    FROM read_parquet('{pc}') p JOIN ivb ON CAST(floor(p.mass * 100) AS BIGINT) = ivb.bin
                    WHERE p.mass BETWEEN ivb.lo AND ivb.hi""").df()
    w = w.drop_duplicates(["scen", "mid", "ik"]).reset_index(drop=True)
    w.to_parquet(out, index=False)
    log(f"windows: {len(w):,} (molecule, structure) pairs, {w.ik.nunique():,} unique structures, "
        f"median {int(w.groupby(['scen', 'mid']).size().median())} per molecule")
    return w


def _fp_init():
    import prep_data as PD
    PD._fp_init(str(ROOT / "external/bench_inputs/coco/fp_bits.npy"))


def _fp_one(smi):
    import prep_data as PD
    r = PD.fp_and_mass(smi)
    return None if r is None else r[0]  # already packed (np.packbits) by prep_data


def stage_fps(a, w):
    """Unique structures -> packed 6,930-bit fingerprints, in resumable chunks of 50k."""
    from multiprocessing import Pool
    u = w.drop_duplicates("ik")[["ik", "smiles"]].sort_values("ik").reset_index(drop=True)
    (WORK / "fp").mkdir(exist_ok=True)
    CH = 50000
    if all((WORK / "fp" / f"chunk_{i // CH:05d}.npz").exists() for i in range(0, len(u), CH)):
        return sorted((WORK / "fp").glob("chunk_*.npz"))
    with Pool(a.workers, initializer=_fp_init) as pool:
        for i in range(0, len(u), CH):
            f = WORK / "fp" / f"chunk_{i // CH:05d}.npz"
            if f.exists():
                continue
            part = u.iloc[i:i + CH]
            fps = pool.map(_fp_one, part.smiles.tolist(), chunksize=200)
            ok = np.array([x is not None for x in fps])
            nb = len(next(x for x in fps if x is not None))
            arr = np.stack([x if x is not None else np.zeros(nb, np.uint8) for x in fps])
            np.savez(f, ik=np.asarray(part.ik.tolist(), dtype="U14"), fp=arr, ok=ok)
            log(f"fingerprints {min(i + CH, len(u)):,}/{len(u):,}")
    return [f for f in sorted((WORK / "fp").glob("chunk_*.npz"))]


def stage_scores(a, L, w, chunks):
    """score = fp @ zlog for every (molecule, structure) pair in w, streamed chunk by chunk (all 4.9M fingerprints
    at once would need ~4.3 GB). Unparseable structures get -inf. Cached as scores_<ppm>ppm.npy (aligned with w)."""
    out = WORK / f"scores_{a.ppm}ppm.npy"
    if out.exists():
        return np.load(out)
    gk = list(zip(w.scen, w.mid))
    keys = sorted(set(gk))
    gid = pd.Series(range(len(keys)), index=pd.MultiIndex.from_tuples(keys)).reindex(gk).values
    Z = np.stack([L[sc][m][1] for sc, m in keys])
    nz = Z.shape[1]
    # global position of each w row's structure in the ik-sorted chunk order
    ik_all = np.concatenate([np.load(f, allow_pickle=True)["ik"].astype("U14") for f in chunks])
    pos = pd.Series(np.arange(len(ik_all)), index=ik_all).reindex(w.ik.values).values.astype(np.int64)
    del ik_all
    sc = np.full(len(w), -np.inf, np.float32)
    order = np.argsort(pos, kind="stable")
    bounds = np.searchsorted(pos[order], np.arange(len(chunks) + 1) * 50000)
    for c, f in enumerate(chunks):
        z = np.load(f, allow_pickle=True)
        fp, ok = z["fp"], z["ok"]
        rows = order[bounds[c]:bounds[c + 1]]
        for j in range(0, len(rows), 4000):
            r = rows[j:j + 4000]
            loc = pos[r] - c * 50000
            F = np.unpackbits(fp[loc], axis=1)[:, :nz].astype(np.float32)
            v = np.einsum("ij,ij->i", F, Z[gid[r]])
            sc[r] = np.where(ok[loc], v, -np.inf)
        if c % 10 == 0:
            log(f"scores: chunk {c + 1}/{len(chunks)}")
    np.save(out, sc)
    log(f"scores: {np.isfinite(sc).sum():,}/{len(sc):,} pairs scored")
    return sc


def stage_lists(a, L, w, sc):
    import bench
    bench.setup_env(ROOT / "research/bench/eng", a.workers)
    import casmi_engine as E
    smi_of = dict(zip(w.ik, w.smiles))
    w = w.assign(score=sc)
    w = w[np.isfinite(w.score)].sort_values(["scen", "mid", "score"], ascending=[True, True, False], kind="mergesort")
    for scen in SCENS:
        out = WORK / f"pc_lists_{scen}.pkl"
        if out.exists():
            continue
        top = {mid: list(zip(g.ik.values[:60], g.score.values[:60].astype(float)))
               for mid, g in w[w.scen == scen].groupby("mid", sort=False)}
        C = bench.canon_many(E, sorted({smi_of[k] for v in top.values() for k, _ in v}), a.workers)
        res = {}
        for mid, v in top.items():
            rec = {"keys": [], "raw": [], "smiles": [], "score": []}
            for k, s in v:
                key = C.get(smi_of[k]) or k
                if key in rec["keys"]:
                    continue
                rec["keys"].append(key); rec["raw"].append(k); rec["smiles"].append(smi_of[k]); rec["score"].append(s)
                if len(rec["keys"]) >= 40:
                    break
            res[mid] = rec
        pickle.dump(res, open(out, "wb"), protocol=4)
        log(f"{scen}: PubChem lists for {len(res)} molecules")


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nets", default=str(ROOT / "models" / "fp_ho1_akriti"))
    ap.add_argument("--work", default=str(WORK), help="cache/output dir (default: the original ho1 cache)")
    a = ap.parse_args()
    WORK = Path(a.work)
    WORK.mkdir(parents=True, exist_ok=True)
    L = stage_logits(a)
    w = stage_windows(a, L)
    chunks = stage_fps(a, w)
    sc = stage_scores(a, L, w, chunks)
    stage_lists(a, L, w, sc)
    log("done")


if __name__ == "__main__":
    main()

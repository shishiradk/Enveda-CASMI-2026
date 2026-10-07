"""Local test bench for the E1 engine (research/kaggle_e1/casmi_e1.ipynb) on the leak-controlled proxy scenarios.

    python research/bench/bench.py all  --tag e1            # prepare + run + eval  (the one command)
    python research/bench/bench.py run  --tag e1 [--scen S1,S2,S3] [--limit 20] [--workers 3] [--eng-dir DIR]
    python research/bench/bench.py rescore --tag e1          # new rankers / blends from the saved channel records
    python research/bench/bench.py eval --tag e1 [--score blend]

Scenarios (same molecules, held-out keys and "correct" sets as research/kaggle_v2/proxy_eval2.py):
  S1  250 enveda-np-examples molecules, Class-1-like: every enveda-np-examples spectrum leaves the library.
  S2  same queries as Class 2: every spectrum of the targets and their aliases (results/kaggle_v1_proxy/held_keys)
      leaves the library; their structures stay in the pool only if COCONUT / ChEBI / LIPID MAPS has them.
  S3  300 natural products that are not in COCONUT (results/kaggle_v2_proxy/s3_held): all their spectra leave the
      library. E1 has no PubChem pool, so the truth is reachable only if ChEBI / LIPID MAPS has it.
  S3i (written together with S3 unless --no-inject) S3 with the true structures put back into the pool WITHOUT spectra (an oracle structure database):
      a second Class-2 population drawn from gnps/riken/mona/massbank/msdial instead of enveda-np-examples.

The engine code is the notebook's own (extract_engine.py); this file only replaces E.main / eng_runner.py by the same
call sequence with a row mask instead of a filtered train.parquet (see mask_library) and keeps every per-candidate score
so that variants and the error analysis are computed offline by `eval`.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "bench"
INPUTS = ROOT / "external" / "bench_inputs"            # local copies of the notebook's Kaggle datasets
ALT_FP = ROOT / "external" / "bench_inputs_alt" / "megayak_fp"  # leak-free FPNets (megayak, np-examples held out)
TRAIN = ROOT / "train.parquet"
V1P, V2P = ROOT / "results" / "kaggle_v1_proxy", ROOT / "results" / "kaggle_v2_proxy"
NPX = "enveda-np-examples"
W_PV = 0.88
SCORE_SETS = ("blend", "pv", "ours", "pv_nofp", "ours_cv", "blend_cv", "pv_cv", "clean", "pv_mk", "blend_mk",
              "blend_mk_cv", "pv_mk_cv", "clean_mk", "fp_only", "fp_only_mk",
              "pv_kf", "ours_kf", "clean_kf")  # the *_kf sets are added by bench_cvrank.py


def setup_env(eng_dir, threads):
    os.environ["CASMI_ROOTS"] = str(INPUTS)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(eng_dir), str(HERE), os.environ.get("PYTHONPATH", "")])
    for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    (OUT / "cache" / "numba").mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(OUT / "cache" / "numba")
    sys.path.insert(0, str(eng_dir))


def log(m):
    print(f"[bench {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------------------------
# prepare: queries with every column the engine reads, truth table
# ------------------------------------------------------------------------------------------------------------------
def prepare():
    import duckdb
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    src = {"S12": V1P / "proxy_test.parquet", "S3": V2P / "s3_test.parquet"}
    for name, q in src.items():
        dst = OUT / f"queries_{name}.parquet"
        if dst.exists():
            continue
        duckdb.sql(f"""COPY (SELECT q.molecule_id, q.spectrum_id, t.ms2_mzs, t.ms2_normalized_intensities, t.adduct,
                                    t.ionization_mode, t.instrument_type, t.precursor_mz, t.collision_energy_ev
                             FROM read_parquet('{TRAIN.as_posix()}', file_row_number=true) t
                             JOIN '{q.as_posix()}' q ON ('r' || t.file_row_number) = q.spectrum_id
                             ORDER BY q.molecule_id, q.spectrum_id) TO '{dst.as_posix()}' (FORMAT PARQUET)""")
        n = duckdb.sql(f"SELECT count(*), count(DISTINCT molecule_id) FROM '{dst.as_posix()}'").fetchone()
        n0 = duckdb.sql(f"SELECT count(*) FROM '{q.as_posix()}'").fetchone()[0]
        assert n[0] == n0, (n, n0)
        log(f"queries {name}: {n[0]} spectra / {n[1]} molecules")
    tp = OUT / "truth.parquet"
    if not tp.exists():
        rows = []
        for scen, cache in (("S12", V2P / "cache_S1.pkl"), ("S3", V2P / "cache_S3.pkl")):
            correct = pickle.load(open(cache, "rb"))["correct"]
            keys = ",".join(f"'{k}'" for k in correct)
            smi = dict(duckdb.sql(f"""SELECT inchikey14, any_value(normalized_smiles), any_value(molecular_formula)
                                      FROM '{TRAIN.as_posix()}' WHERE inchikey14 IN ({keys}) GROUP BY 1""").df()
                       .set_index("inchikey14").apply(tuple, axis=1))
            for t, ok in correct.items():
                rows.append(dict(qset=scen, mid=t, smiles=smi[t][0], formula=smi[t][1], correct=";".join(sorted(ok))))
        pd.DataFrame(rows).to_parquet(tp, index=False)
        log(f"truth table: {len(rows)} molecules")


def load_truth():
    import pandas as pd
    import duckdb as _ddb
    # Use duckdb reader: avoids pyarrow "Repetition level histogram size mismatch" on
    # parquet files written with older pyarrow versions.
    def _read(p):
        return _ddb.sql(f"SELECT * FROM '{p.as_posix()}'").df()
    T = _read(OUT / "truth.parquet")
    if (OUT / "truth_SV.parquet").exists():  # scenario SV (research/scripts/sv_prepare.py)
        T = pd.concat([T, _read(OUT / "truth_SV.parquet")], ignore_index=True)
    return {q: {r.mid: dict(smiles=r.smiles, formula=r.formula, correct=set(r.correct.split(";")))
                for r in g.itertuples()} for q, g in T.groupby("qset")}


def held_keys(scen):
    import pandas as pd
    if scen == "S1":
        return None
    return set(pd.read_parquet((V1P / "held_keys.parquet") if scen == "S2" else (V2P / "s3_held.parquet"))["ik"])


# ------------------------------------------------------------------------------------------------------------------
# (a) library leak. The engine reads 9 train columns (casmi_engine.load_library) and derives BOTH the spectral library
# and the train part of the candidate pool from them. Removing rows from the file would
#   - library: drop the rows from the exact-mass index and the analog representatives  (wanted)
#   - pool: drop every structure whose rows are all gone                               (WRONG for Class 2: the
#     COCONUT candidate file shipped with the engine excludes train structures, so a filtered file deletes the truth
#     of 99.6% of the S2 targets from the pool although the real COCONUT contains them).
# The mask therefore works in memory: row_p = -1 for the removed rows, and a train-origin structure stays in the pool
# iff a row remains OR the real COCONUT has it. One 3 GB library load serves every scenario. write_filtered_train()
# writes the plain filtered file; it is only valid for runs where no held structure has to stay (see the report).
# ------------------------------------------------------------------------------------------------------------------
def row_mask(scen, ik, is_npx):
    import numpy as np
    import pandas as pd
    if scen == "S1":
        return is_npx.copy()
    return pd.Series(ik).isin(held_keys("S2" if scen == "S2" else "S3")).values


def write_filtered_train(scen, dst):
    import duckdb
    cols = ("inchikey14, normalized_smiles, adduct, precursor_mz, ionization_mode, ms2_mzs, "
            "ms2_normalized_intensities, molecular_formula, instrument_type")
    if scen == "S1":
        where = f"ingest_lib <> '{NPX}'"
    else:
        hk = ((V1P / "held_keys.parquet") if scen == "S2" else (V2P / "s3_held.parquet")).as_posix()
        where = f"inchikey14 NOT IN (SELECT ik FROM '{hk}')"
    duckdb.sql("SET preserve_insertion_order=true")
    duckdb.sql(f"COPY (SELECT {cols} FROM '{TRAIN.as_posix()}' WHERE {where}) TO '{Path(dst).as_posix()}' "
               f"(FORMAT PARQUET, ROW_GROUP_SIZE 120000)")


def read_is_npx(path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    if "ingest_lib" not in f.schema_arrow.names:
        return np.zeros(f.metadata.num_rows, bool)
    out = []
    for i in range(f.num_row_groups):
        c = f.read_row_group(i, columns=["ingest_lib"]).column(0).cast(pa.string())
        out.append(np.asarray(c.to_pylist(), dtype=object) == NPX)
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------------------------
def fit_or_load(name, fn, deps):
    h = hashlib.md5(json.dumps(deps, sort_keys=True, default=str).encode()).hexdigest()[:10]
    p = OUT / "cache" / f"rankers_{name}_{h}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    r = fn()
    pickle.dump(r, open(p, "wb"))
    return r


def alt_zlogs(E, pv_fp, models, te):
    """Fingerprint logits from another FP model pair, computed exactly as casmi_engine.compute_channels does."""
    import numpy as np
    out = {}
    for mid, sub in te.groupby("molecule_id", sort=True):
        specs = []
        for r in sub.itertuples():
            a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)
            k_ = a_ <= float(r.precursor_mz) + 2.0
            a_, b_ = a_[k_], b_[k_]
            specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))
        modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)
        ces = np.array([E.parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)
        z = pv_fp.molecule_logits(models, specs, sub.precursor_mz.values.astype(np.float32),
                                  list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),
                                  ces, modes)
        out[mid] = None if z is None else z.astype(np.float32)
    return out


def canon_many(E, smiles, workers):
    """Tautomer-canonical InChIKey14 (the metric key) with a disk cache shared by all runs."""
    from multiprocessing import Pool
    p = OUT / "cache" / "canon.pkl"
    C = pickle.load(open(p, "rb")) if p.exists() else {}
    need = sorted(set(smiles) - set(C))
    if need:
        t0 = time.time()
        with Pool(workers) as mp:
            C.update(zip(need, mp.map(E.canon_key, need, chunksize=32)))
        pickle.dump(C, open(p, "wb"))
        log(f"canonicalised {len(need):,} new SMILES in {time.time() - t0:.0f}s (cache {len(C):,})")
    return C


def run(a):
    eng_dir = Path(a.eng_dir).resolve()
    setup_env(eng_dir, a.workers)
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    torch.set_num_threads(a.workers)
    import casmi_engine as E
    import pv_fp
    import rdkit
    # A NULL collision_energy_ev makes casmi_engine.parse_ce return NaN -> NaN fingerprint logits for the whole molecule.
    # The competition test has an energy on every row; the S3 queries (public-library spectra) mostly do not. Read a
    # missing energy as 25 eV, the trainer's default (same repair as bench_fixfp.py applied to the first S3 run).
    _pce = E.parse_ce
    E.parse_ce = lambda v: (lambda x: x if np.isfinite(x) else 25.0)(_pce(v))
    prepare()
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    # eng_runner.py settings (kept in sync by hand; the values are asserted against the extracted runner text)
    runner = (eng_dir / "eng_runner.py").read_text()
    assert "E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200" in runner and "w_pv=0.88" in runner
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    train = Path(a.train) if a.train else TRAIN
    T0 = time.time()
    L = E.load_library(str(train))
    log(f"library {len(L['prec']):,} spectra from {train.name}")
    ik_all = L["ik"].copy()
    is_npx = read_is_npx(train)
    pc = OUT / "cache" / f"pool_{train.stem}_{train.stat().st_size}_{rdkit.__version__}.pkl"
    if pc.exists():  # 255k train structures take ~15 min to fingerprint on 3 workers
        E.POOL = pickle.load(open(pc, "rb"))
        log(f"pool {len(E.POOL['mass']):,} structures (cache)")
    else:
        E.POOL = E.build_pool(L, a.workers)
        pickle.dump(E.POOL, open(pc, "wb"), protocol=4)
    P = E.POOL
    ext = set(pickle.load(open(INPUTS / "coco" / "coco_meta.pkl", "rb"))["keys"]) | \
        set(pickle.load(open(INPUTS / "bio" / "bio_meta.pkl", "rb"))["keys"])
    from_train = ~pd.Series(P["keys"]).isin(ext).values
    import duckdb
    coco_real = set(duckdb.sql(f"SELECT ik FROM '{(ROOT / 'results/kaggle_v1_assets/universe.parquet').as_posix()}' "
                               "WHERE src = 'coconut'").df()["ik"])
    in_coco = pd.Series(P["keys"]).isin(coco_real).values
    row_p_full = P["k2i"].reindex(ik_all).fillna(-1).values.astype(np.int64)
    del L["ik"], L["smi"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fp_public = sorted(str(p) for p in (INPUTS / "fp").glob("fp_*.pt"))
    single, merged, _ = pv_fp.load_fp_models(fp_public, dev)
    models = (single, merged, dev)
    models_mk = None
    alt_fp = Path(a.alt_fp) if a.alt_fp else ALT_FP
    if alt_fp.exists() and not a.no_alt_fp:
        s2, m2, _ = pv_fp.load_fp_models(sorted(str(p) for p in alt_fp.glob("*.pt")), dev)
        models_mk = (s2, m2, dev)
    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}; alt {alt_fp if models_mk else 'no'}"
        f"{f' ({len(s2)} single + {len(m2)} merged)' if models_mk else ''}")
    RK = load_rankers(E)
    truth = load_truth()
    TM = {}

    def timed(mod, name):
        f = getattr(mod, name)

        def w(*x, **k):
            t = time.time(); r = f(*x, **k); TM[name] = TM.get(name, 0.0) + time.time() - t
            return r
        setattr(mod, name, w)
    for nm in ("search", "search_shift_rows", "search_shift_pre"):
        timed(E.pv, nm)
    timed(pv_fp, "molecule_logits"); timed(E, "top_analogs")
    meta = dict(tag=a.tag, eng_dir=str(eng_dir), rdkit=rdkit.__version__, torch=torch.__version__, dev=dev,
                workers=a.workers, limit=a.limit, train=str(train), scen={},
                alt_fp=str(alt_fp) if models_mk else None)

    for scen in a.scen.split(","):
        t0 = time.time()
        base = scen[:2]
        qset = "S12" if base in ("S1", "S2") else ("SV" if scen == "SV" else "S3")
        if scen in ("T", "SV"):  # SV: the same queries, their exact train duplicates masked (sv_prepare.py)  # parity check: the real test queries, nothing masked, against the Kaggle run's eng_lists.json
            te = pq.read_table(ROOT / "test.parquet").to_pandas()
            a.limit = a.limit or (40 if scen == "T" else 0)
        else:
            te = pq.read_table(OUT / f"queries_{qset}.parquet").to_pandas()
        if a.limit:
            keep = sorted(te.molecule_id.unique())[:a.limit]
            te = te[te.molecule_id.isin(keep)]
        if a.premasked or scen == "T":
            excl = np.zeros(len(ik_all), bool)
        elif scen == "SV":
            excl = np.zeros(len(ik_all), bool)
            excl[np.load(OUT / "sv_dup_rows.npy")] = True
        else:
            excl = row_mask(base, ik_all, is_npx)
        L["row_p"] = np.where(excl, -1, row_p_full)
        alive_keys = set(pd.unique(ik_all[~excl]))
        # prvsiyan's coco_* files hold only the COCONUT structures that are NOT in train, so a held-out molecule's
        # pool entry is train-origin even when it is a COCONUT structure. It must stay in the pool exactly when the
        # real COCONUT has it (our own COCONUT universe decides), otherwise Class 2 would lose its truth by construction.
        alive = ~from_train | pd.Series(P["keys"]).isin(alive_keys).values | in_coco
        alive_strict = alive
        if a.inject and base == "S3":  # oracle structure DB: channels are computed once on the superset pool
            alive = np.ones(len(alive), bool)
        hk = held_keys(base) if scen not in ("T", "SV") else None
        if scen in ("T", "SV"):
            alive = alive_strict = np.ones(len(alive), bool)
        elif hk is not None:  # leak gates
            assert not (set(pd.unique(ik_all[L["row_p"] >= 0])) & hk), "held key still has library rows"
            if True:
                assert not (from_train & alive_strict & ~in_coco & pd.Series(P["keys"]).isin(hk).values).any(),                     "held train-only structure in pool"
        elif not a.premasked:
            assert not is_npx[L["row_p"] >= 0].any()
        _pw = E.pool_window.__wrapped__ if hasattr(E.pool_window, "__wrapped__") else E.pool_window

        def pool_window(t, ppm, _pw=_pw, alive=alive):
            idx = _pw(t, ppm)
            return idx[alive[idx]]
        pool_window.__wrapped__ = _pw
        E.pool_window = pool_window
        log(f"{scen}: {int(excl.sum()):,} library rows masked, {int((~alive).sum()):,} train structures out of the pool, "
            f"{te.molecule_id.nunique()} molecules / {len(te)} spectra")
        I = E.build_index(L)
        mols, recs = E.compute_channels(L, I, models, te)
        del I
        zmk = alt_zlogs(E, pv_fp, models_mk, te) if models_mk else {}
        log(f"{scen}: channels done {time.time() - t0:.0f}s; cumulative timers {json.dumps({k: round(v) for k, v in TM.items()})}")
        E.compute_frag(recs, a.workers)
        log(f"{scen}: frag done {time.time() - t0:.0f}s")
        if scen == "T":
            parity(E, P, recs, RK['ours'][0], RK['ours'][1], RK['pv'][0], out, a.workers)
            continue
        # everything the rankers need, so `rescore` can add score sets without recomputing the channels
        pickle.dump(dict(recs=recs, zmk=zmk, alive=np.packbits(alive), alive_strict=np.packbits(alive_strict),
                         n=len(alive), excl_rows=int(excl.sum())), open(out / f"recs_{scen}.pkl", "wb"), protocol=4)
        variants = variants_of(scen, recs, alive, alive_strict)
        for scen, recs, alive in variants:
            save_scenario(E, P, a.workers, out, scen, qset, {m: int(n) for m, n in te.molecule_id.value_counts().items()},
                          recs, alive, from_train, truth, meta, t0, RK, zmk, int(excl.sum()))
    meta["total_sec"] = round(time.time() - T0, 1)
    old = json.load(open(out / "run_meta.json")) if (out / "run_meta.json").exists() else {"scen": {}}
    meta["scen"] = {**old.get("scen", {}), **meta["scen"]}
    json.dump(meta, open(out / "run_meta.json", "w"), indent=1)


def parity(E, P, recs, ours, blocks, pvr, out, workers):
    """Same molecules through eng_runner.py's list construction, compared with the lists the Kaggle notebook wrote."""
    import numpy as np
    K = json.load(open(OUT / "kaggle_e1_output" / "eng_lists.json"))
    bl = E.blend_scores(E.score_molecules(recs, pvr, ["base"], use_fp=True),
                        E.score_molecules(recs, ours, blocks, use_fp=False), W_PV)
    order = {r["mid"]: r["cand"][np.argsort(-bl[r["mid"]], kind="mergesort")[:80]] for r in recs if r["mid"] in bl}
    C = canon_many(E, [P["smiles"][c] for o in order.values() for c in o], workers)
    rows = []
    for mid, o in order.items():
        keys, seen = [], set()
        for c in o:
            k = C.get(P["smiles"][c])
            if k is None or k in seen:
                continue
            seen.add(k); keys.append(k)
            if len(keys) >= 40:
                break
        kk = K.get(str(mid), {}).get("keys", [])
        rows.append(dict(mid=str(mid), n_local=len(keys), n_kaggle=len(kk), top1_same=bool(keys and kk and keys[0] == kk[0]),
                         top25_jaccard=len(set(keys[:25]) & set(kk[:25])) / max(1, len(set(keys[:25]) | set(kk[:25]))),
                         top25_identical_order=keys[:25] == kk[:25], top5_identical_order=keys[:5] == kk[:5]))
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = {c: round(float(df[c].mean()), 4) for c in ("top1_same", "top25_jaccard", "top25_identical_order", "top5_identical_order")}
    summ["n"] = len(df)
    df.to_csv(out / "parity.csv", index=False)
    json.dump(summ, open(out / "parity.json", "w"), indent=1)
    log(f"parity vs Kaggle E1 run: {json.dumps(summ)}")


def variants_of(scen, recs, alive, alive_strict):
    import numpy as np
    if np.array_equal(alive, alive_strict):
        return [(scen, recs, alive)]
    sub = []  # S3 proper = the S3i records restricted to the structures E1 really has
    for r in recs:
        k = alive_strict[r["cand"]] if len(r["cand"]) else np.zeros(0, bool)
        sub.append({f: (v[k] if f in PER_CAND and isinstance(v, np.ndarray) and len(v) == len(k) else v)
                    for f, v in r.items()})
    return [(scen, sub, alive_strict), (scen + "i", recs, alive)]


def rescore(a):
    """Recompute every score set from the saved channel records (new rankers / blends) without the library."""
    eng_dir = Path(a.eng_dir).resolve()
    setup_env(eng_dir, a.workers)
    import numpy as np
    import pandas as pd
    import rdkit
    import casmi_engine as E
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    out = OUT / a.tag
    pc = OUT / "cache" / f"pool_{TRAIN.stem}_{TRAIN.stat().st_size}_{rdkit.__version__}.pkl"
    E.POOL = P = pickle.load(open(pc, "rb"))
    ext = set(pickle.load(open(INPUTS / "coco" / "coco_meta.pkl", "rb"))["keys"]) | \
        set(pickle.load(open(INPUTS / "bio" / "bio_meta.pkl", "rb"))["keys"])
    from_train = ~pd.Series(P["keys"]).isin(ext).values
    RK = load_rankers(E)
    truth = load_truth()
    meta = json.load(open(out / "run_meta.json")) if (out / "run_meta.json").exists() else {"scen": {}}
    for scen in a.scen.split(","):
        rp = out / f"recs_{scen}.pkl"
        if not rp.exists():
            continue
        t0 = time.time()
        D = pickle.load(open(rp, "rb"))
        alive = np.unpackbits(D["alive"])[:D["n"]].astype(bool)
        strict = np.unpackbits(D["alive_strict"])[:D["n"]].astype(bool)
        qset = "S12" if scen[:2] in ("S1", "S2") else "S3"
        prev = pickle.load(open(out / f"{scen}.pkl", "rb"))
        n_spec = {m: r["n_spec"] for m, r in prev.items()}
        for sc, recs, al in variants_of(scen, D["recs"], alive, strict):
            save_scenario(E, P, a.workers, out, sc, qset, n_spec, recs, al, from_train, truth, meta, t0, RK, D["zmk"],
                          D["excl_rows"])
    json.dump(meta, open(out / "run_meta.json", "w"), indent=1)


PER_CAND = ("cand", "lib", "lib_mean", "libsh", "lib_cnt", "frag", "frag_ad", "frag_mean", "frag_pol")


def load_rankers(E):
    """Rankers of the notebook (ours, pv) plus the leak-controlled refits (ours_cv, pv_cv); fitted once and cached."""
    import numpy as np
    sim_path, pv_path = E.find("sim_rank_rows_nofp.npz"), E.find("rank_train.npz")
    gbm = dict(W_A=E.RANK.W_A, SEEDS=E.RANK.SEEDS, GBM=E.RANK.GBM, PVGBM=E.CFG.GBM, sk=__import__("sklearn").__version__)
    ours, blocks = fit_or_load("ours", lambda: E.fit_rankers(sim_path), gbm)
    pvr, _ = fit_or_load("pv", lambda: E.fit_pv_rankers(pv_path), gbm)

    def fit_ours_cv():  # (b) ranker-row leak: megayak rows of the enveda-np-examples queries (= S1/S2 targets) removed
        z = np.load(sim_path, allow_pickle=True)
        keep = z["Q"] != list(z["sets"]).index("np")
        X, Y, S = z["X"][keep], z["Y"][keep], z["S"][keep]
        r = [E._hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, E.RANK.GBM) for wA in E.RANK.W_A for sd in E.RANK.SEEDS]
        log(f"ours_cv ranker: {len(r)} GBMs on {X.shape[0]:,} rows (np-examples rows dropped)")
        return r
    RK = {"ours": (ours, blocks), "pv": (pvr, ["base"]), "ours_cv": (fit_or_load("ours_cv", fit_ours_cv, gbm), blocks)}
    gp = OUT / "cache" / "pv_np_groups.json"  # written by bench_pvgroups.py: rank_train.npz groups = our S1/S2 molecules
    if gp.exists():
        npg = json.load(open(gp))["groups"]

        def fit_pv_cv():
            z = np.load(pv_path)
            keep = ~np.isin(z["G"], npg)
            X, Y, M = z["X"][keep], z["Y"][keep], z["M"][keep]
            r = [E._hgb(X, Y, np.where(M == 0, w1, 1.0 - w1), sd, E.CFG.GBM) for w1 in (0.30, 0.60) for sd in (0, 1, 2, 3)]
            log(f"pv_cv ranker: {len(r)} GBMs on {X.shape[0]:,} rows ({len(npg)} np-examples query groups dropped)")
            return r
        RK["pv_cv"] = (fit_or_load("pv_cv", fit_pv_cv, dict(gbm, groups=len(npg))), ["base"])
    return RK


def save_scenario(E, P, workers, out, scen, qset, n_spec, recs, alive, from_train, truth, meta, t0, RK, zmk, n_excl):
        import numpy as np
        mk = [dict(r, zlog=zmk.get(r["mid"])) for r in recs] if zmk else None
        S = {"ours": E.score_molecules(recs, *RK["ours"], use_fp=False),
             "pv": E.score_molecules(recs, *RK["pv"], use_fp=True),
             "pv_nofp": E.score_molecules(recs, *RK["pv"], use_fp=False),
             "ours_cv": E.score_molecules(recs, *RK["ours_cv"], use_fp=False)}
        if mk:
            S["pv_mk"] = E.score_molecules(mk, *RK["pv"], use_fp=True)
        if "pv_cv" in RK:
            S["pv_cv"] = E.score_molecules(recs, *RK["pv_cv"], use_fp=True)
            S["clean"] = E.blend_scores(S["pv_cv"], S["ours_cv"], W_PV)
            if mk:
                S["pv_mk_cv"] = E.score_molecules(mk, *RK["pv_cv"], use_fp=True)
                S["clean_mk"] = E.blend_scores(S["pv_mk_cv"], S["ours_cv"], W_PV)
        S["blend"] = E.blend_scores(S["pv"], S["ours"], W_PV)
        S["blend_cv"] = E.blend_scores(S["pv"], S["ours_cv"], W_PV)
        if mk:
            S["blend_mk"] = E.blend_scores(S["pv_mk"], S["ours"], W_PV)
            S["blend_mk_cv"] = E.blend_scores(S["pv_mk"], S["ours_cv"], W_PV)
        S["fp_only"], S["fp_only_mk"] = {}, {}
        for r in recs:
            if not len(r["cand"]):
                continue
            cf = E.pool_fps(r["cand"]).astype(np.float32)
            for nm, z in (("fp_only", r.get("zlog")), ("fp_only_mk", zmk.get(r["mid"]))):
                if z is not None:
                    S[nm][r["mid"]] = cf @ z
        log(f"{scen}: scored {time.time() - t0:.0f}s")
        # candidates whose metric key is needed: top-80 of every score set (the notebook dedups the blend top-80)
        top = {}
        for r in recs:
            if not len(r["cand"]):
                continue
            u = set()
            for nm, sc in S.items():
                if r["mid"] in sc:
                    u |= set(r["cand"][np.argsort(-sc[r["mid"]], kind="mergesort")[:80]].tolist())
            top[r["mid"]] = sorted(u)
        tsmi = truth[qset]
        C = canon_many(E, [P["smiles"][c] for u in top.values() for c in u] + [t["smiles"] for t in tsmi.values()],
                       workers)
        pool_key_set = set(P["keys"][alive])
        res = {}
        for r in recs:
            mid = r["mid"]
            ok = tsmi[mid]["correct"]
            rec = dict(n_spec=n_spec.get(mid, 0), truth_in_pool=bool(ok & pool_key_set),
                       truth_canon=C.get(tsmi[mid]["smiles"]))
            if len(r["cand"]):
                rec.update(target=r["target"], top_sim=float(r["ana"][0][1]) if r["ana"] else 0.0, keys=P["keys"][r["cand"]].astype(str), mass=P["mass"][r["cand"]],
                           from_train=from_train[r["cand"]], lib=r["lib"],
                           scores={nm: sc[mid].astype(np.float32) for nm, sc in S.items() if mid in sc},
                           top={int(np.where(r["cand"] == c)[0][0]): (P["smiles"][c], C.get(P["smiles"][c]))
                                for c in top[mid]})
            res[mid] = rec
        pickle.dump(res, open(out / f"{scen}.pkl", "wb"))
        meta["scen"][scen] = dict(molecules=len(res), sec=round(time.time() - t0, 1), masked_rows=n_excl,
                                  pool_removed=int((~alive).sum()))
        log(f"{scen}: saved {len(res)} molecules, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "run", "rescore", "eval", "all", "write-train"])
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--scen", default="S1,S2,S3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eng-dir", default=str(HERE / "eng"))
    ap.add_argument("--train", default="", help="alternative train parquet (e.g. a filtered file)")
    ap.add_argument("--premasked", action="store_true", help="--train is already filtered for the scenario: no row mask")
    ap.add_argument("--no-alt-fp", action="store_true")
    ap.add_argument("--alt-fp", default="", help="folder with the alternative FPNets for the *_mk score sets (default: megayak)")
    ap.add_argument("--no-inject", dest="inject", action="store_false", help="skip the S3i oracle-pool variant")
    ap.add_argument("--score", default="blend")
    ap.add_argument("--dst", default="")
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare()
    elif a.stage == "write-train":
        write_filtered_train(a.scen, a.dst)
    elif a.stage == "rescore":
        rescore(a)
    else:
        if a.stage in ("run", "all"):
            run(a)
        if a.stage in ("eval", "all"):
            sys.path.insert(0, str(HERE))
            import bench_eval
            bench_eval.main(a.tag, a.score)

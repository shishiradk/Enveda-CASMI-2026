"""EXP-008 performance-only implementation (optimization v1) of the locked pseudo-C3 experiment.

The scientific reference remains research/scripts/exp008_c3_smoke.py (unchanged). This file imports
all of its scientific functions (universe, parent groups, pools, representative rule, tie-aware RR,
chance RR, analogue score S(c), failure categories, bootstrap) and replaces ONLY the execution of the
wide-window ModifiedCosineGreedy scoring:

  reference: per (query, evidence) pair -> pandas .loc -> matchms Spectrum (metadata harmonization)
             -> ModifiedCosineGreedy.pair (Python wrapper around numba helpers)
  fast:      evidence peaks materialized once into a flat, pre-sorted store; one numba kernel per
             query arm that calls matchms's OWN compiled helpers (collect_peak_pairs,
             score_best_matches) with the same argument order, array layout, shift sign, cosine
             fallback (|shift| <= tol) and stable-mergesort-reversed pair ordering as matchms.pair.
             Parallel over evidence spectra (prange); each score is independent, so results do not
             depend on thread count or scheduling.

Baseline B (direct ranker, ~hundreds of pairs/query) uses the reference code path unchanged.

Stages
  build-store     flat peak store of all enveda-180 monomer-adduct representatives (cache)
  prepare-full    407-query full-run population (same eligibility/tier/decoy rules), no scoring
  regress         re-run the 30 smoke queries and compare field-by-field to the smoke oracle
  full            the 407-query run (resumable). Requires --authorized AND a passing regression
                  report for this exact script sha256.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from numba import njit, prange
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp008_c3_smoke as X  # noqa: E402  (scientific reference, unchanged)
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs, score_best_matches  # noqa: E402

ROOT = X.ROOT
OUT = ROOT / "results" / "exp008_c3_full"
CACHE = OUT / "cache"
STORE = CACHE / "rep_store.npz"
FULL_MANIFEST = OUT / "manifest.json"
REGRESSION = OUT / "regression_report.json"
FULL_JSONL = OUT / "query_results.jsonl"
FULL_SUMMARY = OUT / "summary.json"
REFERENCE_SCRIPT = Path(X.__file__)
OPT_VERSION = "exp008-fast-v1"
TOL, MZP, IP = 0.1, 0.0, 1.0  # locked scorer params (identical to reference)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ numba kernel
@njit
def _pairs(spec1, spec2, tol, shift, mz_power, intensity_power):
    """matchms collect_peak_pairs with its None result mapped to an empty (0, 3) array (as matchms.pair does)."""
    r = collect_peak_pairs(spec1, spec2, tol, shift, mz_power, intensity_power)
    if r is None:
        return np.zeros((0, 3))
    return r.copy()


@njit(parallel=True)
def batch_modcos(q_mz, q_int, q_pm, mz_flat, int_flat, offsets, pms, idx, tol, mz_power, intensity_power):
    """ModifiedCosineGreedy(query, evidence[k]) for every k in idx, mirroring matchms 0.33.1 .pair():
    reference = query (spec1), query = evidence (spec2), mass_shift = pm_query - pm_evidence.
    |shift| <= tol falls back to CosineGreedy.pair (zero-shift pairs only), exactly as matchms does."""
    out = np.zeros(idx.shape[0])
    spec1 = np.vstack((q_mz, q_int)).T
    for j in prange(idx.shape[0]):
        k = idx[j]
        a = offsets[k]
        b = offsets[k + 1]
        spec2 = np.vstack((mz_flat[a:b].copy(), int_flat[a:b].copy())).T
        shift = q_pm - pms[k]
        zp = _pairs(spec1, spec2, tol, 0.0, mz_power, intensity_power)
        if abs(shift) <= tol:
            pairs = zp
        else:
            nz = _pairs(spec1, spec2, tol, shift, mz_power, intensity_power)
            pairs = np.concatenate((zp, nz))
        if pairs.shape[0] == 0:
            out[j] = 0.0
            continue
        order = np.argsort(pairs[:, 2], kind="mergesort")[::-1]
        sorted_pairs = pairs[order, :]
        score, _ = score_best_matches(sorted_pairs, spec1, spec2, mz_power, intensity_power)
        out[j] = score
    return out


# ------------------------------------------------------------------ store
def sorted_peaks(mzs, ints):
    """Exactly spectra.spectrum_io.make_spectrum's ordering (np.argsort default kind)."""
    mzs = np.asarray(mzs, dtype=float)
    ints = np.asarray(ints, dtype=float)
    order = np.argsort(mzs)
    return mzs[order], ints[order]


def cmd_build_store(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = X.connect()
    meta = X.load_meta(con)
    reps = X.representative_table(meta)
    reps = reps[reps["adduct"].isin(X.MONOMER_DELTA)].reset_index(drop=True)
    print(f"{len(reps)} monomer-adduct representatives; fetching peaks")
    mz_parts, int_parts, offsets, pms = [], [], [0], []
    for c0 in range(0, len(reps), 50_000):  # bounded memory
        chunk = reps.iloc[c0:c0 + 50_000]
        pk = X.fetch_peaks(con, chunk["rid"].values)
        for rid in chunk["rid"].values:
            row = pk.loc[rid]
            m, i = sorted_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
            mz_parts.append(m); int_parts.append(i)
            offsets.append(offsets[-1] + len(m)); pms.append(float(row["pm"]))
        del pk
        print(f"  {min(c0 + 50_000, len(reps))}/{len(reps)} ({time.time() - t0:.0f}s)")
    np.savez(STORE, rid=reps["rid"].values.astype(np.int64), mz=np.concatenate(mz_parts),
             inten=np.concatenate(int_parts), offsets=np.asarray(offsets, dtype=np.int64),
             pm=np.asarray(pms, dtype=np.float64))
    print(f"store written: {STORE} ({STORE.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


class Store:
    def __init__(self, reps):
        z = np.load(STORE)
        self.rid = z["rid"]; self.mz = z["mz"]; self.inten = z["inten"]
        self.offsets = z["offsets"]; self.pm = z["pm"]
        self.pos = {int(r): k for k, r in enumerate(self.rid)}
        r = reps.set_index("rid").loc[self.rid]
        self.ik = r["ik"].values
        self.adduct = r["adduct"].values
        assert np.array_equal(r["pm"].values, self.pm), "store precursor mismatch vs representative table"

    def peaks(self, rid):
        k = self.pos[int(rid)]
        a, b = self.offsets[k], self.offsets[k + 1]
        return self.mz[a:b], self.inten[a:b], float(self.pm[k])


def wide_hits_fast(store, q_mz, q_int, q_pm, adduct, excluded):
    """Same selection and ordering as X.wide_hits; scoring via batch_modcos."""
    lo, hi = q_pm - X.WIDE_WINDOW_DA, q_pm + X.WIDE_WINDOW_DA
    sel = np.nonzero((store.adduct == adduct) & (store.pm >= lo) & (store.pm <= hi)
                     & ~np.isin(store.ik, list(excluded)))[0].astype(np.int64)
    scores = batch_modcos(q_mz, q_int, float(q_pm), store.mz, store.inten, store.offsets, store.pm, sel,
                          TOL, MZP, IP)
    scored = sorted(zip(scores.tolist(), store.ik[sel].tolist()), key=lambda x: (-x[0], x[1]))
    return [(ik, s) for s, ik in scored[:X.TOP_K_HITS]], int(len(sel))


# ------------------------------------------------------------------ per-query (mirrors X.cmd_smoke)
def run_query(m, ctx):
    from matchms.similarity import ModifiedCosineGreedy
    uni, meta, store, bpeaks = ctx["uni"], ctx["meta"], ctx["store"], ctx["bpeaks"]
    sim = ctx.setdefault("sim", ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0))
    tq = time.time()
    t, d, a = m["ik"], m["decoy_ik"], m["adduct"]
    grp_t, grp_d = uni.group(t), uni.group(d)
    pool = uni.pool(m["neutral_mass"], X.PPM_PRIMARY)
    pool10 = uni.pool(m["neutral_mass"], X.PPM_SENSITIVITY)
    qrow = bpeaks.loc[m["rid"]]
    q_mz, q_int = sorted_peaks(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"])
    qspec = X.make_spec(qrow)
    in_pool = t in pool

    rr_a = X.chance_rr(len(pool)) if in_pool else 0.0

    ev_b = meta[meta["ik"].isin(pool) & ~meta["ik"].isin(grp_t) & (meta["adduct"] == a)]
    direct = {c: -np.inf for c in pool}
    for rid, ik in zip(ev_b["rid"].values, ev_b["ik"].values):
        s = float(sim.pair(qspec, X.make_spec(bpeaks.loc[rid]))["score"])
        direct[ik] = max(direct[ik], s)
    n_direct_target = int((ev_b["ik"] == t).sum())
    rr_b, span_b = X.tie_aware_rr(direct, t)

    hits, n_ev = wide_hits_fast(store, q_mz, q_int, m["pm"], a, grp_t)
    sc_c = X.r1_scores(uni, pool, hits)
    rr_c, span_c = X.tie_aware_rr(sc_c, t)
    top1_c = max(sc_c, key=lambda c: (sc_c[c], c)) if sc_c else None

    d_mz, d_int, d_pm = store.peaks(m["decoy_rid"])
    hits_d, n_ev_d = wide_hits_fast(store, d_mz, d_int, d_pm, a, grp_t | grp_d)
    sc_d = X.r1_scores(uni, pool, hits_d)
    rr_decoy_t, _ = X.tie_aware_rr(sc_d, t)
    rr_decoy_d, _ = X.tie_aware_rr(sc_d, d)

    return {
        **{k: m[k] for k in ("rid", "ik", "adduct", "tier", "nn_tc", "decoy_ik", "decoy_rid")},
        "pool_size_5ppm": len(pool), "pool_size_10ppm": len(pool10),
        "target_in_pool_5ppm": in_pool, "target_in_pool_10ppm": t in pool10,
        "check_target_evidence_rows": int(meta.loc[~meta["ik"].isin(grp_t), "ik"].eq(t).sum()),
        "check_hits_in_target_group": sum(h in grp_t for h, _ in hits),
        "check_decoy_hits_in_t_or_d_group": sum(h in (grp_t | grp_d) for h, _ in hits_d),
        "A_chance_rr": rr_a,
        "B_direct_rr": rr_b, "B_rank_span": span_b, "B_n_direct_spectra_target": n_direct_target,
        "B_n_pool_with_direct_evidence": int(sum(np.isfinite(v) for v in direct.values())),
        "C_r1_rr": rr_c, "C_rank_span": span_c, "C_top1": top1_c,
        "C_n_evidence_scored": n_ev,
        "C_hits": [{"ik": h, "s": s, "tc_to_truth": uni.tc(t, h)} for h, s in hits],
        "C_top25": sorted(sc_c.items(), key=lambda kv: (-kv[1], kv[0]))[:25],
        "C_failure": X.failure_category(uni, t, pool, span_c, hits, top1_c) if in_pool else "F0_not_in_pool",
        "decoy_rr_target": rr_decoy_t, "decoy_rr_decoy": rr_decoy_d, "decoy_n_evidence_scored": n_ev_d,
        "tc_target_decoy": uni.tc(t, d),
        "tc_top1_truth": uni.tc(t, top1_c) if top1_c else None,
        "tc_pool_mean_truth": float(np.mean([uni.tc(t, c) for c in pool if c != t])) if len(pool) > 1 else None,
        "sec": time.time() - tq,
    }


def load_context(queries):
    con = X.connect()
    meta = X.load_meta(con)
    uni = X.Universe(X.build_universe(con))
    reps = X.representative_table(meta)
    store = Store(reps)
    rep_idx = reps.set_index(["ik", "adduct"])["rid"]
    need = set()
    for m in queries:
        m["decoy_rid"] = int(rep_idx.loc[(m["decoy_ik"], m["adduct"])])
        need.add(m["rid"])
        pool = uni.pool(m["neutral_mass"], X.PPM_PRIMARY)
        grp = uni.group(m["ik"])
        b = meta[meta["ik"].isin(pool) & ~meta["ik"].isin(grp) & (meta["adduct"] == m["adduct"])]
        need |= set(b["rid"].values.tolist())
    bpeaks = X.fetch_peaks(con, need)
    return {"uni": uni, "meta": meta, "store": store, "bpeaks": bpeaks}


def json_roundtrip(r):
    return json.loads(json.dumps(r, default=str))


# ------------------------------------------------------------------ regression
def compare(a, b, path, diffs, fdiff):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append((path, "keys", sorted(set(a) ^ set(b))))
        for k in set(a) & set(b):
            compare(a[k], b[k], f"{path}.{k}", diffs, fdiff)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, "len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", diffs, fdiff)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            diffs.append((path, a, b))
            fdiff.append(abs(a - b))
    elif a != b:
        diffs.append((path, a, b))


def cmd_regress(args):
    t0 = time.time()
    man = json.load(open(X.MANIFEST))
    oracle = {r["rid"]: r for r in map(json.loads, open(X.SMOKE_JSONL))}
    queries = [dict(q) for q in man["queries"]]
    ctx = load_context(queries)
    t_setup = time.time() - t0
    # JIT warm-up on one pair (excluded from per-query timing, reported separately)
    tw = time.time()
    q = queries[0]
    mz, it, pm = ctx["store"].peaks(q["decoy_rid"])
    batch_modcos(mz, it, pm, ctx["store"].mz, ctx["store"].inten, ctx["store"].offsets, ctx["store"].pm,
                 np.arange(2, dtype=np.int64), TOL, MZP, IP)
    t_jit = time.time() - tw
    per, all_diffs, fdiff, secs = [], [], [], []
    for m in queries:
        r = json_roundtrip(run_query(m, ctx))
        secs.append(r["sec"])
        o = oracle[r["rid"]]
        d = []
        compare({k: v for k, v in r.items() if k != "sec"}, {k: v for k, v in o.items() if k != "sec"},
                f"rid{r['rid']}", d, fdiff)
        all_diffs += d
        per.append({"rid": r["rid"], "sec_fast": r["sec"], "sec_reference": o["sec"], "n_diffs": len(d)})
        print(f"  rid {r['rid']}: {r['sec']:.1f}s (reference {o['sec']:.0f}s) diffs={len(d)}")
    rep = {
        "script": "research/scripts/exp008_c3_fast.py", "script_sha256": sha(__file__),
        "reference_script_sha256": sha(REFERENCE_SCRIPT), "optimization_version": OPT_VERSION,
        "numba_threads": int(os.environ["NUMBA_NUM_THREADS"]),
        "n_queries": len(per), "fields_compared": "all record fields except 'sec'",
        "n_field_differences": len(all_diffs), "max_abs_float_difference": max(fdiff) if fdiff else 0.0,
        "differences": [list(map(str, x)) for x in all_diffs[:200]],
        "PASS": len(all_diffs) == 0,
        "runtime": {"setup_s": t_setup, "jit_warmup_s": t_jit,
                    "median_sec_per_query_fast": float(np.median(secs)),
                    "max_sec_per_query_fast": float(np.max(secs)),
                    "median_sec_per_query_reference": float(np.median([p["sec_reference"] for p in per])),
                    "total_scoring_s_fast": float(np.sum(secs))},
        "per_query": per,
    }
    json.dump(rep, open(REGRESSION, "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_query", "differences")}, indent=2))


# ------------------------------------------------------------------ full-run population (no scoring)
def cmd_prepare_full(args):
    """Same eligibility (E1-E4), tiers and decoy rule as X.cmd_preflight, for ALL eligible queries."""
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    con = X.connect()
    meta = X.load_meta(con)
    uni = X.Universe(X.build_universe(con))
    reps = X.representative_table(meta)
    q = meta.set_index("rid").loc[X.load_population()].reset_index()
    rep_by_adduct = {a: set(g["ik"]) for a, g in reps.groupby("adduct")}
    elig = []
    for _, r in q.iterrows():
        if r["adduct"] not in X.MONOMER_DELTA or r["ik"] not in uni.row_of:
            continue
        M = X.neutral_mass(r["pm"], r["adduct"])
        pool = uni.pool(M, X.PPM_PRIMARY)
        if len(pool) < 2:
            continue
        grp = uni.group(r["ik"])
        decoys = sorted(c for c in pool if c not in grp and uni.parent_of[c] != uni.parent_of[r["ik"]]
                        and c in rep_by_adduct.get(r["adduct"], ()))
        if not decoys:
            continue
        elig.append({"rid": int(r["rid"]), "ik": r["ik"], "adduct": r["adduct"], "pm": float(r["pm"]),
                     "neutral_mass": M, "pool_size_5ppm": len(pool), "decoy_candidates": decoys})
    fps_by_adduct = {}
    for e in elig:
        a = e["adduct"]
        if a not in fps_by_adduct:
            iks = sorted(ik for ik in rep_by_adduct[a] if ik in uni.row_of)
            fps_by_adduct[a] = (iks, [uni.fp(ik) for ik in iks])
        iks, fps = fps_by_adduct[a]
        grp = uni.group(e["ik"])
        sims = np.array(DataStructs.BulkTanimotoSimilarity(uni.fp(e["ik"]), fps))
        mask = np.array([ik not in grp for ik in iks])
        e["nn_tc"] = float(sims[mask].max())
        e["tier"] = next(name for name, lo, hi in X.TIERS if lo <= e["nn_tc"] < hi)
    queries = []
    for e in sorted(elig, key=lambda e: e["rid"]):
        d = random.Random(X.SEED_DECOY * 10_000_000 + e["rid"]).choice(e["decoy_candidates"])
        queries.append({k: e[k] for k in ("rid", "ik", "adduct", "pm", "neutral_mass", "pool_size_5ppm",
                                          "nn_tc", "tier")} | {"decoy_ik": d,
                                                               "n_decoy_candidates": len(e["decoy_candidates"])})
    # Consistency with the preflight: counts and the 30 smoke rows must be reproduced exactly.
    pre = json.load(open(X.PREFLIGHT_OUT))
    smoke = {m["rid"]: m for m in json.load(open(X.MANIFEST))["queries"]}
    tiers = {n: sum(qq["tier"] == n for qq in queries) for n, _, _ in X.TIERS}
    by = {qq["rid"]: qq for qq in queries}
    smoke_mismatch = [rid for rid, m in smoke.items()
                      if rid not in by or any(by[rid][k] != m[k] for k in m)]
    rng = random.Random(X.SEED_POP)
    resample = []
    for name, _, _ in X.TIERS:
        resample += sorted(rng.sample(sorted(qq["rid"] for qq in queries if qq["tier"] == name), X.PER_TIER))
    consistency = {
        "eligible_count": len(queries), "preflight_eligible_count": pre["counts"]["eligible"],
        "tier_counts": tiers, "preflight_tier_counts": pre["counts"]["eligible_by_tier"],
        "smoke_rows_reproduced_exactly": not smoke_mismatch, "smoke_mismatch_rids": smoke_mismatch,
        "smoke_sample_redrawn_identically": resample == [m["rid"] for m in json.load(open(X.MANIFEST))["queries"]],
    }
    assert len(queries) == pre["counts"]["eligible"] and tiers == pre["counts"]["eligible_by_tier"], consistency
    assert not smoke_mismatch and consistency["smoke_sample_redrawn_identically"], consistency
    json.dump({"queries": queries, "consistency": consistency, "runtime_s": time.time() - t0},
              open(CACHE / "full_population.json", "w"), indent=2)
    print(json.dumps(consistency, indent=2))


def write_manifest():
    import matchms, rdkit, duckdb, numba
    pop = json.load(open(CACHE / "full_population.json"))
    reg = json.load(open(REGRESSION))
    man = {
        "experiment_id": "EXP-008 full run (pseudo-C3)",
        "status": "PREPARED - NOT EXECUTED (requires separate authorization)",
        "script": "research/scripts/exp008_c3_fast.py", "script_sha256": sha(__file__),
        "reference_script": "research/scripts/exp008_c3_smoke.py", "reference_script_sha256": sha(REFERENCE_SCRIPT),
        "optimization_version": OPT_VERSION,
        "population": {"source": "EXP-007 manifest all_600_sorted -> eligibility E1-E4 (unchanged)",
                       "count": len(pop["queries"]), "consistency": pop["consistency"],
                       "query_ids": [q["rid"] for q in pop["queries"]]},
        "queries": pop["queries"],
        "seeds": {"population": X.SEED_POP, "decoy": X.SEED_DECOY, "bootstrap": X.SEED_BOOT},
        "adduct_table": {"source": "research/03_exp002_design.md §5 (monomers, sign-fixed)", "delta": X.MONOMER_DELTA},
        "pool_rule": f"|ExactMolWt(c) - (precursor_mz - delta)| <= {X.PPM_PRIMARY} ppm of neutral mass (10 ppm sensitivity)",
        "wide_window_rule": f"same adduct, {X.EVIDENCE_LIB} representatives, |pm - pm_query| <= {X.WIDE_WINDOW_DA} Da, target parent group excluded",
        "H": X.TOP_K_HITS,
        "fingerprint": "RDKit Morgan radius=2, 2048 bits, Tanimoto",
        "scorer": {"name": "matchms ModifiedCosineGreedy", "tolerance": TOL, "mz_power": MZP, "intensity_power": IP},
        "candidate_score": "S(c) = max_{h in H, parent(h) != parent(c)} s_h * Tc(c,h); tie-aware expected RR",
        "software": {"python": sys.version.split()[0], "matchms": matchms.__version__, "rdkit": rdkit.__version__,
                     "duckdb": duckdb.__version__, "numba": numba.__version__, "numpy": np.__version__},
        "regression": {k: reg[k] for k in ("PASS", "n_queries", "n_field_differences", "max_abs_float_difference",
                                           "script_sha256")},
        "runtime_benchmark": reg["runtime"],
        "outputs": {"per_query": "results/exp008_c3_full/query_results.jsonl (append, resumable)",
                    "summary": "results/exp008_c3_full/summary.json",
                    "report": "research/analysis/exp008_c3_full_run.md"},
        "command": "python research/scripts/exp008_c3_fast.py full --authorized",
    }
    json.dump(man, open(FULL_MANIFEST, "w"), indent=2)
    print(f"wrote {FULL_MANIFEST}")


def cmd_manifest(args):
    write_manifest()


# ------------------------------------------------------------------ full run (gated, resumable)
def cmd_full(args):
    if not args.authorized:
        sys.exit("full run requires explicit authorization (--authorized). Not executed.")
    man = json.load(open(FULL_MANIFEST))
    reg = json.load(open(REGRESSION))
    me = sha(__file__)
    assert reg["PASS"] and reg["script_sha256"] == me == man["script_sha256"], "regression not passed for this script"
    assert sha(REFERENCE_SCRIPT) == man["reference_script_sha256"], "reference script changed"
    done = set()
    if FULL_JSONL.exists():
        done = {json.loads(l)["rid"] for l in open(FULL_JSONL) if l.strip()}
    todo = [dict(q) for q in man["queries"] if q["rid"] not in done]
    print(f"{len(done)} already complete, {len(todo)} to run")
    if todo:
        ctx = load_context(todo)
        with open(FULL_JSONL, "a") as f:
            for m in todo:
                r = run_query(m, ctx)
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
                print(f"  rid {r['rid']} {r['tier']}: C={r['C_r1_rr']:.3f} A={r['A_chance_rr']:.3f} "
                      f"decoy={r['decoy_rr_target']:.3f} ({r['sec']:.0f}s)", flush=True)
    rows = {json.loads(l)["rid"]: json.loads(l) for l in open(FULL_JSONL) if l.strip()}
    df = pd.DataFrame([rows[q["rid"]] for q in man["queries"]])  # deterministic manifest order
    checks = {
        "L1_target_evidence_rows": int(df["check_target_evidence_rows"].sum()),
        "L1_hits_in_target_group": int(df["check_hits_in_target_group"].sum()),
        "L7_decoy_hits_in_excluded_groups": int(df["check_decoy_hits_in_t_or_d_group"].sum()),
        "L6_canary_direct_spectra_for_target": int(df["B_n_direct_spectra_target"].sum()),
    }

    def summary(sub):
        span = sub["C_rank_span"].map(lambda s: s if isinstance(s, list) else None)
        rk = span.map(lambda s: np.mean(s) if s else np.inf)
        return {
            "n": len(sub),
            "pool_recall_5ppm": float(sub["target_in_pool_5ppm"].mean()),
            "pool_size_5ppm_median": float(sub["pool_size_5ppm"].median()),
            "A_chance_mrr": float(sub["A_chance_rr"].mean()),
            "B_direct_mrr": float(sub["B_direct_rr"].mean()),
            "C_r1_mrr": float(sub["C_r1_rr"].mean()),
            **{f"C_R@{k}": float((rk <= k).mean()) for k in (1, 5, 10, 25)},
            "decoy_mrr_target": float(sub["decoy_rr_target"].mean()),
            "lift_C_minus_A": X.bootstrap_ci(sub["C_r1_rr"] - sub["A_chance_rr"], X.SEED_BOOT),
            "lift_decoy_minus_A": X.bootstrap_ci(sub["decoy_rr_target"] - sub["A_chance_rr"], X.SEED_BOOT),
            "lift_C_minus_decoy": X.bootstrap_ci(sub["C_r1_rr"] - sub["decoy_rr_target"], X.SEED_BOOT),
            "failure_categories": sub["C_failure"].value_counts().to_dict(),
        }
    out = {"n_complete": len(df), "n_manifest": len(man["queries"]), "checks": checks,
           "checks_pass": all(v == 0 for v in checks.values()),
           "overall": summary(df), "by_tier": {t: summary(g) for t, g in df.groupby("tier")},
           "median_sec_per_query": float(df["sec"].median())}
    json.dump(out, open(FULL_SUMMARY, "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build-store", "prepare-full", "regress", "manifest", "full"])
    p.add_argument("--authorized", action="store_true")
    args = p.parse_args()
    {"build-store": cmd_build_store, "prepare-full": cmd_prepare_full, "regress": cmd_regress,
     "manifest": cmd_manifest, "full": cmd_full}[args.stage](args)


if __name__ == "__main__":
    main()

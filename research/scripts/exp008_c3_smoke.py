"""EXP-008 pseudo-Class-3: preflight, pool census (8a) and 30-query ranking smoke (8b).

Design: research/analysis/exp008_c3_design.md. Implementation spec: research/analysis/exp008_c3_smoke.md.

Stages
  preflight  Construction checks + eligibility + structural tiers + seeded sampling + manifest.
             No spectral scoring, no ranking, no truth-in-pool measurement.
  census     EXP-008a (pool recall / pool size on all eligible queries). Requires --authorized.
  smoke      EXP-008b (Baselines A/B/C + decoy control on the 30 manifest queries). Requires --authorized.

Two sources, never mixed:
  STRUCTURE UNIVERSE  every train inchikey14 (one SMILES each), built once, target included.
  SPECTRAL EVIDENCE   train spectra minus every spectrum of the target's parent group.

Usage
  python research/scripts/exp008_c3_smoke.py preflight
  python research/scripts/exp008_c3_smoke.py census --authorized
  python research/scripts/exp008_c3_smoke.py smoke  --authorized
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RDLogger.DisableLog("rdApp.*")

TRAIN = (ROOT / "train.parquet").as_posix()
RES = ROOT / "results"
UNIVERSE_CACHE = RES / "exp008_structure_universe.parquet"
PREFLIGHT_OUT = RES / "exp008_c3_preflight.json"
MANIFEST = RES / "exp008_c3_smoke_manifest.json"
CENSUS_OUT = RES / "exp008_c3_census.json"
SMOKE_JSONL = RES / "exp008_c3_smoke_query_results.jsonl"
SMOKE_OUT = RES / "exp008_c3_smoke.json"

# ---- Locked parameters (design §5; L8: never tuned on outcomes) ----
SEED_POP = 20260924
SEED_DECOY = 20260925
SEED_BOOT = 20260926
PPM_PRIMARY = 5.0
PPM_SENSITIVITY = 10.0
WIDE_WINDOW_DA = 150.0
TOP_K_HITS = 20
TIERS = (("T-low", 0.0, 0.60), ("T-mid", 0.60, 0.70), ("T-high", 0.70, 1.0 + 1e-9))
PER_TIER = 10
N_BOOT = 1000
F1_SIM_FLOOR = 0.30
F2_TC_FLOOR = 0.40
EVIDENCE_LIB = "enveda-180"  # R1 wide-search library (same instrument as the queries)

# EXP-002 §5 table, monomers only (sign-fixed). neutral_mass = precursor_mz - delta.
MONOMER_DELTA = {
    "[M+H]+": 1.007276,
    "[M-H]-": -1.007276,
    "[M+Na]+": 22.989221,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+Cl]-": 34.969402,
    "[M+CH2O2-H]-": 44.998203,
    "[M+C2H4O2-H]-": 59.013853,
}
RID_SANITY = [(773739, "RCWXXMNGYWDMMO", "[M+H]+"), (733288, "QHALUOAFNBWZED", "[2M+Na]+")]

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ------------------------------------------------------------------ data access
def connect():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")  # deterministic row order (EXP-007 §13)
    return con


def load_meta(con):
    """Spectrum-level metadata for all train rows; rid = 0-based file row (== EXP-007 rid)."""
    df = con.execute(
        f"""SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm, num_peaks,
                   ingest_lib, collision_energy_orig AS ce
            FROM read_parquet('{TRAIN}', file_row_number=true)"""
    ).fetchdf()
    for rid, ik, ad in RID_SANITY:
        row = df.iloc[rid]
        assert int(row["rid"]) == rid and row["ik"] == ik and row["adduct"] == ad, f"rid check failed at {rid}"
    return df


def fetch_peaks(con, rids):
    con.register("want", pd.DataFrame({"rid": sorted(set(int(r) for r in rids))}))
    df = con.execute(
        f"""SELECT file_row_number AS rid, precursor_mz AS pm, ms2_mzs, ms2_normalized_intensities
            FROM read_parquet('{TRAIN}', file_row_number=true)
            WHERE file_row_number IN (SELECT rid FROM want)"""
    ).fetchdf()
    con.unregister("want")
    return df.set_index("rid")


# ------------------------------------------------------------------ structure universe
def parent_key(mol):
    """Connectivity key after largest-fragment + neutralisation (design L2: salt/charge forms)."""
    try:
        m = rdMolStandardize.LargestFragmentChooser().choose(mol)
        m = rdMolStandardize.Uncharger().uncharge(m)
        return Chem.MolToInchiKey(m)[:14] or None
    except Exception:
        return None


def build_universe(con):
    if UNIVERSE_CACHE.exists():
        return pd.read_parquet(UNIVERSE_CACHE)
    t0 = time.time()
    mol = con.execute(
        f"""SELECT inchikey14 AS ik, min(normalized_smiles) AS smiles, min(molecular_formula) AS formula
            FROM read_parquet('{TRAIN}') GROUP BY 1 ORDER BY 1"""
    ).fetchdf()
    mass, pkey, charge, ok = [], [], [], []
    for smi in mol["smiles"]:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        ok.append(m is not None)
        mass.append(Descriptors.ExactMolWt(m) if m is not None else np.nan)
        charge.append(Chem.GetFormalCharge(m) if m is not None else 0)
        pkey.append(parent_key(m) if m is not None else None)
    mol["mass"], mol["parent"], mol["charge"], mol["parse_ok"] = mass, pkey, charge, ok
    mol["parent"] = mol["parent"].fillna(mol["ik"])  # unparseable -> own key
    mol.to_parquet(UNIVERSE_CACHE, index=False)
    print(f"structure universe: {len(mol)} molecules in {time.time() - t0:.0f}s")
    return mol


class Universe:
    def __init__(self, df):
        self.df = df[df["parse_ok"]].sort_values("mass").reset_index(drop=True)
        self.mass = self.df["mass"].values
        self.ik = self.df["ik"].values
        self.row_of = {k: i for i, k in enumerate(self.ik)}
        self.parent_of = dict(zip(df["ik"], df["parent"]))
        self.formula_of = dict(zip(df["ik"], df["formula"]))
        self.smiles_of = dict(zip(df["ik"], df["smiles"]))
        self.members_of_parent = df.groupby("parent")["ik"].apply(set).to_dict()
        self._fp = {}

    def pool(self, neutral_mass, ppm):
        """Candidate STRUCTURES with |mass - M| <= ppm * 1e-6 * M (Da)."""
        tol = neutral_mass * ppm * 1e-6
        i0 = np.searchsorted(self.mass, neutral_mass - tol, side="left")
        i1 = np.searchsorted(self.mass, neutral_mass + tol, side="right")
        return list(self.ik[i0:i1])

    def group(self, ik):
        """Target's exclusion group: itself + every molecule with the same parent key."""
        return {ik} | self.members_of_parent.get(self.parent_of.get(ik, ik), set())

    def fp(self, ik):
        if ik not in self._fp:
            self._fp[ik] = MORGAN.GetFingerprint(Chem.MolFromSmiles(self.smiles_of[ik]))
        return self._fp[ik]

    def tc(self, a, b):
        return DataStructs.TanimotoSimilarity(self.fp(a), self.fp(b))


def neutral_mass(precursor_mz, adduct):
    return precursor_mz - MONOMER_DELTA[adduct]


# ------------------------------------------------------------------ evidence library
def representative_table(meta):
    """One label-free representative spectrum per (molecule, adduct) in EVIDENCE_LIB:
    merged-CE spectrum (ce contains ',') first, then max num_peaks, then lowest rid."""
    e = meta[meta["ingest_lib"] == EVIDENCE_LIB].copy()
    e["merged"] = e["ce"].astype(str).str.contains(",")
    e = e.sort_values(["ik", "adduct", "merged", "num_peaks", "rid"], ascending=[True, True, False, False, True])
    return e.drop_duplicates(["ik", "adduct"])[["rid", "ik", "adduct", "pm"]].reset_index(drop=True)


# ------------------------------------------------------------------ population
def load_population():
    man = json.load(open(RES / "exp007_c2_full_run_manifest.json"))
    rids = man["query_ids"]["all_600_sorted"]
    assert len(rids) == 600
    return rids


def tie_aware_rr(scores: dict, target, cutoff=25):
    """Expected reciprocal rank of target under uniform random tie-breaking; 0 if absent."""
    if target not in scores:
        return 0.0, None
    s = scores[target]
    vals = np.array(list(scores.values()), dtype=float)
    better = int(np.sum(vals > s))
    tied = int(np.sum(vals == s))
    ranks = np.arange(better + 1, better + tied + 1)
    return float(np.mean(np.where(ranks <= cutoff, 1.0 / ranks, 0.0))), (better + 1, better + tied)


def chance_rr(n, cutoff=25):
    return float(sum(1.0 / r for r in range(1, min(cutoff, n) + 1)) / n) if n else 0.0


# ------------------------------------------------------------------ preflight
def cmd_preflight(args):
    t0 = time.time()
    con = connect()
    meta = load_meta(con)
    uni = Universe(build_universe(con))
    reps = representative_table(meta)
    rids = load_population()
    q = meta.set_index("rid").loc[rids].reset_index()
    report = {"stage": "preflight", "checks": {}, "counts": {}}

    # C. neutral-mass unit checks (round trip on the table; units Da; ppm relative to M)
    rt = {a: abs(neutral_mass(300.0 + d, a) - 300.0) for a, d in MONOMER_DELTA.items()}
    assert max(rt.values()) < 1e-9
    r0 = meta.iloc[RID_SANITY[0][0]]
    m_truth = uni.df.loc[uni.row_of[r0["ik"]], "mass"]
    m_query = neutral_mass(r0["pm"], r0["adduct"])
    report["checks"]["C_neutral_mass"] = {
        "round_trip_max_abs_err_da": max(rt.values()),
        "sanity_rid": int(r0["rid"]), "adduct": r0["adduct"], "precursor_mz": float(r0["pm"]),
        "neutral_mass_query_da": float(m_query), "exact_mass_truth_da": float(m_truth),
        "ppm_error": float((m_query - m_truth) / m_truth * 1e6),
    }
    # D. the target can appear in its own pool (sanity rid only; population-wide recall is EXP-008a)
    report["checks"]["D_sanity_target_in_pool_5ppm"] = r0["ik"] in uni.pool(m_query, PPM_PRIMARY)

    # Eligibility (no scoring, no truth-in-pool use)
    rep_by_adduct = {a: set(g["ik"]) for a, g in reps.groupby("adduct")}
    elig, reasons = [], {"E1_not_monomer": 0, "E2_smiles_unparsed": 0, "E3_pool_lt_2": 0, "E4_no_decoy": 0}
    for _, r in q.iterrows():
        if r["adduct"] not in MONOMER_DELTA:
            reasons["E1_not_monomer"] += 1; continue
        if r["ik"] not in uni.row_of:
            reasons["E2_smiles_unparsed"] += 1; continue
        M = neutral_mass(r["pm"], r["adduct"])
        pool = uni.pool(M, PPM_PRIMARY)
        if len(pool) < 2:
            reasons["E3_pool_lt_2"] += 1; continue
        grp = uni.group(r["ik"])
        decoys = sorted(c for c in pool if c not in grp and uni.parent_of[c] != uni.parent_of[r["ik"]]
                        and c in rep_by_adduct.get(r["adduct"], ()))
        if not decoys:
            reasons["E4_no_decoy"] += 1; continue
        elig.append({"rid": int(r["rid"]), "ik": r["ik"], "adduct": r["adduct"], "pm": float(r["pm"]),
                     "neutral_mass": M, "pool_size_5ppm": len(pool), "decoy_candidates": decoys})
    report["counts"]["population"] = len(q)
    report["counts"]["excluded"] = reasons
    report["counts"]["eligible"] = len(elig)

    # Structural tiers: NN-Tc of target to molecules with a same-adduct EVIDENCE_LIB representative,
    # excluding the target's parent group. Truth structure used for stratification only (L5).
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
        e["tier"] = next(name for name, lo, hi in TIERS if lo <= e["nn_tc"] < hi)
    tier_counts = {name: sum(e["tier"] == name for e in elig) for name, _, _ in TIERS}
    report["counts"]["eligible_by_tier"] = tier_counts
    short = {k: v for k, v in tier_counts.items() if v < PER_TIER}
    if short:
        report["STOP"] = f"insufficient eligible queries in tier(s) {short}; need {PER_TIER} each"
        json.dump(report, open(PREFLIGHT_OUT, "w"), indent=2)
        print(json.dumps(report, indent=2))
        sys.exit(2)

    # Seeded sampling, fixed tier order, no replacement / substitution
    rng = random.Random(SEED_POP)
    sample = []
    for name, _, _ in TIERS:
        pool_rids = sorted(e["rid"] for e in elig if e["tier"] == name)
        sample += sorted(rng.sample(pool_rids, PER_TIER))
    by_rid = {e["rid"]: e for e in elig}
    manifest_q = []
    for rid in sample:
        e = by_rid[rid]
        d = random.Random(SEED_DECOY * 10_000_000 + rid).choice(e["decoy_candidates"])
        manifest_q.append({k: e[k] for k in ("rid", "ik", "adduct", "pm", "neutral_mass", "pool_size_5ppm",
                                             "nn_tc", "tier")} | {"decoy_ik": d,
                                                                   "n_decoy_candidates": len(e["decoy_candidates"])})

    # A/B. exclusion + universe checks on the sampled 30 (metadata only)
    ev_ik = meta["ik"].values
    a_viol, grp_sizes, b_viol = [], [], []
    for m in manifest_q:
        grp = uni.group(m["ik"]) | uni.group(m["decoy_ik"])  # decoy arm excludes both
        grp_sizes.append(len(uni.group(m["ik"])))
        remaining = meta[~np.isin(ev_ik, list(grp))]
        if np.isin(remaining["ik"].values, [m["ik"], m["decoy_ik"]]).any():
            a_viol.append(m["rid"])
        if m["ik"] not in uni.row_of:
            b_viol.append(m["rid"])
    report["checks"]["A_target_absent_from_evidence"] = {"violations": a_viol,
                                                         "parent_group_size_hist": pd.Series(grp_sizes).value_counts().to_dict()}
    report["checks"]["B_target_in_structure_universe"] = {"violations": b_viol}
    report["counts"]["universe_molecules"] = int(len(uni.df))
    report["counts"]["evidence_spectra_total"] = int(len(meta))
    report["counts"]["representatives_by_adduct"] = {a: len(v) for a, v in rep_by_adduct.items() if a in MONOMER_DELTA}

    manifest = {
        "experiment": "EXP-008b smoke", "status": "MANIFEST ONLY - not executed",
        "design": "research/analysis/exp008_c3_design.md", "impl": "research/analysis/exp008_c3_smoke.md",
        "params": {"seed_population": SEED_POP, "seed_decoy": SEED_DECOY, "seed_bootstrap": SEED_BOOT,
                   "ppm_primary": PPM_PRIMARY, "ppm_sensitivity": PPM_SENSITIVITY,
                   "wide_window_da": WIDE_WINDOW_DA, "top_k_hits": TOP_K_HITS,
                   "tiers": TIERS, "per_tier": PER_TIER, "evidence_lib_R1": EVIDENCE_LIB,
                   "adduct_delta": MONOMER_DELTA, "morgan": "r=2, 2048 bits"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "queries": manifest_q,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    report["runtime_s"] = time.time() - t0
    json.dump(report, open(PREFLIGHT_OUT, "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


# ------------------------------------------------------------------ scoring (gated)
def make_spec(row):
    from spectra.spectrum_io import make_spectrum
    return make_spectrum(row["ms2_mzs"], row["ms2_normalized_intensities"], row["pm"])


def r1_scores(uni, pool, hits):
    """Baseline C. S(c) = max_{h in H, parent(h) != parent(c)} s_h * Tc(c, h); 0 if no admissible h."""
    out = {}
    for c in pool:
        pc = uni.parent_of[c]
        vals = [s * uni.tc(c, h) for h, s in hits if uni.parent_of[h] != pc]
        out[c] = max(vals) if vals else 0.0
    return out


def wide_hits(sim, qspec, q_pm, adduct, reps, peaks, excluded):
    """Top-K molecules by ModifiedCosine over same-adduct EVIDENCE_LIB representatives within
    +/- WIDE_WINDOW_DA of the query precursor, excluding the given molecules."""
    r = reps[(reps["adduct"] == adduct) & (reps["pm"].between(q_pm - WIDE_WINDOW_DA, q_pm + WIDE_WINDOW_DA))
             & (~reps["ik"].isin(excluded))]
    scored = []
    for rid, ik in zip(r["rid"].values, r["ik"].values):
        scored.append((float(sim.pair(qspec, make_spec(peaks.loc[rid]))["score"]), ik))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(ik, s) for s, ik in scored[:TOP_K_HITS]], len(r)


def failure_category(uni, target, pool, rank_span, hits, top1):
    if target not in pool:
        return "F0_not_in_pool"
    if not hits or max(s for _, s in hits) < F1_SIM_FLOOR:
        return "F1_no_informative_neighbours"
    if max(uni.tc(target, h) for h, _ in hits) < F2_TC_FLOOR:
        return "F2_neighbours_unrelated"
    if rank_span[0] == 1:
        return "SUCCESS_rank1"
    if rank_span[0] <= 25:
        return "F4_top25_not_rank1"
    return "F3_isomer_displacer" if uni.formula_of.get(top1) == uni.formula_of.get(target) else "F3b_other_displacer"


def bootstrap_ci(x, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def cmd_smoke(args):
    from matchms.similarity import ModifiedCosineGreedy
    import matchms, rdkit
    man = json.load(open(MANIFEST))
    assert man["script_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), \
        "script changed since manifest was written; re-run preflight"
    con = connect()
    meta = load_meta(con)
    uni = Universe(build_universe(con))
    reps = representative_table(meta)
    sim = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    t0 = time.time()

    # Peaks needed: queries, decoy representatives, same-adduct reps in every wide window, pool spectra (B)
    need = set()
    rep_idx = reps.set_index(["ik", "adduct"])["rid"]
    for m in man["queries"]:
        need.add(m["rid"])
        d_rid = int(rep_idx.loc[(m["decoy_ik"], m["adduct"])])
        m["decoy_rid"] = d_rid
        need.add(d_rid)
        d_pm = float(reps.loc[reps["rid"] == d_rid, "pm"].iloc[0])
        for centre in (m["pm"], d_pm):
            w = reps[(reps["adduct"] == m["adduct"]) & reps["pm"].between(centre - WIDE_WINDOW_DA, centre + WIDE_WINDOW_DA)]
            need |= set(w["rid"].values.tolist())
        pool = uni.pool(m["neutral_mass"], PPM_PRIMARY)
        grp = uni.group(m["ik"])
        b = meta[meta["ik"].isin(pool) & ~meta["ik"].isin(grp) & (meta["adduct"] == m["adduct"])]
        need |= set(b["rid"].values.tolist())
    print(f"fetching peaks for {len(need)} spectra")
    peaks = fetch_peaks(con, need)

    rows = []
    for m in man["queries"]:
        tq = time.time()
        t, d, a = m["ik"], m["decoy_ik"], m["adduct"]
        grp_t, grp_d = uni.group(t), uni.group(d)
        pool = uni.pool(m["neutral_mass"], PPM_PRIMARY)
        pool10 = uni.pool(m["neutral_mass"], PPM_SENSITIVITY)
        qspec = make_spec(peaks.loc[m["rid"]])
        in_pool = t in pool

        # Baseline A: chance within the pool
        rr_a = chance_rr(len(pool)) if in_pool else 0.0

        # Baseline B: existing direct ranker; each pool molecule scored only by its OWN same-adduct spectra
        ev_b = meta[meta["ik"].isin(pool) & ~meta["ik"].isin(grp_t) & (meta["adduct"] == a)]
        direct = {c: -np.inf for c in pool}
        for rid, ik in zip(ev_b["rid"].values, ev_b["ik"].values):
            s = float(sim.pair(qspec, make_spec(peaks.loc[rid]))["score"])
            direct[ik] = max(direct[ik], s)
        n_direct_target = int((ev_b["ik"] == t).sum())  # canary L6: must be 0
        rr_b, span_b = tie_aware_rr(direct, t)

        # Baseline C: analogue propagation, target parent group excluded from evidence
        hits, n_ev = wide_hits(sim, qspec, m["pm"], a, reps, peaks, grp_t)
        sc_c = r1_scores(uni, pool, hits)
        rr_c, span_c = tie_aware_rr(sc_c, t)
        top1_c = max(sc_c, key=lambda c: (sc_c[c], c)) if sc_c else None

        # Decoy control: d's spectrum as query; evidence excludes t and d groups; SAME pool; rank of t
        d_row = peaks.loc[m["decoy_rid"]]
        dspec = make_spec(d_row)
        hits_d, n_ev_d = wide_hits(sim, dspec, float(d_row["pm"]), a, reps, peaks, grp_t | grp_d)
        sc_d = r1_scores(uni, pool, hits_d)
        rr_decoy_t, _ = tie_aware_rr(sc_d, t)
        rr_decoy_d, _ = tie_aware_rr(sc_d, d)

        rows.append({
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
            "C_failure": failure_category(uni, t, pool, span_c, hits, top1_c) if in_pool else "F0_not_in_pool",
            "decoy_rr_target": rr_decoy_t, "decoy_rr_decoy": rr_decoy_d, "decoy_n_evidence_scored": n_ev_d,
            "tc_target_decoy": uni.tc(t, d),
            "tc_top1_truth": uni.tc(t, top1_c) if top1_c else None,
            "tc_pool_mean_truth": float(np.mean([uni.tc(t, c) for c in pool if c != t])) if len(pool) > 1 else None,
            "sec": time.time() - tq,
        })
        print(f"  rid {m['rid']} {m['tier']}: A={rr_a:.3f} B={rr_b:.3f} C={rr_c:.3f} decoy={rr_decoy_t:.3f} "
              f"({time.time() - tq:.0f}s)")

    with open(SMOKE_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    df = pd.DataFrame(rows)
    checks = {
        "L1_target_evidence_rows": int(df["check_target_evidence_rows"].sum()),
        "L1_hits_in_target_group": int(df["check_hits_in_target_group"].sum()),
        "L7_decoy_hits_in_excluded_groups": int(df["check_decoy_hits_in_t_or_d_group"].sum()),
        "L6_canary_direct_spectra_for_target": int(df["B_n_direct_spectra_target"].sum()),
    }
    def summary(sub):
        return {
            "n": len(sub),
            "pool_recall_5ppm": float(sub["target_in_pool_5ppm"].mean()),
            "pool_recall_10ppm": float(sub["target_in_pool_10ppm"].mean()),
            "pool_size_5ppm_median": float(sub["pool_size_5ppm"].median()),
            "A_chance_mrr": float(sub["A_chance_rr"].mean()),
            "B_direct_mrr": float(sub["B_direct_rr"].mean()),
            "C_r1_mrr": float(sub["C_r1_rr"].mean()),
            "C_r1_R@1": float((sub["C_rank_span"].map(lambda s: s is not None and s[0] == 1)).mean()),
            "decoy_mrr_target": float(sub["decoy_rr_target"].mean()),
            "lift_C_minus_A": bootstrap_ci(sub["C_r1_rr"] - sub["A_chance_rr"], SEED_BOOT),
            "lift_decoy_minus_A": bootstrap_ci(sub["decoy_rr_target"] - sub["A_chance_rr"], SEED_BOOT),
            "failure_categories": sub["C_failure"].value_counts().to_dict(),
        }
    out = {
        "status": "SMOKE (n=30) - integrity only, not powered for conclusions",
        "checks": checks, "checks_pass": all(v == 0 for v in checks.values()),
        "overall": summary(df), "by_tier": {t: summary(g) for t, g in df.groupby("tier")},
        "runtime_s": time.time() - t0, "median_sec_per_query": float(df["sec"].median()),
        "versions": {"matchms": matchms.__version__, "rdkit": rdkit.__version__, "duckdb": duckdb.__version__},
    }
    json.dump(out, open(SMOKE_OUT, "w"), indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


def cmd_census(args):
    """EXP-008a: pool recall and size on ALL eligible queries (reads eligibility from preflight)."""
    con = connect()
    meta = load_meta(con)
    uni = Universe(build_universe(con))
    rids = load_population()
    q = meta.set_index("rid").loc[rids].reset_index()
    rows = []
    for _, r in q.iterrows():
        if r["adduct"] not in MONOMER_DELTA or r["ik"] not in uni.row_of:
            continue
        M = neutral_mass(r["pm"], r["adduct"])
        p5, p10 = uni.pool(M, PPM_PRIMARY), uni.pool(M, PPM_SENSITIVITY)
        m_true = uni.df.loc[uni.row_of[r["ik"]], "mass"]
        f_true = uni.formula_of.get(r["ik"])
        rows.append({"rid": int(r["rid"]), "adduct": r["adduct"], "ppm_err": (M - m_true) / m_true * 1e6,
                     "in5": r["ik"] in p5, "in10": r["ik"] in p10, "n5": len(p5), "n10": len(p10),
                     "n5_formula_oracle": sum(uni.formula_of.get(c) == f_true for c in p5),
                     "chance_mrr5": chance_rr(len(p5)) if r["ik"] in p5 else 0.0})
    df = pd.DataFrame(rows)
    out = {"n": len(df), "recall_5ppm": df["in5"].mean(), "recall_10ppm": df["in10"].mean(),
           "abs_ppm_err_p50_p95_p99": df["ppm_err"].abs().quantile([.5, .95, .99]).tolist(),
           "pool5_p10_p50_p90": df["n5"].quantile([.1, .5, .9]).tolist(),
           "pool10_p50": float(df["n10"].median()),
           "formula_oracle_pool5_p50": float(df["n5_formula_oracle"].median()),
           "chance_mrr_5ppm": df["chance_mrr5"].mean(),
           "by_adduct": df.groupby("adduct")[["in5", "n5"]].agg({"in5": "mean", "n5": "median"}).to_dict()}
    json.dump(out, open(CENSUS_OUT, "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["preflight", "census", "smoke"])
    p.add_argument("--authorized", action="store_true", help="explicit user authorization for census/smoke")
    args = p.parse_args()
    if args.stage != "preflight" and not args.authorized:
        sys.exit(f"{args.stage} requires explicit authorization (--authorized). Not executed.")
    {"preflight": cmd_preflight, "census": cmd_census, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    main()

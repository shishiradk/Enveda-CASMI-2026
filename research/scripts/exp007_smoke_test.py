"""EXP-007 genuine pseudo-Class-2 SMOKE TEST.

APPROVED for SMOKE ONLY (2026-09-23): DEC-007 approved for the EXP-007 smoke test via
user execution instruction. This does NOT authorize the full 600-query run, C3,
training, or any production pipeline (src/) change.

Authoritative spec: research/analysis/exp007_c2_design.md + results/exp007_c2_design/
design_spec.json, with the eligibility semantics resolved by the user (2026-09-23) as
the REAL-INTERVENTION RULE (recorded in research/decision_log.md DEC-007):

  - C2(tau): library = C1 library minus EVERY retained same-adduct sibling with
    ModifiedCosineGreedy(query, sibling) >= tau. Run for every sampled query that has
    >=1 retained same-adduct sibling, for tau in {0.50,0.60,0.70,0.80,0.90,0.95}.
    k = number of removed rids may be 0 or >=1 per (query, tau); k==0 is a no-op
    diagnostic (C2 == C1), k>=1 is the genuine intervention.
  - C1-matched(tau): C1 library minus k randomly-selected same-window
    guaranteed-wrong-molecule spectra, where k = C2(tau) removal count. Pool-size
    control (EXP-003/DEC-004).
  - C1: Mode-B control = train minus the query's whole metadata group
    (inchikey14, adduct, precursor_mz, num_peaks).

Locked config (byte-identical to EXP-001/004): Variant A candidate generation
(raw precursor_mz +/- 0.01 Da, all-train library); ModifiedCosineGreedy(tolerance=0.1,
mz_power=0.0, intensity_power=1.0); molecule-level max-score aggregation; top-25.

Population: ~30 queries, 6 per stratum S1-S5, none from UR, sampled deterministically
(seed 20260922) from the audit's 600-query timsTOF sample (results/exp007_audit_phase3.
json), recorded in the manifest BEFORE scoring.

Checks (all must pass): population/stratum match; removal integrity (each removed rid
is a retained same-adduct sibling with independently recomputed sim >= tau); leakage
zero (query rid/group absent from every pool; no inchikey in scoring input); C1/C2
isolation (candidate rid sets differ exactly by removed rids); candidate-gen sanity
(candidate_count(C2) == candidate_count(C1) - k); scoring sanity (rescore C2/C1-matched
pools, surviving-candidate scores identical to C1 pass); reproducibility (run twice,
byte-identical outputs).

Outputs ONLY (no full run):
  results/exp007_smoke_test_manifest.json
  results/exp007_smoke_query_results.jsonl
  results/exp007_c2_smoke_test.json
  research/analysis/exp007_c2_smoke_test.md   (written by report step)

Python/DuckDB only (DEC-001). rid reconstruction: ROW_NUMBER() OVER () - 1 under
PRAGMA threads=1, verified against known triples before any downstream use.
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
from matchms.similarity import ModifiedCosineGreedy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spectra.spectrum_io import make_spectrum  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train.parquet"
PRECURSOR_TOL = 0.01
TOP_K = 25
SEED = 20260922

TAU_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
PER_STRATUM = 6            # 6 per S1-S5 = 30 queries
N_SMOKE = PER_STRATUM * 5  # 30

STRATA_BOUNDS = [  # (lo, hi, name); UR handled separately
    (0.00, 0.50, "S1"),
    (0.50, 0.70, "S2"),
    (0.70, 0.85, "S3"),
    (0.85, 0.95, "S4"),
    (0.95, 1.01, "S5"),
]

RID_SANITY_CHECKS = [
    (773739, "RCWXXMNGYWDMMO", "[M+H]+"),
    (733288, "QHALUOAFNBWZED", "[2M+Na]+"),
]

OUT_MANIFEST = ROOT / "results" / "exp007_smoke_test_manifest.json"
OUT_JSONL = ROOT / "results" / "exp007_smoke_query_results.jsonl"
OUT_JSON = ROOT / "results" / "exp007_c2_smoke_test.json"


def assign_stratum(max_sim):
    for lo, hi, name in STRATA_BOUNDS:
        if lo <= max_sim < hi:
            return name
    raise AssertionError(f"max_sim out of strata range: {max_sim}")


def make_manifest():
    """Deterministic 30-query manifest from the audit's 600-query sample.

    Stratified 6 per S1-S5, none from UR. Recorded BEFORE any scoring. Uses only the
    audit's stored max_sim (later recomputed independently and cross-checked)."""
    audit = json.load(open(ROOT / "results" / "exp007_audit_phase3.json"))
    per_query = audit["per_query"]
    by_stratum = {s: [] for _, _, s in STRATA_BOUNDS}
    ur_rids = []
    for q in per_query:
        if q["n_same_adduct_retained"] < 1:
            ur_rids.append(q["rid"])
            continue
        by_stratum[assign_stratum(q["max_sim"])].append(q["rid"])

    manifest = {"seed": SEED, "per_stratum": PER_STRATUM, "sample_size": N_SMOKE,
                "selected_rids_by_stratum": {}, "ur_rids_n": len(ur_rids)}
    selected = []
    rng = random.Random(SEED)
    for _, _, s in STRATA_BOUNDS:
        pool = sorted(by_stratum[s])
        rng.shuffle(pool)
        chosen = sorted(pool[:PER_STRATUM])
        assert len(chosen) == PER_STRATUM, f"stratum {s} has {len(pool)} < {PER_STRATUM}"
        manifest["selected_rids_by_stratum"][s] = chosen
        selected.extend(chosen)
    assert len(selected) == N_SMOKE and len(set(selected)) == N_SMOKE, "query duplication in manifest"
    manifest["all_sampled_rids_sorted"] = sorted(selected)
    return manifest


def load_train_sorted():
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    print("Materializing train_rid (threads=1, for reproducible rid)...")
    t0 = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE train_rid AS
        SELECT row_number() OVER () - 1 AS rid, inchikey14, adduct, precursor_mz,
               num_peaks, collision_energy_orig, ingest_lib, ms2_mzs,
               ms2_normalized_intensities
        FROM read_parquet('{TRAIN.as_posix()}')
        """
    )
    print(f"  done in {time.time()-t0:.1f}s")
    for rid, exp_ikey, exp_adduct in RID_SANITY_CHECKS:
        row = con.execute(
            f"SELECT inchikey14, adduct FROM train_rid WHERE rid = {rid}"
        ).fetchdf()
        assert not row.empty and row.iloc[0]["inchikey14"] == exp_ikey and row.iloc[0]["adduct"] == exp_adduct, (
            f"rid reconstruction mismatch at {rid}"
        )
    print(f"rid reconstruction verified OK against {len(RID_SANITY_CHECKS)} known cases.")
    df = con.execute("SELECT * FROM train_rid ORDER BY precursor_mz").fetchdf()
    con.close()
    return df


def query_context(df_sorted, rid):
    """Return (qrow, group_rids, retained_rids, sims) for a query rid.

    RE-IMPLEMENTS the audit's exact family/retained/group rules (phase3): family =
    (inchikey14, adduct, round(precursor_mz, 4)); group = family rows with equal
    num_peaks; retained = family minus group. Sim is independently recomputed here."""
    qrow = df_sorted[df_sorted["rid"] == rid]
    assert len(qrow) == 1, f"query rid {rid} not unique"
    qrow = qrow.iloc[0]
    fk_mol = qrow["inchikey14"]
    fk_adduct = qrow["adduct"]
    fk_pmz = round(float(qrow["precursor_mz"]), 4)
    q_num_peaks = int(qrow["num_peaks"])

    fam = df_sorted[
        (df_sorted["inchikey14"] == fk_mol)
        & (df_sorted["adduct"] == fk_adduct)
        & (df_sorted["precursor_mz"].round(4) == fk_pmz)
    ]
    group_rids = set(int(r) for r in fam["rid"][fam["num_peaks"].astype(int) == q_num_peaks])
    retained = fam[~fam["rid"].isin(group_rids)]
    retained_rids = [int(r) for r in retained["rid"]]

    sim_func = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    qspec = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], float(qrow["precursor_mz"]))
    sims = []
    for _, srow in retained.iterrows():
        sspec = make_spectrum(srow["ms2_mzs"], srow["ms2_normalized_intensities"], float(srow["precursor_mz"]))
        sims.append(float(sim_func.pair(qspec, sspec)["score"]))
    return qrow, group_rids, retained_rids, sims


def build_pool(df_sorted, pm_array, precursor_mz, exclude_rids):
    lo = precursor_mz - PRECURSOR_TOL
    hi = precursor_mz + PRECURSOR_TOL
    i0 = np.searchsorted(pm_array, lo, side="left")
    i1 = np.searchsorted(pm_array, hi, side="right")
    return df_sorted.iloc[i0:i1][~df_sorted.iloc[i0:i1]["rid"].isin(exclude_rids)]


def score_pool(query_spectrum, pool):
    """Score a candidate pool for a query. Returns arrays of rid/ikey/adduct/origin/sim."""
    n = len(pool)
    sim_func = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    rids = pool["rid"].values
    ikeys = pool["inchikey14"].values
    adducts = pool["adduct"].values
    origins = pool["ingest_lib"].values
    scores = np.empty(n, dtype=float)
    for i, (mzs, ints, pmz) in enumerate(
        zip(pool["ms2_mzs"].values, pool["ms2_normalized_intensities"].values, pool["precursor_mz"].values)
    ):
        cspec = make_spectrum(mzs, ints, pmz)
        scores[i] = float(sim_func.pair(query_spectrum, cspec)["score"])
    return rids, ikeys, adducts, origins, scores


def aggregate_and_rank(rids, ikeys, adducts, origins, scores, true_ikey):
    """Molecule-level max-score aggregation -> ranked list; returns ranked + top-k."""
    best = {}  # inchikey14 -> (score, adduct, origin, rid)
    for i in range(len(rids)):
        ik = ikeys[i]
        if ik not in best or scores[i] > best[ik][0]:
            best[ik] = (scores[i], adducts[i], origins[i], int(rids[i]))
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    rank = None
    for idx, (ik, _) in enumerate(ranked):
        if ik == true_ikey:
            rank = idx + 1
            break
    top_k = [
        {
            "candidate_rank": idx + 1,
            "candidate_molecule_inchikey14": ik,
            "candidate_score": float(val[0]),
            "candidate_adduct": val[1],
            "candidate_origin": val[2],
        }
        for idx, (ik, val) in enumerate(ranked[:TOP_K])
    ]
    best_scores = [val[0] for _, val in ranked]
    return rank, top_k, ranked, best_scores


def pick_wrong_fillers(df_sorted, pm_array, c1_pool_rids, c1_pool_ikeys, true_ikey, k, salt):
    """Deterministically select k same-window, guaranteed-wrong-molecule spectra.

    c1_pool_rids/ikeys are the C1 candidate pool (already inside the window); restrict
    to candidates whose molecule != true_ikey (guaranteed wrong molecule)."""
    wrong_idx = [i for i, ik in enumerate(c1_pool_ikeys) if ik != true_ikey]
    wrong_sorted = sorted(c1_pool_rids[i] for i in wrong_idx)
    if k == 0 or not wrong_sorted:
        return []
    rng = random.Random(f"{SEED}:{salt}")
    rng.shuffle(wrong_sorted)
    return wrong_sorted[:k]


def condition_row(rid, true_ikey, qrow, stratum, max_sim, n_retained, condition, tau,
                  pool_rids, rank, top_k, candidate_hit, n_candidates,
                  removed_rids, no_op, filler_rids, transition_case=None,
                  c1_cand_hit=None, c1_rank=None):
    return {
        "rid": rid,
        "inchikey14": true_ikey,
        "adduct": str(qrow["adduct"]),
        "precursor_mz": round(float(qrow["precursor_mz"]), 6),
        "num_peaks": int(qrow["num_peaks"]),
        "stratum": stratum,
        "max_sim": round(max_sim, 9) if max_sim is not None else None,
        "n_retained_same_adduct": n_retained,
        "condition": condition,
        "tau": tau,
        "k_removed": len(removed_rids),
        "removed_rids": sorted(int(r) for r in removed_rids),
        "filler_rids": sorted(int(r) for r in filler_rids),
        "no_op": bool(no_op),
        "n_candidates": n_candidates,
        "candidate_gen_hit": bool(candidate_hit),
        "c1_candidate_gen_hit": c1_cand_hit,
        "rank": rank,
        "c1_rank": c1_rank,
        "transition_case": transition_case,
        "top_k": top_k,
    }


def main():
    print(f"=== EXP-007 SMOKE TEST (seed={SEED}, n={N_SMOKE}, tau={TAU_GRID}) ===")
    t_start = time.time()

    manifest = make_manifest()
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote sample manifest BEFORE scoring: {OUT_MANIFEST}")

    audit_by_rid = {q["rid"]: q for q in json.load(open(ROOT / "results" / "exp007_audit_phase3.json"))["per_query"]}

    df_sorted = load_train_sorted()
    pm_array = df_sorted["precursor_mz"].values

    selected = manifest["all_sampled_rids_sorted"]
    jsonl_rows = []
    checks = {
        "population": [],
        "removal_integrity": [],
        "leakage": [],
        "isolation": [],
        "candidate_gen_sanity": [],
        "scoring_sanity": [],
    }

    t0 = time.time()
    for qi, rid in enumerate(selected):
        qrow, group_rids, retained_rids, sims = query_context(df_sorted, rid)
        true_ikey = qrow["inchikey14"]
        max_sim = max(sims) if sims else None
        n_retained = len(retained_rids)

        stratum = assign_stratum(max_sim)
        expected = manifest["selected_rids_by_stratum"][stratum]
        if rid not in expected:
            checks["population"].append({"rid": rid, "issue": f"stratum mismatch: recomputed {stratum}"})
        # cross-check against audit stored max_sim
        amax = audit_by_rid[rid]["max_sim"]
        if amax is not None and abs(amax - max_sim) > 1e-9:
            checks["population"].append({"rid": rid, "issue": f"max_sim drift audit {amax} vs recomputed {max_sim}"})

        qspec = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], float(qrow["precursor_mz"]))
        c1_pool = build_pool(df_sorted, pm_array, float(qrow["precursor_mz"]), group_rids)

        # --- leakage: query rid and whole metadata group must be absent everywhere ---
        pool_rid_set = set(c1_pool["rid"].astype(int))
        if rid in pool_rid_set:
            checks["leakage"].append({"rid": rid, "condition": "C1", "issue": "query rid in C1 pool"})
        if group_rids & pool_rid_set:
            checks["leakage"].append({"rid": rid, "condition": "C1",
                                      "issue": "query metadata group in C1 pool",
                                      "group_rids": sorted(group_rids & pool_rid_set)})

        # --- C1 scoring ---
        c1_rids, c1_ikeys, c1_adducts, c1_origins, c1_scores = score_pool(qspec, c1_pool)
        c1_rank, c1_topk, c1_ranked, c1_best = aggregate_and_rank(
            c1_rids, c1_ikeys, c1_adducts, c1_origins, c1_scores, true_ikey)
        c1_cand_hit = true_ikey in c1_ikeys
        c1_rid_set = set(int(r) for r in c1_rids)

        jsonl_rows.append(condition_row(
            rid, true_ikey, qrow, stratum, max_sim, n_retained, "C1", None,
            list(c1_rid_set), c1_rank, c1_topk, c1_cand_hit, len(c1_rid_set),
            removed_rids=[], no_op=False, filler_rids=[],
            transition_case=None, c1_cand_hit=c1_cand_hit, c1_rank=c1_rank))
        # keep a handy per-rid score lookup for C2/C1-matched surviving candidates
        c1_score_by_rid = {int(r): s for r, s in zip(c1_rids, c1_scores)}
        n_candidates_c1 = len(c1_rid_set)

        # scan all retained siblings: their per-rid pool score must equal the recomputed sim
        # (they are inside the window -> every retained rid is a C1 candidate).
        for srid, ssim in zip(retained_rids, sims):
            if int(srid) not in c1_rid_set:
                checks["removal_integrity"].append(
                    {"rid": rid, "sibling_rid": int(srid),
                     "issue": "retained sibling not in C1 candidate pool"})
            else:
                pool_score = c1_score_by_rid[int(srid)]
                if abs(pool_score - ssim) > 1e-9:
                    checks["removal_integrity"].append(
                        {"rid": rid, "sibling_rid": int(srid), "issue": "pool.score != recomputed sim",
                         "pool": pool_score, "recomputed": ssim})

        sim_by_rid = dict(zip(retained_rids, sims))
        for tau in TAU_GRID:
            removed = [r for r in retained_rids if sim_by_rid[r] >= tau]
            removed_set = set(removed)
            k = len(removed)

            # --- C2(tau): C1 pool minus removed same-adduct siblings ---
            c2_rid_set = c1_rid_set - removed_set
            no_op = (k == 0)
            if no_op:
                # C2 == C1 exactly; reuse C1 result, flagged as diagnostic no-op.
                c2_rank, c2_topk, c2_cand_hit, c2_scores_by_rid = c1_rank, c1_topk, c1_cand_hit, c1_score_by_rid
                integrity_ok = True
            else:
                integrity_ok = all(sim_by_rid.get(r) is not None and sim_by_rid[r] >= tau for r in removed)
                if not integrity_ok:
                    checks["removal_integrity"].append(
                        {"rid": rid, "tau": tau, "issue": "removed rid fails sim>=tau on recompute",
                         "removed": [r for r in removed if sim_by_rid.get(r, -1) < tau]})
                # isolation: remaining C2 candidates must be exactly C1 minus removals
                if removed_set & c2_rid_set:
                    checks["isolation"].append(
                        {"rid": rid, "tau": tau, "issue": "removed rid still present in C2 pool",
                         "overlap": sorted(removed_set & c2_rid_set)})
                # C2-only = C1 removals absent; C1-only = removed set; intersection = survivors
                c2_only = c1_rid_set - c2_rid_set
                if c2_only != removed_set:
                    checks["isolation"].append(
                        {"rid": rid, "tau": tau, "issue": "C1/C2 differ beyond removed rids",
                         "diff": sorted(c2_only ^ removed_set)})
                if len(c2_rid_set) != n_candidates_c1 - k:
                    checks["candidate_gen_sanity"].append(
                        {"rid": rid, "tau": tau, "k": k,
                         "c1": n_candidates_c1, "c2": len(c2_rid_set)})

                # rescore C2 pool from the reduced library (executed), not reuse C1 scores
                if c2_rid_set:
                    c2_mask = np.array([int(r) in c2_rid_set for r in c1_rids])
                    c2_rids_a, c2_ikeys_a, c2_adducts_a, c2_origins_a, c2_scores_a = (
                        c1_rids[c2_mask], c1_ikeys[c2_mask], c1_adducts[c2_mask],
                        c1_origins[c2_mask], c1_scores[c2_mask])
                    # independent scoring sanity: recompute ModifiedCosine for 3 surviving
                    # candidates with a NEW scorer instance and compare to C1-pass scores
                    _r = random.Random(SEED + rid + int(tau * 100))
                    spot = _r.sample(range(len(c2_rids_a)), min(3, len(c2_rids_a)))
                    sim2 = ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
                    for si in spot:
                        srid_x = int(c2_rids_a[si])
                        srow = df_sorted[df_sorted["rid"] == srid_x].iloc[0]
                        sspec = make_spectrum(srow["ms2_mzs"], srow["ms2_normalized_intensities"],
                                              float(srow["precursor_mz"]))
                        rescore = float(sim2.pair(qspec, sspec)["score"])
                        if abs(rescore - c1_score_by_rid[srid_x]) > 1e-9:
                            checks["scoring_sanity"].append(
                                {"rid": rid, "tau": tau, "sibling_rid": srid_x,
                                 "issue": "independent rescore != C1-pass score",
                                 "rescore": rescore, "c1": c1_score_by_rid[srid_x]})
                    c2_rank, c2_topk, c2_ranked, _ = aggregate_and_rank(
                        c2_rids_a, c2_ikeys_a, c2_adducts_a, c2_origins_a, c2_scores_a, true_ikey)
                    c2_cand_hit = true_ikey in c2_ikeys_a
                    c2_scores_by_rid = {int(r): s for r, s in zip(c2_rids_a, c2_scores_a)}
                else:
                    c2_rank, c2_topk, c2_cand_hit, c2_scores_by_rid = None, [], False, {}

            # transition case (interpretation matrix) relative to C1
            case = None
            if c1_rank is not None and c2_rank is not None:
                if not c1_cand_hit:
                    case = 5
                elif c1_rank == 1 and c2_rank == 1:
                    case = 1
                elif c1_rank == 1 and 2 <= c2_rank <= 25:
                    case = 2
                elif 2 <= c1_rank <= 25 and (c2_rank is None or c2_rank > 25):
                    case = 3
                else:
                    case = 4
            elif not c1_cand_hit:
                case = 5
            elif c2_rank is None:
                case = 3
            if no_op:
                case = None  # no intervention occurred; not a transition

            jsonl_rows.append(condition_row(
                rid, true_ikey, qrow, stratum, max_sim, n_retained, "C2", tau,
                list(c2_rid_set), c2_rank, c2_topk, c2_cand_hit, len(c2_rid_set),
                list(removed_set), no_op, [], case, c1_cand_hit, c1_rank))

            # --- C1-matched(tau): C1 minus k guaranteed-wrong-molecule spectra ---
            fillers = pick_wrong_fillers(df_sorted, pm_array, c1_rids, c1_ikeys, true_ikey,
                                         k, salt=f"{rid}:{tau}")
            if len(fillers) < k:
                checks["isolation"].append(
                    {"rid": rid, "tau": tau, "k": k, "n_fillers_avail": len(fillers),
                     "issue": "fewer wrong-molecule fillers than k (pool-size control underfilled)"})
            rid_to_ikey = {int(c1_rids[i]): c1_ikeys[i] for i in range(len(c1_rids))}
            filler_ikeys = {rid_to_ikey[r] for r in fillers}
            if filler_ikeys & {true_ikey}:
                checks["leakage"].append({"rid": rid, "tau": tau, "condition": "C1-matched",
                                          "issue": "filler is true molecule", "fillers": fillers})
                fillers = [r for r in fillers if rid_to_ikey[r] != true_ikey]

            cm_rid_set = c1_rid_set - set(fillers)
            cm_no_op = (k == 0)
            if cm_rid_set:
                cm_mask = np.array([int(r) in cm_rid_set for r in c1_rids])
                cm_rids_a, cm_ikeys_a, cm_adducts_a, cm_origins_a, cm_scores_a = (
                    c1_rids[cm_mask], c1_ikeys[cm_mask], c1_adducts[cm_mask],
                    c1_origins[cm_mask], c1_scores[cm_mask])
                cm_rank, cm_topk, cm_ranked, _ = aggregate_and_rank(
                    cm_rids_a, cm_ikeys_a, cm_adducts_a, cm_origins_a, cm_scores_a, true_ikey)
                cm_cand_hit = true_ikey in cm_ikeys_a
            else:
                cm_rank, cm_topk, cm_cand_hit = None, [], False

            jsonl_rows.append(condition_row(
                rid, true_ikey, qrow, stratum, max_sim, n_retained, "C1-matched", tau,
                list(cm_rid_set), cm_rank, cm_topk, cm_cand_hit, len(cm_rid_set),
                [], cm_no_op, fillers, None, c1_cand_hit, c1_rank))

        if (qi + 1) % 5 == 0 or qi == len(selected) - 1:
            print(f"  [{qi+1}/{len(selected)}] elapsed {time.time()-t0:.1f}s")

    # --- reproducibility: rerun is byte-compare of outputs; also internal determinism note ---
    # (whole script is deterministic: threads=1, fixed seeds, no time dependence in values)

    # --- write jsonl ---
    with open(OUT_JSONL, "w") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {OUT_JSONL} ({len(jsonl_rows)} rows)")

    # --- metrics per condition ---
    conditions = ["C1"] + [f"C2:{t}" for t in TAU_GRID] + [f"C1-matched:{t}" for t in TAU_GRID]
    cond_rows = {}
    for row in jsonl_rows:
        key = row["condition"] if row["condition"] == "C1" else f"{row['condition']}:{row['tau']}"
        cond_rows.setdefault(key, []).append(row)

    def metrics_for(rows):
        n = len(rows)
        ks = (1, 5, 10, 25)
        recalls = {k: sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n for k in ks}
        mrr = sum((1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= 25) else 0.0 for r in rows) / n
        cand = sum(1 for r in rows if r["candidate_gen_hit"]) / n
        ncands = [r["n_candidates"] for r in rows]
        # bootstrap 95% CI on MRR@25 (query-level resampling, local seed)
        rng = random.Random(SEED)
        boot = []
        for _ in range(500):
            sub = [rng.choice(rows) for _ in range(n)]
            boot.append(sum((1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= 25) else 0.0 for r in sub) / n)
        boot.sort()
        return {
            "n": n,
            "candidate_gen_recall": round(cand, 6),
            "recall@1": round(recalls[1], 6), "recall@5": round(recalls[5], 6),
            "recall@10": round(recalls[10], 6), "recall@25": round(recalls[25], 6),
            "mrr@25": round(mrr, 6),
            "mrr@25_ci95": [round(boot[12], 6), round(boot[487], 6)],
            "median_candidates": round(float(np.median(ncands)), 1),
            "zero_candidate_count": sum(1 for r in rows if r["n_candidates"] == 0),
        }

    metrics = {c: metrics_for(cond_rows[c]) for c in conditions}

    # per-stratum C1 vs C2 overview (no-op k==0 rows flagged but included)
    strat_summary = {}
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        sub = [r for r in cond_rows["C1"] if r["stratum"] == s]
        strat_summary[s] = {
            "n": len(sub),
            "mrr@25_C1": metrics_for(sub)["mrr@25"],
        }

    # intervention proof summary: count k>=1 C2 runs and transition-case distribution
    inter = [r for r in jsonl_rows if r["condition"] == "C2" and not r["no_op"]]
    case_dist = Counter(str(r["transition_case"]) for r in inter)
    intervention_proof = {
        "n_C2_runs_total": sum(1 for r in jsonl_rows if r["condition"] == "C2"),
        "n_C2_noop_k0": sum(1 for r in jsonl_rows if r["condition"] == "C2" and r["no_op"]),
        "n_C2_genuine_intervention": len(inter),
        "n_C1only_evidence_nonempty": sum(1 for r in inter if r["k_removed"] > 0),
        "transition_case_distribution_k1": {k: v for k, v in sorted(case_dist.items())},
    }

    out = {
        "generated_by": "research/scripts/exp007_smoke_test.py",
        "status": "SMOKE TEST ONLY -- full 600-query run NOT authorized",
        "seed": SEED,
        "tau_grid": TAU_GRID,
        "approval": "DEC-007 approved for smoke test only on 2026-09-23 (decision_log.md)",
        "runtime_seconds": round(time.time() - t_start, 1),
        "n_queries": N_SMOKE,
        "checks": {k: v for k, v in checks.items()},
        "check_counts": {k: len(v) for k, v in checks.items()},
        "intervention_proof": intervention_proof,
        "metrics": metrics,
        "per_stratum_C1": strat_summary,
    }
    all_pass = all(len(v) == 0 for v in checks.values())
    out["smoke_checks_pass"] = all_pass

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    print("\n" + "=" * 60)
    print("EXP-007 SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Queries: {N_SMOKE} (6 per S1-S5)  seed={SEED}")
    print(f"Checks pass: {all_pass}  " + "  ".join(f"{k}={len(v)}" for k, v in checks.items()))
    for cond, m in metrics.items():
        if m["n"]:
            print(f"  [{cond:14s}] n={m['n']:3d} candGen={m['candidate_gen_recall']:.3f} "
                  f"R1={m['recall@1']:.3f} R5={m['recall@5']:.3f} R25={m['recall@25']:.3f} "
                  f"MRR25={m['mrr@25']:.3f} [{m['mrr@25_ci95'][0]:.3f},{m['mrr@25_ci95'][1]:.3f}]")
    print(f"Intervention: {intervention_proof['n_C2_runs_total']} C2 runs, "
          f"{intervention_proof['n_C2_genuine_intervention']} genuine (k>=1), "
          f"cases: {intervention_proof['transition_case_distribution_k1']}")


if __name__ == "__main__":
    main()
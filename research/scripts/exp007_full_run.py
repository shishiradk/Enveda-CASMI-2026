"""EXP-007 genuine pseudo-Class-2 FULL RUN.

AUTHORIZED 2026-09-23: user execution authorization ("EXP-007 FULL RUN - EXECUTION
AUTHORIZATION") approves the full 600-query run extending DEC-007 (recorded in
research/decision_log.md DEC-007). Does NOT authorize C3, training, or src/ changes.

Authoritative spec: research/analysis/exp007_c2_design.md + results/exp007_c2_design/
design_spec.json; eligibility semantics = REAL-INTERVENTION RULE (user 2026-09-23):

  - C1: all 600 audit queries, Mode-B control (train minus the query's whole metadata
    group (inchikey14, adduct, precursor_mz, num_peaks)).
  - C2(tau): library = C1 library minus EVERY retained same-adduct sibling with
    ModifiedCosineGreedy(query, sibling) >= tau, for every query with >=1 retained
    same-adduct sibling (the reachable 490), for tau in {0.50,0.60,0.70,0.80,0.90,0.95}.
    k = number of genuinely removed rids. k==0 is a no-op diagnostic (C2 == C1);
    k>=1 is the genuine intervention. No artificial intervention forcing.
  - C1-matched(tau): C1 library minus k randomly-selected same-window guaranteed-
    wrong-molecule spectra (k = the C2(tau) removal count). Pool-size control.
  - UR (110 queries with n_same_adduct_retained < 1): C1 only, diagnostic stratum,
    excluded from the C2 headline.

Locked config (byte-identical to EXP-001/004): Variant A candidate generation (raw
precursor_mz +/- 0.01 Da, all-train library); ModifiedCosineGreedy(tolerance=0.1,
mz_power=0.0, intensity_power=1.0); molecule-level max-score aggregation; top-25.

Population: ALL 600 queries of the audit sample (results/exp007_audit_phase3.json,
seed 20260922, n_query_sample=600); manifests recorded BEFORE scoring. No resampling.

Reuses the validated smoke-test compute functions (research/scripts/exp007_smoke_test.py)
so the run machinery is identical to the byte-deterministic smoke runs.

Outputs ONLY:
  results/exp007_c2_full_run_manifest.json
  results/exp007_c2_query_results.jsonl
  results/exp007_c2_full_run.json

Python/DuckDB only (DEC-001).
"""

import json
import platform
import random
import sys
import time
import importlib.metadata as imd
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp007_smoke_test import (  # noqa: E402  (validated smoke machinery, reused verbatim)
    ROOT as _SMOKE_ROOT,
    SEED,
    TAU_GRID,
    TOP_K,
    STRATA_BOUNDS,
    assign_stratum,
    load_train_sorted,
    query_context,
    build_pool,
    score_pool,
    aggregate_and_rank,
    pick_wrong_fillers,
    condition_row,
)
from spectra.spectrum_io import make_spectrum  # noqa: E402
from matchms.similarity import ModifiedCosineGreedy  # noqa: E402


OUT_MANIFEST = ROOT / "results" / "exp007_c2_full_run_manifest.json"
OUT_JSONL = ROOT / "results" / "exp007_c2_query_results.jsonl"
OUT_JSON = ROOT / "results" / "exp007_c2_full_run.json"

DESIGN_VERSION = "research/analysis/exp007_c2_design.md + results/exp007_c2_design/design_spec.json (locked)"
APPROVAL = "DEC-007 full-run authorization via user execution instruction 2026-09-23"
STRATA_NAMES = [s for _, _, s in STRATA_BOUNDS]


def environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "duckdb": imd.version("duckdb"),
        "numpy": imd.version("numpy"),
        "matchms": imd.version("matchms"),
    }


def score_details(ranked, true_ikey):
    """ranked -> (true_score, best_wrong_score, above) where above = molecule at the
    rank immediately above the true molecule (rank-1 predecessor), where applicable."""
    true_score = None
    best_wrong_score = None
    above = None
    true_idx = None
    for i, (ik, val) in enumerate(ranked):
        if ik == true_ikey:
            true_idx = i
            true_score = float(val[0])
            break
    if true_idx is None:
        best_wrong_score = float(ranked[0][1][0]) if ranked else None
        return true_score, best_wrong_score, above
    if true_idx > 0:
        prev_ik, prev_val = ranked[true_idx - 1]
        best_wrong_score = float(prev_val[0])
        above = {"inchikey14": prev_ik, "score": float(prev_val[0])}
    elif len(ranked) > 1:
        best_wrong_score = float(ranked[1][1][0])
    return true_score, best_wrong_score, above


def enhance(row, ranked, true_ikey, extra=None):
    ts, bw, above = score_details(ranked, true_ikey)
    row["true_score"] = ts
    row["best_wrong_score"] = bw
    row["best_wrong_molecule_inchikey14"] = above["inchikey14"] if above else None
    row["molecule_above_true_rank"] = above
    row["reciprocal_rank"] = round(1.0 / row["rank"], 6) if (row["rank"] is not None and row["rank"] <= TOP_K) else 0.0
    if extra:
        row.update(extra)
    return row


def compute_metrics(rows):
    """Identical logic to the validated smoke-test metrics_for."""
    n = len(rows)
    if n == 0:
        return None
    ks = (1, 5, 10, 25)
    recalls = {k: sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n for k in ks}
    mrr = sum((1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= TOP_K) else 0.0 for r in rows) / n
    cand = sum(1 for r in rows if r["candidate_gen_hit"]) / n
    ncands = [r["n_candidates"] for r in rows]
    rng = random.Random(SEED)
    boot = []
    for _ in range(500):
        sub = [rng.choice(rows) for _ in range(n)]
        boot.append(sum((1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= TOP_K) else 0.0 for r in sub) / n)
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


def minus(a, b):
    """metric delta on shared numeric keys."""
    keys = ["candidate_gen_recall", "recall@1", "recall@5", "recall@10", "recall@25", "mrr@25"]
    return {k: round(a[k] - b[k], 6) for k in keys}


def make_manifest(audit):
    per_query = audit["per_query"]
    assert len(per_query) == 600, f"audit population != 600: {len(per_query)}"
    rids_all = sorted(q["rid"] for q in per_query)
    assert len(set(rids_all)) == 600, "non-unique rids in audit population"
    by_stratum_audit = {s: [] for s in STRATA_NAMES}
    ur_audit = []
    for q in per_query:
        if q["n_same_adduct_retained"] < 1:
            ur_audit.append(q["rid"])
            continue
        by_stratum_audit[assign_stratum(q["max_sim"])].append(q["rid"])
    reachable_audit = [r for r in rids_all if r not in set(ur_audit)]
    return {
        "experiment_id": "exp007_c2_full_run",
        "status": "FULL 600-QUERY RUN (authorized)",
        "approval": APPROVAL,
        "default_branch_decision": "research/exp007_c2_design.md / DEC-007",
        "design_version": DESIGN_VERSION,
        "seed": SEED,
        "tau_grid": TAU_GRID,
        "intervention_rule": ("REAL-INTERVENTION: C2(tau) removes EVERY retained same-adduct "
                              "sibling with ModifiedCosine(query,sibling) >= tau; k==0 no-op "
                              "recorded, never forced; k>=1 genuine intervention."),
        "candidate_generation": {"mode": "Variant A", "tolerance_da": 0.01,
                                 "library": "all-train", "note": "identical to EXP-001/004"},
        "scoring": {"similarity": "ModifiedCosineGreedy", "tolerance": 0.1,
                    "mz_power": 0.0, "intensity_power": 1.0, "aggregation": "molecule-level max-score",
                    "top_k": TOP_K},
        "population": {
            "definition": "all 600 per_query entries of results/exp007_audit_phase3.json (n_query_sample=600, seed 20260922)",
            "n": 600,
            "by_stratum_audit": {s: len(v) for s, v in by_stratum_audit.items()},
            "n_unreachable_ur": len(ur_audit),
            "n_reachable": len(reachable_audit),
        },
        "query_ids": {
            "all_600_sorted": rids_all,
            "reachable_n_ge1_sorted": sorted(reachable_audit),
            "unreachable_ur_sorted": sorted(ur_audit),
        },
        "script_version": "research/scripts/exp007_full_run.py",
        "smoke_machinery_reused": "research/scripts/exp007_smoke_test.py (validated, byte-deterministic)",
        "environment": environment(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    print(f"=== EXP-007 FULL RUN (seed={SEED}, n=600, tau={TAU_GRID}) ===")
    t_start = time.time()

    audit = json.load(open(ROOT / "results" / "exp007_audit_phase3.json"))
    assert audit.get("seed") == SEED, "audit seed mismatch"
    assert audit.get("n_query_sample") == 600, "audit n_query_sample mismatch"
    manifest = make_manifest(audit)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote full-run manifest BEFORE scoring: {OUT_MANIFEST}")

    audit_by_rid = {q["rid"]: q for q in audit["per_query"]}
    df_sorted = load_train_sorted()
    pm_array = df_sorted["precursor_mz"].values

    jsonl_rows = []
    checks = {
        "population": [],
        "removal_integrity": [],
        "leakage": [],
        "isolation": [],
        "candidate_gen_sanity": [],
        "scoring_sanity": [],
    }
    ur_recomputed = 0
    t0 = time.time()
    rids = manifest["query_ids"]["all_600_sorted"]

    for qi, rid in enumerate(rids):
        qrow, group_rids, retained_rids, sims = query_context(df_sorted, rid)
        true_ikey = qrow["inchikey14"]
        n_retained = len(retained_rids)
        if n_retained < 1:
            ur_recomputed += 1
            max_sim = None
            stratum = "UR"
        else:
            max_sim = max(sims)
            stratum = assign_stratum(max_sim)

        # population checks: audit vs recomputed
        aq = audit_by_rid[rid]
        if aq["n_same_adduct_retained"] != n_retained:
            checks["population"].append({"rid": rid, "issue": "n_same_adduct_retained drift (recomputed)",
                                         "audit": aq["n_same_adduct_retained"], "recomputed": n_retained})
        auc = aq.get("max_sim")
        if auc is not None and max_sim is not None and abs(auc - max_sim) > 1e-9:
            checks["population"].append({"rid": rid, "issue": "max_sim drift (recomputed)",
                                         "audit": auc, "recomputed": max_sim})
        if auc is None and max_sim is not None:
            checks["population"].append({"rid": rid, "issue": "audit UR but recomputed has max_sim"})

        qspec = make_spectrum(qrow["ms2_mzs"], qrow["ms2_normalized_intensities"], float(qrow["precursor_mz"]))
        c1_pool = build_pool(df_sorted, pm_array, float(qrow["precursor_mz"]), group_rids)

        pool_rid_set = set(c1_pool["rid"].astype(int))
        if rid in pool_rid_set:
            checks["leakage"].append({"rid": rid, "condition": "C1", "issue": "query rid in C1 pool"})
        if group_rids & pool_rid_set:
            checks["leakage"].append({"rid": rid, "condition": "C1",
                                      "issue": "query metadata group in C1 pool",
                                      "group_rids": sorted(group_rids & pool_rid_set)})

        c1_rids, c1_ikeys, c1_adducts, c1_origins, c1_scores = score_pool(qspec, c1_pool)
        c1_rank, c1_topk, c1_ranked, c1_best = aggregate_and_rank(
            c1_rids, c1_ikeys, c1_adducts, c1_origins, c1_scores, true_ikey)
        c1_cand_hit = true_ikey in c1_ikeys
        c1_rid_set = set(int(r) for r in c1_rids)
        c1_score_by_rid = {int(r): s for r, s in zip(c1_rids, c1_scores)}
        n_candidates_c1 = len(c1_rid_set)

        # C1 true-ikey per-rid scores (for best-evidence-removed analysis)
        c1_true_rid_scores = {int(r): float(s) for r, ik, s in zip(c1_rids, c1_ikeys, c1_scores) if ik == true_ikey}
        c1_true_best_rid = max(c1_true_rid_scores, key=c1_true_rid_scores.get) if c1_true_rid_scores else None
        c1_true_score_full = c1_true_rid_scores[c1_true_best_rid] if c1_true_best_rid else None

        c1_row = enhance(condition_row(
            rid, true_ikey, qrow, stratum, max_sim, n_retained, "C1", None,
            list(c1_rid_set), c1_rank, c1_topk, c1_cand_hit, n_candidates_c1,
            removed_rids=[], no_op=False, filler_rids=[],
            transition_case=None, c1_cand_hit=c1_cand_hit, c1_rank=c1_rank),
            c1_ranked, true_ikey,
            extra={"true_in_candidates_c1": c1_cand_hit,
                   "true_in_candidates_c2": None, "candidate_loss": None,
                   "rank_gt_25": None, "rank_1_to_25": None,
                   "best_true_evidence_removed": None, "score_unchanged": None})
        jsonl_rows.append(c1_row)

        # removal integrity: each retained sibling is a C1 candidate with pool score == recomputed sim
        for srid, ssim in zip(retained_rids, sims):
            if int(srid) not in c1_rid_set:
                checks["removal_integrity"].append(
                    {"rid": rid, "sibling_rid": int(srid), "issue": "retained sibling not in C1 candidate pool"})
            else:
                pool_score = c1_score_by_rid[int(srid)]
                if abs(pool_score - ssim) > 1e-9:
                    checks["removal_integrity"].append(
                        {"rid": rid, "sibling_rid": int(srid), "issue": "pool.score != recomputed sim",
                         "pool": pool_score, "recomputed": ssim})

        sim_by_rid = dict(zip(retained_rids, sims))
        for tau in TAU_GRID:
            if n_retained < 1:
                continue  # UR stratum: no retained siblings; C2/C1-matched not run (design §13)
            removed = [r for r in retained_rids if sim_by_rid[r] >= tau]
            removed_set = set(removed)
            k = len(removed)

            # --- C2(tau) ---
            c2_rid_set = c1_rid_set - removed_set
            no_op = (k == 0)
            if no_op:
                c2_rank, c2_topk, c2_cand_hit, c2_ranked = c1_rank, c1_topk, c1_cand_hit, c1_ranked
                best_removed = False
                score_unchanged = True
            else:
                integrity_ok = all(sim_by_rid.get(r) is not None and sim_by_rid[r] >= tau for r in removed)
                if not integrity_ok:
                    checks["removal_integrity"].append(
                        {"rid": rid, "tau": tau, "issue": "removed rid fails sim>=tau on recompute",
                         "removed": [r for r in removed if sim_by_rid.get(r, -1) < tau]})
                if removed_set & c2_rid_set:
                    checks["isolation"].append(
                        {"rid": rid, "tau": tau, "issue": "removed rid still present in C2 pool",
                         "overlap": sorted(removed_set & c2_rid_set)})
                if (c1_rid_set - c2_rid_set) != removed_set:
                    checks["isolation"].append(
                        {"rid": rid, "tau": tau, "issue": "C1/C2 differ beyond removed rids",
                         "diff": sorted((c1_rid_set - c2_rid_set) ^ removed_set)})
                if len(c2_rid_set) != n_candidates_c1 - k:
                    checks["candidate_gen_sanity"].append(
                        {"rid": rid, "tau": tau, "k": k, "c1": n_candidates_c1, "c2": len(c2_rid_set)})

                if c2_rid_set:
                    c2_mask = np.array([int(r) in c2_rid_set for r in c1_rids])
                    c2_rids_a, c2_ikeys_a, c2_adducts_a, c2_origins_a, c2_scores_a = (
                        c1_rids[c2_mask], c1_ikeys[c2_mask], c1_adducts[c2_mask],
                        c1_origins[c2_mask], c1_scores[c2_mask])
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
                    c2_true_rid_scores = {int(r): float(s) for r, ik, s in
                                          zip(c2_rids_a, c2_ikeys_a, c2_scores_a) if ik == true_ikey}
                    best_removed = (c1_true_best_rid in removed_set)
                    score_unchanged = (max(c2_true_rid_scores.values()) == c1_true_score_full) if c2_true_rid_scores else False
                else:
                    c2_rank, c2_topk, c2_cand_hit, c2_ranked = None, [], False, []
                    best_removed = (c1_true_best_rid in removed_set)
                    score_unchanged = False

            case = None
            if c1_rank is not None and c2_rank is not None:
                if not c1_cand_hit:
                    case = 5
                elif c1_rank == 1 and c2_rank == 1:
                    case = 1
                elif c1_rank == 1 and 2 <= c2_rank <= TOP_K:
                    case = 2
                elif 2 <= c1_rank <= TOP_K and (c2_rank is None or c2_rank > TOP_K):
                    case = 3
                else:
                    case = 4
            elif not c1_cand_hit:
                case = 5
            elif c2_rank is None:
                case = 3
            if no_op:
                case = None

            jsonl_rows.append(enhance(condition_row(
                rid, true_ikey, qrow, stratum, max_sim, n_retained, "C2", tau,
                list(c2_rid_set), c2_rank, c2_topk, c2_cand_hit, len(c2_rid_set),
                list(removed_set), no_op, [], case, c1_cand_hit, c1_rank),
                c2_ranked, true_ikey,
                extra={"true_in_candidates_c1": c1_cand_hit,
                       "true_in_candidates_c2": c2_cand_hit,
                       "candidate_loss": bool(c1_cand_hit and not c2_cand_hit),
                       "rank_gt_25": bool(c2_cand_hit and (c2_rank is None or c2_rank > TOP_K)),
                       "rank_1_to_25": bool(c2_cand_hit and c2_rank is not None and c2_rank <= TOP_K),
                       "best_true_evidence_removed": best_removed,
                       "score_unchanged": bool(score_unchanged)}))

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
                cm_rank, cm_topk, cm_cand_hit, cm_ranked = None, [], False, []

            jsonl_rows.append(enhance(condition_row(
                rid, true_ikey, qrow, stratum, max_sim, n_retained, "C1-matched", tau,
                list(cm_rid_set), cm_rank, cm_topk, cm_cand_hit, len(cm_rid_set),
                [], cm_no_op, fillers, None, c1_cand_hit, c1_rank),
                cm_ranked, true_ikey,
                extra={"true_in_candidates_c1": c1_cand_hit,
                       "true_in_candidates_c2": None, "candidate_loss": None,
                       "rank_gt_25": None, "rank_1_to_25": None,
                       "best_true_evidence_removed": None, "score_unchanged": None}))

        if (qi + 1) % 50 == 0 or qi == len(rids) - 1:
            print(f"  [{qi+1}/{len(rids)}] elapsed {time.time()-t0:.1f}s")

    assert ur_recomputed == manifest["population"]["n_unreachable_ur"], \
        f"UR reconciliation mismatch: recomputed {ur_recomputed}"

    with open(OUT_JSONL, "w") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {OUT_JSONL} ({len(jsonl_rows)} rows)")

    conditions = ["C1"] + [f"C2:{t}" for t in TAU_GRID] + [f"C1-matched:{t}" for t in TAU_GRID]
    cond_rows = {}
    for row in jsonl_rows:
        key = row["condition"] if row["condition"] == "C1" else f"{row['condition']}:{row['tau']}"
        cond_rows.setdefault(key, []).append(row)

    metrics = {c: compute_metrics(cond_rows[c]) for c in conditions}

    # like-for-like C1 restricted to reachable (490) for deltas
    reach_rids_for_tau = {t: {r["rid"] for r in cond_rows[f"C2:{t}"]} for t in TAU_GRID}
    c1_reach = {t: [r for r in cond_rows["C1"] if r["rid"] in reach_rids_for_tau[t]] for t in TAU_GRID}
    changed_rids = {t: {r["rid"] for r in cond_rows[f"C2:{t}"] if not r["no_op"]} for t in TAU_GRID}
    noop_rids = {t: {r["rid"] for r in cond_rows[f"C2:{t}"] if r["no_op"]} for t in TAU_GRID}

    deltas = {}
    for t in TAU_GRID:
        c1_r = c1_reach[t]
        c2_r = cond_rows[f"C2:{t}"]
        cm_r = cond_rows[f"C1-matched:{t}"]
        d = {
            "c2_minus_c1": minus(compute_metrics(c2_r), compute_metrics(c1_r)),
            "control_minus_c1": minus(compute_metrics(cm_r), compute_metrics(c1_r)),
        }
        c1_ch = [r for r in c1_r if r["rid"] in changed_rids[t]]
        c2_ch = [r for r in c2_r if not r["no_op"]]
        cm_ch = [r for r in cm_r if r["rid"] in changed_rids[t]]
        if c2_ch:
            d["changed_only_c2_minus_c1"] = minus(compute_metrics(c2_ch), compute_metrics(c1_ch))
            d["changed_only_control_minus_c1"] = minus(compute_metrics(cm_ch), compute_metrics(c1_ch))
        c1_np = [r for r in c1_r if r["rid"] in noop_rids[t]]
        c2_np = [r for r in c2_r if r["no_op"]]
        cm_np = [r for r in cm_r if r["rid"] in noop_rids[t]]
        if c2_np:
            d["noop_only_c2_minus_c1"] = minus(compute_metrics(c2_np), compute_metrics(c1_np))
            d["noop_only_control_minus_c1"] = minus(compute_metrics(cm_np), compute_metrics(c1_np))
        deltas[f"tau_{t:.2f}"] = d

    # per-stratum x condition (incl. per-stratum x tau changed/no-op where present)
    per_stratum = {}
    all_strata = STRATA_NAMES + ["UR"]
    for s in all_strata:
        ps = {}
        for c in conditions:
            sub = [r for r in cond_rows[c] if r["stratum"] == s]
            if sub:
                ps[c] = compute_metrics(sub)
                if c.startswith("C2:"):
                    ch = [r for r in sub if not r["no_op"]]
                    if ch:
                        ps[f"{c}__changed_only"] = compute_metrics(ch)
                    np_ = [r for r in sub if r["no_op"]]
                    if np_:
                        ps[f"{c}__noop_only"] = compute_metrics(np_)
        per_stratum[s] = ps

    # intervention proof per tau
    inter_by_tau = {}
    for t in TAU_GRID:
        inter = [r for r in cond_rows[f"C2:{t}"] if not r["no_op"]]
        case_dist = Counter(str(r["transition_case"]) for r in inter)
        inter_by_tau[f"tau_{t:.2f}"] = {
            "n_C2_runs": sum(1 for r in cond_rows[f"C2:{t}"]),
            "n_genuine_k1": len(inter),
            "n_noop_k0": sum(1 for r in cond_rows[f"C2:{t}"] if r["no_op"]),
            "transition_case_distribution": {k: v for k, v in sorted(case_dist.items())},
            "median_k": round(float(np.median([r["k_removed"] for r in cond_rows[f"C2:{t}"]])), 1),
        }

    all_inter = [r for r in jsonl_rows if r["condition"] == "C2" and not r["no_op"]]
    k_dist = Counter(r["k_removed"] for r in all_inter)

    # control power analysis (per tau). C1-matched rows store fillers (not removed rids),
    # so the effective number of removed spectra is len(filler_rids); no_op is the C2 k==0 flag.
    control_power = {}
    for t in TAU_GRID:
        rows = cond_rows[f"C1-matched:{t}"]
        changes = [r for r in rows if r["rank"] is not None and r["rank"] != r["c1_rank"]]
        cset_changes = [r for r in rows if not r["no_op"]]
        best_removed = [r for r in cond_rows[f"C2:{t}"] if r["best_true_evidence_removed"]]
        k_rank = Counter()
        k_n = Counter()
        for r in rows:
            kk = len(r["filler_rids"])
            k_n[kk] += 1
            if r["rank"] is not None and r["rank"] != r["c1_rank"]:
                k_rank[kk] += 1
        control_power[f"tau_{t:.2f}"] = {
            "n_queries": len(rows),
            "rank_changes_vs_C1": len(changes),
            "candidate_set_changes": len(cset_changes),
            "changed_true_rank": len(changes),
            "c1_true_rank_histogram": dict(sorted(Counter(
                (r["c1_rank"] if r["c1_rank"] is not None else "out_of_candidates")
                for r in rows).items(), key=lambda kv: (isinstance(kv[0], int), kv[0]))),
            "k_distribution": dict(sorted(k_n.items())),
            "k_vs_rank_change": {k: {"n": k_n[k], "rank_changes": k_rank.get(k, 0)} for k in sorted(k_n)},
            "best_true_evidence_removed": len(best_removed),
            "no_effect_another_true_remained": len([r for r in all_inter if r["tau"] == t and r["score_unchanged"]]),
        }

    # candidate pool stats
    nc = [r["n_candidates"] for r in cond_rows["C1"]]
    pool_stats = {
        "C1": {"median": round(float(np.median(nc)), 1), "min": int(min(nc)),
               "max": int(max(nc)), "mean": round(float(np.mean(nc)), 1), "n_zero": sum(1 for x in nc if x == 0)},
    }
    for t in TAU_GRID:
        for c in (f"C2:{t}", f"C1-matched:{t}"):
            v = [r["n_candidates"] for r in cond_rows[c]]
            pool_stats[c] = {"median": round(float(np.median(v)), 1), "min": int(min(v)),
                             "max": int(max(v)), "mean": round(float(np.mean(v)), 1),
                             "n_zero": sum(1 for x in v if x == 0)}

    # failure decomposition per tau (candidate-loss vs ranking-loss vs no-op)
    failure_decomp = {}
    for t in TAU_GRID:
        rows = cond_rows[f"C2:{t}"]
        A = sum(1 for r in rows if r["candidate_loss"])
        B = sum(1 for r in rows if r["rank_gt_25"])
        Cc = sum(1 for r in rows if r["rank_1_to_25"])
        D = sum(1 for r in rows if r["no_op"])
        C_c1_1 = sum(1 for r in rows if r["c1_rank"] is not None and r["c1_rank"] == 1)
        failure_decomp[f"tau_{t:.2f}"] = {
            "A_candidate_loss_true_lost_from_C2": A,
            "B_candidate_present_rank_gt_25": B,
            "C_candidate_present_rank_1_to_25": Cc,
            "D_noop_k0": D,
            "n_c2_runs": len(rows),
            "n_c1_rank1_before": C_c1_1,
        }

    all_pass = all(len(v) == 0 for v in checks.values())
    check_counts = {k: len(v) for k, v in checks.items()}
    failed_checks = {k: [x for x in v] for k, v in checks.items() if v}
    passed_checks = [k for k, v in checks.items() if not v]

    out = {
        "generated_by": "research/scripts/exp007_full_run.py",
        "status": "COMPLETE" if all_pass else "FAILED (stop: do not proceed)",
        "approval": APPROVAL,
        "seed": SEED,
        "tau_grid": TAU_GRID,
        "runtime_seconds": round(time.time() - t_start, 1),
        "n_queries": 600,
        "n_reachable_c2": 600 - manifest["population"]["n_unreachable_ur"],
        "n_unreachable_ur": manifest["population"]["n_unreachable_ur"],
        "jsonl_rows": len(jsonl_rows),
        "checks": {k: v for k, v in checks.items()},
        "check_counts": check_counts,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "intervention_proof_total": {
            "n_C2_runs_total": sum(1 for r in jsonl_rows if r["condition"] == "C2"),
            "n_genuine_k1": len(all_inter),
            "n_noop_k0": sum(1 for r in jsonl_rows if r["condition"] == "C2" and r["no_op"]),
            "k_distribution": {k: v for k, v in sorted(k_dist.items())},
        },
        "intervention_by_tau": inter_by_tau,
        "failure_decomposition_by_tau": failure_decomp,
        "metrics": metrics,
        "deltas_like_for_like": deltas,
        "per_stratum": per_stratum,
        "control_power": control_power,
        "candidate_pool_stats": pool_stats,
        "reproducibility_note": ("Single full run (per authorization). Determinism: threads=1, fixed seeds, "
                                 "same machinery as the 3x byte-identical smoke runs. Replay of the 30 smoke "
                                 "queries is produced by research/scripts/exp007_smoke_replay.py."),
        "environment": environment(),
    }
    if not all_pass:
        out["stop_reason"] = "Non-empty check buckets: full-run STOP per fail-closed rule."

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    print("\n" + "=" * 60)
    print("EXP-007 FULL RUN SUMMARY")
    print("=" * 60)
    print(f"Queries: 600 (reachable C2: {out['n_reachable_c2']}, UR: {out['n_unreachable_ur']})  seed={SEED}")
    print(f"Checks pass: {all_pass}  " + "  ".join(f"{k}={len(v)}" for k, v in checks.items()))
    if not all_pass:
        print("STOPPED: full-run fail-closed; do not proceed to reporting/conclusions.")
        raise SystemExit(2)
    for cond in ["C1"] + [f"C2:{t}" for t in TAU_GRID] + [f"C1-matched:{t}" for t in TAU_GRID]:
        m = metrics[cond]
        if m:
            print(f"  [{cond:16s}] n={m['n']:4d} candGen={m['candidate_gen_recall']:.3f} "
                  f"R1={m['recall@1']:.3f} R5={m['recall@5']:.3f} R25={m['recall@25']:.3f} "
                  f"MRR25={m['mrr@25']:.3f} [{m['mrr@25_ci95'][0]:.3f},{m['mrr@25_ci95'][1]:.3f}]")
    ip = out["intervention_proof_total"]
    print(f"Intervention total: {ip['n_C2_runs_total']} C2 runs, {ip['n_genuine_k1']} genuine (k>=1), "
          f"{ip['n_noop_k0']} no-ops (k=0)")


if __name__ == "__main__":
    main()
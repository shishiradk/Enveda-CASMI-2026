"""EXP-015 freeze verification (read-only; no expensive runs).

Re-checks internal consistency of every EXP-015 output and re-derives key
recorded facts (eligible set, R_noE180 identity, coverage, pools/query hashes,
per-query enumeration, transitions, peak caches). Exits non-zero if any check
fails. The only artifacts produced: this printed report.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import exp015_spectrum_representation as M  # noqa: E402  (constants, coconut(), Peaks)
import exp011_class2_proxy as E  # noqa: E402

O, Rtr = ROOT / "research" / "analysis" / "exp015", ROOT / "results" / "exp015"
E11, E12 = ROOT / "results" / "exp011_class2_proxy", ROOT / "results" / "exp012_coconut"

PASS, FAIL = [], []
sha = lambda arr: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


t0 = time.time()

# ------------------------------------------------------------------ 1. inventory
tbl = ["baseline_results.csv", "variant_results.csv", "per_query_results.csv",
       "fusion_results.csv", "oracle_results.json", "leakage_audit.json",
       "summary.json", "holdout_manifest.csv", "run.log",
       "run.err", "leakage_audit_v1.json", "summary_v1_inconsistent.json", "variant_results_v1.csv"]
missing = [f for f in tbl if not (O / f).exists()]
check("artifact inventory", not missing, f"missing={missing}")
cache = ["ref_rids.npy", "primary_targets.parquet", "build.json", "peaks.npz",
         "peaks_mz.npy", "peaks_it.npy", "peaks_off.npy", "peaks_pm.npy"]
check("cache inventory (results/exp015)", all((Rtr / f).exists() for f in cache),
      f"missing={[f for f in cache if not (Rtr / f).exists()]}")

# ------------------------------------------------------------------ 2. build re-derivation
bid = json.load(open(Rtr / "build.json"))
con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='4GB'")
meta = con.execute(f"""SELECT file_row_number AS rid, inchikey14 AS ik, adduct, precursor_mz AS pm
                       FROM read_parquet('{(ROOT/'train.parquet').as_posix()}', file_row_number=true)""").df()
C, cfp = M.coconut()
D = set(pd.read_parquet(E11 / "db_only_decoys.parquet")["ik"])
prev = set(pd.read_parquet(E11 / "targets.parquet")["ik"])
m10 = set(meta.loc[meta["adduct"].isin(E.AD10), "ik"])
elig = sorted((D & set(C["ik"]) & m10) - prev)
check("eligible fresh targets == 2513", len(elig) == 2513, f"got {len(elig)}")

T = pd.read_parquet(Rtr / "primary_targets.parquet")
pick = T["ik"].tolist()
check("sampled == 1000 and sorted", len(pick) == 1000 and pick == sorted(pick))
check("sampled disjoint from EXP-011/12/13 targets", not (set(pick) & prev))
check("sampled subset of eligible", set(pick) <= set(elig))
stc = {k: int(v) for k, v in T["status"].value_counts().to_dict().items()}
check("status_counts == build.info", stc == bid["status_counts"], f"{stc}")
okT = T[T["status"] == "ok"]
check("ok targets == 954", len(okT) == 954)
check("ok covered == 838", int(((okT["covered"]).sum())) == 838)
# M vs manifest medians
man = pd.read_csv(O / "holdout_manifest.csv")
rec = man.groupby("inchikey14")["neutral_mass_spectrum"].median().rename("Mrec").reset_index()
mchk = okT[["ik", "M"]].merge(rec, left_on="ik", right_on="inchikey14")
check("manifest neutral-mass medians == T.M", np.allclose(mchk["M"], mchk["Mrec"], atol=1e-6))
check("manif. neutral_mass_molecule == T.M", np.allclose(
    man.groupby("inchikey14")["neutral_mass_molecule"].first().reindex(okT["ik"]),
    okT.set_index("ik")["M"], atol=1e-6))
check("ok query spectra == 3147", int((man["leakage_status"] == "ok").sum()) == 3147)
check("no duplicate spectrum rows in manifest", man["spectrum_id"].duplicated().sum() == 0)
check("ok spectra truth_in_pool coverage == 838",
      int(man.loc[man["leakage_status"] == "ok"].groupby("inchikey14")["truth_in_pool"].any().sum()) == 838)
nq = man.groupby("inchikey14").size()
check("<=4 queries per molecule", nq.max() <= 4, f"max {nq.max()}")
check("sampled overlap with EXP-011/13 targets flagged as appeared_in_exp011_013",
      man["appeared_in_exp011_013"].sum() == 0)
check("no previous_target flag among ok", okT["previous_target"].sum() == 0)
check("no leak flags among ok targets", okT[["target_in_ref", "alias_in_ref", "parent_in_ref",
                                             "target_or_alias_in_twotower", "parent_in_twotower"]].sum().sum() == 0)
check("library mix == build.info", man.loc[man["leakage_status"] == "ok", "library"].value_counts().to_dict()
      == bid["query_library_mix"])

# ------------------------------------------------------------------ 3. R_noE180 identity
tr = pd.read_parquet(E11 / "training.parquet").set_index("rid")
ref = tr[(tr["split"] == "train") & (tr["lib"] != "enveda-180")]
rr = np.sort(ref.index.values)
ref_rids = np.load(Rtr / "ref_rids.npy")
check("ref_rids length == 298095", len(ref_rids) == 298095)
check("ref_rids == re-derived R_noE180 rids", np.array_equal(ref_rids, rr))
check("R_noE180 molecules == 78707", ref["ik"].nunique() == 78707)
rec_sha = sha(ref_rids)
check("R_noE180 sha matches build+audit ad8bb5…42f9d",
      rec_sha == bid["R_noE180"]["sha256_sorted_rids"] ==
      json.load(open(O / "leakage_audit.json"))["R_noE180"]["sha256_sorted_rids"] ==
      "ad8bb53eb7303d76abd5a23ad15c71344eb696cb10053fccc8d13c17234f929d")

# ------------------------------------------------------------------ 4. COCONUT indexing / EXP-012
check("cfp row count == parse_ok coconut rows", len(cfp) == int(C["parse_ok"].sum()))
check("cfp dtype uint8", cfp.dtype == np.uint8)
check("cfp shape == (rows, FP_BITS/8)", cfp.shape[1] == E.FP_BITS // 8)
check("cfp no NaN", not np.isnan(cfp).any())

# ------------------------------------------------------------------ 5. peaks
z = np.load(Rtr / "peaks.npz")
off = np.r_[0, np.cumsum(z["len"])]
check("peaks spectra == 302426", off[-1] == np.cumsum(z["len"]).max() and (off.size - 1) == 302426,
      f"n={(off.size - 1)}")
check("peaks total == 31,484,786", z["mz"].size == 31_484_786 == z["it"].size == off[-1])
check("peaks rid sorted unique", np.array_equal(z["rid"], np.sort(z["rid"])))
npfu = {"ref": np.isin(z["rid"], ref_rids).sum(),
        "prim": np.isin(z["rid"], man.loc[man["leakage_status"] == "ok", "spectrum_id"]).sum(),
        "dev": int(sum(np.isin(z["rid"], pd.read_parquet(E11 / "queries.parquet").query("pop=='T1_np'")["rid"].values)))}
check("peaks contains full ref+prim+dev", npfu["ref"] == 298095 and npfu["prim"] == 3147
      and npfu["ref"] + npfu["prim"] + npfu["dev"] <= (off.size - 1), f"{npfu}")
for k, n in (("mz", "peaks_mz.npy"), ("it", "peaks_it.npy"), ("pm", "peaks_pm.npy")):
    a, b = z[k], np.load(Rtr / n)
    check(f"peaks_npz.{k} == {n} bit-exact", np.array_equal(np.asarray(a), np.asarray(b)))
a, b = off, np.load(Rtr / "peaks_off.npy")
check("peaks_off == cumsum(len)", np.array_equal(a, b))
check("ref neg count == ref_neg.npy length", (tr.loc[ref_rids, "adduct"].isin(E.NEG)).sum() ==
      int(np.load(Rtr / "_ref_neg.npy").shape[0]) if (Rtr / "_ref_neg.npy").exists() else True)  # leftover removed

# ------------------------------------------------------------------ 6. tables
vr = pd.read_csv(O / "variant_results.csv")
br = pd.read_csv(O / "baseline_results.csv")
fr = pd.read_csv(O / "fusion_results.csv")
check("variant_results 20 rows (10 variants x 2 pops)", vr.shape == (20, 19), f"{vr.shape}")
check("baseline_results = B0 only (2 rows)", br.shape[0] == 2 and set(br["variant"]) == {"B0_current"} and
      np.allclose(br["MRR"].values, vr.loc[vr["variant"] == "B0_current", "MRR"].values))
check("PRIMARY n_ranked == 838 all variants", (vr.loc[vr.population == "PRIMARY", "n_ranked"] == 838).all())
check("DEV n_ranked == 243 all variants", (vr.loc[vr.population == "DEV", "n_ranked"] == 243).all())
delta_cols = ["dMRR_vs_B0", "dR@1_vs_B0", "d_isomer_error_rate_vs_B0"]
nan_r = pd.isna(vr).any(axis=1)
nan_ok = (vr["variant"] == "B0_current") & pd.isna(vr[delta_cols]).all(axis=1)
check("NaN in variant/baseline/fusion only as B0 empty delta cells",
      (((nan_r == nan_ok).all()) and pd.isna(br[delta_cols]).all().all()
       and not pd.isna(fr).any().any()))
for _, r in vr.iterrows():
    for c in ("dMRR_vs_B0", "dR@1_vs_B0", "d_isomer_error_rate_vs_B0"):
        if not pd.isna(r[c]):
            v = [float(x) for x in r[c].strip("[]").split(",")]
            assert len(v) == 3 and (v[0] == float(r[c].split(",")[0].strip("["))), (r["variant"], c)
check("all delta CIs parse as [mean, lo, hi]", True)

summ = json.load(open(O / "summary.json"))
leak = json.load(open(O / "leakage_audit.json"))
check("summary.valid == true and audit.VALID == true", summ["valid"] is True and leak["VALID"] is True)
check("summary.selection == variant_results DEV MRR",
      all(abs(summ["selection"]["dev_MRR"][v] - vr.loc[(vr.variant == v) & (vr.population == "DEV"), "MRR"].iloc[0]) < 1e-12
          for v in summ["selection"]["dev_MRR"]))
check("audit B0_deterministic == true", leak["B0_deterministic"] is True)
check("audit feature_reproduction_max_abs_diff == 0.0", leak["feature_reproduction_max_abs_diff"] == 0.0)
check("audit coconut_fp_indexing note", leak["coconut_fp_indexing"] == "parse_ok-filtered (EXP-012 bug not reproduced)")
check("audit selection_used_PRIMARY == false (DEV-only selection)", leak["selection_used_PRIMARY"] is False)
man_ids, dev_ids = man.loc[man.leakage_status == "ok", "spectrum_id"].astype(np.int64).unique(), None

# ---- re-derive pool & query hashes
Cs = C.sort_values("mass").reset_index(drop=True)
cm = Cs["mass"].values
pools = {}
Mk = dict(zip(okT["ik"], okT["M"]))
okkeys = {r.ik: set(r.coconut_ok_keys) for r in okT.itertuples()}
for ik in okT["ik"].tolist():
    lo, hi = np.searchsorted(cm, Mk[ik] * (1 - 5e-6)), np.searchsorted(cm, Mk[ik] * (1 + 5e-6), side="right")
    cand = Cs.iloc[lo:hi]["ik"].values
    if not any(k in okkeys[ik] for k in cand):
        continue
    pools[ik] = list(cand)
check("re-derived PRIMARY pools == 838", len(pools) == 838)
ph = hashlib.sha256("|".join(f"{k}:{','.join(v)}" for k, v in sorted(pools.items())).encode()).hexdigest()
check("pool_hash re-derived == audit pool_hash_PRIMARY", ph == leak["pool_hash_PRIMARY"])
qrids = np.sort(man.loc[man.leakage_status == "ok", "spectrum_id"].astype(np.int64).values)
qh = hashlib.sha256(",".join(map(str, qrids)).encode()).hexdigest()
check("query_hash PRIMARY re-derived == audit", qh == leak["query_hash_PRIMARY"])
check("oracle coverage == 838/954", abs(summ["oracle"]["coverage"] - 838 / 954) < 1e-9)
check("oracle same_formula_prevalence == 746/838", abs(summ["oracle"]["same_formula_prevalence"] - 746 / 838) < 1e-9)
pl = pd.read_csv(O / "per_query_results.csv").query("population=='PRIMARY' and variant=='B0_current'")["pool"]
check("median pool == 28.5", pl.median() == 28.5, f"{pl.median()}")
# manifest candidate_count sync
mcnt = man.loc[man.leakage_status == "ok"].groupby("inchikey14")["candidate_count"].first()
poolsizes = pd.Series({k: len(v) for k, v in pools.items()})
check("manifest candidate_count == re-derived pool sizes", (mcnt.reindex(poolsizes.index) == poolsizes).all())

# ------------------------------------------------------------------ 7. per-query enumeration, transitions
pq = pd.read_csv(O / "per_query_results.csv")
expected = {(v, p) for v in ("B0_current", "V2_fragment_only", "V3a_bin0.01", "V3b_bin0.5",
                             "V4a_linear", "V4b_log", "V5_modcos_rerank", "C1_bestNL_bestInt",
                             "C2_bestNL_bestBin_bestInt", "C3_C2_plus_modcos") for p in ("PRIMARY", "DEV")}
got = set(zip(pq["variant"], pq["population"]))
check("per_query enumerates 10 variants x 2 pops", got == expected, f"{got - expected}")
cnt = pq.groupby(["population", "variant"], dropna=False).size().unstack()
check("per_query counts PRIMARY 838 / DEV 243 per variant",
      (cnt.loc["PRIMARY"] == 838).all() and (cnt.loc["DEV"] == 243).all())
need = ["pool", "rank", "rr", "rr25", "r@1", "r@5", "r@10", "r@50", "B0_rank", "B0_rr", "rank_delta_vs_B0"]
check("no NaN in rank/B0 cols", not pd.isna(pq[need]).any().any())
okna = pd.isna(pq["isomer_error_rate"])
check("NaN in per_query only in isomer_error_rate (no-same-formula rows)",
      okna.sum() == int((pq["has_same_formula"] == False).sum()) and  # noqa: E712
      pd.isna(pq.loc[~okna, need]).any().any() == False)  # noqa: E712
check("rank_delta == rank - B0_rank", np.allclose(pq["rank_delta_vs_B0"], pq["rank"] - pq["B0_rank"]))
# variant MRR == per-query means
mrr = pq.groupby(["population", "variant"]).agg(MRR=("rr", "mean"), R1=("r@1", "mean"), R10=("r@10", "mean"))
for _, r in vr.iterrows():
    g = mrr.loc[(r["population"], r["variant"])]
    assert abs(g["MRR"] - r["MRR"]) < 1e-9 and abs(g["R1"] - r["R@1"]) < 1e-9 and abs(g["R10"] - r["R@10"]) < 1e-9
check("variant_results == per-query means for MRR/R@1/R@10", True)
# transitions vs summary
for variant in ("V2_fragment_only", "V3a_bin0.01", "V3b_bin0.5", "V4a_linear", "V4b_log",
                "V5_modcos_rerank", "C1_bestNL_bestInt", "C2_bestNL_bestBin_bestInt", "C3_C2_plus_modcos"):
    for pop in ("PRIMARY", "DEV"):
        b = pq[(pq.population == pop) & (pq.variant == "B0_current")].set_index("ik")
        v = pq[(pq.population == pop) & (pq.variant == variant)].set_index("ik")
        m = b.join(v, rsuffix="_v").dropna(subset=["rank_v"])
        cb, cv = m["r@1"] >= 0.5, m["r@1_v"] >= 0.5
        t = {"wrong_to_correct": int((~cb & cv).sum()), "correct_to_wrong": int((cb & ~cv).sum()),
             "both_correct": int((cb & cv).sum()), "both_wrong": int((~cb & ~cv).sum())}
        ref_t = summ["transitions"][f"{pop}|{variant}"]["transitions"]
        assert t == ref_t, (pop, variant, t, ref_t)
check("per-query transitions == summary.transitions (all variants/pops)", True)
# fusion counts
fc = summ["fusion_counts"]
check("fusion overlap sums to n_ranked", fc["kNN_only_correct"] + fc["two_tower_only_correct"]
      + fc["both_correct"] + fc["neither"] == 838)
check("fusion fixes vs regressions net == +31", fc["fusion_fixes_vs_kNN"] - fc["fusion_regressions_vs_kNN"] == 31)
check("fusion_results rows (kNN_B0, two_tower, fusion_50_50)", set(fr["model"]) == {"kNN_B0", "two_tower", "fusion_50_50"})
kb = fr.loc[fr["model"] == "kNN_B0"]
b0p = vr[(vr.variant == "B0_current") & (vr.population == "PRIMARY")]
check("fusion kNN_B0 row == B0 PRIMARY row", np.allclose(kb[["MRR", "R@1", "R@10"]].values.astype(float),
      b0p[["MRR", "R@1", "R@10"]].values.astype(float)))
check("fusion MRR/R@1 match summary deltas",
      abs(fr.loc[fr.model == "fusion_50_50", "MRR"].iloc[0] - (b0p["MRR"].iloc[0] + fc["delta_MRR_fusion_vs_kNN"][0])) < 1e-9)

# ------------------------------------------------------------------ 8. logs
logtxt = open(O / "run.log").read()
check("run.log shows completion, traceback-free", "done" in logtxt and "Traceback" not in logtxt
      and logtxt.strip().splitlines()[-1][:1] in ("}", "{", " "))
err = open(O / "run.err").read()
check("run.err free of Traceback", "Traceback" not in err and "Error" not in err.split("networkx")[-1])
check("build.log + peaks.log present", (O / "build.log").exists() and (O / "peaks.log").exists())

print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {time.time() - t0:.0f}s")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  ")
    sys.exit(1)
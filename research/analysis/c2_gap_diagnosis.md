# C2 gap diagnosis: the PubChem re-ranker

**Bottom line:** the headline is the **channel** MRR (PC 0.41 -> 0.68), but the metric is MRR@25 on the **merged** 25-item list, where the gain is ~6x smaller — the E6 defect is the `m1_alt` interleave, not the PubChem channel.

## 1. Experiment x bucket (FACT, `gate.json`; weights SV .146 / S1 .21 / PC .111 / S2 0)

| experiment | SV | S1 | S2 | PC | proxy | d(proxy) | dSV | dS1 | dPC |
|---|---|---|---|---|---|---|---|---|---|
| E6 as shipped (real E6 ranks) | 0.9211 | 0.8584 | 0.6338 | 0.3092 | **0.3491** | - | - | - | - |
| `ho1+struct` re-rank only | 0.9214 | 0.8665 | 0.6649 | 0.3291 | **0.3530** | +0.0039 | +0.0003 | +0.0081 | +0.0199 |
| `ho1+struct` + cross-fit gate | 0.9047 | 0.8992 | 0.7294 | 0.4161 | **0.3671** | +0.0180 | **-0.0164** | +0.0408 | +0.1069 |
| `ho1+struct+pop` re-rank only | 0.9216 | 0.8721 | 0.6935 | 0.3579 | **0.3574** | +0.0083 | +0.0005 | +0.0137 | +0.0487 |
| `ho1+struct+pop` + cross-fit gate | 0.8914 | 0.9405 | 0.8421 | 0.5483 | **0.3885** | +0.0394 | **-0.0297** | +0.0821 | +0.2391 |

**FACT.** Both gated rows regress SV, and the gate supplies 4/5 of the best number (.0311 of .0394 for
`ho1+struct+pop`) — that is the whole trade.
**FACT.** The merge absorbs ~80% of every channel gain, so **channel MRR is the wrong quantity to optimise** (PC, 213 mol, `rerank.json`/`ablate.json`): PC channel 0.4074 -> 0.6801 for `ho1+struct+pop`, but PC merged only 0.3030 -> 0.3611 (`__cf` = truth's popularity replaced: 0.3943 channel / 0.3013 merged). The PC channel alone is 0.4074 while E6 merged is 0.3092 — **`m1_alt` demotes the channel**; `merge.json` pattern `EP` lifts PC 0.3092 -> 0.3512 for only +0.0016 proxy, and `pop_only` PC channel 0.4290 collapses to 0.0533 (`__cf`).

## 2. Leakage audit (the important part)

| check | verdict | evidence |
|---|---|---|
| CV grouped by molecule? | **Yes** | `c2gap_rerank.py:99-100`, `c2gap_gate.py:61-62` fold over unique `(scen,mid)`/`mid`; a molecule's ~100 candidates never straddle a split |
| `is_truth` / truth-derived feature? | **No** | `is_truth` is the label only (`:120`) plus the counterfactual donor pick (`:109-113`), absent from every list in `SETS` (`:67-75`); truths deeper than rank 100 are dropped so rank 101 cannot leak (`:45-47`); `ppm` uses the query target mass (`:59-62`) |
| Trained on the buckets it scores? | **No** | PC+S2 only (`:144`, `c2gap_gate.py:68`); SV never trained on (`c2gap_gate.py:7`); `is_truth` never shipped, it lives only in `results/c2gap/cand.parquet` |
| Gate threshold honest? | **Yes** | cross-fit 2-fold by molecule (`c2gap_gate.py:113-126`); the in-sample grid is 2-4x higher, ignore it |
| Scoring nets hold out the scored molecules? | **Nearly** | 2/206 PC, 8/249 S2 leak — below |
| Any CFT score feature? | **No CFT at all** | addendum answered: `grep -i cft research/scripts/c2gap_*.py` -> 0 hits, so leaky `models/cft_kaggle` is unused |

**FACT (residual net leak).** Scoring nets are the held-out **fingerprint** nets `models/fp_ho1_akriti` (`e6_pc_channel.py:3`) and `models/fp_ho2_akriti` (`c2gap_ho2.py:32`), holdout per `scratch_wf/ho2_overlap.py:14-16` (ho1 585 keys, ho2 1,185 = 585 + 600 S4; `train.parquet` holds 275,810 distinct `ik`). Testing every truth `ik` in `cand.parquet` against those sets *and* `train.parquet`: **PC 2/206, S2 8/249 truth keys were outside the holdout yet in `train.parquet`**, so the nets trained on them. Bounded by 2/213 = 0.009 PC MRR, and the 8 S2 keys carry proxy weight 0, so the estimate is unaffected — but "leak-free" is not literally true.

**FACT (the real risk: a feature that does not transfer).** `pop.json`, identical code and feature, 3 buckets:

| bucket | what it is | truth `n_sid` q1/med/q3 | isomer med | `pop_only_mrr25` sid | truth top 10% |
|---|---|---|---|---|---|
| S2 (246) | enveda-np-examples, "closest library to the test set" | 252 / 478 / 861 | 2 | **0.7931** | 1.00 |
| PC (213) | GNPS-type PubChem-only | 5 / 28 / 75 | 2 | **0.3332** | 0.75 |
| S4PC (300) | enveda-180 timsTOF (`c2gap_pop.py:5-6`) | 2 / 3 / 5 | 2 | **0.0035** | 0.20 |

**INFERENCE.** Popularity's power tracks NP-likeness: 0.80 MRR on the NP-like bucket, 0.003 on the non-NP one. Not leakage — allowed public PubChem data — but a bet that hidden-test compounds are popular relative to their own isomers. `c3_s4_scenarios.md:46-51` says the hidden test *is* NP-like and S12 is the chemistry reference, so the bet is reasonable; the `__cf` collapse shows the model has no fallback if it loses.

## 3. Feature eligibility (step 3) — every feature, with its source

| feature | source | eligible |
|---|---|---|
| `score`, `score_gap/z/rk` | ho1 fp net (`fp @ zlog`), `models/fp_ho1_akriti` -> `results/c3/e6_work/scores_5.0ppm.npy` | yes (train.parquet-only) |
| `s_ho2`, `ens*` | ho2 nets `models/fp_ho2_akriti` -> `results/c2gap/ho2_scores.parquet` | yes, but drop (below) |
| `lsid`, `lpmid`, `lcid`, `*_rel`, `ppm`, `abs_ppm` | `external/pubchem/pubchem_rows_pop.parquet`, `sum(n_sid)/sum(n_pmid)/count(*)` per `ik`, `mass` vs the query target mass (`c2gap_extract.py:59-73`) | yes (public PubChem) |
| `np`, `np_rel`, `rings arom fsp3 nstereo multi charge isotope hbd logp` | RDKit on candidate SMILES; NP via `npscorer.readNPModel()` (`c2gap_rerank.py:28`) | yes |
| — | any CFT / NIST / FRIGID / third-party Kaggle data | **absent** |

**FACT.** `ens_window.json`: PC window MRR ho1 0.4155 / ho2 0.4505 / mean 0.4410, (mean - ho1) CI **[+0.0256, -0.0003, +0.0515]** includes 0 and `ens+struct+pop` (0.6560) loses to `ho1+struct+pop` (0.6801): **drop ho2.**

## 4. Recommendation

**Ship `ho1+struct+pop` re-rank, E6 merge pattern unchanged, NO gate** — the only variant where no bucket regresses (SV 0.9216 vs 0.9211, on a bucket never trained on). The gate buys +0.0311 proxy but costs **-0.0302 SV** (0.9216 -> 0.8914), needs a fragile `share > T` threshold, and its all-rows variant is catastrophic (`sim_rerank_PE_all`: SV -> 0.6955).
**Expected LB gain +0.004** (INFERENCE: +0.0083 proxy with the mandated 2x haircut, E6 having delivered half its prediction); ~0 if the popularity bet loses, several times that if it wins. Do not budget the gate's +0.020.
**Go/no-go before submitting (180 mol):** re-score SV with the **full-data** nets, confirm re-ranked SV MRR >= 0.9211 — **FACT** the model was fit on held-out-net scores but E6 deploys `e6net_merged_full.pt` / `e6net_single_full.pt`, a real train/deploy shift in `score_gap/z/rk`.

## 5. E8 kernel steps (E8 = E7 + re-ranked PubChem channel)

1. Fit the lambdarank booster on all bench PC+S2 molecules with `SETS["ho1+struct+pop"]` (2 workers, 200 rounds, `lambdarank_truncation_level=25`); ship as text.
2. Upload a per-`ik` popularity map `(ik, n_sid, n_pmid, n_cid)` trimmed from `pubchem_rows_pop.parquet` (2.6 GB, 101M rows, ~74.8M distinct `ik` approx); the test window is unknown at build time so the whole map must ship (`results/kaggle_v3_proxy/pop_S{12,3}.parquet` are the bench-only precedent, per query, 17-23 MB).
3. In E7's PubChem stage score the window with the **full-data** nets, order the top-60 by the re-ranker, then keep the existing metric-key de-dup and 40-keep so E7's merge cell is untouched.
4. Wrap the re-rank in `try/except`; on failure keep the un-re-ranked E6 list (same contract as E7's C3 stage).
5. Re-run E7's visible check (top-3 identical 400/400, 25 unique per row) — the re-ranker must not touch ranks 1-3.

**Not recommended:** the PE gate; the ho2/ens ensemble; changing the merge pattern (out of scope — `EP` moves ranks 1-3 and needs its own visible check).

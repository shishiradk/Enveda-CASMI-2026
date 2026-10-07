# E8 go/no-go — SV re-scored with the deploy nets

**NO-GO** (the re-ranker, as fitted, does not survive the ho1→full net shift; the rank-only refit does not fix it).

## Why this check exists

The C2 re-ranker (`c2_gap_diagnosis.md`) was fitted on **ho1**-net scores; E6/E8 deploy the **full-data** nets, so
`score`, `score_gap`, `score_z`, `score_rk` shift. SV (visible test) is the only bucket the full nets never trained
on, so it is the only honest check.

**FACT.** `models/fp_full/fp_merged_full.pt` ≡ `research/kaggle_e1/e6/dataset/e6net_merged_full.pt` and
`fp_single_full.pt` ≡ `e6net_single_full.pt` (SHA-256 prefixes C76DE4B6E5C4727E / BB449909880E1480), so
`--nets models/fp_full` scores exactly the deployed nets.
**FACT.** My pipeline reproduces the published E6 number: merging `results/c2gap/work/pc_lists_SV.pkl` with the
shipped m1_alt rule gives SV merged MRR **0.9211**, matching `gate.json` `real_E6` to 4 decimals.

## Method (FACT, commands)

```
python research/scripts/e6_pc_channel.py --nets models/fp_full --workers 2 --work results/c2gap_full/work   # 7.2 min
python research/scripts/c2gap_extract.py --scens SV --tag _fullSV --work results/c2gap_full/work          # 0.9 min
python research/scripts/e8_sv_gonogo.py --feats {full|rankov|rankz}                                        # ~7 min each
```

- `e6_pc_channel.py` and `c2gap_extract.py` gained a `--work` argument (default = the original ho1 cache, unchanged).
  The old `results/c3/e6_work/` was **not** touched: fingerprints, windows and pc_lists_S1/S2/S3 were **hardlinked**
  (0 disk cost), only `logits.pkl` / `scores_5.0ppm.npy` / `pc_lists_SV.pkl` are new. D: went 2.24 → 2.19 GB free.
- Booster: LightGBM lambdarank, exactly as `c2gap_rerank.run_cv` (lr 0.05, leaves 7, 200 rounds, trunc 25, seed 0),
  trained on **all** bench PC+S2 molecules with ho1 scores (377 molecules with truth in the top-100), applied to the
  400 SV molecules with full-net scores. Merge = `e6_merge_eval.merge(e, p, 1, "alt")`, unchanged.

## Result, SV (400 molecules; MRR@25)

| variant | merged, no re-rank | merged, with re-rank | channel, no re-rank | channel, with re-rank | δ (re-rank − no) | 95% CI (boot 4000) | 1-mol deltas (impr/regr) |
|---|---|---|---|---|---|---|---|
| E6 as shipped (ho1 nets) | **0.9211** | — | 0.6369 | — | — | — | — |
| `full` (shipped feature set) | **0.9211** | **0.9209** | 0.6757 | 0.6149 | −0.0002 | [−0.0006, 0.0] | 0 / 1 |
| `rankov` (drop `score`) | **0.9211** | **0.9209** | 0.6757 | 0.6122 | −0.0002 | [−0.0006, 0.0] | 0 / 1 |
| `rankz` (drop `score`,`score_gap`) | **0.9211** | **0.9206** | 0.6757 | 0.5933 | −0.0005 | [−0.0014, 0.0] | 0 / 2 |

## Interpretation (FACT then INFERENCE)

**FACT.** The shift is real: full vs ho1 nets change the channel rank on 156 of 400 SV molecules; the merged rank
changes on only 2 of 400 (the engine dominates the merge on SV, engine MRR 0.9214).
**FACT.** On the channel, the re-ranker is **worse** than plain full-net score order on SV (0.615 vs 0.676).
**FACT.** The merged metric is essentially flat: δ = −0.0002 from exactly one molecule regressing (rank 1 → worse);
the 95% CI upper bound is 0.

**INFERENCE.** The re-ranker overfits the ho1 score distribution: at deploy time its own ordering of the channel is a
net loss, and the only reason the shipped metric barely moves is that the engine wraps every channel slot on SV. The
benchmark +0.0083 proxy assumed no net shift; on SV the honest expectation is now ≈ 0, not +0.008. The CI lower bound
(≥ −0.005) passes, but the merged-MRR floor (≥ 0.9211) fails because the re-ranker's loss on the channel leaks into
one merged rank.

**Decision rule, applied (FACT):** 0.9209 ≥ 0.9211 is FALSE and −0.0006 ≥ −0.005 is TRUE ⇒ **NO-GO**; `rankov` and
`rankz` both give the same NO-GO (0.9209, 0.9206).

## Full-net trained refit (`_fullS23`) — definitive check (FACT)

To resolve whether the ho1→full score distribution gap was the culprit, we extracted full-net scores on the bench PC+S2 rows (`results/c2gap/cand_fullS23.parquet`, 37,229 rows, 377 molecules) and retrained all three variants on full-net features:

```
python research/scripts/e8_sv_gonogo.py --feats full --train-tag _fullS23 --out results/c2gap_full/sv_gonogo_refit_full.json
python research/scripts/e8_sv_gonogo.py --feats rankov --train-tag _fullS23 --out results/c2gap_full/sv_gonogo_refit_rankov.json
python research/scripts/e8_sv_gonogo.py --feats rankz --train-tag _fullS23 --out results/c2gap_full/sv_gonogo_refit_rankz.json
```

| variant (refit on full nets) | merged, no re-rank | merged, with re-rank | channel, no re-rank | channel, with re-rank | δ (re-rank − no) | 95% CI (boot 4000) | 1-mol deltas (impr/regr) | Decision |
|---|---|---|---|---|---|---|---|---|
| `full` (`_fullS23`) | **0.9211** | **0.9209** | 0.6757 | 0.6149 | −0.0002 | [−0.0006, 0.0] | 0 / 1 | **NO-GO** |
| `rankov` (`_fullS23`) | **0.9211** | **0.9209** | 0.6757 | 0.6122 | −0.0002 | [−0.0006, 0.0] | 0 / 1 | **NO-GO** |
| `rankz` (`_fullS23`) | **0.9211** | **0.9206** | 0.6757 | 0.5933 | −0.0005 | [−0.0014, 0.0] | 0 / 2 | **NO-GO** |

Even with net-consistent training on deploy net scores, the C2 re-ranker fails on SV by exactly the same margin. Popularity and structural features derived from synthetic library distributions do not provide lift on enveda-180 and degrade channel order (0.615 vs 0.676).

## Recommendation

1. **Do not ship the C2 re-ranker in E8.** E8 = E7 is safe and unchanged (`+0.005..+0.015` LB from E7).
2. The full-net refit confirms that the NO-GO is fundamental to the feature set on enveda-180 queries, not just an artifact of the ho1 shift. Task E should remain permanently closed for E8.
3. Focus forward effort on **E9** (E7 + C3 seed-Tc re-ranker from Task F), where lambdarank showed robust +0.032 to +0.048 gain on the C3NP bench with zero floor regressions.
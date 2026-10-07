# E3c study: a real-score ranker term for the forward-model re-ordering

Date: 2026-10-02. Code: `research/bench/fm/fm_real.py`, `fm_real_learned.py`, `fm_real_extra.py`, `visible_real_check.py`.
Outputs: `results/bench/fm/real_score.*`, `real_learned.*`, `real_extra.txt`, `visible_real_check.json`.
No Kaggle push, upload or submission. No existing file was changed. No forward-model score was recomputed.

## 1. Conclusion

- **Do not push an E3c. It was not built.** No variant has a credible gain over E3b that the leaderboard could show.
- The real-score term does fix root defect #1: on the labelled visible test molecules it demotes 0 of 39 correct
  answers without any library protection (E3's rule: 6 of 39). It is an equally valid fix, not a better one.
- On the bench every real-score family, gated or not, ends within +-0.005 of E3b on the S1/S2 mean when the setting is
  chosen on one half and reported on the other. The 95% intervals are about +-0.015.
- What the real-score rules change is the split: about +0.015 more on S1 and about -0.015 on S2 than E3b. The learned
  combiner pushes this further (+0.035 / -0.028).
- **The S2 gain of the bench does not transfer.** The three leaderboard results fit a test set where at most about 10%
  of the molecules behave like S2, with a point estimate near 0 (section 5). E3b's +0.078 on S2 is worth at most about
  +0.008 on the leaderboard; the differences between variants on S2 are worth at most 0.002.
- Expected leaderboard effect of the best real-score rule over E3b: **0 to +0.006**, against noise of +-0.016.

## 2. What the rankers output

`research/bench/eng/casmi_engine.py:391-431`.

| | prvsiyan ranker (`pv`) | our ranker (`ours`) |
|---|---|---|
| model | 8 `HistGradientBoostingClassifier`: class-1 weight 0.30 and 0.60, 4 seeds each | 4: class-A weight 0.35 and 0.55, 2 seeds each |
| target | candidate is the true structure (one row per candidate, pointwise) | same |
| output per candidate | mean over the ensemble of `predict_proba[:, 1]` | same |

- The outputs are probabilities of single candidates. They are not normalised over the list and do not sum to 1.
- Blend (`blend_scores`): `0.88 * (1 - rank_norm(pv)) + 0.12 * (1 - rank_norm(ours)) + 1e-6 * pv`. The ranks are taken
  over **all candidates of the mass window**, not over the top 60. Only this value is exported to the forward stage.
- Example (S1, one molecule, top 4): blend 1.00 / 0.96 / 0.93 / 0.92; `pv` 0.76 / 0.012 / 0.006 / 0.003. The blend
  shows four near-equal candidates where the ranker is certain of the first.
- On the visible test molecules the truth's blended logit is 5.2 above the best other candidate (median; minimum 3.3).

### Honest raw scores: how this was ensured

The raw scores used are `pv_kf` and `ours_kf`, written by `bench_cvrank.py` into `results/bench/e1/S1.pkl` / `S2.pkl`.
`fm_real_extra.py` checks:

- `clean_kf` equals `blend_scores(pv_kf, ours_kf)` to 6e-8 for all 250 molecules in S1 and S2.
- The 50 fold-0 molecules of S1, re-scored with the cached fold-0 rankers, reproduce the stored `pv_kf` to 3e-8 (50 of 50).
- The same molecules scored with the fold-1 rankers differ for 50 of 50. The stored values are therefore the
  held-out fold's, not another fold's.
- The stored shipped-ranker `pv` differs from `pv_kf` for 50 of 50 (median largest difference 0.07).
- The fold-0 fit excludes 9,700 of 142,762 rows of `rank_train.npz` (groups below 250 with `G % 5 == 0`), as the
  code of `bench_cvrank.py:60` states.

Not re-checked for `ours_kf` beyond the first item; it comes from the same loop.
The base list of every comparison is the `clean_kf` list, so "no re-ordering" is the honest E1 (S1 0.863, S2 0.647).

## 3. Variants

All rules are slot-preserving inside formula groups of the top 60, like E3. Forward term: `lam * (z(ICEBERG) + z(GLACIER))`,
z over the group. `L = 0.88 * logit(pv) + 0.12 * logit(ours)`.

| ranker term | definition |
|---|---|
| `rank` | z over the group of the exported rank blend (E3, E3b) |
| `zl_pv` | z over the group of `logit(pv)` (the diagnosis variant) |
| `zl_bl` | z over the group of `L` |
| `zl_list` | `L` standardised over the whole top 60 |
| `abs` | `L` in absolute logit units; one parameter `a` = logit units per forward z |

- (a) each term with lam 0.1 to 2.0 (`abs`: a 0.25 to 4), with and without E3b's library protection (>= 0.6): 54 real-score settings.
- (b) the forward term applied only when the ranker margin between the group's two best is below a threshold
  (probability of `pv` 0.1 / 0.2 / 0.3, or `L` 1 / 2 / 3): 84 settings.
- (c) learned combiner: logistic regression and a small LightGBM LambdaRank on up to 16 features (ranker logits and
  margins, forward scores raw and z, library similarity, group size).

## 4. Results

Paired differences, MRR@25, 10,000-resample bootstrap over the 250 molecules. E3b on this sample: S1 +0.004, S2 +0.078.

### Full-sample values of selected settings (not held out)

| setting | S1 vs E1 | S2 vs E1 | S1 vs E3b | S2 vs E3b | mean vs E3b |
|---|---|---|---|---|---|
| `zl_bl` lam 0.5 | +0.024 | +0.065 | +0.020 [+0.005, +0.037] | -0.013 [-0.033, +0.006] | +0.004 [-0.010, +0.017] |
| `zl_bl` lam 1.0 | +0.010 | +0.081 | +0.006 [-0.016, +0.028] | +0.003 [-0.017, +0.023] | +0.004 [-0.012, +0.020] |
| `abs` a 1.5 | +0.013 | +0.085 | +0.010 [-0.013, +0.033] | +0.006 [-0.015, +0.028] | +0.008 [-0.010, +0.026] |
| `abs` a 2.0 | +0.001 | +0.091 | -0.003 [-0.029, +0.023] | +0.013 [-0.010, +0.036] | +0.005 [-0.015, +0.025] |
| `zl_bl` lam 1.0, protect 0.6 | +0.004 | +0.076 | +0.001 [+0.000, +0.002] | -0.002 [-0.017, +0.012] | -0.001 [-0.008, +0.006] |
| `abs` a 2.0, gate `L` < 2 | +0.010 | +0.086 | +0.007 [-0.016, +0.029] | +0.008 [-0.015, +0.031] | +0.007 [-0.012, +0.026] |
| raw-score order inside groups, forward term off | -0.005 | -0.005 | | | -0.046 [-0.062, -0.031] |

The best full-sample cell is +0.008 on the mean and its interval includes 0.

### Held out: setting chosen on a random half, reported on the other half

20 random splits, both directions; every molecule is held out 20 times.

| family (settings) | criterion | S1 vs E1 | S2 vs E1 | S1 vs E3b | S2 vs E3b | mean vs E3b |
|---|---|---|---|---|---|---|
| rank term, E3 / E3b (10) | mean | +0.004 [-0.004, +0.014] | +0.064 [+0.041, +0.088] | +0.000 [-0.006, +0.007] | -0.014 [-0.025, -0.005] | -0.007 [-0.013, -0.001] |
| (a) real-score, no gate (54) | mean | +0.006 [-0.013, +0.024] | +0.075 [+0.046, +0.105] | +0.002 [-0.016, +0.019] | -0.004 [-0.022, +0.014] | -0.001 [-0.016, +0.014] |
| (a) real-score, no gate (54) | worse of S1, S2 | +0.019 [+0.003, +0.036] | +0.063 [+0.039, +0.088] | +0.015 [-0.000, +0.031] | -0.016 [-0.035, +0.002] | -0.000 [-0.013, +0.012] |
| (a) absolute logit only (14) | no S1 loss, then S2 | +0.003 [-0.018, +0.024] | +0.083 [+0.053, +0.115] | -0.001 [-0.021, +0.019] | +0.005 [-0.015, +0.024] | +0.002 [-0.014, +0.018] |
| (b) gated (84) | mean | +0.007 [-0.011, +0.025] | +0.071 [+0.042, +0.100] | +0.003 [-0.013, +0.019] | -0.008 [-0.026, +0.010] | -0.002 [-0.017, +0.012] |
| (a) + (b) (126) | mean | +0.007 [-0.011, +0.026] | +0.073 [+0.044, +0.103] | +0.004 [-0.014, +0.021] | -0.006 [-0.024, +0.013] | -0.001 [-0.016, +0.013] |
| (c) logistic regression, 16 features | trained on the other half | +0.038 [+0.021, +0.058] | +0.051 [+0.026, +0.076] | +0.035 [+0.018, +0.053] | -0.028 [-0.049, -0.008] | +0.003 [-0.010, +0.017] |
| (c) LightGBM LambdaRank, 16 features | trained on the other half | +0.022 [+0.004, +0.040] | +0.071 [+0.042, +0.101] | +0.018 [+0.002, +0.035] | -0.007 [-0.028, +0.013] | +0.005 [-0.009, +0.020] |

- (a) and (b) leave nothing measurable on the table against E3b; the gate adds nothing to the real-score term.
- (c) with 4 or 8 features is worse than E3b on the mean (-0.006 to -0.011). With 16 features it is level.
- The "E3b" column compares against E3b's full-sample value, a single member of its family; the first row shows what
  selection costs inside E3b's own family (-0.007).

### Where the S1 gain of the unprotected real-score rules comes from

| rule | S1: correct top-1 demoted / truths promoted to top-1 | S2 |
|---|---|---|
| E3 | 39 / 16 | 16 / 40 |
| E3b | 0 / 1 | 2 / 27 |
| `zl_bl` lam 0.5 | 2 / 11 | 5 / 23 |
| `abs` a 1.5 | 9 / 14 | 6 / 32 |

- The promoted S1 truths have library similarity 0.81 (median) and sat behind a candidate with library similarity 0.
  E3b cannot promote them: a protected candidate keeps its slot in both directions.
- The same is visible in the combiner: a model without any forward feature (`L`, its margin, library similarity) gains
  +0.021 on S1 and loses 0.032 on S2. Part of the S1 gain is a library prior, not forward-model evidence. S1 is built
  so that every truth has a library spectrum; a prior tuned on it is the kind of local preference the engine's author
  reports did not transfer.

### Visible test molecules (real test spectra, labelled)

39 molecules of the parity run; raw probabilities recomputed locally with the shipped rankers; Kaggle lists and Kaggle forward scores.

| rule | MRR change | correct top-1 demoted |
|---|---|---|
| E3 (rank, lam 1) | -0.089 | 6 of 39 |
| rank, lam 0.5 / 0.1, no protection | -0.063 / -0.013 | 4 / 1 |
| E3b | 0 | 0 |
| `zl_bl`, `zl_pv`, `zl_list` at lam 0.5 and 1.0; `abs` a 0.5 to 2.0; no protection | 0 | 0 |
| `abs` a 4.0, no protection | -0.013 | 1 |

### As-submitted rankers and S3

| rule | S1 | S2 | S3 (46 listed truths of 300) |
|---|---|---|---|
| E3b | +0.004 [-0.001, +0.012] | +0.065 [+0.038, +0.094] | +0.003 [+0.000, +0.008] |
| `zl_bl` lam 0.5 | +0.024 [+0.009, +0.040] | +0.068 [+0.043, +0.093] | +0.003 [+0.000, +0.007] |
| `abs` a 1.5 | +0.014 [-0.005, +0.034] | +0.084 [+0.054, +0.114] | +0.004 [-0.002, +0.012] |

S3 has no query-held-out raw scores; these rows use the shipped rankers. S3 says "no harm seen", nothing more.

## 5. Reality check against the leaderboard

| | bench S1 | bench S2 | visible set | leaderboard |
|---|---|---|---|---|
| E1 level | 0.863 | 0.647 | 1.000 | 0.353 |
| E3 - E1 | -0.057 | +0.075 | -0.084 | -0.031 |
| E3b - E1 | +0.004 | +0.078 | 0 | +0.002 |

Model: a share `v` of the hidden molecules behaves like the visible set, `w1` like S1, `w2` like S2; the rest scores
about 0 and cannot change.

- Exact solution with S1 and S2 only (two differences): `w1` = 0.54, `w2` = -0.002. That mix would give E1
  0.54 * 0.863 = 0.47, not 0.353. S1 alone is not the right picture of the scoring part of the test set.
- With the E1 level as a third equation the best non-negative fit is `v` = 0.34, `w1` = 0, `w2` = 0.03. It predicts E1
  0.353, E3 -0.026, E3b +0.002.
- `v` and `w1` cannot be told apart: moving 0.4 of the test set from one to the other changes the predicted E3 loss by
  0.004. Any `w1` from 0 to 0.4 fits.
- `w2` is bounded. With `w2` = 0.1 the model predicts E3 -0.012 to -0.017 (observed -0.031) and E3b +0.009 (observed
  +0.002). With `w2` = 0.2 it predicts E3 about 0, which is two noise units from the observation.

Reading:

- About a third of the hidden molecules are exact or near-exact library hits that E1 already ranks first. Almost all of
  E1's 0.353 comes from them. E3 lost there; E3b and the real-score rules are neutral there.
- Molecules of the S2 kind (truth listed at rank 2-25, no library spectrum) are at most about 10% of the test set and
  possibly close to none. **A bench S2 gain of x shows as at most 0.1 x on the leaderboard**: +0.078 becomes at most
  +0.008, and the +-0.015 differences between variants become at most +-0.002.
- The bench's S2 scenario answers "can the forward models rank isomers when the truth is listed": yes. It does not
  describe the hidden test, where the truth is mostly either first already or not listed at all.

Selection and report under such mixes (same split protocol; criterion and reported number are `w1 * S1 + w2 * S2`):

| mix `w1`, `w2` | best real-score rule, held out, vs E1 | vs E3b | E3b vs E1 |
|---|---|---|---|
| 0.4, 0 | +0.007 [+0.001, +0.014] | +0.006 [-0.000, +0.012] | +0.001 [-0.000, +0.005] |
| 0.3, 0.05 | +0.008 [+0.002, +0.014] | +0.003 [-0.002, +0.008] | +0.005 [+0.003, +0.008] |
| 0.2, 0.1 | +0.009 [+0.004, +0.014] | +0.000 [-0.004, +0.005] | +0.009 [+0.006, +0.012] |
| 0, 0.1 | +0.008 [+0.005, +0.011] | +0.000 [-0.002, +0.002] | +0.008 [+0.005, +0.011] |

The most favourable mix gives +0.006 over E3b, and that mix assumes all Class-1 test molecules look like S1 rather than
like the visible set, where the gain is 0 by construction.

## 6. Recommendation

- Keep E3b as the forward-model variant. Do not spend a submission on E3c.
- If the forward stage is touched again for another reason, replace the rank term by `zl_bl` with lam 0.5 to 1.0 and
  export the two raw probabilities from the engine runner. It removes the defect at its source and no longer needs
  the 0.6 threshold. Expected leaderboard change against E3b: 0 to +0.006.
- Re-ordering a top-60 list has no room left on this leaderboard. The 60% of the score that is missing sits in
  molecules whose truth is not listed; that is a pool and coverage problem, not a ranking one.

## 7. Limits and unverified points

- Leaderboard noise of +-0.016 was taken as given; the mix bounds in section 5 move with it.
- The mix model assumes the bench deltas apply unchanged to the hidden molecules of each kind. Bench molecules are
  popular natural products; the collision energies differ from the test's.
- The visible-set check covers 39 molecules, with raw probabilities from a local refit of the shipped rankers
  (fit noise against Kaggle: same rank 1 for 40 of 40, lower ranks differ).
- S1 and S2 are the same 250 molecules; the two columns are not independent.
- The learned combiner was trained and tested on bench molecules only. Its S1 gain was not tested on any real test data.
- `ours_kf` provenance was checked through the blend identity only.
- The stored `clean_kf` is float32; its 1e-6 tie-break term is lost, so the stored order differs from a float64
  recomputation inside tied ranks for 21% of the molecules. The lists of this study and of the diagnosis use the stored order.

## 8. Commands

```
python research/bench/fm/fm_real.py                        # sections 3-4: families (a), (b), half-split selection   (1 min)
python research/bench/fm/fm_real_learned.py                # (c) learned combiner, molecule-level held out            (1 min)
python research/bench/fm/fm_real_extra.py 3                # provenance, class-mix selection, as-submitted, S3        (15 s)
python research/bench/fm/visible_real_check.py <E3 kernel output dir> 3      # visible test molecules                 (10 s)
```

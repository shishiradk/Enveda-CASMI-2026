# Why E3 (E1 + forward-model re-ordering) lost 0.031 on the leaderboard

Date: 2026-10-02. Code: `research/bench/fm/`. Data: `results/bench/fm/`. Recommended variant: `research/kaggle_e1/e3b/` (built, not pushed).
No leaderboard submission was used. E1 / E2 / E3 files were only read.

## 1. Conclusion

- **The step is not harmful in itself; E3's combination rule is.** It lets the forward models overrule the ranker
  whatever the ranker's evidence is, including an exact library match.
- **Direct proof on E3's own Kaggle run.** The 400 visible test molecules are placeholder rows of train (enveda-180), so
  their structures are known and E1 has their spectra in its library. On them E1 scores 1.000 and E3 0.916:
  **-0.084 [-0.106, -0.064]**. All 55 top-1 changes of that run replaced a correct answer whose library similarity was
  1.0; none fixed anything.
- **On the bench E3's rule loses on Class 1 and gains on Class 2**: S1 -0.057 [-0.094, -0.019], S2 +0.075 [+0.037, +0.114]
  (honest ranker scores, paired, MRR@25). The leaderboard loss (-0.031, about 9% of E1's 0.353) matches the Class-1
  pattern (S1 loses 6.6% of its MRR, the visible set 8.4%).
- **Fix (E3b):** a candidate with its own library match (library similarity >= 0.6) keeps its slot and is not scored;
  lam 0.5 / 0.5. Bench: **S1 +0.004 [-0.001, +0.011], S2 +0.078 [+0.052, +0.106]**; visible set 0 by construction.
  With the setting chosen on one half and applied to the other (20 random splits): S1 +0.001 [-0.012, +0.014],
  S2 +0.060 [+0.035, +0.087].
- **Confidence.** High that E3's loss comes from overruling library-backed answers. Moderate that E3b does not lose
  against E1. Low that E3b gains something the leaderboard can show: the S2 gain needs the truth to be listed at rank
  2-25 without a library spectrum, and E1's 0.353 says such molecules are much rarer in the hidden test than in S2.
  Expected leaderboard effect: between 0 and about +0.02, inside the +-0.016 noise.

## 2. Code review of E3

Paths are under `research/kaggle_e1/` unless stated.

### Defects

| # | Where | Defect | Effect |
|---|---|---|---|
| 1 | `build_e3.py:77`, `:91`; `research/bench/eng/casmi_engine.py:425-431`; `e3/fm_rerank.py:80-86` | The exported "ranker score" is the engine's blend, which is a **within-molecule rank** (0.88 rank(pv) + 0.12 rank(ours)), not a probability. Its z-score inside a group is a straight line in list position: neighbours differ by about 3.5 / n z-units (0.06-0.08 for n = 40-60) whether the ranker was certain or not. | **Root cause.** A forward z-difference of 1 at lam 0.1 already swaps neighbours. On the visible set even lam 0.1 / 0.1 demotes 16 of 400 correct answers that have library similarity 1.0 (section 4). |
| 2 | `e3/fm_stage.py:97-112`, `:167`; `build_e3.py:94-96` | Library evidence is not used. `lib_max` only orders the work, and it is the maximum over the whole candidate window. | Candidates with a measured library match are re-ordered by a predicted spectrum. |
| 3 | `build_e3.py:40`; `e3/fm_stage.py:150`; `e3/fm_rerank.py:86` | lam 1.0 for each of two models that are 0.90 correlated within a molecule (median, Kaggle run). | The forward term has a spread of about 1.95 against 1 for the ranker term: the order inside a group is almost the forward order. |
| 4 | `e3/fm_stage.py:28-40`, `:104-111` | "Same formula" does not restrict anything: the engine's window is a mass window, so the largest formula group holds 75% of the list on Kaggle (median) and 93% on the bench. 19,390 of about 20,600 listed candidates were in a group. | The step is a re-ranking of nearly the whole list, not a tie-break among a few isomers. |
| 5 | `build_e3.py:39`, `:83-93` | The list is extended from E1's 40 to 60 and re-ordered as one. | On Kaggle 3.6 of the 25 submitted candidates per molecule came from ranks 26-60 (0.9 from 41-60). No measurable effect on the bench (top-N 25 / 40 / 60 agree within the CI). |
| 6 | `e3/fm_rerank.py:42-48` | Groups of two: with ddof = 1 every z is +-0.707, for the ranker and for each model. | A pair is swapped whenever both models disagree with the ranker, whatever the margins. Minor. |
| 7 | `research/forward_model/ice_runner.py:91-92`, `:388` | Observed spectra are used unfiltered (`obs_top_k = 0`, `obs_min_rel = 0`) and peaks above the precursor are kept, while E1's engine cuts them (`casmi_engine.py:338`). Test spectra: median 271 peaks, 19 of them >= 1% of the base peak; 48% have peaks above precursor + 1.5. | Lowers the similarity level (median 0.14 on Kaggle) for all candidates alike. No measurable effect on the ranking (section 5c). |
| 8 | `e3/test_fm_rerank.py` | The tests cover the permutation mechanics only. The rule was never run on labelled lists of our ranker (`forward_model_runner.md` section 5 says so). | Process defect: the visible set and the bench would both have shown the loss before submitting. |

### Checked and correct

- Higher is better for all three terms. Scores of a group come from one molecule's list and are comparable.
- Slots: a group is written back into its own positions (`e3/fm_rerank.py:88-92`). The 400 submitted rows are
  reproduced exactly by running `rerank_molecule` locally on the downloaded `eng_lists.json` and runner scores.
- Zero variance or fewer than two scored members: the model contributes 0; a group covered by no model is untouched.
- Unscored candidates get z = 0 (the mean). In the Kaggle run every sent candidate was scored.
- Formula: `CalcMolFormula` of the candidate SMILES as listed (`e3/fm_stage.py:28-40`); a charged form gets its own formula
  string and group. Tautomers share a formula.
- Spectra: only `[M+H]+` / `[M+Na]+` spectra are scored. Per spectrum the candidate is predicted at the spectrum's own
  energies, the predictions are scaled to max 1 and summed, top 100 peaks; similarity = unweighted spectral entropy,
  greedy one-to-one matching within max(0.01 Da, 20 ppm); mean over the molecule's covered spectra.
- Collision energy: `test.parquet` has `collision_energy_ev` as `DOUBLE[]` with [20], [40], [60] or [20, 40, 60] for
  all 1,213 rows; these values are fed unchanged. Instrument token `QTOF`; all test rows are timsTOF.
- Not relevant for the test data: energies of 0 are dropped and more than three energies are thinned
  (`ice_runner.py:212`, `:226-228`).

## 3. Forward-model scores for the bench

- Dataset `shishiradhikari11/casmi-bench-fm-input` (private): for each of the 250 S1/S2 molecules the union of the
  top-60 lists (E3's list builder) under `clean_kf` and `blend` in S1 and S2, and the 46 S3 molecules whose truth is
  listed; query spectra inside the JSON. 259 molecules have a covered spectrum (216 of the 250), 12,229 candidates.
- Notebook `shishiradhikari11/casmi-bench-fm` (private, version 1, T4): E3's `fm_stage.locate / install_site / run_model`,
  same command line as E3 (adducts, `QTOF`, default energies, RDKit 2025.03.6). One addition to the copied runner:
  every raw prediction is pickled, so other merges are computed offline with the runner's own functions.
- Run: 2,907 s = **0.81 GPU h**. GLACIER 58,258 predictions at 46.9 /s, ICEBERG 58,230 at 38.4 /s; 55,687 of 55,691
  entry-candidates scored (4 with an unsupported element). All molecules scored, not only formula groups.
- One entry per spectrum, so per-spectrum scores exist; the mean over a molecule's entries is E3's score.
  Offline recomputation from the float32 dump differs from the runner by at most 0.009 (GLACIER) / 0.039 (ICEBERG);
  the bench deltas are identical to two decimals.
- Baselines reproduced: S1 0.863, S2 0.647 (`clean_kf`), 0.905 / 0.695 (`blend`).

## 4. The actual Kaggle run, scored with labels

All 1,213 visible test spectra are exact copies of enveda-180 train rows (`competition_intel.md` item 15), which gives
labels for the E3 run (`research/bench/fm/visible_test_check.py`; labels in `results/bench/fm/visible_test_labels.json`, from a DuckDB join of test and train rows on precursor m/z, peak count and first m/z: 1,213 of 1,213 rows match, one structure per molecule). This is an easy Class-1 set: every truth has library
similarity about 1.0 and E1 ranks all 400 first.

| lam_ice / lam_gl (E3 rule) | MRR change | correct top-1 demoted |
|---|---|---|
| 1 / 1 (submitted) | -0.084 [-0.106, -0.064] | 55 of 400 |
| 0.5 / 0.5 | -0.065 [-0.084, -0.047] | 44 |
| 0.25 / 0.25 | -0.046 [-0.062, -0.031] | 32 |
| 0.1 / 0.1 | -0.022 [-0.034, -0.013] | 16 |
| 0.1 / 0 | -0.015 [-0.023, -0.007] | 11 |

- No lam is safe with the rank-based ranker term.
- Forward models alone, inside the truth's formula group (358 groups, median 40.5 members): truth first for 75%
  (ICEBERG) and 81% (GLACIER); the ranker 100%.

## 5. Bench analysis

Paired differences versus no re-ordering, MRR@25, 10,000-resample bootstrap over the 250 molecules, honest ranker
scores (`clean_kf`) unless stated. The 34 molecules without a covered spectrum count with 0. S1 and S2 are the same
molecules, so the two columns are not independent.

### (a) E3's exact rule

| | S1 | S2 | mean |
|---|---|---|---|
| E3 rule, honest ranker | -0.057 [-0.094, -0.019] | +0.075 [+0.037, +0.114] | +0.009 [-0.022, +0.041] |
| E3 rule, as-submitted ranker (`blend`) | -0.071 [-0.106, -0.035] | +0.066 [+0.029, +0.103] | -0.003 [-0.033, +0.028] |
| correct top-1 demoted / truths promoted to top-1 | 39 / 16 | 16 / 40 | |

S3 (46 listed truths of 300): +0.004 [-0.002, +0.012].
The bench shows the loss where the truth has a library spectrum (S1) and a gain where it has none (S2).

### (b) lam_ice x lam_gl (E3 rule otherwise)

S1 (rows lam_ice, columns lam_gl):

| | 0 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| 0 | 0 | +0.012 | +0.008 | -0.024 | -0.045 | -0.090 |
| 0.1 | +0.017 | +0.016 | +0.009 | -0.015 | -0.045 | -0.093 |
| 0.25 | +0.017 | +0.015 | +0.002 | -0.014 | -0.046 | -0.087 |
| 0.5 | -0.011 | -0.005 | -0.012 | -0.032 | -0.054 | -0.086 |
| 1.0 | -0.038 | -0.035 | -0.035 | -0.043 | -0.057 | -0.081 |
| 2.0 | -0.069 | -0.067 | -0.065 | -0.062 | -0.073 | -0.088 |

S2:

| | 0 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| 0 | 0 | +0.043 | +0.059 | +0.050 | +0.052 | +0.060 |
| 0.1 | +0.041 | +0.063 | +0.065 | +0.059 | +0.065 | +0.058 |
| 0.25 | +0.055 | +0.069 | +0.073 | +0.067 | +0.072 | +0.056 |
| 0.5 | +0.068 | +0.085 | +0.077 | +0.080 | +0.073 | +0.061 |
| 1.0 | +0.073 | +0.079 | +0.086 | +0.084 | +0.075 | +0.057 |
| 2.0 | +0.061 | +0.061 | +0.070 | +0.072 | +0.069 | +0.063 |

- S2 is flat at +0.05 to +0.085 for any lam >= 0.1; S1 falls steadily with lam. Typical CI half-width 0.02-0.04.
- Each model alone: ICEBERG 0.25 gives +0.017 / +0.055, GLACIER 0.25 gives +0.008 / +0.059; at 1.0 they give
  -0.038 / +0.073 and -0.045 / +0.052. Neither model is the problem; the two are interchangeable within the CI.
- lam pair chosen on one half, applied to the other (20 splits): S1 +0.004 [-0.015, +0.024], S2 +0.068 [+0.045, +0.092].
  Picks: 0.5/0.1 and 0.25/0.1 (13 each), 0.1/0.1 (7).
- Small lam looks fine on S1 (+0.016 at 0.1 / 0.1) but still loses 0.022 on the visible set (section 4). S1 truths have
  library similarity 0.87 (median) from other instruments, the visible set 1.0 from the same instrument; a rank-based
  ranker term cannot tell either from a guess.

### (c) Variants (one change to E3's rule; lam 0.25 / 0.5 / 1.0 for both models)

| variant | S1 | S2 | reading |
|---|---|---|---|
| ranker term = z(logit of the pv probability) instead of the rank | +0.007 / +0.018 / -0.003 | +0.046 / +0.072 / +0.084 | removes the S1 loss at every lam |
| only when the pv margin between the group's two best is < 0.2 | +0.010 / +0.002 / -0.007 | +0.059 / +0.064 / +0.059 | helps; < 0.4 is too loose (-0.033 at lam 1) |
| only the top-5 of the group | +0.010 / +0.013 / -0.006 | +0.030 / +0.044 / +0.057 | safer, loses part of the S2 gain |
| only the top-10 of the group | +0.012 / +0.002 / -0.021 | +0.047 / +0.065 / +0.074 | |
| z over the whole top-N instead of the group | +0.001 / -0.028 / -0.055 | +0.076 / +0.082 / +0.075 | same as E3 (groups are nearly the whole list) |
| whole top-N as one group | +0.004 / -0.023 / -0.050 | +0.088 / +0.094 / +0.088 | same |
| rank sum instead of z | -0.012 / -0.038 / -0.053 | +0.051 / +0.054 / +0.059 | not better |
| reciprocal-rank fusion, K = 3 | +0.005 / +0.005 / -0.043 | +0.036 / +0.065 / +0.074 | not better |
| only molecules with best library similarity < 0.7 | +0.009 / +0.008 / +0.011 | +0.055 / +0.064 / +0.066 | safe on S1 at every lam |
| only molecules with best library similarity >= 0.7 | -0.007 / -0.040 / -0.068 | +0.018 / +0.017 / +0.009 | this is where E3 loses |
| candidates with own library similarity >= 0.6 keep their slot | +0.004 / +0.004 / +0.005 | +0.062 / +0.078 / +0.073 | safe on S1 at every lam; chosen |
| cosine instead of entropy | -0.001 / -0.033 / -0.065 | +0.071 / +0.068 / +0.068 | no difference |
| best single energy of 20 / 40 / 60 instead of the spectrum's own | +0.003 / -0.018 / -0.057 | +0.066 / +0.072 / +0.074 | no difference |
| mean over 20 / 40 / 60 | +0.013 / -0.019 / -0.063 | +0.061 / +0.072 / +0.068 | no difference |
| observed peaks >= 1% of base peak, none above the precursor | +0.000 / -0.032 / -0.063 | +0.079 / +0.077 / +0.078 | no difference |
| the same, precursor peak removed on both sides | +0.005 / -0.025 / -0.047 | +0.073 / +0.082 / +0.093 | no difference within the CI |

All CIs are in `results/bench/fm/analysis.txt` and `reco_v0.txt`; half-widths are 0.01-0.04.
Spectrum treatment, similarity and energy merge do not matter. What matters is how much the ranker's evidence counts.

### (d) Diagnostics

Formula groups that contain the truth, all members scored (206 in S1, 207 in S2, median 42-43 members):

| truth first in its group | S1 | S2 |
|---|---|---|
| ranker | 84% | 54% |
| ICEBERG alone | 55% | 56% |
| GLACIER alone | 52% | 52% |
| z(ICEBERG) + z(GLACIER) | 59% | 58% |
| ranker first, forward not | 33% | 16% |
| forward first, ranker not | 7% | 21% |

- The forward models are as good as the ranker when the truth has no library spectrum and far worse when it has one.
  Their accuracy does not depend on the scenario (the spectra and candidates are the same); the ranker's does.
- They are much weaker here than in the stand-alone validation (top-1 0.80 with 20 PubChem decoys): real lists are
  larger and closer. Truth first by group size 2-9 / 10-29 / 30-60: 71% / 66% / 52%.
- No dependence on adduct (with an `[M+Na]+` spectrum 58%, `[M+H]+` only 59%) or on the number of covered spectra
  (one: 62%, four or more: 56%).

Truths the ranker had first and E3 demoted:

| | S1 demoted (39) | S1 kept first (133) | S2 demoted (16) | S2 kept first (93) |
|---|---|---|---|---|
| truth has a library spectrum | 100% | 99% | 25% | 2% |
| truth's library similarity, median | 0.88 | 0.87 | 0 | 0 |
| group size, median | 56 | 33 | 56 | 23 |
| pv margin between the group's two best, median | 0.28 | 0.40 | 0.12 | 0.50 |
| truth's rank by forward score, median | 5 | 1 | 3 | 1 |
| new rank, median | 2 | 1 | 2 | 1 |
| with an `[M+Na]+` spectrum | 41% | 27% | 31% | 23% |
| covered spectra, median | 4 | 3 | 3 | 3 |

- Common to the demotions: a large group, and in S1 a truth that had a library match of 0.88 and a comfortable ranker
  margin. The forward models did not rank it badly (median 5th of 56); the rule simply gave the ranker no weight.

## 6. Recommended setting: E3b

Rule: E3's rule with two changes.
1. A candidate whose own library similarity (the engine's per-candidate `lib`) is >= 0.6 belongs to no group: it is
   not sent to the forward models and keeps its slot.
2. lam 0.5 / 0.5.

| | S1 | S2 | mean |
|---|---|---|---|
| E3b, honest ranker | +0.004 [-0.001, +0.011] | +0.078 [+0.052, +0.106] | +0.041 [+0.028, +0.055] |
| E3b, as-submitted ranker | +0.004 [-0.001, +0.012] | +0.065 [+0.038, +0.094] | |
| E3b, random half A / half B | +0.001 / +0.006 | +0.084 [+0.048, +0.124] / +0.073 [+0.036, +0.113] | |
| setting chosen on one half among 68 (rank or logit term, lam, gate, protection), applied to the other; 20 splits | +0.001 [-0.012, +0.014] | +0.060 [+0.035, +0.087] | +0.031 [+0.014, +0.047] |
| correct top-1 demoted / truths promoted to top-1 | 0 / 1 | 2 / 27 | |

S3: +0.003 [+0.000, +0.008]. Visible set: 0 (every truth is protected).
The cross-fitted row is the honest estimate; the E3b row is the full-sample value of one member of that family.
The protection threshold (0.6 / 0.7 / 0.8) and lam (0.25 / 0.5 / 1.0) all give S1 +0.003 to +0.012 and S2 +0.062 to
+0.078, so the choice inside the family is not critical.

Why protection rather than the logit ranker term or a small lam:
- it is the only variant that is exactly neutral on the visible set, where the damage was measured;
- the logit term could not be checked on the visible set (E3 exported ranks only);
- it also cuts the GPU work slightly (protected candidates are not predicted).

Files (nothing pushed):

| path | content |
|---|---|
| `research/kaggle_e1/e3b/fm_stage.py` | E3's stage with `protect_lib` (default 0.6), lam defaults 0.5 / 0.5 |
| `research/kaggle_e1/e3b/build_e3b.py` | builder; the engine runner also exports the per-candidate `lib`; kernel id `shishiradhikari11/casmi-e3b-eligible-fm` |
| `research/kaggle_e1/e3b/casmi_e3b.ipynb`, `kernel-metadata.json` | built notebook |
| `research/kaggle_e1/e3b/test_e3b.py` | unit test of the protection; bench numbers through the stage's own `select` and E3's `rerank_molecule` (reproduces the table) |

`e3/fm_rerank.py` is used unchanged. If the list has no `lib`, the stage raises and the notebook keeps E1's lists.

## 7. Limits

- **The bench cannot say how large the gain is on the hidden test.** S2 is the best case for this step (truth in the
  pool and listed, no library spectrum). E1's 0.353 means most hidden molecules either score through a library match
  (protected, no change) or have no reachable truth (no change). The S2 gain applies to the remainder only.
- The reading "the leaderboard loss is the Class-1 loss" is an inference from three consistent relative losses
  (S1 6.6%, visible set 8.4%, leaderboard 8.8%); it was not tested on the hidden data.
- 250 molecules, the same in S1 and S2; popular natural products; bench query energies (35, 80, 20/50 eV among them)
  differ from the test's 20 / 40 / 60.
- The honest ranker scores remove the ranker-row leak only (`e1_bench_report.md` section 7).
- E3b's notebook was built and parsed, and its stage functions were run on the bench; **it was not run on Kaggle**.
  The engine-side change (exporting `lib`) is three lines and was not executed end to end.
- The logit ranker term (pv probability) would also need the runner to export the probabilities; not implemented.
- S3 has only 43 scored molecules; its intervals say nothing beyond "no harm seen".

## 8. Kaggle resources and commands

- Created: private dataset `shishiradhikari11/casmi-bench-fm-input`, private notebook `shishiradhikari11/casmi-bench-fm`
  (one run, 0.81 GPU h of the 4 h budget). Nothing else was created, changed or deleted; no submission.
- Read only: output of `shishiradhikari11/casmi-e3-eligible-fm` (downloaded to the session scratchpad).

```
python research/bench/fm/build_fm_input.py        # lists.pkl, input/fm_input.json
python research/bench/fm/build_fm_kernel.py       # kernel/ (pushed once)
python research/bench/fm/fm_rescore.py 3          # rescore.pkl: other merges from the raw predictions
python research/bench/fm/fm_analysis.py           # sections 5a-5d -> results/bench/fm/analysis.txt, analysis.json
python research/bench/fm/fm_reco.py               # section 6 family and split -> reco_v0.txt, reco.json
python research/bench/fm/visible_test_check.py <E3 kernel output dir>     # section 4
python research/kaggle_e1/e3b/test_e3b.py         # E3b unit + bench check
python research/kaggle_e1/e3b/build_e3b.py        # rebuild the E3b notebook
```

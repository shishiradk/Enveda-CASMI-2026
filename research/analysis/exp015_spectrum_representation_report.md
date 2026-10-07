# EXP-015 report — spectrum representation ablation + kNN/two-tower complementarity

Date: 2026-09-27. Design: `research/analysis/exp015_spectrum_representation_design.md`.
Script: `research/scripts/exp015_spectrum_representation.py` (version of 2026-09-26 23:51, see Audit §2).
Run: executed by another agent session (build PID 4896 from 21:22; peaks 22:02–22:03; final `run` started 23:52 and ended
00:30:52). This report audits that run. Nothing was rerun or modified. EXP-014 (fragmentation features) is untouched.
The buggy EXP-012 value (0.244) is not used anywhere.

## Executive conclusion
**Yes, modestly and reproducibly.** On a fresh, leakage-audited holdout of 838 ranked natural products, a **finer m/z
resolution (0.01 Da instead of 0.1 Da) raises kNN MRR from 0.507 to 0.554 (+0.047 [+0.029, +0.065])**, R@1 from
0.359 to 0.415 (+0.057 [+0.031, +0.082]), same-formula MRR from 0.466 to 0.513, and lowers the isomer error rate from
0.199 to 0.178 (−0.020 [−0.033, −0.009]). log-intensity (+0.032) and ModifiedCosine re-scoring (+0.035) also help. Linear
intensity (−0.080) and coarse 0.5 Da bins (−0.030) hurt. Removing neutral losses is neutral overall (−0.005) but increases
the isomer error rate (+0.016 [+0.001, +0.030]). The fixed 50/50 kNN + two-tower fusion shows clearly complementary
errors: 222 targets are solved by exactly one model, and the isomer error rate falls from 0.199 to 0.151. Its MRR gain is
not significant (+0.016 [−0.004, +0.034]).

## Dataset
| | PRIMARY (fresh) | SECONDARY / DEV |
|---|---|---|
| Source | EXP-011 DB-only set D ∩ COCONUT 2026-09, never an evaluation target | EXP-013 243 enveda-np-examples targets |
| Eligible → sampled (seed 20260927) | 2,513 → 1,000 | — |
| Excluded | 43 alias in R_noE180 and two-tower training; 2 alias in two-tower training; 1 no query after de-dup | — |
| Accepted targets / query spectra | 954 / 3,147 (15 query spectra removed: byte-identical to a reference) | 250 / 1,184 |
| Truth in 5 ppm COCONUT pool (coverage) | **838 / 954 = 87.8%** | 243 / 250 = 97.2% |
| Median pool (ranked targets) | 28.5 | 36 |
| Same-formula alternative present | 89.0% | 95.5% |
| Query libraries | GNPS 966, RIKEN 649, MSnLib 648, MoNA 295, MassBank 206, Spectraverse 190, MS-DIAL 163, enveda-180 24, other 6 | enveda-np-examples (timsTOF) |
| Overlap with EXP-011/012/013 targets | 0 (and 0 with the 243) | — |

Caveat: PRIMARY queries are mostly public-library (Orbitrap/QTOF) spectra, not the hidden test's timsTOF. Lower
coverage (87.8%) reflects more precursor/adduct inconsistency in those libraries at 5 ppm.

## Baseline (B0 = current representation, R_noE180, EXP-011 kNN rule)
Representation: fragment + neutral-loss blocks, 0.1 Da bins, √intensity, top-150 peaks, L2-normalised. Reference R_noE180 =
298,095 spectra / 78,707 molecules, sha256(sorted rids) `ad8bb53e…29d` (identical to EXP-013's R_noE180). kNN: same
polarity, top-20 by cosine, weights max(sim, 0) + 1e-6, Morgan r2/2048 prediction, molecule = mean over spectra, COCONUT
5 ppm pools.

| | MRR | MRR@25 | R@1 | R@5 | R@10 | R@50 | same-formula MRR / R@1 / R@10 (n) | isomer error rate |
|---|---|---|---|---|---|---|---|---|
| PRIMARY B0 (838) | **0.507** | 0.505 | 0.359 | 0.697 | 0.790 | 0.935 | 0.466 / 0.308 / 0.770 (746) | 0.199 |
| DEV B0 (243) | 0.543 | 0.542 | 0.383 | 0.733 | 0.831 | 0.967 | 0.525 / 0.358 / 0.823 (232) | 0.153 |

DEV B0 reproduces EXP-013's R_noE180 result (0.543259 vs 0.543266; the difference is tie-ordering noise).

## Representation results (PRIMARY, n = 838; paired vs B0, bootstrap 95% CI, 2,000 resamples)
| Variant | MRR | R@1 | R@5 | R@10 | R@50 | Same-formula MRR | ΔMRR [CI] | ΔR@1 [CI] | Δ isomer error rate [CI] |
|---|--:|--:|--:|--:|--:|--:|---|---|---|
| B0 current | 0.507 | 0.359 | 0.697 | 0.790 | 0.935 | 0.466 | — | — | — |
| V2 fragment-only (no NL) | 0.501 | 0.364 | 0.667 | 0.767 | 0.931 | 0.459 | −0.005 [−0.025, +0.014] | +0.005 [−0.021, +0.033] | +0.016 [+0.001, +0.030] |
| **V3a bin 0.01 Da** | **0.554** | **0.415** | **0.718** | **0.813** | 0.940 | **0.513** | **+0.047 [+0.029, +0.065]** | **+0.057 [+0.031, +0.082]** | **−0.020 [−0.033, −0.009]** |
| V3b bin 0.5 Da | 0.477 | 0.332 | 0.649 | 0.766 | 0.931 | 0.439 | −0.030 [−0.044, −0.016] | −0.027 [−0.047, −0.007] | +0.020 [+0.010, +0.031] |
| V4a linear intensity | 0.427 | 0.285 | 0.594 | 0.715 | 0.922 | 0.388 | −0.080 [−0.098, −0.063] | −0.074 [−0.099, −0.051] | +0.062 [+0.049, +0.077] |
| V4b log intensity | 0.538 | 0.398 | 0.714 | 0.805 | 0.936 | 0.498 | +0.032 [+0.017, +0.046] | +0.039 [+0.018, +0.060] | −0.018 [−0.029, −0.008] |
| V5 ModCos re-score (top-500 prefilter) | 0.541 | 0.401 | 0.714 | 0.800 | 0.938 | 0.505 | +0.035 [+0.015, +0.055] | +0.042 [+0.014, +0.070] | −0.014 [−0.027, −0.000] |
| C1 no NL + log ‡ | 0.538 | 0.400 | 0.695 | 0.787 | 0.943 | 0.496 | +0.031 [+0.011, +0.050] | +0.041 [+0.012, +0.069] | −0.011 [−0.025, +0.003] |
| C2 no NL + 0.01 Da + log ‡ | 0.548 | 0.408 | 0.711 | 0.805 | 0.943 | 0.505 | +0.041 [+0.020, +0.062] | +0.049 [+0.022, +0.077] | −0.015 [−0.031, −0.001] |
| C3 = C2 + ModCos ‡ | 0.547 | 0.411 | 0.719 | 0.800 | 0.946 | 0.509 | +0.040 [+0.020, +0.059] | +0.053 [+0.025, +0.079] | −0.007 [−0.022, +0.007] |

‡ **Non-primary.** The C-levels were selected on the old 243 targets (DEV: no NL, 0.01 Da, log), which the current
instructions forbid as a selection basis. They are reported for completeness only and support no conclusion. They are also
not additive: C2 < V3a alone, because removing neutral losses (chosen on DEV) costs on PRIMARY.
DEV direction check (secondary): V3a +0.063 [+0.033, +0.096]; V4b +0.010 (n.s.); V5 +0.034 (n.s.); V4a −0.126;
V3b −0.039. The two significant PRIMARY effects (V3a positive, V4a negative) replicate on the timsTOF DEV set.

## Error transitions (PRIMARY; correct ⇔ tie-aware P(rank = 1) ≥ 0.5)
| Variant | wrong→correct | correct→wrong | both correct | both wrong |
|---|---|---|---|---|
| V2 fragment-only | 65 | 62 | 240 | 471 |
| **V3a 0.01 Da** | **84** | **37** | 265 | 452 |
| V3b 0.5 Da | 24 | 47 | 255 | 512 |
| V4a linear | 24 | 87 | 215 | 512 |
| V4b log | 57 | 24 | 278 | 479 |
| V5 ModCos | 90 | 55 | 247 | 446 |

By the category of B0's top-1 error: targets improved / worsened, and category MRR B0 → variant. These categories are defined
by B0's own errors, so improvements in failure categories are partly regression to the mean. The symmetric
transition counts above are the unbiased comparison.

| Variant | distant isomer (n=295) | near isomer (n=208) | unrelated (n=42) | B0-correct (n=293) |
|---|---|---|---|---|
| V3a 0.01 Da | 156/82, 0.173 → 0.308 | 60/36, 0.337 → 0.391 | 31/6, 0.252 → 0.497 | 0/36, 1.0 → 0.924 |
| V4b log | 153/74, 0.173 → 0.265 | 44/36, 0.337 → 0.371 | 21/8, 0.252 → 0.408 | 0/23, 1.0 → 0.951 |
| V5 ModCos | 155/89, 0.173 → 0.286 | 72/39, 0.337 → 0.434 | 24/6, 0.252 → 0.463 | 0/52, 1.0 → 0.886 |

V3a's net gain is concentrated in distant-isomer confusions (156 improved vs 82 worsened). ModCos is the only variant with a
clear net gain on near-isomer confusions (72 vs 39).

## Fingerprint oracle (PRIMARY, diagnostic only)
Coverage 87.8%; oracle MRR **0.958**, R@1 0.944 (EXP-013 old set: 0.996). Same-formula alternatives in 89.0% of pools.
At the fingerprint level the candidates remain almost fully separable, so the remaining gap is in the spectrum → structure mapping.

## Complementarity (PRIMARY, fixed 50/50 average rank; no retraining; weight not tuned)
| Model | MRR | R@1 | R@10 | same-formula MRR | isomer error rate |
|---|---|---|---|---|---|
| kNN (B0) | 0.507 | 0.359 | 0.790 | 0.466 | 0.199 |
| two-tower (EXP-013 D_hardneg) | 0.481 | 0.333 | 0.764 | 0.426 | 0.200 |
| **fusion 50/50** | **0.522** | **0.375** | **0.809** | **0.478** | **0.151** |

Fusion vs kNN: ΔMRR +0.016 [−0.004, +0.034]; ΔR@1 +0.016 [−0.011, +0.043]. Two-tower vs kNN ΔMRR −0.026 [−0.051, +0.000].
Correct-at-rank-1 overlap: kNN only 122, two-tower only 100, both 180, neither 436. Fusion fixes 92 and breaks 61 kNN results.
The errors are genuinely complementary (222 targets solved by exactly one model), but at equal weights the MRR gain is not
significant. No CI is available for the fusion isomer-error drop, because fusion per-target rows were not saved by the run.

## Leakage and audit status: VALID
1. Holdout: D ∩ COCONUT, seed 20260927 (sampling and query choice), deterministic build (`results/exp015/build.json`).
2. Disjointness: 0 accepted targets in EXP-011/012/013 target sets, 0 in the 243.
3. Coverage: 838/954 (87.8%).
4. R_noE180: 298,095 / 78,707, sha `ad8bb53e…`, equal to EXP-013's R_noE180 counts.
5. Leakage: 0 accepted targets with any flag (target / alias / parent in R_noE180 or two-tower training, previous target).
   15 byte-duplicate queries removed. COCONUT fingerprints indexed parse_ok-filtered (EXP-012 bug not reproduced).
6. B0 reproduction: feature re-derivation max |Δ| = 0.0 vs EXP-011 X_train (2,000 sampled references). DEV B0 = 0.5433.
7. Variant definitions match the design. The script was edited by the other agent at 23:51, before the final run, with two
   non-scientific fixes: (i) parameter key `"bin"` → `"bin_w"` (the earlier version would not have run the variants);
   (ii) B0 top-20 neighbours computed directly instead of sliced from the top-500 (only exact-tie ordering can differ).
8. All 9 variants + oracle + fusion executed (run.log). `run.err` contains only networkx import warnings.
9. Paired evaluation: identical pool hash (PRIMARY `f2c46e2f…`, DEV `2efae89b…`) and query hash for every variant.
   Paired bootstrap 2,000 resamples, seed 20260927.
10. Same-formula and isomer analysis present for every variant. 11. Oracle computed. 12. Fusion computed.
13. Schema: 838 PRIMARY and 243 DEV rows per variant in `per_query_results.csv` (10,810 rows). No NaN in rank/rr/R@k.
    isomer_error_rate is NaN exactly where no same-formula candidate exists.
14. No failed or partial jobs. B0 determinism gate passed.
15. Provenance: all reported numbers come from the 00:30 outputs of the run started 23:52, apart from `build.json`, `holdout_manifest.csv`,
    `primary_targets.parquet` and `ref_rids.npy` (22:02, same PID-4896 build) and `peaks.npz` (22:03). The `peaks_*.npy` caches
    (22:21, written by the earlier run) were verified byte-equal to `peaks.npz`. The earlier run's outputs (22:41) are kept
    under different names (`*_v1*`) and are not used; its MRRs differ from the final run by ≤ 0.00056 (tie ordering).

## Decision gate
- **GATE A: met, modestly.** A representation change (0.01 Da resolution) materially and significantly improves same-formula
  discrimination: overall +0.047 MRR, same-formula MRR +0.047, isomer error rate −0.020, all with CIs excluding 0.
  Log intensity and ModCos give smaller significant gains. The direction replicates on the timsTOF DEV set.
- Gate B evidence (secondary): complementary errors exist (222 exclusive solves; isomer error 0.199 → 0.151), but equal-weight
  fusion is not significant on MRR.
- Gate C: not supported.
→ **Investigate a learned spectrum encoder using the high-resolution representation**, keeping the kNN as the control
and the fusion as a candidate hybrid.

## Exact next experiment (EXP-016; pre-registered, no selection on any evaluation set)
Held-out evaluation on **fresh holdout #2**: the 1,513 D ∩ COCONUT molecules with spectra not sampled for EXP-015
(disjoint from EXP-015 PRIMARY, which has now informed the choice of representation). Arms, all fixed in advance:
1. kNN, 2×2 factorial on the 0.01 Da representation: {√, log} intensity × {with, without} neutral losses
   (V3a+log was never tested together; V2 showed neutral losses matter for isomers).
2. Two-tower retrained with EXP-013's protocol and hard negatives, spectrum tower on the 0.01 Da + √ representation
   (the only change), compared with the EXP-013 two-tower.
3. Fixed 50/50 fusion of arm-1 (0.01 Da, √, NL; the EXP-015 winner) and arm-2, reporting per-target rows so that
   CIs can be given for the fusion isomer-error rate.
Also report a timsTOF-only stratum and keep the 243 as a secondary check. No Kaggle submission.

## Files used for every reported number
`research/analysis/exp015/{variant_results.csv, baseline_results.csv, per_query_results.csv, fusion_results.csv,
oracle_results.json, leakage_audit.json, summary.json, holdout_manifest.csv, run.log}`;
`results/exp015/{build.json, primary_targets.parquet, ref_rids.npy, peaks.npz, peaks_*.npy}`;
cross-checks: `results/exp013_np_forensics/C_report.json`, `results/exp011_class2_proxy/{targets.parquet, X_train.npz}`,
`results/exp012_coconut/target_aliases.json`. Not used: `*_v1*` files (earlier superseded run).

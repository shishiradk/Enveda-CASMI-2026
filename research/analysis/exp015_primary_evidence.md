# EXP-015 Frozen Evidence Package

Status: **EXPERIMENT FROZEN.** All numbers below are read verbatim from
`research/analysis/exp015/*` of the final VALID run. Do not re-run, re-rank, or
re-train on this corpus. Feed this package into the EXP-014 joint decision for
EXP-016.

Full run details: `research/analysis/exp015_spectrum_representation_report.md`.
Read-only consistency audit: 70/70 PASS, `research/analysis/exp015/verification.txt`
(rerunnable via `research/scripts/exp015_verify.py`, ~10 s, no expensive steps).

Population: fresh 954-ok/838-covered targets holdout (sampled 1,000 from 2,513
eligible; disjoint from all prior experiments), 3,147 query spectra,
reference R_noE180 = 298,095 spectra / 78,707 molecules. All CIs are paired
per-query bootstrap (10,000 draws), `[mean, lo, hi]`, vs the frozen B0 baseline
`CURRENT = {bin_w: 0.1, nl: True, inten: sqrt}`. B0 DEV MRR 0.5433 reproduces
EXP-013's 0.543 (determinism gate: `B0_deterministic: true`).

---

## 1. Primary evidence table (PRIMARY / SECONDARY)

Levels marked **PRIMARY** were pre-registered single-factor variations of the
frozen CURRENT spec — no DEV feedback. Levels marked **SECONDARY** were composed
from EXP-013 243-target DEV rankings (`best_nl=False, best_bin=0.01,
best_intensity=log`) and are reported for transparency only; they are excluded
from all primary conclusions.

| evidence | variant | change vs B0 | MRR | dMRR [lo, hi] | R@1 | R@10 | iso_err | iso d [lo, hi] | sf MRR |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | B0_current | frozen baseline | 0.5068 | — | 0.3587 | 0.7895 | 0.1988 | — | 0.4661 |
| PRIMARY | V3a_bin0.01 | bin 0.01 Da | 0.5536 | **+0.0468** [0.0291, 0.0647] | 0.4152 | 0.8133 | 0.1784 | −0.0205 [−0.0326, −0.0087] | 0.5131 |
| PRIMARY | V4b_log | log1p intensity | 0.5383 | **+0.0315** [0.0167, 0.0459] | 0.3980 | 0.8046 | 0.1805 | −0.0183 [−0.0287, −0.0081] | 0.4981 |
| PRIMARY | V5_modcos_rerank | ModCos top-20 | 0.5415 | **+0.0347** [0.0149, 0.0550] | 0.4010 | 0.8001 | 0.1851 | −0.0137 [−0.0266, 0.0000] | 0.5047 |
| PRIMARY | V2_fragment_only | no neutral losses | 0.5014 | −0.0054 [−0.0249, +0.0136] | 0.3636 | 0.7666 | 0.2144 | **+0.0156 [0.0015, +0.0297]** | 0.4593 |
| PRIMARY | V3b_bin0.5 | bin 0.5 Da | 0.4769 | −0.0299 [−0.0440, −0.0158] | 0.3315 | 0.7655 | 0.2192 | +0.0204 [0.0102, 0.0314] | 0.4389 |
| PRIMARY | V4a_linear | linear intensity | 0.4265 | −0.0803 [−0.0985, −0.0626] | 0.2847 | 0.7150 | 0.2608 | +0.0620 [0.0486, 0.0771] | 0.3879 |
| SECONDARY | C2_bestNL_bestBin_bestInt | bin0.01+noNL+log | 0.5478 | +0.0410 [0.0199, 0.0618] | 0.4076 | 0.8048 | 0.1835 | −0.0153 [−0.0306, −0.0006] | 0.5045 |
| SECONDARY | C3_C2_plus_modcos | C2 + ModCos | 0.5472 | +0.0404 [0.0201, 0.0590] | 0.4113 | 0.8004 | 0.1917 | −0.0072 [−0.0217, +0.0072] | 0.5087 |
| SECONDARY | C1_bestNL_bestInt | noNL+log | 0.5382 | +0.0314 [0.0111, 0.0502] | 0.4000 | 0.7874 | 0.1883 | −0.0105 [−0.0246, +0.0033] | 0.4958 |

Sorted machine-readable copy: `research/analysis/exp015/primary_evidence_table.csv`.

**Clean statements (no overstatement):**
- Binning at **0.01 Da raises PRIMARY MRR by +0.047 and R@1 by +0.057, both
  significant, and lowers isomer-error by 2.1 pp (CI excludes 0)**. 0.5 Da and
  linear intensity are significantly worse.
- Log intensity is a further significant +0.031 MRR.
- ModCos rerank gains **+0.035 MRR (significant)**; its iso-error CI touches 0,
  so do not claim an isomer benefit for it.
- **Neutral-loss removal is the DECEPTIVE finding**: DEV MRR +0.028 (best on
  EXP-013 targets) but PRIMARY MRR neutral (−0.005, NS) — and PRIMARY
  isomer-error gets *significantly worse* (+1.6 pp). The EXP-013 spread did not
  transfer; NL features matter for isomer resolution.
- SECONDARY rows confirm direction but are DEV-composed; their apparent
  iso-benefit for C2 (CI excludes 0) is not additional independent evidence.

---

## 2. Same-formula evidence (includes near/distant isomers)

`same_formula_prevalence` on PRIMARY = 746/838 (89.0%). All deltas below are
within-pool (same neutral formula) and therefore isomer-resolution evidence.

Isomer-error rate (truth not ranked #1 among same-formula candidates):

| variant | iso_err | Δ [lo, hi] |
|---|---|---|
| B0_current | 0.1988 | — |
| V3a_bin0.01 | 0.1784 | **−0.0205 [−0.0326, −0.0087]** |
| V4b_log | 0.1805 | **−0.0183 [−0.0287, −0.0081]** |
| V5_modcos_rerank | 0.1851 | −0.0137 [−0.0266, 0.0000] |
| C2 (SECONDARY) | 0.1835 | −0.0153 [−0.0306, −0.0006] |
| V2_fragment_only | 0.2144 | **+0.0156 [+0.0015, +0.0297]** |

MRR by B0-top1 category, and B0→variant transitions (PRIMARY, rank-1):

| category (by B0 top1) | n | B0 MRR | V3a MRR | V5 MRR | V4b MRR | V2 MRR |
|---|---|---|---|---|---|---|
| correct | 293 | 1.000 | 0.924 | 0.886 | 0.951 | 0.862 |
| distant isomer | 295 | 0.173 | **0.308** | 0.286 | 0.265 | 0.251 |
| near isomer | 208 | 0.337 | 0.391 | **0.434** | 0.371 | 0.377 |
| unrelated | 42 | 0.252 | **0.497** | 0.463 | 0.408 | 0.363 |

| B0→variant | wrong→correct | correct→wrong | both correct | net |
|---|---|---|---|---|
| V3a_bin0.01 | 84 | 37 | 265 | **+47** |
| V5_modcos_rerank | 90 | 55 | 247 | +35 |
| V4b_log | 57 | 24 | 278 | +33 |
| V2_fragment_only | 65 | 62 | 240 | +3 |

Interpretation (observed, not CI-backed for the by-category split): gains are
concentrated in pools where B0 was wrong — 0.01 Da binning roughly **doubles
distant-isomer and unrelated-candidate MRR** (0.17→0.31, 0.25→0.50) while
spoiling only 36/293 previously-correct pools; removal of neutral losses (V2)
helps the same confused pools less and harms prior correct answers more, which
is why its overall Δ is neutral yet the isomer CI is significantly worse.

---

## 3. Fusion (kNN_B0 + EXP-013 hard-negative two-tower) — complementarity only

Computed once on PRIMARY with the frozen EXP-013 tower; fixed 50/50 average
rank. **Not proof of a better model**: both ΔMRR and ΔR@1 CIs cross zero.

| model | MRR | R@1 | iso_err |
|---|---|---|---|
| kNN_B0 | 0.5068 | 0.3587 | 0.1988 |
| two_tower (alone) | 0.4811 | 0.3330 | 0.2002 |
| fusion_50_50 | 0.5223 | 0.3749 | 0.1513 |

Paired deltas vs kNN_B0:

| metric | Δ [lo, hi] | reading |
|---|---|---|
| MRR two_tower | −0.026 [−0.051, +0.000] | alone no better than kNN |
| MRR fusion | +0.0155 [−0.0035, +0.0338] | crosses 0 → supports complementarity, not superiority |
| R@1 fusion | +0.0162 [−0.0113, +0.0427] | crosses 0 → same |

Correct-overlap at R@1: kNN-only 122, two-tower-only 100, both 180, neither 436.
Net R@1 fixes vs kNN: **92 fixed, 61 regressed (+31)**.

Isomer-error transition (observed, no CI): 0.199 (kNN) → 0.151 (fusion), 4.8 pp
better; treat as a hypothesis for EXP-016, not a claimed effect. The two
correct-sets overlap only partly (kNN correct 302, two-tower correct 280, both
180): 122 pools only-kNN and 100 only-two-tower, so ~40% of their correct
answers are unique to one ranker — the raw material a principled fusion would
exploit.

---

## 4. Oracle freeze — structural-separability ceiling

The "fingerprint oracle" ranks each pool's candidates by CF-Tanimoto similarity
to the query's exact class fingerprint (perfect molecular knowledge, no spectral
distance). It is an upper bound on *inter-class separability*, NOT an achievable
performance target and NOT a submitted-mode result.

| metric | value |
|---|---|
| coverage | 0.878 (838/954) |
| MRR | 0.9576 |
| R@1 | 0.9443 |
| same_formula_prevalence | 0.890 (746/838) |
| median pool size | 28.5 |

Reading (frozen): within a ±5 ppm mass window, when the exact correct
fingerprint is known, the truth is uniquely rank-1 in 94% of pools — i.e. the
*structural* confusability is small. The gap to our 35.9% R@1 is dominated by
*spectral-feature* failure to recover that fingerprint, which is exactly the
problem arena for EXP-016.

---

## 5. Leakage & internal-consistency verification (frozen)

Read-only audit, all re-derivable from stored outputs (`verification.txt`,
`exp015_verify.py`):

- **Fresh holdout / zero overlap**: 1,000 targets sampled seeded from 2,513
  fresh-eligible (coalesced over EXP-011/12/13 parent molecules); disjoint from
  EXP-011/12/13 targets; zero overlap between PRIMARY and DEV; ok-954 set
  carries no leak flags; 43/2/1 targets excluded for reference/two-tower leaks; 1
  no-query-after-dedup; 15 duplicate query spectra dropped.
- **R_noE180 identity**: census (298,095 spectra / 78,707 molecules) re-derived;
  byte-sha256 `ad8bb53e…` matches `build.json` and `leakage_audit.json`.
- **COCONUT indexing**: `coconut_fp_packed.npy` rows == parse-ok filtered rows;
  `parse_ok-filtered; EXP-012 bug not reproduced`; feature reproduction max abs
  diff 0.0 (B0-identical trajectories on DEV vs EXP-013).
- **Pool / query hashes**: `pool_hash_PRIMARY` and `query_hash_PRIMARY`
  re-derived from the ±5 ppm mass-window pools and ok query rid sets — identical
  to audit.
- **No NaN / partial jobs**: NaN appears only in legitimate positions (B0 empty
  deltas; per-query `isomer_error_rate` on rows without a same-formula
  candidate); `run.log` completes, `run.err` and `build.log`/`peaks.log`
  traceback-free; 70/70 consistency checks PASS and exit code 0.
- **Determinism**: `B0_deterministic: true` (top-20 via the identical K_NN path;
  the earlier v1 run's determinism-gate mismatch is documented and superseded —
  see `*_v1_*` backups; final numbers supersede v1).

No EXP-013/014 files were modified. No runs were launched to produce this package.
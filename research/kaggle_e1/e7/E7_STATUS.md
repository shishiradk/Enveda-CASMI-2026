# E7 status (Task A) — 2026-10-06

E7 = E6 list, ranks 1-3 unchanged; ranks 4-8 = first 5 `rr_bt` candidates (biotransform+mmp_edit, biotransform
first) whose metric key is not in the E6 top-25; then E6 ranks 4+; de-dup on metric key; 25 per row. Ungated.
**Build-ready; not pushed (the user pushes).** Dry run complete, all visible checks pass.

## Blocker found and fixed: the work budgets were far too tight

**FACT.** `W_GEN=1.6e5` truncated **519/550** C3NP bench molecules, `W_S2=3.0e6` truncated 74/550 (measured
against `results/c3gen/e7/bt_calib_nolimit_dump.pkl`). On rows 0-59 that gave MRR@25 **0.5746** vs the 0.7335
target (hit@1 34 vs 41, hit@200 39 vs 52). The unlimited calibration run truncated 0/550 and observed:
max `W_GEN` 1,047,477 | max `W_S2` 22,405,392 | max `t_gen` 20.2 s | max `t_total` 39.0 s.
**Fix** (`biotransform.py:86-101` + shipped `dataset/biotransform.py`; `e7_c3_channel.py:38` matches):
`W_GEN -> 2.0e6`, `W_S2 -> 4.0e7` (~1.9x the observed maxima, so they bind only on a pathological input),
`T_GEN/T_TOTAL 25/35 -> 90/120` s (failsafes only). A comment records the 0.159 MRR cost of tightening these.

## Step 2 — determinism verified

`biotransform.py --c3np --n 60 --offset 0 --workers 2`, run three times (`bt_o0_runA/B.json`, `bt_o0_check.json`):

| Check | Result | Target |
|---|---|---|
| Output SHA-256, runA / runB / check | all `6571A875…C821` | identical |
| MRR@25 (`c3np_eval.py --only_present`) | **0.7335**, ci999 [0.556, 0.896] | 0.7335 +/- 0.01 |
| hit@1 / hit@25 / hit@200 / mean_cand | 41 / 50 / 52 / 181.9 | match the r2 reference |
| `truncated` (work/time/cap/stage2), `guard_fired` (gen/all), `\|guard=` in provenance | all 0 | 0 |

**FACT.** With no guard fired and no `|guard=` tag anywhere, the output does not depend on machine speed.

## Step 3 — dry run (`results/kaggle_e7_dry/dry_report.json`; `ctx`/`ana` already complete, 400/400, full context)

| Item | Value |
|---|---|
| Stage `c3` runtime / molecules | 1204 s / 400 (2 workers) |
| `sec` per molecule mean / max | 5.99 / 11.98 |
| Lists / failures / missing | 399 / **1** / 0 |
| `budget_hit` / `coco_ok` / `bt_guard` / `mmp_guard` | False / True / 0 / 0 |

**FACT.** Visible check, all pass: top-3 identical to `e6_lists.json` **400/400**; 25 unique SMILES per row **400/400**;
25 unique **metric keys** per row **400/400**; rows with Class-3 insertions **399** (exactly 5 each, at ranks 4-8
and nowhere else); unparsable SMILES 0; `n_post` 0; E6 ranks 21-25 pushed past 25 per row (by design);
`visible_mrr_e6` / `visible_mrr_e7` = 1.0 / 1.0.

**INFERENCE.** The visible MRR cannot show an E7 gain — the truth already sits at rank 1, so insertions at
ranks 4-8 cannot move it. The check proves E7 does no harm, not that it helps.

**FACT.** The 1 failure is a single RDKit `Range Error` (`ROMol.cpp:202`, `22 < 22`) in biotransform's hydrolysis
branch for `m_0d08be`; caught per molecule, so that row keeps its E6 list (the designed fallback). Not caused by
the budget change; left alone deliberately, since a fix would invalidate the verified 3-run determinism check
for 0.25% of rows.

## Step 4 — build + dataset completeness

`build_e7.py` (idempotent) produces `casmi_e7.ipynb` (6 cells, all compile) and `kernel-metadata.json`
(`shishiradhikari11/casmi-e7-c3-channel`, private, T4, competition `enveda-CASMI26-molecule-id-mass-spectra`).
Grepping `open(` / `np.load` / `read_parquet` in the shipped code: **all 15 files present** in `dataset/`
(7 assets + 6 code + README + metadata); COCONUT/PubChem reads come from other `dataset_sources`. Bench-only
paths (`np_pool.parquet`, `biotx_pool_np.npy`, `RULES_PATH`) are correctly bypassed in deploy mode, and the
`fp_*.pt` glob cannot pick up our nets, which are named `e6net_*`. Step 5: `UPLOAD_STEPS.md`, dataset
**655.9 MB / 15 files** (588.9 MB of it `c3_pool_fp.npy`; `__pycache__/` adds 0.18 MB of dead `.pyc`).

## Other changes
- `e7_c3_channel.py`: pre-flight asset check. **FACT.** A missing asset raises inside the Pool initializer and
  `multiprocessing.Pool` then respawns the worker forever, so the run *hangs* rather than failing (it hung 590 s
  until I killed it, racing a second run over the same outputs). Now it fails fast with `assets_missing`.
- `parity_e7.py:151`: fixed `TypeError: dict() got multiple values for keyword argument 'sec'`. Deploy parity,
  20 bench molecules (`parity_deploy.json`): 0 guards, 0 truncated, `n_ana` 50/50, `n_win` 200 (19) / 164 (1),
  `n_rr` 200/20, top-25 overlap with bench `rr_bt` 17-25 (mean 23.2). `bt_same_as_e7_full` **not** checked —
  `results/c3gen/e7/bt_full_e7_c*.json` does not exist.
- **Disk:** D: was 100% full (0 bytes free), blocking all writes. With the user's approval, `proxmox-ve_9.2-1.iso`
  (1.59 GB) and `OllamaSetup.exe` (1.30 GB) were **moved** to
  `C:\Users\LENOVO\AppData\Local\Temp\opencode\d_drive_installers`; D: now has 2,297 MB free. Nothing deleted.
  An earlier dedupe attempt on `results/kaggle_v2_pubchem/pubchem_rows.parquet` vs
  `external/pubchem/pubchem_rows.parquet` was a **no-op — already hardlinked**; both remain valid and
  byte-identical (MD5 `E4FC8787…45A`).

## Expected LB effect

**INFERENCE.** DEC-010 predicted +0.01 to +0.03 (central) against a 0.0037 LB cost from displacing ranks 4-8.
E6 delivered about half its predicted gain, so apply a 2x haircut: **+0.005 to +0.015**, not closing 0.432.

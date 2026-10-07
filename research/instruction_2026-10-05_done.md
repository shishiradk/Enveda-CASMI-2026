# opencode instructions (2026-10-05, written by Claude before its token budget ran out)

Project: Kaggle "Enveda CASMI 2026 — Molecule ID From Mass Spectra", repo `D:\Enveda-CASMI-2026` (Windows 11, Git Bash,
Python 3.14 with rdkit, duckdb, pandas, numpy, torch). Metric: MRR@25 on tautomer-canonical InChIKey14.
Best prize-eligible public LB: **0.360 (E6)**. Prize line (5th place) is about 0.432.

**Read first:** `research/decision_log.md` (DEC-008, DEC-009 + CORRECTION, DEC-010), `research/analysis/c3gen_tournament.md`.

## Rules for every task
- Label every claim **FACT** (you measured it: give the command and the number) or **INFERENCE**.
- **Prize eligibility is mandatory.** Never use Ahmed Berat Ozer's Kaggle datasets, FRIGID weights, NIST or any
  non-commercial data or weights. Allowed: train.parquet and models trained on it, PubChem, COCONUT, RDKit, our code.
- **RAM:** 16 GB machine shared with other apps. Keep each process under 1.5 GB and use at most 2 worker processes.
  Never load `train.parquet` or `external/pubchem/pubchem_rows.parquet` fully into pandas (use duckdb with
  `SET memory_limit='1GB'`).
- Do not delete files. Do not push to Kaggle, upload or submit (the user does that). Do not run `.bat` files.
  Do not kill processes you did not start.
- Each task writes **only** to the files it lists, so tasks A, B and C can run at the same time.
- Finish with a short report file (path given in the task). Keep it ≤ 80 lines and put numbers in tables.

---

## TASK A — finish the E7 package (Class-3 channel) — opencode app, `opencode/big-pickle`
**Goal:** E7 = E6 list, ranks 1-3 unchanged; ranks 4-8 = the first 5 `rr_bt` candidates (round-robin of
`research/c3gen/biotransform.py` and `research/c3gen/mmp_edit.py`, biotransform first) whose metric key is not
already in the E6 top-25; then E6 ranks 4+; de-duplicated; exactly 25 per row. If the Class-3 channel fails in any
way, the E6 list is kept.

**State (a Claude agent was stopped mid-work at about 18:00):**
- Already written: `research/kaggle_e1/e7/` (`build_e7.py`, `casmi_e7.ipynb`, `e7_c3_channel.py`, `dryrun_e7.py`,
  `dataset/`).
- Local dry-run context stage done: `results/kaggle_e7_dry/` (`ctx.log`, `ctx_check.json`, `e6_ctx.pkl`, `eng_ana.json`).
- It was measuring biotransform's runtime without the time guard: `results/c3gen/e7/bt_calib_nolimit.*`. That run was
  killed, so it may be incomplete.

**Steps:**
1. Read the four files in `research/kaggle_e1/e7/` and `results/kaggle_e7_dry/ctx.log` and `ctx_check.json`. Write
   down what works and what is missing.
2. **biotransform time guard:** the wall-clock time guard must not decide the output (results must not depend on
   machine speed). Mirror what `mmp_edit.py` does (search it for `guard`): deterministic work budgets as the real
   limit, and wall-clock only as a hard failsafe that tags the provenance with `|guard=...`. Check: run
   `python research/c3gen/biotransform.py --c3np --n 60 --offset 0 --workers 2 --out results/c3gen/e7/bt_o0_check.json`,
   twice. Both outputs must be identical, and MRR@25 (from `python research/scripts/c3np_eval.py <json> --name x --only_present`)
   must be within ±0.01 of 0.7335.
3. Finish the dry run: `python research/kaggle_e1/e7/dryrun_e7.py` (read its docstring for the arguments). Report:
   - runtime;
   - number of failures;
   - **visible check:** top-3 identical to `results/kaggle_e6/e6_lists.json` on 400/400 rows, 25 unique SMILES per
     row, and the number of rows that got Class-3 insertions.
4. Run `python research/kaggle_e1/e7/build_e7.py`. It must produce `casmi_e7.ipynb`, `kernel-metadata.json` and the
   `dataset/` folder. Check that the dataset folder holds every file `e7_c3_channel.py` opens; grep it for `open(`,
   `np.load` and `read_parquet`.
5. Write `research/kaggle_e1/e7/UPLOAD_STEPS.md`: the exact `kaggle datasets create -p ...` and
   `kaggle kernels push -p ...` commands for the user's account `shishiradhikari11`, plus the dataset size.

**Report:** `research/kaggle_e1/e7/E7_STATUS.md`.
**Files you may change:** `research/kaggle_e1/e7/*`, `research/c3gen/biotransform.py`, `results/c3gen/e7/*`,
`results/kaggle_e7_dry/*`.

---

## TASK B — Class-2 (PubChem-only) re-ranker write-up — opencode CLI, `opencode/big-pickle`
**Goal:** E6 puts the right answer at rank 1 for only 37/213 molecules of the PubChem-only (PC) bench bucket
(MRR 0.309). A stopped Claude agent built experiments that seem to raise PC MRR to about 0.44-0.63. Verify them and
write them up.

**Existing files:**
- Scripts: `research/scripts/c2gap_extract.py`, `c2gap_rerank.py`, `c2gap_gate.py`, `c2gap_ablate.py`.
- Outputs in `results/c2gap/`: `gate.log`, `gate.json`, `rerank.json`, `ablate.json`, `pop.json`, `merge.json`,
  `ens_window.json`, `ho2.log`.

**Steps:**
1. Read the four scripts and the json/log outputs. Make a table: experiment → MRR per bucket (SV, S1, S2, PC), and the
   LB estimate using calibration weights SV 0.146, S1 0.21, PC 0.111 (S2 weight 0). The gain on a bucket is that
   bucket's ΔMRR times its weight.
2. **Leakage audit (most important).**
   - Is the re-ranker trained and evaluated with cross-validation **grouped by molecule**?
   - Does any feature use the truth: rank of the truth, `is_truth` columns, or popularity computed with the truth key?
   - Do the nets that give the scores (ho1 / ho2) hold out the molecules being scored?
   - Rerun `c2gap_ablate.py` if it did not finish (≤ 30 min, `timeout 1800`).
3. **Eligibility of each feature.** Popularity (PubChem patent and literature counts) is public-domain PubChem data,
   so it is allowed. Name the source file of every feature.
4. Recommend: the feature set, whether to gate, and the expected LB gain (INFERENCE, with a 2x haircut because E6
   delivered half of its predicted gain). Add the steps to put it into a Kaggle kernel (E8 = E7 + re-ranked PubChem
   channel).

**Report:** `research/analysis/c2_gap_diagnosis.md`.
**Files you may change:** `research/scripts/c2gap_*.py`, `results/c2gap/*`, that report.

---

## TASK C — review of TASK A's package (read-only) — CLI with `google/gemini-3-flash-preview`
**Run only after TASK A wrote `E7_STATUS.md`.** It has a 20-request budget, so be efficient. Read
`research/kaggle_e1/e7/e7_c3_channel.py`, `build_e7.py` and the merge cell in `casmi_e7.ipynb`. Look for:
1. Paths that won't exist on Kaggle.
2. A glob that could pick up the wrong `.pt` files. The engine loads `fp_*.pt`, so our nets are named `e6net_*`.
3. Missing try/except around the Class-3 stage, so that a failure would not keep the E6 lists.
4. Top-3 being changed.
5. Duplicates by metric key.
6. Rows with fewer than 25 entries.
7. Any non-eligible input.

**Report:** `research/analysis/e7_review_gemini.md`.

---

## TASK B addendum (optional, only if TASK B finishes early)
CFT-ho2 (`models/cft_ho2_akriti/cft_ho2.pt`, S4 held out) is now available. If `c2gap` features include a CFT score,
check that the CFT used holds out the molecules being scored. `models/cft_kaggle` was trained without the ho1
holdout, so it is LEAKY on S1/S2/S3/PC. CFT-ho2 is leak-free only on S4.

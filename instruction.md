# opencode instructions — 2026-10-07 (Claude is offline until tomorrow)

Project: Kaggle "Enveda CASMI 2026 — Molecule ID From Mass Spectra", repo `D:\Enveda-CASMI-2026` (Windows 11, Git Bash,
Python 3.14 with rdkit, duckdb, pandas, numpy, torch, lightgbm). Metric: MRR@25 on tautomer-canonical InChIKey14.
Best prize-eligible public LB: **0.360 (E6)**. The 5th-place prize line is about 0.432.
The previous round (TASK A = E7 build, TASK B = PubChem re-ranker) is **done**. Its instructions are archived in
`research/instruction_2026-10-05_done.md`.

**Read first:** `research/decision_log.md` (DEC-010 and both addenda, at the end of the file),
`research/kaggle_e1/e7/E7_STATUS.md`, `research/analysis/c2_gap_diagnosis.md`, `research/analysis/c3gen_tournament.md`.

## Rules for every task
- Label every claim **FACT** (you measured it: give the command and the number) or **INFERENCE**.
- **Prize eligibility is mandatory.** Never use Ahmed Berat Ozer's Kaggle datasets, FRIGID weights,
  `prvsiyan/chebi-lipidmaps-casmi26`, NIST, or any other non-commercial data or weights. Allowed: train.parquet and
  models trained on it, PubChem, COCONUT, RDKit, our own code, and the CC0/CC-BY sources listed in
  `research/analysis/fork_licence_rebuild_plan.md` (table rows marked "keep").
- **RAM:** the machine has 16 GB, shared with other apps. Keep each process under 1.5 GB and use at most 2 worker
  processes per task. Never load `train.parquet` or `external/pubchem/*.parquet` fully into pandas. Use duckdb with
  `SET memory_limit='1GB'`.
- **DISK: D: has only about 2.3 GB free.** Check it before every large write (`df -h /d`). Stop and report if it
  falls below 1 GB. Put large scratch files (>200 MB) in `C:\casmi_tmp\` (37 GB free), not on D:.
- **Never write into `results/c3/e6_work/`.** Every c2gap result depends on the ho1 caches stored there.
- Do not delete files. Do not push to Kaggle, upload or submit (the user does that). Do not run `.bat` files.
  Do not kill processes you did not start.
- Each task writes **only** to the files it lists, so the tasks can run at the same time.
- Finish with a short report file (path given in the task). Keep it ≤ 80 lines and put numbers in tables.

## Schedule for the day

| When | Who / tool | Task | Status |
|---|---|---|---|
| Morning, in parallel | CLI `google/gemini-3-flash-preview` | **TASK C**: review the E7 package (read-only, ~15 requests) | **DONE** (no blockers, `e7_review_gemini.md`) |
| Morning, in parallel | opencode app, `opencode/big-pickle` | **TASK D**: E8 go/no-go, re-score SV with the deploy nets | **DONE: NO-GO** (0.9209 < 0.9211 floor, `e8_sv_gonogo.md`) |
| Morning, in parallel | opencode CLI, `opencode/big-pickle` | **TASK F**: seed-Tc-aware Class-3 re-ranker (bench research) | **DONE: RECOMMEND** (folded to E9, `c3_seedtc_reranker.md`) |
| After C reports no blocker | **Shishir / Agent** | Push and submit E7 (see "Shishir's checklist" below) | **SUBMITTED** (id `56883907`, pending LB) |
| After D reports **GO** | opencode app, `opencode/big-pickle` | **TASK E**: build the E8 assets (ranker + compact popularity map) | **CLOSED** (gated on D=GO; D confirmed NO-GO) |

Prompt for each: `Do TASK C in instruction.md` (or D, E, F).

---

## Shishir's checklist (by hand, not opencode)
1. **Wait for TASK C.** If `research/analysis/e7_review_gemini.md` names a blocker, do not push. Leave it for Claude.
2. **Push E7.** Follow `research/kaggle_e1/e7/UPLOAD_STEPS.md`: first `kaggle datasets create -p research/kaggle_e1/e7/dataset`
   (656 MB), then `kaggle kernels push -p research/kaggle_e1/e7`. Delete `research/kaggle_e1/e7/dataset/__pycache__/`
   first if you want a clean upload.
3. **Check the log** against the pass table in UPLOAD_STEPS.md §3. You need top-3 unchanged 400/400, 25 unique per row
   400/400, about 399 rows with C3 insertions, `bt_guard` and `mmp_guard` both 0, and `coco_ok` True.
   If all pass, submit with the description
   `E7: E6 + Class-3 rr_bt block at ranks 4-8 (biotransform+mmp_edit), ungated. Prize-eligible.`
4. **Akriti's Kaggle seed nets** (`akritirijal04/casmi-fp-train-full-s10` and `-s20`, started 2026-10-06, about 9 h each).
   If they have finished, download them with her key, as in `research/JOBS_2026-10-05.md` §2, to `models/fp_fulls10`
   and `models/fp_fulls20` (about 280 MB each, so check D: space first).
   Each `check_summary.txt` must say `RESULT: ALL OK`. If a run stopped at 660 min, re-run it; it resumes.
5. Akriti's laptop ensemble (35–45 h) is not due yet. Nothing to do.

---

## TASK C — review the E7 package before it is pushed (read-only) — CLI, `google/gemini-3-flash-preview`
The free tier allows 20 requests per day, so read each file once and do not wander.
Read `research/kaggle_e1/e7/e7_c3_channel.py`, `build_e7.py`, `e7_merge.py`, `kernel-metadata.json`, and the merge
cell plus the last cell of `casmi_e7.ipynb`. Look for:
1. Paths that won't exist on Kaggle (`/kaggle/input/<slug>/...` must match a `dataset_sources` entry).
2. A glob that could pick up the wrong `.pt` files. The engine loads `fp_*.pt`, so our nets are named `e6net_*`.
3. A missing try/except around the Class-3 stage or the merge. Any failure must fall back to the E6 lists.
4. Any code path that could change ranks 1-3.
5. Duplicates by metric key, or rows with fewer or more than 25 entries.
6. **Eligibility of every `dataset_sources` entry.** Compare each against the table in
   `research/analysis/fork_licence_rebuild_plan.md`. `prvsiyan/chebi-lipidmaps-casmi26` must **not** appear
   (`shishiradhikari11/casmi-bio-clean` is our CC-BY rebuild). Flag anything not covered by that table.
7. Wall-clock limits that could make the output depend on machine speed (`time.time()`, `perf_counter` used in a
   decision rather than only in logging).

Rate each finding **BLOCKER** (do not push), **FIX-LATER**, or **OK**. Give the file and line number.
**Report:** `research/analysis/e7_review_gemini.md`. **Files you may change:** that report only.

---

## TASK D — E8 go/no-go: re-score SV with the deploy nets — opencode app, `opencode/big-pickle`
**Why:** the PubChem re-ranker (`ho1+struct+pop`, report `research/analysis/c2_gap_diagnosis.md`) was fitted on scores
from the held-out **ho1** nets. E6/E8 deploy the **full-data** nets, so the features `score_gap`, `score_z` and
`score_rk` will shift. SV (the visible-test bucket) is the only bucket the full nets have not trained on, so it is
the only honest check.

**FACT (Claude checked on 2026-10-06):** `models/fp_full/fp_merged_full.pt` and `fp_single_full.pt` are byte-identical
(SHA-256) to the deployed `research/kaggle_e1/e6/dataset/e6net_*_full.pt`, so use `--nets models/fp_full`.

**Steps:**
1. `research/scripts/e6_pc_channel.py` hard-codes `WORK = results/c3/e6_work` (line 20) and would **overwrite the ho1
   score cache**. Add a `--work` argument whose default is the old path, so the default behaviour does not change.
   Do the same for `research/scripts/c2gap_extract.py` (it reads `W = results/c3/e6_work`). Use `results/c2gap_full/work/`
   as the new work dir. It needs `windows_5.0ppm.parquet` (234 MB): **hardlink** it there (`ln`, which takes no disk
   space) rather than copying it. Check `df -h /d` after.
2. Score with the full nets: `python research/scripts/e6_pc_channel.py --nets models/fp_full --workers 2 --work results/c2gap_full/work`
   (read its docstring first; you may need only the scoring stage). Report the runtime.
3. Re-extract SV with the new scores into `results/c2gap_full/` (`--scens SV`, with a tag so nothing in
   `results/c2gap/` is overwritten).
4. Apply the re-ranker exactly as in `c2gap_rerank.py` with `SETS["ho1+struct+pop"]`, **trained on the bench PC+S2
   rows with ho1 scores** (as shipped) and **applied to SV rows with full-net scores**. Then the E6 merge, unchanged.
5. Report this table for SV (180 molecules): E6 as shipped (0.9211); full-net channel without re-rank; full-net
   channel with re-rank. Add the bootstrap CI of (re-rank − no re-rank).
6. **Decision rule:** **GO** if re-ranked SV merged MRR ≥ 0.9211 **and** the CI lower bound of (re-rank − no re-rank)
   is ≥ −0.005. Otherwise **NO-GO**. If NO-GO, also try a re-ranker refit on rank-only score features (drop raw `score`,
   keep `score_rk` and `score_z`, which are less sensitive to the net) and report whether that passes.

**Report:** `research/analysis/e8_sv_gonogo.md`, with GO or NO-GO on the first line.
**Files you may change:** `research/scripts/e6_pc_channel.py` and `c2gap_extract.py` (only the `--work` argument),
new scripts `research/scripts/e8_*.py`, `results/c2gap_full/*`, that report.

---

## TASK E — E8 assets: final ranker + compact popularity map — opencode app, `opencode/big-pickle`
**Run only after TASK D says GO** (or after its rank-only variant passes; then use that feature set).
Claude will wire E8 into the kernel tomorrow. Your job is to get the assets ready and measured.

**Steps:**
1. **Final ranker.** Fit the lambdarank booster on **all** bench PC+S2 molecules with the feature set TASK D approved
   (2 workers, 200 rounds, `lambdarank_truncation_level=25`, fixed seed). Save it as text:
   `research/kaggle_e1/e8/dataset/c2_ranker.txt`, plus `c2_ranker_features.json` (ordered feature names, LightGBM
   version, seed). Refit twice and confirm the two text files are byte-identical.
2. **Find out how the Kaggle PubChem channel reads candidates.** Read `research/kaggle_e1/e6/` (the PubChem stage) and
   the `casmi-v2-pubchem` dataset source. Which file, which columns, how many rows? Does every row carry an
   InChIKey14? Report it.
3. **Compact popularity map.** The source `external/pubchem/pubchem_rows_pop.parquet` is 2.6 GB with 101M rows. The
   ranker needs `lsid`, `lpmid`, `lcid` (log counts) per candidate (see `c2gap_extract.py:59-73`). Shipping a key→count
   table for about 75M keys is too big. Instead:
   - **Preferred:** row-aligned arrays matching the file the Kaggle channel already reads (no keys needed), stored as
     `uint8` log-quantised counts (for example `round(log1p(n) * 16)`, clipped at 255). Three `.npy` files.
   - If the rows do not line up with a row-aligned layout, propose the smallest alternative and measure its size.
   - Build with duckdb (`memory_limit='1GB'`), writing intermediates to `C:\casmi_tmp\`. Put only the final arrays
     in `research/kaggle_e1/e8/dataset/`, and only if D: still has more than 1.5 GB free afterwards. Otherwise leave
     them in `C:\casmi_tmp\e8\` and say so.
4. **Check that quantisation is harmless.** Re-run the TASK D SV evaluation, and the bench PC/S2 grouped-CV
   evaluation, with popularity features rebuilt from the quantised arrays. The MRR change must be under 0.002 on every
   bucket. Report the table.
5. **Draft the kernel stage** as a stand-alone module `research/kaggle_e1/e8/e8_pc_rerank.py`. It has one function
   that takes the PubChem channel's top-60 per query (the E7 structure) and returns the re-ordered list. The whole
   function sits in try/except; on any error it returns the input unchanged and logs `E8 RERANK FAILED`. Add a local
   test `research/kaggle_e1/e8/test_e8_pc_rerank.py` that runs it on the SV candidates and checks: ranks 1-3 of the
   final merged list unchanged 400/400, 25 unique metric keys per row, and that the fallback triggers if the ranker
   file is missing. **Do not copy or edit the E7 folder.** The E7 dataset is 656 MB and D: has no room for it.

**Report:** `research/kaggle_e1/e8/E8_ASSETS.md` (sizes, determinism, quantisation table, open questions for Claude).
**Files you may change:** `research/kaggle_e1/e8/*` (new folder), `research/scripts/e8_*.py`, `results/c2gap_full/*`,
`C:\casmi_tmp\*`.

---

## TASK F — seed-Tc-aware Class-3 re-ranker (bench research) — opencode CLI, `opencode/big-pickle`
**Why:** `rr_bt` (the E7 Class-3 generator) scores MRR 0.757 when the truth has a ±1-heavy-atom congener in the seed
sources (nb1-yes) but only **0.166** when it doesn't (nb1-no). Real hidden Class 3 is probably mostly nb1-no. A
zlog re-rank hurts overall (0.253) but helps nb1-no (0.191). The idea (`c3gen_tournament.md` §8, item 3.1): mix the
spectrum score in **more** as the best seed similarity **falls**.

**Steps:**
1. Find the bench inputs. Candidate lists are in `results/c3gen/full/*.json` (biotransform, mmp_edit, rr_bt), the
   context is in `results/c3np/context.pkl`, and the evaluator is `research/scripts/c3np_eval.py`, which reports nb1
   strata. Find the script that produced the zlog re-rank number 0.253 (`grep -rn zlog research/c3gen research/scripts`)
   and reuse its scoring rather than writing new scoring.
2. Per candidate, build these features: rr position, rank within each generator, which generator(s) produced it,
   z(zlog) within the molecule, best seed Tc (Tanimoto to the closest seed), rule frequency / rule support, and the
   heavy-atom delta to the seed. **No truth-derived feature.** `nb1` itself is a truth-side label and must not be a
   feature, though it may be used for stratification.
3. Baselines first: a hand rule `score = rr_rank_score + w(seedTc) * z(zlog)` with `w` falling linearly in seed Tc.
   Grid over 2 parameters and evaluate with 5-fold CV **grouped by molecule** and stratified by nb1. Then a LightGBM
   lambdarank on the same features with the same CV.
4. Report MRR@25 overall, nb1-yes, nb1-no, and under `--strip edit1` (the realistic mode), for: rr_bt as shipped, the
   zlog re-rank, the hand rule, and lambdarank. The goal is nb1-no clearly above 0.191 **without** losing more than
   0.01 overall in strip edit1 mode.
5. Estimate the LB effect as an INFERENCE. Show the arithmetic, use a 2x haircut, and assume about 10-20% of the
   hidden test is Class 3 (see DEC-010).

**Report:** `research/analysis/c3_seedtc_reranker.md`.
**Files you may change:** new scripts `research/c3gen/seedtc_*.py`, `results/c3gen/seedtc/*`, that report.
Do **not** edit `biotransform.py`, `mmp_edit.py` or anything in `research/kaggle_e1/e7/`; E7's determinism check
depends on them.

---

## When Claude is back (tomorrow)
Paste: "resume: read instruction.md (2026-10-07) and the reports e7_review_gemini.md, e8_sv_gonogo.md,
E8_ASSETS.md, c3_seedtc_reranker.md, plus the E7 LB score if submitted". Claude then: wires E8 into the kernel,
benches the s10/s20 seed nets, and decides whether the seed-Tc re-ranker goes into E9.

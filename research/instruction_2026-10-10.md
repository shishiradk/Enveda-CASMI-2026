# Instruction for opencode, 2026-10-10: what are the 0.45+ teams doing?

Context: competition `enveda-CASMI26-molecule-id-mass-spectra`. Our best eligible public score is 0.379. The public
leaderboard top is 0.484, 5th place is 0.459, and 11th is 0.449. We want to know what the 0.44+ public notebooks do
that we don't, and whether their inputs are prize-eligible.

**Prize-eligibility rubric (use it for every input):** ELIGIBLE = open licence (MIT, Apache, CC0, CC BY) or data/weights
we trained ourselves from competition data. NOT ELIGIBLE = NIST, METLIN, any non-commercial (NC) licence, weights
trained on those, or a dataset whose licence is not stated (mark it UNKNOWN, not eligible).

**Rules:** read-only research. Do NOT submit anything and do NOT change any file outside the output files named
below. Throw-away scripts go in `research/scratch_opencode/`. Label each statement FACT (seen in code/metadata),
INFERENCE, or HYPOTHESIS. Do not repeat what is already in `research/analysis/competition_intel.md`; read it first and only add new things.
Keep each report under ~150 lines. Free-tier quota is small, so pull at most 8 notebooks per task.

## Task 1a (CLI, `-m google/gemini-3-flash-preview`): top public notebooks
Output: `research/analysis/pubnb_top.md`

1. Run `kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --sort-by scoreDescending --page-size 20`
   (also try `--sort-by voteCount`). Record the order. A score is shown only if it is in the title or metadata; otherwise write "?".
2. Pull the first 8 notebooks: `kaggle kernels pull <owner/slug> -p research/scratch_opencode/pubnb/<slug> -m`.
3. For each notebook write one table row plus a short block:
   - owner/slug, title, score (if visible), last run date, GPU or CPU, runtime if visible
   - attached datasets, models and kernels (from kernel-metadata.json)
   - pipeline in at most 6 lines (retrieval, candidate source, ranker/model, fusion, post-processing)
   - anything that looks novel or that we do not do: pretrained weights, extra candidate databases (PubChem,
     COCONUT, ChEBI, HMDB...), forward spectrum models, ensembling, test-time tricks
   - eligibility of every attached input per the rubric, with the licence you saw (FACT) or UNKNOWN

## Task 1b (app, `opencode/big-pickle`): the 0.40 to 0.449 band and the discussion trail
Output: `research/analysis/pubnb_mid.md`

1. Same listing command as 1a, but take notebooks ranked 9 to 16 (use `--page 2` or a larger `--page-size`).
   Pull at most 8 into `research/scratch_opencode/pubnb/`. Skip any that 1a already pulled (check for the folder).
2. Same table and blocks as 1a.
3. Run `kaggle competitions` / `kaggle datasets list --search casmi` style searches to find public datasets that
   notebooks depend on and that we do NOT have. List the ten most-used ones with owner, size, licence.

## Task 2 (run AFTER 1a and 1b finish): synthesis
Output: `research/analysis/pubnb_synthesis.md`

Read `pubnb_top.md`, `pubnb_mid.md`, and our pipeline summary `research/kaggle_e1/e7/E7_STATUS.md`.
Produce:
1. The short list of techniques or inputs that appear in 2 or more of the 0.44+ notebooks and that we lack.
2. For each: eligible / not eligible / unknown, and which licence evidence supports that.
3. Of the eligible ones, rank by (expected gain, effort). Say what could be built from our own data in under a day.
4. Anything that looks like it uses NIST or NC data, with the exact file or dataset name.

When done, print the three report paths. Do not summarize in chat beyond that.

# Agent task briefs — CASMI 2026 (updated 2026-09-30)

Goal: beat the public 0.399 notebook clearly with **our own** pipeline (V3), then climb. Claude builds the core (the
learned ranker and its validation). The tasks below are independent pieces that feed it.

| Task | Agent | Why this agent | CPU budget | Start |
|---|---|---|---|---|
| A. V1 code review (read-only) | opencode | small, well-specified, no heavy compute | 1 core | now |
| C. Extra structure databases | opencode | mechanical download + parse | ≤2 processes | after A |
| E. Competition intelligence digest | Hermes | web reading, no compute | none | now |
| D. Forward-model (ICEBERG) feasibility spike | Antigravity | install + benchmark an external repo | ≤2 processes | now |
| B. EXP-016 fragmentation confirmation | Antigravity | pre-registered run, long CPU job | ≤4 processes | after the user says the PubChem build is done |

Paste the **shared rules** and **one task** into the agent.

## Shared rules (every agent)

- Project root: `D:\Enveda-CASMI-2026`. Read `research/analysis/submission_v1.md` first (current pipeline, LB 0.237), then
  `research/analysis/methods_survey_2026-09-30.md` (what the field is doing).
- **Never edit** `research/kaggle_v0/`, `research/kaggle_v1/`, `research/kaggle_v2/`, `results/kaggle_v*`,
  `external/pubchem/`, or any existing report. Claude works there. Write only to the output paths your task names.
- **No Kaggle submissions, no Kaggle dataset or notebook pushes, and never run other people's Kaggle notebooks.** You
  may read public notebooks and discussions for ideas.
- Load `train.parquet` with DuckDB (`duckdb.sql("SELECT ... FROM 'train.parquet'")`). pandas/pyarrow fails on it
  (`Repetition level histogram size mismatch`).
- Shared machine: 8 logical CPUs, 15.6 GB RAM, Windows, Python 3.14 (matchms 0.33.1, RDKit 2026.03, DuckDB 1.5.5, scipy,
  numpy, lightgbm if installed). **Respect your task's CPU budget and stay under 4 GB RAM.** Claude's jobs run at the same
  time.
- If you need a new Python package, install it into a virtual environment under `research/envs/<task>/`, not globally.
- Label every claim FACT (measured, with the file or command that shows it), INFERENCE, or HYPOTHESIS. Never invent
  numbers; if you could not measure something, say so.
- Metric: MRR@25 per molecule; a guess is correct when its tautomer-canonical InChIKey first block (InChIKey14)
  equals the answer's.
- When done, tell the user the output path. Claude verifies the key numbers before using anything.

---

## Task A — opencode: independent review of the V1 pipeline (read-only)

**Goal:** find real bugs before V2/V3 build on this code.

Read `research/kaggle_v1/casmi_v1_kaggle.py`, `research/kaggle_v1/proxy_eval.py` and the functions they call in
`research/kaggle_v0/casmi_v0_kaggle.py`. Report:

1. **Correctness:** row alignment between `results/kaggle_v1_assets/universe.parquet` and `universe_fp.npy` (build:
   `research/kaggle_v1/build_assets.py`); ppm window math in `Universe.window`; polarity mask and top-k in `_topk`;
   `featurize` (peak ranking per row, neutral-loss bins, L2 normalisation); fingerprint weighting in `knn_scores`; tie
   ordering; molecule-level mean; spectra without peaks.
2. **Leakage in `proxy_eval.py` scenario S2:** can a held-out target, its tautomer alias or its parent group still reach
   the kNN references, the library candidates or the candidate universe?
3. **Hidden-rerun robustness:** ~1,500 spectra / 400 molecules; unknown adduct; empty or NaN peak lists; duplicated
   spectrum ids; empty 5 ppm window; memory on Kaggle (30 GB, 4 CPUs).

For each finding: `file:line`, a concrete input that triggers it, what goes wrong, severity (critical/major/minor), and
a suggested fix (text only, do not apply). Throw-away proof scripts may go in `research/scratch_opencode/`. No style
issues.

**Output:** `research/analysis/review_v1_opencode.md`.

---

## Task C — opencode: extra candidate-structure databases

**Goal:** more Class-2 coverage. The public stack gained +0.009 LB from adding ChEBI + LIPID MAPS structures.

Download the **latest public releases** of these databases (record the URL, release date, file sha256 and licence of
each):

| Database | What to get | Notes |
|---|---|---|
| ChEBI | complete structures SDF or TSV with SMILES | CC BY 4.0 |
| LIPID MAPS LMSD | full SDF | check the licence page and record it |
| NPAtlas | full download (TSV/JSON with SMILES) | microbial natural products, CC BY 4.0 |
| HMDB | metabolite structures SDF (`structures.zip`) | check the licence; record it |
| LOTUS | latest Zenodo/Wikidata dump with SMILES, only if it is a single freely downloadable file | mostly overlaps COCONUT |

Put raw downloads under `external/structdb/<db>/`. Then write `research/structdb/build_structdb.py` that produces one
table **`results/structdb/structdb.parquet`** with columns:

`ik` (InChIKey first 14 characters, computed with RDKit from the parsed molecule, not copied from the file), `smiles`
(RDKit canonical), `mass` (RDKit `Descriptors.ExactMolWt`, neutral), `formula`, `sources` (comma-separated database
names), `n_sources`.

Rules: one row per `ik` (merge sources); keep only single-component, neutral, non-radical molecules with elements in
C H N O P S F Cl Br I, containing carbon, mass 150–1170 Da; take the largest fragment of salts
(`rdMolStandardize.LargestFragmentChooser`) and neutralise (`Uncharger`) before computing `ik`; drop anything RDKit
cannot parse. Sort by mass.

Also write `results/structdb/structdb_report.json`:
- rows per source;
- overlap of each source with the COCONUT keys (`results/kaggle_v1_assets/universe.parquet`, `src == 'coconut'`) and
  with the train structures (`src == 'train'`);
- the number of **new** keys not already in `universe.parquet`;
- the RDKit parse failures per source.

Use at most 2 worker processes.

**Output:** `results/structdb/structdb.parquet`, `results/structdb/structdb_report.json`,
`research/structdb/build_structdb.py`, and a short summary in `research/analysis/structdb_report.md`.

---

## Task D — Antigravity: forward-model (ICEBERG) isomer re-scoring feasibility

**Goal:** decide whether we can run a forward MS/MS simulator ourselves to break same-formula isomer ties (95% of our
top-1 errors are same-formula isomers). Public stacks gain about +0.02 LB from ICEBERG + GLACIER re-scoring.

1. Use **only the official sources**: the ms-pred repository by the Coley group (`github.com/coleygroup/ms-pred`, MIT)
   and its officially released pretrained checkpoints. Record the commit hash, checkpoint URLs and their licences.
   Checkpoints trained on NIST are **not allowed** (the host disallows NIST-, METLIN- and vendor-trained weights). Use
   only models trained on public data (e.g. MassSpecGym / NPLIB-free). Write down which training data each checkpoint
   used, with a source. If you cannot establish that a checkpoint is NIST-free, stop and report.
2. Install it in `research/envs/iceberg/` (CPU is fine). Get a working prediction: SMILES + adduct + collision energy →
   predicted spectrum.
3. **Benchmark on our proxy:** take 60 molecules from `research/kaggle_v1/proxy_eval.py` scenario S2 (the cached
   candidates are in `results/kaggle_v1_proxy/cache_base_S2.pkl`: `pickle.load(...)["knn"][target] = {ik: cosine}`, and
   the query spectra are in `results/kaggle_v1_proxy/proxy_test.parquet`, `molecule_id` = target InChIKey14). For each
   molecule, take the top 20 kNN candidates, keep those with the target's molecular formula plus the true structure if
   it is in the pool, predict their spectra for each query spectrum's adduct, and score each candidate by the mean
   cosine (matchms `CosineGreedy`, tolerance 0.02) between predicted and observed spectra. Candidate SMILES:
   `results/kaggle_v1_assets/universe.parquet` (`ik`, `smiles`).
4. Report, on those 60 molecules:
   - MRR of kNN alone, of the forward score alone, and of `z(kNN) + λ·z(forward)` within same-formula groups for
     λ ∈ {0.5, 1.0}, with a paired bootstrap 95% CI for the differences (2,000 resamples, seed 20260930);
   - runtime per candidate spectrum on this CPU;
   - peak RAM;
   - the problems you hit.
5. Adducts: the test uses `[M+H]+`, `[M+NH4]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+Na]+`, `[M+K]+`, `[M-H]-`,
   `[M-H2O-H]-`, `[M+CH2O2-H]-`, `[M+Cl]-`. Report which ones the model supports.

At most 2 worker processes. Do not touch the proxy caches (read only).

**Output:** `research/analysis/iceberg_feasibility.md` plus scripts in `research/forward_model/`.

---

## Task B — Antigravity (after Task D, and only once the user says the PubChem build is done): EXP-016

**Goal:** decide whether in-silico fragmentation features (EXP-014) improve ranking of same-formula isomers on a fresh
natural-product population. They are a candidate feature for the V3 ranker.

- Design (pre-registered, locked): `research/analysis/exp016_fragmentation_confirmation_design.md`. **Do not change
  hypotheses, gates, seeds or scorers after seeing any result.**
- Script: `research/scripts/exp016_fragmentation_confirmation.py` (written, never run). Background:
  `research/analysis/exp014_fragmentation_features_report.md`, `research/scripts/exp014_fragmentation_features_v2.py`.
- Steps:
  1. Check the script against the design and list any mismatch *before* running. If a fix is needed, copy it to
     `research/scripts/exp016_fragmentation_confirmation_fixed.py`, fix the copy, and say why.
  2. Smoke test first (small n), then the full run (~20–40 min of CPU, at most 4 processes).
  3. The design's leakage gates are zero-tolerance: if one fails, stop and report INVALID.
- Outputs: `results/exp016/`, `research/analysis/exp016/`, and `research/analysis/exp016_fragmentation_confirmation_report.md`
  with the H1/H2/H3 verdicts, paired bootstrap CIs, isomer-error table, runtime and leakage audit, in the style of the
  EXP-014 report.

---

## Task E — Hermes: competition intelligence digest (read-only, no code execution)

**Goal:** a complete, sourced map of what has been tried in this competition and what it scored, so we build on
evidence instead of re-discovering it.

Read the public notebooks and the discussion forum of the Kaggle competition
`enveda-CASMI26-molecule-id-mass-spectra` (reading only; do not fork, copy or run notebooks). Cover at least the 25
most-voted notebooks, all notebooks with a stated LB ≥ 0.33, and every discussion thread with host posts or ≥ 5
replies.

Produce `research/analysis/competition_intel.md` with:

1. A table of notebooks: author, title, link, stated public LB, date, licence, main components (candidate sources,
   fingerprint model, ranker, forward models, PubChem handling, class-3 generation), GPU/CPU, runtime, and which datasets
   and pretrained models it depends on (with licences).
2. An **ablation ledger**: every "X changed the LB from A to B" statement you find, with the link and the date. Include
   negative results (things that hurt).
3. **Host rulings and clarifications** (allowed and disallowed data or models, the class definitions, the size of the
   public/private split, metric details), quoted with links.
4. What the top teams (LB ≥ 0.42) have disclosed about their approaches, if anything.
5. Open questions and contradictions between sources.

Quote numbers exactly as written and link each one. Mark anything you infer as INFERENCE.

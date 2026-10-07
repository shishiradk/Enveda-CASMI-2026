# opencode tasks: review of the evaluation-bench package

Two independent, read-only review tasks on the same package. They do not overlap, so they can run at the same time:

| Task | Run it with | Report file |
|---|---|---|
| A. Windows scripts and runsheet | opencode CLI, model `google/gemini-3-flash-preview` | `research/analysis/review_bench_pkg_A.md` |
| B. Package completeness, caches, versions | opencode app, model `opencode/big-pickle` | `research/analysis/review_bench_pkg_B.md` |

How to start (from `D:\Enveda-CASMI-2026`):
- CLI: `opencode run -m google/gemini-3-flash-preview "Read research/bench_pkg/instruction.md and do Task A exactly as written."`
- App: open the folder `D:\Enveda-CASMI-2026`, choose `opencode/big-pickle`, and send:
  `Read research/bench_pkg/instruction.md and do Task B exactly as written.`

---

## Shared context (read for either task)

**What the package is.** `research/bench_pkg/` builds a package that lets a teammate run the team's evaluation
bench on her own PC. The bench scores a pair of trained fingerprint networks (`fp_single*.pt`, `fp_merged*.pt`)
on four proxy scenarios (S1, S2, S3, S3i) and reports MRR@25.

**Who runs it.** The teammate is not a programmer. She has:
- a Windows 11 laptop with an RTX 5050 and 16 GB RAM;
- Python 3.12 installed, and a working Kaggle API token from an earlier job.

She will download the package from a private Kaggle dataset, then double-click, in order:
`setup_bench.bat`, `smoke_bench.bat`, `run_bench.bat`, `send_bench_results.bat`. She follows `RUNSHEET_BENCH.md`.
An earlier package in the same style, `research/train_pkg/`, worked for her without problems.

**Files under review** (in `research/bench_pkg/`):
- `RUNSHEET_BENCH.md`, `requirements_bench.txt`;
- `setup_bench.bat`, `smoke_bench.bat`, `run_bench.bat`, `send_bench_results.bat`;
- `check_env_bench.py`, `bench_check.py`, `make_bench_pkg.py`.

**Staged copy.** `D:\casmi_bench_pkg` is what `make_bench_pkg.py` produced. It mirrors the project root, so it holds
`research/bench`, `external/bench_inputs` and `results/...`. Its `train.parquet` is a local hard link and is NOT
uploaded.

**Code the package runs.**
- `research/bench/bench.py`: the `run` and `eval` stages. The `--alt-fp <folder>` option was just added and is
  used by `run_bench.bat`.
- `research/bench/bench_eval.py`: `main()`.
- `research/bench/eng/casmi_engine.py`: `find()` looks files up under the folder in the `CASMI_ROOTS` environment
  variable.

## Rules (both tasks)

- **Do NOT modify, move or delete any existing file**, in the repo or in `D:\casmi_bench_pkg`.
- **Do NOT run `bench.py run`, `bench.py all`, or any `.bat` file.** Another bench run is using most of the
  machine's RAM.
- You may read files and list directories. You may run tiny Python one-liners (under 1 GB RAM, 1 core, under
  1 minute).
- Throw-away scripts go in `research/scratch_opencode/`, with names starting `bp_A_` or `bp_B_`.
- Report concrete defects only: things that would make a step fail, give wrong results, or confuse her. Skip
  style and refactoring opinions.
- Label every finding:
  - **FACT**: you checked it; say how (command, file:line).
  - **INFERENCE**: follows from a FACT.
  - **HYPOTHESIS**: not checked.

**Report format.**
- First line: a verdict, either `READY`, `READY AFTER FIXES` or `NOT READY`.
- Then a numbered list of findings, most severe first. Each finding gives:
  - its label;
  - the file:line;
  - what goes wrong for her, as a concrete scenario;
  - a suggested fix, as text only (do not apply it).
- Under 120 lines.

---

## Task A: Windows scripts and runsheet (CLI, gemini-3-flash-preview)

Review the four `.bat` files, `bench_check.py` and `RUNSHEET_BENCH.md`. Check at least the following.

**1. Batch pitfalls.**
- A folder path with spaces, e.g. `C:\Users\Akriti Rijal\Downloads\export`. Check quoting of `%MODELS%`, and of
  `%~dp0` when it holds spaces.
- `%VAR%` expanded inside `( ... )` blocks: the value set before the block is used, not the one set inside.
- `set /p` when she just presses Enter: is the variable empty or left over from before?
- `for /f %%s in ('venv\Scripts\python bench_check.py todo %TAG%')` must capture `S1,S2,S3`. Commas are delimiters
  in `for /f`; does the whole string survive?
- `|| goto :fail` after a `python` call: does the exit code propagate?
- `tar -xf train.parquet.zip` on Windows: does built-in `tar` extract `.zip` files?
- `> file echo {...}` with JSON braces and quotes in `send_bench_results.bat`.
- Behaviour when the run name contains spaces, or is typed differently on a re-run.

**2. Resume logic.**
- `bench_check.py todo` says S3 is done only when both `S3.pkl` and `S3i.pkl` exist. Confirm that `bench.py` writes
  both files in one S3 run. Find the line that writes them.
- Confirm that `bench.py run --scen S2,S3` (a subset) works.

**3. Smoke parity.**
- `smoke_bench.bat` runs S2 with `--limit 5`. `bench_check.py smoke` compares its non-`_mk` columns with
  `results/bench/e1/per_molecule_blend.csv`.
- Confirm that `--limit 5` selects the same 5 molecules that exist in that reference file.
- Confirm that the column names line up.

**4. Runsheet.**
- Does every step match what the scripts actually print (`SETUP OK`, `RESULT: ALL OK`, `FINISHED`)?
- Are any steps missing? For example: where she gets the `.pt` folder; what to do if the smoke test is skipped.
- Are any instructions wrong? Check in particular the `--workers 1` fallback command in the table.

Write `research/analysis/review_bench_pkg_A.md`.

---

## Task B: package completeness, caches, versions (app, big-pickle)

**1. Missing files.**
- Trace every file path that `bench.py run` (S1, S2 and S3 with S3i), `bench.py eval` and `bench_eval.main` read at
  runtime. Look for: `ROOT / ...`, `OUT / ...`, `INPUTS / ...`, `V1P` / `V2P`, `E.find(...)`, `np.load`,
  `pickle.load`, `read_parquet`, duckdb `FROM '...'`.
- Mark each one present or absent in `D:\casmi_bench_pkg`. Only paths that are actually reached count. If a read is
  skipped because a cache exists, check that the cache file is in the staged copy.
- Pay attention to `results/kaggle_v2_pubchem/pubchem_rows.parquet` (not shipped; is `results/bench/cache/ours_*.pkl`
  enough?), `S3i` handling, and `pv_np_groups.json`.

**2. Cache keys that would miss on her machine.**
- The pool cache name `pool_{train.stem}_{size}_{rdkit.__version__}.pkl`.
- The ranker cache hash in `fit_or_load`: what goes into the hash? sklearn version, anything path-dependent?
- If a cache misses, what happens: a silent refit with different results, or a crash? How long does it take?

**3. Version pins.**
- Do the pins in `requirements_bench.txt` have wheels for CPython 3.12 on Windows x64? Check PyPI file listings
  (e.g. `pip download <pkg>==<ver> --only-binary=:all: --python-version 3.12 --platform win_amd64 --no-deps -d
  research/scratch_opencode/bp_B_wheels`, then delete the downloads), or the PyPI JSON API.
- Does anything need `pip install` that is missing from the list? Compare with the imports in `research/bench/*.py`
  and `research/bench/eng/*.py`.
- Is torch 2.11.0 (cu128) compatible with the rest?

**4. Kaggle layout.**
- `make_bench_pkg.py` uploads with `kaggle datasets create -r zip`.
- What will she see after the website **Download** and after `kaggle datasets download --unzip`? A clean folder
  tree, or nested `research.zip` / `external.zip` / `results.zip`?
- Does the runsheet's "Extract All into the same folder" instruction produce the right tree in both cases?

**5. GPU path.**
- With CUDA available, `bench.py` puts the FP networks on `cuda`.
- Check `research/bench/eng/pv_fp.py` (`molecule_logits`, `load_fp_models`) for anything that assumes CPU: numpy
  conversion without `.cpu()`, dtype mismatches, AMP.

Write `research/analysis/review_bench_pkg_B.md`.

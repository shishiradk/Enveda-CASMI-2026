# E7 upload steps (NOT run by the agent — the user pushes)

Package built by `python research/kaggle_e1/e7/build_e7.py` (verified: notebook 6 cells, all cells compile).

## 0. Prerequisites

Run from the repo root `D:\Enveda-CASMI-2026`, with the Kaggle CLI authenticated **as `shishiradhikari11`**:

```
kaggle config view          # check the username is shishiradhikari11
```

The slugs are already baked into the metadata files, so no flags are needed:

| File | Contains |
|---|---|
| `research/kaggle_e1/e7/dataset/dataset-metadata.json` | `id: shishiradhikari11/casmi-e7-c3assets` |
| `research/kaggle_e1/e7/kernel-metadata.json` | `id: shishiradhikari11/casmi-e7-c3-channel`, private, T4 |

## 1. Upload the assets dataset (do this FIRST)

```
kaggle datasets create -p research/kaggle_e1/e7/dataset
```

- Target: **`shishiradhikari11/casmi-e7-c3assets`**
- **Size: 655.9 MB in 15 files.** `c3_pool_fp.npy` alone is 588.9 MB (89.8% of it).
- License declared `CC-BY-4.0` (the pool contains COCONUT-derived structures); the code is MIT. See `dataset/README.md`.
- Optional cleanup first: the folder also holds `__pycache__/` (4 `.pyc` files, 0.18 MB) left over from local
  imports. `kaggle datasets create -p` uploads it; it is harmless (wrong Python version, never imported) but
  can be removed to keep the upload clean.
- If the dataset already exists, replace its contents with:
  `kaggle datasets version -p research/kaggle_e1/e7/dataset -m "E7 assets rebuild" -d`

## 2. Push the kernel

```
kaggle kernels push -p research/kaggle_e1/e7
```

- Target: **`shishiradhikari11/casmi-e7-c3-channel`** (private, `NvidiaTeslaT4`).
- 11 `dataset_sources`: 6 yours (`casmi-bio-clean`, `casmi-fm-runner`, `casmi-v2-pubchem`,
  `casmi-rdkit2025-cp313`, `casmi-e6-pcnets`, plus the new `casmi-e7-c3assets`) and 5 third-party /
  Kaggle-managed (`prvsiyan/*` x3, `megayak/*`, `metric/rdkit-2026-3-3-wheel`). All 11 must be public or owned
  by the account, or the run fails at startup.
- Do **not** switch the kernel env away from the RDKit wheel source: `metric/rdkit-2026-3-3-wheel` pins
  RDKit 2026.03.3, and the Class-3 channel depends on that version's behaviour.
- Re-push the same id to create a new version.

## 3. What to check in the kernel log

The Class-3 stage prints one line per 25 molecules and a final summary. Expect, in order:

```
E7 C3: 400 test molecules, 400 with full context, COCONUT True | ...
E7 C3 25/400 molecules | lists 25 | fail 0 | ...
...
E7 C3 channel done {'n_lists': 399, 'n_fail': 1, 'budget_hit': False, 'coco_ok': True,
                    'bt_guard': 0, 'mmp_guard': 0, ...}
E7 merge: {'n_c3': 399, 'n_rows_ins': 399, 'n_ins': 1995, 'n_err': 0, 'n_post': 0}
E7 visible check: top-3 unchanged vs E6 400 / 400 | 25 unique per row 400 / 400 | rows with C3 insertions 399
```

Pass criteria (all met in the local dry run, `results/kaggle_e7_dry/dry_report.json`):

| Check | Expected |
|---|---|
| `top-3 unchanged vs E6` | 400 / 400 |
| `25 unique per row` | 400 / 400 |
| `rows with C3 insertions` | ~399 (1 molecule falls back to E6, see E7_STATUS.md) |
| `bt_guard` / `mmp_guard` | 0 — if non-zero, results depend on machine speed |
| `coco_ok` | `True`; if `False`, **no** molecule gets a C3 list and E6 is kept everywhere |

If `E7 C3 CHANNEL FAILED` or `E7 MERGE FAILED` appears, the kernel still writes a valid E6-equivalent
`submission.csv` — that is the intended fallback, not a crash.

## 4. Sizes at a glance

| Item | Size |
|---|---|
| `dataset/` upload (15 files, excl. `__pycache__`) | 655.9 MB |
| `casmi_e7.ipynb` | 89.8 KB |
| `kernel-metadata.json` | 898 B |

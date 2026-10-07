# E7 package review (Task C) — 2026-10-06

Reviewer: Claude Sonnet 4.6 (Thinking). Files read: `research/kaggle_e1/e7/e7_c3_channel.py`,
`research/kaggle_e1/e7/build_e7.py`, `research/kaggle_e1/e7/e7_merge.py`,
`research/kaggle_e1/e7/casmi_e7.ipynb` (cell 4 / E7CELL source in `build_e7.py` lines 94-131),
`research/kaggle_e1/e6/kernel-metadata.json`, `research/kaggle_e1/e7/E7_STATUS.md`.

---

## Checklist

| # | Check | Finding | Verdict |
|---|---|---|---|
| 1 | Paths won't exist on Kaggle | See detail below | **CLEAN** |
| 2 | `fp_*.pt` glob picks up wrong nets | `e7_c3_channel.py` never loads `.pt`; glob only in cell-0 print | **CLEAN** |
| 3 | Missing try/except around Class-3 stage | Full try/except wraps subprocess + json.load; merge has its own guard | **CLEAN** |
| 4 | Top-3 being changed | `e7_merge.py:31-33` post-condition check + `n_post` counter | **CLEAN** |
| 5 | Duplicates by metric key | `e7_merge.py:27` `if k and k not in keys` before append | **CLEAN** |
| 6 | Rows with fewer than 25 entries | seq = 3 E6 + <=5 C3 + 22 E6 tail -> cap=40 -> submission writes first 25; dry run confirmed 400/400 | **CLEAN** |
| 7 | Non-eligible inputs | All assets from train + COCONUT (CC BY 4.0); no NIST / FRIGID / Ahmed Berat Ozer data | **CLEAN** |

**Overall: no blocking issue found.**

---

## 1  Paths on Kaggle — detail

| Path / resource | How it arrives on Kaggle | OK? |
|---|---|---|
| `e7_c3_channel.py`, `biotransform.py`, `mmp_edit.py`, `assets.py`, `e7_merge.py` | `dataset/` -> `shishiradhikari11/casmi-e7-c3assets`; `find('e7_c3_channel.py')` locates it | yes |
| `c3_pool_*.npy`, `c3_pool_smiles.txt`, `c3_fp_bits.npy`, `c3_mmp_rules.parquet`, `c3_pool_np.npy` (7 asset files) | same dataset; pre-flight check in `e7_c3_channel.py:286-293` fails fast if missing | yes |
| `coco_fp.npy`, `coco_mass.npy`, `coco_meta.pkl` | `prvsiyan/coconut-casmi26-candidates` (already in E6 kernel sources, inherited by E7) | yes |
| `/kaggle/working/e6_ctx.pkl` | written by `e7_e6_channel.py --dump-ctx` in cell 3; cell 4 checks `os.path.exists` before running | yes |
| `/kaggle/working/eng_ana.json` | written by the modified engine runner in cell 1; same pre-flight check | yes |
| `os.path.join(COMP, 'test.parquet')` | `COMP = os.path.dirname(find('train.parquet'))` -> competition source | yes |
| `--assets os.path.dirname(_c3py)` | `_c3py = find('e7_c3_channel.py')` -> dataset dir; same dir holds all `c3_*` assets and code | yes |
| `e7_e6_channel.py` | `find('e7_e6_channel.py')` with try/except fallback to bare `e6_channel.py` | yes |

**No broken path found.**

## 2  `fp_*.pt` glob

`casmi_e7.ipynb` cell 0 (line 30) prints:

    sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))

This is a diagnostic print only; the matched paths are never passed to any loader in E7 code.
`e7_c3_channel.py` loads no `.pt` files at all (no `torch` import, no `load_fp_models` call).
The E6 forward-model stage (cell 2) calls `load_fp_models(paths, dev)` on these paths; it is
unchanged from E6. Our own nets are named `e6net_*`, so `fp_*.pt` cannot match them.
**No issue.**

## 3  try/except around the Class-3 stage

`build_e7.py` E7CELL structure (lines 100-130):

    try:                                   # outer — catches everything
        _c3py = find(...)                  # FileNotFoundError -> outer except -> C3L = {}
        if _budget < 600: raise ...        # low budget -> outer except -> C3L = {}
        for _f in (...):                   # missing ctx/ana -> outer except -> C3L = {}
            if not os.path.exists(_f): raise FileNotFoundError(_f)
        _cmd = [...]
        try:                               # inner — subprocess.TimeoutExpired only
            rr = subprocess.run(_cmd, ...)
        except subprocess.TimeoutExpired as _e:
            print(...)                     # falls through; reads whatever was written
        C3L = json.load(open(_c3out))      # JSONDecodeError / FileNotFoundError -> outer except
        E7_STATS = C3L.pop('_stats', {})
        C3L.pop('_info', None)
    except Exception as e:                 # catches all of the above
        C3L = {}
        print('E7 C3 CHANNEL FAILED -> E6 lists kept:', repr(e))
    ...
    try:                                   # merge has its own guard
        ENG, E7_MERGE = e7_merge(ENG, C3L)
    except Exception as e:
        print('E7 MERGE FAILED -> E6 lists kept:', repr(e))

COCONUT failure path in `e7_c3_channel.py` writes `{}` to `OUT_JSON` and returns, so
`json.load` on a timeout-partial file might see incomplete JSON -> `JSONDecodeError` ->
outer except -> `C3L = {}` -> E6 lists kept. **All failure modes correctly protected.**

## 4  Top-3 being changed

`e7_merge.py:31-33`:

    if keys[:keep_top] != [k for k in ek if k][:keep_top]:
        st['n_post'] += 1
        continue          # does NOT update out[mid]; E6 entry kept

`keep_top=3` by default. Post-condition check is conservative: if the merged list's first
3 metric keys differ from E6's non-null first 3, the whole merge is discarded for that
molecule. Dry run reported `n_post = 0` on 400 molecules. **No top-3 mutation possible.**

## 5  Duplicates by metric key

`e7_merge.py:25-29`:

    keys, smis = [], []
    for k, s in seq:
        if k and k not in keys:        # list-based uniqueness check
            keys.append(k); smis.append(s)
        if len(keys) >= cap:
            break

`seq` = E6 top-3 (unique by E6's own guarantee) + <=5 C3 inserts (already checked against
`ref = set(ek[:25])` and a local `seen` set) + E6 tail. The `if k not in keys` guard
removes any residual clash. Dry run confirmed 25 unique metric keys per row 400/400.
**No duplicates possible.**

## 6  Rows with fewer than 25 entries

`seq` = 3 (E6 top-3) + <=5 (C3 inserts, filtered to not duplicate E6 top-25) + 22 (E6 ranks
4-25). Before dedup: 3 + 5 + 22 = 30 entries. C3 inserts are keys **not** in E6 top-25,
so only the 22-entry E6 tail can produce duplicates (if any C3 insert coincidentally matches
an E6 rank > 25 entry). After dedup the minimum realistic count is 25. `cap=40` is generous.
The submission cell writes `[:25]`. Dry run: 400/400 rows have exactly 25 unique SMILES.
**No under-length rows possible in practice or in the dry run.**

## 7  Non-eligible inputs

| Source | License / eligibility | Used for |
|---|---|---|
| `train.parquet` (competition data) | competition rules allow derived artifacts | pool structures, MMP rules mined by us |
| COCONUT (`prvsiyan/coconut-casmi26-candidates`) | CC BY 4.0 | pool structures |
| RDKit | BSD-3-Clause | chemistry ops |
| Our code (`biotransform.py`, `mmp_edit.py`, `assets.py`, `e7_*.py`) | MIT | generators |
| `casmi-e6-pcnets` (our E6 nets) | trained by us on `train.parquet` only | zlog in ctx (E6 stage) |

No NIST data. No Ahmed Berat Ozer / FRIGID weights. No `casmi26-v4b`, `casmi26-v3`,
`fpnet_full1`, `casmi26-iceberg`, `casmi26-glacier` (asserted absent in `build_e7.py:34-35`).
**Fully prize-eligible.**

---

## Minor notes (non-blocking)

1. **`__pycache__/` in dataset dir** — E7_STATUS.md step 4 notes 0.18 MB of dead `.pyc` files
   in `dataset/`. Inert but slightly enlarges the upload. Low priority.

2. **`c3_mmp_rules.parquet` pre-flight vs. build** — `e7_c3_channel.py:287` checks for this
   file in the asset pre-flight. `build_e7.py:181` copies it from `results/c3gen/mmp_rules.parquet`.
   The source path is not checked by `build_e7.py`. **FACT (E7_STATUS.md step 4):** all 15
   files confirmed present in `dataset/`, so already resolved before upload.

3. **`e7_e6_channel.py` not reviewed** — Task C scope covers `e7_c3_channel.py`, `build_e7.py`,
   and the E7CELL merge. `e7_e6_channel.py` (the ctx-dump E6 channel copy) is out of scope.

---

**Conclusion:** The E7 package is structurally sound on all 7 Task C checks. No blocking
issue was found. The package is ready for the user to push per `UPLOAD_STEPS.md`.

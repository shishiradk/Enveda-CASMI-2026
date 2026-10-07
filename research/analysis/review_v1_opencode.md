# Task A — independent review of the V1 pipeline

**Reviewer:** opencode (read-only). **Date:** 2026-09-30.
**Target:** `research/kaggle_v1/casmi_v1_kaggle.py` (365 lines), `research/kaggle_v1/proxy_eval.py`, and the
`research/kaggle_v0/casmi_v0_kaggle.py` functions they call.
**Budget:** 1 CPU core, < 4 GB RAM. Peak process working set observed 2.12 GB (`a18_pools_mem.json`).

Every claim is labelled **FACT** (measured; the proving script is named), **INFERENCE** (follows from a measured
fact), or **HYPOTHESIS** (not measured). Proof scripts are throw-away files in `research/scratch_opencode/`.
No pipeline file was modified.

---

## Verdict

The shipped run is clean and reproducible, and the science is sound where it matters most — I could not find an
error that changes the current `submission.csv`.

```
FACT  results/kaggle_v1_local/v1_run_report.json
      knn.spectra_used=1213  spectra_non_ad10=0  reference_without_fp=0
      empty_pool_molecules=[]  molecules_scored=400  placeholder_rows=0  validation.fatal_ok=true
```

What I did find is a class of **hidden-rerun risks**: the pipeline degrades to a silently wrong submission, or
crashes outright, on inputs that the visible `test.parquet` happens not to contain. Six of the fourteen findings
are worth fixing before V2/V3 build on this code, because each fails **silently** — the run still reports
`fatal_ok: true` and writes a plausible `submission.csv`. One (F1) is worse: it crashes the job before anything
is written.

There is also one **scientific-claim finding**: the similarity-weighted fingerprint average described in
`research/analysis/submission_v1.md` is measurably indistinguishable from an unweighted average, and the
unweighted version scores slightly *better* on the only held-out proxy that exists.

### Findings, most severe first

| # | Sev | Location | One-line summary |
|---|-----|----------|------------------|
| F1 | critical | `casmi_v1_kaggle.py:107-110` | `precursor_mz = +inf` (or any huge value) raises `ValueError: negative axis 1 index` — **no `submission.csv` is written at all** |
| F2 | critical | `casmi_v1_kaggle.py:218` | If no kNN reference resolves against the universe, `reference_without_fp` is only *logged*; the run writes a garbage submission and `fatal_ok: true` |
| F3 | major | `casmi_v1_kaggle.py:207` | Test spectra with an adduct outside the hardcoded `AD10` are dropped silently; a molecule with only such spectra becomes a `"C"` placeholder |
| F4 | major | `casmi_v1_kaggle.py:224-228` | `NaN` or `-inf` precursor → empty 5 ppm window → molecule silently becomes a `"C"` placeholder; one bad spectrum poisons a whole multi-spectrum molecule |
| F5 | major | `casmi_v1_kaggle.py:339,343` + `validate:267-279` | `placeholder_rows` is counted but never made fatal — an all-placeholder submission passes validation |
| F6 | major | `casmi_v1_kaggle.py:216` + `fuse:256` | Empty/zero neighbour weights → NaN predictions → NaN candidate scores; `fuse` then orders them arbitrarily because `-nan` never compares true |
| F7 | major | `research/analysis/submission_v1.md:19` | The similarity weighting is a no-op; the plain unweighted mean is equal-or-better on S2 |
| F8 | minor | `casmi_v1_kaggle.py:175` | `k = min(k, S.shape[1] - 1)` drops one legitimate neighbour when the reference count ≤ k |
| F9 | minor | `casmi_v1_kaggle.py:174` | `-1.0` polarity-masked rows can be selected into the top-k when a polarity class is small |
| F10 | minor | `casmi_v1_kaggle.py:127-131` | `featurize_df` truncates mismatched `ms2_mzs`/`ms2_normalized_intensities` lists instead of failing |
| F11 | minor | `proxy_eval.py:77-82` | The S2 leakage gate checks library candidates and the universe but **not** the kNN references |
| F12 | minor | `casmi_v1_kaggle.py:229` vs `:224` | Fingerprints are averaged across the molecule's spectra, but the mass window uses only the median precursor |
| F13 | minor | `casmi_v1_kaggle.py:196` | `rp.unlink()` raises `PermissionError` on Windows — the local runner is the only affected target |
| F14 | minor | `casmi_v1_kaggle.py:309` | `v1_run_report.json` records no library versions, so a rerun cannot be shown to match |

---

## 1. Correctness

### Row alignment between `universe.parquet` and `universe_fp.npy` — clean

`build_assets.py` writes both from one frame, and `Universe.__init__`
(`casmi_v1_kaggle.py:66-74`) slices them with a single boolean mask, so they cannot drift *within* one load.
I also checked the two independently:

- **FACT** — 300/300 fingerprint rows match the SMILES stored in the same `universe.parquet` row
  (`a2_alignment.py`, `a2_alignment.json`).
- **FACT** — 200/200 sampled neutral masses equal RDKit `Descriptors.ExactMolWt` recomputed from the stored
  SMILES (`a2_alignment.py`).
- **FACT** — `mass` is non-decreasing (`assert` at `:74` passes), `ik` and `smiles` are both unique, `fp` is
  `(729388, 256)` `uint8` (`a1_data_facts.json`).

### ppm window math — correct, and much tighter than it looks

`Universe.window` (`:76-78`) computes `tol = M * ppm * 1e-6` and `searchsorted` on the mass-sorted array.
**FACT** — the half-width is 0.5 mDa at `M=100` and 10 mDa at `M=2000` (`a18_pools_mem.json`). For scale, the
median mass gap between adjacent universe rows is far larger than this.

The consequence is that **pool size is driven by accidental mass collisions, not chemical ambiguity**:

- **FACT** — 167,427 unique masses over 729,388 rows; the largest same-mass cluster is 1,761 rows;
  429,689 rows sit in clusters of ≥10 (`a1_data_facts.json`).
- **FACT** — on the visible test: 0 empty pools, but min pool **1**, 9 molecules with ≤3 candidates, 27 with
  ≤10, and 88/400 (22%) with ≤25 (`a18_pools_mem.json`). Consequently **12 of the 400 submitted rows contain
  fewer than 5 predictions** — a direct consequence of a near-empty 5 ppm window, not of a scoring bug.
- **FACT** — submission hygiene is otherwise good: 0 rows contain a repeated SMILES within the row, the most
  frequently predicted structure appears 4 times across 400 molecules (a bromo-phenyl analog series), and the
  5 multi-component predictions are the only questionable rows.
- **INFERENCE** — for the 22% of molecules whose pool has ≤25 candidates the top-25 set is the pool, so the
  ranking cannot change the submission at all. A pool of 0 is one collision-poor molecule away, and F4 is what
  happens then.

### Polarity mask and top-k — two latent defects

`_topk` (`:171-179`) is correct on the data at hand and wrong in two edge cases.

**F8 (minor)** — `:175` `k = min(k, S.shape[1] - 1)`. `np.argpartition(-S, k)` needs `k < S.shape[1]`, but the
`-1` also *reduces the number of neighbours returned* when the reference is small.
Trigger: 2 reference spectra, `k=3` → `k_eff=1`, so only one neighbour is returned.
**INFERENCE** — inert on Kaggle (353,408 reference spectra ≫ 20), but it makes the function untestable at small
`k` and would silently degrade any future "small reference" configuration. Suggest `k = max(1, min(k, S.shape[1] - 1))`
and assert the returned width.

**F9 (minor)** — `:174` masks opposite-polarity references by setting their similarity to `-1.0`, then selects
the top-k of the *masked* matrix. If fewer than `k` same-polarity references exist, the tail of the returned
neighbours is made of `-1.0` rows.
Impact: `-1.0` survives to `:215` as `w = clip(-1, 0, None) + 1e-6 = 1e-6`, so those slots contribute ~0 weight.
**FACT** — measured on the real reference: 0 opposite-polarity rows are selected for any of the 1,213 visible
spectra (`a13_divzero.py`: `sims` min 0.114, `n_nan=0`, `n_-1=0`). Suggest masking by filtering indices rather
than by sentinel value.

### `featurize` — correct except for the `inf` precursor crash (F1)

Checked piece by piece on the real data:

- **FACT** — peak ranking is per row and intensity-descending: `np.lexsort((-it, row))` at `:95`, then
  `rank < cfg["topk"]` at `:99`. Verified with a hand-built row where two peaks share an m/z bin and intensities
  decide the order; duplicate bins sum correctly via `X.sum_duplicates()` at `:112` (`a4_edge.py`).
- **FACT** — L2 normalisation at `:113-115` is correct, and the `nrm[nrm == 0] = 1.0` guard prevents a 0/0 on
  an all-zero row (`a4_edge.py`, `a12_nl.py`).
- **FACT** — neutral-loss bins land at `n_frag + nb`; `n_frag = 150000` at `bin_w=0.01`, so a 90 Da loss lands
  in column 159000 as expected (`a12_nl.py`).
- **FACT** — non-finite / non-positive intensities and out-of-range m/z are filtered at `:93-94` and `:103`, and
  the 1500 Da fragment cap is exact: 1500.000 is dropped, 1499.999 is kept (`a4_edge.py`).

**F1 (critical)** — `:107` `nb = (nlv / cfg["bin_w"]).astype(np.int64)` casts a non-finite `nlv` to `INT64_MIN`.
For `+inf`, `nlv > 0.5` at `:108` is **True**, so the `INT64_MIN` value survives into `cols` at `:109` and
`csr_matrix` rejects it at `:110`.

```
trigger:  precursor_mz = +inf  (or 1e300) on any one test row
result:   ValueError: negative axis 1 index: -9223372036854775808
          raised from featurize, inside knn_scores, on the whole test frame
          -> main() never reaches sub.to_csv -> NO submission.csv
FACT      a17_modes.json: injected_single_modes["precursor_+inf"].raised
FACT      a17_modes.json: injected_single_modes["precursor_huge_1e300"].raised
```

`-inf` is *not* caught this way (`(-inf) > 0.5` is False) — it falls through to F4 instead. So the one input that
kills the run is not caught by the same guard that catches its mirror image.

**Suggested fix (text only).** Make the finiteness test part of the row filter rather than a side effect of the
comparison. In `featurize`, extend the mask at `:93` to the precursor as well, e.g. fold
`np.isfinite(pmr)` into `ok` (broadcast `pmr` before `row`), or at minimum clip before casting:
`nb = np.clip(nlv / cfg["bin_w"], 0, n_nl).astype(np.int64)` and add `np.isfinite(nlv)` to `m2`.

### Fingerprint weighting in `knn_scores` — works, but is measurably inert (F7)

`:212-216` builds each spectrum's fingerprint as `sum_i w_i * bits_i / sum_i w_i` with
`w_i = clip(sims_i, 0, None) + 1e-6`.

**FACT** — this is not a no-op *in form*: on the 248 scorable S2 proxy molecules the shipped prediction and a
plain unweighted top-20 mean have median pairwise cosine 0.971, and using rank-based weights instead gives
median cosine 0.971 (`a8_ranking.json`).

**FACT** — but it is a no-op *in effect*: the unweighted mean reproduces the shipped top-25 set exactly for
78.6% of molecules (mean Jaccard 0.977), and scores **better** on the proxy:

| variant | S2 MRR@25 | coverage@25 |
|---|---|---|
| A shipped (similarity-weighted) | 0.5773 | 0.9395 |
| B unweighted top-20 mean | **0.5861** | 0.9355 |
| C top-1 neighbour only | 0.4394 | 0.8992 |
| D global mean fingerprint prior | 0.1192 | 0.5323 |
| E inverted (control) | 0.0616 | 0.3669 |

(`a8_ranking.json`, `a9_reconcile.json`; all on the same 248-molecule denominator, tie-break by `fuse`.)

**INFERENCE** — `research/analysis/submission_v1.md:19` describes the prediction as a "similarity-weighted mean".
The measurement does not support that the weighting contributes anything; the `w` term is close to a constant
because neighbour cosines within a ±5 ppm window are uniformly high. Variant C shows the `k=20` average *is*
worth a lot (0.577 vs 0.439), so the gain is from the neighbourhood size, not the weighting.

**Suggested fix (text only).** Either drop `w` (simpler code, no measured loss) or state in `submission_v1.md`
that the weighting is retained for reasons other than proxy MRR. Do not cite it as a contributing component.

### Tie ordering — deterministic, and the one thing that must stay

- **FACT** — `fuse` (`:256`) sorts by `(-score, ik)`. Re-running the whole kNN branch from scratch reproduces
  the cached S2 top-25 **set** for 243/248 molecules and the full **order** for 237/248; the 5 differences are
  exact-cosine ties resolved differently (`np.argsort` stable-by-pool-position vs `fuse` by-InChIKey)
  (`a9_reconcile.json`).
- **FACT** — the reported proxy number reproduces exactly. `submission_v1.md` reports S2 `knn_only` = 0.573;
  `a9_reconcile.json` gets 0.5773 over the 248 molecules that have a non-empty pool, and
  `0.5773 * 248/250 = 0.5726` — the 0.5726 in `results/kaggle_v1_proxy/per_target_base.csv`. The two
  empty-pool molecules score 0 and are included in the denominator.

This also means F6's NaN ordering hazard is the *only* thing standing between the code and a stable tie-break, so
it must be fixed before F2 can be.

### Molecule-level mean vs median precursor (F12, minor)

`:229` averages the per-spectrum fingerprints of every spectrum of a molecule, while `:224` takes the **median**
precursor to define the ±5 ppm window. For a multi-spectrum molecule with two different precursors, the averaged
fingerprint blends fragments measured against two different neutral masses and is then compared against a window
centred on only one of them.

- **FACT** — visible test: 400 molecules / 1,213 spectra, so averaging is genuinely multi-spectrum. The adduct
  mix is asymmetric — 959 `[M+H]+` against 193 `[M-H]-`, 31 `[M+CH2O2-H]-` and 22 `[M+Na]+`
  (`v1_run_report.json`) — and `knn_scores` pools all of a molecule's spectra regardless of polarity, so a
  molecule measured in both modes has its positive- and negative-mode fragments averaged into one vector.
- **HYPOTHESIS** — not measured. I could not construct a clean counterexample without labels, and the effect is
  confounded with the intended multi-spectrum pooling. Flagging for V2, not for a V1 rerun.

### Spectra without peaks — dropped, correctly, but silently

- **FACT** — `:209` `has_peaks = np.diff(Xq.indptr) > 0` and `:222` `if not len(rows): continue`. A molecule
  whose spectra all lack peaks is skipped and becomes a placeholder.
- **FACT** — on the visible test, 1,213/1,213 spectra have peaks and all 400 molecules are scored
  (`v1_run_report.json`).
- **FACT** — the three ways a spectrum can end up with no peaks all reach this path silently
  (`a17_modes.json`): empty `ms2_mzs`, all-zero intensities, all-negative intensities. In each case
  `scored_new_molecule: false` and no warning is emitted.

---

## 2. Leakage in `proxy_eval.py` scenario S2

**Answer: S2 is clean.** The gate is incomplete, but I could not find leakage on any of the three paths the
brief asks about.

| path | question | answer |
|---|---|---|
| kNN references | can a held-out target / tautomer alias / parent group still vote? | **no** — **FACT**: re-running `load_reference`'s SQL with the module's own `AD10` and `ref_exclude_libs`, the un-excluded reference is 353,408 spectra, the exclusion removes exactly 4,783 of them, and the result is **348,625 — an exact match for the pipeline's reported S2 reference**. Since the SQL filters on `inchikey14 NOT IN held_keys` before `rn <= 3`, no held key can survive (`a20_s2ref.py`). 252 of the 285 held keys appear in `train.parquet` at all |
| library candidates | can a held key reach them? | **no** — **FACT**: `leak_library_candidates = 0`; independently, S2/S1 hits are 0/249 versus 249/249 (`a3b_leak.json`) |
| candidate universe | can a held key be predicted? | **no for train-only keys** — **FACT**: `leak_universe_train_rows = 0`; 284 of 285 held keys are legitimately still reachable because they are COCONUT entries |

That last row is by design and correct: S2 removes a molecule from the *library*, not from the candidate set, so
a COCONUT target remains a legitimate Class-2 answer.

**FACT** — alias coverage: all 250 proxy targets appear in `results/exp012_coconut/target_aliases.json`
(1,250 entries), and 229/250 have a non-empty parent group that must be excluded too. 285 held keys are removed
in total.

**F11 (minor)** — `proxy_eval.py:77-82` asserts only `leak_library_candidates` and `leak_universe_train_rows`.
`ref_exclude_sql` is passed at `:69` but **nothing verifies it took effect**, and `run_branches`
(`casmi_v1_kaggle.py:296-297`) returns only the reference *count*, so the gate has no access to the reference
keys. The exclusion is correct today only because the SQL is right.

**Suggested fix (text only).** Have `run_branches` return `len(set(ref[1]) & held)` — or the reference keys
themselves — and add a third `assert rep["leak_knn_references"] == 0` next to the existing two. This is
defence-in-depth: it converts a silent regression in one SQL string into a loud failure.

---

## 3. Hidden-rerun robustness

Target shape from the brief: ~1,500 spectra / 400 molecules, Kaggle 30 GB / 4 CPUs.
Actual visible test: 1,213 spectra / 400 molecules. Injecting 4 rows gave 1,217 spectra, so the tests below run
at the right scale.

### F1 — `+inf` precursor kills the entire run (critical)

See §1. One row is enough. This is the only finding that produces **no output at all**.

### F2 — an unresolvable kNN reference silently produces a garbage submission (critical)

`knn_scores:204-206` maps reference InChIKeys to universe rows and drops the misses:

```python
rrow = np.array([U.row.get(k, -1) for k in rik])
ok_ref = rrow >= 0  # reference molecules without a universe fingerprint cannot vote
Xr, rrow, rneg = Xr[ok_ref], rrow[ok_ref], rneg[ok_ref]
```

`reference_without_fp` is recorded at `:218` and **never asserted**. I forced every reference key to miss:

```
trigger:   any (reference, universe) mismatch -- e.g. the universe asset is rebuilt or
           re-versioned while train.parquet is unchanged, or drop_train_iks is applied
           to the reference instead of the universe
result:    reference_without_fp = 353408  (100% of the reference discarded)
           RuntimeWarning: invalid value encountered in divide   (0/0 at :216)
           every candidate score is NaN
           molecules_scored still 400;  no exception raised
           validate(...) -> fatal_ok = true
           only 1.75% of rows match the correct run
FACT       a17_modes.json: unresolvable_reference
```

The mechanism is a chain of three unasserted steps: `Xr` becomes `(0, nnz)`, so `S` has zero columns, so
`np.argpartition` returns an empty neighbour set, so `w.sum(1) == 0` at `:216` and `preds` is all-NaN, so every
cosine at `:231` is NaN, so `fuse` at `:256` sorts NaNs arbitrarily (F6). The run then writes a
valid-looking `submission.csv` that scores near zero.

**Suggested fix (text only).** Add `assert not (~ok_ref).any(), f"{int((~ok_ref).sum())} reference spectra have no universe fingerprint"`
immediately after `:206`. This one assertion converts the whole F2/F6 failure mode into a loud error.

### F3 — unknown adduct → silent drop → placeholder (major)

`:207` `q = test[test["adduct"].isin(AD10)]`. There is no assertion, and the count is only a log field.

```
trigger:   one molecule whose only spectrum has adduct "[M+2H]2+"
result:    spectra_non_ad10 = 1;  the spectrum is dropped before featurisation;
           the molecule never enters `out`;  its row becomes PLACEHOLDER_SMILES = "C"
FACT       a17_modes.json: injected_single_modes["unknown_adduct_[M+2H]2+"]
```

**FACT** — the visible test uses 7 of the 10 `AD10` adducts (`v1_run_report.json`: `[M+H]+` 959, `[M-H]-` 193,
`[M+CH2O2-H]-` 31, `[M+Na]+` 22, `[M+NH4]+` 4, `[M+Cl]-` 2, `[M+K]+` 2). The other three are never exercised, and
the hidden test is drawn from natural products at a different scale.

**INFERENCE** — the mass is then computed as `precursor_mz - AD10[adduct]` at `:217`, so an unknown adduct also
makes the mass wrong, not just the polarity. Getting this wrong silently zeroes the molecule.

**Suggested fix (text only).** In `main`, after `load_test`, assert
`set(test["adduct"]) <= set(AD10)` and raise with the offending values. If the intent is to tolerate extras, route
them through the neutral-mass map explicitly instead of dropping the row.

### F4 — `NaN` / `-inf` precursor → empty window → placeholder (major)

`Universe.window` with a non-finite `M` returns an empty slice, and `:226-228` records the molecule and moves on:

```
FACT  a11_pin.json: window_probe
      M=300.0  -> 5 candidates
      M=nan    -> 0 candidates
      M=inf    -> 0 candidates
FACT  a17_modes.json: injected_single_modes["precursor_NaN"]
      empty_pool_molecules=["900002"], scored_new_molecule=false,
      warning "invalid value encountered in cast" from :107
```

**FACT** — `np.median([100.0, nan])` is `nan` (`a11_pin.json`), so **one** bad spectrum poisons a multi-spectrum
molecule's window entirely.

**INFERENCE** — combined with the F6/F5 chain, a hidden rerun containing any non-finite precursor converts that
molecule to `"C"` with no error and no non-zero `placeholder_rows` check.

**Suggested fix (text only).** Guard in `main`: drop or quarantine rows where
`~np.isfinite(test["precursor_mz"])`, and count them in the run report. Independently, make
`knn_scores:226-228` escalate: an empty pool is a data problem, so append it to the report *and* fail the run if
it happens for a molecule that did have peaks.

### F5 — placeholders pass validation (major)

`main:338` writes `PLACEHOLDER_SMILES` when a molecule has no predictions; `:343` counts them in
`report["fusion"]["placeholder_rows"]`; `validate` (`:267-279`) has no placeholder check at all, and
`main:347` only gates on `fatal_ok`.

This contradicts `research/analysis/submission_v1.md:23`, which states: *"Placeholder `C` only if a molecule has
no candidates."* In fact a molecule also gets a placeholder when it was **never evaluated at all** — non-AD10
adduct (F3), no usable peaks, or a non-finite precursor (F4) — so the run report's `placeholder_rows: 0` is
being used to mean something stronger than it measures.

```
trigger:   any of F2/F3/F4 applied to every molecule
result:    validate -> fatal_ok = true, so main() writes the file and returns normally
FACT       a22_ph.py: a 400-row submission whose every row is exactly "C" validates as
           {"rows": 400, "min_preds": 1, "max_preds": 1, "empty_fields": 0,
            "invalid_smiles": 0, "any_nan": false, "fatal_ok": true}
           methane is a valid SMILES and "C" is neither empty nor NaN, so every clause
           of the fatal_ok conjunction is satisfied by a submission that scores zero
FACT       a10_robust.json: end-to-end, a run with 3 of 403 molecules lost still
           reported validation_of_degraded_submission.fatal_ok = true
```

**Suggested fix (text only).** Add `"placeholder_rows": 0` to the `fatal_ok` conjunction in `validate`, or raise
in `main` when `report["fusion"]["placeholder_rows"] > 0`. Since the visible run has 0, this costs nothing.

### F6 — NaN propagation through the neighbour weights (major)

See the chain under F2. Two independent defects:

1. `:216` `(w[:, :, None] * bits).sum(1) / w.sum(1, keepdims=True)` — when the neighbour set is empty,
   `w.sum(1) == 0` and `preds` becomes NaN. **FACT** — reproduced as a
   `RuntimeWarning: invalid value encountered in divide` (`a17_modes.json`, `warnings_emitted`).
   **FACT** — on the real reference this never fires: `w.sum` min is 2.507 across all 1,213 spectra, and
   `sims` min is 0.114 with 0 NaN and 0 masked `-1` rows (`a13_divzero.py`, output in `a13_stdout.txt`).
2. `:256` `sorted(..., key=lambda kv: (-kv[1], kv[0]))` — `-nan < x` is `False` for every `x`, so the sort key is
   degenerate and the output order is arbitrary rather than the intended `(score desc, ik asc)`.

**Suggested fix (text only).** At `:216`, guard with `den = np.maximum(w.sum(1, keepdims=True), 1e-12)`. In
`fuse`, replace `(-kv[1], kv[0])` with a key that is total over NaN, e.g.
`key=lambda kv: (0 if np.isnan(kv[1]) else 1, -kv[1] if not np.isnan(kv[1]) else 0.0, kv[0])` so NaN candidates
sort last deterministically instead of arbitrarily.

### Duplicated spectrum ids — handled correctly

**FACT** — the visible test has 0 duplicate `spectrum_id` (`a11_pin.json`: `dupe_before = 0`), and
`main:316-317` makes the key unique before anything reads it: appending
`"#" + groupby("spectrum_id").cumcount()` yields 0 duplicates for the real frame and for a frame with an injected
duplicate (`a11_pin.json`: `dupe_after_main_style_dedup = 0`, `dedup_key_is_molecule_scoped = true`).

Two notes, neither a bug:
- **INFERENCE** — the suffix is keyed on `spectrum_id` alone, not `(molecule_id, spectrum_id)`, so a legitimate
  cross-molecule id collision also gets suffixed. Harmless: with `tau=None` no downstream code reads
  `spectrum_id` (`:321` only uses `molecule_id`).
- **FACT** — `astype(str)` at `:317` turns a null `spectrum_id` into `"nan"`, which `groupby` then treats as one
  group, so nulls are disambiguated too. No latent crash.

### Empty 5 ppm window — same silent-placeholder path as F4

**FACT** — 0 empty pools on the visible test, but the minimum pool is 1 and 9 molecules have ≤3 candidates
(`a18_pools_mem.json`). With a 0.5 mDa half-width at `M=100` and a universe whose median mass gap is much larger,
an empty window is entirely plausible for a molecule whose mass happens to fall in a sparse region. The
behaviour is the F4 path: `empty_pool_molecules` is appended, the molecule becomes `"C"`, `fatal_ok` stays true.

### Memory on Kaggle (30 GB, 4 CPUs) — comfortable

**FACT** (`a18_pools_mem.json`, `a11_pin.json`):

| item | measured |
|---|---|
| reference | 353,408 spectra / 92,410 molecules / 30,399,402 nnz |
| marginal cost of one worker `_init` | **243.5 MB** |
| RSS with 4 worker copies held | 1,754.1 MB |
| universe fingerprints | 186.7 MB |
| parent peak working set, full `knn_scores` on the real test | **2,121.8 MB** |
| `knn_scores` wall time, `workers=1` | 3.2 s warm / 60.1 s cold |

**INFERENCE** — with `workers=4` the pool path costs the parent (which does *not* call `_init`) plus four 243.5 MB
copies, and the `workers=1` path measured 2,121.8 MB including one copy; the whole job therefore needs roughly
2 GB, about 6% of the Kaggle 30 GB budget. Memory is not a risk, even at the measured worst case of four
simultaneous reference copies.

**FACT** — `reference_without_fp = 0` and all 353,408 reference keys resolve (`a17_modes.json`:
`valid_reference.unresolved_keys = 0`), so the F2 path is not live today.

### F13 — Windows-only unlink failure (minor, local runner only)

`neighbours:196` `rp.unlink(); npth.unlink()` after `np.load` in `_init`.
**FACT** — on this machine a second `neighbours` call in the same `work` directory raises
`PermissionError: [WinError 32] ... 'D:\\...\\_ref_neg.npy'`, *after* all the compute is done, so the result is
lost. Kaggle is Linux, where `unlink` on an open file succeeds; this only affects `--mode local` on Windows.
**Suggested fix (text only).** Wrap in `try/except OSError: pass`, or use a `tempfile.TemporaryDirectory()`.

### F10 — mismatched peak list lengths are truncated, not rejected (minor)

`featurize_df:129` `lens = np.array([min(len(a), len(b)) for a, b in zip(mzs, its)])`.
**FACT** — with 3 m/z and 1 intensity, `featurize_df` silently drops the 2 surplus peaks and the molecule is
scored normally (`a17_modes.json`: `mismatched_list_lengths.scored_new_molecule = true`, `n_pool = 41`).
`featurize_arrow` (used for the reference) has no such guard — **FACT** it raises
`ValueError: operands could not be broadcast together with shapes (3,) (2,)`.
**FACT** — the real data has 0 mismatched rows in either train or test (`a1_data_facts.json`), so this is latent.
**INFERENCE** — `featurize_df` is the safer behaviour and `featurize_arrow` is the dangerous one: the two
entry points disagree on what a malformed row means. Suggest asserting `len(a) == len(b)` in both, or matching
`featurize_arrow`'s strictness.

### F14 — the run report cannot prove a rerun matched (minor)

`main:309` writes `cfg`, `workers` and timings, but no library versions.
**FACT** — `results/kaggle_v1_local/v1_run_report.json` has no `versions` key. Measured locally: Python 3.14.0,
NumPy 2.4.4, pandas 3.0.5, SciPy 1.16.3, DuckDB 1.5.5, RDKit 2026.03, matchms 0.33.1.
**Suggested fix (text only).** Add a `{name: module.__version__}` block for numpy/scipy/pandas/duckdb/rdkit and
the sha256 of `universe.parquet` / `universe_fp.npy` to the report. Fingerprints silently change the output, and
their hashes are the cheapest reproducibility guard available.

---

## Calibration: how much of the proxy score is real

Worth stating plainly, because the number in `submission_v1.md` reads stronger than it is.

- **FACT** — S2 pool coverage (truth anywhere in the candidate pool) is 0.972, and the median pool is 49
  (`a9_reconcile.json`). Drawing 25 of 49 at random and asking for MRR@25 gives **0.142** (sd 0.245 over 200
  draws/molecule), with top-25 coverage 0.560.
- **FACT** — the shipped rule scores 0.5773 on the same pools: **4.07× the random baseline**
  (`a9_reconcile.json`, `random_in_pool_baseline`).
- **FACT** — 33.9% of S2 molecules and 22% of visible-test molecules have pools of ≤25
  (`a9_reconcile.json`, `a18_pools_mem.json`), where the ranking cannot affect the top-25 set at all.
- **FACT** — a global constant prior scores 0.1192 and an inverted control 0.0616 (`a8_ranking.json`), so the
  spectrum is carrying real signal; this is not a degenerate setup.

**INFERENCE** — roughly 0.43 of the reported 0.573 is attributable to discrimination above a same-pool random
baseline, and the rest is pool size. Any future comparison between V1 and a V2 candidate should be run against
that 0.142 baseline too, or a change that only reorders an already-correct small pool will look like progress.

---

## What I could not measure

- **Whether F12** (mean across spectra vs median precursor) costs accuracy. Needs labels; not constructed.
- **The actual effect of the +inf/NaN robustness fixes** — they are latent on the visible data, so I can only
  show that the trigger crashes or degrades, not that the fix recovers score.
- **Whether the hidden test's adduct set matches the visible one.** The brief's premise is a different scale and
  composition, so F3 is a risk I can demonstrate but cannot quantify.
- **Anything about V2/V3.** Out of scope for this task.
- **A `1e300`-precursor filter cost.** No performance measurement; the fix is a comparison, not a loop.

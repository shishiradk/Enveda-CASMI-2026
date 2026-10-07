# E4 = E3b + PubChem popularity prior

Date: 2026-10-02. Code: `research/kaggle_e1/e4/`. Data: `results/kaggle_e4/`.
Kaggle (both private): dataset `shishiradhikari11/casmi-pop-lookup`, notebook `shishiradhikari11/casmi-e4-eligible-pop`.
No competition submission was made. No existing file, dataset or notebook was changed.

## 1. Conclusion

- **E4 ran cleanly on Kaggle.** Notebook `shishiradhikari11/casmi-e4-eligible-pop` **version 2** (full visible test,
  T4, internet off, private) is ready to submit. It was not submitted.
- **Visible test: 0 of 400 first places changed** against E3b; labelled MRR@25 stays 1.000. Every first place is a
  library-protected candidate. Below it the lists change a lot: order of the top 25 in 395 rows, membership in 364.
- **Lookup coverage.** 91.9% of the 772,653 pool structures have a PubChem record, 18.6% a PubMed link. In the run
  20,510 of 21,381 listed candidates (95.9%) had a record; none was unknown to the table.
- **Design choice.** The prior is added to the z-score of a real score (logit blend of the two raw ranker
  probabilities, made monotone along the engine's list), not to the z-score of the exported rank blend. On the bench
  this halves the loss for unpopular truths at the same gain for popular ones.
- **Expected effect is not known.** The bench gain (+0.088 on S2) comes from the popularity of the bench truths; with
  COCONUT-typical counts for the truth the same setting loses 0.057 on S2. The public +0.010 was measured on another
  stack. Realistic range on the leaderboard: about -0.01 to +0.01, inside the noise.
- GPU time used: 1.66 h (smoke 0.22 h, full run 1.44 h).

## 2. The method in the public notebook

Source: `casmi26-sub-v4b-0-409-on-the-public` (huseyinemreaksoy), module `fusion_core.py` and the last code cell. Read for
the method only; none of its code or data is used.

- **Feature.** One number per pool structure: `log1p(substance records) + log1p(PubMed links)`, from two arrays aligned to
  the rows of that stack's own candidate pool (a non-eligible dataset).
- **Where.** First step of the final stage, per molecule, on the ranker's top 60 (de-duplicated), before the
  forward-model re-ordering.
- **Score it is added to.** The ranker output (mean of LightGBM boosters, a real-valued score), standardised over all
  top-60 entries of the molecule: `z = (s - mean) / (population sd + 1e-9)`. Then `f = z + mu * pop`.
- **What is re-sorted.** All pool candidates of the top 60 are sorted by `f` and written into the slots that pool
  candidates occupied. It is a re-sort of the whole list, not of formula groups. Candidates made by the structure
  generator have no pool row: they get no prior, keep their slots and are put into formula groups of their own.
- **Carried forward.** `f` replaces the ranker score. The forward-model step that follows computes
  `z(f) + lam * z(ICEBERG) + lam * z(GLACIER)` inside same-formula groups, so the prior also counts there.
- **Library hits.** No exception for the prior: the switch that would turn it off for molecules with a library match
  >= 0.9 exists (`POP_LIB_OFF`) and is off. Only the forward models are switched off for those molecules.
- **Candidates without counts.** A pool row without a PubChem record has the value 0 and competes with it.
- **Other channels.** The PubChem-only lists and the second engine's lists get no prior.
- **Reported effect.** 0.399 -> 0.404 with mu 0.15 (together with a fragment re-score), 0.409 with mu 0.25. Single
  submissions. The author states that the prior is not spectrum evidence and was worse than no prior on a synthetic
  library.

## 3. Our implementation

Files in `research/kaggle_e1/e4/`:

| file | content |
|---|---|
| `build_lookup.py` | builds the lookup table and `coverage.json` |
| `check_tautomer_miss.py` | sample check of PubChem records missed because of the tautomer form |
| `pop_stage.py` | the prior: lookup, ranker term, permutation (embedded in the notebook) |
| `e4_stage.py` | prior + E3b's rule on the adjusted lists (embedded in the notebook) |
| `build_e4.py` | notebook builder; patches the pushed E3b notebook by asserted text replacements |
| `test_e4.py` | unit tests and the replay of E3b's Kaggle run |
| `bench_e4.py` | the stage on the bench lists of the E3 diagnosis |
| `check_run.py` | checks of a downloaded kernel output |

### Order of the steps (per molecule)

1. Engine list (up to 60 candidates) as in E3b. The runner now also exports the two raw ranker probabilities
   (`pv`, `ours`) of every listed candidate; nothing else in the engine changed.
2. E3b's cell runs unchanged: GLACIER and ICEBERG score the same-formula groups of the unprotected candidates and the
   E3b lists are formed. They are the fallback and the reference.
3. Popularity prior on the engine list (`pop_stage.run_pop_stage`):
   - `pop = log1p(n_sid) + log1p(n_pmid)`; 0 for a candidate without a PubChem record.
   - `f = z(ranker term) + mu * pop`, z over all entries of the top 60 (population sd, as in the public notebook).
   - Candidates with their own library similarity >= 0.6 keep their slots (E3b's protection). All other candidates are
     sorted by `f` into the slots the unprotected candidates occupied. Ties keep their order.
4. E3b's rule is applied again, to the adjusted list, with the forward scores of step 2 (`e4_stage.apply_forward`,
   which calls E3b's `select` and E3's `rerank_molecule`). The prior only permutes unprotected candidates inside the
   top 60, so the set of candidates that E3b sends to the forward models is the same and no second GPU pass is needed.
5. Safety: the replay of step 4 on the unadjusted list must give the E3b lists of step 2 key for key, otherwise the
   stage raises. Any failure in steps 3-5 keeps the E3b lists; a failure of the forward stage leaves the prior alone
   on the engine lists.

`POP_MU = 0.25` is the single setting (`build_e4.py --mu`). `POP_MU = 0` skips the stage and gives E3b.

### Design choice: the ranker term

The engine exports a within-molecule rank blend (0.88 rank(pv) + 0.12 rank(ours)). Its z-score is a straight line in
list position whatever the ranker's certainty; `e3_diagnosis.md` names this as the root cause of E3's loss. A prior
added to it would overrule a confident ranker as easily as a guess.

Chosen: **`score_mode = 'logit'`**.
- `L = 0.88 * logit(pv) + 0.12 * logit(ours)` (probabilities clipped to [1e-6, 1 - 1e-6]); this is the `zl_bl`
  quantity of `e3c_real_score.md`.
- `L` is made non-increasing along the engine's list by isotonic regression (pool-adjacent-violators). The engine's
  order is kept exactly; only the margins come from `L`. Where `L` disagrees with the list order the candidates are
  tied, and popularity decides between them.
- This keeps "mu = 0 is E3b" exact, which a plain sort by `L` would not.

Bench comparison of the two terms (section 5): same gain when the truth is popular, less than half the loss when it is
not. `score_mode = 'rank'` (the exported blend) remains as an option.

### Differences from the public notebook

| point | public notebook | E4 | reason |
|---|---|---|---|
| ranker term | z of the LightGBM score | z of the isotonic logit blend | our export is a rank; see above |
| library matches | prior applies to every molecule | candidates with own library similarity >= 0.6 keep their slot | E3b's protection, required |
| score seen by the forward step | `f` (prior included, with its margins) | the rank-blend score of the slot | keeps mu = 0 identical to E3b; the forward step treats the adjusted order as it treats the engine's order. The forward models can therefore move a popular candidate back down more easily than in the public notebook |
| candidates without a pool row | generated candidates keep their slots | none exist; every candidate is a pool structure | |
| counts | one CID per pool row | summed over all CIDs with the same InChIKey first block | task definition |

### Tests (`test_e4.py`)

- Lookup: present, zero-count and absent keys; unsorted table rejected; plain key before engine key.
- Missing counts: candidates unknown to the table get 0 and compete; equal counts or an empty table change nothing.
- Protected candidates never move, even without a record; without protection they are overtaken.
- Ties keep their order; entries after the window never move; a pair the logits order the other way is tied.
- Missing or non-finite raw scores: the molecule keeps its order; if no molecule has them the stage raises.
- mu = 0 is the identity on 50 random lists, in both score modes.
- Forward replay: mu = 0 gives E3b exactly; a tampered E3b list is detected; forward stage off gives the prior alone.
- Real data: the replay on E3b's Kaggle output reproduces **400 of 400** submitted rows of E3b version 1.

## 4. Lookup table

`results/kaggle_e4/pop_lookup/pop_lookup.npz` (7.3 MB): 772,653 sorted keys (S14), `n_sid` and `n_pmid` (int32).
Dataset `shishiradhikari11/casmi-pop-lookup`, private, CC0-1.0, with `ATTRIBUTION.txt`.

- **Pool.** COCONUT 436,389 structures, ChEBI / LIPID MAPS 141,718, train 275,810 distinct `inchikey14`
  (277,566 distinct SMILES; read with DuckDB). 772,653 distinct keys in total.
- **Keys.** The pool keys equal the plain InChIKey first block of the pool SMILES for 100.00% of the rows. The key the
  engine exports at run time is the tautomer-canonical one; it equals the pool key for 94.1% (sample of 3,000, RDKit
  2026.03.6).
- **Join.** Plain InChIKey first block of the pool SMILES = `ik` of `pubchem_rows_pop.parquet`; counts summed over all
  CIDs with that `ik` (1,872,059 CIDs for 710,210 keys; median 2 CIDs per matched key). At run time each candidate is
  looked up by the plain key of its SMILES, then by the engine key. Zero-count rows are stored, so "no record" and
  "unknown to the table" are told apart.
- **Tautomers not caught.** PubChem rows were not tautomer-canonicalised (101 M rows). Sample check: of 225 pool
  structures without a match (mass < 900, at most 4,000 PubChem isomers), 25 (11%) have a PubChem row under another
  plain key with the same tautomer-canonical key (median 22 substance records). About 0.9% of the pool therefore has
  the value 0 although PubChem knows the compound. The control on matched structures did not finish (30-minute limit).

| source | structures | with a PubChem record | with a PubMed link | n_sid quartiles (matched) | n_pmid 75% / 90% / 99% (matched) | n_sid <= 3 (matched) | pop median (all) |
|---|---|---|---|---|---|---|---|
| COCONUT | 436,389 | 91.7% | 21.0% | 4 / 9 / 18 | 0 / 2 / 17 | 24.7% | 2.30 |
| ChEBI / LIPID MAPS | 141,718 | 95.5% | 41.1% | 7 / 15 / 41 | 4 / 15 / 1,498 | 6.5% | 3.00 |
| train | 275,810 | 91.9% | 12.3% | 2 / 4 / 9 | 0 / 2 / 404 | 41.2% | 1.61 |
| all | 772,653 | 91.9% | 18.6% | 3 / 7 / 16 | 0 / 2 / 78 | 29.7% | 1.95 |

`pop` over all pool structures: 10% 0.69, 25% 1.10, 50% 1.95, 75% 3.00, 90% 4.49, 99% 9.43, maximum 22.

## 5. Bench check (`bench_e4.py`)

E4's own functions on the 250 bench molecules of the E3 diagnosis (honest ranker scores, Kaggle forward scores).
Paired MRR@25 difference against E3b (S1 0.867, S2 0.725), 10,000-resample bootstrap.

| scenario | term | mu | actual counts | truth with 1 substance record, no PubMed link | truth with the counts of a random COCONUT structure |
|---|---|---|---|---|---|
| S1 | logit | 0.15 | +0.008 [+0.001, +0.018] | -0.004 [-0.009, -0.001] | +0.000 |
| S1 | logit | 0.25 | +0.011 [+0.001, +0.025] | -0.004 [-0.009, -0.001] | -0.001 |
| S1 | rank | 0.25 | +0.013 [+0.003, +0.027] | -0.008 [-0.015, -0.002] | -0.001 |
| S2 | logit | 0.15 | +0.066 [+0.042, +0.091] | -0.059 [-0.079, -0.041] | -0.032 |
| S2 | logit | 0.25 | +0.088 [+0.058, +0.118] | -0.100 [-0.125, -0.076] | -0.057 |
| S2 | logit | 0.40 | +0.081 [+0.050, +0.112] | -0.159 [-0.190, -0.129] | -0.090 |
| S2 | rank | 0.15 | +0.072 [+0.044, +0.102] | -0.182 [-0.216, -0.150] | -0.097 |
| S2 | rank | 0.25 | +0.083 [+0.053, +0.115] | -0.261 [-0.301, -0.223] | -0.147 |

- The "actual" column is an upper bound, not an estimate: the bench truths are textbook natural products (`pop`
  median 13.0; the other listed candidates 2.5; the COCONUT pool 2.3). See `exp020_popularity_prior.md` section 4.
- The two counterfactual columns replace the truth's counts only. They show the cost when the truth is an ordinary
  compound: at mu 0.25 the S2 loss is as large as the S2 gain.
- S1 barely moves in either direction: its truths have a library match and are mostly protected.
- The logit term has the same upside as the rank term and 40% of its downside. This is the basis of the design choice.
- Correct top-1 answers demoted at mu 0.25, logit term: S1 0, S2 3 (29 truths promoted to top-1); with one substance
  record for the truth: S1 1, S2 26.
- Full grid: `results/kaggle_e4/bench_e4.json`.

## 6. Kaggle run

Notebook `shishiradhikari11/casmi-e4-eligible-pop`, private, T4, internet off; E3b's seven datasets plus
`shishiradhikari11/casmi-pop-lookup`. The pushed source equals `research/kaggle_e1/e4/casmi_e4.ipynb`.

| version | setting | wall time | result |
|---|---|---|---|
| 1 | smoke, first 16 molecules | 792 s (0.22 GPU h) | all stages ok |
| **2** | **full visible test, SMOKE off, mu 0.25, logit term** | **5,166 s (1.44 GPU h)** | **all stages ok; ready to submit** |

Version 2, from the log and `check_run.py` on the downloaded output:

- **Stages.** Engine 400 lists (2,984 s). Forward stage ok: GLACIER and ICEBERG each scored 18,617 of 18,617 sent
  candidates in 368 molecules; 812 candidates protected. Popularity stage ok (21 s); the E3b replay check passed.
- **Submission.** 400 rows, 2 / 25 / 25 SMILES per row (min / median / max), all unique within the row, no empty row,
  no fallback row.
- **Counts.** 21,381 candidates in the top-60 windows: 21,381 found by the plain key of their SMILES, 0 unknown to the
  table. 20,510 (95.9%) have a PubChem record, 2,155 (10.1%) a PubMed link. All 400 molecules have at least one
  candidate with a record. `pop` of the listed candidates: quartiles 1.10 / 1.61 / 2.20, 90% 3.30, 99% 6.67, maximum 16.7.
- **Same engine output as E3b's run.** The engine lists equal those of E3b version 1 for 400 of 400 molecules, and the
  E3b reference computed inside the E4 run equals E3b's submitted rows for 400 of 400.

| E4 against | top-1 changed | top-25 order changed | top-25 membership changed |
|---|---|---|---|
| E3b (same run, and E3b's own run) | **0** | 395 | 364 |
| E1 (engine order) | 0 | 395 | 367 |
| for comparison: E3b against E1 | 0 | 360 | 326 |

- **Top-1.** Nothing to list: the engine's first candidate is library-protected in 400 of 400 molecules, and all 812
  protected candidates are in their slots after E4. Labelled MRR@25 on the 400 visible molecules: 1.000 before and after.
- **Below the first place.** Per row, 3.8 of E4's 25 candidates are not in E3b's 25 (median 4, maximum 12); 26% of
  the slots hold the same candidate as in E3b; 5.1 of the 25 come from engine ranks 26-60 (E3b: 2.7).
- **Isotonic ties.** 8,474 neighbouring pairs (40%) are tied in the ranker term, i.e. the logit blend orders them
  differently from the rank blend; popularity decides those pairs.
- The 5 rows without any change have lists of 2 to 6 candidates.

Not done: no leaderboard submission; mu and the score term were not varied on Kaggle.

## 7. Risks

- **The prior is not spectrum evidence.** It ranks the well-documented isomer first. The public author reports that it
  hurts ordinary and synthetic compounds; our bench shows the same size of loss as of gain on S2 when the truth has
  COCONUT-typical counts. The sign on the hidden test depends on how popular its truths are, which we cannot measure.
- **Transfer from the public result is an assumption.** The +0.010 was measured on another stack (a real-valued ranker
  score, generated candidates, a PubChem channel, no library protection of single candidates) with single submissions;
  leaderboard noise is about +-0.016.
- **Where E4 can act.** Library-matched candidates are fixed, and `e3c_real_score.md` bounds the molecules with a
  listed, library-free truth at about 10% of the hidden test. A bench S2 effect of x is at most about 0.1 x on the
  leaderboard: roughly -0.01 to +0.009 at mu 0.25.
- **Strength inside the list.** One confident candidate inflates the standard deviation of the top 60, so the rest of
  the list is compressed: on the smoke run the median z gap between neighbours was 0.009, while one unit of `pop`
  is worth 0.25. Below a confident first candidate the list is in effect sorted by popularity, then re-ordered
  inside formula groups by the forward models. The public rule has the same property.
- **Only unprotected first places are exposed.** The visible test cannot show this risk: all of its first places are
  exact library matches.
- **Visible test says nothing about the gain.** It only shows that library answers are not touched.
- **Counts are undercounted for about 0.9% of the pool** (other tautomer in PubChem) and are sums over stereoisomers
  and isotopologues that share the InChIKey first block.
- **mu was not tuned by us.** 0.25 is the public value; the bench cannot choose it because of the popularity bias.
- **PubMed link types** in `CID-PMID.gz` were not checked (as in EXP-020).

## 8. Commands

```
python research/kaggle_e1/e4/build_lookup.py 3            # lookup + coverage.json                      (5 min)
python research/kaggle_e1/e4/check_tautomer_miss.py       # tautomer sample check                       (> 30 min)
python research/kaggle_e1/e4/test_e4.py <E3b output dir>  # unit tests + replay of E3b's Kaggle run
python research/kaggle_e1/e4/bench_e4.py                  # section 5                                   (15 min)
python research/kaggle_e1/e4/build_e4.py [--smoke N] [--mu MU] [--score logit|rank]
kaggle kernels push -p research/kaggle_e1/e4
python research/kaggle_e1/e4/check_run.py <E4 output dir> <E3b output dir>
```

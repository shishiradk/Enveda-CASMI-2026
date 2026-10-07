# C3 re-rank: seed-Tc-aware lambdarank on the C3NP bench (Task F)

Status: RECOMMEND — ship the lambdarank re-ranker on top of the rr_bt union. The 2-parameter hand rule family fails and is not recommended.

## Setup (fact)

- Bench inputs: mmp_edit/biotransform full + `_strip_edit1` candidate json, shipped `combo_rr_bt{sfx}.json` unions (= rr_bt), c3np `context.pkl` (zlog, analogs, window), evaluator `c3np_eval.py` (MRR@25; canonical-key match; warm `eval_canon_cache`, 316k keys).
- Feature tables (new, `results/c3gen/seedtc/feat_{default,strip_edit1}.parquet`, this repo): per candidate of the rr_bt union (<=200, order asserted == shipped json for every molecule): `rank_rr`, `rk_a/rk_b` (generator rank), `gen_a/gen_b`, `z` (6930-bit fp @ zlog, `_zfull_many` recipe), `z_z` (within-molecule z), `seed_tc` (max Morgan r2/2048 Tc to analog+window seeds, stripped set for the `edit1` mode), `ha_delta` (heavy atoms to closest seed). Truth only as training label; nb1/nb2/tc_band only for stratification.
- 550 mols; rows 106,488 (default) / 106,526 (strip); positives 416 / 387.
- CV: grouped (by molecule) 5-fold, stratified on nb1 (fold nb1 shares 0.577-0.582), deterministic seed. Lambdarank trained per fold on default-mode features only (deploy-realistic), applied to held-out molecules in both benches; fold-pooled ranked jsons re-scored by `c3np_eval`.

## Baselines (fact; from results/c3np/eval_*.json)

| method (bench) | overall | nb1-yes | nb1-no | hit25 |
|---|---|---|---|---|
| rr_bt (default)          | 0.5077 | 0.7568 | 0.1664 | 0.6527 |
| zlog (default)           | 0.2532 | 0.3236 | 0.1567 | 0.4691 |
| rr_bt (strip edit1)      | 0.3993 | 0.5676 | 0.1685 | 0.5818 |
| zlog_rrf nb1-no (tbl)    | 0.414  | -      | 0.191  | -      |

## Hand rule (fact)

2-param family `score = rrs + w0*max(0,(t1-seed_tc)/t1)*z_z`, `rrs=1/(rank_rr+10)`; grid w0 in {0.5,1,2,4} x t1 in {0.5,0.7,0.85,1.05}, same CV. Every cell degrades the rr_bt order: best cell (0.5,0.5) 0.212/0.166 nb1-no (default), 0.176/0.076 (strip); worst cells ~0.034 all. Z-score term saturates `rrs` (z_z in [-5,5] vs rrs in [0.005,0.09]) and seed_tc alone cannot be blended linearly without destroying order. Family rejected; the goal (nb1-no > 0.191) is unreachable inside it.

## Lambdarank (fact; results/c3gen/seedtc/cv_summary.json + eval_seedtc_lmb_*.json)

LGBMRanker: n_est 150, lr .05, 15 leaves, min_child 20, subsample/.colsample .9, reg_lambda 1, ndcg@25, features: rank_rr, rrs, rk_a, rk_b, rma, rmb, gen_a, gen_b, z, z_z, seed_tc, ha_delta.

| method (bench) | overall | nb1-yes | nb1-no | hit25 |
|---|---|---|---|---|
| lambdarank (default) | 0.5394 | 0.7760 | 0.2150 | 0.6527 |
| lambdarank (strip)  | 0.4418 | 0.6072 | 0.2150 | 0.5891 |

Goal check: nb1-no 0.2150 > 0.191 (zlog_rrf) and > 0.1664 (rr_bt) — clear. Strip overall 0.4418 >= 0.3893 floor (rr_bt 0.3993 - 0.01) — clear. top1 0.4945, hit10 0.62 (default).

Paired per-molecule bootstrap deltas (5000, seed 0, 95% CI):
- default overall +0.0317 [+0.0170, +0.0467]; default nb1-no +0.0487 [+0.0271, +0.0729]
- strip overall +0.0428 [+0.0281, +0.0581]; strip nb1-no +0.0465 [+0.0250, +0.0696]
- hand rule mrr25 = -0.295 [-0.329, -0.264] (family harm confirmed)

Feature importances (fold-averaged, split counts): seed_tc 436, z_z 373, z 299, rank_rr 265, rma 170, rmb 157, rk_b 116, rk_a 106, ha_delta 98, rrs 80, gen_* ~0.

## Deploy (inference)

Per molecule: build fp for the <=200-union candidates (r2/r3/RDKitFP/MACCS -> 6930 bits), `z = fp @ zlog`, `z_z`, `seed_tc`/`ha_delta` vs ctx seeds (edit themselves; seeds-B atoms ignored), rank by lambdarank pred. Runtime: 40 RDKit fp + a 64-bit-dot + GBM, ~ms/mol; no truth use. Miss the first-occurrence sweeps (few ms) and GPU net (excluded). One model, mode-agnostic features.

## LB estimate (inference)

Cache holds only the recipe; composite estimates use arithmetic + 2x haircut, hidden = Class-3-like + samples. If hidden nb1 share ~ the known bench (0.57 yes): lambdarank 0.5394 vs rr_bt 0.5077 -> ~+0.03 (n small); pessimistic nb1=0.25: 0.355 vs 0.314. Expectation: modest LB gain, no regression risk below rr_bt.

## Caveats

- Rule frequency/support (count of generator seeds/positive seen) not in shipped candidate jsons — the model instead uses `rk_*`/`rma` etc. Sub-optimal vs provenance, but deployable.
- Seeds-B (heavy-atom-shifted seed molecules) not reconstructible cheaply from context; `seed_tc` uses analog+window-atom seeds only (disclosed).
- Z-scored `z_z` needs the molecule's candidate list; at deploy the list is the own union (as in bench).
- Strip edit1 nb1-no identical to default nb1-no (92 mid rr changes, none nb1-no): the model is robust to removing edit1 seeds.

## Outputs (this repo)

- research/c3gen/seedtc_build.py, seedtc_rank.py
- results/c3gen/seedtc/feat_{default,strip_edit1}.parquet, cv_summary.json, rank_{hand_best,lmb}_{default,strip_edit1}.json
- results/c3np/eval_seedtc_{lmb,hand}_{default,strip}.json + _per_mol.csv
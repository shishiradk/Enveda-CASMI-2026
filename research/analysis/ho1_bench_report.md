# Held-out FPNet retrain (ho1) on the bench

Date: 2026-10-03.

**What was tested.** The two networks `fp_single_ho1.pt` and `fp_merged_ho1.pt`:
- trained by a teammate on an RTX 5050 with `research/train_pkg`;
- results in Kaggle dataset `akritirijal04/casmi-fp-ho1-results`, local copy in `models/fp_ho1_akriti/`;
- sha256 checked against her `check_summary.txt`.

**Validation during training (hard-negative top-1).**

| Network | Best | At step | Notes |
|---|---|---|---|
| single | 0.626 | 29k | stopped by the patience rule |
| merged | 0.673 | 39k | flat at about 0.665 from step 21k on |

The learning rate was still 1.6e-4 at step 40k, so the cosine schedule was cut off before it decayed.

**How the bench was run.**
- Command: `bench.py all --tag ho1 --alt-fp models/fp_ho1_akriti`.
- Her networks fill the "alt" slot, so they appear as the `*_mk` score sets. The public networks stay in the main slot.
- All public-net rows equal run e1 exactly, so the run is reproducible.
- Runtime: 79 min on CPU.

## Results: MRR@25, ho1 against the public networks on the same candidates

Paired differences with 95% bootstrap confidence intervals.

| Score set | S1 / S2 (enveda-np-examples, timsTOF natural products) | S3 | S3i |
|---|---|---|---|
| FP channel alone (`fp_only_mk - fp_only`) | **+0.073 [+0.043, +0.103]** (0.486 -> 0.559) | -0.005 (n.s.) | **-0.030 [-0.050, -0.012]** |
| Leak-controlled ranker (`pv_mk_cv - pv_cv`, S2) | **+0.057 [+0.030, +0.083]** | | -0.010 [-0.021, 0.000] |
| Full pv ranker (`pv_mk - pv`) | S1 +0.004 (n.s.), S2 **+0.026 [+0.004, +0.047]** | -0.001 | -0.008 (n.s.) |
| Clean (`clean_mk - clean`) | S2 **+0.030 [+0.005, +0.055]** | -0.003 | -0.005 |

## Reading

- **FACT:** on the population closest to the hidden test (timsTOF natural products, S1/S2), the ho1 networks beat the
  public networks clearly. The FP channel alone gains about +0.07 MRR, and the gain survives inside the ranker.
- **INFERENCE:** the S3i loss is most likely memorisation by the public networks, not a weakness of ho1:
  - ho1 held out the S3 molecules;
  - the public networks were very likely trained on them (`train_pkg_report.md` §4, item 6: the public nets score
    0.79/0.84 on our validation spectra, against 0.45-0.51 reported by their author).
  - Not proven.
- **INFERENCE:** the rankers were fitted on public-net features. Refitting them on ho1 features may add more.
- **Decision implication:** our own retrained FPNets are at least as good as the public ones and are prize-eligible
  by construction. Next step: a full-data retrain (no proxy hold-out) for submission use, and a refit of the
  ranker on our own features.

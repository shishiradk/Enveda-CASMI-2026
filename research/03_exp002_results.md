# 03b — EXP-002 Results: Adduct-Aware Candidate Generation

Status: **COMPLETE.** Design in `research/03_exp002_design.md`. Full run artifact:
`results/exp002_checkpoint.json`. Reused unchanged from EXP-001:
`results/exp001_query_sets.json` (Mode B queries) and `results/exp001_checkpoint.json`
(Variant A's scoring results).

**Headline finding, stated up front per the required interpretation rule: candidate
generation was fixed completely (recall 97%→100%), but final ranking got net worse
(MRR@25 −0.033, Recall@1 −4.5pp). Candidate generation was not the whole bottleneck —
fixing it exposed that the ranking function does not scale gracefully to a larger,
chemistry-aware candidate pool. This is not declared a success.**

---

## 1. Configuration (unchanged from design)

- Same 400 Mode B queries, same leakage exclusions, as EXP-001 — zero re-sampling.
- Variant A (baseline) scoring results reused directly from EXP-001's
  `B|C_all_train|ModifiedCosine` condition — not recomputed, guaranteeing perfect
  consistency with the original result. Variant A's candidate *rid sets* were
  recomputed fresh (cheap, integers only) purely for the set-membership comparison
  against Variant B; a consistency check against EXP-001's recorded `n_candidates`
  found **0 mismatches** across all 400 queries.
- Variant B (adduct-aware): `neutral_mass = (precursor_mz - delta) / n` per
  `research/03_exp002_design.md` §5's corrected 13-adduct table, tolerance 0.01 Da.
- Scorer: identical `matchms.similarity.ModifiedCosineGreedy(tolerance=0.1, mz_power=0.0,
  intensity_power=1.0)` for both variants. The ranker was not touched.
- Explosion safety cap (50,000 candidates/query): **never triggered** (0 flags).

## 2. Candidate-count composition

| Adduct class | n queries | mean A | mean B | median A | median B | max B |
|---|---|---|---|---|---|---|
| Monomer | 314 | 1076.2 | 1464.9 | 880 | 1193 | 5,020 |
| Multimer | 86 | 251.0 | 1660.6 | 216.5 | 1270 | 4,763 |
| Unsupported | 0 | — | — | — | — | — |

Overall candidate-count percentiles (50/75/90/95/99): A = [690, 1340, 2013, 2355, 2889];
B = [1224, 2252, 3215, 3722, 4408]. No unsupported-adduct queries occurred in this
sample (all 400 Mode B queries had a modeled adduct). Growth is substantial (roughly
1.4–6.6x depending on percentile) but bounded — never approached the 50,000 explosion
cap, consistent with the pre-run smoke test in the design doc.

## 3. An important correction to the design's assumption: B is *not* a superset of A

The design assumed Variant B's candidate set would be a superset of Variant A's for
same-adduct pairs. In the actual run, **398 of 400 queries** had Variant-A candidates
that Variant B correctly excluded. Manual inspection (rid 773739, query adduct
`[M+H]+`, precursor_mz 311.1622) explains this cleanly: Variant A's raw-precursor rule
also admitted 227 `[M-H]-` candidates whose raw precursor_mz happened to sit within
0.01 Da of the query's — but a real `[M-H]-`/`[M+H]+` pair of the *same* molecule would
differ in raw precursor_mz by ~2.01 Da (two proton masses), not <0.01 Da. These are
almost certainly **unrelated molecules at unrelated adducts, coincidentally close in
raw precursor value** — false positives Variant A's adduct-naive rule let in, that
Variant B's chemistry-aware comparison correctly rejects. **Variant B is not just more
recall-generous than A; it is also more precise.** This is a genuine, explicable
finding, not a bug, and it means part of Variant B's candidate-pool "growth" numbers in
§2 understate the underlying churn — the pool is being substantially reshuffled, not
purely expanded.

## 4. Classification (candidate-set membership, before ranking)

| Category | n |
|---|---|
| 1. Already a candidate under A | 388 |
| 2. Newly admitted under B | 12 |
| 3. Remains absent under B | 0 |
| 4. (sub-analysis of category 1, below) | — |
| 5. Unexpected (A-hit, B-miss) | 0 |

All 12 of EXP-001's candidate-generation failures (11 "molecule absent from library" +
1 "zero candidates") were recovered — **candidate-generation recall reached a perfect
100%** (up from 97.0%).

## 5. Final metrics: Variant A vs Variant B

| Metric | Variant A | Variant B | Delta (B−A) |
|---|---|---|---|
| candidate-gen recall | 0.9700 | 1.0000 | **+0.0300** |
| Recall@1 | 0.5350 | 0.4900 | **−0.0450** |
| Recall@5 | 0.8050 | 0.7800 | −0.0250 |
| Recall@10 | 0.8725 | 0.8550 | −0.0175 |
| Recall@25 | 0.9150 | 0.9200 | +0.0050 |
| MRR@25 | 0.6510 | 0.6178 | **−0.0332** |

Runtime: 467.9s (~7.8 min) for the full scoring stage (8 batches of 50 queries).
Peak RSS: 1,395 MB (batch 2), always releasing back to ~800–885 MB between batches —
memory stayed bounded throughout, no risk of the EXP-001 memory-pressure incident
recurring.

## 6. Category-1 decomposition: candidate present under both, does ranking change?

For the 388 queries where the correct candidate was already present under A, ranks
were compared query-by-query:

| Outcome | n |
|---|---|
| Same rank | 270 |
| Improved (rank_B < rank_A) | 25 |
| **Worsened (rank_B > rank_A)** | **84** |
| Newly failed (was in top-25 under A, fell out entirely under B) | 8 |
| Newly succeeded (was ranked >25 under A, now in top-25 under B) | 1 |

Of queries that were **rank 1 under A**: 184 stayed rank 1, but **30 lost rank-1
status** under B. Median candidate count for this group grew from 702 (A) to 1,236 (B)
— a mean growth of **2.74x**. The larger, more heterogeneous candidate pool is
correlated with rank degradation even for queries whose correct candidate was already
present and previously well-ranked.

## 7. The 12 previously-absent queries: outcome under B

| rid | true molecule | rank under B | shift branch changed score? |
|---|---|---|---|
| 395356 (zero-candidate case) | JFHPQARHIIJGOD | **1** | n/a (only candidate found) |
| 689395 | PJAAZAXBSLFNNK | **1** | Yes |
| 431909 | JZQMDNUYGZOPNW | **1** | Yes |
| 6296 | ACBPYGWTRFBYMP | **1** | No (matches added, score unchanged) |
| 1029257 | XGQSLHOHBLGVCV | 3 | Yes |
| 8769 | ADGUJIPJRXOWSH | 5 | Yes |
| 156672 | DFSLKFZMYMXXGX | 6 | Yes |
| 27493 | ANGIDRSDQGRWQD | 12 | Yes |
| 626670 | OBFWTNNDCLKJFP | 15 | Yes |
| 733288 | QHALUOAFNBWZED | **None (still fails)** | Yes |
| 410447 | JNVFBVQKGYTOAH | **None (still fails)** | Yes |
| 318845 | HPQRSZWHZSXIAR | **None (still fails)** | No |

**8 of 12 reach the top 25** (4 at rank 1); **3 of 12 are recovered as candidates but
still fail to rank in the top 25** — for these, candidate generation is no longer the
bottleneck, ranking now is, exactly the scenario the interpretation rule anticipated.

## 8. Modified Cosine analysis: did the shift branch actually help?

For the 12 previously-absent queries' newly-admitted true-molecule candidates, direct
Cosine and Modified Cosine scores were compared per-candidate (not assumed):
**9 of 12 queries show the shift branch changing the score** (i.e. Modified Cosine's
shifted-peak matching found matches direct Cosine could not, and those matches were
weighted highly enough to move the score) — consistent with, and now measured
in-situ rather than on one illustrative example, the standalone diagnostic from
EXP-001. 2 of 12 (rid 318845, 6296) show matches were added but the score was
unaffected (low-intensity matched peaks, same pattern observed in EXP-001's original
diagnostic). This confirms Modified Cosine's shift-matching is doing real, measurable
work once legitimate cross-adduct candidates are actually in the pool — the scorer
was never the limiting factor; candidate generation was, for these specific cases.

## 9. Interpretation

**Do not declare success.** Candidate-generation recall reaching 100% is a genuine,
confirmed improvement, and Modified Cosine's shift-matching demonstrably contributes
to it. But the net effect on the metric that matters (MRR@25) is **negative**: the
gains from 8 newly-successful queries are outweighed by ranking degradation spread
across the 388 queries that were already succeeding, particularly the 84 that got
worse ranks and the 8 that fell out of the top 25 entirely despite being correctly
found before. Recall@25 barely moved (+0.005) because the gains and losses at that
specific cutoff happen to roughly cancel; but Recall@1, Recall@5, Recall@10, and
MRR@25 — which are sensitive to rank quality, not just top-25 presence — all
declined.

**The mechanism, evidenced, not assumed:** Variant B's candidate pools grew ~2.74x on
median for already-succeeding queries, and §3 established this growth is partly
"real" (legitimate cross-adduct matches) and partly a reshuffling of noise (removing
some raw-precursor coincidences, admitting others via the wider effective neutral-mass
net, especially for multimer adducts). Modified Cosine's raw, uncalibrated score is
not robust to this larger, more heterogeneous pool — more candidates means more
chances for a spuriously higher-scoring wrong molecule to outrank the correct one in
the simple max-per-molecule aggregation used throughout EXP-001/EXP-002.

## 10. What this tells us about the research question

> How much Class-1 retrieval recall is lost because candidate generation compares
> reported precursor m/z directly instead of accounting for adducts?

**Answer: 12/400 queries (3%) had their candidate-generation step fixed, and 8/400
(2%) net new molecules reached the top 25 as a direct result** — a real but modest
gain, achieved at the cost of degrading rank quality for a much larger population of
already-correct queries. Adduct-aware candidate generation, used with the *current*
raw max-score ranking, is **not a net-positive change to deploy as-is**. The
bottleneck this experiment actually isolates is: **ranking does not degrade
gracefully as candidate pool size and heterogeneity grow** — a finding the
candidate-generation question surfaced, but did not itself answer how to fix.

## 11. What remains unknown / recommended next step

1. Whether a **precision-focused refinement of Variant B** (e.g. §4's Variant C
   pair-curation from the design doc, or a tighter neutral-mass tolerance) would
   recover most of the 8 net-positive top-25 gains while shrinking the noise growth
   that hurt category-1 queries — untested.
2. Whether the degradation is really about **pool size** or about **pool
   composition** (e.g. do the specific new candidates admitted for multimer queries,
   which saw the largest growth, disproportionately explain the 84 worsened ranks?) —
   not yet decomposed by adduct class.
3. Whether a **ranking-side fix** (e.g. a secondary tie-break, a calibrated score, or
   simply capping/truncating the candidate pool before scoring by a cheap
   pre-filter) would let candidate-generation gains through without the ranking cost
   — this reframes the next experiment from "better candidate generation" to "make
   ranking robust to a larger candidate pool," which is a **ranking** question and is
   explicitly out of scope until reviewed and approved separately (per the
   instruction not to touch the ranker in this experiment).

**No further action taken. Awaiting review.**

SSR design: scaffold-and-substituent recombination generator for Class-3 (`research/c3gen/ssr.py`, not implemented)

Bottom line: (INFERENCE) Built well, this angle gets about 10% recall@200 and about 0.03 MRR@25 on true Class-3, and adds about +0.003 to +0.01 on the LB. That is at or below the ±0.006–0.02 noise floor. It cannot close the 0.07 gap to 5th place by itself. It is still worth building because it is cheap (about 1 day plus a 15-minute asset build) and it adds to E6 without touching it. Run the cheap C3NP check (pass/fail rule at the end) before any integration work.

**0. New measurements for this angle (FACT).** Truth-key analogs were removed and stereo was stripped.
- Scripts: `research/c3gen/ssr_probe_extract.py` and `ssr_probe_ceiling.py`. Output: `results/c3gen/ssr_probe_ceiling.txt`.
- The sibling MMP probe's results are in `results/c3gen/mmp_reach_S1.json` and `mmp_reach_S3.json`.

| Measure (top-20 analogs unless noted) | S1 (n=250) | S3 (n=300) |
|---|---|---|
| Some analog shares the truth's exact Murcko scaffold, top-1 / top-5 / top-20 | 0.16 / 0.35 / 0.48 | 0.37 / 0.48 / 0.53 |
| Same, truths with ≥1 ring only (S3 has 18% acyclic truths, which match trivially) | ≈0.46 | ≈0.43 |
| Same, truths with ≥3 rings | 0.43 | 0.29 |
| Same scaffold and the truth is ≤1 substituent swap from that analog (multiset distance ≤2) | 81/250 = 32% | 88/300 = 29% |
| Analog has the same scaffold and the same formula (pure positional isomer) | 14% | 15% |
| Truth's substituents all appear among those same-scaffold analogs | 19% | 8% |
| MMP probe: one single-cut edit away from an analog | 40% | 42% |
| MMP probe: same, with the changed part ≤4 heavy atoms | 30% | 34% |
| MMP probe: index of the first analog that hits (S1 hits: 0 / 1–4 / 5–19 = 42 / 35 / 24 of 101) | — | — |

- The most common MMP edits are ±CH3, ±hexose, ±OH, OH↔OMe, ±deoxyhexose, ±prenyl and acetyl.
- Because the analogs alone rarely carry the truth's substituents (third-to-last row), the substituent vocabulary has to come from train ∪ COCONUT statistics, not from the analogs.

**1. Algorithm** (`generate(ctx, k=200, forbidden)`, deterministic, sorted iteration, no random numbers)
1. **Templates.**
   - Take the top 10 `ctx['analogs']` by spectral similarity and the top 10 `ctx['window']` by fpscore. The window structures have Δ≈0, so they seed isomer and swap edits.
   - Drop any whose ik14 is in `forbidden`. This filters database inputs only; outputs are never filtered.
   - Strip stereo and neutralise.
2. **Core and decorations.**
   - Use a sugar-aware Murcko core: cut O- and C-glycosidic pyranose/furanose units off first and treat them as substituents. Plain Murcko keeps sugars in the scaffold, which hides the frequent ±hexose and sugar-swap edits.
   - Decorations D_t are pairs of (core atom, substituent SMILES with a dummy atom). Every H-bearing core atom is a potential site; sites are de-duplicated by symmetry using canonical ranks with breakTies=False.
3. **Site evidence from MS2, after ModiFinder** (Shahneh et al., JASMS 2024). The idea is reimplemented from the paper; no code is copied.
   - Δ = target − M(template).
   - Enumerate template fragments with ≤2 bond cuts (a ring opening costs 2 cuts), as atom sets plus mass.
   - Use the top 30 merged query peaks, at ±10 ppm or 3 mDa, plus the adduct ion mass:
     - a fragment matched unshifted is evidence that its atoms are not the site;
     - a fragment matched shifted by +Δ is evidence that the site is in it.
   - Per-atom site score: s_a = log((1 + Σ shifted intensity) / (1 + Σ unshifted intensity)).
   - Template support = fraction of intensity explained (shifted plus unshifted).
4. **Edits.** Each edit is built with RWMol, sanitised, and kept only if RDKit ExactMolWt is within 10 ppm of the target.
   - E0, positional isomer (Δ≈0): move one decoration to another symmetry-unique site of the same element and aromaticity class.
   - E1, single swap, add or remove at one site: sub_old (H or an existing decoration) → sub_new, where |M_t + Δsub − target| ≤ 10 ppm. Candidates come from the Δ-indexed transform table (A2), limited to the top 30 by P(sub | environment, class).
   - E2, two small edits: from {±CH2, ±O, ±H2, +C2H2O, ±hexose, ±deoxyhexose, ±C5H8}, only at the 4 best sites by s_a, and only when E0 and E1 give fewer than k/2 candidates for that template.
   - E3, cross-recombination: when two templates share a core, put B's decorations onto A, then check the mass.
5. **De-duplication.**
   - Cheap pass first: canonical SMILES, then standard InChIKey14.
   - Then the tautomer-canonical key (`prep_data.canon_key`) on the final ≤250 only.
6. **Score.**
   - S = z(fp·zlog) + a·max s_a over edited atoms + b·log P(sub_new | env, class) + c·template spectral sim + d·template support.
   - The fingerprint is `prep_data.fp_and_mass` with `fp_bits`.
   - Defaults: a=0.5, b=0.3, c=0.5, d=0.3. Fit them with a coarse grid on half of C3NP and report on the other half.
7. **Output.** Return the top k as (smiles, S, provenance), e.g. `ssr:E1:tpl=<ik14>:site=<i>:[*]O>[*]OC`.

**Runtime per molecule (INFERENCE).**
- ≤4,000 RWMol builds at about 0.5 ms each: 2 s.
- ≤2,000 fingerprints at about 1 ms each: 2 s.
- Fragment trees for 20 templates: about 4 s.
- 250 tautomer keys at about 20 ms each: 5 s.
- Total about 13 s, under the 20 s budget.
- RAM under 300 MB, with A1 loaded as top-50 substituents per environment.

**2. Data assets** (built to `results/c3gen/ssr_assets/`, under the resource rules)
- **A1 `sub_stats.parquet`.**
  - Source: `results/train_pkg/data/pool_smiles.txt` plus `pool_key.npy` (711,626 structures from COCONUT and train), read line by line in 20k-line chunks with 3 workers.
  - For each structure, record each (generic-core class hash, atom-environment code, substituent ≤12 heavy atoms). The environment code is element, aromaticity, ring size and neighbour heteroatoms.
  - Exclude the union of all bench truth keys (S12, S3, and C3NP `forbidden.parquet` once it exists) at build time. This is the leak guard: otherwise the truth's own counts would boost its exact substituent.
  - Aggregate in DuckDB (memory_limit 2GB) into P(sub | env, class) with back-off to P(sub | env).
  - Cost: about 3 ms per molecule, so about 12–15 min on 3 workers, under 400 MB per worker; the output is about 50 MB.
- **A2 `transforms.parquet`.** For the top 300 global substituents (H included), every pairwise Δmass, sorted for window lookup. About 90k rows, seconds to build.
- **A3.** Import `fp_bits.npy`, `prep_data.fp_and_mass` and `canon_key` unchanged.
- **A4.** A new fragment enumerator that keeps atom sets; `pv.frag_masses_safe` returns masses only.

**3. Expected performance (INFERENCE, with reasons)**
- **Bench reach.**
  - About 30% of bench truths are one small edit from a top-20 analog (both probes agree).
  - About 85% of those have the right template among the top 10.
  - About 0.7 survive the k=200 cut: each template gives roughly 20–60 Δ-matched candidates, and 10–20 templates compete for 200 slots.
  - This gives recall@200 of about 0.18 on S1/S3-like data.
- **True Class-3.**
  - Class-3 structures are absent from PubChem, which holds most known NP analogue series, so their library neighbourhoods are sparser.
  - Udam (rank 6) found that delete-truth simulations "overstated reach about 3×".
  - Discounting by 0.4–0.6 gives recall@200 of about 0.10 (range 0.05–0.20).
- **Ranking within the list.**
  - The decoys are positional and substituent isomers on the same core; same-formula isomers are already the main error source (intel §0.6).
  - P(rank 1 | in list) is about 0.15–0.25, so MRR given in-list is about 0.3.
  - Class-3 MRR@25 alone is therefore about 0.03 (range 0.015–0.06); on the bench it is about 0.05.
- **Published reference points.**

| Source | Result | Relevance here |
|---|---|---|
| MassSpecGym de novo (Bushuiev 2024) | 0% top-1 for all baselines | What blind generation achieves |
| DiffMS (Bohde 2025) | 2.30% top-1 | Same |
| MADGEN (ICLR 2025) | 1.31% top-1 with a predicted scaffold; **10.5% top-1 with an oracle scaffold** on MassSpecGym (49% on NIST23) | Closest analogue: our analogs give a noisy oracle scaffold for about 45–50% of bench molecules, so 0.45 × 0.105 ≈ 5% top-1 on the bench, consistent with the estimate above |
| MSNovelist (Stravs 2022) | 25% top-1, 45% retrieved on GNPS, given the correct formula | The structures were database-known |
| MIST + MolForge (arXiv 2508.04180) | 28% top-1 | Decoder trained on 2M structures including COCONUT and HMDB. An audit (arXiv 2606.19624) found leakage or evaluation problems in 17 of 26 MassSpecGym papers. Not transferable to Class-3 |

- I found no verified numbers for BioTransformer- or MetFrag-style expansion, so none are claimed.
- **Competition evidence (FACT, intel §2.2).**
  - prvsiyan's blind ±O/±CH2/±hexose enumeration: 0.337 → 0.335.
  - francisco's BRICS generation: 0.046 → 0.050.
  - Udam: generation "has not yet paid off", even when gated.
  - SSR differs from those through MS2 site localisation, scaffold-class substituent priors, and append-only merging. This has not been tested.

**4. Eligibility of every input**
- train.parquet structures and our own nets: allowed per DEC-008.
- COCONUT: CC0 / CC BY 4.0, with attribution.
- RDKit, including rdMMPA: BSD-3.
- ModiFinder: the idea only, reimplemented from the paper; its code licence is unverified, so none of its code is used.
- No PubChem, NIST, NC data, NC weights, FRIGID or Ahmed assets.
- The bench pool cache is used for evaluation only.

**5. Main failure modes**
1. **No analog shares the scaffold** (about 52–55% of bench molecules, more on Class-3). The output is wrong-core noise. It must carry low S so the merge cap leaves it out (require template support above a threshold).
2. **Positional ambiguity.** Glycosides have 5–7 OH sites and flavonoids/coumarins have many aromatic C–H sites. fp·zlog has weak position sensitivity, so the truth is diluted among near-ties.
3. **One Δ, many chemistries.** +O can be C-hydroxylation, an N-oxide or an epoxide. ±CH2 can be on C, N or O. This inflates candidate counts.
4. **Bench optimism.** COCONUT truths have dense neighbour series. A wrong target from adduct misassignment makes every candidate mass-wrong.
5. **Leakage paths.** Substituent statistics that include the truth, and stereo or tautomer twins left in the analog list. Guards: the forbidden union is excluded from A1, and key-based removal stays in the CLI.

**6. Combining with the pipeline (append-only, E6 rule, DEC-008/009)**
- **Final list:** engine top-1, then a 3-way round robin of engine[1:], the E6 PubChem channel and SSR, de-duplicated by key, capped at 8 SSR entries in the top 25.
- **Variants to sweep:** SSR enters only after engine position k_e ∈ {3, 5, 10}, or in the round robin.
- **Acceptance:** SV and S1 each drop ≤0.003, and C3NP gains ≥0.02.
- **Optional promotion:** SSR's top-1 to slot 2 only on the generator's own evidence (template similarity ≥0.8 and support ≥0.5), never on a library-similarity gate (E5 lesson), and validated on SV's re-measured hits first.
- **LB arithmetic:**
  - In the round robin, an SSR hit at its own rank r lands at about position 1 + 3r, so its reciprocal rank shrinks about 3–4×.
  - Class-3 is about 0.5 of the test, so the gain is 0.5 × 0.03 / 3.5 ≈ +0.004, or about +0.01 with promotion.
- **Optional later step:** re-score SSR's top 50 with ICEBERG/GLACIER (MIT, MassSpecGym checkpoints) within same-formula groups, if CPU time allows.

**Pass/fail rule:** the first check is A1 plus the CLI on C3NP, S1 and S3, n=30 each, at about 1 hour of compute. If recall@200 < 0.08 or hit@1 < 0.02 on C3NP, drop this angle and spend the effort on Class-1/2 isomer ranking with fragmentation features plus a GBDT ranker. That is the only approach above 0.40 that a team has disclosed (Udam, 0.414).

Files:
- `D:\Enveda-CASMI-2026\research\c3gen\ssr_probe_extract.py`
- `D:\Enveda-CASMI-2026\research\c3gen\ssr_probe_ceiling.py`
- `D:\Enveda-CASMI-2026\results\c3gen\ssr_probe_ceiling.txt`
- `D:\Enveda-CASMI-2026\results\c3gen\ssr_probe_ana.pkl`
- `D:\Enveda-CASMI-2026\results\c3gen\mmp_reach_S1.json`
- `D:\Enveda-CASMI-2026\results\c3gen\mmp_reach_S3.json`

Sources: [MADGEN](https://arxiv.org/pdf/2501.01950) · [ModiFinder](https://www.biorxiv.org/content/10.1101/2024.02.17.580849.full.pdf) · [MSNovelist](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9262714/) · [MIST+MolForge](https://arxiv.org/html/2508.04180v3) · [MassSpecGym audit](https://arxiv.org/pdf/2606.19624)
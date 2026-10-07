DESIGN: `mmpedit`, a Class-3 generator that edits library analogs using mined matched-molecular-pair (MMP) rules
(research/c3gen/mmpedit.py. Design only; nothing implemented. One measurement probe was run, see section 0.)

0. MEASURED REACHABILITY (FACT; probe research/c3gen/mmp_reach_probe.py; outputs results/c3gen/mmp_reach_S1.json and results/c3gen/mmp_reach_S3.json)
- Setup: ho1 records, the top-20 spectral analogs per molecule (`rec['ana']`), and the truth removed by key.
- A molecule counts as a hit when the truth is exactly one single-cut MMP edit from at least one of those analogs. The edit may also be an H-substitution (one H replaced by a group). The shared constant part must cover at least 60% of the truth's heavy atoms, and each variable part must have at most 10 heavy atoms.

| Bench | Single-cut hit | Variable part ≤ 4 heavy atoms | Any edit, incl. 2-cut on the top-10 analogs | First hit at analog rank 0 / 1-4 / 5-19 |
|---|---|---|---|---|
| S1 (250 enveda-np, timsTOF) | 101 (40%) | 74 | 109 (44%) | 42 / 35 / 24 |
| S3 (300 GNPS NPs) | 127 (42%) | 101 | 147 (49%) | 77 / 28 / 22 |

- Same-mass isomers among the top-20 analogs: S1 66, S3 60.
- Median Morgan2 Tanimoto between the truth and its best analog: about 0.57.
- Most frequent edits on S1:
  - CH3 <-> H (methylation / demethylation)
  - hexosyl <-> H, and O-hexosyl / O-deoxyhexosyl <-> OH (glycosylation)
  - H <-> OH (hydroxylation)
  - OMe <-> OH
  - prenyl <-> H
  - O-acetyl <-> OH
  - NMe <-> NMe2
- On S3 the leading edits are alkyl homologation (lipid-like chains).
- INFERENCE: these S1/S3 truths are mostly Class 2. The ceiling is valid for Class 3 only if novel NPs relate to library NPs the way library NPs relate to each other. The host's description of Class 3 ("NP analogs, hypothesised NPs") supports this, but it is not proven.

1. ALGORITHM (per molecule; deterministic; inputs are ctx only)
1. Seeds:
   - Take ctx['analogs'][:20] (best first) and drop any analog whose key is in `forbidden`.
   - Optional, phase 2: the top 5 entries of ctx['window'] as seeds for edits that keep the mass the same.
2. Fragment each seed:
   - Single cut: `rdMMPA.FragmentMol(maxCuts=1)`, plus H-cuts (a dummy atom on every heavy atom that carries an H).
   - Each result is a triple (constant, var_from, env), where env is a hash of the radius-1 atom environment at the attachment point (mmpdb-style).
   - Seeds with more than 70 heavy atoms get H-cuts only.
3. Target mass difference:
   - Δ = target − ExactMolWt(seed).
   - Tolerance = 10 ppm × target, plus 0.0005 Da.
4. One-step edits:
   - Look up rules keyed by (var_from, env) and sorted by dmass; keep those with |dmass − Δ| ≤ tol.
   - If there are none, back off to rules keyed by var_from alone.
   - Build the product with `Chem.molzip(constant, var_to)`.
5. Two-step edits, only if step 4 yields fewer than 50 products:
   - Use pairs from the 300 most frequent small rules (≤ 4 heavy atoms each, plus all sugar and acyl rules) with d1 + d2 ≈ Δ. Examples: OH + Me = +30.011, Hex + Me = +176.068.
   - Apply them in sequence on different cut sites.
   - Same-mass case (|Δ| < tol): "move" edits that take a substituent off at one site and put it on another (mined rules var→H and H→var, same var).
6. Validate each product:
   - Sanitize, strip stereo, and check |ExactMolWt − target| ≤ 10 ppm.
   - Dedupe by InChIKey14 first; compute the tautomer-canonical key only on the final top k.
   - Cap the pool at 3,000 products. If over, keep the highest prior: sim(seed) × log1p(rule support).
7. Score each product:
   - S = z(fp·zlog) + λ1·log1p(support_env) + λ2·sim_seed − λ3·[two-step] − λ4·[env backoff].
   - fp·zlog uses the prep_data `fp_and_mass` fingerprint, with z taken within the molecule's candidate set.
   - Optional, phase 2: a fragment-explanation term on the top 300. Count the query peaks explained (via `frag_masses`) by fragments that contain the edit site but not by the seed's own fragments. This is the ModiFinder idea (Shahneh et al., 2024) of using shifted peaks to localize the modification site. It is the only term that separates positional isomers that zlog cannot tell apart.
8. Return the top k = 200 best-first.
   - Provenance string: `mmp:s{seed_rank}:{var_from}>>{var_to}:env{0|1}:n{support}:d{1|2}`.
   - Fit λ1-λ4 on a grid over S1 even mids and check them on S3 / C3NP (C3NP has a forbidden list).

2. DATA ASSETS (built once, offline, within the resource rules)
A. Fragment table:
   - Input: pool SMILES from results/train_pkg/data/pool_smiles.txt (about 711k COCONUT + train structures).
   - 3 workers, each streaming batches of 20k SMILES through steps 1.2 and 1.3. Each batch is written as parquet rows (const_hash u64, var_id u32, env u32, src u8, mol_idx u32).
   - The var vocabulary (≤ 10 heavy atoms, canonical) is interned in a per-worker dict and merged at the end.
   - Estimate (INFERENCE from the probe's timing): about 8 ms per molecule, so about 1.6 CPU-hours, or about 35 min on 3 workers. RAM is under 0.6 GB per worker. About 25M rows, about 0.4 GB on disk.
B. Rule table, built in DuckDB with memory_limit 2GB:
   - Self-join the fragment table on const_hash. Count (var_i, var_j, env) pairs by the number of distinct constants.
   - Cap each constant group at 64 members (deterministic choice by hash) to stop the explosion from generic constants.
   - Keep rules with support ≥ 2 that were seen in ≥ 2 distinct constants.
   - Store dmass, support_all and support_coco. COCONUT support is weighted higher, because train includes enveda-180 synthetic chemistry.
   - Expected size is 0.3-1M rules (results/c3gen/mmp_rules.parquet, under 100 MB), loaded as a dict of sorted numpy arrays.
   - Estimate: 10-20 min, at most 2 GB.
C. Leak control (needed for honest bench numbers):
   - The evaluation rule DB must be built with every bench truth key excluded from the fragment input: the S1, S3 and C3NP forbidden keys and held_S4.
   - Otherwise the pair (truth, analog) is itself a rule with support ≥ 1.
   - The submission DB may use everything.
D. 2-cut (linker-swap) rules are phase 2; they would roughly triple the cost of asset A.
- Runtime per molecule (INFERENCE): fragmenting the seeds takes about 0.3 s, products 1-5 s, fingerprints for 3k products 3-6 s, and fragment explanation on 300 products about 4 s. Total ≤ 15 s; well under the 20 s budget.

3. EXPECTED PERFORMANCE ON NP CLASS 3 (INFERENCE built on the FACT numbers above)
- Recall@200 is a product of factors:
  - Reachability: 0.40-0.45 at one step and about 0.50 with two steps and 2-cut edits.
  - Rule coverage with matching Δ: about 0.85. The dominant edits are high-support sugar, methyl, hydroxyl and acyl rules.
  - Survival into the top 200 after the 3k cap and scoring: about 0.5-0.7. For Δ = CH2 on a seed with 30 C-H/N-H/O-H sites, 20 seeds give about 600 positional variants.
  - Result: recall@200 ≈ 0.18-0.30 (point estimate 0.22).
- MRR@25 within Class 3:
  - The competing candidates are near-twins that differ by a few fingerprint bits, so zlog separates sites poorly.
  - For calibration, MassSpecGym retrieval with mass-matched PubChem candidate lists (≤ 256 candidates) reports best-model hit@1 of roughly 15-30% (MIST about 15%, from memory; approximate).
  - Positional isomers of one seed are harder than PubChem decoys. Estimate truth in the top 25 for 30-45% of reachable cases and MRR given reachable 0.10-0.20.
  - Result: Class-3 MRR@25 ≈ 0.04-0.09.
- Comparison with de novo generation (published numbers, quoted from memory, approximate):
  - MassSpecGym de novo baselines: about 0% top-1.
  - DiffMS (2025): about 2% top-1 and about 4% top-10 on MassSpecGym; about 8% top-1 on NPLIB1.
  - MSNovelist (Stravs 2022): correct structure among outputs in about 45% of GNPS cases, but at about 25% rank-1, and that is when the formula is known.
- Library-relative expansion has a precedent: MINE (Jeffryes 2015), BioTransformer 3.0 (Wishart 2022), and NAP / molecular-network propagation. These report that most annotation gains come from one-step biotransformations of a known neighbour, which matches the probe above.
- The public fork's `EngineCfg(generate=True)` ("derive from a close spectral analogue") is the same idea. Its v1 stack scored 0.354, so it gives no evidence of a large standalone gain.
- Leaderboard estimate: 0.5 Class-3 share × 0.04-0.09 × a placement factor of 0.4-0.7 ≈ **+0.01 to +0.03**. That does not close the 0.072 gap alone; it is one of the stacked Class-3 generators.

4. ELIGIBILITY (all inputs)
- RDKit (rdMMPA, molzip, fingerprints): BSD-3.
- COCONUT: CC0 / CC BY.
- train.parquet structures: competition data.
- zlog: our own leak-free ho1 / full FPNet nets.
- The mmpdb and Hussain-Rea (2010) algorithms are reimplemented on rdMMPA from the papers; no code is copied.
- Hand-written mass-delta sanity list (hexose 162.0528, which is distinct from caffeoyl 162.0317 at 10 ppm, and similar): an own list.
- Nothing here is NC or NIST. No Ahmed or FRIGID code or weights; reading the fork's docstring only.

5. MAIN FAILURE MODES
1. The seed has the wrong scaffold: in 50-60% of molecules no top-20 analog is within one edit. The generator returns nothing useful, and the merge is a no-op.
2. Positional-isomer explosion with weak site discrimination: zlog alone puts the truth at ranks 5-50. This needs the fragment-explanation term, which is unproven.
3. Edits outside MMP grammar: C=C reduction (+2.016), epoxidation, ring closure or opening, and core oxidation. Fix: about 10 mined atom-level SMIRKS edits as a third family.
4. Multi-site NP decoration (≥ 3 edits) is unreachable.
5. Rule bias toward enveda-180 synthetic chemistry through train. Fix: weight support_coco.
6. Benchmark leakage if C is skipped, which would inflate recall by an unknown amount.
7. Tautomer and stereo duplicates using up slots. Fix: final dedupe on the tautomer-canonical key.

6. INTEGRATION WITH THE CURRENT PIPELINE (E6 style; DEC-008 design rule)
- Merge order:
  1. Engine top-1 is never displaced.
  2. Engine[1:] and the E6 PubChem channel alternate as now.
  3. `mmpedit` candidates are appended only below engine top-k (k = 10 to start), interleaved with the remaining channel slots, after dropping keys already in the list.
- No gating on library similarity (E5 and the E6 gating test showed it hurts).
- Acceptance before any submission:
  - SV must stay ≥ 0.92 and S1 must not fall by more than 0.003 (DEC-008).
  - C3NP MRR gain must be > 0, with the leak-free rule DB.
  - Kaggle runtime: 400 molecules × ≤ 15 s ≈ 1.7 h on one core, or about 35 min on 3 cores, which fits beside the 2,452 s E6 channel.
- Order of experiments:
  1. Build assets A, B and C (about 1 h).
  2. CLI smoke test: `--bench S1 --n 30`, reporting hit@1, hit@25 and hit@200.
  3. Full S1 and S3, then C3NP.
  4. Fit λ, and decide whether the fragment-explanation term earns its runtime.

Files written: /d/Enveda-CASMI-2026/research/c3gen/mmp_reach_probe.py, /d/Enveda-CASMI-2026/results/c3gen/mmp_reach_S1.json, /d/Enveda-CASMI-2026/results/c3gen/mmp_reach_S3.json (S3 file saved from console output; top_transforms omitted)
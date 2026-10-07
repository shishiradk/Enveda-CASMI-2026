# EXP-008 — Pseudo-C3 smoke: exact implementation

Date: 2026-09-24. Status: **PREPARED, NOT EXECUTED.** The census and smoke stages refuse to run without
`--authorized`. Only the preflight (construction checks, eligibility, tiers, seeded manifest) has been run; it does
no spectral scoring or ranking.
Script: `research/scripts/exp008_c3_smoke.py`. Design: `research/analysis/exp008_c3_design.md`.
No `src/` change, no research-state edit, no DEC-008, no EXP-009.

## 0. Binding EXP-007 caveat

The EXP-007 audit used exact precursor matching to 4 decimal places, while candidate generation used a ±0.01 Da
window. Some same-adduct spectra can therefore remain in the candidate pool uncounted as retained evidence.
The 171/72/247 decomposition and the 124/47 other-adduct/no-evidence split are **not exact**, and EXP-007's
magnitudes **are affected**. The C1-matched control and the leakage/replay/determinism checks are unaffected.
EXP-008 does not rely on any affected EXP-007 number: it uses only the EXP-007 rid list (the manifest's `all_600_sorted`).

## 1. Two separate sources

| | Structure universe U | Spectral evidence library |
|---|---|---|
| Content | one entry per train `inchikey14`; SMILES = `min(normalized_smiles)`; formula = `min(molecular_formula)` | train spectra (rows) |
| Target | **retained** | **all spectra of the target's parent group removed** |
| Built | once, cached as `results/exp008_structure_universe.parquet`, no per-target edits | filtered per query |
| Used for | the candidate pool (final search space = **structures**) | scoring only |

Parent group G(t) = {t} ∪ {m ∈ U : parent(m) = parent(t)}, where
parent(m) = InChIKey14 of `LargestFragmentChooser` → `Uncharger` applied to m. This covers the salt and charge forms.

## 2. Neutral mass and candidate pool (exact)

- Query neutral mass (Da): **M_q = precursor_mz − Δ(adduct)**, n = 1 (monomers only). Δ is taken verbatim from EXP-002 §5:
  [M+H]+ +1.007276 · [M-H]- −1.007276 · [M+Na]+ +22.989221 · [M+K]+ +38.963158 · [M+NH4]+ +18.033823 ·
  [M+Cl]- +34.969402 · [M+CH2O2-H]- +44.998203 · [M+C2H4O2-H]- +59.013853.
- Candidate mass (Da): RDKit `Descriptors.ExactMolWt` (monoisotopic) of the universe SMILES.
- Pool: **P(q) = { c ∈ U : |mass(c) − M_q| ≤ 5·10⁻⁶ · M_q }**. It is a sorted-array range query. A 10 ppm pool is recorded as sensitivity only.
- Only the query's own precursor and adduct enter M_q. Candidates are structures, so no candidate precursor or adduct is involved.

## 3. The three baselines (reported separately, never merged)

For target t, query spectrum q (the EXP-007 query rid), and query adduct a:

**Baseline A: chance.** Expected RR@25 under a uniform random order of P(q):
`RR_A = (1/|P|) Σ_{r=1}^{min(25,|P|)} 1/r` if t ∈ P, else 0.

**Baseline B: existing direct spectral ranker.** Each c ∈ P is scored only from its own spectra:
`D(c) = max_{s ∈ Spec(c), adduct(s)=a} ModCos(q, s)`, over all libraries with G(t) removed. Molecules with no spectra get −∞.
Expectation: t has no spectra, so D(t) = −∞ and t sits in the bottom tie group. The **L6 canary** is
"direct spectra counted for t = 0". RR_B can be slightly > 0 only through random tie-breaking inside that bottom group.

**Baseline C: analogue propagation (R1).**
1. Evidence set for q: `E(q) = { rep(m, a) : m ∉ G(t), |pm(rep) − pm_q| ≤ 150 Da }`, where rep(m, a) is one enveda-180 spectrum
   of molecule m at adduct a. It is chosen label-free: the merged-CE spectrum (`collision_energy_orig` contains ",") first,
   then max `num_peaks`, then lowest rid.
2. `s_m = ModifiedCosineGreedy(q, rep(m,a))` with tol 0.1, mz_power 0, intensity_power 1: the validated EXP-001 scorer.
3. `H` = the top K = 20 molecules by s_m (ties broken by `inchikey14` ascending).
4. For each candidate structure c ∈ P:
   **S(c) = max_{h ∈ H, parent(h) ≠ parent(c)} s_h · Tc(c, h)**, and S(c) = 0 if no admissible h.
   Tc is Tanimoto on Morgan r=2, 2048-bit fingerprints.
5. Aggregation is at molecule level by construction (one S per candidate structure).

**Own-spectrum prevention:** h is inadmissible for c when parent(h) = parent(c). This covers c itself and its
salt/charge forms. So **no candidate is ever scored by its own spectra**, and the ranking is on an equal footing between
spectrum-bearing candidates and the spectrum-less target. G(t) is removed from E(q), so the target's score comes
**entirely** from spectra of other molecules plus structural similarity to them.

All RRs are **tie-aware**: the expected RR over uniform tie-breaking within t's tie group. This keeps A, B and C
comparable and prevents any deterministic tie order from favouring t.

## 4. Decoy-query control

- **Generation:** d is drawn uniformly (seed `20260925·10⁷ + rid`) from the pool members c ∈ P(q) with c ∉ G(t),
  parent(c) ≠ parent(t), and an enveda-180 representative at the **same adduct a**. The query spectrum is replaced by rep(d, a).
- **Preserved:** the pool P(q) (the same candidate structures, so the same chance baseline), the adduct, the neutral mass (d lies in P(q)),
  the target's absence from evidence, the scorer, K, the window, and the Tc definition.
- **Changed:** only the spectrum, which now comes from a different real molecule of the same mass and adduct. G(d) is also removed
  from evidence (symmetric with t), so d cannot retrieve itself.
- **Why it is informative:** every pool-level property that could make t rank well without spectral information is
  held fixed. That includes t being structurally "central" to the spectrum-bearing neighbourhood, isomer-family bias, and pool size.
  If R1's lift on t comes from t's spectrum, it must disappear when t's spectrum is swapped for d's.
- **Reading:** real-query lift > 0 with decoy lift ≈ 0 is evidence that analogue propagation extracts target-specific spectral
  information. Decoy lift ≈ real lift means the "signal" is a pool artefact, not evidence. Decoy failure (t still ranked well)
  therefore invalidates a positive R1 result.
- **Known limit:** d can be structurally similar to t, in which case its spectrum legitimately carries t-like information.
  Tc(t, d) is recorded per query for stratified reading. d is not re-drawn on that basis (no outcome-driven selection).
- **Secondary (recorded, not primary):** the RR of d itself under the decoy query, which amounts to a second C3 test.

## 5. Population, tiers, sampling

- Source: the EXP-007 manifest `query_ids.all_600_sorted` (600 timsTOF rids).
- Eligibility: E1 monomer adduct in the §2 table · E2 truth SMILES parses · E3 |P(q)| ≥ 2 · E4 at least one decoy candidate.
  None of these uses truth-in-pool.
- Tier = NN-Tc(t) = max Tc(t, m) over molecules m ∉ G(t) that have an enveda-180 representative at the query's adduct.
  **T-low < 0.60 · T-mid [0.60, 0.70) · T-high ≥ 0.70**. The truth structure is used for stratification only.
- Sampling: `random.Random(20260924)`, tiers in the fixed order low→mid→high, 10 each without replacement.
  If a tier has fewer than 10 eligible queries, preflight **STOPS**, reports the counts and writes no manifest. There is no substitution.
- The manifest (`results/exp008_c3_smoke_manifest.json`) stores the rids, tiers, NN-Tc, decoys, all parameters, and the script sha256.
  The smoke stage refuses to run if the script has changed since.

## 6. Checks

| ID | Check | Where |
|---|---|---|
| rid | file_row_number == EXP-007 rid on 2 known triples | every stage |
| C | adduct round-trip error < 1e-9 Da for all 8 adducts; sanity rid ppm error | preflight |
| D | sanity target lies in its own 5 ppm pool | preflight |
| A | after G(t) ∪ G(d) removal, 0 evidence rows of t or d; parent-group size histogram | preflight (30) and smoke |
| B | target ∈ universe for all 30 | preflight |
| L1 | 0 target rows in evidence; 0 hits in G(t) | smoke |
| L6 | 0 direct spectra for t (canary) | smoke |
| L7 | 0 decoy-arm hits in G(t) ∪ G(d) | smoke |

## 7. Outputs

- `results/exp008_c3_preflight.json`: checks, eligibility, tier counts
- `results/exp008_c3_smoke_manifest.json`: written by preflight
- `results/exp008_c3_census.json`: 8a (authorized only)
- `results/exp008_c3_smoke_query_results.jsonl`, `results/exp008_c3_smoke.json`: 8b (authorized only)

## 8. Stopping criteria (smoke)

The smoke stops (reported, not interpreted) if any L1/L6/L7 check is non-zero, pool recall at 5 ppm is < 0.95, or the median
runtime is > 2 min/query. n=30 validates integrity only. The full run needs separate authorization.

## 9. Preflight result

Source: `results/exp008_c3_preflight.json`. Runtime 2,379 s, mostly the one-off universe build, now cached.

- **Checks, all pass:**
  - C: round-trip error 0.0 Da. Sanity rid 773739 [M+H]+: M_q 310.154924 vs ExactMolWt 310.154209, +2.30 ppm.
  - D: the sanity target is in its 5 ppm pool.
  - A: 0 violations. The parent-group size is 1 for all 30, so no salt or charge forms are present in this sample.
  - B: 0 violations.
- **Eligibility (600 queries):** E1 not monomer 175 · E2 0 · E3 pool<2 2 · E4 no decoy 16 → **407 eligible**.
- **Tiers (eligible):** T-low 141 · T-mid 128 · T-high 138. All tiers are ≥ 10, so there was no stop.
- **Manifest:** 30 queries (10/10/10), adducts [M+H]+ 24 / [M-H]- 6, median 5 ppm pool 55 structures.
  NN-Tc ranges: T-low 0.406–0.586, T-mid 0.603–0.698, T-high 0.719–0.929.
  Script sha256 `a49fa4a0a72a0109…` matches the file on disk. Any edit to the script requires re-running preflight.
- **Not yet measured:** pool recall (EXP-008a) and all ranking quantities.

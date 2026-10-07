# EXP-008 preflight — independent construction verification

Date: 2026-09-24. Scope: preflight verification only (authorized). **No census, no ranking, no spectral
scoring, no `src/` change, no DEC-008, no research-state edit, no EXP-009.** The existing preflight outputs
(`results/exp008_c3_preflight.json`, `results/exp008_c3_smoke_manifest.json`) were **not** overwritten. Every
number below was recomputed with separate code (rid via `row_number() OVER () − 1` as in EXP-007, not
`file_row_number`; an independently written representative table, pool, tier and sampling code; RDKit masses
from the 2026-09-24 census). Scratch scripts: `v008*.py` in the session scratchpad.

## Reproduction

| Item | Preflight | Independent | Match |
|---|---|---|---|
| Script sha256 vs manifest | a49fa4a0… | recomputed | yes |
| Population | 600 | 600 unique rids | yes |
| E1 not monomer | 175 | 175 = [2M+Na]+ 91 + [2M+H]+ 52 + [2M-H]- 32 | yes |
| E2 / E3 / E4 | 0 / 2 / 16 | 0 / 2 / 16 | yes |
| Eligible | 407 | 407 | yes |
| Tiers low/mid/high | 141/128/138 | 141/128/138 | yes |
| 30-query manifest (rids, ik, tier, NN-Tc, pool size, decoy, #decoy candidates) | — | identical, 0 field mismatches | yes |
| 5 ppm pool sizes of the 30 | — | identical using independent RDKit masses | yes |
| Adduct Δ vs atomic-mass theory | — | all within 0.003 mDa ([M+NH4]+ −0.0026 mDa ≈ 0.01 ppm) | yes |

## Checks the preflight script does not perform (added here)

- **Alias leakage:** 30 targets + 30 decoys. Each molecule's tautomer-canonical InChIKey14 (RDKit
  `TautomerEnumerator`, which is the competition's matching) and its salt/charge-parent + tautomer key were
  compared against every universe structure within ±3 mDa (median 86 each). **0 aliases.**
- **R1 reach of the tier neighbour:** for all 30, the NN-Tc neighbour lies within the ±150 Da R1 window, so the
  tiers describe evidence R1 can actually retrieve.
- **Universe mass stability:** 70/275,810 inchikey14 have SMILES-choice-dependent mass (min vs any SMILES,
  up to 5.03 Da). None is a target or decoy in the manifest.

## Findings

1. **Check A (target absent from evidence) is tautological.** It removes G(t) ∪ G(d) and then asserts that t and d are
   absent. It cannot fail, and says nothing about aliases. The alias check above is the substantive test.
2. **L6 canary is tautological and differs from the design.**
   - `B_n_direct_spectra_target` counts target rows in `ev_b`, and `ev_b` was already filtered to exclude G(t), so the value is always 0.
   - The design §5.2 states the criterion as "RR_B = 0", but tie-aware RR_B is > 0 by construction: t ties with the pool members that have no spectra.
   - A leak through an alias would not show up in either form, because the alias is a separate candidate key.
   - A working canary would be a pair:
     - positive control: B without exclusion ranks t with a finite score, consistent with EXP-007 C1;
     - negative control: with exclusion, no pool candidate whose tautomer or parent key equals t's has a finite score.
3. **Decoy reading rule.** Under d's spectrum, R1 promotes d and its analogues. The null for t's RR is therefore
   plausibly *below* chance, not equal to it. "Decoy lift CI ∋ 0" is the wrong null. The pre-registered contrast
   should be paired: RR(t | q_t) − RR(t | q_d). Also, the decoy query is a representative spectrum (merged-CE first)
   while the real query is an arbitrary EXP-007 rid. That asymmetry needs recording.
4. Population is the EXP-007 C2-stratified sample, not a random timsTOF sample. Report it as such.

# S4: timsTOF query sets for the Class-3 and PubChem-only buckets

Date: 2026-10-04. Builder: `research/scripts/c3_s4_build2.py` (replaces the stalled `c3_s4_build.py`).
Statistics: `research/scripts/c3_s4_report.py` → `results/c3/s4_work/report.json`. Log: `research/analysis/c3_s4_run_log.txt`.
Outputs: `results/c3/queries_S4C3.parquet`, `queries_S4PC.parquet`, `truth_S4.parquet` (qsets `S4C3`, `S4PC`), `held_S4.parquet`.

## 1. Counts (FACT, c3_s4_build2.py)

| Step | Class 3 (C3) | PubChem-only (PC) |
|---|---|---|
| enveda-180, ≥1 timsTOF spectrum, not in COCONUT, ≥1 `[M+H]+`/`[M-H]-` spectrum | 5,112 | 171,862 |
| PubChem/COCONUT isomers within ±1e-5 Da of exact mass | 25.7M pairs (10.4M SMILES, median 3,038 per candidate) | – |
| Isomers with the same ElementGraph hash (tautomer-invariant skeleton) | 121 pairs | – |
| **Dropped: canonical-tautomer key + formula equal to a PubChem structure** | **116** (all PubChem, none COCONUT) | – |
| Verified pool | 4,996 | 171,862 |
| Sampled (seed 20261003, 75 per exact-mass quartile) | 300 | 300 |
| Query spectra (all enveda-180 timsTOF, `[2M…]` excluded, ≤16 per molecule) | 1,390 | 1,423 |

- Held keys: 600 (the sampled keys; no other train InChIKey14 shares a canonical-tautomer key with them).
- 1 C3 truth has a tautomer alias in `correct`; 0 PC truths.
- **Method (FACT):** PubChem `mass` is exact monoisotopic mass (isomers agree to 1e-9 Da), so ±1e-5 Da selects
  the same-formula set without parsing 100M SMILES. Tautomers share heavy-atom connectivity, so the ElementGraph hash
  is a safe pre-filter; only the 121 skeleton matches were tautomer-canonicalised.
- **Deviation from the instruction:** `[2M+H]+`, `[2M+Na]+`, `[2M-H]-` spectra (≈25% of the first build) were
  dropped, because the test has no dimer adducts.

## 2. Distributions (FACT, c3_s4_report.py)

| Set | Molecules | Spectra | Precursor m/z quartiles | Spectra/molecule median (mean, max) | `[M+H]+` / `[M-H]-` |
|---|---|---|---|---|---|
| S4C3 | 300 | 1,390 | 312 / 334 / 355 | 4 (4.63, 11) | 0.70 / 0.24 |
| S4PC | 300 | 1,423 | 306 / 331 / 355 | 4 (4.74, 12) | 0.72 / 0.22 |
| S12 (bench) | 250 | 1,184 | 291 / 388 / 485 | 4 (4.74, 17) | 0.48 / 0.23 |
| visible test | 400 | 1,213 | 300 / 329 / 347 | 3 (3.03) | 0.79 / 0.16 |

## 3. Natural-product likeness (FACT; RDKit Contrib NP_Score; share with score > 0, median)

| Set | Share NP > 0 | Median |
|---|---|---|
| S4C3 | 0.017 | −1.16 |
| S4PC | 0.003 | −1.50 |
| **visible test (truth_SV)** | **0.007** | **−1.50** |
| S12 bench | 0.864 | +1.43 |
| S3 bench | 0.397 | −0.33 |

**CORRECTION (same day):** an earlier version inferred from this table that the test is synthetic. That was wrong.
- The visible test.parquet is placeholder data copied from enveda-180 (data page).
- The data page states that the hidden test focuses on natural-product-like molecules, and that enveda-180 is "a
  different region of chemical space from the test molecules".
- S4 therefore matches the test's instrument (timsTOF), adducts and mass range, but **not its chemistry**. S12
  (enveda-np-examples, "the closest library to the test set") remains the chemistry reference.

## 4. Examples (first 10 of each set)

- **S4C3:** `Cc1ccc(N2CCCN(C(=O)C(O)C3CC3)CC2)nn1`, `CC(C)(C)n1cnc(C(=O)NCC2(C(=O)O)CC2)n1`,
  `CC(C)NC(=O)Cn1cc2c(c(C#N)c1=O)CCCC2`, `Cc1cc(CNc2ncc(C)c(-c3ccccc3F)n2)ncn1`,
  `Cc1cc(Cc2nc3ccsc3c(=O)[nH]2)ccc1[N+](=O)[O-]`, `CC(=O)N1CC(NC2CCCOCC2)CCC1C(F)(F)F`,
  `CN(C)c1cncc(NC(=O)C2CCC(C(=O)O)N2)c1`, `CC1(C)CN(c2nc3c(F)cccc3o2)C2COCC21`,
  `c1cnc(-c2ccnc(NC3CCC4CCCCC4C3)n2)cn1`, `O=C(Cc1cnc2ccccc2c1)N1CCC(O)(C2CC2)C1`
- **S4PC:** `CCN(CC)C(=O)CSCC(=O)Nc1ccccc1`, `Cc1oc(C)c(C(=O)NCCc2ccccc2)c1C`, `CCCOc1nccnc1N1CC2CCC(O)CC2C1`,
  `CC(C)n1ccc(CN2CCCC2C(C)(C)O)n1`, `CN(C)c1ccnc(C(=O)NC(C)(C)CC(N)=O)c1`, `Cc1c(C(=O)Nc2cc(CC(C)C)nn2C)cnn1C`,
  `CC(C)c1nncn1CCNC(=O)C1CC1C1CC1`, `CC(C)c1nc(NC(=O)c2c(O)cccc2F)n(C)n1`, `Cc1cccc(Cc2noc(CCNC(=O)CO)n2)c1F`,
  `CC(C)NC1CCCN(C(=O)c2ccc3cccnc3c2)C1`

## 5. Limits

- **Learned nets:** S4 molecules were training molecules for the ho1 nets, the full-data nets and CFT (they hold out
  only S1/S2/S3 keys). S4 is leak-free for the library side (mask `held_S4`), but **not** for any learned scorer
  until a net is retrained with `held_S4.parquet` excluded (INFERENCE).
- **Bench wiring:** `research/bench/bench.py` does not read S4 yet; it needs a scenario entry using these queries,
  `truth_S4` and `held_S4`.

# Licence-clean ChEBI + LIPID MAPS structure table (replacement for prvsiyan/chebi-lipidmaps-casmi26)

Date of all web accesses and downloads: 2026-10-01.

## 1. Verdict

A licence-clean replacement is possible and has been built. Both primary sources are CC BY 4.0 (commercial use
and redistribution allowed, attribution required). The Kaggle dataset `prvsiyan/chebi-lipidmaps-casmi26` is
labelled CC-BY-NC-SA-4.0 by its uploader (Kaggle metadata, `licenses: [{"name": "CC-BY-NC-SA-4.0"}]`); that
label is the uploader's choice, not a restriction inherited from ChEBI or LIPID MAPS.

## 2. Licence evidence (primary sources)

### ChEBI (EMBL-EBI) - CC BY 4.0
- https://www.ebi.ac.uk/chebi/aboutChebiForward.do : "The data on this website is available under the Creative
  Commons License (CC BY 4.0)" (link target https://creativecommons.org/licenses/by/4.0/deed.en).
- https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/README (ChEBI Release 255, last update 2026-09-09):
  "LICENSE  License data is released under: Creative Commons Attribution 4.0 International License."
- https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/LICENSE : first line "Attribution 4.0 International" (full CC BY
  4.0 legal code). Copies saved as `external/structdb/CHEBI_LICENSE.txt` and `CHEBI_SDF_README.txt`.
- Inconsistency to be aware of: https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/README contains an older
  paragraph saying the content is under "CC Attribution-ShareAlike 4.0 International License", while the same
  README's file list and the LICENSE file in that same directory say "Creative Commons Attribution 4.0
  International License". We used only the SDF directory, whose README and LICENSE both say CC BY 4.0 with no
  ShareAlike wording. Either way commercial use and redistribution are allowed.
- Requested attribution (flat_files README): "ChEBI data is from http://www.ebi.ac.uk/chebi - the version of
  ChEBI is 255."
- Files available: `SDF/` chebi.sdf.gz (139M, complete), chebi_3_stars.sdf.gz (57M), chebi_lite.sdf.gz (64M:
  molfile, ChEBI ID, name, star), chebi_lite_3_stars.sdf.gz (15M); `flat_files/` structures.tsv.gz (88M),
  compounds.tsv.gz, chemical_data.tsv.gz, names.tsv.gz, etc. The old `Flat_file_tab_delimited/` path is 404.

### LIPID MAPS LMSD - CC BY 4.0
- https://www.lipidmaps.org/terms-of-use : "All LIPID MAPS(R) databases are licensed under a Creative Commons
  Attribution 4.0 International License. Use the How to Link & Cite page for information on providing
  attribution to LIPID MAPS(R)."
- https://www.lipidmaps.org/databases/lmsd/download : "The SDF file contains exports of structures and
  annotations from LMSD active records. This file is available as CC BY 4.0." (links to
  https://creativecommons.org/licenses/by/4.0/). The IDs/InChIKey/SMILES TSV is "available as CC0".
  Both pages saved as HTML in `external/structdb/`.
- Files available: LMSD SDF zip (20.9 MB, dated 2026-10-01), verbose SDF zip (21.2 MB), RDF/TTL (35.5 MB),
  IDs+structures TSV (7.5 MB, CC0).

## 3. What the non-commercial dataset contains and how the notebook uses it

Files: `bio_fp.npy` (62,744 x 867 uint8 = 6,930 bits packed with np.packbits), `bio_mass.npy` (62,744 float64,
ascending, 100.01-1299.30 Da), `bio_meta.pkl` (dict: `keys` object array of InChIKey first blocks, all unique;
`smiles` object array, stereo-free; `nbits` = 6930).

Consumption (engine string in the cell "# Our engine (two-ranker + BIO + AFIX)", function `build_pool`):
`bd = dirname(find("bio_fp.npy"))`, loads `bio_meta.pkl`, keeps rows whose key is not already in the COCONUT
pool, and vstack/concatenates fp, mass, keys, smiles onto the pool. The fingerprint is
Morgan r2 4096 + Morgan r3 4096 + RDKit path FP 2048 (maxPath 6) + MACCS 167, indexed by `fp_bits.npy` (6,930
selected bit positions) and packed. `fp_bits.npy` comes from `prvsiyan/coconut-casmi26-candidates`, which is
labelled CC BY 4.0 on Kaggle.

The dataset was downloaded to the session scratch directory only (not into the project) and used solely for
schema inspection, key comparison and fingerprint-recipe validation.

## 4. What was built

Code: `research/structdb/build_structdb.py` (build, 4 workers, ~9 min) and `research/structdb/coverage.py`.

Inputs in `external/structdb/` (about 90 MB total):
- `chebi_lite.sdf.gz` from https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi_lite.sdf.gz, release 255,
  sha256 45a7f612ca3a6d7ae1cd802f2826874a57d2b81a0edac4a369a11e45d33d9fd7, 199,606 records.
- `LMSD.sdf.zip` from https://www.lipidmaps.org/files/?file=LMSD&ext=sdf.zip, dated 2026-10-01,
  sha256 3350fda0088b3e22b31ecc69c49fa86fe2d2c5c671758da69a00ce77ff2a736a, 50,586 records.
- `fp_bits.npy` (single file from the CC BY 4.0 Kaggle dataset above), only for the notebook-layout output.

Filters (RDKit 2026.03.6, from the molfiles): parses; single component; elements within C H N O P S F Cl Br I;
net formal charge 0; no radical electrons; no isotope labels; monoisotopic mass 150-1,170 Da. Key = first 14
characters of the InChIKey; SMILES = canonical with stereochemistry removed; one row per key (ChEBI record
preferred as representative, lowest id).

| reason | ChEBI | LIPID MAPS |
|---|---|---|
| records read | 199,606 | 50,586 |
| RDKit parse failure | 112 | 415 |
| multi-component | 5,007 | 3 |
| element outside set (incl. R-groups) | 13,313 | 147 |
| net charge != 0 | 11,597 | 457 |
| radical | 185 | 0 |
| isotope-labelled | 300 | 221 |
| mass outside 150-1,170 | 14,498 | 6,996 |
| key failure | 1 | 0 |
| kept records | 154,593 | 42,347 |
| unique InChIKey-14 | 118,899 | 37,567 |

Outputs in `results/structdb/`:
- `chebi_lipidmaps_clean.parquet` - 141,718 rows, mass-sorted; columns ik, smiles, mass, src, src_id, chebi_id,
  lm_id, name. src: chebi only 104,151; lipidmaps only 22,819; both 14,748.
- `dropin/bio_fp.npy` (141,718 x 867 uint8), `dropin/bio_mass.npy`, `dropin/bio_meta.pkl` (keys, smiles, nbits)
  - same file names, dtypes and dict layout that `build_pool` reads.
- `build_stats.json`, `build_log.txt`, `coverage.json`.

## 5. Coverage

| src | rows | in our universe | in PubChem | in neither |
|---|---|---|---|---|
| all | 141,718 | 81,278 (57.4%) | 135,263 (95.4%) | 3,636 |
| chebi only | 104,151 | 60,952 | 99,979 | 2,637 |
| lipidmaps only | 22,819 | 9,593 | 21,648 | 748 |
| both | 14,748 | 10,733 | 13,636 | 251 |

Universe = `results/kaggle_v1_assets/universe.parquet` (729,388 keys: coconut 479,717, train 249,671).
60,440 keys are new relative to the universe. PubChem = `external/pubchem/pubchem_rows.parquet`, semi-joined on
ik with DuckDB.

Against the non-commercial dataset (62,744 keys):
- 60,425 of their keys are in ours = 96.3% of all, 99.5% of the 60,722 that fall in our 150-1,170 Da range.
- 2,022 of theirs are outside our mass range (their range is 100-1,300 Da); 297 in-range keys are not
  reproduced (1 of those is in our universe anyway).
- 81,293 of our keys are not in theirs: ours is 2.26x larger.
- Fingerprint recipe check: on the 60,425 shared keys, 60,398 packed rows are byte-identical to theirs; the 27
  others differ by a median of 42 bits (most likely a different tautomer/stereo-free SMILES for the same
  connectivity). The recipe and bit selection are therefore reproduced.

## 6. What remains for a true drop-in swap

1. Not run in the notebook. The file layout matches, but nothing was uploaded to Kaggle (by instruction), so
   the swap is untested end to end and its leaderboard effect is unknown.
2. Size mismatch. Ours adds 141,718 rows versus 62,744; a larger pool changes candidate lists and may move the
   score in either direction. How the original was filtered is not documented (the size suggests a curated
   subset, e.g. ChEBI 3-star, but this was not verified). To get closer, rebuild from
   `chebi_lite_3_stars.sdf.gz` or keep the STAR tag (the lite SDF carries it; the build script currently drops
   it) and filter.
3. Mass range. Ours is 150-1,170 Da per the task spec; theirs is 100-1,300 Da. Widen `MASS_LO/MASS_HI` if the
   2,022 out-of-range keys matter.
4. Pickle/numpy version. `bio_meta.pkl` was written with the local numpy and protocol 4, same structure as the
   original (object arrays referencing `numpy._core`); it needs numpy 2.x on the Kaggle image, as the original
   does.
5. Attribution. A released solution must carry: "ChEBI data is from http://www.ebi.ac.uk/chebi - the version of
   ChEBI is 255" (CC BY 4.0) and a LIPID MAPS LMSD citation per its How to Link & Cite page (CC BY 4.0), plus
   credit for `fp_bits.npy` (prvsiyan/coconut-casmi26-candidates, CC BY 4.0).
6. Other dependencies of the 0.399 stack were not audited here. This work clears only the ChEBI/LIPID MAPS
   input; the remaining datasets in the kernel metadata each need their own licence check (Kaggle labels seen in
   passing: coconut-casmi26-candidates CC BY 4.0, casmi26-fp-models-v2 CC0, casmi26-ranker-features CC0).
7. Kaggle licence labels are uploader-declared. The CC BY 4.0 label on the dataset holding `fp_bits.npy` was
   taken at face value.
8. Zwitterions with net charge 0 are kept; molecules with any isotope label are dropped rather than
   de-labelled. Both are choices of this build, not requirements of the notebook.

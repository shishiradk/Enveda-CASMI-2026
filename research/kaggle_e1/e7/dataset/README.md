# casmi-e7-c3assets (CASMI 2026, team shishiradhikari11)

Class-3 candidate generation for the E7 notebook (`shishiradhikari11/casmi-e7-c3-channel`): per test molecule, two
generators edit known structures toward the measured mass and the merged list (`rr_bt`) is inserted at ranks 4-8 of
our E6 lists.

## Code (MIT licence, written by our team)
- `e7_c3_channel.py` - the channel (context builder, both generators, round-robin union).
- `biotransform.py` - hand-written biotransformation rules (glycosylation, acylation, methylation, oxidation, ...).
- `mmp_edit.py` - single-cut matched-molecular-pair edits with rules mined by us.
- `assets.py` - shared fragment/rule helpers.
- `e7_e6_channel.py` - copy of our E6 PubChem channel (`casmi-e6-pcnets/e6_channel.py`) plus a context dump.
- `e7_merge.py` - the E7 list merge.
Dependencies: RDKit (BSD-3-Clause), NumPy, pandas, pyarrow.

## Data assets (CC BY 4.0, because they contain COCONUT-derived structures)
- `c3_pool_{fp,mass,key}.npy`, `c3_pool_smiles.txt` - 712,199 structures: 436,389 COCONUT natural products
  (COCONUT, https://coconut.naturalproducts.net, CC BY 4.0; obtained via prvsiyan/coconut-casmi26-candidates) and
  275,810 structures of the competition `train.parquet`. Mass-sorted; 6,930-bit packed fingerprints (ECFP4 | ECFP6 |
  RDKit FP | MACCS, prvsiyan's bit selection `c3_fp_bits.npy`), computed by us with RDKit.
- `c3_pool_np.npy` - boolean mask: natural-product-like pool rows (COCONUT, plus train structures with RDKit Contrib
  NP-likeness > 0, or > -0.5 when found in a natural-product library).
- `c3_mmp_rules.parquet` - matched-molecular-pair transformation rules mined by us from train and COCONUT structures
  (used with support >= 3).
No NIST data, no non-commercial datasets and no third-party model weights are included. Competition train structures
are used as the host allows for derived artifacts; please attribute COCONUT (Sorokina et al., J. Cheminform. 2021) when
reusing the structure files.

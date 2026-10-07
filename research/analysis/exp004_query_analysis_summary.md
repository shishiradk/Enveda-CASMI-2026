# EXP-004 Query-Level Analysis Summary

- Total query count: 400
- Retained query count (retained_evidence_count > 0): 400
- Unreachable count (variant A): 12
- Only-cross adduct count: 12
- Same-adduct-only count: 149
- Same+cross count: 239

## Evidence-count distribution
- Mean: 7.35
- Median: 5
- Min: 1
- Max: 580
- Unique values: 23

## max Modified-Cosine distribution
- Mean: 0.1383
- Median: 0.0556
- Min: 0.0000
- Max: 0.9987

## Variant A rank distribution (for reachable queries)
- Mean: 2.72
- Median: 1.00
- Min: 1
- Max: 25

## Note on margin distribution
Margin distribution (correct_minus_top_wrong) is not available in the saved artifacts; requires per-candidate scores from EXP-003 inspection (92 queries only).

## Note on precursor_mz and molecular_formula
These fields are not available in the existing JSON artifacts and would require reading the train.parquet file, which was not possible due to missing parquet support in the environment.

import pandas as pd
a = pd.read_parquet("results/submission_v1/scores_smoke20.parquet")
b = pd.read_parquet("results/submission_v0/scores_sanity20.parquet")
print("V1 scores_smoke20 schema:\n", a.dtypes)
print("\nrows:", len(a), "cols:", list(a.columns))
print("n unique spectra:", a["spectrum_id"].nunique())
print("\nidentity V1 smoke20 vs V0 sanity20 (should be byte-identical):")
print("  shape equal:", a.shape == b.shape)
try:
    same = a.merge(b, on=["spectrum_id", "inchikey14", "cand_adduct"], suffixes=("_a", "_b"))
    print("  merges to", len(same), "/", len(a))
    if len(same):
        d = (same["score_a"] - same["score_b"]).abs().max()
        print("  max |score diff|:", d)
except Exception as e:
    print("  merge compare failed:", e)
print(a.head(3).to_string())
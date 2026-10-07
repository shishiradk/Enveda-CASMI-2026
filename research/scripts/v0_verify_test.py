import json
import duckdb

TRAIN = "D:/Enveda-CASMI-2026/train.parquet".replace("\\", "/")
TEST = "D:/Enveda-CASMI-2026/test.parquet".replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit='1GB'; SET threads=1")

cols = con.execute(f"DESCRIBE SELECT * FROM '{TEST}'").fetchall()
print("TEST COLUMNS:")
for c in cols:
    print(" ", c[0], c[1])

n_rows = con.execute(f"SELECT count(*) FROM '{TEST}'").fetchone()[0]
n_mid = con.execute(f"SELECT count(DISTINCT molecule_id) FROM '{TEST}'").fetchone()[0]
print(f"test rows: {n_rows}, distinct molecule_id: {n_mid}")

dup_mid = con.execute(
    f"SELECT molecule_id, count(*) n FROM '{TEST}' GROUP BY 1 HAVING count(*) > 1"
).fetchall()
print(f"molecule_ids with >1 row: {len(dup_mid)}")
if dup_mid:
    for d in dup_mid[:10]:
        print("  dup:", d)

nulls = con.execute(
    f"""SELECT
        sum(CASE WHEN molecule_id IS NULL THEN 1 ELSE 0 END) mol_null,
        sum(CASE WHEN spectrum_id IS NULL THEN 1 ELSE 0 END) spec_null,
        sum(CASE WHEN adduct IS NULL THEN 1 ELSE 0 END) adduct_null,
        sum(CASE WHEN precursor_mz IS NULL THEN 1 ELSE 0 END) pm_null,
        sum(CASE WHEN ms2_mzs IS NULL THEN 1 ELSE 0 END) mz_null,
        sum(CASE WHEN ms2_normalized_intensities IS NULL THEN 1 ELSE 0 END) int_null
      FROM '{TEST}'"""
).fetchone()
print("nulls:", nulls)

distinct_adduct = con.execute(f"SELECT adduct, count(*) n FROM '{TEST}' GROUP BY 1 ORDER BY 2 DESC").fetchall()
print("test adducts:")
for a in distinct_adduct:
    print(" ", a)

train_cols = con.execute(f"DESCRIBE SELECT * FROM '{TRAIN}'").fetchall()
print("TRAIN COLUMNS:")
for c in train_cols:
    print(" ", c[0], c[1])

n_train = con.execute(f"SELECT count(*) FROM '{TRAIN}'").fetchone()[0]
n_train_mid = con.execute(f"SELECT count(DISTINCT molecule_id) FROM '{TRAIN}'").fetchone()[0]
print(f"train rows: {n_train}, distinct molecule_id: {n_train_mid}")

import duckdb
try:
    import rdkit
    print("rdkit version:", rdkit.__version__)
except Exception as e:
    print("rdkit NOT available:", e)

try:
    import matchms
    print("matchms version:", matchms.__version__)
except Exception as e:
    print("matchms NOT available:", e)

con.close()
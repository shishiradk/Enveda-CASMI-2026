import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=1")

print("=== SCHEMA ===")
cols = con.execute("DESCRIBE SELECT * FROM read_parquet('train.parquet')").fetchall()
for c in cols:
    print(c[0], "|", c[1])

print("\n=== sample timsTOF rows ===")
rows = con.execute("""
SELECT inchikey14, adduct, precursor_mz, num_peaks, collision_energy_ev,
       collision_energy_orig, instrument_type, ingest_lib, ionization_mode,
       precursor_error_ppm
FROM read_parquet('train.parquet')
WHERE instrument_type = 'timsTOF'
LIMIT 5
""").fetchall()
for r in rows:
    print(r)

print("\n=== timsTOF totals ===")
print("rows:", con.execute("SELECT count(*) FROM read_parquet('train.parquet') WHERE instrument_type='timsTOF'").fetchone()[0])
print("molecules:", con.execute("SELECT count(DISTINCT inchikey14) FROM read_parquet('train.parquet') WHERE instrument_type='timsTOF'").fetchone()[0])

print("\n=== distinct CE formats (timsTOF) ===")
rows = con.execute("""
SELECT collision_energy_orig_units, count(*) n
FROM read_parquet('train.parquet')
WHERE instrument_type='timsTOF'
GROUP BY collision_energy_orig_units ORDER BY n DESC
""").fetchall()
for r in rows:
    print(r)

print("\n=== distinct collision_energy_orig samples (timsTOF) ===")
rows = con.execute("""
SELECT collision_energy_orig, count(*) n
FROM read_parquet('train.parquet')
WHERE instrument_type='timsTOF'
GROUP BY collision_energy_orig ORDER BY n DESC LIMIT 15
""").fetchall()
for r in rows:
    print(r)
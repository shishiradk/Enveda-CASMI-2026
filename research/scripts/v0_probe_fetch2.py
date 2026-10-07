import time
import duckdb
import pandas as pd

LO = 245.0831
HI = 460.1654

# --- Approach A: keep connection alive, in-memory table from one scan ---
con = duckdb.connect()
con.execute("SET threads=2")
t = time.time()
con.execute(
    f"""CREATE TABLE train_pool AS
        SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
        FROM 'train.parquet'
        WHERE precursor_mz BETWEEN {LO} AND {HI}"""
)
print(f"[A] in-memory pool build: {time.time()-t:.1f}s")
con.execute("PRAGMA disable_progress_bar")
# warm range
con.execute("SELECT count(*) FROM train_pool WHERE precursor_mz BETWEEN 300.0 AND 300.02").fetchall()

test = con.execute("SELECT spectrum_id, precursor_mz FROM 'test.parquet'").fetchdf().sort_values("precursor_mz").reset_index(drop=True)
CHUNK = 60
ts = []
for c0 in range(0, len(test), CHUNK):
    chunk = test.iloc[c0:c0 + CHUNK]
    con.register("q", chunk[["spectrum_id", "precursor_mz"]])
    t = time.time()
    lib = con.execute(
        f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
            FROM train_pool t SEMI JOIN q
            ON t.precursor_mz BETWEEN q.precursor_mz - 0.01 AND q.precursor_mz + 0.01
            WHERE TRUE"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    ts.append(time.time() - t)
    con.unregister("q")
print(f"[A] 21 in-memory chunk queries: total {sum(ts):.2f}s  avg {sum(ts)/len(ts):.3f}s")

# --- Approach B: persistent on-disk pool .duckdb, range query ---
t = time.time()
con2 = duckdb.connect("submit_v0_cache.duckdb")
con2.execute("SET threads=2")
con2.execute(
    f"""CREATE OR REPLACE TABLE train_pool AS
        SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
        FROM 'train.parquet'
        WHERE precursor_mz BETWEEN {LO} AND {HI}"""
)
print(f"[B] persistent pool build: {time.time()-t:.1f}s  (file lives in cwd)")
con2.execute("PRAGMA disable_progress_bar")
con2.execute("SELECT count(*) FROM train_pool WHERE precursor_mz BETWEEN 300.0 AND 300.02").fetchall()
ts = []
for c0 in range(0, len(test), CHUNK):
    chunk = test.iloc[c0:c0 + CHUNK]
    con2.register("q", chunk[["spectrum_id", "precursor_mz"]])
    t = time.time()
    lib = con2.execute(
        f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
            FROM train_pool t SEMI JOIN q
            ON t.precursor_mz BETWEEN q.precursor_mz - 0.01 AND q.precursor_mz + 0.01
            WHERE TRUE"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    ts.append(time.time() - t)
    con2.unregister("q")
print(f"[B] 21 persistent chunk queries: total {sum(ts):.2f}s  avg {sum(ts)/len(ts):.3f}s")
con2.close()
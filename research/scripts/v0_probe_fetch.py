import time
import duckdb
import pandas as pd

LO = 245.0831
HI = 460.1654

con = duckdb.connect()
con.execute("SET threads=2")
test = con.execute("SELECT spectrum_id, precursor_mz FROM 'test.parquet'").fetchdf()
test = test.sort_values("precursor_mz").reset_index(drop=True)
CHUNK = 60

print("--- per-chunk SEMI JOIN over train.parquet (current) ---")
tots = []
t0_all = time.time()
for c0 in range(0, len(test), CHUNK):
    chunk = test.iloc[c0:c0 + CHUNK]
    con.register("q", chunk[["spectrum_id", "precursor_mz"]])
    t = time.time()
    lib = con.execute(
        f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
            FROM 'train.parquet' t SEMI JOIN q
            ON t.precursor_mz BETWEEN q.precursor_mz - 0.01 AND q.precursor_mz + 0.01
            WHERE TRUE"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    dt = time.time() - t
    tots.append(dt)
    con.unregister("q")
    print(f"  chunk@core {c0}: {dt:.1f}s rows={len(lib)}")
print(f"TOTAL 21-chunk fetch: {time.time()-t0_all:.1f}s  (avg {sum(tots)/len(tots):.1f}s)")

print()
print("--- one-time scan to persistent .duckdb, then range query ---")
t = time.time()
con2 = duckdb.connect(str("submit_v0_cache.duckdb"))
con2.execute("SET threads=2")
con2.execute(
    f"""CREATE OR REPLACE TABLE train_pool AS
        SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
        FROM 'train.parquet'
        WHERE precursor_mz BETWEEN {LO} AND {HI}"""
)
print(f"  build pool table: {time.time()-t:.1f}s")
t = time.time()
n = con2.execute("SELECT count(*) FROM train_pool").fetchone()[0]
print(f"  count: {n}")
con2.execute("PRAGMA disable_progress_bar")
# warm
con2.execute("SELECT count(*) FROM train_pool WHERE precursor_mz BETWEEN 300.0 AND 300.02").fetchall()

ts = []
for i in range(5):
    loq = 245.09 + i * 43.0
    t = time.time()
    rows = con2.execute(f"SELECT precursor_mz FROM train_pool WHERE precursor_mz BETWEEN {loq-0.01} AND {loq+0.01}").fetchall()
    ts.append(time.time() - t)
    print(f"  range query @ {loq:.2f}: {ts[-1]:.4f}s rows={len(rows)}")
print(f"  avg range query: {sum(ts)/len(ts):.4f}s")
con2.close()
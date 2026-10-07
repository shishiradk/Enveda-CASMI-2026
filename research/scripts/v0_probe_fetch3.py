import time
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LO = 245.0831
HI = 460.1654
POOL = "results/submission_v0/pool.parquet"

con = duckdb.connect()
con.execute("SET threads=4")
test = con.execute("SELECT spectrum_id, precursor_mz FROM 'test.parquet'").fetchdf().sort_values("precursor_mz").reset_index(drop=True)

# one-time build: scan train.parquet once, keep pool, write sorted with small row groups
t = time.time()
df = con.execute(
    f"""SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
        FROM 'train.parquet'
        WHERE precursor_mz BETWEEN {LO} AND {HI}
        ORDER BY precursor_mz"""
).fetchdf()
print(f"one-time pool scan+sort: {time.time()-t:.1f}s  rows={len(df)}")

t = time.time()
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, POOL, row_group_size=4096, compression="zstd")
print(f"write pool.parquet (row_group=4096, zstd): {time.time()-t:.1f}s "
      f"size={__import__('os').path.getsize(POOL)/1e6:.0f}MB")

# per-chunk range query against sorted pool.parquet
con.execute("PRAGMA disable_progress_bar")
CHUNK = 60
ts = []
rows_tot = 0
for c0 in range(0, len(test), CHUNK):
    chunk = test.iloc[c0:c0 + CHUNK]
    loq = float(chunk["precursor_mz"].min()) - 0.01
    hiq = float(chunk["precursor_mz"].max()) + 0.01
    t = time.time()
    lib = con.execute(
        f"""SELECT precursor_mz, adduct, inchikey14, ms2_mzs, ms2_normalized_intensities
            FROM '{POOL}'
            WHERE precursor_mz BETWEEN {loq} AND {hiq}"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    ts.append(time.time() - t)
    rows_tot += len(lib)
print(f"21 pool.parquet range queries: total {sum(ts):.2f}s  avg {sum(ts)/len(ts):.3f}s  rows={rows_tot}")
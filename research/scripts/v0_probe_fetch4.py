import time
import duckdb
import pandas as pd

POOL = "results/submission_v0/pool.parquet"

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("PRAGMA disable_progress_bar")

# full run: 21 chunks of 60 (contiguous pm bands)
test = con.execute("SELECT spectrum_id, precursor_mz FROM 'test.parquet'").fetchdf().sort_values("precursor_mz").reset_index(drop=True)
CHUNK = 60
ts = []
rows_tot = 0
for c0 in range(0, len(test), CHUNK):
    chunk = test.iloc[c0:c0 + CHUNK]
    con.register("q", chunk[["spectrum_id", "precursor_mz"]])
    t = time.time()
    lib = con.execute(
        f"""SELECT t.precursor_mz, t.adduct, t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities
            FROM '{POOL}' t SEMI JOIN q
            ON t.precursor_mz BETWEEN q.precursor_mz - 0.01 AND q.precursor_mz + 0.01
            WHERE TRUE"""
    ).fetchdf().sort_values("precursor_mz").reset_index(drop=True)
    ts.append(time.time() - t)
    rows_tot += len(lib)
    con.unregister("q")
print(f"[SEMI JOIN over pool.parquet] 21 chunks: total {sum(ts):.2f}s avg {sum(ts)/len(ts):.3f}s rows={rows_tot}")

# range-band mode for the 21 contiguous chunks (narrow per band)
ts = []
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
print(f"[RANGE-BAND over pool.parquet] 21 chunks: total {sum(ts):.2f}s avg {sum(ts)/len(ts):.3f}s")
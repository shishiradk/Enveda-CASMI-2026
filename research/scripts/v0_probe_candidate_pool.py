import duckdb
con = duckdb.connect()
con.execute("SET threads=2")
t = con.execute("SELECT precursor_mz FROM 'test.parquet'").fetchdf()
lo = float(t["precursor_mz"].min() - 0.01)
hi = float(t["precursor_mz"].max() + 0.01)
n = con.execute(f"SELECT count(*) FROM 'train.parquet' WHERE precursor_mz BETWEEN {lo} AND {hi}").fetchone()[0]
nn = con.execute("SELECT count(*) FROM 'train.parquet'").fetchone()[0]
print(f"test pm: {float(t['precursor_mz'].min()):.2f} .. {float(t['precursor_mz'].max()):.2f}")
print(f"train rows in [lo-0.01, hi+0.01]: {n} / {nn} = {n/max(nn,1):.3f}")
print("lo hi:", lo, hi)
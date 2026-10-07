"""Add PubChem popularity counts to the candidate table: n_sid (substance records) and n_pmid (PubMed links) per CID.

    python research/kaggle_v3/build_pop.py

Reads external/pubchem/CID-SID.gz, CID-PMID.gz and pubchem_rows.parquet; writes pubchem_rows_pop.parquet (same rows and
order, plus n_sid, n_pmid as int32).
"""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = Path(__file__).resolve().parents[2] / "external" / "pubchem"
N = 200_000_000


def count(path, ncol):
    cols = ", ".join(f"'c{i}': 'VARCHAR'" for i in range(ncol))
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='3GB'")
    rd = con.execute(f"SELECT TRY_CAST(c0 AS BIGINT) AS cid FROM read_csv('{path.as_posix()}', delim='\t', header=false, "
                     f"columns={{{cols}}}, null_padding=true, ignore_errors=true)").fetch_record_batch(2_000_000)
    out = np.zeros(N, np.int32)
    n = 0
    for b in rd:
        c = b.column(0).to_numpy(zero_copy_only=False)
        c = c[np.isfinite(c.astype(float))].astype(np.int64)
        c = c[(c >= 0) & (c < N)]
        if len(c):
            bc = np.bincount(c)
            out[:len(bc)] += bc.astype(np.int32)
        n += len(c)
    con.close()
    print(time.strftime("%H:%M:%S"), path.name, "rows", n, "cids", int((out > 0).sum()), flush=True)
    return out


def main():
    t0 = time.time()
    sid = count(D / "CID-SID.gz", 3)
    pmid = count(D / "CID-PMID.gz", 3)
    src = pq.ParquetFile(D / "pubchem_rows.parquet")
    schema = src.schema_arrow.append(pa.field("n_sid", pa.int32())).append(pa.field("n_pmid", pa.int32()))
    w = pq.ParquetWriter(D / "pubchem_rows_pop.parquet", schema, compression="zstd")
    rows = 0
    for i in range(src.metadata.num_row_groups):
        t = src.read_row_group(i)
        cid = np.clip(t["cid"].to_numpy(), 0, N - 1)
        t = t.append_column("n_sid", pa.array(sid[cid])).append_column("n_pmid", pa.array(pmid[cid]))
        w.write_table(t, row_group_size=100_000)
        rows += t.num_rows
    w.close()
    info = {"rows": rows, "sec": round(time.time() - t0, 1)}
    json.dump(info, open(D / "build_pop.json", "w"))
    print(info)


if __name__ == "__main__":
    main()

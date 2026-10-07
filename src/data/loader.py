"""Reusable data access for the CASMI 2026 train/test parquet files.

IMPORTANT: pandas.read_parquet / pyarrow.parquet.read_table fail on these files in
this environment with:

    OSError: Repetition level histogram size mismatch

(observed with pyarrow==19.0.0, on both train.parquet and test.parquet). This is a
pyarrow reader bug related to the repetition-level statistics written into the
Parquet footer for the nested list<double> columns (ms2_mzs, ms2_normalized_intensities,
collision_energy_ev). duckdb reads the same files without issue, so it is used as the
canonical reader here. Re-check whether pyarrow still has this bug before relying on
pandas.read_parquet directly in the final Kaggle notebook -- Kaggle's pyarrow version
may differ from the local one.
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


def connect() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory duckdb connection pointed at the competition files."""
    return duckdb.connect()


def query(sql: str, con: duckdb.DuckDBPyConnection | None = None):
    """Run a SQL query (referencing 'train.parquet' / 'test.parquet' by relative
    path, or use train()/test() helpers below to build queries) and return a
    pandas DataFrame."""
    owns_con = con is None
    con = con or connect()
    try:
        return con.execute(sql).fetchdf()
    finally:
        if owns_con:
            con.close()


def train_df(columns: str = "*", where: str = "", limit: int | None = None):
    """Load (a slice of) train.parquet as a pandas DataFrame via duckdb.

    Avoid calling with columns='*' and no limit/where on the full 2.5M-row table
    unless you actually need everything in memory (the list columns are large).
    """
    sql = f"SELECT {columns} FROM '{TRAIN_PATH.as_posix()}'"
    if where:
        sql += f" WHERE {where}"
    if limit is not None:
        sql += f" LIMIT {limit}"
    return query(sql)


def test_df(columns: str = "*", where: str = ""):
    sql = f"SELECT {columns} FROM '{TEST_PATH.as_posix()}'"
    if where:
        sql += f" WHERE {where}"
    return query(sql)


def sample_submission_df():
    import pandas as pd

    return pd.read_csv(SAMPLE_SUBMISSION_PATH)

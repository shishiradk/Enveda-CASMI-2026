"""EXP-017 asset: PubChem candidate table for Class-2 retrieval.

    python research/kaggle_v1/build_pubchem.py smiles   # stage 1: CID-SMILES.gz -> filtered parts (cid, smiles, mass)
    python research/kaggle_v1/build_pubchem.py join     # stage 2: + CID-InChI-Key.gz -> one row per InChIKey14

Filters (the competition states test monoisotopic masses 157-1,159 Da, one of 10 singly charged adducts):
  - text: single component (no '.'), no isotope labels, no charged or bracketed exotic atoms, elements limited to
    C H N O P S F Cl Br I, at least one carbon, SMILES length <= 300;
  - RDKit: parses and sanitises, no radicals, exact neutral mass in [150, 1170] Da (5 ppm-window margin).
Stage 2 merges PubChem's stereo / isotope-free duplicates per InChIKey14: representative = lowest CID (its SMILES is
what would be submitted), n_cid = number of merged CIDs (a weak popularity prior, kept as a column, not yet used).

Outputs (external/pubchem/): parts/part_*.parquet, pubchem_ik14.parquet (ik, cid, smiles, mass, n_cid; sorted by mass),
build_*.json.
"""
import gzip
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "external" / "pubchem"
PARTS = D / "parts"
MASS_LO, MASS_HI = 150.0, 1170.0
MAXLEN = 300
CHUNK = 200_000

BRACKET = re.compile(r"\[([^\]]*)\]")
BR_OK = re.compile(r"^(?:C|N|O|P|S|F|Cl|Br|I|c|n|o|s|p)(?:@{0,2})(?:H\d?)?(?:@{0,2})$")
OUTSIDE_BAD = re.compile(r"B(?!r)|b|A|D|E|G|K|L|M|R|T|U|V|W|X|Y|Z|a|d|e|f|g|h|i|k|l(?<!Cl)|m|q|r(?<!Br)|t|u|v|w|x|y|z")


def text_ok(s):
    if len(s) > MAXLEN or "." in s or ("C" not in s and "c" not in s):
        return False
    for b in BRACKET.findall(s):
        if not BR_OK.match(b):  # isotopes ([13C]), charges ([N+], [O-]), metals, [2H] ... all rejected
            return False
    return not OUTSIDE_BAD.search(BRACKET.sub("", s))


def rd(batch):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    out = []
    for cid, s in batch:
        m = Chem.MolFromSmiles(s)
        if m is None or Descriptors.NumRadicalElectrons(m):
            continue
        w = Descriptors.ExactMolWt(m)
        if MASS_LO <= w <= MASS_HI:
            out.append((cid, s, w))
    return out


def stage_smiles():
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    n_in = n_text = n_out = part = 0

    def batches():
        nonlocal n_in, n_text
        buf = []
        with gzip.open(D / "CID-SMILES.gz", "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                n_in += 1
                cid, _, s = line.rstrip("\n").partition("\t")
                if s and text_ok(s):
                    n_text += 1
                    buf.append((int(cid), s))
                    if len(buf) == 20_000:
                        yield buf
                        buf = []
        if buf:
            yield buf

    acc = []
    with Pool(6) as pool:
        for res in pool.imap(rd, batches(), chunksize=1):
            acc.extend(res)
            if len(acc) >= CHUNK:
                pd.DataFrame(acc, columns=["cid", "smiles", "mass"]).to_parquet(PARTS / f"part_{part:05d}.parquet", index=False)
                n_out += len(acc); part += 1; acc = []
                if part % 10 == 0:
                    print(f"{time.strftime('%H:%M:%S')} in {n_in:,} text-ok {n_text:,} kept {n_out:,} ({time.time() - t0:.0f}s)", flush=True)
    if acc:
        pd.DataFrame(acc, columns=["cid", "smiles", "mass"]).to_parquet(PARTS / f"part_{part:05d}.parquet", index=False)
        n_out += len(acc)
    info = {"cids_in": n_in, "text_ok": n_text, "kept": n_out, "sec": round(time.time() - t0, 1)}
    json.dump(info, open(D / "build_smiles.json", "w"), indent=1)
    print(info)


def stage_join():
    import duckdb
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='8GB'; SET threads=6; SET temp_directory='{(D / 'duck_tmp').as_posix()}'")
    con.execute(f"""
        CREATE TABLE ik AS SELECT column0::BIGINT AS cid, left(column2, 14) AS ik
        FROM read_csv('{(D / "CID-InChI-Key.gz").as_posix()}', delim='\t', header=false, quote='',
                      columns={{'column0': 'VARCHAR', 'column1': 'VARCHAR', 'column2': 'VARCHAR'}})""")
    con.execute(f"""
        COPY (SELECT ik, min(cid) AS cid, arg_min(smiles, cid) AS smiles, arg_min(mass, cid) AS mass, count(*) AS n_cid
              FROM read_parquet('{(PARTS / "*.parquet").as_posix()}') JOIN ik USING (cid)
              GROUP BY ik ORDER BY mass)
        TO '{(D / "pubchem_ik14.parquet").as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION ZSTD)""")
    n = con.execute(f"SELECT count(*), sum(n_cid) FROM '{(D / 'pubchem_ik14.parquet').as_posix()}'").fetchone()
    parts = con.execute(f"SELECT count(*) FROM read_parquet('{(PARTS / '*.parquet').as_posix()}')").fetchone()[0]
    info = {"filtered_cids": int(parts), "ik14_rows": int(n[0]), "cids_joined": int(n[1]), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(D / "build_join.json", "w"), indent=1)
    print(info)


def stage_join2():
    """Low-memory replacement for stage_join (which exceeded RAM): streaming merge-join by CID (both inputs are in
    ascending CID order), 1-Da mass partitions, then per-partition sort into one mass-sorted parquet. Stereo/isotope
    duplicates are NOT merged globally: rows keep (ik, cid); consumers merge per mass window (min cid, count = n_cid)."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    J = D / "joined"
    J.mkdir(exist_ok=True)
    parts = sorted(PARTS.glob("part_*.parquet"))
    kf = gzip.open(D / "CID-InChI-Key.gz", "rt", encoding="utf-8", errors="replace")
    kc, kk = -1, None
    n_in = n_out = 0
    for i, p in enumerate(parts):
        df = pd.read_parquet(p)
        assert df["cid"].is_monotonic_increasing
        iks = []
        for c in df["cid"].values:
            while kc < c:
                line = kf.readline()
                if not line:
                    kc = 1 << 62
                    break
                a, _, rest = line.partition("\t")
                kc, kk = int(a), rest.rsplit("\t", 1)[-1].strip()
            iks.append(kk[:14] if kc == c and kk else None)
        df["ik"] = iks
        df = df[df["ik"].notna()]
        n_in += len(iks); n_out += len(df)
        df.to_parquet(J / f"j_{i:05d}.parquet", index=False)
        if i % 50 == 0:
            print(f"{time.strftime('%H:%M:%S')} joined {i + 1}/{len(parts)} parts, rows {n_out:,}/{n_in:,}", flush=True)
    kf.close()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='4GB'; SET threads=4; SET preserve_insertion_order=false; "
                f"SET temp_directory='{(D / 'duck_tmp').as_posix()}'")
    B = D / "buckets"
    con.execute(f"""COPY (SELECT ik, cid, smiles, mass, floor(mass)::INT AS b FROM read_parquet('{(J / '*.parquet').as_posix()}'))
                    TO '{B.as_posix()}' (FORMAT PARQUET, PARTITION_BY (b), OVERWRITE_OR_IGNORE true)""")
    out = D / "pubchem_rows.parquet"
    schema = pa.schema([("ik", pa.string()), ("cid", pa.int64()), ("smiles", pa.string()), ("mass", pa.float64())])
    w = pq.ParquetWriter(out, schema, compression="zstd")
    n_final = 0
    for bd in sorted(B.glob("b=*"), key=lambda x: int(x.name[2:])):
        t = con.execute(f"SELECT ik, cid, smiles, mass FROM read_parquet('{(bd / '*.parquet').as_posix()}') ORDER BY mass, cid").arrow()
        if not hasattr(t, "column_names"):
            t = t.read_all()
        w.write_table(t.cast(schema), row_group_size=100_000)
        n_final += t.num_rows
    w.close()
    info = {"filtered_cids": n_in, "with_inchikey": n_out, "final_rows": n_final, "sec": round(time.time() - t0, 1)}
    json.dump(info, open(D / "build_join2.json", "w"), indent=1)
    print(info)


if __name__ == "__main__":
    {"smiles": stage_smiles, "join": stage_join, "join2": stage_join2}[sys.argv[1]]()

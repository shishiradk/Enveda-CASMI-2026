"""Class census of the training molecules: which ones are in COCONUT, in PubChem, or in neither (real Class 3).

    python research/scripts/c3_census.py

Keys are the stored train inchikey14 against the raw InChIKey14 of PubChem (external/pubchem/pubchem_rows.parquet,
100.9M filtered CIDs) and of COCONUT (results/kaggle_v1_assets/universe.parquet, src = coconut). Tautomer variants are
NOT reconciled here, so "absent" is an upper bound; c3_verify.py checks the absent set with the metric's key.
Writes results/c3/census.json and results/c3/train_classes.parquet (one row per train inchikey14).
"""
import json, time
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3"
OUT.mkdir(parents=True, exist_ok=True)
T = (ROOT / "train.parquet").as_posix()
PC = (ROOT / "external/pubchem/pubchem_rows.parquet").as_posix()
U = (ROOT / "results/kaggle_v1_assets/universe.parquet").as_posix()

t0 = time.time()
con = duckdb.connect()
con.sql("SET preserve_insertion_order = false")
con.sql(f"""CREATE TABLE mol AS
    SELECT inchikey14 AS ik,
           any_value(normalized_smiles) AS smiles,
           any_value(molecular_formula) AS formula,
           list_distinct(list(ingest_lib)) AS libs,
           count(*) AS n_spec,
           sum(CASE WHEN instrument_type ILIKE '%tims%' THEN 1 ELSE 0 END) AS n_tims
    FROM read_parquet('{T}') GROUP BY 1""")
con.sql(f"CREATE TABLE coco AS SELECT DISTINCT ik FROM read_parquet('{U}') WHERE src = 'coconut'")
con.sql(f"""CREATE TABLE inpc AS SELECT DISTINCT p.ik FROM read_parquet('{PC}') p
            WHERE p.ik IN (SELECT ik FROM mol)""")
con.sql("""CREATE TABLE cls AS SELECT m.*,
           m.ik IN (SELECT ik FROM coco) AS in_coco,
           m.ik IN (SELECT ik FROM inpc) AS in_pc
           FROM mol m""")
con.sql(f"COPY cls TO '{(OUT / 'train_classes.parquet').as_posix()}' (FORMAT parquet)")


def tab(where=""):
    q = f"""SELECT count(*) n,
              sum((in_coco)::int) coco, sum((NOT in_coco AND in_pc)::int) pubchem_only,
              sum((NOT in_coco AND NOT in_pc)::int) neither
            FROM cls {where}"""
    return dict(zip(["n", "in_coconut", "pubchem_only", "neither"], con.sql(q).fetchone()))


res = {"all": tab(), "timstof": tab("WHERE n_tims > 0"),
       "np_examples": tab("WHERE list_contains(libs, 'enveda-np-examples')"),
       "enveda_180": tab("WHERE list_contains(libs, 'enveda-180')")}
res["by_lib"] = {}
for lib, in con.sql("SELECT DISTINCT unnest(libs) FROM cls").fetchall():
    res["by_lib"][lib] = tab(f"WHERE list_contains(libs, '{lib}')")
res["sec"] = round(time.time() - t0, 1)
json.dump(res, open(OUT / "census.json", "w"), indent=1)
print(json.dumps(res, indent=1))

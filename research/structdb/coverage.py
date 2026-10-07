"""Coverage / overlap report for results/structdb/chebi_lipidmaps_clean.parquet.

Usage: python research/structdb/coverage.py [dir_with_noncommercial_bio_files]
The optional directory (bio_fp.npy / bio_mass.npy / bio_meta.pkl of prvsiyan/chebi-lipidmaps-casmi26) is only
read to compare keys and to validate the fingerprint recipe; nothing from it is written to our outputs.
"""
import json, os, pickle, sys

import duckdb
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "structdb")
P = lambda *a: os.path.join(ROOT, *a).replace("\\", "/")


class NumpyOnly(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] == "numpy" and name in ("_reconstruct", "ndarray", "dtype"):
            return super().find_class(mod, name)
        raise pickle.UnpicklingError(f"blocked {mod}.{name}")


def main():
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='4GB'")
    con.execute(f"CREATE TABLE c AS SELECT * FROM read_parquet('{P('results/structdb/chebi_lipidmaps_clean.parquet')}')")
    q = lambda s: con.execute(s).fetchall()
    rep = {"rows": q("SELECT count(*) FROM c")[0][0], "by_src": dict(q("SELECT src, count(*) FROM c GROUP BY 1"))}
    uni = P("results/kaggle_v1_assets/universe.parquet")
    rep["universe_rows"] = q(f"SELECT count(*), count(DISTINCT ik) FROM read_parquet('{uni}')")[0]
    rep["universe_src"] = dict(q(f"SELECT src, count(*) FROM read_parquet('{uni}') GROUP BY 1 ORDER BY 2 DESC"))
    con.execute(f"CREATE TABLE in_uni AS SELECT ik FROM c SEMI JOIN read_parquet('{uni}') u ON c.ik = u.ik")
    pc = P("external/pubchem/pubchem_rows.parquet")
    con.execute(f"""CREATE TABLE in_pc AS SELECT DISTINCT p.ik FROM read_parquet('{pc}') p
                    SEMI JOIN c ON p.ik = c.ik WHERE p.mass BETWEEN 149.9 AND 1170.1""")
    rep["overlap_by_src"] = [dict(zip(["src", "n", "in_universe", "in_pubchem", "in_neither"], r)) for r in q("""
        SELECT coalesce(src, 'ALL'), count(*), count(u.ik), count(p.ik),
               sum(CASE WHEN u.ik IS NULL AND p.ik IS NULL THEN 1 ELSE 0 END)
        FROM c LEFT JOIN in_uni u ON c.ik = u.ik LEFT JOIN in_pc p ON c.ik = p.ik
        GROUP BY ROLLUP(src) ORDER BY 2 DESC""")]
    if len(sys.argv) > 1:
        d = sys.argv[1]
        meta = NumpyOnly(open(os.path.join(d, "bio_meta.pkl"), "rb")).load()
        keys = [str(k) for k in meta["keys"]]
        mass = np.load(os.path.join(d, "bio_mass.npy"))
        con.execute("CREATE TABLE nc(ik VARCHAR, mass DOUBLE)")
        con.executemany("INSERT INTO nc VALUES (?, ?)", list(zip(keys, mass.tolist())))
        r = q("""SELECT count(*), count(c.ik),
                        sum(CASE WHEN nc.mass BETWEEN 150 AND 1170 THEN 1 ELSE 0 END),
                        sum(CASE WHEN nc.mass BETWEEN 150 AND 1170 AND c.ik IS NOT NULL THEN 1 ELSE 0 END)
                 FROM nc LEFT JOIN c ON nc.ik = c.ik""")[0]
        rep["noncommercial"] = dict(zip(["their_rows", "their_keys_in_ours", "their_rows_150_1170",
                                         "their_150_1170_keys_in_ours"], r))
        rep["noncommercial"]["ours_not_in_theirs"] = q("SELECT count(*) FROM c ANTI JOIN nc ON c.ik = nc.ik")[0][0]
        rep["noncommercial"]["their_missing_in_universe_or_pubchem"] = q(f"""
            SELECT count(*), sum(CASE WHEN u.ik IS NOT NULL THEN 1 ELSE 0 END)
            FROM (SELECT nc.ik FROM nc ANTI JOIN c ON nc.ik = c.ik WHERE nc.mass BETWEEN 150 AND 1170) m
            LEFT JOIN (SELECT DISTINCT ik FROM read_parquet('{uni}')) u ON m.ik = u.ik""")[0]
        # fingerprint recipe check: for keys present in both, compare our packed rows with theirs
        fpo = os.path.join(OUT, "dropin", "bio_fp.npy")
        if os.path.exists(fpo):
            ours = np.load(fpo, mmap_mode="r")
            okeys = NumpyOnly(open(os.path.join(OUT, "dropin", "bio_meta.pkl"), "rb")).load()["keys"]
            theirs = np.load(os.path.join(d, "bio_fp.npy"), mmap_mode="r")
            pos = {k: i for i, k in enumerate(okeys)}
            pairs = [(pos[k], j) for j, k in enumerate(keys) if k in pos]
            a = np.asarray(ours[[p[0] for p in pairs]]); b = np.asarray(theirs[[p[1] for p in pairs]])
            same = (a == b).all(axis=1)
            diff = np.unpackbits(a ^ b, axis=1).sum(1)
            rep["fp_check"] = {"shape_ours": list(ours.shape), "shape_theirs": list(theirs.shape),
                               "shared_keys": len(pairs), "identical_rows": int(same.sum()),
                               "median_bit_diff_nonidentical": float(np.median(diff[~same])) if (~same).any() else 0.0}
    json.dump(rep, open(os.path.join(OUT, "coverage.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()

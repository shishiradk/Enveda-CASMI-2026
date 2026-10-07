"""EXP-007 dataset audit, Phase 1: metadata census of per-molecule spectrum inventory.

Goal: establish how many molecules have multiple genuinely distinct spectra under
various definitions, scoped to timsTOF queries (matching the real test set) with
all-train retained evidence (per DEC-003).

Runs cheap duckdb aggregations over the full 2.5M-row train set — no scoring.

Definitions used here (metadata-level proxies; peak-level similarity is Phase 2):
  nominal spectrum = (adduct, round(precursor_mz, 4), collision_energy_orig)
  metadata group   = (inchikey14, adduct, round(precursor_mz, 4), num_peaks)
                     [mirrors EXP-001's group but with rounded precursor to merge
                      ~0.0001-level float wobble]

Outputs: results/exp007_audit_phase1.json + printed summary.
"""

import json
import duckdb

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TRAIN = str(ROOT / "train.parquet")
OUT = str(ROOT / "results" / "exp007_audit_phase1.json")

con = duckdb.connect()
con.execute("PRAGMA threads=1")
con.execute("CREATE TEMP TABLE t AS SELECT * FROM read_parquet('train.parquet')")

print("rows:", con.execute("SELECT count(*) FROM t").fetchone()[0])

def histogram(q, value_col, label):
    rows = con.execute(q).fetchall()
    counts = [int(r[1]) for r in rows]
    total = sum(counts)
    n = len(rows)
    # simple percentile over the per-item counts
    def pct(p):
        s = sorted(counts)
        k = (n - 1) * p
        f = int(k)
        return s[f]
    print(f"{label}: n={n}, p10={pct(.10)}, p25={pct(.25)}, median={pct(.5)}, "
          f"p75={pct(.75)}, p90={pct(.90)}, max={max(counts)}, "
          f"sum={total}")
    return {"n": n, "p10": pct(.10), "p25": pct(.25), "median": pct(.5),
            "p75": pct(.75), "p90": pct(.90), "max": max(counts), "sum": total}

report = {}

# ---------- A. per-molecule spectra counts, full train ----------
print("\n=== A. spectra (rows) per molecule, FULL train ===")
report["spectra_per_molecule_full"] = histogram(
    "SELECT inchikey14, count(*) FROM t GROUP BY inchikey14", None, "spectra/molecule")

# ---------- B. per-molecule nominal spectra (adduct, prec, CE) ----------
print("\n=== B. nominal spectra per molecule (adduct, round(precursor,4), CE) ===")
report["nominal_spectra_per_molecule_full"] = histogram(
    """SELECT inchikey14, count(*) FROM (
           SELECT inchikey14, adduct, round(precursor_mz, 4) p, collision_energy_orig
           FROM t GROUP BY inchikey14, adduct, round(precursor_mz, 4), collision_energy_orig
       ) GROUP BY inchikey14""", None, "nominal spectra/molecule")

# ---------- C. per-molecule distinct adducts ----------
print("\n=== C. distinct adducts per molecule (full train) ===")
report["adducts_per_molecule_full"] = histogram(
    "SELECT inchikey14, count(DISTINCT adduct) FROM t GROUP BY inchikey14", None, "adducts/molecule")

# ---------- D. per-molecule distinct CE strings ----------
print("\n=== D. distinct collision_energy_orig per molecule (full train) ===")
report["ces_per_molecule_full"] = histogram(
    "SELECT inchikey14, count(DISTINCT collision_energy_orig) FROM t GROUP BY inchikey14", None, "CE/molecule")

# ---------- E. per-molecule distinct ingest_lib ----------
print("\n=== E. distinct ingest_lib per molecule (full train) ===")
report["libs_per_molecule_full"] = histogram(
    "SELECT inchikey14, count(DISTINCT ingest_lib) FROM t GROUP BY inchikey14", None, "libs/molecule")

# ---------- F. molecules with >=2 spectra at SAME adduct (any CE) ----------
print("\n=== F. same-adduct spectra per (molecule, adduct) — max over adducts ===")
report["max_same_adduct_spectra_per_molecule"] = histogram(
    """SELECT inchikey14, max(n) FROM (
            SELECT inchikey14, adduct, count(*) n FROM t GROUP BY inchikey14, adduct
        ) GROUP BY inchikey14""", None, "max same-adduct spectra/molecule")

# ---------- G. molecules with >=2 spectra at SAME (adduct, CE) → replicate detection seed ----------
print("\n=== G. same (adduct, CE) rows per molecule — no precursor grouping (potential replicates) ===")
report["max_same_adduct_ce_per_molecule"] = histogram(
    """SELECT inchikey14, max(n) FROM (
            SELECT inchikey14, adduct, collision_energy_orig, count(*) n
            FROM t GROUP BY inchikey14, adduct, collision_energy_orig
        ) GROUP BY inchikey14""", None, "max same-adduct+CE rows/molecule")

# ---------- H. counts of molecules satisfying key thresholds ----------
print("\n=== H. molecule counts by threshold ===")
def count_where(label, sql):
    c = con.execute(sql).fetchone()[0]
    report[label] = c
    print(f"{label}: {c}")

count_where("mol_ge_2_spectra_full",
    "SELECT count(*) FROM (SELECT inchikey14 FROM t GROUP BY inchikey14 HAVING count(*) >= 2)")
count_where("mol_ge_3_spectra_full",
    "SELECT count(*) FROM (SELECT inchikey14 FROM t GROUP BY inchikey14 HAVING count(*) >= 3)")
count_where("mol_with_ge2_nominal_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            FROM t GROUP BY inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
        ) GROUP BY inchikey14 HAVING count(*) >= 2)""")
count_where("mol_with_ge2_adducts",
    "SELECT count(*) FROM (SELECT inchikey14 FROM t GROUP BY inchikey14 HAVING count(DISTINCT adduct) >= 2)")
count_where("mol_with_ge2_collision_energies",
    "SELECT count(*) FROM (SELECT inchikey14 FROM t GROUP BY inchikey14 HAVING count(DISTINCT collision_energy_orig) >= 2)")
count_where("mol_with_same_adduct_ge2_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct FROM t GROUP BY inchikey14, adduct HAVING count(*) >= 2
        ) GROUP BY inchikey14)""")
count_where("mol_with_same_adduct_ge2_nominal_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            FROM t GROUP BY inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            HAVING count(*) >= 2
        ) GROUP BY inchikey14)""")

# ---------- I. same analysis restricted to timsTOF-enveda (the test-relevant population) ----------
print("\n=== I. timsTOF-only per-molecule censuses ===")
con.execute("CREATE TEMP TABLE timstof AS SELECT * FROM t WHERE instrument_type = 'timsTOF'")
report["timsTOF_rows"] = con.execute("SELECT count(*) FROM timstof").fetchone()[0]
report["timsTOF_molecules"] = con.execute("SELECT count(DISTINCT inchikey14) FROM timstof").fetchone()[0]

count_where("timsTOF_mol_ge_2_spectra",
    "SELECT count(*) FROM (SELECT inchikey14 FROM timstof GROUP BY inchikey14 HAVING count(*) >= 2)")
count_where("timsTOF_mol_ge_3_spectra",
    "SELECT count(*) FROM (SELECT inchikey14 FROM timstof GROUP BY inchikey14 HAVING count(*) >= 3)")
report["timsTOF_nominal_spectra_per_molecule"] = histogram(
    """SELECT inchikey14, count(*) FROM (
           SELECT inchikey14, adduct, round(precursor_mz, 4), collision_energy_orig
           FROM timstof GROUP BY inchikey14, adduct, round(precursor_mz, 4), collision_energy_orig
       ) GROUP BY inchikey14""", None, "timsTOF nominal spectra/molecule")
count_where("timsTOF_mol_ge_2_nominal_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            FROM timstof GROUP BY inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
        ) GROUP BY inchikey14 HAVING count(*) >= 2)""")
count_where("timsTOF_mol_same_adduct_ge2_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct FROM timstof GROUP BY inchikey14, adduct HAVING count(*) >= 2
        ) GROUP BY inchikey14)""")
count_where("timsTOF_mol_same_adduct_ge2_nominal_spectra",
    """SELECT count(*) FROM (SELECT inchikey14 FROM (
            SELECT inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            FROM timstof GROUP BY inchikey14, adduct, round(precursor_mz,4), collision_energy_orig
            HAVING count(*) >= 2
        ) GROUP BY inchikey14)""")
report["timsTOF_adducts_per_molecule"] = histogram(
    "SELECT inchikey14, count(DISTINCT adduct) FROM timstof GROUP BY inchikey14", None, "timsTOF adducts/molecule")
report["timsTOF_ces_per_molecule"] = histogram(
    "SELECT inchikey14, count(DISTINCT collision_energy_orig) FROM timstof GROUP BY inchikey14", None, "timsTOF CE/molecule")

# ---------- J. replicate census: how many (molecule, adduct, CE, prec) cells have >1 row ----------
print("\n=== J. within-(molecule,adduct,CE,prec) row multiplicity (replicate seed) ===")
rows = con.execute("""
    SELECT mult, count(*) n_cells
    FROM (SELECT inchikey14, adduct, collision_energy_orig,
                 round(precursor_mz, 4), count(*) mult
          FROM timstof GROUP BY inchikey14, adduct, collision_energy_orig, round(precursor_mz, 4))
    GROUP BY mult ORDER BY mult
""").fetchall()
report["timsTOF_replicate_cells"] = {str(m): c for m, c in rows}
print(report["timsTOF_replicate_cells"])

# ---------- K. what is the source of multi-spectra?  CE-variation vs other ----------
print("\n=== K. For molecules with >=2 same-adduct nominal spectra (timsTOF), is the pair at different CE? ===")
rows = con.execute("""
    SELECT
      count(*) FILTER (WHERE same_adduct_pairs >= 1 OR same_adduct_diffce_pairs >= 1) AS n_mol_same_adduct_multispectra,
      count(*) FILTER (WHERE same_adduct_diffce_pairs >= 1) AS n_mol_same_adduct_diff_ce
    FROM (
      SELECT inchikey14,
             count(*) FILTER (WHERE adduct_a = adduct_b) AS same_adduct_pairs,
             count(*) FILTER (WHERE adduct_a = adduct_b AND ce_a != ce_b) AS same_adduct_diffce_pairs
      FROM (
        SELECT a.inchikey14, a.adduct adduct_a, a.ce_a, b.adduct adduct_b, b.ce_a ce_b
        FROM (SELECT inchikey14, adduct, collision_energy_orig ce_a, round(precursor_mz,4) p
              FROM timstof GROUP BY inchikey14, adduct, collision_energy_orig, round(precursor_mz,4)) a
        JOIN (SELECT inchikey14, adduct, collision_energy_orig ce_a, round(precursor_mz,4) p
              FROM timstof GROUP BY inchikey14, adduct, collision_energy_orig, round(precursor_mz,4)) b
          ON a.inchikey14 = b.inchikey14
         AND (a.adduct, a.ce_a, a.p) > (b.adduct, b.ce_a, b.p)
      )
      GROUP BY inchikey14
    )
""").fetchone()
report["K"] = {"n_mol_same_adduct_multispectra": rows[0], "n_mol_same_adduct_diff_ce": rows[1]}
print(report["K"])

with open(OUT, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nWrote {OUT}")
"""Task S4 (research/c3_scenarios_instruction.md), rewritten with stage checkpoints: timsTOF query sets for the
Class-3 (truth in neither PubChem nor COCONUT) and PubChem-only buckets, drawn from enveda-180.

    python research/scripts/c3_s4_build2.py [--workers 4]

Replaces c3_s4_build.py, which RDKit-parsed all 100M PubChem SMILES. Here the PubChem `mass` column is used as what it
is, the exact monoisotopic mass (isomers share it to 1e-9 Da), so a +-1e-5 Da window returns the same-formula isomer
set (~3,000 per candidate); an RDKit ElementGraph hash (tautomer-invariant heavy-atom connectivity) then keeps
only the isomers sharing the candidate's skeleton, and only those are tautomer-canonicalised. Every stage writes to results/c3/s4_work/ and is skipped
when its output exists. Final outputs: results/c3/{queries_S4C3,queries_S4PC,truth_S4,held_S4}.parquet.
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3"
WK = OUT / "s4_work"
TR = (ROOT / "train.parquet").as_posix()
PC = (ROOT / "external/pubchem/pubchem_rows.parquet").as_posix()
UNI = (ROOT / "results/kaggle_v1_assets/universe.parquet").as_posix()
CLS = (OUT / "train_classes.parquet").as_posix()
SEED, N_PER_SET, MAX_SPEC, TOL = 20261003, 300, 16, 1e-5
QCOLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "adduct", "ionization_mode",
         "instrument_type", "precursor_mz", "collision_energy_ev"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def _key_mass(smi, _te=[]):
    """(canonical-tautomer InChIKey14, exact mass, formula) or (None, nan, None)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if not _te:
        RDLogger.DisableLog("rdApp.*")
        _te.append(rdMolStandardize.TautomerEnumerator())
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None, float("nan"), None
        return (Chem.MolToInchiKey(_te[0].Canonicalize(m))[:14], Descriptors.ExactMolWt(m),
                rdMolDescriptors.CalcMolFormula(m))
    except Exception:
        return None, float("nan"), None


def _eg(smi):
    """Heavy-atom element graph (connectivity, no H / bond orders / charges): tautomers always share it."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolHash
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return rdMolHash.MolHash(m, rdMolHash.HashFunction.ElementGraph) if m is not None else None
    except Exception:
        return None


def hashed(smiles, workers, tag):
    """ElementGraph hash per SMILES, checkpointed in chunks of 200k under s4_work/eg_<tag>_*.parquet."""
    smiles = list(dict.fromkeys(smiles))
    parts, CH = [], 200000
    with Pool(workers) as pool:
        for i in range(0, len(smiles), CH):
            f = WK / f"eg_{tag}_{i // CH:05d}.parquet"
            if not f.exists():
                part = smiles[i:i + CH]
                pd.DataFrame({"smiles": part, "eg": pool.map(_eg, part, chunksize=500)}).to_parquet(f, index=False)
                log(f"element graphs[{tag}] {min(i + CH, len(smiles)):,}/{len(smiles):,}")
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def keyed(smiles, workers, tag):
    """Canonical keys for a list of SMILES, checkpointed in chunks of 20k under s4_work/keys_<tag>_*.parquet."""
    smiles = list(dict.fromkeys(smiles))
    parts, CH = [], 20000
    with Pool(workers) as pool:
        for i in range(0, len(smiles), CH):
            f = WK / f"keys_{tag}_{i // CH:05d}.parquet"
            if not f.exists():
                part = smiles[i:i + CH]
                r = pool.map(_key_mass, part, chunksize=100)
                pd.DataFrame({"smiles": part, "ckey": [x[0] for x in r], "mass": [x[1] for x in r],
                              "formula": [x[2] for x in r]}).to_parquet(f, index=False)
                log(f"keys[{tag}] {min(i + CH, len(smiles)):,}/{len(smiles):,}")
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["smiles", "ckey", "mass", "formula"])


def stage_cands(workers):
    """enveda-180 timsTOF molecules outside COCONUT with >=1 [M+H]+/[M-H]- spectrum; set C3 or PC; mass; C3 keys."""
    f = WK / "cands.parquet"
    if f.exists():
        return pd.read_parquet(f)
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '4GB'; SET temp_directory = '{(WK / 'duck_tmp').as_posix()}'")
    c = con.sql(f"""
        WITH ok AS (SELECT DISTINCT inchikey14 AS ik FROM read_parquet('{TR}')
                    WHERE ingest_lib = 'enveda-180' AND adduct IN ('[M+H]+', '[M-H]-')),
             k AS (SELECT k.ik, k.smiles, k.formula, CASE WHEN k.in_pc THEN 'PC' ELSE 'C3' END AS qset
                   FROM read_parquet('{CLS}') k JOIN ok USING (ik)
                   WHERE list_contains(k.libs, 'enveda-180') AND k.n_tims >= 1 AND NOT k.in_coco)
        SELECT k.*, p.pc_mass FROM k LEFT JOIN
          (SELECT ik, any_value(mass) AS pc_mass FROM read_parquet('{PC}') WHERE ik IN (SELECT ik FROM k) GROUP BY ik) p
        USING (ik)""").df()
    log(f"candidates with an [M+H]+/[M-H]- spectrum: {c.qset.value_counts().to_dict()}")
    K = keyed(c[c.qset == "C3"].smiles.tolist(), workers, "cand").drop_duplicates("smiles").set_index("smiles")
    c["ckey"] = c.smiles.map(K.ckey)
    c["mass"] = c.pc_mass.fillna(c.smiles.map(K.mass))
    c.drop(columns="pc_mass").to_parquet(f, index=False)
    return pd.read_parquet(f)


def stage_isomers(c):
    """Every PubChem / COCONUT structure within TOL Da of a C3 candidate's exact mass (equi-join on 1e-4 Da bins)."""
    f = WK / "isomers.parquet"
    if f.exists():
        return pd.read_parquet(f)
    q = c[c.qset == "C3"][["ik", "mass"]].dropna()
    qb = pd.DataFrame([(r.ik, r.mass, b) for r in q.itertuples()
                       for b in range(int(np.floor((r.mass - TOL) * 1e4)), int(np.floor((r.mass + TOL) * 1e4)) + 1)],
                      columns=["cand", "cmass", "bin"])
    con = duckdb.connect()
    con.sql(f"SET memory_limit = '5GB'; SET temp_directory = '{(WK / 'duck_tmp').as_posix()}'")
    con.register("qb", qb)
    iso = con.sql(f"""
        SELECT qb.cand, s.ik, s.smiles, s.src FROM
          (SELECT ik, smiles, mass, 'pubchem' AS src FROM read_parquet('{PC}')
           UNION ALL SELECT ik, smiles, mass, 'coconut' FROM read_parquet('{UNI}') WHERE src = 'coconut') s
        JOIN qb ON CAST(floor(s.mass * 1e4) AS BIGINT) = qb.bin
        WHERE abs(s.mass - qb.cmass) <= {TOL}""").df().drop_duplicates(["cand", "ik", "src"])
    iso.to_parquet(f, index=False)
    log(f"isomers: {len(iso):,} (candidate, structure) pairs, {iso.smiles.nunique():,} unique SMILES, "
        f"median {int(iso.groupby('cand').size().median()) if len(iso) else 0} per candidate")
    return iso


def stage_verify(c, iso, workers):
    f = WK / "verified.json"
    if f.exists():
        return json.load(open(f))
    c3 = c[c.qset == "C3"].set_index("ik")
    H = hashed(iso.smiles.tolist() + c3.smiles.tolist(), workers, "iso").drop_duplicates("smiles").set_index("smiles")
    iso = iso.assign(eg=iso.smiles.map(H.eg))
    n_pairs = len(iso)
    iso = iso[iso.eg.notna() & (iso.eg.values == c3.smiles.map(H.eg).reindex(iso.cand).values)]
    log(f"element-graph prefilter: {n_pairs:,} isomer pairs -> {len(iso):,} share the candidate's skeleton")
    K = keyed(iso.smiles.tolist(), workers, "iso").drop_duplicates("smiles").set_index("smiles")
    iso = iso.assign(ckey=iso.smiles.map(K.ckey), formula=iso.smiles.map(K.formula))
    hit = iso[(iso.ckey.notna()) & (iso.ckey.values == c3.ckey.reindex(iso.cand).values)
              & (iso.formula.values == c3.formula.reindex(iso.cand).values)]
    drop = sorted(set(hit.cand))
    res = dict(n_c3=len(c3), n_isomer_pairs=n_pairs, n_same_skeleton=len(iso), n_no_key=int(c3.ckey.isna().sum()), n_dropped=len(drop),
               dropped_by_src=hit.drop_duplicates("cand").src.value_counts().to_dict(),
               dropped_examples=hit.drop_duplicates("cand").head(10)[["cand", "ik", "src", "smiles"]].values.tolist(),
               verified=sorted(set(c3.index[c3.ckey.notna()]) - set(drop)))
    json.dump(res, open(f, "w"), indent=1)
    log(f"tautomer check: {len(c3)} C3 candidates, {len(drop)} dropped, {len(res['verified'])} verified")
    return res


def sample(c, ver):
    rng = np.random.default_rng(SEED)
    out = []
    for qset, pool in (("C3", c[c.ik.isin(ver["verified"])]), ("PC", c[c.qset == "PC"])):
        pool = pool.dropna(subset=["mass"]).sort_values("ik").reset_index(drop=True)
        pool["q"] = pd.qcut(pool.mass, 4, labels=False)
        for q, g in pool.groupby("q"):
            out.append(g.iloc[rng.choice(len(g), min(N_PER_SET // 4, len(g)), replace=False)].assign(qset=qset))
    return pd.concat(out, ignore_index=True)


def write_outputs(c, S, workers):
    rng = np.random.default_rng(SEED)
    duckdb.register("S", S[["ik", "qset"]])
    sp = duckdb.sql(f"""SELECT S.qset, t.inchikey14 AS molecule_id, 'r' || t.file_row_number AS spectrum_id,
                               t.ms2_mzs, t.ms2_normalized_intensities, t.adduct, t.ionization_mode,
                               t.instrument_type, t.precursor_mz, t.collision_energy_ev
                        FROM read_parquet('{TR}', file_row_number = true) t JOIN S ON t.inchikey14 = S.ik
                        WHERE t.ingest_lib = 'enveda-180' AND t.instrument_type = 'timsTOF'
                          AND t.adduct NOT LIKE '[2M%'""").df()  # the test has no dimer adducts
    sp = sp.sort_values(["molecule_id", "spectrum_id"]).reset_index(drop=True)
    sel = []  # at most MAX_SPEC spectra per molecule, seeded random subset (groupby.apply drops the key in pandas 3)
    for _, ix in sp.groupby("molecule_id").indices.items():
        sel.extend(ix if len(ix) <= MAX_SPEC else np.sort(rng.choice(ix, MAX_SPEC, replace=False)))
    keep = sp.iloc[np.sort(np.asarray(sel))].reset_index(drop=True)
    for qset in ("C3", "PC"):
        keep[keep.qset == qset][QCOLS].to_parquet(OUT / f"queries_S4{qset}.parquet", index=False)
    T = S.assign(correct=[";".join(sorted({r.ik} | ({r.ckey} if isinstance(r.ckey, str) else set())))
                          for r in S.itertuples()])
    T.rename(columns={"ik": "mid"}).assign(qset=lambda d: "S4" + d.qset)[
        ["qset", "mid", "smiles", "formula", "correct"]].to_parquet(OUT / "truth_S4.parquet", index=False)
    # held: sampled keys + every train molecule with the same formula whose canonical key matches a sampled one
    cls = pd.read_parquet(CLS, columns=["ik", "smiles", "formula"])
    same = cls[cls.formula.isin(set(S.formula)) & ~cls.ik.isin(set(S.ik))]
    K = keyed(same.smiles.tolist(), workers, "held").drop_duplicates("smiles").set_index("smiles")
    sk = set(S.ckey.dropna()) | set(S.ik)
    extra = same[same.smiles.map(K.ckey).isin(sk)].ik
    pd.DataFrame({"ik": sorted(set(S.ik) | set(extra))}).to_parquet(OUT / "held_S4.parquet", index=False)
    log(f"queries: {keep.groupby('qset').molecule_id.nunique().to_dict()} molecules, "
        f"{keep.qset.value_counts().to_dict()} spectra; held keys {len(set(S.ik) | set(extra))} "
        f"({len(extra)} train tautomer aliases)")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    WK.mkdir(parents=True, exist_ok=True)
    c = stage_cands(a.workers)
    iso = stage_isomers(c)
    ver = stage_verify(c, iso, a.workers)
    S = sample(c, ver)
    K = keyed(S[S.qset == "PC"].smiles.tolist(), a.workers, "pcsamp").drop_duplicates("smiles").set_index("smiles")
    S["ckey"] = S.ckey.where(S.qset == "C3", S.smiles.map(K.ckey))
    S.to_parquet(WK / "sampled.parquet", index=False)
    write_outputs(c, S, a.workers)
    log("done; report: python research/scripts/c3_s4_report.py")


if __name__ == "__main__":
    main()

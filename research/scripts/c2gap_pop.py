"""C2 gap diagnosis, stage 4: how popular is the truth among its same-formula PubChem isomers?

    python research/scripts/c2gap_pop.py

Populations: S3 PubChem-only truths (GNPS-type), S2 truths (enveda-np-examples), S4PC truths (enveda-180, i.e.
compounds Enveda bought and measured on timsTOF, like the hidden test). Isomers = PubChem structures (plain ik14) whose
monoisotopic mass is within 2e-4 Da of the truth's (a same-formula proxy). Popularity from
external/pubchem/pubchem_rows_pop.parquet, summed over CIDs per ik14: n_sid, n_pmid.
Reports per population: isomer count, truth percentile, popularity-only MRR@25 among isomers (ties: mean rank).
DuckDB memory_limit 1GB, 2 threads. Writes results/c2gap/pop.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bench"))
import bench  # noqa: E402
POP = (ROOT / "external/pubchem/pubchem_rows_pop.parquet").as_posix()


def main():
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = pd.read_parquet(ROOT / "results/c2gap/mol.parquet")
    truth = bench.load_truth()
    rows = []
    for r in mol[mol.b.isin(["PC", "S2"])].itertuples():
        T = truth["S12" if r.scen == "S2" else "S3"][r.mid]
        rows.append((r.b, r.mid, sorted(T["correct"]), T["smiles"]))
    t4 = pd.read_parquet(ROOT / "results/c3/truth_S4.parquet")
    for r in t4[t4.qset == "S4PC"].itertuples():
        rows.append(("S4PC", r.mid, sorted(r.correct.split(";")), r.smiles))
    q = []
    for b, mid, cor, smi in rows:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        q.append(dict(b=b, mid=mid, cor=";".join(cor), mass=ExactMolWt(m)))
    q = pd.DataFrame(q)
    q["bin"] = np.floor(q.mass * 1000).astype(np.int64)
    qb = pd.concat([q.assign(bin=q.bin + d) for d in (-1, 0, 1)], ignore_index=True)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='1GB'; SET threads=2; SET temp_directory='{(ROOT / 'results/c2gap/duck_tmp').as_posix()}'")
    con.register("qb", qb[["b", "mid", "mass", "bin"]])
    iso = con.sql(f"""SELECT qb.b, qb.mid, p.ik, sum(p.n_sid) n_sid, sum(p.n_pmid) n_pmid
                      FROM read_parquet('{POP}') p JOIN qb ON CAST(floor(p.mass * 1000) AS BIGINT) = qb.bin
                      WHERE abs(p.mass - qb.mass) < 2e-4 GROUP BY 1, 2, 3""").df()
    out = {}
    per = []
    for (b, mid), g in iso.groupby(["b", "mid"]):
        cor = set(q[(q.b == b) & (q.mid == mid)].cor.iloc[0].split(";"))
        hit = g.ik.isin(cor).values
        if not hit.any():
            per.append(dict(b=b, mid=mid, n_iso=len(g), found=False)); continue
        res = dict(b=b, mid=mid, n_iso=len(g), found=True)
        for f, v in (("sid", np.log1p(g.n_sid.values)), ("pmid", np.log1p(g.n_pmid.values)),
                     ("sum", np.log1p(g.n_sid.values) + np.log1p(g.n_pmid.values))):
            tv = v[hit].max()
            gt, eq = int((v > tv).sum()), int((v == tv).sum())
            rank = gt + (eq + 1) / 2.0  # mean rank among ties
            res[f"rank_{f}"] = rank
            res[f"pct_{f}"] = 1.0 - gt / len(g)
            res[f"truth_{f}"] = float(tv)
        res["n_sid_truth"] = int(g.n_sid.values[hit].max()); res["n_pmid_truth"] = int(g.n_pmid.values[hit].max())
        res["n_sid_iso_med"] = float(np.median(g.n_sid.values[~hit])) if (~hit).any() else np.nan
        per.append(res)
    per = pd.DataFrame(per)
    per.to_parquet(ROOT / "results/c2gap/pop_per_mol.parquet", index=False)
    for b, g in per.groupby("b"):
        f = g[g.found]
        d = dict(n=len(g), found=int(len(f)), n_iso_q=f.n_iso.quantile([.25, .5, .75]).tolist(),
                 truth_n_sid_q=f.n_sid_truth.quantile([.25, .5, .75]).tolist(),
                 truth_n_pmid_q=f.n_pmid_truth.quantile([.25, .5, .75]).tolist(),
                 iso_n_sid_median_q=f.n_sid_iso_med.quantile([.25, .5, .75]).tolist())
        for k in ("sid", "pmid", "sum"):
            rk = f[f"rank_{k}"].values
            d[f"pop_only_mrr25_{k}"] = round(float(np.where(rk <= 25, 1 / rk, 0).sum() / len(g)), 4)
            d[f"pop_only_top1_{k}"] = round(float((rk <= 1).sum() / len(g)), 4)
            d[f"truth_pct_median_{k}"] = round(float(f[f"pct_{k}"].median()), 4)
            d[f"truth_in_top10pct_{k}"] = round(float((f[f"pct_{k}"] >= 0.9).mean()), 4)
        out[b] = d
    json.dump(out, open(ROOT / "results/c2gap/pop.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

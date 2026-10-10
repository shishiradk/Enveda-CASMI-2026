"""Class-3 generation recall as a function of how many spectral analogs are used as parents.

For simulated class-3 queries: run the engine's own analog search (spectra only, truth masked exactly as in the
simulation), then ask for each analog in rank order whether derive() (engine budget: <=2 steps, 60 products) reaches
the truth.  recall@k = share of queries whose truth is derivable from one of the first k analogs.

    python research/scratch/analog_rank_c3.py --n 800 --out results/v4n/analog_rank_c3.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "v4n_rebuild"))
sys.path.insert(0, str(ROOT / "research" / "v4n_rebuild" / "sim"))
KS = (1, 3, 6, 12, 25, 50, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tables", default=str(ROOT / "results/v4n/tables"))
    ap.add_argument("--ckpt", default=str(ROOT / "models/cft_hoR_a/out_cft/export/cft_hoR_a.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/v4n/analog_rank_c3.json"))
    a = ap.parse_args()
    import simulate as S
    from engine import derive
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDLogger.DisableLog("rdApp.*")
    S._init(a.tables, [a.ckpt], 4, True)
    E, L, ex = S.W["E"], S.W["L"], S.W["ex"]
    cfg = E.cfg
    IK = pd.read_parquet(Path(a.tables) / 'train_structs.parquet', columns=['inchikey14']).inchikey14.values
    q = pd.read_parquet(ROOT / "results/v4n/sim/hoR_RA/queries.parquet")
    q = q[q.regime == "c3"].sample(a.n, random_state=a.seed).reset_index(drop=True)
    out, t0 = [], time.time()
    for i, r in q.iterrows():
        spectra = S._query_spectra(L, r["spec_idx"])
        target = float(r["target"]); sid0 = int(r["sid0"]); tik = IK[sid0]
        mi = np.asarray(r["mask_idx"], np.int64)
        ex[mi] = True
        try:
            an = E.analogs(spectra, target, ex, sid0, int(r["exclude_lib"]))
        finally:
            ex[mi] = False
        rec = dict(qid=int(r.qid), n_analogs=len(an), n_sim03=int(sum(1 for x in an if x[1] >= cfg.gen_min_sim)),
                   first_rank=None, first_sim=None, top_sim=float(an[0][1]) if an else None)
        for rank, (sid, sim, shift, _) in enumerate(an):
            if sid == sid0:
                continue
            delta = target - float(L.struct_mass[sid])
            try:
                prods = derive.derive(L.struct_smiles[sid], delta, tol=cfg.gen_tol_da, max_steps=2,
                                      max_out=cfg.gen_max_per_parent)
            except Exception:
                continue
            hit = False
            for smi, combo in prods:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                try:
                    if inchi.MolToInchiKey(m)[:14] == tik:
                        hit = True
                        break
                except Exception:
                    pass
            if hit:
                rec["first_rank"], rec["first_sim"] = rank, float(sim)
                break
        out.append(rec)
        if (i + 1) % 50 == 0:
            d = pd.DataFrame(out)
            print(f"{i+1}/{len(q)} {time.time()-t0:.0f}s | " +
                  " ".join(f"@{k} {(d.first_rank.fillna(10**9) < k).mean():.3f}" for k in KS), flush=True)
    d = pd.DataFrame(out)
    fr = d.first_rank.fillna(10**9)
    rep = dict(n=len(d), recall_at={str(k): float((fr < k).mean()) for k in KS},
               engine_rule=float(((fr < cfg.gen_n_analog) & (d.first_sim.fillna(0) >= cfg.gen_min_sim)).mean()),
               any_in_100=float(d.first_rank.notna().mean()),
               median_n_analogs=float(d.n_analogs.median()), median_n_sim03=float(d.n_sim03.median()),
               share_no_analog_ge_0p3=float((d.n_sim03 == 0).mean()), seconds=round(time.time() - t0))
    json.dump(dict(summary=rep, rows=out), open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

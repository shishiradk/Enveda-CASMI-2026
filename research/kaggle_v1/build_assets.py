"""Build the V1 structure universe for the Kaggle dataset "casmi-v1-assets".

    python research/kaggle_v1/build_assets.py

Universe = COCONUT 2026-09 (EXP-012 table, parse_ok) + train structures not in COCONUT (EXP-008 universe, parse_ok),
the same "COCONUT+train" construction as EXP-012. One row per InChIKey14, sorted by neutral mass.
Fingerprints: Morgan r2 / 2048 bits (rdFingerprintGenerator), packed, row-aligned with universe.parquet.
COCONUT rows reuse the EXP-012 fingerprints (parse_ok-filtered, i.e. the EXP-013-corrected indexing).

Outputs (results/kaggle_v1_assets/): universe.parquet (ik, smiles, mass, src), universe_fp.npy, build.json.
"""
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "kaggle_v1_assets"
FP_BITS = 2048


def _fp(smi):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS).GetFingerprint(m), arr)
    return np.packbits(arr)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C = pd.read_parquet(ROOT / "results" / "exp012_coconut" / "coconut_db.parquet")
    cfp = np.load(ROOT / "results" / "exp012_coconut" / "coconut_fp_packed.npy")
    mask = C["parse_ok"].values
    C, cfp = C[mask].reset_index(drop=True), cfp[mask]
    T = pd.read_parquet(ROOT / "results" / "exp008_structure_universe.parquet")
    T = T[T["parse_ok"] & ~T["ik"].isin(set(C["ik"]))].reset_index(drop=True)
    with Pool(6) as pool:
        tfp = np.stack(pool.map(_fp, T["smiles"].tolist(), chunksize=2000))
    # spot check: recomputed COCONUT fingerprints must equal the stored (parse_ok-aligned) ones
    rng = np.random.default_rng(20260930)
    samp = rng.choice(len(C), 300, replace=False)
    fp_check = int(sum(np.array_equal(_fp(C["smiles"].iat[i]), cfp[i]) for i in samp))
    U = pd.concat([C[["ik", "smiles", "mass"]].assign(src="coconut"), T[["ik", "smiles", "mass"]].assign(src="train")],
                  ignore_index=True)
    fp = np.concatenate([cfp, tfp])
    order = np.argsort(U["mass"].values, kind="stable")
    U, fp = U.iloc[order].reset_index(drop=True), fp[order]
    assert U["ik"].is_unique and np.isfinite(U["mass"]).all()
    U.to_parquet(OUT / "universe.parquet", index=False)
    np.save(OUT / "universe_fp.npy", fp)
    info = {"rows": len(U), "coconut": int((U["src"] == "coconut").sum()), "train_only": int((U["src"] == "train").sum()),
            "coconut_fp_recompute_match": f"{fp_check}/300",
            "sha256_fp": hashlib.sha256(fp.tobytes()).hexdigest(), "sec": round(time.time() - t0, 1)}
    json.dump(info, open(OUT / "build.json", "w"), indent=1)
    print(info)
    assert fp_check == 300


if __name__ == "__main__":
    main()

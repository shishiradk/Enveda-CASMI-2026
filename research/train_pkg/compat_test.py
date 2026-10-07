"""Compatibility proof (run on the project machine, not by the teammate).

    python research/train_pkg/compat_test.py --exp <out>/export [--public external/bench_inputs/fp]

1. loads the exported files with the ENGINE'S OWN loader, research/bench/eng/pv_fp.load_fp_models;
2. runs the engine's molecule_logits() on raw spectra read from train.parquet;
3. checks that the engine's per-spectrum logits on RAW peaks equal our network on the PREPARED arrays
   (proves the prepared data uses the engine's featurisation);
4. compares checkpoint structure (top-level keys, tensor names / shapes / dtypes) with the public files.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default=str(ROOT / "results/train_pkg/data"))
    ap.add_argument("--eng", default=str(ROOT / "research/bench/eng"))
    ap.add_argument("--public", default=str(ROOT / "external/bench_inputs/fp"))
    ap.add_argument("--train", default=str(ROOT / "train.parquet"))
    ap.add_argument("--n_mol", type=int, default=40)
    a = ap.parse_args()
    import torch, duckdb
    sys.path.insert(0, a.eng)
    import pv_fp                                    # the engine module, untouched
    import train_fp as T, fp_model as M
    R = {"engine_module": str(Path(pv_fp.__file__).resolve())}

    paths = sorted(str(p) for p in Path(a.exp).glob("fp_*.pt"))
    single, merged, nbits = pv_fp.load_fp_models(paths, "cpu")
    R["loaded"] = dict(paths=[Path(p).name for p in paths], single=len(single), merged=len(merged), nbits=nbits)
    assert len(single) == 1 and len(merged) == 1 and nbits == 6930

    pub = sorted(Path(a.public).glob("fp_*.pt")); R["structure"] = {}
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        for q in pub:
            pk = torch.load(q, map_location="cpu", weights_only=False)
            same = (set(ck) == set(pk) and list(ck["model"]) == list(pk["model"])
                    and all(ck["model"][k].shape == pk["model"][k].shape and ck["model"][k].dtype == pk["model"][k].dtype
                            for k in pk["model"])
                    and all(type(ck[k]) is type(pk[k]) for k in ("step", "nbits", "d", "layers")))
            R["structure"][f"{Path(p).name} vs {q.name}"] = bool(same)
            assert same
    R["top_level_keys"] = sorted(ck)

    D = T.Data(a.data, mmap=True, log=lambda *x: None)
    rng = np.random.default_rng(0)
    mols = rng.choice(np.unique(D.six[D.va]), size=a.n_mol, replace=False)
    rows = np.load(Path(a.data) / "spec_row.npy")
    con = duckdb.connect(); con.execute("set threads=2; set memory_limit='2GB'")
    maxd = 0.0; nspec = 0; fin = True; models = (single, merged, "cpu")
    for mth in mols:
        ids = D._grp_order[D._grp_bounds[mth]:D._grp_bounds[mth + 1]][:8]
        ids = np.sort(ids)
        rr = ",".join(str(int(rows[i])) for i in ids)
        q = con.sql(f"""select file_row_number r, ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct,
                        instrument_type, ionization_mode, collision_energy_ev
                        from read_parquet('{Path(a.train).as_posix()}', file_row_number=true)
                        where file_row_number in ({rr}) order by r""").fetchall()
        assert [x[0] for x in q] == [int(rows[i]) for i in ids]
        specs = [(np.array(x[1], float), np.array(x[2], float)) for x in q]
        prec = np.array([x[3] for x in q], np.float32); adduct = [x[4] for x in q]; instr = [x[5] for x in q]
        modes = np.array([1.0 if x[6] == "positive" else -1.0 for x in q])
        ces = np.array([float(np.mean(np.abs(x[7]))) if x[7] else 25.0 for x in q], np.float32)
        z = pv_fp.molecule_logits(models, specs, prec, adduct, instr, ces, modes)      # the engine's full path
        fin &= z is not None and z.shape == (6930,) and bool(np.isfinite(z).all())
        P = [pv_fp.prep_peaks(x, y, p) for (x, y), p in zip(specs, prec)]
        za = pv_fp.logits_batch(P, single, "cpu", prec, adduct, instr, ces, modes)      # engine, raw peaks
        inp, _ = D.batch(ids, 1, rng, aug=False, merge_p=0.0)                           # ours, prepared arrays
        with torch.no_grad():
            zb = single[0](*[torch.as_tensor(x) for x in inp]).float().mean(0).numpy()
        maxd = max(maxd, float(np.abs(za - zb).max())); nspec += len(ids)
    R["engine_molecule_logits"] = dict(molecules=int(len(mols)), spectra=int(nspec), all_finite_6930=bool(fin))
    R["featurisation_parity_max_abs_logit_diff"] = maxd
    assert fin and maxd < 1e-3, maxd
    R["verdict"] = "PASS"
    print(json.dumps(R, indent=1))
    json.dump(R, open(Path(a.exp) / "compat_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()

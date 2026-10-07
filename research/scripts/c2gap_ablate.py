"""C2 gap: ablation of the structure re-ranker (which feature carries the gain). -> results/c2gap/ablate.json"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
from c2gap_rerank import featurize, add_rel, run_cv  # noqa
D = ROOT / "results/c2gap"
c = pd.read_parquet(D / "cand.parquet"); mol = pd.read_parquet(D / "mol.parquet")
mol = mol[mol.b.isin(["PC", "S2"])].reset_index(drop=True); c = c[c.b.isin(["PC", "S2"])].reset_index(drop=True)
c = add_rel(featurize(c), mol)
S = ["score", "score_gap", "score_z", "score_rk"]
sets = {"score_feats": S, "+ppm": S + ["ppm", "abs_ppm"], "+np": S + ["np", "np_rel"], "+np+ppm": S + ["np", "np_rel", "ppm", "abs_ppm"],
        "+desc_no_np": S + ["ppm", "abs_ppm", "rings", "arom", "fsp3", "nstereo", "multi", "charge", "isotope", "hbd", "logp"]}
out = {k: run_cv(c, mol, v) for k, v in sets.items()}
# PC-only training (does S2/np-examples training carry the PC gain?)
out["+np+ppm_trainPConly"] = run_cv(c[c.b == "PC"].reset_index(drop=True), mol[mol.b == "PC"].reset_index(drop=True), S + ["np", "np_rel", "ppm", "abs_ppm"])
json.dump(out, open(D / "ablate.json", "w"), indent=1)
for k, v in out.items(): print(k, v)

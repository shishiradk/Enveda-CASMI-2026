"""For simulated class-3 queries: which structural changes (parent -> truth) do the 45 transforms NOT cover?
Writes results/v4n/c3_delta_pairs.csv (parent_smiles, truth_smiles, delta_formula, tanimoto, reachable) and prints top deltas."""
import sys, json, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/v4n_rebuild"))
from engine import derive
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog("rdApp.*")
LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
T = ROOT / "results/v4n/tables"
st = pd.read_parquet(T / "train_structs.parquet", columns=["sid", "inchikey14", "mass", "smiles"])
fp = np.load(T / "train_fp_raw.npy", mmap_mode="r")
mass = st.mass.values; order = np.argsort(mass, kind="mergesort"); ms = mass[order]; ik = st.inchikey14.values
cm = np.array([m for _, m in derive.combos()])
q = pd.read_parquet(ROOT / "results/v4n/sim/hoR_RA/queries.parquet", columns=["qid", "regime", "sid0"])
q = q[q.regime == "c3"].sample(3000, random_state=0)
rep = {r["qid"]: r["reach_top30"] for r in json.load(open(ROOT / "results/v4n/oracle_c3.json"))["rows"]}
def counts(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); c = collections.Counter(a.GetSymbol() for a in m.GetAtoms()); return c
rows = []
for _, r in q.iterrows():
    sid = int(r.sid0); M = mass[sid]
    pars = set()
    for c in cm:
        lo = np.searchsorted(ms, M - c - 0.003, "left"); hi = np.searchsorted(ms, M - c + 0.003, "right")
        if hi > lo: pars.update(order[lo:hi].tolist())
    pars = np.array([p for p in pars if ik[p] != ik[sid]], np.int64)
    if not len(pars): continue
    a = np.asarray(fp[sid]); B = np.asarray(fp[pars])
    inter = LUT[np.bitwise_and(B, a)].sum(1, dtype=np.int32); tan = inter / np.maximum(int(LUT[a].sum()) + LUT[B].sum(1, dtype=np.int32) - inter, 1)
    j = int(np.argmax(tan)); p = int(pars[j])
    try:
        ct, cp = counts(st.smiles.values[sid]), counts(st.smiles.values[p])
    except Exception:
        continue
    d = {e: ct.get(e, 0) - cp.get(e, 0) for e in set(ct) | set(cp) if ct.get(e, 0) != cp.get(e, 0) and e != "H"}
    dh = ct.get("H", 0) - cp.get("H", 0)
    f = "".join(f"{'+' if v > 0 else '-'}{e}{abs(v) if abs(v) > 1 else ''}" for e, v in sorted(d.items())) + (f"{'+' if dh > 0 else '-'}H{abs(dh)}" if dh else "")
    rows.append(dict(qid=int(r.qid), parent_smiles=st.smiles.values[p], truth_smiles=st.smiles.values[sid], delta_formula=f,
                     tanimoto=round(float(tan[j]), 3), reachable=int(rep.get(int(r.qid), -1))))
df = pd.DataFrame(rows); df.to_csv(ROOT / "results/v4n/c3_delta_pairs.csv", index=False)
g = df[(df.reachable == 0) & (df.tanimoto >= 0.7)]
print(len(df), "queries;", len(g), "unreachable with a parent Tanimoto>=0.7")
print(g.delta_formula.value_counts().head(25).to_string())

import sys, time, json
shims = "--real" not in sys.argv
import pathlib
if not shims: pathlib.PosixPath = pathlib.PurePosixPath
import fm_env; fm_env.setup(shims=shims)
import torch, numpy as np
if not shims:
    import dgl
    _g=dgl.graph
    def _g64(data,*a,**k):
        return _g(tuple(torch.as_tensor(x).long() for x in data),*a,**k)
    dgl.graph=_g64
torch.set_num_threads(4)
import ms_pred.common as common
from ms_pred.iceberg import joint_model
ck = "../../external/forward_model/ckpt/iceberg/"
t=time.time()
m = joint_model.JointModel.from_checkpoints(ck+"gen/best.ckpt", ck+"inten_contr/best.ckpt")
m.eval()
print("load", time.time()-t, "dgl", __import__("dgl").__version__, "rdkit", __import__("rdkit").__version__)
smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1", "O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c12", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "COc1cc(/C=C/C(=O)O)ccc1O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N", "OC[C@H]1O[C@@H](Oc2cc(O)c3c(c2)oc(-c2ccc(O)cc2)cc3=O)[C@H](O)[C@@H](O)[C@@H]1O"]
out={}
t=time.time()
with torch.no_grad():
    r = m.predict_mol(smis, collision_eng=[30.0]*len(smis), precursor_mz=[common.mass_from_smi(s)+common.ion2mass["[M+H]+"] for s in smis],
                      adduct=["[M+H]+"]*len(smis), threshold=0.0, device="cpu", max_nodes=100, instrument=["QTOF"]*len(smis), binned_out=False)
print("predict", time.time()-t)
for s, sp in zip(smis, r["spec"]):
    sp = sp.numpy(); o = np.argsort(-sp[:,1])[:100]; sp = sp[o]
    out[s] = sp.tolist()
    print(s, sp.shape, [(round(a,4), round(b,4)) for a,b in sp[:5]])
json.dump(out, open(sys.argv[-1], "w"))

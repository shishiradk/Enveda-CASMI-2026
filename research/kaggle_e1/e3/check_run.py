"""Check a downloaded E3 kernel output directory against the E1 kernel output (no labels involved).

    python research/kaggle_e1/e3/check_run.py <e3 output dir> <e1 output dir>

Verifies submission.csv (rows, <= 25 unique parseable SMILES per row, no empties), that the pre-re-order lists
equal E1's submission, and recounts the E3-vs-E1 changes on the tautomer-independent list of SMILES strings.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
e3, e1 = Path(sys.argv[1]), Path(sys.argv[2])
sub = pd.read_csv(e3 / "submission.csv")
ref = pd.read_csv(e3 / "e1_order_reference.csv")
old = pd.read_csv(e1 / "submission.csv")
sample = pd.read_csv(Path(__file__).resolve().parents[3] / "sample_submission.csv")
out = {"rows": len(sub), "ids_match_sample": list(sub.molecule_id) == list(sample.molecule_id),
       "columns": list(sub.columns)}
lists = sub.smiles.fillna("").str.split(";")
out["empty_rows"] = int((sub.smiles.fillna("") == "").sum())
out["fallback_CCO_rows"] = int((sub.smiles == "CCO").sum())
out["max_per_row"] = int(lists.map(len).max())
out["min_per_row"] = int(lists.map(len).min())
out["rows_with_25"] = int((lists.map(len) == 25).sum())
out["rows_with_duplicate_smiles"] = int((lists.map(len) != lists.map(lambda v: len(set(v)))).sum())
out["empty_tokens"] = int(lists.map(lambda v: sum(1 for s in v if not s)).sum())
out["unparseable_smiles"] = int(sum(1 for v in lists for s in v if Chem.MolFromSmiles(s) is None))
out["pre_reorder_rows_equal_to_E1_kernel_output"] = int((ref.set_index("molecule_id").smiles == old.set_index("molecule_id").smiles).sum())
a, b = old.set_index("molecule_id").smiles.str.split(";"), sub.set_index("molecule_id").smiles.str.split(";")
out["vs_E1_kernel_output"] = {
    "top1_changed": int(sum(a[m][0] != b[m][0] for m in a.index)),
    "top25_order_changed": int(sum(a[m] != b[m] for m in a.index)),
    "top25_membership_changed": int(sum(set(a[m]) != set(b[m]) for m in a.index)),
    "mean_new_candidates_in_top25": round(float(sum(len(set(b[m]) - set(a[m])) for m in a.index)) / len(a), 2),
}
st = json.load(open(e3 / "fm_stats.json"))
out["fm_status"] = st.get("status")
print(json.dumps(out, indent=1))
print(json.dumps(st, indent=1))

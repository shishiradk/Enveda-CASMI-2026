"""Extract the engine modules embedded in an E1-style notebook into a directory the bench can import.

    python research/bench/extract_engine.py [NOTEBOOK=research/kaggle_e1/casmi_e1.ipynb] [OUT=research/bench/eng]

Writes pv.py, pv_fp.py, casmi_engine.py (the ENG_FILES dict of the engine cell) and eng_runner.py verbatim, with ONE
patch to casmi_engine.py: the hard-coded input root becomes overridable through the CASMI_ROOTS environment variable
(os.pathsep-separated), so worker processes spawned on Windows also find the local copies of the Kaggle datasets.
Nothing else is touched, so the bench always runs the code that is in the notebook.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research/kaggle_e1/casmi_e1.ipynb"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "research/bench/eng"
out.mkdir(parents=True, exist_ok=True)
nb = json.load(open(nb_path, encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"] if "ENG_FILES" in "".join(c["source"]))
files = {}
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ENG_FILES":
        files.update(ast.literal_eval(node.value))
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "write" and node.args \
            and isinstance(node.args[0], ast.Constant) and "casmi_engine as E" in str(node.args[0].value):
        files["eng_runner.py"] = node.args[0].value
OLD = 'ROOTS = [r"C:\\Users\\HW-LEE\\Desktop\\CASMI"] if LOCAL else ["/kaggle/input"]'
NEW = ('ROOTS = (os.environ["CASMI_ROOTS"].split(os.pathsep) if os.environ.get("CASMI_ROOTS")  # BENCH PATCH\n'
       '         else [r"C:\\Users\\HW-LEE\\Desktop\\CASMI"] if LOCAL else ["/kaggle/input"])')
assert files["casmi_engine.py"].count(OLD) == 1, "ROOTS line not found: engine changed, update the patch"
files["casmi_engine.py"] = files["casmi_engine.py"].replace(OLD, NEW)
for name, text in files.items():
    (out / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"{name}: {len(text):,} chars")

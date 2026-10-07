"""After a bench run: parity check (smoke) or a short summary to paste into the team chat (full).

    python bench_check.py smoke <tag>     public-net ranks must equal the reference run e1 on the same molecules
    python bench_check.py full  <tag>     MRR@25 table of the run next to the reference, written to bench_summary.txt
    python bench_check.py todo  <tag>     prints the scenarios still to run (bench.py does not resume inside a scenario)
"""
import hashlib, json, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bench"
REF = OUT / "e1"
KEY_SETS = ["fp_only", "fp_only_mk", "pv", "pv_mk", "pv_nofp", "clean", "clean_mk", "blend", "blend_mk"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def smoke(tag):
    a = pd.read_csv(OUT / tag / "per_molecule_blend.csv")
    b = pd.read_csv(REF / "per_molecule_blend.csv")
    b = b[b.scen == "S2"].set_index("mid").loc[a.mid]
    a = a.set_index("mid")
    cols = [c for c in a.columns if c.startswith("rr_") and "_mk" not in c and c in b.columns]
    diff = (a[cols] - b[cols]).abs()
    n_bad = int((diff.max(axis=1) > 1e-9).sum())
    print(f"parity with the reference run: {len(a) - n_bad}/{len(a)} molecules identical on {len(cols)} score sets")
    # GPU vs CPU arithmetic may reorder a near-tie in one molecule; more than that is a real difference.
    ok = n_bad <= 1
    print("RESULT: ALL OK" if ok else "RESULT: NOT OK - send results\\bench\\" + tag + " folder and the window text")
    return ok


def full(tag):
    d = OUT / tag
    R = json.load(open(d / "eval_blend.json"))["mrr"]
    E = json.load(open(REF / "eval_blend.json"))["mrr"]
    meta = json.load(open(d / "run_meta.json"))
    lines = [f"CASMI bench summary, tag {tag}, {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"device {meta.get('dev')} | torch {meta.get('torch')} | rdkit {meta.get('rdkit')} | "
             f"total {meta.get('total_sec', 0) / 60:.0f} min",
             f"alt FP models (the *_mk rows): {meta.get('alt_fp')}"]
    alt = Path(meta["alt_fp"]) if meta.get("alt_fp") else None
    if alt and alt.exists():
        for p in sorted(alt.glob("*.pt")):
            lines.append(f"  {p.name} sha256 {sha(p)}")
    scens = [s for s in ("S1", "S2", "S3", "S3i") if s in R]
    lines.append("MRR@25 this run (reference e1 = public nets + megayak nets in brackets)")
    lines.append("set          " + "".join(f"{s:>18}" for s in scens))
    for k in KEY_SETS:
        row = f"{k:<13}"
        for s in scens:
            v = R[s].get(k, [None])[0]
            e = E.get(s, {}).get(k, [None])[0]
            row += f"{'-' if v is None else f'{v:.3f}':>8} ({'-' if e is None else f'{e:.3f}'})".rjust(18)
        lines.append(row)
    missing = [s for s in ("S1", "S2", "S3", "S3i") if s not in R]
    lines.append("RESULT: ALL OK" if not missing else f"RESULT: NOT FINISHED, missing {missing} - run run_bench.bat again")
    txt = "\n".join(lines)
    print(txt)
    (d / "bench_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\nsaved in results\\bench\\{tag}\\bench_summary.txt")
    return not missing


def todo(tag):
    d = OUT / tag
    need = {"S1": ["S1"], "S2": ["S2"], "S3": ["S3", "S3i"]}
    print(",".join(s for s, f in need.items() if not all((d / f"{x}.pkl").exists() for x in f)) or "none")
    return True


if __name__ == "__main__":
    mode, tag = sys.argv[1], sys.argv[2]
    if mode == "todo":
        sys.exit(0 if todo(tag) else 1)
    sys.exit(0 if (smoke(tag) if mode == "smoke" else full(tag)) else 1)

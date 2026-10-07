"""Build a licence-clean ChEBI + LIPID MAPS structure table from the primary-source SDF files.

Inputs (downloaded from the primary sites into external/structdb/):
  chebi_lite.sdf.gz   https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi_lite.sdf.gz      (CC BY 4.0)
  LMSD.sdf.zip        https://www.lipidmaps.org/files/?file=LMSD&ext=sdf.zip               (CC BY 4.0)
  fp_bits.npy         kaggle prvsiyan/coconut-casmi26-candidates (CC BY 4.0) - bit selection, only for the
                      optional notebook-layout output.

Outputs (results/structdb/):
  chebi_lipidmaps_clean.parquet   ik, smiles, mass, src, src_id, chebi_id, lm_id, name   (mass-sorted)
  dropin/bio_fp.npy, bio_mass.npy, bio_meta.pkl   same layout the engine's build_pool() reads
  build_stats.json
"""
import gzip, io, json, os, pickle, zipfile, time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXT = os.path.join(ROOT, "external", "structdb")
OUT = os.path.join(ROOT, "results", "structdb")
ALLOWED = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
MASS_LO, MASS_HI = 150.0, 1170.0
WORKERS = 4
CHUNK = 500


def sdf_records(fh):
    """Yield (molblock, {tag: value}) from a text SDF stream."""
    buf = []
    for line in fh:
        if line.startswith("$$$$"):
            txt = "".join(buf); buf = []
            cut = txt.find("M  END")
            if cut < 0: continue
            block = txt[:cut + 6] + "\n"
            tags, key = {}, None
            for l in txt[cut + 6:].splitlines():
                if l.startswith("> <"):
                    key = l[3:l.index(">", 3)]; tags[key] = ""
                elif key is not None and l.strip():
                    tags[key] = (tags[key] + " " + l.strip()).strip()
            yield block, tags
        else:
            buf.append(line)


def chunks(source):
    if source == "chebi":
        fh = gzip.open(os.path.join(EXT, "chebi_lite.sdf.gz"), "rt", encoding="utf-8", errors="replace")
        idk, nmk = "ChEBI ID", "ChEBI NAME"
    else:
        z = zipfile.ZipFile(os.path.join(EXT, "LMSD.sdf.zip"))
        fh = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace")
        idk, nmk = "LM_ID", "NAME"
    cur = []
    for block, tags in sdf_records(fh):
        nm = tags.get(nmk) or tags.get("SYSTEMATIC_NAME") or ""
        cur.append((source, tags.get(idk, ""), nm, block))
        if len(cur) >= CHUNK:
            yield cur; cur = []
    if cur: yield cur
    fh.close()


def process(chunk):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    rows, why = [], Counter()
    for source, sid, name, block in chunk:
        why[source + ":total"] += 1
        try:
            m = Chem.MolFromMolBlock(block)
        except Exception:
            m = None
        if m is None or m.GetNumAtoms() == 0:
            why[source + ":rdkit_parse_fail"] += 1; continue
        if len(Chem.GetMolFrags(m)) != 1:
            why[source + ":multi_component"] += 1; continue
        if any(a.GetSymbol() not in ALLOWED for a in m.GetAtoms()):
            why[source + ":element"] += 1; continue
        if sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0:
            why[source + ":charged"] += 1; continue
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            why[source + ":radical"] += 1; continue
        if any(a.GetIsotope() for a in m.GetAtoms()):
            why[source + ":isotope_labelled"] += 1; continue
        try:
            mass = float(ExactMolWt(m))
            if not (MASS_LO <= mass <= MASS_HI):
                why[source + ":mass_range"] += 1; continue
            ik = Chem.MolToInchiKey(m)
            m2 = Chem.Mol(m); Chem.RemoveStereochemistry(m2)
            smi = Chem.MolToSmiles(m2)
            if not ik or not smi or Chem.MolFromSmiles(smi) is None:
                why[source + ":key_fail"] += 1; continue
        except Exception:
            why[source + ":key_fail"] += 1; continue
        why[source + ":kept"] += 1
        rows.append((ik[:14], smi, mass, source, sid, name))
    return rows, why


_g = {}


def fp_row(smi):
    """6,930-bit packed fingerprint, same recipe as the engine's fp_and_mass()."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    if not _g:
        RDLogger.DisableLog("rdApp.*")
        _g["bits"] = np.load(os.path.join(EXT, "fp_bits.npy"))
        _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    m = Chem.MolFromSmiles(smi)
    fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                         _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                         _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                         np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]
    return np.packbits(fp)


def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "dropin"), exist_ok=True)
    rows, why = [], Counter()
    with Pool(WORKERS) as mp:
        for source in ("chebi", "lipidmaps"):
            for r, w in mp.imap(process, chunks(source), chunksize=1):
                rows += r; why.update(w)
            print(source, "done", len(rows), f"{time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows, columns=["ik", "smiles", "mass", "source", "sid", "name"])
        per_src = {s: int(df[df.source == s].ik.nunique()) for s in ("chebi", "lipidmaps")}
        # one row per InChIKey-14: representative = ChEBI record if any, else LIPID MAPS; lowest id wins
        df["pri"] = (df.source != "chebi").astype(int)
        df = df.sort_values(["ik", "pri", "sid"], kind="stable")
        ids = df.groupby(["ik", "source"]).sid.first().unstack().reindex(columns=["chebi", "lipidmaps"])
        rep = df.drop_duplicates("ik").set_index("ik")
        ids = ids.reindex(rep.index)
        has_c, has_l = ids["chebi"].notna().values, ids["lipidmaps"].notna().values
        out = pd.DataFrame({
            "ik": rep.index, "smiles": rep.smiles.values, "mass": rep.mass.values,
            "src": np.where(has_c & has_l, "chebi|lipidmaps", np.where(has_c, "chebi", "lipidmaps")),
            "src_id": rep.sid.values,
            "chebi_id": ids["chebi"].values,
            "lm_id": ids["lipidmaps"].values,
            "name": rep.name.values,
        }).sort_values(["mass", "ik"], kind="stable").reset_index(drop=True)
        out.to_parquet(os.path.join(OUT, "chebi_lipidmaps_clean.parquet"), index=False, compression="zstd")
        print("table", len(out), flush=True)
        nbits = None
        if os.path.exists(os.path.join(EXT, "fp_bits.npy")):
            nbits = int(len(np.load(os.path.join(EXT, "fp_bits.npy"))))
            fp = np.vstack(mp.map(fp_row, out.smiles.tolist(), chunksize=256))
            np.save(os.path.join(OUT, "dropin", "bio_fp.npy"), fp)
            np.save(os.path.join(OUT, "dropin", "bio_mass.npy"), out.mass.values.astype(np.float64))
            with open(os.path.join(OUT, "dropin", "bio_meta.pkl"), "wb") as f:
                pickle.dump({"keys": out.ik.values.astype(object), "smiles": out.smiles.values.astype(object),
                             "nbits": nbits}, f, protocol=4)
            print("dropin", fp.shape, flush=True)
    stats = {"filters": dict(sorted(why.items())), "unique_ik_per_source_after_filter": per_src,
             "rows_final": int(len(out)), "src_counts": out.src.value_counts().to_dict(),
             "mass_min": float(out.mass.min()), "mass_max": float(out.mass.max()), "nbits": nbits,
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(OUT, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

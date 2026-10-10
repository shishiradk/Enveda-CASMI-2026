"""Chemistry core of the v4r engine (our code, MIT).

* Adducts: parsed generically from the adduct string ("[2M+Na]+", "[M-2H2O+H]+", "[M+CH2O2-H]-", ...) instead of a
  hand-written table, so every adduct that occurs in train/test gets a neutral mass.  Atom masses come from the RDKit
  periodic table (the same numbers ExactMolWt uses for pool masses), electron mass from CODATA.
* Score key = the competition metric key: RDKit TautomerEnumerator().Canonicalize -> MolToInchiKey -> first 14 chars
  (identical definition to research/bench/eng/casmi_engine.canon_key).
* Standardiser: largest fragment, uncharge, remove stereo, canonical SMILES; None if the molecule stays charged.
* featurize(): ONE canonicalisation per structure gives the score key AND the tautomer-canonical molecule, which is the
  only form fingerprints and fragments are ever computed from (REBUILD_SPEC 3.0 provenance precaution).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

ELECTRON = 0.000548579909
ELEMS10 = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")      # element-vector order (= CFT element head order)

# ------------------------------------------------------------------------------------------- element masses
_MASS: Dict[str, float] = {}


def mass_of(el: str) -> float:
    if el not in _MASS:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        if el == "D":
            _MASS[el] = 2.01410177812
        else:
            _MASS[el] = float(pt.GetMostCommonIsotopeMass(el))
    return _MASS[el]


H_MASS = 1.00782503223          # == RDKit most-common-isotope mass of H (checked in validate.v0)
PROTON = H_MASS - ELECTRON

_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(f) -> Optional[Dict[str, int]]:
    """'C10H12N2O' -> {'C': 10, ...}; None for anything that is not a plain element/count string."""
    if not isinstance(f, str) or not f:
        return None
    f = f.replace("+", "").replace("-", "")
    out: Dict[str, int] = {}
    pos = 0
    for m in _TOK.finditer(f):
        if m.start() != pos:
            return None
        pos = m.end()
        el = m.group(1)
        try:
            mass_of(el)
        except Exception:
            return None
        out[el] = out.get(el, 0) + (int(m.group(2)) if m.group(2) else 1)
    return out if pos == len(f) and out else None


def formula_mass(counts: Dict[str, int]) -> float:
    return float(sum(mass_of(e) * n for e, n in counts.items()))


def formula_vec(f) -> Optional[np.ndarray]:
    """Counts of the 10 elements C H N O P S F Cl Br I; None if the formula has any other element."""
    c = parse_formula(f)
    if c is None or any(e not in ELEMS10 for e in c):
        return None
    return np.array([c.get(e, 0) for e in ELEMS10], np.int64)


# ------------------------------------------------------------------------------------------- adducts
_ADDUCT_RE = re.compile(r"^\[(\d*)M((?:[+-]\d*[A-Za-z0-9]+)*)\](\d*)([+-])$")
_TERM_RE = re.compile(r"([+-])(\d*)([A-Z][A-Za-z0-9]*)")
_ALIAS = {"HCOO": "CHO2", "CH3COO": "C2H3O2", "CF3COO": "C2F3O2"}


def parse_adduct(a) -> Optional[Tuple[int, int, float, int]]:
    """'[2M+Na]+' -> (n_M, |z|, delta, polarity) with  m/z * |z| = n_M * M + delta  (electrons inside delta)."""
    if not isinstance(a, str):
        return None
    m = _ADDUCT_RE.match(a.strip())
    if m is None:
        return None
    n = int(m.group(1)) if m.group(1) else 1
    z = int(m.group(3)) if m.group(3) else 1
    pol = 1 if m.group(4) == "+" else -1
    delta = 0.0
    terms = m.group(2)
    pos = 0
    for t in _TERM_RE.finditer(terms):
        if t.start() != pos:
            return None
        pos = t.end()
        f = _ALIAS.get(t.group(3), t.group(3))
        c = parse_formula(f)
        if c is None:
            return None
        k = int(t.group(2)) if t.group(2) else 1
        delta += (1 if t.group(1) == "+" else -1) * k * formula_mass(c)
    if pos != len(terms):
        return None
    return n, z, delta - pol * z * ELECTRON, pol


TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
# Our adduct index: the ten test adducts, then the frequent train adducts; everything else -> '<unk>'.
# Only equality of indices is used by the engine (lib_same_adduct); the FP model receives the adduct string.
ADDUCT_LIST = TEST_ADDUCTS + [
    "[2M+Na]+", "[2M+H]+", "[2M-H]-", "[M]+", "[M+C2H4O2-H]-", "[M+2H]2+", "[2M+Na-2H]-", "[2M+CH2O2-H]-",
    "[M-H2O]+", "[M-CH3]-", "[M-2H]-", "[M]-", "[2M+NH4]+", "[M-H5O3]+", "[2M+C2H4O2-H]-", "[M+C2H4N]+",
    "[M+3H]3+", "[M+2Na-H]+", "[2M+K]+", "[M-H2]2-", "[M+Ca]2+", "[M-C3H6NO2]-", "[M+CH5O]+", "[M-H2N]+",
    "[M+Br]-", "[M+HNa]2+", "[M+Li]+", "[M+Na-2H]-", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
UNK_ADDUCT = ADDUCT_IX["<unk>"]


def adduct_index(a) -> int:
    return ADDUCT_IX.get(a, UNK_ADDUCT)


_AD_CACHE: Dict[str, Optional[Tuple[int, int, float, int]]] = {}


def adduct_info(a):
    if a not in _AD_CACHE:
        _AD_CACHE[a] = parse_adduct(a)
    return _AD_CACHE[a]


def neutral_mass(mz, adduct) -> np.ndarray:
    """Neutral monoisotopic mass of M from precursor m/z and adduct string(s); NaN if the adduct cannot be parsed."""
    mz = np.atleast_1d(np.asarray(mz, np.float64))
    ad = np.atleast_1d(np.asarray(adduct, dtype=object))
    out = np.full(len(mz), np.nan)
    for a in set(ad.tolist()):
        info = adduct_info(a)
        if info is None:
            continue
        n, z, d, _ = info
        sel = ad == a
        out[sel] = (mz[sel] * z - d) / n
    return out


def precursor_mz(mass: float, adduct: str) -> float:
    n, z, d, _ = adduct_info(adduct)
    return (n * mass + d) / z


# ------------------------------------------------------------------------------------------- RDKit helpers
_rd: dict = {}


def rd():
    if not _rd:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        _rd.update(Chem=Chem, Descriptors=Descriptors, rdMD=rdMolDescriptors,
                   taut=rdMolStandardize.TautomerEnumerator(), lfc=rdMolStandardize.LargestFragmentChooser(),
                   unch=rdMolStandardize.Uncharger())
    return _rd


def score_key(smiles) -> Optional[str]:
    """Competition metric key (tautomer-canonical InChIKey first block)."""
    r = rd()
    try:
        m = r["Chem"].MolFromSmiles(smiles)
        if m is None:
            return None
        ik = r["Chem"].MolToInchiKey(r["taut"].Canonicalize(m))
        return ik[:14] if ik else None
    except Exception:
        return None


def raw_key(smiles) -> Optional[str]:
    r = rd()
    try:
        m = r["Chem"].MolFromSmiles(smiles)
        ik = r["Chem"].MolToInchiKey(m) if m is not None else ""
        return ik[:14] if ik else None
    except Exception:
        return None


def standardize(smiles, keep_charged: bool = False) -> Optional[str]:
    r = rd()
    Chem = r["Chem"]
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = r["unch"].uncharge(r["lfc"].choose(m))
        if not keep_charged and Chem.GetFormalCharge(m) != 0:
            return None
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def mol_props(mol) -> Tuple[str, float, int]:
    r = rd()
    return r["rdMD"].CalcMolFormula(mol), float(r["Descriptors"].ExactMolWt(mol)), int(mol.GetNumHeavyAtoms())


def canonical(smiles):
    """(mol, tautomer-canonical mol, score key) or None. One Canonicalize call."""
    r = rd()
    Chem = r["Chem"]
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        tm = r["taut"].Canonicalize(m)
        ik = Chem.MolToInchiKey(tm)
        return m, tm, (ik[:14] if ik else None)
    except Exception:
        return None


def featurize(smiles, fpr=None, frags: bool = True, max_depth: int = 2):
    """Everything the tables need for one (already standardised) structure SMILES, or None.
    Returns dict(key, smiles_tc, formula, mass, n_heavy, fp_raw (packed cfp1 raw, uint8[2048]), frags float32)."""
    from . import fingerprint, fragments
    c = canonical(smiles)
    if c is None or c[2] is None:
        return None
    m, tm, key = c
    try:
        formula, mass, nh = mol_props(m)
        fpr = fpr or fingerprint.raw_fingerprinter()
        fp = np.packbits(fpr.raw(tm))
        smi_tc = rd()["Chem"].MolToSmiles(tm)
        fr = fragments.fragments_of_mol(tm, max_depth) if frags else None
    except Exception:
        return None
    return dict(key=key, smiles_tc=smi_tc, formula=formula, mass=mass, n_heavy=nh, fp_raw=fp, frags=fr)

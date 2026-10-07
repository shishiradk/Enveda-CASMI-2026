"""biotransform: a Class-3 generator that applies hand-written biotransformation rules to (A) library analogs,
(B) mass-shifted database neighbours of the target and (C) same-mass window structures, then ranks the products by
the leak-free spectrum->fingerprint logits (zlog) plus a rule-path prior.

Design: research/c3gen/designs/biotransform.md (deviations are listed in DEVIATIONS below).

Interface (common C3 generator interface):
    generate(ctx, k=200, forbidden=frozenset()) -> [(smiles, score, provenance), ...] best first, <= k items
ctx keys: mid, target (neutral monoisotopic mass), adducts, zlog (float32[6930]), analogs (list of
{smiles, ik14, sim, ...} best first, truth removed), window (list of {ik, smiles, fpscore|score}, truth removed),
spectra (list of {mz, intensity, precursor_mz, adduct}; unused by this generator).

Rules of the interface honoured here:
  * every output is sanitizable and within 10 ppm of ctx['target'] (RDKit Descriptors.ExactMolWt, re-checked);
  * database INPUTS (analogs, window, pool seeds) are filtered by `forbidden` (plain and tautomer keys as given);
    outputs are NOT filtered by it;
  * deterministic, one core, <= ~20 s per molecule for k=200 (hard time guards inside).

CLI (C3NP bench):
    python research/c3gen/biotransform.py --c3np --n 60 --offset 0 --workers 2 --out results/c3gen/biotransform_c3np.json
writes {mid: [smiles, ...]} for research/scripts/c3np_eval.py plus <out>_summary.json (hit@1/@25/@200 by
tautomer-canonical key, mean candidates, s/molecule, mass-valid share). --dump also writes the scored component table
(<out>_dump.pkl) for offline weight tuning. --strip {tc085,tc070,edit1} removes the truth's congeners from the seed
sources (research/scripts/c3np_strip.py; ctx['exclude'] is a seed-only exclusion); the summary also reports
by-subset scores without chain-homolog hits (+-CH2 / +-C2H4 on chains). Bench MRR is not a Class-3 forecast.

Assets (read-only): results/train_pkg/data/{pool_mass,pool_fp,pool_key,pool_smiles.txt,fp_bits}.npy (COCONUT + train
structures, sorted by mass, 6930-bit packed fingerprints of the zlog recipe) and results/c3gen/np_pool.parquet
(only its ik14 column: restricts pool seeds to the natural-product-like subset). A boolean mask is cached at
results/c3gen/work/biotx_pool_np.npy.

DEVIATIONS from the design (see the report for reasons):
  * no pool tautomer-key build: seeds are checked against `forbidden` by plain key (the forbidden table already lists
    raw aliases of every tautomer alias found in COCONUT / PubChem / the engine pool) and by ckey when the input has one;
  * no shift localisation (ModiFinder-like) and no mined rule priors / fitted a,b,c,d: priors are hand-set per rule;
  * stage 1 uses both Morgan blocks (r2 + r3) of the zlog recipe, stage 2 the full recipe on the top N2;
  * two-step paths only from the top 20 analog seeds (A) and the window relocation seeds (C); pool seeds (B) 1 step;
  * zlog is z-scored per molecule before adding the prior / seed / multiplicity terms (weights W, chosen on the C3NP
    rows 0-59 and 300-359 smoke dumps, i.e. in-sample there; rows 120-179 / 420-479 were held out);
  * extra rules beyond the design list: chain +-CH2 / +-C2H4, aliphatic C-methyl, +-OMe on arenes, +-CO2,
    flavone/coumarin C=C hydrogenation (pseudo-aromatic ring), disaccharides, apiose / furanose, angeloyl-type acyls;
  * window seeds are dropped from the output by plain key (design section 9), and so are all seed structures;
  * isotope-labelled database seeds are skipped.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CASMI_ROOT", Path(__file__).resolve().parents[2]))
TD = ROOT / "results" / "train_pkg" / "data"
NP_POOL = ROOT / "results" / "c3gen" / "np_pool.parquet"
NP_MASK = ROOT / "results" / "c3gen" / "work" / "biotx_pool_np.npy"
DEPLOY_DIR = None      # set_asset_dir(): deploy mode (Kaggle), every asset from one directory, no repo paths


def set_asset_dir(d):
    """Deploy mode: read every asset from directory d, named with a 'c3_' prefix so no other Kaggle input's file-name
    lookups can collide: c3_pool_mass.npy, c3_pool_fp.npy, c3_pool_key.npy, c3_pool_smiles.txt (the full train_pkg
    pool: COCONUT + all train structures, no held-key exclusions), c3_fp_bits.npy, c3_pool_np.npy (NP-like mask over
    the pool rows). Nothing is written. Call before the first generate()."""
    global DEPLOY_DIR
    DEPLOY_DIR = Path(d)
    for k in ("pool_mass", "pool_fp", "pool_key", "pool_np", "pool_smi_off", "pool_smi_raw"):
        _G.pop(k, None)
    if "Chem" in _G:
        _G["bits"] = np.load(_ap("fp_bits.npy"))


def _ap(name):
    """Path of asset `name` (bench: results/train_pkg/data/<name>; deploy: <DEPLOY_DIR>/c3_<name>)."""
    return DEPLOY_DIR / ("c3_" + name) if DEPLOY_DIR is not None else TD / name

PPM = 10.0
N_SEED_A, MIN_SIM_A, N_2STEP_A = 30, 0.2, 20
N_SEED_B_PER_D = 6
N_SEED_C = 10
CAP_SEED, CAP_TOTAL = 400, 6000
N_STAGE2 = 400
# Deterministic work budgets are the real limits (E7: they replace the old 9 s / 17 s wall-clock guards, which fired on
# 6/550 C3NP molecules and made those rows machine-speed dependent). Units: heavy atoms touched.
#   W_GEN : generation   = sum over expand() calls of heavy(seed) * (1 + rules tried) + heavy(product) per product
#                          (+ 3 * heavy(seed) + heavy(fragment) for hydrolysis)
#   W_S2  : stage 2      = sum heavy(product) ** 2.5 over the full-recipe rescored products
# Calibrated on the unlimited 550-molecule calibration run (results/c3gen/e7/bt_calib_nolimit_dump.pkl), which
# truncated 0/550: observed max W_GEN 1,047,477, max W_S2 22,405,392, max t_gen 20.2 s, max t_total 39.0 s. The
# budgets below sit ~1.9x above those maxima, so they bind only on a pathological input, never on a bench row: that is
# what makes the output independent of machine speed. The first attempt (W_GEN 1.6e5 / W_S2 3.0e6) truncated 519/550
# and 74/550 and cost 0.159 MRR@25 on rows 0-59 (0.7335 -> 0.5746); do not tighten these without re-measuring.
# The wall-clock limits below are only a hard failsafe for an overloaded machine (~4x / ~3x the observed maxima).
# When one fires, info['guard'] records it and every output provenance gets a '|guard=<gen|all>' suffix; if T_TOTAL
# fires during stage 2, the WHOLE list falls back to the stage-1 order (no partial mixing; deterministic unless
# T_GEN also fired).
W_GEN = 2.0e6
W_S2 = 4.0e7
T_GEN, T_TOTAL = 90.0, 120.0         # seconds: failsafes only (never reached on the C3NP bench, see E7 report)
# final score = (zfull - max zfull) / sd(zfull)   [per molecule, over the stage-2 scored products]
#               + W_PRIOR * path prior + W_SEED * seed term + W_MULT * log(n distinct seeds) - W_STEP * (steps - 1)
# Weights picked on the C3NP rows 0-59 dump (flat optimum region; see the report), not fitted further.
W = dict(prior=1.0, seed=2.0, mult=2.0, step=0.25)
if os.environ.get("BIOTX_CFG"):          # experiment overrides, e.g. BIOTX_CFG='{"N_SEED_B_PER_D": 5}'
    for _k, _v in json.loads(os.environ["BIOTX_CFG"]).items():
        if _k == "W":
            W.update(_v)
        else:
            globals()[_k] = _v

AM = {"C": 12.0, "H": 1.00782503207, "O": 15.99491461956, "N": 14.0030740048, "S": 31.97207100, "P": 30.97376163}
_G = {}


def fmass(f):
    """'C6H10O5' -> mass; '-CH2' -> negative; 'O-H2' handled as O + (-H2)."""
    import re
    tot, sign = 0.0, 1
    for tok in re.findall(r"[+-]|[A-Z][a-z]?\d*", f):
        if tok in "+-":
            sign = 1 if tok == "+" else -1
            continue
        el, n = re.match(r"([A-Z][a-z]?)(\d*)", tok).groups()
        tot += sign * AM[el] * (int(n) if n else 1)
    return tot


# ---------------------------------------------------------------------------------------------------------- RDKit
def _rd():
    if "Chem" not in _G:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
        from rdkit.Chem.Descriptors import ExactMolWt
        RDLogger.DisableLog("rdApp.*")
        _G.update(Chem=Chem, MACCS=MACCSkeys, EMW=ExactMolWt, BT=Chem.BondType,
                  m2=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096),
                  m3=rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096),
                  rk=rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6))
        _G["bits"] = np.load(_ap("fp_bits.npy"))
        _init_rules()
    return _G["Chem"]


def normalize(m):
    """Stereo-free copy whose neutral non-aromatic-N atoms carry implicit H only (so edits re-derive H counts).
    Falls back to the stereo-free original when that would change any atom's H count."""
    Chem = _rd()
    if any(a.GetIsotope() for a in m.GetAtoms()):      # isotope-labelled database entries are not natural products
        return None
    m = Chem.Mol(m)
    Chem.RemoveStereochemistry(m)
    before = [a.GetTotalNumHs() for a in m.GetAtoms()]
    w = Chem.Mol(m)
    for a in w.GetAtoms():
        if a.GetFormalCharge() == 0 and a.GetNumRadicalElectrons() == 0 and not (
                a.GetIsAromatic() and a.GetSymbol() in ("N", "P")):
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(w)
        if [a.GetTotalNumHs() for a in w.GetAtoms()] == before:
            return w
    except Exception:
        pass
    return m


def _lose_h(a):
    n = a.GetNumExplicitHs()
    if n > 0:
        a.SetNumExplicitHs(n - 1)


def _gain_h(a):
    if a.GetNoImplicit() or (a.GetIsAromatic() and a.GetSymbol() != "C"):
        a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)


def _fin(rw):
    Chem = _G["Chem"]
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def _group(smi):
    Chem = _G["Chem"]
    g = Chem.MolFromSmiles(smi)
    d = [a.GetIdx() for a in g.GetAtoms() if a.GetAtomicNum() == 0][0]
    nb = g.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx()
    return g, d, nb


def _attach(m, site, grp):
    Chem = _G["Chem"]
    g, d, nb = grp
    off = m.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(m, g))
    rw.AddBond(site, off + nb, _G["BT"].SINGLE)
    rw.RemoveAtom(off + d)
    _lose_h(rw.GetAtomWithIdx(site))
    return _fin(rw)


# ---------------------------------------------------------------------------------------------------------- rules
# Each rule: name, delta formula, prior (log scale, 0 = most common step), kind, params.
# kinds: add (site SMARTS atom 0 must carry H, group SMILES with [*]), remove (SMARTS, atom 0 kept, rest deleted),
# bond (SMARTS 2 atoms -> new order), custom (function name).
OH_SITE = "[OX2H1;$(O-[#6]);!$(O-C=O)]"            # alcohol / phenol OH (not acid)
OH_ANY = "[OX2H1;$(O-[#6])]"                          # incl. acid OH
GROUPS = {
    "Hex": "[*]C1OC(CO)C(O)C(O)C1O", "dHex": "[*]C1OC(C)C(O)C(O)C1O", "Pen": "[*]C1OCC(O)C(O)C1O",
    "PenF": "[*]C1OC(CO)C(O)C1O", "Api": "[*]C1OCC(O)(CO)C1O", "GlcA": "[*]C1OC(C(=O)O)C(O)C(O)C1O",
    "Hex6Hex": "[*]C1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O",
    "Hex2Hex": "[*]C1OC(CO)C(O)C(O)C1OC2OC(CO)C(O)C(O)C2O",
    "Rut": "[*]C1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O", "NeoHsp": "[*]C1OC(CO)C(O)C(O)C1OC2OC(C)C(O)C(O)C2O",
    "Hex6Pen": "[*]C1OC(COC2OCC(O)C(O)C2O)C(O)C(O)C1O",
    "Ac": "[*]C(C)=O", "Mal": "[*]C(=O)CC(=O)O", "Cou": "[*]C(=O)C=Cc1ccc(O)cc1", "Caf": "[*]C(=O)C=Cc1ccc(O)c(O)c1",
    "Fer": "[*]C(=O)C=Cc1ccc(O)c(OC)c1", "Sin": "[*]C(=O)C=Cc1cc(OC)c(O)c(OC)c1", "Gal": "[*]C(=O)c1cc(O)c(O)c(O)c1",
    "Bz": "[*]C(=O)c1ccccc1", "Tig": "[*]C(=O)C(C)=CC", "iVal": "[*]C(=O)CC(C)C", "MeBu": "[*]C(=O)C(C)CC",
    "Et": "[*]CC", "COOH": "[*]C(=O)O", "OMe": "[*]OC",
    "Prenyl": "[*]CC=C(C)C", "Geranyl": "[*]CC=C(C)CCC=C(C)C", "Me": "[*]C", "OH": "[*]O", "SO3": "[*]S(=O)(=O)O",
}
GROUP_FORMULA = {"Hex": "C6H10O5", "dHex": "C6H10O4", "Pen": "C5H8O4", "PenF": "C5H8O4", "Api": "C5H8O4",
                 "GlcA": "C6H8O6", "Hex6Hex": "C12H20O10", "Hex2Hex": "C12H20O10", "Rut": "C12H20O9",
                 "NeoHsp": "C12H20O9", "Hex6Pen": "C11H18O9", "Ac": "C2H2O", "Mal": "C3H2O3", "Cou": "C9H6O2",
                 "Caf": "C9H6O3", "Fer": "C10H8O3", "Sin": "C11H10O4", "Gal": "C7H4O4", "Bz": "C7H4O",
                 "Tig": "C5H6O", "iVal": "C5H8O", "MeBu": "C5H8O", "Prenyl": "C5H8", "Geranyl": "C10H16",
                 "Me": "CH2", "OH": "O", "SO3": "SO3", "Et": "C2H4", "COOH": "CO2", "OMe": "CH2O"}

RULES = [
    # name, delta, prior, kind, params
    ("+OH@ar", "O", 0.0, "add", ("[c;!H0]", "OH")),
    ("+OH@al", "O", -0.7, "add", ("[CX4;!H0;!$(C-[!#6;!#1])]", "OH")),
    ("+O@N-oxide", "O", -2.5, "custom", "noxide"),
    ("+O@epoxide", "O", -2.5, "custom", "epoxide"),
    ("-OH", "-O", -0.3, "remove", "[#6;!$(C=O):1]-[OX2H1]"),
    ("+Me@O", "CH2", 0.0, "add", (OH_ANY, "Me")),
    ("+Me@N", "CH2", -0.7, "add", ("[N,n;!H0;!$(N-C=O)]", "Me")),
    ("+Me@c", "CH2", -1.5, "add", ("[c;!H0]", "Me")),
    ("-Me@O", "-CH2", 0.0, "remove", "[O:1]-[CH3]"),
    ("-Me@N", "-CH2", -0.7, "remove", "[N,n:1]-[CH3]"),
    ("-Me@c", "-CH2", -1.5, "remove", "[c:1]-[CH3]"),
    ("+H2@C=C", "H2", -0.5, "bond", ("[C:1]=[C:2]", 1)),
    ("+CH2@chain", "CH2", -0.8, "add", ("[CH3;$(C-[CX4,CX3;#6])]", "Me")),
    ("+Me@C", "CH2", -1.5, "add", ("[CX4;H1,H2;!$(C-[!#6;!#1])]", "Me")),
    ("-CH2@chain", "-CH2", -0.8, "remove", "[CX4,CX3;#6:1]-[CH3]"),
    ("+C2H4@chain", "C2H4", -1.2, "add", ("[CH3;$(C-[CX4,CX3;#6])]", "Et")),
    ("+Et@O,N", "C2H4", -1.5, "add", ("[OX2H1,NX3;!H0;$(*-[#6])]", "Et")),
    ("-C2H4", "-C2H4", -1.2, "remove", "[#6,O,N:1]-[CH2][CH3]"),
    ("+OMe@c", "CH2O", -0.8, "add", ("[c;!H0]", "OMe")),
    ("-OMe@c", "-CH2O", -0.8, "remove", "[c:1]-[OX2][CH3]"),
    ("+CO2@c", "CO2", -1.8, "add", ("[c;!H0]", "COOH")),
    ("+CO2@Ca", "CO2", -1.8, "add", ("[CX4;!H0;$(C-N)]", "COOH")),
    ("-CO2", "-CO2", -1.0, "remove", "[#6:1]-[CX3](=O)[OX2H1]"),
    ("+H2@pyrone", "H2", -0.8, "custom", "dearom"),
    ("+H2@C=O", "H2", -0.7, "bond", ("[CX3;$(C([#6])([#6])=O),$([CH1]([#6])=O):1]=[O:2]", 1)),
    ("-H2@CC", "-H2", -0.8, "bond", ("[CX4;!H0:1]-[CX4;!H0:2]", 2)),
    ("-H2@CHOH", "-H2", -0.7, "bond", ("[CX4;!H0:1]-[OX2H1:2]", 2)),
    ("-H2O", "-H2O", -0.8, "custom", "dehydrate"),
    ("+H2O", "H2O", -0.8, "custom", "hydrate"),
    ("+O-H2@CH2OH>COOH", "O-H2", -1.5, "custom", "acid"),
    ("-H2O@lactone", "-H2O", -1.0, "custom", "lactone"),
    ("+C@OCH2O", "C", -1.0, "custom", "mdo_catechol"),
    ("-H2@OCH2O", "-H2", -1.0, "custom", "mdo_guaiacol"),
    ("-H2@chromene", "-H2", -1.5, "custom", "chromene"),
    ("0@chroman", "", -2.0, "custom", "chroman"),
    ("-prenyl", "-C5H8", -1.0, "remove", "[#6,#8:1]-[CH2][CH1]=[CX3]([CH3])[CH3]"),
    ("-SO3", "-SO3", -1.0, "remove", "[O:1]-S(=O)(=O)[OX2H1,OX1-]"),
    ("+Prenyl@c", "C5H8", -1.0, "add", ("[c;!H0]", "Prenyl")),
    ("+Prenyl@O", "C5H8", -1.3, "add", ("[OX2H1;$(O-c)]", "Prenyl")),
    ("+Geranyl@c", "C10H16", -2.0, "add", ("[c;!H0]", "Geranyl")),
    ("+SO3@O", "SO3", -1.5, "add", (OH_SITE, "SO3")),
    ("+Hex@C-glyc", "C6H10O5", -1.5, "add", ("[c;!H0]", "Hex")),
] + [("+%s@O" % g, GROUP_FORMULA[g], p, "add", (OH_ANY if g in ("Hex",) else OH_SITE, g)) for g, p in [
    ("Hex", 0.0), ("dHex", -0.7), ("Pen", -0.7), ("PenF", -1.5), ("Api", -1.5), ("GlcA", -0.7),
    ("Hex6Hex", -1.2), ("Hex2Hex", -1.5), ("Rut", -1.0), ("NeoHsp", -1.5), ("Hex6Pen", -1.8),
    ("Ac", -1.0), ("Mal", -1.2), ("Cou", -1.0), ("Caf", -1.0), ("Fer", -1.0), ("Sin", -1.5), ("Gal", -1.0),
    ("Bz", -1.5), ("Tig", -1.5), ("iVal", -1.8), ("MeBu", -1.8)]] + [
    ("+%s@N" % g, GROUP_FORMULA[g], p, "add", ("[NX3;!H0;!$(N-C=O);!$(N-S=O)]", g)) for g, p in [
        ("Ac", -1.2), ("Cou", -1.5), ("Caf", -1.8), ("Fer", -1.5)]]
HYDROLYSIS_PRIOR = 0.0   # generic: ester / amide / glycoside / sulfate cleavage, keeping either fragment
# seed-B removal deltas (truth = pool structure minus a group, done by hydrolysis)
B_REMOVE = ["Hex", "dHex", "Pen", "GlcA", "Hex6Hex", "Rut", "Hex6Pen", "Ac", "Mal", "Cou", "Caf", "Fer", "Sin",
            "Gal", "Bz", "Tig", "iVal", "SO3"]
H2O = fmass("H2O")


def _init_rules():
    Chem = _G["Chem"]
    rules = []
    for name, df, prior, kind, par in RULES:
        d = fmass(df) if df else 0.0
        r = dict(name=name, delta=d, prior=prior, kind=kind)
        if kind == "add":
            r["patt"] = Chem.MolFromSmarts(par[0])
            r["grp"] = _group(GROUPS[par[1]])
        elif kind == "remove":
            r["patt"] = Chem.MolFromSmarts(par)
        elif kind == "bond":
            r["patt"] = Chem.MolFromSmarts(par[0])
            r["order"] = {1: _G["BT"].SINGLE, 2: _G["BT"].DOUBLE}[par[1]]
        else:
            r["fn"] = par
        assert r.get("patt", 1) is not None, name
        rules.append(r)
    _G["rules"] = rules
    _G["deltas"] = np.array([r["delta"] for r in rules])
    P = {k: Chem.MolFromSmarts(v) for k, v in dict(
        hyd_acyl="[CX3:1](=O)-[$([OX2;D2]),$([NX3;D2,D3]):2]", hyd_glyc="[C;R:1](-[O;R])-[O,N;!R:2]-[#6]",
        hyd_sulf="[#6]-[O:2]-[S,P:1](=O)", noxide="[NX3;H0;!$(N-C=[O,N,S]);!$(N-a);!$(N-[!#6])]",
        noxide_ar="[nX2;r6]", alkene="[C:1]=[C:2]", dehyd="[OX2H1]-[CX4:1]-[CX4;!H0:2]",
        acid="[CX4H2;$(C-[#6]):1]-[OX2H1]", lac_acid="[CX3:1](=O)-[OX2H1]",
        lac_oh="[OX2H1;$(O-[#6;!$(C=O)])]", cat="[OX2H1]-c:c-[OX2H1]", gua="[OX2H1]-c:c-[OX2]-[CH3]",
        chrom="[OX2H1]-c:c-[CH2]-[CH1]=[CX3]([CH3])[CH3]").items()}
    _G["P"] = P


def _sites(m, patt, ranks, ordered=False, need_h=None):
    """Matches deduplicated by atom symmetry class (ranks of the matched atoms)."""
    seen, out = set(), []
    for mt in m.GetSubstructMatches(patt, uniquify=True, maxMatches=2000):
        if need_h is not None and m.GetAtomWithIdx(mt[need_h]).GetTotalNumHs() < 1:
            continue
        key = tuple(ranks[i] for i in mt)
        if not ordered:
            key = tuple(sorted(key))
        if key in seen:
            continue
        seen.add(key)
        out.append(mt)
    return out


def _apply_fixed(m, r, ranks):
    """Yield product Mols of fixed-delta rule r at every symmetry-unique site of m."""
    Chem, BT = _G["Chem"], _G["BT"]
    k = r["kind"]
    if k == "add":
        for mt in _sites(m, r["patt"], ranks, need_h=0):
            p = _attach(m, mt[0], r["grp"])
            if p is not None:
                yield p, mt[0]
    elif k == "remove":
        for mt in _sites(m, r["patt"], ranks, ordered=True):
            keep, dele = mt[0], sorted(mt[1:], reverse=True)
            ds = set(mt[1:])
            if any(nb.GetIdx() not in ds and nb.GetIdx() != keep
                   for i in mt[1:] for nb in m.GetAtomWithIdx(i).GetNeighbors()):
                continue
            rw = Chem.RWMol(m)
            for i in dele:
                rw.RemoveAtom(i)
            _gain_h(rw.GetAtomWithIdx(keep))
            p = _fin(rw)
            if p is not None:
                yield p, keep
    elif k == "bond":
        for mt in _sites(m, r["patt"], ranks):
            b = m.GetBondBetweenAtoms(mt[0], mt[1])
            if b is None or b.GetIsAromatic():
                continue
            rw = Chem.RWMol(m)
            rw.GetBondBetweenAtoms(mt[0], mt[1]).SetBondType(r["order"])
            for i in mt:
                a = rw.GetAtomWithIdx(i)
                if r["order"] == BT.DOUBLE:
                    _lose_h(a)
                else:
                    _gain_h(a) if a.GetNoImplicit() else None
            p = _fin(rw)
            if p is not None:
                yield p, mt[0]
    else:
        yield from _custom(m, r["fn"], ranks)


def _custom(m, fn, ranks):
    Chem, BT, P = _G["Chem"], _G["BT"], _G["P"]
    if fn == "noxide":
        for pat in (P["noxide"], P["noxide_ar"]):
            for mt in _sites(m, pat, ranks):
                rw = Chem.RWMol(m)
                o = rw.AddAtom(Chem.Atom(8))
                rw.GetAtomWithIdx(o).SetFormalCharge(-1)
                rw.GetAtomWithIdx(mt[0]).SetFormalCharge(1)
                rw.AddBond(mt[0], o, BT.SINGLE)
                p = _fin(rw)
                if p is not None:
                    yield p, mt[0]
    elif fn in ("epoxide", "hydrate"):
        for mt in _sites(m, P["alkene"], ranks, ordered=(fn == "hydrate")):
            b = m.GetBondBetweenAtoms(*mt)
            if b.GetIsAromatic() or b.GetBondType() != BT.DOUBLE:
                continue
            rw = Chem.RWMol(m)
            rw.GetBondBetweenAtoms(*mt).SetBondType(BT.SINGLE)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(mt[0], o, BT.SINGLE)
            if fn == "epoxide":
                rw.AddBond(mt[1], o, BT.SINGLE)
            else:
                a = rw.GetAtomWithIdx(mt[1])
                _gain_h(a) if a.GetNoImplicit() else None
            p = _fin(rw)
            if p is not None:
                yield p, mt[0]
    elif fn == "dearom":
        # C=C hydrogenation inside pseudo-aromatic 6-rings (chromone / pyrone / pyridone / coumarin):
        # flavone -> flavanone, coumarin -> dihydrocoumarin. The reverse (-H2) is covered by -H2@CC.
        ri = m.GetRingInfo()
        k = None
        for ring in ri.BondRings():
            if len(ring) != 6:
                continue
            atoms = {i for b in ring for i in (m.GetBondWithIdx(b).GetBeginAtomIdx(), m.GetBondWithIdx(b).GetEndAtomIdx())}
            if not all(m.GetBondWithIdx(b).GetIsAromatic() for b in ring):
                continue
            hetero = any(m.GetAtomWithIdx(i).GetSymbol() in ("O", "N") for i in atoms)
            exo = any(nb.GetSymbol() == "O" and m.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondType() == BT.DOUBLE
                      for i in atoms for nb in m.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in atoms)
            if not (hetero and exo):
                continue
            if k is None:
                k = Chem.Mol(m)
                try:
                    Chem.Kekulize(k, clearAromaticFlags=True)
                except Exception:
                    return
            seen = set()
            for b in ring:
                bk = k.GetBondWithIdx(b)
                i, j = bk.GetBeginAtomIdx(), bk.GetEndAtomIdx()
                if bk.GetBondType() != BT.DOUBLE or k.GetAtomWithIdx(i).GetSymbol() != "C" or                         k.GetAtomWithIdx(j).GetSymbol() != "C" or m.GetBondWithIdx(b).IsInRingSize(6) and                         sum(1 for r in ri.BondRings() if b in r) > 1:
                    continue
                key = tuple(sorted((ranks[i], ranks[j])))
                if key in seen:
                    continue
                seen.add(key)
                rw = Chem.RWMol(k)
                rw.GetBondWithIdx(b).SetBondType(BT.SINGLE)
                p = _fin(rw)
                if p is not None:
                    yield p, i
    elif fn == "dehydrate":
        for mt in _sites(m, P["dehyd"], ranks, ordered=True):
            o, c1, c2 = mt
            rw = Chem.RWMol(m)
            rw.GetBondBetweenAtoms(c1, c2).SetBondType(BT.DOUBLE)
            _lose_h(rw.GetAtomWithIdx(c2))
            rw.RemoveAtom(o)
            p = _fin(rw)
            if p is not None:
                yield p, c1
    elif fn == "acid":
        for mt in _sites(m, P["acid"], ranks):
            rw = Chem.RWMol(m)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(mt[0], o, BT.DOUBLE)
            p = _fin(rw)
            if p is not None:
                yield p, mt[0]
    elif fn == "lactone":
        acids = m.GetSubstructMatches(P["lac_acid"])
        ohs = [x[0] for x in m.GetSubstructMatches(P["lac_oh"])]
        seen = set()
        for c, _, oa in acids:
            for ob in ohs:
                if ob == oa:
                    continue
                n = len(Chem.GetShortestPath(m, c, ob))
                if n not in (5, 6, 7):
                    continue
                key = (ranks[c], ranks[ob])
                if key in seen:
                    continue
                seen.add(key)
                rw = Chem.RWMol(m)
                rw.AddBond(c, ob, BT.SINGLE)
                _lose_h(rw.GetAtomWithIdx(ob))
                rw.RemoveAtom(oa)
                p = _fin(rw)
                if p is not None:
                    yield p, c
    elif fn == "mdo_catechol":
        for mt in _sites(m, P["cat"], ranks):
            rw = Chem.RWMol(m)
            c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(mt[0], c, BT.SINGLE)
            rw.AddBond(mt[3], c, BT.SINGLE)
            for i in (mt[0], mt[3]):
                _lose_h(rw.GetAtomWithIdx(i))
            p = _fin(rw)
            if p is not None:
                yield p, mt[0]
    elif fn == "mdo_guaiacol":
        for mt in _sites(m, P["gua"], ranks, ordered=True):
            rw = Chem.RWMol(m)
            rw.AddBond(mt[0], mt[4], BT.SINGLE)
            _lose_h(rw.GetAtomWithIdx(mt[0]))
            p = _fin(rw)
            if p is not None:
                yield p, mt[0]
    elif fn in ("chromene", "chroman"):
        for mt in _sites(m, P["chrom"], ranks, ordered=True):
            o, _, _, c2, c3, c4 = mt[:6]
            rw = Chem.RWMol(m)
            rw.AddBond(o, c4, BT.SINGLE)
            _lose_h(rw.GetAtomWithIdx(o))
            rw.GetBondBetweenAtoms(c3, c4).SetBondType(BT.SINGLE)
            if fn == "chromene":
                rw.GetBondBetweenAtoms(c2, c3).SetBondType(BT.DOUBLE)
            p = _fin(rw)
            if p is not None:
                yield p, c4


def _hydrolyse(m, ranks):
    """Generic hydrolysis: cut an ester / amide / glycosidic / sulfate-phosphate bond, OH onto the acyl / anomeric /
    S-P side; yields every resulting fragment (ring opening yields the single +H2O product)."""
    Chem, BT, P = _G["Chem"], _G["BT"], _G["P"]
    seen = set()
    for name in ("hyd_acyl", "hyd_glyc", "hyd_sulf"):
        for mt in m.GetSubstructMatches(P[name], maxMatches=500):
            a, b = (mt[2], mt[1]) if name == "hyd_sulf" else (mt[0], mt[2])   # a: acyl / anomeric / S,P side
            key = tuple(sorted((ranks[a], ranks[b])))
            if key in seen or m.GetBondBetweenAtoms(a, b) is None:
                continue
            seen.add(key)
            rw = Chem.RWMol(m)
            rw.RemoveBond(a, b)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(a, o, BT.SINGLE)
            _gain_h(rw.GetAtomWithIdx(b))
            p = _fin(rw)
            if p is None:
                continue
            frags = Chem.GetMolFrags(p, asMols=True, sanitizeFrags=True)
            for f in frags:
                if f.GetNumHeavyAtoms() >= 3:
                    yield f, a


# ---------------------------------------------------------------------------------------------------------- engine
class _Gen:
    def __init__(self, target, tol, t0):
        self.target, self.tol, self.t0 = target, tol, t0
        self.prod = {}           # canonical smiles -> dict(best path info, n paths)
        self.n_attempt = 0
        self.truncated = ""
        self.work = 0.0
        self.guard = set()

    def time_ok(self):
        """False once a generation limit is reached: product cap, work budget (deterministic) or the wall-clock
        failsafe (machine dependent; recorded in self.guard)."""
        if len(self.prod) >= CAP_TOTAL:
            self.truncated = self.truncated or "cap"
            return False
        if self.work >= W_GEN:
            self.truncated = self.truncated or "work"
            return False
        if time.time() - self.t0 >= T_GEN:
            self.truncated = "time"           # only this path makes the output machine-speed dependent
            self.guard.add("gen")
            return False
        return True

    def add(self, p, mass, path, prior, seed_term, prov):
        Chem = _G["Chem"]
        if abs(mass - self.target) > self.tol:
            return
        s = Chem.MolToSmiles(p)
        e = self.prod.get(s)
        comp = W["prior"] * prior + W["seed"] * seed_term - W["step"] * (len(path) - 1)
        src = prov.split(":")[0]
        if e is None:
            self.prod[s] = dict(mol=p, mass=mass, prior=prior, seed=seed_term, steps=len(path), prov=prov, n=1,
                                srcs={src}, comp=comp)
            return
        if src not in e["srcs"]:
            e["srcs"].add(src)
            e["n"] += 1
        if comp > e["comp"]:
            e.update(prior=prior, seed=seed_term, steps=len(path), prov=prov, comp=comp)

    def expand(self, m, mass, depth, seed_term, tag, path=(), prior=0.0, budget=None, rmin=0):
        """Mass-first expansion of seed m (mass = ExactMolWt) by up to `depth` steps toward the target."""
        Chem, EMW = _G["Chem"], _G["EMW"]
        rules, deltas = _G["rules"], _G["deltas"]
        if budget is None:
            budget = [CAP_SEED]
        dlt = self.target - mass
        tol = self.tol
        nh = m.GetNumHeavyAtoms()
        self.work += nh
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        # fixed rules: final step or intermediate step
        for ri, r in enumerate(rules):
            if ri < rmin:
                continue
            if budget[0] <= 0 or not self.time_ok():
                return
            final = abs(dlt - r["delta"]) <= tol
            inter = False
            if depth >= 2 and not final:
                rest = dlt - r["delta"]
                inter = bool(np.any(np.abs(rest - deltas) <= tol)) or rest < -40.0
                if inter and path == () and r["delta"] == 0.0:
                    inter = False
            if not (final or inter):
                continue
            self.work += nh
            for p, site in _apply_fixed(m, r, ranks):
                budget[0] -= 1
                self.n_attempt += 1
                self.work += p.GetNumHeavyAtoms()
                pm = mass + r["delta"]
                pp = path + ((r["name"], site),)
                if final:
                    self.add(p, float(EMW(p)), pp, prior + r["prior"], seed_term,
                             tag + ":" + ">".join("%s@%d" % x for x in pp))
                if inter:
                    self.expand(p, pm, depth - 1, seed_term, tag, pp, prior + r["prior"], budget, rmin=ri)
                if budget[0] <= 0 or not self.time_ok():
                    return
        # hydrolysis (variable delta): only when the seed must lose mass
        if dlt < -15.0 and budget[0] > 0 and self.time_ok():
            self.work += 3 * nh
            for f, site in _hydrolyse(m, ranks):
                budget[0] -= 1
                self.n_attempt += 1
                self.work += f.GetNumHeavyAtoms()
                fm = float(EMW(f))
                pp = path + (("hydrolysis", site),)
                if abs(fm - self.target) <= tol:
                    self.add(f, fm, pp, prior + HYDROLYSIS_PRIOR, seed_term,
                             tag + ":" + ">".join("%s@%d" % x for x in pp))
                elif depth >= 2:
                    rest = self.target - fm
                    if np.any(np.abs(rest - deltas) <= tol) or rest < -40.0:
                        self.expand(f, fm, depth - 1, seed_term, tag, pp, prior + HYDROLYSIS_PRIOR, budget)
                if budget[0] <= 0 or not self.time_ok():
                    return


def _pool():
    if "pool_mass" not in _G:
        _G["pool_mass"] = np.load(_ap("pool_mass.npy"))
        _G["pool_fp"] = np.load(_ap("pool_fp.npy"), mmap_mode="r")
        _G["pool_key"] = np.load(_ap("pool_key.npy"))
        _G["pool_np"] = _np_mask(_G["pool_key"])
        _G["pool_smi_off"] = None
    return _G


def _np_mask(pool_key):
    if DEPLOY_DIR is not None:              # deploy: shipped mask, never rebuilt
        mk = np.load(_ap("pool_np.npy"))
        if len(mk) != len(pool_key):
            raise ValueError("c3_pool_np.npy does not match c3_pool_key.npy")
        return mk.astype(bool)
    if NP_MASK.exists():
        mk = np.load(NP_MASK)
        if len(mk) == len(pool_key):
            return mk
    import pyarrow.parquet as pq
    keys = set(pq.read_table(NP_POOL, columns=["ik14"]).column("ik14").to_pylist())
    mk = np.array([k.decode() in keys for k in pool_key], bool)
    NP_MASK.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(NP_MASK) + ".tmp.npy"
    np.save(tmp, mk)
    os.replace(tmp, NP_MASK)
    return mk


def _pool_smiles(rows):
    """SMILES of pool rows (line offsets of pool_smiles.txt indexed once per process)."""
    g = _G
    if g.get("pool_smi_off") is None:
        raw = _ap("pool_smiles.txt").read_bytes()
        nl = np.frombuffer(raw, np.uint8) == 10
        ends = np.flatnonzero(nl)
        starts = np.concatenate([[0], ends[:-1] + 1])
        g["pool_smi_raw"], g["pool_smi_off"] = raw, (starts, ends)
    st, en = g["pool_smi_off"]
    raw = g["pool_smi_raw"]
    return [raw[st[i]:en[i]].decode().rstrip("\r") for i in rows]


def _zfull_many(mols, zlog):
    Chem, MACCS, bits = _G["Chem"], _G["MACCS"], _G["bits"]
    out = np.empty(len(mols), np.float32)
    for i, m in enumerate(mols):
        fp = np.concatenate([_G["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _G["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _G["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCS.GenMACCSKeys(m), dtype=np.uint8)])[bits]
        out[i] = float(fp.astype(np.float32) @ zlog)
    return out


def _stage1_weights(zlog):
    bits = _G["bits"]
    w2 = np.zeros(4096, np.float32)
    w3 = np.zeros(4096, np.float32)
    s2 = bits < 4096
    w2[bits[s2]] = zlog[s2]
    s3 = (bits >= 4096) & (bits < 8192)
    w3[bits[s3] - 4096] = zlog[s3]
    return w2, w3


def _keys_of(entry, *names):
    return {entry.get(n) for n in names if entry.get(n)}


def seeds_b(target, tol, zlog, forbidden):
    """Pool structures (NP-like subset) at target - d for every single rule delta d (and target + group for the
    hydrolysis removals); top N_SEED_B_PER_D per delta by packed-fp @ zlog."""
    g = _pool()
    pm, pfp, pkey, pnp = g["pool_mass"], g["pool_fp"], g["pool_key"], g["pool_np"]
    ds = sorted({round(float(d), 6) for d in _G["deltas"] if abs(d) > 1e-6} |
                {round(-fmass(GROUP_FORMULA[x]), 6) for x in B_REMOVE})
    out = []
    for d in ds:
        mt = target - d
        if mt < 60:
            continue
        lo, hi = np.searchsorted(pm, mt - tol), np.searchsorted(pm, mt + tol, "right")
        if hi <= lo:
            continue
        rows = np.arange(lo, hi)
        rows = rows[pnp[lo:hi]]
        rows = np.array([r for r in rows if pkey[r].decode() not in forbidden], int)
        if not len(rows):
            continue
        sc = np.unpackbits(np.asarray(pfp[rows]), axis=1)[:, :len(zlog)].astype(np.float32) @ zlog
        o = np.lexsort((rows, -sc))[:N_SEED_B_PER_D]
        for j in o:
            out.append((int(rows[j]), float(sc[j]), d))
    return out


def generate(ctx, k=200, forbidden=frozenset(), return_table=False):
    t0 = time.time()
    Chem = _rd()
    EMW = _G["EMW"]
    target = float(ctx["target"])
    tol = target * PPM * 1e-6
    zlog = np.asarray(ctx["zlog"], np.float32)
    forbidden = set(forbidden)
    excl = forbidden | set(ctx.get("exclude") or ())   # seed-only exclusion (+ bench strip keys, c3np_strip.py)
    G = _Gen(target, tol * 0.98, t0)
    # keys never to output: window structures (already in E6/engine) and the seeds themselves (in a database)
    known = set()
    for w in ctx.get("window", []) or []:
        known |= _keys_of(w, "ik", "ik14", "ckey")

    # ---- seeds A: library analogs
    seedsA = []
    for a in (ctx.get("analogs") or []):
        if len(seedsA) >= N_SEED_A:
            break
        if (a.get("sim") or 0) < MIN_SIM_A:
            continue
        if _keys_of(a, "ik14", "ik", "ckey") & excl:
            continue
        m = Chem.MolFromSmiles(a["smiles"])
        if m is None:
            continue
        m = normalize(m)
        if m is None:
            continue
        seedsA.append((m, float(a.get("sim") or 0)))
        known |= _keys_of(a, "ik14", "ik", "ckey")
    # ---- seeds C: same-mass window structures (relocations, zero net delta, 2 steps)
    seedsC = []
    win = sorted(ctx.get("window", []) or [], key=lambda w: -float(w.get("fpscore", w.get("score", 0)) or 0))
    for w in win[:N_SEED_C]:
        if _keys_of(w, "ik", "ik14", "ckey") & excl:
            continue
        m = Chem.MolFromSmiles(w["smiles"])
        m = normalize(m) if m is not None else None
        if m is not None:
            seedsC.append(m)
    # ---- seeds B
    sb = seeds_b(target, tol, zlog, excl)
    smi_b = _pool_smiles([r for r, _, _ in sb]) if sb else []
    if sb:
        scs = np.array([s for _, s, _ in sb])
        mu, sd = float(scs.mean()), float(scs.std() + 1e-6)
    seedsB = []
    for (row, sc, d), s in zip(sb, smi_b):
        m = Chem.MolFromSmiles(s)
        m = normalize(m) if m is not None else None
        if m is None:
            continue
        seedsB.append((m, 0.25 + 0.1 * max(-2.0, min(2.0, (sc - mu) / sd)), sc, row))
        known.add(_G["pool_key"][row].decode())
    t_seed = time.time() - t0

    # ---- expansion (priority order: analogs 1-2 steps, pool seeds 1 step, window relocations)
    for i, (m, sim) in enumerate(seedsA):
        if not G.time_ok():
            break
        G.expand(m, float(EMW(m)), 2 if i < N_2STEP_A else 1, sim, "A%d" % i)
    order = sorted(range(len(seedsB)), key=lambda i: -seedsB[i][2])
    for i in order:
        if not G.time_ok():
            break
        m, st, sc, row = seedsB[i]
        G.expand(m, float(EMW(m)), 1, st, "B%d" % row)
    for i, m in enumerate(seedsC):
        if not G.time_ok():
            break
        G.expand(m, float(EMW(m)), 2, 0.15, "C%d" % i, budget=[CAP_SEED // 2])
    t_gen = time.time() - t0

    if not G.prod:
        return ([], dict(n_prod=0, t_gen=t_gen, t_seed=t_seed, work=G.work, guard="+".join(sorted(G.guard)),
                         truncated=G.truncated)) if return_table else []
    smis = sorted(G.prod)
    ents = [G.prod[s] for s in smis]
    # ---- stage 1: Morgan blocks
    w2, w3 = _stage1_weights(zlog)
    s1 = np.empty(len(ents), np.float32)
    for i, e in enumerate(ents):
        m = e["mol"]
        s1[i] = _G["m2"].GetFingerprintAsNumPy(m).astype(np.float32) @ w2 + \
            _G["m3"].GetFingerprintAsNumPy(m).astype(np.float32) @ w3
    pre = np.array([W["prior"] * e["prior"] + W["seed"] * e["seed"] + W["mult"] * math.log(e["n"])
                    - W["step"] * (e["steps"] - 1) for e in ents], np.float32)
    s1n = (s1 - s1.max()) / (s1.std() + 1e-6)
    o1 = np.lexsort((np.arange(len(ents)), -(s1n + pre)))[:N_STAGE2]
    t_s1 = time.time() - t0
    # ---- stage 2: full recipe, bounded by the deterministic W_S2 budget (wall clock = failsafe only)
    zf = np.full(len(ents), np.nan, np.float32)
    s2work = 0.0
    for j in o1:
        h = float(ents[j]["mol"].GetNumHeavyAtoms()) ** 2.5
        if s2work + h > W_S2:
            G.truncated = G.truncated or "stage2_work"
            break
        if time.time() - t0 > T_TOTAL:
            G.guard.add("all")
            break
        s2work += h
        zf[j] = _zfull_many([ents[j]["mol"]], zlog)[0]
    if "all" in G.guard:                      # failsafe: whole list in stage-1 order (no partial mixing)
        zf[:] = np.nan
        G.truncated = G.truncated or "stage2_time"
    done = ~np.isnan(zf)
    t_s2 = time.time() - t0
    zd = zf[done]
    zn = (zf - zd.max()) / (zd.std() + 1e-6) if done.any() else zf
    # products not stage-2 scored rank after the scored ones (by stage-1)
    final = np.where(done, zn + pre, -1e6 + s1n + pre)
    order = np.lexsort((np.arange(len(ents)), -final))
    gsuf = ("|guard=" + "+".join(sorted(G.guard))) if G.guard else ""
    out, seen, table = [], set(), []
    for j in order:
        if len(out) >= k:
            break
        e = ents[j]
        if abs(float(EMW(e["mol"])) - target) > tol:
            continue
        try:
            ik = Chem.MolToInchiKey(e["mol"])[:14]
        except Exception:
            ik = None
        if not ik or ik in seen or ik in known:
            continue
        seen.add(ik)
        smi = Chem.MolToSmiles(e["mol"], isomericSmiles=False)
        chk = Chem.MolFromSmiles(smi)                 # what the consumer will parse must pass the 10 ppm rule
        if chk is None or abs(float(EMW(chk)) - target) > tol:
            continue
        out.append((smi, float(final[j]), e["prov"] + gsuf))
        if return_table:
            table.append(dict(smiles=smi, ik14=ik, z=float(zf[j]) if done[j] else None, s1=float(s1[j]),
                              prior=e["prior"], seed=e["seed"], n=e["n"], steps=e["steps"], prov=e["prov"]))
    if return_table:
        return out, dict(n_prod=len(ents), n_attempt=G.n_attempt, truncated=G.truncated, n_stage2=int(done.sum()), t_gen=t_gen,
                         t_total=time.time() - t0, nA=len(seedsA), nB=len(seedsB), nC=len(seedsC), table=table,
                         work=G.work, s2work=s2work, guard="+".join(sorted(G.guard)), t_seed=t_seed, t_s1=t_s1,
                         t_s2=t_s2)
    return out


# ---------------------------------------------------------------------------------------------------------- CLI
def _load_spectra(mids_by_file):
    import duckdb
    out = {}
    for f, mids in mids_by_file.items():
        p = (ROOT / f).as_posix()
        q = (f"select molecule_id, ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct from '{p}' "
             f"where molecule_id in ({','.join(repr(m) for m in mids)})")
        for mid, mz, it, pmz, ad in duckdb.sql(q).fetchall():
            out.setdefault(mid, []).append(dict(mz=np.array(mz or []), intensity=np.array(it or []),
                                                precursor_mz=pmz, adduct=ad))
    return out


def _worker(job):
    mid, ctx, forb = job
    t = time.time()
    try:
        res, info = generate(ctx, k=200, forbidden=forb, return_table=True)
        err = None
    except Exception as ex:          # keep the run going; report
        import traceback
        res, info, err = [], {}, traceback.format_exc(limit=3)
    return mid, res, info, time.time() - t, err


CHAIN_RULES = ("+CH2@chain", "-CH2@chain", "+C2H4@chain", "-C2H4@")


def _rank_index(res, rank, r):
    """Index into res of the candidate at de-duplicated rank `rank` (same de-dup as _truth_eval)."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    sys.path.insert(0, str(ROOT / "research" / "c3gen"))
    import assets as A
    seen, pos = set(), 0
    for i, (smi, _, _) in enumerate(res):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        key = Chem.MolToInchiKey(m)[:14]
        if key not in set(r["correct"].split(";")) | {r["ik14"], r["ckey"]} and CalcMolFormula(m) == r["formula"]:
            key = A.taut_key(m) or key
        if key in seen:
            continue
        seen.add(key)
        pos += 1
        if pos == rank:
            return i
    return len(res) - 1


def _truth_eval(results, M):
    """hit@1/@25/@200 by tautomer-canonical key (plain-key match, plus taut key for same-formula products)."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    sys.path.insert(0, str(ROOT / "research" / "c3gen"))
    import assets as A
    rows = []
    for mid, res in results.items():
        r = M.loc[mid]
        ok = set(r["correct"].split(";")) | {r["ik14"], r["ckey"]}
        rank, seen = 0, set()
        pos = 0
        for smi, _, _ in res:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            ik = Chem.MolToInchiKey(m)[:14]
            key = ik
            hit = ik in ok
            if not hit and CalcMolFormula(m) == r["formula"]:
                tk = A.taut_key(m)
                key = tk or ik
                hit = tk in ok
            if key in seen:
                continue
            seen.add(key)
            pos += 1
            if hit:
                rank = pos
                break
        rows.append(dict(mid=mid, subset=r["subset"], rank=rank))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3np", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/c3gen/biotransform_c3np.json")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--no_spectra", action="store_true")
    ap.add_argument("--strip", default="none", choices=["none", "tc085", "tc070", "edit1"],
                    help="bench realism mode: strip the truth's congeners from the seed sources (c3np_strip.py)")
    a = ap.parse_args()
    if not a.c3np:
        ap.error("only --c3np is implemented")
    import pandas as pd
    C = pickle.load(open(ROOT / "results/c3np/context.pkl", "rb"))["mols"]
    F = pd.read_parquet(ROOT / "results/c3np/forbidden.parquet")
    forb = F.groupby("mid").key.apply(set).to_dict()
    M = pd.read_parquet(ROOT / "results/c3np/molecules.parquet").set_index("mid", drop=False)
    mids = list(M.mid)[a.offset:a.offset + a.n]
    spec = {}
    if not a.no_spectra:
        byf = {}
        for mid in mids:
            s = C[mid].get("spectra") or {}
            if isinstance(s, dict) and s.get("file"):
                byf.setdefault(s["file"], []).append(s.get("molecule_id", mid))
        spec = _load_spectra(byf)
    sys.path.insert(0, str(ROOT / "research" / "scripts"))
    import c3np_strip
    skeys = c3np_strip.strip_keys(a.strip)
    jobs = []
    for mid in mids:
        c = C[mid]
        ctx = dict(mid=mid, target=c["target"], adducts=c["adducts"], zlog=c["zlog"], analogs=c["analogs"],
                   window=[dict(w, fpscore=w.get("score")) for w in c["window"]], spectra=spec.get(mid, []))
        ctx = c3np_strip.apply_strip(ctx, skeys.get(mid, frozenset()))
        jobs.append((mid, ctx, frozenset(forb.get(mid, set()))))
    t0 = time.time()
    res, infos, times, errs = {}, {}, {}, {}
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as mp:
            it = mp.imap(_worker, jobs, chunksize=1)
            for i, (mid, r, info, dt, err) in enumerate(it):
                res[mid], infos[mid], times[mid] = r, info, dt
                if err:
                    errs[mid] = err
                if (i + 1) % 10 == 0:
                    print(time.strftime("%H:%M:%S"), f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs):
            mid, r, info, dt, err = _worker(j)
            res[mid], infos[mid], times[mid] = r, info, dt
            if err:
                errs[mid] = err
            if (i + 1) % 10 == 0:
                print(time.strftime("%H:%M:%S"), f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({m: [s for s, _, _ in r] for m, r in res.items()}, open(out, "w"))
    if a.dump:
        pickle.dump(dict(res=res, infos=infos, times=times), open(str(out).replace(".json", "_dump.pkl"), "wb"))
    # summary (the generator never saw the truth; this reads it only to score)
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    nv = nt = 0
    for mid, r in res.items():
        tg = float(C[mid]["target"])
        for s, _, _ in r:
            nt += 1
            m = Chem.MolFromSmiles(s)
            nv += int(m is not None and abs(ExactMolWt(m) - tg) <= tg * PPM * 1e-6)
    rows = _truth_eval(res, M)
    D = pd.DataFrame(rows)
    D["n_cand"] = [len(res[m]) for m in D.mid]
    # chain-homolog hits (+-CH2 / +-C2H4 on a chain anywhere in the winning path): reported separately because
    # homolog series (lipids, N-acyl amides) are over-represented in s3none (review r2)
    D["chain"] = [bool(0 < rk and any(t in res[m][_rank_index(res[m], rk, M.loc[m])][2] for t in CHAIN_RULES))
                  for m, rk in zip(D.mid, D["rank"])]
    D["sec"] = [times[m] for m in D.mid]

    def summ(d):
        rk = d["rank"].values
        return dict(n=len(d), mrr25=round(float(np.where((rk > 0) & (rk <= 25), 1.0 / np.maximum(rk, 1), 0).mean()), 4),
                    hit1=int((rk == 1).sum()), hit25=int(((rk > 0) & (rk <= 25)).sum()),
                    hit200=int(((rk > 0) & (rk <= 200)).sum()), mean_cand=round(float(d.n_cand.mean()), 1),
                    s_per_mol=round(float(d.sec.mean()), 2), s_max=round(float(d.sec.max()), 2))
    D_nc = D.assign(rank=np.where(D.chain, 0, D["rank"]))
    S = dict(name=Path(a.out).stem, n=len(D), offset=a.offset, workers=a.workers, strip=a.strip,
             wall_s=round(time.time() - t0, 1),
             overall=summ(D), by_subset={k: summ(d) for k, d in D.groupby("subset")},
             by_subset_without_chain_homolog_hits={k: summ(d) for k, d in D_nc.groupby("subset")},
             n_chain_homolog_hits=int(D.chain.sum()),
             mass_valid_share=round(nv / max(nt, 1), 4), n_errors=len(errs), errors=dict(list(errs.items())[:3]),
             mean_products=round(float(np.mean([i.get("n_prod", 0) for i in infos.values()])), 1),
             truncated={t: sum(1 for i in infos.values() if i.get("truncated") == t)
                        for t in ("time", "cap", "work", "stage2_work", "stage2_time")},
             guard_fired={g: sum(1 for i in infos.values() if g in (i.get("guard") or "").split("+"))
                          for g in ("gen", "all")},
             budgets=dict(W_GEN=W_GEN, W_S2=W_S2, T_GEN=T_GEN, T_TOTAL=T_TOTAL),
             hits=[dict(r, chain=bool(c)) for r, c in zip(rows, D.chain) if r["rank"] > 0])
    json.dump(S, open(str(out).replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k != "hits"}, indent=1))


if __name__ == "__main__":
    main()

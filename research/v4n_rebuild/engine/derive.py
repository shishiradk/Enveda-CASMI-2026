"""Class-3 candidate generation: one- and two-step biosynthetic transformations of an analog structure whose mass
change matches (target - analog mass).  Our grammar (MIT): 45 transforms written as reaction SMARTS with their
formula change; the combination set is every single transform plus every unordered pair (with repetition).

Deliberate difference to v1: no duplicated 'dehexose' entry (v1 lists the same reaction twice; the duplicate can only
consume budget, never add a product).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from . import chem

# (name, reaction SMARTS, formula change).  Order matters only when budgets bind (kept stable on purpose).
_OH = "[O;H1;!$(OC=O):1]"                 # hydroxyl that is not part of a carboxylic acid
_HEX = "C1OC(CO)C(O)C(O)C1O"
_PEN = "C1OCC(O)C(O)C1O"
_DHX = "C1OC(C)C(O)C(O)C1O"
GRAMMAR: List[Tuple[str, str, str]] = [
    # ---- additions
    ("OH_on_aromatic_CH", "[c;H1:1]>>[c:1]O", "+O"),
    ("OH_on_CH2_CH3", "[C;H2,H3;!$(C=O):1]>>[C:1]O", "+O"),
    ("OH_on_CH", "[C;H1;!$(C=O):1]>>[C:1]O", "+O"),
    ("O_methylation", f"{_OH}>>[O:1]C", "+CH2"),
    ("N_methylation", "[N;H1,H2:1]>>[N:1]C", "+CH2"),
    ("C_methylation_aromatic", "[c;H1:1]>>[c:1]C", "+CH2"),
    ("methoxy_on_aromatic", "[c;H1:1]>>[c:1]OC", "+CH2O"),
    ("O_acetylation", f"{_OH}>>[O:1]C(C)=O", "+C2H2O"),
    ("N_acetylation", "[N;H1,H2;!$(NC=O):1]>>[N:1]C(C)=O", "+C2H2O"),
    ("O_malonylation", f"{_OH}>>[O:1]C(=O)CC(=O)O", "+C3H2O3"),
    ("O_hexosylation", f"{_OH}>>[O:1]{_HEX}", "+C6H10O5"),
    ("O_pentosylation", f"{_OH}>>[O:1]{_PEN}", "+C5H8O4"),
    ("O_deoxyhexosylation", f"{_OH}>>[O:1]{_DHX}", "+C6H10O4"),
    ("O_glucuronidation", f"{_OH}>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O", "+C6H8O6"),
    ("C_hexosylation_aromatic", f"[c;H1:1]>>[c:1]{_HEX}", "+C6H10O5"),
    ("O_sulfation", f"{_OH}>>[O:1]S(=O)(=O)O", "+SO3"),
    ("C_prenylation_aromatic", "[c;H1:1]>>[c:1]CC=C(C)C", "+C5H8"),
    ("O_prenylation", f"{_OH}>>[O:1]CC=C(C)C", "+C5H8"),
    ("O_ethylation", f"{_OH}>>[O:1]CC", "+C2H4"),
    ("reduce_C=C", "[C:1]=[C:2]>>[C:1][C:2]", "+H2"),
    ("reduce_C=O", "[C:1]=[O:2]>>[C:1][O:2]", "+H2"),
    ("desaturate_C-C", "[C;H1,H2:1][C;H1,H2:2]>>[C:1]=[C:2]", "-H2"),
    ("oxidise_CH-OH", "[C;H1:1][O;H1:2]>>[C:1]=[O:2]", "-H2"),
    ("carboxyl_on_aromatic", "[c;H1:1]>>[c:1]C(=O)O", "+CO2"),
    ("oxidise_CH3_to_COOH", "[C;H3:1]>>[C:1](=O)O", "+O2"),
    ("amino_on_aromatic", "[c;H1:1]>>[c:1]N", "+NH"),
    ("chloro_on_aromatic", "[c;H1:1]>>[c:1]Cl", "+Cl-H"),
    ("O_coumaroylation", f"{_OH}>>[O:1]C(=O)C=Cc1ccc(O)cc1", "+C9H6O2"),
    ("O_feruloylation", f"{_OH}>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1", "+C10H8O3"),
    ("O_caffeoylation", f"{_OH}>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1", "+C9H6O3"),
    ("O_galloylation", f"{_OH}>>[O:1]C(=O)c1cc(O)c(O)c(O)c1", "+C7H4O4"),
    ("O_benzoylation", f"{_OH}>>[O:1]C(=O)c1ccccc1", "+C7H4O"),
    # ---- removals
    ("O_demethylation", "[O:1][CH3]>>[O:1]", "-CH2"),
    ("N_demethylation", "[N:1][CH3]>>[N:1]", "-CH2"),
    ("dehydroxylate_aromatic", "[c:1][OH]>>[c:1]", "-O"),
    ("dehydroxylate_aliphatic", "[C;!$(C=O):1][OH]>>[C:1]", "-O"),
    ("O_deacetylation", "[O:1]C(=O)[CH3]>>[O:1]", "-C2H2O"),
    ("O_dehexosylation", f"[O:1]{_HEX}>>[O:1]", "-C6H10O5"),
    ("O_depentosylation", f"[O:1]{_PEN}>>[O:1]", "-C5H8O4"),
    ("O_dedeoxyhexosylation", f"[O:1]{_DHX}>>[O:1]", "-C6H10O4"),
    ("O_demalonylation", "[O:1]C(=O)CC(=O)O>>[O:1]", "-C3H2O3"),
    ("O_desulfation", "[O:1]S(=O)(=O)O>>[O:1]", "-SO3"),
    ("deprenylate_aromatic", "[c:1]CC=C(C)C>>[c:1]", "-C5H8"),
    ("decarboxylation", "[C,c:1]C(=O)[OH]>>[C,c:1]", "-CO2"),
    ("demethoxylate_aromatic", "[c:1]O[CH3]>>[c:1]", "-CH2O"),
]
assert len({g[0] for g in GRAMMAR}) == len(GRAMMAR)


def _delta(change: str) -> Dict[str, int]:
    """'+C2H2O' -> {'C':2,'H':2,'O':1}; '+Cl-H' -> {'Cl':1,'H':-1}."""
    out: Dict[str, int] = {}
    import re
    for sign, f in re.findall(r"([+-])([A-Za-z0-9]+)", change):
        for el, n in chem.parse_formula(f).items():
            out[el] = out.get(el, 0) + (n if sign == "+" else -n)
    return out


DELTA = {name: _delta(ch) for name, _, ch in GRAMMAR}
NAMES = [g[0] for g in GRAMMAR]
_SMARTS = {name: s for name, s, _ in GRAMMAR}
_RX: dict = {}


def _reaction(name):
    if name not in _RX:
        from rdkit.Chem import AllChem
        _RX[name] = AllChem.ReactionFromSmarts(_SMARTS[name])
    return _RX[name]


def _combo_mass(names) -> float:
    tot: Dict[str, int] = {}
    for n in names:
        for el, k in DELTA[n].items():
            tot[el] = tot.get(el, 0) + k
    return sum(chem.mass_of(e) * k for e, k in tot.items())


_COMBOS: List[Tuple[Tuple[str, ...], float]] = []


def combos():
    if not _COMBOS:
        for n in NAMES:
            _COMBOS.append(((n,), _combo_mass((n,))))
        for a, b in itertools.combinations_with_replacement(NAMES, 2):
            _COMBOS.append(((a, b), _combo_mass((a, b))))
    return _COMBOS


def matching(delta: float, tol: float, max_steps: int = 2):
    return [c for c, m in combos() if abs(m - delta) <= tol and len(c) <= max_steps]


def _products(mol, name, cap):
    from rdkit import Chem
    try:
        res = _reaction(name).RunReactants((mol,))
    except Exception:
        return []
    out, seen = [], set()
    for tup in res:
        p = tup[0]
        try:
            Chem.SanitizeMol(p)
            s = Chem.MolToSmiles(p)
        except Exception:
            continue
        if s not in seen:
            seen.add(s); out.append(p)
            if len(out) >= cap:
                break
    return out


def derive(smiles: str, delta: float, tol: float = 0.003, max_steps: int = 2, max_out: int = 300,
           budget: int = 800) -> List[Tuple[str, Tuple[str, ...]]]:
    """[(product SMILES, combo)] for the combos matching `delta` (first `max_out`, de-duplicated by SMILES)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    found: Dict[str, Tuple[str, ...]] = {}
    for combo in matching(delta, tol, max_steps):
        cur = [mol]
        for depth, name in enumerate(combo):
            nxt = []
            for m in cur[:24]:
                nxt += _products(m, name, 32 if depth == 0 else 16)
                budget -= 1
                if budget <= 0:
                    break
            cur = nxt
            if not cur or budget <= 0:
                break
        for m in cur:
            s = Chem.MolToSmiles(m)
            if s != smiles and s not in found:
                found[s] = combo
        if len(found) >= max_out or budget <= 0:
            break
    return list(found.items())[:max_out]

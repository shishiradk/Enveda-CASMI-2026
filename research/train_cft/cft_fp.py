"""CFT structure fingerprint ("cfp1"): our own multi-block binary fingerprint, computed with RDKit.

Raw layout (16,384 bits), all standard RDKit generators with default settings unless stated:
    block     bits   generator
    ecfp4     4096   Morgan radius 2
    ecfp6     4096   Morgan radius 3
    fcfp4     2048   Morgan radius 2 with feature atom invariants
    path      2048   RDKit path fingerprint, maxPath 6
    apair     2048   atom pair   (count simulation, RDKit default)
    tors      2048   topological torsion (count simulation, RDKit default)
The model predicts only the bits listed in cfp_bits.npy (bit frequency within [0.005, 0.995] on a structure
sample; chosen by prep_cft.py).

The SAME function must be used for the training pool and for candidates at inference:
    fp = Fingerprinter(bits)            # bits = np.load("cfp_bits.npy")
    packed = fp.packed(smiles)          # np.uint8 [ceil(nbits/8)] (np.packbits, MSB first) or None
No stereo information is used (chirality flags off), no standardisation is applied to the SMILES.
"""
import numpy as np

BLOCKS = (("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("path", 2048), ("apair", 2048), ("tors", 2048))
RAW_BITS = sum(n for _, n in BLOCKS)
ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
_EL_IX = {e: i for i, e in enumerate(ELEMENTS)}


class Fingerprinter:
    def __init__(self, bits=None):
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator as G
        RDLogger.DisableLog("rdApp.*")
        self.gens = [G.GetMorganGenerator(radius=2, fpSize=4096),
                     G.GetMorganGenerator(radius=3, fpSize=4096),
                     G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                     G.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
                     G.GetAtomPairGenerator(fpSize=2048),
                     G.GetTopologicalTorsionGenerator(fpSize=2048)]
        self.bits = None if bits is None else np.asarray(bits, np.int64)
        self.nbits = RAW_BITS if bits is None else len(self.bits)

    def mol(self, smiles):
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles

    def raw(self, m):
        """0/1 uint8 vector of the RAW_BITS raw bits for an RDKit molecule."""
        return np.concatenate([g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in self.gens])

    def bits01(self, smiles):
        m = self.mol(smiles)
        if m is None: return None
        v = self.raw(m)
        return v if self.bits is None else v[self.bits]

    def packed(self, smiles):
        v = self.bits01(smiles)
        return None if v is None else np.packbits(v)

    def describe(self, smiles):
        """(packed fingerprint, exact mass, formula string, element counts uint8[10]) or None."""
        from rdkit.Chem.Descriptors import ExactMolWt
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        m = self.mol(smiles)
        if m is None: return None
        try:
            v = self.raw(m)
            if self.bits is not None: v = v[self.bits]
            el = np.zeros(len(ELEMENTS), np.int64)
            for a in m.GetAtoms():
                i = _EL_IX.get(a.GetSymbol())
                if i is not None: el[i] += 1
                el[1] += a.GetTotalNumHs()
            return np.packbits(v), float(ExactMolWt(m)), CalcMolFormula(m), np.minimum(el, 255).astype(np.uint8)
        except Exception:
            return None


def unpack(packed, nbits):
    """np.packbits rows -> 0/1 uint8 [..., nbits]."""
    return np.unpackbits(np.asarray(packed), axis=-1)[..., :nbits]

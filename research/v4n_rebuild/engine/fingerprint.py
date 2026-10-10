"""cfp1 structure fingerprint (same definition as research/train_cft/cft_fp.py, our code).

Raw layout, 16,384 bits:  ecfp4 4096 | ecfp6 4096 | fcfp4 2048 | rdkit path (maxPath 6) 2048 | atom pair 2048 |
topological torsion 2048.  Tables store the RAW layout (np.packbits, MSB first, 2,048 bytes per row) so that any FP
model, whatever its informative-bit selection (cft_ho2: 11,184 bits, R models: 11,171 bits), can be served from the
same pool; `select(packed_raw, bits)` projects to a model layout.
"""
from __future__ import annotations

import numpy as np

BLOCKS = (("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("path", 2048), ("apair", 2048), ("tors", 2048))
RAW_BITS = sum(n for _, n in BLOCKS)
RAW_BYTES = RAW_BITS // 8
BLOCK_START = np.cumsum([0] + [n for _, n in BLOCKS])


class RawFingerprinter:
    def __init__(self):
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator as G
        RDLogger.DisableLog("rdApp.*")
        self.gens = [G.GetMorganGenerator(radius=2, fpSize=4096),
                     G.GetMorganGenerator(radius=3, fpSize=4096),
                     G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                     G.GetRDKitFPGenerator(fpSize=2048, maxPath=6),
                     G.GetAtomPairGenerator(fpSize=2048),
                     G.GetTopologicalTorsionGenerator(fpSize=2048)]

    def raw(self, mol) -> np.ndarray:
        return np.concatenate([g.GetFingerprintAsNumPy(mol).astype(np.uint8) for g in self.gens])


_FPR = []


def raw_fingerprinter() -> RawFingerprinter:
    if not _FPR:
        _FPR.append(RawFingerprinter())
    return _FPR[0]


def unpack(packed, nbits=RAW_BITS) -> np.ndarray:
    return np.unpackbits(np.asarray(packed), axis=-1)[..., :nbits]


def select(packed_raw, bits) -> np.ndarray:
    """packed raw rows [n, 2048] -> 0/1 uint8 [n, len(bits)] in the model layout."""
    return unpack(packed_raw)[..., bits]


def block_counts(bits) -> dict:
    """Kept bits per cfp1 block for a selection."""
    b = np.asarray(bits)
    return {name: int(((b >= BLOCK_START[i]) & (b < BLOCK_START[i + 1])).sum()) for i, (name, _) in enumerate(BLOCKS)}

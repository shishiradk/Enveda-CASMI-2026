"""Conversion between raw (mz, intensity) arrays and matchms Spectrum objects."""

import numpy as np
from matchms import Spectrum


def make_spectrum(mzs, intensities, precursor_mz: float) -> Spectrum:
    """Build a matchms Spectrum, sorted by m/z ascending (required for matchms'
    peak-matching routines, which assume sorted input)."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    order = np.argsort(mzs)
    return Spectrum(
        mz=mzs[order],
        intensities=intensities[order],
        metadata={"precursor_mz": float(precursor_mz)},
    )

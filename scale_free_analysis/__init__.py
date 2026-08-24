"""Spectral, oscillatory-bout, and cycle-by-cycle EEG analysis."""

from .metrics import (
    APERIODIC_FEATURES,
    BAND_FEATURES,
    ebosc_wavelet_power,
    fit_specparam_spectrum,
)

__all__ = [
    "APERIODIC_FEATURES",
    "BAND_FEATURES",
    "ebosc_wavelet_power",
    "fit_specparam_spectrum",
]

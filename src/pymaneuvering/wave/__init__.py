"""Wave dynamics module with spectral separation architecture.

Provides:
    - Wave spectra (JONSWAP, Pierson-Moskowitz)
    - Directional spreading functions (cosine-squared)
    - WaveModel with dual-output: raw_load (1st + 2nd order) and
      mean_drift_load (2nd-order mean drift only)

NOTE: This is a simplified stochastic disturbance model, not a
vessel-specific RAO / QTF hydrodynamic model.
"""

from .spectrum import WaveSpectrum, JONSWAP, PiersonMoskowitz
from .spreading import cosine_squared_spreading, discretize_directions
from .wave_model import WaveModel, WaveLoad

__all__ = [
    "WaveSpectrum",
    "JONSWAP",
    "PiersonMoskowitz",
    "cosine_squared_spreading",
    "discretize_directions",
    "WaveModel",
    "WaveLoad",
]

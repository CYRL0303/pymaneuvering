"""OCIMF 2010 wind aerodynamics module.

Replaces the legacy trigonometric wind approximation with engineering-grade
coefficient lookup tables per OCIMF 2010 standards.

Provides:
    - OCIMFWindModel: 36-node lookup table with linear interpolation
    - WindCoefficients: non-dimensional (Cx, Cy, Cn) dataclass
    - LoadingCondition: "loaded" / "ballasted" selector
    - compute_from_true_wind(): apparent wind conversion built in

NOTE: The ballasted-condition tables are approximate (scaled from loaded)
and should NOT be treated as validated OCIMF engineering data.
"""

from .ocimf import (
    OCIMFWindModel,
    WindCoefficients,
    LoadingCondition,
)

__all__ = [
    "OCIMFWindModel",
    "WindCoefficients",
    "LoadingCondition",
]

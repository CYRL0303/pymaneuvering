"""
Directional spreading functions for short-crested wave fields.

The cosine-2s spreading function distributes wave energy across
directions around the dominant wave heading.
"""

import numpy as np
from scipy.special import gamma as gamma_func


def cosine_squared_spreading(
    theta: np.ndarray,
    theta_0: float,
    s: float,
) -> np.ndarray:
    """Cosine-2s directional spreading function D(θ).

    Normalised so that ∫_{-π}^{π} D(θ) dθ = 1.

    D(θ) = C(s) · cos²ˢ( (θ − θ₀) / 2 )

    with the normalisation constant

        C(s) = 2^{2s−1} / π · Γ(s+1)² / Γ(2s+1)

    Args:
        theta: Array of wave directions [rad].
        theta_0: Dominant (mean) wave direction [rad].
        s: Spreading factor (≥ 1).  Higher *s* gives a narrower spread.
           Typical values: 1 (very wide), 2–4 (moderate), 10+ (swell).

    Returns:
        Normalised directional spreading density D(θ) [1/rad].
    """
    if s < 1.0:
        raise ValueError(f"spreading_factor (s) must be ≥ 1, got {s}")

    theta = np.asarray(theta, dtype=np.float64)
    dtheta = theta - theta_0

    # Normalisation constant
    # C = Γ(s+1) / (2√π Γ(s+1/2))   (common alternative form)
    # Using the standard form:
    C = (2.0 ** (2.0 * s - 1.0) / np.pi) \
        * (gamma_func(s + 1.0) ** 2) / gamma_func(2.0 * s + 1.0)

    D = C * (np.cos(0.5 * dtheta)) ** (2.0 * s)

    # Clip small negative values from floating-point
    return np.maximum(D, 0.0)


def discretize_directions(
    theta_0: float,
    n_directions: int,
    s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize directions and compute spreading weights.

    Args:
        theta_0: Dominant wave direction [rad].
        n_directions: Number of directional bins.
        s: Spreading factor.

    Returns:
        (theta_array, weights): Direction angles [rad] and normalised
        spreading weights summing to 1.
    """
    theta = np.linspace(-np.pi, np.pi, n_directions, endpoint=False)
    weights = cosine_squared_spreading(theta, theta_0, s)
    weights /= weights.sum()
    return theta, weights

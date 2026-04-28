"""
Wave energy spectra: JONSWAP and Pierson-Moskowitz.

References:
    - Hasselmann, K. et al. (1973) "Measurements of wind-wave growth and
      swell decay during JONSWAP." Ergänzungsheft 8-12.
    - Pierson, W.J. & Moskowitz, L. (1964) "A proposed spectral form for
      fully developed wind seas." J. Geophys. Res. 69(24).
"""

from abc import ABC, abstractmethod
import numpy as np

GRAVITY = 9.81  # [m/s²]


class WaveSpectrum(ABC):
    """Abstract base class for 1D wave energy spectra.

    All spectra are functions of angular frequency ω [rad/s] and return
    spectral density S(ω) in [m²·s].
    """

    @abstractmethod
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the spectral density.

        Args:
            omega: Angular frequency or array of frequencies [rad/s].

        Returns:
            Spectral density S(ω) [m²·s], same shape as *omega*.
        """
        ...

    @property
    @abstractmethod
    def significant_wave_height(self) -> float:
        ...

    @property
    @abstractmethod
    def peak_frequency(self) -> float:
        ...


class JONSWAP(WaveSpectrum):
    """JONSWAP spectrum for developing seas.

    Args:
        Hs: Significant wave height [m].
        Tp: Peak period [s].
        gamma: Peak enhancement factor (1.0 = PM spectrum, 3.3 = standard JONSWAP).
    """

    def __init__(self, Hs: float, Tp: float, gamma: float = 3.3) -> None:
        if Hs <= 0:
            raise ValueError(f"Hs must be positive, got {Hs}")
        if Tp <= 0:
            raise ValueError(f"Tp must be positive, got {Tp}")
        if gamma < 1.0:
            raise ValueError(f"gamma must be >= 1.0, got {gamma}")

        self._Hs = Hs
        self._Tp = Tp
        self._gamma = gamma
        self._omega_p = 2.0 * np.pi / Tp

        # Normalisation factor α following Goda (2010)
        self._compute_alpha()

    def _compute_alpha(self) -> None:
        """Compute Phillips constant α from Hs constraint: Hm0 ≈ 4√(m0)."""
        # Iterative approach — integrate numerically
        omegas = np.linspace(0.01, 5.0 * self._omega_p, 2000)
        S = self._raw_spectrum(omegas)
        m0 = np.trapz(S, omegas)
        # Hs = 4 * sqrt(m0) → adjust alpha to match
        # We integrate at call time; store the raw shape and let alpha
        # be determined by the Hs constraint
        self._alpha_scale = (self._Hs / 4.0) ** 2 / max(m0, 1e-12)

    def _raw_spectrum(self, omega: np.ndarray) -> np.ndarray:
        """Un-normalised JONSWAP shape (array-safe, supports scalars)."""
        omega = np.asarray(omega, dtype=np.float64)
        scalar = omega.ndim == 0
        if scalar:
            omega = omega[np.newaxis]

        S = np.zeros_like(omega, dtype=np.float64)
        mask = omega > 0.0
        if not np.any(mask):
            return S.item() if scalar else S

        om = omega[mask]
        sigma = np.where(om <= self._omega_p, 0.07, 0.09)
        r = np.exp(
            -((om - self._omega_p) ** 2) / (2.0 * sigma**2 * self._omega_p**2)
        )
        S[mask] = (
            GRAVITY**2
            * om ** (-5)
            * np.exp(-1.25 * (self._omega_p / om) ** 4)
            * self._gamma ** r
        )
        return S.item() if scalar else S

    def __call__(self, omega: np.ndarray) -> np.ndarray:
        S = self._raw_spectrum(omega) * self._alpha_scale
        S = np.maximum(S, 0.0)
        return S

    @property
    def significant_wave_height(self) -> float:
        return self._Hs

    @property
    def peak_frequency(self) -> float:
        return self._omega_p

    @property
    def peak_period(self) -> float:
        return self._Tp

    @property
    def gamma(self) -> float:
        return self._gamma

    def __repr__(self) -> str:
        return (f"JONSWAP(Hs={self._Hs:.2f}, Tp={self._Tp:.2f}, "
                f"gamma={self._gamma:.2f})")


class PiersonMoskowitz(JONSWAP):
    """Pierson-Moskowitz spectrum — a JONSWAP with gamma=1 (fully developed sea)."""

    def __init__(self, Hs: float, Tp: float) -> None:
        super().__init__(Hs=Hs, Tp=Tp, gamma=1.0)

    def __repr__(self) -> str:
        return f"PiersonMoskowitz(Hs={self._Hs:.2f}, Tp={self._Tp:.2f})"

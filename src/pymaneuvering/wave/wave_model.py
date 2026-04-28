"""
Wave load model — simplified stochastic disturbance model.

This module computes first-order (wave-frequency) and second-order
(mean drift) wave loads on a marine vessel using a discretised
frequency-direction wave spectrum.

**IMPORTANT**: This is a simplified disturbance model based on wave
spectra and directional spreading. It does NOT use vessel-specific
RAOs or full second-order difference-frequency QTFs. For rigorous
hydrodynamic analysis, replace the transfer functions with actual
RAO / QTF data for the target vessel.

Architecture:
    - ``compute_raw_load()``: returns *combined* 1st + 2nd order forces
      for use by the dynamics engine (full physics simulation).
    - ``compute_mean_drift_load()``: returns *only* 2nd-order mean drift
      forces for use by control feedforward / DP systems.

Both outputs are available at every time step — no extra computation
is incurred because the internal spectral decomposition is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .spectrum import WaveSpectrum, GRAVITY

# ---------------------------------------------------------------------------
# Force container types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveLoad:
    """Horizontal-plane wave force/moment vector.

    All quantities in SI: forces in [N], moment in [N·m].
    """
    surge: float    # X-direction force (body-fixed)
    sway: float     # Y-direction force (body-fixed)
    yaw: float      # Yaw moment about midship

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.surge, self.sway, self.yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Transfer-function helpers
# ---------------------------------------------------------------------------

def _deep_water_wavenumber(omega: np.ndarray) -> np.ndarray:
    """Deep-water wave number k = ω²/g [rad/m]."""
    return omega ** 2 / GRAVITY


def _first_order_surge_tf(omega_rel: np.ndarray, ship_beam: float) -> np.ndarray:
    """Non-dimensional first-order surge force transfer function.

    Approximate Froude-Krylov + diffraction envelope.
    At low ω → flat plate reflection; at high ω → decay ∝ ω^{-2}.
    """
    k = _deep_water_wavenumber(omega_rel)
    kb2 = np.clip(0.5 * k * ship_beam, 1e-6, None)
    return np.abs(np.sin(kb2) / kb2) * np.exp(-0.5 * kb2)


def _first_order_sway_tf(omega_rel: np.ndarray, ship_length: float) -> np.ndarray:
    """Non-dimensional first-order sway force transfer function.

    Strip-theory like envelope: long-wave dominance with high-frequency cut-off.
    """
    kl = np.clip(omega_rel ** 2 * ship_length / GRAVITY, 1e-6, None)
    return np.abs(1.0 - np.exp(-kl)) * np.exp(-0.3 * kl)


def _first_order_yaw_tf(omega_rel: np.ndarray, ship_length: float) -> np.ndarray:
    """Non-dimensional first-order yaw moment transfer function.

    Peaked at intermediate frequencies (ship-length resonant band).
    """
    kl = np.clip(omega_rel ** 2 * ship_length / GRAVITY, 1e-6, None)
    return kl * np.exp(-0.5 * kl)


def _drift_surge_tf(omega_rel: np.ndarray, ship_beam: float) -> np.ndarray:
    """Wave drift force coefficient for surge (diagonal QTF approximation).

    Returns non-dimensional coefficient that, when multiplied by
    ½ ρ g B ζ_a², gives the mean surge drift force.
    """
    k = _deep_water_wavenumber(omega_rel)
    return 1.0 - np.exp(-k * ship_beam)


def _drift_sway_tf(omega_rel: np.ndarray, ship_length: float) -> np.ndarray:
    """Wave drift force coefficient for sway."""
    k = _deep_water_wavenumber(omega_rel)
    return 1.0 - np.exp(-0.5 * k * ship_length)


def _drift_yaw_tf(omega_rel: np.ndarray, ship_length: float) -> np.ndarray:
    """Wave drift moment coefficient for yaw."""
    k = _deep_water_wavenumber(omega_rel)
    kl = np.clip(k * ship_length, 1e-6, None)
    return (1.0 - np.exp(-0.25 * kl)) * np.exp(-1.5 / kl)


# ---------------------------------------------------------------------------
# Main wave model
# ---------------------------------------------------------------------------

class WaveModel:
    """Spectral wave-load engine with dual-output architecture.

    This class discretises a directional wave spectrum into *Nf* frequency
    and *Nd* directional components, pre-computes random phases and
    force amplitudes, and evaluates both first-order (raw oscillatory) and
    second-order (slow-drift) wave forces at each call.

    Args:
        spectrum: A ``WaveSpectrum`` instance (e.g. ``JONSWAP``).
        ship_length: Ship length (Lpp or L) [m].
        ship_beam: Ship beam (B) [m].
        ship_draft: Ship draft [m].
        rho: Water density [kg/m³] (default 1025 seawater).
        n_frequencies: Number of frequency components for spectral
            discretization (default 80).
        n_directions: Number of directional bins (default 36 = 10° spacing).
        spreading_factor: Cosine-2s spreading exponent (default 2.0).
        dominant_wave_direction: Mean wave direction [rad], 0 = from North
            (default 0.0).
        seed: Random seed for reproducible phase generation.
        frequency_range: ``(omega_min, omega_max)`` override for the
            frequency discretization range.  Defaults to an automatic
            range covering 99.9 % of spectral energy.
    """

    def __init__(
        self,
        spectrum: WaveSpectrum,
        *,
        ship_length: float,
        ship_beam: float,
        ship_draft: float,
        rho: float = 1025.0,
        n_frequencies: int = 80,
        n_directions: int = 36,
        spreading_factor: float = 2.0,
        dominant_wave_direction: float = 0.0,
        seed: Optional[int] = None,
        frequency_range: Optional[tuple[float, float]] = None,
    ) -> None:
        if n_frequencies < 10:
            raise ValueError("n_frequencies must be >= 10")
        if n_directions < 4:
            raise ValueError("n_directions must be >= 4")
        if spreading_factor < 1.0:
            raise ValueError("spreading_factor must be >= 1")

        self._spectrum = spectrum
        self._L = ship_length
        self._B = ship_beam
        self._d = ship_draft
        self._rho = rho
        self._Nf = n_frequencies
        self._Nd = n_directions
        self._spreading_factor = spreading_factor
        self._theta_0 = dominant_wave_direction

        # --- Frequency discretization ------------------------------------
        omega_p = spectrum.peak_frequency
        if frequency_range is not None:
            omega_min, omega_max = frequency_range
        else:
            omega_min = 0.3 * omega_p
            omega_max = 4.0 * omega_p

        self._omega = np.linspace(omega_min, omega_max, n_frequencies)
        self._domega = self._omega[1] - self._omega[0]

        # --- Directional discretization ----------------------------------
        # Convention: discretized directions self._theta represent wave
        # FROM directions (oceanographic).  self._theta_0 = 0 means waves
        # come FROM North.  Convert to wave-going-TO direction in
        # _relative_wave_direction() via θ_to = θ + π.
        from .spreading import cosine_squared_spreading
        raw_theta = np.linspace(-np.pi, np.pi, n_directions, endpoint=False)
        self._theta = raw_theta
        self._dtheta = raw_theta[1] - raw_theta[0]

        # Build directional probability weights: D(θ) * dθ per bin,
        # normalized to sum to 1.
        D = cosine_squared_spreading(raw_theta, self._theta_0, spreading_factor)
        self._weights_dir = D * self._dtheta
        self._weights_dir /= self._weights_dir.sum()

        # --- Spectral energy distribution --------------------------------
        # E_ij = S(ω_i) * w_j * dω  (w_j already includes dθ, see above)
        S = spectrum(self._omega)
        self._wave_amplitudes = np.sqrt(
            2.0 * np.maximum(S[:, np.newaxis] * self._weights_dir[np.newaxis, :], 0.0)
            * self._domega
        )

        # --- Random phases (fixed for reproducibility) -------------------
        rng = np.random.RandomState(seed)
        self._phases_initial = rng.uniform(0.0, 2.0 * np.pi, size=(n_frequencies, n_directions))
        # Incremental phase tracking for ODE-friendly time evolution
        self._current_phases = self._phases_initial.copy()
        self._phase_t: Optional[float] = None

        # --- Pre-compute frequency-dependent transfer function scales -----
        self._precompute_transfer_functions()

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_transfer_functions(self) -> None:
        """Pre-compute frequency-dependent scaling for all TF types.

        All transfer functions use the intrinsic wave frequency ω
        (NOT encounter frequency ω_e).  This is a simplified disturbance
        model — for a more rigorous treatment, TFs should depend on
        encounter frequency and relative wave direction jointly.
        """
        omega = self._omega[:, np.newaxis]  # (Nf, 1) for broadcasting

        # First-order TF amplitudes
        self._tf1_surge = _first_order_surge_tf(omega, self._B).ravel()
        self._tf1_sway = _first_order_sway_tf(omega, self._L).ravel()
        self._tf1_yaw = _first_order_yaw_tf(omega, self._L).ravel()

        # Second-order TF amplitudes (drift coefficients)
        self._tf2_surge = _drift_surge_tf(omega, self._B).ravel()  # (Nf,)
        self._tf2_sway = _drift_sway_tf(omega, self._L).ravel()
        self._tf2_yaw = _drift_yaw_tf(omega, self._L).ravel()

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _encounter_frequency(
        self, heading: float, speed: float
    ) -> NDArray[np.float64]:
        """Encounter frequency ω_e = ω − ω² U cos(β−ψ) / g.

        β = wave-going-TO direction = self._theta + π.
        ψ = ship heading.

        **Signed** ω_e — negative values indicate waves overtaken by the ship
        (following seas where ship speed exceeds wave celerity).

        Args:
            heading: Ship heading ψ [rad], 0 = North.
            speed: Ship forward speed U [m/s].
        """
        omega_grid = self._omega[:, np.newaxis]  # (Nf, 1)
        # Wave propagation direction (TO): θ_from + π
        theta_to = self._theta[np.newaxis, :] + np.pi  # (1, Nd)
        cos_term = np.cos(theta_to - heading)
        omega_e = omega_grid - (omega_grid ** 2) * speed * cos_term / GRAVITY
        return omega_e  # signed — no abs()

    def _relative_wave_direction(
        self, heading: float
    ) -> NDArray[np.float64]:
        """Relative wave direction per directional bin [rad].

        0 = following sea, π = head sea.

        self._theta stores wave-FROM directions (oceanographic convention).
        Convert to wave-going-TO for the relative direction computation.
        """
        wave_dir_to = self._theta + np.pi  # FROM → TO
        rel_dir = wave_dir_to - heading
        return rel_dir

    def compute_loads(
        self,
        t: float,
        heading: float,
        speed: float,
    ) -> tuple[WaveLoad, WaveLoad]:
        """Compute both raw and drift wave loads at time *t*.

        Uses **incremental phase tracking** for ODE-friendly time evolution.
        Phase evolves as Φ += ω_e·dt rather than the retroactive ω_e·t + φ,
        which avoids phase-jump discontinuities when ship speed or heading
        changes during a maneuvering simulation.

        The drift (second-order) component uses the correct factor ½ ρ g
        (time-averaging of the quadratic pressure term).

        Args:
            t: Current simulation time [s].
            heading: Ship heading ψ [rad] (0 = North).
            speed: Ship forward speed U [m/s].

        Returns:
            ``(raw_load, drift_load)`` — each a ``WaveLoad``.

            *raw_load* = 1st-order (wave-frequency) + 2nd-order (drift).
            *drift_load* = 2nd-order drift forces only.
        """
        omega_e_grid = self._encounter_frequency(heading, speed)  # (Nf, Nd)
        rel_dir = self._relative_wave_direction(heading)  # (Nd,)

        # --- Incremental phase evolution ----------------------------------
        if self._phase_t is None:
            # First call — initialise from stored random phases
            self._current_phases = self._phases_initial.copy()
            dt = 0.0
        else:
            dt = t - self._phase_t
            if dt < 0:
                # ODE solver backtracked — reset phases to avoid corruption
                self._current_phases = self._phases_initial.copy()
                dt = 0.0
        self._phase_t = t

        # Advance phases: dΦ = ω_e · dt  (signed — handles following seas)
        if dt > 0:
            self._current_phases += omega_e_grid * dt
        # Wrap to [0, 2π) for numerical stability
        self._current_phases %= (2.0 * np.pi)

        # --- Directional modulation for 1st-order forces -----------------
        cos_rel = np.cos(rel_dir)[np.newaxis, :]  # (1, Nd)
        sin_rel = np.sin(rel_dir)[np.newaxis, :]
        sin2_rel = np.sin(2.0 * rel_dir)[np.newaxis, :]

        # --- First-order forces (oscillatory) ----------------------------
        cos_phase = np.cos(self._current_phases)

        # Amplitude = wave_amplitude * TF(ω) * directional_factor
        amps = self._wave_amplitudes  # (Nf, Nd)

        # Dimensional scale for 1st-order forces [N] and [N·m]
        scale_surge_1 = self._rho * GRAVITY * self._B * self._d
        scale_sway_1 = self._rho * GRAVITY * self._L * self._d
        scale_yaw_1 = self._rho * GRAVITY * self._L * self._d * self._L / 4.0

        # Sum over frequency × direction
        F1_surge = scale_surge_1 * np.sum(
            amps * self._tf1_surge[:, np.newaxis] * cos_rel * cos_phase
        )
        F1_sway = scale_sway_1 * np.sum(
            amps * self._tf1_sway[:, np.newaxis] * sin_rel * cos_phase
        )
        F1_yaw = scale_yaw_1 * np.sum(
            amps * self._tf1_yaw[:, np.newaxis] * sin2_rel * cos_phase
        )

        # --- Second-order drift forces (slowly-varying) ------------------
        # Mean drift force = ½ ρ g × drift_TF(ω) × ζ_a²
        # The ½ comes from time-averaging ⟨cos²(ωt)⟩ = ½.
        amps_sq = self._wave_amplitudes ** 2  # (Nf, Nd) — ζ_a² per component

        scale_drift = 0.5 * self._rho * GRAVITY

        F2_surge = scale_drift * self._B * np.sum(
            amps_sq * self._tf2_surge[:, np.newaxis] * cos_rel
        )
        F2_sway = scale_drift * self._L * np.sum(
            amps_sq * self._tf2_sway[:, np.newaxis] * sin_rel
        )
        F2_yaw = scale_drift * self._L ** 2 * np.sum(
            amps_sq * self._tf2_yaw[:, np.newaxis] * sin2_rel
        )

        raw_load = WaveLoad(
            surge=F1_surge + F2_surge,
            sway=F1_sway + F2_sway,
            yaw=F1_yaw + F2_yaw,
        )
        drift_load = WaveLoad(
            surge=F2_surge,
            sway=F2_sway,
            yaw=F2_yaw,
        )
        return raw_load, drift_load

    def compute_raw_load(
        self, t: float, heading: float, speed: float
    ) -> WaveLoad:
        """First-order + second-order combined wave load.

        Use this as the forcing term fed to the vessel dynamics ODE.
        """
        raw, _ = self.compute_loads(t, heading, speed)
        return raw

    def compute_mean_drift_load(
        self, t: float, heading: float, speed: float
    ) -> WaveLoad:
        """Mean (time-averaged) second-order wave drift force.

        This is a **mean drift approximation** — it sums over spectral
        components independently and does NOT include difference-frequency
        QTFs or slowly-varying phase coupling.  For true slow-drift
        dynamics, a full QTF-based model is required.

        Use this as a steady feedforward term for DP / heading control.
        """
        _, drift = self.compute_loads(t, heading, speed)
        return drift

    # Backward-compatible alias — use compute_mean_drift_load() for new code.
    compute_drift_load = compute_mean_drift_load

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spectrum(self) -> WaveSpectrum:
        return self._spectrum

    @property
    def spreading_factor(self) -> float:
        return self._spreading_factor

    @spreading_factor.setter
    def spreading_factor(self, value: float) -> None:
        """Update the spreading factor dynamically.

        Re-computes directional weights.  Phases are preserved
        so force time-histories remain continuous across the change.
        """
        if value < 1.0:
            raise ValueError("spreading_factor must be >= 1")
        from .spreading import cosine_squared_spreading
        self._spreading_factor = value
        D = cosine_squared_spreading(self._theta, self._theta_0, value)
        self._weights_dir = D * self._dtheta
        self._weights_dir /= self._weights_dir.sum()
        # Rebuild wave amplitudes with new directional weights
        S = self._spectrum(self._omega)
        self._wave_amplitudes = np.sqrt(
            2.0 * np.maximum(S[:, np.newaxis] * self._weights_dir[np.newaxis, :], 0.0)
            * self._domega
        )

    @property
    def significant_wave_height(self) -> float:
        return self._spectrum.significant_wave_height

    @property
    def peak_period(self) -> float:
        return self._spectrum.peak_period

    @property
    def dominant_wave_direction(self) -> float:
        return self._theta_0

    @property
    def frequencies(self) -> NDArray[np.float64]:
        return self._omega.copy()

    @property
    def directions(self) -> NDArray[np.float64]:
        return self._theta.copy()

    def __repr__(self) -> str:
        return (
            f"WaveModel({self._spectrum!r}, "
            f"L={self._L:.0f}m, B={self._B:.1f}m, "
            f"Nf={self._Nf}, Nd={self._Nd}, "
            f"s={self._spreading_factor:.1f})"
        )

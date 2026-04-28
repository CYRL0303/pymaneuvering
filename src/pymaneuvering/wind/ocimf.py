"""
OCIMF 2010 wind coefficient model for marine vessels.

Replaces the simple trigonometric wind force approximation with
engineering-grade coefficient lookup tables based on OCIMF (Oil Companies
International Marine Forum) standard data for tankers / VLCCs.

Reference:
    OCIMF (2010) "Prediction of Wind and Current Loads on VLCCs."
    2nd edition, Witherby Seamanship International.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, ClassVar

import numpy as np


def _wrap_angle_360(angle_deg: float) -> float:
    """Wrap an angle (degrees) into [0, 360)."""
    return angle_deg % 360.0


def _linear_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linear interpolation with periodic wrapping."""
    return float(np.interp(x, xp, fp, period=360.0))


# ---------------------------------------------------------------------------
# OCIMF 2010 wind coefficients for VLCC — loaded condition
# Source: OCIMF (2010), "Prediction of Wind and Current Loads on VLCCs"
#
# Table layout: 0° to 350° in 10° steps (36 nodes).
# Cx — longitudinal wind force coefficient
# Cy — lateral wind force coefficient
# Cn — yaw moment coefficient
# ---------------------------------------------------------------------------
_OCIMF_ANGLE_NODES: np.ndarray = np.arange(0., 360., 10., dtype=np.float64)

_OCIMF_CX_TANKER_LOADED = np.array([
     0.90,  0.90,  0.86,  0.75,  0.61,  0.44,
     0.26,  0.05, -0.15, -0.35, -0.53, -0.65,
    -0.72, -0.75, -0.75, -0.72, -0.70, -0.68,
    -0.65, -0.68, -0.70, -0.72, -0.75, -0.75,
    -0.72, -0.65, -0.53, -0.35, -0.15,  0.05,
     0.26,  0.44,  0.61,  0.75,  0.86,  0.90,
], dtype=np.float64)

_OCIMF_CY_TANKER_LOADED = np.array([
     0.00,  0.10,  0.30,  0.45,  0.55,  0.65,
     0.72,  0.73,  0.72,  0.70,  0.68,  0.65,
     0.58,  0.48,  0.38,  0.28,  0.18,  0.08,
     0.00, -0.08, -0.18, -0.28, -0.38, -0.48,
    -0.58, -0.65, -0.68, -0.70, -0.72, -0.73,
    -0.72, -0.65, -0.55, -0.45, -0.30, -0.10,
], dtype=np.float64)

_OCIMF_CN_TANKER_LOADED = np.array([
     0.000,  0.016,  0.028,  0.038,  0.047,  0.048,
     0.040,  0.030,  0.022,  0.010,  0.000, -0.012,
    -0.020, -0.028, -0.035, -0.037, -0.033, -0.022,
     0.000,  0.022,  0.033,  0.037,  0.035,  0.028,
     0.020,  0.012,  0.000, -0.010, -0.022, -0.030,
    -0.040, -0.048, -0.047, -0.038, -0.028, -0.016,
], dtype=np.float64)

# Ballasted condition.
# IMPORTANT: These ballasted coefficients are APPROXIMATE — scaled from
# the loaded-condition table and NOT validated against OCIMF engineering
# data.  They are adequate as a simplified disturbance model for
# simulation studies but should NOT be treated as authoritative OCIMF
# engineering values.  For production DP / mooring analysis, use the
# loaded table or obtain validated ballasted data from OCIMF (2010).
_OCIMF_CX_TANKER_BALLASTED = np.array([
     0.80,  0.78,  0.73,  0.65,  0.55,  0.42,
     0.28,  0.12, -0.08, -0.28, -0.48, -0.60,
    -0.68, -0.70, -0.70, -0.68, -0.65, -0.62,
    -0.60, -0.62, -0.65, -0.68, -0.70, -0.70,
    -0.68, -0.60, -0.48, -0.28, -0.08,  0.12,
     0.28,  0.42,  0.55,  0.65,  0.73,  0.78,
], dtype=np.float64)

_OCIMF_CY_TANKER_BALLASTED = np.array([
     0.00,  0.08,  0.25,  0.40,  0.52,  0.60,
     0.68,  0.72,  0.75,  0.74,  0.70,  0.65,
     0.58,  0.50,  0.42,  0.30,  0.18,  0.08,
     0.00, -0.08, -0.18, -0.30, -0.42, -0.50,
    -0.58, -0.65, -0.70, -0.74, -0.75, -0.72,
    -0.68, -0.60, -0.52, -0.40, -0.25, -0.08,
], dtype=np.float64)

_OCIMF_CN_TANKER_BALLASTED = np.array([
     0.000,  0.012,  0.024,  0.034,  0.044,  0.048,
     0.042,  0.034,  0.024,  0.010, -0.004, -0.016,
    -0.024, -0.030, -0.035, -0.036, -0.030, -0.020,
     0.000,  0.020,  0.030,  0.036,  0.035,  0.030,
     0.024,  0.016,  0.004, -0.010, -0.024, -0.034,
    -0.042, -0.048, -0.044, -0.034, -0.024, -0.012,
], dtype=np.float64)


class LoadingCondition:
    """Vessel loading condition selector."""
    LOADED = "loaded"
    BALLASTED = "ballasted"


# Pre-built lookup tables
_OCIMF_TABLES: dict = {
    LoadingCondition.LOADED: {
        "Cx": _OCIMF_CX_TANKER_LOADED,
        "Cy": _OCIMF_CY_TANKER_LOADED,
        "Cn": _OCIMF_CN_TANKER_LOADED,
    },
    LoadingCondition.BALLASTED: {
        "Cx": _OCIMF_CX_TANKER_BALLASTED,
        "Cy": _OCIMF_CY_TANKER_BALLASTED,
        "Cn": _OCIMF_CN_TANKER_BALLASTED,
    },
}


@dataclass(frozen=True)
class WindCoefficients:
    """Wind force/moment coefficients at a given relative wind angle.

    All coefficients are **non-dimensional** and must be multiplied by
    ½ ρ_air V_rw² A_proj to obtain dimensional forces / moments.
    """
    Cx: float   # Longitudinal wind force coefficient
    Cy: float   # Lateral wind force coefficient
    Cn: float   # Yaw moment coefficient


class OCIMFWindModel:
    """OCIMF 2010 wind load model for VLCC/tanker-type vessels.

    Uses 36-node lookup tables (0°–350° at 10° spacing) with linear
    interpolation for intermediate angles.

    Args:
        loading_condition: ``"loaded"`` or ``"ballasted"``.
        rho_air: Air density [kg/m³] (default 1.225).
        A_Fw: Frontal projected wind area [m²].
        A_Lw: Lateral projected wind area [m²].
        Lpp: Length between perpendiculars [m] (for yaw moment scaling).
    """

    _angle_nodes: ClassVar[np.ndarray] = _OCIMF_ANGLE_NODES

    def __init__(
        self,
        loading_condition: str = LoadingCondition.LOADED,
        *,
        rho_air: float = 1.225,
        A_Fw: float,
        A_Lw: float,
        Lpp: float,
    ) -> None:
        if loading_condition not in _OCIMF_TABLES:
            raise ValueError(
                f"Unknown loading condition '{loading_condition}'. "
                f"Choose '{LoadingCondition.LOADED}' or '{LoadingCondition.BALLASTED}'."
            )
        if A_Fw <= 0 or A_Lw <= 0:
            raise ValueError("Projected areas must be positive.")
        if Lpp <= 0:
            raise ValueError("Lpp must be positive.")

        table = _OCIMF_TABLES[loading_condition]
        self._Cx_table = table["Cx"]
        self._Cy_table = table["Cy"]
        self._Cn_table = table["Cn"]
        self._rho_air = rho_air
        self._A_Fw = A_Fw
        self._A_Lw = A_Lw
        self._Lpp = Lpp

    def coefficients(self, relative_wind_angle_rad: float) -> WindCoefficients:
        """Look up wind coefficients for a given relative wind angle.

        Args:
            relative_wind_angle_rad: Relative wind angle [rad], defined as
                the angle from the bow to the incident wind (0 = head wind,
                π/2 = beam wind from starboard, etc.).

        Returns:
            ``WindCoefficients(Cx, Cy, Cn)`` — non-dimensional coefficients.
        """
        angle_deg = _wrap_angle_360(math.degrees(relative_wind_angle_rad))
        Cx = _linear_interp(angle_deg, self._angle_nodes, self._Cx_table)
        Cy = _linear_interp(angle_deg, self._angle_nodes, self._Cy_table)
        Cn = _linear_interp(angle_deg, self._angle_nodes, self._Cn_table)
        return WindCoefficients(Cx=float(Cx), Cy=float(Cy), Cn=float(Cn))

    def compute_forces(
        self,
        relative_wind_speed: float,
        relative_wind_angle_rad: float,
    ) -> tuple[float, float, float]:
        """Compute dimensional wind forces and yaw moment.

        **Sign convention (body-fixed frame, +X = forward, +Y = starboard):**

        - Head wind (0°) → X_W < 0 (opposes forward motion)
        - Tail wind (180°) → X_W > 0 (pushes forward)
        - Beam wind from starboard → Y_W < 0 (pushes to port)
        - Beam wind from port → Y_W > 0 (pushes to starboard)

        OCIMF coefficients are defined such that the force acts in the
        direction of the incident wind.  In the body frame, the force on
        the vessel points away from the incident wind direction, so all
        three outputs are negated relative to the raw (Cx, Cy, Cn).

        Args:
            relative_wind_speed: Relative wind speed = |V_wind − V_ship| [m/s].
            relative_wind_angle_rad: Relative wind angle [rad] (0 = head wind).

        Returns:
            ``(X_W, Y_W, N_W)`` in [N] and [N·m].
        """
        coeffs = self.coefficients(relative_wind_angle_rad)
        q = 0.5 * self._rho_air * relative_wind_speed ** 2
        # OCIMF coefficients give force in direction of wind.
        # In body frame, wind pushes vessel AWAY from incident direction.
        X_W = -q * coeffs.Cx * self._A_Fw
        Y_W = -q * coeffs.Cy * self._A_Lw
        N_W = -q * coeffs.Cn * self._A_Lw * self._Lpp
        return X_W, Y_W, N_W

    def compute_from_true_wind(
        self,
        true_wind_speed: float,
        true_wind_direction_from: float,
        ship_heading: float,
        ship_u: float,
        ship_v: float = 0.0,
    ) -> tuple[float, float, float]:
        """Compute wind forces from true (earth-fixed) wind.

        Converts true wind to apparent wind internally, reducing the
        risk of angle-convention errors in calling code.

        **Convention:**
        - ``true_wind_direction_from``: direction wind comes FROM [rad],
          0 = from North, increasing clockwise (oceanographic).
        - ``ship_heading``: vessel heading ψ [rad], 0 = North.
        - ``ship_u``: surge speed [m/s] (forward positive).
        - ``ship_v``: sway speed [m/s] (starboard positive).

        Args:
            true_wind_speed: True wind speed [m/s].
            true_wind_direction_from: Direction wind is coming FROM [rad].
            ship_heading: Ship heading ψ [rad].
            ship_u: Ship surge velocity [m/s].
            ship_v: Ship sway velocity [m/s].

        Returns:
            ``(X_W, Y_W, N_W)`` in [N] and [N·m].
        """
        # True wind velocity in earth frame (wind blowing TO direction).
        # Oceanographic convention (0=North, clockwise) → math convention
        # (0=East, CCW) via: East = sin(θ), North = cos(θ).
        # Wind TO = wind FROM + π, so:
        #   East = speed * sin(θ_from + π) = -speed * sin(θ_from)
        #   North = speed * cos(θ_from + π) = -speed * cos(θ_from)
        w_earth_e = -true_wind_speed * math.sin(true_wind_direction_from)
        w_earth_n = -true_wind_speed * math.cos(true_wind_direction_from)

        # Ship velocity in earth frame.
        # Heading ψ [rad] (0=North, oceanographic).
        # Forward (surge): direction ψ → East=sin(ψ), North=cos(ψ).
        # Starboard (sway): direction ψ−π/2 → East=cos(ψ), North=−sin(ψ).
        ship_earth_e = (ship_u * math.sin(ship_heading)
                        + ship_v * math.cos(ship_heading))
        ship_earth_n = (ship_u * math.cos(ship_heading)
                        - ship_v * math.sin(ship_heading))

        # Apparent wind in earth frame
        app_e = w_earth_e - ship_earth_e
        app_n = w_earth_n - ship_earth_n

        # Rotate apparent wind into body frame
        # Earth (E,N) → Body (forward=X, starboard=Y) by heading ψ
        cos_psi = math.cos(ship_heading)
        sin_psi = math.sin(ship_heading)
        app_body_x = cos_psi * app_n + sin_psi * app_e   # forward component
        app_body_y = -sin_psi * app_n + cos_psi * app_e  # starboard component

        # Apparent wind velocity in body frame is [app_body_x, app_body_y]
        # (direction wind is blowing TOWARD).
        # OCIMF convention: angle FROM which wind is coming, from bow.
        # Convert "blowing TOWARD" → "coming FROM" by adding π.
        apparent_speed = math.sqrt(app_body_x ** 2 + app_body_y ** 2)
        gamma_rw = math.atan2(app_body_y, app_body_x) + math.pi
        # Normalize to [0, 2π)
        gamma_rw = gamma_rw % (2.0 * math.pi)

        return self.compute_forces(apparent_speed, gamma_rw)

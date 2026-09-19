"""
geometry.py — angles-only observation geometry (Pastor 2022, Eqs. 2.1-2.3).

r = R + rho*L relates a topocentric angle-only observation (ra, dec) to an
inertial position vector, given the observer's position R and an assumed
range rho. This module provides that relation and its inverse (predicted
angles from an inertial position, used to compute residuals against
observations during the double-r iteration).

All angles are degrees at the public boundary (matching every other
angle-only interface in this repo -- catalog_store, associator, etc.);
internally everything is radians.

Author: Peter Thomas
"""
from __future__ import annotations

import numpy as np


def line_of_sight(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Unit pointing vector L(alpha, delta) -- Eq. 2.2."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])


def slant_range(R: np.ndarray, L: np.ndarray, r_mag: float) -> float:
    """
    Solve |R + rho*L| = r_mag for rho (> 0) -- Eq. 2.3.

    Raises ValueError if r_mag is too small for the line of sight from R to
    reach a sphere of that radius (the ray's closest approach to the origin
    exceeds r_mag).
    """
    c = 2.0 * np.dot(R, L)
    discriminant = c * c - 4.0 * (np.dot(R, R) - r_mag * r_mag)
    if discriminant < 0.0:
        raise ValueError(
            f"no real slant range for r_mag={r_mag}: line of sight never reaches that radius"
        )
    rho = (-c + np.sqrt(discriminant)) / 2.0
    if rho <= 0.0:
        # The other root can be the physical one for an observer already
        # past closest approach along -L; fall back to it if positive.
        rho_alt = (-c - np.sqrt(discriminant)) / 2.0
        if rho_alt > 0.0:
            return rho_alt
        raise ValueError(f"no positive slant range solution for r_mag={r_mag}")
    return rho


def position_vector(R: np.ndarray, rho: float, L: np.ndarray) -> np.ndarray:
    """r = R + rho*L -- Eq. 2.1."""
    return R + rho * L


def radec_from_position(r: np.ndarray, R: np.ndarray) -> tuple:
    """
    Inverse of position_vector: predicted (ra_deg, dec_deg) an observer at R
    would measure for an object at inertial position r. Used to form
    measurement residuals against real observations (the H(.) operator in
    Pastor 2022 Eq. 2.9).
    """
    los = np.asarray(r) - np.asarray(R)
    los_mag = np.linalg.norm(los)
    ra = np.degrees(np.arctan2(los[1], los[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(los[2] / los_mag, -1.0, 1.0)))
    return ra, dec


def wrapped_residual_deg(observed_deg: float, predicted_deg: float) -> float:
    """
    Angular residual observed - predicted, wrapped to (-180, 180] deg to
    avoid discontinuities near the 0/360 seam -- Eq. 2.10.
    """
    diff = np.radians(observed_deg - predicted_deg)
    return np.degrees(np.arctan2(np.sin(diff), np.cos(diff)))

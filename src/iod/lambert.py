"""
lambert.py — two-body Lambert boundary-value solver via the universal-
variable formulation (Vallado, Algorithm 58/59): given two position
vectors and a time of flight, find the two velocity vectors of the
connecting Keplerian orbit.

This is the operator L referenced throughout Pastor (2022) Ch. 2 --
double_r_lambert.py calls it each iteration with the current (r1, rn)
position-vector guesses. The paper cites Izzo's solver as its concrete
implementation choice; this module solves the identical boundary-value
problem via the classical universal-variable formulation instead, which
is independently well-established and is validated here against the
kepler.py propagator (propagate a known orbit to get two truth position
vectors, then check Lambert recovers the known velocities).

Scope: single-revolution (0-rev) transfers only, transfer angle != pi,
prograde motion assumed (consistent with the "direct-orbit" convention
used throughout this package, and with the overwhelmingly prograde
cataloged-object population Pastor cites in the Circular method section).
Multi-revolution and retrograde transfers are not implemented.

Author: Peter Thomas
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from iod.elements import MU_EARTH_KM3_S2


def _stumpff_cs(z: float) -> tuple:
    """Stumpff functions C(z), S(z) (distinct convention from kepler.py's C2/C3, same functions)."""
    if z > 1e-6:
        sqrt_z = np.sqrt(z)
        c = (1.0 - np.cos(sqrt_z)) / z
        s = (sqrt_z - np.sin(sqrt_z)) / sqrt_z ** 3
    elif z < -1e-6:
        sqrt_nz = np.sqrt(-z)
        c = (1.0 - np.cosh(sqrt_nz)) / z
        s = (np.sinh(sqrt_nz) - sqrt_nz) / sqrt_nz ** 3
    else:
        c = 1.0 / 2.0 - z / 24.0 + z ** 2 / 720.0
        s = 1.0 / 6.0 - z / 120.0 + z ** 2 / 5040.0
    return c, s


def solve(
    r1: np.ndarray, r2: np.ndarray, dt: float,
    mu: float = MU_EARTH_KM3_S2, prograde: bool = True,
) -> tuple:
    """
    Solve Lambert's problem for the velocities at r1 and r2 of the
    Keplerian orbit connecting them in time dt (> 0).

    Returns (v1, v2), each a 3-vector in the same units as r1/r2 per dt.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if dt <= 0.0:
        raise ValueError("Lambert time of flight must be positive")

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1.0, 1.0)
    cross_z = np.cross(r1, r2)[2]

    if prograde:
        dnu = np.arccos(cos_dnu) if cross_z >= 0 else 2.0 * np.pi - np.arccos(cos_dnu)
    else:
        dnu = np.arccos(cos_dnu) if cross_z < 0 else 2.0 * np.pi - np.arccos(cos_dnu)

    sin_dnu = np.sin(dnu)
    if abs(sin_dnu) < 1e-10:
        raise ValueError("Lambert transfer angle too close to 0 or pi (singular geometry)")

    A = sin_dnu * np.sqrt(r1_mag * r2_mag / (1.0 - cos_dnu))
    if A == 0.0:
        raise ValueError("Lambert 'A' parameter is zero (degenerate geometry)")

    sqrt_mu = np.sqrt(mu)

    def _y(z):
        c, s = _stumpff_cs(z)
        return r1_mag + r2_mag + A * (z * s - 1.0) / np.sqrt(c)

    def _tof(z):
        c, s = _stumpff_cs(z)
        y = _y(z)
        chi = np.sqrt(y / c)
        return (chi ** 3 * s + A * np.sqrt(y)) / sqrt_mu

    def _residual(z):
        return _tof(z) - dt

    # For 0-rev transfers, tof(z) increases monotonically from ~0 (very
    # hyperbolic, z -> -inf) to +inf as z approaches (2*pi)^2 (the next
    # multi-rev boundary), so this fixed bracket covers every physically
    # reachable positive dt without needing adaptive bracket search.
    z_lo, z_hi = -4.0 * (2.0 * np.pi) ** 2, (2.0 * np.pi) ** 2 * 0.999999
    while _y(z_lo) < 0.0:
        z_lo *= 0.5  # move z_lo closer to 0 until y(z) is valid there too

    f_lo, f_hi = _residual(z_lo), _residual(z_hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            f"Lambert solver could not bracket a 0-rev solution for dt={dt}s "
            f"(residuals {f_lo:.3g}, {f_hi:.3g} same sign)"
        )

    z = brentq(_residual, z_lo, z_hi, xtol=1e-12, rtol=1e-14, maxiter=200)

    c, s = _stumpff_cs(z)
    y = _y(z)

    f = 1.0 - y / r1_mag
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2_mag

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g

    return v1, v2

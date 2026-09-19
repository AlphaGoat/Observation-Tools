"""
circular.py — Circular method (Pastor 2022, Sec. 2.3.1 / Algorithm 2),
restricted to the two-observation case double_r_lambert.py needs to seed
its iteration (Algorithm 3, step 1: "solve the circular problem between
observation 1 and n").

Assumes the orbit is (near-)circular, i.e. |r1| = |r2| = r, and solves for
the single scalar r that makes the circular-orbit time of flight between
the two line-of-sight directions equal the true elapsed time between the
observations (Eq. 2.6). This is *only* used as an initial guess -- the
double-r gradient descent refines r1 and rn independently afterwards, so a
rough circular approximation is sufficient even for eccentric orbits (as
demonstrated by Pastor's own GTO validation case, e=0.71).

Scope: direct (prograde) motion, zero completed revolutions between the
two observations -- consistent with lambert.py's scope in this package.
Retrograde detection and the multi-revolution correction described in the
paper are not implemented (documented limitation, acceptable since this
is only ever used as a starting guess for gradient descent refinement).

Author: Peter Thomas
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from iod.elements import MU_EARTH_KM3_S2
from iod.geometry import position_vector, slant_range


def _delta_nu(r_mag: float, R1: np.ndarray, L1: np.ndarray, R2: np.ndarray, L2: np.ndarray) -> float:
    """True-anomaly difference between the two circular-radius position vectors -- Eq. 2.7."""
    rho1 = slant_range(R1, L1, r_mag)
    rho2 = slant_range(R2, L2, r_mag)
    r1_vec = position_vector(R1, rho1, L1)
    r2_vec = position_vector(R2, rho2, L2)

    cos_dnu = np.clip(
        np.dot(r1_vec, r2_vec) / (np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec)), -1.0, 1.0,
    )
    dnu = np.arccos(cos_dnu)
    if np.cross(r1_vec, r2_vec)[2] < 0.0:
        # arccos cannot distinguish > pi transfers; a negative z cross
        # product for an assumed-prograde orbit means the true angle is
        # the reflex angle.
        dnu = 2.0 * np.pi - dnu
    return dnu


def circular_radius(
    R1: np.ndarray, L1: np.ndarray, R2: np.ndarray, L2: np.ndarray, dt: float,
    mu: float = MU_EARTH_KM3_S2, r_min: float = None, r_max: float = 60_000.0,
) -> float:
    """
    Solve Eq. 2.6, f(r) = dt - mu^-0.5 * r^1.5 * delta_nu(r) = 0, for r.

    r_min defaults to just past the larger of the two lines of sight'
    closest approach to the origin (below which no real slant range
    exists); r_max defaults to a bound comfortably past GEO/GTO apogee --
    the domain this package targets (LEO through GEO/GTO RSO cataloguing).
    """
    if r_min is None:
        # Minimum |R + rho*L| reachable at rho >= 0: if the ray already
        # heads away from closest approach (R.L >= 0), that's rho=0, i.e.
        # |R| itself; otherwise it's the perpendicular distance |R x L|.
        def _min_reachable(R, L):
            return np.linalg.norm(R) if np.dot(R, L) >= 0.0 else np.linalg.norm(np.cross(R, L))

        r_min = max(_min_reachable(R1, L1), _min_reachable(R2, L2)) * 1.0001 + 1.0

    def f(r_mag):
        dnu = _delta_nu(r_mag, R1, L1, R2, L2)
        return dt - dnu * np.sqrt(r_mag ** 3 / mu)

    # Eq. 2.6 can have zero, one, or several roots in r depending on the
    # observation geometry and eccentricity (Pastor 2022 notes the same
    # for the general n-pair case: "at most one of them corresponds to the
    # orbit, being the remaining spurious solutions"). Since this is only
    # ever used to seed Algorithm 3's gradient descent, not as a final
    # answer, we don't need the exact root: scan for the smallest-radius
    # sign change (the true solution is the smallest root in practice,
    # larger ones correspond to spurious additional-revolution solutions)
    # and fall back to the closest-to-zero sample if no sign change exists
    # at all in range, rather than failing outright.
    n_samples = 60
    radii = np.geomspace(r_min, r_max, n_samples)
    values = []
    for r in radii:
        try:
            values.append(f(r))
        except ValueError:
            values.append(np.nan)
    values = np.array(values)

    for i in range(len(radii) - 1):
        if np.isnan(values[i]) or np.isnan(values[i + 1]):
            continue
        if values[i] * values[i + 1] <= 0.0:
            return brentq(f, radii[i], radii[i + 1], xtol=1e-6, rtol=1e-12, maxiter=200)

    valid = ~np.isnan(values)
    if not np.any(valid):
        raise RuntimeError(f"circular method: no valid range geometry found in [{r_min:.1f}, {r_max:.1f}] km")
    return float(radii[valid][np.argmin(np.abs(values[valid]))])

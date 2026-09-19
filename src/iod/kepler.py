"""
kepler.py — two-body Kepler propagator via the universal-variable
formulation (Vallado, "Fundamentals of Astrodynamics and Applications",
Algorithm 8). Handles ellipse, parabola and hyperbola with one code path.

This is the propagator ℱ referenced throughout Pastor (2022) Ch. 2 --
double_r_lambert.py propagates the Lambert-derived state at the first
observation forward to every middle observation's epoch to compute
predicted measurements each iteration.

Future work: J2 perturbation
-------------------------------
Two-body dynamics is the right choice for double_r_lambert.py's inner
loop specifically (cheap, closed-form, called many times per solve via
finite differences) -- matching Pastor's own framing of two-body as the
appropriate cost/accuracy tradeoff for IOD, with higher-fidelity models
belonging to the later OD refinement stage (Sec. 2.2.2). But once a
catalog object exists and needs to be *propagated* over longer spans
(correlator predictions days out, scheduling follow-up observations),
neglecting J2 (Earth's oblateness) introduces secular drift this
propagator cannot represent: several deg/day of RAAN/argument-of-perigee
precession for a typical LEO object, essentially unbounded position error
over a multi-day gap.

The natural extension is a J2-perturbed variant of this propagator, not a
replacement:
  - Cheapest, and the better fit for this package's existing style:
    apply the classical J2 secular-rate corrections (Brouwer mean-element
    theory secular equations for RAAN, argument of perigee and mean
    anomaly drift -- the same order of theory as Brouwer-Lyddane/SGP,
    which Pastor lists as a suitable OD-stage compromise) on top of the
    existing closed-form two-body propagate() call, i.e. propagate
    two-body then rotate/advance by the accumulated J2 secular drift for
    that dt. Keeps the same (r, v) in / (r, v) out signature, so every
    caller (including double_r_lambert.py, if ever wanted there) keeps
    working unchanged.
  - More accurate but heavier: numerically integrate the two-body
    equations of motion plus the J2 acceleration term (e.g. via
    scipy.integrate.solve_ivp, already an implicit dependency via scipy's
    presence elsewhere in this package). This breaks the current
    closed-form Kepler-equation solve and is meaningfully more expensive
    per call -- a poor fit for double_r_lambert.py's iterate-many-times
    inner loop, but fine for a one-shot "propagate this confirmed object
    forward N days" call elsewhere (e.g. the correlator or a scheduler).
  - Not the right fit here at all: full SGP4. It expects TLE mean
    elements from a dedicated fitting process, not an arbitrary osculating
    (r, v) -- see "TLE-fitting bridge" below for what that would actually
    require.

Future work: TLE-fitting bridge to SGP4/skyfield
----------------------------------------------------
src/tasking/ already uses skyfield's SGP4 (via EarthSatellite) for
TLE-sourced targets, but nothing here produces a TLE from an IOD/OD
result -- a confirmed catalog object has a Cartesian (r, v) state, and
SGP4 needs mean orbital elements (+ B* drag) at a reference epoch, which
real TLEs get from a dedicated fit against a tracking arc, not a
closed-form osculating-to-mean transform. Two ways to bridge this, in
increasing order of fidelity and implementation cost:
  1. Approximate osculating-to-mean conversion (Brouwer's transformation
     equations) as a one-shot, closed-form estimate of the 6 mean
     elements. Fast, but only as good as a single-epoch approximation can
     be -- no use of the tracking history, no B* estimate.
  2. A proper differential-correction fit: treat the 6 mean elements + B*
     as unknowns, propagate candidate values through SGP4 itself, and
     adjust via Gauss-Newton batch least squares against a set of tracked
     (r, v) or angles -- structurally the same pattern double_r_lambert.py
     already uses (iterate a propagator, linearize via finite differences,
     solve weighted normal equations), just with SGP4 in place of the
     two-body propagator and 7 unknowns in place of 2. This would live in
     its own module (e.g. src/iod/tle_fit.py), not inside kepler.py/
     double_r_lambert.py, since it's a genuinely different propagator and
     a different (harder, higher-dimensional) fitting problem.

Author: Peter Thomas
"""
from __future__ import annotations

import numpy as np

from iod.elements import MU_EARTH_KM3_S2


def _stumpff(psi: float) -> tuple:
    """Stumpff functions C2(psi), C3(psi), robust across psi's sign."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        c2 = (1.0 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / sqrt_psi ** 3
    elif psi < -1e-6:
        sqrt_neg = np.sqrt(-psi)
        c2 = (1.0 - np.cosh(sqrt_neg)) / psi
        c3 = (np.sinh(sqrt_neg) - sqrt_neg) / sqrt_neg ** 3
    else:
        c2 = 1.0 / 2.0 - psi / 24.0 + psi ** 2 / 720.0
        c3 = 1.0 / 6.0 - psi / 120.0 + psi ** 2 / 5040.0
    return c2, c3


def propagate(
    r0: np.ndarray, v0: np.ndarray, dt: float,
    mu: float = MU_EARTH_KM3_S2, tol: float = 1e-10, max_iter: int = 100,
) -> tuple:
    """
    Propagate a Cartesian two-body state (r0, v0) forward (dt > 0) or
    backward (dt < 0) by dt seconds. Returns (r, v) in the same units.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if dt == 0.0:
        return r0.copy(), v0.copy()

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    rdotv0 = np.dot(r0, v0)
    sqrt_mu = np.sqrt(mu)

    alpha = 2.0 / r0_mag - v0_mag ** 2 / mu  # 1/a

    if alpha > 1e-6:  # ellipse (incl. circle)
        chi = sqrt_mu * dt * alpha
    elif alpha < -1e-6:  # hyperbola
        a = 1.0 / alpha
        sign = 1.0 if dt > 0 else -1.0
        chi = sign * np.sqrt(-a) * np.log(
            (-2.0 * mu * alpha * dt)
            / (np.dot(r0, v0) + sign * np.sqrt(-mu * a) * (1.0 - r0_mag * alpha))
        )
    else:  # parabola
        h = np.cross(r0, v0)
        p = np.dot(h, h) / mu
        s = 0.5 * np.arctan(1.0 / (3.0 * np.sqrt(mu / p ** 3) * dt))
        w = np.arctan(np.cbrt(np.tan(s)))
        chi = np.sqrt(p) * 2.0 / np.tan(2.0 * w)

    for _ in range(max_iter):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        r_pred = (
            chi * chi * c2
            + (rdotv0 / sqrt_mu) * chi * (1.0 - psi * c3)
            + r0_mag * (1.0 - psi * c2)
        )
        # F(chi) = sqrt(mu) * t(chi); dF/dchi = r_pred, so the Newton step
        # below is (target - F(chi)) / F'(chi) without an extra 1/sqrt_mu.
        f_val = (
            chi ** 3 * c3
            + (rdotv0 / sqrt_mu) * chi * chi * c2
            + r0_mag * chi * (1.0 - psi * c3)
        )
        chi_new = chi + (sqrt_mu * dt - f_val) / r_pred
        if abs(chi_new - chi) < tol:
            chi = chi_new
            break
        chi = chi_new
    else:
        raise RuntimeError(f"Kepler propagator did not converge after {max_iter} iterations")

    psi = chi * chi * alpha
    c2, c3 = _stumpff(psi)

    f = 1.0 - (chi * chi / r0_mag) * c2
    g = dt - (chi ** 3 / sqrt_mu) * c3
    r = f * r0 + g * v0
    r_mag = np.linalg.norm(r)

    fdot = (sqrt_mu / (r_mag * r0_mag)) * (alpha * chi ** 3 * c3 - chi)
    gdot = 1.0 - (chi * chi / r_mag) * c2
    v = fdot * r0 + gdot * v0

    return r, v

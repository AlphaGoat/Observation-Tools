"""
double_r_lambert.py — Double r-iteration Lambert method (Pastor 2022,
Sec. 2.3.2, Algorithm 3): angles-only initial orbit determination from
n >= 3 topocentric (ra, dec, epoch) observations, requiring no initial
guess or a-priori orbit information.

Unlike classical IOD methods (Gauss, Gooding/double-r) that use exactly
three observations and treat angles as noise-free, this method estimates
just two parameters -- the position-vector magnitudes at the first and
last observation, r1 and rn -- via a Gauss-Newton-style batch least
squares fit against *every* available observation (not just three),
using Lambert's problem to turn a (r1, rn) guess into a full trajectory
via two-body propagation. That parameter minimality (2 unknowns, not 6)
is what keeps the problem well-conditioned even for a single track's worth
of angles, where a full 6-parameter state estimate is usually
ill-conditioned.

Algorithm (Pastor 2022, Algorithm 3)
--------------------------------------
1. Seed r1, rn from the Circular method (circular.py) between the first
   and last observation.
2. Iterate:
   a. Turn (r1, rn) into position vectors via the range equation
      (geometry.slant_range/position_vector).
   b. Solve Lambert's problem for the connecting velocity at t1
      (lambert.solve).
   c. Propagate that state to every middle observation's epoch
      (kepler.propagate) and predict its (ra, dec) (geometry.radec_from_position).
   d. Form weighted angular residuals against the real observations and a
      Jacobian of those residuals w.r.t. (r1, rn) (finite differences here
      -- see note below), then take a Gauss-Newton correction step.
   e. Stop once the weighted objective stops improving.

Jacobian: analytic partials exist (Pastor cites [70]) but require
differentiating through the Lambert solve and Kepler propagation
symbolically. This implementation uses central finite differences
instead -- a standard, well-validated substitute for exactly this kind of
composed-solver Jacobian, at the cost of a few extra function evaluations
per iteration (cheap here: n is small and IOD is not a hot path).

Scope: inherits lambert.py's and circular.py's 0-revolution restriction --
the observation span (first to last epoch) must stay under one orbital
period, or the Lambert/circular solves silently converge to a spurious
same-endpoint-angle orbit from a *different* number of completed
revolutions rather than raising an error. For a GEO/GTO-scale semi-major
axis this means keeping the total track-to-track baseline to at most
several hours; a caller spanning tracks days apart around a lower orbit
(where the period is shorter, so more revolutions fit in the same
wall-clock gap) needs to check this explicitly -- there is no internal
guard for it, since the true period isn't known until after solving.

Output: this method's stated deliverable is not a single-point estimate
but "a full estimation (mean and covariance)" of (r1, rn) (Pastor 2022,
p.29) -- reported here as range_covariance, a 2x2 matrix from the batch
least-squares normal equations, alongside the Cartesian state vector
(r, v) at the first observation's epoch that (r1, rn) implies. A full 6x6
Cartesian state covariance is not derived (would need an additional
Jacobian chain from (r1, rn) to (r, v)) -- a documented gap, not required
by the paper's own definition of the method's output.

Author: Peter Thomas
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from iod import kepler, lambert
from iod.circular import circular_radius
from iod.elements import MU_EARTH_KM3_S2
from iod.geometry import line_of_sight, position_vector, radec_from_position, slant_range, wrapped_residual_deg


@dataclass
class Observation:
    ra_deg: float
    dec_deg: float
    epoch_s: float
    site_position_km: np.ndarray

    def __post_init__(self):
        self.site_position_km = np.asarray(self.site_position_km, dtype=float).reshape(3)


@dataclass
class IODResult:
    epoch_s: float
    r_km: np.ndarray
    v_km_s: np.ndarray
    r1_mag_km: float
    rn_mag_km: float
    range_covariance: Optional[np.ndarray]
    rms_deg: float
    n_iterations: int
    converged: bool


class IODError(RuntimeError):
    """Raised when the double-r iteration fails to produce a usable solution."""


def _predict(r1: float, rn: float, obs: List[Observation], L: List[np.ndarray], mu: float):
    """
    Given scalar range-magnitude guesses (r1, rn), build the implied
    trajectory and predict (ra, dec) at every middle observation.

    Returns (r1_vec, v1_vec, [(ra_pred, dec_pred), ...]) for obs[1:-1].
    """
    R0, Rn = obs[0].site_position_km, obs[-1].site_position_km
    dt_total = obs[-1].epoch_s - obs[0].epoch_s

    rho1 = slant_range(R0, L[0], r1)
    rhon = slant_range(Rn, L[-1], rn)
    r1_vec = position_vector(R0, rho1, L[0])
    rn_vec = position_vector(Rn, rhon, L[-1])

    v1_vec, _ = lambert.solve(r1_vec, rn_vec, dt_total, mu=mu)

    predictions = []
    for i in range(1, len(obs) - 1):
        r_i, _ = kepler.propagate(r1_vec, v1_vec, obs[i].epoch_s - obs[0].epoch_s, mu=mu)
        predictions.append(radec_from_position(r_i, obs[i].site_position_km))

    return r1_vec, v1_vec, predictions


def _safe_inv(matrix: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def _fd_jacobian_column(r1, rn, is_r1, obs, L, mu, h0):
    """
    Central-difference d(residual)/dr for one of the two parameters. Falls
    back to a one-sided difference if one direction lands on an
    unreachable geometry (e.g. a range below the line of sight's minimum
    reachable radius -- geometry.slant_range's ValueError), which happens
    routinely near that boundary early in the iteration; shrinks the step
    first in case the boundary is merely nearby rather than immediately
    adjacent to the current estimate.
    """
    base_residual = None

    def _pred(dr1, drn):
        return _residuals(_predict(r1 + dr1, rn + drn, obs, L, mu)[2], obs)

    h = h0
    for _ in range(6):
        dr = (h, 0.0) if is_r1 else (0.0, h)
        try:
            res_plus = _pred(*dr)
        except (ValueError, RuntimeError):
            res_plus = None
        try:
            res_minus = _pred(*(-dr[0], -dr[1]))
        except (ValueError, RuntimeError):
            res_minus = None

        if res_plus is not None and res_minus is not None:
            return (res_minus - res_plus) / (2.0 * h)
        if res_plus is not None:
            if base_residual is None:
                base_residual = _pred(0.0, 0.0)
            return (base_residual - res_plus) / h
        if res_minus is not None:
            if base_residual is None:
                base_residual = _pred(0.0, 0.0)
            return (res_minus - base_residual) / h
        h *= 0.5
    raise IODError("could not compute finite-difference Jacobian column: perturbation always invalid")


def _residuals(predictions, obs: List[Observation]) -> np.ndarray:
    """Flattened [Δra_1, Δdec_1, Δra_2, Δdec_2, ...] against obs[1:-1] -- Eq. 2.10."""
    out = []
    for (ra_pred, dec_pred), o in zip(predictions, obs[1:-1]):
        out.append(wrapped_residual_deg(o.ra_deg, ra_pred))
        out.append(wrapped_residual_deg(o.dec_deg, dec_pred))
    return np.array(out)


def solve(
    observations: List[Observation],
    mu: float = MU_EARTH_KM3_S2,
    sigma_ra_deg: float = 1.0 / 3600.0,
    sigma_dec_deg: float = 1.0 / 3600.0,
    max_iter: int = 30,
    tol: float = 1e-10,
    fd_step_km: float = 1.0,
    max_step_frac: float = 0.5,
) -> IODResult:
    """
    Run the double r-iteration Lambert method on n >= 3 angles-only
    observations. Observations are sorted by epoch internally; the first
    and last anchor the Lambert solve, all others (including any beyond
    the first/last) contribute residuals refining (r1, rn).
    """
    obs = sorted(observations, key=lambda o: o.epoch_s)
    if len(obs) < 3:
        raise IODError(f"double r-iteration Lambert method requires n >= 3 observations, got {len(obs)}")

    L = [line_of_sight(o.ra_deg, o.dec_deg) for o in obs]
    dt_total = obs[-1].epoch_s - obs[0].epoch_s
    if dt_total <= 0.0:
        raise IODError("observations must span a positive time interval")

    r1 = rn = circular_radius(obs[0].site_position_km, L[0], obs[-1].site_position_km, L[-1], dt_total, mu=mu)

    W_diag = np.tile([1.0 / sigma_ra_deg ** 2, 1.0 / sigma_dec_deg ** 2], len(obs) - 2)

    n_iter = 0
    converged = False
    range_covariance = None

    for k in range(1, max_iter + 1):
        n_iter = k
        _, _, predictions = _predict(r1, rn, obs, L, mu)
        residual = _residuals(predictions, obs)
        J = float(np.sum(W_diag * residual ** 2))

        # Finite-difference Jacobian of the residual vector w.r.t. (r1, rn).
        H = np.zeros((len(residual), 2))
        H[:, 0] = _fd_jacobian_column(r1, rn, True, obs, L, mu, fd_step_km)
        H[:, 1] = _fd_jacobian_column(r1, rn, False, obs, L, mu, fd_step_km)

        W = np.diag(W_diag)
        HtW = H.T @ W
        normal_matrix = HtW @ H
        try:
            delta = np.linalg.solve(normal_matrix, HtW @ residual)
        except np.linalg.LinAlgError:
            raise IODError("normal matrix is singular -- observations may be insufficiently spaced")

        max_delta_r1 = max_step_frac * r1
        max_delta_rn = max_step_frac * rn
        delta[0] = np.clip(delta[0], -max_delta_r1, max_delta_r1)
        delta[1] = np.clip(delta[1], -max_delta_rn, max_delta_rn)

        # Backtracking line search: only accept a step that actually
        # reduces the weighted objective. The raw Gauss-Newton step can
        # badly overshoot in the strongly non-linear regions Pastor's
        # Figure 2.4 illustrates, especially from a rough circular seed;
        # a fixed fractional clip alone isn't enough to prevent divergence
        # there since a wrong-direction step just keeps re-clipping and
        # walking further away every iteration.
        accepted = False
        step_scale = 1.0
        r1_trial, rn_trial, J_trial = r1, rn, J
        for _ in range(20):
            r1_trial = max(r1 + step_scale * delta[0], fd_step_km * 2)
            rn_trial = max(rn + step_scale * delta[1], fd_step_km * 2)
            try:
                _, _, pred_trial = _predict(r1_trial, rn_trial, obs, L, mu)
                resid_trial = _residuals(pred_trial, obs)
                J_trial = float(np.sum(W_diag * resid_trial ** 2))
            except (ValueError, RuntimeError):
                J_trial = np.inf
            if J_trial < J:
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            # Stuck: no step, however small, improves the objective --
            # treat the current (r1, rn) as the converged solution.
            converged = True
            range_covariance = _safe_inv(normal_matrix)
            break

        improved = abs(J - J_trial) < tol
        r1, rn = r1_trial, rn_trial
        if improved:
            converged = True
            range_covariance = _safe_inv(normal_matrix)
            break

    if range_covariance is None:
        range_covariance = _safe_inv(normal_matrix)

    r1_vec, v1_vec, predictions = _predict(r1, rn, obs, L, mu)
    final_residual = _residuals(predictions, obs)
    rms_deg = float(np.sqrt(np.mean(final_residual ** 2))) if len(final_residual) else 0.0

    return IODResult(
        epoch_s=obs[0].epoch_s,
        r_km=r1_vec,
        v_km_s=v1_vec,
        r1_mag_km=r1,
        rn_mag_km=rn,
        range_covariance=range_covariance,
        rms_deg=rms_deg,
        n_iterations=n_iter,
        converged=converged,
    )

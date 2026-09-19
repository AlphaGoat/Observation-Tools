"""
linking.py — track-to-track compatibility scoring for the orbit correlator.

Pure math, no HTTP or database access (mirrors the associator/MHT.py split:
the algorithm is independently testable from the service wrapping it).

Compatibility test
-------------------
Two attributables (ra, dec, ra_dot, dec_dot) are tested for whether they
could plausibly belong to the same object by linearly propagating the
earlier one to the later one's epoch under a constant-velocity model, then
computing a chi-square distance between the propagated state and the
observed one -- the same measurement-space compatibility test Siminski
(2016) and Pastor (2022 Sec. 3.2) use to gate track-to-track generation
before running any orbit determination. This is deliberately *not* an
orbit fit: it is the cheap linear pre-filter that keeps the number of
hypotheses needing a real IOD (Lambert, Gauss, etc. -- not yet implemented
in this repo) small. Over the short baselines this project currently
produces tracklets for (same-night to few-night gaps), the linear model is
a reasonable proxy per Siminski's own initial-value-vs-boundary-value
comparison; it degrades for multi-week gaps, where curvature the linear
model can't see becomes significant -- a real IOD-based test should
replace or supplement this once one exists.

Process noise
--------------
Rates are held constant during propagation, but real orbital motion is not
perfectly linear over gaps of hours to days, so a discrete white-noise-
acceleration (DWNA) process noise term is added on top of the linear
covariance propagation to avoid being overconfident about compatibility at
long dt. This is the same process-noise structure used in a standard
constant-velocity Kalman filter (matching the associator's own within-
collect filter in spirit, just with a much larger sigma since dt here is
minutes to days rather than inter-frame seconds).

Author: Peter Thomas
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TrackState:
    """A 4-D attributable (ra, dec, ra_dot, dec_dot) at a single epoch."""
    epoch: float
    state: np.ndarray       # shape (4,): [ra_deg, dec_deg, ra_dot_deg_s, dec_dot_deg_s]
    covariance: np.ndarray  # shape (4, 4)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float).reshape(4)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(4, 4)


def track_to_state(track: dict) -> TrackState:
    """Build a TrackState from a catalog-store track row (see db.py)."""
    return TrackState(
        epoch=0.5 * (track["t_start"] + track["t_end"]),
        state=[track["ra"], track["dec"], track["ra_dot"], track["dec_dot"]],
        covariance=track["covariance"],
    )


def object_to_state(obj: dict) -> Optional[TrackState]:
    """
    Build a TrackState from a catalog-store object row, if it carries a
    4-D representative state (state_dim == 4). Returns None for objects
    with no state yet, or a 6-D orbital state_dim -- neither is directly
    comparable to a raw attributable by this linear test.
    """
    if obj.get("state_dim") != 4 or obj.get("state_vector") is None or obj.get("epoch") is None:
        return None
    covariance = obj.get("covariance")
    if covariance is None:
        return None
    return TrackState(epoch=obj["epoch"], state=obj["state_vector"], covariance=covariance)


def _transition_matrix(dt: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _process_noise(dt: float, sigma_accel_deg_s2: float) -> np.ndarray:
    """DWNA process noise for one axis, block-diagonal across (ra, dec)."""
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    q = sigma_accel_deg_s2 ** 2
    block = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = block   # ra, ra_dot
    Q[np.ix_([1, 3], [1, 3])] = block   # dec, dec_dot
    return Q


def propagate(ts: TrackState, epoch: float, sigma_accel_deg_s2: float) -> TrackState:
    """Propagate a TrackState to a new epoch under constant velocity + DWNA."""
    dt = epoch - ts.epoch
    F = _transition_matrix(dt)
    state = F @ ts.state
    covariance = F @ ts.covariance @ F.T + _process_noise(dt, sigma_accel_deg_s2)
    return TrackState(epoch=epoch, state=state, covariance=covariance)


def _wrap_ra_residual(delta_ra_deg: float) -> float:
    """Wrap a RA difference into [-180, 180) so matches near the 0/360 seam score correctly."""
    return (delta_ra_deg + 180.0) % 360.0 - 180.0


def chi2_distance(a: TrackState, b: TrackState, sigma_accel_deg_s2: float) -> float:
    """
    Chi-square compatibility distance (4 dof) between two attributables,
    propagating whichever is earlier forward to the other's epoch.
    """
    if a.epoch <= b.epoch:
        early, late = a, b
    else:
        early, late = b, a
    propagated = propagate(early, late.epoch, sigma_accel_deg_s2)

    residual = late.state - propagated.state
    residual[0] = _wrap_ra_residual(residual[0])

    cov_total = propagated.covariance + late.covariance
    return float(residual @ np.linalg.solve(cov_total, residual))


@dataclass
class Candidate:
    kind: str    # "object" | "track"
    id: int
    chi2: float
    dt: float


def rank_candidates(
    new_state: TrackState,
    object_candidates: Sequence[tuple],   # (object_id, TrackState)
    track_candidates: Sequence[tuple],    # (track_id, TrackState)
    chi2_gate: float,
    sigma_accel_deg_s2: float,
    k_best: int,
) -> list:
    """
    Generation + scoring + pruning in one pass: compute chi2 for every
    candidate, keep those under the gate, and return at most k_best sorted
    ascending by chi2 (best match first). This caps the branching factor of
    hypotheses generated per new track, matching Pastor's use of a k-best
    cutoff to bound hypothesis-tree growth (2022, Sec. 3.2.3).
    """
    scored: list = []
    for object_id, ref in object_candidates:
        chi2 = chi2_distance(new_state, ref, sigma_accel_deg_s2)
        if chi2 <= chi2_gate:
            scored.append(Candidate("object", object_id, chi2, abs(new_state.epoch - ref.epoch)))
    for track_id, ref in track_candidates:
        chi2 = chi2_distance(new_state, ref, sigma_accel_deg_s2)
        if chi2 <= chi2_gate:
            scored.append(Candidate("track", track_id, chi2, abs(new_state.epoch - ref.epoch)))

    scored.sort(key=lambda c: c.chi2)
    return scored[:k_best]

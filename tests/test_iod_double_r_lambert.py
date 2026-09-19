"""
Integration tests for the double r-iteration Lambert method
(src/iod/double_r_lambert.py, Pastor 2022 Algorithm 3).

Strategy: build a known Keplerian orbit, propagate it to n synthetic
observation epochs from a fixed ground site (a fixed inertial position --
real sidereal site rotation is irrelevant to validating the IOD math
itself), compute the exact topocentric (ra, dec) at each epoch, then check
that the method recovers the true state vector at the first epoch from
angles alone.

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iod import kepler
from iod.double_r_lambert import Observation, solve
from iod.elements import coe_to_rv
from iod.geometry import radec_from_position

R_SITE = np.array([6378.0 * np.cos(np.radians(30)) * np.cos(np.radians(45)),
                    6378.0 * np.cos(np.radians(30)) * np.sin(np.radians(45)),
                    6378.0 * np.sin(np.radians(30))])


def _make_observations(r0, v0, epochs, site=R_SITE):
    obs = []
    for t in epochs:
        r, _ = kepler.propagate(r0, v0, t - epochs[0])
        ra, dec = radec_from_position(r, site)
        obs.append(Observation(ra_deg=ra, dec_deg=dec, epoch_s=t, site_position_km=site))
    return obs


def test_recovers_near_circular_geo_orbit_from_four_tracks():
    a = 42164.0
    r0, v0 = coe_to_rv(a, 0.001, np.radians(3.0), np.radians(10.0), np.radians(50.0), np.radians(15.0))
    epochs = [0.0, 3600.0 * 2, 3600.0 * 5, 3600.0 * 9]
    obs = _make_observations(r0, v0, epochs)

    result = solve(obs)

    assert result.converged
    assert np.linalg.norm(result.r_km - r0) < 5.0       # km
    assert np.linalg.norm(result.v_km_s - v0) < 0.005    # km/s


def test_recovers_gto_orbit_matching_pastor_figure_2_4_eccentricity():
    # a=24460 km, e=0.71 -- same GTO shape as the paper's own worked example
    # (their Fig 2.4 caption). This package's lambert.py/circular.py are
    # explicitly 0-revolution only (documented scope), so the observation
    # span must stay under the ~10.6h orbital period at this semi-major
    # axis -- the paper's own dt12~7h + dt23~5h totalling 12h would in fact
    # be a multi-rev transfer for this specific a, outside what either
    # solver here claims to handle.
    a, e = 24460.0, 0.71
    r0, v0 = coe_to_rv(a, e, np.radians(7.0), np.radians(20.0), np.radians(40.0), np.radians(60.0))
    epochs = [0.0, 3600.0 * 3, 3600.0 * 6, 3600.0 * 9]
    obs = _make_observations(r0, v0, epochs)

    result = solve(obs, max_iter=50)

    assert result.converged
    assert np.linalg.norm(result.r_km - r0) < 1.0        # km
    assert np.linalg.norm(result.v_km_s - v0) < 0.001     # km/s


def test_recovers_leo_orbit_from_single_short_track_plus_extra_obs():
    a, e = 7000.0, 0.05
    r0, v0 = coe_to_rv(a, e, np.radians(51.6), np.radians(0.0), np.radians(0.0), np.radians(0.0))
    epochs = [0.0, 60.0, 130.0]
    obs = _make_observations(r0, v0, epochs)

    result = solve(obs)

    assert np.linalg.norm(result.r_km - r0) < 5.0
    assert np.linalg.norm(result.v_km_s - v0) < 0.01


def test_rejects_fewer_than_three_observations():
    from iod.double_r_lambert import IODError
    obs = [
        Observation(ra_deg=10.0, dec_deg=5.0, epoch_s=0.0, site_position_km=R_SITE),
        Observation(ra_deg=11.0, dec_deg=5.1, epoch_s=600.0, site_position_km=R_SITE),
    ]
    with pytest.raises(IODError):
        solve(obs)


def test_range_covariance_is_reported_and_positive_definite():
    a = 42164.0
    r0, v0 = coe_to_rv(a, 0.0, np.radians(2.0), 0.0, 0.0, np.radians(10.0))
    epochs = [0.0, 3600.0 * 3, 3600.0 * 6, 3600.0 * 9]
    obs = _make_observations(r0, v0, epochs)

    result = solve(obs)

    assert result.range_covariance is not None
    assert result.range_covariance.shape == (2, 2)
    eigvals = np.linalg.eigvalsh(result.range_covariance)
    assert np.all(eigvals > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for the universal-variable Kepler propagator (src/iod/kepler.py).

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iod import kepler
from iod.elements import MU_EARTH_KM3_S2, coe_to_rv


def _energy(r, v, mu=MU_EARTH_KM3_S2):
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


def _ang_momentum(r, v):
    return np.cross(r, v)


@pytest.mark.parametrize("a,e", [
    (42164.0, 0.0),      # circular GEO
    (42164.0, 0.001),    # near-circular GEO
    (24460.0, 0.71),     # GTO-like, matches Pastor 2022 Fig 2.4
    (7000.0, 0.3),       # eccentric LEO
])
def test_round_trip_forward_then_backward(a, e):
    r0, v0 = coe_to_rv(a, e, np.radians(10.0), np.radians(30.0), np.radians(60.0), np.radians(20.0))
    dt = 3600.0

    r1, v1 = kepler.propagate(r0, v0, dt)
    r2, v2 = kepler.propagate(r1, v1, -dt)

    assert np.allclose(r2, r0, rtol=1e-8, atol=1e-6)
    assert np.allclose(v2, v0, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("a,e", [(42164.0, 0.0), (24460.0, 0.71), (7000.0, 0.3)])
def test_conserves_energy_and_angular_momentum(a, e):
    r0, v0 = coe_to_rv(a, e, np.radians(5.0), np.radians(0.0), np.radians(0.0), np.radians(0.0))
    e0 = _energy(r0, v0)
    h0 = _ang_momentum(r0, v0)

    for dt in (60.0, 1800.0, 3600.0 * 5, -1800.0):
        r, v = kepler.propagate(r0, v0, dt)
        assert _energy(r, v) == pytest.approx(e0, rel=1e-9)
        assert np.allclose(_ang_momentum(r, v), h0, rtol=1e-8)


def test_circular_quarter_period_is_a_90_degree_rotation():
    a = 42164.0
    r0, v0 = coe_to_rv(a, 0.0, 0.0, 0.0, 0.0, 0.0)
    period = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH_KM3_S2)

    r, v = kepler.propagate(r0, v0, period / 4.0)

    assert np.linalg.norm(r) == pytest.approx(a, rel=1e-8)
    assert r[0] == pytest.approx(0.0, abs=1e-4)
    assert r[1] == pytest.approx(a, rel=1e-6)


def test_hyperbolic_orbit_propagates():
    # a<0 hyperbola: build via vis-viva at a chosen r0 with v0 above escape speed.
    r0 = np.array([7000.0, 0.0, 0.0])
    v_esc = np.sqrt(2 * MU_EARTH_KM3_S2 / 7000.0)
    v0 = np.array([0.0, v_esc * 1.2, 0.0])

    r1, v1 = kepler.propagate(r0, v0, 600.0)
    r2, v2 = kepler.propagate(r1, v1, -600.0)

    assert np.allclose(r2, r0, rtol=1e-7, atol=1e-5)
    assert np.allclose(v2, v0, rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

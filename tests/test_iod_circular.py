"""
Tests for the Circular method IOD seed (src/iod/circular.py).

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iod import kepler
from iod.circular import circular_radius
from iod.elements import MU_EARTH_KM3_S2, coe_to_rv
from iod.geometry import line_of_sight, radec_from_position

R_SITE = np.array([6378.0 * np.cos(np.radians(30)) * np.cos(np.radians(45)),
                    6378.0 * np.cos(np.radians(30)) * np.sin(np.radians(45)),
                    6378.0 * np.sin(np.radians(30))])


def _observe(r_true, R):
    ra, dec = radec_from_position(r_true, R)
    return line_of_sight(ra, dec)


def test_recovers_exact_radius_for_truly_circular_orbit():
    a = 42164.0
    r0, v0 = coe_to_rv(a, 0.0, np.radians(2.0), 0.0, 0.0, np.radians(10.0))
    dt = 3600.0 * 3
    r1, _ = kepler.propagate(r0, v0, dt)

    L1 = _observe(r0, R_SITE)
    L2 = _observe(r1, R_SITE)

    r_est = circular_radius(R_SITE, L1, R_SITE, L2, dt)
    assert r_est == pytest.approx(a, rel=1e-6)


def test_gives_reasonable_order_of_magnitude_for_eccentric_orbit():
    # Not expected to be accurate for e=0.71 (GTO) -- it's only a seed for
    # the double-r gradient descent -- but should land in a sane ballpark
    # rather than diverging or erroring out.
    a = 24460.0
    r0, v0 = coe_to_rv(a, 0.71, np.radians(5.0), 0.0, 0.0, np.radians(30.0))
    dt = 3600.0 * 7
    r1, _ = kepler.propagate(r0, v0, dt)

    L1 = _observe(r0, R_SITE)
    L2 = _observe(r1, R_SITE)

    r_est = circular_radius(R_SITE, L1, R_SITE, L2, dt)
    assert 5000.0 < r_est < 50000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

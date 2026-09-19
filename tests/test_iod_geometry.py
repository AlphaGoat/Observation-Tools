"""
Tests for angles-only observation geometry (src/iod/geometry.py).

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iod import geometry


def test_position_and_radec_round_trip():
    R = np.array([6378.0, 0.0, 0.0])
    ra, dec = 123.4, -17.8
    L = geometry.line_of_sight(ra, dec)
    r = geometry.position_vector(R, 40000.0, L)

    ra2, dec2 = geometry.radec_from_position(r, R)
    assert ra2 == pytest.approx(ra, abs=1e-9)
    assert dec2 == pytest.approx(dec, abs=1e-9)


def test_slant_range_recovers_known_rho():
    R = np.array([6378.0, 0.0, 0.0])
    L = geometry.line_of_sight(45.0, 10.0)
    true_rho = 36000.0
    r = geometry.position_vector(R, true_rho, L)

    rho = geometry.slant_range(R, L, np.linalg.norm(r))
    assert rho == pytest.approx(true_rho, rel=1e-9)


def test_slant_range_raises_for_unreachable_radius():
    R = np.array([6378.0, 0.0, 0.0])
    L = geometry.line_of_sight(0.0, 89.9)  # nearly straight up from a point near equator
    with pytest.raises(ValueError):
        geometry.slant_range(R, L, 1.0)  # radius smaller than Earth itself


def test_wrapped_residual_handles_seam():
    assert geometry.wrapped_residual_deg(359.9, 0.1) == pytest.approx(-0.2, abs=1e-9)
    assert geometry.wrapped_residual_deg(0.1, 359.9) == pytest.approx(0.2, abs=1e-9)
    assert geometry.wrapped_residual_deg(10.0, 8.0) == pytest.approx(2.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

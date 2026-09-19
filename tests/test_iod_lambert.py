"""
Tests for the universal-variable Lambert solver (src/iod/lambert.py).

Validation strategy: build a known orbit, propagate it with kepler.py to
get two truth position vectors r1, r2 separated by a known dt, then check
that lambert.solve(r1, r2, dt) recovers the *true* v1, v2 from that same
orbit -- this cross-validates the Lambert solver against the independently
tested Kepler propagator using self-consistent physical data.

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iod import kepler, lambert
from iod.elements import coe_to_rv


@pytest.mark.parametrize("a,e,dt", [
    (42164.0, 0.0, 3600.0 * 6),      # circular GEO, ~90 deg-ish arc
    (42164.0, 0.0, 1800.0),          # circular GEO, short arc
    (42164.0, 0.3, 3600.0 * 4),      # eccentric GEO-ish
    (24460.0, 0.71, 3600.0 * 7),     # GTO, matches Pastor 2022 Fig 2.4 Delta t12 ~ 7h
    (7000.0, 0.1, 1200.0),           # LEO, short arc
])
def test_recovers_true_velocities(a, e, dt):
    r0, v0 = coe_to_rv(a, e, np.radians(10.0), np.radians(30.0), np.radians(60.0), np.radians(20.0))
    r1_true, v1_true = r0, v0
    r2_true, v2_true = kepler.propagate(r0, v0, dt)

    v1, v2 = lambert.solve(r1_true, r2_true, dt)

    assert np.allclose(v1, v1_true, rtol=1e-6, atol=1e-8)
    assert np.allclose(v2, v2_true, rtol=1e-6, atol=1e-8)


def test_rejects_nonpositive_dt():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([0.0, 7000.0, 0.0])
    with pytest.raises(ValueError):
        lambert.solve(r1, r2, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

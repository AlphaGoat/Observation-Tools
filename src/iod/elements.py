"""
elements.py — classical orbital elements <-> Cartesian state vector.

Only used to build synthetic ground-truth orbits for testing the IOD
methods in this package (kepler.py, lambert.py, double_r_lambert.py); the
IOD algorithms themselves work entirely in Cartesian state vectors, never
touch classical elements.

Author: Peter Thomas
"""
from __future__ import annotations

import numpy as np

MU_EARTH_KM3_S2 = 398600.4418  # Earth gravitational parameter, km^3/s^2


def coe_to_rv(
    a: float, e: float, i: float, raan: float, argp: float, nu: float,
    mu: float = MU_EARTH_KM3_S2,
) -> tuple:
    """
    Classical orbital elements -> Cartesian position/velocity (km, km/s).

    a     semi-major axis (km)
    e     eccentricity
    i     inclination (rad)
    raan  right ascension of ascending node (rad)
    argp  argument of perigee (rad)
    nu    true anomaly (rad)
    """
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu))

    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    co, so = np.cos(argp), np.sin(argp)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])

    return R @ r_pqw, R @ v_pqw

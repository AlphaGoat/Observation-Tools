"""
KD-tree index and plate solver for astrometric calibration.

Two scipy.spatial.cKDTree instances back the index:
  CodeSpaceTree    — 4D tree over (xC, yC, xD, yD) hash codes for one scale tier
  StarPositionTree — 3D tree over XYZ unit-sphere positions (avoids cos(dec) distortion near poles)

Indices are built per scale tier (each spanning a √2 factor in quad AB diameter).
PlateSolver accepts a list of CodeSpaceTrees and selects the relevant tier(s) at
solve time based on the image FOV.

Build a tiered index (offline, once):
    from astrometry.kd_tree import build_index_tiered, SCALE_TIERS
    build_index_tiered(
        sky_region=dict(min_ra=0, max_ra=10, min_dec=-5, max_dec=5),
        tier_indices=[8, 9, 10, 11],           # ~0.8–4.5° quad AB diameter
        index_dir="indices/",
        star_index_path="indices/stars.joblib",
    )

Field solve (online, per image):
    from astrometry.kd_tree import CodeSpaceTree, StarPositionTree, PlateSolver
    import glob
    code_trees = [CodeSpaceTree.load(p) for p in sorted(glob.glob("indices/codes_tier*.joblib"))]
    star_tree  = StarPositionTree.load("indices/stars.joblib")
    solver = PlateSolver(code_trees, star_tree)
    wcs = solver.solve(detected_stars, image_height=2048, image_width=2048, fov_deg=2.0)

Authors: Peter Thomas
"""

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import joblib
import numpy as np
from astropy_healpix import HEALPix, nside_to_pixel_resolution, pixel_resolution_to_nside
from scipy.spatial import cKDTree

from astrometry.feature_generation import compute_hash_code
from astrometry.bayesian_decision_maker import bayesian_decision_maker


# ── Scale tier definitions ────────────────────────────────────────────────────

# Each tier covers a √2 factor in quad AB angular diameter (degrees).
# Tier N spans [base * (√2)^N, base * (√2)^(N+1)] where base ≈ 0.05°.
# These match Astrometry.net's preset index scale bands.
SCALE_TIERS: dict = {
    0:  (0.050, 0.071),
    1:  (0.071, 0.100),
    2:  (0.100, 0.141),
    3:  (0.141, 0.200),
    4:  (0.200, 0.283),
    5:  (0.283, 0.400),
    6:  (0.400, 0.566),
    7:  (0.566, 0.800),
    8:  (0.800, 1.131),
    9:  (1.131, 1.600),
    10: (1.600, 2.263),
    11: (2.263, 3.200),
    12: (3.200, 4.525),
    13: (4.525, 6.400),
    14: (6.400, 9.051),
    15: (9.051, 12.80),
}


def _ab_sep_deg(ra: np.ndarray, dec: np.ndarray) -> float:
    """
    Return the angular separation in degrees of the most-separated star pair
    in a 4-star group, using the flat-sky cos(dec) approximation (valid < ~10°).
    """
    mean_dec = (dec[:, None] + dec[None, :]) / 2
    dist = np.sqrt(
        ((ra[:, None] - ra[None, :]) * np.cos(np.radians(mean_dec))) ** 2
        + (dec[:, None] - dec[None, :]) ** 2
    )
    return float(dist.max())


def _sep2_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Angular separation (degrees) between two sky positions (flat-sky approx)."""
    mean_dec_rad = np.radians((dec1 + dec2) / 2)
    return float(np.sqrt(
        ((ra1 - ra2) * np.cos(mean_dec_rad)) ** 2 + (dec1 - dec2) ** 2
    ))


def _gnomonic(
    ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gnomonic (TAN) projection: (RA, Dec) → tangent-plane (ξ, η) in degrees.

    Tangent point is (ra0, dec0).  Returns (xi_deg, eta_deg).
    Valid for separations up to ~60° from the tangent point; the plate solver
    uses it for FOVs up to ~10° where it is essentially exact.
    """
    ra_r  = np.radians(np.asarray(ra,  dtype=np.float64))
    dec_r = np.radians(np.asarray(dec, dtype=np.float64))
    r0 = np.radians(ra0)
    d0 = np.radians(dec0)
    cos_c = (np.sin(d0) * np.sin(dec_r)
             + np.cos(d0) * np.cos(dec_r) * np.cos(ra_r - r0))
    xi  = -np.cos(dec_r) * np.sin(ra_r - r0) / cos_c
    eta = (np.cos(d0) * np.sin(dec_r)
           - np.sin(d0) * np.cos(dec_r) * np.cos(ra_r - r0)) / cos_c
    return np.degrees(xi), np.degrees(eta)


def _gnomonic_inv(
    xi_deg: np.ndarray, eta_deg: np.ndarray, ra0: float, dec0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse gnomonic: tangent-plane (ξ, η) in degrees → (RA, Dec) in degrees.
    """
    xi  = np.radians(np.asarray(xi_deg,  dtype=np.float64))
    eta = np.radians(np.asarray(eta_deg, dtype=np.float64))
    r0 = np.radians(ra0)
    d0 = np.radians(dec0)
    D   = np.cos(d0) - eta * np.sin(d0)
    ra  = r0 + np.arctan2(-xi, D)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.sqrt(xi ** 2 + D ** 2))
    return np.degrees(ra) % 360.0, np.degrees(dec)


# ── WCS ──────────────────────────────────────────────────────────────────────

@dataclass
class WCS:
    """
    Gnomonic (TAN) WCS.

    Forward:  pixel = A @ [ξ, η, 1]ᵀ
    where (ξ, η) are gnomonic tangent-plane coordinates in degrees, projected
    from the tangent point (ra0, dec0).  A is shape (2, 3), float64.

    Backward: (ξ, η) = A[:,:2]⁻¹ @ (pixel − A[:,2])  then gnomonic⁻¹
    """
    A:    np.ndarray
    ra0:  float = 0.0
    dec0: float = 0.0

    def radec_to_pix(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Return (N, 2) pixel [x, y] for arrays of RA, Dec in degrees."""
        ra  = np.asarray(ra,  dtype=float).ravel()
        dec = np.asarray(dec, dtype=float).ravel()
        xi, eta = _gnomonic(ra, dec, self.ra0, self.dec0)
        coords  = np.vstack([xi, eta, np.ones(len(xi))])   # (3, N)
        return (self.A @ coords).T                           # (N, 2)

    def pix_to_radec(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ra_deg, dec_deg) arrays for pixel coordinates (0-indexed)."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        M_inv = np.linalg.inv(self.A[:, :2])
        t     = self.A[:, 2]
        xi_eta = M_inv @ np.vstack([x - t[0], y - t[1]])   # (2, N) tangent-plane
        return _gnomonic_inv(xi_eta[0], xi_eta[1], self.ra0, self.dec0)


def _fit_wcs(
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
) -> Optional[WCS]:
    """
    Fit a gnomonic WCS from n matched (ra, dec) → (x, y) pairs via an
    orthogonal-Procrustes (Kabsch) similarity fit: rotation + one uniform
    scale + translation, mapping tangent-plane (ξ, η) to pixels. This is the
    same method astrometry.net's own solver uses (fit_tan_wcs / GSL SVD of
    the field/star cross-covariance -- see util/fit-wcs.c), and it matches
    the physical model our own quad hash codes already assume: real camera
    images are similarity transforms of the sky, not general affine ones
    (shear/anisotropic scale). It also needs only 2 non-coincident points to
    be well-posed, versus 3 for a general affine fit.

    An earlier version of this function fit an unconstrained 6-parameter
    affine map (independent x/y scale + shear) via two separate least-squares
    regressions. That has *fewer* effective constraints per free parameter
    (4 points give 8 equations for 6 unknowns, vs. 8 equations for 4 unknowns
    here) and can silently absorb positional noise as spurious shear instead
    of averaging it into rotation/scale error -- confirmed empirically to
    give comparable-or-worse accuracy on real matched quads, never
    meaningfully better.

    Sets the tangent point at the centroid of the catalogue stars. Requires
    n ≥ 2 non-coincident pairs.
    """
    n = len(cat_ra)
    if n < 2:
        return None
    ra0  = float(np.mean(cat_ra))
    dec0 = float(np.mean(cat_dec))
    xi, eta = _gnomonic(cat_ra, cat_dec, ra0, dec0)

    p = np.column_stack([xi, eta]).astype(np.float64)       # tangent-plane, degrees
    f = np.column_stack([pix_x, pix_y]).astype(np.float64)  # pixels

    p_cm = p.mean(axis=0)
    f_cm = f.mean(axis=0)
    pc = p - p_cm
    fc = f - f_cm

    # Cross-covariance of centered (sky, pixel) offsets; SVD gives the
    # rotation minimizing ||fc - R @ pc||^2 (orthogonal Procrustes).
    cov = pc.T @ fc
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # Reflection, not a rotation -- flip the smallest-singular-value
        # axis to recover a proper rotation (standard Kabsch correction).
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    pvar = float(np.sum(pc ** 2))
    fvar = float(np.sum(fc ** 2))
    if pvar <= 0 or not np.isfinite(pvar) or not np.isfinite(fvar):
        return None
    scale = np.sqrt(fvar / pvar)  # pixels per degree

    CD = R * scale
    t  = f_cm - CD @ p_cm
    A  = np.column_stack([CD, t])
    return WCS(A=A, ra0=ra0, dec0=dec0) if np.all(np.isfinite(A)) else None


# ── Canonical quad ordering ───────────────────────────────────────────────────

def _sky_abcd(ra: np.ndarray, dec: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the canonical [a, b, c, d] index ordering for a 4-star quad
    in sky (RA/Dec) coordinates, using the same cos(dec)-corrected distance
    and similarity transform as compute_hash_code.

    Returns None if C or D lies outside the AB inscribed circle.
    """
    ra  = np.asarray(ra,  dtype=float)
    dec = np.asarray(dec, dtype=float)

    # Pairwise angular distances with cos(dec) correction
    mean_dec = (dec[:, None] + dec[None, :]) / 2
    dist = np.sqrt(
        ((ra[:, None] - ra[None, :]) * np.cos(np.radians(mean_dec))) ** 2
        + (dec[:, None] - dec[None, :]) ** 2
    )
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    c, d = [i for i in range(4) if i not in (a, b)]

    # Inscribed-circle validity
    c_ra  = (ra[a]  + ra[b])  / 2
    c_dec = (dec[a] + dec[b]) / 2
    radius = dist[a, b] / 2
    for idx in (c, d):
        cd = np.cos(np.radians((dec[idx] + c_dec) / 2))
        d_star = np.sqrt(((ra[idx] - c_ra) * cd) ** 2 + (dec[idx] - c_dec) ** 2)
        if d_star >= radius:
            return None

    # Similarity transform to code space. Negated to match _gnomonic's tangent-
    # plane chirality -- see the matching comment in feature_generation.py's
    # compute_hash_code (this function must stay bit-for-bit consistent with it).
    cos_ab = np.cos(np.radians(c_dec))
    proj   = -ra * cos_ab
    dpr    = proj[b] - proj[a]
    ddec   = dec[b]  - dec[a]
    theta  = np.pi / 4 - np.arctan2(ddec, dpr)
    lam    = np.sqrt(2) / np.sqrt(dpr ** 2 + ddec ** 2)
    t_x    = lam * (-proj[a] * np.cos(theta) + dec[a] * np.sin(theta))
    t_y    = lam * (-proj[a] * np.sin(theta) - dec[a] * np.cos(theta))
    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ],
    ])
    coords = np.vstack([proj, dec, np.ones(4)])
    tc     = T @ coords

    xc, xd = tc[0, c], tc[0, d]

    # Canonical C/D ordering
    if xc > xd:
        c, d = d, c
        xc, xd = xd, xc

    # Canonical A/B orientation
    if xc + xd > 1:
        a, b = b, a
        xc, xd = 1.0 - xd, 1.0 - xc
        if xc > xd:
            c, d = d, c

    return np.array([a, b, c, d])


def _pix_abcd(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Return canonical [a, b, c, d] ordering for a 4-star quad in pixel
    coordinates (Euclidean distances, no cos-dec correction).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx ** 2 + dy ** 2

    a, b = np.unravel_index(np.argmax(dist2), dist2.shape)
    c, d = [i for i in range(4) if i not in (a, b)]

    # Inscribed circle
    cx, cy = (x[a] + x[b]) / 2, (y[a] + y[b]) / 2
    r2 = dist2[a, b] / 4
    if (x[c] - cx) ** 2 + (y[c] - cy) ** 2 >= r2:
        return None
    if (x[d] - cx) ** 2 + (y[d] - cy) ** 2 >= r2:
        return None

    # Similarity transform to code space
    dab_x  = x[b] - x[a]
    dab_y  = y[b] - y[a]
    theta  = np.pi / 4 - np.arctan2(dab_y, dab_x)
    lam    = np.sqrt(2) / np.sqrt(dab_x ** 2 + dab_y ** 2)
    t_x    = lam * (-(x[a]) * np.cos(theta) + (y[a]) * np.sin(theta))
    t_y    = lam * (-(x[a]) * np.sin(theta) - (y[a]) * np.cos(theta))
    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ],
    ])
    coords = np.vstack([x, y, np.ones(4)])
    tc     = T @ coords

    xc, xd = tc[0, c], tc[0, d]

    if xc > xd:
        c, d = d, c
        xc, xd = xd, xc

    if xc + xd > 1:
        a, b = b, a
        xc, xd = 1.0 - xd, 1.0 - xc
        if xc > xd:
            c, d = d, c

    return np.array([a, b, c, d])


def _pix_quad_info(
    x: np.ndarray, y: np.ndarray
) -> Optional[Tuple[Tuple[float, float, float, float], np.ndarray]]:
    """
    Compute pixel-space hash code AND canonical ABCD indices in one consistent pass.

    Running the canonical flip logic once (rather than separately in _pix_hash_code
    and _pix_abcd) prevents near-boundary quads (xC + xD ≈ 1.0) from getting
    inconsistent A/B orderings due to floating-point differences between two
    independent calls.

    Returns
    -------
    (code, abcd) where
        code  : (xc, yc, xd, yd) float tuple — the 4D hash code
        abcd  : (4,) int array   — canonical [A, B, C, D] indices into x/y
    or None if the quad is geometrically invalid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx ** 2 + dy ** 2

    a, b = np.unravel_index(np.argmax(dist2), dist2.shape)
    c, d = [i for i in range(4) if i not in (a, b)]

    cx_mid, cy_mid = (x[a] + x[b]) / 2, (y[a] + y[b]) / 2
    r2 = dist2[a, b] / 4
    if (x[c] - cx_mid) ** 2 + (y[c] - cy_mid) ** 2 >= r2:
        return None
    if (x[d] - cx_mid) ** 2 + (y[d] - cy_mid) ** 2 >= r2:
        return None

    dab_x = x[b] - x[a]
    dab_y = y[b] - y[a]
    theta = np.pi / 4 - np.arctan2(dab_y, dab_x)
    lam   = np.sqrt(2) / np.sqrt(dab_x ** 2 + dab_y ** 2)
    t_x   = lam * (-x[a] * np.cos(theta) + y[a] * np.sin(theta))
    t_y   = lam * (-x[a] * np.sin(theta) - y[a] * np.cos(theta))
    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ],
    ])
    tc = T @ np.vstack([x, y, np.ones(4)])

    xc, yc = tc[0, c], tc[1, c]
    xd, yd = tc[0, d], tc[1, d]

    # Canonical C/D: ensure xc ≤ xd
    if xc > xd:
        c, d = d, c
        xc, xd = xd, xc
        yc, yd = yd, yc

    # Canonical A/B orientation: ensure xc + xd ≤ 1 (one consistent check)
    if xc + xd > 1.0:
        a, b = b, a
        xc, xd = 1.0 - xd, 1.0 - xc
        yc, yd = 1.0 - yd, 1.0 - yc
        if xc > xd:
            c, d = d, c
            xc, xd = xd, xc
            yc, yd = yd, yc

    return (xc, yc, xd, yd), np.array([a, b, c, d])


def _pix_hash_code(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, ...]]:
    """Pixel-space hash code (Euclidean, no cos correction). Delegates to _pix_quad_info."""
    result = _pix_quad_info(x, y)
    return result[0] if result is not None else None


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert (RA, Dec) in degrees to unit-sphere XYZ. Returns (N, 3)."""
    ra  = np.radians(np.asarray(ra_deg,  dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


# ── CodeSpaceTree ─────────────────────────────────────────────────────────────

class CodeSpaceTree:
    """
    cKDTree over 4D hash codes (xC, yC, xD, yD) for a single scale tier.

    quad_source_ids[i] holds the four Gaia source IDs of the stars in quad i,
    stored in canonical ABCD order (matching the ordering of the hash code
    coordinates). Shape: (N_quads, 4), dtype int64.

    scale_lower_deg / scale_upper_deg record the AB angular diameter range
    (degrees) covered by this tier so PlateSolver can select the right trees
    for a given image FOV.
    """

    def __init__(
        self,
        codes: np.ndarray,
        quad_source_ids: np.ndarray,
        scale_lower_deg: Optional[float] = None,
        scale_upper_deg: Optional[float] = None,
    ) -> None:
        if codes.ndim != 2 or codes.shape[1] != 4:
            raise ValueError("codes must be shape (N, 4)")
        if len(codes) != len(quad_source_ids):
            raise ValueError("codes and quad_source_ids must have the same length")
        self.codes           = np.asarray(codes,           dtype=np.float32)
        self.quad_source_ids = np.asarray(quad_source_ids, dtype=np.int64)
        self.scale_lower_deg = scale_lower_deg
        self.scale_upper_deg = scale_upper_deg
        self._tree           = cKDTree(self.codes)

    def range_search(
        self, query_code: np.ndarray, radius: float
    ) -> List[np.ndarray]:
        """
        Return all catalogue quad source-ID arrays within `radius` of `query_code`
        in L2 distance over the 4D code space.

        Parameters
        ----------
        query_code : (4,) float  — detected hash code (xC, yC, xD, yD)
        radius     : float       — search tolerance, typical 0.01–0.05

        Returns
        -------
        List of (4,) int64 arrays, one per matching catalogue quad.
        """
        q = np.asarray(query_code, dtype=np.float32)
        indices = self._tree.query_ball_point(q, r=radius)
        return [self.quad_source_ids[i] for i in indices]

    def save(self, path: str) -> None:
        """Persist the tree and data arrays. Loads in zero rebuild time via joblib mmap."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CodeSpaceTree":
        """
        Restore from disk. mmap_mode='r' is what actually makes the large
        arrays memory-mapped rather than fully materialized -- joblib.dump's
        default compress=0 (used by save()) is what makes the file mmap-able
        in the first place, but joblib.load only does so if told to; its own
        default (mmap_mode=None) reads everything into RAM like a plain
        fread, same as astrometry.net avoids via mmap() in its own index
        loader (see fitsbin.c / kdtree_fits_io.c). Backed by a read-only PVC
        mount or shared page cache, this lets multiple gunicorn workers (no
        --preload here, so each loads independently post-fork) or replicas
        share physical pages of the same index file instead of each holding
        a private full copy.
        """
        return joblib.load(path, mmap_mode="r")


# ── StarPositionTree ──────────────────────────────────────────────────────────

class StarPositionTree:
    """
    cKDTree over catalogue star positions on the unit sphere (3D XYZ).

    Storing XYZ rather than raw (RA, Dec) means Euclidean distance in the tree
    is a faithful proxy for angular separation everywhere on the sky, including
    near the poles where a 2D (RA, Dec) tree has cos(dec) distortion.

    RA/Dec arrays are still kept for box queries and WCS fitting; the cKDTree
    itself operates on the derived XYZ coordinates.
    """

    def __init__(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        source_ids: np.ndarray,
        mV: np.ndarray,
    ) -> None:
        self.ra         = np.asarray(ra,         dtype=np.float64)
        self.dec        = np.asarray(dec,        dtype=np.float64)
        self.source_ids = np.asarray(source_ids, dtype=np.int64)
        self.mV         = np.asarray(mV,         dtype=np.float64)
        self._tree      = cKDTree(_radec_to_xyz(self.ra, self.dec))

    def nearest(self, ra: float, dec: float) -> Tuple[int, float]:
        """Return (source_id, angular_distance_deg) of the nearest catalogue star."""
        q = _radec_to_xyz(np.array([ra]), np.array([dec]))[0]
        d_chord, i = self._tree.query(q)
        # Chord length → central angle: θ = 2·arcsin(chord/2)
        ang_deg = float(np.degrees(2.0 * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))))
        return int(self.source_ids[i]), ang_deg

    def stars_in_box(
        self, ra_min: float, ra_max: float, dec_min: float, dec_max: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (ra, dec, mV) arrays for catalogue stars inside the bounding box.
        """
        mask = (
            (self.ra  >= ra_min) & (self.ra  <= ra_max) &
            (self.dec >= dec_min) & (self.dec <= dec_max)
        )
        return self.ra[mask], self.dec[mask], self.mV[mask]

    def save(self, path: str) -> None:
        """Persist the tree and data arrays. Loads in zero rebuild time via joblib mmap."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "StarPositionTree":
        """Restore from disk. mmap_mode='r' -- see CodeSpaceTree.load for why this matters."""
        return joblib.load(path, mmap_mode="r")


# Indices built via this file's own CLI (`python kd_tree.py build ...`, below)
# were pickled while this module was running as __main__, so joblib recorded
# CodeSpaceTree/StarPositionTree as living in __main__ rather than
# astrometry.kd_tree. Loading such a file from a different entry point --
# the api.py Flask service under gunicorn, in particular -- fails with
# "Can't get attribute 'CodeSpaceTree' on <module '__main__' ...>" unless
# those names are also reachable from whatever __main__ actually is at load
# time. This doesn't change how new indices are pickled going forward; it
# only patches the loading process's environment so existing __main__-tagged
# files keep working.
import sys as _sys
_main_module = _sys.modules.get("__main__")
if _main_module is not None:
    _main_module.__dict__.setdefault("CodeSpaceTree", CodeSpaceTree)
    _main_module.__dict__.setdefault("StarPositionTree", StarPositionTree)


# ── PlateSolver ───────────────────────────────────────────────────────────────

class PlateSolver:
    """
    Blind plate solver using geometric hashing and Bayesian verification.

    Accepts one or more CodeSpaceTrees (one per scale tier) and a single
    StarPositionTree.  At solve time, if fov_deg is provided only the tier(s)
    whose scale range overlaps the expected FOV are searched, which dramatically
    reduces false matches compared to searching a single untiered index.

    For each 4-star combo from the detected stars (brightest 20 used):
      1. Compute a pixel-space hash code.
      2. Range-search the selected CodeSpaceTrees for matching catalogue quads.
      3. Match detected ABCD order to catalogue ABCD order.
      4. Fit an affine WCS from the 4 correspondence pairs.
      5. Project nearby catalogue stars to pixels under the candidate WCS.
      6. Accept via bayesian_decision_maker; return the first accepted WCS.
    """

    def __init__(
        self,
        code_trees: "CodeSpaceTree | List[CodeSpaceTree]",
        star_tree: StarPositionTree,
        *,
        code_radius: float = 0.02,
        max_quads: int = 300,
        model: str = "asymmetric",
        variance: float = 9.0,
        distractors: float = 0.25,
        field_span_multiplier: float = 4.0,
    ) -> None:
        # Accept a single tree or a list for backward compatibility
        if isinstance(code_trees, CodeSpaceTree):
            code_trees = [code_trees]
        self.code_trees  = code_trees
        self.star_tree   = star_tree
        self.code_radius = code_radius
        self.max_quads   = max_quads
        self.model       = model
        self.variance    = variance
        self.distractors = distractors
        self.field_span  = field_span_multiplier

        # Fast source_id → (ra, dec) lookup
        self._sid_ra  = dict(zip(star_tree.source_ids, star_tree.ra))
        self._sid_dec = dict(zip(star_tree.source_ids, star_tree.dec))

    def _select_trees(self, fov_deg: Optional[float]) -> List[CodeSpaceTree]:
        """
        Return the subset of code trees relevant for a given FOV.

        A tier is relevant when its scale band overlaps [fov_deg*0.1, fov_deg].
        A quad's AB pair can span from ~10% of the FOV (tight quad) up to the
        full FOV diagonal.  When fov_deg is None all trees are searched.
        """
        if fov_deg is None:
            return self.code_trees
        selected = []
        for ct in self.code_trees:
            lo = ct.scale_lower_deg
            hi = ct.scale_upper_deg
            if lo is None or hi is None:
                selected.append(ct)   # no metadata — include unconditionally
                continue
            # Overlap condition: tier [lo, hi] ∩ [fov*0.1, fov] ≠ ∅
            if hi >= fov_deg * 0.1 and lo <= fov_deg:
                selected.append(ct)
        return selected or self.code_trees   # fall back to all if nothing matched

    def _resolve_sids(
        self, source_ids: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Map source IDs to (ra, dec) arrays. Returns None if any ID is unknown."""
        try:
            ra  = np.array([self._sid_ra[s]  for s in source_ids])
            dec = np.array([self._sid_dec[s] for s in source_ids])
        except KeyError:
            return None
        return ra, dec

    def solve(
        self,
        detected_stars: np.ndarray,
        image_height: int,
        image_width: int,
        sort_by: str = "snr",
        fov_deg: Optional[float] = None,
    ) -> Optional[WCS]:
        """
        Attempt to determine the astrometric solution for a field.

        Parameters
        ----------
        detected_stars : (K, 3) array  — [y_pix, x_pix, brightness_metric]
        image_height, image_width : int
        sort_by : {"snr", "magnitude"}
        fov_deg : approximate image field-of-view diagonal in degrees.
            When provided, only code trees whose scale tier overlaps
            [fov_deg*0.1, fov_deg] are searched, sharply reducing false
            matches.  When None all trees are searched.

        Returns
        -------
        WCS if a solution is accepted, None otherwise.
        """
        stars = np.asarray(detected_stars, dtype=float)
        if len(stars) < 4:
            return None

        # Work on the N brightest stars to limit combinations
        if sort_by == "snr":
            order = np.argsort(stars[:, 2])[::-1]
        else:
            order = np.argsort(stars[:, 2])
        stars = stars[order[:20]]
        n = len(stars)

        active_trees = self._select_trees(fov_deg)

        quads_tried = 0
        for quad_idx in itertools.combinations(range(n), 4):
            if quads_tried >= self.max_quads:
                break

            det   = stars[list(quad_idx)]
            pix_y = det[:, 0]
            pix_x = det[:, 1]

            # One consistent pass: code and ABCD indices share the same canonical flip
            quad_info = _pix_quad_info(pix_x, pix_y)
            if quad_info is None:
                continue

            query_code, det_abcd = quad_info
            quads_tried += 1
            q = np.array(query_code, dtype=np.float32)

            # Search each selected scale tier
            for code_tree in active_trees:
                matches = code_tree.range_search(q, radius=self.code_radius)

                for cat_sids in matches:
                    result = self._resolve_sids(cat_sids)
                    if result is None:
                        continue
                    cat_ra, cat_dec = result

                    # Try both A/B orientations; near xC+xD=1 the sky canonical
                    # and pixel canonical orderings can disagree due to noise.
                    det_pix_x = pix_x[det_abcd]
                    det_pix_y = pix_y[det_abcd]
                    ab_swap   = np.array([1, 0, 2, 3])

                    wcs = None
                    best_res2 = np.inf
                    for c_ra, c_dec in [
                        (cat_ra, cat_dec),
                        (cat_ra[ab_swap], cat_dec[ab_swap]),
                    ]:
                        w = _fit_wcs(c_ra, c_dec, det_pix_x, det_pix_y)
                        if w is None:
                            continue
                        pred = w.radec_to_pix(c_ra, c_dec)
                        res2 = ((pred[:, 0] - det_pix_x) ** 2
                                + (pred[:, 1] - det_pix_y) ** 2).sum()
                        if res2 < best_res2:
                            wcs, best_res2 = w, res2
                    if wcs is None:
                        continue

                    ra_c  = float(np.mean(cat_ra))
                    dec_c = float(np.mean(cat_dec))
                    span  = float(max(np.ptp(cat_ra), np.ptp(cat_dec), 0.05)) * self.field_span
                    ref_ra, ref_dec, ref_mV = self.star_tree.stars_in_box(
                        ra_c - span, ra_c + span, dec_c - span, dec_c + span
                    )
                    if len(ref_ra) == 0:
                        continue

                    ref_pix   = wcs.radec_to_pix(ref_ra, ref_dec)
                    ref_stars = np.column_stack([
                        ref_pix[:, 1], ref_pix[:, 0], ref_mV,
                    ])

                    if bayesian_decision_maker(
                        reference_stars=ref_stars,
                        test_stars=stars,
                        image_height=image_height,
                        image_width=image_width,
                        model=self.model,
                        variance=self.variance,
                        distractors=self.distractors,
                        sort_by=sort_by,
                    ):
                        return wcs

        return None


# ── Index builder ─────────────────────────────────────────────────────────────

def _find_one_quad(
    pts: np.ndarray,
    star_ra: np.ndarray,
    star_dec: np.ndarray,
    source_ids: np.ndarray,
    times_used: np.ndarray,
    reuse_cap: int,
    spatial_tree: cKDTree,
    r_hi: float,
    scale_lo: float,
    scale_hi: Optional[float],
    healpix: HEALPix,
    pix: int,
    used_quads: set,
    min_pair_sep_frac: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """
    Find the first valid, not-yet-used quad (brightest-first A, then B, then
    C/D) whose centroid lands in this HEALPix cell, using only stars under
    `reuse_cap`.

    `min_pair_sep_frac`, if set, rejects quads where any of the 6 pairwise
    star separations falls below that fraction of the A-B separation --
    e.g. 0.05 rejects a quad if two of its stars are within 5% of the quad's
    own scale of each other. Catches near-degenerate quads (confirmed in
    testing: a real registered quad with two catalog stars 0.14 arcsec
    apart produced a ~270 arcsec WCS center error even with zero positional
    noise, because two near-coincident points barely constrain the fit).
    None (default) disables the check -- see `filter_ill_conditioned` on
    build_index/build_index_tiered.

    One quad per call — the caller sweeps multiple passes to build up a
    per-cell density, rather than this function exhausting every C/D
    combination for every A-B pair in one shot. `used_quads` (a set of
    frozenset({i,j,c,d}) local-index tuples already returned this cell) is
    required, not optional: without it, since this search is deterministic
    and times_used only blocks a star once it hits `reuse_cap`, a fresh call
    just rediscovers the *same* first-valid quad every pass until one of its
    4 stars is finally capped -- wasting most of the per-cell quota on
    `reuse_cap` duplicate copies of one quad instead of `passes` distinct
    ones (caught empirically: 16 "quads" in a cell collapsing to 2 distinct
    ones under `max_times_used=8`).

    `healpix`/`pix` define the cell: equal-area HEALPix pixels, matching
    astrometry.net's own build-astrometry-index grid (a plain RA/Dec box
    grid has non-uniform physical area -- shrinking by cos(dec) toward the
    poles -- which biases quad density away from uniform sky coverage).

    Returns (code, ordered_source_ids, local_indices) or None if no valid,
    unused quad exists under the current reuse cap.
    """
    n = len(star_ra)
    for i in range(n):
        if times_used[i] >= reuse_cap:
            continue

        # B candidates: all stars within r_hi of A in projected space.
        # Requiring j > i ensures each unordered {A, B} pair is visited once.
        # query_ball_point returns indices in tree-traversal order, not
        # brightness order -- sort them (star arrays are pre-sorted
        # brightest-first, so ascending index == ascending brightness) so the
        # first valid quad found actually favours bright stars, matching the
        # "start with the brightest stars" design (Lang et al. 2010 sec 2.3).
        # Without this, only star A was ever brightness-prioritized; B/C/D
        # could be arbitrarily faint even when much brighter alternatives
        # existed among the candidates.
        b_candidates = sorted(spatial_tree.query_ball_point(pts[i], r=r_hi))

        for j in b_candidates:
            if j <= i or times_used[j] >= reuse_cap:
                continue

            sep = _sep2_deg(star_ra[i], star_dec[i], star_ra[j], star_dec[j])
            if sep < scale_lo:
                continue
            if scale_hi is not None and sep > scale_hi:
                continue

            # C/D candidates must lie within the A-B inscribed circle
            # (centre = midpoint of A-B, radius = sep/2).
            mid_proj = (pts[i] + pts[j]) * 0.5
            r_ins    = sep / 2  # degrees (same scale as projected coords)

            cd_raw = spatial_tree.query_ball_point(mid_proj, r=r_ins)
            cd_candidates = sorted(
                k for k in cd_raw
                if k != i and k != j and times_used[k] < reuse_cap
            )
            if len(cd_candidates) < 2:
                continue

            # Assign quads to cells by centroid. Compute every C/D pair's
            # centroid HEALPix pixel in one vectorized call rather than one
            # astropy_healpix call per pair.
            pairs = list(itertools.combinations(cd_candidates, 2))
            c_idx = np.array([p[0] for p in pairs])
            d_idx = np.array([p[1] for p in pairs])
            mean_ra  = (star_ra[i]  + star_ra[j]  + star_ra[c_idx]  + star_ra[d_idx])  / 4
            mean_dec = (star_dec[i] + star_dec[j] + star_dec[c_idx] + star_dec[d_idx]) / 4
            centroid_pix = healpix.lonlat_to_healpix(mean_ra * u.deg, mean_dec * u.deg)
            in_cell = np.nonzero(centroid_pix == pix)[0]
            if len(in_cell) == 0:
                continue

            for idx in in_cell:
                c, d = int(c_idx[idx]), int(d_idx[idx])

                local = [i, j, c, d]
                if frozenset(local) in used_quads:
                    continue

                if min_pair_sep_frac is not None:
                    quad_pts = pts[local]
                    diffs = quad_pts[:, None, :] - quad_pts[None, :, :]
                    pairwise = np.hypot(diffs[..., 0], diffs[..., 1])
                    np.fill_diagonal(pairwise, np.inf)
                    if pairwise.min() < min_pair_sep_frac * sep:
                        continue

                q_ra  = star_ra[local]
                q_dec = star_dec[local]

                code = compute_hash_code(q_ra, q_dec)
                if code is None:
                    continue

                abcd = _sky_abcd(q_ra, q_dec)
                if abcd is None:
                    continue

                ordered_sids = np.array(
                    [source_ids[local[abcd[k]]] for k in range(4)],
                    dtype=np.int64,
                )
                return np.array(code, dtype=np.float32), ordered_sids, local

    return None


def _generate_indexed_features(
    star_ra: np.ndarray,
    star_dec: np.ndarray,
    star_mv: np.ndarray,
    source_ids: np.ndarray,
    healpix: HEALPix,
    pix: int,
    max_times_used: int = 8,
    scale_lower_deg: Optional[float] = None,
    scale_upper_deg: Optional[float] = None,
    passes: int = 16,
    max_reuses: Optional[int] = None,
    min_pair_sep_frac: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-cell quota, multi-pass quad builder — mirrors astrometry.net's
    build-astrometry-index (-p/-R/-L): each of `passes` sweeps tries to add
    ONE quad centered in this cell, so the cell ends up with at most
    `passes` quads rather than an uncontrolled, star-density-dependent count.

    Earlier versions of this function emitted every valid C/D combination for
    every qualifying A-B pair in a single sweep, which could produce a
    combinatorial burst of near-duplicate quads (sharing 2-3 of 4 stars) in
    dense fields and badly oversubscribe code space there, while sparse
    fields got comparatively few quads -- the opposite of the uniform
    per-cell density the reference index-building strategy targets
    (Lang et al. 2010 sec 2.3; index-building controls the false-match rate
    by controlling quad density, sec 3.1.6).

    Cells are HEALPix pixels (`healpix`/`pix`), not RA/Dec boxes: a lat/lon
    grid has non-uniform physical cell area (shrinking by cos(dec) toward
    the poles), which biases quad density away from the uniform sky
    coverage astrometry.net's own HEALPix-gridded index-building targets.

    Each pass only considers stars used fewer than `max_times_used` times so
    far (reuse is scoped to one grid-cell build, matching build_index's
    per-cell catalog query -- it does not persist across cells). If a given
    pass can't find a quad under the base cap -- typically because earlier
    passes in this same cell have already exhausted the local candidate
    stars, not because none exist at all -- and `max_reuses` is given, that
    pass retries with the reuse cap escalated one star-use at a time
    (astrometry.net's -L) until it succeeds or `max_reuses` is reached --
    better to reuse a star than skip the pass.

    `min_pair_sep_frac`, if set, rejects geometrically ill-conditioned quads
    (see `_find_one_quad`) -- opt-in via `filter_ill_conditioned` on
    build_index/build_index_tiered while this is being evaluated.

    Returns
    -------
    codes           : (M, 4) float32 — hash codes for valid quads
    quad_source_ids : (M, 4) int64   — source IDs in canonical ABCD order
    """
    star_ra    = np.asarray(star_ra,    dtype=float)
    star_dec   = np.asarray(star_dec,   dtype=float)
    star_mv    = np.asarray(star_mv,    dtype=float)
    source_ids = np.asarray(source_ids, dtype=np.int64)

    if len(star_ra) < 4:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

    # Brightest-first ordering (smallest mV = brightest)
    order      = np.argsort(star_mv)
    star_ra    = star_ra[order]
    star_dec   = star_dec[order]
    source_ids = source_ids[order]

    scale_lo = scale_lower_deg if scale_lower_deg is not None else 0.0
    scale_hi = scale_upper_deg  # may be None (no upper bound)

    # Flat-sky projected KD-tree: x = ra · cos(dec_center), y = dec
    # Valid for cells up to ~10° (HEALPix cell resolution is typically 0.5°–2°).
    pix_lon, pix_lat = healpix.healpix_to_lonlat(pix)
    dec_center = pix_lat.to(u.deg).value
    cos_dec    = np.cos(np.radians(dec_center))
    pts = np.column_stack([star_ra * cos_dec, star_dec])
    spatial_tree = cKDTree(pts)

    # Upper bound for the KD-tree radius query.
    # When scale_hi is finite use it directly; otherwise cap at 3× the cell's
    # angular resolution (the search region fed into this function spans the
    # cell plus a padded neighbourhood).
    if scale_hi is not None:
        r_hi = scale_hi
    else:
        cell_res_deg = nside_to_pixel_resolution(healpix.nside).to(u.deg).value
        r_hi = 3.0 * cell_res_deg

    times_used = np.zeros(len(star_ra), dtype=int)
    used_quads: set = set()
    codes_out: List[np.ndarray] = []
    sids_out:  List[np.ndarray] = []

    def _attempt(reuse_cap: int) -> bool:
        result = _find_one_quad(
            pts, star_ra, star_dec, source_ids, times_used, reuse_cap,
            spatial_tree, r_hi, scale_lo, scale_hi, healpix, pix, used_quads,
            min_pair_sep_frac,
        )
        if result is None:
            return False
        code, ordered_sids, local = result
        codes_out.append(code)
        sids_out.append(ordered_sids)
        used_quads.add(frozenset(local))
        for k in local:
            times_used[k] += 1
        return True

    # Reuse-cap escalation must happen per failed pass, not once at the end:
    # times_used starts at zero, so the *first* pass always succeeds under
    # the base cap if any valid quad exists at all -- escalation only ever
    # matters once earlier passes in this cell have exhausted the local
    # candidate stars, which happens partway through the `passes` sweep.
    for _ in range(passes):
        if _attempt(max_times_used):
            continue
        if max_reuses is not None and max_reuses > max_times_used:
            cap = max_times_used
            while cap < max_reuses:
                cap += 1
                if _attempt(cap):
                    break

    if codes_out:
        return (
            np.array(codes_out, dtype=np.float32),
            np.array(sids_out,  dtype=np.int64),
        )
    return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.int64)


def _healpix_cells_for_region(
    sky_region: dict, cell_deg: float
) -> Tuple[HEALPix, np.ndarray, float]:
    """
    HEALPix grid + the cell indices overlapping a sky_region box.

    Shared by build_index and build_index_tiered so both iterate exactly the
    same cells for a given (sky_region, cell_deg).

    Returns (healpix, cell_pix, cell_res_deg).
    """
    min_ra,  max_ra  = sky_region["min_ra"],  sky_region["max_ra"]
    min_dec, max_dec = sky_region["min_dec"], sky_region["max_dec"]

    nside = pixel_resolution_to_nside(cell_deg * u.deg, round="up")
    healpix = HEALPix(nside=nside, order="nested")
    cell_res_deg = nside_to_pixel_resolution(nside).to(u.deg).value

    # Find HEALPix cells overlapping the region via a cone query (fast, no
    # need to materialize every pixel on the sky), then trim to the exact
    # box -- the cone is a superset since it bounds the box's diagonal.
    # For a full-RA-range region (e.g. a declination band), fall back to
    # enumerating every pixel and filtering by dec -- a single cone can't
    # tightly bound a 360°-wide box, and cone-searching wastefully wide
    # radii is slower than just listing the (still cheap) full pixel set.
    if max_ra - min_ra >= 359.999:
        all_idx = np.arange(healpix.npix)
        lon, lat = healpix.healpix_to_lonlat(all_idx)
        cand_dec = lat.to(u.deg).value
        cell_pix = all_idx[(cand_dec >= min_dec) & (cand_dec <= max_dec)]
        return healpix, cell_pix, cell_res_deg

    center_ra  = (min_ra + max_ra) / 2
    center_dec = (min_dec + max_dec) / 2
    half_diag_deg  = 0.5 * np.hypot(max_ra - min_ra, max_dec - min_dec)
    search_radius  = (half_diag_deg + 2 * cell_res_deg) * u.deg
    candidate_pix  = healpix.cone_search_lonlat(center_ra * u.deg, center_dec * u.deg, search_radius)

    cand_lon, cand_lat = healpix.healpix_to_lonlat(candidate_pix)
    cand_ra  = cand_lon.to(u.deg).value % 360.0
    cand_dec = cand_lat.to(u.deg).value
    in_region = (
        (cand_ra  >= min_ra)  & (cand_ra  <= max_ra) &
        (cand_dec >= min_dec) & (cand_dec <= max_dec)
    )
    return healpix, candidate_pix[in_region], cell_res_deg


def build_index(
    sky_region: dict,
    cell_deg: float,
    code_index_path: str,
    star_index_path: str,
    catalog_name: str = "Gaia",
    catalog_path: Optional[str] = None,
    row_limit: int = 1000,
    max_times_used: int = 8,
    scale_lower_deg: Optional[float] = None,
    scale_upper_deg: Optional[float] = None,
    passes: int = 16,
    max_reuses: Optional[int] = None,
    filter_ill_conditioned: bool = False,
    min_pair_sep_frac: float = 0.05,
    verbose: bool = False,
) -> Tuple[CodeSpaceTree, StarPositionTree]:
    """
    Build and persist the code-space and star-position indices for a sky region.

    Parameters
    ----------
    sky_region       : dict — keys: min_ra, max_ra, min_dec, max_dec (degrees)
    cell_deg         : target grid cell size (degrees). Cells are equal-area
        HEALPix pixels, not an RA/Dec box grid -- a lat/lon grid has
        non-uniform physical cell area (shrinking by cos(dec) toward the
        poles), which would bias quad density away from uniform sky
        coverage. The actual Nside is chosen so cell resolution is at or
        below `cell_deg` (astrometry.net's -N, specified here by size).
    code_index_path  : output path for CodeSpaceTree  (.joblib)
    star_index_path  : output path for StarPositionTree (.joblib)
    catalog_name     : "Gaia" (live network) or "sstrc7" (local catalog directory)
    catalog_path     : sstrc7 catalog directory (only meaningful when
        catalog_name == "sstrc7"; None resolves from $SSTRC7_PATH / ~/.sstrc7)
    scale_lower_deg  : only include quads with AB separation ≥ this (degrees)
    scale_upper_deg  : only include quads with AB separation ≤ this (degrees)
    passes           : target quads per grid cell (astrometry.net's -p, default 16)
    max_reuses       : if a cell still has zero quads after `passes` sweeps,
        escalate the per-star reuse cap up to this value to rescue it
        (astrometry.net's -L). None (default) disables escalation.
    filter_ill_conditioned : reject quads with a near-duplicate/near-collinear
        star pair (see `_find_one_quad`) -- confirmed to cause large WCS
        errors even with zero positional noise. Off by default while this is
        being evaluated; turn on to compare index quality with it enabled.
    min_pair_sep_frac : minimum pairwise star separation, as a fraction of
        the quad's own A-B separation, required when `filter_ill_conditioned`
        is True. Ignored otherwise.
    verbose          : print a full "cell N: stars/codes" line per cell
        (via tqdm.write, so it doesn't corrupt the progress bar) in addition
        to the bar. Off by default -- for a production build spanning
        thousands of cells, one line per cell floods the terminal; the bar
        alone (with running totals in its postfix) is the useful signal.

    Returns
    -------
    (CodeSpaceTree, StarPositionTree)
    """
    healpix, cell_pix, cell_res_deg = _healpix_cells_for_region(sky_region, cell_deg)
    nside = healpix.nside

    all_codes:    List[np.ndarray] = []
    all_quad_ids: List[np.ndarray] = []
    all_ra:       List[float]      = []
    all_dec:      List[float]      = []
    all_src_ids:  List[int]        = []
    all_mV:       List[float]      = []
    seen_src_ids: set              = set()

    from astrometry.catalog_queries import query_catalog  # deferred to avoid astroquery at import time
    from tqdm import tqdm  # deferred so solve-only users don't need it installed

    print(f"[build_index] {len(cell_pix)} HEALPix cells (nside={nside}, "
          f"~{cell_res_deg:.3f}° resolution) over "
          f"RA [{sky_region['min_ra']}, {sky_region['max_ra']}]  "
          f"Dec [{sky_region['min_dec']}, {sky_region['max_dec']}]")

    total_codes = 0
    total_stars = 0
    bar = tqdm(cell_pix, desc="[build_index] cells", unit="cell")
    for pix in bar:
        pix = int(pix)
        p_lon, p_lat = healpix.healpix_to_lonlat(pix)
        cell_ra  = p_lon.to(u.deg).value
        cell_dec = p_lat.to(u.deg).value

        # Padded box around the cell centre to avoid edge effects: half the
        # cell's own extent plus one full cell of margin, matching the old
        # RA/Dec grid's 3×3-neighbourhood padding.
        pad = 1.5 * cell_res_deg
        search_ra_min  = cell_ra  - pad
        search_ra_max  = cell_ra  + pad
        search_dec_min = np.clip(cell_dec - pad, -90.0, 90.0)
        search_dec_max = np.clip(cell_dec + pad, -90.0, 90.0)

        # catalog_path is only forwarded when set -- query_gaia_catalog
        # has no such parameter and would raise on an unexpected kwarg.
        extra_kwargs = {"row_limit": row_limit}
        if catalog_path is not None:
            extra_kwargs["catalog_path"] = catalog_path

        try:
            stars = query_catalog(
                catalog_name, cell_ra, cell_dec,
                fov_width  = search_ra_max  - search_ra_min,
                fov_height = search_dec_max - search_dec_min,
                **extra_kwargs,
            )
        except Exception as exc:
            tqdm.write(f"  cell {pix}: catalog query failed — {exc}")
            continue

        s_ra   = np.asarray(stars["ra"],        dtype=float)
        s_dec  = np.asarray(stars["dec"],       dtype=float)
        s_mV   = np.asarray(stars["mV"],        dtype=float)
        s_sids = np.asarray(stars["source_id"], dtype=np.int64)

        codes, quad_sids = _generate_indexed_features(
            s_ra, s_dec, s_mV, s_sids,
            healpix=healpix,
            pix=pix,
            max_times_used=max_times_used,
            scale_lower_deg=scale_lower_deg,
            scale_upper_deg=scale_upper_deg,
            passes=passes,
            max_reuses=max_reuses,
            min_pair_sep_frac=min_pair_sep_frac if filter_ill_conditioned else None,
        )

        if len(codes):
            all_codes.append(codes)
            all_quad_ids.append(quad_sids)

        # Accumulate unique catalogue stars for the position tree
        new_mask = np.array([sid not in seen_src_ids for sid in s_sids])
        if new_mask.any():
            all_ra.extend(s_ra[new_mask].tolist())
            all_dec.extend(s_dec[new_mask].tolist())
            all_src_ids.extend(s_sids[new_mask].tolist())
            all_mV.extend(s_mV[new_mask].tolist())
            seen_src_ids.update(int(s) for s in s_sids[new_mask])

        total_codes += len(codes)
        total_stars = len(seen_src_ids)
        bar.set_postfix(codes=total_codes, stars=total_stars, refresh=False)
        if verbose:
            tqdm.write(f"  cell {pix:8d}: {len(s_ra):4d} stars  "
                       f"{len(codes):5d} codes")

    if not all_codes:
        raise RuntimeError(
            "No hash codes generated. Check sky region bounds and catalog access."
        )

    codes_arr    = np.vstack(all_codes)
    quad_ids_arr = np.vstack(all_quad_ids)
    ra_arr       = np.array(all_ra,      dtype=np.float64)
    dec_arr      = np.array(all_dec,     dtype=np.float64)
    src_ids_arr  = np.array(all_src_ids, dtype=np.int64)
    mV_arr       = np.array(all_mV,      dtype=np.float64)

    code_tree = CodeSpaceTree(
        codes_arr, quad_ids_arr,
        scale_lower_deg=scale_lower_deg,
        scale_upper_deg=scale_upper_deg,
    )
    star_tree = StarPositionTree(ra_arr, dec_arr, src_ids_arr, mV_arr)

    Path(code_index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(star_index_path).parent.mkdir(parents=True, exist_ok=True)
    code_tree.save(code_index_path)
    star_tree.save(star_index_path)

    print(f"\n[build_index] wrote {len(codes_arr):,} codes  → {code_index_path}")
    print(f"[build_index] wrote {len(ra_arr):,} stars   → {star_index_path}")

    return code_tree, star_tree


def build_index_tiered(
    sky_region: dict,
    tier_indices: List[int],
    index_dir: str,
    star_index_path: str,
    cell_deg: float = 2.0,
    catalog_name: str = "Gaia",
    catalog_path: Optional[str] = None,
    row_limit: int = 500,
    max_times_used: int = 8,
    passes: int = 16,
    max_reuses: Optional[int] = None,
    filter_ill_conditioned: bool = False,
    min_pair_sep_frac: float = 0.05,
    verbose: bool = False,
) -> Tuple[List[CodeSpaceTree], StarPositionTree]:
    """
    Build one CodeSpaceTree per scale tier for a sky region.

    Queries the catalog exactly ONCE per cell and reuses that same star data
    to build every tier's quads, instead of re-querying per tier. Tiers only
    differ in which quads get built from a cell's stars (scale_lower/upper),
    not in the underlying star field -- so an earlier version of this
    function, which called build_index once per tier, paid the catalog-query
    cost (measured as the dominant cost: 0.2-2s/cell, vs 15-55ms/cell for
    quad-building) N times over for N tiers. Sharing the query cuts total
    build time by roughly a factor of N.

    Parameters
    ----------
    sky_region    : dict — min_ra, max_ra, min_dec, max_dec (degrees)
    tier_indices  : list of ints into SCALE_TIERS (e.g. [8, 9, 10, 11])
    index_dir     : directory to write per-tier code index files
    star_index_path : path for the shared StarPositionTree
    cell_deg      : target HEALPix grid cell size in degrees (see build_index)
    catalog_name  : "Gaia" (live network) or "sstrc7" (local catalog directory)
    catalog_path  : sstrc7 catalog directory (only meaningful when
        catalog_name == "sstrc7"; None resolves from $SSTRC7_PATH / ~/.sstrc7)
    row_limit     : max stars per catalog query
    max_times_used: max quads per star per tier
    passes        : target quads per grid cell per tier (astrometry.net's -p)
    max_reuses    : escalate the reuse cap to rescue empty cells, up to this
        value (astrometry.net's -L). None (default) disables escalation.
    filter_ill_conditioned : reject near-degenerate quads (see build_index).
        Off by default while this is being evaluated.
    min_pair_sep_frac : threshold used when `filter_ill_conditioned` is True.
    verbose       : print a full per-cell-per-tier line (see build_index).
        Off by default.

    Returns
    -------
    (list of CodeSpaceTree, StarPositionTree)
    """
    for tier in tier_indices:
        if tier not in SCALE_TIERS:
            raise ValueError(f"Tier {tier} not in SCALE_TIERS (valid: 0–15)")

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    Path(star_index_path).parent.mkdir(parents=True, exist_ok=True)

    healpix, cell_pix, cell_res_deg = _healpix_cells_for_region(sky_region, cell_deg)

    all_codes:    dict = {tier: [] for tier in tier_indices}
    all_quad_ids: dict = {tier: [] for tier in tier_indices}
    all_ra:       List[float] = []
    all_dec:      List[float] = []
    all_src_ids:  List[int]   = []
    all_mV:       List[float] = []
    seen_src_ids: set = set()

    from astrometry.catalog_queries import query_catalog  # deferred to avoid astroquery at import time
    from tqdm import tqdm  # deferred so solve-only users don't need it installed

    print(f"[build_index_tiered] {len(cell_pix)} HEALPix cells (nside={healpix.nside}, "
          f"~{cell_res_deg:.3f}° resolution) × {len(tier_indices)} tier(s) over "
          f"RA [{sky_region['min_ra']}, {sky_region['max_ra']}]  "
          f"Dec [{sky_region['min_dec']}, {sky_region['max_dec']}]")

    total_codes = 0
    t0 = time.perf_counter()
    bar = tqdm(cell_pix, desc="[build_index_tiered] cells", unit="cell")
    for pix in bar:
        pix = int(pix)
        p_lon, p_lat = healpix.healpix_to_lonlat(pix)
        cell_ra  = p_lon.to(u.deg).value
        cell_dec = p_lat.to(u.deg).value

        pad = 1.5 * cell_res_deg
        search_ra_min  = cell_ra  - pad
        search_ra_max  = cell_ra  + pad
        search_dec_min = np.clip(cell_dec - pad, -90.0, 90.0)
        search_dec_max = np.clip(cell_dec + pad, -90.0, 90.0)

        extra_kwargs = {"row_limit": row_limit}
        if catalog_path is not None:
            extra_kwargs["catalog_path"] = catalog_path

        try:
            stars = query_catalog(
                catalog_name, cell_ra, cell_dec,
                fov_width  = search_ra_max  - search_ra_min,
                fov_height = search_dec_max - search_dec_min,
                **extra_kwargs,
            )
        except Exception as exc:
            tqdm.write(f"  cell {pix}: catalog query failed — {exc}")
            continue

        s_ra   = np.asarray(stars["ra"],        dtype=float)
        s_dec  = np.asarray(stars["dec"],       dtype=float)
        s_mV   = np.asarray(stars["mV"],        dtype=float)
        s_sids = np.asarray(stars["source_id"], dtype=np.int64)

        # Accumulate unique catalogue stars for the shared position tree --
        # once per cell, not once per (cell, tier).
        new_mask = np.array([sid not in seen_src_ids for sid in s_sids])
        if new_mask.any():
            all_ra.extend(s_ra[new_mask].tolist())
            all_dec.extend(s_dec[new_mask].tolist())
            all_src_ids.extend(s_sids[new_mask].tolist())
            all_mV.extend(s_mV[new_mask].tolist())
            seen_src_ids.update(int(s) for s in s_sids[new_mask])

        for tier in tier_indices:
            lo, hi = SCALE_TIERS[tier]
            codes, quad_sids = _generate_indexed_features(
                s_ra, s_dec, s_mV, s_sids,
                healpix=healpix,
                pix=pix,
                max_times_used=max_times_used,
                scale_lower_deg=lo,
                scale_upper_deg=hi,
                passes=passes,
                max_reuses=max_reuses,
                min_pair_sep_frac=min_pair_sep_frac if filter_ill_conditioned else None,
            )
            if len(codes):
                all_codes[tier].append(codes)
                all_quad_ids[tier].append(quad_sids)
                total_codes += len(codes)
            if verbose:
                tqdm.write(f"  cell {pix:8d} tier {tier:2d}: {len(codes):5d} codes")

        bar.set_postfix(codes=total_codes, stars=len(seen_src_ids), refresh=False)

    if not any(all_codes[t] for t in tier_indices):
        raise RuntimeError(
            "No hash codes generated. Check sky region bounds and catalog access."
        )

    ra_arr      = np.array(all_ra,      dtype=np.float64)
    dec_arr     = np.array(all_dec,     dtype=np.float64)
    src_ids_arr = np.array(all_src_ids, dtype=np.int64)
    mV_arr      = np.array(all_mV,      dtype=np.float64)
    star_tree = StarPositionTree(ra_arr, dec_arr, src_ids_arr, mV_arr)
    star_tree.save(star_index_path)
    print(f"\n[build_index_tiered] wrote {len(ra_arr):,} stars → {star_index_path}")

    code_trees: List[CodeSpaceTree] = []
    for tier in tier_indices:
        lo, hi = SCALE_TIERS[tier]
        if all_codes[tier]:
            codes_arr    = np.vstack(all_codes[tier])
            quad_ids_arr = np.vstack(all_quad_ids[tier])
        else:
            codes_arr    = np.empty((0, 4), dtype=np.float32)
            quad_ids_arr = np.empty((0, 4), dtype=np.int64)
        code_tree = CodeSpaceTree(
            codes_arr, quad_ids_arr, scale_lower_deg=lo, scale_upper_deg=hi,
        )
        code_path = str(Path(index_dir) / f"codes_tier{tier:02d}.joblib")
        code_tree.save(code_path)
        print(f"[build_index_tiered] wrote {len(codes_arr):,} codes → {code_path}  "
              f"[tier {tier}, {lo:.3f}°–{hi:.3f}°]")
        code_trees.append(code_tree)

    print(f"\n[build_index_tiered] all {len(tier_indices)} tier(s) done in "
          f"{time.perf_counter()-t0:.0f}s")
    return code_trees, star_tree


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m astrometry.kd_tree",
        description="Build or query the astrometric hash-code index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- build ---------------------------------------------------------------
    p_build = sub.add_parser(
        "build",
        help="Build a single-tier or multi-tier index for a sky region.",
    )
    p_build.add_argument("--min_ra",  type=float, required=True)
    p_build.add_argument("--max_ra",  type=float, required=True)
    p_build.add_argument("--min_dec", type=float, required=True)
    p_build.add_argument("--max_dec", type=float, required=True)
    p_build.add_argument("--cell_deg", type=float, default=2.0,
                         help="Target HEALPix grid cell size (deg, default 2.0). "
                              "Equal-area cells, not an RA/Dec box.")
    p_build.add_argument("--tiers", type=int, nargs="+", default=None,
                         help="Scale tier indices to build (e.g. --tiers 8 9 10 11). "
                              "If omitted, builds a single untiered index.")
    p_build.add_argument("--index_dir", type=str, default="indices/",
                         help="Output directory for tiered builds (default: indices/).")
    p_build.add_argument("--code_index", type=str, default="index_codes.joblib",
                         help="Output path for single-tier CodeSpaceTree.")
    p_build.add_argument("--star_index", type=str, default="index_stars.joblib",
                         help="Output path for StarPositionTree.")
    p_build.add_argument("--catalog",   type=str, default="Gaia",
                         help="Catalog to query: 'Gaia' (live network) or "
                              "'sstrc7' (local catalog directory).")
    p_build.add_argument("--catalog_path", type=str, default=None,
                         help="sstrc7 catalog directory (only used when "
                              "--catalog sstrc7; defaults to $SSTRC7_PATH "
                              "or ~/.sstrc7).")
    p_build.add_argument("--row_limit", type=int, default=500,
                         help="Stars per catalog query (default 500).")
    p_build.add_argument("--max_times_used", type=int, default=8,
                         help="Max quads per star, ie reuse cap (default 8).")
    p_build.add_argument("--passes", type=int, default=16,
                         help="Target quads per grid cell (default 16).")
    p_build.add_argument("--max_reuses", type=int, default=None,
                         help="Escalate the reuse cap up to this value to "
                              "rescue cells left empty after --passes sweeps "
                              "(default: no escalation).")
    p_build.add_argument("--filter_ill_conditioned", action="store_true",
                         help="Reject quads with a near-duplicate/near-"
                              "collinear star pair (default: off, opt-in "
                              "while this is being evaluated).")
    p_build.add_argument("--min_pair_sep_frac", type=float, default=0.05,
                         help="Minimum pairwise star separation as a "
                              "fraction of the quad's A-B separation, used "
                              "only when --filter_ill_conditioned is set "
                              "(default 0.05).")
    p_build.add_argument("--scale_lower", type=float, default=None,
                         help="Min AB separation for single-tier build (deg).")
    p_build.add_argument("--scale_upper", type=float, default=None,
                         help="Max AB separation for single-tier build (deg).")
    p_build.add_argument("--verbose", action="store_true",
                         help="Print a full per-cell line for every cell "
                              "(default: off -- just the progress bar).")

    # ---- query ---------------------------------------------------------------
    p_query = sub.add_parser(
        "query", help="Range-search a saved CodeSpaceTree with a hash code."
    )
    p_query.add_argument("--code_index", type=str, required=True)
    p_query.add_argument("--code", type=float, nargs=4, required=True,
                         metavar=("XC", "YC", "XD", "YD"))
    p_query.add_argument("--radius", type=float, default=0.02)

    # ---- info ----------------------------------------------------------------
    p_info = sub.add_parser("info", help="Print summary of a saved index.")
    p_info.add_argument("--code_index", type=str, default=None)
    p_info.add_argument("--star_index", type=str, default=None)

    args = parser.parse_args()

    if args.command == "build":
        region = dict(
            min_ra=args.min_ra,  max_ra=args.max_ra,
            min_dec=args.min_dec, max_dec=args.max_dec,
        )
        if args.tiers:
            build_index_tiered(
                sky_region=region,
                tier_indices=args.tiers,
                index_dir=args.index_dir,
                star_index_path=args.star_index,
                cell_deg=args.cell_deg,
                catalog_name=args.catalog,
                catalog_path=args.catalog_path,
                row_limit=args.row_limit,
                max_times_used=args.max_times_used,
                passes=args.passes,
                max_reuses=args.max_reuses,
                filter_ill_conditioned=args.filter_ill_conditioned,
                min_pair_sep_frac=args.min_pair_sep_frac,
                verbose=args.verbose,
            )
        else:
            build_index(
                sky_region=region,
                cell_deg=args.cell_deg,
                code_index_path=args.code_index,
                star_index_path=args.star_index,
                catalog_name=args.catalog,
                catalog_path=args.catalog_path,
                row_limit=args.row_limit,
                max_times_used=args.max_times_used,
                scale_lower_deg=args.scale_lower,
                scale_upper_deg=args.scale_upper,
                passes=args.passes,
                max_reuses=args.max_reuses,
                filter_ill_conditioned=args.filter_ill_conditioned,
                min_pair_sep_frac=args.min_pair_sep_frac,
                verbose=args.verbose,
            )

    elif args.command == "query":
        tree = CodeSpaceTree.load(args.code_index)
        q    = np.array(args.code, dtype=np.float32)
        hits = tree.range_search(q, radius=args.radius)
        print(f"Query code  : {q.tolist()}")
        print(f"Radius      : {args.radius}")
        print(f"Index size  : {len(tree.codes):,} quads")
        print(f"Matches     : {len(hits)}")
        for k, sids in enumerate(hits[:20]):
            print(f"  [{k:3d}]  source_ids = {sids.tolist()}")
        if len(hits) > 20:
            print(f"  ... and {len(hits) - 20} more")

    elif args.command == "info":
        if args.code_index:
            t = CodeSpaceTree.load(args.code_index)
            print(f"CodeSpaceTree  : {args.code_index}")
            print(f"  quads        : {len(t.codes):,}")
            print(f"  code range   : min={t.codes.min():.4f}  max={t.codes.max():.4f}")
        if args.star_index:
            s = StarPositionTree.load(args.star_index)
            print(f"StarPositionTree : {args.star_index}")
            print(f"  stars          : {len(s.ra):,}")
            print(f"  RA range       : [{s.ra.min():.2f}, {s.ra.max():.2f}] deg")
            print(f"  Dec range      : [{s.dec.min():.2f}, {s.dec.max():.2f}] deg")

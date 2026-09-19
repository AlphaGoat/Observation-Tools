"""
Projects observations from pixel space to RADEC coordinates on the celestial
sphere, given the WCS for a frame.

In a rate-tracked observation the plate solver returns a WCS fitted to the
background star trails.  This module inverts that WCS to convert satellite
detections — which come out of SEP as pixel (x, y) positions — into celestial
(RA, Dec) coordinates that the MHT associator can consume directly.

Supported WCS types
-------------------
kd_tree.WCS (gnomonic TAN model)
    Forward:  [x, y]ᵀ = A @ [ξ, η, 1]ᵀ          A is (2, 3)
    where (ξ, η) are gnomonic tangent-plane coordinates (degrees) relative to
    the tangent point (wcs.ra0, wcs.dec0).
    Inverse:  (ξ, η) = M⁻¹ @ ([x, y]ᵀ − t)  then gnomonic⁻¹

    This model is accurate across wide FOVs (multiple degrees) because it
    correctly handles spherical curvature and RA compression near the poles.

    Legacy kd_tree.WCS without ra0/dec0 attributes falls back to the flat
    affine approximation (still accurate to < 0.1″ for FOVs ≤ ~1–2°).

astropy.wcs.WCS
    Inverted via all_pix2world(), which supports full TAN/SIN projections
    and distortion corrections.  Pass origin=0 (0-indexed, matching SEP).

Public API
----------
pix_to_radec(wcs, x_pix, y_pix)
    Low-level array conversion.  Returns (ra_array, dec_array) in degrees.

project_satellite_detections(wcs, satellites, t_start_s, t_end_s)
    Convert a list of SatelliteDetection objects (from sep_extractor) to
    (ra, dec, t_start, t_end) observation tuples ready for the associator.

project_star_streaks(wcs, streaks)
    Convert star-trail midpoints to sky coordinates.  Useful for verifying
    the WCS solution: projected midpoints should match the catalogue.

project_frame(wcs, satellite_detections, t_start_s, t_end_s)
    Convenience wrapper — combines pix_to_radec and observation formatting
    into a single call that returns the associator-ready observation list
    for one image frame.

affine_A_from_astropy_wcs(astropy_wcs, x_ref, y_ref)
    Build a 2×3 affine A matrix from an astropy.wcs.WCS by sampling it at
    three reference pixels and fitting with least squares.  Use this to obtain
    a JSON-serializable WCS when calling the /project Flask API from code that
    already has an astropy WCS.

    Why not a direct attribute?
    ---------------------------
    Astropy stores a WCS as (CRPIX, CRVAL, CD-matrix), which is a gnomonic
    (TAN) projection — an inherently non-linear sky-to-pixel model.  There is
    no single attribute that gives our flat affine A directly.  The closest
    candidates and why they do not work as drop-ins:

      wcs.wcs.cd    — the 2×2 CD matrix maps pixel *offsets* from CRPIX to
                      intermediate world coordinates (IWC), not to RA/Dec.
                      IWC folds in a cos(dec) projection factor and is not the
                      same as a RA/Dec offset.

      wcs.wcs.crpix — 1-indexed reference pixel (FITS convention).

      wcs.wcs.crval — reference (RA, Dec) in degrees.

    Fitting via sampling is exact over any small FOV (< ~1–2°) and stays
    correct regardless of which FITS WCS variant (CD, PC+CDELT, SIP) the
    caller holds.

Author: Peter Thomas
Date: 10 September 2026
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np


# ── Type aliases ──────────────────────────────────────────────────────────────

# (ra_deg, dec_deg, t_start_s, t_end_s) — one observation for the associator
Observation = Tuple[float, float, float, float]


# ── WCS inversion ─────────────────────────────────────────────────────────────

def _affine_inv(wcs_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute the inverse of an affine WCS matrix once so it can be reused
    for many point projections without repeated np.linalg.inv calls.

    Parameters
    ----------
    wcs_A : (2, 3) float array
        The A matrix from kd_tree.WCS:  [x, y]ᵀ = A @ [ra, dec, 1]ᵀ

    Returns
    -------
    (M_inv, t)
        M_inv : (2, 2) — inverse of the pixel Jacobian A[:, :2]
        t     : (2,)   — translation vector A[:, 2]
    """
    M = wcs_A[:, :2]
    t = wcs_A[:, 2]
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "WCS pixel Jacobian is singular — the WCS solution is degenerate."
        ) from exc
    return M_inv, t


def pix_to_radec(
    wcs: object,
    x_pix: Union[float, Sequence[float], np.ndarray],
    y_pix: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to celestial (RA, Dec) using a plate WCS.

    Accepts either our affine kd_tree.WCS or an astropy.wcs.WCS object.
    All pixel coordinates are 0-indexed, matching SEP output.

    Parameters
    ----------
    wcs : kd_tree.WCS or astropy.wcs.WCS
        The plate solution for this image frame.
    x_pix, y_pix : array-like
        Pixel column(s) and row(s) to project.  Scalars or 1-D arrays.

    Returns
    -------
    ra  : (N,) float64 array — right ascension in degrees [0, 360)
    dec : (N,) float64 array — declination in degrees [−90, +90]
    """
    x = np.asarray(x_pix, dtype=np.float64).ravel()
    y = np.asarray(y_pix, dtype=np.float64).ravel()

    # ── Our kd_tree.WCS (gnomonic or legacy flat) ────────────────────────────
    if hasattr(wcs, "A"):
        M_inv, t = _affine_inv(np.asarray(wcs.A, dtype=np.float64))
        xi_eta = M_inv @ np.vstack([x - t[0], y - t[1]])   # (2, N) tangent-plane
        ra0  = getattr(wcs, "ra0",  None)
        dec0 = getattr(wcs, "dec0", None)
        if ra0 is not None and dec0 is not None:
            # Gnomonic WCS: deproject from tangent plane to sky
            from astrometry.kd_tree import _gnomonic_inv
            return _gnomonic_inv(xi_eta[0], xi_eta[1], float(ra0), float(dec0))
        # Legacy flat WCS: tangent-plane coords treated directly as RA/Dec
        ra  = xi_eta[0] % 360.0
        dec = np.clip(xi_eta[1], -90.0, 90.0)
        return ra, dec

    # ── astropy WCS ───────────────────────────────────────────────────────────
    if hasattr(wcs, "all_pix2world"):
        pix = np.column_stack([x, y])
        sky = wcs.all_pix2world(pix, 0)   # origin=0 → 0-indexed pixels
        ra  = sky[:, 0] % 360.0
        dec = np.clip(sky[:, 1], -90.0, 90.0)
        return ra, dec

    raise TypeError(
        f"Unsupported WCS type {type(wcs).__name__!r}. "
        "Expected kd_tree.WCS (with .A) or astropy.wcs.WCS "
        "(with .all_pix2world)."
    )


# ── Observation projection ────────────────────────────────────────────────────

def project_satellite_detections(
    wcs: object,
    satellites: Sequence,
    t_start_s: float,
    t_end_s: float,
) -> List[Observation]:
    """
    Project satellite point-source detections to associator observation tuples.

    Each element of ``satellites`` may be:
      - a SatelliteDetection (from sep_extractor) with .x / .y attributes, or
      - a dict with "x" and "y" keys (e.g. from the /extract API response), or
      - a 2-element sequence [x_pix, y_pix].

    Parameters
    ----------
    wcs : kd_tree.WCS or astropy.wcs.WCS
    satellites : sequence of detections
        Satellite point sources from one image frame.
    t_start_s : float
        UTC epoch of the shutter opening in seconds from any consistent epoch.
    t_end_s : float
        UTC epoch of the shutter closing.  t_end_s > t_start_s.

    Returns
    -------
    List of (ra_deg, dec_deg, t_start_s, t_end_s) tuples — one per detection.
    Ready to pass as one frame in run_multiple_hypothesis_tracking(obs=...).
    """
    if not satellites:
        return []

    xs, ys = [], []
    for sat in satellites:
        if hasattr(sat, "x"):
            xs.append(float(sat.x)); ys.append(float(sat.y))
        elif isinstance(sat, dict):
            xs.append(float(sat["x"])); ys.append(float(sat["y"]))
        else:
            seq = list(sat)
            xs.append(float(seq[0])); ys.append(float(seq[1]))

    ra, dec = pix_to_radec(wcs, xs, ys)

    return [
        (float(ra[i]), float(dec[i]), float(t_start_s), float(t_end_s))
        for i in range(len(ra))
    ]


def project_star_streaks(
    wcs: object,
    streaks: Sequence,
) -> np.ndarray:
    """
    Project star-trail midpoints to sky coordinates.

    The midpoint of each trail is the star's pixel position at mid-exposure,
    so projecting it gives the star's (RA, Dec) at the exposure midpoint.
    This is primarily useful for verifying the WCS solution — the projected
    sky positions should closely match the astrometric catalogue.

    Parameters
    ----------
    wcs : kd_tree.WCS or astropy.wcs.WCS
    streaks : sequence of StarStreak or dicts
        Star trails from sep_extractor (or the /extract API response).
        Each element must expose .x_mid / .y_mid attributes or "x_mid" /
        "y_mid" dict keys.

    Returns
    -------
    (N, 3) float64 array — columns: [ra_deg, dec_deg, flux]
        Rows correspond to elements of ``streaks`` in input order.
    """
    if not streaks:
        return np.empty((0, 3), dtype=np.float64)

    xs, ys, fluxes = [], [], []
    for s in streaks:
        if hasattr(s, "x_mid"):
            xs.append(float(s.x_mid)); ys.append(float(s.y_mid))
            fluxes.append(float(s.flux))
        elif isinstance(s, dict):
            xs.append(float(s["x_mid"])); ys.append(float(s["y_mid"]))
            fluxes.append(float(s.get("flux", 0.0)))
        else:
            seq = list(s)
            xs.append(float(seq[0])); ys.append(float(seq[1]))
            fluxes.append(float(seq[2]) if len(seq) > 2 else 0.0)

    ra, dec = pix_to_radec(wcs, xs, ys)
    return np.column_stack([ra, dec, fluxes])


def project_frame(
    wcs: object,
    satellite_detections: Sequence,
    t_start_s: float,
    t_end_s: float,
) -> List[Observation]:
    """
    Project all satellite detections in one image frame to sky observations.

    Convenience wrapper around project_satellite_detections that matches the
    terminology of the broader pipeline (one "frame" = one exposure).

    Parameters
    ----------
    wcs : kd_tree.WCS or astropy.wcs.WCS
        Plate solution for this frame, derived from the star trails.
    satellite_detections : sequence
        Satellite point sources from sep_extractor or the /extract API.
    t_start_s : float
        Shutter-open epoch in seconds.
    t_end_s : float
        Shutter-close epoch in seconds.

    Returns
    -------
    List of (ra_deg, dec_deg, t_start_s, t_end_s) tuples.

    Example
    -------
    >>> from astrometry.kd_tree import CodeSpaceTree, StarPositionTree
    >>> from source_extraction.sep_extractor import extract_sources, star_array
    >>> from obs.project_obs_from_pixel_to_radec import project_frame
    >>>
    >>> sats, streaks = extract_sources(image)
    >>> result = solver.solve(star_array(streaks), image_height=H, image_width=W)
    >>> if result.solved:
    ...     obs = project_frame(result.wcs, sats, t_start_s=0.0, t_end_s=1.0)
    ...     # obs is ready for run_multiple_hypothesis_tracking(obs=[obs, ...])
    """
    return project_satellite_detections(wcs, satellite_detections, t_start_s, t_end_s)


# ── Astropy WCS → affine A ────────────────────────────────────────────────────

def affine_A_from_astropy_wcs(
    astropy_wcs: object,
    x_ref: float,
    y_ref: float,
    offset_px: float = 50.0,
) -> np.ndarray:
    """
    Build a 2×3 affine A matrix from an astropy.wcs.WCS object.

    Samples the astropy WCS at three pixels near (x_ref, y_ref) and fits the
    affine model [x, y]ᵀ = A @ [ra, dec, 1]ᵀ with least squares.  The result
    can be passed directly to the /project Flask API as ``"wcs": {"A": A.tolist()}``.

    This is the correct approach because astropy's CD/PC + CRPIX + CRVAL
    parameterisation has no single attribute equivalent to our A matrix — see
    module docstring for a full explanation.

    Parameters
    ----------
    astropy_wcs : astropy.wcs.WCS
        Any valid astropy WCS (CD, PC+CDELT, SIP distortions all work because
        we sample via all_pix2world rather than reading internal attributes).
    x_ref, y_ref : float
        Pixel coordinates of the field centre (0-indexed, matching SEP output).
        Use image_width/2 and image_height/2 if unsure.
    offset_px : float
        Step size in pixels used for the three sample points.  Should be large
        enough to avoid numerical noise but small enough that the affine
        approximation holds (50 px is a safe default for fields < 2°).

    Returns
    -------
    A : (2, 3) float64 ndarray
        Affine WCS matrix.  Valid for the region around (x_ref, y_ref) to
        within < 0.1″ for FOVs up to ~1–2°.

    Example
    -------
    >>> from astropy.wcs import WCS as AstropyWCS
    >>> import requests
    >>>
    >>> astropy_wcs = AstropyWCS(fits_header)
    >>> A = affine_A_from_astropy_wcs(astropy_wcs, x_ref=1024, y_ref=1024)
    >>>
    >>> resp = requests.post("http://projector:5003/project", json={
    ...     "wcs":       {"A": A.tolist()},
    ...     "t_start_s": 0.0,
    ...     "t_end_s":   1.0,
    ...     "satellites": [{"x": 1100.0, "y": 980.0}],
    ... })
    """
    test_pix = np.array([
        [x_ref,             y_ref            ],
        [x_ref + offset_px, y_ref            ],
        [x_ref,             y_ref + offset_px],
    ], dtype=np.float64)

    # all_pix2world with origin=0 matches SEP's 0-indexed pixel convention
    test_sky = astropy_wcs.all_pix2world(test_pix, 0)  # (3, 2): [ra, dec]

    # Fit [x, y]ᵀ = A @ [ra, dec, 1]ᵀ  via least squares (same as _fit_wcs)
    M = np.column_stack([test_sky, np.ones(3)])         # (3, 3)
    Ax, *_ = np.linalg.lstsq(M, test_pix[:, 0], rcond=None)
    Ay, *_ = np.linalg.lstsq(M, test_pix[:, 1], rcond=None)
    return np.vstack([Ax, Ay])

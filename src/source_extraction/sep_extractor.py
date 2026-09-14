"""
sep_extractor.py — Satellite and star-streak extraction for rate-tracked observations.

In a rate-track collect the telescope slews to follow the satellite under
observation.  This means:

  - The satellite (and any nearby RSOs) appears as a *compact point source*,
    because the telescope is keeping up with it.
  - Background stars appear as *elongated streaks*, because they drift across
    the detector while the telescope slews.

Classification is therefore the opposite of a sidereal observation:

  point source  →  satellite detection
  streak        →  star trail (used for astrometric plate-solving)

The midpoint of each star streak corresponds to the star's sky position at
mid-exposure.  Passing those midpoints to the plate solver yields a WCS that
is valid at the exposure midpoint — the standard reference time for satellite
astrometry.

Pipeline
--------
1. Estimate and subtract a spatially varying sky background (sep.Background).
2. Detect all sources above a configurable sigma threshold (sep.extract).
3. Classify by elongation (a/b) and semi-major axis length:
     satellite : elongation <  elong_thresh                    (point source)
     star trail: elongation ≥  elong_thresh  AND
                 semi-major ≥  min_streak_px                   (long streak)
     discard   : everything else (cosmic rays, artefacts, …)
4. Optionally filter star trails that deviate from the consensus streak
   direction — all stars should trail at the same angle because the telescope
   has a single slew rate.

Output formats
--------------
  Satellites : list of SatelliteDetection — centroid, flux, FWHM, SNR.
  Star trails: list of StarStreak — midpoint, endpoints, angle, length, flux.

  star_array(streaks) → (N, 3) float array [[y_mid, x_mid, flux], …]
    Drop-in input for plate_solve.solve_field() and PlateSolver.solve().
    The midpoints are the plate-solver's "detected star positions".

Usage (library)
---------------
    from source_extraction.sep_extractor import extract_sources, star_array
    sats, streaks = extract_sources(image_array)
    plate_input   = star_array(streaks)   # → plate solver

Usage (CLI)
-----------
    python -m source_extraction.sep_extractor image.fits \\
        --threshold 3.0 --elong_thresh 3.0 --min_streak 20 --show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import sep
except ImportError as exc:
    raise ImportError("SEP is required: pip install sep") from exc


# ── Detection types ───────────────────────────────────────────────────────────

@dataclass
class SatelliteDetection:
    """
    A compact point source — the satellite under observation or a nearby RSO.

    The telescope is tracking this object, so its image is a round (or
    slightly PSF-elongated) spot rather than a trail.
    """
    x: float          # centroid pixel column (0-indexed)
    y: float          # centroid pixel row    (0-indexed)
    flux: float       # total flux inside the detection isophote (ADU)
    snr: float        # peak / local background RMS
    fwhm_px: float    # estimated FWHM in pixels  (2.355 × sqrt(a · b))
    a: float          # semi-major axis from second moments (pixels)
    b: float          # semi-minor axis from second moments (pixels)
    theta_deg: float  # major-axis position angle, degrees from x-axis


@dataclass
class StarStreak:
    """
    An elongated star trail produced by the telescope tracking a satellite.

    The midpoint (x_mid, y_mid) is the star's pixel position at mid-exposure,
    which is what the plate solver needs.  The endpoints bracket the full trail.
    """
    x_mid: float      # trail midpoint, pixel column (0-indexed)
    y_mid: float      # trail midpoint, pixel row    (0-indexed)
    x1: float         # endpoint 1, column
    y1: float         # endpoint 1, row
    x2: float         # endpoint 2, column
    y2: float         # endpoint 2, row
    length_px: float  # trail length in pixels  (2 × semi-major axis)
    angle_deg: float  # trail angle in degrees from x-axis  (−90 to +90)
    flux: float       # total flux inside the detection isophote (ADU)
    snr: float        # peak / local background RMS
    a: float          # semi-major axis (pixels)
    b: float          # semi-minor axis — encodes the stellar PSF width


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_sources(
    image: np.ndarray,
    *,
    threshold: float = 3.0,
    minarea: int = 5,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.005,
    elong_thresh: float = 3.0,
    min_streak_px: float = 20.0,
    sat_fwhm_max_px: float = 20.0,
    bkg_box_size: int = 64,
    bkg_filter_size: int = 3,
    gain: float = 1.0,
    filter_kernel: Optional[np.ndarray] = None,
    angle_filter_sigma: Optional[float] = 2.0,
) -> Tuple[List[SatelliteDetection], List[StarStreak]]:
    """
    Extract satellite point sources and star trails from a rate-tracked image.

    Parameters
    ----------
    image : 2-D numpy array
        Raw pixel values in any numeric dtype.
    threshold : float
        Detection threshold in multiples of the local background RMS.
    minarea : int
        Minimum number of connected pixels above threshold for a valid source.
    deblend_nthresh : int
        Number of deblending sub-thresholds (SEP default 32).
    deblend_cont : float
        Minimum contrast ratio for deblending (SEP default 0.005).
    elong_thresh : float
        a/b ratio at or above which a detection is a star trail.
        Stars typically show elong >> 3; satellite PSFs < 2.
    min_streak_px : float
        Minimum semi-major axis (pixels) required to accept a star trail.
        Short highly-elongated artefacts (hot pixels, cosmic rays) are
        discarded if their semi-major axis is below this limit.
    sat_fwhm_max_px : float
        Maximum FWHM (pixels) accepted for a satellite point source.
        Objects larger than this are likely extended and discarded.
    bkg_box_size : int
        Background estimation grid cell size in pixels.
    bkg_filter_size : int
        Spatial filter applied to the background grid (in cells).
    gain : float
        Detector gain in electrons/ADU for Poisson noise modelling.
    filter_kernel : (H, W) array or None
        Convolution kernel applied before detection.  None → SEP default
        3×3 Gaussian matched filter.
    angle_filter_sigma : float or None
        If not None, star trails whose angle deviates more than this many
        standard deviations from the population median are rejected.  All
        stars trail at the same angle (one telescope slew rate), so outliers
        are likely artefacts.  Set to None to skip this filter.

    Returns
    -------
    (satellites, star_streaks)
        satellites   : list of SatelliteDetection (point sources).
        star_streaks : list of StarStreak (background star trails).
    """
    # ── Normalise image ───────────────────────────────────────────────────────
    img = np.asarray(image, dtype=np.float64)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    # Handle big-endian arrays from astropy FITS reads
    if img.dtype.byteorder not in ("=", "<", "|"):
        img = img.byteswap().newbyteorder()

    # ── Background estimation and subtraction ─────────────────────────────────
    bkg = sep.Background(
        img,
        bw=bkg_box_size, bh=bkg_box_size,
        fw=bkg_filter_size, fh=bkg_filter_size,
    )
    img_sub = img - bkg.back()
    bkg_rms = bkg.rms()

    # ── Source detection ──────────────────────────────────────────────────────
    kwargs: dict = dict(
        thresh=threshold,
        minarea=minarea,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        gain=gain,
    )
    if filter_kernel is not None:
        kwargs["filter_kernel"] = np.asarray(filter_kernel, dtype=np.float64)

    sources = sep.extract(img_sub, err=bkg_rms, **kwargs)

    # ── Classify detections ───────────────────────────────────────────────────
    sats:    List[SatelliteDetection] = []
    streaks: List[StarStreak]         = []

    for src in sources:
        a         = float(src["a"])
        b         = float(src["b"])
        theta     = float(src["theta"])   # radians, −π/2 to π/2
        x         = float(src["x"])
        y         = float(src["y"])
        flux      = float(src["flux"])
        peak      = float(src["peak"])

        if a <= 0 or b <= 0:
            continue

        elong     = a / b
        fwhm_px   = 2.355 * np.sqrt(a * b)
        theta_deg = np.degrees(theta)

        yi = int(np.clip(round(y), 0, bkg_rms.shape[0] - 1))
        xi = int(np.clip(round(x), 0, bkg_rms.shape[1] - 1))
        local_rms = float(bkg_rms[yi, xi]) or float(bkg.globalrms)
        snr = peak / local_rms if local_rms > 0 else 0.0

        if elong >= elong_thresh and a >= min_streak_px:
            # Star trail — telescope tracked past this star
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            streaks.append(StarStreak(
                x_mid=x,
                y_mid=y,
                x1=x - a * cos_t,
                y1=y - a * sin_t,
                x2=x + a * cos_t,
                y2=y + a * sin_t,
                length_px=2.0 * a,
                angle_deg=theta_deg,
                flux=flux,
                snr=snr,
                a=a,
                b=b,
            ))

        elif elong < elong_thresh and fwhm_px <= sat_fwhm_max_px:
            # Point source — satellite under observation (or nearby RSO)
            sats.append(SatelliteDetection(
                x=x, y=y,
                flux=flux,
                snr=snr,
                fwhm_px=fwhm_px,
                a=a, b=b,
                theta_deg=theta_deg,
            ))

    # ── Angle consistency filter ──────────────────────────────────────────────
    # All star trails share the same direction (one telescope slew rate).
    # Detections that deviate significantly are almost certainly not real stars.
    if angle_filter_sigma is not None and len(streaks) >= 3:
        angles = np.array([s.angle_deg for s in streaks])
        # Wrap to [−90, +90] and use circular-mean logic
        # (angles are already in this range from SEP's theta convention)
        med   = np.median(angles)
        mad   = np.median(np.abs(angles - med))
        sigma = mad * 1.4826  # consistent estimator of std
        if sigma > 0:
            streaks = [s for s in streaks
                       if abs(s.angle_deg - med) <= angle_filter_sigma * sigma]

    return sats, streaks


# ── Convenience helpers ───────────────────────────────────────────────────────

def star_array(streaks: List[StarStreak]) -> np.ndarray:
    """
    Convert a StarStreak list to a (N, 3) float array [[y_mid, x_mid, flux], …].

    This is the format expected by plate_solve.solve_field() and
    PlateSolver.solve().  The midpoints of the star trails are the star
    positions at mid-exposure — the correct reference epoch for WCS fitting.
    """
    if not streaks:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(
        [[s.y_mid, s.x_mid, s.flux] for s in streaks],
        dtype=np.float64,
    )


def streak_stats(streaks: List[StarStreak]) -> dict:
    """
    Summarise the population of star streaks to characterise the telescope
    slew during the exposure.

    Returns a dict with:
      angle_deg      — median trail direction (degrees from x-axis)
      angle_std_deg  — scatter in trail direction (degrees)
      length_px      — median trail length (pixels)
      length_std_px  — scatter in trail length (pixels)
      n              — number of trails used
    """
    if not streaks:
        return dict(angle_deg=None, angle_std_deg=None,
                    length_px=None, length_std_px=None, n=0)
    angles  = np.array([s.angle_deg  for s in streaks])
    lengths = np.array([s.length_px  for s in streaks])
    return dict(
        angle_deg=float(np.median(angles)),
        angle_std_deg=float(np.std(angles)),
        length_px=float(np.median(lengths)),
        length_std_px=float(np.std(lengths)),
        n=len(streaks),
    )


def background_subtract(
    image: np.ndarray,
    box_size: int = 64,
    filter_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate and subtract the sky background.

    Returns
    -------
    (subtracted_image, rms_map, global_rms)
    """
    img = np.asarray(image, dtype=np.float64)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    if img.dtype.byteorder not in ("=", "<", "|"):
        img = img.byteswap().newbyteorder()
    bkg = sep.Background(img, bw=box_size, bh=box_size, fw=filter_size, fh=filter_size)
    return img - bkg.back(), bkg.rms(), float(bkg.globalrms)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _plot(
    image: np.ndarray,
    sats: List[SatelliteDetection],
    streaks: List[StarStreak],
    title: str = "",
) -> None:
    """Show satellite point sources (circles) and star trails (lines) on the image."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.percentile(image, [1, 99])
    ax.imshow(image, origin="lower", cmap="gray",
              vmin=vmin, vmax=vmax, interpolation="nearest")

    for sat in sats:
        circ = mpatches.Circle(
            (sat.x, sat.y), radius=max(sat.fwhm_px, 3.0),
            edgecolor="red", facecolor="none", linewidth=1.2,
        )
        ax.add_patch(circ)
        ax.plot(sat.x, sat.y, "r+", markersize=8, markeredgewidth=1.2)

    for s in streaks:
        ax.add_line(Line2D(
            [s.x1, s.x2], [s.y1, s.y2],
            color="cyan", linewidth=1.0,
        ))
        ax.plot(s.x_mid, s.y_mid, "c.", markersize=4)

    sat_patch    = mpatches.Patch(color="red",  label=f"Satellites ({len(sats)})")
    streak_patch = mpatches.Patch(color="cyan", label=f"Star trails ({len(streaks)})")
    ax.legend(handles=[sat_patch, streak_patch], loc="upper right", framealpha=0.7)

    ax.set_title(title or f"{len(sats)} satellite(s)  /  {len(streaks)} star trails")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: list, header: str) -> None:
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                              for v in row) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m source_extraction.sep_extractor",
        description=(
            "Extract satellite point sources and star trails from a rate-tracked "
            "astronomical image.  Stars appear as streaks; the satellite is compact."
        ),
    )
    parser.add_argument("image", help="FITS image file path.")
    parser.add_argument("--threshold",    type=float, default=3.0,
                        help="Detection threshold in sigma (default 3.0).")
    parser.add_argument("--minarea",      type=int,   default=5,
                        help="Minimum detection area in pixels (default 5).")
    parser.add_argument("--elong_thresh", type=float, default=3.0,
                        help="a/b elongation at or above which = star trail (default 3.0).")
    parser.add_argument("--min_streak",   type=float, default=20.0,
                        help="Min semi-major axis (px) to accept a star trail (default 20).")
    parser.add_argument("--sat_fwhm_max", type=float, default=20.0,
                        help="Max FWHM (px) for satellite point-source class (default 20).")
    parser.add_argument("--bkg_box",      type=int,   default=64,
                        help="Background grid cell size in pixels (default 64).")
    parser.add_argument("--gain",         type=float, default=1.0,
                        help="Detector gain in e-/ADU (default 1.0).")
    parser.add_argument("--no_angle_filter", action="store_true",
                        help="Disable the streak angle consistency filter.")
    parser.add_argument("--out_sats",     type=str, default=None,
                        help="CSV path for satellite detections.")
    parser.add_argument("--out_stars",    type=str, default=None,
                        help="CSV path for star trail midpoints (plate-solver format).")
    parser.add_argument("--hdu",          type=int, default=0,
                        help="FITS HDU index to read (default 0).")
    parser.add_argument("--show",         action="store_true",
                        help="Display detection plot.")

    args = parser.parse_args(argv)

    try:
        from astropy.io import fits as pyfits
    except ImportError as exc:
        sys.exit(f"astropy is required to read FITS files: {exc}")

    with pyfits.open(args.image) as hdul:
        image = hdul[args.hdu].data.astype(np.float64)

    if image is None or image.ndim != 2:
        sys.exit(f"HDU {args.hdu} in {args.image!r} does not contain a 2-D image.")

    print(f"Image shape : {image.shape[1]}×{image.shape[0]} px  "
          f"(min={image.min():.1f}  max={image.max():.1f})")

    sats, streaks = extract_sources(
        image,
        threshold=args.threshold,
        minarea=args.minarea,
        elong_thresh=args.elong_thresh,
        min_streak_px=args.min_streak,
        sat_fwhm_max_px=args.sat_fwhm_max,
        bkg_box_size=args.bkg_box,
        gain=args.gain,
        angle_filter_sigma=None if args.no_angle_filter else 2.0,
    )

    print(f"Detected    : {len(sats)} satellite(s)  /  {len(streaks)} star trail(s)")

    if sats:
        for k, sat in enumerate(sats):
            print(f"  Sat {k+1:2d}  — ({sat.x:.1f}, {sat.y:.1f})  "
                  f"flux={sat.flux:.0f}  fwhm={sat.fwhm_px:.1f}px  SNR={sat.snr:.1f}")

    stats = streak_stats(streaks)
    if stats["n"] > 0:
        print(f"  Trails  — angle {stats['angle_deg']:+.1f}° ± {stats['angle_std_deg']:.1f}°  "
              f"length {stats['length_px']:.0f} ± {stats['length_std_px']:.0f} px")

    if args.out_sats:
        _write_csv(
            args.out_sats,
            [(s.x, s.y, s.flux, s.snr, s.fwhm_px, s.a, s.b, s.theta_deg) for s in sats],
            "x_pix,y_pix,flux,snr,fwhm_px,a,b,theta_deg",
        )
        print(f"Satellites → {args.out_sats}")

    if args.out_stars:
        _write_csv(
            args.out_stars,
            [(s.y_mid, s.x_mid, s.flux, s.snr, s.length_px, s.angle_deg) for s in streaks],
            "y_mid,x_mid,flux,snr,length_px,angle_deg",
        )
        print(f"Star trails → {args.out_stars}")

    if args.show:
        _plot(image, sats, streaks, title=Path(args.image).name)


if __name__ == "__main__":
    main()

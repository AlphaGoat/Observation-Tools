#!/usr/bin/env python3
"""
Plot WCS residuals: our plate solver vs ground-truth FITS header WCS.

For every FITS image the script:
  1. Extracts sources with SEP.
       Rate-track mode (default): stars are streaks; their midpoints are fed
       to the plate solver.  Satellites appear as point sources and are ignored
       for astrometry.
       Sidereal mode (--sidereal): stars are point sources; all compact sources
       are fed to the plate solver.
  2. Runs our KD-tree plate solver to obtain a WCS solution.
  3. For each detected star, converts its pixel position to (RA, Dec) using
       (a) our solved WCS, and (b) the header WCS (ground truth).
  4. Computes ΔRA · cos(Dec) and ΔDec in arcseconds.
  5. Writes per-image diagnostic figures and a cross-image summary.

Per-image figure (4-panel)
--------------------------
  Top-left  : image with quiver overlay — residual vectors at each star
  Top-right : ΔRA vs ΔDec scatter with 1-σ ellipse
  Bottom-left : histogram of total residual magnitude
  Bottom-right: residual magnitude vs angular distance from field centre

Summary figure (3-panel)
------------------------
  Left  : per-image solve rate and star count
  Centre: box plot of residual magnitudes per image (arcsec)
  Right : per-image RMS in RA and Dec separately

Usage
-----
  # Plate-solver mode (requires pre-built index files):
  python scripts/plot_wcs_residuals.py data/*.fits \\
      --index-dir indices/ --star-index indices/stars.joblib \\
      --fov-deg 1.5 --output-dir plots/

  # Fit-mode (no index needed — tests WCS model accuracy only):
  python scripts/plot_wcs_residuals.py data/*.fits --fit-mode --output-dir plots/

  # Sidereal images (stars are point sources, not streaks):
  python scripts/plot_wcs_residuals.py data/*.fits --sidereal \\
      --index-dir indices/ --star-index indices/stars.joblib
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # safe for headless runs; --display overrides
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── type alias ────────────────────────────────────────────────────────────────

ImageResult = Dict  # per-image result dict, described below


# ── residual computation ──────────────────────────────────────────────────────

def _angular_sep_arcsec(
    ra1: np.ndarray, dec1: np.ndarray,
    ra2: np.ndarray, dec2: np.ndarray,
) -> np.ndarray:
    """Great-circle angular separation in arcseconds."""
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    cos_c = (np.sin(dec1) * np.sin(dec2)
             + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cos_c = np.clip(cos_c, -1.0, 1.0)
    return np.degrees(np.arccos(cos_c)) * 3600.0


def _pixel_scale_arcsec(header_wcs) -> float:
    """Estimate pixel scale in arcsec/pixel from header WCS (used for display only)."""
    try:
        pix = np.array([[0.0, 0.0], [1.0, 0.0]])
        sky = header_wcs.all_pix2world(pix, 0)
        dra  = (sky[1, 0] - sky[0, 0]) * np.cos(np.radians(sky[0, 1]))
        ddec = sky[1, 1] - sky[0, 1]
        return float(np.hypot(dra, ddec) * 3600.0)
    except Exception:
        return 1.0


def _compute_residuals(
    our_wcs,
    header_wcs,
    star_x: np.ndarray,
    star_y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    For each star at pixel (x, y) compute residuals between our WCS and the header WCS.

    Returns a dict with keys:
        ra_truth, dec_truth : header WCS sky positions (degrees)
        ra_ours,  dec_ours  : our WCS sky positions    (degrees)
        d_ra_as             : ΔRA · cos(Dec) in arcsec (ours − truth)
        d_dec_as            : ΔDec in arcsec           (ours − truth)
        sep_as              : great-circle separation in arcsec
        dist_from_centre_deg: angular distance from field centre (header CRVAL)
    """
    from obs.project_obs_from_pixel_to_radec import pix_to_radec

    # Ground truth from FITS header
    sky_truth = header_wcs.all_pix2world(
        np.column_stack([star_x, star_y]), 0
    )
    ra_truth  = sky_truth[:, 0]
    dec_truth = sky_truth[:, 1]

    # Our plate solver
    ra_ours, dec_ours = pix_to_radec(our_wcs, star_x, star_y)

    d_ra_as  = (ra_ours - ra_truth) * np.cos(np.radians(dec_truth)) * 3600.0
    d_dec_as = (dec_ours - dec_truth) * 3600.0
    sep_as   = _angular_sep_arcsec(ra_truth, dec_truth, ra_ours, dec_ours)

    # Distance from field centre (header CRVAL)
    try:
        ra0, dec0 = header_wcs.wcs.crval[0], header_wcs.wcs.crval[1]
    except Exception:
        ra0, dec0 = float(np.mean(ra_truth)), float(np.mean(dec_truth))

    dist_from_centre = _angular_sep_arcsec(
        np.full(len(ra_truth), ra0), np.full(len(dec_truth), dec0),
        ra_truth, dec_truth,
    ) / 3600.0   # degrees

    return dict(
        ra_truth=ra_truth, dec_truth=dec_truth,
        ra_ours=ra_ours,   dec_ours=dec_ours,
        d_ra_as=d_ra_as,   d_dec_as=d_dec_as,
        sep_as=sep_as,
        dist_from_centre_deg=dist_from_centre,
    )


# ── source extraction ─────────────────────────────────────────────────────────

def _detect_stars(
    image: np.ndarray,
    sidereal: bool,
    threshold: float,
    elong_thresh: float,
    min_streak_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect stars in an image and return (x_pix, y_pix, flux) arrays.

    Rate-track mode: streak midpoints — x_mid, y_mid from StarStreak objects.
    Sidereal mode:   compact point sources — x, y from SatelliteDetection, but
                     we invert the classification (everything compact = star).
    """
    from source_extraction.sep_extractor import extract_sources, star_array

    if sidereal:
        # In sidereal mode stars are point sources — we still use extract_sources
        # but treat *all* compact detections as stars by setting elong_thresh very
        # high (no elongation requirement) and grabbing point sources.
        import sep
        import numpy as _np

        img = _np.asarray(image, dtype=_np.float64)
        if not img.flags["C_CONTIGUOUS"]:
            img = _np.ascontiguousarray(img)
        if img.dtype.byteorder not in ("=", "<", "|"):
            img = img.byteswap().newbyteorder()

        bkg = sep.Background(img, bw=64, bh=64, fw=3, fh=3)
        img_sub = img - bkg.back()
        bkg_rms = bkg.rms()

        sources = sep.extract(img_sub, thresh=threshold, minarea=5, err=bkg_rms)
        x   = sources["x"].astype(float)
        y   = sources["y"].astype(float)
        flx = sources["flux"].astype(float)
        # Exclude very elongated detections (cosmic rays)
        a, b = sources["a"].astype(float), sources["b"].astype(float)
        good = (a / np.maximum(b, 1e-3)) < elong_thresh
        return x[good], y[good], flx[good]

    else:
        # Rate-track: streak midpoints → plate solver
        _, streaks = extract_sources(
            image,
            threshold=threshold,
            elong_thresh=elong_thresh,
            min_streak_px=min_streak_px,
        )
        arr = star_array(streaks)   # [[y_mid, x_mid, flux], …]
        if len(arr) == 0:
            return np.array([]), np.array([]), np.array([])
        return arr[:, 1], arr[:, 0], arr[:, 2]   # x, y, flux


# ── plate solver ──────────────────────────────────────────────────────────────

def _run_plate_solver(
    star_x: np.ndarray,
    star_y: np.ndarray,
    star_flux: np.ndarray,
    image_h: int,
    image_w: int,
    index_dir: Optional[str],
    star_index: Optional[str],
    fov_deg: Optional[float],
):
    """
    Run our KD-tree plate solver.  Returns a kd_tree.WCS or None.

    In fit-mode (index_dir is None) the function returns None and the caller
    falls back to fitting directly from the header WCS.
    """
    if index_dir is None or star_index is None:
        return None

    import glob as _glob
    from astrometry.kd_tree import CodeSpaceTree, StarPositionTree, PlateSolver

    code_files = sorted(_glob.glob(str(Path(index_dir) / "codes_tier*.joblib")))
    if not code_files:
        raise FileNotFoundError(
            f"No codes_tier*.joblib files found in {index_dir!r}. "
            "Run build_index_tiered() to generate them first."
        )

    code_trees = [CodeSpaceTree.load(p) for p in code_files]
    star_tree  = StarPositionTree.load(star_index)
    solver     = PlateSolver(code_trees, star_tree)

    detected = np.column_stack([star_y, star_x, star_flux])   # [y, x, flux]
    return solver.solve(detected, image_h, image_w, fov_deg=fov_deg)


def _fit_wcs_from_header(
    header_wcs,
    star_x: np.ndarray,
    star_y: np.ndarray,
):
    """
    Fit-mode fallback: derive our WCS by fitting _fit_wcs to (pixel, sky) pairs
    from the header WCS.  Tests the gnomonic model accuracy in isolation.
    """
    from astrometry.kd_tree import _fit_wcs

    sky = header_wcs.all_pix2world(np.column_stack([star_x, star_y]), 0)
    return _fit_wcs(sky[:, 0], sky[:, 1], star_x, star_y)


# ── per-image figure ──────────────────────────────────────────────────────────

def _cmap_resid(sep_as: np.ndarray):
    """Colour map: blue (small error) → red (large error)."""
    norm = Normalize(vmin=0, vmax=np.percentile(sep_as, 95) or 1.0)
    return plt.cm.RdYlBu_r, norm


def _plot_image(
    image: np.ndarray,
    res: Dict,
    ax: plt.Axes,
    pixel_scale_as: float,
    title: str,
) -> None:
    """Quiver overlay: residual arrows at each star on the image background."""
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    ax.imshow(image, origin="lower", cmap="gray",
              vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")

    # Convert arcsec residuals to pixel offsets for display (scale arrow length)
    dx_px = res["d_ra_as"]  / pixel_scale_as
    dy_px = res["d_dec_as"] / pixel_scale_as

    # Colour arrows by magnitude
    cmap, norm = _cmap_resid(res["sep_as"])
    colours = cmap(norm(res["sep_as"]))

    # star_x, star_y were stored in res
    x, y = res["star_x"], res["star_y"]

    q = ax.quiver(
        x, y, dx_px, dy_px,
        color=colours,
        angles="xy", scale_units="xy", scale=1.0,   # 1 arrow unit = 1 pixel
        width=0.002,
        headwidth=4, headlength=4,
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Residual (arcsec)", fraction=0.03, pad=0.01)

    # Arrow scale reference: show a 1-arcsec arrow in the corner
    ref_px = 1.0 / pixel_scale_as
    ax.quiver(
        [x.min() + 30], [y.min() + 30],
        [ref_px], [0],
        color="lime", angles="xy", scale_units="xy", scale=1.0,
        width=0.003, headwidth=4, headlength=4,
    )
    ax.text(x.min() + 35, y.min() + 40, '1"', color="lime", fontsize=7, va="bottom")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x  (pixels)", fontsize=8)
    ax.set_ylabel("y  (pixels)", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_scatter(res: Dict, ax: plt.Axes) -> None:
    """ΔRA vs ΔDec scatter with 1-σ covariance ellipse."""
    d_ra  = res["d_ra_as"]
    d_dec = res["d_dec_as"]
    cmap, norm = _cmap_resid(res["sep_as"])

    ax.scatter(d_ra, d_dec, c=res["sep_as"], cmap=cmap, norm=norm,
               s=18, linewidths=0, alpha=0.8)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.axvline(0, color="0.6", lw=0.8, ls="--")

    # 1-σ ellipse
    if len(d_ra) >= 3:
        cov  = np.cov(d_ra, d_dec)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h  = 2.0 * np.sqrt(np.abs(vals))
        ell = Ellipse(
            (d_ra.mean(), d_dec.mean()),
            width=w, height=h, angle=angle,
            edgecolor="k", facecolor="none", lw=1.2, ls="--",
        )
        ax.add_patch(ell)

    ax.set_xlabel(r"$\Delta$RA·cos(Dec)  (arcsec)", fontsize=8)
    ax.set_ylabel(r"$\Delta$Dec  (arcsec)", fontsize=8)
    ax.set_title("Residual scatter", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    stats_txt = (
        f"n = {len(d_ra)}\n"
        f"rms RA   = {np.sqrt(np.mean(d_ra**2)):.3f}\"\n"
        f"rms Dec  = {np.sqrt(np.mean(d_dec**2)):.3f}\"\n"
        f"median |Δ| = {np.median(res['sep_as']):.3f}\""
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))


def _plot_histogram(res: Dict, ax: plt.Axes, pixel_scale_as: float) -> None:
    """Histogram of total residual magnitude in arcseconds."""
    sep_as = res["sep_as"]
    bins = np.linspace(0, max(np.percentile(sep_as, 99) * 1.2, 0.5), 30)
    ax.hist(sep_as, bins=bins, color="#3b82f6", edgecolor="white",
            linewidth=0.5, alpha=0.9)

    med = np.median(sep_as)
    rms = float(np.sqrt(np.mean(sep_as ** 2)))
    ax.axvline(med, color="k",     lw=1.2, ls="--", label=f"median = {med:.3f}\"")
    ax.axvline(rms, color="#ef4444", lw=1.2, ls=":",  label=f"rms    = {rms:.3f}\"")

    # Second x-axis in pixels
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(lambda a: a / pixel_scale_as, lambda p: p * pixel_scale_as),
    )
    ax2.set_xlabel("pixels", fontsize=7)
    ax2.tick_params(labelsize=7)

    ax.set_xlabel("Total residual  (arcsec)", fontsize=8)
    ax.set_ylabel("Stars", fontsize=8)
    ax.set_title("Residual magnitude distribution", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.tick_params(labelsize=7)


def _plot_vs_radius(res: Dict, ax: plt.Axes) -> None:
    """Residual magnitude vs angular distance from field centre."""
    dist = res["dist_from_centre_deg"] * 60.0    # → arcmin
    sep  = res["sep_as"]
    cmap, norm = _cmap_resid(sep)

    ax.scatter(dist, sep, c=sep, cmap=cmap, norm=norm, s=18, alpha=0.8)

    # Running median
    if len(dist) >= 5:
        order  = np.argsort(dist)
        d_sort = dist[order]
        s_sort = sep[order]
        n = max(5, len(dist) // 5)
        d_med, s_med = [], []
        for k in range(0, len(dist) - n + 1, max(1, n // 2)):
            d_med.append(np.median(d_sort[k:k + n]))
            s_med.append(np.median(s_sort[k:k + n]))
        ax.plot(d_med, s_med, "k-", lw=1.5, label="running median")
        ax.legend(fontsize=7)

    ax.set_xlabel("Distance from field centre  (arcmin)", fontsize=8)
    ax.set_ylabel("Total residual  (arcsec)", fontsize=8)
    ax.set_title("Residual vs radius", fontsize=9)
    ax.tick_params(labelsize=7)


def make_per_image_figure(
    image: np.ndarray,
    res: Dict,
    pixel_scale_as: float,
    stem: str,
    solve_mode: str,
) -> plt.Figure:
    """4-panel diagnostic figure for one image."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{stem}  —  {solve_mode}  —  n={len(res['sep_as'])} stars\n"
        f"median {np.median(res['sep_as']):.3f}\"  "
        f"rms {np.sqrt(np.mean(res['sep_as']**2)):.3f}\"  "
        f"max {res['sep_as'].max():.3f}\"",
        fontsize=10,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.90, bottom=0.07)
    ax_img  = fig.add_subplot(gs[0, 0])
    ax_scat = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_rad  = fig.add_subplot(gs[1, 1])

    _plot_image(image, res, ax_img, pixel_scale_as, stem)
    _plot_scatter(res, ax_scat)
    _plot_histogram(res, ax_hist, pixel_scale_as)
    _plot_vs_radius(res, ax_rad)

    return fig


# ── summary figure ────────────────────────────────────────────────────────────

def make_summary_figure(results: List[ImageResult]) -> plt.Figure:
    """Cross-image summary: star counts, residual box plots, per-axis RMS."""
    solved = [r for r in results if r["solved"]]
    failed = [r for r in results if not r["solved"]]

    stems     = [r["stem"]    for r in solved]
    n_stars   = [r["n_stars"] for r in solved]
    all_sep   = [r["res"]["sep_as"]    for r in solved]
    all_d_ra  = [r["res"]["d_ra_as"]  for r in solved]
    all_d_dec = [r["res"]["d_dec_as"] for r in solved]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"WCS Residual Summary  —  {len(solved)}/{len(results)} images solved",
        fontsize=11,
    )

    # Panel 1: star count per image
    ax = axes[0]
    colours = ["#3b82f6"] * len(stems)
    bars = ax.bar(range(len(stems)), n_stars, color=colours, edgecolor="white")
    ax.set_xticks(range(len(stems)))
    ax.set_xticklabels(stems, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Stars used for WCS", fontsize=8)
    ax.set_title("Stars per image", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    if failed:
        ax.set_title(
            f"Stars per image  ({len(failed)} failed: "
            + ", ".join(r['stem'] for r in failed[:3])
            + ("…" if len(failed) > 3 else "") + ")",
            fontsize=8,
        )

    # Panel 2: box plot of residual magnitudes
    ax = axes[1]
    if all_sep:
        bp = ax.boxplot(
            all_sep,
            labels=stems,
            patch_artist=True,
            medianprops=dict(color="k", lw=2),
            boxprops=dict(facecolor="#93c5fd", linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        ax.set_xticks(range(1, len(stems) + 1))
        ax.set_xticklabels(stems, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Total residual  (arcsec)", fontsize=8)
        ax.set_title("Residual distribution per image", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    # Panel 3: per-axis RMS
    ax = axes[2]
    x_pos = np.arange(len(stems))
    rms_ra  = [float(np.sqrt(np.mean(d ** 2))) for d in all_d_ra]
    rms_dec = [float(np.sqrt(np.mean(d ** 2))) for d in all_d_dec]
    w = 0.35
    ax.bar(x_pos - w / 2, rms_ra,  w, label="ΔRA·cos(Dec)", color="#3b82f6",
           edgecolor="white")
    ax.bar(x_pos + w / 2, rms_dec, w, label="ΔDec",         color="#f97316",
           edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stems, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMS  (arcsec)", fontsize=8)
    ax.set_title("Per-axis RMS", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    return fig


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_image(
    fits_path: str,
    *,
    index_dir: Optional[str],
    star_index: Optional[str],
    fov_deg: Optional[float],
    hdu: int,
    threshold: float,
    elong_thresh: float,
    min_streak_px: float,
    sidereal: bool,
    fit_mode: bool,
) -> ImageResult:
    """
    Run the full pipeline for one FITS file.

    Returns a dict with keys:
        stem      : filename without extension
        path      : full FITS path
        solved    : bool — did we get a WCS solution?
        n_stars   : number of stars used
        res       : residual dict from _compute_residuals (if solved)
        error_msg : description of failure (if not solved)
    """
    from astropy.io import fits as pyfits
    from astropy.wcs import WCS as AstropyWCS, FITSFixedWarning

    stem = Path(fits_path).stem
    print(f"\n── {stem} ──")

    # ── Load image and header WCS ─────────────────────────────────────────────
    with pyfits.open(fits_path) as hdul:
        hdr  = hdul[hdu].header
        data = hdul[hdu].data

    if data is None or data.ndim != 2:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=0, res=None,
                    error_msg=f"HDU {hdu} has no 2-D image data")

    image = np.asarray(data, dtype=np.float64)
    h, w  = image.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        try:
            header_wcs = AstropyWCS(hdr, naxis=2)
        except Exception as exc:
            return dict(stem=stem, path=fits_path, solved=False,
                        n_stars=0, res=None,
                        error_msg=f"Could not parse header WCS: {exc}")

    if not header_wcs.has_celestial:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=0, res=None,
                    error_msg="Header contains no celestial WCS axes")

    print(f"  Image     : {w}×{h} px")

    # ── Extract stars ─────────────────────────────────────────────────────────
    try:
        star_x, star_y, star_flux = _detect_stars(
            image, sidereal, threshold, elong_thresh, min_streak_px,
        )
    except Exception as exc:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=0, res=None,
                    error_msg=f"Source extraction failed: {exc}")

    if len(star_x) < 4:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=len(star_x), res=None,
                    error_msg=f"Too few stars detected ({len(star_x)}); need ≥ 4")

    print(f"  Stars     : {len(star_x)}")

    # ── Solve / fit ───────────────────────────────────────────────────────────
    our_wcs = None
    solve_mode_label = ""

    if fit_mode:
        # Bypass plate solver — fit our WCS model directly from the header WCS
        try:
            our_wcs = _fit_wcs_from_header(header_wcs, star_x, star_y)
            solve_mode_label = "fit-mode (no index)"
        except Exception as exc:
            return dict(stem=stem, path=fits_path, solved=False,
                        n_stars=len(star_x), res=None,
                        error_msg=f"WCS fit failed: {exc}")
    else:
        # Full plate solver
        try:
            our_wcs = _run_plate_solver(
                star_x, star_y, star_flux, h, w,
                index_dir, star_index, fov_deg,
            )
            solve_mode_label = "plate solver"
        except FileNotFoundError as exc:
            return dict(stem=stem, path=fits_path, solved=False,
                        n_stars=len(star_x), res=None,
                        error_msg=str(exc))
        except Exception as exc:
            return dict(stem=stem, path=fits_path, solved=False,
                        n_stars=len(star_x), res=None,
                        error_msg=f"Plate solver error: {exc}")

    if our_wcs is None:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=len(star_x), res=None,
                    error_msg="Plate solver did not converge")

    print(f"  WCS       : solved  ({solve_mode_label})")

    # ── Compute residuals ─────────────────────────────────────────────────────
    try:
        res = _compute_residuals(our_wcs, header_wcs, star_x, star_y)
    except Exception as exc:
        return dict(stem=stem, path=fits_path, solved=False,
                    n_stars=len(star_x), res=None,
                    error_msg=f"Residual computation failed: {exc}")

    res["star_x"] = star_x
    res["star_y"] = star_y

    print(
        f"  Residuals : median {np.median(res['sep_as']):.3f}\"  "
        f"rms {np.sqrt(np.mean(res['sep_as']**2)):.3f}\"  "
        f"max {res['sep_as'].max():.3f}\""
    )

    return dict(
        stem=stem, path=fits_path, solved=True,
        n_stars=len(star_x), res=res,
        image=image,
        header_wcs=header_wcs,
        pixel_scale_as=_pixel_scale_arcsec(header_wcs),
        solve_mode=solve_mode_label,
        error_msg=None,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="plot_wcs_residuals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "fits_files", nargs="+",
        help="FITS file paths or glob patterns (e.g. data/*.fits).",
    )

    # ── Index files ───────────────────────────────────────────────────────────
    solver_grp = parser.add_argument_group("plate solver (ignored in --fit-mode)")
    solver_grp.add_argument(
        "--index-dir", metavar="DIR",
        help="Directory containing codes_tier*.joblib index files.",
    )
    solver_grp.add_argument(
        "--star-index", metavar="FILE",
        help="Path to the StarPositionTree .joblib file.",
    )
    solver_grp.add_argument(
        "--fov-deg", type=float, default=None, metavar="N",
        help="Expected image FOV diagonal in degrees (speeds up tier selection).",
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode_grp = parser.add_argument_group("mode")
    mode_grp.add_argument(
        "--fit-mode", action="store_true",
        help=(
            "Skip the plate solver.  Instead, fit our WCS model directly from "
            "the header WCS at the detected star positions.  Tests the gnomonic "
            "model accuracy without requiring index files."
        ),
    )
    mode_grp.add_argument(
        "--sidereal", action="store_true",
        help=(
            "Treat stars as point sources rather than streaks.  Use for standard "
            "sidereal-tracking images.  Default is rate-track (stars are streaks)."
        ),
    )

    # ── Extraction parameters ─────────────────────────────────────────────────
    ext_grp = parser.add_argument_group("source extraction")
    ext_grp.add_argument("--hdu",          type=int,   default=0,   metavar="N")
    ext_grp.add_argument("--threshold",    type=float, default=3.0, metavar="σ",
                         help="SEP detection threshold in σ (default 3.0).")
    ext_grp.add_argument("--elong-thresh", type=float, default=3.0, metavar="N",
                         help="a/b ratio threshold for streak classification (default 3.0).")
    ext_grp.add_argument("--min-streak-px", type=float, default=20.0, metavar="N",
                         help="Min semi-major axis in pixels to accept a star trail (default 20).")

    # ── Output ────────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--output-dir", metavar="DIR", default="plots/",
        help="Directory to write figures into (default: plots/).",
    )
    out_grp.add_argument(
        "--display", action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    out_grp.add_argument(
        "--dpi", type=int, default=150, metavar="N",
        help="Figure DPI for saved images (default 150).",
    )

    args = parser.parse_args(argv)

    if args.display:
        matplotlib.use("TkAgg")

    if not args.fit_mode and (args.index_dir is None or args.star_index is None):
        parser.error(
            "Plate-solver mode requires --index-dir and --star-index.\n"
            "Run with --fit-mode to test WCS model accuracy without an index."
        )

    # ── Expand file globs ─────────────────────────────────────────────────────
    fits_paths: List[str] = []
    for pat in args.fits_files:
        expanded = glob.glob(pat)
        if not expanded:
            fits_paths.append(pat)   # let the open() fail with a clear message
        else:
            fits_paths.extend(sorted(expanded))
    fits_paths = sorted(set(fits_paths))

    if not fits_paths:
        sys.exit("No FITS files found.")

    print(f"Processing {len(fits_paths)} image(s)  →  output dir: {args.output_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each image ────────────────────────────────────────────────────
    all_results: List[ImageResult] = []

    for fits_path in fits_paths:
        result = process_image(
            fits_path,
            index_dir=args.index_dir,
            star_index=args.star_index,
            fov_deg=args.fov_deg,
            hdu=args.hdu,
            threshold=args.threshold,
            elong_thresh=args.elong_thresh,
            min_streak_px=args.min_streak_px,
            sidereal=args.sidereal,
            fit_mode=args.fit_mode,
        )
        all_results.append(result)

        if not result["solved"]:
            print(f"  SKIP      : {result['error_msg']}")
            continue

        # Per-image figure
        fig = make_per_image_figure(
            result["image"],
            result["res"],
            result["pixel_scale_as"],
            stem=result["stem"],
            solve_mode=result["solve_mode"],
        )
        fig_path = out_dir / f"{result['stem']}_residuals.png"
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved     : {fig_path}")
        if args.display:
            plt.show()
        plt.close(fig)

    # ── Summary figure ────────────────────────────────────────────────────────
    solved = [r for r in all_results if r["solved"]]
    print(f"\n{'─'*60}")
    print(f"Solved {len(solved)}/{len(all_results)} images.")

    if solved:
        all_sep = np.concatenate([r["res"]["sep_as"] for r in solved])
        print(
            f"Overall  —  median {np.median(all_sep):.3f}\"  "
            f"rms {np.sqrt(np.mean(all_sep**2)):.3f}\"  "
            f"max {all_sep.max():.3f}\""
        )

        summary_fig = make_summary_figure(all_results)
        summary_path = out_dir / "summary.png"
        summary_fig.savefig(summary_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Summary  →  {summary_path}")
        if args.display:
            plt.show()
        plt.close(summary_fig)

    for r in all_results:
        if not r["solved"]:
            print(f"  FAILED: {r['stem']}  —  {r['error_msg']}")


if __name__ == "__main__":
    main()

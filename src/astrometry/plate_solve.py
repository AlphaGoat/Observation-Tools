"""
plate_solve.py — End-to-end astrometric plate solver.

Pipeline
--------
  1. Load pre-built CodeSpaceTree(s) and StarPositionTree from disk.
  2. Sort detected stars by brightness; select the N brightest.
  3. Enumerate 4-star combinations (quads) from the detected stars.
  4. Compute a pixel-space geometric hash code for each quad.
  5. Range-search the relevant scale-tier CodeSpaceTree(s) for matching catalogue quads.
  6. For each catalogue match, resolve star IDs → (RA, Dec) and fit a candidate WCS.
  7. Project nearby catalogue stars under the candidate WCS onto the image plane.
  8. Run bayesian_decision_maker to accept or reject the candidate.
  9. On acceptance, return the WCS and a verification report.

Usage
-----
  As a library:
      from astrometry.plate_solve import solve_field, load_index
      code_trees, star_tree = load_index("indices/")
      result = solve_field(
          detections,          # (K, 3) array: [y_pix, x_pix, flux]
          code_trees, star_tree,
          image_height=2048, image_width=2048,
          fov_deg=2.0,
          verbose=True,
      )
      if result.solved:
          print(result.wcs)

  As a script:
      python -m astrometry.plate_solve \\
          --detections stars.npy \\
          --index_dir  indices/ \\
          --star_index indices/stars.joblib \\
          --image_height 2048 --image_width 2048 \\
          --fov_deg 2.0

  Detection file formats:
      .npy  — (K, 3) float array [y_pix, x_pix, flux]
      .csv  — columns: y_pix, x_pix, flux  (header optional)

Authors: Peter Thomas
"""

from __future__ import annotations

import glob
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from astrometry.kd_tree import (
    CodeSpaceTree,
    StarPositionTree,
    WCS,
    PlateSolver,
    _pix_quad_info,
    _fit_wcs,
    _radec_to_xyz,
)
from astrometry.bayesian_decision_maker import bayesian_decision_maker


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    """
    Complete output from a plate-solve attempt.

    Attributes
    ----------
    solved          : True if a WCS was accepted.
    wcs             : Accepted WCS (None if not solved).
    n_detected      : Total detected stars fed to the solver.
    n_used          : Stars actually used (brightest N).
    quads_tried     : Number of pixel-space quads evaluated.
    quads_matched   : Total catalogue-quad matches found across all quads.
    elapsed_s       : Wall-clock time for the solve.
    match_ra        : RA of the 4 anchor stars in the accepted quad (deg).
    match_dec       : Dec of the 4 anchor stars in the accepted quad (deg).
    match_pix       : Pixel [x, y] of the 4 anchor stars in the accepted quad.
    residuals_arcsec: Residuals (arcsec) of all detected stars against the WCS,
                      populated by verify() after a successful solve.
    log             : Step-by-step log lines accumulated during the solve.
    """
    solved: bool = False
    wcs: Optional[WCS] = None
    n_detected: int = 0
    n_used: int = 0
    quads_tried: int = 0
    quads_matched: int = 0
    elapsed_s: float = 0.0
    match_ra: Optional[np.ndarray] = None
    match_dec: Optional[np.ndarray] = None
    match_pix: Optional[np.ndarray] = None
    residuals_arcsec: Optional[np.ndarray] = None
    log: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'SOLVED' if self.solved else 'FAILED'} in {self.elapsed_s:.2f} s",
            f"  Detections   : {self.n_detected} total, {self.n_used} used",
            f"  Quads tried  : {self.quads_tried}",
            f"  Quad matches : {self.quads_matched}",
        ]
        if self.solved and self.wcs is not None:
            A = self.wcs.A
            lines += [
                f"  WCS matrix   :",
                f"    x_pix = {A[0,0]:+.6f}·ra + {A[0,1]:+.6f}·dec + {A[0,2]:+.4f}",
                f"    y_pix = {A[1,0]:+.6f}·ra + {A[1,1]:+.6f}·dec + {A[1,2]:+.4f}",
            ]
        if self.residuals_arcsec is not None and len(self.residuals_arcsec):
            lines += [
                f"  Residuals    : median={np.median(self.residuals_arcsec):.2f}\"  "
                f"rms={np.sqrt(np.mean(self.residuals_arcsec**2)):.2f}\"  "
                f"max={np.max(self.residuals_arcsec):.2f}\"",
            ]
        return "\n".join(lines)


# ── Index loader ──────────────────────────────────────────────────────────────

def load_index(
    index_dir: str,
    star_index_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[CodeSpaceTree], StarPositionTree]:
    """
    Load all CodeSpaceTrees from *index_dir* and the StarPositionTree.

    Parameters
    ----------
    index_dir        : Directory containing ``codes_tier*.joblib`` files.
    star_index_path  : Path to ``stars.joblib``.  If None, searched inside
                       *index_dir* for ``stars.joblib``.
    verbose          : Print loading summary.

    Returns
    -------
    (list of CodeSpaceTree, StarPositionTree)
    """
    code_paths = sorted(glob.glob(str(Path(index_dir) / "codes_tier*.joblib")))
    if not code_paths:
        raise FileNotFoundError(
            f"No codes_tier*.joblib files found in '{index_dir}'"
        )

    if star_index_path is None:
        star_index_path = str(Path(index_dir) / "stars.joblib")
    if not Path(star_index_path).exists():
        raise FileNotFoundError(f"Star index not found: '{star_index_path}'")

    t0 = time.perf_counter()
    code_trees = [CodeSpaceTree.load(p) for p in code_paths]
    star_tree  = StarPositionTree.load(star_index_path)
    elapsed    = time.perf_counter() - t0

    if verbose:
        print(f"[load_index] Loaded {len(code_trees)} code tier(s) + star index "
              f"in {elapsed*1e3:.1f} ms")
        for ct, p in zip(code_trees, code_paths):
            lo = ct.scale_lower_deg
            hi = ct.scale_upper_deg
            tier_str = (f"{lo:.3f}°–{hi:.3f}°"
                        if lo is not None and hi is not None else "no scale info")
            print(f"  {Path(p).name}: {len(ct.codes):,} quads  [{tier_str}]")
        print(f"  stars.joblib : {len(star_tree.ra):,} catalogue stars")

    return code_trees, star_tree


# ── Detection loader ──────────────────────────────────────────────────────────

def load_detections(path: str) -> np.ndarray:
    """
    Load a detection array from a .npy or .csv file.

    Expected shape: (K, 3) with columns [y_pix, x_pix, flux].

    For CSV files the first row is skipped if it contains non-numeric text.
    """
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in (".csv", ".txt"):
        try:
            arr = np.loadtxt(p, delimiter=",")
        except ValueError:
            arr = np.loadtxt(p, delimiter=",", skiprows=1)
    else:
        raise ValueError(f"Unsupported detection file format: '{p.suffix}'")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(
            f"Detection array must have ≥ 3 columns [y_pix, x_pix, flux]; "
            f"got shape {arr.shape}"
        )
    return arr[:, :3].astype(np.float64)


# ── Core solve function ───────────────────────────────────────────────────────

def solve_field(
    detections: np.ndarray,
    code_trees: List[CodeSpaceTree],
    star_tree: StarPositionTree,
    image_height: int,
    image_width: int,
    fov_deg: Optional[float] = None,
    sort_by: str = "flux",
    max_stars: int = 20,
    max_quads: int = 300,
    code_radius: float = 0.02,
    model: str = "asymmetric",
    variance: float = 9.0,
    distractors: float = 0.25,
    field_span_multiplier: float = 4.0,
    run_verify: bool = True,
    verbose: bool = True,
) -> SolveResult:
    """
    Plate-solve an image given pixel-space star detections and a pre-built index.

    Parameters
    ----------
    detections       : (K, 3) array — [y_pix, x_pix, flux].
    code_trees       : List of CodeSpaceTree (one per scale tier).
    star_tree        : StarPositionTree (full catalogue).
    image_height,
    image_width      : Image dimensions in pixels.
    fov_deg          : Approximate field-of-view diagonal (degrees).
                       When provided, only relevant scale tiers are searched.
    sort_by          : "flux"  — brightest first (highest flux value).
                       "magnitude" — sort ascending (lowest magnitude first).
    max_stars        : Number of brightest stars passed to the quad enumerator.
    max_quads        : Hard cap on pixel-space quads evaluated.
    code_radius      : L2 radius in 4D code space for catalogue match (default 0.02).
    model            : Bayesian model — "simple_independence" or "asymmetric".
    variance         : Bayesian positional variance (pixels²).
    distractors      : Bayesian distractor fraction.
    field_span_multiplier : Multiplier on the catalogue quad span when fetching
                       nearby reference stars for Bayesian verification.
    run_verify       : If True, compute per-star residuals after a successful solve.
    verbose          : Print step-by-step progress.

    Returns
    -------
    SolveResult
    """
    result = SolveResult()
    log    = result.log
    t0     = time.perf_counter()

    def _log(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    # ── Step 1: prepare detections ────────────────────────────────────────────
    stars = np.asarray(detections, dtype=float)
    result.n_detected = len(stars)

    if len(stars) < 4:
        _log(f"[solve] Only {len(stars)} detections — need ≥ 4.  Aborting.")
        result.elapsed_s = time.perf_counter() - t0
        return result

    if sort_by == "magnitude":
        order = np.argsort(stars[:, 2])          # ascending: dim → bright
    else:
        order = np.argsort(stars[:, 2])[::-1]    # descending: bright → dim

    stars = stars[order[:max_stars]]
    n = len(stars)
    result.n_used = n
    _log(f"[solve] {result.n_detected} detections → using brightest {n}")

    # ── Step 2: select relevant scale tiers ───────────────────────────────────
    solver = PlateSolver(
        code_trees, star_tree,
        code_radius=code_radius,
        max_quads=max_quads,
        model=model,
        variance=variance,
        distractors=distractors,
        field_span_multiplier=field_span_multiplier,
    )
    active_trees = solver._select_trees(fov_deg)

    if fov_deg is not None:
        tier_info = ", ".join(
            f"{ct.scale_lower_deg:.3f}°–{ct.scale_upper_deg:.3f}°"
            for ct in active_trees
            if ct.scale_lower_deg is not None
        )
        _log(f"[solve] FOV={fov_deg}° → {len(active_trees)} active tier(s): {tier_info}")
    else:
        _log(f"[solve] No FOV hint — searching all {len(active_trees)} tier(s)")

    # ── Step 3: enumerate quads and search ────────────────────────────────────
    _log(f"[solve] Enumerating quads (max {max_quads})...")
    quads_tried   = 0
    quads_matched = 0

    for quad_idx in itertools.combinations(range(n), 4):
        if quads_tried >= max_quads:
            break

        det   = stars[list(quad_idx)]
        pix_y = det[:, 0]
        pix_x = det[:, 1]

        # ── Step 4: compute pixel-space hash code and ABCD ordering ───────────
        # One consistent pass prevents near-boundary quads from getting
        # inconsistent A/B orderings due to independent floating-point checks.
        quad_info = _pix_quad_info(pix_x, pix_y)
        if quad_info is None:
            continue   # C or D outside inscribed circle → invalid quad geometry

        query_code, det_abcd = quad_info
        quads_tried += 1
        q = np.array(query_code, dtype=np.float32)

        # ── Step 5: range-search catalogue code trees ─────────────────────────
        for code_tree in active_trees:
            matches = code_tree.range_search(q, radius=code_radius)
            if not matches:
                continue

            quads_matched += len(matches)
            _log(
                f"  quad {quads_tried:4d} [{quad_idx}] "
                f"code=({q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f}) "
                f"→ {len(matches)} match(es) in tier "
                f"{code_tree.scale_lower_deg:.3f}°–{code_tree.scale_upper_deg:.3f}°"
            )

            for cat_sids in matches:
                # ── Step 6: resolve source IDs → (RA, Dec) ───────────────────
                result_rd = solver._resolve_sids(cat_sids)
                if result_rd is None:
                    continue
                cat_ra, cat_dec = result_rd

                # ── Fit candidate WCS from 4 correspondence pairs ─────────────
                # Try both A/B orientations: quads near the xC+xD=1 boundary can
                # have inconsistent canonical A/B assignment between sky (exact
                # floating-point) and pixel (noisy), flipping the WCS.  Picking
                # the orientation with the smaller 4-point residual fixes this at
                # negligible extra cost.
                det_pix_x  = pix_x[det_abcd]
                det_pix_y  = pix_y[det_abcd]
                ab_swap    = np.array([1, 0, 2, 3])

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

                # ── Step 7: fetch reference stars around the candidate field ───
                ra_c  = float(np.mean(cat_ra))
                dec_c = float(np.mean(cat_dec))
                span  = float(
                    max(np.ptp(cat_ra), np.ptp(cat_dec), 0.05)
                ) * field_span_multiplier

                ref_ra, ref_dec, ref_mV = star_tree.stars_in_box(
                    ra_c - span, ra_c + span,
                    dec_c - span, dec_c + span,
                )
                if len(ref_ra) == 0:
                    continue

                ref_pix   = wcs.radec_to_pix(ref_ra, ref_dec)
                ref_stars = np.column_stack([ref_pix[:, 1], ref_pix[:, 0], ref_mV])

                # ── Step 8: Bayesian decision ─────────────────────────────────
                accepted = bayesian_decision_maker(
                    reference_stars=ref_stars,
                    test_stars=stars,
                    image_height=image_height,
                    image_width=image_width,
                    model=model,
                    variance=variance,
                    distractors=distractors,
                    sort_by="snr" if sort_by == "flux" else sort_by,
                )

                if accepted:
                    result.quads_tried   = quads_tried
                    result.quads_matched = quads_matched
                    result.solved        = True
                    result.wcs           = wcs
                    result.match_ra      = cat_ra
                    result.match_dec     = cat_dec
                    result.match_pix     = np.column_stack([
                        pix_x[det_abcd], pix_y[det_abcd]
                    ])
                    result.elapsed_s     = time.perf_counter() - t0
                    _log(
                        f"[solve] ACCEPTED at quad {quads_tried}  "
                        f"anchor stars: RA={cat_ra.tolist()}  Dec={cat_dec.tolist()}"
                    )

                    # ── Step 9: verification residuals ────────────────────────
                    if run_verify:
                        result.residuals_arcsec = _compute_residuals(
                            wcs, star_tree,
                            stars[:, 1], stars[:, 0],  # x_pix, y_pix
                            search_radius_pix=10.0,
                        )

                    return result

    result.quads_tried   = quads_tried
    result.quads_matched = quads_matched
    result.elapsed_s     = time.perf_counter() - t0
    _log(f"[solve] No solution found after {quads_tried} quads ({quads_matched} catalogue matches).")
    return result


# ── Residual computation ──────────────────────────────────────────────────────

def _compute_residuals(
    wcs: WCS,
    star_tree: StarPositionTree,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    search_radius_pix: float = 10.0,
) -> np.ndarray:
    """
    For each detected star, find the nearest catalogue star (via WCS inverse),
    and return angular residuals in arcseconds.

    The WCS is used forward-only here: we project each detected pixel position
    to an approximate (RA, Dec) via a 2D grid search around the WCS-predicted
    sky position, then look up the true catalogue star.
    """
    # Approximate pixel scale: arcsec/pixel from the WCS matrix
    A = wcs.A
    # Plate scale: degrees per pixel in x and y directions
    scale_x = np.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)  # deg/pix
    scale_y = np.sqrt(A[0, 1] ** 2 + A[1, 1] ** 2)
    scale_deg_per_pix = (scale_x + scale_y) / 2

    # Invert the affine WCS: (ra, dec) → (x, y) is A @ [ra, dec, 1]
    # We need the pseudo-inverse for (x, y) → (ra, dec).
    # A is (2, 3): stack homogeneous to get a 3×3 system.
    # Simple least-squares inverse:
    A2 = wcs.A[:, :2]   # (2, 2)
    b  = wcs.A[:, 2]    # (2,)
    try:
        A2_inv = np.linalg.inv(A2)
    except np.linalg.LinAlgError:
        return np.array([])

    residuals = []
    for xi, yi in zip(x_pix, y_pix):
        pix_vec = np.array([xi, yi]) - b
        radec   = A2_inv @ pix_vec   # approximate (ra, dec) for this pixel

        # Nearest catalogue match within a loose sky radius
        radius_deg = search_radius_pix * scale_deg_per_pix * 3
        sid, ang_deg = star_tree.nearest(float(radec[0]), float(radec[1]))
        if ang_deg < radius_deg:
            residuals.append(ang_deg * 3600.0)   # convert to arcsec

    return np.array(residuals, dtype=float)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m astrometry.plate_solve",
        description="Plate-solve an image from pixel-space star detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--detections", required=True,
                        help="Path to detections file (.npy or .csv).  "
                             "Columns: y_pix, x_pix, flux.")
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--image_width",  type=int, required=True)

    # Index
    parser.add_argument("--index_dir",   default="indices/",
                        help="Directory containing codes_tier*.joblib files.")
    parser.add_argument("--star_index",  default=None,
                        help="Path to stars.joblib.  "
                             "Defaults to <index_dir>/stars.joblib.")

    # Solve options
    parser.add_argument("--fov_deg",     type=float, default=None,
                        help="Image FOV diagonal (degrees).  "
                             "Enables scale-tier filtering.")
    parser.add_argument("--sort_by",     choices=["flux", "magnitude"],
                        default="flux",
                        help="How to rank detected stars (brightest first).")
    parser.add_argument("--max_stars",   type=int, default=20,
                        help="Max detected stars passed to quad enumerator.")
    parser.add_argument("--max_quads",   type=int, default=300,
                        help="Hard cap on pixel-space quads evaluated.")
    parser.add_argument("--code_radius", type=float, default=0.02,
                        help="L2 search radius in 4D code space.")
    parser.add_argument("--model",
                        choices=["simple_independence", "asymmetric"],
                        default="simple_independence",
                        help="Bayesian verification model.")
    parser.add_argument("--variance",    type=float, default=9.0,
                        help="Positional variance for Bayesian model (pixels²).")
    parser.add_argument("--distractors", type=float, default=0.25,
                        help="Distractor fraction for Bayesian model.")
    parser.add_argument("--no_verify",   action="store_true",
                        help="Skip residual computation after a successful solve.")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress step-by-step log; print only the summary.")

    args = parser.parse_args()

    # Load
    code_trees, star_tree = load_index(
        args.index_dir,
        star_index_path=args.star_index,
        verbose=not args.quiet,
    )

    detections = load_detections(args.detections)
    if not args.quiet:
        print(f"[plate_solve] Loaded {len(detections)} detections from '{args.detections}'")

    # Solve
    result = solve_field(
        detections,
        code_trees, star_tree,
        image_height=args.image_height,
        image_width=args.image_width,
        fov_deg=args.fov_deg,
        sort_by=args.sort_by,
        max_stars=args.max_stars,
        max_quads=args.max_quads,
        code_radius=args.code_radius,
        model=args.model,
        variance=args.variance,
        distractors=args.distractors,
        run_verify=not args.no_verify,
        verbose=not args.quiet,
    )

    print()
    print("── Result " + "─" * 60)
    print(result.summary())
    sys.exit(0 if result.solved else 1)


if __name__ == "__main__":
    _main()

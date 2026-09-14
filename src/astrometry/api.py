"""
api.py — Flask REST API for the astrometric plate solver.

Indices are loaded once at startup from the directory specified by the
INDEX_DIR environment variable (default: /indices).  Every subsequent
request is served from in-memory KD-trees with no I/O on the hot path.

Endpoints
---------
GET  /health        Liveness + index summary.
GET  /info          Full index metadata (tiers, star count, RA/Dec ranges).
POST /solve         Blind plate-solve a set of detected stars.

Solve request body (JSON)
--------------------------
{
  "detections":  [[y_pix, x_pix, flux], ...],   // required
  "image_height": 2048,                          // required
  "image_width":  2048,                          // required
  "fov_deg":      2.0,                           // optional
  "sort_by":      "flux",                        // optional: "flux"|"magnitude"
  "max_stars":    20,                            // optional
  "max_quads":    300,                           // optional
  "code_radius":  0.02,                          // optional
  "model":        "asymmetric",                  // optional: "asymmetric"|"simple_independence"
  "variance":     9.0,                           // optional
  "distractors":  0.25,                          // optional
  "verbose":      false                          // optional: include step log in response
}

Solve response body (JSON)
---------------------------
{
  "solved":           true,
  "elapsed_s":        0.034,
  "n_detected":       12,
  "n_used":           9,
  "quads_tried":      7,
  "quads_matched":    2,
  "wcs": {                                       // null when not solved
    "A":    [[ax, ay, a1], [bx, by, b1]],       // pix = A @ [ξ, η, 1]  (gnomonic)
    "ra0":  83.82,                               // tangent-point RA  (degrees)
    "dec0": -5.39                                // tangent-point Dec (degrees)
  },
  "match_ra":         [83.75, 84.25, 83.80, 84.20],  // null when not solved
  "match_dec":        [-5.5, -5.0, -5.0, -5.5],
  "match_pix":        [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
  "residuals_arcsec": [0.5, 0.8, ...],          // null when not solved
  "log":              [...]                      // only present when verbose=true
}

Environment variables
---------------------
INDEX_DIR    Directory containing codes_tier*.joblib files (default: /indices).
STAR_INDEX   Full path to stars.joblib (default: INDEX_DIR/stars.joblib).
HOST         Bind host (default: 0.0.0.0).
PORT         Bind port (default: 5000).

Authors: Peter Thomas
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from flask import Flask, jsonify, request

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrometry.kd_tree import CodeSpaceTree, StarPositionTree
from astrometry.plate_solve import solve_field

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Index state ───────────────────────────────────────────────────────────────

_code_trees: List[CodeSpaceTree] = []
_star_tree: Optional[StarPositionTree] = None
_index_dir: str = ""
_load_error: Optional[str] = None


def _load_indices() -> None:
    global _code_trees, _star_tree, _index_dir, _load_error

    index_dir  = os.environ.get("INDEX_DIR", "/indices")
    star_path  = os.environ.get("STAR_INDEX", str(Path(index_dir) / "stars.joblib"))
    _index_dir = index_dir

    code_paths = sorted(glob.glob(str(Path(index_dir) / "codes_tier*.joblib")))

    if not code_paths:
        _load_error = (
            f"No codes_tier*.joblib files found in '{index_dir}'. "
            "Mount the index directory at that path before starting the server."
        )
        log.warning(_load_error)
        return

    if not Path(star_path).exists():
        _load_error = f"Star index not found: '{star_path}'"
        log.warning(_load_error)
        return

    t0 = time.perf_counter()
    try:
        _code_trees = [CodeSpaceTree.load(p) for p in code_paths]
        _star_tree  = StarPositionTree.load(star_path)
    except Exception as exc:
        _load_error = f"Failed to load indices: {exc}"
        log.error(_load_error)
        return

    elapsed = (time.perf_counter() - t0) * 1e3
    log.info(
        "Loaded %d code tier(s) + star index in %.1f ms  "
        "(%s stars, %s quads total)",
        len(_code_trees),
        elapsed,
        f"{len(_star_tree.ra):,}",
        f"{sum(len(ct.codes) for ct in _code_trees):,}",
    )
    for ct, p in zip(_code_trees, code_paths):
        lo = ct.scale_lower_deg
        hi = ct.scale_upper_deg
        tier_str = (f"{lo:.3f}°–{hi:.3f}°"
                    if lo is not None and hi is not None else "no scale info")
        log.info("  %s: %d quads  [%s]", Path(p).name, len(ct.codes), tier_str)

    _load_error = None


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    """Quick liveness check — always 200, reports index readiness in body."""
    ready = _star_tree is not None and len(_code_trees) > 0
    body = {
        "status":         "ok",
        "index_ready":    ready,
        "index_dir":      _index_dir,
        "n_code_tiers":   len(_code_trees),
        "n_stars":        int(len(_star_tree.ra)) if _star_tree else 0,
        "n_quads_total":  int(sum(len(ct.codes) for ct in _code_trees)),
    }
    if _load_error:
        body["load_error"] = _load_error
    return jsonify(body), 200


@app.get("/info")
def info():
    """Detailed index metadata."""
    if _star_tree is None or not _code_trees:
        return jsonify({"error": _load_error or "Indices not loaded"}), 503

    tiers = []
    for ct in _code_trees:
        tiers.append({
            "n_quads":        int(len(ct.codes)),
            "scale_lower_deg": ct.scale_lower_deg,
            "scale_upper_deg": ct.scale_upper_deg,
        })

    return jsonify({
        "index_dir":  _index_dir,
        "n_stars":    int(len(_star_tree.ra)),
        "ra_range":   [float(_star_tree.ra.min()),  float(_star_tree.ra.max())],
        "dec_range":  [float(_star_tree.dec.min()), float(_star_tree.dec.max())],
        "code_tiers": tiers,
    }), 200


@app.post("/solve")
def solve():
    """
    Blind plate-solve a set of detected stars.

    See module docstring for the full request/response schema.
    """
    if _star_tree is None or not _code_trees:
        return jsonify({"error": _load_error or "Indices not loaded"}), 503

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate required fields ──────────────────────────────────────────────
    missing = [k for k in ("detections", "image_height", "image_width")
               if k not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        detections = np.asarray(body["detections"], dtype=float)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid detections: {exc}"}), 400

    if detections.ndim != 2 or detections.shape[1] < 3:
        return jsonify({
            "error": (
                "detections must be a 2-D array with ≥ 3 columns "
                f"[y_pix, x_pix, flux]; got shape {list(detections.shape)}"
            )
        }), 400

    try:
        image_height = int(body["image_height"])
        image_width  = int(body["image_width"])
    except (ValueError, TypeError):
        return jsonify({"error": "image_height and image_width must be integers"}), 400

    # ── Optional parameters ───────────────────────────────────────────────────
    def _opt(key, cast, default):
        v = body.get(key)
        return cast(v) if v is not None else default

    fov_deg      = _opt("fov_deg",      float, None)
    sort_by      = _opt("sort_by",      str,   "flux")
    max_stars    = _opt("max_stars",    int,   20)
    max_quads    = _opt("max_quads",    int,   300)
    code_radius  = _opt("code_radius",  float, 0.02)
    model        = _opt("model",        str,   "asymmetric")
    variance     = _opt("variance",     float, 9.0)
    distractors  = _opt("distractors",  float, 0.25)
    verbose      = bool(body.get("verbose", False))

    if sort_by not in ("flux", "magnitude"):
        return jsonify({"error": "sort_by must be 'flux' or 'magnitude'"}), 400
    if model not in ("asymmetric", "simple_independence"):
        return jsonify({"error": "model must be 'asymmetric' or 'simple_independence'"}), 400

    # ── Solve ─────────────────────────────────────────────────────────────────
    log.info(
        "Solve request: %d detections  %dx%d  fov=%s°  model=%s",
        len(detections), image_width, image_height,
        fov_deg if fov_deg is not None else "?", model,
    )

    result = solve_field(
        detections,
        code_trees=_code_trees,
        star_tree=_star_tree,
        image_height=image_height,
        image_width=image_width,
        fov_deg=fov_deg,
        sort_by=sort_by,
        max_stars=max_stars,
        max_quads=max_quads,
        code_radius=code_radius,
        model=model,
        variance=variance,
        distractors=distractors,
        run_verify=True,
        verbose=False,
    )

    log.info(
        "Solve %s in %.3f s  (%d quads tried, %d matched)",
        "ACCEPTED" if result.solved else "FAILED",
        result.elapsed_s, result.quads_tried, result.quads_matched,
    )

    # ── Build response ────────────────────────────────────────────────────────
    resp: dict = {
        "solved":          result.solved,
        "elapsed_s":       result.elapsed_s,
        "n_detected":      result.n_detected,
        "n_used":          result.n_used,
        "quads_tried":     result.quads_tried,
        "quads_matched":   result.quads_matched,
        "wcs":             None,
        "match_ra":        None,
        "match_dec":       None,
        "match_pix":       None,
        "residuals_arcsec": None,
    }

    if result.solved:
        resp["wcs"] = {
            "A":    result.wcs.A.tolist(),
            "ra0":  float(getattr(result.wcs, "ra0",  0.0)),
            "dec0": float(getattr(result.wcs, "dec0", 0.0)),
        }
        if result.match_ra is not None:
            resp["match_ra"]  = result.match_ra.tolist()
            resp["match_dec"] = result.match_dec.tolist()
        if result.match_pix is not None:
            resp["match_pix"] = result.match_pix.tolist()
        if result.residuals_arcsec is not None:
            resp["residuals_arcsec"] = result.residuals_arcsec.tolist()

    if verbose:
        resp["log"] = result.log

    return jsonify(resp), 200 if result.solved else 422


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Application factory — loads indices and returns the Flask app."""
    _load_indices()
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    _load_indices()
    app.run(host=host, port=port, debug=False)

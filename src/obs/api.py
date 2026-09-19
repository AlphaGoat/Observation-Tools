"""
api.py — Flask REST API for pixel-to-RA/Dec observation projection.

Accepts the affine WCS from the plate solver, a list of satellite pixel
detections, and the frame timing, then returns the projected (RA, Dec)
observations ready for the MHT associator.

The WCS A matrix is taken directly from the plate solver's /solve response
(the "wcs.A" field), so the two services chain together with no intermediate
transformation.

Endpoints
---------
GET  /health    Liveness check.
POST /project   Project pixel detections to celestial coordinates.

Project request body (JSON)
----------------------------
{
  "wcs": {
    "A":   [[ax, ay, a1],    // required — 2×3 WCS matrix from plate solver
            [bx, by, b1]],   //   forward: [x, y]ᵀ = A @ [ξ, η, 1]ᵀ (gnomonic)
    "ra0":  83.85,           // optional — tangent-point RA  (degrees, default 0)
    "dec0": -5.375           // optional — tangent-point Dec (degrees, default 0)
  },
  "t_start_s":  0.0,         // required — shutter-open epoch (seconds, any consistent origin)
  "t_end_s":    1.0,         // required — shutter-close epoch (seconds)
  "satellites": [            // required — satellite point-source detections
    {"x": 256.0, "y": 256.0},               // dict form
    {"x": 180.0, "y": 320.0, "flux": 3200}, // optional extra fields are ignored
    [128.0, 220.0]                           // [x, y] array form also accepted
  ],
  "star_trails": [           // optional — star-trail midpoints for WCS verification
    {"x_mid": 120.0, "y_mid": 100.0, "flux": 12000.0},
    [300.0, 250.0, 9000.0]   // [x_mid, y_mid, flux] array form also accepted
  ]
}

Project response body (JSON)
-----------------------------
{
  "n_observations":  2,
  "elapsed_s":       0.001,
  "observations": [
    {"ra": 83.850, "dec": -5.375, "t_start": 0.0, "t_end": 1.0},
    {"ra": 83.899, "dec": -5.409, "t_start": 0.0, "t_end": 1.0}
  ],
  "star_positions": [        // null when star_trails not provided
    {"ra": 83.600, "dec": -5.600, "flux": 12000.0},
    ...
  ]
}

observations is a list of (ra, dec, t_start, t_end) dicts that can be passed
directly as one frame in the associator's /associate request.

Chaining plate solver → projection → associator
-----------------------------------------------
    # 1. Extract from image
    ext = requests.post("http://extractor:5002/extract", json={...}).json()

    # 2. Plate solve on the star trails
    sol = requests.post("http://platesolver:5000/solve", json={
        "detections":   ext["plate_solver_input"],
        "image_height": H, "image_width": W,
    }).json()

    # 3. Project satellite detections to sky
    proj = requests.post("http://projector:5003/project", json={
        "wcs":        sol["wcs"],
        "t_start_s":  t_start,
        "t_end_s":    t_end,
        "satellites": ext["satellites"],
    }).json()

    # 4. Associate across frames
    assoc = requests.post("http://associator:5001/associate", json={
        "frames": [proj["observations"], ...],
        "exposure_time": 1.0, "gap_time": 2.5, "sensor_fov": 2.0,
    }).json()

Environment variables
---------------------
HOST    Bind host (default 0.0.0.0).
PORT    Bind port (default 5003).

Authors: Peter Thomas
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obs.project_obs_from_pixel_to_radec import (
    project_satellite_detections,
    project_star_streaks,
)
from astrometry.kd_tree import WCS

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/project")
def project():
    """
    Project satellite pixel detections to celestial (RA, Dec) coordinates.

    See module docstring for the full request/response schema.
    """
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate required fields ──────────────────────────────────────────────
    missing = [k for k in ("wcs", "t_start_s", "t_end_s", "satellites")
               if k not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # ── Parse WCS ─────────────────────────────────────────────────────────────
    wcs_body = body["wcs"]
    if not isinstance(wcs_body, dict) or "A" not in wcs_body:
        return jsonify({"error": "'wcs' must be an object with an 'A' field"}), 400

    try:
        A = np.asarray(wcs_body["A"], dtype=np.float64)
        if A.shape != (2, 3):
            raise ValueError(f"Expected shape (2, 3), got {list(A.shape)}")
        ra0  = float(wcs_body.get("ra0",  0.0))
        dec0 = float(wcs_body.get("dec0", 0.0))
        wcs  = WCS(A=A, ra0=ra0, dec0=dec0)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid WCS matrix: {exc}"}), 400

    # ── Parse timing ──────────────────────────────────────────────────────────
    try:
        t_start_s = float(body["t_start_s"])
        t_end_s   = float(body["t_end_s"])
    except (TypeError, ValueError):
        return jsonify({"error": "t_start_s and t_end_s must be numbers"}), 400

    if t_end_s <= t_start_s:
        return jsonify({"error": "t_end_s must be greater than t_start_s"}), 400

    # ── Parse satellite detections ────────────────────────────────────────────
    raw_sats = body["satellites"]
    if not isinstance(raw_sats, list):
        return jsonify({"error": "'satellites' must be a list"}), 400

    # Validate each entry is parseable (dict or 2-element sequence)
    for k, entry in enumerate(raw_sats):
        try:
            if isinstance(entry, dict):
                _ = float(entry["x"]), float(entry["y"])
            else:
                seq = list(entry)
                if len(seq) < 2:
                    raise ValueError("need at least [x, y]")
                _ = float(seq[0]), float(seq[1])
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({
                "error": (
                    f"satellites[{k}] is not a valid detection. "
                    f"Expected {{x, y}} or [x, y]. Detail: {exc}"
                )
            }), 400

    # ── Parse optional star trails ────────────────────────────────────────────
    raw_trails: Optional[list] = body.get("star_trails")
    if raw_trails is not None and not isinstance(raw_trails, list):
        return jsonify({"error": "'star_trails' must be a list when provided"}), 400

    log.info(
        "Project request: %d satellite(s)  %s trail(s)  t=[%.3f, %.3f]",
        len(raw_sats),
        len(raw_trails) if raw_trails else 0,
        t_start_s,
        t_end_s,
    )

    # ── Project ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        obs = project_satellite_detections(wcs, raw_sats, t_start_s, t_end_s)
    except Exception as exc:
        log.exception("Satellite projection failed: %s", exc)
        return jsonify({"error": f"Projection error: {exc}"}), 500

    star_sky = None
    if raw_trails:
        try:
            arr = project_star_streaks(wcs, raw_trails)
            star_sky = [
                {"ra": float(row[0]), "dec": float(row[1]), "flux": float(row[2])}
                for row in arr
            ]
        except Exception as exc:
            log.warning("Star-trail projection failed (non-fatal): %s", exc)

    elapsed = time.perf_counter() - t0

    log.info(
        "Projection complete: %d observation(s) in %.3f s",
        len(obs), elapsed,
    )

    return jsonify({
        "n_observations": len(obs),
        "elapsed_s":      elapsed,
        "observations": [
            {"ra": o[0], "dec": o[1], "t_start": o[2], "t_end": o[3]}
            for o in obs
        ],
        "star_positions": star_sky,
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5003"))
    app.run(host=host, port=port, debug=False)

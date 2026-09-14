"""
api.py — Flask REST API for SEP-based source extraction.

In a rate-tracked observation the telescope slews to follow a satellite, so
stars trail across the detector while the satellite appears as a compact point
source.  This service wraps sep_extractor to classify those two populations
and return catalogs ready for the downstream plate solver and track associator.

Endpoints
---------
GET  /health     Liveness check.
POST /extract    Run SEP extraction on one image frame.

Extract request body (JSON)
----------------------------
{
  "image":  "<base64>",     // required — base64-encoded image bytes (see "format")
  "format": "npy",          // "npy" (default) — base64(numpy .npy bytes)
                            // "fits"           — base64(FITS file bytes)
  "hdu":    0,              // FITS HDU index (default 0; ignored for npy)

  // Detection parameters (all optional)
  "threshold":          3.0,   // sigma above local background
  "minarea":            5,     // minimum connected pixels for a valid source
  "elong_thresh":       3.0,   // a/b ratio separating star trails from point sources
  "min_streak_px":     20.0,   // minimum semi-major axis (px) to accept a star trail
  "sat_fwhm_max_px":   20.0,   // maximum FWHM (px) for a satellite point source
  "bkg_box_size":      64,     // background grid cell size (pixels)
  "gain":               1.0,   // detector gain (e-/ADU)
  "angle_filter_sigma": 2.0    // streak-angle consistency filter (null = disabled)
}

Encoding the image (Python client example)
------------------------------------------
    import io, base64, numpy as np

    # From a numpy array
    buf = io.BytesIO()
    np.save(buf, image_array)
    payload = base64.b64encode(buf.getvalue()).decode()
    body = {"image": payload, "format": "npy"}

    # From a FITS file
    payload = base64.b64encode(open("image.fits", "rb").read()).decode()
    body = {"image": payload, "format": "fits"}

Extract response body (JSON)
-----------------------------
{
  "n_satellites":  2,
  "n_star_trails": 15,
  "elapsed_s":     0.045,
  "satellites": [
    {"x": 256.0, "y": 256.0, "flux": 5000.0, "snr": 40.0,
     "fwhm_px": 4.8, "a": 2.1, "b": 2.0, "theta_deg": 2.9},
    ...
  ],
  "star_trails": [
    {"x_mid": 120.0, "y_mid": 100.0,
     "x1": 65.0, "y1": 68.0, "x2": 175.0, "y2": 132.0,
     "length_px": 110.0, "angle_deg": 30.0, "flux": 12000.0, "snr": 80.0},
    ...
  ],
  "streak_stats": {
    "angle_deg": 30.0, "angle_std_deg": 0.2,
    "length_px": 110.0, "length_std_px": 11.0, "n": 15
  },
  "plate_solver_input": [[y_mid, x_mid, flux], ...]
}

plate_solver_input is the [[y, x, flux], …] array that /solve on the plate
solver service accepts directly as its "detections" field.

Environment variables
---------------------
HOST            Bind host  (default 0.0.0.0).
PORT            Bind port  (default 5002).
EXTRACTOR_TYPE  Controls which detections are included in the response:
                  "both"       (default) — satellites + star trails
                  "stars"      — only star trails (for the star-extractor deployment)
                  "satellites" — only satellite point sources (for satellite-extractor)
                When running as two separate Kubernetes deployments, set this env var
                on each so each service only does the work its consumers need.

Authors: Peter Thomas
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source_extraction.sep_extractor import (
    SatelliteDetection,
    StarStreak,
    extract_sources,
    star_array,
    streak_stats,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Image decoding ────────────────────────────────────────────────────────────

def _decode_image(b64: str, fmt: str, hdu: int) -> np.ndarray:
    """
    Decode a base64-encoded image payload to a 2-D float64 numpy array.

    Supported formats
    -----------------
    "npy"  : numpy .npy bytes written by np.save()
    "fits" : a FITS file; the primary (or requested) HDU's data is used
    """
    raw = base64.b64decode(b64)

    if fmt == "npy":
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(
                f"numpy array must be 2-D; got shape {arr.shape}"
            )
        return arr.astype(np.float64)

    if fmt == "fits":
        from astropy.io import fits as pyfits
        with pyfits.open(io.BytesIO(raw)) as hdul:
            data = hdul[hdu].data
        if data is None or data.ndim != 2:
            raise ValueError(
                f"FITS HDU {hdu} does not contain a 2-D image"
            )
        return data.astype(np.float64)

    raise ValueError(f"Unknown image format {fmt!r}; expected 'npy' or 'fits'")


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _ser_sat(s: SatelliteDetection) -> dict:
    return {
        "x": s.x, "y": s.y,
        "flux": s.flux, "snr": s.snr,
        "fwhm_px": s.fwhm_px,
        "a": s.a, "b": s.b,
        "theta_deg": s.theta_deg,
    }


def _ser_streak(s: StarStreak) -> dict:
    return {
        "x_mid": s.x_mid, "y_mid": s.y_mid,
        "x1": s.x1, "y1": s.y1,
        "x2": s.x2, "y2": s.y2,
        "length_px": s.length_px,
        "angle_deg": s.angle_deg,
        "flux": s.flux, "snr": s.snr,
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# What this instance returns — set via EXTRACTOR_TYPE env var.
# "both" (default) keeps backward compatibility with the original API.
_EXTRACTOR_TYPE = os.environ.get("EXTRACTOR_TYPE", "both").lower()
if _EXTRACTOR_TYPE not in ("both", "stars", "satellites"):
    log.warning(
        "Unknown EXTRACTOR_TYPE=%r — defaulting to 'both'.", _EXTRACTOR_TYPE
    )
    _EXTRACTOR_TYPE = "both"

log.info("EXTRACTOR_TYPE = %r", _EXTRACTOR_TYPE)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "extractor_type": _EXTRACTOR_TYPE}), 200


@app.post("/extract")
def extract():
    """
    Run SEP source extraction on one image frame.

    See module docstring for the full request/response schema.
    """
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    if "image" not in body:
        return jsonify({"error": "Missing required field: 'image'"}), 400

    fmt = str(body.get("format", "npy")).lower()
    if fmt not in ("npy", "fits"):
        return jsonify({"error": "'format' must be 'npy' or 'fits'"}), 400

    try:
        hdu = int(body.get("hdu", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "'hdu' must be an integer"}), 400

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        image = _decode_image(body["image"], fmt, hdu)
    except Exception as exc:
        return jsonify({"error": f"Image decode failed: {exc}"}), 400

    # ── Parse optional extraction parameters ──────────────────────────────────
    def _opt(key, cast, default):
        v = body.get(key)
        return cast(v) if v is not None else default

    threshold    = _opt("threshold",          float, 3.0)
    minarea      = _opt("minarea",            int,   5)
    elong_thresh = _opt("elong_thresh",       float, 3.0)
    min_streak   = _opt("min_streak_px",      float, 20.0)
    sat_fwhm_max = _opt("sat_fwhm_max_px",    float, 20.0)
    bkg_box      = _opt("bkg_box_size",       int,   64)
    gain         = _opt("gain",               float, 1.0)

    raw_asig = body.get("angle_filter_sigma", 2.0)
    angle_filter_sigma: Optional[float] = None if raw_asig is None else float(raw_asig)

    log.info(
        "Extract request: %dx%d image  thresh=%.1f  elong=%.1f  "
        "min_streak=%.0fpx  fmt=%s",
        image.shape[1], image.shape[0],
        threshold, elong_thresh, min_streak, fmt,
    )

    # ── Run extraction ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        sats, streaks = extract_sources(
            image,
            threshold=threshold,
            minarea=minarea,
            elong_thresh=elong_thresh,
            min_streak_px=min_streak,
            sat_fwhm_max_px=sat_fwhm_max,
            bkg_box_size=bkg_box,
            gain=gain,
            angle_filter_sigma=angle_filter_sigma,
        )
    except Exception as exc:
        log.exception("SEP extraction failed: %s", exc)
        return jsonify({"error": f"Extraction error: {exc}"}), 500

    elapsed = time.perf_counter() - t0

    log.info(
        "Extraction complete: %d satellite(s)  %d star trail(s)  %.3f s",
        len(sats), len(streaks), elapsed,
    )

    # ── Build response ────────────────────────────────────────────────────────
    plate_input = star_array(streaks).tolist()  # [[y_mid, x_mid, flux], …]
    stats       = streak_stats(streaks)

    # Always include image dimensions so the plate solver can be called without
    # the client needing to parse the image header separately.
    h, w = image.shape

    if _EXTRACTOR_TYPE == "stars":
        return jsonify({
            "n_star_trails":     len(streaks),
            "elapsed_s":         elapsed,
            "image_height":      h,
            "image_width":       w,
            "star_trails":       [_ser_streak(s) for s in streaks],
            "streak_stats":      stats,
            "plate_solver_input": plate_input,
        }), 200

    if _EXTRACTOR_TYPE == "satellites":
        return jsonify({
            "n_satellites": len(sats),
            "elapsed_s":    elapsed,
            "image_height": h,
            "image_width":  w,
            "satellites":   [_ser_sat(s) for s in sats],
        }), 200

    # "both" — original full response, backward-compatible
    return jsonify({
        "n_satellites":      len(sats),
        "n_star_trails":     len(streaks),
        "elapsed_s":         elapsed,
        "image_height":      h,
        "image_width":       w,
        "satellites":        [_ser_sat(s) for s in sats],
        "star_trails":       [_ser_streak(s) for s in streaks],
        "streak_stats":      stats,
        "plate_solver_input": plate_input,
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5002"))
    app.run(host=host, port=port, debug=False)

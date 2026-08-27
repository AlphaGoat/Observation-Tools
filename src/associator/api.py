"""
api.py — Flask REST API for the MHT observation associator.

Each request is a stateless batch job: the client sends all frames for one
collect and receives the best track per seed detection.  No server-side session
state is maintained between requests.

Endpoints
---------
GET  /health        Liveness check.
POST /associate     Run MHT on a multi-frame observation set; return tracks.

Associate request body (JSON)
------------------------------
{
  "frames": [                          // required — list of frames, oldest first
    [                                  // frame 0
      {"ra": 10.0, "dec": 20.0, "t_start": 0.0, "t_end": 1.0},
      ...
    ],
    [                                  // frame 1
      [10.1, 20.1, 3.5, 4.5],         // arrays [ra, dec, t_start, t_end] also accepted
      ...
    ]
  ],
  "exposure_time":       1.0,          // required — frame exposure in seconds
  "gap_time":            2.5,          // required — inter-frame gap in seconds
  "sensor_fov":          2.0,          // required — FOV diagonal in degrees (clutter volume)
  "distance_threshold":  9.21,         // optional — Mahalanobis gate (chi² 99th pct)
  "k_best":              10,           // optional — max hypotheses per tree per frame
  "w_motion":            1.0,          // optional — weight on motion log-likelihood
  "w_appearance":        0.0           // optional — weight on appearance score (placeholder)
}

Associate response body (JSON)
-------------------------------
{
  "n_tracks":  2,
  "elapsed_s": 0.012,
  "tracks": [
    {
      "score":        12.34,
      "n_frames":     3,
      "observations": [
        {"frame": 0, "ra": 10.00, "dec": 20.00, "t_start": 0.0, "t_end": 1.0},
        {"frame": 1, "ra": 10.10, "dec": 20.10, "t_start": 3.5, "t_end": 4.5},
        {"frame": 2, "ra": 10.20, "dec": 20.20, "t_start": 7.0, "t_end": 8.0}
      ],
      "final_state": {
        "ra":      10.20,
        "dec":     20.20,
        "ra_dot":  0.0286,
        "dec_dot": 0.0286
      },
      "final_covar": [[...], [...], [...], [...]]
    }
  ]
}

Environment variables
---------------------
HOST     Bind host (default: 0.0.0.0).
PORT     Bind port (default: 5001).

Authors: Peter Thomas
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from associator.MHT import run_multiple_hypothesis_tracking, Track

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Observation: (ra_deg, dec_deg, t_start_s, t_end_s)
Observation = Tuple[float, float, float, float]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_observation(raw) -> Optional[Observation]:
    """
    Accept an observation as either a dict or a 4-element array.

    Dict form:  {"ra": float, "dec": float, "t_start": float, "t_end": float}
    Array form: [ra, dec, t_start, t_end]

    Returns None on bad input.
    """
    try:
        if isinstance(raw, dict):
            return (
                float(raw["ra"]),
                float(raw["dec"]),
                float(raw["t_start"]),
                float(raw["t_end"]),
            )
        seq = list(raw)
        if len(seq) < 4:
            return None
        return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
    except (KeyError, TypeError, ValueError):
        return None


def _serialize_track(track: Track) -> dict:
    """Convert a Track object to a JSON-serializable dict."""
    obs_list = []
    for node in track.nodes:
        ra, dec, t_start, t_end = node.ob
        obs_list.append({
            "frame":   node.frame_num,
            "ra":      ra,
            "dec":     dec,
            "t_start": t_start,
            "t_end":   t_end,
        })

    sv = track.k_filter.state_vector
    cov = track.k_filter.covar

    return {
        "score":        float(track.calculate_score()),
        "n_frames":     len(track.nodes),
        "observations": obs_list,
        "final_state": {
            "ra":      float(sv[0]),
            "dec":     float(sv[1]),
            "ra_dot":  float(sv[2]),
            "dec_dot": float(sv[3]),
        },
        "final_covar": cov.tolist(),
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/associate")
def associate():
    """
    Run Multiple Hypothesis Tracking on a multi-frame observation set.

    See module docstring for the full request/response schema.
    """
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate required fields ──────────────────────────────────────────────
    missing = [k for k in ("frames", "exposure_time", "gap_time", "sensor_fov")
               if k not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    raw_frames = body.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) == 0:
        return jsonify({"error": "'frames' must be a non-empty list"}), 400

    # ── Parse frames ──────────────────────────────────────────────────────────
    frames: List[List[Observation]] = []
    for frame_idx, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, list):
            return jsonify({
                "error": f"frames[{frame_idx}] must be a list of observations"
            }), 400
        parsed_frame: List[Observation] = []
        for ob_idx, raw_ob in enumerate(raw_frame):
            ob = _parse_observation(raw_ob)
            if ob is None:
                return jsonify({
                    "error": (
                        f"frames[{frame_idx}][{ob_idx}] is not a valid observation. "
                        "Expected {ra, dec, t_start, t_end} or [ra, dec, t_start, t_end]."
                    )
                }), 400
            parsed_frame.append(ob)
        frames.append(parsed_frame)

    if not frames[0]:
        return jsonify({"error": "frames[0] (the first frame) must not be empty"}), 400

    # ── Parse scalar parameters ───────────────────────────────────────────────
    try:
        exposure_time = float(body["exposure_time"])
        gap_time      = float(body["gap_time"])
        sensor_fov    = float(body["sensor_fov"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "exposure_time, gap_time, and sensor_fov must be numbers"
        }), 400

    if exposure_time <= 0 or gap_time < 0 or sensor_fov <= 0:
        return jsonify({
            "error": "exposure_time and sensor_fov must be positive; gap_time must be ≥ 0"
        }), 400

    def _opt(key, cast, default):
        v = body.get(key)
        return cast(v) if v is not None else default

    distance_threshold = _opt("distance_threshold", float, 9.21)
    k_best             = _opt("k_best",             int,   10)
    w_motion           = _opt("w_motion",            float, 1.0)
    w_appearance       = _opt("w_appearance",        float, 0.0)

    n_detections = sum(len(f) for f in frames)
    log.info(
        "Associate request: %d frame(s)  %d total detections  "
        "exp=%.1fs  gap=%.1fs  fov=%.2f°",
        len(frames), n_detections, exposure_time, gap_time, sensor_fov,
    )

    # ── Run MHT ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        tracks = run_multiple_hypothesis_tracking(
            obs=frames,
            exposure_time=exposure_time,
            gap_time=gap_time,
            sensor_fov=sensor_fov,
            distance_threshold=distance_threshold,
            w_motion=w_motion,
            w_appearance=w_appearance,
            k_best=k_best,
        )
    except Exception as exc:
        log.exception("MHT failed: %s", exc)
        return jsonify({"error": f"MHT error: {exc}"}), 500

    elapsed = time.perf_counter() - t0

    log.info(
        "Association complete: %d track(s) in %.3f s",
        len(tracks), elapsed,
    )

    # ── Serialise ─────────────────────────────────────────────────────────────
    serialised = []
    for track in tracks:
        try:
            serialised.append(_serialize_track(track))
        except Exception as exc:
            log.warning("Could not serialise track: %s", exc)

    return jsonify({
        "n_tracks":  len(serialised),
        "elapsed_s": elapsed,
        "tracks":    serialised,
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host=host, port=port, debug=False)

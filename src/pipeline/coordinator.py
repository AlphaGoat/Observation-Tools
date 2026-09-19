"""
coordinator.py — Pipeline orchestrator for the Observation-Tools service graph.

Drives one complete observation pipeline run:

    ┌──────────────────── per frame (concurrent) ──────────────────────────┐
    │                                                                       │
    │  ┌─────────────────┐       ┌────────────────────┐                   │
    │  │  star-extractor  │──────▶│   plate-solver     │──┐                │
    │  └─────────────────┘       └────────────────────┘  │                │
    │         (parallel)                                   ▼               │
    │  ┌─────────────────┐                         ┌──────────────┐       │
    │  │ satellite-extr. │────────────────────────▶│   projector  │       │
    │  └─────────────────┘                         └──────┬───────┘       │
    │                                                      │               │
    └──────────────────────────────────────────────────────┼───────────────┘
                                                           │
                          ┌────────────────────────────────▼────────────────┐
                          │               associator                         │
                          │  (receives all frame observations together)      │
                          └────────────────────────────────┬────────────────┘
                                                             │  (opt-in, "correlate": true)
                          ┌────────────────────────────────▼────────────────┐
                          │               correlator                         │
                          │  (one call per track, sequential -- see below)   │
                          └─────────────────────────────────────────────────┘

For each frame:
  1. Star extractor  ── extracts star trail midpoints → plate solver input
  2. Satellite extractor ── extracts satellite point sources
     (steps 1 and 2 run in parallel for each frame; all frames run concurrently)
  3. Plate solver ── fits gnomonic WCS from star trails
  4. Projector ── converts satellite pixel positions to (RA, Dec)

After all frames:
  5. Associator ── fits tracks across all frames using MHT
  6. Correlator (opt-in) ── ingests each track into catalog-store and
     correlates it against the catalog (generation/scoring/promotion, see
     correlator/correlate.py). Tracks are sent one at a time, in order, not
     concurrently: two tracks from the same run can legitimately need to
     attribute to the same in-progress hypothesis, and catalog-store's
     generation-phase writes (as opposed to promote_object) aren't
     cross-request atomic -- sending them concurrently risked two tracks
     racing to spawn separate, duplicate hypotheses instead of one seeing
     the other's already-created one.

Endpoints
---------
GET  /health          Liveness check + downstream service status.
POST /pipeline/run    Execute the pipeline on a sequence of frames.

Run request body (JSON)
------------------------
{
  "frames": [
    {
      "image":     "<base64>",   // required — base64-encoded image bytes
      "format":    "fits",       // optional — "fits" (default) | "npy"
      "hdu":       0,            // optional — FITS HDU index (default 0)
      "t_start_s": 0.0,         // required — shutter-open epoch (seconds)
      "t_end_s":   1.0,         // required — shutter-close epoch (seconds)
      "frame_id":  "frame_001"  // optional — identifier used in logs + response
    },
    ...
  ],

  // ── Association parameters (required) ───────────────────────────────────
  "exposure_time": 1.0,         // frame exposure in seconds
  "gap_time":      2.5,         // inter-frame gap in seconds
  "sensor_fov":    2.0,         // sensor FOV diagonal in degrees

  // ── Solver / extraction parameters (optional) ────────────────────────────
  "fov_deg":       2.0,         // expected image FOV (degrees) for tier selection
  "threshold":     3.0,         // SEP detection threshold (sigma)
  "elong_thresh":  3.0,         // a/b elongation threshold for streak classification
  "min_streak_px": 20.0,        // minimum star trail length (pixels)

  // ── Correlation (optional, opt-in) ───────────────────────────────────────
  // Ingests each associator track into catalog-store and correlates it.
  // Off by default: unlike every earlier stage, this writes to shared,
  // persistent state (the catalog), so it isn't turned on implicitly.
  "correlate":        false,          // optional, default false
  "sensor_id":        "rubin",        // required if correlate=true
  "site_position_km": [x, y, z]       // optional even if correlate=true; applied
                                       // to every track in this run as a fixed
                                       // observer position (see module docstring
                                       // "Correlation" section for the caveat this
                                       // implies), needed for IOD-gated promotion
}

Run response body (JSON)
-------------------------
{
  "n_frames":       10,
  "n_solved":       8,
  "n_observations": 45,
  "solve_rate":     0.8,
  "elapsed_s":      12.5,
  "tracks": [...],              // associator output (see associator API docs)
  "frame_results": [
    {
      "frame_id":       "frame_001",
      "t_start_s":      0.0,
      "t_end_s":        1.0,
      "n_stars":        15,
      "n_satellites":   2,
      "solved":         true,
      "n_observations": 2,
      "error":          null     // error description if frame failed, else null
    }
  ],
  "correlation_results": [        // present (possibly empty) iff "correlate" was requested
    {
      "track_idx":  0,            // index into the "tracks" array above
      "result":     {...} | null, // the correlator's own /correlate response, see
                                   // correlator/api.py -- action, branches, promoted, iod
      "error":      null          // set instead of "result" if this track's
                                   // correlate call itself failed (never aborts the run)
    }
  ]
}

Correlation
-------------
Off by default ("correlate": false) because, unlike every other stage
here, it writes to shared persistent state (the catalog via catalog-
store) rather than just computing a response -- re-running the same
frames with correlation on will ingest and attempt to correlate the same
tracks again, is not idempotent, and is meant to be turned on
deliberately, not as a side effect of testing/re-running a pipeline call.

Each associator track becomes one catalog-store attributable: ra/dec/
ra_dot/dec_dot/covariance from the track's final_state/final_covar, and
t_start/t_end spanning its first to last observation. site_position_km,
if given, is applied identically to every track in the run -- a
simplification, not a real per-epoch observer position: nothing in this
pipeline currently computes true site geodesy (lat/lon/alt -> inertial
position over time), so a single fixed vector is the best available
approximation for now (see src/correlator/correlate.py and
src/iod/double_r_lambert.py for how it's used, and src/iod/kepler.py's
"Future work" notes for the larger gap this simplification stands in
for). Omitting it entirely is always safe -- correlation still runs, just
without the option of an IOD-based promotion (see correlator/correlate.py).

Environment variables
---------------------
STAR_EXTRACTOR_URL      http://star-extractor:5002
SATELLITE_EXTRACTOR_URL http://satellite-extractor:5002
PLATE_SOLVER_URL        http://plate-solver:5000
PROJECTOR_URL           http://projector:5003
ASSOCIATOR_URL          http://associator:5001
CORRELATOR_URL          http://correlator:5005
MAX_FRAME_WORKERS       Max concurrent frame threads (default 8)
EXTRACT_TIMEOUT_S       Per-service HTTP timeout for extraction (default 120)
SOLVE_TIMEOUT_S         Plate solver HTTP timeout (default 60)
PROJECT_TIMEOUT_S       Projector HTTP timeout (default 30)
ASSOC_TIMEOUT_S         Associator HTTP timeout (default 120)
CORRELATE_TIMEOUT_S     Per-track correlator HTTP timeout (default 30)
HOST                    Bind host (default 0.0.0.0)
PORT                    Bind port (default 8080)

Authors: Peter Thomas
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Downstream service URLs ───────────────────────────────────────────────────

_STAR_EXTRACTOR_URL    = os.environ.get("STAR_EXTRACTOR_URL",
                                        "http://star-extractor:5002")
_SATELLITE_EXTRACTOR_URL = os.environ.get("SATELLITE_EXTRACTOR_URL",
                                          "http://satellite-extractor:5002")
_PLATE_SOLVER_URL      = os.environ.get("PLATE_SOLVER_URL",
                                        "http://plate-solver:5000")
_PROJECTOR_URL         = os.environ.get("PROJECTOR_URL",
                                        "http://projector:5003")
_ASSOCIATOR_URL        = os.environ.get("ASSOCIATOR_URL",
                                        "http://associator:5001")
_CORRELATOR_URL        = os.environ.get("CORRELATOR_URL",
                                        "http://correlator:5005")

_MAX_FRAME_WORKERS = int(os.environ.get("MAX_FRAME_WORKERS", "8"))

_EXTRACT_TIMEOUT   = int(os.environ.get("EXTRACT_TIMEOUT_S",  "120"))
_SOLVE_TIMEOUT     = int(os.environ.get("SOLVE_TIMEOUT_S",    "60"))
_PROJECT_TIMEOUT   = int(os.environ.get("PROJECT_TIMEOUT_S",  "30"))
_ASSOC_TIMEOUT     = int(os.environ.get("ASSOC_TIMEOUT_S",    "120"))
_CORRELATE_TIMEOUT = int(os.environ.get("CORRELATE_TIMEOUT_S", "30"))


# ── Service call helpers ──────────────────────────────────────────────────────

def _post(url: str, body: dict, timeout: int) -> Tuple[bool, dict]:
    """POST JSON to url. Returns (ok, response_dict)."""
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except requests.exceptions.Timeout:
        return False, {"error": f"timeout after {timeout}s calling {url}"}
    except requests.exceptions.ConnectionError as exc:
        return False, {"error": f"connection error to {url}: {exc}"}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = {}
        return False, {"error": f"HTTP {exc.response.status_code} from {url}",
                       "detail": detail}
    except Exception as exc:
        return False, {"error": f"unexpected error calling {url}: {exc}"}


# ── Per-frame pipeline ────────────────────────────────────────────────────────

def _process_frame(
    frame_payload: dict,
    idx: int,
    extract_params: dict,
    fov_deg: Optional[float],
) -> dict:
    """
    Run the per-frame pipeline stages for one image.

    Returns a frame result dict consumed by the coordinator endpoint.
    """
    image_b64 = frame_payload["image"]
    fmt       = frame_payload.get("format", "fits")
    hdu       = int(frame_payload.get("hdu", 0))
    t_start   = float(frame_payload["t_start_s"])
    t_end     = float(frame_payload["t_end_s"])
    frame_id  = frame_payload.get("frame_id") or f"frame_{idx:04d}"

    result: dict = {
        "frame_id":       frame_id,
        "idx":            idx,
        "t_start_s":      t_start,
        "t_end_s":        t_end,
        "n_stars":        0,
        "n_satellites":   0,
        "solved":         False,
        "n_observations": 0,
        "observations":   [],
        "error":          None,
    }

    # ── Stage 1: extract stars + satellites in parallel ───────────────────────
    extract_body = {
        "image":  image_b64,
        "format": fmt,
        "hdu":    hdu,
        **extract_params,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        star_future = ex.submit(
            _post,
            f"{_STAR_EXTRACTOR_URL}/extract",
            extract_body, _EXTRACT_TIMEOUT,
        )
        sat_future = ex.submit(
            _post,
            f"{_SATELLITE_EXTRACTOR_URL}/extract",
            extract_body, _EXTRACT_TIMEOUT,
        )
        star_ok, star_data = star_future.result()
        sat_ok,  sat_data  = sat_future.result()

    if not star_ok:
        result["error"] = f"star extraction failed: {star_data.get('error')}"
        log.warning("[%s] %s", frame_id, result["error"])
        return result

    if not sat_ok:
        # Missing satellite data is non-fatal — we can still plate-solve
        log.warning("[%s] satellite extraction failed: %s",
                    frame_id, sat_data.get("error"))
        sat_data = {"satellites": [], "n_satellites": 0}

    plate_input = star_data.get("plate_solver_input", [])
    satellites  = sat_data.get("satellites", [])

    result["n_stars"]     = star_data.get("n_star_trails", 0)
    result["n_satellites"] = sat_data.get("n_satellites", 0)

    log.info("[%s] %d star trails  %d satellites",
             frame_id, result["n_stars"], result["n_satellites"])

    if not plate_input:
        result["error"] = "no star trails detected — cannot plate-solve"
        return result

    # ── Stage 2: plate solve ──────────────────────────────────────────────────
    # image_height / image_width come from the extractor response so the client
    # doesn't need to parse the FITS header separately.
    solve_ok, solve_data = _post(
        f"{_PLATE_SOLVER_URL}/solve",
        {
            "detections":   plate_input,
            "image_height": star_data.get("image_height", 2048),
            "image_width":  star_data.get("image_width",  2048),
            "fov_deg":      fov_deg,
        },
        _SOLVE_TIMEOUT,
    )

    if not solve_ok:
        result["error"] = f"plate solver error: {solve_data.get('error')}"
        return result

    if not solve_data.get("solved"):
        result["error"] = "plate solve did not converge"
        return result

    result["solved"] = True
    wcs_payload = solve_data["wcs"]

    log.info("[%s] solved — ra0=%.3f  dec0=%.3f",
             frame_id,
             wcs_payload.get("ra0", 0.0),
             wcs_payload.get("dec0", 0.0))

    if not satellites:
        # Solved but no satellites detected — nothing to project
        return result

    # ── Stage 3: project satellite detections to (RA, Dec) ───────────────────
    proj_ok, proj_data = _post(
        f"{_PROJECTOR_URL}/project",
        {
            "wcs":        wcs_payload,
            "t_start_s":  t_start,
            "t_end_s":    t_end,
            "satellites": satellites,
        },
        _PROJECT_TIMEOUT,
    )

    if not proj_ok:
        result["error"] = f"projection failed: {proj_data.get('error')}"
        return result

    obs = proj_data.get("observations", [])
    result["observations"]   = obs
    result["n_observations"] = len(obs)

    log.info("[%s] %d observation(s) projected", frame_id, len(obs))
    return result


# ── Correlation (opt-in, Stage 6) ───────────────────────────────────────────────

def _build_track_attributable(track: dict, sensor_id: str, site_position_km: Optional[list]) -> dict:
    """Convert one associator track into a catalog-store attributable body."""
    observations = track["observations"]
    body = {
        "sensor_id": sensor_id,
        "t_start":   min(o["t_start"] for o in observations),
        "t_end":     max(o["t_end"] for o in observations),
        "ra":        track["final_state"]["ra"],
        "dec":       track["final_state"]["dec"],
        "ra_dot":    track["final_state"]["ra_dot"],
        "dec_dot":   track["final_state"]["dec_dot"],
        "covariance": track["final_covar"],
    }
    if site_position_km is not None:
        body["site_position_km"] = site_position_km
    return body


def _correlate_tracks(tracks: List[dict], sensor_id: str, site_position_km: Optional[list]) -> List[dict]:
    """
    Ingest and correlate each track against the catalog, one at a time.

    Sequential, not concurrent (see module docstring's Stage 6 note): two
    tracks from the same run can legitimately need to attribute to the
    same in-progress hypothesis, and catalog-store's generation-phase
    writes aren't cross-request atomic the way promote_object is.
    """
    results: List[dict] = []
    for idx, track in enumerate(tracks):
        try:
            attributable = _build_track_attributable(track, sensor_id, site_position_km)
        except (KeyError, IndexError) as exc:
            results.append({"track_idx": idx, "result": None, "error": f"malformed track: {exc}"})
            continue

        ok, data = _post(f"{_CORRELATOR_URL}/correlate", attributable, _CORRELATE_TIMEOUT)
        if ok:
            results.append({"track_idx": idx, "result": data, "error": None})
        else:
            error = data.get("error", "unknown correlation error")
            results.append({"track_idx": idx, "result": None, "error": error})
            log.warning("Correlation failed for track %d: %s", idx, error)

    return results


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    """Liveness check.  Probes all downstream services and reports their status."""
    services = {
        "star_extractor":    f"{_STAR_EXTRACTOR_URL}/health",
        "satellite_extractor": f"{_SATELLITE_EXTRACTOR_URL}/health",
        "plate_solver":      f"{_PLATE_SOLVER_URL}/health",
        "projector":         f"{_PROJECTOR_URL}/health",
        "associator":        f"{_ASSOCIATOR_URL}/health",
        "correlator":        f"{_CORRELATOR_URL}/health",
    }
    statuses: Dict[str, Any] = {}
    all_ok = True
    for name, url in services.items():
        try:
            r = requests.get(url, timeout=5)
            statuses[name] = {"status": "ok", "code": r.status_code}
        except Exception as exc:
            statuses[name] = {"status": "unreachable", "error": str(exc)}
            all_ok = False

    return jsonify({
        "status":   "ok" if all_ok else "degraded",
        "services": statuses,
    }), 200


@app.post("/pipeline/run")
def pipeline_run():
    """
    Execute the full observation pipeline on a sequence of image frames.

    See module docstring for the complete request / response schema.
    """
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate required top-level fields ────────────────────────────────────
    frames = body.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        return jsonify({"error": "'frames' must be a non-empty list"}), 400

    for req in ("exposure_time", "gap_time", "sensor_fov"):
        if req not in body:
            return jsonify({"error": f"Missing required field: '{req}'"}), 400

    for k, frame in enumerate(frames):
        for req in ("image", "t_start_s", "t_end_s"):
            if req not in frame:
                return jsonify({
                    "error": f"frames[{k}] is missing required field '{req}'"
                }), 400

    correlate = bool(body.get("correlate", False))
    if correlate and "sensor_id" not in body:
        return jsonify({"error": "'sensor_id' is required when 'correlate' is true"}), 400

    # ── Parse optional parameters ─────────────────────────────────────────────
    fov_deg      = body.get("fov_deg")
    exposure_time = float(body["exposure_time"])
    gap_time      = float(body["gap_time"])
    sensor_fov    = float(body["sensor_fov"])
    sensor_id     = body.get("sensor_id")
    site_position_km = body.get("site_position_km")

    extract_params = {k: body[k] for k in (
        "threshold", "elong_thresh", "min_streak_px",
    ) if k in body}

    log.info(
        "Pipeline run: %d frame(s)  exposure=%.1fs  gap=%.1fs  fov=%.1f°",
        len(frames), exposure_time, gap_time, sensor_fov,
    )

    t0 = time.perf_counter()

    # ── Process all frames concurrently ──────────────────────────────────────
    n_workers = min(_MAX_FRAME_WORKERS, len(frames))
    frame_results: List[dict] = [None] * len(frames)  # type: ignore[list-item]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_process_frame, frame, idx, extract_params, fov_deg): idx
            for idx, frame in enumerate(frames)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                frame_results[idx] = fut.result()
            except Exception as exc:
                log.exception("Frame %d raised an unhandled exception: %s", idx, exc)
                f = frames[idx]
                frame_results[idx] = {
                    "frame_id":       f.get("frame_id", f"frame_{idx:04d}"),
                    "idx":            idx,
                    "t_start_s":      f.get("t_start_s"),
                    "t_end_s":        f.get("t_end_s"),
                    "n_stars":        0,
                    "n_satellites":   0,
                    "solved":         False,
                    "n_observations": 0,
                    "observations":   [],
                    "error":          str(exc),
                }

    # ── Collect observations across frames for the associator ─────────────────
    # The associator expects frames in chronological order.
    frame_results.sort(key=lambda r: r["idx"])

    obs_per_frame: List[List[dict]] = [
        r["observations"] for r in frame_results
    ]
    n_solved = sum(1 for r in frame_results if r["solved"])
    n_obs    = sum(r["n_observations"] for r in frame_results)

    log.info(
        "Frame processing done: %d/%d solved  %d total observations",
        n_solved, len(frames), n_obs,
    )

    # ── Stage 5: associate tracks across all frames ───────────────────────────
    tracks: List[dict] = []
    assoc_error: Optional[str] = None

    if any(obs_per_frame):
        assoc_ok, assoc_data = _post(
            f"{_ASSOCIATOR_URL}/associate",
            {
                "frames":        obs_per_frame,
                "exposure_time": exposure_time,
                "gap_time":      gap_time,
                "sensor_fov":    sensor_fov,
            },
            _ASSOC_TIMEOUT,
        )
        if assoc_ok:
            tracks = assoc_data.get("tracks", [])
            log.info("%d track(s) found", len(tracks))
        else:
            assoc_error = assoc_data.get("error", "unknown association error")
            log.warning("Association failed: %s", assoc_error)
    else:
        log.warning("No observations across any frame — skipping association")

    # ── Stage 6: correlate tracks against the catalog (opt-in) ────────────────
    correlation_results: List[dict] = []
    if correlate and tracks:
        correlation_results = _correlate_tracks(tracks, sensor_id, site_position_km)
        n_promoted = sum(1 for r in correlation_results if r["result"] and r["result"].get("promoted"))
        log.info("Correlation done: %d track(s) processed, %d promoted", len(correlation_results), n_promoted)
    elif correlate:
        log.info("Correlation requested but no tracks to correlate")

    elapsed = time.perf_counter() - t0

    # ── Build response ────────────────────────────────────────────────────────
    resp: dict = {
        "n_frames":       len(frames),
        "n_solved":       n_solved,
        "n_observations": n_obs,
        "solve_rate":     n_solved / len(frames) if frames else 0.0,
        "elapsed_s":      elapsed,
        "n_tracks":       len(tracks),
        "tracks":         tracks,
        "frame_results": [
            {
                "frame_id":       r["frame_id"],
                "t_start_s":      r["t_start_s"],
                "t_end_s":        r["t_end_s"],
                "n_stars":        r["n_stars"],
                "n_satellites":   r["n_satellites"],
                "solved":         r["solved"],
                "n_observations": r["n_observations"],
                "error":          r["error"],
            }
            for r in frame_results
        ],
        "correlation_results": correlation_results,
    }
    if assoc_error:
        resp["association_error"] = assoc_error

    return jsonify(resp), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host=host, port=port, debug=False)

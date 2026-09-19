"""
api.py — Flask REST API for the track-to-track orbit correlator.

Stateless: this service holds no data of its own. Every request reads and
writes against the catalog-store service (CATALOG_STORE_URL). See
correlate.py's module docstring for the generation/scoring/pruning/
promotion algorithm.

Endpoints
---------
GET  /health       Liveness check + reachability of catalog-store.
POST /correlate     Correlate one track against the catalog.

POST /correlate body
----------------------
Either an already-ingested track:
{"track_id": 42}

or a raw attributable, which is inserted into catalog-store first:
{
  "sensor_id": "rubin", "t_start": 1737331200.0, "t_end": 1737331205.0,
  "ra": 150.234, "dec": -3.112, "ra_dot": 0.0021, "dec_dot": -0.0004,
  "covariance": [[...], [...], [...], [...]],
  "site_position_km": [x, y, z]   // optional; needed for IOD-gated
                                  // promotion, see correlate.py
}

Optional tuning overrides (all have defaults, see correlate.py):
  "lookback_s", "chi2_gate", "k_best", "sigma_accel_deg_s2",
  "promote_min_tracks", "promote_chi2_max",
  "iod_sigma_ra_deg", "iod_sigma_dec_deg", "iod_max_rms_deg"

-> {"track_id": 42, "action": "branched"|"new_uct", "branches": [...],
    "promoted": object_id | null, "iod": {"status": ...} | absent}

Environment variables
-----------------------
CATALOG_STORE_URL   Base URL of the catalog-store service
                     (default: http://catalog-store:5004).
HOST / PORT          Bind host/port (default 0.0.0.0 / 5005).

Authors: Peter Thomas
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from correlator import catalog_client, correlate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
_catalog_store_url = os.environ.get("CATALOG_STORE_URL", "http://catalog-store:5004").rstrip("/")

_RAW_TRACK_FIELDS = ("sensor_id", "t_start", "t_end", "ra", "dec", "ra_dot", "dec_dot", "covariance")


def _opt(body: dict, key: str, cast, default=None):
    v = body.get(key)
    return cast(v) if v is not None else default


@app.get("/health")
def health():
    ok, data = catalog_client.list_objects(_catalog_store_url, timeout=5.0)
    if not ok:
        return jsonify({"status": "degraded", "catalog_store_url": _catalog_store_url,
                         "error": data}), 503
    return jsonify({"status": "ok", "catalog_store_url": _catalog_store_url}), 200


@app.post("/correlate")
def correlate_endpoint():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    if "track_id" in body:
        ok, track = catalog_client.get_track(_catalog_store_url, int(body["track_id"]))
        if not ok:
            return jsonify({"error": f"could not fetch track {body['track_id']}: {track}"}), 404
    else:
        missing = [k for k in _RAW_TRACK_FIELDS if k not in body]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        track_body = {k: body[k] for k in _RAW_TRACK_FIELDS}
        if "site_position_km" in body:
            track_body["site_position_km"] = body["site_position_km"]
        ok, inserted = catalog_client.insert_track(_catalog_store_url, track_body)
        if not ok:
            return jsonify({"error": f"could not ingest track: {inserted}"}), 502
        ok, track = catalog_client.get_track(_catalog_store_url, inserted["id"])
        if not ok:
            return jsonify({"error": f"track ingested but could not be re-fetched: {track}"}), 502

    kwargs = {}
    for key, cast in (
        ("lookback_s", float), ("chi2_gate", float), ("k_best", int),
        ("sigma_accel_deg_s2", float), ("promote_min_tracks", int), ("promote_chi2_max", float),
        ("iod_sigma_ra_deg", float), ("iod_sigma_dec_deg", float), ("iod_max_rms_deg", float),
    ):
        if key in body:
            kwargs[key] = cast(body[key])

    try:
        result = correlate.correlate_new_track(_catalog_store_url, track, **kwargs)
    except correlate.CorrelationError as exc:
        log.exception("Correlation failed for track %s", track.get("id"))
        return jsonify({"error": str(exc)}), 502

    return jsonify(result), 200


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5005"))
    app.run(host=host, port=port, debug=False)

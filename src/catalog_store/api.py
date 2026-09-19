"""
api.py — Flask REST API for the RSO catalog store.

This is the only service that writes to the catalog database. It is
deliberately dumb storage: track-to-track generation, scoring, pruning and
promotion *decisions* belong to the correlator service, which calls this
API (including, when a hypothesis has enough tracks with known observer
positions, a real orbit fit via src/iod/double_r_lambert.py -- this
service just stores whatever state/covariance the correlator hands it,
4-D attributable or 6-D orbital, and never runs IOD itself). The one
exception is POST /objects/<id>/promote, which must
flip one object to 'confirmed' and invalidate every conflicting candidate
in a single transaction -- that can't safely be decomposed into several
calls from a stateless caller without a race window, so it lives here (see
db.py's module docstring for the full reasoning).

Data model summary (see db.py for full schema)
------------------------------------------------
tracks   -- raw ingested attributables (ra, dec, ra_dot, dec_dot, covariance)
objects  -- hypotheses and confirmed catalog entries, distinguished by
            `status` ('candidate' | 'confirmed' | 'invalidated'). An object
            with status='candidate' is an uncorrelated track (UCT) by
            definition -- GET /objects?status=candidate is the UCT list.

Endpoints
---------
GET  /health                     Liveness + row counts.
POST /tracks                     Ingest a raw attributable.
GET  /tracks/<id>                Fetch one track.
GET  /tracks                     List tracks (optional sensor_id, since).
POST /objects                    Create a hypothesis from one or more tracks.
GET  /objects/<id>               Fetch one object with its track/parent ids.
GET  /objects                    List objects incl. track_ids (optional status, updated_before/after).
PATCH /objects/<id>              Update mutable fields (not status='confirmed').
POST /objects/<id>/tracks        Link additional tracks to an existing hypothesis.
POST /objects/<id>/promote       Confirm a hypothesis; invalidate conflicts.

POST /tracks body
------------------
{
  "sensor_id": "rubin",                     // required
  "t_start":   1737331200.0,                // required, unix epoch seconds
  "t_end":     1737331205.0,                // required
  "ra":        150.234,                     // required, degrees
  "dec":       -3.112,                      // required, degrees
  "ra_dot":    0.0021,                      // required, deg/s
  "dec_dot":  -0.0004,                      // required, deg/s
  "covariance": [[...], [...], [...], [...]], // required, 4x4
  "site_position_km": [x, y, z]             // optional, observer inertial position;
                                             // required for real IOD (see db.py)
}
-> {"id": 42}

POST /objects body
--------------------
{
  "track_ids":  [42, 43],           // required, at least one
  "status":     "candidate",        // optional, default "candidate"
  "state_vector":     [...],        // optional, 4-D attributable or 6-D orbital state
  "covariance":       [[...], ...], // optional
  "epoch":            1737331202.5, // optional, unix epoch seconds
  "figure_of_merit":  0.42,         // optional
  "parent_ids":       [17, 19]      // optional, hypotheses merged to form this one
}
-> {"id": 7}

PATCH /objects/<id> body
--------------------------
Same optional fields as POST /objects (minus track_ids/parent_ids), plus:
{
  "clear_covariance": true   // optional -- explicitly null out a stale
                              // covariance (e.g. when a state_vector
                              // update changes dimensionality)
}

POST /objects/<id>/promote
----------------------------
-> {"promoted": 7, "invalidated": [11, 14]}

Environment variables
----------------------
DB_PATH   Path to the SQLite database file (default: /data/catalog.db).
HOST      Bind host (default: 0.0.0.0).
PORT      Bind port (default: 5004).

Authors: Peter Thomas
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from flask import Flask, g, jsonify, request

from catalog_store import db

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
_db_path = os.environ.get("DB_PATH", "/data/catalog.db")


def _get_conn():
    if "conn" not in g:
        g.conn = db.connect(_db_path)
    return g.conn


@app.teardown_appcontext
def _close_conn(exception=None):
    conn = g.pop("conn", None)
    if conn is not None:
        conn.close()


def _opt(body: dict, key: str, cast, default=None):
    v = body.get(key)
    return cast(v) if v is not None else default


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    conn = _get_conn()
    try:
        n_tracks = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]
        n_objects = conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"]
        n_candidates = conn.execute(
            "SELECT COUNT(*) AS n FROM objects WHERE status = 'candidate'"
        ).fetchone()["n"]
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500

    return jsonify({
        "status": "ok",
        "db_path": _db_path,
        "n_tracks": n_tracks,
        "n_objects": n_objects,
        "n_candidates": n_candidates,
    }), 200


# ── Tracks ───────────────────────────────────────────────────────────────────

@app.post("/tracks")
def post_track():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ("sensor_id", "t_start", "t_end", "ra", "dec", "ra_dot", "dec_dot", "covariance")
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        track_id = db.insert_track(
            _get_conn(),
            sensor_id=str(body["sensor_id"]),
            t_start=float(body["t_start"]),
            t_end=float(body["t_end"]),
            ra=float(body["ra"]),
            dec=float(body["dec"]),
            ra_dot=float(body["ra_dot"]),
            dec_dot=float(body["dec_dot"]),
            covariance=body["covariance"],
            site_position_km=body.get("site_position_km"),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid track: {exc}"}), 400

    return jsonify({"id": track_id}), 201


@app.get("/tracks/<int:track_id>")
def get_track(track_id: int):
    track = db.get_track(_get_conn(), track_id)
    if track is None:
        return jsonify({"error": f"track {track_id} not found"}), 404
    return jsonify(track), 200


@app.get("/tracks")
def list_tracks():
    sensor_id = request.args.get("sensor_id")
    since = request.args.get("since", type=float)
    tracks = db.list_tracks(_get_conn(), sensor_id=sensor_id, since=since)
    return jsonify({"n": len(tracks), "tracks": tracks}), 200


# ── Objects ──────────────────────────────────────────────────────────────────

@app.post("/objects")
def post_object():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    track_ids = body.get("track_ids")
    if not isinstance(track_ids, list) or not track_ids:
        return jsonify({"error": "'track_ids' must be a non-empty list"}), 400

    try:
        object_id = db.create_object(
            _get_conn(),
            track_ids=[int(t) for t in track_ids],
            status=_opt(body, "status", str, "candidate"),
            state_vector=body.get("state_vector"),
            covariance=body.get("covariance"),
            epoch=_opt(body, "epoch", float),
            figure_of_merit=_opt(body, "figure_of_merit", float),
            parent_ids=body.get("parent_ids"),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid object: {exc}"}), 400

    return jsonify({"id": object_id}), 201


@app.get("/objects/<int:object_id>")
def get_object(object_id: int):
    obj = db.get_object(_get_conn(), object_id)
    if obj is None:
        return jsonify({"error": f"object {object_id} not found"}), 404
    return jsonify(obj), 200


@app.get("/objects")
def list_objects():
    status = request.args.get("status")
    updated_before = request.args.get("updated_before", type=float)
    updated_after = request.args.get("updated_after", type=float)

    try:
        objects = db.list_objects(
            _get_conn(), status=status, updated_before=updated_before, updated_after=updated_after,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"n": len(objects), "objects": objects}), 200


@app.patch("/objects/<int:object_id>")
def patch_object(object_id: int):
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        obj = db.update_object(
            _get_conn(),
            object_id,
            status=body.get("status"),
            state_vector=body.get("state_vector"),
            covariance=body.get("covariance"),
            epoch=_opt(body, "epoch", float),
            figure_of_merit=_opt(body, "figure_of_merit", float),
            clear_covariance=_opt(body, "clear_covariance", bool, False),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if obj is None:
        return jsonify({"error": f"object {object_id} not found"}), 404
    return jsonify(obj), 200


@app.post("/objects/<int:object_id>/tracks")
def post_object_tracks(object_id: int):
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    track_ids = body.get("track_ids")
    if not isinstance(track_ids, list) or not track_ids:
        return jsonify({"error": "'track_ids' must be a non-empty list"}), 400

    if db.get_object(_get_conn(), object_id) is None:
        return jsonify({"error": f"object {object_id} not found"}), 404

    db.add_tracks_to_object(_get_conn(), object_id, [int(t) for t in track_ids])
    return jsonify(db.get_object(_get_conn(), object_id)), 200


@app.post("/objects/<int:object_id>/promote")
def post_object_promote(object_id: int):
    try:
        result = db.promote_object(_get_conn(), object_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409

    log.info("Promoted object %d (invalidated %d conflicting hypotheses)",
              object_id, len(result["invalidated"]))
    return jsonify(result), 200


# ── Entry point ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Application factory — ensures the schema exists and returns the Flask app."""
    db.init_db(_db_path)
    log.info("Catalog store using database at %s", _db_path)
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5004"))
    db.init_db(_db_path)
    app.run(host=host, port=port, debug=False)

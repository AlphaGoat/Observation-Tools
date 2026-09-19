"""
db.py — SQLite-backed storage layer for the RSO catalog store.

This module owns the schema and all direct database access. It is the only
place in the service that touches the sqlite3 connection; the Flask API
(api.py) calls these functions and never issues SQL itself.

Data model
----------
tracks
    Raw ingested attributables (ra, dec, ra_dot, dec_dot + covariance) from
    the associator's MHT output. Immutable once written.

    site_position_km is the observer's inertial-frame position vector at
    the track's epoch, nullable -- nothing upstream (associator/pipeline)
    populates it yet, so it's optional at ingestion. It's required for
    real angles-only IOD (see src/iod/double_r_lambert.py): the correlator
    only attempts an IOD-based promotion when every constituent track of a
    hypothesis has it set, and falls back to the linear/attributable
    promotion path otherwise. Must be in the same inertial frame as the
    ra/dec measurements themselves (e.g. GCRS) -- this is a caller
    contract, not something this table validates.

objects
    Both in-progress hypotheses and confirmed catalog entries live here,
    distinguished by `status`, not split into separate tables -- a promoted
    hypothesis just flips a flag rather than migrating rows. An object with
    status='candidate' and no confirmed orbit is, by definition, an
    uncorrelated track (UCT): every newly-generated hypothesis starts this
    way (see Pastor 2022, Algorithm 4), and it stays a UCT until it either
    grows into a promoted object or gets folded into an existing one via
    track-to-orbit correlation. There is deliberately no separate "UCT"
    table -- `SELECT * FROM objects WHERE status = 'candidate'` is the UCT
    list.

    For hypotheses with too few tracks for a real orbit determination
    (fewer than an IOD method's minimum observation count), state_vector
    holds the raw 4-D attributable (ra, dec, ra_dot, dec_dot) carried over
    from the associator instead of a 6-D orbital state -- state_dim records
    which case applies.

object_tracks
    Many-to-many junction between objects and tracks. A single track can
    belong to multiple competing hypotheses simultaneously until pruning or
    promotion resolves the ambiguity, so this cannot be a simple foreign key
    on `tracks`.

hypothesis_lineage
    Parent/child links between objects, recording which hypotheses were
    merged to produce a new one (Pastor's F(.) traceability). Used to
    reconstruct why a track ended up (or didn't end up) in a promoted
    object.

Concurrency
-----------
SQLite in WAL mode with a busy_timeout allows multiple writers without
application-level locking; writers block-and-retry at the SQLite layer
instead of erroring out. The one operation that must be atomic beyond a
single-row write is `promote_object`, which flips one object to 'confirmed'
and invalidates every other candidate sharing a track with it in a single
transaction (Pastor's Algorithm 7) -- this is why that logic lives here in
the storage layer rather than in a future correlator service issuing several
separate HTTP calls.

Authors: Peter Thomas
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

VALID_STATUSES = ("candidate", "confirmed", "invalidated")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_id          TEXT    NOT NULL,
    t_start            REAL    NOT NULL,
    t_end              REAL    NOT NULL,
    ra                 REAL    NOT NULL,
    dec                REAL    NOT NULL,
    ra_dot             REAL    NOT NULL,
    dec_dot            REAL    NOT NULL,
    covariance         TEXT    NOT NULL,
    site_position_km   TEXT,
    created_at         REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    status            TEXT    NOT NULL CHECK (status IN ('candidate', 'confirmed', 'invalidated')),
    n_tracks          INTEGER NOT NULL,
    epoch             REAL,
    state_dim         INTEGER,
    state_vector      TEXT,
    covariance        TEXT,
    figure_of_merit   REAL,
    created_at        REAL    NOT NULL,
    updated_at        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS object_tracks (
    object_id  INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    track_id   INTEGER NOT NULL REFERENCES tracks(id)  ON DELETE CASCADE,
    PRIMARY KEY (object_id, track_id)
);

CREATE TABLE IF NOT EXISTS hypothesis_lineage (
    child_object_id   INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    parent_object_id  INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    PRIMARY KEY (child_object_id, parent_object_id)
);

CREATE INDEX IF NOT EXISTS idx_object_tracks_track  ON object_tracks(track_id);
CREATE INDEX IF NOT EXISTS idx_object_tracks_object ON object_tracks(object_id);
CREATE INDEX IF NOT EXISTS idx_objects_status        ON objects(status);
"""


# ── Connection management ───────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    """
    Open a connection with WAL journaling and foreign keys enabled.

    Safe to call once per request -- SQLite handles multiple connections
    (even across processes) to the same file under WAL mode.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: str) -> None:
    """Create the schema if it does not already exist."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ── JSON helpers ─────────────────────────────────────────────────────────────

def _dump(value: Optional[Sequence]) -> Optional[str]:
    return None if value is None else json.dumps(value)


def _load(value: Optional[str]) -> Optional[Any]:
    return None if value is None else json.loads(value)


# ── Row -> dict helpers ──────────────────────────────────────────────────────

def _track_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id":               row["id"],
        "sensor_id":        row["sensor_id"],
        "t_start":          row["t_start"],
        "t_end":            row["t_end"],
        "ra":               row["ra"],
        "dec":              row["dec"],
        "ra_dot":           row["ra_dot"],
        "dec_dot":          row["dec_dot"],
        "covariance":       _load(row["covariance"]),
        "site_position_km": _load(row["site_position_km"]),
        "created_at":       row["created_at"],
    }


def _object_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id":               row["id"],
        "status":           row["status"],
        "n_tracks":         row["n_tracks"],
        "epoch":            row["epoch"],
        "state_dim":        row["state_dim"],
        "state_vector":     _load(row["state_vector"]),
        "covariance":       _load(row["covariance"]),
        "figure_of_merit":  row["figure_of_merit"],
        "created_at":       row["created_at"],
        "updated_at":       row["updated_at"],
    }


# ── Tracks ───────────────────────────────────────────────────────────────────

def insert_track(
    conn: sqlite3.Connection,
    sensor_id: str,
    t_start: float,
    t_end: float,
    ra: float,
    dec: float,
    ra_dot: float,
    dec_dot: float,
    covariance: Sequence[Sequence[float]],
    site_position_km: Optional[Sequence[float]] = None,
) -> int:
    """Insert a raw attributable. Returns the new track id."""
    now = time.time()
    cur = conn.execute(
        """
        INSERT INTO tracks
            (sensor_id, t_start, t_end, ra, dec, ra_dot, dec_dot, covariance, site_position_km, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sensor_id, t_start, t_end, ra, dec, ra_dot, dec_dot, _dump(covariance), _dump(site_position_km), now),
    )
    conn.commit()
    return cur.lastrowid


def get_track(conn: sqlite3.Connection, track_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
    return None if row is None else _track_row_to_dict(row)


def list_tracks(
    conn: sqlite3.Connection,
    sensor_id: Optional[str] = None,
    since: Optional[float] = None,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM tracks WHERE 1 = 1"
    params: List[Any] = []
    if sensor_id is not None:
        query += " AND sensor_id = ?"
        params.append(sensor_id)
    if since is not None:
        query += " AND created_at >= ?"
        params.append(since)
    query += " ORDER BY created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [_track_row_to_dict(r) for r in rows]


# ── Objects ──────────────────────────────────────────────────────────────────

def create_object(
    conn: sqlite3.Connection,
    track_ids: Sequence[int],
    status: str = "candidate",
    state_vector: Optional[Sequence[float]] = None,
    covariance: Optional[Sequence[Sequence[float]]] = None,
    epoch: Optional[float] = None,
    figure_of_merit: Optional[float] = None,
    parent_ids: Optional[Sequence[int]] = None,
) -> int:
    """
    Create a hypothesis (or a confirmed object, if called directly with
    status='confirmed' -- normally reached via promote_object instead).

    Links every track in `track_ids` via object_tracks, and every id in
    `parent_ids` via hypothesis_lineage, in the same transaction as the
    object insert.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}, must be one of {VALID_STATUSES}")
    if not track_ids:
        raise ValueError("track_ids must contain at least one track")

    state_dim = None if state_vector is None else len(state_vector)
    now = time.time()

    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            INSERT INTO objects
                (status, n_tracks, epoch, state_dim, state_vector, covariance,
                 figure_of_merit, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                status, len(track_ids), epoch, state_dim,
                _dump(state_vector), _dump(covariance), figure_of_merit,
                now, now,
            ),
        )
        object_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO object_tracks (object_id, track_id) VALUES (?, ?)",
            [(object_id, tid) for tid in track_ids],
        )
        if parent_ids:
            conn.executemany(
                "INSERT OR IGNORE INTO hypothesis_lineage (child_object_id, parent_object_id) VALUES (?, ?)",
                [(object_id, pid) for pid in parent_ids],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return object_id


def get_object(conn: sqlite3.Connection, object_id: int) -> Optional[Dict[str, Any]]:
    """Fetch an object with its linked track ids and parent hypothesis ids."""
    row = conn.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
    if row is None:
        return None

    obj = _object_row_to_dict(row)
    obj["track_ids"] = [
        r["track_id"] for r in
        conn.execute("SELECT track_id FROM object_tracks WHERE object_id = ?", (object_id,)).fetchall()
    ]
    obj["parent_ids"] = [
        r["parent_object_id"] for r in
        conn.execute(
            "SELECT parent_object_id FROM hypothesis_lineage WHERE child_object_id = ?", (object_id,)
        ).fetchall()
    ]
    return obj


def list_objects(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    updated_before: Optional[float] = None,
    updated_after: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    List objects, optionally filtered by status and/or last-update time.

    `status='candidate'` with no other filters is the UCT list. Combined
    with `updated_before`, it's the "stale UCTs, candidates for a follow-up
    observation" query a scheduler would run.

    Each object includes `track_ids` (batched in one extra query, not N+1)
    so a caller -- e.g. the correlator deciding which loose tracks are
    already claimed by an in-progress hypothesis -- doesn't need a separate
    get_object() round trip per row. `parent_ids` is not included here;
    fetch a single object via get_object() for full lineage detail.
    """
    query = "SELECT * FROM objects WHERE 1 = 1"
    params: List[Any] = []
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}, must be one of {VALID_STATUSES}")
        query += " AND status = ?"
        params.append(status)
    if updated_before is not None:
        query += " AND updated_at < ?"
        params.append(updated_before)
    if updated_after is not None:
        query += " AND updated_at >= ?"
        params.append(updated_after)
    query += " ORDER BY updated_at DESC"
    rows = conn.execute(query, params).fetchall()
    objects = [_object_row_to_dict(r) for r in rows]
    if not objects:
        return objects

    ids = [o["id"] for o in objects]
    placeholders = ",".join("?" * len(ids))
    track_rows = conn.execute(
        f"SELECT object_id, track_id FROM object_tracks WHERE object_id IN ({placeholders})",
        ids,
    ).fetchall()
    tracks_by_object: Dict[int, List[int]] = {i: [] for i in ids}
    for r in track_rows:
        tracks_by_object[r["object_id"]].append(r["track_id"])
    for o in objects:
        o["track_ids"] = tracks_by_object[o["id"]]
    return objects


def update_object(
    conn: sqlite3.Connection,
    object_id: int,
    status: Optional[str] = None,
    state_vector: Optional[Sequence[float]] = None,
    covariance: Optional[Sequence[Sequence[float]]] = None,
    epoch: Optional[float] = None,
    figure_of_merit: Optional[float] = None,
    clear_covariance: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Update mutable fields on a hypothesis (e.g. after re-scoring with a new
    track, or growing its track set).

    Refuses to set status='confirmed' here -- that transition must go
    through promote_object() so the conflicting-hypothesis invalidation
    always happens atomically alongside it.

    covariance=None (the default) leaves the stored covariance untouched,
    matching every other optional field here -- but that means there is
    normally no way to *remove* a stale one. clear_covariance=True does
    that explicitly: e.g. when the correlator upgrades a hypothesis from a
    4-D attributable state to a real 6-D IOD state vector, the old 4x4
    covariance is dimensionally wrong for the new state and must be
    cleared, not silently left in place.
    """
    if status is not None and status == "confirmed":
        raise ValueError("use promote_object() to confirm a hypothesis")
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}, must be one of {VALID_STATUSES}")

    fields, params = [], []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if state_vector is not None:
        fields.append("state_vector = ?")
        params.append(_dump(state_vector))
        fields.append("state_dim = ?")
        params.append(len(state_vector))
    if clear_covariance:
        fields.append("covariance = ?")
        params.append(None)
    elif covariance is not None:
        fields.append("covariance = ?")
        params.append(_dump(covariance))
    if epoch is not None:
        fields.append("epoch = ?")
        params.append(epoch)
    if figure_of_merit is not None:
        fields.append("figure_of_merit = ?")
        params.append(figure_of_merit)

    if not fields:
        return get_object(conn, object_id)

    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(object_id)

    conn.execute(f"UPDATE objects SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return get_object(conn, object_id)


def add_tracks_to_object(conn: sqlite3.Connection, object_id: int, track_ids: Sequence[int]) -> None:
    """Link additional tracks to an existing hypothesis and bump n_tracks."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT OR IGNORE INTO object_tracks (object_id, track_id) VALUES (?, ?)",
            [(object_id, tid) for tid in track_ids],
        )
        n_tracks = conn.execute(
            "SELECT COUNT(*) AS n FROM object_tracks WHERE object_id = ?", (object_id,)
        ).fetchone()["n"]
        conn.execute(
            "UPDATE objects SET n_tracks = ?, updated_at = ? WHERE id = ?",
            (n_tracks, time.time(), object_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def promote_object(conn: sqlite3.Connection, object_id: int) -> Dict[str, Any]:
    """
    Confirm a hypothesis as a catalog object and invalidate every other
    candidate hypothesis that shares a track with it (Pastor 2022,
    Algorithm 7) -- a single track cannot belong to two different objects.

    Runs as one transaction so no other request can promote a conflicting
    hypothesis in the race window between reading and writing.

    Raises ValueError if the object does not exist or is not a candidate
    (guards against double-promotion and promoting an already-invalidated
    hypothesis).
    """
    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute("SELECT status FROM objects WHERE id = ?", (object_id,)).fetchone()
        if row is None:
            raise ValueError(f"object {object_id} does not exist")
        if row["status"] != "candidate":
            raise ValueError(f"object {object_id} has status {row['status']!r}, expected 'candidate'")

        now = time.time()
        conn.execute(
            "UPDATE objects SET status = 'confirmed', updated_at = ? WHERE id = ?",
            (now, object_id),
        )

        conflicting = conn.execute(
            """
            SELECT DISTINCT o.id
            FROM objects o
            JOIN object_tracks ot ON ot.object_id = o.id
            WHERE o.status = 'candidate'
              AND o.id != ?
              AND ot.track_id IN (SELECT track_id FROM object_tracks WHERE object_id = ?)
            """,
            (object_id, object_id),
        ).fetchall()
        invalidated_ids = [r["id"] for r in conflicting]

        if invalidated_ids:
            conn.executemany(
                "UPDATE objects SET status = 'invalidated', updated_at = ? WHERE id = ?",
                [(now, oid) for oid in invalidated_ids],
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {"promoted": object_id, "invalidated": invalidated_ids}

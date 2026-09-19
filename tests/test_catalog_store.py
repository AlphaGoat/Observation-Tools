"""
Tests for the catalog store: the SQLite DAO layer (db.py) directly, plus a
smoke test of the Flask API (api.py) end to end.

Author: Peter Thomas
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catalog_store import db


# ── db.py: fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def conn(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    db.init_db(db_path)
    c = db.connect(db_path)
    yield c
    c.close()


def _covar4():
    return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def _insert_track(conn, ra=150.0, dec=-3.0, sensor_id="rubin"):
    return db.insert_track(
        conn, sensor_id=sensor_id, t_start=0.0, t_end=1.0,
        ra=ra, dec=dec, ra_dot=0.001, dec_dot=-0.0005, covariance=_covar4(),
    )


# ── Tracks ───────────────────────────────────────────────────────────────────

def test_insert_and_get_track(conn):
    track_id = _insert_track(conn)
    track = db.get_track(conn, track_id)

    assert track is not None
    assert track["sensor_id"] == "rubin"
    assert track["ra"] == 150.0
    assert track["covariance"] == _covar4()


def test_get_missing_track_returns_none(conn):
    assert db.get_track(conn, 999) is None


def test_insert_track_without_site_position_defaults_to_none(conn):
    track_id = _insert_track(conn)
    assert db.get_track(conn, track_id)["site_position_km"] is None


def test_insert_and_get_track_with_site_position(conn):
    track_id = db.insert_track(
        conn, sensor_id="rubin", t_start=0.0, t_end=1.0,
        ra=150.0, dec=-3.0, ra_dot=0.001, dec_dot=-0.0005, covariance=_covar4(),
        site_position_km=[6378.0, 0.0, 0.0],
    )
    assert db.get_track(conn, track_id)["site_position_km"] == [6378.0, 0.0, 0.0]


def test_list_tracks_filters_by_sensor(conn):
    _insert_track(conn, sensor_id="rubin")
    _insert_track(conn, sensor_id="haleakala")
    _insert_track(conn, sensor_id="rubin")

    rubin_tracks = db.list_tracks(conn, sensor_id="rubin")
    assert len(rubin_tracks) == 2
    assert all(t["sensor_id"] == "rubin" for t in rubin_tracks)

    all_tracks = db.list_tracks(conn)
    assert len(all_tracks) == 3


def test_list_tracks_filters_by_since(conn):
    _insert_track(conn)
    cutoff = time.time()
    _insert_track(conn)

    recent = db.list_tracks(conn, since=cutoff)
    assert len(recent) == 1


# ── Objects: create / read ────────────────────────────────────────────────────

def test_create_object_defaults_to_candidate(conn):
    t1 = _insert_track(conn)
    object_id = db.create_object(conn, track_ids=[t1])

    obj = db.get_object(conn, object_id)
    assert obj["status"] == "candidate"
    assert obj["n_tracks"] == 1
    assert obj["track_ids"] == [t1]
    assert obj["state_vector"] is None
    assert obj["state_dim"] is None


def test_create_object_requires_at_least_one_track(conn):
    with pytest.raises(ValueError):
        db.create_object(conn, track_ids=[])


def test_create_object_rejects_invalid_status(conn):
    t1 = _insert_track(conn)
    with pytest.raises(ValueError):
        db.create_object(conn, track_ids=[t1], status="bogus")


def test_create_object_with_attributable_state(conn):
    """A low-track-count hypothesis should be able to carry a 4-D attributable
    (no real orbit yet) rather than a 6-D state -- this is how a single-track
    UCT gets a usable prediction basis without running an IOD."""
    t1 = _insert_track(conn)
    object_id = db.create_object(
        conn, track_ids=[t1], state_vector=[150.0, -3.0, 0.001, -0.0005],
        covariance=_covar4(), epoch=1.0,
    )

    obj = db.get_object(conn, object_id)
    assert obj["state_dim"] == 4
    assert obj["state_vector"] == [150.0, -3.0, 0.001, -0.0005]


def test_create_object_records_lineage(conn):
    t1, t2, t3 = (_insert_track(conn) for _ in range(3))
    parent_a = db.create_object(conn, track_ids=[t1])
    parent_b = db.create_object(conn, track_ids=[t2])
    child = db.create_object(conn, track_ids=[t1, t2, t3], parent_ids=[parent_a, parent_b])

    obj = db.get_object(conn, child)
    assert sorted(obj["parent_ids"]) == sorted([parent_a, parent_b])


def test_get_missing_object_returns_none(conn):
    assert db.get_object(conn, 999) is None


# ── Objects: listing / UCT query ──────────────────────────────────────────────

def test_list_objects_status_filter_is_the_uct_query(conn):
    t1, t2 = (_insert_track(conn) for _ in range(2))
    candidate_id = db.create_object(conn, track_ids=[t1])
    confirmed_id = db.create_object(conn, track_ids=[t2], status="confirmed")

    candidates = db.list_objects(conn, status="candidate")
    assert [o["id"] for o in candidates] == [candidate_id]

    confirmed = db.list_objects(conn, status="confirmed")
    assert [o["id"] for o in confirmed] == [confirmed_id]


def test_list_objects_updated_before_finds_stale_candidates(conn):
    t1 = _insert_track(conn)
    stale_id = db.create_object(conn, track_ids=[t1])

    cutoff = time.time() + 1  # stale_id's updated_at is strictly before this
    t2 = _insert_track(conn)
    db.create_object(conn, track_ids=[t2])

    stale = db.list_objects(conn, status="candidate", updated_before=cutoff)
    assert stale_id in [o["id"] for o in stale]


# ── Objects: update ────────────────────────────────────────────────────────────

def test_update_object_rejects_direct_confirmation(conn):
    t1 = _insert_track(conn)
    object_id = db.create_object(conn, track_ids=[t1])
    with pytest.raises(ValueError):
        db.update_object(conn, object_id, status="confirmed")


def test_update_object_sets_figure_of_merit_and_bumps_updated_at(conn):
    t1 = _insert_track(conn)
    object_id = db.create_object(conn, track_ids=[t1])
    before = db.get_object(conn, object_id)["updated_at"]

    time.sleep(0.01)
    after_update = db.update_object(conn, object_id, figure_of_merit=0.42)

    assert after_update["figure_of_merit"] == 0.42
    assert after_update["updated_at"] > before


def test_update_object_clear_covariance_nulls_a_stale_covariance(conn):
    """When the correlator upgrades a hypothesis from a 4-D attributable
    state to a real 6-D IOD state, the old 4x4 covariance is dimensionally
    wrong for the new state and must be explicitly clearable, not just
    silently left in place (covariance=None alone means "leave unchanged")."""
    t1 = _insert_track(conn)
    object_id = db.create_object(
        conn, track_ids=[t1], state_vector=[150.0, -3.0, 0.001, -0.0005], covariance=_covar4(),
    )
    assert db.get_object(conn, object_id)["covariance"] == _covar4()

    updated = db.update_object(conn, object_id, state_vector=[1.0] * 6, clear_covariance=True)

    assert updated["covariance"] is None
    assert updated["state_dim"] == 6


def test_add_tracks_to_object_bumps_n_tracks(conn):
    t1, t2 = (_insert_track(conn) for _ in range(2))
    object_id = db.create_object(conn, track_ids=[t1])

    db.add_tracks_to_object(conn, object_id, [t2])

    obj = db.get_object(conn, object_id)
    assert obj["n_tracks"] == 2
    assert sorted(obj["track_ids"]) == sorted([t1, t2])


# ── Objects: promotion ─────────────────────────────────────────────────────────

def test_promote_object_confirms_status(conn):
    t1 = _insert_track(conn)
    object_id = db.create_object(conn, track_ids=[t1])

    result = db.promote_object(conn, object_id)

    assert result["promoted"] == object_id
    assert result["invalidated"] == []
    assert db.get_object(conn, object_id)["status"] == "confirmed"


def test_promote_object_invalidates_conflicting_candidates(conn):
    """Two competing hypotheses share track t2 (e.g. an ambiguous association
    during catalog build-up). Promoting one must invalidate the other,
    per Pastor 2022 Algorithm 7."""
    t1, t2, t3, t4 = (_insert_track(conn) for _ in range(4))
    winner = db.create_object(conn, track_ids=[t1, t2])
    loser = db.create_object(conn, track_ids=[t2, t3])
    unrelated = db.create_object(conn, track_ids=[t4])

    result = db.promote_object(conn, winner)

    assert result["invalidated"] == [loser]
    assert db.get_object(conn, winner)["status"] == "confirmed"
    assert db.get_object(conn, loser)["status"] == "invalidated"
    assert db.get_object(conn, unrelated)["status"] == "candidate"


def test_promote_object_does_not_touch_already_confirmed_objects(conn):
    """An already-confirmed object sharing no interest in this promotion
    should never be flipped to invalidated even if it happens to reference
    an overlapping track (shouldn't happen in practice, but the query only
    ever targets status='candidate')."""
    t1, t2 = (_insert_track(conn) for _ in range(2))
    already_confirmed = db.create_object(conn, track_ids=[t1], status="confirmed")
    new_winner = db.create_object(conn, track_ids=[t1, t2])

    db.promote_object(conn, new_winner)

    assert db.get_object(conn, already_confirmed)["status"] == "confirmed"


def test_promote_object_rejects_missing_object(conn):
    with pytest.raises(ValueError):
        db.promote_object(conn, 999)


def test_promote_object_rejects_double_promotion(conn):
    t1 = _insert_track(conn)
    object_id = db.create_object(conn, track_ids=[t1])
    db.promote_object(conn, object_id)

    with pytest.raises(ValueError):
        db.promote_object(conn, object_id)


# ── api.py: end-to-end smoke test ─────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "catalog.db"))
    # api.py reads DB_PATH into a module-level constant at import time, so it
    # must be imported (or reloaded) only after the env var is set.
    import importlib
    from catalog_store import api as api_module
    importlib.reload(api_module)

    app = api_module.create_app()
    app.testing = True
    with app.test_client() as c:
        yield c


def test_api_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_track_and_object_roundtrip(client):
    track_resp = client.post("/tracks", json={
        "sensor_id": "rubin", "t_start": 0.0, "t_end": 1.0,
        "ra": 150.0, "dec": -3.0, "ra_dot": 0.001, "dec_dot": -0.0005,
        "covariance": _covar4(),
    })
    assert track_resp.status_code == 201
    track_id = track_resp.get_json()["id"]

    object_resp = client.post("/objects", json={"track_ids": [track_id]})
    assert object_resp.status_code == 201
    object_id = object_resp.get_json()["id"]

    uct_resp = client.get("/objects?status=candidate")
    assert uct_resp.status_code == 200
    assert object_id in [o["id"] for o in uct_resp.get_json()["objects"]]

    promote_resp = client.post(f"/objects/{object_id}/promote")
    assert promote_resp.status_code == 200
    assert promote_resp.get_json()["promoted"] == object_id

    uct_resp_after = client.get("/objects?status=candidate")
    assert object_id not in [o["id"] for o in uct_resp_after.get_json()["objects"]]


def test_api_rejects_object_without_track_ids(client):
    resp = client.post("/objects", json={})
    assert resp.status_code == 400


def test_api_promote_missing_object_returns_conflict(client):
    resp = client.post("/objects/999/promote")
    assert resp.status_code == 409


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

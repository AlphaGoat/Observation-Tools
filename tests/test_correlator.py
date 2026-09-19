"""
Tests for the correlator: pure math in linking.py, plus an end-to-end test
that runs the correlator's orchestration (and its Flask API) against a
real, live catalog-store server.

Author: Peter Thomas
"""
from __future__ import annotations

import importlib
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correlator import catalog_client, correlate, linking


# ── linking.py: propagation ───────────────────────────────────────────────────

def test_propagate_zero_dt_is_identity():
    ts = linking.TrackState(epoch=100.0, state=[10.0, 20.0, 0.01, -0.02], covariance=np.eye(4) * 1e-4)
    out = linking.propagate(ts, epoch=100.0, sigma_accel_deg_s2=1e-8)
    assert np.allclose(out.state, ts.state)
    assert np.allclose(out.covariance, ts.covariance)


def test_propagate_advances_position_by_rate_times_dt():
    ts = linking.TrackState(epoch=0.0, state=[10.0, 20.0, 0.01, -0.02], covariance=np.eye(4) * 1e-4)
    out = linking.propagate(ts, epoch=100.0, sigma_accel_deg_s2=0.0)
    assert out.state[0] == pytest.approx(10.0 + 0.01 * 100.0)
    assert out.state[1] == pytest.approx(20.0 - 0.02 * 100.0)
    assert out.state[2] == pytest.approx(0.01)
    assert out.state[3] == pytest.approx(-0.02)


def test_propagate_covariance_grows_with_process_noise():
    ts = linking.TrackState(epoch=0.0, state=[10.0, 20.0, 0.0, 0.0], covariance=np.eye(4) * 1e-6)
    out = linking.propagate(ts, epoch=1000.0, sigma_accel_deg_s2=1e-6)
    assert out.covariance[0, 0] > ts.covariance[0, 0]
    assert out.covariance[1, 1] > ts.covariance[1, 1]


# ── linking.py: chi2 distance ─────────────────────────────────────────────────

def test_chi2_zero_for_identical_states_same_epoch():
    a = linking.TrackState(epoch=0.0, state=[10.0, 20.0, 0.0, 0.0], covariance=np.eye(4) * 1e-4)
    b = linking.TrackState(epoch=0.0, state=[10.0, 20.0, 0.0, 0.0], covariance=np.eye(4) * 1e-4)
    assert linking.chi2_distance(a, b, sigma_accel_deg_s2=0.0) == pytest.approx(0.0, abs=1e-9)


def test_chi2_near_zero_for_a_track_consistent_with_constant_velocity():
    ra_dot, dec_dot = 0.001, -0.0005
    a = linking.TrackState(epoch=0.0, state=[150.0, -3.0, ra_dot, dec_dot], covariance=np.eye(4) * 1e-6)
    dt = 600.0
    b = linking.TrackState(
        epoch=dt, state=[150.0 + ra_dot * dt, -3.0 + dec_dot * dt, ra_dot, dec_dot],
        covariance=np.eye(4) * 1e-6,
    )
    assert linking.chi2_distance(a, b, sigma_accel_deg_s2=0.0) == pytest.approx(0.0, abs=1e-6)


def test_chi2_large_for_incompatible_states():
    a = linking.TrackState(epoch=0.0, state=[150.0, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-6)
    b = linking.TrackState(epoch=10.0, state=[151.0, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-6)
    # 1 deg apart against ~1e-6 deg^2 variance is wildly incompatible -- well
    # past any sane chi2 gate (e.g. the correlator's default of ~18).
    assert linking.chi2_distance(a, b, sigma_accel_deg_s2=1e-8) > 1000.0


def test_chi2_handles_ra_wraparound():
    a = linking.TrackState(epoch=0.0, state=[359.9, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-2)
    b = linking.TrackState(epoch=0.0, state=[0.1, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-2)
    # True angular separation is 0.2 deg (chi2 = 0.2^2 / 0.02 = 2.0), not the
    # ~6.5M an unwrapped 359.8 deg "difference" would give -- proves wraparound
    # is applied rather than testing an arbitrary tolerance.
    assert linking.chi2_distance(a, b, sigma_accel_deg_s2=0.0) == pytest.approx(2.0)


# ── linking.py: ranking ───────────────────────────────────────────────────────

def test_rank_candidates_filters_sorts_and_caps():
    new_state = linking.TrackState(epoch=0.0, state=[150.0, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-6)

    def _at(ra_offset):
        return linking.TrackState(
            epoch=0.0, state=[150.0 + ra_offset, -3.0, 0.0, 0.0], covariance=np.eye(4) * 1e-6,
        )

    object_candidates = [(1, _at(0.0001)), (2, _at(50.0))]   # id 2 is way off, should be gated out
    track_candidates = [(10, _at(0.0002)), (11, _at(0.0003)), (12, _at(0.0004))]

    ranked = linking.rank_candidates(
        new_state, object_candidates, track_candidates,
        chi2_gate=100.0, sigma_accel_deg_s2=1e-8, k_best=2,
    )

    assert len(ranked) == 2
    assert ranked[0].chi2 <= ranked[1].chi2
    assert all(c.id != 2 for c in ranked)


# ── Integration: correlate.py + api.py against a live catalog-store ──────────

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def catalog_store_url(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "catalog.db"))
    from catalog_store import api as catalog_api
    importlib.reload(catalog_api)

    app = catalog_api.create_app()
    port = _free_port()
    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if requests.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.05)

    yield base_url

    server.shutdown()
    thread.join(timeout=5)


def _mk_track(ra, dec, ra_dot, dec_dot, mean_epoch, duration=1.0, sensor_id="test"):
    return {
        "sensor_id": sensor_id,
        "t_start": mean_epoch - duration / 2,
        "t_end": mean_epoch + duration / 2,
        "ra": ra, "dec": dec, "ra_dot": ra_dot, "dec_dot": dec_dot,
        "covariance": [[1e-6, 0, 0, 0], [0, 1e-6, 0, 0], [0, 0, 1e-10, 0], [0, 0, 0, 1e-10]],
    }


def _insert(catalog_store_url, track):
    ok, data = catalog_client.insert_track(catalog_store_url, track)
    assert ok, data
    return {**track, "id": data["id"]}


def test_correlate_creates_uct_when_no_candidates(catalog_store_url):
    track = _insert(catalog_store_url, _mk_track(150.0, -3.0, 0.001, -0.0005, 1_800_000_000.0))

    result = correlate.correlate_new_track(catalog_store_url, track)

    assert result["action"] == "new_uct"
    object_id = result["branches"][0]["object_id"]
    ok, obj = catalog_client.get_object(catalog_store_url, object_id)
    assert ok and obj["status"] == "candidate" and obj["track_ids"] == [track["id"]]


def test_correlate_pairs_compatible_loose_track(catalog_store_url):
    t0, ra0, dec0, ra_dot, dec_dot = 1_800_000_000.0, 150.0, -3.0, 0.001, -0.0005
    track_a = _insert(catalog_store_url, _mk_track(ra0, dec0, ra_dot, dec_dot, t0))

    dt = 600.0
    track_b = _insert(
        catalog_store_url,
        _mk_track(ra0 + ra_dot * dt, dec0 + dec_dot * dt, ra_dot, dec_dot, t0 + dt),
    )

    result = correlate.correlate_new_track(catalog_store_url, track_b)

    assert result["action"] == "branched"
    assert result["branches"][0]["kind"] == "track"
    assert result["branches"][0]["source_id"] == track_a["id"]

    object_id = result["branches"][0]["object_id"]
    ok, obj = catalog_client.get_object(catalog_store_url, object_id)
    assert ok and obj["n_tracks"] == 2 and obj["status"] == "candidate"
    assert sorted(obj["track_ids"]) == sorted([track_a["id"], track_b["id"]])


def test_correlate_promotes_after_third_compatible_track(catalog_store_url):
    t0, ra0, dec0, ra_dot, dec_dot = 1_800_000_000.0, 150.0, -3.0, 0.001, -0.0005

    _insert(catalog_store_url, _mk_track(ra0, dec0, ra_dot, dec_dot, t0))
    track_b = _insert(
        catalog_store_url, _mk_track(ra0 + ra_dot * 600, dec0 + dec_dot * 600, ra_dot, dec_dot, t0 + 600),
    )
    result_b = correlate.correlate_new_track(catalog_store_url, track_b)
    object_id = result_b["branches"][0]["object_id"]

    track_c = _insert(
        catalog_store_url, _mk_track(ra0 + ra_dot * 1200, dec0 + dec_dot * 1200, ra_dot, dec_dot, t0 + 1200),
    )
    result_c = correlate.correlate_new_track(catalog_store_url, track_c)

    assert result_c["promoted"] == object_id
    assert result_c["iod"]["status"] == "unavailable"  # no site_position_km on these tracks
    ok, obj = catalog_client.get_object(catalog_store_url, object_id)
    assert ok and obj["status"] == "confirmed" and obj["n_tracks"] == 3


def test_correlate_rejects_incompatible_track_as_separate_uct(catalog_store_url):
    t0 = 1_800_000_000.0
    _insert(catalog_store_url, _mk_track(150.0, -3.0, 0.001, -0.0005, t0))
    # Wildly different sky position / rate -- should not link to the above.
    other = _insert(catalog_store_url, _mk_track(10.0, 40.0, -0.02, 0.03, t0 + 5.0))

    result = correlate.correlate_new_track(catalog_store_url, other)

    assert result["action"] == "new_uct"


# ── Integration: IOD-gated promotion (src/iod/double_r_lambert.py) ───────────

R_SITE = np.array([6378.0 * np.cos(np.radians(30)) * np.cos(np.radians(45)),
                    6378.0 * np.cos(np.radians(30)) * np.sin(np.radians(45)),
                    6378.0 * np.sin(np.radians(30))])


def _synthetic_orbit_track(r0, v0, t0, mean_epoch, sensor_id="test", rate_dt=1.0):
    """A track built from a real two-body orbit, observed from R_SITE, with
    ra_dot/dec_dot fit from a 1s finite difference -- as a real short
    tracklet fit would produce. Unlike _mk_track, this is real orbital
    motion (slightly non-linear), not synthetic linear motion."""
    from iod import kepler
    from iod.geometry import radec_from_position

    r, _ = kepler.propagate(r0, v0, mean_epoch - t0)
    ra, dec = radec_from_position(r, R_SITE)
    r_next, _ = kepler.propagate(r0, v0, mean_epoch - t0 + rate_dt)
    ra_next, dec_next = radec_from_position(r_next, R_SITE)

    return {
        "sensor_id": sensor_id,
        "t_start": mean_epoch - 0.5, "t_end": mean_epoch + 0.5,
        "ra": ra, "dec": dec,
        "ra_dot": (ra_next - ra) / rate_dt, "dec_dot": (dec_next - dec) / rate_dt,
        "covariance": [[1e-6, 0, 0, 0], [0, 1e-6, 0, 0], [0, 0, 1e-10, 0], [0, 0, 0, 1e-10]],
        "site_position_km": R_SITE.tolist(),
    }


def test_correlate_promotes_with_real_iod_when_site_positions_available(catalog_store_url):
    from iod.elements import coe_to_rv

    a = 42164.0  # near-circular GEO, same shape as test_iod_double_r_lambert.py's own case
    r0, v0 = coe_to_rv(a, 0.001, np.radians(3.0), np.radians(10.0), np.radians(50.0), np.radians(15.0))
    t0 = 1_800_000_000.0
    epochs = [t0, t0 + 1200.0, t0 + 2400.0]  # 20 min spacing: passes both the
                                              # linear generation gate and IOD

    track_a = _insert(catalog_store_url, _synthetic_orbit_track(r0, v0, t0, epochs[0]))
    track_b = _insert(catalog_store_url, _synthetic_orbit_track(r0, v0, t0, epochs[1]))
    result_b = correlate.correlate_new_track(catalog_store_url, track_b)
    object_id = result_b["branches"][0]["object_id"]

    track_c = _insert(catalog_store_url, _synthetic_orbit_track(r0, v0, t0, epochs[2]))
    result_c = correlate.correlate_new_track(catalog_store_url, track_c)

    assert result_c["iod"]["status"] == "promoted"
    assert result_c["promoted"] == object_id

    ok, obj = catalog_client.get_object(catalog_store_url, object_id)
    assert ok and obj["status"] == "confirmed"
    assert obj["state_dim"] == 6
    assert obj["covariance"] is None  # stale 4-D covariance must be cleared, not left mismatched

    r_fit = np.array(obj["state_vector"][:3])
    v_fit = np.array(obj["state_vector"][3:])
    assert np.linalg.norm(r_fit - r0) < 1.0       # km
    assert np.linalg.norm(v_fit - v0) < 0.001      # km/s


def test_attempt_iod_promotion_unavailable_without_site_position(catalog_store_url):
    t1, t2, t3 = (_insert(catalog_store_url, _mk_track(150.0 + 0.01 * i, -3.0, 0.001, -0.0005, 1e9 + 600 * i))
                  for i in range(3))
    ok, created = catalog_client.create_object(catalog_store_url, track_ids=[t1["id"], t2["id"], t3["id"]])
    assert ok

    outcome = correlate._attempt_iod_promotion(catalog_store_url, created["id"], 1 / 3600.0, 1 / 3600.0, 5 / 3600.0, 10.0)

    assert outcome["status"] == "unavailable"
    ok, obj = catalog_client.get_object(catalog_store_url, created["id"])
    assert obj["status"] == "candidate"  # untouched


def test_attempt_iod_promotion_rejected_when_fit_does_not_converge(catalog_store_url, monkeypatch):
    from iod import double_r_lambert

    def _mk_track_with_site(i):
        t = _mk_track(150.0 + 0.01 * i, -3.0, 0.001, -0.0005, 1_800_000_000.0 + 600 * i)
        t["site_position_km"] = R_SITE.tolist()
        return t

    tracks = [_insert(catalog_store_url, _mk_track_with_site(i)) for i in range(3)]
    ok, created = catalog_client.create_object(catalog_store_url, track_ids=[t["id"] for t in tracks])
    assert ok

    def _fake_solve(observations, **kwargs):
        return double_r_lambert.IODResult(
            epoch_s=observations[0].epoch_s, r_km=np.zeros(3), v_km_s=np.zeros(3),
            r1_mag_km=0.0, rn_mag_km=0.0, range_covariance=None,
            rms_deg=1.0, n_iterations=1, converged=False,
        )
    monkeypatch.setattr(double_r_lambert, "solve", _fake_solve)

    outcome = correlate._attempt_iod_promotion(catalog_store_url, created["id"], 1 / 3600.0, 1 / 3600.0, 5 / 3600.0, 10.0)

    assert outcome["status"] == "rejected"
    ok, obj = catalog_client.get_object(catalog_store_url, created["id"])
    assert obj["status"] == "candidate"  # promotion withheld despite the linear gate having passed


# ── Integration: correlator's own Flask API ───────────────────────────────────

@pytest.fixture()
def correlator_client(catalog_store_url, monkeypatch):
    monkeypatch.setenv("CATALOG_STORE_URL", catalog_store_url)
    from correlator import api as correlator_api
    importlib.reload(correlator_api)

    app = correlator_api.create_app()
    app.testing = True
    with app.test_client() as c:
        yield c


def test_api_health_reports_catalog_store_reachable(correlator_client):
    resp = correlator_client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_correlate_ingests_raw_attributable_and_correlates(correlator_client):
    resp = correlator_client.post("/correlate", json=_mk_track(150.0, -3.0, 0.001, -0.0005, 1_800_000_000.0))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["action"] == "new_uct"


def test_api_correlate_ingests_raw_attributable_with_site_position(correlator_client, catalog_store_url):
    """Regression test: the raw-attributable ingestion path used to filter
    the POST body down to a fixed required-field list, silently dropping
    site_position_km even when the caller supplied it -- which meant IOD
    could never be attempted for anything ingested this way."""
    body = _mk_track(150.0, -3.0, 0.001, -0.0005, 1_800_000_000.0)
    body["site_position_km"] = R_SITE.tolist()

    resp = correlator_client.post("/correlate", json=body)
    assert resp.status_code == 200

    ok, tracks = catalog_client.list_tracks(catalog_store_url)
    assert ok and len(tracks) == 1
    assert tracks[0]["site_position_km"] == R_SITE.tolist()


def test_api_correlate_by_track_id(correlator_client, catalog_store_url):
    track = _insert(catalog_store_url, _mk_track(150.0, -3.0, 0.001, -0.0005, 1_800_000_000.0))
    resp = correlator_client.post("/correlate", json={"track_id": track["id"]})
    assert resp.status_code == 200
    assert resp.get_json()["action"] == "new_uct"


def test_api_correlate_missing_fields_returns_400(correlator_client):
    resp = correlator_client.post("/correlate", json={"sensor_id": "test"})
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

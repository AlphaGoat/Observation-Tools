"""
Tests for the pipeline coordinator's correlation wiring (Stage 6, opt-in):
converting an associator track into a catalog-store attributable, calling
the correlator sequentially per track, and the /pipeline/run request
validation and response shape around "correlate": true.

Does not re-test the pre-existing extract/solve/project/associate stages
(unchanged, untested before this work) -- scope is the new correlation
integration only.

Author: Peter Thomas
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import coordinator


def _mk_associator_track(ra=150.0, dec=-3.0, ra_dot=0.001, dec_dot=-0.0005, t_start=0.0, t_end=2.0):
    return {
        "score": 12.3,
        "n_frames": 2,
        "observations": [
            {"frame": 0, "ra": ra, "dec": dec, "t_start": t_start, "t_end": t_start + 1.0},
            {"frame": 1, "ra": ra + 0.01, "dec": dec + 0.01, "t_start": t_end - 1.0, "t_end": t_end},
        ],
        "final_state": {"ra": ra, "dec": dec, "ra_dot": ra_dot, "dec_dot": dec_dot},
        "final_covar": [[1e-6, 0, 0, 0], [0, 1e-6, 0, 0], [0, 0, 1e-10, 0], [0, 0, 0, 1e-10]],
    }


# ── _build_track_attributable ─────────────────────────────────────────────────

def test_build_track_attributable_spans_first_to_last_observation():
    track = _mk_associator_track(t_start=100.0, t_end=110.0)
    body = coordinator._build_track_attributable(track, "rubin", None)

    assert body["sensor_id"] == "rubin"
    assert body["t_start"] == 100.0
    assert body["t_end"] == 110.0
    assert body["ra"] == track["final_state"]["ra"]
    assert body["covariance"] == track["final_covar"]
    assert "site_position_km" not in body


def test_build_track_attributable_includes_site_position_when_given():
    track = _mk_associator_track()
    body = coordinator._build_track_attributable(track, "rubin", [6378.0, 0.0, 0.0])
    assert body["site_position_km"] == [6378.0, 0.0, 0.0]


# ── _correlate_tracks ────────────────────────────────────────────────────────

def test_correlate_tracks_calls_correlator_sequentially_in_order(monkeypatch):
    calls = []

    def _fake_post(url, body, timeout):
        calls.append((url, body["ra"]))
        return True, {"action": "new_uct", "track_id": len(calls), "promoted": None}

    monkeypatch.setattr(coordinator, "_post", _fake_post)

    tracks = [_mk_associator_track(ra=r) for r in (10.0, 20.0, 30.0)]
    results = coordinator._correlate_tracks(tracks, "rubin", None)

    assert [c[1] for c in calls] == [10.0, 20.0, 30.0]  # strictly in order, one at a time
    assert all(url.endswith("/correlate") for url, _ in calls)
    assert [r["track_idx"] for r in results] == [0, 1, 2]
    assert all(r["error"] is None for r in results)


def test_correlate_tracks_reports_per_track_failure_without_aborting(monkeypatch):
    def _fake_post(url, body, timeout):
        if body["ra"] == 20.0:
            return False, {"error": "catalog-store unreachable"}
        return True, {"action": "new_uct"}

    monkeypatch.setattr(coordinator, "_post", _fake_post)

    tracks = [_mk_associator_track(ra=r) for r in (10.0, 20.0, 30.0)]
    results = coordinator._correlate_tracks(tracks, "rubin", None)

    assert len(results) == 3
    assert results[1]["error"] == "catalog-store unreachable"
    assert results[1]["result"] is None
    assert results[0]["error"] is None and results[2]["error"] is None


def test_correlate_tracks_reports_malformed_track_without_aborting(monkeypatch):
    monkeypatch.setattr(coordinator, "_post", lambda *a, **k: (True, {"action": "new_uct"}))

    good = _mk_associator_track()
    malformed = {"final_state": {}, "final_covar": []}  # missing "observations"
    results = coordinator._correlate_tracks([good, malformed], "rubin", None)

    assert results[0]["error"] is None
    assert "malformed track" in results[1]["error"]


# ── /pipeline/run request validation ──────────────────────────────────────────

@pytest.fixture()
def client():
    app = coordinator.create_app()
    app.testing = True
    with app.test_client() as c:
        yield c


def test_correlate_true_without_sensor_id_returns_400(client):
    resp = client.post("/pipeline/run", json={
        "frames": [{"image": "AAAA", "t_start_s": 0.0, "t_end_s": 1.0}],
        "exposure_time": 1.0, "gap_time": 1.0, "sensor_fov": 2.0,
        "correlate": True,
    })
    assert resp.status_code == 400
    assert "sensor_id" in resp.get_json()["error"]


def test_correlate_false_by_default_no_sensor_id_required(monkeypatch, client):
    # Stub every downstream call so this only exercises validation +
    # response shape, not the real extract/solve/project/associate stages.
    monkeypatch.setattr(coordinator, "_post", lambda url, body, timeout: (False, {"error": "stub: not reached"}))

    resp = client.post("/pipeline/run", json={
        "frames": [{"image": "AAAA", "t_start_s": 0.0, "t_end_s": 1.0}],
        "exposure_time": 1.0, "gap_time": 1.0, "sensor_fov": 2.0,
    })
    assert resp.status_code == 200
    assert resp.get_json()["correlation_results"] == []


def test_pipeline_run_correlates_each_track_when_opted_in(monkeypatch, client):
    """End-to-end through the Flask layer: stub every downstream service call
    so this exercises the real /pipeline/run control flow, including that
    Stage 6 fires exactly when "correlate": true and tracks exist."""
    track = _mk_associator_track()

    def _fake_post(url, body, timeout):
        if url.endswith("/extract"):
            # Star- and satellite-extractor share the /extract suffix; a
            # single stub response covering both sets of fields is fine --
            # each caller only reads the keys it needs.
            return True, {"plate_solver_input": [{"x": 1, "y": 1}], "n_star_trails": 1,
                           "image_height": 2048, "image_width": 2048,
                           "satellites": [{"x": 5, "y": 5}], "n_satellites": 1}
        if url.endswith("/solve"):
            return True, {"solved": True, "wcs": {"ra0": 150.0, "dec0": -3.0}}
        if url.endswith("/project"):
            return True, {"observations": [{"ra": 150.0, "dec": -3.0, "t_start": 0.0, "t_end": 1.0}]}
        if url.endswith("/associate"):
            return True, {"tracks": [track]}
        if url.endswith("/correlate"):
            return True, {"action": "new_uct", "promoted": None}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(coordinator, "_post", _fake_post)

    resp = client.post("/pipeline/run", json={
        "frames": [{"image": "AAAA", "t_start_s": 0.0, "t_end_s": 1.0}],
        "exposure_time": 1.0, "gap_time": 1.0, "sensor_fov": 2.0,
        "correlate": True, "sensor_id": "rubin",
    })

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_tracks"] == 1
    assert len(body["correlation_results"]) == 1
    assert body["correlation_results"][0]["result"]["action"] == "new_uct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
correlate.py — orchestrates track-to-track correlation against catalog-store.

This is the "decision" layer Pastor's hypothesis-tree architecture (2022,
Ch. 3) describes: generation, scoring, pruning, and promotion of
track-to-track linkage hypotheses. catalog-store itself is deliberately
dumb storage (see catalog_store/db.py); every decision about which tracks
should link, and when a hypothesis is confident enough to confirm, is made
here, then executed against catalog-store's API.

Algorithm, per new track
-------------------------
1. Fetch in-progress hypotheses (status='candidate' objects) and recently
   ingested tracks within a lookback window.
2. Figure out which of those tracks are "loose" -- not yet claimed by any
   in-progress hypothesis -- since a track already inside a hypothesis
   should only be extended via that hypothesis, not re-paired individually.
3. Score the new track against every in-progress hypothesis and every loose
   track (linking.rank_candidates: chi-square compatibility gate + k-best
   cutoff -- the generation, scoring, and pruning steps).
4. For each surviving candidate (up to k_best -- Pastor's branching
   hypotheses):
     - a hypothesis match attributes the new track to that object;
     - a loose-track match spawns a brand-new 2-track hypothesis.
   If nothing survives the gate, the track becomes its own single-track
   hypothesis -- by catalog-store's design this *is* an uncorrelated track
   (UCT), discoverable via GET /objects?status=candidate.
5. The single best-scoring branch is checked against a promotion
   threshold (minimum track count + chi2 ceiling); if it passes, an actual
   orbit determination is attempted (Pastor 2022's Double r-iteration
   Lambert method, src/iod/double_r_lambert.py) using every constituent
   track's angles -- but only if every one of them carries an observer
   position (catalog_store's track.site_position_km, nullable since
   nothing upstream populates it yet). If IOD converges with an
   acceptably low residual, the hypothesis is promoted with that real 6-D
   Cartesian state; if it doesn't converge or fits poorly, promotion is
   *withheld* even though the cheaper linear chi2 gate passed, since a
   real orbit disagreeing is a stronger, more informative signal than the
   linear approximation agreeing. If IOD can't be attempted at all
   (missing observer positions), promotion falls back to the linear-only
   decision (the bootstrap behavior this module used before IOD existed).
   Either way, catalog-store's promote_object confirms the hypothesis and
   atomically invalidates any other candidate sharing a track with it
   (Pastor's Algorithm 7 -- implemented in catalog_store/db.py, not
   duplicated here).

What "representative state" means before promotion
------------------------------------------------------
Every hypothesis below the promotion threshold still carries a
"representative state" for scoring purposes, but it's just the most
recent constituent track's raw 4-D attributable (ra, dec, ra_dot,
dec_dot) -- a linear stand-in, not a real orbital state. Only promotion
itself calls into real orbit determination (see above); everything before
that (generation, scoring, pruning) stays in the cheap linear
approximation, which is the right tradeoff since most candidate branches
never reach promotion at all.

Author: Peter Thomas
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from correlator import catalog_client, linking
from iod import double_r_lambert

log = logging.getLogger(__name__)


class CorrelationError(RuntimeError):
    """Raised when a required catalog-store call fails."""


def _require(ok: bool, data: Any, what: str) -> Any:
    if not ok:
        raise CorrelationError(f"{what} failed: {data}")
    return data


def correlate_new_track(
    base_url: str,
    track: Dict[str, Any],
    *,
    lookback_s: float = 7 * 24 * 3600.0,
    chi2_gate: float = 18.47,          # chi2, 4 dof, ~99.9th percentile
    k_best: int = 3,
    sigma_accel_deg_s2: float = 1e-8,
    promote_min_tracks: int = 3,       # Milani's "confirm via 3rd-tracklet attribution"
    promote_chi2_max: float = 9.49,    # chi2, 4 dof, ~95th percentile
    iod_sigma_ra_deg: float = 1.0 / 3600.0,
    iod_sigma_dec_deg: float = 1.0 / 3600.0,
    iod_max_rms_deg: float = 5.0 / 3600.0,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Correlate one already-ingested track (a catalog-store track row, with
    'id') against in-progress hypotheses and other recent tracks.

    Returns a summary dict: {"track_id", "branches": [...], "action",
    "promoted": object_id | None}.
    """
    track_id = track["id"]
    now = time.time()
    since = now - lookback_s

    ok, candidate_objects = catalog_client.list_objects(
        base_url, status="candidate", updated_after=since, timeout=timeout,
    )
    _require(ok, candidate_objects, "list candidate objects")

    ok, recent_tracks = catalog_client.list_tracks(base_url, since=since, timeout=timeout)
    _require(ok, recent_tracks, "list recent tracks")

    attached_ids = {tid for obj in candidate_objects for tid in obj.get("track_ids", [])}
    loose_tracks = [t for t in recent_tracks if t["id"] != track_id and t["id"] not in attached_ids]

    new_state = linking.track_to_state(track)

    object_pairs = []
    for obj in candidate_objects:
        ref = linking.object_to_state(obj)
        if ref is not None:
            object_pairs.append((obj["id"], ref))
    track_pairs = [(t["id"], linking.track_to_state(t)) for t in loose_tracks]

    ranked = linking.rank_candidates(
        new_state, object_pairs, track_pairs, chi2_gate, sigma_accel_deg_s2, k_best,
    )

    result: Dict[str, Any] = {"track_id": track_id, "branches": [], "action": None, "promoted": None}

    if not ranked:
        obj_id = _create_singleton(base_url, track, new_state, timeout)
        result["action"] = "new_uct"
        result["branches"] = [{"kind": "new_object", "object_id": obj_id, "chi2": None}]
        log.info("Track %d: no compatible candidates -> new UCT (object %d)", track_id, obj_id)
        return result

    for rank, cand in enumerate(ranked):
        if cand.kind == "object":
            resulting_object_id = _attribute_to_object(base_url, cand.id, track, new_state, cand.chi2, timeout)
        else:
            resulting_object_id = _spawn_pair(base_url, track, cand.id, cand.chi2, timeout)
        result["branches"].append({
            "kind": cand.kind, "source_id": cand.id, "chi2": cand.chi2,
            "object_id": resulting_object_id,
        })

    result["action"] = "branched"
    best = ranked[0]
    best_object_id = result["branches"][0]["object_id"]

    n_tracks = _object_track_count(base_url, best_object_id, timeout)
    if best.kind == "object" and n_tracks >= promote_min_tracks and best.chi2 <= promote_chi2_max:
        iod_outcome = _attempt_iod_promotion(
            base_url, best_object_id, iod_sigma_ra_deg, iod_sigma_dec_deg, iod_max_rms_deg, timeout,
        )
        result["iod"] = iod_outcome

        if iod_outcome["status"] == "unavailable":
            # No IOD to check against (missing observer positions) --
            # fall back to the pre-IOD linear-only promotion decision.
            ok, promotion = catalog_client.promote_object(base_url, best_object_id, timeout=timeout)
            if ok:
                result["promoted"] = best_object_id
                result["invalidated"] = promotion.get("invalidated", [])
                log.info(
                    "Track %d: promoted object %d on linear chi2 only (IOD unavailable: %s)",
                    track_id, best_object_id, iod_outcome.get("reason"),
                )
            else:
                log.warning("Track %d: promotion of object %d failed: %s", track_id, best_object_id, promotion)
        elif iod_outcome["status"] == "promoted":
            result["promoted"] = best_object_id
            result["invalidated"] = iod_outcome.get("invalidated", [])
            log.info(
                "Track %d: promoted object %d on a converged IOD fit (rms=%.2g deg)",
                track_id, best_object_id, iod_outcome.get("rms_deg"),
            )
        else:  # "rejected" -- IOD ran but didn't converge or fit poorly; withhold promotion
            log.info(
                "Track %d: withheld promotion of object %d -- IOD rejected (%s)",
                track_id, best_object_id, iod_outcome.get("reason"),
            )

    return result


def _create_singleton(base_url: str, track: Dict[str, Any], state: linking.TrackState, timeout: float) -> int:
    ok, data = catalog_client.create_object(
        base_url,
        track_ids=[track["id"]],
        status="candidate",
        state_vector=state.state.tolist(),
        covariance=state.covariance.tolist(),
        epoch=state.epoch,
        timeout=timeout,
    )
    return _require(ok, data, "create singleton UCT")["id"]


def _attribute_to_object(
    base_url: str, object_id: int, track: Dict[str, Any], state: linking.TrackState,
    chi2: float, timeout: float,
) -> int:
    ok, data = catalog_client.add_tracks_to_object(base_url, object_id, [track["id"]], timeout=timeout)
    _require(ok, data, f"attribute track {track['id']} to object {object_id}")

    catalog_client.update_object(
        base_url, object_id,
        state_vector=state.state.tolist(),
        covariance=state.covariance.tolist(),
        epoch=state.epoch,
        figure_of_merit=chi2,
        timeout=timeout,
    )
    return object_id


def _spawn_pair(base_url: str, track: Dict[str, Any], other_track_id: int, chi2: float, timeout: float) -> int:
    ok, other = catalog_client.get_track(base_url, other_track_id, timeout=timeout)
    _require(ok, other, f"fetch track {other_track_id}")

    later = track if track["t_end"] >= other["t_end"] else other
    later_state = linking.track_to_state(later)

    ok, data = catalog_client.create_object(
        base_url,
        track_ids=[track["id"], other_track_id],
        status="candidate",
        state_vector=later_state.state.tolist(),
        covariance=later_state.covariance.tolist(),
        epoch=later_state.epoch,
        figure_of_merit=chi2,
        timeout=timeout,
    )
    return _require(ok, data, "create paired hypothesis")["id"]


def _attempt_iod_promotion(
    base_url: str, object_id: int,
    sigma_ra_deg: float, sigma_dec_deg: float, max_rms_deg: float, timeout: float,
) -> Dict[str, Any]:
    """
    Try to promote object_id on the strength of a real orbit fit rather
    than the linear chi2 gate alone.

    Returns {"status": "promoted"|"rejected"|"unavailable", ...}:
      - "unavailable": IOD could not be attempted (a constituent track is
        missing site_position_km) -- caller should fall back to the
        linear-only promotion decision.
      - "rejected": IOD ran but did not converge, or converged with a
        residual above max_rms_deg -- caller should withhold promotion.
      - "promoted": IOD converged acceptably; the object has already been
        updated with the real 6-D state (state_vector=[r_km, v_km_s],
        stale 4-D covariance cleared) and promoted via catalog-store.
    """
    ok, obj = catalog_client.get_object(base_url, object_id, timeout=timeout)
    if not ok:
        return {"status": "unavailable", "reason": f"could not fetch object: {obj}"}

    tracks = []
    for track_id in obj["track_ids"]:
        ok, t = catalog_client.get_track(base_url, track_id, timeout=timeout)
        if not ok:
            return {"status": "unavailable", "reason": f"could not fetch track {track_id}: {t}"}
        tracks.append(t)

    if any(t.get("site_position_km") is None for t in tracks):
        return {"status": "unavailable", "reason": "one or more tracks lack site_position_km"}

    observations = [
        double_r_lambert.Observation(
            ra_deg=t["ra"], dec_deg=t["dec"],
            epoch_s=0.5 * (t["t_start"] + t["t_end"]),
            site_position_km=t["site_position_km"],
        )
        for t in tracks
    ]

    try:
        iod_result = double_r_lambert.solve(observations, sigma_ra_deg=sigma_ra_deg, sigma_dec_deg=sigma_dec_deg)
    except (double_r_lambert.IODError, ValueError, RuntimeError) as exc:
        return {"status": "rejected", "reason": f"IOD raised: {exc}"}

    if not iod_result.converged:
        return {"status": "rejected", "reason": "IOD did not converge"}
    if iod_result.rms_deg > max_rms_deg:
        return {"status": "rejected", "reason": f"IOD rms {iod_result.rms_deg:.3g} deg exceeds {max_rms_deg:.3g} deg",
                "rms_deg": iod_result.rms_deg}

    state_vector = list(iod_result.r_km) + list(iod_result.v_km_s)
    ok, updated = catalog_client.update_object(
        base_url, object_id,
        state_vector=state_vector, epoch=iod_result.epoch_s,
        figure_of_merit=iod_result.rms_deg, clear_covariance=True, timeout=timeout,
    )
    if not ok:
        return {"status": "unavailable", "reason": f"could not store IOD state: {updated}"}

    ok, promotion = catalog_client.promote_object(base_url, object_id, timeout=timeout)
    if not ok:
        return {"status": "unavailable", "reason": f"promote_object failed: {promotion}"}

    return {"status": "promoted", "rms_deg": iod_result.rms_deg, "invalidated": promotion.get("invalidated", [])}


def _object_track_count(base_url: str, object_id: int, timeout: float) -> int:
    ok, data = catalog_client.get_object(base_url, object_id, timeout=timeout)
    obj = _require(ok, data, f"fetch object {object_id}")
    return obj["n_tracks"]

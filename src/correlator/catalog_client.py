"""
catalog_client.py — thin HTTP client for the catalog-store service.

Mirrors the `_post` helper pattern in pipeline/coordinator.py: every
function returns (ok, data) instead of raising, so the caller decides how
to handle a downstream failure rather than every call site needing its own
try/except around requests.

Author: Peter Thomas
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

JsonResult = Tuple[bool, Any]


def _request(method: str, url: str, timeout: float, **kwargs) -> JsonResult:
    try:
        r = requests.request(method, url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return True, (r.json() if r.content else {})
    except requests.exceptions.Timeout:
        return False, {"error": f"timeout after {timeout}s calling {url}"}
    except requests.exceptions.ConnectionError as exc:
        return False, {"error": f"connection error to {url}: {exc}"}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = {}
        return False, {"error": f"HTTP {exc.response.status_code} from {url}", "detail": detail}
    except Exception as exc:
        return False, {"error": f"unexpected error calling {url}: {exc}"}


def insert_track(base_url: str, track: Dict[str, Any], timeout: float = 10.0) -> JsonResult:
    return _request("POST", f"{base_url}/tracks", timeout, json=track)


def get_track(base_url: str, track_id: int, timeout: float = 10.0) -> JsonResult:
    return _request("GET", f"{base_url}/tracks/{track_id}", timeout)


def list_tracks(
    base_url: str, since: Optional[float] = None, sensor_id: Optional[str] = None,
    timeout: float = 10.0,
) -> JsonResult:
    params = {}
    if since is not None:
        params["since"] = since
    if sensor_id is not None:
        params["sensor_id"] = sensor_id
    ok, data = _request("GET", f"{base_url}/tracks", timeout, params=params)
    return ok, (data.get("tracks", []) if ok else data)


def get_object(base_url: str, object_id: int, timeout: float = 10.0) -> JsonResult:
    return _request("GET", f"{base_url}/objects/{object_id}", timeout)


def create_object(
    base_url: str,
    track_ids: List[int],
    status: str = "candidate",
    state_vector: Optional[List[float]] = None,
    covariance: Optional[List[List[float]]] = None,
    epoch: Optional[float] = None,
    figure_of_merit: Optional[float] = None,
    parent_ids: Optional[List[int]] = None,
    timeout: float = 10.0,
) -> JsonResult:
    body = {"track_ids": track_ids, "status": status}
    if state_vector is not None:
        body["state_vector"] = state_vector
    if covariance is not None:
        body["covariance"] = covariance
    if epoch is not None:
        body["epoch"] = epoch
    if figure_of_merit is not None:
        body["figure_of_merit"] = figure_of_merit
    if parent_ids is not None:
        body["parent_ids"] = parent_ids
    return _request("POST", f"{base_url}/objects", timeout, json=body)


def list_objects(
    base_url: str,
    status: Optional[str] = None,
    updated_before: Optional[float] = None,
    updated_after: Optional[float] = None,
    timeout: float = 10.0,
) -> JsonResult:
    params = {}
    if status is not None:
        params["status"] = status
    if updated_before is not None:
        params["updated_before"] = updated_before
    if updated_after is not None:
        params["updated_after"] = updated_after
    ok, data = _request("GET", f"{base_url}/objects", timeout, params=params)
    return ok, (data.get("objects", []) if ok else data)


def add_tracks_to_object(base_url: str, object_id: int, track_ids: List[int], timeout: float = 10.0) -> JsonResult:
    return _request("POST", f"{base_url}/objects/{object_id}/tracks", timeout, json={"track_ids": track_ids})


def update_object(
    base_url: str,
    object_id: int,
    state_vector: Optional[List[float]] = None,
    covariance: Optional[List[List[float]]] = None,
    epoch: Optional[float] = None,
    figure_of_merit: Optional[float] = None,
    clear_covariance: bool = False,
    timeout: float = 10.0,
) -> JsonResult:
    body = {}
    if state_vector is not None:
        body["state_vector"] = state_vector
    if clear_covariance:
        body["clear_covariance"] = True
    elif covariance is not None:
        body["covariance"] = covariance
    if epoch is not None:
        body["epoch"] = epoch
    if figure_of_merit is not None:
        body["figure_of_merit"] = figure_of_merit
    return _request("PATCH", f"{base_url}/objects/{object_id}", timeout, json=body)


def promote_object(base_url: str, object_id: int, timeout: float = 10.0) -> JsonResult:
    return _request("POST", f"{base_url}/objects/{object_id}/promote", timeout)

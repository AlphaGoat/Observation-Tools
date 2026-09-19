"""
Utilities for grabbing TLEs from Space-Track.org

Author: Peter Thomas
Date: 2025-10-12

---
Fix (2026-09-17): "active" and "visual" used to add a "CURRENT/Y" predicate.
Verified against Space-Track's live /basicspacedata/modeldef/class/gp --
there is no "CURRENT" field on the `gp` class, so that predicate 500'd on
every real request (caught by actually running this against Space-Track
while testing scheduler-service's tle-refresh CronJob, not by inspection).
`gp` already returns only the latest elset per object -- that's what
distinguishes it from `gp_history` -- so the predicate was never doing
anything but breaking the query. Removed it; "active" and "all" are now
identical queries (both plain `class/gp`) until a real distinction (e.g.
filtering DECAY_DATE) is designed.
"""
import os
import json
import getpass
import argparse
import requests
from typing import Optional


# Maps the catalog argument to the Space-Track GP class and predicate string.
# "active" - most recent TLE for every currently tracked object
# "visual"  - same, but filtered to payloads only (most likely optically observable)
# "all"     - most recent TLE for all objects including debris and rocket bodies
_CATALOG_QUERY_MAP = {
    "active": "class/gp",
    "visual": "class/gp/OBJECT_TYPE/PAYLOAD",
    "all":    "class/gp",
}


def _get_credentials(username: Optional[str], password: Optional[str]):
    """
    Resolve Space-Track credentials using a priority-ordered fallback:
      1. Explicit arguments passed by the caller
      2. Environment variables SPACETRACK_USER and SPACETRACK_PASS
      3. Interactive getpass prompts (password is never echoed)
    """
    username = username or os.environ.get("SPACETRACK_USER") or input("Space-Track username: ")
    password = password or os.environ.get("SPACETRACK_PASS") or getpass.getpass("Space-Track password: ")
    return username, password


def get_latest_tles(username: Optional[str] = None, password: Optional[str] = None,
                    catalog: str = "active",
                    fmt: str = "tle", limit: int = 100) -> str:
    """
    Get the latest TLEs from Space-Track.org

    Parameters:
    username (str): Space-Track.org username.
    password (str): Space-Track.org password.
    catalog (str): Catalog to query. Options are "active", "visual", or "all". Default is "active".
    fmt (str): Format of the returned data. Options are "tle" or "json". Default is "tle".
    limit (int): Maximum number of records to return.

    Returns:
    str: TLE data as a string.
    """
    if catalog not in _CATALOG_QUERY_MAP:
        raise ValueError(f"Unknown catalog '{catalog}'. Choose from: {list(_CATALOG_QUERY_MAP.keys())}")

    username, password = _get_credentials(username, password)

    base_url = "https://www.space-track.org"
    login_url = f"{base_url}/ajaxauth/login"

    with requests.Session() as session:
        response = session.post(login_url, data={"identity": username, "password": password})
        response.raise_for_status()
        if "Failed" in response.text or "Invalid" in response.text:
            raise RuntimeError("Space-Track authentication failed — check credentials.")

        class_predicates = _CATALOG_QUERY_MAP[catalog]
        query = (
            f"{base_url}/basicspacedata/query"
            f"/{class_predicates}"
            f"/orderby/NORAD_CAT_ID asc"
            f"/limit/{limit}"
            f"/format/{fmt}"
            f"/emptyresult/show"
        )
        response = session.get(query)
        response.raise_for_status()

    return response.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest TLEs from Space-Track.org")
    parser.add_argument("--catalog", type=str, default="active", choices=list(_CATALOG_QUERY_MAP.keys()),
                        help="Catalog to query")
    parser.add_argument("--format", type=str, default="tle", choices=["tle", "json"],
                        help="Format of the returned data")
    parser.add_argument("--limit", type=int, default=100, help="Number of TLEs to fetch")
    parser.add_argument("--output", type=str, default="tles.txt", help="Output file to save TLE data")
    args = parser.parse_args()

    tle_data = get_latest_tles(username=None, password=None, 
                               catalog=args.catalog, fmt=args.format, 
                               limit=args.limit)

    if args.format == "json":
        with open(args.output, "w") as f:
            json.dump(json.loads(tle_data), f, indent=4)
    else:
        with open(args.output, "w") as f:
            f.write(tle_data)

    print(f"TLE data saved to {args.output}")

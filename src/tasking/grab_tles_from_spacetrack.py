"""
Utilities for grabbing TLEs from Space-Track.org

Author: Peter Thomas
Date: 2025-10-12
"""
import urllib.parse


def get_latest_tles(username: str, password: str, catalog: str = "active", format: str = "tle", limit: int=100) -> str:
    """
    Get the latest TLEs from Space-Track.org

    Parameters:
    username (str): Space-Track.org username
    password (str): Space-Track.org password
    catalog (str): Catalog to query. Options are "active", "historical", "all", or "visual". Default is "active".
    format (str): Format of the returned data. Options are "tle" or "json". Default is "tle".

    Returns:
    str: TLE data as a string
    """
    import requests

    base_url = "https://www.space-track.org"
    login_url = f"{base_url}/ajaxauth/login"

    # Start a session to persist cookies
    with requests.Session() as session:

        # Login to Space-Track.org
        login_payload = {
            'identity': username,
            'password': password,
        }
        response = session.post(login_url, data=login_payload)
        response.raise_for_status()

        # Fetch the TLE data
        query = urllib.parse.urljoin(base_url, urllib.parse.quote(f'/basicspacedata/query/class/tle_latest/orderby/NORAD_CAT_ID asc/limit/{limit}/format/{format}/emptyresult/show'))
        response = session.get(query)
        response.raise_for_status()

    return response.text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch latest TLEs from Space-Track.org")
    parser.add_argument('--username', type=str, required=True, help='Space-Track.org username')
    parser.add_argument('--password', type=str, required=True, help='Space-Track.org password')
    parser.add_argument('--catalog', type=str, default='active', choices=['active', 'historical', 'all', 'visual'], help='Catalog to query')
    parser.add_argument('--format', type=str, default='tle', choices=['tle', 'json'], help='Format of the returned data')
    parser.add_argument("--limit", type=int, default=100, help="Number of TLEs to fetch")
    parser.add_argument('--output', type=str, default='tles.txt', help='Output file to save TLE data')

    args = parser.parse_args()

    tle_data = get_latest_tles(args.username, args.password, args.catalog, args.format, args.limit)
    if args.format == "json":
        import json
        tle_json = json.loads(tle_data)
        with open(args.output, 'w') as f:
            json.dump(tle_json, f, indent=4)
    else:
        with open(args.output, 'w') as f:
            f.write(tle_data)

    print(f"TLE data saved to {args.output}")
"""
Utilities for getting satellite visibility information.

Author: Peter Thomas
Date: 2025-10-12
"""
from skyfield.api import load, wgs84
from skyfield.sgp4lib import EarthSatellite


def get_satellite_visibility(tle_line1: str, tle_line2: str, observer_lat: float, observer_lon: float, observer_elevation_m: float,
                             start_time_utc: str, end_time_utc: str, time_step_minutes: int=1):
    """
    Calculate satellite visibility from a given observer location and time range.

    Parameters:
    tle_line1 (str): First line of the TLE data.
    tle_line2 (str): Second line of the TLE data.
    observer_lat (float): Observer latitude in degrees.
    observer_lon (float): Observer longitude in degrees.
    observer_elevation_m (float): Observer elevation in meters.
    start_time_utc (str): Start time in UTC (ISO format).
    end_time_utc (str): End time in UTC (ISO format).
    time_step_minutes (int): Time step in minutes for visibility calculation.

    Returns:
    list of tuples: Each tuple contains (time, altitude, azimuth) when the satellite is visible.
    """
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2)
    observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)

    start_time = ts.utc(*map(int, start_time_utc.replace('T', '-').replace(':', '-').split('-')))
    end_time = ts.utc(*map(int, end_time_utc.replace('T', '-').replace(':', '-').split('-')))

    times = ts.utc_range(start_time, end_time, step=time_step_minutes * 60)

    visibility_data = []

    for t in times:
        difference = satellite - observer
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()

        if alt.degrees > 0:  # Satellite is above the horizon
            visibility_data.append((t.utc_iso(), alt.degrees, az.degrees))

    return visibility_data


def is_satellite_illuminated(tle_line1: str, tle_line2: str, time_utc: str) -> bool:
    """
    Determine if the satellite is illuminated by the Sun at a given time.

    Parameters:
    tle_line1 (str): First line of the TLE data.
    tle_line2 (str): Second line of the TLE data.
    time_utc (str): Time in UTC (ISO format).

    Returns:
    bool: True if the satellite is illuminated, False otherwise.
    """
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2)
    eph = load('de421.bsp')
    sun = eph['sun']
    earth = eph['earth']

    t = ts.utc(*map(int, time_utc.replace('T', '-').replace(':', '-').split('-')))
    sat_at_time = satellite.at(t)
    sun_at_time = sun.at(t)
    earth_at_time = earth.at(t)

    sat_pos = sat_at_time.position.km
    sun_pos = sun_at_time.position.km
    earth_pos = earth_at_time.position.km

    # Vector from Earth to Satellite
    earth_to_sat = sat_pos - earth_pos
    # Vector from Earth to Sun
    earth_to_sun = sun_pos - earth_pos

    # Calculate the angle between the two vectors
    dot_product = sum(a * b for a, b in zip(earth_to_sat, earth_to_sun))
    mag_earth_to_sat = sum(a**2 for a in earth_to_sat) ** 0.5
    mag_earth_to_sun = sum(a**2 for a in earth_to_sun) ** 0.5

    cos_angle = dot_product / (mag_earth_to_sat * mag_earth_to_sun)

    # If the angle is less than 90 degrees, the satellite is illuminated
    return cos_angle > 0.


def get_sensor_slew_rate(satellite, sensor_fov_deg: float, t_start: float, t_end: float) -> float:
    """
    Calculate the required slew rate for a sensor to center target satellite over collection.

    Parameters:
    sensor_fov_deg (float): Sensor field of view in degrees.
    exposure_time_s (float): Exposure time in seconds.

    Returns:
    float: Required slew rate in degrees per second.
    """
    if t_end <= t_start:
        raise ValueError("End time must be greater than start time.")
    duration_s = t_end - t_start

    # Get rate of change of satellite position in degrees per second
    sat_at_start = satellite.at(t_start)
    sat_at_end = satellite.at(t_end)
    ra_start, dec_start, _ = sat_at_start.radec()
    ra_end, dec_end, _ = sat_at_end.radec()

    return sensor_fov_deg / duration_s


if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser(description="Calculate satellite visibility and illumination.")
    parser.add_argument('--tle_file', type=str, required=True, help='Path to the TLE file.')
    parser.add_argument('--observer_lat', type=float, required=True, help='Observer latitude in degrees.')
    parser.add_argument('--observer_lon', type=float, required=True, help='Observer longitude in degrees.')
    parser.add_argument('--observer_elevation_m', type=float, required=True, help='Observer elevation in meters.')
    parser.add_argument('--start_time_utc', type=str, required=True, help='Start time in UTC (ISO format).')
    parser.add_argument('--end_time_utc', type=str, required=True, help='End time in UTC (ISO format).')

    args = parser.parse_args()

    with open(args.tle_file, 'r') as f:
        satellites = json.load(f)

    for sat in satellites:
        tle_line1 = sat['TLE_LINE1']
        tle_line2 = sat['TLE_LINE2']
        visibility = get_satellite_visibility(
            tle_line1, tle_line2,
            args.observer_lat, args.observer_lon, args.observer_elevation_m,
            args.start_time_utc, args.end_time_utc
        )
        print(f"Satellite: {sat['name']}")
        for time, alt, az in visibility:
            illuminated = is_satellite_illuminated(tle_line1, tle_line2, time)
            illum_status = "Illuminated" if illuminated else "In Earth's Shadow"
            print(f"Time: {time}, Altitude: {alt:.2f}°, Azimuth: {az:.2f}°, Status: {illum_status}")
        print("\n")
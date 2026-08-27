"""
Utilities for getting satellite visibility information.

Author: Peter Thomas
Date: 2025-10-12
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from typing import List, Tuple, Optional

from skyfield.api import load, wgs84, EarthSatellite


def _parse_utc(time_str: str) -> datetime:
    """Parse an ISO 8601 UTC string to a timezone-aware datetime."""
    return datetime.fromisoformat(time_str.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def _build_time_array(ts, start_time_utc: str, end_time_utc: str, time_step_minutes: int):
    """Return a skyfield Time array spanning the given UTC window at the requested cadence."""
    start_dt = _parse_utc(start_time_utc)
    end_dt = _parse_utc(end_time_utc)
    total_seconds = (end_dt - start_dt).total_seconds()
    num_steps = max(2, int(total_seconds / (time_step_minutes * 60)) + 1)
    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(end_dt)
    return ts.tt_jd(np.linspace(t0.tt, t1.tt, num_steps))


def get_satellite_visibility(
    tle_line1: str,
    tle_line2: str,
    observer_lat: float,
    observer_lon: float,
    observer_elevation_m: float,
    start_time_utc: str,
    end_time_utc: str,
    time_step_minutes: int = 1,
    min_altitude_deg: float = 0.0,
) -> List[Tuple[str, float, float]]:
    """
    Calculate satellite visibility from a given observer location and time range.

    Parameters:
    tle_line1 (str): First line of the TLE data.
    tle_line2 (str): Second line of the TLE data.
    observer_lat (float): Observer latitude in degrees.
    observer_lon (float): Observer longitude in degrees.
    observer_elevation_m (float): Observer elevation in metres.
    start_time_utc (str): Start time in UTC (ISO 8601 format).
    end_time_utc (str): End time in UTC (ISO 8601 format).
    time_step_minutes (int): Sampling cadence in minutes.
    min_altitude_deg (float): Minimum altitude above horizon to count as visible.

    Returns:
    List of (utc_iso, altitude_deg, azimuth_deg) tuples for each visible epoch.
    """
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, ts=ts)
    observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)

    times = _build_time_array(ts, start_time_utc, end_time_utc, time_step_minutes)

    topocentric = (satellite - observer).at(times)
    alt, az, _ = topocentric.altaz()

    visible_mask = alt.degrees > min_altitude_deg
    return [
        (times[i].utc_iso(), float(alt.degrees[i]), float(az.degrees[i]))
        for i in range(len(times))
        if visible_mask[i]
    ]


def is_satellite_illuminated(
    tle_line1: str,
    tle_line2: str,
    time_utc: str,
    eph=None,
) -> bool:
    """
    Determine if the satellite is illuminated by the Sun at a given time.

    Parameters:
    tle_line1 (str): First line of the TLE data.
    tle_line2 (str): Second line of the TLE data.
    time_utc (str): Time in UTC (ISO 8601 format).
    eph: Pre-loaded skyfield ephemeris. If None, de421.bsp is loaded automatically.

    Returns:
    bool: True if the satellite is sunlit, False if it is in Earth's shadow.
    """
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, ts=ts)
    if eph is None:
        eph = load("de421.bsp")
    t = ts.from_datetime(_parse_utc(time_utc))
    return bool(satellite.at(t).is_sunlit(eph))


def get_sensor_slew_rate(
    satellite: EarthSatellite,
    t_start,
    t_end,
) -> float:
    """
    Calculate the mean angular rate of a satellite across the sky between two skyfield
    Time objects, in degrees per second. This gives the minimum slew rate the telescope
    must sustain to track the satellite over that interval.

    Parameters:
    satellite: Skyfield EarthSatellite object.
    t_start: Skyfield Time object for the start of the interval.
    t_end: Skyfield Time object for the end of the interval.

    Returns:
    float: Mean angular rate in degrees per second.
    """
    if t_end.tt <= t_start.tt:
        raise ValueError("End time must be greater than start time.")

    ra_start, dec_start, _ = satellite.at(t_start).radec()
    ra_end, dec_end, _ = satellite.at(t_end).radec()

    delta_ra = (ra_end.degrees - ra_start.degrees) * np.cos(np.radians(dec_start.degrees))
    delta_dec = dec_end.degrees - dec_start.degrees
    angular_distance = np.sqrt(delta_ra**2 + delta_dec**2)

    duration_s = (t_end.tt - t_start.tt) * 86400.0
    return angular_distance / duration_s


def plot_visibility(
    tle_list: List[Tuple[str, str, str]],
    observer_lat: float,
    observer_lon: float,
    observer_elevation_m: float,
    start_time_utc: str,
    end_time_utc: str,
    min_altitude_deg: float = 15.0,
    time_step_minutes: int = 1,
    eph=None,
) -> plt.Figure:
    """
    Plot altitude vs time for all satellites that are visible above min_altitude_deg
    during the specified observation window.

    Illuminated passes are drawn as solid lines; passes in Earth's shadow are dashed.
    A shaded band marks the portion of the window where the Sun is below the horizon
    (night-time at the observer location).

    Parameters:
    tle_list: List of (name, tle_line1, tle_line2) tuples.
    observer_lat (float): Observer latitude in degrees.
    observer_lon (float): Observer longitude in degrees.
    observer_elevation_m (float): Observer elevation in metres.
    start_time_utc (str): Start of observation window (ISO 8601 UTC).
    end_time_utc (str): End of observation window (ISO 8601 UTC).
    min_altitude_deg (float): Minimum altitude above horizon to plot.
    time_step_minutes (int): Sampling cadence in minutes.
    eph: Pre-loaded skyfield ephemeris. Loaded automatically if None.

    Returns:
    matplotlib Figure.
    """
    ts = load.timescale()
    if eph is None:
        eph = load("de421.bsp")

    observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elevation_m)
    times = _build_time_array(ts, start_time_utc, end_time_utc, time_step_minutes)

    # Convert skyfield times to Python datetimes for matplotlib
    time_dts = [t.utc_datetime() for t in times]

    # Compute solar altitude to shade the night-time window
    sun = eph["sun"]
    earth = eph["earth"]
    sun_topo = (earth + observer).at(times).observe(sun).apparent()
    sun_alt, _, _ = sun_topo.altaz()
    night_mask = sun_alt.degrees < 0.0

    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade night-time periods
    in_night = False
    night_start = None
    for i, is_night in enumerate(night_mask):
        if is_night and not in_night:
            night_start = time_dts[i]
            in_night = True
        elif not is_night and in_night:
            ax.axvspan(night_start, time_dts[i], color="navy", alpha=0.08, label="_nolegend_")
            in_night = False
    if in_night:
        ax.axvspan(night_start, time_dts[-1], color="navy", alpha=0.08)

    # Plot each satellite
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    any_visible = False

    for sat_idx, (name, line1, line2) in enumerate(tle_list):
        satellite = EarthSatellite(line1, line2, ts=ts)
        topocentric = (satellite - observer).at(times)
        alt, _, _ = topocentric.altaz()
        sunlit = satellite.at(times).is_sunlit(eph)

        alt_deg = alt.degrees
        color = prop_cycle[sat_idx % len(prop_cycle)]
        label_added = False

        # Walk through time steps and draw segments coloured by illumination state
        i = 0
        while i < len(times):
            if not (alt_deg[i] >= min_altitude_deg):  # handles NaN safely
                i += 1
                continue

            # Find the end of this continuous visible window
            j = i
            while j < len(times) and alt_deg[j] >= min_altitude_deg:
                j += 1

            if j == i:  # degenerate: NaN slipped through — skip
                i += 1
                continue

            seg_times = time_dts[i:j]
            seg_alt = alt_deg[i:j]
            seg_sunlit = sunlit[i:j]

            # Split the segment further by illumination state
            k = 0
            while k < len(seg_times):
                lit = seg_sunlit[k]
                m = k
                while m < len(seg_times) and seg_sunlit[m] == lit:
                    m += 1

                linestyle = "-" if lit else "--"
                lbl = name if (not label_added) else "_nolegend_"
                ax.plot(
                    seg_times[k:m],
                    seg_alt[k:m],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=lbl,
                )
                label_added = True
                any_visible = True
                k = m

            i = j

    if not any_visible:
        ax.text(0.5, 0.5, "No satellites visible above threshold during this window.",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)

    ax.axhline(min_altitude_deg, color="gray", linewidth=0.8, linestyle=":", label=f"Min altitude ({min_altitude_deg}°)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (degrees)")
    ax.set_title(
        f"Satellite Visibility  |  "
        f"Lat {observer_lat:.2f}°  Lon {observer_lon:.2f}°  "
        f"Elev {observer_elevation_m:.0f} m\n"
        f"{start_time_utc}  →  {end_time_utc}"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Calculate and plot satellite visibility.")
    parser.add_argument("--tle_file", type=str, required=True, help="Path to JSON TLE file.")
    parser.add_argument("--observer_lat", type=float, required=True, help="Observer latitude in degrees.")
    parser.add_argument("--observer_lon", type=float, required=True, help="Observer longitude in degrees.")
    parser.add_argument("--observer_elevation_m", type=float, default=0.0, help="Observer elevation in metres.")
    parser.add_argument("--start_time_utc", type=str, required=True, help="Start time (ISO 8601 UTC).")
    parser.add_argument("--end_time_utc", type=str, required=True, help="End time (ISO 8601 UTC).")
    parser.add_argument("--min_altitude_deg", type=float, default=15.0, help="Minimum visible altitude in degrees.")
    parser.add_argument("--time_step_minutes", type=int, default=1, help="Sampling cadence in minutes.")
    parser.add_argument("--plot_output", type=str, default=None, help="Path to save visibility plot (optional).")
    args = parser.parse_args()

    with open(args.tle_file, "r") as f:
        satellites = json.load(f)

    # Skip decayed objects — their TLEs produce degenerate SGP4 states
    # when propagated far past the decay date.
    tle_list = [
        (sat["OBJECT_NAME"], sat["TLE_LINE1"], sat["TLE_LINE2"])
        for sat in satellites
        if sat.get("TLE_LINE1") and sat.get("TLE_LINE2") and not sat.get("DECAY_DATE")
    ]
    print(f"{len(tle_list)} active satellites loaded (decayed records excluded)")

    # Print visibility passes to stdout
    eph = load("de421.bsp")
    ts = load.timescale()
    for name, line1, line2 in tle_list:
        passes = get_satellite_visibility(
            line1, line2,
            args.observer_lat, args.observer_lon, args.observer_elevation_m,
            args.start_time_utc, args.end_time_utc,
            time_step_minutes=args.time_step_minutes,
            min_altitude_deg=args.min_altitude_deg,
        )
        if passes:
            # Batch illumination check across all visible epochs at once
            sat_sf = EarthSatellite(line1, line2, ts=ts)
            pass_times = ts.from_datetime([_parse_utc(t) for t, _, _ in passes])
            sunlit_flags = sat_sf.at(pass_times).is_sunlit(eph)
            print(f"\n{name}")
            for (t, alt, az), lit in zip(passes, sunlit_flags):
                status = "sunlit" if lit else "in shadow"
                print(f"  {t}  Alt: {alt:.1f}°  Az: {az:.1f}°  [{status}]")

    # Generate and optionally save visibility plot
    fig = plot_visibility(
        tle_list,
        args.observer_lat, args.observer_lon, args.observer_elevation_m,
        args.start_time_utc, args.end_time_utc,
        min_altitude_deg=args.min_altitude_deg,
        time_step_minutes=args.time_step_minutes,
        eph=eph,
    )
    if args.plot_output:
        fig.savefig(args.plot_output, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot_output}")
    else:
        plt.show()
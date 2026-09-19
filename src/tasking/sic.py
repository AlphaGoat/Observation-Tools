"""
Shannon Information Content (SIC) computation for satellite follow-up scheduling.

Implements the orbit information-gain metric from Hinze et al. (AMOS 2016):
  SIC = ½ (ln|P⁻_pos| − ln|P⁺_pos|)

where P is the 6×6 orbit error covariance in ECI (m, m/s units), updated via
a Kalman filter after one angle-only observation (az, el tracklet).

Two covariance initialisation modes are supported:
  "uniform"  — identical diagonal P for all satellites; SIC reflects geometry only.
  "tle_age"  — P scaled by TLE age; older TLEs yield larger uncertainty and
               higher scheduling priority.

Authors: Peter Thomas
Date: 2025-10-12
"""
import datetime as _dt
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Tuple

from skyfield.api import load, wgs84, EarthSatellite


# ─── Public type aliases ─────────────────────────────────────────────────────

CovarianceMode = Literal["uniform", "tle_age"]


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Sensor:
    """Ground-based optical telescope."""
    name: str
    lat: float                          # degrees, positive north
    lon: float                          # degrees, positive east
    elevation_m: float
    min_elevation_deg: float = 20.0     # hard visibility floor
    max_phase_angle_deg: float = 100.0  # illumination constraint
    min_lunar_distance_deg: float = 20.0
    exposure_time_s: float = 120.0      # tracklet duration
    readout_time_s: float = 10.0
    slew_rate_deg_s: float = 2.0        # maximum slew speed
    sigma_az_arcsec: float = 1.0        # astrometric noise, 1-sigma
    sigma_el_arcsec: float = 1.0


@dataclass
class SatelliteTarget:
    """
    GEO space object.  covariance_eci is a 6×6 matrix in ECI with units
    [m², m², m², (m/s)², (m/s)², (m/s)²] on the diagonal.
    It is populated by one of the init_covariance_* functions below.
    """
    name: str
    norad_id: int
    tle_line1: str
    tle_line2: str
    covariance_eci: np.ndarray = field(default_factory=lambda: np.eye(6))


# ─── Covariance initialisation ───────────────────────────────────────────────

def parse_tle_epoch(tle_line1: str) -> datetime:
    """
    Parse the TLE epoch from line 1 (columns 18–32, format YYDDD.DDDDDDDD).
    Returns a timezone-aware datetime in UTC.
    """
    epoch_str = tle_line1[18:32].strip()
    year_2d = int(epoch_str[:2])
    year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
    day_frac = float(epoch_str[2:])          # 1-indexed day of year
    jan1 = _dt.datetime(year, 1, 1, tzinfo=timezone.utc)
    return jan1 + timedelta(days=day_frac - 1.0)


def init_covariance_uniform(
    sigma_pos_m: float = 1000.0,
    sigma_vel_mps: float = 1.0,
) -> np.ndarray:
    """
    Return a diagonal 6×6 covariance matrix, identical for all satellites.
    SIC computed from this matrix reflects observation geometry only.

    Parameters
    ----------
    sigma_pos_m : float
        1-sigma position uncertainty in metres (default 1 km).
    sigma_vel_mps : float
        1-sigma velocity uncertainty in m/s (default 1 m/s).
    """
    diag = [sigma_pos_m ** 2] * 3 + [sigma_vel_mps ** 2] * 3
    return np.diag(diag).astype(float)


def init_covariance_tle_age_scaled(
    tle_line1: str,
    t_eval: datetime,
    sigma_pos_ref_m: float = 100.0,
    sigma_vel_ref_mps: float = 0.1,
    growth_rate_per_day: float = 2.0,
) -> np.ndarray:
    """
    Return a diagonal 6×6 covariance matrix scaled by the age of the TLE.

        σ_pos(age) = σ_pos_ref × (1 + growth_rate × age_days)

    Satellites with older TLEs receive larger P and therefore higher SIC,
    naturally prioritising them in the GA.

    Parameters
    ----------
    tle_line1 : str
        First line of the TLE (epoch in columns 18–32).
    t_eval : datetime
        Time at which the schedule is being built (typically t_start of night).
    sigma_pos_ref_m : float
        Reference 1-sigma position uncertainty at TLE epoch (default 100 m).
    sigma_vel_ref_mps : float
        Reference 1-sigma velocity uncertainty at TLE epoch (default 0.1 m/s).
    growth_rate_per_day : float
        Daily growth factor for position uncertainty (default 2 km/day).
    """
    if t_eval.tzinfo is None:
        t_eval = t_eval.replace(tzinfo=timezone.utc)
    t_epoch = parse_tle_epoch(tle_line1)
    age_days = max(0.0, (t_eval - t_epoch).total_seconds() / 86400.0)
    scale = 1.0 + growth_rate_per_day * age_days
    diag = [( sigma_pos_ref_m * scale) ** 2] * 3 + [(sigma_vel_ref_mps * scale) ** 2] * 3
    return np.diag(diag).astype(float)


# ─── Observation geometry (H matrix) ─────────────────────────────────────────

def _eci_to_azel(
    r_sat_km: np.ndarray,
    r_obs_km: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    gast_rad: float,
) -> Tuple[float, float]:
    """
    Convert a satellite ECI position to topocentric (az, el) at the observer.

    Uses a simple z-rotation by GAST for ECI→ECEF conversion, then standard
    ENU decomposition.  Accuracy is sufficient for the numerical Jacobian.

    Parameters
    ----------
    r_sat_km, r_obs_km : ndarray, shape (3,)
        ECI positions in kilometres.
    lat_deg, lon_deg : float
        Observer geodetic coordinates in degrees.
    gast_rad : float
        Greenwich Apparent Sidereal Time in radians.

    Returns
    -------
    (az_rad, el_rad)
    """
    rho_eci = r_sat_km - r_obs_km

    # ECI → ECEF (rotation around z-axis by GAST)
    c, s = np.cos(gast_rad), np.sin(gast_rad)
    rho_ecef = np.array([
        c * rho_eci[0] + s * rho_eci[1],
        -s * rho_eci[0] + c * rho_eci[1],
        rho_eci[2],
    ])

    # ECEF → ENU
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)

    E = -slo * rho_ecef[0] + clo * rho_ecef[1]
    N = -sl * clo * rho_ecef[0] - sl * slo * rho_ecef[1] + cl * rho_ecef[2]
    U =  cl * clo * rho_ecef[0] + cl * slo * rho_ecef[1] + sl * rho_ecef[2]

    rho_mag = np.sqrt(E ** 2 + N ** 2 + U ** 2)
    el = np.arcsin(np.clip(U / rho_mag, -1.0, 1.0))
    az = np.arctan2(E, N) % (2.0 * np.pi)
    return float(az), float(el)


def compute_observation_geometry(
    satellite: SatelliteTarget,
    sensor: Sensor,
    t_obs: datetime,
    ts=None,
    eps_m: float = 1.0,
) -> np.ndarray:
    """
    Compute the 2×6 observation geometry matrix H = ∂(az, el)/∂(rx, ry, rz, vx, vy, vz).

    Velocity columns (3–5) are zero for an instantaneous angle-only measurement.
    Position columns (0–2) are computed by central differences, perturbing each
    ECI position component by eps_m metres and measuring the change in (az, el).

    Units: H[i, j] in rad/m for j < 3; 0 for j ≥ 3.

    Parameters
    ----------
    satellite : SatelliteTarget
    sensor : Sensor
    t_obs : datetime
        UTC observation time (timezone-aware preferred).
    ts : Skyfield Timescale, optional
        Reuse an existing timescale to avoid repeated file I/O.
    eps_m : float
        Central-difference step size in metres (default 1 m).
    """
    if ts is None:
        ts = load.timescale()
    if t_obs.tzinfo is None:
        t_obs = t_obs.replace(tzinfo=timezone.utc)

    t_sf = ts.from_datetime(t_obs)
    sat_sf = EarthSatellite(satellite.tle_line1, satellite.tle_line2, ts=ts)
    obs_sf = wgs84.latlon(sensor.lat, sensor.lon, elevation_m=sensor.elevation_m)

    r_sat_km = sat_sf.at(t_sf).position.km          # (3,) ECI km
    r_obs_km = obs_sf.at(t_sf).position.km           # (3,) ECI km
    gast_rad = t_sf.gast * (2.0 * np.pi / 24.0)     # hours → rad

    eps_km = eps_m / 1000.0
    H = np.zeros((2, 6))
    for i in range(3):
        r_plus = r_sat_km.copy();  r_plus[i]  += eps_km
        r_minus = r_sat_km.copy(); r_minus[i] -= eps_km
        az_p, el_p = _eci_to_azel(r_plus,  r_obs_km, sensor.lat, sensor.lon, gast_rad)
        az_m, el_m = _eci_to_azel(r_minus, r_obs_km, sensor.lat, sensor.lon, gast_rad)
        H[0, i] = (az_p - az_m) / (2.0 * eps_m)    # rad/m
        H[1, i] = (el_p - el_m) / (2.0 * eps_m)    # rad/m
    return H


# ─── SIC computation ─────────────────────────────────────────────────────────

def compute_sic(
    satellite: SatelliteTarget,
    sensor: Sensor,
    t_obs: datetime,
    ts=None,
) -> float:
    """
    Shannon Information Content of one follow-up tracklet (Hinze et al. 2016).

        K   = P⁻ Hᵀ (W + H P⁻ Hᵀ)⁻¹
        P⁺  = (I − K H) P⁻
        SIC = ½ (ln|P⁻_pos| − ln|P⁺_pos|)

    where P_pos denotes the 3×3 position submatrix of P.

    A positive SIC indicates the tracklet reduces position uncertainty.
    Returns 0.0 if the Kalman update is degenerate (e.g. H is all-zero).

    Parameters
    ----------
    satellite : SatelliteTarget
        Must have covariance_eci set (call one of the init_covariance_* functions).
    sensor : Sensor
    t_obs : datetime
        UTC start of the tracklet.
    ts : Skyfield Timescale, optional
    """
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    W = np.diag([
        (sensor.sigma_az_arcsec * arcsec_to_rad) ** 2,
        (sensor.sigma_el_arcsec * arcsec_to_rad) ** 2,
    ])

    H = compute_observation_geometry(satellite, sensor, t_obs, ts=ts)
    P = satellite.covariance_eci

    S = W + H @ P @ H.T                             # innovation covariance (2×2)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return 0.0

    K = P @ H.T @ S_inv                             # Kalman gain (6×2)
    P_plus = (np.eye(6) - K @ H) @ P               # updated covariance (6×6)

    P_pos_before = P[:3, :3]
    P_pos_after  = P_plus[:3, :3]

    sign_b, logdet_b = np.linalg.slogdet(P_pos_before)
    sign_a, logdet_a = np.linalg.slogdet(P_pos_after)

    if sign_b <= 0 or sign_a <= 0:
        return 0.0
    return float(0.5 * (logdet_b - logdet_a))

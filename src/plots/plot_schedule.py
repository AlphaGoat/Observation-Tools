"""
plot_schedule.py — Schedule diagram for multi-telescope GEO observation tasking.

Builds a Gantt chart showing which sensor observes which satellite and when.
Can be driven by the GA scheduler (end-to-end) or by a pre-computed schedule
passed directly.

Usage
-----
# Run the GA and plot in one call:
    python src/plots/plot_schedule.py \
        --tles spacetracks.json \
        --t_start 2025-10-12T20:00:00Z \
        --t_end   2025-10-13T04:00:00Z \
        --covariance_mode tle_age \
        --out     src/plots/schedule.png

# Plot a schedule that was already computed in Python:
    from src.plots.plot_schedule import plot_schedule
    fig = plot_schedule(schedule, sensors, satellites)
    fig.savefig("schedule.png", dpi=150)

Authors: Peter Thomas
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tasking.sic import CovarianceMode, Sensor, SatelliteTarget
from tasking.scheduler import (
    Chromosome,
    Gene,
    ObservationOpportunity,
    scheduler_optimizer,
)


# ── Colour helpers ────────────────────────────────────────────────────────────

# Distinct palette up to 20 satellites; cycles after that.
_CMAP = plt.get_cmap("tab20")


def _sat_colour(sat_idx: int):
    return _CMAP(sat_idx % 20)


# ── Core plot function ────────────────────────────────────────────────────────

def plot_schedule(
    schedule: Chromosome,
    sensors: List[Sensor],
    satellites: List[SatelliteTarget],
    t_start: Optional[datetime] = None,
    t_end: Optional[datetime] = None,
    title: str = "Observation Schedule",
) -> plt.Figure:
    """
    Gantt-chart visualisation of an observation schedule.

    One horizontal lane per sensor.  Each observation is drawn as a coloured
    bar whose width equals the sensor's exposure time.  Satellites are
    distinguished by colour; the bar is labelled with the satellite name and
    its SIC contribution.

    Unscheduled satellites are listed below the chart as a legend entry with
    a cross-hatched patch.

    Parameters
    ----------
    schedule : Chromosome
        One entry per satellite: ObservationOpportunity or None (not scheduled).
    sensors : list of Sensor
    satellites : list of SatelliteTarget
    t_start, t_end : datetime, optional
        Axis limits.  Inferred from the schedule if omitted.
    title : str

    Returns
    -------
    matplotlib Figure
    """
    scheduled_genes: List[ObservationOpportunity] = [
        g for g in schedule if g is not None
    ]

    # Axis time bounds
    if t_start is None and scheduled_genes:
        t_start = min(g.t_obs for g in scheduled_genes) - timedelta(minutes=30)
    if t_end is None and scheduled_genes:
        t_end = (
            max(
                g.t_obs + timedelta(seconds=sensors[g.sensor_idx].exposure_time_s)
                for g in scheduled_genes
            )
            + timedelta(minutes=30)
        )

    n_sensors = len(sensors)
    row_h = 0.7
    fig_h = max(3.5, n_sensors * 1.4 + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    scheduled_sat_indices = {g.satellite_idx for g in scheduled_genes}
    unscheduled_names: List[str] = [
        satellites[i].name
        for i in range(len(satellites))
        if i not in scheduled_sat_indices
    ]

    # ── Bars ─────────────────────────────────────────────────────────────────
    for gene in scheduled_genes:
        sensor = sensors[gene.sensor_idx]
        sat = satellites[gene.satellite_idx]
        dur_h = sensor.exposure_time_s / 3600.0
        left = mdates.date2num(gene.t_obs)
        color = _sat_colour(gene.satellite_idx)

        bar = ax.barh(
            gene.sensor_idx,
            dur_h,
            left=left,
            height=row_h,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )

        # Readout window — narrow grey stripe immediately after exposure
        readout_left = left + dur_h
        readout_dur_h = sensor.readout_time_s / 3600.0
        ax.barh(
            gene.sensor_idx,
            readout_dur_h,
            left=readout_left,
            height=row_h,
            color=color,
            alpha=0.25,
            edgecolor="none",
        )

        # Label inside the bar (skip if bar too narrow to fit)
        mid = mdates.date2num(gene.t_obs + timedelta(seconds=sensor.exposure_time_s / 2))
        if dur_h / ((mdates.date2num(t_end) - mdates.date2num(t_start)) or 1) > 0.03:
            ax.text(
                mid,
                gene.sensor_idx,
                f"{sat.name}\nSIC {gene.sic:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                clip_on=True,
            )

    # ── Axes decoration ───────────────────────────────────────────────────────
    ax.set_yticks(range(n_sensors))
    ax.set_yticklabels([s.name for s in sensors], fontsize=9)
    ax.set_ylim(-0.6, n_sensors - 0.4)

    if t_start and t_end:
        ax.set_xlim(mdates.date2num(t_start), mdates.date2num(t_end))

    ax.xaxis_date()
    span_h = (t_end - t_start).total_seconds() / 3600.0 if t_start and t_end else 8
    if span_h <= 2:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Time (UTC)", fontsize=9)

    total_sic = sum(g.sic for g in scheduled_genes)
    ax.set_title(
        f"{title}\n"
        f"{len(scheduled_genes)}/{len(schedule)} satellites scheduled  ·  "
        f"Total SIC = {total_sic:.3f}",
        fontsize=10,
        pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_sat_colour(i), label=satellites[i].name)
        for i in sorted(scheduled_sat_indices)
    ]
    if unscheduled_names:
        legend_patches.append(
            mpatches.Patch(
                facecolor="lightgrey",
                edgecolor="grey",
                hatch="//",
                label=f"Not scheduled: {', '.join(unscheduled_names)}",
            )
        )

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=7,
            ncol=max(1, len(legend_patches) // 8),
            framealpha=0.8,
        )

    # ── SIC summary annotation ────────────────────────────────────────────────
    summary_lines = [f"{'Satellite':<22} {'Sensor':<18} {'t_obs':>8}  SIC"]
    summary_lines.append("─" * 60)
    for i, gene in enumerate(schedule):
        sat = satellites[i]
        if gene is None:
            summary_lines.append(f"{sat.name:<22} {'—':<18} {'—':>8}  —")
        else:
            sen = sensors[gene.sensor_idx]
            summary_lines.append(
                f"{sat.name:<22} {sen.name:<18} "
                f"{gene.t_obs.strftime('%H:%M'):>8}  {gene.sic:.4f}"
            )
    summary_lines.append("─" * 60)
    summary_lines.append(f"{'Total SIC':<50} {total_sic:.4f}")
    print("\n".join(summary_lines))

    plt.tight_layout()
    return fig


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def _build_demo_sensors() -> List[Sensor]:
    """Three geographically distributed optical telescopes."""
    return [
        Sensor(
            name="Zimmerwald",
            lat=46.877,
            lon=7.465,
            elevation_m=907,
            min_elevation_deg=20.0,
            exposure_time_s=120.0,
            readout_time_s=10.0,
        ),
        Sensor(
            name="Tenerife",
            lat=28.299,
            lon=-16.510,
            elevation_m=2390,
            min_elevation_deg=20.0,
            exposure_time_s=120.0,
            readout_time_s=10.0,
        ),
        Sensor(
            name="Canberra",
            lat=-35.288,
            lon=149.194,
            elevation_m=812,
            min_elevation_deg=20.0,
            exposure_time_s=120.0,
            readout_time_s=10.0,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a GA-optimised GEO observation schedule as a Gantt chart."
    )
    parser.add_argument(
        "--tles", required=True,
        help="JSON TLE file (Space-Track OMM format or list of "
             "{OBJECT_NAME, NORAD_CAT_ID, TLE_LINE1, TLE_LINE2}).",
    )
    parser.add_argument(
        "--sensors", default=None,
        help="Optional JSON file with sensor definitions. "
             "Uses built-in demo sensors (Zimmerwald, Tenerife, Canberra) if omitted.",
    )
    parser.add_argument(
        "--t_start", required=True,
        help="Observation window start, ISO 8601 UTC (e.g. 2025-10-12T20:00:00Z).",
    )
    parser.add_argument(
        "--t_end", required=True,
        help="Observation window end, ISO 8601 UTC.",
    )
    parser.add_argument(
        "--covariance_mode", choices=["uniform", "tle_age"], default="uniform",
        help="Covariance initialisation for SIC (default: uniform).",
    )
    parser.add_argument(
        "--max_sats", type=int, default=20,
        help="Cap on number of satellites to schedule (default 20).",
    )
    parser.add_argument(
        "--generations", type=int, default=600,
        help="GA generations (default 600).",
    )
    parser.add_argument(
        "--pop_size", type=int, default=50,
        help="GA population size (default 50).",
    )
    parser.add_argument(
        "--time_step", type=int, default=10,
        help="Legal table time resolution in minutes (default 10).",
    )
    parser.add_argument(
        "--out", default="src/plots/schedule.png",
        help="Output path for the schedule diagram (default: src/plots/schedule.png).",
    )
    parser.add_argument(
        "--title", default="GEO Observation Schedule",
        help="Plot title.",
    )
    args = parser.parse_args()

    # Load sensors
    if args.sensors:
        with open(args.sensors) as f:
            sensors = [Sensor(**cfg) for cfg in json.load(f)]
    else:
        sensors = _build_demo_sensors()
        print(f"[plot_schedule] using built-in demo sensors: "
              f"{[s.name for s in sensors]}")

    # Load TLEs (Space-Track OMM JSON)
    with open(args.tles) as f:
        tle_records = json.load(f)

    # Filter to records that have TLE lines
    tle_records = [r for r in tle_records if r.get("TLE_LINE1") and r.get("TLE_LINE2")]
    tle_records = tle_records[: args.max_sats]
    satellites = [
        SatelliteTarget(
            name=rec["OBJECT_NAME"],
            norad_id=int(rec.get("NORAD_CAT_ID", 0)),
            tle_line1=rec["TLE_LINE1"],
            tle_line2=rec["TLE_LINE2"],
        )
        for rec in tle_records
    ]
    print(f"[plot_schedule] scheduling {len(satellites)} satellites "
          f"across {len(sensors)} sensors")

    t_start = _parse_utc(args.t_start)
    t_end   = _parse_utc(args.t_end)

    # Run GA
    print("Optimizing schedule...")
    schedule = scheduler_optimizer(
        sensors,
        satellites,
        t_start,
        t_end,
        covariance_mode=args.covariance_mode,
        num_generations=args.generations,
        population_size=args.pop_size,
        time_step_minutes=args.time_step,
    )
    print("Schedule calculated. Plotting schedule...")

    # Plot
    matplotlib.use("Agg")
    fig = plot_schedule(schedule, sensors, satellites, t_start, t_end, title=args.title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot_schedule] diagram saved to {out_path}")


if __name__ == "__main__":
    main()

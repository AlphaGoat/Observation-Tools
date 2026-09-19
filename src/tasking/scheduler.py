"""
Genetic algorithm scheduler for multi-telescope GEO satellite observation tasking.

Fitness metric: Shannon Information Content (SIC) summed across all scheduled
follow-up tracklets.  Each satellite is observed at most once per schedule.
GA architecture follows Hinze et al. (AMOS 2016): rank-based selection with
stochastic universal sampling, conditionally accepted crossover and mutation,
and elitism.

Authors: Peter Thomas
Date: 2025-10-12
"""
import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from skyfield.api import load, wgs84, EarthSatellite

from .sic import (
    CovarianceMode,
    Sensor,
    SatelliteTarget,
    compute_sic,
    init_covariance_uniform,
    init_covariance_tle_age_scaled,
)


# ─── Core data types ─────────────────────────────────────────────────────────

@dataclass
class ObservationOpportunity:
    """One valid (satellite, sensor, time) triple with its pre-computed SIC."""
    satellite_idx: int
    sensor_idx: int
    t_obs: datetime        # UTC start of tracklet
    sic: float


# One slot per satellite: which opportunity it takes, or None (not observed).
Gene       = Optional[ObservationOpportunity]
Chromosome = List[Gene]                         # len == len(satellites)

# [sat_idx][sen_idx] → list of opportunities sorted by SIC descending
LegalTable = Dict[int, Dict[int, List[ObservationOpportunity]]]


# ─── Constraint helpers ───────────────────────────────────────────────────────

def _conflict(gene_a: Gene, gene_b: Gene, sensors: List[Sensor]) -> bool:
    """True if two genes use the same sensor with overlapping windows."""
    if gene_a is None or gene_b is None:
        return False
    if gene_a.sensor_idx != gene_b.sensor_idx:
        return False
    sensor = sensors[gene_a.sensor_idx]
    dur = timedelta(seconds=sensor.exposure_time_s + sensor.readout_time_s)
    return gene_a.t_obs < gene_b.t_obs + dur and gene_b.t_obs < gene_a.t_obs + dur


def _slew_clears(
    gene_prev: Gene,
    gene_next: Gene,
    sensors: List[Sensor],
    r_sat: Dict[Tuple[int, int], np.ndarray],
) -> bool:
    """
    True if there is enough time between consecutive observations on the same
    sensor to complete the slew between the two satellite positions.
    r_sat maps (sat_idx, sen_idx) → (3,) ECI position km at observation time.
    (Used as a secondary check; omit r_sat lookup by passing {} to skip.)
    """
    if gene_prev is None or gene_next is None:
        return True
    if gene_prev.sensor_idx != gene_next.sensor_idx:
        return True
    sensor = sensors[gene_prev.sensor_idx]
    gap_s = (gene_next.t_obs - gene_prev.t_obs).total_seconds() - (
        sensor.exposure_time_s + sensor.readout_time_s
    )
    if gap_s <= 0:
        return False
    return True  # detailed angle check omitted; conflict() is the hard gate


# ─── Fitness ─────────────────────────────────────────────────────────────────

def _fitness(chromosome: Chromosome, sensors: List[Sensor]) -> float:
    """
    F = Σ SIC(gene)  −  10 × Σ SIC(conflicting genes)

    The large penalty keeps conflicting schedules in the population but far
    below any valid schedule, letting the GA recover them via crossover.
    """
    genes = [g for g in chromosome if g is not None]
    total = sum(g.sic for g in genes)
    penalty = 0.0
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if _conflict(genes[i], genes[j], sensors):
                penalty += genes[i].sic + genes[j].sic
    return total - 10.0 * penalty


# ─── Scheduler ───────────────────────────────────────────────────────────────

class GeneticAlgorithmScheduler:
    """
    Multi-telescope GEO follow-up scheduler.

    Parameters
    ----------
    sensors : list of Sensor
    satellites : list of SatelliteTarget
        covariance_eci will be overwritten according to covariance_mode.
    t_start, t_end : datetime
        Observation window (UTC).
    covariance_mode : {"uniform", "tle_age"}
        How to initialise P for each satellite.
    population_size : int
        Number of chromosomes per generation (default 50).
    num_generations : int
        Maximum GA iterations (default 600).
    crossover_prob : float
        Per-pair crossover probability (default 0.6).
    mutation_prob : float
        Per-gene mutation probability (default 0.1).
    e_max : float
        Baker rank-selection parameter E_max ∈ (1, 2) (default 1.1).
    time_step_minutes : int
        Time resolution of the legal observation table (default 10 min).
    stagnation_limit : int
        Stop early if best fitness does not improve for this many generations.
    """

    def __init__(
        self,
        sensors: List[Sensor],
        satellites: List[SatelliteTarget],
        t_start: datetime,
        t_end: datetime,
        covariance_mode: CovarianceMode = "uniform",
        population_size: int = 50,
        num_generations: int = 600,
        crossover_prob: float = 0.6,
        mutation_prob: float = 0.1,
        e_max: float = 1.1,
        time_step_minutes: int = 10,
        stagnation_limit: int = 100,
    ):
        self.sensors = sensors
        self.satellites = satellites
        self.t_start = t_start if t_start.tzinfo else t_start.replace(tzinfo=timezone.utc)
        self.t_end   = t_end   if t_end.tzinfo   else t_end.replace(tzinfo=timezone.utc)
        self.covariance_mode = covariance_mode
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob  = crossover_prob
        self.mutation_prob   = mutation_prob
        self.e_max           = e_max
        self.time_step_minutes = time_step_minutes
        self.stagnation_limit  = stagnation_limit

        self.ts  = load.timescale()
        self.eph = load("de421.bsp")

        self._init_covariances()
        print(f"[scheduler] building legal observation table …")
        self.legal_table: LegalTable = self._build_legal_table()
        n_opps = sum(
            len(v) for sat in self.legal_table.values() for v in sat.values()
        )
        print(f"[scheduler] {n_opps} valid opportunities across "
              f"{len(satellites)} satellites × {len(sensors)} sensors")

        self.population: List[Chromosome] = self._init_population()
        self.best_chromosome: Chromosome = max(
            self.population, key=lambda c: _fitness(c, self.sensors)
        )

    # ── Covariance initialisation ────────────────────────────────────────────

    def _init_covariances(self) -> None:
        for sat in self.satellites:
            if self.covariance_mode == "uniform":
                sat.covariance_eci = init_covariance_uniform()
            else:
                sat.covariance_eci = init_covariance_tle_age_scaled(
                    sat.tle_line1, self.t_start
                )

    # ── Legal observation table ──────────────────────────────────────────────

    def _build_legal_table(self) -> LegalTable:
        """
        Pre-compute all valid (satellite, sensor, t_obs) triples and their SIC.

        Constraints checked (vectorised over the full time array per sat/sensor):
          1. Satellite is sunlit.
          2. Altitude above sensor horizon ≥ min_elevation_deg.
          3. Phase angle (angle at satellite between observer and sun) ≤ max_phase_angle_deg.
          4. Angular separation between satellite and moon ≥ min_lunar_distance_deg.
        """
        table: LegalTable = {
            i: {j: [] for j in range(len(self.sensors))}
            for i in range(len(self.satellites))
        }

        # Build Skyfield time array
        t0 = self.ts.from_datetime(self.t_start)
        t1 = self.ts.from_datetime(self.t_end)
        n_steps = max(
            2,
            int((self.t_end - self.t_start).total_seconds()
                / (self.time_step_minutes * 60)) + 1,
        )
        times_sf = self.ts.tt_jd(np.linspace(t0.tt, t1.tt, n_steps))

        # Corresponding Python datetimes (used in ObservationOpportunity)
        t_obs_list = [
            self.t_start + timedelta(minutes=i * self.time_step_minutes)
            for i in range(n_steps)
        ]

        # Pre-compute sun and moon ECI positions (same for all satellites)
        earth = self.eph["earth"]
        r_sun  = earth.at(times_sf).observe(self.eph["sun"]).position.km   # (3, N)
        r_moon = earth.at(times_sf).observe(self.eph["moon"]).position.km  # (3, N)

        for sat_idx, sat in enumerate(self.satellites):
            sat_sf    = EarthSatellite(sat.tle_line1, sat.tle_line2, ts=self.ts)
            sat_state = sat_sf.at(times_sf)
            r_sat     = sat_state.position.km                              # (3, N)
            sunlit    = sat_state.is_sunlit(self.eph)                      # (N,) bool

            for sen_idx, sensor in enumerate(self.sensors):
                obs_sf = wgs84.latlon(
                    sensor.lat, sensor.lon, elevation_m=sensor.elevation_m
                )
                alt_deg = (sat_sf - obs_sf).at(times_sf).altaz()[0].degrees  # (N,)
                r_obs   = obs_sf.at(times_sf).position.km                     # (3, N)

                # Phase angle at satellite (vectorised)
                to_obs_s = r_obs - r_sat                                    # (3, N) sat→obs
                to_sun_s = r_sun - r_sat                                    # (3, N) sat→sun
                norm_obs = np.linalg.norm(to_obs_s, axis=0)
                norm_sun = np.linalg.norm(to_sun_s, axis=0)
                cos_phase = (
                    np.einsum("ij,ij->j", to_obs_s, to_sun_s)
                    / np.where(norm_obs * norm_sun > 0, norm_obs * norm_sun, 1.0)
                )
                phase_deg = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))

                # Lunar distance from observer (vectorised)
                to_moon_o = r_moon - r_obs                                  # (3, N) obs→moon
                to_sat_o  = r_sat  - r_obs                                  # (3, N) obs→sat
                norm_moon = np.linalg.norm(to_moon_o, axis=0)
                norm_sat  = np.linalg.norm(to_sat_o,  axis=0)
                cos_lunar = (
                    np.einsum("ij,ij->j", to_moon_o, to_sat_o)
                    / np.where(norm_moon * norm_sat > 0, norm_moon * norm_sat, 1.0)
                )
                lunar_deg = np.degrees(np.arccos(np.clip(cos_lunar, -1.0, 1.0)))

                valid_mask = (
                    sunlit
                    & (alt_deg   >= sensor.min_elevation_deg)
                    & (phase_deg <= sensor.max_phase_angle_deg)
                    & (lunar_deg >= sensor.min_lunar_distance_deg)
                )

                for i in np.where(valid_mask)[0]:
                    sic_val = compute_sic(sat, sensor, t_obs_list[i], ts=self.ts)
                    if sic_val > 0.0:
                        table[sat_idx][sen_idx].append(
                            ObservationOpportunity(sat_idx, sen_idx, t_obs_list[i], sic_val)
                        )

        # Sort each list best-first
        for sat_idx in table:
            for sen_idx in table[sat_idx]:
                table[sat_idx][sen_idx].sort(key=lambda o: o.sic, reverse=True)

        return table

    def _all_opportunities(self, sat_idx: int) -> List[ObservationOpportunity]:
        """Flat list of all opportunities for one satellite across all sensors."""
        return [
            opp
            for sen_opps in self.legal_table[sat_idx].values()
            for opp in sen_opps
        ]

    # ── Population initialisation ────────────────────────────────────────────

    def _random_chromosome(self) -> Chromosome:
        chromosome: Chromosome = []
        for sat_idx in range(len(self.satellites)):
            opps = self._all_opportunities(sat_idx)
            chromosome.append(random.choice(opps) if opps else None)
        return chromosome

    def _priority_chromosome(self) -> Chromosome:
        """
        Seed chromosome: assign opportunities in priority order (largest
        position uncertainty first) choosing the highest-SIC slot that does
        not conflict with already-assigned observations.
        """
        priority_order = sorted(
            range(len(self.satellites)),
            key=lambda i: float(np.trace(self.satellites[i].covariance_eci[:3, :3])),
            reverse=True,
        )
        chromosome: Chromosome = [None] * len(self.satellites)
        # Track assigned windows: list of (sensor_idx, t_start, t_end)
        assigned: List[Tuple[int, datetime, datetime]] = []

        for sat_idx in priority_order:
            opps = sorted(
                self._all_opportunities(sat_idx), key=lambda o: o.sic, reverse=True
            )
            for opp in opps:
                sensor = self.sensors[opp.sensor_idx]
                dur = timedelta(seconds=sensor.exposure_time_s + sensor.readout_time_s)
                opp_end = opp.t_obs + dur
                clash = any(
                    s == opp.sensor_idx and opp.t_obs < end and opp_end > start
                    for s, start, end in assigned
                )
                if not clash:
                    chromosome[sat_idx] = opp
                    assigned.append((opp.sensor_idx, opp.t_obs, opp_end))
                    break

        return chromosome

    def _init_population(self) -> List[Chromosome]:
        pop = [self._priority_chromosome()]
        pop += [self._random_chromosome() for _ in range(self.population_size - 1)]
        return pop

    # ── GA operators ─────────────────────────────────────────────────────────

    def _rank_select(self) -> List[int]:
        """
        Baker (1985) rank-based selection with stochastic universal sampling.
        Returns population_size indices into self.population.
        """
        n = len(self.population)
        fitnesses = [_fitness(c, self.sensors) for c in self.population]
        ranks = np.argsort(np.argsort(fitnesses)) + 1          # 1=worst, n=best
        e_min = 2.0 - self.e_max
        expected = e_min + (self.e_max - e_min) * (ranks - 1) / max(n - 1, 1)
        probs = expected / expected.sum()

        # Stochastic universal sampling
        cumsum = np.cumsum(probs)
        step   = 1.0 / n
        ptr    = random.uniform(0.0, step)
        selected: List[int] = []
        for i, cs in enumerate(cumsum):
            while ptr <= cs and len(selected) < n:
                selected.append(i)
                ptr += step

        random.shuffle(selected)
        return selected[:n]

    def _crossover(
        self,
        parent_a: Chromosome,
        parent_b: Chromosome,
        mean_fitness: float,
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Swap gene j (one satellite's opportunity) between two parents.
        Accept the swap only if at least one child exceeds mean_fitness.
        (Hinze 2016 acceptance criterion.)
        """
        if random.random() > self.crossover_prob:
            return parent_a, parent_b

        j = random.randrange(len(parent_a))
        child_a = parent_a.copy()
        child_b = parent_b.copy()
        child_a[j], child_b[j] = child_b[j], child_a[j]

        fa = _fitness(child_a, self.sensors)
        fb = _fitness(child_b, self.sensors)
        if max(fa, fb) > mean_fitness:
            return child_a, child_b
        return parent_a, parent_b

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        For each gene with probability mutation_prob, replace it with a
        different opportunity from the legal table — accepted only if the
        new SIC is higher than the current gene's SIC.
        """
        for sat_idx in range(len(chromosome)):
            if random.random() > self.mutation_prob:
                continue
            opps = self._all_opportunities(sat_idx)
            if not opps:
                continue
            candidate = random.choice(opps)
            current_sic = chromosome[sat_idx].sic if chromosome[sat_idx] is not None else 0.0
            if candidate.sic > current_sic:
                chromosome[sat_idx] = candidate
        return chromosome

    # ── Main loop ────────────────────────────────────────────────────────────

    def optimize(self) -> Chromosome:
        """
        Run the GA and return the best chromosome found.

        Terminates after num_generations iterations or after stagnation_limit
        consecutive generations without improvement in best fitness.
        """
        best_fitness = _fitness(self.best_chromosome, self.sensors)
        stagnation   = 0

        for gen in range(self.num_generations):
            fitnesses    = [_fitness(c, self.sensors) for c in self.population]
            mean_fitness = float(np.mean(fitnesses))
            best_idx     = int(np.argmax(fitnesses))

            if fitnesses[best_idx] > best_fitness:
                best_fitness         = fitnesses[best_idx]
                self.best_chromosome = copy.deepcopy(self.population[best_idx])
                stagnation           = 0
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit:
                print(f"[scheduler] early stop at generation {gen} "
                      f"(no improvement for {stagnation} generations)")
                break

            if gen % 50 == 0:
                scheduled = sum(1 for g in self.best_chromosome if g is not None)
                print(f"[scheduler] gen {gen:4d}  best_fitness={best_fitness:.4f}  "
                      f"scheduled={scheduled}/{len(self.satellites)}")

            # Selection
            indices  = self._rank_select()
            selected = [copy.deepcopy(self.population[i]) for i in indices]

            # Crossover (pairs)
            next_gen: List[Chromosome] = []
            for i in range(0, len(selected) - 1, 2):
                ca, cb = self._crossover(selected[i], selected[i + 1], mean_fitness)
                next_gen.extend([ca, cb])
            if len(selected) % 2 == 1:
                next_gen.append(selected[-1])

            # Mutation
            next_gen = [self._mutate(c) for c in next_gen]

            # Elitism: overwrite worst individual with current best
            next_gen[0] = copy.deepcopy(self.best_chromosome)
            self.population = next_gen

        return self.best_chromosome


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def scheduler_optimizer(
    sensors: List[Sensor],
    satellites: List[SatelliteTarget],
    t_start: datetime,
    t_end: datetime,
    covariance_mode: CovarianceMode = "uniform",
    **kwargs,
) -> Chromosome:
    """
    One-call interface: build the scheduler and run the GA.

    Returns the best chromosome (List[Optional[ObservationOpportunity]]).
    """
    ga = GeneticAlgorithmScheduler(
        sensors, satellites, t_start, t_end,
        covariance_mode=covariance_mode, **kwargs
    )
    return ga.optimize()


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_schedule(
    schedule: Chromosome,
    sensors: List[Sensor],
    satellites: List[SatelliteTarget],
) -> "plt.Figure":
    """
    Gantt-chart visualisation of an observation schedule.

    One row per sensor; horizontal bars coloured by satellite index, labelled
    with the satellite name and the SIC value of that observation.

    Parameters
    ----------
    schedule : Chromosome
        Output of GeneticAlgorithmScheduler.optimize().
    sensors : list of Sensor
    satellites : list of SatelliteTarget

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, max(3, len(sensors) * 1.2)))

    cmap   = plt.get_cmap("tab20")
    y_ticks = list(range(len(sensors)))
    colors  = [cmap(i % 20) for i in range(len(satellites))]

    for gene in schedule:
        if gene is None:
            continue
        sensor  = sensors[gene.sensor_idx]
        sat     = satellites[gene.satellite_idx]
        color   = colors[gene.satellite_idx]
        dur     = timedelta(seconds=sensor.exposure_time_s)
        t_start = gene.t_obs
        t_end   = t_start + dur

        ax.barh(
            gene.sensor_idx,
            (t_end - t_start).total_seconds() / 3600.0,
            left=mdates.date2num(t_start),
            height=0.6,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            mdates.date2num(t_start + dur / 2),
            gene.sensor_idx,
            f"{sat.name}\n{gene.sic:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([s.name for s in sensors])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    fig.autofmt_xdate()

    scheduled = [g for g in schedule if g is not None]
    total_sic = sum(g.sic for g in scheduled)
    ax.set_title(
        f"Observation Schedule  |  "
        f"{len(scheduled)}/{len(schedule)} satellites scheduled  |  "
        f"Total SIC = {total_sic:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Time (UTC)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GA-based multi-telescope GEO satellite observation scheduler."
    )
    parser.add_argument(
        "--sensors", required=True,
        help="Path to JSON file with sensor definitions. "
             "Each entry: {name, lat, lon, elevation_m, [optional fields…]}.",
    )
    parser.add_argument(
        "--tles", required=True,
        help="Path to JSON TLE file "
             "(list of {OBJECT_NAME, NORAD_CAT_ID, TLE_LINE1, TLE_LINE2}).",
    )
    parser.add_argument(
        "--t_start", required=True,
        help="Observation window start (ISO 8601 UTC, e.g. 2025-04-08T20:00:00Z).",
    )
    parser.add_argument(
        "--t_end", required=True,
        help="Observation window end (ISO 8601 UTC).",
    )
    parser.add_argument(
        "--covariance_mode", choices=["uniform", "tle_age"], default="uniform",
        help="Covariance initialisation mode (default: uniform). "
             "'uniform'  — identical 1 km / 1 m/s diagonal P for all satellites. "
             "'tle_age'  — P scaled by time since TLE epoch; older TLEs → higher priority.",
    )
    parser.add_argument(
        "--generations", type=int, default=600,
        help="Maximum GA generations (default 600).",
    )
    parser.add_argument(
        "--pop_size", type=int, default=50,
        help="Population size (default 50).",
    )
    parser.add_argument(
        "--time_step", type=int, default=10,
        help="Legal table time resolution in minutes (default 10).",
    )
    parser.add_argument(
        "--plot_output", default=None,
        help="Save schedule Gantt chart to this path (optional).",
    )
    args = parser.parse_args()

    # Load sensors
    with open(args.sensors) as f:
        sensor_configs = json.load(f)
    sensors = [Sensor(**cfg) for cfg in sensor_configs]

    # Load TLEs
    with open(args.tles) as f:
        tle_records = json.load(f)
    satellites = [
        SatelliteTarget(
            name=rec["OBJECT_NAME"],
            norad_id=int(rec.get("NORAD_CAT_ID", 0)),
            tle_line1=rec["TLE_LINE1"],
            tle_line2=rec["TLE_LINE2"],
        )
        for rec in tle_records
    ]

    # Parse times
    def _parse_utc(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)

    t_start = _parse_utc(args.t_start)
    t_end   = _parse_utc(args.t_end)

    # Run
    schedule = scheduler_optimizer(
        sensors, satellites, t_start, t_end,
        covariance_mode=args.covariance_mode,
        num_generations=args.generations,
        population_size=args.pop_size,
        time_step_minutes=args.time_step,
    )

    # Print results
    print("\n── Optimised Schedule ──────────────────────────────────")
    total_sic = 0.0
    for sat_idx, gene in enumerate(schedule):
        sat = satellites[sat_idx]
        if gene is None:
            print(f"  {sat.name:<20}  NOT SCHEDULED")
        else:
            sensor = sensors[gene.sensor_idx]
            print(
                f"  {sat.name:<20}  sensor={sensor.name:<15}  "
                f"t={gene.t_obs.strftime('%H:%M')} UTC  SIC={gene.sic:.4f}"
            )
            total_sic += gene.sic
    scheduled = sum(1 for g in schedule if g is not None)
    print(f"\n  Total: {scheduled}/{len(satellites)} scheduled  |  "
          f"Total SIC = {total_sic:.4f}")

    # Plot
    if args.plot_output:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_schedule(schedule, sensors, satellites)
        fig.savefig(args.plot_output, dpi=150, bbox_inches="tight")
        print(f"\n  Schedule plot saved to {args.plot_output}")

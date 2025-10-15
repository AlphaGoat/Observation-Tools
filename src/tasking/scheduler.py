"""
Scheduler module for tasking observations.

Author: Peter Thomas
Date: 2025-10-12
"""
import copy
import random
from typing import List, Tuple
from skyfield.api import EarthSatellite


Chromosome = List[Tuple[Sensor, EarthSatellite, int]]


class GeneticAlgorithmScheduler:
    def __init__(self, sensors, satellites, t_start, t_end, num_generations=1000):
        self.sensors = sensors
        self.satellites = satellites
        self.t_start = t_start
        self.t_end = t_end

        self.num_generations = num_generations

        # Table to hold legal observations
        self.legal_observation_table = self._get_legal_observation_table()

        # Generate a chromosome representation of the scheduling problem
        self.chromosome = self._generate_naive_chromosome()

    def optimize(self):
        """
        Optimize the scheduling of observations to maximize coverage and efficiency.
        """
        for _ in range(self.num_generations):
            pass
        pass

    def _vertical_crossover(self, chromosome_0: Chromosome, chromosome_1: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform vertical crossover operation on chromosome. Vertical crossover works
        by selecting a site at random in chromosome 0 and swapping it with a site
        in chromosome 1 where the same task (observing same object) is being performed.
        """
        # Select a random site in chromosome 0
        site_idx_0 = random.choice(list(range(len(chromosome_0))))

        # Get object under observation at randomly chosen site
        _, sat, _ = chromosome_0[site_idx_0]

        # Get location of object observation in chromosome 1
        site_idx_1 = [idx for idx in range(len(chromosome_1)) if chromosome_1[idx][1] == sat][0]

        # Switch sites in chromosomes
        # NOTE: May just swap satellite under observation instead of entire "site" in future
        site_0 = chromosome_0[site_idx_0]
        chromosome_0[site_idx_0] = chromosome_1[site_idx_1]
        chromosome_1[site_idx_1] = site_0

        return chromosome_0, chromosome_1

    def _horizontal_crossover(self, chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform horizontal crossover operation on chromosome. Horizontal crossover
        works by selecting two sites on the same chromosome and swapping same task
        (observing samee object)
        """
        new_chromosome = copy.deepcopy(chromosome)
        idx0 = random.choice(list(range(len(chromosome))))
        sensor0, sat0, timeslot0 = chromosome[idx0]
        idx1 = random.choice(list(range(len(chromosome))).remove(idx0))
        sensor1, sat1, timeslot1 = chromosome[idx1]
        new_chromosome[idx0] = (sensor0, sat1, timeslot0)
        new_chromosome[idx1] = (sensor1, sat0, timeslot1)
        return new_chromosome

    def _generate_naive_chromosome(self) -> Chromosome:
        """
        Generate a chromosome representation of the scheduling problem.
        """
        chromosome = []
        num_times_observed = {sat: 0 for sat in self.satellites}

        for sensor_idx, sensor in enumerate(self.sensors):
            # Get number of available time windows for this sensor
            go_to_next_sensor = False
            num_time_windows = self._get_number_sensor_time_windows(sensor, self.t_start, self.t_end)
            curr_time_window = 0

            for sat_idx, sat in enumerate(self.satellites):
                sat = self.satellites[sat_idx]

                # Want to generate at least three observation slots per satellite
                for _ in range(num_times_observed[sat], 3):
                    if curr_time_window >= num_time_windows:
                        go_to_next_sensor = True
                        break
                    chromosome.append((sensor, sat, curr_time_window))
                    curr_time_window += 1
                    num_times_observed[sat] += 1

                if go_to_next_sensor:
                    break

        return chromosome

    def _get_legal_observation_table(self):
        """
        Generate a table of legal observation operations 
        """
        pass

    def _get_number_sensor_time_windows(self, sensor, t_start, t_end):
        """
        Get available time windows for a sensor between t_start and t_end.
        """
        t_total = (t_end - t_start).total_seconds() # in seconds
        num_slots = int(t_total / ((sensor.exposure_time + sensor.readout_time) * self.num_frames_per_observation))

        return num_slots

    def _calc_fitness_criterion(self, chromosome: Chromosome) -> float:
        """
        Calculates fitness criterion for a given chromosome.

        TODO: criterion should heavily penalize incomplete collections 
        (< 3 observations per satellite) as well as erroneous taskings
        (> 3 observations per satellite). In addition, criterion should
        penalize observations of same satellite that are taken too close
        together (< ~10 minutes). 
        """
        # Get unique satellite objects in current chromosome
        satellites = set(sat for _, sat, _ in chromosome)
        pass


def scheduler_optimizer(sensors, satellites, t_start, t_end):
    """
    Given a set of sensors, and satellites with provided visibilities,
    optimize the scheduling of observations to maximize coverage and efficiency.
    """
    pass


def plot_schedule():
    import matplotlib.pyplot as plt
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scheduler module for tasking observations.")
    parser.add_argument("--sensors", type=str, nargs="+", required=True, help="Path to sensors configuration file.")
    parser.add_argument("--tles", type=str, required=True, help="Path to TLE files for satellites.")
    parser.add_argument("--t_start", type=str, required=True, help="Start time for scheduling (ISO format).")
    parser.add_argument("--t_end", type=str, required=True, help="End time for scheduling (ISO format).")
    args = parser.parse_args()
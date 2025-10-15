"""
Associate obs over multiple frames

Author: Peter Thomas
Date: 2025-10-12
"""
import copy
import itertools
import numpy as np
from typing import List, Set


def determine_linearity(ra, dec, t_ob_start, t_ob_end) -> bool:
    """
    Determine if a set of observations are linear in motion via principal component analysis.

    Parameters:
    ra (List[float]): List of right ascension values.
    dec (List[float]): List of declination values.
    t_ob_start (List[float]): List of observation start times.
    t_ob_end (List[float]): List of observation end times.

    Returns:
    bool: True if the observations are linear, False otherwise.
    """
    # Get rates as well as positions
    t_obs = [(t_ob_start[i] + t_ob_end[i]) / 2. for i in range(len(t_ob_start))]
    ra_rates = [(ra[i+1] - ra[i]) / (t_obs[i+1] - t_obs[i]) if (t_obs[i+1] - t_obs[i]) != 0 else 0. for i in range(len(ra)-1)]
    dec_rates = [(dec[i+1] - dec[i]) / (t_obs[i+1] - t_obs[i]) if (t_obs[i+1] - t_obs[i]) != 0 else 0. for i in range(len(dec)-1)]

    return True


def associate_observations(obs: List[Set[float, float, float, float, int]], exposure_time: float) -> np.ndarray:
    """
    Associate observations projected into real-world coordinates over multiple frames.

    Parameters:
    obs (List[Set[float, float, float, float, int]]): List of sets of observations. Each set contains observations from a single frame,
        with each observation containing (ra, dec, t_ob_start, t_ob_end, frame_id).

    Returns:
    np.ndarray: Array of associated observations.
    """
    associated_obs = []
    obs_to_be_associated = copy.deepcopy(obs)

    while len(obs_to_be_associated) > 0:
        # Collect observations from the first frame
        start_frame_idx = min([observation[4] for observation in obs_to_be_associated])
        obs_in_start_frame = [observation for observation_set in obs_to_be_associated for observation in observation_set if observation[4] == start_frame_idx]

        end_frame_idx = max([observation[4] for observation in obs_to_be_associated])
        obs_in_end_frame = [observation for observation_set in obs_to_be_associated for observation in observation_set if observation[4] == end_frame_idx]

        # Generate set of seed points
        seed_points = list(itertools.product(obs_in_start_frame, obs_in_end_frame))
        for seed in seed_points:
            ra1, dec1, t_ob_start1, t_ob_end1, frame_id1 = seed[0]
            ra2, dec2, t_ob_start2, t_ob_end2, frame_id2 = seed[1]
            tracklet = [seed[0]] + [None] * (end_frame_idx - start_frame_idx - 1) + [seed[1]]

            # Collect intermediate frames' observations and test for linearity
            for intermediate_frame_idx in range(start_frame_idx + 1, end_frame_idx):
                obs_in_intermediate_frame = [observation for observation_set in obs_to_be_associated for observation in observation_set if observation[4] == intermediate_frame_idx]
                if len(obs_in_intermediate_frame) == 0:
                    continue

                # Find the observation in the intermediate frame that is closest to the line defined by the seed points
                closest_observation = None
                min_distance = float('inf')
                for obs_intermediate in obs_in_intermediate_frame:
                    ra_int, dec_int, t_ob_start_int, t_ob_end_int, frame_id_int = obs_intermediate

                    # Calculate expected position at the time of the intermediate observation using linear interpolation
                    time_fraction = (t_ob_start_int - t_ob_start1) / (t_ob_start2 - t_ob_start1) if (t_ob_start2 - t_ob_start1) != 0 else 0
                    expected_ra = ra1 + time_fraction * (ra2 - ra1)
                    expected_dec = dec1 + time_fraction * (dec2 - dec1)

                    # Calculate distance from expected position
                    distance = np.sqrt((ra_int - expected_ra)**2 + (dec_int - expected_dec)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_observation = obs_intermediate

                # If a close enough observation is found, update the seed to include it
                if closest_observation and min_distance < 0.01:  # Threshold can be adjusted
                    tracklet[intermediate_frame_idx - start_frame_idx] = closest_observation

            # If the tracklet is complete, evaluate its validity


            # Calculate angular distance between the two observations
            angular_distance = np.sqrt((ra2 - ra1)**2 + (dec2 - dec1)**2)

            # Calculate time difference between the two observations
            time_difference = abs(t_ob_start2 - t_ob_end1)

            # Estimate velocity (degrees per second)
            if time_difference > 0:
                estimated_velocity = angular_distance / time_difference
            else:
                estimated_velocity = float('inf')

            # Define a threshold for maximum allowable velocity (e.g., 0.01 degrees per second)
            max_allowable_velocity = 0.01

            if estimated_velocity <= max_allowable_velocity:
                # If the estimated velocity is within the allowable range, associate the observations
                associated_obs.append((ra1, dec1, t_ob_start1, t_ob_end2, frame_id1))

                # Remove associated observations from the list
                obs_to_be_associated = [
                    observation_set - {seed[0], seed[1]} if observation_set & {seed[0], seed[1]} else observation_set
                    for observation_set in obs_to_be_associated
                ]
                obs_to_be_associated = [observation_set for observation_set in obs_to_be_associated if len(observation_set) > 0]

                break
        else:
            # If no associations were made, remove the first frame's observations to avoid infinite loop
            obs_to_be_associated = [
                observation_set - {observation for observation in observation_set if observation[4] == start_frame_idx}
                if any(observation[4] == start_frame_idx for observation in observation_set) else observation_set
                for observation_set in obs_to_be_associated
            ]
            obs_to_be_associated = [observation_set for observation_set in obs_to_be_associated if len(observation_set) > 0]
        pass
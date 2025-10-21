"""
Module for implementing multiple hypothesis tracking based observation
association.

Author: Peter Thomas
Date: 20 October 2025
"""
import copy
import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Tuple


def mahanalobis_distance(ob, pred, covar):
    return (pred - ob).T @ np.inv(covar) @ (pred - ob)


class TrackTreeNode:
    def __init__(self, ob: Tuple[float, float, float, float], frame_num: int):
        self.ob = ob
        self.frame_num = frame_num

        # Empty list for children nodes
        self.children: List[TrackTreeNode] = []

    def add_child(self, ob: Tuple[float, float, float, float]) -> None:
        self.children.append((TrackTreeNode(ob, self.frame_num + 1)))


class Track:
    def __init__(self, init_node: TrackTreeNode, exposure_time: float, gap_time: float, sensor_fov: float):
        self.root = init_node
        self.sensor_fov = sensor_fov
        self.k_filter = KalmanFilter(init_node.ob[0], init_node.ob[1], exposure_time, gap_time)

        # Keep a historical record of the states and covariance matrices of the kalman filter
        # after each time step. This will be used to calculate track score when all nodes have
        # been added to the track
        self.pred_positions = [self.k_filter.state_vector]
        self.covars = [self.k_filter.covar]

    def add_node_to_track(self, node: TrackTreeNode):
        # Proceed to end of track
        last_node = self.root
        while last_node.child is not None:
            last_node = last_node.child
        last_node.child = node

    def get_list_of_nodes_in_track(self):
        # Return track as a list of observations in track
        track_obs = [self.root.ob]
        next_node = self.root
        while next_node.child is not None:
            next_node = next_node.child
            track_obs.append(next_node.ob)
        return track_obs

    def calculate_score(self) -> float:
        """
        Calculate the score of this track, which is a weighted
        sum of the motion score and appearance score
        """
        motion_score = self.__calculate_motion_score()
        appearance_score = self.__calculate_appearance_score()

        return self.w_motion * motion_score + self.w_appearance * appearance_score


    def __calculate_motion_score(self) -> float:
        measured_positions = self.get_list_of_nodes_in_track()

        target_hypothesis = None
        for i, ob in enumerate(measured_positions):
            # Liklihood of each location measurement at time t under target hypothesis
            # is assumed to be gaussian
            prob = multivariate_normal.pdf((ob[0], ob[1]), mean=self.pred_positions[i], cov=self.covars[i])
            if target_hypothesis is None:
                target_hypothesis = prob
            else:
                target_hypothesis *= prob

        # Calculate null hypothesis, which is one over dimensions of the measurement space
        # (in this case, we'll take it to be the FOV of the scene)
        V = self.__calculate_measurement_space()
        null_hypothesis = (1. / V)**len(measured_positions)

        motion_score = np.log(target_hypothesis / null_hypothesis)

        return motion_score

    def __calculate_measurement_space(self):
        """
        Calculate the measurement space of the scene, which is the area
        carved out of the skybox by all frames in the collection  
        """
        pass
        


class TrackTree:
    def __init__(self, root: TrackTreeNode, exposure_time: float, gap_time: float):
        self.root = root
        self.tracks = []

    def update(self, obs):
        # Add observations at frame 'k' to tracks in tree
        for ob in obs:
            pass



class KalmanFilter:
    """
    Kalman filter to filter hypotheses.
    """
    def __init__(self, initial_ra: float, initial_dec: float, 
                 init_ra_velocity: float, init_dec_velocity: float,
                 exposure_time: float, gap_time: float):

        self.state_vector = np.array([initial_ra, initial_dec, init_ra_velocity, init_dec_velocity])
        self.covar = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.state_transition_matrix = np.array([
            [1.0, 0.0, exposure_time + gap_time, 0.0],
            [0.0, 1.0, 0.0, exposure_time + gap_time],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.noise_matrix = np.array()

        self.pred_state_vector: np.ndarray = None
        self.pred_covar: np.ndarray = None
        self.predict_step_flag: bool = False

    def predict(self):
        """
        Perform prediction step
        """
        self.pred_state_vector = self.state_transition_matrix @ self.state_vector.T
        self.pred_covar = self.state_transition_matrix @ self.covar @ self.state_transition_matrix.T + self.noise_matrix
        self.predict_step_flag = True

        return self.pred_state_vector, self.pred_covar

    def update(self, measurement):
        """
        Perform update step 
        """
        # Perform sanity check, make sure that predict step was performed before update
        if not self.predict_step_flag:
            raise RuntimeError("Predicted state vectors and covariances have not been initialized. Was predict step run before this?")

        # Transition matrix to take us from state space (position & velocity) to measurement
        # space (just position)
        H = np.array([1., 1., 0., 0.])
        pred_pos = np.dot(H, self.predicted_state_vector)
        innovation = measurement - pred_pos

        kalman_gain = np.linalg.inv(H @ self.pred_covar @ H.T + R) @ H @ self.pred_covar

        pass


def run_multiple_hypothesis_tracking(obs: List[List[Tuple[float, float, float, float]]],
                                     exposure_time: float,
                                     gap_time: float,
                                     distance_threshold: float,
                                     w_motion: float,
                                     w_appearance: float):
    """
    Run Multiple hypothesis tracking algorithm on a list of observations from all
    frames in collection 

    Arguments:
        obs (List[List[Tuple[float, float, float, float]]]): A nested list, with the outer 
            list representing the list of frames in the collection and inner list representing
            observations collected in each frame. Each observation is a tuple consisting of
            observed right ascension (degrees), declination (degrees), start of frame exposure,
            and end of frame exposure.
        exposure_time (float): Exposure time in seconds.
        gap_time (float): Gap time between frames, in seconds.
    """
    # Initialize first set of Track trees from first frame
    first_frame_obs = obs[0]
    track_trees = []
    for ob in first_frame_obs:
        root = TrackTreeNode(ob, 0)
        track_trees.append(TrackTree(root))

    for k, obs_k in enumerate(obs[1:]):
        for tree in track_trees:
            new_tracks = []
            for track in tree:
                for ob in obs_k:
                    # Decide whether or not to add observation to tree
                    # based on mahanalanobis distance between predicted
                    # track and observed position
                    pred_state, pred_covar = track.k_filter.predict()
                    dist = mahanalobis_distance(obs_k, pred_state, pred_covar)

                    if dist <= distance_threshold:
                        # Generate a new track with this observation added to it
                        new_track = copy.deepcopy(track)
                        new_track.add_node(ob)
                        new_tracks.append(new_track)

                    # Construct a new track starting with this ob and add to tree
                    init_track = Track(TrackTreeNode(ob, frame_num=k), exposure_time, gap_time)
                    tree.append(init_track)

                if new_tracks:
                    tree.remove(track)
                    tree.extend(new_tracks)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Built a set of test frames 
    pass
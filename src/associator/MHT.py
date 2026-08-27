"""
Module for implementing multiple hypothesis tracking based observation
association.

Author: Peter Thomas
Date: 20 October 2025
"""
import copy
import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Tuple, Optional

# Observation: (ra_deg, dec_deg, t_start_s, t_end_s)
Observation = Tuple[float, float, float, float]


def mahalanobis_distance(ob: np.ndarray, pred: np.ndarray, covar: np.ndarray) -> float:
    """Squared Mahalanobis distance between an observation and a predicted position."""
    diff = ob - pred
    return float(diff.T @ np.linalg.inv(covar) @ diff)


class TrackTreeNode:
    def __init__(self, ob: Observation, frame_num: int):
        self.ob = ob
        self.frame_num = frame_num
        self.children: List["TrackTreeNode"] = []

    def add_child(self, ob: Observation) -> "TrackTreeNode":
        child = TrackTreeNode(ob, self.frame_num + 1)
        self.children.append(child)
        return child


class KalmanFilter:
    """
    Linear Kalman filter tracking state vector [ra, dec, ra_dot, dec_dot]
    (all in degrees / degrees-per-second) under a constant angular-rate motion model.
    """

    # Default astrometric precision: ~10 arcseconds in degrees
    _DEFAULT_SIGMA_OBS: float = 10.0 / 3600.0
    # Default angular acceleration uncertainty in deg/s² (tunes process noise)
    _DEFAULT_SIGMA_ACCEL: float = 1e-4

    def __init__(
        self,
        initial_ra: float,
        initial_dec: float,
        init_ra_velocity: float,
        init_dec_velocity: float,
        exposure_time: float,
        gap_time: float,
        sigma_obs: float = _DEFAULT_SIGMA_OBS,
        sigma_accel: float = _DEFAULT_SIGMA_ACCEL,
    ):
        dt = exposure_time + gap_time

        self.state_vector = np.array(
            [initial_ra, initial_dec, init_ra_velocity, init_dec_velocity]
        )

        # Initial covariance: tight on position, uninformative on velocity
        self.covar = np.diag([sigma_obs**2, sigma_obs**2, 1.0, 1.0])

        # Constant-velocity state transition: position advances by velocity * dt
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Process noise: continuous white-noise acceleration model
        q = sigma_accel**2
        self.Q = q * np.array([
            [dt**4 / 4, 0.0,        dt**3 / 2, 0.0      ],
            [0.0,        dt**4 / 4, 0.0,        dt**3 / 2],
            [dt**3 / 2, 0.0,        dt**2,      0.0      ],
            [0.0,        dt**3 / 2, 0.0,        dt**2    ],
        ])

        # Measurement matrix: observe RA and Dec only (not velocities)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Measurement noise covariance (2×2)
        self.R = (sigma_obs**2) * np.eye(2)

        self.pred_state_vector: Optional[np.ndarray] = None
        self.pred_covar: Optional[np.ndarray] = None
        self._predicted: bool = False

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward one time step."""
        self.pred_state_vector = self.F @ self.state_vector
        self.pred_covar = self.F @ self.covar @ self.F.T + self.Q
        self._predicted = True
        return self.pred_state_vector, self.pred_covar

    def update(self, measurement: np.ndarray) -> None:
        """
        Assimilate a 2-element position measurement [ra_deg, dec_deg].
        Must be called after predict().
        """
        if not self._predicted:
            raise RuntimeError("predict() must be called before update().")

        innovation = measurement - self.H @ self.pred_state_vector
        S = self.H @ self.pred_covar @ self.H.T + self.R       # innovation covariance (2×2)
        K = self.pred_covar @ self.H.T @ np.linalg.inv(S)      # Kalman gain (4×2)

        self.state_vector = self.pred_state_vector + K @ innovation
        self.covar = (np.eye(4) - K @ self.H) @ self.pred_covar
        self._predicted = False

    def propagate_without_update(self) -> None:
        """Advance state using the predicted values when no measurement is available (missed detection)."""
        if not self._predicted:
            raise RuntimeError("predict() must be called before propagate_without_update().")
        self.state_vector = self.pred_state_vector.copy()
        self.covar = self.pred_covar.copy()
        self._predicted = False


class Track:
    """
    A single track hypothesis: a linear sequence of per-frame observation
    associations, plus a Kalman filter tracking the object state.
    """

    def __init__(
        self,
        init_node: TrackTreeNode,
        exposure_time: float,
        gap_time: float,
        sensor_fov: float,
        w_motion: float = 1.0,
        w_appearance: float = 0.0,
        init_ra_velocity: float = 0.0,
        init_dec_velocity: float = 0.0,
    ):
        self.nodes: List[TrackTreeNode] = [init_node]
        self.sensor_fov = sensor_fov
        self.w_motion = w_motion
        self.w_appearance = w_appearance

        self.k_filter = KalmanFilter(
            init_node.ob[0], init_node.ob[1],
            init_ra_velocity, init_dec_velocity,
            exposure_time, gap_time,
        )

        # pred_positions[i] / pred_covars[i] are the filter predictions made
        # immediately before nodes[i] was associated. For the root node there
        # was no prediction, so we seed with the initial filter state.
        self.pred_positions: List[np.ndarray] = [self.k_filter.state_vector[:2].copy()]
        self.pred_covars: List[np.ndarray] = [self.k_filter.covar[:2, :2].copy()]

    def add_observation(
        self,
        node: TrackTreeNode,
        pred_state: np.ndarray,
        pred_covar: np.ndarray,
    ) -> None:
        """
        Associate a new observation with this track and run the Kalman update.
        pred_state and pred_covar are the filter outputs from predict() that were
        used to gate this observation; they are recorded for scoring.
        """
        self.nodes.append(node)
        self.pred_positions.append(pred_state[:2].copy())
        self.pred_covars.append(pred_covar[:2, :2].copy())
        self.k_filter.update(np.array([node.ob[0], node.ob[1]]))

    def get_observations(self) -> List[Observation]:
        return [n.ob for n in self.nodes]

    def calculate_score(self) -> float:
        """Weighted sum of motion and appearance log-likelihood scores."""
        motion_score = self._calculate_motion_score()
        appearance_score = self._calculate_appearance_score()
        return self.w_motion * motion_score + self.w_appearance * appearance_score

    def _calculate_motion_score(self) -> float:
        """
        Log-likelihood ratio of the target hypothesis vs. a uniform clutter model.
        A positive score means the sequence of detections is more consistent with
        a real moving object than with random clutter.
        """
        V = self._measurement_space_volume()
        log_score = 0.0
        for i, ob in enumerate(self.get_observations()):
            z = np.array([ob[0], ob[1]])
            target_ll = multivariate_normal.logpdf(
                z, mean=self.pred_positions[i], cov=self.pred_covars[i]
            )
            null_ll = -np.log(V)
            log_score += target_ll - null_ll
        return log_score

    def _calculate_appearance_score(self) -> float:
        # Placeholder — magnitude / SNR-based appearance scoring would go here.
        return 0.0

    def _measurement_space_volume(self) -> float:
        """Approximate clutter volume as the solid-angle area of the sensor FOV (deg²)."""
        return self.sensor_fov**2


class TrackTree:
    """All track hypotheses that share a common root observation."""

    def __init__(
        self,
        root_node: TrackTreeNode,
        exposure_time: float,
        gap_time: float,
        sensor_fov: float,
        w_motion: float = 1.0,
        w_appearance: float = 0.0,
    ):
        self.root_node = root_node
        self.exposure_time = exposure_time
        self.gap_time = gap_time
        self.sensor_fov = sensor_fov
        self.w_motion = w_motion
        self.w_appearance = w_appearance

        self.tracks: List[Track] = [
            Track(root_node, exposure_time, gap_time, sensor_fov, w_motion, w_appearance)
        ]

    def best_track(self) -> Optional[Track]:
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: t.calculate_score())


def run_multiple_hypothesis_tracking(
    obs: List[List[Observation]],
    exposure_time: float,
    gap_time: float,
    sensor_fov: float,
    distance_threshold: float = 9.21,
    w_motion: float = 1.0,
    w_appearance: float = 0.0,
    k_best: int = 10,
) -> List[Track]:
    """
    Run Multiple Hypothesis Tracking on a sequence of per-frame observation sets.

    Arguments:
        obs: Nested list — obs[frame][i] = (ra_deg, dec_deg, t_start_s, t_end_s).
        exposure_time: Frame exposure time in seconds.
        gap_time: Gap between consecutive frames in seconds.
        sensor_fov: Sensor field of view in degrees (square FOV assumed for clutter volume).
        distance_threshold: Squared Mahalanobis distance gate. The chi-squared 2-DOF
            distribution gives ~95th percentile at 6.0, ~99th percentile at 9.21.
        w_motion: Weight on the motion log-likelihood score component.
        w_appearance: Weight on the appearance score component.
        k_best: Maximum number of hypotheses to retain per TrackTree after each frame
            (k-best pruning to prevent exponential hypothesis growth).

    Returns:
        List of the highest-scoring Track hypothesis from each TrackTree.
    """
    if not obs or not obs[0]:
        return []

    # Initialise one TrackTree per detection in the first frame
    track_trees: List[TrackTree] = [
        TrackTree(
            TrackTreeNode(ob, frame_num=0),
            exposure_time, gap_time, sensor_fov, w_motion, w_appearance,
        )
        for ob in obs[0]
    ]

    for frame_idx, obs_k in enumerate(obs[1:], start=1):
        for tree in track_trees:
            new_tracks: List[Track] = []

            for track in tree.tracks:
                # Predict forward to this frame
                pred_state, pred_covar = track.k_filter.predict()
                pos_pred = pred_state[:2]
                pos_covar = pred_covar[:2, :2]

                gated_branches: List[Track] = []
                for ob in obs_k:
                    z = np.array([ob[0], ob[1]])
                    if mahalanobis_distance(z, pos_pred, pos_covar) <= distance_threshold:
                        # Branch: clone the track (with its predicted state) and update
                        branch = copy.deepcopy(track)
                        branch.add_observation(
                            TrackTreeNode(ob, frame_num=frame_idx),
                            pred_state,
                            pred_covar,
                        )
                        gated_branches.append(branch)

                if gated_branches:
                    new_tracks.extend(gated_branches)
                else:
                    # Missed detection: propagate state without a measurement update
                    track.k_filter.propagate_without_update()
                    new_tracks.append(track)

            # TODO: seed new single-observation tracks from detections that fell
            # outside the gate of every existing track in this tree. This is needed
            # for catalog-building but not required for single-collection association.

            # k-best pruning: retain only the top-k hypotheses by score
            new_tracks.sort(key=lambda t: t.calculate_score(), reverse=True)
            tree.tracks = new_tracks[:k_best]

    return [tree.best_track() for tree in track_trees if tree.best_track() is not None]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MHT on a sequence of observation frames.")
    parser.add_argument("--exposure_time", type=float, required=True, help="Frame exposure time in seconds.")
    parser.add_argument("--gap_time", type=float, required=True, help="Gap between frames in seconds.")
    parser.add_argument("--sensor_fov", type=float, required=True, help="Sensor field of view in degrees.")
    parser.add_argument("--distance_threshold", type=float, default=9.21,
                        help="Squared Mahalanobis gate threshold (default: 9.21 ~ chi2 99th pct).")
    parser.add_argument("--k_best", type=int, default=10, help="Max hypotheses to retain per tree.")
    args = parser.parse_args()

    # Minimal smoke test: two frames, two detections each, one linearly moving object
    test_obs = [
        [(10.0, 20.0, 0.0, 1.0), (15.0, 25.0, 0.0, 1.0)],
        [(10.1, 20.1, 3.5, 4.5), (15.1, 25.1, 3.5, 4.5)],
        [(10.2, 20.2, 7.0, 8.0), (15.2, 25.2, 7.0, 8.0)],
    ]
    best_tracks = run_multiple_hypothesis_tracking(
        test_obs,
        exposure_time=args.exposure_time,
        gap_time=args.gap_time,
        sensor_fov=args.sensor_fov,
        distance_threshold=args.distance_threshold,
        k_best=args.k_best,
    )
    for i, track in enumerate(best_tracks):
        print(f"Track {i}: {len(track.nodes)} frames, score={track.calculate_score():.3f}")
        for node in track.nodes:
            print(f"  frame {node.frame_num}: ra={node.ob[0]:.4f} dec={node.ob[1]:.4f}")

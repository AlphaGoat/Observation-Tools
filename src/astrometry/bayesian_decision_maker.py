"""
Bayesian decision maker for selecting optimal astrometric fit based on stellar features.

Implements the verification framework from Lang et al. 2010 (Astrometry.net).
For each candidate WCS hypothesis, computes the log Bayes factor K comparing a
foreground model (true match) against a background model (random coincidence),
then accepts or rejects via a utility-weighted decision threshold.

Authors: Peter Thomas
Date: 2025-10-15
"""
import numpy as np
from numpy.typing import ArrayLike


def _sort_brightest_first(stars: np.ndarray, sort_by: str) -> np.ndarray:
    """
    Return stars sorted so the brightest row comes first.

    Column layout: [y, x, brightness_metric]
      - sort_by="snr":       column 2 is SNR — sort descending (highest SNR first).
      - sort_by="magnitude": column 2 is visual magnitude — sort ascending (lowest mag first).
    """
    if sort_by == "snr":
        order = np.argsort(stars[:, 2])[::-1]
    elif sort_by == "magnitude":
        order = np.argsort(stars[:, 2])
    else:
        raise ValueError(f"sort_by must be 'snr' or 'magnitude', got '{sort_by}'.")
    return stars[order]


def simple_independence_model(
    reference_stars: ArrayLike,
    test_stars: ArrayLike,
    image_height: int,
    image_width: int,
    variance: float = 10.0,
    distractors: float = 0.25,
) -> float:
    """
    Log Bayes factor K under the simple independence model (Lang et al. 2010 §4.2).

    Each test star is evaluated independently against its nearest reference star using
    a 2D Gaussian foreground model vs. a uniform background model:

        log p_F(z_i) = log((1-d) / (2π·σ²·N_R))  -  min_dist²_i / (2σ²)
        log p_B      = log(1 / (H·W))

        log K = Σ_i  [log p_F(z_i) - log p_B]

    Star ordering does not affect this model — all stars are evaluated independently.

    Parameters
    ----------
    reference_stars : array-like, shape (N_R, 3)
        Reference catalog stars [y, x, brightness_metric].
    test_stars : array-like, shape (N_T, 3)
        Detected stars [y, x, brightness_metric].
    image_height, image_width : int
        Image dimensions in pixels (sets the background density).
    variance : float
        Astrometric position variance σ² in pixels².
    distractors : float
        Expected fraction of test stars that are distractors (typical: 0.25).

    Returns
    -------
    float
        log K.  Positive = evidence for a correct alignment.
    """
    reference_stars = np.asarray(reference_stars)
    test_stars = np.asarray(test_stars)
    N_R = len(reference_stars)
    if N_R == 0 or len(test_stars) == 0:
        return 0.0

    # Pairwise squared distances using y (col 0) and x (col 1): shape (N_test, N_ref)
    diffs = test_stars[:, np.newaxis, :2] - reference_stars[np.newaxis, :, :2]
    dist2 = (diffs ** 2).sum(axis=2)
    min_dist2 = dist2.min(axis=1)  # (N_test,)

    # Foreground: 2D Gaussian from nearest reference star, normalised by N_R
    log_p_F = (
        np.log((1.0 - distractors) / (2.0 * np.pi * variance * N_R))
        - min_dist2 / (2.0 * variance)
    )

    # Background: uniform over the image
    log_p_B = -np.log(image_height * image_width)

    return float(np.sum(log_p_F - log_p_B))


def asymmetric_model(
    reference_stars: ArrayLike,
    test_stars: ArrayLike,
    image_height: int,
    image_width: int,
    variance: float = 10.0,
    distractors: float = 0.25,
    sort_by: str = "snr",
) -> float:
    """
    Log Bayes factor K under the asymmetric distractor-aware model (Lang et al. 2010 §4.3).

    Stars are processed brightest-first.  Each star is assigned to the foreground Gaussian
    model or the distractor model, whichever gives higher probability.  The distractor
    log-probability evolves as matches accumulate (μ increases), reflecting that a field
    with many already-claimed matches has fewer unclaimed reference stars:

        log p_distractor(z_i) = log(d + (1-d)·μ/N_R) + log p_B

    Parameters
    ----------
    reference_stars : array-like, shape (N_R, 3)
        Reference catalog stars [y, x, brightness_metric].
    test_stars : array-like, shape (N_T, 3)
        Detected stars [y, x, brightness_metric].  Sorted internally by brightness.
    image_height, image_width : int
    variance : float
        Astrometric position variance σ² in pixels².
    distractors : float
        Base distractor fraction d.
    sort_by : {"snr", "magnitude"}
        Convention for column 2.  "snr": sorted descending (highest SNR first).
        "magnitude": sorted ascending (lowest magnitude = brightest first).

    Returns
    -------
    float
        log K.
    """
    reference_stars = np.asarray(reference_stars, dtype=float)
    test_stars = np.asarray(test_stars, dtype=float)
    N_R = len(reference_stars)
    if N_R == 0 or len(test_stars) == 0:
        return 0.0

    # Sort test stars brightest-first so μ accumulates in the correct order
    test_stars = _sort_brightest_first(test_stars, sort_by)

    log_p_B = -np.log(image_height * image_width)

    # Pre-compute per-star foreground Gaussian log-probabilities
    diffs = test_stars[:, np.newaxis, :2] - reference_stars[np.newaxis, :, :2]
    dist2 = (diffs ** 2).sum(axis=2)
    min_dist2 = dist2.min(axis=1)
    log_gaussian_peak = np.log((1.0 - distractors) / (2.0 * np.pi * variance * N_R))
    log_p_F_all = log_gaussian_peak - min_dist2 / (2.0 * variance)

    log_K = 0.0
    mu = 0  # number of test stars claimed as matches so far

    for log_p_F in log_p_F_all:
        # Distractor probability grows as μ increases (fewer unclaimed reference stars remain)
        d_term = distractors + (1.0 - distractors) * mu / N_R
        log_p_distractor = np.log(d_term) + log_p_B

        if log_p_F >= log_p_distractor:
            log_K += log_p_F - log_p_B
            mu += 1
        else:
            log_K += log_p_distractor - log_p_B

    return float(log_K)


def bayesian_decision_maker(
    reference_stars: ArrayLike,
    test_stars: ArrayLike,
    image_height: int,
    image_width: int,
    u_tp: float = 1.0,
    u_fp: float = -1999.0,
    u_fn: float = -1.0,
    u_tn: float = 1.0,
    model: str = "simple_independence",
    variance: float = 10.0,
    distractors: float = 0.25,
    sort_by: str = "snr",
) -> bool:
    """
    Decide whether a candidate astrometric fit is correct using Bayesian decision theory.

    Compares log K against a utility-weighted threshold:

        log threshold = log(p_B/p_F)  +  log((u_TN - u_FP) / (u_TP - u_FN))

    The prior p_B/p_F = 10^6 reflects that the vast majority of candidate quad matches
    are false alarms.  The large |u_FP| enforces near-zero false-positive tolerance.

    Parameters
    ----------
    reference_stars : array-like, shape (N_R, 3)
        Reference catalog stars [y, x, brightness_metric].
    test_stars : array-like, shape (N_T, 3)
        Detected stars [y, x, brightness_metric].
    image_height, image_width : int
    u_tp, u_fp, u_fn, u_tn : float
        Utility matrix: true-positive, false-positive, false-negative, true-negative.
    model : {"simple_independence", "asymmetric"}
    variance : float
        Astrometric position variance σ² in pixels².
    distractors : float
        Expected distractor fraction passed to the chosen model.
    sort_by : {"snr", "magnitude"}
        Brightness convention for column 2 of the star arrays.  Used by the asymmetric
        model to process stars brightest-first.  Ignored by the simple independence model.

    Returns
    -------
    bool
        True if the alignment is accepted as correct.
    """
    if model == "simple_independence":
        log_K = simple_independence_model(
            reference_stars, test_stars, image_height, image_width,
            variance=variance, distractors=distractors,
        )
    elif model == "asymmetric":
        log_K = asymmetric_model(
            reference_stars, test_stars, image_height, image_width,
            variance=variance, distractors=distractors, sort_by=sort_by,
        )
    else:
        raise ValueError(
            f"Unknown model '{model}'. Choose 'simple_independence' or 'asymmetric'."
        )

    # log threshold = log(prior odds) + log(utility ratio)
    log_prior_odds = np.log(1e6)  # p_B / p_F: ~1 in 10^6 candidate quads is a true match
    log_utility_ratio = np.log((u_tn - u_fp) / (u_tp - u_fn))
    log_threshold = log_prior_odds + log_utility_ratio

    return bool(log_K > log_threshold)

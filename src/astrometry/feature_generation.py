"""
Module for generating astrometric features from stellar data.

Authors: Peter Thomas
Date: 2025-10-10
"""
import argparse
import itertools
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike

from astrometry.catalog_queries import query_catalog


def compute_hash_code(quad_ra: ArrayLike[float], quad_dec: ArrayLike[float], debug=False) -> Tuple[float, float, float, float] | None:
    """
    Compute a hash code for a quad of stars based on their relative positions.

    Parameters:
        quad_ra (ArrayLike[float]): Right ascension of the four stars in degrees.
        quad_dec (ArrayLike[float]): Declination of the four stars in degrees.
    Returns:
        Tuple[float, float, float, float] | None: The hash code as (x_c, y_c, x_d, y_d) or None if the quad is invalid.
    """
    # Get two most widely separated stars in the quad, which will be used to define local coordinate system
    dist = np.sqrt((quad_ra[:, np.newaxis] - quad_ra[np.newaxis, :])**2 + (quad_dec[:, np.newaxis] - quad_dec[np.newaxis, :])**2)
    max_dist_indices = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    star_a_idx, star_b_idx = max_dist_indices
    star_c_idx, star_d_idx = [i for i in range(4) if i not in max_dist_indices]

    # Determine if both star c and star d fall in the circle defined by star a and star b
    diam = dist[star_a_idx, star_b_idx]
    center_ra = (quad_ra[star_a_idx] + quad_ra[star_b_idx]) / 2
    center_dec = (quad_dec[star_a_idx] + quad_dec[star_b_idx]) / 2
    radius = diam / 2

    def is_within_circle(star_idx):
        d = np.sqrt((quad_ra[star_idx] - center_ra)**2 + (quad_dec[star_idx] - center_dec)**2)
        return d < radius

    if not (is_within_circle(star_c_idx) and is_within_circle(star_d_idx)):
        return None  # Quad is invalid

    # Define local coordinate system with star A at origin and star B at (1, 1)
    star_a_ra = quad_ra[star_a_idx]
    star_a_dec = quad_dec[star_a_idx]
    star_b_ra = quad_ra[star_b_idx]
    star_b_dec = quad_dec[star_b_idx]

    # Compute affine transformation parameters
    # Reference: https://math.stackexchange.com/questions/2982975/geometric-hash-code-or-is-there-a-unique-affine-transformation-mapping-two-2d-po
    delta_ra = star_b_ra - star_a_ra
    delta_dec = star_b_dec - star_a_dec

    theta = (np.pi / 4) - np.arctan2(delta_dec, delta_ra)
    lam = np.sqrt(2) * (1 / np.sqrt(delta_ra**2 + delta_dec**2))

    t_x = lam * (-star_a_ra * np.cos(theta) + star_a_dec * np.sin(theta))
    t_y = lam * (-star_a_dec * np.sin(theta) - star_a_ra * np.cos(theta))

    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ]
    ])

    # Apply transformation to all stars in quad
    coords = np.vstack((quad_ra, quad_dec, np.ones(4)))
    transformed_coords = T @ coords
    x_c = transformed_coords[0, star_c_idx]
    y_c = transformed_coords[1, star_c_idx]
    x_d = transformed_coords[0, star_d_idx]
    y_d = transformed_coords[1, star_d_idx]

    # Ensure consistent ordering of stars c and d
    if x_c > x_d:
        x_c, x_d = x_d, x_c
        y_c, y_d = y_d, y_c
        star_c_idx, star_d_idx = star_d_idx, star_c_idx

    # Ensure x_c + x_d <= 1
    if x_c + x_d > 1:
        return None  # Quad is invalid

    if debug:
        # Plot quad and transformed coordinates for debugging
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot([quad_ra[star_a_idx], quad_ra[star_d_idx]], [quad_dec[star_a_idx], quad_dec[star_d_idx]], 'go-')
        ax[0].plot([quad_ra[star_d_idx], quad_ra[star_b_idx]], [quad_dec[star_d_idx], quad_dec[star_b_idx]], 'go-')
        ax[0].plot([quad_ra[star_b_idx], quad_ra[star_c_idx]], [quad_dec[star_b_idx], quad_dec[star_c_idx]], 'go-')
        ax[0].plot([quad_ra[star_c_idx], quad_ra[star_a_idx]], [quad_dec[star_c_idx], quad_dec[star_a_idx]], 'go-')

        ax[0].text(quad_ra[star_c_idx], quad_dec[star_c_idx], 'C', color='green', fontsize=12)
        ax[0].text(quad_ra[star_d_idx], quad_dec[star_d_idx], 'D', color='green', fontsize=12)
        ax[0].text(quad_ra[star_a_idx], quad_dec[star_a_idx], 'A', color='green', fontsize=12)
        ax[0].text(quad_ra[star_b_idx], quad_dec[star_b_idx], 'B', color='green', fontsize=12)

        ax[0].set_title('Original Quad')
        ax[0].set_xlabel('RA (deg)')
        ax[0].set_ylabel('Dec (deg)')

        ax[1].scatter(transformed_coords[0, :], transformed_coords[1, :])
        ax[1].plot([transformed_coords[0, star_a_idx], transformed_coords[0, star_d_idx]], [transformed_coords[1, star_a_idx], transformed_coords[1, star_d_idx]], 'go-')
        ax[1].plot([transformed_coords[0, star_d_idx], transformed_coords[0, star_b_idx]], [transformed_coords[1, star_d_idx], transformed_coords[1, star_b_idx]], 'go-')
        ax[1].plot([transformed_coords[0, star_b_idx], transformed_coords[0, star_c_idx]], [transformed_coords[1, star_b_idx], transformed_coords[1, star_c_idx]], 'go-')
        ax[1].plot([transformed_coords[0, star_c_idx], transformed_coords[0, star_a_idx]], [transformed_coords[1, star_c_idx], transformed_coords[1, star_a_idx]], 'go-')

        ax[1].text(x_c, y_c, 'C', color='green', fontsize=12)
        ax[1].text(x_d, y_d, 'D', color='green', fontsize=12)
        ax[1].text(transformed_coords[0, star_a_idx], transformed_coords[1, star_a_idx], 'A', color='green', fontsize=12)
        ax[1].text(transformed_coords[0, star_b_idx], transformed_coords[1, star_b_idx], 'B', color='green', fontsize=12)

        ax[1].set_title('Transformed Quad')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        plt.show()

    return (x_c, y_c, x_d, y_d)


def generate_astrometric_features(star_ra: ArrayLike[float], star_dec: ArrayLike[float], 
                                  star_mv: ArrayLike[float], grid_ra: Tuple[float, float], 
                                  grid_dec: Tuple[float, float], num_passes: int=16, 
                                  max_times_used: int=8) -> np.ndarray:
    """
    Generate astrometric features from star positions for a given grid
    """
    # Order stars by magnitude (brightest first)
    sorted_indices = np.argsort(star_mv)
    star_ra = star_ra[sorted_indices]
    star_dec = star_dec[sorted_indices]
    star_mv = star_mv[sorted_indices]

    # Array to keep track of number of times a star has been used to generate a feature
    num_times_used = np.zeros(len(star_ra), dtype=int)

    geometric_hash_codes = []
    for _ in range(num_passes):
        possible_star_quads = itertools.product(range(len(star_ra)), repeat=4)
        for quad in possible_star_quads:
            # Check if any star in the quad has been used too many times
            if any(num_times_used[star_idx] >= max_times_used for star_idx in quad):
                continue

            # Determine if center of quad falls within grid
            quad_ra = star_ra[list(quad)]
            quad_dec = star_dec[list(quad)]
            center_ra = np.mean(quad_ra)
            center_dec = np.mean(quad_dec)
            if not (grid_ra[0] <= center_ra <= grid_ra[1] and grid_dec[0] <= center_dec <= grid_dec[1]):
                continue

            # Generate geometric hash code for this quad
            hash_code = compute_hash_code(quad_ra, quad_dec)
            if hash_code is None:
                continue  # Invalid quad

            geometric_hash_codes.append(hash_code)

            # Update the number of times each star in the quad has been used
            for star_idx in quad:
                num_times_used[star_idx] += 1

    return np.array(geometric_hash_codes)


def iterate_over_celestial_grids(grid_size: Tuple[float, float], max_ra: float, max_dec: float, min_ra: float, min_dec: float):
    """
    Iterate over celestial grids defined by right ascension (RA) and declination (Dec).

    NOTE: coordinates are specified relative to International Celestial Reference System (ICRS).

    Parameters:
    grid_size (Tuple[float, float]): Size of each grid cell in degrees (ra_size, dec_size).
    max_ra (float): Maximum right ascension in degrees.
    max_dec (float): Maximum declination in degrees.
    min_ra (float): Minimum right ascension in degrees.
    min_dec (float): Minimum declination in degrees.
    """
    # Divide the skybox into grids and iterate over them
    ra_size, dec_size = grid_size
    ra_steps = int((max_ra - min_ra) / ra_size)
    dec_steps = int((max_dec - min_dec) / dec_size)
    for i in range(ra_steps):
        for j in range(dec_steps):
            grid_ra = min_ra + i * ra_size
            grid_dec = min_dec + j * dec_size

            # Pull in stars from this grid as well as all adjacent grids
            min_search_ra = np.clip(min_ra + (i - 1) * ra_size, 0.0, 360.0)
            max_search_ra = np.clip(min_ra + (i + 1) * ra_size, 0.0, 360.0)
            min_search_dec = np.clip(min_dec + (j - 1) * dec_size, -90.0, 90.0)
            max_search_dec = np.clip(min_dec + (j + 1) * dec_size, -90.0, 90.0)

            fov_width = max_search_ra - min_search_ra
            fov_height = max_search_dec - min_search_dec
            center_ra = (max_search_ra + min_search_ra) / 2
            center_dec = (max_search_dec + min_search_dec) / 2

            stars = query_catalog(center_ra, center_dec, fov_width, fov_height, catalog_name="Gaia", row_limit=1000)

            # Generate astrometric features for this grid
            hash_codes = generate_astrometric_features(stars['ra'], stars['dec'],
                                                      (grid_ra, grid_ra + ra_size),
                                                      (grid_dec, grid_dec + dec_size))

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate astrometric features from stellar data.")
    parser.add_argument("--grid_size", type=float, nargs=2, required=True, help="Size of each grid cell in degrees (ra_size, dec_size).")
    parser.add_argument("--max_ra", type=float, required=True, help="Maximum right ascension in degrees.")
    parser.add_argument("--max_dec", type=float, required=True, help="Maximum declination in degrees.")
    parser.add_argument("--min_ra", type=float, required=True, help="Minimum right ascension in degrees.")
    parser.add_argument("--min_dec", type=float, required=True, help="Minimum declination in degrees.")
    parser.add_argument("--catalog_name", type=str, required=True, help="Name of the astronomical catalog to query (e.g., 'Gaia', 'SDSS').")
    args = parser.parse_args()    
    iterate_over_celestial_grids(args.grid_size, args.max_ra, args.max_dec, args.min_ra, args.min_dec)
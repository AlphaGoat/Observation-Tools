"""
Module for generating astrometric features from stellar data.

Authors: Peter Thomas
Date: 2025-10-10
"""
import argparse
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike

from astrometry.catalog_queries import query_catalog


def compute_hash_code(quad_ra: ArrayLike[float], quad_dec: ArrayLike[float]) -> Tuple[float, float, float, float]:
    """
    Compute a hash code for a quad of stars based on their relative positions.
    """
    # Get two most widely separated stars in the quad, which will be used to define local coordinate system
    dist = np.sqrt((quad_ra[:, np.newaxis] - quad_ra[np.newaxis, :])**2 + (quad_dec[:, np.newaxis] - quad_dec[np.newaxis, :])**2)
    max_dist_indices = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    star1_idx, star2_idx = max_dist_indices

    # Define local coordinate system with star1 at origin and star2 at (1, 1)
    star1_ra = quad_ra[star1_idx]
    star1_dec = quad_dec[star1_idx]
    star2_ra = quad_ra[star2_idx]
    star2_dec = quad_dec[star2_idx]





def generate_astrometric_features(star_ra: ArrayLike[float], star_dec: ArrayLike[float], 
                                  star_mv: ArrayLike[float], grid_ra: Tuple[float, float], 
                                  grid_dec: Tuple[float, float], num_passes: int=16) -> np.ndarray:
    """
    Generate astrometric features from star positions for a given grid
    """
    # Order stars by magnitude (brightest first)
    sorted_indices = np.argsort(star_mv)
    star_ra = star_ra[sorted_indices]
    star_dec = star_dec[sorted_indices]
    star_mv = star_mv[sorted_indices]

    # Array to keep track of number of times a star has been used to generate a feature
    used_stars = np.zeros(len(star_ra), dtype=int)


    pass


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

            stars = query_catalog(center_ra, center_dec, fov_width, fov_height, catalog_name="Gaia")

            # Generate astrometric features for this grid
            features = generate_astrometric_features(stars['ra'], stars['dec'],
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
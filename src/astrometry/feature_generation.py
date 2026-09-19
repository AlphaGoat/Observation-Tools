"""
Module for generating astrometric features from stellar data.

Authors: Peter Thomas
Date: 2025-10-10
"""
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from numpy.typing import ArrayLike

def compute_hash_code(
    quad_ra: ArrayLike,
    quad_dec: ArrayLike,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute a rotation- and scale-invariant geometric hash code for a quad of stars.

    Stars A and B are the two most widely separated (the baseline).  A similarity
    transform maps A → (0,0) and B → (1,1) in flat-sky projected coordinates
    (RA scaled by cos(dec) of the A-B midpoint).  The normalised positions of
    the interior stars C and D form the hash code (x_c, y_c, x_d, y_d).

    Canonical ordering guarantees a unique code per geometric configuration:
      - x_c ≤ x_d  (C/D label disambiguation)
      - x_c + x_d ≤ 1  (A/B orientation disambiguation via reflection)

    Parameters
    ----------
    quad_ra  : array-like, shape (4,)   RA of the four stars in degrees.
    quad_dec : array-like, shape (4,)   Dec of the four stars in degrees.

    Returns
    -------
    Tuple (x_c, y_c, x_d, y_d) or None if the quad is invalid
    (C or D lies outside the inscribed circle with diameter AB).
    """
    quad_ra  = np.asarray(quad_ra,  dtype=float)
    quad_dec = np.asarray(quad_dec, dtype=float)

    # Pairwise angular distances with cos(dec) correction (flat-sky approximation)
    mean_dec_pair = (quad_dec[:, np.newaxis] + quad_dec[np.newaxis, :]) / 2
    dist = np.sqrt(
        ((quad_ra[:, np.newaxis] - quad_ra[np.newaxis, :])
         * np.cos(np.radians(mean_dec_pair))) ** 2
        + (quad_dec[:, np.newaxis] - quad_dec[np.newaxis, :]) ** 2
    )

    # A and B: the pair with the greatest angular separation
    star_a_idx, star_b_idx = np.unravel_index(np.argmax(dist), dist.shape)
    star_c_idx, star_d_idx = [i for i in range(4)
                               if i not in (star_a_idx, star_b_idx)]

    # Inscribed circle: C and D must lie within the circle whose diameter is AB
    center_ra  = (quad_ra[star_a_idx]  + quad_ra[star_b_idx])  / 2
    center_dec = (quad_dec[star_a_idx] + quad_dec[star_b_idx]) / 2
    radius = dist[star_a_idx, star_b_idx] / 2

    def within_circle(idx: int) -> bool:
        cd = np.cos(np.radians((quad_dec[idx] + center_dec) / 2))
        d  = np.sqrt(((quad_ra[idx] - center_ra) * cd) ** 2
                     + (quad_dec[idx] - center_dec) ** 2)
        return bool(d < radius)

    if not (within_circle(star_c_idx) and within_circle(star_d_idx)):
        return None

    # Project RA to flat sky using the declination of the A-B midpoint.
    # This makes angular distances isotropic for the affine transform.
    # Negated to match _gnomonic's tangent-plane chirality (kd_tree.py):
    # xi = -cos(dec)*sin(ra-ra0)/cos_c increases with *decreasing* RA. Without
    # this flip, a pixel-space quad code can never simultaneously match a
    # stored catalog code and yield an accurate _fit_wcs solution -- confirmed
    # empirically: same quad, opposite pixel parities, one gives an exact code
    # match with a garbage (~49,000 px^2 residual) WCS fit, the other an exact
    # WCS fit (0 residual) with no code match at all.
    cos_ab    = np.cos(np.radians(center_dec))
    proj_ra   = -quad_ra * cos_ab
    delta_ra  = proj_ra[star_b_idx]  - proj_ra[star_a_idx]
    delta_dec = quad_dec[star_b_idx] - quad_dec[star_a_idx]

    # Similarity transform: A → (0,0), B → (1,1)
    theta = (np.pi / 4) - np.arctan2(delta_dec, delta_ra)
    lam   = np.sqrt(2) / np.sqrt(delta_ra ** 2 + delta_dec ** 2)

    t_x = lam * (-proj_ra[star_a_idx] * np.cos(theta)
                 + quad_dec[star_a_idx] * np.sin(theta))
    t_y = lam * (-proj_ra[star_a_idx] * np.sin(theta)   # Fixed: ra then dec (not swapped)
                 - quad_dec[star_a_idx] * np.cos(theta))

    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ],
    ])

    coords = np.vstack((proj_ra, quad_dec, np.ones(4)))
    tc = T @ coords

    x_c, y_c = tc[0, star_c_idx], tc[1, star_c_idx]
    x_d, y_d = tc[0, star_d_idx], tc[1, star_d_idx]

    # Canonical C/D ordering: ensure x_c ≤ x_d
    if x_c > x_d:
        x_c, x_d = x_d, x_c
        y_c, y_d = y_d, y_c

    # Canonical A/B orientation: enforce x_c + x_d ≤ 1.
    # If x_c + x_d > 1 the quad is in the non-canonical A↔B orientation.
    # Swapping A and B maps every interior coordinate (x, y) → (1-x, 1-y),
    # so we apply that reflection instead of discarding the quad.
    if x_c + x_d > 1:
        x_c, x_d = 1.0 - x_d, 1.0 - x_c
        y_c, y_d = 1.0 - y_d, 1.0 - y_c
        if x_c > x_d:       # Re-check C/D ordering after reflection
            x_c, x_d = x_d, x_c
            y_c, y_d = y_d, y_c

    return (x_c, y_c, x_d, y_d)


def plot_quad(
    quad_ra: ArrayLike,
    quad_dec: ArrayLike,
) -> plt.Figure:
    """
    Visualize a star quad in sky (RA-Dec) space and in hash feature space.

    Left panel: the four stars on the sky with the A-B baseline and the
    inscribed circle that defines quad validity.
    Right panel: the same quad in the normalized coordinate system where
    A → (0,0) and B → (1,1), with the hash code coordinates of C and D labelled.

    This is the primary debugging tool for the quad hash algorithm.  Call it on
    any four-star tuple to inspect whether the transform is working correctly.

    Parameters
    ----------
    quad_ra  : array-like, shape (4,)   RA of the four stars in degrees.
    quad_dec : array-like, shape (4,)   Dec of the four stars in degrees.

    Returns
    -------
    matplotlib Figure.
    """
    quad_ra  = np.asarray(quad_ra,  dtype=float)
    quad_dec = np.asarray(quad_dec, dtype=float)

    # ---- Reproduce full transform (mirrors compute_hash_code exactly) -------
    mean_dec_pair = (quad_dec[:, np.newaxis] + quad_dec[np.newaxis, :]) / 2
    dist = np.sqrt(
        ((quad_ra[:, np.newaxis] - quad_ra[np.newaxis, :])
         * np.cos(np.radians(mean_dec_pair))) ** 2
        + (quad_dec[:, np.newaxis] - quad_dec[np.newaxis, :]) ** 2
    )

    star_a_idx, star_b_idx = np.unravel_index(np.argmax(dist), dist.shape)
    star_c_idx, star_d_idx = [i for i in range(4)
                               if i not in (star_a_idx, star_b_idx)]

    center_ra  = (quad_ra[star_a_idx]  + quad_ra[star_b_idx])  / 2
    center_dec = (quad_dec[star_a_idx] + quad_dec[star_b_idx]) / 2
    radius = dist[star_a_idx, star_b_idx] / 2

    def within_circle(idx: int) -> bool:
        cd = np.cos(np.radians((quad_dec[idx] + center_dec) / 2))
        d  = np.sqrt(((quad_ra[idx] - center_ra) * cd) ** 2
                     + (quad_dec[idx] - center_dec) ** 2)
        return bool(d < radius)

    valid = within_circle(star_c_idx) and within_circle(star_d_idx)

    cos_ab    = np.cos(np.radians(center_dec))
    proj_ra   = quad_ra * cos_ab
    delta_ra  = proj_ra[star_b_idx]  - proj_ra[star_a_idx]
    delta_dec = quad_dec[star_b_idx] - quad_dec[star_a_idx]

    theta = (np.pi / 4) - np.arctan2(delta_dec, delta_ra)
    lam   = np.sqrt(2) / np.sqrt(delta_ra ** 2 + delta_dec ** 2)
    t_x   = lam * (-proj_ra[star_a_idx] * np.cos(theta) + quad_dec[star_a_idx] * np.sin(theta))
    t_y   = lam * (-proj_ra[star_a_idx] * np.sin(theta) - quad_dec[star_a_idx] * np.cos(theta))

    T = np.array([
        [lam * np.cos(theta), -lam * np.sin(theta), t_x],
        [lam * np.sin(theta),  lam * np.cos(theta), t_y],
        [0.,                   0.,                  1. ],
    ])
    coords = np.vstack((proj_ra, quad_dec, np.ones(4)))
    tc = T @ coords

    x_c, y_c = tc[0, star_c_idx], tc[1, star_c_idx]
    x_d, y_d = tc[0, star_d_idx], tc[1, star_d_idx]

    if x_c > x_d:
        x_c, x_d = x_d, x_c
        y_c, y_d = y_d, y_c
        star_c_idx, star_d_idx = star_d_idx, star_c_idx

    # Track canonical A/B assignment after potential reflection
    canonical_a_idx, canonical_b_idx = star_a_idx, star_b_idx
    if x_c + x_d > 1:
        x_c, x_d = 1.0 - x_d, 1.0 - x_c
        y_c, y_d = 1.0 - y_d, 1.0 - y_c
        canonical_a_idx, canonical_b_idx = star_b_idx, star_a_idx  # A↔B swap
        if x_c > x_d:
            x_c, x_d = x_d, x_c
            y_c, y_d = y_d, y_c
            star_c_idx, star_d_idx = star_d_idx, star_c_idx

    labels  = {
        canonical_a_idx: 'A', canonical_b_idx: 'B',
        star_c_idx: 'C',      star_d_idx: 'D',
    }
    colours = {'A': '#e74c3c', 'B': '#3498db', 'C': '#2ecc71', 'D': '#f39c12'}
    feat_pos = {
        canonical_a_idx: (0.0, 0.0),
        canonical_b_idx: (1.0, 1.0),
        star_c_idx:      (x_c, y_c),
        star_d_idx:      (x_d, y_d),
    }

    status_str = "Valid" if valid else "INVALID — C or D outside inscribed circle"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{status_str}  |  hash: "
        f"(x_C={x_c:.4f},  y_C={y_c:.4f},  x_D={x_d:.4f},  y_D={y_d:.4f})",
        fontsize=10,
    )

    # ---- Left: sky (RA-Dec) space ------------------------------------------
    ax = axes[0]

    # Inscribed circle — RA is horizontally stretched by 1/cos(dec)
    theta_c  = np.linspace(0, 2 * np.pi, 300)
    circ_ra  = center_ra  + (radius / cos_ab) * np.cos(theta_c)
    circ_dec = center_dec + radius              * np.sin(theta_c)
    ax.plot(circ_ra, circ_dec, '--', color='lightgray', linewidth=1.0,
            label='Inscribed circle')

    # Quad outline: A–C–B–D–A
    quad_order = [canonical_a_idx, star_c_idx, canonical_b_idx, star_d_idx]
    poly_ra  = [quad_ra[i]  for i in quad_order] + [quad_ra[quad_order[0]]]
    poly_dec = [quad_dec[i] for i in quad_order] + [quad_dec[quad_order[0]]]
    ax.plot(poly_ra, poly_dec, '-', color='gray', linewidth=0.8, alpha=0.6)

    # Baseline A–B
    ax.plot([quad_ra[canonical_a_idx], quad_ra[canonical_b_idx]],
            [quad_dec[canonical_a_idx], quad_dec[canonical_b_idx]],
            'k-', linewidth=2.0, label='Baseline A–B', zorder=3)

    for idx in range(4):
        lbl = labels[idx]
        col = colours[lbl]
        ax.scatter(quad_ra[idx], quad_dec[idx], color=col, s=110, zorder=5)
        ax.annotate(f'  {lbl}', (quad_ra[idx], quad_dec[idx]),
                    fontsize=12, color=col, fontweight='bold')

    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('Dec (degrees)')
    ax.set_title('Sky (RA–Dec) space')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Right: feature (hash) space ---------------------------------------
    ax = axes[1]

    # Unit square boundary
    sq = plt.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], fill=False,
                     edgecolor='lightgray', linestyle='--', linewidth=1.0)
    ax.add_patch(sq)

    # Diagonal (A→B = (0,0)→(1,1)) for reference
    ax.plot([0, 1], [0, 1], color='lightgray', linewidth=0.6, linestyle=':')

    # Quad outline in feature space
    poly_fx = [feat_pos[i][0] for i in quad_order] + [feat_pos[quad_order[0]][0]]
    poly_fy = [feat_pos[i][1] for i in quad_order] + [feat_pos[quad_order[0]][1]]
    ax.plot(poly_fx, poly_fy, '-', color='gray', linewidth=0.8, alpha=0.6)

    # Baseline A–B
    ax.plot([feat_pos[canonical_a_idx][0], feat_pos[canonical_b_idx][0]],
            [feat_pos[canonical_a_idx][1], feat_pos[canonical_b_idx][1]],
            'k-', linewidth=2.0, label='Baseline A–B', zorder=3)

    for idx in range(4):
        lbl = labels[idx]
        col = colours[lbl]
        fx, fy = feat_pos[idx]
        ax.scatter(fx, fy, color=col, s=110, zorder=5)
        ax.annotate(f'  {lbl}\n  ({fx:.3f}, {fy:.3f})', (fx, fy),
                    fontsize=9, color=col, fontweight='bold')

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Feature (hash) space')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_astrometric_features(
    star_ra: ArrayLike,
    star_dec: ArrayLike,
    star_mv: ArrayLike,
    grid_ra: Tuple[float, float],
    grid_dec: Tuple[float, float],
    max_times_used: int = 8,
) -> np.ndarray:
    """
    Generate geometric quad hash codes for all valid 4-star combinations in a sky grid cell.

    Stars are enumerated brightest-first (ascending magnitude) so that bright stars
    participate in more quads before hitting the `max_times_used` cap.  Only quads
    whose centroid falls within `grid_ra` × `grid_dec` contribute to the output.

    Parameters
    ----------
    star_ra  : array-like (N,)   Star right ascensions in degrees.
    star_dec : array-like (N,)   Star declinations in degrees.
    star_mv  : array-like (N,)   Visual magnitudes (lower = brighter).
    grid_ra  : (ra_min, ra_max)  RA bounds of the target grid cell in degrees.
    grid_dec : (dec_min, dec_max) Dec bounds of the target grid cell in degrees.
    max_times_used : int
        Maximum number of quads a single star may appear in.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Array of hash codes (x_c, y_c, x_d, y_d) for all M valid quads found.
        Returns shape (0, 4) if no valid quads exist.
    """
    star_ra  = np.asarray(star_ra,  dtype=float)
    star_dec = np.asarray(star_dec, dtype=float)
    star_mv  = np.asarray(star_mv,  dtype=float)

    # Enumerate brightest-first so the max_times_used cap favours bright stars
    order    = np.argsort(star_mv)
    star_ra  = star_ra[order]
    star_dec = star_dec[order]

    num_times_used = np.zeros(len(star_ra), dtype=int)
    hash_codes: List[Tuple[float, float, float, float]] = []

    # Fixed: combinations gives unique 4-star subsets; product (with repeat=4)
    # generated n^4 tuples including quads with repeated star indices.
    for quad in itertools.combinations(range(len(star_ra)), 4):
        if any(num_times_used[i] >= max_times_used for i in quad):
            continue

        quad_ra  = star_ra[list(quad)]
        quad_dec = star_dec[list(quad)]

        # Only include quads whose centroid falls within the target grid cell
        if not (grid_ra[0]  <= np.mean(quad_ra)  <= grid_ra[1]
                and grid_dec[0] <= np.mean(quad_dec) <= grid_dec[1]):
            continue

        code = compute_hash_code(quad_ra, quad_dec)
        if code is None:
            continue

        hash_codes.append(code)
        for i in quad:
            num_times_used[i] += 1

    return np.array(hash_codes) if hash_codes else np.empty((0, 4))


def iterate_over_celestial_grids(
    grid_size: Tuple[float, float],
    min_ra: float,
    max_ra: float,
    min_dec: float,
    max_dec: float,
    catalog_name: str = "Gaia",
    row_limit: int = 1000,
) -> np.ndarray:
    """
    Build a database of geometric quad hash codes spanning a sky region.

    Divides the sky region into a grid and, for each cell, queries the star catalog
    over the cell plus its immediate neighbours (a 3×3 neighbourhood), then generates
    hash codes for all valid 4-star quads whose centroid falls within the cell.

    Parameters
    ----------
    grid_size        : (ra_size_deg, dec_size_deg)
    min_ra, max_ra   : RA bounds of the sky region in degrees.
    min_dec, max_dec : Dec bounds of the sky region in degrees.
    catalog_name     : Catalog to query ("Gaia" supported).
    row_limit        : Maximum stars returned per catalog query.

    Returns
    -------
    np.ndarray, shape (M, 4)
        All hash codes from all grid cells concatenated.
    """
    ra_size, dec_size = grid_size
    ra_steps  = int((max_ra  - min_ra)  / ra_size)
    dec_steps = int((max_dec - min_dec) / dec_size)

    all_hash_codes: List[np.ndarray] = []

    for i in range(ra_steps):
        for j in range(dec_steps):
            grid_ra_min  = min_ra  + i * ra_size
            grid_dec_min = min_dec + j * dec_size

            # Search a 3×3 neighbourhood (current cell ± one cell in each axis)
            # Fixed: upper bound uses i+2 / j+2 to include the full adjacent cell,
            # not just its starting edge.
            search_ra_min  = np.clip(min_ra  + (i - 1) * ra_size,  0.0,  360.0)
            search_ra_max  = np.clip(min_ra  + (i + 2) * ra_size,  0.0,  360.0)
            search_dec_min = np.clip(min_dec + (j - 1) * dec_size, -90.0,  90.0)
            search_dec_max = np.clip(min_dec + (j + 2) * dec_size, -90.0,  90.0)

            fov_width  = search_ra_max  - search_ra_min
            fov_height = search_dec_max - search_dec_min
            center_ra  = (search_ra_min  + search_ra_max)  / 2
            center_dec = (search_dec_min + search_dec_max) / 2

            from astrometry.catalog_queries import query_catalog  # deferred: astroquery only needed here
            # Fixed: catalog_name is the first positional argument of query_catalog.
            stars = query_catalog(
                catalog_name, center_ra, center_dec, fov_width, fov_height,
                row_limit=row_limit,
            )

            # Fixed: pass star_mv (stars['mV']); previously the grid tuples were
            # shifted one position to the left and star_mv was omitted entirely.
            codes = generate_astrometric_features(
                stars['ra'], stars['dec'], stars['mV'],
                grid_ra=(grid_ra_min,  grid_ra_min  + ra_size),
                grid_dec=(grid_dec_min, grid_dec_min + dec_size),
            )

            if len(codes):
                all_hash_codes.append(codes)

    return np.vstack(all_hash_codes) if all_hash_codes else np.empty((0, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate astrometric quad hash codes from a star catalog."
    )
    parser.add_argument("--grid_size", type=float, nargs=2, required=True,
                        metavar=("RA_DEG", "DEC_DEG"),
                        help="Grid cell size in degrees (RA Dec).")
    parser.add_argument("--max_ra",  type=float, required=True,   # Fixed: was "--max_ragithub login with gmail"
                        help="Maximum right ascension in degrees.")
    parser.add_argument("--max_dec", type=float, required=True,
                        help="Maximum declination in degrees.")
    parser.add_argument("--min_ra",  type=float, required=True,
                        help="Minimum right ascension in degrees.")
    parser.add_argument("--min_dec", type=float, required=True,
                        help="Minimum declination in degrees.")
    parser.add_argument("--catalog_name", type=str, default="Gaia",
                        help="Astronomical catalog to query (default: Gaia).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the hash code array as a .npy file.")
    args = parser.parse_args()

    hash_db = iterate_over_celestial_grids(
        args.grid_size,
        args.min_ra, args.max_ra,
        args.min_dec, args.max_dec,
        catalog_name=args.catalog_name,
    )
    print(f"Generated {len(hash_db)} hash codes.")
    if args.output:
        np.save(args.output, hash_db)
        print(f"Saved to {args.output}")

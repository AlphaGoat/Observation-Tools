"""
Module for helper functions for querying from a variety of astronomical catalogs.

Authors: Peter Thomas
Date: 2025-10-10
"""
import argparse
import numpy as np

from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord


def plot_stars(stars, center_ra, center_dec, fov_height, fov_width):
    """
    Plot stars on a 2D scatter plot.

    Parameters:
    stars (Table): Astropy Table containing star data with 'ra' and 'dec' columns.
    center_ra (float): Right Ascension at center of FOV in degrees.
    center_dec (float): Declination at center of FOV in degrees.
    fov_height (float): Height of the field of view in degrees.
    fov_width (float): Width of the field of view in degrees.
    """
    import matplotlib.pyplot as plt

    ra = stars['ra']
    dec = stars['dec']

    plt.figure(figsize=(8, 8))
    plt.scatter(ra, dec, s=1, color='blue')
    plt.xlim(center_ra - (fov_width / 2), center_ra + (fov_width / 2))
    plt.ylim(center_dec - (fov_height / 2), center_dec + (fov_height / 2))
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.title('Star Positions')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


def query_gaia_catalog(center_ra: float, center_dec: float, fov_width: float, fov_height: float, row_limit: int = 10000):
    """
    Query the Gaia catalog for stars within a specified radius of given RA and Dec.

    References: https://ysbach.github.io/SNU_AOpython/chaps/05-query.html

    Parameters:
    center_ra (float): Right Ascension at center of FOV in degrees.
    center_dec (float): Declination at center of FOV in degrees.
    fov_width (float): Width of the field of view in degrees.
    fov_height (float): Height of the field of view in degrees.

    Returns:
    Table: Astropy Table containing the query results.
    """
    # Set row limit for Gaia queries
    Gaia.ROW_LIMIT = row_limit

    # Perform the query
    center_coordinate = SkyCoord(ra=center_ra * u.degree, dec=center_dec * u.degree, frame='icrs')
    q_gaia = Gaia.query_object_async(coordinate=center_coordinate, width=fov_width * u.degree, height=fov_height * u.degree)

    # Add useful columns
    q_gaia['bp_snr'] = 1. / q_gaia["phot_bp_mean_flux_over_error"] 
    q_gaia['rp_snr'] = 1. / q_gaia["phot_rp_mean_flux_over_error"]
    q_gaia['g_snr'] = 1. / q_gaia["phot_g_mean_flux_over_error"]

    # Calculate color error
    q_gaia['dC'] = 2.5 / np.log(10) * np.sqrt(q_gaia['bp_snr']**2 + q_gaia['rp_snr']**2)

    # Magnitude conversion coefficients (https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html)
    coef = np.array([-0.02704, 0.1425, -0.2156, 0.01426])
    rmse = 0.03017 

    # Calculate V-mag and error
    q_gaia['mV'] = (
        q_gaia['phot_g_mean_mag'] + coef[0] + coef[1] * q_gaia['bp_rp'] + 
        coef[2] * q_gaia['bp_rp']**2 + coef[3] * q_gaia['bp_rp']**3
    )
    q_gaia['dV'] = np.sqrt(
        2.5 / np.log(10) * q_gaia['g_snr']**2 +
        (coef[1] + 2 * coef[2] * q_gaia['bp_rp'] + 3 * coef[3] * q_gaia['bp_rp']**2)**2 * q_gaia['dC']**2 +
        rmse**2
    )

    # Only include columns we care about
    q_gaia = q_gaia['source_id', 'ra', 'dec', 'bp_snr', 'rp_snr', 'g_snr', 'mV', 'dV']
    return q_gaia


def query_catalog(catalog_name: str, center_ra: float, center_dec: float, fov_width: float, fov_height: float, **kwargs):
    """
    Query a specified astronomical catalog for stars within a given radius of RA and Dec.

    Parameters:
    catalog_name (str): Name of the catalog to query (e.g., 'Gaia', 'Mast').
    center_ra (float): Right Ascension at center of FOV in degrees.
    center_dec (float): Declination at center of FOV in degrees.
    radius (float): Search radius in arcseconds.

    Returns:
    Table: Astropy Table containing the query results.
    """
    if catalog_name.lower() == "gaia":
        return query_gaia_catalog(center_ra, center_dec, fov_width=fov_width, fov_height=fov_height, **kwargs)
    elif catalog_name.lower() == "mast":
        return NotImplementedError("Mast catalog query not implemented yet.")
#        return Catalogs.query_region(f"{center_ra} {center_dec}", radius=radius * u.arcsec, catalog="TIC")
    else:
        raise ValueError(f"Catalog '{catalog_name}' is not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query astronomical catalogs for stellar data.")
    parser.add_argument("--catalog_name", type=str, required=True, default="Gaia", help="Name of the catalog to query (e.g., 'Gaia', 'Mast').")
    parser.add_argument("--center_ra", type=float, required=True, help="Right Ascension at center of FOV in degrees.")
    parser.add_argument("--center_dec", type=float, required=True, help="Declination at center of FOV in degrees.")
    parser.add_argument("--fov_width", type=float, default=5.0, help="Width of the field of view in degrees.")
    parser.add_argument("--fov_height", type=float, default=5.0, help="Height of the field of view in degrees.")
    args = parser.parse_args()
    stars = query_catalog(args.catalog_name, args.center_ra, args.center_dec, args.fov_width, args.fov_height)

    # Plot the results
    plot_stars(stars, args.center_ra, args.center_dec, args.fov_height, args.fov_width)
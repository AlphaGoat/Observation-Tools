"""
Source Extraction. Very basic point-like source detection, will need
to implement better methods going forward, but this will do for now...

Author: Peter Thomas
Date: October 19, 2025
"""
import warnings
import numpy as np

from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks, DAOStarFinder
from photutils.segmentation import detect_sources, make_2dgaussian_kernel



def gaussian_fitting_source_extraction(img_frame: np.ndarray):
    # Subtract background from image
    mean, median, std = sigma_clipped_stats(img_frame, sigma=3.0, maxiters=5)
    subtracted_frame = img_frame - median

    with warnings.catch_warnings(action="ignore"): # Generates a lot of warnings when fitting 2D gaussian to source
        sources_findpeaks = find_peaks(subtracted_frame,
                                       threshold=5. * std, box_size=29)
    return sources_findpeaks
        

def DAOStarFinder_source_extraction(img_frame: np.ndarray, fwhm: float=5.0):
    mean, median, std = sigma_clipped_stats(img_frame, sigma=3.0, maxiters=5)
    subtracted_frame = img_frame - median
    daofind = DAOStarFinder(fwhm=fwhm, threshold=5. * std)
    sources_dao = daofind(subtracted_frame)
    return sources_dao


def image_segmentation(img_frame: np.ndarray, threshold: float=3., npixels: int=10, fwhm: float=3., 
                       kernel_size: int=5):
    """
    Basic implementation of image segmentation for source detection.

    Parameters:
        img_frame: 
    """
    mean, median, std = sigma_clipped_stats(img_frame, sigma=3.0, maxiters=5)
    subtracted_frame = img_frame - median

    kernel = make_2dgaussian_kernel(fwhm, size=kernel_size)

    # Make a convolved image
    convolved_data = convolve(subtracted_frame, kernel)

    # Create a segmentation image from the convolved image
    seg = detect_sources(convolved_data, threshold, npixels)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_frame_path", type=str, required=True,
                        help="Path to frame we want to perform source extraction on (NOTE: needs to be FITS image...).")

    args = parser.parse_args()

    with fits.open(args.test_frame_path) as hdul:
        img_data = hdul[0].data

    # Find sources using both methods
    sources_findpeaks = gaussian_fitting_source_extraction(img_data)
    sources_dao = DAOStarFinder_source_extraction(img_data)

    # Normalize image input
    norm = ImageNormalize(stretch=SqrtStretch())

    # Prepare plot to compare source detection methods
    fig, ax1 = plt.subplot(1, 1, figsize=(8, 8))
    fitsplot = ax1.imshow(img_data, norm=norm, cmap="Greys", origin="lower")

    marker_size = 60
    ax1.scatter(sources_findpeaks['x_peak'], sources_findpeaks['y_peak'], s=marker_size, marker='s',
                lw=1, alpha=1, facecolor="None", edgecolor='r', label='Found by find_peaks method')
    ax1.scatter(sources_dao['xcentroid'], sources_dao['ycentroid'], s= 2 * marker_size, marker='D',
                lw=1, alpha=1, facecolor="None", edgecolor='#0077BB', label='Found by DAOfind method')
    ax1.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.35))

    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title("Sources found by Different Methods")
    
    
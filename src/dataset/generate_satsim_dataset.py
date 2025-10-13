"""
Generate a synthetic dataset for satellite image simulation.

Author: Peter Thomas
Date: 2025-10-10
"""
import os
import json
import argparse
import random
from tqdm import tqdm


def construct_satsim_config(
        num_frames: int=10,
        fpa_height: int=1024,
        fpa_width: int=1024,
        y_fov: float=0.5,
        x_fov: float=0.5,
        dark_current: float=0.3,
        exposure_time: float=1.0,
        gap_time: float=2.5,
        gain: float=1.0,
        bias: float=0.0,
        read_noise: int=5,
        electronic_noise: int=0
):
    config = {}
    config["version"] = 1
    config["sim"] = {
        "mode": "fftconv2p", # specifies convolution mode
        "spacial_osf": 15, # number of pixels to upsample the image in each axis,
        "temporal_osf": 100, # Number of transformations to apply to smear the background
        "padding": 100, # number of real pixels to pad each side of the image,
        "samples": 1, # number of sets to generate
    }

    config["fpa"] = {
        "height": fpa_height, # number of pixels in the y direction
        "width": fpa_width, # number of pixels in the x direction
        "y_fov": y_fov, # field of view in degrees in the y direction
        "x_fov": x_fov, # field of view in degrees in the x direction
        "dark_current": dark_current, # dark current in electrons per second per pixel
        "gain": gain, # gain of the detector (electrons per ADU)
        "bias": bias, # bias level of the detector (ADU)
        "zeropoint": 20.6663, # zeropoint of the detector (magnitude at 1 ADU per second)
        "a2d": {
            "response": "linear", # response of the analog to digital converter (linear or nonlinear)
            "fwc": 100000, # full well capacity in electrons
            "gain": 1.0, # gain of the A2D (electrons per ADU)
            "bias": 1500, # bias level of the A2D (ADU) in digitical counts
        },
        "noise": {
            "read": read_noise, # read noise in electrons,
            "electronic": electronic_noise, # electronic noise in electrons,
        },
        "psf": {
            "mode": "gaussian", # point spread function mode (gaussian, airy, moffat, or empirical)
            "eod": 0.15, # energy on detector or ensquared energy from 0.0-1.0 of the gaussian
        },
        "time": {
            "exposure": exposure_time, # exposure time in seconds
            "gap": gap_time, # time between exposures in seconds
        },
        "num_frames": num_frames, # number of frames per collection
    }

    config["background"] = {
        "stray": {
            "mode": "none"
        },
        "galactic": 19.5
    }

    config["geometry"] = {
        "stars": {
            "mode": "bins",        # star generation type
            "mv": {
                "bins":            # visual magnitude bins
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "density":         # density of stars in number/degrees^2 per bin
                    [0.019444,0,0.0055556,0.016667,0.036111,0.038889,0.097222,0.66944,2.4778,5.0028,10.269,24.328,35.192,60.017,110.06,180.28,285.53,446.14]
            },
            "motion": {
                "mode": "affine",  # transform type (placeholder)
                "rotation": 0,     # clockwise star rotation rate in radians/seconds
                "translation":     # drift rate in pixels/second [row, column]
                    [0.4,7.0]
            }
        },
        "obs": {
            "mode": "list",            # obj generation type
            "list": {
                "$sample": "random.list",
                "length":              # number of targets to sample
                    { "$sample": "random.randint", "low": 0, "high": 15 },
                "value": {
                    "mode": "line",    # draw type
                    "origin": [        # starting location of object in normalized array coordinates [row, col]
                        { "$sample": "random.uniform", "low": 0.1, "high": 0.9 },
                        { "$sample": "random.uniform", "low": 0.1, "high": 0.9 }
                    ],
                    "velocity": [      # velocity of the object in pixels/second [row, col]
                        { "$sample": "random.uniform", "low": -0.01, "high": 0.01 },
                        { "$sample": "random.uniform", "low": -0.01, "high": 0.01 }
                    ],
                    "mv":              # visual magnitude of the object
                        { "$sample": "random.uniform", "low": 5.0, "high": 22.0 }
                }
            }
        }
    }

    return config


def generate_random_configs(num_collections: int=10):
    for _ in tqdm(range(num_collections), desc="Generating configurations"):
        num_frames = random.randint(5, 20)
        fpa_height = random.choice([512, 1024, 2048])
        fpa_width = random.choice([512, 1024, 2048])
        y_fov = random.uniform(0.1, 2.0)
        x_fov = random.uniform(0.1, 2.0)
        dark_current = random.uniform(0.1, 1.0)
        exposure_time = random.uniform(0.5, 5.0)
        gap_time = random.uniform(1.0, 10.0)
        gain = random.uniform(0.5, 2.0)
        bias = random.uniform(1000, 2000)
        read_noise = random.randint(5, 15)
        electronic_noise = random.randint(0, 2)

        config = construct_satsim_config(
            num_frames=num_frames,
            fpa_height=fpa_height,
            fpa_width=fpa_width,
            y_fov=y_fov,
            x_fov=x_fov,
            dark_current=dark_current,
            exposure_time=exposure_time,
            gap_time=gap_time,
            gain=gain,
            bias=bias,
            read_noise=read_noise,
            electronic_noise=electronic_noise
        )

        yield config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for satellite image simulation.")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the generated dataset.')
    parser.add_argument('--num_collections', type=int, default=1000, help='Number of synthetic collections to generate.')
    args = parser.parse_args()

    # Placeholder for dataset generation logic
    num_collections = args.num_collections
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = list(generate_random_configs(num_collections))

    # Save the generated dataset to a file
    for i, config in enumerate(dataset):
        with open(os.path.join(args.output_dir, f'simulated_collection_{i:04d}.json'), 'w') as f:
            json.dump(config, f, indent=4)

    print(f"Synthetic dataset with {num_collections} collections saved to {args.output_dir}.")
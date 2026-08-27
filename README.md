# Observation-Tools

A Python toolkit for observing artificial satellites with ground-based telescopes. The end-to-end pipeline takes a set of target satellites and sensor locations, schedules observations, detects objects in imagery, solves the astrometric plate, and links detections across frames into satellite tracks.

---

## Pipeline Stages

### 1. Data Ingestion — `src/tasking/grab_tles_from_spacetrack.py`

Fetch current Two-Line Element (TLE) orbital data from [Space-Track.org](https://www.space-track.org). TLEs are the input to all downstream orbital propagation.

```bash
python src/tasking/grab_tles_from_spacetrack.py \
    --username <user> --password <pass> \
    --catalog active --format json --limit 500 \
    --output tles.json
```

---

### 2. Visibility & Tasking — `src/tasking/`

Determine when and where each satellite is observable, then schedule sensors to maximise coverage.

**Visibility** (`get_visibility.py`): Given a TLE and a ground observer location, compute the satellite's altitude and azimuth over a time window and flag passes where it is above the horizon and sunlit.

**Scheduling** (`scheduler.py`): A genetic algorithm that assigns (sensor, satellite, time slot) triples — a *chromosome* — across the available observation windows and iteratively optimises for coverage.

```bash
python src/tasking/scheduler.py \
    --sensors sensor_config.json \
    --tles tles.json \
    --t_start 2025-11-01T00:00:00 \
    --t_end 2025-11-01T06:00:00
```

---

### 3. Source Extraction — `src/source_extraction/physics_based/source_extractor.py`

Detect point-like sources in each raw FITS image frame. Three methods are available:

| Method | Function | Notes |
|---|---|---|
| Gaussian peak fitting | `gaussian_fitting_source_extraction` | Fast; uses `find_peaks` |
| DAOStarFinder | `DAOStarFinder_source_extraction` | Better for crowded fields |
| Image segmentation | `image_segmentation` | Most robust; Gaussian kernel convolution |

All methods subtract a sigma-clipped background before detection.

```bash
python src/source_extraction/physics_based/source_extractor.py \
    --test_frame_path frame.fits
```

---

### 4. Astrometry (Plate Solving) — `src/astrometry/`

Map pixel-space detections to celestial coordinates (RA/Dec) by matching them against a reference star catalog.

**Catalog queries** (`catalog_queries.py`): Query Gaia DR3 for stars within the sensor field of view and convert Gaia photometry to V-band magnitudes.

**Feature generation** (`feature_generation.py`): Build a database of *geometric quad hash codes* — rotation- and scale-invariant 4-tuples derived from groups of four stars — that can be matched against the same features computed from image detections.

**KD-tree** (`kd_tree.py`): Spatial index over the hash code database for efficient nearest-neighbour lookup during matching.

**Bayesian decision** (`bayesian_decision_maker.py`): A likelihood ratio test that decides whether a candidate match (image detections vs. catalog stars) is a true astrometric solution or a false alarm.

```bash
python src/astrometry/feature_generation.py \
    --catalog_name Gaia \
    --grid_size 1.0 1.0 \
    --min_ra 0.0 --max_ra 360.0 \
    --min_dec -90.0 --max_dec 90.0
```

---

### 5. Observation Association — `src/associator/`

Link per-frame detections into multi-frame tracklets corresponding to individual satellites.

**Simple associator** (`association.py`): Seeds candidate tracklets from first/last frame pairs, fills intermediate frames by linear interpolation, and filters by angular velocity threshold.

**Multiple Hypothesis Tracker** (`MHT.py`): A more principled approach that maintains a tree of candidate tracks, propagates each with a Kalman filter, and scores hypotheses using a Gaussian likelihood ratio against a uniform clutter model. Exposes a Flask API for integration with other services.

```bash
# Start the association service
python src/associator/flask.py --host 0.0.0.0 --port 5050
```

---

### 6. Synthetic Dataset Generation — `src/dataset/generate_satsim_dataset.py`

Generate randomised [SatSim](https://github.com/satsim/satsim) configuration files to produce synthetic satellite imagery for testing the pipeline. Randomises sensor parameters (resolution, FOV, noise, timing) and object populations (count, velocity, visual magnitude).

```bash
python src/dataset/generate_satsim_dataset.py \
    --output_dir ./synthetic_data \
    --num_collections 1000
```

---

## Dependencies

- `astropy`, `astroquery` — catalog queries and FITS I/O
- `photutils` — source detection
- `skyfield` — orbital propagation and visibility
- `numpy`, `scipy` — numerical routines
- `flask` — association service API
- `tqdm` — progress bars

---

## Status

This repository is a work in progress. Most modules have the correct algorithmic structure in place but are not yet fully implemented or tested.

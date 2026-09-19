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

### 6. Initial Orbit Determination — `src/iod/`

A pure numerical library (no HTTP service) implementing Pastor's Double r-iteration Lambert method (2022, Sec. 2.3.2): given n ≥ 3 angles-only observations (ra, dec, epoch, observer position), estimates a 6-D Cartesian state (position + velocity) at the first observation's epoch, with no initial guess required. It's a batch least-squares fit of just two parameters — the position-vector magnitudes at the first and last observation — rather than a full 6-parameter state fit, which is what keeps it well-conditioned on a single track's worth of angles.

Built from three independently-tested layers: `kepler.py` (universal-variable two-body propagator), `lambert.py` (universal-variable Lambert boundary-value solver), `circular.py` (circular-orbit seed for the gradient descent), composed in `double_r_lambert.py`. Validated against synthetic ground-truth orbits recovering both position and velocity to sub-km / sub-mm/s accuracy, including the paper's own GTO example (a=24460 km, e=0.71).

Scope limitation: both `lambert.py` and `circular.py` handle single-revolution (0-rev) transfers only — the observation span must stay under the orbit's actual period, which isn't known ahead of time (see `double_r_lambert.py`'s module docstring). `kepler.py`'s module docstring documents two future extensions not yet built: a J2-perturbed propagator variant (for longer-span propagation of confirmed objects, where the two-body model here would accumulate real secular drift error) and a differential-correction bridge from an IOD/OD Cartesian state to an SGP4-compatible TLE (for handing a confirmed object off to `src/tasking/`'s existing skyfield/SGP4 usage).

```python
from iod.double_r_lambert import Observation, solve

result = solve([
    Observation(ra_deg=..., dec_deg=..., epoch_s=..., site_position_km=[...]),
    ...  # n >= 3, any order
])
result.r_km, result.v_km_s   # Cartesian state at the first (earliest) observation's epoch
```

---

### 7. Catalog Store — `src/catalog_store/`

Persistent storage for tracks, candidate hypotheses, and confirmed catalog objects — the one stateful service in the pipeline (SQLite, WAL mode). Written to by the correlator (below), which the pipeline coordinator calls, on an opt-in basis, once per associator track.

A candidate object with too few tracks to support an orbit determination (an uncorrelated track, UCT) is stored the same way as any other hypothesis, just with `status='candidate'` — `GET /objects?status=candidate` is the UCT list, useful as a cue for scheduling follow-up observations.

```bash
# Start the catalog store service
DB_PATH=./catalog.db python src/catalog_store/api.py
```

| Endpoint | Description |
|---|---|
| `POST /tracks` | Ingest a raw attributable (ra, dec, ra_dot, dec_dot, covariance, optional site_position_km) |
| `POST /objects` | Create a hypothesis from one or more tracks |
| `GET /objects?status=candidate` | List uncorrelated tracks / in-progress hypotheses |
| `POST /objects/<id>/promote` | Confirm a hypothesis; atomically invalidates conflicting candidates |

---

### 8. Correlator — `src/correlator/`

Decides, for each newly ingested track, whether it belongs to an existing catalog hypothesis, pairs with another loose track to start a new one, or stands alone as an uncorrelated track — the generation / scoring / pruning / promotion cycle from Pastor (2022, Ch. 3). Stateless: all state lives in catalog-store, which this service calls over HTTP.

Generation, scoring and pruning use a linear (constant-velocity + process-noise) chi-square gate in measurement space — cheap, and the only option below 3 tracks anyway. Promotion is stricter: once a hypothesis has enough tracks and passes the linear gate, the correlator attempts a real orbit fit (`src/iod/double_r_lambert.py`) using every constituent track's angles. If every track has an observer position (`site_position_km` — nullable, nothing upstream populates it yet) and the fit converges with an acceptable residual, the hypothesis is promoted with that real 6-D Cartesian state instead of the linear 4-D attributable. If the fit doesn't converge or fits poorly, promotion is *withheld* even though the linear gate passed — a real orbit disagreeing is a stronger signal than the linear approximation agreeing. If IOD can't be attempted at all (missing observer positions on any track), promotion falls back to the linear-only decision.

Called from `src/pipeline/coordinator.py`'s `POST /pipeline/run`, opt-in via `"correlate": true` in the request (off by default — unlike every earlier pipeline stage, this writes to shared persistent state, so it isn't turned on implicitly). Each associator track becomes one catalog-store attributable and is sent to the correlator one at a time, in order — not concurrently, since two tracks from the same run can need to attribute to the same in-progress hypothesis, and catalog-store's generation-phase writes aren't cross-request atomic the way `promote_object` is. `site_position_km` — needed for IOD-gated promotion — is nullable and nothing upstream currently computes real per-epoch observer geodesy; the coordinator accepts an optional single fixed vector applied to every track in a run as a documented simplification (see `pipeline/coordinator.py`'s module docstring).

```bash
# Start standalone (requires catalog-store running)
CATALOG_STORE_URL=http://localhost:5004 python src/correlator/api.py
```

| Endpoint | Description |
|---|---|
| `POST /correlate` | Correlate one track (`{"track_id": N}` or a raw attributable) against the catalog |

---

### 9. Synthetic Dataset Generation — `src/dataset/generate_satsim_dataset.py`

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

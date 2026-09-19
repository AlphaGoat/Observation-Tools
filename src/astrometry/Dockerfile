# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
        numpy>=1.24 \
        scipy>=1.10 \
        joblib>=1.3 \
        astropy>=5.3 \
        astropy-healpix>=1.0 \
        matplotlib>=3.7 \
        flask>=3.0 \
        gunicorn>=21.0


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Plate Solver"
LABEL org.opencontainers.image.description="Astrometric plate solving against a pre-built quad-hash star index"

COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# /indices is a read-only mount for the pre-built CodeSpaceTree + StarPositionTree
# files (codes_tier*.joblib, stars.joblib) -- see kd_tree.py's build_index_tiered
# for how they're generated offline. Not baked into the image: the index for a
# given sky region/catalog epoch is large (multi-GB) and independent of the
# service code, so it's supplied at deploy time (docker-compose bind mount, or
# a Kubernetes PersistentVolumeClaim -- see k8s/index-pvc.yaml + plate-solver.yaml).
ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=5000 \
    WORKERS=2 \
    TIMEOUT=120 \
    INDEX_DIR=/indices

EXPOSE 5000

# start-period is long: loading the full tiered index into memory (KD-trees over
# hundreds of thousands of quads) takes real time, and the /health endpoint
# itself always returns 200 -- see api.py -- so this only guards against the
# process not being up at all, not against a still-loading index.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "astrometry.api:create_app()"

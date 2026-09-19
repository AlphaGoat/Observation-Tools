# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for scipy/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a prefix we can layer-copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Plate Solver"
LABEL org.opencontainers.image.description="Blind astrometric plate solver API"

# Runtime shared libs for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# ── Index mount point ─────────────────────────────────────────────────────────
# Indices are NOT baked into the image — they are mounted at runtime.
# Expected layout under the mount:
#   /indices/codes_tier08.joblib   (one file per scale tier)
#   /indices/codes_tier09.joblib
#   ...
#   /indices/stars.joblib
#
# Build indices with:
#   python -m astrometry.kd_tree build \
#       --min_ra 83 --max_ra 85 --min_dec -6 --max_dec -4 \
#       --tiers 8 9 10 11 --index_dir /indices --star_index /indices/stars.joblib
#
# Run with:
#   docker run -p 5000:5000 \
#       -v /path/to/your/indices:/indices:ro \
#       plate-solver
VOLUME /indices

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app/src \
    INDEX_DIR=/indices \
    STAR_INDEX=/indices/stars.joblib \
    HOST=0.0.0.0 \
    PORT=5000 \
    WORKERS=2 \
    TIMEOUT=120

EXPOSE 5000

# ── Health check ──────────────────────────────────────────────────────────────
# Uses /health endpoint; start period gives time for index loading.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker by default: the KD-trees are large in-memory structures that
# don't benefit from multiprocessing within a single container.  Increase
# WORKERS if the indices are small and you need concurrent solves.
CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "astrometry.api:create_app()"

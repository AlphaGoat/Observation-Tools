# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
        flask>=3.0 \
        gunicorn>=21.0


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Catalog Store"
LABEL org.opencontainers.image.description="SQLite-backed RSO catalog storage API"

COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# ── Environment ───────────────────────────────────────────────────────────────
# /data is a writable volume for the SQLite database file -- this is the one
# service in the pipeline that owns persistent, mutable state, so unlike the
# other services it needs a volume that survives container restarts.
ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=5004 \
    WORKERS=4 \
    TIMEOUT=60 \
    DB_PATH=/data/catalog.db

VOLUME ["/data"]

EXPOSE 5004

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# SQLite's own WAL-mode locking (with busy_timeout set in db.py) serialises
# concurrent writers safely, so multiple gunicorn workers sharing the same
# database file are fine -- no need to force WORKERS=1.
CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "catalog_store.api:create_app()"

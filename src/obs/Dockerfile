# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
        numpy>=1.24 \
        flask>=3.0 \
        gunicorn>=21.0


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Obs Projector"
LABEL org.opencontainers.image.description="Projects satellite pixel detections to RA/Dec observations using a plate WCS"

COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# ── Environment ───────────────────────────────────────────────────────────────
# Stateless — all data arrives in the request body; no volumes required.
ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=5003 \
    WORKERS=4 \
    TIMEOUT=30

EXPOSE 5003

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Fully stateless and CPU-light (pure numpy linear algebra), so 4 workers
# comfortably handle concurrent frame projections.
CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "obs.api:create_app()"

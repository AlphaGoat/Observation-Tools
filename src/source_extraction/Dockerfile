# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# sep is a C extension — needs a compiler at build time only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --prefix=/install \
        numpy>=1.24 \
        sep>=1.2 \
        astropy>=5.3 \
        flask>=3.0 \
        gunicorn>=21.0


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Source Extractor"
LABEL org.opencontainers.image.description="SEP-based satellite and star-trail extraction API for rate-tracked observations"

# libgomp1 is needed at runtime by numpy/sep for OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# ── Environment ───────────────────────────────────────────────────────────────
# No VOLUME — the extractor is stateless; all data arrives in the request body.
ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=5002 \
    WORKERS=2 \
    TIMEOUT=120

EXPOSE 5002

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Two workers: the service is stateless, but each request holds a full image
# array in memory, so we keep the worker count modest.
CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "source_extraction.api:create_app()"

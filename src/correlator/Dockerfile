# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
        numpy>=1.24 \
        scipy>=1.10 \
        flask>=3.0 \
        gunicorn>=21.0 \
        requests>=2.31


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Correlator"
LABEL org.opencontainers.image.description="Track-to-track orbit correlation against the catalog store"

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/

# Stateless -- holds no data of its own, everything lives in catalog-store.
ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=5005 \
    WORKERS=4 \
    TIMEOUT=60 \
    CATALOG_STORE_URL=http://catalog-store:5004

EXPOSE 5005

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "correlator.api:create_app()"

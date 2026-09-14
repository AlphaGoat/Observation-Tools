# Pipeline coordinator — lightweight image (no SEP / astropy / joblib needed).
# Only requires flask, gunicorn, and requests.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --prefix=/install \
        flask>=3.0 \
        gunicorn>=21.0 \
        requests>=2.31


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Observation-Tools Pipeline Coordinator"
LABEL org.opencontainers.image.description="Orchestrates the full observation pipeline: extract → solve → project → associate"

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/

ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8080 \
    WORKERS=4 \
    TIMEOUT=600 \
    STAR_EXTRACTOR_URL=http://star-extractor:5002 \
    SATELLITE_EXTRACTOR_URL=http://satellite-extractor:5002 \
    PLATE_SOLVER_URL=http://plate-solver:5000 \
    PROJECTOR_URL=http://projector:5003 \
    ASSOCIATOR_URL=http://associator:5001 \
    MAX_FRAME_WORKERS=8

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# Long timeout: a full multi-frame pipeline run can take several minutes.
CMD gunicorn \
        --workers  ${WORKERS} \
        --bind     ${HOST}:${PORT} \
        --timeout  ${TIMEOUT} \
        --access-logfile - \
        --error-logfile  - \
        "pipeline.coordinator:create_app()"

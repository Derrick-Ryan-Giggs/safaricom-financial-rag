# airflow/Dockerfile
#
# Extends the official Airflow image with `uv` and this project's own
# dependencies (via `uv sync` at build time), so DAG tasks can shell out to
# `uv run python -m ingestion.X` exactly like you already do by hand --
# one environment, no separate DockerOperator/container-per-task layer
# needed for a project this size.
#
# Build context is the project root (see docker-compose.yml's
# `context: ..`), so COPY paths below are relative to
# safaricom-financial-rag/, not airflow/.
#
# NOTE ON SYSTEM DEPENDENCIES: ingestion/extract.py's hi_res PDF strategy
# (unstructured + YOLOX layout detection + Tesseract OCR) needs system
# packages beyond plain Python deps. The apt-get list below is a
# best-effort starting point based on unstructured's documented hi_res
# requirements -- it has NOT been verified to build cleanly in this exact
# environment (I don't have your pyproject.toml/uv.lock to know exact
# pinned versions). Treat the first `docker compose build` as a real test,
# not a formality -- expect to iterate on this list if it fails.
#
# Airflow version tag below is illustrative -- check for a more current
# stable release before building, since Airflow ships frequently.
FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir uv

WORKDIR /opt/airflow/project

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen || uv sync
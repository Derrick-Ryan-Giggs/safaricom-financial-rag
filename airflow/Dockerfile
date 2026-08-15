# airflow/Dockerfile — Airflow 2.9.3 + project deps via uv. Build context is project root.
#
# Extends the official Airflow image with `uv` and this project's own
# dependencies (via `uv sync` at build time), so DAG tasks can shell out to
# `uv run python -m ingestion.X` exactly like you already do by hand.
#
# Build context is the project root (see docker-compose.yml's
# `context: .`), so COPY paths below are relative to
# safaricom-financial-rag/, not airflow/.
#
# LOCKFILE / CACHE / TIMEOUT: same reasoning as docker/Dockerfile --
# pyproject.toml + uv.lock copied together before `uv sync --frozen`
# (never regenerates the lock), a cache mount so a failed/timed-out
# download doesn't lose already-fetched packages on retry, and a raised
# HTTP timeout for large wheels (thinc, spacy deps) on slower connections.

FROM apache/airflow:2.9.3-python3.12

USER root
RUN rm -f /etc/apt/sources.list.d/*postgresql* /etc/apt/sources.list.d/*pgdg* /etc/apt/sources.list.d/*mariadb* \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir uv==0.9.25

WORKDIR /opt/airflow/project

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=1200
RUN --mount=type=cache,target=/home/airflow/.cache/uv,uid=50000 uv sync --frozen --extra extract

# Source itself comes from the bind mount in docker-compose.yml
# (.:/opt/airflow/project) at runtime, not baked in here.

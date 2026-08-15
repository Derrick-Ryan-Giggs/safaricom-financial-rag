# docker/Dockerfile
#
# Streamlit app image shared by the `app` and `dashboard` services (see
# x-app-common in docker-compose.yml). Build context is the project root
# (context: . in compose), so COPY paths below are relative to
# safaricom-financial-rag/, not docker/.
#
# `app` runs this image's default CMD (ui/app.py, the RAG chat interface).
# `dashboard` overrides command in compose to run monitoring/dashboard.py
# instead -- same image, same deps, different entry script.
#
# WORKDIR is /app to match the mount targets in x-app-common's volumes
# (./embeddings:/app/embeddings, ./models:/app/models) -- those bind
# mounts overlay whatever's baked in here at container start.
#
# LOCKFILE: pyproject.toml + the real uv.lock are copied together BEFORE
# sync runs, and --frozen refuses to touch the lock at all -- so the venv
# is always built against the actual committed lock, never a regenerated
# one that could drift and trigger a silent re-download later.
#
# CACHE MOUNT: --mount=type=cache keeps downloaded wheels on the host
# between build attempts. If a download times out partway through (as
# happened with thinc's wheel), a retry only re-fetches what's missing,
# not all 201 packages from zero.
#
# TIMEOUT: raised from the default 600s -- one large wheel timed out at
# 600s on a real-world connection, 1200s gives more headroom.
#
# --no-sync in CMD stops `uv run` from re-checking/re-downloading at
# every container start -- it just uses whatever's already in the venv.

FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.9.25

WORKDIR /app

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=1200
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

COPY . .

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

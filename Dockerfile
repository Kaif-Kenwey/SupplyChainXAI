# syntax=docker/dockerfile:1
# SupplyChainXAI — API + dashboard image.
#
#   docker build -t supplychainxai .
#   docker run -p 8000:8000 supplychainxai
#
# The image ships with the committed artifacts + SQLite store, so the API is
# immediately usable. Point DATABASE_URL at PostgreSQL for the production-like
# mode (see docker-compose.yml).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libpq for psycopg2; curl for the container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependency layer first — cached between code changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# application + data + artifacts (dashboard needs echarts vendor file)
COPY supplychainxai/ ./supplychainxai/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# non-root runtime user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/processed /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# initialise the store from the shipped CSVs (no-op for SQLite if present),
# then serve. DATABASE_URL selects SQLite vs PostgreSQL transparently.
CMD ["sh", "-c", "python scripts/init_store.py && exec uvicorn supplychainxai.api.app:app --host 0.0.0.0 --port 8000"]

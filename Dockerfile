# LiveOps Agent — multi-stage container
#
# Stage 1 (builder): compile wheels for everything in requirements.txt — including
# Prophet's cmdstan model, psycopg2, and any other deps that need a C++ toolchain.
# Stage 2 (runtime): copy the pre-built wheels into a slim runtime that only carries
# what's needed at run time (libpq5, curl for healthcheck). Final image is ~600MB
# smaller than the single-stage version and rebuilds skip Prophet's stan compile.
#
# Build:    docker build -t liveops-agent .
# Run UI:   docker run --rm -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" liveops-agent
# Run loop: docker run --rm --env-file .env -v "$PWD/data:/app/data" liveops-agent python auto_agent.py

# --------------------------------------------------------------------------- #
# Stage 1 — builder
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Toolchain for native wheels (Prophet/cmdstanpy, psycopg2, cryptography rust bits).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Build wheels for every dep into /wheels — runtime stage will install from here.
RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir=/wheels -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Runtime deps only — libpq5 for psycopg2, curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Install pre-built wheels from the builder stage — no toolchain in the runtime image.
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy app code last so layer cache is reused when only code changes.
COPY --chown=appuser:appuser . .

# Persist DB + user uploads under /app/data — mount as a volume in prod.
RUN mkdir -p /app/data /app/user_data \
    && chown -R appuser:appuser /app/data /app/user_data
VOLUME ["/app/data", "/app/user_data"]

USER appuser

EXPOSE 8000

# Healthcheck pings the FastAPI /healthz endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# Default = the FastAPI website. Override CMD to run auto_agent.py instead.
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

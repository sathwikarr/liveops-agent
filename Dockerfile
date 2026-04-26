# LiveOps Agent — single-stage container
#
# Build:   docker build -t liveops-agent .
# Run UI:  docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" liveops-agent
# Run loop: docker run --rm --env-file .env -v "$PWD/data:/app/data" liveops-agent python auto_agent.py
#
# Prophet/cmdstanpy needs a C++ toolchain at build time, so we install build-essential
# and clean it up after pip install to keep the image lean. Final image ~1.4 GB.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps for Prophet/cmdstanpy + curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Drop the build toolchain after Prophet's stan model is compiled.
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# App code.
COPY . .

# Persist DB + uploads under /app/data — mount this as a volume.
RUN mkdir -p /app/data /app/user_data
VOLUME ["/app/data", "/app/user_data"]

EXPOSE 8501

# Healthcheck pings Streamlit's internal status endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default = the dashboard. Override CMD to run auto_agent.py instead.
CMD ["streamlit", "run", "app.py"]

# Auto-trade container image
#
# Builds a single image that contains:
#   - Python 3.12 + ML deps (pandas, scikit-learn, joblib, flask)
#   - Taishin SDK (Linux manylinux wheel)
#   - All app code
#
# At runtime, different services use different entrypoints (see docker-compose.yml).

# Taishin SDK is x86_64-only (cp37-abi3-manylinux_2_17_x86_64).
# Build for amd64 even on Apple Silicon (qemu emulation).
FROM --platform=linux/amd64 python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Taipei

# System deps: curl for healthchecks; libgomp1 needed by scikit-learn / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tzdata libgomp1 \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy SDK wheel(s) first so layer caching is friendly
COPY taishin_sdk-*-manylinux*.whl /tmp/

# Install Python deps
RUN pip install --upgrade pip && \
    pip install \
      pandas==2.2.* \
      scikit-learn==1.8.* \
      joblib==1.5.* \
      flask==3.1.* \
      requests==2.32.* \
      beautifulsoup4==4.12.* && \
    pip install /tmp/taishin_sdk-*-manylinux*.whl && \
    rm /tmp/taishin_sdk-*.whl

# Copy app source (volumes will override data/, logs/, artifacts/ at runtime)
COPY *.py /app/
COPY stocks.json /app/

# Empty default directories — bind-mounted in compose
RUN mkdir -p /app/data /app/artifacts /app/logs

# Default command (overridden by each service in compose)
CMD ["python3", "dashboard.py"]

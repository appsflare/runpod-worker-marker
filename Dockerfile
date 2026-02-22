FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# --------------------------------------------------------------------------- #
# System dependencies
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Install UV
# --------------------------------------------------------------------------- #
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=python3.11 \
    TORCH_DEVICE=cuda \
    PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------- #
# Install Python dependencies via UV
# --------------------------------------------------------------------------- #
WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# --------------------------------------------------------------------------- #
# Copy worker source
# --------------------------------------------------------------------------- #
COPY handler.py ./

CMD ["python3.11", "handler.py"]

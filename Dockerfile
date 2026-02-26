FROM ollama/ollama:latest AS ollama

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# --------------------------------------------------------------------------- #
# System dependencies
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        git \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Install Ollama (copied from official image to avoid install script / systemd)
# --------------------------------------------------------------------------- #
COPY --from=ollama /usr/local/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/local/lib/ollama /usr/local/lib/ollama

# --------------------------------------------------------------------------- #
# Install UV
# --------------------------------------------------------------------------- #
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=python3.12 \  
    TORCH_DEVICE=cuda \
    MODEL_CACHE_DIR=/models \
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
COPY handler.py ollama_runner.py test_input.json ./

CMD ["uv", "run", "python3", "-u", "handler.py"]

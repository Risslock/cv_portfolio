FROM python:3.12-slim

# CUDA/cuDNN come from the `tensorflow[and-cuda]` pip wheels (see ../pyproject.toml).
# The NVIDIA driver itself is provided by the host through the container GPU
# runtime (`docker run --gpus=all`), so no CUDA toolkit is installed in this image.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Venv lives outside /workspace (which gets bind-mounted over at container start)
# so it's baked into the image layer and never collides with a native-Windows
# .venv living in the same project folder on the host.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --locked --no-install-project

# tensorflow[and-cuda]'s pip-installed CUDA/cuDNN libraries aren't found via
# rpath when TensorFlow lives in a uv-managed venv (confirmed while building
# this image). Registering their directories with ldconfig at build time makes
# the dynamic linker find them permanently, regardless of shell environment or
# how the container is later started.
RUN find /opt/venv/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib \
        > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig

WORKDIR /workspace

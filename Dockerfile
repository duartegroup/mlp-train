FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (nvidia driver + curl to bootstrap pixi)
RUN apt-get update && \
    apt-get install -y curl nvidia-driver-535-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Resolve the MACE environment. Copy the manifest + lockfile first so this layer
# is cached unless the dependencies change, then copy the source for the editable
# install of mlptrain.
COPY pixi.toml pixi.lock /app/
COPY . /app
# CONDA_OVERRIDE_CUDA lets the CUDA builds (locked via system-requirements
# cuda = "12") install on this GPU-less build host; they run on GPU at runtime.
RUN CONDA_OVERRIDE_CUDA=12.0 pixi install --locked -e mace && \
    rm -rf ~/.cache/rattler

ENTRYPOINT ["pixi", "run", "-e", "mace"]
CMD ["pytest"]

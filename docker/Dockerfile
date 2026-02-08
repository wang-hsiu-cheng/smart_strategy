# Use Ubuntu as base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Set the maintainer
LABEL maintainer="wang.tw.ted@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-tk \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# ---- build wheels only ----
# Install Gymnasium (with classic control + box2d + mujoco just in case)
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
    gymnasium[all] \
    stable-baselines3 \
    tensorboard \
    # 確保 SB3 與 tensorboard 相容
    shimmy \
    numpy \
    matplotlib

FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-tk \
    libgl1-mesa-glx \
    libosmesa6 \
    libglew2.2 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch GPU ----
RUN pip install --no-cache-dir \
    torch --extra--index-url https://download.pytorch.org/whl/cu122

# ---- install prebuilt wheels ----
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Working directory
WORKDIR /home/user/rl_main

ENV SHELL=/bin/bash
ENV TERM=xterm-256color

CMD ["/bin/bash", "-ic"]
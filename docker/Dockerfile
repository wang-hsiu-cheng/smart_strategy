# ==========================================
# 階段 1: Builder (編譯 wheels)
# ==========================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential \
    swig cmake curl libglew-dev libosmesa6-dev libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# 預先下載並編譯所有套件
RUN pip3 wheel --no-cache-dir --wheel-dir=/wheels \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple \
    torch==2.13.0 \
    gymnasium[classic_control,box2d,mujoco] \
    stable-baselines3 \
    sb3-contrib \
    tensorboard \
    shimmy \
    numpy \
    matplotlib

# ==========================================
# 階段 2: Base Runtime (共用底層)
# ==========================================
FROM ubuntu:22.04 AS base_runtime

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=ted
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# 1. 第一步：先只安裝套件（確保 sudo 已存在）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip sudo gosu curl lsb-release \
    libgl1 libglib2.0-0 libosmesa6 libglx-mesa0 libglfw3 \
    python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# 2. 第二步：建立使用者與設定權限 (分開執行避免語法解析混亂)
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip3 install --no-cache-dir /wheels/* \
    && find /usr/local/lib/python3.10/dist-packages/ -name "__pycache__" -type d -exec rm -rf {} +

# ==========================================
# 階段 3: Train Image (rl_main_train)
# ==========================================
FROM base_runtime AS rl_train

ARG USERNAME=ted
USER $USERNAME
WORKDIR /home/$USERNAME/rl_main/train
CMD ["/bin/bash", "-ic"]

# ==========================================
# 階段 4: Deploy Image (rl_main_deploy)
# ==========================================
FROM base_runtime AS rl_deploy
USER root
ARG USERNAME=ted

# 安裝 ROS2 Humble (直接裝在 NVIDIA 鏡像上)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rqt-plot ros-humble-rqt-tf-tree \
    ros-humble-nav2-msgs ros-humble-nav2-util \
    ros-humble-behaviortree-cpp ros-humble-generate-parameter-library \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
WORKDIR /home/$USERNAME/tdk_slam_ws
ARG WS_PATH=/home/$USERNAME/tdk_slam_ws

COPY entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-ic"]
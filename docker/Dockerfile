FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    TERM=xterm-256color \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
    LIBVA_DRIVER_NAME=iHD

ARG USERNAME=ted
ARG USER_UID=1000
ARG USER_GID=1000

# 1. 系統依賴、開發工具、GUI/虛擬渲染支援與 ROS 2 核心/感知/導航套件
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo gosu git curl wget cmake python3-pip usbutils build-essential tree \
        clinfo locales ca-certificates \
        python3-colcon-common-extensions \
        # GUI 與 Gymnasium Headless 虛擬渲染依賴
        libgl1 libglx-mesa0 libglib2.0-0 libosmesa6 libglfw3 python3-tk \
        # ROS 2 Jazzy 通訊、感知、可視化與除錯工具
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-tf2-ros \
        ros-jazzy-tf2-tools \
        ros-jazzy-rviz2 \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-laser-filters \
        ros-jazzy-pcl-ros \
        ros-jazzy-teleop-twist-keyboard \
        ros-jazzy-rqt-plot \
        ros-jazzy-rqt-graph \
        ros-jazzy-rqt-tf-tree \
        # 導航與決策控制（Nav2 / 行為樹）
        ros-jazzy-nav2-msgs \
        ros-jazzy-nav2-util \
        ros-jazzy-behaviortree-cpp \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Intel GPU 運算驅動鏈：本地 packages 優先，不存在或為空時自動透過 GitHub API 下載
RUN --mount=type=bind,source=packages,target=/tmp/packages \
    apt-get update && apt-get install -y --no-install-recommends curl jq wget ca-certificates \
    && mkdir -p /tmp/intel_install \
    && if ls /tmp/packages/*.deb 1> /dev/null 2>&1; then \
           echo "[+] 偵測到本地 packages 目錄中有 .deb，優先使用本地安裝包..." \
           && cp /tmp/packages/*.deb /tmp/intel_install/; \
       else \
           echo "[-] 本地 packages 目錄為空，透過 GitHub API 自動抓取最新驅動鏈..." \
           && IGC_URLS=$(curl -s https://api.github.com/repos/intel/intel-graphics-compiler/releases/latest \
               | jq -r '.assets[] | select(.name | test("intel-igc-(core|opencl).*amd64\\.deb$")) | .browser_download_url') \
           && for url in $IGC_URLS; do \
                  echo "下載 IGC: $url" && wget -q -P /tmp/intel_install "$url"; \
              done \
           && NEO_URLS=$(curl -s https://api.github.com/repos/intel/compute-runtime/releases/latest \
               | jq -r '.assets[] | select(.name | test("(intel-opencl-icd|libigdgmm12).*amd64\\.deb$")) | .browser_download_url') \
           && for url in $NEO_URLS; do \
                  echo "下載 NEO: $url" && wget -q -P /tmp/intel_install "$url"; \
              done; \
       fi \
    && (dpkg -i /tmp/intel_install/*.deb || apt-get install -f -y --no-install-recommends) \
    && rm -rf /tmp/intel_install \
    && echo "/usr/lib/wsl/lib" > /etc/ld.so.conf.d/wsl.conf \
    && ldconfig \
    && rm -f /etc/ld.so.preload \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/*

# 3. 安裝 Python 核心生態系：PyTorch XPU (Intel GPU 加速)、強化學習、模型導出與 MLOps 工具
# Step 3.1: 安裝針對 Intel XPU 硬體加速的 PyTorch 與 Torchvision
RUN pip3 install --no-cache-dir --break-system-packages --root-user-action=ignore \
        --index-url https://download.pytorch.org/whl/xpu \
        torch torchvision \
    # Step 3.2: 安裝強化學習、推論轉換 (含 onnxscript 防止導出錯誤)、版本管理與配置解析工具
    && pip3 install --no-cache-dir --break-system-packages --root-user-action=ignore \
        gymnasium[classic_control,box2d] \
        stable-baselines3 \
        sb3-contrib \
        tensorboard \
        huggingface_hub \
        shimmy \
        numpy \
        matplotlib \
        openvino \
        onnx \
        onnxruntime \
        onnxscript \
        python-dotenv \
        pydantic-settings \
        pyyaml \
        dvc \
    && find /usr/local/lib/python3.12/dist-packages/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 5. 精確設定使用者身分、GID 990 (消除啟動警告) 與工作空間擁有權
RUN touch /var/mail/ubuntu 2>/dev/null || true \
    && if id -u ubuntu >/dev/null 2>&1; then userdel -r -f ubuntu 2>/dev/null || true; fi \
    && if getent group ubuntu >/dev/null 2>&1; then groupdel ubuntu 2>/dev/null || true; fi \
    && (getent group $USER_GID >/dev/null 2>&1 || groupadd --gid $USER_GID $USERNAME) \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && (getent group 990 >/dev/null 2>&1 || groupadd -g 990 render_host) \
    && (getent group dialout >/dev/null 2>&1 || groupadd -r dialout) \
    && (getent group video >/dev/null 2>&1 || groupadd -r video) \
    && (getent group render >/dev/null 2>&1 || groupadd -r render) \
    && usermod -aG sudo,dialout,video,render,render_host $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME/rl_main \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# 6. 配置 entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

USER $USERNAME
ENV PATH="/home/ted/.local/bin:${PATH}"
WORKDIR /home/$USERNAME/rl_main
ARG WS_PATH=/home/$USERNAME/rl_main

# 7. 環境自動載入與常用工具別名
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f ${WS_PATH}/install/setup.bash ] && source ${WS_PATH}/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "alias foxglove='ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765'" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
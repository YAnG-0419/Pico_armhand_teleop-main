FROM ros:humble-ros-base

ARG LIBFRANKA_VERSION=0.19.0
ARG FRANKA_DESCRIPTION_TAG=1.0.1
ARG ROSDISTRO_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/rosdistro

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget \
    python3-pip python3-yaml python3-numpy \
    iproute2 \
    ros-humble-hardware-interface \
    ros-humble-controller-manager \
    ros-humble-controller-interface \
    ros-humble-ros2-control-test-assets \
    ros-humble-ros2controlcli \
    ros-humble-joint-state-broadcaster \
    ros-humble-generate-parameter-library \
    ros-humble-realtime-tools \
    ros-humble-backward-ros \
    ros-humble-xacro \
    libgflags-dev nlohmann-json3-dev libdw-dev libssl-dev \
    libgoogle-glog-dev libpoco-dev libeigen3-dev libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q \
      "https://github.com/frankarobotics/libfranka/releases/download/${LIBFRANKA_VERSION}/libfranka_${LIBFRANKA_VERSION}_jammy_amd64.deb" \
    && apt-get update \
    && apt-get install -y "./libfranka_${LIBFRANKA_VERSION}_jammy_amd64.deb" \
    && rm "libfranka_${LIBFRANKA_VERSION}_jammy_amd64.deb" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vendor_ws/src
RUN git clone --depth 1 --branch v2.0.4 \
        https://github.com/frankarobotics/franka_ros2.git \
    && git clone --depth 1 --branch "${FRANKA_DESCRIPTION_TAG}" \
        https://github.com/frankarobotics/franka_description.git \
    && rm -rf \
        franka_ros2/franka_gazebo \
        franka_ros2/franka_fr3_moveit_config \
        franka_ros2/franka_example_controllers \
        franka_ros2/franka_bringup \
        franka_ros2/integration_launch_testing \
        franka_ros2/franka_ros2

ENV ROSDISTRO_INDEX_URL=${ROSDISTRO_MIRROR}/index-v4.yaml
RUN sed -i \
      "s|https://raw.githubusercontent.com/ros/rosdistro/master|${ROSDISTRO_MIRROR}|g" \
      /etc/ros/rosdep/sources.list.d/20-default.list \
    && apt-get update \
    && (rosdep update || rosdep update || rosdep update) \
    && rosdep install --from-paths . --ignore-src -y \
        --dependency-types build \
        --dependency-types buildtool \
        --dependency-types build_export \
        --skip-keys "libfranka pinocchio" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vendor_ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && rm -rf build log

COPY patches/franka_ros2_restore_thread_scheduler.patch /tmp/
RUN git -C src/franka_ros2 apply /tmp/franka_ros2_restore_thread_scheduler.patch \
    && source /opt/ros/humble/setup.bash \
    && colcon build --packages-select franka_hardware \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && rm -rf build log /tmp/franka_ros2_restore_thread_scheduler.patch

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/pico_armhand_teleop
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

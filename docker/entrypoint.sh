#!/bin/bash
set -e

if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi
if ! grep -q "source /home/ted/rl_main/deploy/setup.bash" ~/.bashrc; then
    echo "source /home/ted/rl_main/deploy/setup.bash" >> ~/.bashrc
fi
if ! grep -q "alias groot=/home/ted/rl_main/groot/groot.AppImage" ~/.bashrc; then
    echo "alias groot=/home/ted/rl_main/groot/groot.AppImage" >> ~/.bashrc
fi
if ! grep -q "alias foxglove='ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765'" ~/.bashrc; then
    echo "alias foxglove='ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765'" >> ~/.bashrc
fi

source /opt/ros/humble/setup.bash
source /home/ted/rl_main/deploy/setup.bash
alias groot=/home/ted/rl_main/groot/groot.AppImage
alias foxglove='ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765'

exec "$@"
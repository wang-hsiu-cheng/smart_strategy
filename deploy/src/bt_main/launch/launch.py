import os
import launch_ros.actions

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
# for including other launch file
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
# for DeclareLaunchArgument
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    robot_config = os.path.join(
        get_package_share_directory('ppo_model'),
        'robot_config',
        'robot_config.yaml'
    )
    map_config = os.path.join(
        get_package_share_directory('ppo_model'),
        'map_config',
        'map_config.yaml'
    )

    model_server_node = Node(
        parameters=[
            robot_config,
            map_config
        ],
        package = 'ppo_model',
        executable = 'model_server',
        name = 'model_server'
    )
    bt_main_node = Node(
        parameters=[
            robot_config,
            map_config,
            {"frame_id": "base_footprint"},
            {"tree_name": "MainTree"}
        ],
        package='bt_main',
        executable='bt_main',
        name='bt_main',
        output='screen'
    )
    return LaunchDescription([
        model_server_node, 
        bt_main_node
    ])

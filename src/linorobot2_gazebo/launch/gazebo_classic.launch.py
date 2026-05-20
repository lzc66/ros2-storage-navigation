"""
Gazebo Classic launch for linorobot2 (ROS 2 Humble compatible).

Usage:
  LINOROBOT2_BASE=mecanum ros2 launch linorobot2_gazebo gazebo_classic.launch.py

Differences from gazebo.launch.py (Jazzy/gz_sim):
  - Uses gzserver + gzclient instead of ros_gz_sim
  - Uses spawn_entity.py instead of ros_gz_sim create
  - Uses libgazebo_ros_* plugins instead of ros_gz_bridge
"""
import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    TimerAction, SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = True

    robot_base = os.getenv('LINOROBOT2_BASE', 'mecanum')
    pkg_gazebo = FindPackageShare('linorobot2_gazebo')
    pkg_desc   = FindPackageShare('linorobot2_description')

    ekf_config_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_base"), "config", "ekf.yaml"]
    )

    urdf_path = PathJoinSubstitution(
        [pkg_desc, "urdf/robots", f"{robot_base}.urdf.xacro"]
    )

    description_launch_path = PathJoinSubstitution(
        [pkg_desc, 'launch', 'description.launch.py']
    )

    world_path = PathJoinSubstitution(
        [pkg_gazebo, 'worlds', 'turtlebot3_world.sdf']
    )

    # Read URDF via xacro
    from ament_index_python.packages import get_package_share_directory
    desc_share = get_package_share_directory('linorobot2_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'robots', f'{robot_base}.urdf.xacro')
    import xacro
    robot_desc = xacro.process_file(urdf_file).toprettyxml(indent='  ')

    return LaunchDescription([
        DeclareLaunchArgument(
            name='gui',
            default_value='true',
            description='Enable Gazebo Client'
        ),

        DeclareLaunchArgument(
            name='world_name',
            default_value='turtlebot3_world',
            description='Gazebo world name'
        ),

        DeclareLaunchArgument(
            name='spawn_x', default_value='0.0',
            description='Robot spawn X'
        ),
        DeclareLaunchArgument(
            name='spawn_y', default_value='0.0',
            description='Robot spawn Y'
        ),
        DeclareLaunchArgument(
            name='spawn_z', default_value='0.0',
            description='Robot spawn Z'
        ),
        DeclareLaunchArgument(
            name='spawn_yaw', default_value='0.0',
            description='Robot spawn yaw'
        ),

        # Gazebo server
        ExecuteProcess(
            cmd=['gzserver', '--verbose',
                 LaunchConfiguration('world_path', default=[pkg_gazebo, '/worlds/', LaunchConfiguration('world_name'), '.sdf']),
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so'],
            output='screen',
        ),

        # Gazebo client (conditional)
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('gui')),
        ),

        # Spawn robot
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', 'linorobot2',
                        '-topic', 'robot_description',
                        '-x', LaunchConfiguration('spawn_x'),
                        '-y', LaunchConfiguration('spawn_y'),
                        '-z', LaunchConfiguration('spawn_z'),
                        '-Y', LaunchConfiguration('spawn_yaw'),
                        '-unpause',
                    ],
                    output='screen',
                ),
            ],
        ),

        # Robot state publisher with URDF
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'robot_description': robot_desc,
                    }],
                    output='screen',
                ),
            ],
        ),

        # EKF for odometry
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node',
                    output='screen',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        ekf_config_path,
                    ],
                    remappings=[("odometry/filtered", "/odom")],
                ),
            ],
        ),
    ])

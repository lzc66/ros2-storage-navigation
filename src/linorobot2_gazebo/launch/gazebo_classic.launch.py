"""
Gazebo Classic launch for linorobot2 (ROS 2 Humble + ros2_control).

Usage:
  LINOROBOT2_BASE=mecanum ros2 launch linorobot2_gazebo gazebo_classic.launch.py

Features:
  - Gazebo Classic (gzserver + gzclient)
  - ros2_control: mecanum_drive_controller + lift_controller + joint_state_broadcaster
  - Prismatic lift joint (0.0 – 0.8m, position-controlled)
  - Spawns linorobot2 with URDF via spawn_entity.py
"""
import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, TimerAction, RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = True

    robot_base = os.getenv('LINOROBOT2_BASE', 'mecanum')
    pkg_gazebo = FindPackageShare('linorobot2_gazebo')
    pkg_desc   = FindPackageShare('linorobot2_description')

    # Resolve URDF via xacro at launch time
    from ament_index_python.packages import get_package_share_directory
    desc_share = get_package_share_directory('linorobot2_description')
    urdf_file = os.path.join(desc_share, 'urdf', 'robots', f'{robot_base}.urdf.xacro')
    import xacro
    robot_desc = xacro.process_file(urdf_file).toprettyxml(indent='  ')

    controller_yaml = PathJoinSubstitution(
        [pkg_gazebo, 'config', 'controller_manager.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            name='gui', default_value='true',
            description='Enable Gazebo Client'
        ),
        DeclareLaunchArgument(
            name='world_name', default_value='turtlebot3_world',
            description='Gazebo world name'
        ),
        DeclareLaunchArgument(
            name='spawn_x', default_value='0.0'),
        DeclareLaunchArgument(
            name='spawn_y', default_value='0.0'),
        DeclareLaunchArgument(
            name='spawn_z', default_value='0.0'),
        DeclareLaunchArgument(
            name='spawn_yaw', default_value='0.0'),

        # Gazebo server (ros2_control plugin uses --ros-args for ROS interface)
        ExecuteProcess(
            cmd=['gzserver', '--verbose',
                 LaunchConfiguration('world_path',
                     default=[pkg_gazebo, '/worlds/', LaunchConfiguration('world_name'), '.sdf']),
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

        # robot_state_publisher loads URDF for gazebo_ros2_control to read
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
            }],
        ),

        # Spawn robot entity (reads robot_description from topic)
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

        # joint_state_broadcaster (publishes /joint_states for all joints)
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_broadcaster',
                               '--controller-manager', '/controller_manager'],
                    output='screen',
                ),
            ],
        ),

        # mecanum_drive_controller (handles /cmd_vel, publishes /odom and /tf)
        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['mecanum_controller',
                               '--controller-manager', '/controller_manager'],
                    output='screen',
                ),
            ],
        ),

        # lift_controller (position controller for prismatic lift joint)
        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['lift_controller',
                               '--controller-manager', '/controller_manager'],
                    output='screen',
                ),
            ],
        ),
    ])

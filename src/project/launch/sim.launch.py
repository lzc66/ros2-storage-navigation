import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('project')

    # GAZEBO_MODEL_PATH: include our custom models and system models
    model_path = os.path.join(pkg_share, 'models')
    sys_model_path = os.path.expanduser('~/.gazebo/models')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        model_path += ':' + os.environ['GAZEBO_MODEL_PATH']
    model_path += ':' + sys_model_path

    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path)

    world_file = os.path.join(pkg_share, 'worlds', 'indoor_scene.world')

    # Start Gazebo server + client
    gazebo = ExecuteProcess(
        cmd=['gzserver', '--verbose', world_file, '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    # Spawn mbot robot in Gazebo
    mbot_sdf = os.path.join(pkg_share, 'models', 'mbot', 'model.sdf')

    spawn_mbot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', mbot_sdf,
            '-entity', 'mbot',
            '-x', '0.0', '-y', '0.0', '-z', '0.1',
            '-robot_namespace', 'mbot',
            '-unpause'
        ],
        output='screen',
    )

    # Start item_spawner node
    item_spawner = Node(
        package='project',
        executable='item_spawner.py',
        name='item_spawner',
        output='screen',
    )

    return LaunchDescription([
        set_model_path,
        gazebo,
        gzclient,
        TimerAction(
            period=3.0,
            actions=[spawn_mbot]
        ),
        TimerAction(
            period=5.0,
            actions=[item_spawner]
        ),
    ])

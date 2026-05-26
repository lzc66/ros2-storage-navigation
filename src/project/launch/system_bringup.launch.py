"""One-click bringup: Gazebo AWS Warehouse + Linorobot2 Mecanum + Nav2 + Vision + Brain."""
import os
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, SetEnvironmentVariable, TimerAction,
)
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('project')
    aws_share = get_package_share_directory('aws_robomaker_small_warehouse_world')

    # Force linorobot2 mecanum as the only hardware platform
    os.environ['LINOROBOT2_BASE'] = 'mecanum'

    # Model paths
    aws_model_path = os.path.join(aws_share, 'models')
    model_path = os.path.join(pkg_share, 'models')
    sys_model_path = os.path.expanduser('~/.gazebo/models')
    lino_gazebo_share = get_package_share_directory('linorobot2_gazebo')
    lino_desc_share = get_package_share_directory('linorobot2_description')
    full_model_path = (
        f'{model_path}:{aws_model_path}:{sys_model_path}:'
        f'{os.path.join(lino_gazebo_share, "models")}'
    )
    env_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    if env_model_path:
        full_model_path += f':{env_model_path}'

    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', full_model_path)
    set_py_unbuf = SetEnvironmentVariable('PYTHONUNBUFFERED', '1')
    set_robot_base = SetEnvironmentVariable('LINOROBOT2_BASE', 'mecanum')

    # AWS RoboMaker Small Warehouse (no roof)
    world_file = os.path.join(aws_share, 'worlds', 'no_roof_small_warehouse',
                              'no_roof_small_warehouse.world')
    # AWS warehouse map (5cm resolution, aligned with no_roof_small_warehouse)
    map_file = os.path.join(aws_share, 'maps', '005', 'map.yaml')
    nav2_params = os.path.join(pkg_share, 'params', 'nav2_params.yaml')
    # sim_time only for Gazebo nodes; Nav2 uses wall time (avoids /clock QoS mismatch)
    use_sim_time = SetParameter(name='use_sim_time', value=False)

    # World XML: inject gazebo_ros_state plugin
    dyn_world = '/tmp/dynamic_world.world'
    with open(world_file, 'r') as f:
        world_xml = f.read()
    world_xml = world_xml.replace(
        '</world>',
        '  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"/>\n</world>'
    )
    with open(dyn_world, 'w') as f:
        f.write(world_xml)

    # Gazebo server + client
    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose', dyn_world,
             '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen',
    )
    gzclient = ExecuteProcess(
        cmd=['gzclient'], output='screen',
    )

    # --- Linorobot2 Mecanum: xacro → clean SDF → factory spawn ---
    import xacro, re, subprocess as _sp
    urdf_file = os.path.join(lino_desc_share, 'urdf', 'robots', 'mecanum.urdf.xacro')
    robot_desc = xacro.process_file(urdf_file).toprettyxml(indent='  ')
    urdf_tmp = '/tmp/linorobot2_mecanum.urdf'
    with open(urdf_tmp, 'w') as f:
        f.write(robot_desc)
    sdf_text = _sp.run(['gz', 'sdf', '-p', urdf_tmp],
                       check=True, timeout=30, capture_output=True, text=True).stdout
    sdf_text = re.sub(r"\s+gz:[a-z_]+='[^']*'", '', sdf_text)
    sdf_text = re.sub(r'\s+gz:[a-z_]+="[^"]*"', '', sdf_text)
    sdf_tmp = '/tmp/linorobot2_mecanum.sdf'
    with open(sdf_tmp, 'w') as f:
        f.write(sdf_text)

    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-entity', 'linorobot2',
            '-file', sdf_tmp,
            '-x', '-2.0', '-y', '-0.5', '-z', '0.1', '-unpause',
        ],
        output='screen',
    )

    # Robot state publisher (TF tree)
    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'use_sim_time': False, 'robot_description': robot_desc}],
        output='screen',
    )

    # --- Nav2 nodes ---
    map_server = Node(package='nav2_map_server', executable='map_server',
                      parameters=[nav2_params,
                                  {'use_sim_time': False, 'yaml_filename': map_file}],
                      output='screen')
    # Ground-truth localization: static map→odom identity (no AMCL needed)
    static_map_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )
    lifecycle_mgr_loc = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_localization',
                             parameters=[{'use_sim_time': False,
                                          'node_names': ['map_server'],
                                          'autostart': True,
                                          'bond_timeout': 10.0}],
                             output='screen')
    planner_server = Node(package='nav2_planner', executable='planner_server',
                          parameters=[nav2_params], output='screen')
    controller_server = Node(package='nav2_controller', executable='controller_server',
                             parameters=[nav2_params], output='screen')
    bt_navigator = Node(package='nav2_bt_navigator', executable='bt_navigator',
                        parameters=[nav2_params], output='screen')
    behavior_server = Node(package='nav2_behaviors', executable='behavior_server',
                           parameters=[nav2_params], output='screen')
    waypoint_follower = Node(package='nav2_waypoint_follower', executable='waypoint_follower',
                             parameters=[nav2_params], output='screen')
    velocity_smoother = Node(package='nav2_velocity_smoother', executable='velocity_smoother',
                             parameters=[nav2_params], output='screen')
    lifecycle_mgr_nav = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_navigation',
                             parameters=[{'use_sim_time': False,
                                          'node_names': ['planner_server', 'controller_server',
                                                         'bt_navigator', 'behavior_server',
                                                         'waypoint_follower', 'velocity_smoother'],
                                          'autostart': True}],
                             output='screen')

    # --- Vision + Brain (camera topics aligned with Gazebo Classic output) ---
    vision_node = Node(package='project', executable='vision_node.py',
                       name='vision_node', output='screen',
                       remappings=[
                           ('/camera/color/image_raw', '/camera/image_raw'),
                           ('/camera/color/camera_info', '/camera/camera_info'),
                       ])
    brain_node = Node(package='project', executable='brain_node.py',
                      name='brain_node', output='screen')
    item_spawner = Node(package='project', executable='item_spawner.py',
                        name='item_spawner', output='screen')
    dynamic_obstacle = Node(package='project', executable='dynamic_obstacle.py',
                            name='dynamic_obstacle', output='screen')

    return LaunchDescription([
        use_sim_time, set_model_path, set_py_unbuf, set_robot_base,
        gzserver, gzclient,
        TimerAction(period=1.0, actions=[robot_state_pub]),
        TimerAction(period=25.0, actions=[spawn_robot]),
        # Nav2: lifecycle managers start AFTER managed nodes for service discovery
        TimerAction(period=30.0, actions=[
            map_server, static_map_odom,
        ]),
        TimerAction(period=32.0, actions=[lifecycle_mgr_loc]),
        TimerAction(period=33.0, actions=[
            planner_server, controller_server, bt_navigator,
            behavior_server, waypoint_follower, velocity_smoother,
            lifecycle_mgr_nav,
        ]),
        TimerAction(period=37.0, actions=[vision_node, brain_node, item_spawner]),
        TimerAction(period=40.0, actions=[dynamic_obstacle]),
    ])

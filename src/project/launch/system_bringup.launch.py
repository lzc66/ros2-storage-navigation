"""One-click bringup: Gazebo + Nav2 + Vision + Brain + Dynamic Obstacle."""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('project')

    model_path = os.path.join(pkg_share, 'models')
    sys_model_path = os.path.expanduser('~/.gazebo/models')
    tb3_model_path = '/opt/ros/humble/share/turtlebot3_gazebo/models'
    env_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    full_model_path = f'{model_path}:{sys_model_path}:{tb3_model_path}'
    if env_model_path:
        full_model_path += f':{env_model_path}'

    set_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', full_model_path)

    world_file = os.path.join(pkg_share, 'worlds', 'turtlebot3_world.world')
    map_file = os.path.join(pkg_share, 'maps', 'map.yaml')
    nav2_params = os.path.join(pkg_share, 'params', 'nav2_params.yaml')

    use_sim_time = SetParameter(name='use_sim_time', value=True)

    # === Stage 8: Dynamic World Injection ===
    # Inject gazebo_ros_state plugin directly into the world XML
    dyn_world = '/tmp/dynamic_world.world'
    with open(world_file, 'r') as f:
        world_xml = f.read()
    world_xml = world_xml.replace(
        '</world>',
        '  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"/>\n</world>'
    )
    with open(dyn_world, 'w') as f:
        f.write(world_xml)

    # Gazebo (no -s state plugin — injected into world XML instead)
    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose', dyn_world,
             '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    # Spawn waffle
    tb3_sdf = '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf'
    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', tb3_sdf, '-entity', 'turtlebot3',
                   '-x', '-2.0', '-y', '-0.5', '-z', '0.1', '-unpause'],
        output='screen',
    )

    # Robot state publisher
    urdf_path = '/opt/ros/humble/share/turtlebot3_gazebo/urdf/turtlebot3_waffle.urdf'
    robot_desc = open(urdf_path, 'r').read()
    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'use_sim_time': True, 'robot_description': robot_desc}],
        output='screen',
    )

    # Nav2 nodes
    map_server = Node(package='nav2_map_server', executable='map_server',
                      parameters=[nav2_params,
                                  {'use_sim_time': True, 'yaml_filename': map_file}],
                      output='screen')
    amcl = Node(package='nav2_amcl', executable='amcl', name='amcl',
                parameters=[nav2_params], output='screen')
    lifecycle_mgr_loc = Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                             name='lifecycle_manager_localization',
                             parameters=[{'use_sim_time': True,
                                          'node_names': ['map_server', 'amcl'], 'autostart': True}],
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
                             parameters=[{'use_sim_time': True,
                                          'node_names': ['planner_server', 'controller_server',
                                                         'bt_navigator', 'behavior_server',
                                                         'waypoint_follower', 'velocity_smoother'],
                                          'autostart': True}],
                             output='screen')

    # Stage 3: Vision + Brain
    vision_node = Node(package='project', executable='vision_node.py',
                       name='vision_node', output='screen')
    brain_node = Node(package='project', executable='brain_node.py',
                      name='brain_node', output='screen')

    # Item spawner (from Stage 1)
    item_spawner = Node(package='project', executable='item_spawner.py',
                        name='item_spawner', output='screen')

    # Stage 6: Dynamic obstacle (moving worker)
    dynamic_obstacle = Node(package='project', executable='dynamic_obstacle.py',
                            name='dynamic_obstacle', output='screen')

    return LaunchDescription([
        use_sim_time, set_model_path,
        gzserver, gzclient,
        TimerAction(period=3.0, actions=[spawn_robot]),
        TimerAction(period=5.0, actions=[robot_state_pub]),
        TimerAction(period=8.0, actions=[
            map_server, amcl, lifecycle_mgr_loc,
            planner_server, controller_server, bt_navigator,
            behavior_server, waypoint_follower, velocity_smoother,
            lifecycle_mgr_nav,
        ]),
        TimerAction(period=12.0, actions=[vision_node, brain_node, item_spawner]),
        TimerAction(period=14.0, actions=[dynamic_obstacle]),
    ])

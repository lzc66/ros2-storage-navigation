#!/usr/bin/python3
"""
Dynamic Obstacle: a moving worker cylinder that oscillates across the central corridor,
testing Nav2's DWA local planner ability to avoid dynamic obstacles.
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, SetEntityState
import math
import time


class DynamicObstacle(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle')
        self._state_cli = self.create_client(SetEntityState, '/set_entity_state')
        self._spawn_worker()
        self._last_time = time.time()
        self._y_pos = 0.5
        self._dy = -0.2   # speed m/s (negative = moving toward -0.5)
        self._timer = self.create_timer(0.1, self._move_worker)
        self.get_logger().info('Dynamic Obstacle: worker oscillating y=[-0.5, 0.5]')

    def _spawn_worker(self):
        cli = self.create_client(SpawnEntity, '/spawn_entity')
        cli.wait_for_service()
        sdf = '''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="moving_worker">
    <pose>0.0 0.5 0.75 0 0 0</pose>
    <link name="link">
      <kinematic>true</kinematic>
      <inertial><mass>100.0</mass><inertia><ixx>10</ixx><iyy>10</iyy><izz>10</izz></inertia></inertial>
      <collision name="c">
        <geometry><cylinder><radius>0.2</radius><length>1.5</length></cylinder></geometry>
        <surface>
          <contact>
            <collide_without_contact>true</collide_without_contact>
          </contact>
        </surface>
      </collision>
      <visual name="v">
        <geometry><cylinder><radius>0.2</radius><length>1.5</length></cylinder></geometry>
        <material><ambient>1 1 0 1</ambient><diffuse>1 1 0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>'''
        req = SpawnEntity.Request()
        req.name = 'moving_worker'
        req.xml = sdf
        req.initial_pose.position.x = 0.0
        req.initial_pose.position.y = 0.5
        req.initial_pose.position.z = 0.75
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().success:
            self.get_logger().info('Worker spawned at (0.0, 0.5)')
        else:
            self.get_logger().warn('Worker spawn failed (may already exist)')

    def _move_worker(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        self._y_pos += self._dy * dt
        if self._y_pos > 0.5:
            self._y_pos = 0.5
            self._dy = -abs(self._dy)
        elif self._y_pos < -0.5:
            self._y_pos = -0.5
            self._dy = abs(self._dy)

        if not self._state_cli.service_is_ready():
            return

        req = SetEntityState.Request()
        req.state.name = 'moving_worker'
        req.state.pose.position.x = 0.0
        req.state.pose.position.y = self._y_pos
        req.state.pose.position.z = 0.75
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'
        self._state_cli.call_async(req)  # reuse single client


def main():
    rclpy.init()
    node = DynamicObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

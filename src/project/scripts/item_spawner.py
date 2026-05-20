#!/usr/bin/python3
import threading
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from project.srv import SpawnItem

# Anti-explosion: spawn items slightly above target Z to avoid mesh overlap,
# relying on gravity to settle them onto the shelf surface below.
Z_CLEARANCE = 0.05  # meters


class GazeboClient(Node):
    """Separate node running in its own thread for calling Gazebo services."""
    def __init__(self):
        super().__init__('gazebo_client')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('GazeboClient waiting for /spawn_entity...')

    def spawn(self, name, xml, x, y, z):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.robot_namespace = ''
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.done() and future.result() is not None:
            return future.result().success
        return False


class ItemSpawner(Node):
    def __init__(self):
        super().__init__('item_spawner')

        self._gazebo_client = GazeboClient()
        self._gazebo_thread = threading.Thread(
            target=rclpy.spin, args=(self._gazebo_client,), daemon=True
        )
        self._gazebo_thread.start()

        self.srv = self.create_service(SpawnItem, 'spawn_item', self.spawn_item_callback)
        self.get_logger().info('Item Spawner ready. Service: /spawn_item')
        self.count = 0

    def spawn_item_callback(self, request, response):
        target_type = request.target_type.lower()
        x, y, z = request.x, request.y, request.z

        if target_type not in ('box', 'barrel'):
            self.get_logger().error(f'Unknown target_type: {target_type}. Use "box" or "barrel".')
            response.success = False
            response.message = f'Unknown target_type: {target_type}'
            return response

        self.count += 1
        name = f'{target_type}_{self.count}'
        sdf = self._make_sdf(name, target_type)

        # Apply Z-clearance: spawn above shelf surface, let gravity settle.
        # Prevents physics explosion from collision-mesh overlap on multi-level shelves.
        safe_z = z + Z_CLEARANCE
        ok = self._gazebo_client.spawn(name, sdf, x, y, safe_z)

        if ok:
            self.get_logger().info(
                f'Spawned [{target_type}] {name} at ({x:.1f}, {y:.1f}, {z:.1f})'
            )
            response.success = True
            response.message = f'Spawned {target_type} at ({x:.1f}, {y:.1f}, {z:.1f}) [clearance Z={safe_z:.2f}]'
        else:
            self.get_logger().error(f'Failed to spawn {name}')
            response.success = False
            response.message = f'Failed to spawn {name}'

        return response

    def _make_sdf(self, name, target_type):
        if target_type == 'box':
            return self._box_sdf(name)
        else:
            return self._barrel_sdf(name)

    def _box_sdf(self, name):
        return f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <iyy>0.01</iyy>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.3 0.3 0.3</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.3 0.3 0.3</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''

    def _barrel_sdf(self, name):
        return f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.04</ixx>
          <iyy>0.04</iyy>
          <izz>0.02</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.55 0.27 0.07 1</ambient>
          <diffuse>0.55 0.27 0.07 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''


def main(args=None):
    rclpy.init(args=args)
    node = ItemSpawner()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/python3
"""
Mock Vision Node — ground-truth position injection.

Publishes a fixed target position as PointStamped on /target_object
to feed brain_node for E2E pipeline verification.

The target position matches e2e_test_box spawn coordinates (1.5, 0.0).
For different targets, republish with updated coordinates.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


TARGET_X = 1.5
TARGET_Y = 0.0
PUB_HZ = 10.0


class MockVisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self._pub = self.create_publisher(PointStamped, '/target_object', 10)
        self._timer = self.create_timer(1.0 / PUB_HZ, self._publish)
        self.get_logger().info(
            f'Mock Vision: publishing ({TARGET_X:.1f},{TARGET_Y:.1f}) '
            f'on /target_object @ {PUB_HZ}Hz'
        )

    def _publish(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = TARGET_X
        msg.point.y = TARGET_Y
        msg.point.z = 1.0  # type code: 1.0 = red
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = MockVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
